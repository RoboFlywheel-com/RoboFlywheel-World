#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure a built room shell from its Blender geometry.

    blender -b <out>/scene.blend -P python -m rfw_w.scenes.workers.blender.measure_built_shell -- \
        --plan tests/fixtures/office_room_plan.json --out /tmp/measured.json

Run under **Infinigen's bundled Blender Python**, not under any conda interpreter.

Why this runs in Blender rather than over the exported USD: Infinigen deliberately does not
export its `MaskTag` face attribute (`infinigen/core/util/exporting.py:40` is `if False and ...`),
so the exported mesh has no semantics -- only prim names. The `.blend` still carries the tags and
the real geometry, and it is what `-t coarse` leaves behind.

**It measures, it does not judge.** Every number it can establish goes in `measurements`; every
number it could not goes in `not_measured` with a reason. Deciding whether a deviation is
acceptable is `shell.matches_plan`'s job (SPEC-004) -- a ruler that also decided would hide the
cases where the ruler itself was the problem.

Three properties of the built geometry shape how this works, and each was established by reading
the mesh rather than assumed:

1. **`<room>.wall` is the surfaces visible from inside the room**, not a closed solid. Every face's
   normal points into the room and every vertex sits at the *interior* boundary. There is no
   outward-facing face to find, so "pick the outer one" is not a question this geometry can answer.
2. **The interior is inset from the plan outline.** For the office probe the inset is 0.05 m on
   every side: interior faces at x=0.914 where the plan says 0.864, at y=0.796 where it says 0.746,
   floor from z=0.05 where the plan's datum is 0.
3. **Doors and windows are holes in those faces**, so an opening is measured as a gap in coverage
   and never as an object.

Because of (1) and (2) the comparison is done on **wall lines**, not on polygons: each plan edge is
matched to the measured wall that lies nearest and parallel to it, and what matters is the
*spread* of those distances. A builder that reproduced the plan's shape has one constant offset
(the inset); a builder that did not has residuals that scatter. That measure needs no assumption
about what the plan polygon's datum means, which a polygon-to-polygon area comparison would.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


#: Faces are grouped by rounded normal, then by offset with a merge tolerance rather than a
#: rounding. Rounding the offset splits a wall across a bucket boundary: boolean output puts the
#: office's south wall's faces at 0.0498 and 0.0502 alike, and a fixed decimal cut put them in
#: different planes, which then read as two half-walls with a hole between them.
PLANE_KEY_DECIMALS = 3
#: 1 mm. Wide enough to absorb the scatter that motivated merging (the office's south wall faces
#: read 0.0498 and 0.0502), and narrow enough to keep apart two planes that are genuinely
#: different: at 5 mm the notch wall at y=0.796 fused with a window reveal at y=0.799, and the
#: resulting "wall" ran the room's full width with holes in it.
PLANE_MERGE_TOLERANCE_M = 1e-3

#: A wall runs from floor to ceiling. Anything vertical but shorter is a feature *in* a wall -- the
#: side of a doorway or a window -- and the office probe shows both kinds: walls spanning
#: [0.05, 2.9197] and reveals spanning [0.9669, 2.0663] or [0.05, 2.0721].
FULL_HEIGHT_TOLERANCE_M = 0.2

#: How far apart two lengths may be and still count as one: 2 cm. Used to tell a horizontal
#: boundary loop (the room's floor edge) from a vertical one, and to tolerate a loop reaching a
#: hair past the wall it belongs to.
EDGE_TOLERANCE_M = 0.02

#: Two lines count as parallel when their direction vectors agree this closely.
PARALLEL_TOLERANCE = 1e-3


def fail(code: str, detail: str) -> None:
    """Write a named failure and exit non-zero. Named beats a traceback: the caller reads JSON."""
    import bpy
    from mathutils import Vector
    print(json.dumps({"error": code, "detail": detail}))
    raise SystemExit(1)


def canonical_plane(normal: Vector) -> Vector:
    """The normal with its first non-zero component made positive.

    Canonical, because the two faces of one line must land in the same group. The sign is not kept:
    a line's offset is `normal . p` either way, and keeping the sign is what once put the office's
    east wall at x = -4.4024.
    """
    import bpy
    from mathutils import Vector
    # Medium U-room seed 20 exposed 1e-7 noise in nominally zero components:
    # choosing sign before quantisation split one wall into opposite normal buckets.
    unit = Vector(tuple(round(v, PLANE_KEY_DECIMALS) for v in Vector(normal).normalized())).normalized()
    for component in (unit.x, unit.y, unit.z):
        if abs(component) > 1e-9:
            return unit if component > 0 else -unit
    return unit


def mesh_faces(obj) -> list[dict]:
    """World-space vertices, centroid and normal for every face of a mesh object."""
    import bpy
    from mathutils import Vector
    matrix = obj.matrix_world
    vertices = [matrix @ vertex.co for vertex in obj.data.vertices]
    normals = [matrix.to_3x3() @ polygon.normal for polygon in obj.data.polygons]
    faces = []
    for polygon, normal in zip(obj.data.polygons, normals):
        points = [vertices[index] for index in polygon.vertices]
        centre = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
        faces.append({"points": points, "centre": centre, "normal": normal.normalized()})
    return faces



def floor_area_m2(obj) -> float:
    """Area of actual upward-facing floor polygons in world coordinates, not the plan."""
    import bpy
    from mathutils import Vector
    area = 0.0
    for face in mesh_faces(obj):
        if face["normal"].z < 1.0 - PARALLEL_TOLERANCE:
            continue
        points = face["points"]
        cross_sum = sum((p.cross(points[(i + 1) % len(points)])
                         for i, p in enumerate(points)), Vector((0., 0., 0.)))
        area += abs(cross_sum.z) / 2.0
    return area


def wall_lines(faces: list[dict]) -> list[dict]:
    """Group vertical faces into wall lines, each with its extent along itself and its height.

    Only vertical faces: the wall mesh also carries the horizontal soffit and lintel surfaces of
    the openings, and a horizontal face has no place on a wall line.
    """
    import bpy
    from mathutils import Vector
    by_normal: dict[tuple, list[dict]] = defaultdict(list)
    for face in faces:
        if abs(face["normal"].z) > 1e-6:
            continue
        unit = canonical_plane(face["normal"])
        key = (round(unit.x, PLANE_KEY_DECIMALS), round(unit.y, PLANE_KEY_DECIMALS),
               round(unit.z, PLANE_KEY_DECIMALS))
        by_normal[key].append(face)

    lines = []
    for key, items in by_normal.items():
        normal = Vector(key)
        # The line's offset is `normal . p`, taken with the canonical normal and *not* signed by
        # which way the face points: both faces of one line have to land on the same number, and
        # `normal * offset` is then a point on the line. Signing it put the office's east wall at
        # x = -4.4024 and made every distance to it read 8.8 m.
        items.sort(key=lambda face: normal.dot(face["centre"]))
        # Merged against the group's anchor rather than the previous face, so a wall whose faces
        # drift a little does not walk its origin away from where the wall actually is.
        groups: list[tuple[float, list[dict]]] = []
        for face in items:
            offset = normal.dot(face["centre"])
            if groups and abs(offset - groups[-1][0]) <= PLANE_MERGE_TOLERANCE_M:
                groups[-1][1].append(face)
            else:
                groups.append((offset, [face]))

        for anchor, group in groups:
            along = Vector((-normal.y, normal.x, 0.0))
            origin = normal * anchor
            offsets = [(point - origin).dot(along) for face in group for point in face["points"]]
            heights = [point.z for face in group for point in face["points"]]
            low, high = min(offsets), max(offsets)
            lines.append({
                "normal": normal,
                "along": along,
                "origin": origin,
                "point": origin + along * low,
                "end": origin + along * high,
                "offset_of_line": round(anchor, 4),
                "z_min": min(heights),
                "z_max": max(heights),
                "faces": group,
                "face_count": len(group),
            })
    return lines


def split_features(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """(walls, features): vertical lines that reach the ceiling, and those that do not.

    A doorway's side and a window's reveal are vertical planes with horizontal normals, so nothing
    about a single face says they are not walls. What separates them is height: a wall runs the
    room's full interior height, a reveal spans one opening. Both are reported -- the reveals are
    where the wall's depth is read from -- but only the walls are matched against the plan.
    """
    import bpy
    from mathutils import Vector
    if not lines:
        return [], []
    tallest = max(line["z_max"] - line["z_min"] for line in lines)
    walls = [
        line for line in lines
        if line["z_max"] - line["z_min"] >= tallest - FULL_HEIGHT_TOLERANCE_M
    ]
    wallset = {id(line) for line in walls}
    return walls, [line for line in lines if id(line) not in wallset]


def boundary_loops(obj) -> list[list[int]]:
    """The mesh's boundary loops, as lists of vertex indices.

    A hole in a surface is exactly where the surface stops: an edge used by one face and not two.
    That is an exact statement about the mesh, where sweeping heights and looking for missing
    coverage is a sampled approximation of one -- and the sweep proved it: it reported three
    openings on the office's south wall where there is one, because a strip of that wall's faces
    landed in a neighbouring plane group and read as a hole.
    """
    import bpy
    from mathutils import Vector
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for polygon in obj.data.polygons:
        vertices = list(polygon.vertices)
        for index in range(len(vertices)):
            first, second = vertices[index], vertices[(index + 1) % len(vertices)]
            shared[(min(first, second), max(first, second))] += 1

    adjacency: dict[int, list[int]] = defaultdict(list)
    for (first, second), count in shared.items():
        if count == 1:
            adjacency[first].append(second)
            adjacency[second].append(first)

    loops, visited = [], set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        loop, current, visited = [start], start, visited | {start}
        while True:
            following = next((v for v in adjacency[current] if v not in visited), None)
            if following is None:
                break
            loop.append(following)
            visited.add(following)
            current = following
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def opening_loops(obj, entries: list[dict]) -> tuple[dict[int, list[list[Vector]]], list[dict]]:
    """Boundary loops that lie on a wall plane, grouped by the wall they sit in.

    A loop counts only when *every* point of it is on the wall's line: that is what separates a
    hole in a wall from the outline of one of the wall's own panels.
    """
    import bpy
    from mathutils import Vector
    matrix = obj.matrix_world
    vertices = [matrix @ vertex.co for vertex in obj.data.vertices]
    loops = boundary_loops(obj)
    diagnostics: list[dict] = [{"total_loops": len(loops)}]
    by_wall: dict[int, list[list[Vector]]] = defaultdict(list)
    for loop in loops:
        points = [vertices[index] for index in loop]
        diagnostics.append({
            "points": len(loop),
            "z_span": [round(min(p.z for p in points), 3), round(max(p.z for p in points), 3)],
            "xy_span": [round(max(p.x for p in points) - min(p.x for p in points), 3),
                        round(max(p.y for p in points) - min(p.y for p in points), 3)],
        })
        heights = [point.z for point in points]
        if max(heights) - min(heights) < EDGE_TOLERANCE_M:
            continue  # horizontal, so it is the room's floor or ceiling edge, not an opening
        # Matched on the plane's *direction* and the wall's extent, not on its offset. An opening
        # loop sits on the wall's outer face -- the far end of the reveal tunnel -- while the
        # surface this mesh carries is the inner one half a wall-depth away, so an offset test
        # rejects exactly the loops it is looking for.
        loop_normal = newell_normal(points)
        centre = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
        candidates = []
        for entry in entries:
            if abs(loop_normal.dot(entry["normal"])) < 1 - PARALLEL_TOLERANCE:
                continue
            along = entry["along"]
            extents = [(point - entry["origin"]).dot(along) for point in points]
            wall_low, wall_high = sorted(
                ((corner - entry["origin"]).dot(along) for corner in (entry["point"], entry["end"]))
            )
            if min(extents) < wall_low - EDGE_TOLERANCE_M or max(extents) > wall_high + EDGE_TOLERANCE_M:
                continue
            # Distance decides when two walls share a direction: the office's north and south walls
            # are both (0, +-1) and each spans the room, so containment alone lets either answer for
            # the other's openings.
            candidates.append((abs(entry["normal"].dot(centre - entry["origin"])), entry))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            by_wall[id(candidates[0][1])].append(points)
        else:
            diagnostics[-1]["unassigned"] = True
    return by_wall, diagnostics


def newell_normal(points: list[Vector]) -> Vector:
    """A polygon's normal, robust for a loop that is not exactly planar."""
    import bpy
    from mathutils import Vector
    normal = Vector((0.0, 0.0, 0.0))
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        normal.x += (point.y - following.y) * (point.z + following.z)
        normal.y += (point.z - following.z) * (point.x + following.x)
        normal.z += (point.x - following.x) * (point.y + following.y)
    return normal.normalized() if normal.length else normal


def openings_on(entry: dict, loops: list[list[Vector]]) -> list[dict]:
    """This wall's holes, as an offset span and a z span each.

    Each boundary loop is one opening, read straight off its vertices. A door is told from a window
    by where its span starts: a door reaches the wall's bottom, a window starts at its sill, which
    is why the sill is measurable here at all.
    """
    import bpy
    from mathutils import Vector
    along = entry["along"]
    # Each loop is one hole. Its extent along the wall and its z range are read straight off the
    # loop's vertices -- no grouping heuristic, no sweep step to be wrong about.
    found = []
    for points in loops:
        offsets = [(point - entry["origin"]).dot(along) for point in points]
        heights = [point.z for point in points]
        low, high = min(offsets), max(offsets)
        centre = entry["origin"] + along * ((low + high) / 2)
        found.append({
            # Both the wall's own frame (for comparing against the wall) and the room's (for
            # comparing against the plan). Emitting only the former made every reader redo this
            # arithmetic, and get it wrong the first time.
            "offset_low": round(low, 4),
            "offset_high": round(high, 4),
            "centre_xy": [round(centre.x, 4), round(centre.y, 4)],
            "z_min": round(min(heights), 4),
            "z_max": round(max(heights), 4),
        })
    found.sort(key=lambda opening: opening["offset_low"])
    return found


def line_distance(entry: dict, segment) -> float:
    """Perpendicular distance from a plan edge to a measured wall line."""
    import bpy
    from mathutils import Vector
    (ax, ay), (bx, by) = segment
    normal = entry["normal"]
    return abs(normal.x * (ax - entry["origin"].x) + normal.y * (ay - entry["origin"].y))


def direction(segment) -> Vector:
    import bpy
    from mathutils import Vector
    (ax, ay), (bx, by) = segment
    length = math.hypot(bx - ax, by - ay)
    return Vector(((bx - ax) / length, (by - ay) / length, 0.0)) if length else Vector((0, 0, 0))


def measure(plan: dict, room_name: str, report: dict) -> None:
    import bpy
    from mathutils import Vector
    objects = {obj.name: obj for obj in bpy.data.objects}
    if room_name not in objects:
        fail("SHELL_WALL_MESH_NOT_FOUND", f"no object named {room_name!r}")

    wall_object = objects.get(f"{room_name}.wall")
    if wall_object is None:
        fail("SHELL_WALL_MESH_NOT_FOUND", f"no object named {room_name}.wall")

    faces = mesh_faces(wall_object)
    everything = wall_lines(faces)
    entries, features = split_features(everything)
    loops_by_wall, loop_diagnostics = opening_loops(wall_object, entries)
    report["boundary_loops"] = loop_diagnostics
    report["wall_face_count"] = len(faces)
    report["wall_line_count"] = len(entries)
    report["feature_line_count"] = len(features)
    if features:
        # The reveals are the wall seen end-on: each spans the wall's depth, which is the only way
        # to read that depth at all here, because `.wall` carries interior surfaces only and the
        # wall's two faces are therefore not both present to be subtracted.
        lengths = sorted(round((line["end"] - line["point"]).length, 4) for line in features)
        report["measurements"]["wall.depth_m"] = lengths[len(lengths) // 2]
    # Every line, not only the matched ones: a wall the matcher did not pick is how you find out
    # that the matcher is wrong rather than the wall.
    report["all_wall_lines"] = [
        {
            "offset_m": entry["offset_of_line"],
            "normal": [round(entry["normal"].x, 3), round(entry["normal"].y, 3)],
            "point": [round(entry["point"].x, 4), round(entry["point"].y, 4)],
            "end": [round(entry["end"].x, 4), round(entry["end"].y, 4)],
            "length_m": round((entry["end"] - entry["point"]).length, 4),
            "z_span": [round(entry["z_min"], 4), round(entry["z_max"], 4)],
            "faces": entry["face_count"],
            "face_centres": [
                [round(face["centre"].x, 3), round(face["centre"].y, 3), round(face["centre"].z, 3)]
                for face in entry["faces"]
            ],
        }
        for entry in entries
    ]

    floor_obj = objects.get(f"{room_name}.floor")
    if floor_obj is None:
        report["not_measured"].append("room.floor_area_m2: floor mesh missing")
    else:
        report["measurements"]["room.floor_area_m2"] = round(floor_area_m2(floor_obj), 6)

    room_obj = objects[room_name]
    corners = [room_obj.matrix_world @ Vector(corner) for corner in room_obj.bound_box]
    report["measurements"]["room.shell_bbox_m"] = {
        "min": [round(min(c[i] for c in corners), 4) for i in range(3)],
        "max": [round(max(c[i] for c in corners), 4) for i in range(3)],
    }

    room = plan["rooms"][0]
    edges = [(room["polygon_xy"][i], room["polygon_xy"][(i + 1) % len(room["polygon_xy"])])
             for i in range(len(room["polygon_xy"]))]

    # Match plan edges to measured wall lines, one to one, scored on distance **and** length.
    #
    # Both terms are load-bearing. The office probe has a stray 0.9 m vertical line 0.05 m outside
    # its south wall -- the same distance from that plan edge as the real 3.49 m wall -- so a
    # distance-only matcher picks the fragment and then leaves two other edges unmatched. Length
    # agreement is what tells a wall from a piece of one. Assigning greedily from the best score
    # also keeps one measured wall from answering for two plan edges.
    scored = []
    for index, edge in enumerate(edges):
        want = direction(edge)
        planned = math.dist(*edge)
        for entry in entries:
            if abs(want.dot(entry["along"])) < 1 - PARALLEL_TOLERANCE:
                continue
            distance = line_distance(entry, edge)
            built = (entry["end"] - entry["point"]).length
            scored.append((distance + abs(built - planned), index, entry, distance))
    scored.sort(key=lambda item: item[0])

    matched: list[tuple[int, dict, float]] = []
    taken_edges: set[int] = set()
    taken_lines: set[int] = set()
    for _, index, entry, distance in scored:
        if index in taken_edges or id(entry) in taken_lines:
            continue
        taken_edges.add(index)
        taken_lines.add(id(entry))
        matched.append((index, entry, distance))
    matched.sort(key=lambda item: item[0])

    for index in range(len(edges)):
        if index not in taken_edges:
            report["not_measured"].append(
                f"wall[{index}]: no unmatched measured wall line parallel to this plan edge"
            )

    residuals = [round(distance, 4) for _, _, distance in matched]
    report["walls"] = []
    for index, entry, distance in matched:
        openings = openings_on(entry, loops_by_wall.get(id(entry), []))
        report["walls"].append({
            "plan_edge": index,
            "distance_from_plan_edge_m": round(distance, 4),
            "line_offset_m": entry["offset_of_line"],
            "point": [round(entry["point"].x, 4), round(entry["point"].y, 4)],
            "end": [round(entry["end"].x, 4), round(entry["end"].y, 4)],
            "length_m": round((entry["end"] - entry["point"]).length, 4),
            "z_min": round(entry["z_min"], 4),
            "z_max": round(entry["z_max"], 4),
            "face_count": entry["face_count"],
            "openings": openings,
        })

    if residuals:
        report["measurements"]["room.wall_line_offset_m"] = round(
            sorted(residuals)[len(residuals) // 2], 4
        )
        # The shape measure: a constant offset is a datum, a spread is a disagreement.
        report["measurements"]["room.wall_line_offset_spread_m"] = round(
            max(residuals) - min(residuals), 4
        )
    if report["walls"]:
        # The interior floor level, so a reader can turn a hole's `z_min` into a sill height. It is
        # not zero: Infinigen puts the room's floor surface half a slab above the plan's datum.
        report["floor_z_m"] = min(wall["z_min"] for wall in report["walls"])
        report["measurements"]["room.interior_z_span_m"] = [
            min(wall["z_min"] for wall in report["walls"]),
            max(wall["z_max"] for wall in report["walls"]),
        ]
        report["measurements"]["openings.count"] = sum(
            len(wall["openings"]) for wall in report["walls"]
        )
        measured_count = report["measurements"]["openings.count"]
        planned_count = len(plan.get("openings") or [])
        report["measurements"]["openings.plan_count"] = planned_count
        if measured_count != planned_count:
            report["not_measured"].append(
                f"openings.count: the plan names {planned_count} and the mesh shows "
                f"{measured_count}; per-opening identity needs them to correspond"
            )


def deviations(plan: dict, report: dict) -> list[dict]:
    """What the plan asked for against what was measured, item by item.

    Only quantities with a counterpart on both sides appear. A plan field the mesh cannot speak to
    is left to ``not_measured`` -- an absent row is honest, a row with a guessed number is not.
    """
    import bpy
    from mathutils import Vector
    room = plan["rooms"][0]
    measurements = report["measurements"]
    out: list[dict] = []

    bbox = measurements.get("room.shell_bbox_m")
    if bbox:
        built_height = round(bbox["max"][2] - bbox["min"][2], 4)
        measurements["room.height_m"] = built_height
        out.append({
            "field": "room.ceiling_height_m",
            "plan": room["ceiling_height_m"],
            "built": built_height,
            "delta": round(built_height - room["ceiling_height_m"], 4),
        })

    if "room.wall_line_offset_spread_m" in measurements:
        out.append({
            "field": "room.wall_line_offset_spread_m",
            "plan": 0.0,
            "built": measurements["room.wall_line_offset_spread_m"],
            "delta": measurements["room.wall_line_offset_spread_m"],
        })
        # The matched wall lengths, longest first: a plan edge the builder shortened shows up here
        # even when its line is in the right place.
        planned_lengths = sorted(
            round(math.dist(*edge), 4)
            for edge in [(room["polygon_xy"][i], room["polygon_xy"][(i + 1) % len(room["polygon_xy"])])
                         for i in range(len(room["polygon_xy"]))]
        )
        built_lengths = sorted(wall["length_m"] for wall in report["walls"])
        measurements["room.planned_wall_lengths_m"] = planned_lengths
        measurements["room.built_wall_lengths_m"] = built_lengths
        if len(planned_lengths) == len(built_lengths):
            out.append({
                "field": "room.wall_length_max_delta_m",
                "plan": planned_lengths,
                "built": built_lengths,
                "delta": round(
                    max(abs(a - b) for a, b in zip(planned_lengths, built_lengths)), 4
                ),
            })
    return out


def main() -> int:
    import bpy
    from mathutils import Vector
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True, help="a pvscene.room-plan/v1 JSON file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--room-name", default=None,
                        help="the Blender object prefix; defaults to '<kind>_0/0' from the plan")
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    room = plan["rooms"][0]
    room_name = args.room_name or f"{room['kind']}_0/0"

    report: dict = {
        "plan_id": plan.get("plan_id"),
        "plan_hash": plan.get("hash"),
        "room_name": room_name,
        "measurements": {},
        "not_measured": [],
        "walls": [],
    }
    measure(plan, room_name, report)
    report["plan_room"] = {
        "polygon_xy": room["polygon_xy"],
        "ceiling_height_m": room["ceiling_height_m"],
    }
    report["deviations"] = deviations(plan, report)

    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"MEASURE wrote {args.out}")
    print(json.dumps({
        "measurements": report["measurements"],
        "not_measured": report["not_measured"],
        "wall_lines": report["wall_line_count"],
        "walls_matched": len(report["walls"]),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
