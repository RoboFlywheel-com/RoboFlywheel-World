#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render paper overview cameras from one existing, immutable Isaac delivery.

This is a figure-only render: it copies the worker request into a fresh output directory,
changes only the camera set, and keeps the source scene version and geometry bindings.
It never edits the delivery or upgrades its readiness.
"""

from __future__ import annotations
from rfw_w.scenes.visualization.inputs import resolve_input

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path.cwd()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def point_inside(polygon: list[list[float]], x: float, y: float) -> bool:
    """Ray-casting containment test, local to this script so the figure entry
    stays free of the package (this repo's other figure scripts do the same).

    It exists because assuming a rectangular room is wrong: measured on bedroom
    seed 37 the floor plan is L-shaped with a notch at the origin, so the inset
    of the minimum corner lands **outside** the room and the camera renders the
    outside of a wall -- a panel of flat white with no furniture in it.
    """
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index][0], polygon[index][1]
        x2, y2 = polygon[(index + 1) % count][0], polygon[(index + 1) % count][1]
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def choose_vantage(polygon: list[list[float]], height: float, requested: str) -> tuple[list[float], str]:
    """The first corner, in a fixed order, whose inset point is inside the room.

    Preference starts at the requested corner so an existing figure keeps its
    framing when that corner is usable. A corner that is not inside the room
    cannot be a camera position: from there the near wall is between the lens and
    everything the figure is meant to show.
    """
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    order = [requested] + [corner for corner in ("sw", "se", "nw", "ne") if corner != requested]
    for corner in order:
        if corner not in ("sw", "se", "nw", "ne"):
            raise ValueError(f"PAPER_PREVIEW_CAMERA_CORNER_INVALID({corner})")
        x = min_x + 0.34 if "w" in corner else max_x - 0.34
        y = min_y + 0.34 if "s" in corner else max_y - 0.34
        if point_inside(polygon, x, y):
            return [x, y, min(height - 0.22, 2.65)], corner
    raise ValueError("PAPER_PREVIEW_VANTAGE_OUTSIDE_ROOM(no inset corner lies inside the plan)")


# ---------------------------------------------------------------------------
# Dollhouse view (SPEC-013 revision: the eye-level preview was rejected because
# the near wall occluded half the room and the calibrated intensities clipped
# the white walls; the dollhouse looks down into a ceiling-less shell).
# ---------------------------------------------------------------------------

# Eye height as a multiple of room height. 2.2x keeps the camera clearly above
# the wall tops so the roofless shell never blocks the view of the floor; the
# horizontal distance is then solved from the requested elevation.
DOLLHOUSE_HEIGHT_FACTOR = 2.2
# View-direction elevation above horizontal. 52 deg is steep enough that the
# floor plan and furniture layout read at a glance (the first dollhouse renders
# at ~44 deg still let the near-wall stubs hide floor area), shallow enough that
# furniture keeps visible height -- a true top-down view flattens the room into
# a map.
DOLLHOUSE_ELEVATION_DEG = 52.0
# The footprint is fitted to fill the frame, then the focal length is divided by
# this margin so the skirting and furniture that sit just outside the floor
# polygon stay inside the frame edge. 1.05, tightened from 1.10: the first
# dollhouse renders left too much sky in frame (depth-valid fraction 0.35-0.41).
DOLLHOUSE_MARGIN = 1.05
# Aim slightly above the floor -- roughly small-object tabletop height -- so the
# frame centres on the furnished volume rather than on the floor plane.
DOLLHOUSE_TARGET_Z_M = 0.4


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _unit(v):
    length = math.sqrt(_dot(v, v))
    return [v[0] / length, v[1] / length, v[2] / length]


def dollhouse_corner_for(room: dict, requested: str = "auto") -> str:
    """Which corner the dollhouse camera looks from.

    ``auto`` picks the corner diagonally opposite the main (widest) door: the
    door wall is the one a reader already has a mental image of, and looking
    from the far corner keeps the doorway inside the frame instead of behind the
    camera. Without any door opening the fallback is SE, matching the other
    figure cameras' default corner.
    """
    if requested != "auto":
        if requested not in ("ne", "nw", "se", "sw"):
            raise ValueError(f"DOLLHOUSE_CORNER_INVALID({requested})")
        return requested
    polygon = room["polygon_xy"]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    centre_x = (min(xs) + max(xs)) / 2.0
    centre_y = (min(ys) + max(ys)) / 2.0
    doors = [o for o in room.get("openings", []) if o.get("kind") == "door"]
    if not doors:
        return "se"
    door = max(doors, key=lambda o: float(o.get("width_m", 0.0)))
    door_x, door_y = door["center_xy"]
    x_side = "w" if door_x > centre_x else "e"
    y_side = "s" if door_y > centre_y else "n"
    return y_side + x_side


def fit_focal_length(eye, target, points, aperture_mm: float, width: int, height: int,
                     margin: float = DOLLHOUSE_MARGIN) -> float:
    """Longest focal length that still frames every point, divided by ``margin``.

    The renderer ignores the declared vertical aperture and derives it from the
    aspect ratio (square pixels), so both frame directions are fitted here from
    the horizontal aperture alone. ``points`` are world points that must land
    inside the frame.
    """
    forward = _unit(_sub(target, eye))
    right = _unit(_cross(forward, [0.0, 0.0, 1.0]))
    up = _cross(right, forward)
    vertical_aperture = aperture_mm * height / width
    need_h = 0.0
    need_v = 0.0
    for point in points:
        v = _sub(point, eye)
        z = _dot(v, forward)
        if z <= 0:
            raise ValueError("DOLLHOUSE_POINT_BEHIND_CAMERA")
        need_h = max(need_h, abs(_dot(v, right)) / z)
        need_v = max(need_v, abs(_dot(v, up)) / z)
    return min(aperture_mm / (2.0 * need_h), vertical_aperture / (2.0 * need_v)) / margin


def dollhouse_camera(room: dict, corner: str = "auto",
                     elevation_deg: float = DOLLHOUSE_ELEVATION_DEG,
                     aperture_mm: float = 20.955,
                     width: int = 1600, height: int = 1200,
                     margin: float = DOLLHOUSE_MARGIN,
                     fit_wall_tops: bool = True) -> dict:
    """A slightly top-down oblique camera framing the whole room from outside.

    The eye's height is 2.2x the room height; the horizontal distance is solved
    from ``elevation_deg`` so the view direction sits at exactly that angle
    (raising the elevation moves the camera inward and over the room rather
    than just tilting it). The focal length is auto-fitted to the room
    footprint (plus the wall tops, whose far side is what the frame must not
    clip once the ceiling is gone). ``fit_wall_tops=False`` fits the floor
    footprint only: near a straight top view the full-height far walls lean so
    far outward in projection that fitting them shrinks the room to the middle
    of the frame (measured on the 86-degree office render: 78% of the pixels
    were background), and the walls it protects are exactly the ones
    ``--wall-clip-corner all`` has already cut down.
    """
    used = dollhouse_corner_for(room, corner)
    polygon = room["polygon_xy"]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    centre = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0]
    floor_z = float(room.get("floor_z_m", 0.0))
    room_h = float(room["height_m"])
    diagonal = math.hypot(max_x - min_x, max_y - min_y)
    sign_x = 1.0 if "e" in used else -1.0
    sign_y = 1.0 if "n" in used else -1.0
    eye_z = floor_z + room_h * DOLLHOUSE_HEIGHT_FACTOR
    target = [centre[0], centre[1], floor_z + DOLLHOUSE_TARGET_Z_M]
    distance = (eye_z - target[2]) / math.tan(math.radians(elevation_deg))
    eye = [centre[0] + sign_x * distance / math.sqrt(2.0),
           centre[1] + sign_y * distance / math.sqrt(2.0),
           eye_z]
    # Fit the footprint at floor level *and* at the wall tops: with the ceiling
    # gone the far wall's top edge is the highest silhouette the frame must keep.
    fit_zs = (floor_z, floor_z + room_h) if fit_wall_tops else (floor_z,)
    fit_points = [[x, y, z] for x, y in polygon for z in fit_zs]
    focal = fit_focal_length(eye, target, fit_points, aperture_mm, width, height, margin)
    horizontal = math.hypot(eye[0] - target[0], eye[1] - target[1])
    return {
        "eye": eye,
        "target": target,
        "focal_length_mm": focal,
        "horizontal_aperture_mm": aperture_mm,
        "vertical_aperture_mm": aperture_mm * height / width,
        "resolution": [width, height],
        "corner": used,
        "elevation_deg": math.degrees(math.atan2(eye[2] - target[2], horizontal)),
        "eye_distance_m": distance,
        "room_diagonal_m": diagonal,
        "fit_wall_tops": fit_wall_tops,
    }


# ---------------------------------------------------------------------------
# Manual layout overrides, applied to the copied request only (never to the
# delivery). Every refusal is a named gap; every applied change is recorded.
# ---------------------------------------------------------------------------

# Sub-millimetre XY overlap is authoring rounding (a rug under a desk touches its
# AABB); a real collision between two furniture footprints is centimetres.
LAYOUT_OVERLAP_EPS_M = 1e-3
# A moved instance may not cover a doorway's passage: the scene claims the
# opening is a passage, so the figure must not park furniture in it. 0.3 m is
# about half a doorway's walk width -- loose enough that wall-thickness authoring
# noise at the opening edge does not false-fire.
OPENING_CLEARANCE_M = 0.3


def yaw_of(rotation_wxyz) -> float | None:
    """Yaw in degrees of a wxyz quaternion, or None when it is not z-aligned."""
    w, x, y, z = (float(v) for v in rotation_wxyz)
    if abs(x) > 1e-6 or abs(y) > 1e-6:
        return None
    return math.degrees(2.0 * math.atan2(z, w))


def yaw_quat(yaw_deg: float) -> list[float]:
    half = math.radians(yaw_deg) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def local_footprint(instance: dict, bounds) -> tuple[list[float], list[float], float | None]:
    """XY footprint in the instance's own frame: (offset, half, yaw).

    The scene-IR bounds are the AABB in the instance's **local** frame, not a
    world AABB -- measured on the office delivery: the desk's bounds are centred
    on the origin while its translation is not, and the storage shelf's local
    bounds only fit inside the room once rotated by its 90-degree yaw. They are
    therefore used directly, and rotated by the (possibly overridden) yaw when
    the world AABB is computed.
    """
    bmin, bmax = bounds
    return ([(float(bmin[0]) + float(bmax[0])) / 2.0, (float(bmin[1]) + float(bmax[1])) / 2.0],
            [(float(bmax[0]) - float(bmin[0])) / 2.0, (float(bmax[1]) - float(bmin[1])) / 2.0],
            yaw_of(instance["rotation_wxyz"]))


def world_aabb(offset, half, translation, yaw_deg: float | None) -> tuple[list[float], list[float]]:
    rad = math.radians(yaw_deg or 0.0)
    c, s = math.cos(rad), math.sin(rad)
    corners = [(offset[0] + dx * half[0], offset[1] + dy * half[1])
               for dx in (-1.0, 1.0) for dy in (-1.0, 1.0)]
    rotated = [(c * px - s * py + translation[0], s * px + c * py + translation[1])
               for px, py in corners]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    return [min(xs), min(ys)], [max(xs), max(ys)]


def opening_strip(polygon, opening) -> tuple[list[float], list[float], list[float]]:
    """The forbidden strip in front of an opening, in the opening's wall frame.

    Returns (origin_xy, along_unit, inward_unit): the nearest polygon edge is
    the opening's wall (Infinigen walls are authored with their inner surface on
    the plan boundary, so the opening centre sits on an edge), and the inward
    normal points at the polygon centroid.
    """
    cx, cy = (float(v) for v in opening["center_xy"])
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    centroid = [sum(xs) / len(xs), sum(ys) / len(ys)]
    best = None
    count = len(polygon)
    for index in range(count):
        x1, y1 = (float(v) for v in polygon[index])
        x2, y2 = (float(v) for v in polygon[(index + 1) % count])
        ex, ey = x2 - x1, y2 - y1
        length = math.hypot(ex, ey)
        if length == 0:
            continue
        ex, ey = ex / length, ey / length
        px, py = cx - x1, cy - y1
        along = max(0.0, min(length, px * ex + py * ey))
        qx, qy = x1 + ex * along, y1 + ey * along
        d = math.hypot(cx - qx, cy - qy)
        if best is None or d < best[0]:
            best = (d, [ex, ey])
    if best is None:
        raise ValueError(f"LAYOUT_OPENING_EDGE_MISSING({opening.get('opening_id')})")
    along_unit = best[1]
    inward = [-along_unit[1], along_unit[0]]
    if ((centroid[0] - cx) * inward[0] + (centroid[1] - cy) * inward[1]) < 0:
        inward = [-inward[0], -inward[1]]
    return [cx, cy], along_unit, inward


def aabb_blocks_opening(aabb_min, aabb_max, polygon, opening) -> bool:
    origin, along_unit, inward = opening_strip(polygon, opening)
    centre = [(aabb_min[0] + aabb_max[0]) / 2.0 - origin[0],
              (aabb_min[1] + aabb_max[1]) / 2.0 - origin[1]]
    half = [(aabb_max[0] - aabb_min[0]) / 2.0, (aabb_max[1] - aabb_min[1]) / 2.0]
    # Rotate the AABB into the wall frame; the |R| h form is exact for an AABB.
    along_c = centre[0] * along_unit[0] + centre[1] * along_unit[1]
    perp_c = centre[0] * inward[0] + centre[1] * inward[1]
    along_h = half[0] * abs(along_unit[0]) + half[1] * abs(along_unit[1])
    perp_h = half[0] * abs(inward[0]) + half[1] * abs(inward[1])
    half_width = float(opening["width_m"]) / 2.0 + OPENING_CLEARANCE_M
    along_overlap = abs(along_c) < half_width + along_h
    # The strip runs from just outside the wall line to the clearance depth
    # inside; the small outward margin covers wall-thickness authoring noise.
    perp_overlap = (perp_c + perp_h > -0.05) and (perp_c - perp_h < OPENING_CLEARANCE_M)
    return along_overlap and perp_overlap


def apply_layout_overrides(request: dict, overrides: dict, *, bounds_by_instance: dict,
                           protected_ids: set) -> list[dict]:
    """Apply manual pose edits to ``request["instances"]``; return the delta records.

    Validation happens entirely before application, so a refused override leaves
    the request untouched. Bounds come from the delivery's scene_ir (local-frame
    AABBs); an instance the scene_ir does not describe cannot be moved, because
    nothing could then check where it went.

    An override may declare ``"on": "<instance_id>"`` -- the instance rests on
    that support (an alarm clock on a nightstand). Their AABBs legitimately
    graze at the support surface, so a declared pair is exempt from
    LAYOUT_OVERLAP; the declaration is recorded in the delta so the exemption is
    visible in provenance rather than silent.
    """
    instances = request["instances"]
    by_id = {i["instance_id"]: i for i in instances}
    for instance_id in overrides:
        if instance_id not in by_id:
            raise ValueError(f"LAYOUT_UNKNOWN_INSTANCE({instance_id})")
    polygon = request["room"]["polygon_xy"]
    openings = request["room"].get("openings", [])
    new_pose: dict[str, tuple[list[float], float | None]] = {}
    removed: set[str] = set()
    moved_ids: set[str] = set()
    support_of: dict[str, str] = {}
    deltas: list[dict] = []
    for instance_id, change in overrides.items():
        instance = by_id[instance_id]
        if change.get("remove"):
            if instance_id in protected_ids or instance.get("entity_id") in protected_ids:
                raise ValueError(f"LAYOUT_REMOVES_TARGET({instance_id})")
            removed.add(instance_id)
            deltas.append({"instance_id": instance_id,
                           "entity_id": instance.get("entity_id"), "removed": True})
            continue
        support = change.get("on")
        if support is not None:
            if support not in by_id:
                raise ValueError(f"LAYOUT_UNKNOWN_INSTANCE({support})")
            support_of[instance_id] = support
        translation = [float(v) for v in change.get("translation", instance["translation"])]
        yaw = change.get("yaw_deg")
        current_yaw = yaw_of(instance["rotation_wxyz"])
        if yaw is not None and current_yaw is None:
            raise ValueError(f"LAYOUT_YAW_NOT_Z_ALIGNED({instance_id})")
        moved = translation != [float(v) for v in instance["translation"]] or yaw is not None
        if moved and instance_id not in bounds_by_instance:
            raise ValueError(f"LAYOUT_BOUNDS_MISSING({instance_id})")
        new_pose[instance_id] = (translation, float(yaw) if yaw is not None else current_yaw)
        if moved:
            moved_ids.add(instance_id)
    # Declared support pairs are exempt from the overlap gate in both directions:
    # moving the item onto its support and moving the support under its item are
    # the same edit.
    exempt_pairs = {frozenset((item, support)) for item, support in support_of.items()}
    # Footprints of every kept instance at its final pose, for the spatial checks.
    footprints: dict[str, tuple[list[float], list[float], float, float]] = {}
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id in removed or instance_id not in bounds_by_instance:
            continue
        translation, yaw = new_pose.get(
            instance_id, ([float(v) for v in instance["translation"]],
                          yaw_of(instance["rotation_wxyz"])))
        bmin, bmax = bounds_by_instance[instance_id]
        offset, half, _ = local_footprint(instance, (bmin, bmax))
        aabb_min, aabb_max = world_aabb(offset, half, translation, yaw)
        footprints[instance_id] = (aabb_min, aabb_max,
                                   translation[2] + float(bmin[2]),
                                   translation[2] + float(bmax[2]))
    for instance_id in new_pose:
        if instance_id in removed or instance_id not in moved_ids:
            # The spatial gates judge moves; a declaration-only override ("on"
            # with no pose change) must not re-judge the authored pose.
            continue
        aabb_min, aabb_max, z_min, z_max = footprints[instance_id]
        centre = [(aabb_min[0] + aabb_max[0]) / 2.0, (aabb_min[1] + aabb_max[1]) / 2.0]
        if not point_inside(polygon, centre[0], centre[1]):
            raise ValueError(
                f"LAYOUT_OUTSIDE_ROOM({instance_id}, centre=({centre[0]:.3f},{centre[1]:.3f}))")
        for opening in openings:
            if instance_id in support_of:
                # Same asymmetry the additions path carries: the opening rule is
                # about passage, so it judges what stands on the floor. An item
                # declared as resting on a support is placed relative to that
                # support, and where the support stands was the delivery's
                # decision -- measured on the office, the desk is against the
                # window wall, so judging a book on its top by the window strip
                # refuses every book.
                break
            if aabb_blocks_opening(aabb_min, aabb_max, polygon, opening):
                raise ValueError(
                    f"LAYOUT_BLOCKS_OPENING({instance_id}, {opening.get('opening_id')})")
        for other_id, (omin, omax, oz_min, oz_max) in footprints.items():
            if other_id == instance_id:
                continue
            if frozenset((instance_id, other_id)) in exempt_pairs:
                continue
            overlap_x = min(aabb_max[0], omax[0]) - max(aabb_min[0], omin[0])
            overlap_y = min(aabb_max[1], omax[1]) - max(aabb_min[1], omin[1])
            if (overlap_x > LAYOUT_OVERLAP_EPS_M and overlap_y > LAYOUT_OVERLAP_EPS_M
                    and z_max > oz_min and oz_max > z_min):
                raise ValueError(f"LAYOUT_OVERLAP({instance_id}, {other_id})")
    for instance_id, (translation, yaw) in new_pose.items():
        if instance_id in removed:
            continue
        instance = by_id[instance_id]
        delta = {"instance_id": instance_id, "entity_id": instance.get("entity_id")}
        if instance_id in support_of:
            delta["on"] = support_of[instance_id]
        old_translation = [float(v) for v in instance["translation"]]
        if translation != old_translation:
            delta["translation"] = {"from": old_translation, "to": translation}
            instance["translation"] = list(translation)
        old_yaw = yaw_of(instance["rotation_wxyz"])
        if yaw is not None and (old_yaw is None or abs(yaw - old_yaw) > 1e-9):
            delta["yaw_deg"] = {"from": old_yaw, "to": yaw}
            instance["rotation_wxyz"] = yaw_quat(yaw)
        if len(delta) > 2:
            deltas.append(delta)
    if removed:
        request["instances"] = [i for i in instances if i["instance_id"] not in removed]
    return deltas


# ---------------------------------------------------------------------------
# Additions (--additions): robots and extra props put **into** the copied
# request. The layout overrides above can move and remove what the delivery
# placed; nothing could add to it, which is why every paper figure so far shows
# a task with no machine to do it and a work surface with three objects on it.
# ---------------------------------------------------------------------------

# An added item rests on a surface: its lowest point sits on the support's top,
# not inside it and not floating above it. 5 mm is authoring rounding on a
# measured AABB; more than that is a mistake worth naming.
ADDITION_SURFACE_EPS_M = 5e-3

# How far an item may reach past the edge of what it rests on. A book laid on a
# desk really does overhang a couple of centimetres; a robot base that hangs
# 0.10 m off the edge is a mistake the picture shows and no other rule catches.
ADDITION_OVERHANG_M = 0.05

# The height band, above a robot's lowest point, that counts as what it stands
# on -- the same 0.30 m `python -m rfw_w.scenes.visualization.measure_robot_profile` measures the
# profile footprint over, so the placement rule and the profile agree on what a
# robot's footprint is.
ROBOT_SUPPORT_BAND_M = 0.30

# Which measured profile each baked robot is judged against. The mapping is here
# rather than in the declaration so a figure cannot quietly judge a G1 by an
# arm's reach.
ROBOT_PROFILE_OF = {"piper": "piper_x", "g1": "unitree_g1"}


def obj_local_bounds(path: Path, band_m: float | None = None) -> tuple[list[float], list[float]]:
    """Axis-aligned bounds of an OBJ, from its vertex lines.

    Read directly rather than through a mesh library: the figure only needs the
    box, and the box is what the placement rules below test.

    ``band_m`` restricts the box to the vertices within that height of the
    lowest one -- the part that rests on the surface. A robot needs this: its
    whole-body box is a claim that the floor under its outstretched hands is
    occupied, which is how a G1 standing 0.4 m clear of an island still
    "overlaps" it, and how an arm reaching over a desk "overlaps" the mug it is
    reaching for.
    """
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    points: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            point = (float(parts[1]), float(parts[2]), float(parts[3]))
            if band_m is not None:
                points.append(point)
            for axis in range(3):
                lo[axis] = min(lo[axis], point[axis])
                hi[axis] = max(hi[axis], point[axis])
    if not all(math.isfinite(v) for v in lo):
        raise ValueError(f"ADDITION_MESH_EMPTY({path})")
    if band_m is None:
        return lo, hi
    ceiling = lo[2] + band_m
    banded = [p for p in points if p[2] <= ceiling]
    if not banded:
        raise ValueError(f"ADDITION_MESH_EMPTY({path}: nothing within {band_m} m of its base)")
    return ([min(p[axis] for p in banded) for axis in range(3)],
            [max(p[axis] for p in banded) for axis in range(3)])


def apply_normalisation(lo, hi, normalisation: str) -> tuple[list[float], list[float]]:
    """The same frame correction the worker applies, on a box.

    ``y_up_to_z_up`` is a +90 degree rotation about +X (``geometry.py``'s
    ``y_up_to_z_up_matrix``), i.e. (x, y, z) -> (x, -z, y). Skipping it here
    would check the placement of a box the worker never draws -- for a Y-up
    asset that is the difference between standing on the desk and sinking into
    it by its own depth.
    """
    if normalisation == "none":
        return list(lo), list(hi)
    if normalisation != "y_up_to_z_up":
        raise ValueError(f"ADDITION_NORMALISATION_UNKNOWN({normalisation})")
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    mapped = [(x, -z, y) for x, y, z in corners]
    return ([min(p[i] for p in mapped) for i in range(3)],
            [max(p[i] for p in mapped) for i in range(3)])


def expand_path(value: str) -> Path:
    """Resolve ``${VAR}`` in a declared asset path.

    Declarations name libraries by environment variable for the same reason the
    code does: an internal asset path must not enter the repository.
    """
    import os
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"ADDITION_PATH_UNRESOLVED({value})")
    return Path(expanded).expanduser()


def prepare_simready_prop(leaf: str, target_dir: Path) -> tuple[Path, Path, dict]:
    """Prepare one library asset through the **production** provider.

    Reusing ``assets/providers/simready.prepare_rigid`` rather than reading the
    release's OBJ directly is the whole point: v1.3 ships each asset in its own
    frame plus a "normalised" family that is not reliably upright, and the
    provider is where the project's answer to that lives (ADR / the library
    frame table in ``adapters/simready_retrieval.py``). A figure that loaded
    the raw mesh would be the third place to get the up-axis wrong.

    The prepared pair is already in the world frame, so the instance carries
    ``normalisation: none`` -- the same as every retrieved instance in a
    delivery.
    """
    import os
    from rfw_w.scenes.assets.providers.simready import SimreadyPrepareConfig, prepare_rigid

    index_path = os.environ.get("PVSCENE_SIMREADY_INDEX")
    assets_root = os.environ.get("PVSCENE_SIMREADY_ASSETS_ROOT")
    if not index_path or not assets_root:
        raise ValueError("ADDITION_LIBRARY_UNSET(PVSCENE_SIMREADY_INDEX/ASSETS_ROOT)")
    record = json.loads(Path(index_path).read_text()).get(leaf)
    if record is None:
        raise ValueError(f"ADDITION_LIBRARY_LEAF_UNKNOWN({leaf})")
    config = SimreadyPrepareConfig(assets_root=Path(assets_root), prepared_root=target_dir)
    prepared = prepare_rigid(config, prefix=record["assets_prefix"], leaf=leaf, asset_id=leaf)
    # Reported, not gated. The release's `real_bbox` is a coarse annotation whose axis order is not
    # consistent (the vase's reads x,y,z; the mug's does not match its own mesh), so this cannot
    # decide uprightness on its own -- but v1.3's per-asset up-axis is a real trap, and measured
    # 2026-09-26 it lays 10 of 24 ranked candidates on their side after the provider's single fix.
    # Recording both boxes is what let those ten be spotted and swapped before the render.
    prepared_dims = prepared.dimensions_m.to_list()
    real_bbox = record.get("real_bbox")
    upright = None
    if real_bbox:
        upright = {
            "prepared_height_m": round(float(prepared_dims[2]), 4),
            "library_real_bbox": real_bbox,
            "heights_agree_within_25pct": abs(prepared_dims[2] - real_bbox[2]) <= 0.25 * max(
                real_bbox[2], 1e-6),
            "note": "a disagreement means look at the asset, not that the figure is wrong",
        }
    return prepared.visual_obj, prepared.collision_obj, {
        "provider": "simready",
        "library_version": record.get("library_version"),
        "leaf": leaf,
        "super_category": record.get("super_category"),
        "index_dimensions_m": record.get("dimensions_m"),
        "prepared_dimensions_m": prepared.dimensions_m.to_list(),
        "upright_measurement": upright,
        "has_diffuse_texture": prepared.has_diffuse_texture,
        "visual_sha256": digest(prepared.visual_obj),
        "collision_sha256": digest(prepared.collision_obj),
        "prepared_by": "assets/providers/simready.prepare_rigid (the production path)",
    }


def apply_additions(request: dict, additions: list, *, bounds_by_instance: dict,
                    figure_asset_dir: Path, substituted_ids: set | None = None) -> list[dict]:
    """Add declared robots and props to ``request["instances"]``.

    Every item declares the surface it rests on and where on it; the height
    comes from the support's measured top, never from the declaration, so an
    item cannot be authored floating. Validation runs against the request's
    final state -- existing instances at their (possibly overridden) poses and
    the other additions -- and refuses before anything is appended.

    A robot additionally declares the task target it is placed for, and is
    checked against the **measured** reach of its profile
    (``geometry/robot.py``). That check is a placement proxy: it says the target
    is inside the envelope the arm can put its tool point in, and says nothing
    about a grasp existing. ``robot.ik`` stays not-run.
    """
    from rfw_w.scenes.geometry.robot import profile_for

    instances = request["instances"]
    by_id = {i["instance_id"]: i for i in instances}
    polygon = request["room"]["polygon_xy"]
    openings = request["room"].get("openings", [])
    floor_z = float(request["room"].get("floor_z_m", 0.0))

    # Footprints of everything already in the request, at its current pose.
    footprints: dict[str, tuple[list[float], list[float], float, float]] = {}
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id not in bounds_by_instance:
            continue
        bmin, bmax = bounds_by_instance[instance_id]
        offset, half, yaw = local_footprint(instance, (bmin, bmax))
        translation = [float(v) for v in instance["translation"]]
        aabb_min, aabb_max = world_aabb(offset, half, translation, yaw)
        footprints[instance_id] = (aabb_min, aabb_max,
                                   translation[2] + float(bmin[2]),
                                   translation[2] + float(bmax[2]))

    records: list[dict] = []
    staged: list[dict] = []
    for item in additions:
        item_id = str(item["id"])
        if item_id in by_id or any(s["instance"]["instance_id"] == item_id for s in staged):
            raise ValueError(f"ADDITION_DUPLICATE_ID({item_id})")
        kind = item.get("kind", "prop")
        normalisation = str(item.get("normalisation", "none"))

        if kind == "robot":
            robot = str(item["robot"])
            from rfw_w.scenes.visualization.robot_asset import ROBOTS, bake
            if robot not in ROBOTS:
                raise ValueError(f"ADDITION_ROBOT_UNKNOWN({robot})")
            pose_path = resolve_input(item["pose"])
            pose = json.loads(resolve_input(pose_path).read_text()).get("joint_degrees", {})
            baked = bake(robot, pose, figure_asset_dir / item_id)
            visual = figure_asset_dir / item_id / "robot.obj"
            collision = visual
            source_record = {"robot": robot, "pose": str(pose_path),
                             "pose_sha256": digest(pose_path),
                             "source_urdf_sha256": baked["source_urdf_sha256"],
                             "baked_obj_sha256": baked["visual_obj_sha256"],
                             "tcp_in_baked_frame_m": baked["tcp_in_baked_frame_m"],
                             "ik": "not_run", "physics_simulation": "not_run"}
            if normalisation != "none":
                raise ValueError(f"ADDITION_NORMALISATION_UNKNOWN({normalisation}): a baked "
                                 "robot is already Z-up")
        elif kind == "prop":
            if item.get("simready_leaf"):
                visual, collision, source_record = prepare_simready_prop(
                    str(item["simready_leaf"]), figure_asset_dir / item_id)
            else:
                visual = expand_path(item["visual_uri"])
                if not visual.is_file():
                    raise ValueError(f"ADDITION_MESH_MISSING({item_id}: {visual})")
                collision = (expand_path(item["collision_uri"]) if item.get("collision_uri")
                             else visual)
                source_record = {"visual_sha256": digest(visual),
                                 "collision_sha256": digest(collision)}
        else:
            raise ValueError(f"ADDITION_KIND_UNKNOWN({kind})")

        # A robot is judged on what it stands on, not on its whole silhouette:
        # an arm reaching over a desk and a humanoid reaching over an island are
        # the pose the figure is *for*, and the whole-body box calls both of them
        # collisions. The full box is still recorded.
        band = ROBOT_SUPPORT_BAND_M if kind == "robot" else None
        full_lo, full_hi = apply_normalisation(*obj_local_bounds(visual), normalisation)
        lo, hi = ((full_lo, full_hi) if band is None
                  else apply_normalisation(*obj_local_bounds(visual, band), normalisation))
        support = str(item.get("on", "floor"))
        if support == "floor":
            surface_z = floor_z
        else:
            if support not in by_id:
                raise ValueError(f"ADDITION_UNKNOWN_SUPPORT({item_id}, {support})")
            if support in (substituted_ids or set()):
                # The resting height comes from the delivery's scene_ir box, and a
                # figure-only geometry substitution is exactly the case where that
                # box is no longer what renders. Measured 2026-09-26: props declared
                # on the substituted bed landed at the delivered bed's box top --
                # 0.9 m, the headboard -- and floated against it in the picture.
                raise ValueError(
                    f"ADDITION_SUPPORT_SUBSTITUTED({item_id}, {support}): its geometry was "
                    "replaced for the figure, so the delivery's measured top is not where "
                    "this would rest")
            if support not in bounds_by_instance:
                raise ValueError(f"ADDITION_SUPPORT_BOUNDS_MISSING({item_id}, {support})")
            surface_z = footprints[support][3]
        yaw = float(item.get("yaw_deg", 0.0))
        facing_offset = 0.0
        if kind == "robot":
            from rfw_w.scenes.visualization.robot_asset import ROBOTS as _ROBOTS
            # A declaration says which bearing the robot faces; the mesh's own +x may not be its
            # front. Applying the declared yaw raw put the G1's back to the island it was fetching
            # from (measured 2026-09-26, kitchen r03 and bedroom r03).
            facing_offset = float(_ROBOTS[str(item["robot"])].get("facing_offset_deg", 0.0))
            yaw = yaw + facing_offset
        xy = [float(v) for v in item["xy"]]
        # The declaration places the mesh in XY; the height is the support's, so
        # the item rests on the surface by construction. A declared z is allowed
        # only as a check on that, never as an override.
        translation = [xy[0], xy[1], surface_z - lo[2]]
        declared_z = item.get("z_m")
        if declared_z is not None and abs(float(declared_z) - translation[2]) > ADDITION_SURFACE_EPS_M:
            raise ValueError(
                f"ADDITION_FLOATING({item_id}, declared={float(declared_z):.4f} "
                f"resting={translation[2]:.4f})")

        offset = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0]
        half = [(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0]
        aabb_min, aabb_max = world_aabb(offset, half, translation, yaw)
        centre = [(aabb_min[0] + aabb_max[0]) / 2.0, (aabb_min[1] + aabb_max[1]) / 2.0]
        if not point_inside(polygon, centre[0], centre[1]):
            raise ValueError(
                f"ADDITION_OUTSIDE_ROOM({item_id}, centre=({centre[0]:.3f},{centre[1]:.3f}))")
        if support == "floor":
            # The opening rule is about passage, so it judges what stands on the
            # floor. An item on a support cannot reach more than
            # ADDITION_OVERHANG_M past that support's own footprint, and where
            # the support stands was the delivery's decision, not this figure's:
            # measured on the office, the desk is against the window wall, so
            # judging a mug on its top by the window strip refuses every mug.
            for opening in openings:
                if aabb_blocks_opening(aabb_min, aabb_max, polygon, opening):
                    raise ValueError(
                        f"ADDITION_BLOCKS_OPENING({item_id}, {opening.get('opening_id')})")
        else:
            # An item that rests on a surface must stand **on** it. Without this
            # the overlap rule alone is happy to hang an arm's base half off the
            # desk edge: nothing else is there to collide with, and the picture
            # shows a robot floating over the floor.
            smin, smax = footprints[support][0], footprints[support][1]
            overhang = max(smin[0] - aabb_min[0], aabb_max[0] - smax[0],
                           smin[1] - aabb_min[1], aabb_max[1] - smax[1])
            if overhang > ADDITION_OVERHANG_M:
                raise ValueError(
                    f"ADDITION_OFF_SUPPORT({item_id}, {support}, overhang={overhang:.3f} m)")

        exempt = set(item.get("overlap_exempt", [])) | {support}
        if kind == "robot" and item.get("task_target"):
            # The robot stands where it can work on its target; its base band and
            # the target's footprint may share ground under an outstretched hand.
            exempt.add(str(item["task_target"]))
        z_min, z_max = translation[2] + lo[2], translation[2] + hi[2]
        for other_id, (omin, omax, oz_min, oz_max) in footprints.items():
            if other_id in exempt:
                continue
            overlap_x = min(aabb_max[0], omax[0]) - max(aabb_min[0], omin[0])
            overlap_y = min(aabb_max[1], omax[1]) - max(aabb_min[1], omin[1])
            if (overlap_x > LAYOUT_OVERLAP_EPS_M and overlap_y > LAYOUT_OVERLAP_EPS_M
                    and z_max > oz_min + ADDITION_SURFACE_EPS_M
                    and oz_max > z_min + ADDITION_SURFACE_EPS_M):
                raise ValueError(f"ADDITION_OVERLAP({item_id}, {other_id})")

        reach_record = None
        if kind == "robot":
            profile_name = str(item.get("robot_profile") or ROBOT_PROFILE_OF[item["robot"]])
            profile = profile_for(profile_name)
            target_id = str(item["task_target"])
            if target_id not in by_id:
                raise ValueError(f"ADDITION_UNKNOWN_TARGET({item_id}, {target_id})")
            target = [float(v) for v in by_id[target_id]["translation"]]
            distance = math.hypot(target[0] - translation[0], target[1] - translation[1])
            reach_record = {
                "profile": profile_name,
                "reach_m": profile.reach_m,
                "target_instance_id": target_id,
                "horizontal_distance_m": round(distance, 4),
                "claim": "reach proxy: the target lies inside the envelope the tool point can "
                         "be placed in. No grasp is claimed; robot.ik is not_run.",
            }
            if distance > profile.reach_m:
                raise ValueError(
                    f"ROBOT_TARGET_OUT_OF_REACH({item_id}, {target_id}, "
                    f"{distance:.3f} m > {profile.reach_m:.3f} m)")

        instance = {
            "instance_id": item_id,
            "entity_id": str(item.get("entity_id", item_id)),
            "visual_uri": str(visual.resolve()),
            "collision_uri": str(Path(collision).resolve()),
            "translation": translation,
            "rotation_wxyz": yaw_quat(yaw),
            "dynamic": False,
            "normalisation": normalisation,
        }
        staged.append({"instance": instance})
        footprints[item_id] = (aabb_min, aabb_max, z_min, z_max)
        records.append({
            "instance_id": item_id, "kind": kind, "on": support,
            "translation": translation, "yaw_deg": yaw,
            "declared_yaw_deg": float(item.get("yaw_deg", 0.0)),
            "facing_offset_deg": facing_offset,
            "world_aabb_min": aabb_min, "world_aabb_max": aabb_max,
            "footprint_rule": ("support band, the lowest "
                               f"{ROBOT_SUPPORT_BAND_M} m of the mesh" if kind == "robot"
                               else "the whole mesh"),
            "local_box_min": full_lo, "local_box_max": full_hi,
            "overlap_exempt": sorted(exempt),
            "normalisation": normalisation,
            "source": source_record,
            "reach": reach_record,
            "claim": "figure_only_addition",
        })

    instances.extend(item["instance"] for item in staged)
    return records


def apply_support_table(request: dict, declaration: dict, *, bounds_by_instance: dict,
                        figure_asset_dir: Path, resting_ids: set | None = None) -> dict:
    """Swap one room's task support for the shared figure table.

    The three delivered supports are three heights and three footprints, which
    is why the rows do not read as a set. This replaces the named instance's
    geometry with ``make_paper_table``'s, **and** replaces its entry in
    ``bounds_by_instance`` -- placements downstream ask that table, not the
    delivered asset, where its top is. Skipping the second half is how an object
    ends up resting at the old support's height, floating over the new one.

    Pose, instance id and entity id are untouched; the delivery is untouched.
    """
    from rfw_w.scenes.visualization.make_paper_table import TABLE_SIZE_M, make_table, table_bounds

    instance_id = str(declaration["instance"])
    instance = next((i for i in request["instances"] if i["instance_id"] == instance_id), None)
    if instance is None:
        raise ValueError(f"SUPPORT_TABLE_UNKNOWN_INSTANCE({instance_id})")
    size = tuple(float(v) for v in declaration.get("size_m", TABLE_SIZE_M))
    visual, collision = make_table(figure_asset_dir / "support_table", size)
    old_visual, old_collision = instance["visual_uri"], instance.get("collision_uri")
    old_bounds = bounds_by_instance.get(instance_id)
    instance["visual_uri"] = str(visual.resolve())
    instance["collision_uri"] = str(collision.resolve())
    instance["normalisation"] = "none"
    bounds_by_instance[instance_id] = table_bounds(size)
    top_z = float(instance["translation"][2]) + size[2]
    # Whatever the layout declared as resting on this support has to rest on the
    # *new* top. Measured 2026-09-26: the office cup's origin is its centre
    # (bounds_min_m z = -0.046), so re-seating it at "top + 0" buried it 46 mm
    # into the table -- visible from above, gone entirely in the collision pass,
    # and the only thing that said so was a TARGET_NOT_VISIBLE metrics gap two
    # renders later.
    seated = []
    for other_id in sorted(resting_ids or ()):
        other = next((i for i in request["instances"] if i["instance_id"] == other_id), None)
        if other is None or other_id not in bounds_by_instance:
            continue
        base_z = float(other["translation"][2]) + float(bounds_by_instance[other_id][0][2])
        seated.append({"instance_id": other_id, "base_z_m": round(base_z, 4)})
        if abs(base_z - top_z) > ADDITION_SURFACE_EPS_M:
            raise ValueError(
                f"SUPPORT_TABLE_ORPHANED_ITEM({other_id}, base={base_z:.4f} top={top_z:.4f}): "
                "it was declared as resting on this support but does not sit on the new top")
    return {
        "instance_id": instance_id,
        "claim": "figure_only_geometry_substitution: one shared work table for every room",
        "size_m": list(size),
        "source_visual_sha256": digest(Path(old_visual)),
        "source_collision_sha256": digest(Path(old_collision)) if old_collision else None,
        "source_bounds_min_m": None if old_bounds is None else list(old_bounds[0]),
        "source_bounds_max_m": None if old_bounds is None else list(old_bounds[1]),
        "figure_visual_sha256": digest(visual),
        "figure_collision_sha256": digest(collision),
        "figure_material_sha256": digest(visual.with_suffix(".mtl")),
        "visual_path": str(visual),
        "collision_path": str(collision),
        "top_z_m": top_z,
        "items_seated_on_it": seated,
    }


def apply_shell_materials(request: dict, declaration: dict) -> dict:
    """Rebind named shell roles on the copied request; return the before/after record.

    The surfaces travel in ``room_materials`` -- the field the request already
    carries and which a shell-backed room currently ignores entirely -- and the
    role list turns it on. Unknown roles are refused here rather than dropped,
    because a dropped role is an override the provenance claims and the picture
    never got.
    """
    from rfw_w.scenes.workers.isaac.protocol import OVERRIDABLE_SHELL_ROLES

    materials = request.setdefault("room_materials", {})
    record = {"before": json.loads(json.dumps(materials)), "applied": {}}
    for role, surface in declaration.items():
        if role not in OVERRIDABLE_SHELL_ROLES:
            raise ValueError(f"SHELL_MATERIAL_ROLE_UNKNOWN({role}): known roles are "
                             f"{sorted(OVERRIDABLE_SHELL_ROLES)}")
        record["applied"][role] = {"from": materials.get(role), "to": surface}
        materials[role] = surface
    request["shell_material_override"] = sorted(declaration)
    record["roles"] = sorted(declaration)
    record["after"] = json.loads(json.dumps(materials))
    return record


# Pinned to the wire format's own fields: the worker parses lighting with
# ``LightingRequest.from_dict``, which silently ignores keys it does not know --
# a typo would otherwise render with unchanged light while the provenance claims
# an override.
LIGHTING_KEYS = ("dome_intensity", "sun_intensity", "ceiling_fill_intensity",
                 "dome_color", "sun_color")


def apply_lighting_overrides(request: dict, overrides: dict) -> dict:
    """Override lighting keys on the copied request; return the before/after record."""
    lighting = request.setdefault("lighting", {})
    record = {"before": dict(lighting), "applied": {}}
    for key, value in overrides.items():
        if key not in LIGHTING_KEYS:
            raise ValueError(f"LIGHTING_UNKNOWN_KEY({key})")
        record["applied"][key] = {"from": lighting.get(key), "to": value}
        lighting[key] = value
    record["after"] = dict(lighting)
    return record


# ---------------------------------------------------------------------------
# Collision companion pass (--with-collision): the same post-override request,
# rendered once more with every instance drawn as its collision mesh.
# ---------------------------------------------------------------------------


def collision_request(request: dict) -> tuple[dict, list[dict]]:
    """A copy of ``request`` that renders collision geometry instead of visuals.

    Per instance with a collision mesh, ``visual_uri`` is replaced by the
    collision mesh and the ``collision_uri`` field is dropped: the worker
    authors a carried ``collision_uri`` as an extra *invisible* physics child,
    so keeping the field would author the same mesh twice (once visible, once
    invisible). This differs from ``multisim/render_isaac.py``'s collision pass,
    which keeps the field and hard-fails on a missing collision mesh -- here the
    room figure should still render, so an instance without a collision mesh
    keeps its visual and the substitution record names it.

    The shell is left alone in both passes: its visible mesh IS its collision
    mesh. Camera, lighting and every other field are byte-identical, so the two
    passes differ only in which geometry the instances draw.
    """
    variant = json.loads(json.dumps(request))
    substitutions = []
    for instance in variant["instances"]:
        collision = instance.get("collision_uri")
        if not collision:
            substitutions.append({"instance_id": instance["instance_id"],
                                  "entity_id": instance.get("entity_id"),
                                  "substituted": False, "reason": "no_collision_uri"})
            continue
        record = {"instance_id": instance["instance_id"],
                  "entity_id": instance.get("entity_id"), "substituted": True,
                  "visual_uri_before": instance["visual_uri"],
                  "visual_uri_after": collision}
        collision_path = Path(collision)
        record["collision_sha256"] = digest(collision_path) if collision_path.is_file() else None
        instance["visual_uri"] = collision
        instance.pop("collision_uri", None)
        substitutions.append(record)
    return variant, substitutions


# ---------------------------------------------------------------------------
# Post-render metrics: what the produced pixels actually show.
# ---------------------------------------------------------------------------

# A whole-room frame legitimately clips a few pixels on specular highlights, so
# the overexposure gate fires on area. 2% is the line between "a highlight" and
# the SPEC-013 failure mode, where the calibrated sun intensity blew out whole
# white walls.
DEFAULT_METRIC_THRESHOLDS = {
    "max_clipped_fraction": 0.02,
    # A level frame looking into a room should see geometry almost everywhere;
    # under half valid depth means the camera mostly missed the room.
    "min_depth_valid_fraction": 0.5,
    "min_target_pixels": 1,
}

# The dollhouse frame legitimately contains sky around the room silhouette --
# the camera stands outside and above, looking past the roofless shell -- so its
# depth can never approach full validity. Measured on the first dollhouse
# renders: 0.35-0.41 valid with correct framing. 0.25 still catches a camera
# that mostly missed the room without firing on the silhouette's sky.
DOLLHOUSE_METRIC_THRESHOLDS = {"min_depth_valid_fraction": 0.25}


def clipped_fraction(rgb) -> float:
    """Fraction of pixels saturated in every channel.

    250 rather than 255: the renderer's tonemapping plateaus slightly below the
    uint8 ceiling, so a clipped white wall reads as 250-254, not exactly 255.
    """
    arr = np.asarray(rgb)
    if arr.size == 0:
        return 0.0
    return float(np.all(arr[..., :3] >= 250, axis=-1).mean())


def depth_stats(depth) -> dict:
    """Valid fraction and min/max of a depth map; valid matches the worker's own
    definition (finite and positive)."""
    arr = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0)
    stats = {"valid_fraction": float(valid.mean()) if arr.size else 0.0,
             "min_m": None, "max_m": None}
    if valid.any():
        stats["min_m"] = float(arr[valid].min())
        stats["max_m"] = float(arr[valid].max())
    return stats


def target_pixel_counts(instance_ids, id_map: dict, target_ids) -> dict:
    """Pixels per target instance, resolved through the capture's renderer-id map."""
    ids = np.asarray(instance_ids)
    counts = {}
    for target in target_ids:
        renderer_ids = [int(r) for r, name in id_map.items() if name == target]
        counts[target] = int(np.isin(ids, renderer_ids).sum()) if renderer_ids else 0
    return counts


def evaluate_render_metrics(metrics: dict, thresholds: dict | None = None) -> list[str]:
    """Named gaps for a metrics dict; empty means the render passed its own numbers."""
    limits = {**DEFAULT_METRIC_THRESHOLDS, **(thresholds or {})}
    gaps = []
    for camera_id in sorted(metrics.get("cameras", {})):
        camera = metrics["cameras"][camera_id]
        clipped = camera.get("clipped_fraction")
        if clipped is not None and clipped > limits["max_clipped_fraction"]:
            gaps.append(f"OVEREXPOSED({camera_id}, clipped_fraction={clipped:.4f} > "
                        f"{limits['max_clipped_fraction']})")
        depth = camera.get("depth")
        if depth is not None and depth["valid_fraction"] < limits["min_depth_valid_fraction"]:
            gaps.append(f"DEPTH_INVALID({camera_id}, valid_fraction={depth['valid_fraction']:.4f} < "
                        f"{limits['min_depth_valid_fraction']})")
        for target, count in sorted((camera.get("target_pixels") or {}).items()):
            if count < limits["min_target_pixels"]:
                gaps.append(f"TARGET_NOT_VISIBLE({camera_id}, {target}, pixels={count})")
    return gaps


def collect_render_metrics(report_path: Path, target_ids, thresholds: dict | None = None) -> dict:
    """Per-camera exposure/depth/target-visibility numbers from a worker report.

    ``thresholds`` override DEFAULT_METRIC_THRESHOLDS per view (e.g. the
    dollhouse's lower depth-valid floor); the merged limits are recorded so the
    metrics file says which bar it was judged against.
    """
    from PIL import Image

    report = json.loads(report_path.read_text())
    base = report_path.parent
    cameras = {}
    for capture in report.get("captures", []):
        entry: dict = {}
        rgb_file = base / capture["rgb_path"]
        if rgb_file.is_file():
            entry["clipped_fraction"] = clipped_fraction(
                np.asarray(Image.open(rgb_file).convert("RGB")))
        depth_path = capture.get("depth_path")
        if depth_path and (base / depth_path).is_file():
            entry["depth"] = depth_stats(np.load(base / depth_path))
        ids_path = capture.get("instance_id_path")
        if ids_path and (base / ids_path).is_file():
            entry["target_pixels"] = target_pixel_counts(
                np.load(base / ids_path), capture.get("instance_id_map", {}), target_ids)
        cameras[capture["camera_id"]] = entry
    metrics = {"cameras": cameras, "target_ids": list(target_ids),
               "thresholds": {**DEFAULT_METRIC_THRESHOLDS, **(thresholds or {})}}
    metrics["gaps"] = evaluate_render_metrics(metrics, thresholds)
    return metrics


def dollhouse_request(delivery: Path, manifest: dict, request: dict, out: Path,
                      corner: str = "auto", shell_mode: str = "roof",
                      elevation_deg: float = DOLLHOUSE_ELEVATION_DEG,
                      wall_clip_height: float | None = 1.25,
                      wall_clip_corner: str | None = None,
                      width: int = 1600, height: int = 1200,
                      margin: float = DOLLHOUSE_MARGIN,
                      fit_wall_tops: bool = True) -> tuple[dict, dict]:
    """Replace the shell with a roofless (or cutaway) copy and look down into it.

    ``roof`` removes only the ceiling and facade, so the walls still read as
    walls from above; ``cutaway`` also opens the near long wall, which from a
    high oblique angle is only needed when a tall furniture row hugs that wall.
    In roof mode the two walls nearest the camera corner are clipped to
    ``wall_clip_height`` (default 1.25 m, roughly tabletop-plus-a-bit: low enough
    that the 52-degree camera sees the floor over them, high enough that the
    stub still reads as a wall); ``wall_clip_corner`` overrides which walls come
    down -- ``"all"`` clips every wall, which is what a near-top view needs
    (from straight above there is no near side, and any full-height wall leans
    across the frame). ``fit_wall_tops`` is forwarded to the camera fit.
    """
    from rfw_w.scenes.visualization.cutaway_shell import build_cutaway_shell, choose_cut, wall_clip_for

    room = request["room"]
    shell = (delivery / manifest["isaac_request_path"]).parent / room["shell_uri"]
    cut = choose_cut(room) if shell_mode == "cutaway" else None
    figure_shell = out / f"shell_{shell_mode}.usdc"
    camera = dollhouse_camera(room, corner, elevation_deg=elevation_deg,
                              width=width, height=height, margin=margin,
                              fit_wall_tops=fit_wall_tops)
    wall_clip = None
    if shell_mode == "roof" and wall_clip_height is not None:
        wall_clip = wall_clip_for(room, wall_clip_corner or camera["corner"],
                                  wall_clip_height)
    record = build_cutaway_shell(shell.resolve(), figure_shell, cut, mode=shell_mode,
                                 wall_clip=wall_clip)
    base = dict(request["cameras"][0])
    base.update(purpose="export", require_target_visible=False,
                target_instance_ids=[], save_depth=True, save_instance_ids=True,
                resolution=camera["resolution"])
    doll = dict(base, camera_id="paper_dollhouse", eye=camera["eye"], target=camera["target"],
                focal_length_mm=camera["focal_length_mm"],
                horizontal_aperture_mm=camera["horizontal_aperture_mm"],
                vertical_aperture_mm=camera["vertical_aperture_mm"])
    room["shell_uri"] = str(figure_shell.resolve())
    request["cameras"] = [doll]
    record["camera_id"] = "paper_dollhouse"
    record["camera"] = camera
    record["delivery"] = str(delivery)
    record["scene_version"] = manifest["scene_version"]
    (out / "dollhouse.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return request, record


#: How far from the support's own footprint an object still counts as part of the workspace.
#: Measured 2026-09-26 on the three curated rooms: the kitchen stool sits 0.96 m from the
#: island's centre, the office chair 0.83 m from the desk's, and the next-nearest object in
#: either room is 3.13 m away (the kitchen plant). 1.60 m separates those two populations with
#: room to spare and is not near either edge.
WORKSPACE_CONTEXT_RADIUS_M = 1.60

#: Fraction of the half-frame kept clear of the outermost required box, so nothing is cropped
#: at the edge. Same value and same reason as the multisim camera fit.
WORKSPACE_MARGIN = 0.06

#: The same, for a context object's centre. Much wider, because a centre inside the frame only
#: guarantees the object is half in it: at 0.06 the kitchen stool's centre cleared the bottom
#: edge by 3 cm and the stool rendered sliced off. 0.18 keeps roughly the near half of an
#: object this size whole, which is what makes it readable rather than merely present.
WORKSPACE_CONTEXT_MARGIN = 0.18

#: Closest the eye may come to a wall, so a solved distance cannot put the camera in one.
WORKSPACE_WALL_CLEARANCE_M = 0.20


def _workspace_context(by_entity: dict, scene: dict, support_id: str, target_id: str,
                       sx: float, sy: float) -> list[str]:
    """Objects near enough to the support to belong in the same picture.

    The framing used to be a fixed offset from the support, on the stated assumption that the
    three curated workspaces were interchangeable. They are not, and the kitchen is where that
    showed: its island sits in the narrow arm of an L-shaped room, and the fixed offset put the
    eye at (3.34, 2.26) -- 1.07 m from the stool at (4.20, 2.90), effectively on top of it and
    facing away. The close-up came out as an anonymous brown box with a bowl and a jar on it,
    with nothing in frame that says what room this is, while the stool that would have said it
    was just behind the lens.
    """
    context = []
    for entity_id, item in by_entity.items():
        if entity_id in {support_id, target_id}:
            continue
        source = next((s for s in scene["instances"] if s["entity_id"] == entity_id), None)
        if source is None:
            continue
        x, y = map(float, item["translation"][:2])
        if math.hypot(x - sx, y - sy) <= WORKSPACE_CONTEXT_RADIUS_M:
            context.append(entity_id)
    return sorted(context)


def add_workspace_camera(request: dict, manifest: dict, scene: dict) -> dict:
    """Frame the task support, the manipulable target, and whatever else is right there."""
    by_entity = {item["entity_id"]: item for item in request["instances"]}
    support_id = manifest["plan"]["support_id"]
    target_id = manifest["plan"]["target_id"]
    if support_id not in by_entity or target_id not in by_entity:
        raise ValueError(f"WORKSPACE_OBJECT_MISSING({support_id},{target_id})")
    support = by_entity[support_id]
    target = by_entity[target_id]
    source_support = next(item for item in scene["instances"]
                          if item["entity_id"] == support_id)
    surface_z = float(support["translation"][2]) + float(source_support["bounds_max_m"][2])
    sx, sy = map(float, support["translation"][:2])
    tx, ty = map(float, target["translation"][:2])

    context = _workspace_context(by_entity, scene, support_id, target_id, sx, sy)
    required = [support_id, target_id, *context]

    # World-space box the support and the target must sit wholly inside. Context objects are
    # held to their centres instead, further down.
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for entity_id in (support_id, target_id):
        item = by_entity[entity_id]
        source = next(s for s in scene["instances"] if s["entity_id"] == entity_id)
        for axis in range(3):
            base = float(item["translation"][axis])
            lo[axis] = min(lo[axis], base + float(source["bounds_min_m"][axis]))
            hi[axis] = max(hi[axis], base + float(source["bounds_max_m"][axis]))

    # All three curated workspaces sit near the north side, so the view stays south-west of the
    # support: from there the work surface and the seat beside it read as one scene rather than
    # as a slab seen end-on. The BEARING is kept; only the distance is now solved.
    bearing = (-0.85, -1.60)
    norm = math.hypot(*bearing)
    bearing = (bearing[0] / norm, bearing[1] / norm)

    aperture_h = float(request["cameras"][0].get("horizontal_aperture_mm", 20.955))
    width, height = 1600, 1200
    focal = 18.5
    # Square pixels: the renderer derives the vertical aperture from the frame ratio and ignores
    # a declared one. Measured and recorded in AGENTS.md section 6.
    aperture_v = aperture_h * height / width
    half_h = math.atan(aperture_h / (2.0 * focal))
    half_v = math.atan(aperture_v / (2.0 * focal))

    cx, cy = 0.65 * sx + 0.35 * tx, 0.65 * sy + 0.35 * ty
    look_at = [cx, cy, surface_z + 0.04]
    eye_z = surface_z + 1.42

    corners = []
    for axis_x in (lo[0], hi[0]):
        for axis_y in (lo[1], hi[1]):
            for axis_z in (lo[2], hi[2]):
                corners.append((axis_x, axis_y, axis_z))

    # Context objects have to be *recognisable*, not complete. Requiring every corner of every
    # nearby object put the eye 3.44-3.81 m back in all three rooms -- a wide shot in which the
    # jar this task is about covers a handful of pixels -- because a bed and a dining chair are
    # large and the frame had to swallow them whole. Their centre being in frame is what makes
    # the picture say "island with a stool at it" instead of "brown box".
    context_points = []
    for entity_id in context:
        item = by_entity[entity_id]
        source = next(s for s in scene["instances"] if s["entity_id"] == entity_id)
        context_points.append(tuple(
            float(item["translation"][axis])
            + (float(source["bounds_min_m"][axis]) + float(source["bounds_max_m"][axis])) / 2.0
            for axis in range(3)))

    def fits(distance: float) -> bool:
        """Support and target whole, context objects at least centred, margin kept clear.

        Solved by projection rather than by comparing the box's own x/y span to a field of
        view. The span shortcut was tried first and was wrong in a way that looked right: it
        measured the box from `look_at`, which is biased toward the support and is not the
        box's centre, and it used the axis-aligned extent rather than the extent across the
        line of sight. On the kitchen it returned 1.48 m -- nearer than the fixed offset it
        replaced -- and still left the stool 0.93 m outside a 0.59 m half-frame.
        """
        eye = (cx + bearing[0] * distance, cy + bearing[1] * distance, eye_z)
        forward = [look_at[i] - eye[i] for i in range(3)]
        length = math.hypot(*forward)
        forward = [c / length for c in forward]
        right = [forward[1], -forward[0], 0.0]
        right_len = math.hypot(*right)
        if right_len < 1e-9:
            return False
        right = [c / right_len for c in right]
        up = [right[1] * forward[2] - right[2] * forward[1],
              right[2] * forward[0] - right[0] * forward[2],
              right[0] * forward[1] - right[1] * forward[0]]
        for point, margin in ([(c, WORKSPACE_MARGIN) for c in corners]
                              + [(c, WORKSPACE_CONTEXT_MARGIN) for c in context_points]):
            delta = [point[i] - eye[i] for i in range(3)]
            depth = sum(delta[i] * forward[i] for i in range(3))
            if depth <= 1e-3:
                return False
            u = sum(delta[i] * right[i] for i in range(3)) / depth
            v = sum(delta[i] * up[i] for i in range(3)) / depth
            if abs(u) > math.tan(half_h) * (1.0 - margin):
                return False
            if abs(v) > math.tan(half_v) * (1.0 - margin):
                return False
        return True

    polygon = request["room"]["polygon_xy"]
    room_lo = [min(p[0] for p in polygon), min(p[1] for p in polygon)]
    room_hi = [max(p[0] for p in polygon), max(p[1] for p in polygon)]
    # Walk back along the bearing only as far as the room allows. A solved distance that puts
    # the eye in a wall is worse than a slightly tight frame, and the room is the hard bound.
    max_distance = float("inf")
    for axis in range(2):
        centre = cx if axis == 0 else cy
        if bearing[axis] < 0:
            max_distance = min(max_distance,
                               (centre - room_lo[axis] - WORKSPACE_WALL_CLEARANCE_M)
                               / -bearing[axis])
        elif bearing[axis] > 0:
            max_distance = min(max_distance,
                               (room_hi[axis] - WORKSPACE_WALL_CLEARANCE_M - centre)
                               / bearing[axis])
    max_distance = max(0.60, max_distance)

    # Nearest distance that fits, scanned at 1 cm and then reported honestly if none does.
    distance, fitted = max_distance, False
    steps = int((max_distance - 0.60) / 0.01) + 1
    for step in range(steps):
        candidate = 0.60 + step * 0.01
        if fits(candidate):
            distance, fitted = candidate, True
            break

    camera = dict(request["cameras"][0])
    camera.update(camera_id="paper_workspace_detail",
                  eye=[cx + bearing[0] * distance, cy + bearing[1] * distance, eye_z],
                  target=look_at,
                  focal_length_mm=focal,
                  resolution=[width, height],
                  target_instance_ids=[target["instance_id"], support["instance_id"]])
    request["cameras"].append(camera)
    record = {"camera_id": camera["camera_id"], "support_id": support_id,
              "target_id": target_id, "surface_z_m": surface_z,
              "context_ids": context, "required_ids": required,
              "required_box_m": {"min": lo, "max": hi},
              "distance_m": distance, "max_distance_in_room_m": max_distance,
              "eye": camera["eye"], "target": camera["target"]}
    if not fitted:
        # Named, not silently cropped: the room is too small to hold everything in one frame.
        record["gap"] = (f"WORKSPACE_FRAME_ROOM_LIMITED({support_id},"
                         f"{max_distance:.3f}m)")
    return record


def cutaway_request(delivery: Path, manifest: dict, request: dict, out: Path,
                    eye_height_m: float, focal_length_mm: float,
                    height: int = 900) -> tuple[dict, dict, str]:
    """Replace the shell with a cutaway copy and put a level camera outside it.

    The camera is level (eye and target share a z) because the tilted corner view
    was rejected on sight: from inside a room no lens shows it well, so the near
    side is opened and the camera stands outside looking in horizontally.
    """
    from rfw_w.scenes.visualization.cutaway_shell import build_cutaway_shell, choose_cut, level_camera

    shell = (delivery / manifest["isaac_request_path"]).parent / request["room"]["shell_uri"]
    cut = choose_cut(request["room"])
    record = build_cutaway_shell(shell.resolve(), out / "shell_cutaway.usdc", cut)
    camera = level_camera(request["room"], cut, eye_height_m=eye_height_m,
                          focal_length_mm=focal_length_mm, height=height)
    room = request["room"]
    polygon = room["polygon_xy"]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    height = room["height_m"]
    scene = json.loads((delivery / "scene_ir.json").read_text())
    support_id = manifest["plan"]["support_id"]
    support = next((i for i in scene["instances"] if i["entity_id"] == support_id), None)
    if support is None:
        raise ValueError(f"PAPER_PREVIEW_SUPPORT_MISSING({support_id})")
    sx, sy, sz = support["world_transform"]["translation"]
    surface_z = sz + support["bounds_max_m"][2]
    middle = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    chair = next((i for i in scene["instances"] if i["entity_id"] == "chair"), None)
    seat_sign = (1 if chair is None or
                 chair["world_transform"]["translation"][1] >= sy else -1)
    base = dict(request["cameras"][0])
    base.update(purpose="export", require_target_visible=False,
                target_instance_ids=[], save_depth=True, save_instance_ids=True,
                resolution=camera["resolution"])
    level = dict(base, camera_id="paper_room_level", eye=camera["eye"],
                 target=camera["target"], focal_length_mm=camera["focal_length_mm"],
                 horizontal_aperture_mm=camera["horizontal_aperture_mm"],
                 vertical_aperture_mm=camera["vertical_aperture_mm"])
    # The detail shot stays inside the room and is unchanged by the cut: it is a
    # close-up on the support surface, which the removed ceiling and near wall
    # never framed.
    detail = dict(base, camera_id="paper_desk_detail",
                  eye=[min(max(sx + 0.48, min_x + 0.2), max_x - 0.2),
                       min(max(sy + seat_sign * 1.28, min_y + 0.2), max_y - 0.2),
                       min(height - 0.2, surface_z + 0.94)],
                  target=[sx, sy, surface_z], focal_length_mm=13.0)
    request["room"]["shell_uri"] = str((out / "shell_cutaway.usdc").resolve())
    request["cameras"] = [level, detail]
    record["camera_id"] = "paper_room_level"
    record["detail_camera_id"] = "paper_desk_detail"
    record["delivery"] = str(delivery)
    record["scene_version"] = manifest["scene_version"]
    (out / "cutaway.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return request, record, "cutaway"


def build_request(delivery: Path, overview_corner: str = "sw") -> tuple[dict, dict, Path, str]:
    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    source_request = delivery / manifest["isaac_request_path"]
    request = json.loads(source_request.read_text())
    version = manifest["scene_version"]
    if request["scene_version"] != version:
        raise ValueError("PAPER_PREVIEW_SCENE_VERSION_MISMATCH(request)")
    shell_uri = request["room"].get("shell_uri")
    if not shell_uri:
        raise ValueError("PAPER_PREVIEW_SHELL_MISSING")
    shell = (source_request.parent / shell_uri).resolve()
    if not shell.is_file():
        raise ValueError("PAPER_PREVIEW_SHELL_MISSING(file)")
    request["room"]["shell_uri"] = str(shell)
    # The paper preview compares authored geometry. A production request may settle
    # rigid objects for its own gates; copying that duration would move them again.
    request["simulate_seconds"] = 0.0
    room = request["room"]["polygon_xy"]
    xs = [p[0] for p in room]
    ys = [p[1] for p in room]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    height = request["room"]["height_m"]
    middle = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    # The existing overview_corner aims at the task desk. These figure cameras aim
    # at the centre of the room and use a wide lens, so the other furnishings enter.
    base = dict(request["cameras"][0])
    base.update(purpose="export", require_target_visible=False,
                target_instance_ids=[], save_depth=True, save_instance_ids=True,
                resolution=[1600, 1000])
    eye, chosen_corner = choose_vantage(room, height, overview_corner)
    oblique = dict(base, camera_id="paper_room_oblique",
                   eye=eye, target=[middle[0], middle[1], 0.65], focal_length_mm=8.0)
    scene = json.loads((delivery / "scene_ir.json").read_text())
    support_id = manifest["plan"]["support_id"]
    support = next((item for item in scene["instances"] if item["entity_id"] == support_id), None)
    if support is None:
        raise ValueError(f"PAPER_PREVIEW_SUPPORT_MISSING({support_id})")
    sx, sy, sz = support["world_transform"]["translation"]
    surface_z = sz + support["bounds_max_m"][2]
    chair = next((item for item in scene["instances"] if item["entity_id"] == "chair"), None)
    seat_sign = (1 if chair is None or
                 chair["world_transform"]["translation"][1] >= sy else -1)
    # Look at the tabletop from its seat side; the authored chair determines
    # that side, and clipping to the room bounds keeps the camera inside.
    detail = dict(base, camera_id="paper_desk_detail",
                  eye=[min(max(sx + 0.48, min_x + 0.2), max_x - 0.2),
                       min(max(sy + seat_sign * 1.28, min_y + 0.2), max_y - 0.2),
                       min(height - 0.2, surface_z + 0.94)],
                  target=[sx, sy, surface_z], focal_length_mm=13.0)
    request["cameras"] = [oblique, detail]
    return request, manifest, source_request, chosen_corner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument("--overview-corner", choices=("sw", "se", "nw", "ne"), default="sw")
    parser.add_argument("--view", choices=("cutaway", "oblique", "dollhouse"), default="cutaway",
                        help="cutaway: open the ceiling and the near long wall and look in "
                             "level from outside. oblique: the earlier tilted corner view. "
                             "dollhouse: slightly top-down oblique from outside a ceiling-less "
                             "shell, framing the whole room.")
    parser.add_argument("--shell-mode", choices=("cutaway", "roof"), default="roof",
                        help="dollhouse shell: roof removes only the ceiling and facade; "
                             "cutaway also opens the near long wall")
    parser.add_argument("--dollhouse-corner", choices=("auto", "ne", "nw", "se", "sw"),
                        default="auto",
                        help="dollhouse camera corner; auto looks from the corner opposite "
                             "the main door. The same corner drives the roof-mode wall clip.")
    parser.add_argument("--dollhouse-elevation-deg", type=float,
                        default=DOLLHOUSE_ELEVATION_DEG,
                        help="dollhouse view-direction elevation above horizontal; eye height "
                             "stays 2.2x room height, the horizontal distance follows")
    parser.add_argument("--dollhouse-width", type=int, default=1600,
                        help="dollhouse frame width in px")
    parser.add_argument("--dollhouse-height", type=int, default=1200,
                        help="dollhouse frame height in px; match the room bbox aspect to "
                             "fill the frame (top view)")
    parser.add_argument("--dollhouse-margin", type=float, default=DOLLHOUSE_MARGIN,
                        help="frame-fit margin divisor; 1.01 fills the frame edge-to-edge")
    parser.add_argument("--wall-clip-height", type=float, default=1.25,
                        help="dollhouse roof mode: clip the two walls nearest the camera "
                             "corner to this height (0 disables). Default 1.25 m: low enough "
                             "that the camera sees the floor, high enough to read as a wall")
    parser.add_argument("--wall-clip-corner", choices=("ne", "nw", "se", "sw", "all"),
                        default=None,
                        help="which walls the roof-mode clip cuts (default: the camera "
                             "corner's two walls); 'all' cuts every wall -- use it for a "
                             "near-top view, where any full-height wall leans across the frame")
    parser.add_argument("--dollhouse-fit", choices=("wall_tops", "floor"),
                        default="wall_tops",
                        help="what the dollhouse frame fit must contain; 'floor' fits the "
                             "footprint only, for near-top views over fully clipped walls")
    parser.add_argument("--layout-overrides", type=Path, default=None,
                        help='JSON {"overrides": {"<instance_id>": {"translation": [x,y,z], '
                             '"yaw_deg": d, "remove": true, "on": "<support_id>"}}} applied '
                             'to the copied request')
    parser.add_argument("--lighting", type=Path, default=None,
                        help="JSON of lighting keys (dome_intensity, sun_intensity, ...) "
                             "overriding the copied request's lighting")
    parser.add_argument("--additions", type=Path, default=None,
                        help='JSON {"additions": [{"id": .., "kind": "robot"|"prop", ...}], '
                             '"shell_materials": {"wall": {...}}} adding declared robots and '
                             'props to the copied request; each rests on a named support and '
                             'is judged by the same geometry rules as a layout override')
    parser.add_argument("--with-collision", action="store_true",
                        help="after the normal render, run the worker once more on the same "
                             "request with every instance's visual replaced by its collision "
                             "mesh (instances without one keep their visual; both are "
                             "recorded), writing isaac_collision/ beside isaac/")
    parser.add_argument("--workspace-detail", action="store_true",
                        help="add a close camera on the task support and manipulable target")
    parser.add_argument("--replace-bed", action="store_true",
                        help="figure-only upholstered bed in the same footprint; record geometry substitution")
    parser.add_argument("--replace-jar", action="store_true",
                        help="figure-only ceramic jar in the same target footprint")
    parser.add_argument("--eye-height-m", type=float, default=1.45)
    parser.add_argument("--focal-length-mm", type=float, default=18.0)
    parser.add_argument("--height", type=int, default=900,
                        help="cutaway frame height; the room is framed on its width")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--request-only", action="store_true",
                        help="build and validate the figure request, then stop before the worker")
    args = parser.parse_args()
    delivery = args.delivery.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    source_request = delivery / manifest["isaac_request_path"]
    if args.view == "cutaway":
        source = json.loads(source_request.read_text())
        if source["scene_version"] != manifest["scene_version"]:
            raise ValueError("PAPER_PREVIEW_SCENE_VERSION_MISMATCH(request)")
        source["simulate_seconds"] = 0.0
        request, shell_record, _ = cutaway_request(
            delivery, manifest, source, out, args.eye_height_m, args.focal_length_mm,
            args.height)
        chosen_corner = None
    elif args.view == "dollhouse":
        source = json.loads(source_request.read_text())
        if source["scene_version"] != manifest["scene_version"]:
            raise ValueError("PAPER_PREVIEW_SCENE_VERSION_MISMATCH(request)")
        source["simulate_seconds"] = 0.0
        request, shell_record = dollhouse_request(
            delivery, manifest, source, out, corner=args.dollhouse_corner,
            shell_mode=args.shell_mode, elevation_deg=args.dollhouse_elevation_deg,
            wall_clip_height=args.wall_clip_height if args.wall_clip_height > 0 else None,
            wall_clip_corner=args.wall_clip_corner,
            width=args.dollhouse_width, height=args.dollhouse_height,
            margin=args.dollhouse_margin,
            fit_wall_tops=args.dollhouse_fit == "wall_tops")
        chosen_corner = None
    else:
        request, manifest, source_request, chosen_corner = build_request(
            delivery, args.overview_corner)
        shell_record = None
    # Overrides act on the copied request only. Protected ids are read from the
    # *source* request: the figure cameras replace its camera list, so reading
    # them afterwards would silently un-protect the task targets.
    plan = manifest.get("plan", {})
    protected_ids = {v for v in (plan.get("target_id"), plan.get("support_id")) if v}
    for camera in json.loads(source_request.read_text()).get("cameras", []):
        protected_ids.update(camera.get("target_instance_ids", []))
    layout_record = None
    overrides = None
    if args.layout_overrides is not None:
        payload = json.loads(args.layout_overrides.read_text())
        overrides = payload.get("overrides")
        if not isinstance(overrides, dict):
            raise ValueError("LAYOUT_OVERRIDES_SHAPE(expected {\"overrides\": {...}})")
        scene = json.loads((delivery / "scene_ir.json").read_text())
        bounds = {i["instance_id"]: (i["bounds_min_m"], i["bounds_max_m"])
                  for i in scene.get("instances", []) if "bounds_min_m" in i}
        deltas = apply_layout_overrides(request, overrides, bounds_by_instance=bounds,
                                        protected_ids=protected_ids)
        layout_record = {"source": str(args.layout_overrides.resolve()),
                         "source_sha256": digest(args.layout_overrides), "deltas": deltas}
    replacement_record = []
    if args.replace_bed or args.replace_jar:
        from rfw_w.scenes.visualization.make_paper_bed import make_bed, make_jar
        for name, enabled, maker in (("bed", args.replace_bed, make_bed),
                                     ("jar", args.replace_jar, make_jar)):
            if not enabled:
                continue
            instance = next((item for item in request["instances"]
                             if item["entity_id"] == name), None)
            if instance is None:
                raise ValueError(f"ASSET_REPLACEMENT_NO_INSTANCE({name})")
            old_visual, old_collision = instance["visual_uri"], instance.get("collision_uri")
            visual, collision = maker(out / "figure_assets" / name)
            instance["visual_uri"] = str(visual.resolve())
            instance["collision_uri"] = str(collision.resolve())
            replacement_record.append({
                "instance_id": instance["instance_id"],
                "claim": "figure_only_geometry_substitution",
                "source_visual_sha256": digest(Path(old_visual)),
                "source_collision_sha256": digest(Path(old_collision)) if old_collision else None,
                "figure_visual_sha256": digest(visual),
                "figure_collision_sha256": digest(collision),
                "figure_material_sha256": digest(visual.with_suffix(".mtl")),
                "visual_path": str(visual), "collision_path": str(collision),
            })
    additions_record = None
    if args.additions is not None:
        payload = json.loads(args.additions.read_text())
        items = payload.get("additions", [])
        if not isinstance(items, list):
            raise ValueError("ADDITIONS_SHAPE(expected {\"additions\": [...]})")
        scene = json.loads((delivery / "scene_ir.json").read_text())
        bounds = {i["instance_id"]: (i["bounds_min_m"], i["bounds_max_m"])
                  for i in scene.get("instances", []) if "bounds_min_m" in i}
        table_record = None
        if payload.get("support_table"):
            table_record = apply_support_table(
                request, payload["support_table"], bounds_by_instance=bounds,
                figure_asset_dir=out / "figure_assets",
                resting_ids={i for i, change in (overrides or {}).items()
                             if change.get("on") == payload["support_table"]["instance"]})
        added = apply_additions(request, items, bounds_by_instance=bounds,
                                figure_asset_dir=out / "figure_assets",
                                substituted_ids={r["instance_id"] for r in replacement_record})
        shell_materials_record = None
        if payload.get("shell_materials"):
            shell_materials_record = apply_shell_materials(request, payload["shell_materials"])
        additions_record = {
            "source": str(args.additions.resolve()),
            "source_sha256": digest(args.additions),
            "added": added,
            "robots": sorted(r["instance_id"] for r in added if r["kind"] == "robot"),
            "props": sorted(r["instance_id"] for r in added if r["kind"] == "prop"),
            "shell_materials": shell_materials_record,
            "support_table": table_record,
            "claim": "figure_only_addition: nothing here is a pipeline product, and no gate in "
                     "the source delivery judged any of it",
            "robot_physics": "not_run",
            "robot_ik": "not_run",
        }
    workspace_record = None
    if args.workspace_detail:
        if args.view != "dollhouse":
            raise ValueError("WORKSPACE_DETAIL_REQUIRES_DOLLHOUSE")
        workspace_record = add_workspace_camera(
            request, manifest, json.loads((delivery / "scene_ir.json").read_text()))
    lighting_record = None
    if args.lighting is not None:
        lighting_record = apply_lighting_overrides(
            request, json.loads(args.lighting.read_text()))
        lighting_record["source"] = str(args.lighting.resolve())
        lighting_record["source_sha256"] = digest(args.lighting)
    request_path = out / "render_request.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    if args.request_only:
        # Everything above is cheap; the worker is not. Stop here when the point
        # of the run was to find out whether the dressing is admissible at all.
        (out / "provenance_partial.json").write_text(json.dumps(
            {"claim": "request_only: no render was run",
             "figure_request_sha256": digest(request_path),
             "layout_overrides": layout_record, "additions": additions_record,
             "asset_replacement": replacement_record}, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output_dir": str(out), "request_only": True,
                          "instances": len(request["instances"]),
                          "additions": ([r["instance_id"] for r in additions_record["added"]]
                                        if additions_record else [])}, indent=2))
        return 0
    command = [sys.executable, "-m", 'rfw_w.scenes.workers.launch',
               "--request", str(request_path), "--output-dir", str(out / "isaac"),
               "--active-gpu", str(args.active_gpu), "--timeout", str(args.timeout)]
    result = subprocess.run(command, check=False)
    report_path = out / "isaac" / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"PAPER_PREVIEW_RENDER_MISSING: worker exit={result.returncode}")
    report = json.loads(report_path.read_text())
    target_ids = sorted({i["instance_id"] for i in request["instances"]
                         if i.get("entity_id") in protected_ids
                         or i["instance_id"] in protected_ids})
    metrics = collect_render_metrics(
        report_path, target_ids,
        thresholds=DOLLHOUSE_METRIC_THRESHOLDS if args.view == "dollhouse" else None)
    collision_record = None
    collision_ok = True
    if args.with_collision:
        variant, substitutions = collision_request(request)
        collision_request_path = out / "render_request_collision.json"
        collision_request_path.write_text(json.dumps(variant, indent=2, sort_keys=True) + "\n")
        collision_out = out / "isaac_collision"
        collision_command = [sys.executable, "-m", 'rfw_w.scenes.workers.launch',
                             "--request", str(collision_request_path),
                             "--output-dir", str(collision_out),
                             "--active-gpu", str(args.active_gpu),
                             "--timeout", str(args.timeout)]
        started = time.monotonic()
        collision_result = subprocess.run(collision_command, check=False)
        wall_seconds = time.monotonic() - started
        collision_report_path = collision_out / "report.json"
        if not collision_report_path.is_file():
            raise SystemExit(f"PAPER_PREVIEW_RENDER_MISSING(collision): "
                             f"worker exit={collision_result.returncode}")
        collision_report = json.loads(collision_report_path.read_text())
        collision_metrics = collect_render_metrics(
            collision_report_path, target_ids,
            thresholds=DOLLHOUSE_METRIC_THRESHOLDS if args.view == "dollhouse" else None)
        # The collision pass reuses the same camera id, so it is a sub-dict rather
        # than another entry in "cameras": the two passes are never merged.
        metrics["collision"] = collision_metrics
        collision_files = {}
        for capture in collision_report.get("captures", []):
            entry = {}
            for key in ("rgb_path", "depth_path", "instance_id_path"):
                produced = collision_out / capture[key] if capture.get(key) else None
                if produced is not None and produced.is_file():
                    entry[key] = digest(produced)
            collision_files[capture["camera_id"]] = entry
        collision_record = {
            "request_sha256": digest(collision_request_path),
            "report_path": "isaac_collision/report.json",
            "report_sha256": digest(collision_report_path),
            "worker_status": collision_report.get("status"),
            "worker_exit_code": collision_result.returncode,
            "wall_seconds": wall_seconds,
            "substitutions": {
                "substituted": sum(1 for s in substitutions if s["substituted"]),
                "kept_visual": sorted(s["instance_id"] for s in substitutions
                                      if not s["substituted"]),
                "records": substitutions,
            },
            "files": collision_files,
            "render_metrics_gaps": collision_metrics["gaps"],
        }
        collision_ok = (collision_result.returncode == 0
                        and collision_report.get("status") == "passed")
    (out / "render_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    provenance = {
        "source_delivery": str(delivery),
        "source_scene_version": manifest["scene_version"],
        "source_request_sha256": digest(source_request),
        "source_shell_sha256": digest(Path(request["room"]["shell_uri"])),
        "figure_request_sha256": digest(request_path),
        "report_sha256": digest(report_path),
        "capture_state": report.get("capture_state"),
        "worker_status": report.get("status"),
        "worker_exit_code": result.returncode,
        "claim": "figure_only_static_render",
        "view": args.view,
        "overview_corner_requested": args.overview_corner if args.view == "oblique" else None,
        "overview_corner_used": chosen_corner,
        # Only the oblique view stands inside the plan; the cutaway level camera
        # and the dollhouse camera both stand outside it by design.
        "vantage_inside_room": args.view == "oblique",
        "dollhouse": ({"corner_requested": args.dollhouse_corner,
                       "corner_used": shell_record["camera"]["corner"],
                       "shell_mode": args.shell_mode,
                       "elevation_deg": args.dollhouse_elevation_deg,
                       "wall_clip_height_m": (args.wall_clip_height
                                              if args.wall_clip_height > 0 else None),
                       "wall_clip_corner": args.wall_clip_corner,
                       "frame_fit": args.dollhouse_fit,
                       "wall_clip": shell_record.get("wall_clip"),
                       "camera": shell_record["camera"]}
                      if args.view == "dollhouse" else None),
        "cutaway": shell_record,
        "layout_overrides": layout_record,
        "additions": additions_record,
        "asset_replacement": replacement_record,
        "workspace_detail": workspace_record,
        "lighting_override": lighting_record,
        "render_metrics_gaps": metrics["gaps"],
        "collision_pass": collision_record,
    }
    if shell_record is not None:
        # `request` now points at the figure copy, so the digest above would
        # otherwise be filed as the source shell's. Re-digest the real one.
        original_uri = json.loads(source_request.read_text())["room"]["shell_uri"]
        provenance["source_shell_sha256"] = digest(
            (source_request.parent / original_uri).resolve())
        # The cut shell and any figure-only furniture substitution are recorded
        # separately; neither edit changes the source delivery.
        if shell_record.get("mode") == "roof":
            clip = shell_record.get("wall_clip")
            if clip:
                achieved = clip["achieved_max_kept_z_m"]
                note = ("only shell faces were removed: the ceiling, the ceiling skirting and "
                        "the outer facade. " +
                        (f"the two walls nearest the {clip['corner']} corner are clipped to a "
                         f"low stub (requested {clip['height_m']} m, achieved "
                         f"{achieved} m); " if achieved is not None else
                         f"the two walls nearest the {clip['corner']} corner were dropped to "
                         f"the floor: every face straddled the requested {clip['height_m']} m "
                         "clip, and a straddling face is dropped with the cut. ") +
                        "the far walls stand at full height. "
                        "Instance geometry substitutions, if any, are recorded separately.")
            else:
                note = ("only shell faces were removed: the ceiling, the ceiling skirting and "
                        "the outer facade. Every wall stands at full height. "
                        "Instance geometry substitutions, if any, are recorded separately.")
        else:
            note = ("only shell faces were removed: the ceiling, the outer facade and the "
                    "near long wall. Every placed instance is untouched, and no instance "
                    "is hidden by the cut.")
        provenance["shell_substitution"] = {
            "source_shell": shell_record["source_shell"],
            "figure_shell": shell_record["cutaway_shell"],
            "mode": shell_record.get("mode", "cutaway"),
            "open_side": shell_record["open_side"],
            "wall_clip": shell_record.get("wall_clip"),
            "dropped_roles": shell_record["dropped_roles"],
            "faces_in": shell_record["faces_in"],
            "faces_kept": shell_record["faces_kept"],
            "instances_affected": [],
            "note": note,
        }
        provenance["figure_shell_sha256"] = digest(Path(shell_record["cutaway_shell"]))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(out), "status": report.get("status"),
                      "cameras": [x.get("camera_id") for x in report.get("captures", [])],
                      "render_metrics_gaps": metrics["gaps"],
                      "collision": ({"status": collision_record["worker_status"],
                                     "render_metrics_gaps": collision_record["render_metrics_gaps"]}
                                    if collision_record else None)}, indent=2))
    passed = result.returncode == 0 and report.get("status") == "passed" and collision_ok
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
