#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Draw a whole-delivery overview from the manifest, with nothing left out of frame.

    python -m rfw_w.scenes.visualization.render_manifest_overview --delivery outputs/office_pick_cup/m1_005

The delivered renders are photographs: they show what a camera pointed at, and a camera aimed at
the task will not cover the room. Measured on `m1_004`, four of the twelve placed instances --
`floor_lamp`, `plant`, `wall_art` and `ceiling_light` -- had zero pixels in all five delivered
views, so no view-dependent gate and no visual critique could ever have seen them. A plan drawn
from the manifest has no such blind spot: every placed instance is on it by construction. (That
gap is now closed in the pipeline itself, by the two overview cameras; this figure remains the
view that is complete by construction rather than by aiming.)

Everything here is read from the recorded pose. Each piece's outline and height range come from
`position_m`, `rotation_wxyz` and `origin_at_base` through the same `Footprint.placed_extent` the
geometry gates use, so the figure and the gates cannot disagree. Deliveries written before the
pose carried a full rotation fall back to the yaw and a base-origin assumption.

That last point is why this tool is worth keeping honest: an earlier version reconstructed the
vertical origin from a rule of thumb, because the manifest did not say. It drew the wall art as a
flat slab and the pendant from a centre origin, both of which were the actual defects, and the
figure agreed with them instead of exposing them.

Labels are placed greedily against a candidate ring of offsets, so a crowded corner pushes its
labels outward instead of stacking them. Nothing is dropped: an instance that cannot be labelled
cleanly still gets a row in the table on the right.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


from rfw_w.scenes.contracts.units import Quat, Vec3
from rfw_w.scenes.solvers.layout import Footprint, PlacedExtent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle

# One colour per zone kind. Chosen to stay distinguishable and to read as what the zone is for:
# the work zone warm, the display zone cool, circulation the only hatched one.
ZONE_COLOURS = {
    "work": "#f4a261",
    "seating": "#f28ab2",
    "storage": "#b39ddb",
    "display": "#4db6ac",
    "ambient": "#d9d9d9",
    "circulation": "#90caf9",
}
FALLBACK_COLOUR = "#bdbdbd"
# The task target is the instance the request is about; it gets its own colour and a marker so it
# cannot be lost among twelve rectangles.
TARGET_COLOUR = "#d32f2f"
LABEL_FONTSIZE = 6.5
# Offsets tried in order, nearest first: above, below, right, left, then the diagonals, then a
# little further out. Kept short on purpose -- a label pushed far from its piece stops identifying
# it, so a displaced label gets a leader line rather than a larger offset.
CANDIDATE_OFFSETS = (
    (0, 11), (0, -17), (30, 3), (-30, 3),
    (30, 15), (-30, 15), (30, -18), (-30, -18),
    (0, 22), (0, -28), (40, 5), (-40, 5), (0, 32), (0, -34),
)
LEADER_BEYOND_POINTS = 20.0


def _placed_extent(name: str, placement: dict, dimensions) -> "PlacedExtent":
    """Where a piece really is, from its own recorded pose.

    Read from the manifest's `rotation_wxyz` and `origin_at_base` where they are present, and from
    the yaw and a base-origin assumption where they are not -- deliveries written before the pose
    carried a full rotation.

    This used to be a local yaw-and-centre reconstruction, and the module docstring used to argue
    at length about how to read the vertical origin from a rule of thumb. It drew the wall art as
    a flat slab because that is what the yaw said, and it drew the pendant from a centre origin
    because that is what the convention was assumed to be. Both were wrong, and both are now
    answered by the pose rather than guessed at.
    """
    width, depth, height = dimensions
    position = Vec3(*placement["position_m"])
    recorded = placement.get("rotation_wxyz")
    rotation = (
        Quat(*recorded) if recorded
        else Quat.from_yaw(placement.get("yaw_rad", 0.0))
    )
    convention = "base" if placement.get("origin_at_base", True) else "centre"
    return Footprint((width / 2.0, depth / 2.0)).placed_extent(
        height, rotation=rotation, position=position, origin_convention=convention
    )


def _corners(x: float, y: float, yaw: float, width: float, depth: float):
    half_x, half_y = width / 2.0, depth / 2.0
    cos, sin = math.cos(yaw), math.sin(yaw)
    return [
        (x + local_x * cos - local_y * sin, y + local_x * sin + local_y * cos)
        for local_x, local_y in (
            (half_x, half_y), (-half_x, half_y), (-half_x, -half_y), (half_x, -half_y)
        )
    ]


def _place_labels(axis, target_figure, entries) -> None:
    """Write each label at the first candidate offset that does not overlap one already placed.

    A dense room cannot be labelled by a fixed offset: twelve pieces in 14 m2 put several labels on
    the same spot. Measuring each candidate against what is already on the figure and rejecting the
    collisions costs a little work and is the difference between a diagram and a pile of text.
    """
    occupied: list = []
    renderer = target_figure.canvas.get_renderer()

    def _draw(x, y, text, colour, weight, offset):
        # A leader line appears only once the label has been pushed off its piece, so the short
        # offsets stay clean and a displaced label still says which piece it belongs to.
        far = math.hypot(*offset) > LEADER_BEYOND_POINTS
        return axis.annotate(
            text, (x, y), textcoords="offset points", xytext=offset,
            ha="center", va="bottom", fontsize=LABEL_FONTSIZE, color=colour,
            fontweight=weight, zorder=12,
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": colour,
                        "shrinkA": 1.0, "shrinkB": 2.0} if far else None,
        )

    for x, y, text, colour, weight in entries:
        placed = None
        for offset in CANDIDATE_OFFSETS:
            annotation = _draw(x, y, text, colour, weight, offset)
            box = annotation.get_window_extent(renderer).expanded(1.22, 1.75)
            if not any(box.overlaps(other) for other in occupied):
                occupied.append(box)
                placed = annotation
                break
            annotation.remove()
        if placed is None:
            # Every candidate collided. Leaving the label off would silently hide an instance, so
            # the last offset is used anyway: overlapping text can still be read, absent text
            # cannot.
            annotation = _draw(x, y, text, colour, weight, CANDIDATE_OFFSETS[-1])
            occupied.append(annotation.get_window_extent(renderer).expanded(1.22, 1.75))


def _draw_plan(axis, manifest: dict, request: dict | None) -> None:
    zones = manifest["zones"]
    room = [tuple(vertex) for vertex in zones["room_polygon_xy"]]
    # Two patches: an opaque fill under the zones, and the outline kept until last so the wall
    # reads as a wall and not as something the zones were painted over.
    room_fill = MplPolygon(room, closed=True, facecolor="white", edgecolor="none", zorder=1)
    room_outline = MplPolygon(room, closed=True, facecolor="none", edgecolor="black",
                              linewidth=2.2, zorder=11)
    axis.add_patch(room_fill)
    axis.add_patch(room_outline)

    for zone in zones["zones"]:
        colour = ZONE_COLOURS.get(zone["kind"], FALLBACK_COLOUR)
        for polygon in zone["polygons_xy"]:
            patch = MplPolygon(polygon, closed=True, facecolor=colour, edgecolor="none",
                               alpha=0.50, hatch="///" if zone["keepout"] else None, zorder=3)
            axis.add_patch(patch)

    if request:
        for opening in request["room"]["openings"]:
            x, y = opening["center_xy"]
            half = opening["width_m"] / 2.0
            # An opening lies on whichever wall it is nearest; drawn as a thick segment so a door
            # reads differently from a window.
            on_vertical_wall = math.isclose(x, 0.0, abs_tol=1e-6) or math.isclose(
                x, max(vertex[0] for vertex in room), abs_tol=1e-6)
            span = [(x, y - half), (x, y + half)] if on_vertical_wall else [(x - half, y), (x + half, y)]
            axis.plot(*zip(*span), color="#37474f", linewidth=8, solid_capstyle="butt", zorder=5)
            axis.plot(*zip(*span), color="#bbdefb" if opening["kind"] == "window" else "#8d6e63",
                      linewidth=3.5, solid_capstyle="butt", zorder=6)

        # Eyes and aim only. A frustum wedge was drawn here first and taken out: every delivered
        # lens is wide (14-35 mm) and every eye is inside the room, so the wedges blanketed the
        # whole floor and the figure asserted a coverage the renders do not have -- four instances
        # appear in none of the five views. Coverage is measured from the delivered id channels;
        # a picture of the frusta cannot establish it, so it is not drawn here.
        for camera in request["cameras"]:
            eye, aim = camera["eye"], camera["target"]
            forward = math.atan2(aim[1] - eye[1], aim[0] - eye[0])
            axis.plot([eye[0]], [eye[1]], marker="v", color="#2962ff", markersize=5, zorder=8)
            axis.annotate("", xy=(eye[0] + 0.5 * math.cos(forward), eye[1] + 0.5 * math.sin(forward)),
                          xytext=(eye[0], eye[1]),
                          arrowprops={"arrowstyle": "-|>", "color": "#2962ff", "lw": 1.1}, zorder=8)
            axis.annotate(camera["camera_id"], (eye[0], eye[1]), textcoords="offset points",
                          xytext=(0, -13), ha="center", fontsize=5.5, color="#2962ff", zorder=8)

    dimensions = {asset["entity_id"]: asset["dimensions_m"] for asset in manifest["assets_used"]}
    instances = manifest["placement"]["instances"]
    target = manifest.get("targets") or manifest.get("task", {}).get("targets") or ["cup"]

    labels = []
    for name, placement in instances.items():
        x, y, _ = placement["position_m"]
        yaw = placement["yaw_rad"]
        width, depth, _ = dimensions[name]
        is_target = name in target
        extent = _placed_extent(name, placement, dimensions[name])
        axis.add_patch(MplPolygon(
            [(vertex[0], vertex[1]) for vertex in extent.outline.vertices], closed=True,
            facecolor="#ffcdd2" if is_target else "#ffffff",
            edgecolor=TARGET_COLOUR if is_target else "#212121",
            linewidth=2.0 if is_target else 1.2, zorder=9,
        ))
        # The front axis is local +y, so this arrow shows which way a piece faces. Skipped for a
        # piece that is not upright: a hung panel's local +y is world up, and its plan projection
        # is a point, so the arrow would be a zero-length stub pointing nowhere.
        front = (-math.sin(yaw), math.cos(yaw))
        head = (x + front[0] * depth * 0.5, y + front[1] * depth * 0.5)
        if extent.z_high_m - extent.z_low_m <= max(width, depth) + 1e-6:
            axis.annotate("", xy=head, xytext=(x, y),
                          arrowprops={"arrowstyle": "-|>", "color": "#212121", "lw": 0.9},
                          zorder=10)
        axis.plot([x], [y], marker="*" if is_target else ".", markersize=13 if is_target else 2.5,
                  color=TARGET_COLOUR if is_target else "#212121", zorder=11)
        labels.append((x, y, name, TARGET_COLOUR if is_target else "#212121",
                       "bold" if is_target else "normal"))

    xs = [vertex[0] for vertex in room]
    ys = [vertex[1] for vertex in room]
    axis.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    axis.set_ylim(min(ys) - 0.3, max(ys) + 0.3)
    axis.set_aspect("equal")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(f"plan (top view) -- all {len(instances)} placed instances\n"
                   f"every outline is the box the piece was placed as, from its own rotation and origin",
                   fontsize=9)
    _place_labels(axis, axis.figure, labels)


def _draw_elevation(axis, manifest: dict, request: dict | None) -> None:
    """x against z, seen from the doorway wall. Shows the heights a plan cannot."""
    zones = manifest["zones"]
    room = [tuple(vertex) for vertex in zones["room_polygon_xy"]]
    left = min(vertex[0] for vertex in room)
    right = max(vertex[0] for vertex in room)
    far_y = max(vertex[1] for vertex in room)
    height = (request["room"]["height_m"] if request else
              manifest["room"]["validation"]["measured"]["height_m"])

    axis.add_patch(Rectangle((left, 0.0), right - left, height, facecolor="#fafafa",
                             edgecolor="black", linewidth=2.0, zorder=4))
    if request:
        for opening in request["room"]["openings"]:
            x, y = opening["center_xy"]
            # Only openings on the near or far wall project onto an x-z elevation; the window on
            # the side wall would need its own elevation and is named in the table instead.
            if not (math.isclose(y, 0.0, abs_tol=1e-6) or math.isclose(y, far_y, abs_tol=1e-6)):
                continue
            axis.add_patch(Rectangle((x - opening["width_m"] / 2.0, opening["sill_m"]),
                                     opening["width_m"], opening["height_m"],
                                     facecolor="#bbdefb", edgecolor="#1565c0", linewidth=1.2, zorder=5))
            axis.annotate(f"{opening['opening_id']}\n{opening['width_m']:.2f}x{opening['height_m']:.2f} m",
                          (x, opening["sill_m"] + opening["height_m"]), textcoords="offset points",
                          xytext=(0, 3), ha="center", fontsize=6, color="#1565c0", zorder=8)

    support = manifest["placement"]["support_surface"]
    support_xs = [vertex[0] for vertex in support["polygon_xy"]]
    axis.plot([min(support_xs), max(support_xs)], [support["plane_z_m"]] * 2,
              color="#6a1b9a", linewidth=2.0, zorder=6)
    axis.annotate(f"support plane z={support['plane_z_m']:.3f} m", (max(support_xs), support["plane_z_m"]),
                  textcoords="offset points", xytext=(4, 2), fontsize=6, color="#6a1b9a", zorder=8)

    dimensions = {asset["entity_id"]: asset["dimensions_m"] for asset in manifest["assets_used"]}
    target = manifest.get("targets") or manifest.get("task", {}).get("targets") or ["cup"]
    labels = []
    for name, placement in manifest["placement"]["instances"].items():
        is_target = name in target
        extent = _placed_extent(name, placement, dimensions[name])
        projected = [vertex[0] for vertex in extent.outline.vertices]
        bottom, top = extent.z_low_m, extent.z_high_m
        axis.add_patch(Rectangle((min(projected), bottom), max(projected) - min(projected), top - bottom,
                                 facecolor="#ffcdd2" if is_target else "#ffffff",
                                 edgecolor=TARGET_COLOUR if is_target else "#212121",
                                 linewidth=1.8 if is_target else 1.0, zorder=7))
        labels.append(((min(projected) + max(projected)) / 2.0, top, name,
                       TARGET_COLOUR if is_target else "#212121", "bold" if is_target else "normal"))

    axis.set_xlim(left - 0.15, right + 0.15)
    axis.set_ylim(-0.1, height + 0.2)
    axis.set_aspect("equal")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_title("elevation, seen from the doorway wall\n"
                   "heights from each piece's own origin convention, not from a rule of thumb",
                   fontsize=9)
    _place_labels(axis, axis.figure, labels)


def _draw_table(axis, manifest: dict, request: dict | None) -> None:
    """The zone legend and every instance, as text. The panel that cannot lose an instance."""
    axis.set_axis_off()
    zones = manifest["zones"]
    dimensions = {asset["entity_id"]: asset["dimensions_m"] for asset in manifest["assets_used"]}
    instances = manifest["placement"]["instances"]
    target = manifest.get("targets") or manifest.get("task", {}).get("targets") or ["cup"]

    row = 0
    axis.text(0.0, 0.99, "zones", fontsize=9, fontweight="bold", va="top", family="monospace")
    row += 1
    seen: dict[str, float] = {}
    for zone in zones["zones"]:
        seen[zone["kind"]] = seen.get(zone["kind"], 0.0) + zone["area_m2"]
    for kind, area in seen.items():
        axis.add_patch(Rectangle((0.0, 0.925 - row * 0.030), 0.055, 0.020,
                                 facecolor=ZONE_COLOURS.get(kind, FALLBACK_COLOUR), alpha=0.55,
                                 edgecolor="none", transform=axis.transAxes, clip_on=False))
        axis.text(0.08, 0.925 - row * 0.030 + 0.010, f"{kind:12s} {area:6.2f} m2",
                  fontsize=7.5, va="center", family="monospace")
        row += 1
    axis.text(0.0, 0.925 - row * 0.030, f"placeable {zones['placeable_area_m2']:6.2f} m2 of "
              f"{zones['room_area_m2']:6.2f} m2   target {zones['coverage_target']:.2f}",
              fontsize=7.5, va="top", family="monospace")
    row += 2

    axis.text(0.0, 0.925 - row * 0.030, "instances", fontsize=9, fontweight="bold", va="top",
              family="monospace")
    row += 1
    axis.text(0.0, 0.925 - row * 0.030,
              f"{'name':13s}{'W x D x H (m)':>20s}   position (x, y, z) m",
              fontsize=7, va="top", family="monospace", color="#555555")
    row += 1
    for name, placement in instances.items():
        width, depth, piece_height = dimensions[name]
        x, y, z = placement["position_m"]
        mark = "*" if name in target else " "
        axis.text(0.0, 0.925 - row * 0.030,
                  f"{mark}{name:12s}{width:6.2f}x{depth:5.2f}x{piece_height:5.2f}   "
                  f"({x:6.3f},{y:6.3f},{z:6.3f})",
                  fontsize=7, va="top", family="monospace",
                  color=TARGET_COLOUR if name in target else "#212121",
                  fontweight="bold" if name in target else "normal")
        row += 1

    if request:
        row += 1
        axis.text(0.0, 0.925 - row * 0.030, "openings", fontsize=9, fontweight="bold", va="top",
                  family="monospace")
        row += 1
        for opening in request["room"]["openings"]:
            axis.text(0.0, 0.925 - row * 0.030,
                      f"{opening['opening_id']:13s}{opening['kind']:7s}{opening['width_m']:5.2f} x "
                      f"{opening['height_m']:4.2f} m  sill {opening['sill_m']:.2f}",
                      fontsize=7, va="top", family="monospace")
            row += 1
        row += 1
        axis.text(0.0, 0.925 - row * 0.030, "delivered views", fontsize=9, fontweight="bold",
                  va="top", family="monospace")
        row += 1
        for camera in request["cameras"]:
            eye = camera["eye"]
            axis.text(0.0, 0.925 - row * 0.030,
                      f"{camera['camera_id']:15s}{camera['focal_length_mm']:4.0f}mm  eye "
                      f"({eye[0]:5.2f},{eye[1]:5.2f},{eye[2]:5.2f})",
                      fontsize=7, va="top", family="monospace")
            row += 1

    axis.text(0.0, 0.925 - (row + 1) * 0.030,
              "* marks the task target\n"
              "scene_id is the spec's, and is the same for\n"
              "every run: the delivery directory is what\n"
              "identifies the run.",
              fontsize=6.5, va="top", family="monospace", color="#555555")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery", type=Path, required=True,
                        help="a delivery directory containing delivery_manifest.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the figure; defaults to <delivery>/overview")
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    if not (delivery / "delivery_manifest.json").exists():
        parser.error(f"no delivery_manifest.json under {delivery}")

    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    if "zones" not in manifest:
        # m1_001 predates the zone plan. Say so, rather than failing on a missing key halfway
        # through drawing and leaving the reader to work out which delivery is too old.
        parser.error(
            f"{delivery.name} has no `zones` section: it predates the zone plan (WP15), so it "
            f"cannot be drawn from the manifest. The earliest drawable delivery is m1_002."
        )
    out = (args.out or (delivery / "overview")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    request_path = delivery / (manifest.get("isaac_request_path") or "isaac_request.json")
    request = json.loads(request_path.read_text()) if request_path.exists() else None

    figure = plt.figure(figsize=(19.5, 8.4))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.42, 1.02, 0.62), wspace=0.16,
                               left=0.035, right=0.985, top=0.90, bottom=0.07)
    plan = figure.add_subplot(grid[0, 0])
    elevation = figure.add_subplot(grid[0, 1])
    table = figure.add_subplot(grid[0, 2])

    _draw_plan(plan, manifest, request)
    _draw_elevation(elevation, manifest, request)
    _draw_table(table, manifest, request)

    figure.suptitle(
        f"{delivery.name}  --  {manifest['achieved_readiness']}  "
        f"(meets_request={manifest['meets_request']}, {len(manifest['blockers'])} blockers)   "
        f"scene_id={manifest['scene_id']}",
        fontsize=11,
    )

    path = out / "overview.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
