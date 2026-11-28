#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Show what is already on a support, and where the free spots are.

SPEC-016 authoring aid. Placing a dozen props by reading coordinates out of a
scene_ir and doing the arithmetic by hand costs one failed render per mistake;
this prints the occupied boxes on a surface and a coarse free-cell map instead,
so a declaration can be written once and pass.

Read-only: it applies the layout overrides to a copy of the delivery's request
in memory and prints. It renders nothing and writes nothing.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.visualization.show_free_surface \\
      --delivery outputs/office_pick_cup/RUN --support desk \\
      --layout-overrides experiments/paper_dollhouse/layout_office.json --clearance 0.06
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rfw_w.scenes.visualization.render_room_preview import (
    apply_layout_overrides, local_footprint, world_aabb, yaw_of,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--support", required=True,
                        help="instance id of the surface, or 'floor'")
    parser.add_argument("--layout-overrides", type=Path, default=None)
    parser.add_argument("--clearance", type=float, default=0.06,
                        help="a free cell must be this far from every occupied box")
    parser.add_argument("--cell", type=float, default=0.05)
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    request = json.loads((delivery / manifest["isaac_request_path"]).read_text())
    scene = json.loads((delivery / "scene_ir.json").read_text())
    bounds = {i["instance_id"]: (i["bounds_min_m"], i["bounds_max_m"])
              for i in scene.get("instances", []) if "bounds_min_m" in i}
    if args.layout_overrides:
        payload = json.loads(args.layout_overrides.read_text())
        apply_layout_overrides(request, payload["overrides"], bounds_by_instance=bounds,
                               protected_ids=set())

    boxes = {}
    for instance in request["instances"]:
        instance_id = instance["instance_id"]
        if instance_id not in bounds:
            continue
        bmin, bmax = bounds[instance_id]
        offset, half, yaw = local_footprint(instance, (bmin, bmax))
        translation = [float(v) for v in instance["translation"]]
        lo, hi = world_aabb(offset, half, translation, yaw)
        boxes[instance_id] = (lo, hi, translation[2] + float(bmin[2]),
                              translation[2] + float(bmax[2]), yaw_of(instance["rotation_wxyz"]))

    if args.support == "floor":
        polygon = request["room"]["polygon_xy"]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        surface = ([min(xs), min(ys)], [max(xs), max(ys)])
        top = float(request["room"].get("floor_z_m", 0.0))
    else:
        if args.support not in boxes:
            raise SystemExit(f"UNKNOWN_SUPPORT({args.support})")
        lo, hi, _, top, _ = boxes[args.support]
        surface = (lo, hi)

    print(f"support {args.support}: x[{surface[0][0]:.3f}, {surface[1][0]:.3f}] "
          f"y[{surface[0][1]:.3f}, {surface[1][1]:.3f}] top z={top:.3f}")
    print("occupied (boxes whose z range touches the surface):")
    occupied = []
    for instance_id, (lo, hi, z_lo, z_hi, yaw) in sorted(boxes.items()):
        if instance_id == args.support:
            continue
        if z_hi < top - 0.02 or z_lo > top + 1.2:
            continue
        if hi[0] < surface[0][0] or lo[0] > surface[1][0]:
            continue
        if hi[1] < surface[0][1] or lo[1] > surface[1][1]:
            continue
        occupied.append((lo, hi))
        print(f"  {instance_id:16s} x[{lo[0]:.3f}, {hi[0]:.3f}] y[{lo[1]:.3f}, {hi[1]:.3f}] "
              f"z[{z_lo:.3f}, {z_hi:.3f}] yaw={yaw}")

    print(f"free cells ({args.cell} m grid, {args.clearance} m clear of everything above):")
    rows = []
    y = surface[0][1] + args.cell / 2.0
    while y < surface[1][1]:
        row = []
        x = surface[0][0] + args.cell / 2.0
        while x < surface[1][0]:
            free = all(not (lo[0] - args.clearance <= x <= hi[0] + args.clearance
                            and lo[1] - args.clearance <= y <= hi[1] + args.clearance)
                       for lo, hi in occupied)
            row.append((x, y, free))
            x += args.cell
        rows.append(row)
        y += args.cell
    for row in reversed(rows):
        print(f"  y={row[0][1]:5.2f} " + "".join("." if free else "#" for _, _, free in row))
    print("  x=     " + "".join(str(int(round(x * 10)) % 10) for x, _, _ in rows[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
