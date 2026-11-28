#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lay out a room's dressing by rule, instead of by hand.

SPEC-016. Dressing the first three rooms took a dozen iterations each: place a
prop, watch ADDITION_OVERLAP refuse it, move it. That does not scale to ten
scenes, and hand-picked coordinates are not a rule anyone can check.

So this writes the additions declaration from one: substitute the shared work
table, seat the delivery's own task objects on its new top, stand the arm at the
end of the table furthest from the target, and drop the declared props onto the
free cells of the top -- the same free-cell scan `python -m rfw_w.scenes.visualization.show_free_surface`
prints, at the same clearance. The output is an ordinary additions file, judged
by the ordinary gates; nothing here bypasses them.

What it does not decide: which props, or what the shell colours are. Those are
declarations, and they stay in the caller's hands.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.visualization.plan_room_dressing \\
      --delivery outputs/paper_embodiment_matrix/pm01_office_pick_cup/run01 \\
      --props 4_3_0012 29_9_0019 6_0_0028 5_0_0003 25_25_0020 \\
      --out-additions experiments/paper_embodiment_matrix/pm01_office_pick_cup/additions.json \\
      --out-layout experiments/paper_embodiment_matrix/pm01_office_pick_cup/layout.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from rfw_w.scenes.visualization.make_paper_table import TABLE_SIZE_M

#: The arm's base box and the radius its tool sweeps, from `robot_asset`'s bake of
#: `pose_piper_ready` -- half-extents 0.2275 x 0.082 m, tool point 0.500 m out.
ARM_HALF_M = (0.2275, 0.082)
ARM_REACH_M = 0.500
ARM_TCP_BEARING_DEG = -10.2
#: Free-cell scan, matching python -m rfw_w.scenes.visualization.show_free_surface's defaults.
CELL_M = 0.02
CLEARANCE_M = 0.05


def prepared_bounds(leaf: str, cache: Path) -> tuple[list[float], list[float], dict]:
    from rfw_w.scenes.assets.providers.simready import SimreadyPrepareConfig, prepare_rigid

    index = json.loads(Path(os.environ["PVSCENE_SIMREADY_INDEX"]).read_text())
    record = index[leaf]
    config = SimreadyPrepareConfig(assets_root=Path(os.environ["PVSCENE_SIMREADY_ASSETS_ROOT"]),
                                   prepared_root=cache)
    prepared = prepare_rigid(config, prefix=record["assets_prefix"], leaf=leaf, asset_id=leaf)
    return (prepared.bounds_min_m.to_list(), prepared.bounds_max_m.to_list(),
            {"super_category": record.get("super_category"),
             "dimensions_m": prepared.dimensions_m.to_list()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--props", nargs="*", default=[])
    parser.add_argument("--prop-cache", type=Path, default=Path("outputs/_props_cache"))
    parser.add_argument("--table-size", nargs=3, type=float, default=list(TABLE_SIZE_M))
    parser.add_argument("--wall-colour", nargs=3, type=float, default=[0.86, 0.80, 0.70])
    parser.add_argument("--floor-colour", nargs=3, type=float, default=[0.40, 0.27, 0.17])
    parser.add_argument("--out-additions", required=True, type=Path)
    parser.add_argument("--out-layout", required=True, type=Path)
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    request = json.loads((delivery / manifest["isaac_request_path"]).read_text())
    scene = json.loads((delivery / "scene_ir.json").read_text())
    bounds = {i["instance_id"]: (i["bounds_min_m"], i["bounds_max_m"])
              for i in scene.get("instances", []) if "bounds_min_m" in i}
    plan = manifest.get("plan", {})
    support_id, target_id = plan.get("support_id"), plan.get("target_id")
    if not support_id or not target_id:
        raise SystemExit(f"DRESSING_PLAN_INCOMPLETE(support={support_id} target={target_id})")
    by_id = {i["instance_id"]: i for i in request["instances"]}
    support, target = by_id[support_id], by_id[target_id]

    width, depth, height = (float(v) for v in args.table_size)
    cx, cy, cz = (float(v) for v in support["translation"])
    top_z = cz + height
    # The table is authored about the support's own pose, axis-aligned: the
    # delivered supports in these rooms stand square to the room, and a yawed
    # table would need the whole free-cell scan rotated with it.
    x_lo, x_hi = cx - width / 2, cx + width / 2
    y_lo, y_hi = cy - depth / 2, cy + depth / 2

    # Everything the delivery put on this support, re-seated on the new top.
    layout: dict[str, dict] = {}
    seated: list[dict] = []
    for instance_id, instance in by_id.items():
        if instance_id in (support_id,) or instance_id not in bounds:
            continue
        bmin, bmax = bounds[instance_id]
        base = float(instance["translation"][2]) + float(bmin[2])
        old_top = cz + float(bounds[support_id][1][2])
        if abs(base - old_top) > 0.02:
            continue        # not on this support
        seated.append({"instance_id": instance_id, "half": [(float(bmax[0]) - float(bmin[0])) / 2,
                                                            (float(bmax[1]) - float(bmin[1])) / 2],
                       "lift": -float(bmin[2])})

    occupied: list[tuple[float, float, float, float]] = []

    def reserve(px: float, py: float, hx: float, hy: float) -> None:
        occupied.append((px - hx, py - hy, px + hx, py + hy))

    def free(px: float, py: float, hx: float, hy: float) -> bool:
        if not (x_lo + CLEARANCE_M <= px - hx and px + hx <= x_hi - CLEARANCE_M):
            return False
        if not (y_lo + CLEARANCE_M <= py - hy and py + hy <= y_hi - CLEARANCE_M):
            return False
        for ax, ay, bx, by in occupied:
            if (px - hx < bx + CLEARANCE_M and px + hx > ax - CLEARANCE_M
                    and py - hy < by + CLEARANCE_M and py + hy > ay - CLEARANCE_M):
                return False
        return True

    def place(hx: float, hy: float) -> tuple[float, float] | None:
        """First free cell scanning from the target's end of the table outward."""
        steps_x = int((x_hi - x_lo) / CELL_M)
        steps_y = int((y_hi - y_lo) / CELL_M)
        best = None
        for iy in range(steps_y + 1):
            for ix in range(steps_x + 1):
                px, py = x_lo + ix * CELL_M, y_lo + iy * CELL_M
                if free(px, py, hx, hy):
                    score = -(px - x_lo)      # fill from the low-x end first
                    if best is None or score > best[0]:
                        best = (score, px, py)
            if best is not None:
                break
        return None if best is None else (best[1], best[2])

    # 1. the task target keeps the end of the table nearest where it already is.
    target_half = [(float(bounds[target_id][1][i]) - float(bounds[target_id][0][i])) / 2
                   for i in (0, 1)]
    target_x = x_lo + max(target_half[0], 0.06) + CLEARANCE_M
    target_y = (y_lo + y_hi) / 2
    reserve(target_x, target_y, target_half[0], target_half[1])
    layout[target_id] = {"translation": [round(target_x, 4), round(target_y, 4),
                                         round(top_z - float(bounds[target_id][0][2]), 4)],
                         "on": support_id}

    # 2. the arm at the far end, turned so its tool point lands on the target.
    # 0.10 m in from the edge, not flush with it: placed flush, the base-to-target
    # distance came out at 0.803 m against a measured 0.8027 m reach -- inside the
    # gate by a millimetre is not a margin, it is luck.
    arm_x = x_hi - ARM_HALF_M[0] - CLEARANCE_M - 0.10
    arm_y = target_y
    bearing = math.degrees(math.atan2(target_y - arm_y, target_x - arm_x))
    reserve(arm_x, arm_y, ARM_HALF_M[0], ARM_HALF_M[1])
    # The reach band between base and target belongs to the arm.
    band_x = (arm_x + target_x) / 2
    reserve(band_x, arm_y, abs(arm_x - target_x) / 2, ARM_HALF_M[1])
    distance = math.hypot(target_x - arm_x, target_y - arm_y)

    # 3. the delivery's own other surface objects, then the declared props.
    props: list[dict] = []
    unplaced: list[dict] = []
    for item in seated:
        if item["instance_id"] == target_id:
            continue
        spot = place(item["half"][0], item["half"][1])
        if spot is None:
            layout[item["instance_id"]] = {"remove": True}
            unplaced.append({"instance_id": item["instance_id"], "why": "no free cell on the table"})
            continue
        reserve(spot[0], spot[1], item["half"][0], item["half"][1])
        layout[item["instance_id"]] = {"translation": [round(spot[0], 4), round(spot[1], 4),
                                                       round(top_z + item["lift"], 4)],
                                       "on": support_id}
    for leaf in args.props:
        bmin, bmax, meta = prepared_bounds(leaf, args.prop_cache.resolve())
        hx, hy = (bmax[0] - bmin[0]) / 2, (bmax[1] - bmin[1]) / 2
        spot = place(hx, hy)
        if spot is None:
            unplaced.append({"leaf": leaf, "why": "no free cell on the table",
                             "super_category": meta["super_category"]})
            continue
        reserve(spot[0], spot[1], hx, hy)
        props.append({"id": f"prop_{leaf}", "kind": "prop", "simready_leaf": leaf,
                      "on": support_id, "xy": [round(spot[0], 4), round(spot[1], 4)],
                      "yaw_deg": 0, "super_category": meta["super_category"]})

    additions = {
        "comment": ("SPEC-016 dressing, laid out by python -m rfw_w.scenes.visualization.plan_room_dressing: the shared "
                    f"{width} x {depth} x {height} m work table replaces the delivered {support_id}, the "
                    f"task target sits at one end, the Piper stands at the other with its tool point on "
                    f"the bearing to the target ({distance:.3f} m away against a measured 0.8027 m reach), "
                    "and the props fill the free cells of the top. Figure-only: physics and IK are not_run."),
        "shell_materials": {
            "wall": {"colour": [round(v, 3) for v in args.wall_colour], "roughness": 0.88,
                     "metallic": 0.0},
            "floor": {"colour": [round(v, 3) for v in args.floor_colour], "roughness": 0.52,
                      "metallic": 0.0},
        },
        "support_table": {"instance": support_id, "size_m": [width, depth, height],
                          "why": "one shared work table in every scene, so the panels compare"},
        "additions": [
            {"id": "piper_arm", "kind": "robot", "robot": "piper",
             "pose": "experiments/paper_dollhouse/pose_piper_ready.json",
             "on": support_id, "xy": [round(arm_x, 4), round(arm_y, 4)],
             "yaw_deg": round(bearing - ARM_TCP_BEARING_DEG, 1), "task_target": target_id,
             "why": f"at the far end of the shared table, turned toward the task target "
                    f"({distance:.3f} m away)"},
        ] + [{k: v for k, v in p.items() if k != "super_category"} for p in props],
        "layout_note": "the companion layout file re-seats the delivery's own surface objects",
        "unplaced": unplaced,
        "props_placed": [{"id": p["id"], "super_category": p["super_category"], "xy": p["xy"]}
                         for p in props],
    }
    layout_payload = {
        "comment": (f"SPEC-016 dressing layout for {delivery.parent.name}: the delivered {support_id} "
                    f"becomes the shared work table, so everything that stood on its old top is re-seated "
                    "on the new one (or removed when the smaller top has no free cell for it)."),
        "overrides": layout,
    }
    for path, payload in ((args.out_additions, additions), (args.out_layout, layout_payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"delivery": str(delivery), "support": support_id, "target": target_id,
                      "arm_to_target_m": round(distance, 4), "props": len(props),
                      "unplaced": unplaced, "reseated": sorted(layout)}, indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
