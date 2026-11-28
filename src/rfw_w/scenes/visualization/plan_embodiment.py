#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Choose where a robot stands for a delivery's own task, and write the declaration.

SPEC-017. Twenty placements (ten scenes, two embodiments) is too many to author
by hand, and hand-authoring is where the first three went wrong: an arm whose
base hung off the desk, a humanoid inside an island, a humanoid facing a wall.

So the placement is searched, against the same rules `--additions` enforces:

* a table-mounted arm stands on the task's own support, with its base band fully
  on that surface;
* a floor-standing humanoid stands on the floor, clear of every footprint and of
  every opening's passage strip;
* both must have the task target inside the **measured** reach of their profile
  (`geometry/robot.py`), which is a placement proxy and not a grasp;
* among the admissible positions it prefers the one whose bearing to the target
  is clearest -- farthest from the nearest other object -- so the robot is not
  authored nose-to-nose with a bottle.

Writes `<scene>/additions_<arm|g1>.json` plus the search's own evidence: how
many positions were tried, how many each rule refused, and why the winner won.
A scene with no admissible position is a named gap, not a silently skipped row.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.visualization.plan_embodiment \\
      --delivery outputs/paper_embodiment_matrix/pm01_office_pick_cup/pm_20260926_02 \\
      --robot piper --output experiments/paper_embodiment_matrix/pm01_office_pick_cup/additions_arm.json
"""

from __future__ import annotations
from rfw_w.scenes.visualization.inputs import resolve_input
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path.cwd()
from rfw_w.scenes.visualization.render_room_preview import (
    ADDITION_OVERHANG_M, LAYOUT_OVERLAP_EPS_M, ROBOT_PROFILE_OF, ROBOT_SUPPORT_BAND_M,
    aabb_blocks_opening, local_footprint, obj_local_bounds, point_inside, world_aabb, yaw_of,
)
from rfw_w.scenes.visualization.robot_asset import ROBOTS, bake
from rfw_w.scenes.geometry.robot import profile_for

# How far in front of the target the robot should ideally stand: close enough that the tool point
# reaches it comfortably, far enough that the robot is not authored inside the object.
PREFERRED_REACH_FRACTION = 0.75
# The grid the search walks. 5 cm is finer than any placement decision here needs and keeps a
# 5 x 4 m floor under 8000 candidates.
GRID_M = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    parser.add_argument("--pose", type=Path, default=None,
                        help="defaults to experiments/paper_dollhouse/pose_<robot>_ready.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bake-dir", type=Path, default=None,
                        help="where to bake the robot for measuring; defaults to a temp dir")
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    request = json.loads((delivery / manifest["isaac_request_path"]).read_text())
    scene = json.loads((delivery / "scene_ir.json").read_text())
    plan = manifest.get("plan", {})
    target_id = plan.get("target_id")
    support_id = plan.get("support_id")
    if not target_id or not support_id:
        raise SystemExit(f"EMBODIMENT_NO_TASK(delivery names target={target_id} support={support_id})")

    bounds = {i["instance_id"]: (i["bounds_min_m"], i["bounds_max_m"])
              for i in scene.get("instances", []) if "bounds_min_m" in i}
    by_id = {i["instance_id"]: i for i in request["instances"]}
    if target_id not in by_id or support_id not in by_id:
        raise SystemExit(f"EMBODIMENT_NO_TASK({target_id}/{support_id} not in the request)")

    pose_path = args.pose or (data_path('examples') / 'paper_dollhouse' / f'pose_{args.robot}_ready.json')
    pose = json.loads(resolve_input(pose_path).read_text())
    bake_dir = (args.bake_dir or (delivery.parent / "_embodiment_probe" / args.robot)).resolve()
    if bake_dir.exists() and any(bake_dir.iterdir()):
        baked_mesh = bake_dir / "robot.obj"
        baked = json.loads((bake_dir / "robot_asset.json").read_text())
    else:
        baked = bake(args.robot, pose.get("joint_degrees", {}), bake_dir)
        baked_mesh = bake_dir / "robot.obj"
    band_lo, band_hi = obj_local_bounds(baked_mesh, ROBOT_SUPPORT_BAND_M)
    offset = [(band_lo[0] + band_hi[0]) / 2.0, (band_lo[1] + band_hi[1]) / 2.0]
    half = [(band_hi[0] - band_lo[0]) / 2.0, (band_hi[1] - band_lo[1]) / 2.0]
    tcp = baked["tcp_in_baked_frame_m"] or [1.0, 0.0, 0.0]
    tcp_bearing_deg = math.degrees(math.atan2(tcp[1], tcp[0]))

    profile_name = ROBOT_PROFILE_OF[args.robot]
    profile = profile_for(profile_name)
    mount = ROBOTS[args.robot]["mount"]

    # Footprints of everything the delivery placed, at its own pose.
    footprints: dict[str, tuple] = {}
    for instance in request["instances"]:
        instance_id = instance["instance_id"]
        if instance_id not in bounds:
            continue
        bmin, bmax = bounds[instance_id]
        local_offset, local_half, yaw = local_footprint(instance, (bmin, bmax))
        translation = [float(v) for v in instance["translation"]]
        lo, hi = world_aabb(local_offset, local_half, translation, yaw)
        footprints[instance_id] = (lo, hi, translation[2] + float(bmin[2]),
                                   translation[2] + float(bmax[2]))

    polygon = request["room"]["polygon_xy"]
    openings = request["room"].get("openings", [])
    floor_z = float(request["room"].get("floor_z_m", 0.0))
    target_xy = [float(v) for v in by_id[target_id]["translation"][:2]]
    support_box = footprints.get(support_id)
    if support_box is None:
        raise SystemExit(f"EMBODIMENT_NO_SUPPORT_BOUNDS({support_id})")

    if mount == "surface":
        surface_lo, surface_hi = support_box[0], support_box[1]
        surface_z = support_box[3]
        exempt = {support_id, target_id}
        search_lo, search_hi = surface_lo, surface_hi
    else:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        surface_z = floor_z
        exempt = {target_id}
        search_lo = [min(xs), min(ys)]
        search_hi = [max(xs), max(ys)]

    rejected = {"outside_room": 0, "off_support": 0, "overlap": 0, "blocks_opening": 0,
                "out_of_reach": 0, "too_close": 0}
    best = None
    tried = 0
    z_lo = surface_z + band_lo[2]
    z_hi = surface_z + band_hi[2]

    x = search_lo[0]
    while x <= search_hi[0]:
        y = search_lo[1]
        while y <= search_hi[1]:
            tried += 1
            distance = math.hypot(target_xy[0] - x, target_xy[1] - y)
            if distance > profile.reach_m:
                rejected["out_of_reach"] += 1
                y += GRID_M
                continue
            if distance < profile.approach_clearance_m + 0.10:
                rejected["too_close"] += 1
                y += GRID_M
                continue
            yaw = math.degrees(math.atan2(target_xy[1] - y, target_xy[0] - x)) - tcp_bearing_deg
            aabb_min, aabb_max = world_aabb(offset, half, [x, y, 0.0], yaw)
            centre = [(aabb_min[0] + aabb_max[0]) / 2.0, (aabb_min[1] + aabb_max[1]) / 2.0]
            if not point_inside(polygon, centre[0], centre[1]):
                rejected["outside_room"] += 1
                y += GRID_M
                continue
            if mount == "surface":
                overhang = max(surface_lo[0] - aabb_min[0], aabb_max[0] - surface_hi[0],
                               surface_lo[1] - aabb_min[1], aabb_max[1] - surface_hi[1])
                if overhang > ADDITION_OVERHANG_M:
                    rejected["off_support"] += 1
                    y += GRID_M
                    continue
            else:
                if any(aabb_blocks_opening(aabb_min, aabb_max, polygon, opening)
                       for opening in openings):
                    rejected["blocks_opening"] += 1
                    y += GRID_M
                    continue
            clearance = math.inf
            collided = False
            for other_id, (omin, omax, oz_lo, oz_hi) in footprints.items():
                if other_id in exempt:
                    continue
                overlap_x = min(aabb_max[0], omax[0]) - max(aabb_min[0], omin[0])
                overlap_y = min(aabb_max[1], omax[1]) - max(aabb_min[1], omin[1])
                if (overlap_x > LAYOUT_OVERLAP_EPS_M and overlap_y > LAYOUT_OVERLAP_EPS_M
                        and z_hi > oz_lo and oz_hi > z_lo):
                    collided = True
                    break
                if z_hi > oz_lo and oz_hi > z_lo:
                    gap = max(-overlap_x, -overlap_y, 0.0)
                    clearance = min(clearance, gap)
            if collided:
                rejected["overlap"] += 1
                y += GRID_M
                continue
            # Prefer standing at a comfortable fraction of the reach, then prefer room around it.
            comfort = abs(distance - PREFERRED_REACH_FRACTION * profile.reach_m)
            score = (comfort, -min(clearance, 1.0))
            if best is None or score < best["score"]:
                best = {"score": score, "xy": [round(x, 4), round(y, 4)],
                        "yaw_deg": round(yaw, 2), "distance_m": round(distance, 4),
                        "clearance_m": None if clearance is math.inf else round(clearance, 4)}
            y += GRID_M
        x += GRID_M

    evidence = {
        "delivery": str(delivery),
        "scene_version": manifest["scene_version"],
        "robot": args.robot,
        "profile": profile_name,
        "profile_reach_m": profile.reach_m,
        "profile_footprint_m": profile.footprint_m,
        "mount": mount,
        "task_target": target_id,
        "task_support": support_id,
        "grid_m": GRID_M,
        "positions_tried": tried,
        "rejected": rejected,
        "rule": "on the support (arm) or on the floor (humanoid), target inside the measured "
                "reach, clear of every footprint and every opening strip; among those, the "
                f"position closest to {PREFERRED_REACH_FRACTION} of the reach with the most room "
                "around it",
        "ik": "not_run",
        "physics_simulation": "not_run",
    }
    if best is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"comment": "no admissible position", "additions": [],
             "embodiment_gap": f"EMBODIMENT_NO_ADMISSIBLE_POSITION({args.robot})",
             "search": evidence}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "gap": f"EMBODIMENT_NO_ADMISSIBLE_POSITION({args.robot})",
                          "rejected": rejected}, indent=2))
        return 1

    evidence["chosen"] = {k: v for k, v in best.items() if k != "score"}
    payload = {
        "comment": f"SPEC-017 embodiment for {delivery.parent.name}: a {args.robot} placed by "
                   f"search, not by hand. Figure-only; the delivery is untouched and the robot's "
                   f"physics and IK are not_run.",
        "additions": [{
            "id": "arm" if mount == "surface" else "humanoid",
            "kind": "robot",
            "robot": args.robot,
            "pose": str(pose_path),
            "on": support_id if mount == "surface" else "floor",
            "xy": best["xy"],
            "yaw_deg": best["yaw_deg"],
            "task_target": target_id,
            "why": f"searched: {best['distance_m']} m from {target_id} against a measured "
                   f"{profile.reach_m} m reach; {tried} positions tried",
        }],
        "search": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "chosen": evidence["chosen"],
                      "positions_tried": tried, "rejected": rejected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
