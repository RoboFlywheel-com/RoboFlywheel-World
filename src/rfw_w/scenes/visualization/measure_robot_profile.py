#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure a robot's reach and footprint from its own URDF.

SPEC-016. ``ROBOT_PROFILES`` in ``src/rfw_w/scenes/geometry/robot.py`` refuses an
unknown robot rather than guessing its numbers, which is right -- but it means
adding the Piper and the G1 needs measurements, and §9 of AGENTS.md says the
measuring tool is committed **before** the numbers it produces are used.

What is measured, and how:

* ``reach_m`` -- the largest horizontal distance from the base axis to the TCP
  over uniformly sampled joint vectors inside the URDF limits. A *proxy*: it is
  the farthest the tool point can be put, not a guarantee that anything can be
  grasped there. ``robot.ik`` stays ``not_run`` either way.
* ``footprint_m`` -- the diameter of the smallest XY circle around the baked
  mesh at the declared ready pose, centred on the base axis.
* ``approach_clearance_m`` -- **not measured here**. It is declared per robot
  with its reason, because it is a property of the gripper and the approach, and
  nothing in a URDF states it.

Deterministic: the sampler is seeded, so re-running reproduces the numbers.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.visualization.measure_robot_profile \\
      --robot piper --robot g1 --output-dir outputs/_paper/robot_profiles_01
"""

from __future__ import annotations
from rfw_w.scenes.visualization.inputs import resolve_input
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from rfw_w.scenes.visualization.paper_case_common import digest, write
from rfw_w.scenes.visualization.robot_asset import ROBOTS, resolve_root
from rfw_w.scenes.visualization.urdf_kinematics import fk, parse_urdf

# The joints a reach measurement is allowed to move, and the pose the rest hold.
# The G1 walks: sampling its legs would measure where it could *stand*, not how
# far it can reach from where it stands, so the legs stay at the ready pose.
SAMPLED_JOINTS = {
    "piper": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    "g1": ("waist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
           "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint"),
}

# The height band, above the robot's lowest point, that counts as its support:
# the feet of a humanoid, the bolted base of an arm. Everything above it is
# allowed to overhang whatever it is standing next to.
FOOTPRINT_BAND_M = 0.30

# Declared, not measured. Both numbers are the free space a parallel-jaw gripper# needs in front of what it closes on; `franka_panda` has carried 0.15 m since
# the profile existed, and neither of these hands is bulkier than that one.
APPROACH_CLEARANCE_M = {
    "piper": (0.15, "same as franka_panda: a parallel-jaw hand of comparable width; not measured"),
    "g1": (0.18, "three-finger Dex1 hand, wider than a parallel jaw; declared, not measured"),
}


def measure(robot: str, pose_path: Path, samples: int, seed: int) -> dict:
    spec = ROBOTS[robot]
    root = resolve_root(robot)
    urdf = root / spec["urdf"]
    model = parse_urdf(urdf)
    by_name = {j["name"]: j for j in model["joints"]}

    pose = json.loads(resolve_input(pose_path).read_text()).get("joint_degrees", {})
    base_q = {}
    for name, value in pose.items():
        joint = by_name[name]
        base_q[name] = float(value) if joint["type"] == "prismatic" else math.radians(float(value))

    names = [n for n in SAMPLED_JOINTS[robot] if by_name[n]["limits"]]
    missing = [n for n in SAMPLED_JOINTS[robot] if n not in names]
    lows = np.array([by_name[n]["limits"][0] for n in names])
    highs = np.array([by_name[n]["limits"][1] for n in names])

    tcp_link = spec["tcp"]["link"]
    offset = np.asarray(spec["tcp"]["offset_m"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.uniform(lows, highs, size=(samples, len(names)))
    # The joint limits themselves are included: an extremum of reach usually sits
    # on a limit, and uniform sampling alone never lands exactly on one.
    draws = np.vstack([draws, lows, highs])

    best_radius, best_q, reach_z = 0.0, None, None
    for row in draws:
        q = dict(base_q)
        q.update({name: float(value) for name, value in zip(names, row)})
        frames = fk(model, q)
        frame = frames[tcp_link]
        tcp = frame[:3, :3] @ offset + frame[:3, 3]
        radius = float(math.hypot(tcp[0], tcp[1]))
        if radius > best_radius:
            best_radius, best_q, reach_z = radius, q, float(tcp[2])

    ready = fk(model, base_q)
    xs, ys, zs = [], [], []
    import trimesh
    from rfw_w.scenes.visualization.robot_asset import find_mesh
    for link, visuals in model["links"].items():
        if link not in ready:
            continue
        for visual in visuals:
            if not visual.get("mesh"):
                continue
            mesh = trimesh.load(str(find_mesh(root, urdf.parent, visual["mesh"])),
                                force="mesh", process=False)
            from rfw_w.scenes.visualization.urdf_kinematics import transform
            world = ready[link] @ transform(visual["origin_xyz"], visual["origin_rpy"])
            points = np.asarray(mesh.vertices, dtype=np.float64) @ world[:3, :3].T + world[:3, 3]
            xs.append(points[:, 0]); ys.append(points[:, 1]); zs.append(points[:, 2])
    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    # The footprint is the floor (or table) the robot *stands on*, so it is
    # measured on the support band only. Measured over the whole mesh instead,
    # the G1's ready pose reports 0.9297 m -- its outstretched hands, which need
    # no floor under them and would push every placement a third of a metre off
    # the wall for nothing.
    band = z <= (float(z.min()) + FOOTPRINT_BAND_M)
    footprint = 2.0 * float(np.max(np.hypot(x[band], y[band])))
    whole_body = 2.0 * float(np.max(np.hypot(x, y)))

    clearance, clearance_why = APPROACH_CLEARANCE_M[robot]
    return {
        "robot": robot,
        "source_urdf": spec["urdf"],
        "source_urdf_sha256": digest(urdf),
        "ready_pose": str(pose_path),
        "ready_pose_sha256": digest(pose_path),
        "sampled_joints": names,
        "unbounded_joints_skipped": missing,
        "samples": int(len(draws)),
        "seed": seed,
        "reach_m": round(best_radius, 4),
        "reach_measured_at_z_m": None if reach_z is None else round(reach_z, 4),
        "reach_definition": "largest horizontal distance from the base axis to the TCP over "
                            "sampled joint vectors inside the URDF limits; a placement proxy, "
                            "not a grasp guarantee",
        "reach_pose_degrees": None if best_q is None else {
            name: round(math.degrees(best_q[name]), 2) for name in names},
        "footprint_m": round(footprint, 4),
        "footprint_definition": f"diameter of the smallest XY circle about the base axis "
                                f"containing the ready pose's mesh within {FOOTPRINT_BAND_M} m "
                                f"of its lowest point (the support band)",
        "whole_body_swept_diameter_m": round(whole_body, 4),
        "height_m": round(float(z.max() - z.min()), 4),
        "approach_clearance_m": clearance,
        "approach_clearance_source": clearance_why,
        "ik": "not_run",
        "physics_simulation": "not_run",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", action="append", required=True, choices=sorted(ROBOTS))
    parser.add_argument("--pose", action="append", type=Path, default=None,
                        help="ready pose per robot, in the same order; defaults to "
                             "experiments/paper_dollhouse/pose_<robot>_ready.json")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")
    poses = args.pose or [data_path('examples') / 'paper_dollhouse' / f'pose_{r}_ready.json'
                          for r in args.robot]
    if len(poses) != len(args.robot):
        parser.error("give one --pose per --robot, or none at all")

    records = [measure(robot, Path(pose), args.samples, args.seed)
               for robot, pose in zip(args.robot, poses)]
    write(out / "robot_profiles.json", {"robots": records})
    print(json.dumps({r["robot"]: {k: r[k] for k in
                                   ("reach_m", "footprint_m", "height_m",
                                    "approach_clearance_m", "samples")}
                      for r in records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
