# SPDX-License-Identifier: Apache-2.0
"""PyBullet driver: load the aligned URDF, place the object at rest on a ground
plane, and render a 360 orbit video (textured) with PyBullet's TinyRenderer
(headless). Consistent scene/object/angle with the other sim drivers.

  python .../pybullet_orbit.py /path/<uuid>.urdf --out <asset>/validation --frames 180
"""

import argparse
import json
import os
import sys
import time


def run(urdf_path, out_dir, frames=180, fps=30, res=(1280, 960), settle_steps=0):
    import numpy as np
    import pybullet as p
    import pybullet_data

    from rfw_w.assets.simulation import common

    meta = common.parse_urdf_meta(urdf_path)
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "sim": "pybullet",
        "uuid": meta["uuid"],
        "mass": meta["mass"],
        "renderer": "tiny",
        "placement": "static-rest",
    }
    t0 = time.perf_counter()

    # strip non-standard tags (extra_info/custom_data/gazebo) + absolutize mesh
    urdf = common.sanitized_urdf(urdf_path)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    ground = p.loadURDF("plane.urdf")
    # white floor (TinyRenderer has no reflection; white to match the others)
    p.changeVisualShape(ground, -1, rgbaColor=[0.9, 0.9, 0.92, 1.0])

    rest_z = -float(meta["aabb_min"][2])
    body = p.loadURDF(urdf, basePosition=[0, 0, rest_z], useFixedBase=False)
    for _ in range(settle_steps):
        p.stepSimulation()
    # canonical rest pose (consistent with the other sims)
    p.resetBasePositionAndOrientation(body, [0, 0, rest_z], [0, 0, 0, 1])

    # ---- orbit render ----
    framing = common.orbit_framing(meta, table_z=0.0)
    target = [float(x) for x in framing["target"]]
    aspect = res[0] / res[1]
    vfov = np.degrees(common.vertical_fov(framing["fov_deg"], res))
    proj = p.computeProjectionMatrixFOV(
        fov=float(vfov), aspect=aspect, nearVal=0.001, farVal=100.0
    )

    frames_rgb = []
    for eye in common.orbit_positions(framing, frames):
        view = p.computeViewMatrix(
            cameraEyePosition=[float(x) for x in eye],
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
        )
        _, _, rgba, _, _ = p.getCameraImage(
            res[0],
            res[1],
            viewMatrix=view,
            projectionMatrix=proj,
            lightDirection=[0.5, -0.4, 1.0],
            shadow=1,
            renderer=p.ER_TINY_RENDERER,
        )
        frames_rgb.append(np.asarray(rgba, np.uint8).reshape(res[1], res[0], 4))

    p.disconnect(cid)

    mp4 = os.path.join(out_dir, "pybullet_orbit.mp4")
    common.write_mp4(frames_rgb, mp4, fps=fps)
    report.update(
        status="ok",
        n_frames=len(frames_rgb),
        video=mp4,
        seconds=round(time.perf_counter() - t0, 1),
    )
    with open(os.path.join(out_dir, "pybullet_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"  [pybullet] {meta['uuid']} -> {mp4} "
        f"({len(frames_rgb)}f, {report['seconds']}s)"
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--settle", type=int, default=0)
    a = ap.parse_args()
    try:
        run(a.urdf, a.out, frames=a.frames, fps=a.fps, settle_steps=a.settle)
    except Exception as e:
        import traceback

        traceback.print_exc()
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "pybullet_report.json"), "w") as f:
            json.dump(
                {
                    "sim": "pybullet",
                    "status": "fail",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                },
                f,
                indent=2,
            )
        raise


if __name__ == "__main__":
    main()
