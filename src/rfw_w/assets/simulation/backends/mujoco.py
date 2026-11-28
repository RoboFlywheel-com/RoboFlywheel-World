# SPDX-License-Identifier: Apache-2.0
"""MuJoCo driver: load the aligned MJCF, place the object at rest on a ground
plane, and render a 360 orbit video (textured) with MuJoCo's offscreen EGL
renderer. Consistent scene/object/angle with the Genesis and Isaac drivers.

  python .../mujoco_orbit.py /path/<uuid>.urdf --out <asset>/validation --frames 180
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen


def run(urdf_path, out_dir, frames=180, fps=30, res=(1280, 960), settle_steps=0):
    import numpy as np
    import mujoco

    from rfw_w.assets.simulation import common

    meta = common.parse_urdf_meta(urdf_path)
    from pathlib import Path

    candidates = sorted((Path(urdf_path).parent / "mjcf").glob("*.xml"))
    if len(candidates) != 1:
        raise ValueError("expected exactly one canonical MJCF export")
    mjcf_path = str(candidates[0])
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(mjcf_path)
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "sim": "mujoco",
        "uuid": meta["uuid"],
        "mass": meta["mass"],
        "renderer": "mujoco-egl",
        "placement": "static-rest",
    }
    t0 = time.perf_counter()

    # ---- build a scene: object MJCF + gray ground + lights via MjSpec ----
    spec = mujoco.MjSpec.from_file(mjcf_path)
    # offscreen framebuffer must be >= render resolution (default 640x480)
    spec.visual.global_.offwidth = res[0]
    spec.visual.global_.offheight = res[1]
    # vertical fov to match the other sims (framing fov is horizontal); MuJoCo's
    # default 45deg makes the object render smaller than Isaac/SAPIEN/Genesis.
    _fr = common.orbit_framing(meta, table_z=0.0)
    spec.visual.global_.fovy = float(
        np.degrees(common.vertical_fov(_fr["fov_deg"], res))
    )
    # neutral studio background + soft shadows
    spec.visual.headlight.ambient = [0.3, 0.3, 0.32]
    spec.visual.headlight.diffuse = [0.7, 0.68, 0.64]  # warm key
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    # matte gray ground plane (checker texture kept subtle) at z=0
    # white reflective floor (planar reflection)
    grid_mat = spec.add_material(
        name="ground",
        reflectance=0.6,
        rgba=[0.9, 0.9, 0.92, 1.0],
        shininess=0.9,
        specular=0.6,
    )
    # large plane so its far edge stays out of frame (no grazing moire)
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20.0, 20.0, 0.05], material="ground"
    )
    spec.worldbody.add_light(
        pos=[0.4, -0.3, 0.8],
        dir=[-0.5, 0.4, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.9, 0.87, 0.82],
        castshadow=True,
    )

    model = spec.compile()
    data = mujoco.MjData(model)

    # place the object's free joint at the canonical rest pose (AABB bottom on
    # the table, identity orientation) -- consistent with Isaac/Genesis.
    rest_z = -float(meta["aabb_min"][2])
    if model.njnt > 0 and model.nq >= 7:
        data.qpos[:3] = [0.0, 0.0, rest_z]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    # ---- orbit camera (MuJoCo's built-in azimuth/elevation orbit) ----
    framing = common.orbit_framing(meta, table_z=0.0)
    target = framing["target"]
    dz = framing["height"] - target[2]
    # 0.85: MuJoCo's free-camera framing renders slightly smaller than the other
    # engines at the same distance/fov; pull in a touch so object size matches.
    dist = 0.85 * float(np.hypot(framing["radius"], dz))

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = target
    cam.distance = dist

    renderer = mujoco.Renderer(model, height=res[1], width=res[0])
    opt = mujoco.MjvOption()

    # drive azimuth/elevation from the SAME orbit eye positions as the other
    # sims so the per-frame viewpoint matches (MuJoCo forward points at lookat).
    frames_rgb = []
    for eye in common.orbit_positions(framing, frames):
        fwd = np.asarray(target) - np.asarray(eye)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
        renderer.update_scene(data, camera=cam, scene_option=opt)
        frames_rgb.append(renderer.render().copy())

    mp4 = os.path.join(out_dir, "mujoco_orbit.mp4")
    common.write_mp4(frames_rgb, mp4, fps=fps)
    report.update(
        status="ok",
        n_frames=len(frames_rgb),
        video=mp4,
        seconds=round(time.perf_counter() - t0, 1),
    )
    with open(os.path.join(out_dir, "mujoco_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"  [mujoco] {meta['uuid']} -> {mp4} "
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
        with open(os.path.join(a.out, "mujoco_report.json"), "w") as f:
            json.dump(
                {
                    "sim": "mujoco",
                    "status": "fail",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                },
                f,
                indent=2,
            )
        raise


if __name__ == "__main__":
    main()
