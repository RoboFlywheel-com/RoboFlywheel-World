# SPDX-License-Identifier: Apache-2.0
"""Genesis driver: drop an aligned asset (MJCF) on a tabletop, settle, and
render a 360 orbit video. Runs as its own process (gs.init is process-global).

  python -m rfw_w.assets.simulation.backends.genesis \
      /path/<uuid>.urdf --out <asset>/validation --frames 180 --rt

Loads the MJCF (mjcf/<uuid>.xml next to the URDF) because it carries the
textured visual mesh + material and a free root joint (so it drops).
"""

import argparse
import json
import os
import time

import numpy as np

from rfw_w.assets.simulation import common


def run(
    urdf_path,
    out_dir,
    frames=180,
    fps=30,
    res=(1280, 960),
    settle_steps=200,
    use_rt=True,
):
    import genesis as gs

    meta = common.parse_urdf_meta(urdf_path)
    from pathlib import Path

    candidates = sorted((Path(urdf_path).parent / "mjcf").glob("*.xml"))
    if len(candidates) != 1:
        raise ValueError("expected exactly one canonical MJCF export")
    mjcf_path = str(candidates[0])
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(mjcf_path)

    os.makedirs(out_dir, exist_ok=True)
    report = {"sim": "genesis", "uuid": meta["uuid"], "mass": meta["mass"]}
    t0 = time.perf_counter()

    gs.init(backend=gs.gpu, logging_level="warning")

    # renderer: raytracer (LuisaRender) for PBR quality if available, else the
    # rasterizer. The raytracer only fails when the Scene builds its visualizer,
    # so probe the LuisaRenderPy module up front instead of try/except.
    import importlib.util

    have_luisa = importlib.util.find_spec("LuisaRenderPy") is not None
    if use_rt and have_luisa:
        renderer = gs.renderers.RayTracer()
        report["renderer"] = "raytracer"
    else:
        if use_rt and not have_luisa:
            print("  [genesis] LuisaRenderPy not installed; using Rasterizer")
        renderer = gs.renderers.Rasterizer()
        report["renderer"] = "rasterizer"

    scene = gs.Scene(
        show_viewer=False,
        renderer=renderer,
        vis_options=gs.options.VisOptions(
            ambient_light=(0.30, 0.30, 0.32),
            lights=[
                # warm key + cool fill (matches the repo's SAPIEN lighting feel)
                {
                    "type": "directional",
                    "dir": (-0.6, -0.4, -1.0),
                    "color": (1.0, 0.96, 0.90),
                    "intensity": 6.0,
                },
                {
                    "type": "directional",
                    "dir": (0.7, 0.3, -0.5),
                    "color": (0.85, 0.88, 1.0),
                    "intensity": 2.5,
                },
            ],
        ),
        rigid_options=gs.options.RigidOptions(dt=1.0 / 200.0),
    )

    # white floor (Genesis rasterizer has no planar reflection; white to match
    # the reflective-floor look of the RT/MuJoCo sims as closely as it can)
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.9, 0.9, 0.92)),
    )

    spawn, _ = common.rest_pose(meta, table_z=0.0, gap=0.02)
    obj = scene.add_entity(
        gs.morphs.MJCF(file=mjcf_path, pos=tuple(spawn.tolist())),
    )

    framing = common.orbit_framing(meta, table_z=0.0)
    # framing fov is HORIZONTAL (Isaac aperture); Genesis takes a VERTICAL fov.
    vfov = np.degrees(common.vertical_fov(framing["fov_deg"], res))
    cam = scene.add_camera(
        res=res,
        pos=(framing["radius"], 0.0, framing["height"]),
        lookat=tuple(framing["target"].tolist()),
        fov=float(vfov),
        GUI=False,
    )

    scene.build()

    # settle under gravity
    for _ in range(settle_steps):
        scene.step()

    # record settled pose (this IS the physics validation)
    try:
        q = obj.get_quat().cpu().numpy().tolist()  # (w,x,y,z)
        p = obj.get_pos().cpu().numpy().tolist()
        tipped, ang = common.tipped_over(q)
        report.update(settled_pos=p, settled_quat=q, tip_angle_deg=ang, tipped=tipped)
    except Exception as e:
        print(f"  [genesis] pose read failed: {type(e).__name__}: {e}")

    # reset to the canonical rest pose (AABB bottom on the table, yaw=0) so the
    # orbit render matches Isaac exactly -- physics was already validated above.
    try:
        rest_z = -float(meta["aabb_min"][2])
        obj.set_pos(np.array([0.0, 0.0, rest_z], dtype=np.float32))
        obj.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    except Exception as e:
        print(f"  [genesis] canonical reset failed: {type(e).__name__}: {e}")

    # orbit render -- physics is FROZEN here (no scene.step()); the object stays
    # at its settled pose so the 360 orbit is smooth, matching Isaac. Stepping
    # during capture makes contacts micro-jitter -> shaky video.
    frames_rgb = []
    for eye in common.orbit_positions(framing, frames):
        cam.set_pose(pos=tuple(eye.tolist()), lookat=tuple(framing["target"].tolist()))
        rgb = cam.render(rgb=True)
        rgb = rgb[0] if isinstance(rgb, (tuple, list)) else rgb
        frames_rgb.append(np.asarray(rgb))

    mp4 = os.path.join(out_dir, "genesis_orbit.mp4")
    common.write_mp4(frames_rgb, mp4, fps=fps)

    report.update(
        status="ok",
        n_frames=len(frames_rgb),
        video=mp4,
        seconds=round(time.perf_counter() - t0, 1),
    )
    with open(os.path.join(out_dir, "genesis_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"  [genesis] {meta['uuid']} -> {mp4} "
        f"({report['renderer']}, {len(frames_rgb)}f, {report['seconds']}s)"
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--settle", type=int, default=200)
    ap.add_argument("--no-rt", action="store_true", help="skip raytracer")
    a = ap.parse_args()
    try:
        run(
            a.urdf,
            a.out,
            frames=a.frames,
            fps=a.fps,
            settle_steps=a.settle,
            use_rt=not a.no_rt,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "genesis_report.json"), "w") as f:
            json.dump(
                {
                    "sim": "genesis",
                    "status": "fail",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                },
                f,
                indent=2,
            )
        raise


if __name__ == "__main__":
    main()
