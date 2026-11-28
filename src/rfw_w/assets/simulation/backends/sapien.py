# SPDX-License-Identifier: Apache-2.0
"""SAPIEN driver: load the aligned asset's textured visual mesh, place it at
rest on a ground plane, and ray-trace a 360 orbit video. SAPIEN is the repo's
native simulator, so this render is closest to the official Iscene.mp4 look.

  python .../sapien_orbit.py /path/<uuid>.urdf --out <asset>/validation --frames 180
"""

import argparse
import json
import os
import sys
import time


def run(
    urdf_path, out_dir, frames=180, fps=30, res=(1280, 960), spp=32, ray_tracing=True
):
    import numpy as np
    import sapien

    from rfw_w.assets.simulation import common

    meta = common.parse_urdf_meta(urdf_path)
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "sim": "sapien",
        "uuid": meta["uuid"],
        "mass": meta["mass"],
        "placement": "static-rest",
    }
    t0 = time.perf_counter()

    if ray_tracing:
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(spp)
        sapien.render.set_ray_tracing_path_depth(8)
        try:
            sapien.render.set_ray_tracing_denoiser("oidn")
        except Exception:
            pass
        report["renderer"] = "sapien-rt"
    else:
        report["renderer"] = "sapien-raster"

    scene = sapien.Scene()
    scene.set_ambient_light([0.3, 0.3, 0.32])
    scene.add_directional_light([-0.5, 0.4, -1.0], [1.0, 0.96, 0.90], shadow=True)
    scene.add_directional_light([0.7, 0.3, -0.5], [0.5, 0.53, 0.6], shadow=False)

    # white reflective floor (glossy -> RT reflection of the object)
    gmat = sapien.render.RenderMaterial()
    gmat.base_color = [0.9, 0.9, 0.92, 1.0]
    gmat.roughness = 0.12
    gmat.metallic = 0.0
    gmat.specular = 0.6
    scene.add_ground(0.0, render_material=gmat)

    # textured visual mesh at the canonical rest pose (AABB bottom on table)
    visual_obj = os.path.join(os.path.dirname(urdf_path), "mesh", f"{meta['uuid']}.obj")
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(visual_obj)
    actor = builder.build_kinematic(name="obj")
    rest_z = -float(meta["aabb_min"][2])
    actor.set_pose(sapien.Pose(p=[0.0, 0.0, rest_z]))

    # ---- orbit camera ----
    framing = common.orbit_framing(meta, table_z=0.0)
    vfov = common.vertical_fov(framing["fov_deg"], res)  # radians (SAPIEN wants rad)
    cam = scene.add_camera("cam", res[0], res[1], float(vfov), 0.01, 100.0)

    frames_rgb = []
    for eye in common.orbit_positions(framing, frames):
        q = common.look_at_quat_wxyz(eye, framing["target"])  # (w,x,y,z)
        cam.set_pose(sapien.Pose(p=[float(x) for x in eye], q=q))
        scene.update_render()
        cam.take_picture()
        rgba = cam.get_picture("Color")  # HxWx4 float [0,1]
        frames_rgb.append(np.clip(np.asarray(rgba)[:, :, :3], 0, 1))

    mp4 = os.path.join(out_dir, "sapien_orbit.mp4")
    common.write_mp4(frames_rgb, mp4, fps=fps)
    report.update(
        status="ok",
        n_frames=len(frames_rgb),
        video=mp4,
        seconds=round(time.perf_counter() - t0, 1),
    )
    with open(os.path.join(out_dir, "sapien_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"  [sapien] {meta['uuid']} -> {mp4} "
        f"({report['renderer']}, {len(frames_rgb)}f, {report['seconds']}s)"
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--no-rt", action="store_true")
    a = ap.parse_args()
    try:
        run(
            a.urdf,
            a.out,
            frames=a.frames,
            fps=a.fps,
            spp=a.spp,
            ray_tracing=not a.no_rt,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "sapien_report.json"), "w") as f:
            json.dump(
                {
                    "sim": "sapien",
                    "status": "fail",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                },
                f,
                indent=2,
            )
        raise


if __name__ == "__main__":
    main()
