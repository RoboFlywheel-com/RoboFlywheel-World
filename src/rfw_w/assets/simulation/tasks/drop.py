#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Task A -- drop and settle (需求 §8-A).

Release the asset from a set of heights and orientations, let contact dynamics do
the rest, and record what it does.  The object is a free rigid body for the whole
run: nothing is attached, keyframed or teleported, which is the one thing §8 is
most explicit about forbidding.

Two things come out of a run:
  * a **figure trial** -- one pre-registered (height, orientation) pair, rendered
    as the `initial` frame at release and the `response` frame once the object has
    settled, plus a 30 fps mp4 of the whole fall;
  * a **sweep** -- every height x orientation, numbers only, which is what the
    report's pass rate is computed from.

The figure trial is fixed in `thresholds.yaml` BEFORE the run precisely so the
paper frame cannot be the trial that happened to look best (需求 §8: "主图代表样例
不等于总体成功率").  The report labels it that way too.

Run under Isaac Sim's python (see rig.py for why):
    /isaac-sim/python.sh drop.py --key 0_2_0002 --asset-dir <staged asset dir> \
        --out <case>/09_task_dynamics
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


from rfw_w.assets.simulation.tasks import rig as R  # noqa: E402

TASK = "drop"


def run(
    key: str,
    asset_dir: str,
    out_dir: str,
    threshold_path: str = None,
    do_sweep: bool = True,
    video_fps: int = 30,
    res=(768, 1024),
    verbose: bool = True,
) -> dict:
    """Boot Isaac, run the task, return the report dict.  Writes mp4 + pngs too."""
    R.require_isaac()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        return _run(
            app,
            key,
            asset_dir,
            out_dir,
            threshold_path,
            do_sweep,
            video_fps,
            res,
            verbose,
        )
    except BaseException as e:  # noqa: BLE001
        # NOTE: the failure report and the traceback must be written HERE, before
        # `app.close()` -- close() terminates the process, so anything after the
        # finally block (including the caller's own error handling) never runs.
        R.dump_traceback(out_dir, f"task_{TASK}")
        R.write_report(
            out_dir,
            f"task_{TASK}",
            {
                "sim": "isaacsim",
                "task": TASK,
                "key": key,
                "status": "fail",
                "error": f"{type(e).__name__}: {e}",
            },
        )
        raise
    finally:
        app.close()


def _run(
    app, key, asset_dir, out_dir, threshold_path, do_sweep, video_fps, res, verbose
) -> dict:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim

    thr = R.load_thresholds(threshold_path)
    common, dp = thr["common"], thr[TASK]
    dt = float(common["physics_dt"])

    usd = R.build_usd(key, asset_dir)
    sidecar = R.read_sidecar(asset_dir)

    world = World(physics_dt=dt, rendering_dt=dt)
    world.reset()
    stage = omni.usd.get_context().get_stage()
    R.add_lighting(stage)
    R.make_table(world, stage, common["mu_ground"])
    prim = R.load_asset(stage, key, usd)
    # The converted OmniPBR material carries emissive_intensity=10000, which renders
    # the whole asset BLACK. Zeroing it is the fix -- see rig.calm_asset_material.
    R.calm_asset_material(prim)
    world.reset()

    view = RigidPrim(prim_paths_expr=["/World/asset"], name="obj")
    world.reset()
    # Upright, resting on the ground: the extents the camera, the report and the
    # release heights are all defined against. See rig.authoring_up_fix for why the
    # upright pose is not the exported one.
    _, _, lo_u, hi_u = R.place_upright(view, prim, None, 0.0)
    size = hi_u - lo_u
    diag = float(np.linalg.norm(size))

    heights = [float(h) for h in dp["heights_m"]]
    orients = R.orientations(int(common["n_orientations"]), int(common["seed"]))

    # Frame the FIGURE TRIAL's own working volume -- ground up to that trial's
    # release point -- not the tallest trial in the sweep. The camera has to hold
    # for both the initial and the response frame (需求 §3), and sizing it for a
    # 0.30 m drop when the figure uses 0.15 m just shrinks the subject.
    fig = dp.get("figure_trial") or {"height_idx": len(heights) // 2, "orient_idx": 0}
    hi_i = int(fig.get("height_idx", len(heights) // 2)) % len(heights)
    oi_i = int(fig.get("orient_idx", 0)) % len(orients)
    assert hi_i == int(
        fig.get("height_idx", hi_i)
    ), "figure_trial.height_idx is out of range for heights_m"
    assert oi_i == int(
        fig.get("orient_idx", oi_i)
    ), "figure_trial.orient_idx is out of range for n_orientations"

    top = heights[hi_i] + float(size[2]) + 0.06
    shooter = R.Shooter(world, (1.0, -1.0, top * 0.6), (0, 0, top * 0.45), res=res)
    centre = np.array([0.0, 0.0, top * 0.5])
    radius = 0.5 * float(np.linalg.norm(np.array([size[0], size[1], top])))
    framing = R.fit_camera(
        shooter, centre, radius, az_deg=38.0, el_deg=16.0, frame_frac=0.92
    )
    # Decode the asset's MDL texture NOW, while the asset is still parked on the
    # table, so that the release frame needs only a short warm-up. See
    # R.render_warmup: without this the `initial` frame is the untextured black one,
    # because the texture does not land until about render frame 15.
    warmup_setup = R.render_warmup(
        world,
        shooter,
        why="scene setup: decode the asset's texture before the "
        "release frame is taken",
        verbose=verbose,
    )
    if verbose:
        R.log(
            f"[{TASK}] {key}: bbox {np.round(size, 4).tolist()} m  "
            f"camera {framing['distance_m']} m away, "
            f"fov {framing['intrinsics']['vfov_deg']}x"
            f"{framing['intrinsics']['hfov_deg']} deg, "
            f"res {framing['res']}"
        )

    result = {
        "sim": "isaacsim",
        "task": TASK,
        "key": key,
        "status": "ok",
        "engine": {
            "name": "Isaac Sim",
            "version": _isaac_version(),
            "physics": "PhysX GPU",
            "run_under": R.ISAAC_PYTHON,
        },
        "asset": {
            "export_dir": asset_dir,
            "usd": usd,
            "n_collision_hulls": sidecar.get("n_collision_hulls"),
            "total_mass_kg": sidecar.get("total_mass"),
            "whole_mu1": sidecar.get("whole_mu1"),
            "bbox_visual_m": [round(float(v), 6) for v in size],
            "diagonal_m": round(diag, 6),
            "bbox_frame": "world Z-up, asset posed upright (see rig.authoring_up_fix)",
            "note": "bbox measured from the VISUAL mesh (colliders are authored "
            "purpose=guide); mass/inertia/friction are the export's own "
            "part_physics.json values, carried into the USD by "
            "exporters/isaac_usd",
        },
        "thresholds": {
            "version": thr.get("version"),
            "frozen_at": thr.get("frozen_at"),
            "path": threshold_path
            or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "thresholds.yaml"
            ),
            "drop": dp,
        },
        "camera": framing,
        "warmup_setup": warmup_setup,
        "figure_trial": None,
        "trials": [],
    }

    # ── figure trial ────────────────────────────────────────────────────────
    result["figure_trial"] = {
        "height_m": heights[hi_i],
        "orient_idx": oi_i,
        "preregistered_in": "thresholds.yaml::drop.figure_trial",
        "note": "This is ONE representative trial chosen before the run. It is not "
        "the pass rate -- see `sweep` for that (需求 §8).",
    }

    fig_paths = _figure_trial(
        world,
        view,
        shooter,
        prim,
        key,
        out_dir,
        heights[hi_i],
        orients[oi_i],
        dp,
        dt,
        video_fps,
        result,
        verbose,
    )

    # ── sweep ───────────────────────────────────────────────────────────────
    if do_sweep:
        result["sweep"] = _sweep(
            world, view, prim, heights, orients, dp, dt, common, verbose
        )
    else:
        result["sweep"] = {"skipped": True}

    result["outputs"] = fig_paths
    R.write_report(out_dir, f"task_{TASK}", result)
    return result


def _isaac_version() -> str:
    try:
        with open("/isaac-sim/VERSION") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


def _nsteps(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def _figure_trial(
    world,
    view,
    shooter,
    prim,
    key,
    out_dir,
    height,
    quat_xyzw,
    dp,
    dt,
    video_fps,
    result,
    verbose,
) -> dict:
    """One trial, rendered: initial frame, response frame, and a 30 fps mp4."""
    pos, rel_wxyz, lo, hi = R.place_upright(view, prim, quat_xyzw, height)
    view.set_linear_velocities(np.zeros((1, 3)))
    view.set_angular_velocities(np.zeros((1, 3)))
    # Refresh the camera buffer at the RELEASE pose.  Render-only frames: a
    # `world.step(render=True)` here would start the fall before the "initial"
    # frame was taken, and the frame is supposed to be the release instant.
    result["warmup_initial"] = R.render_warmup(
        world, shooter, why="release frame (initial)", verbose=verbose
    )
    # ...and then make the renderer agree with the pose the teleport just set.  The
    # render-only warm-up above does not do that (see R.sync_pose_to_render): without
    # this the release frame is a drawing of the PREVIOUS pose, at the table, while
    # the physics view already reports the asset 0.15 m up.
    result["pose_sync"] = R.sync_pose_to_render(world, steps=3)
    # Record the pose the asset is ACTUALLY in when the release frame is taken, and
    # the height of its lowest point.  Without this the figure pair cannot be checked
    # against the release height, and a frame rendered at the wrong pose looks
    # plausible in a contact sheet.
    _p, _q = view.get_world_poses()
    _p = np.asarray(_p[0], float)
    _q = np.asarray(_q[0], float)
    _lo, _ = R._world_bbox_of(prim, _p, _q)
    result["figure_trial"]["pose_at_initial"] = {
        "position_m": [round(float(v), 6) for v in _p],
        "quat_wxyz": [round(float(v), 6) for v in _q],
        "bbox_bottom_z_m": round(float(_lo[2]), 6),
        "requested_release_bottom_m": float(height),
        "matches_release": bool(abs(float(_lo[2]) - float(height)) < 1e-3),
        "sim_time_s": round(float(world.current_time), 6),
    }
    if verbose:
        R.log(
            f"[{TASK}] release frame: body {np.round(_p, 4).tolist()} "
            f"bottom z={_lo[2]:.4f} (requested {height}), "
            f"t={world.current_time:.4f}s"
        )
    initial = shooter.grab(world)
    R.write_png(os.path.join(out_dir, f"task_{TASK}_initial.png"), initial)

    vid = R.Video(
        os.path.join(out_dir, f"task_{TASK}.mp4"),
        video_fps,
        (initial.shape[1], initial.shape[0]),
    )
    vid.append(initial)

    n = _nsteps(float(dp["duration_s"]), dt)
    t = 0.0
    last_frame_t = 0.0
    # the RELEASE state, in world terms -- both the settle test and the reported
    # displacement are measured against this, not against the authored tilt
    rel_xyzw = np.array([rel_wxyz[1], rel_wxyz[2], rel_wxyz[3], rel_wxyz[0]], float)
    ref_pos, ref_quat = pos.copy(), rel_xyzw
    release_bottom_m = float(lo[2])
    settle_since = None
    settle_dt = float(dp["settle_window_s"])
    speed_tol = float(dp["settle_speed_mps"])

    traj = {"t": [], "pos": [], "quat": [], "vel": [], "angvel": []}
    response, response_t = None, None
    for i in range(n):
        render = (t - last_frame_t) >= (1.0 / video_fps) - 1e-9
        world.step(render=render)
        t += dt
        p, q = view.get_world_poses()  # [1,3], [1,4] wxyz
        p, q = np.asarray(p[0], float), np.asarray(q[0], float)
        lv = np.asarray(view.get_linear_velocities()[0], float)
        traj["t"].append(t)
        traj["pos"].append(p)
        traj["quat"].append(q)
        traj["vel"].append(lv)
        traj["angvel"].append(np.asarray(view.get_angular_velocities()[0], float))
        if render:
            f = shooter.grab(world)
            vid.append(f)
            last_frame_t = t
        # settle detection:横向位移/转角都在容差内且线速度低于阈值，持续 settle_dt
        hdisp = float(np.linalg.norm(p[:2] - ref_pos[:2]))
        rot = R.quat_angle_deg(rel_xyzw, [q[1], q[2], q[3], q[0]])
        tt = max(
            float(dp["trans_abs_m"]),
            float(dp["trans_rel"]) * result["asset"]["diagonal_m"],
        )
        quiet = (
            hdisp <= tt
            and rot <= float(dp["rot_deg"])
            and float(np.linalg.norm(lv)) <= speed_tol
        )
        if quiet:
            settle_since = settle_since if settle_since is not None else t
            if t - settle_since >= settle_dt:
                result["warmup_response"] = R.render_warmup(
                    world,
                    shooter,
                    min_frames=8,
                    why="settled frame (response)",
                    verbose=verbose,
                )
                response = shooter.grab(world)
                response_t = t
                break
        else:
            settle_since = None
    settled_cleanly = response is not None
    if response is None:
        # Ran out of `duration_s` without ever meeting the quiet window. The frame is
        # still taken (the run is over either way), but `settled` below must say
        # FALSE -- reporting it as settled just because the clock ran out makes the
        # flag meaningless.
        result["warmup_response"] = R.render_warmup(
            world,
            shooter,
            min_frames=8,
            why="final frame (never settled)",
            verbose=verbose,
        )
        response = shooter.grab(world)
        response_t = t
    vid.close()

    R.write_png(os.path.join(out_dir, f"task_{TASK}_response.png"), response)
    np.savez_compressed(
        os.path.join(out_dir, f"task_{TASK}_traj.npz"),
        t=np.asarray(traj["t"]),
        pos=np.asarray(traj["pos"]),
        quat=np.asarray(traj["quat"]),
        vel=np.asarray(traj["vel"]),
        angvel=np.asarray(traj["angvel"]),
    )

    _poses = view.get_world_poses()  # (positions[n,3], orientations[n,4])
    pf = np.asarray(_poses[0][0], float)
    qf = np.asarray(_poses[1][0], float)  # wxyz
    metrics = _metrics(
        pf,
        np.array([qf[1], qf[2], qf[3], qf[0]]),
        ref_pos,
        ref_quat,
        dp,
        result["asset"]["diagonal_m"],
    )
    diff = R.frame_diff(initial, response)
    result["figure_trial"].update(
        {
            "release_bottom_m": round(release_bottom_m, 6),
            "response_time_s": round(float(response_t), 4),
            "settled": bool(settled_cleanly),
            "n_video_frames": vid.n,
            "frame_diff_mean_abs": round(float(diff), 4),
            **metrics,
        }
    )
    if verbose:
        R.log(
            f"[{TASK}] figure trial: settled at {response_t:.2f}s, "
            f"hdisp={metrics['horiz_disp_m']:.4f} rot={metrics['rot_deg']:.2f} deg "
            f"stable={metrics['stable']}, frame diff {diff:.2f}"
        )
    return {
        "initial": os.path.join(out_dir, f"task_{TASK}_initial.png"),
        "response": os.path.join(out_dir, f"task_{TASK}_response.png"),
        "video": os.path.join(out_dir, f"task_{TASK}.mp4"),
        "traj": os.path.join(out_dir, f"task_{TASK}_traj.npz"),
    }


def _metrics(p, quat_xyzw, ref_pos, ref_quat, dp, diag) -> dict:
    hdisp = float(
        np.linalg.norm(np.asarray(p[:2], float) - np.asarray(ref_pos[:2], float))
    )
    rot = R.quat_angle_deg(ref_quat, quat_xyzw)
    tt = max(float(dp["trans_abs_m"]), float(dp["trans_rel"]) * diag)
    return {
        "horiz_disp_m": round(hdisp, 6),
        "trans_tol_m": round(tt, 6),
        "rot_deg": round(rot, 6),
        "final_z_m": round(float(p[2]), 6),
        "below_ground": bool(float(p[2]) < float(dp["min_z_m"])),
        "stable": bool(
            hdisp <= tt
            and rot <= float(dp["rot_deg"])
            and float(p[2]) >= float(dp["min_z_m"])
        ),
    }


def _sweep(world, view, prim, heights, orients, dp, dt, common, verbose) -> dict:
    """Every height x orientation, numbers only -- the pass rate comes from here."""
    rows = []
    _, _, lo_u, hi_u = R.place_upright(view, prim, None, 0.0)
    diag = float(np.linalg.norm(hi_u - lo_u))
    n = _nsteps(float(dp["duration_s"]), dt)
    for hidx, height in enumerate(heights):
        for oidx, quat in enumerate(orients):
            world.reset()
            pos, rel_wxyz, _, _ = R.place_upright(view, prim, quat, height)
            rel_xyzw = np.array(
                [rel_wxyz[1], rel_wxyz[2], rel_wxyz[3], rel_wxyz[0]], float
            )
            view.set_linear_velocities(np.zeros((1, 3)))
            view.set_angular_velocities(np.zeros((1, 3)))
            for _ in range(n):
                world.step(render=False)
            p, q = view.get_world_poses()
            p = np.asarray(p[0], float)
            q = np.asarray(q[0], float)  # wxyz
            m = _metrics(p, np.array([q[1], q[2], q[3], q[0]]), pos, rel_xyzw, dp, diag)
            rows.append({"height_m": height, "orient_idx": oidx, **m})
            if verbose:
                R.log(
                    f"[{TASK}] sweep h={height:.2f} o={oidx} "
                    f"stable={m['stable']} hdisp={m['horiz_disp_m']:.4f} "
                    f"rot={m['rot_deg']:.2f}"
                )
    n_ok = sum(1 for r in rows if r["stable"])
    return {
        "n_trials": len(rows),
        "n_stable": n_ok,
        "stable_rate": round(n_ok / max(1, len(rows)), 4),
        "rows": rows,
        # 需求 §8: "保存失败例" -- keep the failures, do not only report the rate
        "failures": [r for r in rows if not r["stable"]],
        "note": "Stable is a per-trial contact outcome, not a quality score. "
        "Whether a given landing pose is the OBJECT's correct rest pose is "
        "a task-definition question (需求 §8-A), not something this rule decides.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", required=True)
    ap.add_argument(
        "--asset-dir",
        required=True,
        help="staged asset export dir (see rig.stage_asset)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    a = ap.parse_args()
    try:
        rep = run(
            a.key,
            a.asset_dir,
            a.out,
            a.thresholds,
            do_sweep=not a.no_sweep,
            video_fps=a.fps,
            res=(a.width, a.height),
        )
        print(
            json.dumps(
                {k: v for k, v in rep.items() if k in ("sim", "task", "key", "status")},
                ensure_ascii=False,
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        os.makedirs(a.out, exist_ok=True)
        R.write_report(
            a.out,
            f"task_{TASK}",
            {
                "sim": "isaacsim",
                "task": TASK,
                "key": a.key,
                "status": "fail",
                "error": f"{type(e).__name__}: {e}",
            },
        )
        raise


if __name__ == "__main__":
    main()
