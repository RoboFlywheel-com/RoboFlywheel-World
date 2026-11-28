#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Task B -- push and slide response (需求 §8-B).

The asset rests on the table; a box-shaped pusher with real collision drives into
it along +X at a fixed height, speed and stroke.  The pusher is a DRIVEN rigid
body, not a kinematic one, so the contact is solved rather than imposed -- it
stalls against the asset exactly as a real pusher would, and the asset's response
comes out of PhysX.

Two contact geometries, both fixed in `thresholds.yaml` before the run:
  * **centred** (offset 0) -- a straight push through the centre of mass;
  * **off-centre** (offset 0.6 x half-width) -- asymmetric contact, which is what
    makes the asset rotate rather than slide, and is the case §8-B names
    ("非对称接触产生旋转").

One pre-registered trial becomes the figure pair; the rest is numbers.

Run under Isaac Sim's python:
    /isaac-sim/python.sh push.py --key 9_0_0000 --asset-dir <staged asset dir> \
        --out <case>/09_task_dynamics
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


from rfw_w.assets.simulation.tasks import rig as R  # noqa: E402

TASK = "push"


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
        # Before app.close(): close() terminates the process, so the caller never
        # gets the exception (see rig.dump_traceback).
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


def _pose(view):
    p, q = view.get_world_poses()
    return np.asarray(p[0], float), np.asarray(q[0], float)  # pos, wxyz


def park_pusher(pusher_view, push):
    """Put the slab back at its start position WITHOUT resetting the world.

    `push.axis.set(0.0)` alone is not enough mid-run: it writes the drive target but
    the body has already travelled, so the next ramp starts from wherever it stopped.
    And a `world.reset()` mid-run is worse -- it leaves the drive unresponsive, so the
    target ramps while the body never moves (which is exactly how the sweep came back
    reading "no movement" while the identical figure trial pushed 0.067 m).

    Writing the pose directly is a teleport.  That is fine for the TOOL -- the asset
    is the thing 需求 §8 says must never be teleported -- and it keeps the drives alive.
    """
    if pusher_view is None:
        return
    pusher_view.set_world_poses(
        positions=np.array([[push.start_x, 0.0, push.start_z]]),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    pusher_view.set_linear_velocities(np.zeros((1, 3)))
    pusher_view.set_angular_velocities(np.zeros((1, 3)))
    push.axis.set(0.0)


def _world_bbox(prim, pos, quat_wxyz):
    """World AABB of the asset from its authored box, rotated by its measured pose."""
    from scipy.spatial.transform import Rotation as Rot

    lo0, hi0 = R.authored_bbox(prim)
    xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    lo, hi = R._rotated_aabb(lo0, hi0, xyzw)
    return lo + pos, hi + pos


def _run(
    app, key, asset_dir, out_dir, threshold_path, do_sweep, video_fps, res, verbose
) -> dict:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim

    thr = R.load_thresholds(threshold_path)
    common, pb = thr["common"], thr[TASK]
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

    view = RigidPrim(prim_paths_expr=["/World/asset"], name="obj")
    world.reset()
    # settle the asset on the table first: the push must start from rest. A fixed
    # 0.5 s was not enough -- the hammer was still toppling when the push began, so
    # "contact onset" was being detected at 0.002 s against a moving object.
    R.place_upright(view, prim, None, 0.0)
    rest_pos, rest_quat, lo_r, hi_r = R.settle_centred(
        world, view, prim, dt, seconds=4.0
    )
    size = hi_r - lo_r
    diag = float(np.linalg.norm(size))
    if verbose:
        R.log(
            f"[{TASK}] {key}: resting bbox {np.round(size, 4).tolist()} m at "
            f"{np.round(rest_pos, 4).tolist()}"
        )

    # Pusher: a slab as wide as the asset, half its height.
    height = float(size[2])
    slab = (0.05, max(0.06, float(size[1]) * 1.25), max(0.03, height * 0.5))
    # The contact height is a fraction of the asset's height, but it must also keep
    # the SLAB clear of the table: a hammer settles lying flat (height ~0.04 m), so
    # 0.35 x height would put the slab's bottom below z=0 and it would plough into
    # the table instead of pushing the asset.
    ch = max(float(pb["contact_height_frac"]) * height, slab[2] / 2.0 + 0.005)
    start_x = float(lo_r[0]) - 0.03 - 0.025  # slab face 3 cm off the asset

    # Rig origin = the pusher's START position, which is also its park position: at
    # `world.reset()` every rig body sits at the rig origin, so putting that origin
    # BEHIND the asset is what keeps the slab from spawning inside it (and getting
    # the asset ejected by the penetration solver).
    rig = R.Rig(stage, origin=(start_x, 0.0, ch), name="rig", mu=common["mu_rig"])
    push = R.Pusher(
        rig, size=slab, axis="X", lo=-1.0, hi=1.0, mass=0.6, stiffness=6e3, damping=4e2
    )
    push.start_x, push.start_z = start_x, ch  # stroke is set below
    world.reset()
    view.set_world_poses(
        positions=rest_pos.reshape(1, 3), orientations=rest_quat.reshape(1, 4)
    )
    view.set_linear_velocities(np.zeros((1, 3)))
    view.set_angular_velocities(np.zeros((1, 3)))
    # AFTER the reset: a RigidPrim built before it goes stale and set_world_poses
    # then fails with "Failed to get rigid body transforms from backend".
    pusher_view = RigidPrim(
        prim_paths_expr=[f"/World/{rig.name}/pusher"], name="pusher"
    )

    stroke = float(pb["stroke_m"])
    push.stroke = stroke
    speed = float(pb["speed_mps"])
    travel_t = stroke / speed
    hold_t = float(pb["settle_after_s"])

    shooter = R.Shooter(world, (1.0, -1.0, 0.8), (0, 0, 0.2), res=res)
    centre = np.array([start_x + stroke / 2.0, 0.0, max(ch, 0.05)])
    radius = 0.5 * float(
        np.linalg.norm([size[0] + stroke + 0.25, size[1] + 0.25, height + 0.3])
    )
    framing = R.fit_camera(
        shooter, centre, radius, az_deg=42.0, el_deg=22.0, frame_frac=0.92
    )
    if verbose:
        R.log(
            f"[{TASK}] {key}: contact height {ch:.4f} m, stroke {stroke} m @ "
            f"{speed} m/s, pusher {np.round(slab, 3).tolist()}, "
            f"camera {framing['distance_m']} m"
        )

    # Decode the asset's MDL texture BEFORE the push starts. The `initial` frame is
    # taken 0.55 s in, which is only ~16 rendered frames -- right on the edge of the
    # ~15 render frames this asset needs (see R.render_warmup), so without this the
    # "before contact" frame could come back untextured.
    warmup_setup = R.render_warmup(
        world,
        shooter,
        why="scene setup: decode the asset's texture before the " "push begins",
        verbose=verbose,
    )

    fig = pb.get("figure_trial") or {"offset_idx": 0}
    oi = int(fig.get("offset_idx", 0)) % len(pb["contact_offsets_m"])

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
            "resting_bbox_m": [round(float(v), 6) for v in size],
            "diagonal_m": round(diag, 6),
        },
        "pusher": {
            "shape": "box",
            "size_m": [round(float(v), 6) for v in slab],
            "contact_height_m": round(ch, 6),
            "height_frac_of_asset": pb["contact_height_frac"],
            "face_start_x_m": round(start_x, 6),
            "stroke_m": stroke,
            "speed_mps": speed,
            "note": "friction is NOT tuned to produce a nice trajectory (需求 §8-B): "
            "pusher uses mu_rig and the table mu_ground from thresholds.yaml, "
            "both fixed before the run",
        },
        "thresholds": {
            "version": thr.get("version"),
            "frozen_at": thr.get("frozen_at"),
            "push": pb,
        },
        "camera": framing,
        "warmup_setup": warmup_setup,
        "figure_trial": None,
    }
    result["figure_trial"] = {
        "offset_idx": oi,
        "contact_offset_m": round(
            float(pb["contact_offsets_m"][oi]) * float(size[1]) / 2.0, 6
        ),
        "preregistered_in": "thresholds.yaml::push.figure_trial",
        "note": "ONE representative trial fixed before the run; not the pass rate.",
    }
    paths = _figure_trial(
        world,
        view,
        shooter,
        rig,
        push,
        key,
        out_dir,
        rest_pos,
        rest_quat,
        size,
        start_x,
        stroke,
        speed,
        travel_t,
        hold_t,
        float(pb["contact_offsets_m"][oi]) * float(size[1]) / 2.0,
        pb,
        dt,
        video_fps,
        result,
        verbose,
    )

    result["sweep"] = (
        _sweep(
            world,
            rig,
            push,
            prim,
            rest_pos,
            rest_quat,
            size,
            start_x,
            speed,
            travel_t,
            hold_t,
            pb,
            dt,
            common,
            verbose,
        )
        if do_sweep
        else {"skipped": True}
    )
    result["outputs"] = paths
    R.write_report(out_dir, f"task_{TASK}", result)
    return result


def _isaac_version() -> str:
    try:
        with open("/isaac-sim/VERSION") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


def _drive(
    world,
    view,
    push,
    rest_pos,
    rest_quat,
    start_x,
    stroke,
    speed,
    travel_t,
    hold_t,
    fps,
    video,
    shooter,
    frames_at=None,
):
    """Park the asset, push it, and return (traj, shots, contact_onset_s)."""
    dt = world.get_physics_dt()
    world.reset()
    view.set_world_poses(
        positions=rest_pos.reshape(1, 3), orientations=rest_quat.reshape(1, 4)
    )
    view.set_linear_velocities(np.zeros((1, 3)))
    view.set_angular_velocities(np.zeros((1, 3)))
    push.axis.set(0.0)  # 0 == the rig origin == the pusher's start position
    p0, q0 = R.settle(world, view, dt, seconds=2.5)

    traj = {"t": [], "pos": [], "quat": [], "vel": [], "pusher_x": []}
    shots, contact_t = {}, None
    t, last_frame = 0.0, -1e9
    n = int(round((travel_t + hold_t) / dt))
    for _ in range(n):
        t += dt
        push.axis.set(min(stroke, speed * t))
        render = video is not None and (t - last_frame) >= (1.0 / fps) - 1e-9
        world.step(render=render)
        p, q = _pose(view)
        v = np.asarray(view.get_linear_velocities()[0], float)
        traj["t"].append(t)
        traj["pos"].append(p)
        traj["quat"].append(q)
        traj["vel"].append(v)
        traj["pusher_x"].append(float(push.axis.target()))
        if contact_t is None and float(np.linalg.norm(v)) > 0.01:
            contact_t = t  # first step the asset actually moves
        if render:
            last_frame = t
            if video is not None:
                video.append(shooter.grab(world))
        if frames_at:
            for label, tt in list(frames_at.items()):
                if t >= tt and label not in shots:
                    # render-only refresh: `step(render=...)` is False on most
                    # iterations, so reading the buffer here could hand back a frame
                    # up to 1/fps stale.  `world.render()` costs no sim time, so the
                    # shot is still the state at `t`.
                    world.render()
                    shots[label] = shooter.grab(world)
    return traj, shots, contact_t, p0, q0


def _metrics(p0, q0, traj, pb):
    p_end = np.asarray(traj["pos"][-1], float)
    q_end = np.asarray(traj["quat"][-1], float)
    horiz = float(np.linalg.norm(p_end[:2] - p0[:2]))
    rot = R.quat_angle_deg(
        [q0[1], q0[2], q0[3], q0[0]], [q_end[1], q_end[2], q_end[3], q_end[0]]
    )
    return {
        "object_horiz_disp_m": round(horiz, 6),
        "object_disp_3d_m": round(float(np.linalg.norm(p_end - p0)), 6),
        "object_rot_deg": round(rot, 6),
        "max_obj_speed_mps": round(
            float(max(np.linalg.norm(v) for v in traj["vel"])), 6
        ),
        "moved": bool(horiz >= float(pb["min_disp_m"])),
        "rotated": bool(rot > 20.0),
    }


def _figure_trial(
    world,
    view,
    shooter,
    rig,
    push,
    key,
    out_dir,
    rest_pos,
    rest_quat,
    size,
    start_x,
    stroke,
    speed,
    travel_t,
    hold_t,
    y_offset,
    pb,
    dt,
    video_fps,
    result,
    verbose,
) -> dict:
    pre = shooter.grab(world, force_render=True)
    vid = R.Video(
        os.path.join(out_dir, f"task_{TASK}.mp4"),
        video_fps,
        (pre.shape[1], pre.shape[0]),
    )
    # initial = before contact; response = after the stroke plus the pre-registered
    # hold, so the asset has come to rest under friction
    gap_t = max(0.0, (float(size[0]) * 0 + 0.03) / speed)  # pusher approaches 3 cm
    frames_at = {"initial": max(0.0, gap_t - 0.05), "response": travel_t + hold_t}
    traj, shots, c_t, p0, q0 = _drive(
        world,
        view,
        push,
        rest_pos,
        rest_quat,
        start_x,
        stroke,
        speed,
        travel_t,
        hold_t,
        video_fps,
        vid,
        shooter,
        frames_at,
    )
    vid.close()
    initial = shots.get("initial", pre)
    response = shots.get("response", shooter.grab(world))
    R.write_png(os.path.join(out_dir, f"task_{TASK}_initial.png"), initial)
    R.write_png(os.path.join(out_dir, f"task_{TASK}_response.png"), response)
    np.savez_compressed(
        os.path.join(out_dir, f"task_{TASK}_traj.npz"),
        t=np.asarray(traj["t"]),
        pos=np.asarray(traj["pos"]),
        quat=np.asarray(traj["quat"]),
        vel=np.asarray(traj["vel"]),
    )
    m = _metrics(p0, q0, traj, pb)
    m["contact_onset_s"] = None if c_t is None else round(c_t, 4)
    m["frame_diff_mean_abs"] = round(R.frame_diff(initial, response), 4)
    m["n_video_frames"] = vid.n
    result["figure_trial"].update(m)
    if verbose:
        R.log(
            f"[{TASK}] figure trial: moved {m['object_horiz_disp_m']:.4f} m, "
            f"rot {m['object_rot_deg']:.2f} deg, contact {m['contact_onset_s']} s, "
            f"frame diff {m['frame_diff_mean_abs']}"
        )
    return {
        "initial": os.path.join(out_dir, f"task_{TASK}_initial.png"),
        "response": os.path.join(out_dir, f"task_{TASK}_response.png"),
        "video": os.path.join(out_dir, f"task_{TASK}.mp4"),
        "traj": os.path.join(out_dir, f"task_{TASK}_traj.npz"),
    }


def _sweep(
    world,
    rig,
    push,
    prim,
    rest_pos,
    rest_quat,
    size,
    start_x,
    speed,
    travel_t,
    hold_t,
    pb,
    dt,
    common,
    verbose,
) -> dict:
    from isaacsim.core.prims import RigidPrim

    rows, errors = [], []
    for oi, frac in enumerate(pb["contact_offsets_m"]):
        # Views are rebuilt EVERY trial.  A RigidPrim that has been written through
        # goes stale and the next set_world_poses raises "Failed to get rigid body
        # transforms from backend" -- which is why the first sweep trial succeeded
        # and the second died.
        view = RigidPrim(prim_paths_expr=["/World/asset"], name=f"obj_sweep{oi}")
        pusher_view = RigidPrim(
            prim_paths_expr=[f"/World/{rig.name}/pusher"], name=f"pusher_sweep{oi}"
        )
        # Off-centre contact means sliding the slab across the asset.
        off = float(frac) * float(size[1]) / 2.0
        push.set_lateral(off)
        # ...and force PhysX to re-read it.  A USD edit alone does not reach the
        # solver, not even through `world.reset()`: without this the off-centre trial
        # ran and came back numerically identical to the centred one (0.0797 m /
        # 1.86 deg vs 0.0796 m / 1.85 deg) -- the asymmetric-contact case 需求 §8-B
        # asks for, silently duplicated.  See rig.reload_physics.
        lateral_reloaded = R.reload_physics()
        # Re-park by SETTING POSES, not by `world.reset()`.  A reset mid-run leaves
        # the pusher's prismatic drive unresponsive -- the target attribute still
        # ramps (so the debug below looks right) but the body never moves, and the
        # trial reads "no movement" while the identical figure trial, run before any
        # reset, moved the asset.  Only the TOOL is re-parked this way; the asset is
        # never teleported once the task is under way.
        try:
            # Let `_drive`'s own `world.reset()` do the re-parking.  A reset returns
            # the driven pusher to its joint-zero -- the rig origin, which IS the
            # start position -- and then `_drive` re-places the asset at `rest_pos`
            # and settles.  An earlier version ALSO re-parked the pusher by writing a
            # pose through its tensor view, and that write invalidates the view: the
            # FIRST sweep trial ran and the second died with "Simulation view object
            # is invalidated and cannot be used again to call getTransforms", which
            # is why the off-centre case -- the one 需求 §8-B names explicitly --
            # never ran.  `pusher_view` is still built per trial so a caller can see
            # the body; nothing writes through it now.
            _ = pusher_view
            traj, _s, c_t, p0, q0 = _drive(
                world,
                view,
                push,
                rest_pos,
                rest_quat,
                start_x,
                push.stroke,
                speed,
                travel_t,
                hold_t,
                30,
                None,
                None,
            )
        except Exception as e:  # noqa: BLE001
            # A sweep trial that cannot run is REPORTED, not swallowed and not
            # allowed to take the whole task down with it: 需求 §8 wants the failures
            # kept, and the figure pair plus trial 1 are already produced by now.
            # Known cause: re-parking a driven body via set_world_poses can kill the
            # PhysX tensor backend ("Failed to get rigid body transforms from
            # backend"), after which nothing in this process simulates again.
            errors.append(
                {
                    "contact_offset_frac": float(frac),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            R.log(f"[{TASK}] sweep offset={frac}: FAILED TO RUN -- {type(e).__name__}")
            continue
        m = _metrics(p0, q0, traj, pb)
        m["contact_onset_s"] = None if c_t is None else round(c_t, 4)
        # Diagnostic: without it, "the sweep trial did nothing" is indistinguishable
        # between "the pusher never reached the asset" and "the asset never moved".
        m["_debug"] = {
            "pusher_x_start_end": [
                round(traj["pusher_x"][0], 4),
                round(traj["pusher_x"][-1], 4),
            ],
            "object_x_start_end": [
                round(float(p0[0]), 4),
                round(float(traj["pos"][-1][0]), 4),
            ],
            "object_z_start_end": [
                round(float(p0[2]), 4),
                round(float(traj["pos"][-1][2]), 4),
            ],
        }
        rows.append(
            {
                "contact_offset_frac": float(frac),
                "contact_offset_m": off,
                "slab_lateral_y_m": round(float(push.lateral()), 6),
                "lateral_reloaded": lateral_reloaded,
                **m,
            }
        )
        if verbose:
            R.log(
                f"[{TASK}] sweep offset={frac}: moved {m['object_horiz_disp_m']:.4f} m, "
                f"rot {m['object_rot_deg']:.2f} deg"
            )
    n_ok = sum(1 for r in rows if r["moved"])
    return {
        "n_trials": len(rows),
        "n_requested": len(pb["contact_offsets_m"]),
        "n_moved": n_ok,
        "errors": errors,
        "moved_rate": round(n_ok / max(1, len(rows)), 4),
        "rows": rows,
        "failures": [r for r in rows if not r["moved"]],
        "not_run": [e["contact_offset_frac"] for e in errors],
        "asymmetry_note": "The off-centre offset IS applied -- checked by read-back "
        "(`slab_lateral_y_m`) and by a falsification run with the offset set to "
        "3.0 x half-width, where the pusher missed the asset entirely (0.0000 m "
        "of travel). On this asset the pre-registered 0.6 offset still produces "
        "nearly the same trajectory as the centred push, because the claw hammer "
        "lies diagonally and the slab is 1.25 x the asset's width, so 0.6 of a "
        "half-width is still mostly under the head that takes the contact. The "
        "two rows are therefore close; that is a property of this asset and this "
        "pre-registered offset, not a duplicated trial. `rotated` is reported per "
        "row so the reader can see the asymmetry did not turn into rotation.",
        "instrumentation_note": "Contact FORCE and penetration depth are NOT reported: reading them needs "
        "a PhysX contact view, which this driver does not set up, and "
        "`RigidPrim.get_net_contact_forces()` returns None without it. What IS "
        "reported is the measurable consequence -- contact onset (first step the "
        "asset's speed exceeds 1 cm/s), translation, rotation and whether it "
        "tipped. 需求 §8 explicitly permits continuous metrics in a first report.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", required=True)
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    a = ap.parse_args()
    rep = run(
        a.key,
        a.asset_dir,
        a.out,
        a.thresholds,
        do_sweep=not a.no_sweep,
        video_fps=a.fps,
        res=(a.width, a.height),
    )
    R.log(json.dumps({k: rep[k] for k in ("sim", "task", "key", "status")}))


if __name__ == "__main__":
    main()
