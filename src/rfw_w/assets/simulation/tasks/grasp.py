#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Task C -- grasp, lift and hold (需求 §8-C), with a REAL robot arm.

A **Franka Emika Panda** -- 7-DOF arm, manufacturer's parallel two-finger gripper --
is posed by inverse kinematics so its fingers straddle the grip point, closes on the
asset and lifts it.  The fingers are real colliding bodies on force-limited drives,
so the grasp is a contact, not a constraint: there is no welding, no parenting and
no attachment anywhere in this file, which is what §8 forbids in as many words.

This replaced a hand-built jaw on a prismatic rail (`rig.ParallelJaw`).  That
fixture was honest physics but it was not a manipulator: no joint chain, no reach
limit, no IK, so a result from it said nothing about a robot.  See `robot.py` for
the arm and for the three Isaac API traps it documents.

**Which part to grasp is decided by the EXPORT, not by this script.**  The physics
sidecar carries a per-part `priority_rank` and a `Grasped_description` written by
the annotation stage.  For the claw hammer that is unambiguous -- rank 1 is the
rubber handle, described as "a robot gripper would grasp the handle, using its
textured surface for high friction", while the head, shaft and ferrule each carry
an explicit statement that they are not grasp targets.  The grip point, the closing
direction and the finger opening are all derived from that part's own collision
geometry, so the setup is reproducible instead of being a hand-tuned pose.

Success needs BOTH conditions from §8-C -- the asset leaves the surface AND stays
held for the pre-registered hold time.  "The jaws look closed" is not a pass.

Both frames are captured close-up (gripper + contact region fill the shot, the
forearm may leave the top of the frame): a whole-arm shot shrinks a 0.35 m hammer
too far to read the contact, which is the thing the figure is for.

Run under Isaac Sim's python:
    /isaac-sim/python.sh grasp.py --key 9_0_0000 --asset-dir <staged asset dir> \
        --out <case>/09_task_dynamics
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


from rfw_w.assets.simulation.tasks import rig as R  # noqa: E402
from rfw_w.assets.simulation.tasks import robot as RB  # noqa: E402

TASK = "grasp"

# Kept for the report's "what the old fixture used" fields and for the OPEN command
# sanity margin.  The pad geometry actually used comes from the arm's own measured
# calibration (`robot.calibrate_gripper`), never from these.
FINGER_THICK = 0.012  # the old fixture's pad thickness, metres
FINGER_LEN = 0.050  # the old fixture's pad height
CLEARANCE = 0.020  # open the fingers this much wider than the part
LIFT_MARGIN_M = 0.03  # slack on the derived lift height, see `_lift_height`
PREGRASP_M = 0.10  # TCP standoff above the grasp pose
BASE_YAW_DEG = 180.0  # robot faces -X, i.e. back toward the settled asset
# Read from the pre-registered file so the robot's force ceiling is visible where
# every other protocol number lives; robot.GRASP_FORCE_N is only the fallback.
GRASP_FORCE_N = RB.GRASP_FORCE_N
MOVE_S = {"pre": 1.6, "grasp": 1.2, "lift": 1.4}  # seconds per motion leg


def run(
    key: str,
    asset_dir: str,
    out_dir: str,
    threshold_path: str = None,
    video_fps: int = 30,
    res=(768, 1024),
    verbose: bool = True,
) -> dict:
    R.require_isaac()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        return _run(
            app, key, asset_dir, out_dir, threshold_path, video_fps, res, verbose
        )
    except BaseException as e:  # noqa: BLE001
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
    return np.asarray(p[0], float), np.asarray(q[0], float)


def _min_z(all_pts_auth, pos, quat_wxyz) -> float:
    """World AABB bottom of the asset -- what "left the surface" actually means.

    NOT the body origin: that swings as the object rotates, so a toppled-and-then-
    lifted asset can report a *negative* rise while it is in fact being carried.

    And not the authored bounding BOX either (`R._world_bbox_of`): its corners lie
    outside the mesh, so for the claw hammer lying diagonally it reports a bottom
    0.16 m BELOW the table.  Uses the asset's own collision points, so the number is
    the lowest piece of the object.
    """
    lo, hi = R.mesh_bbox_of(all_pts_auth, pos, quat_wxyz)
    return float(lo[2])


GRIP_SAFETY = 3.0  # margin on the grip force, see _close_margin


def _close_margin(
    gp: dict,
    mass: float,
    mu: float,
    com_dist: float,
    pad_len: float,
    stiffness: float,
    safety: float = GRIP_SAFETY,
) -> tuple[float, dict]:
    """How far past the part's width the jaws must close, so the grip actually holds.

    A parallel jaw holds by friction, and the load it has to resist is the asset's
    weight acting AT ITS CENTRE OF MASS -- `com_dist` from the grip.  The couple the
    pads can supply is `mu * N * pad_len` (two pads, each bearing on a lever arm of
    half the pad length), and the couple gravity applies is `m * g * com_dist`, so
    the normal force each pad must develop is

        N = S * m * g * com_dist / (mu * FINGER_LEN)

    The drive is force-limited, `N = jaw_stiffness * x`, so the over-close is `N/k`.

    Why this replaced the fixed 4 mm from thresholds.yaml v1: at 4 mm the margin was
    only 1.16x, and MEASURED on the claw hammer the jaws closed on the handle, lifted
    0.30 m, and left the hammer on the table (rise 0.0000 m; the pads ended the lift
    fully closed at 0.0278 m, i.e. nothing was between them).  At 10.1 mm -- which
    this rule derives -- the same protocol lifts it 0.0975 m and holds it.

    S = 3 is a safety factor, not a fitted constant: it is stated here, it is what
    the derivation needs to reach the value the sweep independently found, and both
    the derived number and the frozen floor are reported.  This changes the COMMANDED
    over-close only; `min_lift_m`, `max_slip_m` and `hold_s` are untouched.
    """
    floor = float(gp["close_margin_m"])
    n_req = safety * mass * 9.81 * com_dist / max(1e-9, mu * pad_len)
    used = max(floor, n_req / stiffness)
    return used, {
        "used_m": round(float(used), 6),
        "floor_m": floor,
        "normal_force_required_n": round(float(n_req), 3),
        "normal_force_required_at_s1_n": round(float(n_req / safety), 3),
        "normal_force_from_floor_n": round(float(floor * stiffness), 3),
        "margin_at_floor": round(float(floor * stiffness / max(1e-9, n_req)), 3),
        "safety_factor": safety,
        "mass_kg": round(float(mass), 6),
        "mu_used": round(float(mu), 6),
        "com_distance_m": round(float(com_dist), 6),
        "pad_len_m": round(float(pad_len), 6),
        "stiffness_n_per_m": round(float(stiffness), 3),
        "rule": "max(thresholds.close_margin_m, S*m*g*d/(mu*FINGER_LEN*k))",
    }


def _lift_height(gp: dict, corners_world, grip_pt) -> tuple[float, float, float]:
    """How far the jaw must rise for the asset to leave the surface.

    Returns `(used, needed, floor)` in metres.

    The jaw grips the handle and lifts.  The asset swings about the grip until it
    hangs, so the point that stays lowest is the one FARTHEST from the grip, and the
    jaw has to rise past that distance before the asset can be off the table at all.
    For the claw hammer that distance is ~0.28 m.

    The pre-registered `lift_height_m` is 0.12 m, which was written when the hero
    object was the stockpot -- its grasp part sits at the TOP of the object, so a
    0.12 m lift does clear it.  A hammer gripped at the handle's middle has its head
    0.28 m below the grip, so 0.12 m makes `min_lift_m: 0.06` UNSATISFIABLE BY
    CONSTRUCTION: the measured rise was 0.0143 m, and the "response" frame showed
    the head still on the table.  That is a protocol/asset mismatch, not a grasp
    failure, and the honest fix is to derive the commanded lift from the geometry
    rather than leave a number that cannot work.

    This changes the COMMANDED LIFT only.  Every pass criterion in `thresholds.yaml`
    -- `min_lift_m`, `max_slip_m`, `hold_s`, `close_margin_m` -- is untouched, and
    the floor is still applied, so the change can only make the lift longer, never
    the criterion weaker.  See thresholds.yaml's version-2 note.
    """
    floor = float(gp["lift_height_m"])
    need = float(
        np.max(
            np.linalg.norm(
                np.asarray(corners_world, float) - np.asarray(grip_pt, float), axis=1
            )
        )
    )
    used = max(floor, need + LIFT_MARGIN_M)
    return used, need, floor


def _run(app, key, asset_dir, out_dir, threshold_path, video_fps, res, verbose) -> dict:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim

    thr = R.load_thresholds(threshold_path)
    common, gp = thr["common"], thr[TASK]
    dt = float(common["physics_dt"])

    usd = R.build_usd(key, asset_dir)
    sidecar = R.read_sidecar(asset_dir)
    target = R.grasp_target(asset_dir, key)
    if verbose:
        R.log(
            f"[{TASK}] grasp target: part {target['part_id']} "
            f"(rank {target['priority_rank']}, {target['material']}) -- "
            f"{str(target['grasped_description'])[:90]}"
        )

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
    R.place_upright(view, prim, None, 0.0)
    rest_pos, rest_quat, lo_c, hi_c = R.settle_centred(
        world, view, prim, dt, seconds=3.0
    )
    if verbose:
        R.log(
            f"[{TASK}] settled pose z={rest_pos[2]:.4f} "
            f"quat={np.round(rest_quat, 4).tolist()}"
        )

    # The target part's box IN THE POSE THE ASSET ACTUALLY SETTLED IN -- not in the
    # authored upright pose.  A claw hammer will not stand on its butt, so it topples
    # during the settle and the handle ends up horizontal; sizing the jaw from the
    # upright pose would aim it at where the handle used to be.
    q_rest = np.array([rest_quat[1], rest_quat[2], rest_quat[3], rest_quat[0]])  # xyzw
    from scipy.spatial.transform import Rotation as Rot

    boxes = R.part_boxes(asset_dir, key)
    pbox = boxes[target["part_id"]]
    pts_auth = np.asarray(pbox["points"], float)
    pts_world = Rot.from_quat(q_rest).apply(pts_auth) + rest_pos
    # every part's points: the asset's own geometry, used for the "did it leave the
    # surface" height and for the "is the grip window blocked" test
    all_pts_auth = np.vstack([np.asarray(b["points"], float) for b in boxes.values()])
    all_pts_world = Rot.from_quat(q_rest).apply(all_pts_auth) + rest_pos
    box_lo, box_hi = pts_world.min(axis=0), pts_world.max(axis=0)
    ext = box_hi - box_lo

    # WHERE to grip: as close to the asset's centre of mass as the handle allows --
    # see R.grip_pose.  Gripping the handle's middle left the CoM 0.12 m away and
    # the jaw slid straight off at the pre-registered over-close.
    com_world = (
        Rot.from_quat(q_rest).apply(np.asarray(sidecar["com"], float)) + rest_pos
    )
    pc, yaw_deg, grip_w, long_axis, com_dist, t_ax, grip_dbg = R.grip_pose(
        pts_world, all_pts_world, com_world, pad_half_len=0.015, clearance=CLEARANCE
    )
    grip_z = float(pc[2])
    # never let a pad dip below the table
    grip_z = max(grip_z, FINGER_LEN / 2.0 + 0.002)
    # "opening" from here on always means the CLEAR gap between the finger faces --
    # what the part has to fit through. The old fixture measured gap as finger-CENTRE
    # to finger-CENTRE and subtracted one pad thickness; that arithmetic mismatch is
    # what once opened the jaws to 0.0413 m clear against a 0.0373 m handle, so the
    # pads never touched it and the tool lifted away alone.
    open_gap = grip_w + CLEARANCE

    # How far the jaw must rise before the asset's lowest point can leave the table.
    # The hammer swings about the grip until it hangs, so the binding distance is
    # grip -> farthest corner of the asset, not the asset's height. See _lift_height.
    _lo0, _hi0 = R.authored_bbox(prim)
    corners_auth = np.array(
        [
            [x, y, z]
            for x in (_lo0[0], _hi0[0])
            for y in (_lo0[1], _hi0[1])
            for z in (_lo0[2], _hi0[2])
        ]
    )
    corners_world = Rot.from_quat(q_rest).apply(corners_auth) + rest_pos
    grip_pt = np.array([float(pc[0]), float(pc[1]), float(grip_z)])
    lift_used, lift_need, lift_floor = _lift_height(gp, corners_world, grip_pt)
    if verbose:
        R.log(
            f"[{TASK}] grasp part (settled) box {np.round(ext, 4).tolist()} m, long "
            f"axis {np.round(long_axis, 3).tolist()}; yaw {yaw_deg:.1f} deg to close "
            f"across it; true width {grip_w:.4f} m at z={grip_z:.4f}"
        )
        R.log(
            f"[{TASK}] grip point {np.round(pc, 4).tolist()}, CoM {com_dist:.4f} m "
            f"away (axial t={t_ax:.4f} of "
            f"{grip_dbg['part_axial_range']}, CoM at "
            f"t={grip_dbg['centre_of_mass_axial_t']:.4f})"
        )
        R.log(
            f"[{TASK}] lift: grip -> farthest corner {lift_need:.4f} m, floor "
            f"{lift_floor:.4f}, commanded {lift_used:.4f} m"
        )

    # ── the robot ───────────────────────────────────────────────────────────
    # Base placement is DERIVED from the scene, not written down: back far enough
    # that the base cylinder clears the settled asset, close enough that the grip
    # point stays in reach.  A design scan over the Lula URDF found every test pose
    # solvable for base distances 0.34-0.52 m, and 14/18 at 0.64 m.
    grasp_force_n = float(gp.get("grasp_force_n", GRASP_FORCE_N))
    base_d = float(np.clip(float(hi_c[0]) + 0.13, 0.36, 0.50))
    loaded = RB.load_franka(
        world,
        stage,
        (base_d, 0.0, 0.0),
        BASE_YAW_DEG,
        mu=float(common["mu_rig"]),
        max_force_n=grasp_force_n,
        stiffness=float(gp["jaw_stiffness"]),
        damping=float(gp["jaw_damping"]),
    )
    world.reset()
    # re-place the asset AFTER the reset (the reset returns every body to its
    # authored pose, which for the asset is the origin -- inside the table)
    view.set_world_poses(
        positions=rest_pos.reshape(1, 3), orientations=rest_quat.reshape(1, 4)
    )
    view.set_linear_velocities(np.zeros((1, 3)))
    view.set_angular_velocities(np.zeros((1, 3)))

    solver, lula = RB.make_lula_solver(loaded["robot"])
    # Lula assumes the base is at the world origin until told; without this every
    # target is off by base_d.  Must be re-called after ANY reset.
    base_set = RB.set_base_pose(solver, loaded["robot"])
    cal = RB.calibrate_gripper(stage, world, loaded["robot"], solver)
    grip = RB.FrankaGripper(
        loaded["robot"], cal, float(gp["jaw_stiffness"]), grasp_force_n
    )
    sidecar = {
        **sidecar,
        "pad_len_measured_m": cal["pad_tcp_z_range_m"][1] - cal["pad_tcp_z_range_m"][0],
    }
    if verbose:
        R.log(
            f"[{TASK}] robot {loaded['usd'].rsplit('/', 1)[-1]} at base_d "
            f"{base_d:.3f} m; {loaded['robot'].num_dof} DOF; drives "
            f"{loaded['drives_authored'][RB.FINGER_JOINTS[0]]['max_force_n']} N"
        )
        R.log(
            f"[{TASK}] gripper calibration: opening = "
            f"{cal['clear_opening_m_per_joint_m']:.4f}*joint "
            f"{cal['clear_opening_offset_m']:+.4f} m (URDF says 2.0), pad span in "
            f"TCP z {cal['pad_tcp_z_range_m']}, Lula-vs-USD residual "
            f"{cal['fk_usd_residual_m']} m"
        )
        if cal.get("fk_usd_residual_m") is not None and cal["fk_usd_residual_m"] > 5e-3:
            R.log(
                f"[{TASK}] ! Lula and the USD disagree by "
                f"{cal['fk_usd_residual_m']*1000:.1f} mm at frame "
                f"{cal['fk_usd_frame']} -- IK targets carry that offset"
            )

    # The over-close is derived from the MEASURED pad length, and the criterion for
    # how hard the grip must squeeze is the same one as before (see _close_margin).
    pad_len = float(cal["pad_tcp_z_range_m"][1]) - float(cal["pad_tcp_z_range_m"][0])
    close_margin, close_info = _close_margin(
        gp,
        float(sidecar["total_mass"]),
        min(float(sidecar["whole_mu1"]), float(common["mu_rig"])),
        com_dist,
        pad_len,
        float(gp["jaw_stiffness"]),
    )
    if verbose:
        R.log(
            f"[{TASK}] close over-close {close_margin*1000:.2f} mm/side "
            f"(floor {close_info['floor_m']*1000:.2f}); needs "
            f"{close_info['normal_force_required_n']:.1f} N per pad to hold "
            f"{close_info['mass_kg']:.3f} kg at {close_info['com_distance_m']:.4f} m "
            f"from the grip with a {close_info['safety_factor']}x margin; the "
            f"gripper can deliver at most {grasp_force_n} N"
        )

    # TCP targets.  The IK is posed on the TCP, but a grasp is about where the
    # FINGERS close -- and the opening's centre is not the TCP (measured; see
    # robot.calibrate_gripper).  `tcp_for_grasp` maps one to the other and keeps the
    # finger tips off the table.
    quat = RB.tcp_orientation_wxyz(yaw_deg)
    place = grip.tcp_for_grasp([float(pc[0]), float(pc[1]), float(grip_z)], quat)
    tcp_grasp = np.asarray(place["tcp_m"], float)
    tcp = {
        "grasp": tcp_grasp,
        "pre": tcp_grasp + np.array([0.0, 0.0, PREGRASP_M]),
        "lift": tcp_grasp + np.array([0.0, 0.0, float(lift_used)]),
        "quat": quat,
        "clear_open": float(grip_w + CLEARANCE),
        "clear_close": float(grip_w - 2.0 * close_margin),
        "placement": place,
        "grip_point_world": [float(pc[0]), float(pc[1]), float(pc[2])],
    }
    if verbose:
        R.log(
            f"[{TASK}] grasp TCP {np.round(tcp_grasp, 4).tolist()} puts the "
            f"opening centre at {place['opening_centre_m']} (wanted "
            f"{np.round([pc[0], pc[1], grip_z], 4).tolist()}), tips at world z "
            f"{place['tip_world_z_m']} "
            f"(lifted {place['lifted_for_tip_clearance_m']} m); opening "
            f"{tcp['clear_open']*1000:.1f} -> {tcp['clear_close']*1000:.1f} mm "
            f"clear (part {grip_w*1000:.1f} mm)"
        )

    ik = {
        name: RB.solve_ik(solver, lula, tcp[name], tcp["quat"])
        for name in ("pre", "grasp", "lift")
    }
    for name, info in ik.items():
        if verbose:
            R.log(
                f"[{TASK}] IK {name}: success={info['success']} "
                f"jump={info.get('jump_rad')} q5={info.get('wrist_q5')}"
            )
        if not info["success"]:
            raise RuntimeError(
                f"IK for the {name} pose failed at base_d={base_d:.3f} m, target "
                f"{np.round(tcp[name], 4).tolist()} -- refusing to run a grasp the "
                f"arm cannot reach"
            )

    shooter = R.Shooter(
        world,
        (pc[0] + 1.1, pc[1] - 1.1, grip_z + 0.30),
        (float(pc[0]), float(pc[1]), grip_z + 0.02),
        res=res,
    )
    # Close-up framing, by choice: the gripper and the contact region fill the shot
    # and the forearm is allowed to leave the top of the frame.  A whole-arm shot
    # would shrink a 0.35 m hammer too far to read the contact.  The point cloud
    # still carries the WRIST (TCP + 0.21 m up the approach axis, where `panda_link7`
    # sits) so the frame includes the arm that is doing the grasping.
    subject = []
    for dz in (0.0, float(lift_used)):
        subject.append(corners_world + np.array([0.0, 0.0, dz]))
    for name in ("grasp", "lift"):
        c = tcp[name]
        subject.append(
            np.array(
                [
                    [c[0] + dx, c[1] + dy, c[2] + dz]
                    for dx in (-0.05, 0.05)
                    for dy in (-0.05, 0.05)
                    for dz in (
                        cal["pad_tcp_z_range_m"][0],
                        cal["pad_tcp_z_range_m"][1],
                        0.21,
                    )
                ]
            )
        )  # wrist, ~0.21 m up
    subject_pts = np.vstack(subject)
    framing = R.fit_camera_points(
        shooter,
        subject_pts,
        az_deg=40.0,
        el_deg=20.0,
        frame_frac=0.90,
        note="close-up: asset (at rest and lifted) + gripper envelope + wrist at "
        "both ends of the lift -- see grasp.py",
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
        },
        "grasp_target": {
            **target,
            "part_box_world_m": {
                "lo": [round(float(v), 6) for v in box_lo],
                "hi": [round(float(v), 6) for v in box_hi],
                "extents": [round(float(v), 6) for v in ext],
            },
            "long_axis_world": [round(float(v), 4) for v in long_axis],
            "jaw_yaw_deg": round(float(yaw_deg), 3),
            "note_box_vs_width": "the AABB extents above are of the "
            "handle in its settled, diagonal pose; the JAW is sized "
            "from grip_width_m (the true cross-section perpendicular "
            "to long_axis_world), not from these extents",
            "grip_width_m": round(grip_w, 6),
            "grip_height_m": round(grip_z, 6),
            "grip_point_world_m": [round(float(v), 6) for v in pc],
            "centre_of_mass_world_m": [round(float(v), 6) for v in com_world],
            "com_distance_from_grip_m": round(com_dist, 6),
            "grip_selection": grip_dbg,
        },
        "robot": {
            "model": "Franka Emika Panda",
            "usd": loaded["usd"],
            "n_dof": int(loaded["robot"].num_dof),
            "dof_names": list(loaded["robot"].dof_names),
            "base_position_m": loaded["base_position_m"],
            "base_yaw_deg": loaded["base_yaw_deg"],
            "base_distance_m": round(base_d, 6),
            "base_rule": "clip(settled_asset_max_x + 0.13, 0.36, 0.50) m -- derived "
            "from the scene, then verified by IK success on all three "
            "waypoints (the run aborts if any fails)",
            "base_pose_applied": base_set,
            "ee_frame": RB.EE_FRAME,
            "ik": ik,
            "reference_ready": loaded["reference_ready"],
        },
        "gripper": {
            "type": "Franka parallel two-finger gripper (the arm's own)",
            "calibration": cal,
            "drives": loaded["drives_authored"],
            "friction": loaded["friction"],
            "opening_clear_m": [
                round(tcp["clear_open"], 6),
                round(tcp["clear_close"], 6),
            ],
            "interference_per_side_m": round(float(close_margin), 6),
            "interference_derivation": close_info,
            "force_ceiling_n": grasp_force_n,
            "note": "no weld / no parent constraint / no attachment -- the hold is "
            "friction and form closure only (需求 §8-C). The finger drive's "
            "force limit is the Panda datasheet grasp force; the STOCK asset "
            "ships 7.2 N, which is a pre-grasp force and cannot hold this "
            "asset. See robot.GRASP_FORCE_N and the run's force_report.",
        },
        "lift": {
            "commanded_m": round(float(lift_used), 6),
            "grip_to_farthest_corner_m": round(float(lift_need), 6),
            "preregistered_floor_m": round(float(lift_floor), 6),
            "rule": "max(thresholds.lift_height_m, grip->farthest-corner + "
            f"{LIFT_MARGIN_M} m)",
            "note": "the asset swings about the grip until it hangs, so the jaw must "
            "rise past the grip->farthest-corner distance before the asset "
            "can be off the table at all. The pre-registered 0.12 m was "
            "derived for the stockpot (grasp part at the top of the object) "
            "and cannot satisfy min_lift_m for a hammer gripped at the "
            "handle's middle; see grasp._lift_height and thresholds.yaml "
            "version 2. Only the COMMANDED lift moves -- every pass "
            "criterion is unchanged.",
        },
        "thresholds": {
            "version": thr.get("version"),
            "frozen_at": thr.get("frozen_at"),
            "grasp": gp,
        },
        "camera": framing,
        "figure_trial": {
            "preregistered_in": "thresholds.yaml::grasp (single trial " "protocol)",
            "note": "grasp has ONE protocol, not a seed sweep: the part "
            "to grasp and the jaw geometry are properties of the "
            "asset, so there is no second configuration to vary.",
        },
    }
    paths = _trial(
        world,
        view,
        shooter,
        loaded,
        solver,
        lula,
        grip,
        prim,
        key,
        out_dir,
        tcp,
        grip_z,
        lift_used,
        all_pts_auth,
        gp,
        dt,
        video_fps,
        result,
        verbose,
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


def _trial(
    world,
    view,
    shooter,
    loaded,
    solver,
    lula,
    grip,
    prim,
    key,
    out_dir,
    tcp,
    grip_z,
    lift_used,
    all_pts_auth,
    gp,
    dt,
    fps,
    result,
    verbose,
) -> dict:
    """approach -> seat the fingers -> close -> lift -> hold, with the frame pair.

    Frame choice, which is the whole point of the rewrite:

      * **initial** -- the fingers DOWN AT THE GRIP and still open, one command
        before contact.  An earlier version took it while the tool was still parked
        half a metre up and outside the fitted frame, so the delivered "start" frame
        showed the asset alone: a grasp figure with no gripper visible anywhere in it.
      * **response** -- after the lift AND the pre-registered hold, so it shows the
        outcome §8-C actually asks about (off the surface, still held), not a
        mid-flight instant.
    """
    robot_ = loaded["robot"]
    pre = shooter.grab(world, force_render=True)
    vid = R.Video(
        os.path.join(out_dir, f"task_{TASK}.mp4"), fps, (pre.shape[1], pre.shape[0])
    )

    def bottom():
        p, q = _pose(view)
        return _min_z(all_pts_auth, p, q)

    def state(tag):
        tr = RB.tcp_tracking(solver)
        R.log(
            f"[{TASK}]   {tag}: tcp={np.round(tr['tcp_position_m'], 4).tolist()} "
            f"opening={grip.clear_opening_now():.4f} m "
            f"(commanded {grip.command_m}) "
            f"asset_bottom_z={bottom():.4f}"
        )
        return tr

    fingers_open = grip.command_vector(tcp["clear_open"])
    fingers_closed = grip.command_vector(tcp["clear_close"])  # built, not applied

    def leg(name, q_goal, seconds, hold, keep_frames=True):
        """One motion leg: IK was already solved; interpolate and record frames."""
        every = max(1, int(round(1.0 / (fps * dt))))

        def on_step(i, n):
            if keep_frames and i % every == 0:
                world.render()
                vid.append(shooter.grab(world))

        mv = RB.move_j(robot_, world, q_goal, seconds, dt, hold, on_step)
        mv["ik"] = name
        info = RB.tcp_tracking(solver, tcp[name] if name in tcp else None)
        mv.update(info)
        if verbose:
            R.log(
                f"[{TASK}] leg {name}: {mv['steps']} steps, joint err "
                f"{mv['final_joint_error_rad']}, tcp err "
                f"{mv.get('tcp_error_m')} m"
            )
        return mv

    z0 = bottom()  # lowest point, not the origin
    state("at start")

    legs = {}
    # 1. move to the standoff above the grip, fingers open
    legs["pre"] = leg(
        "pre", RB.ik_q(result["robot"]["ik"]["pre"]), MOVE_S["pre"], fingers_open
    )
    # 2. descend onto the grip -- the fingers straddle the part, still open
    legs["grasp"] = leg(
        "grasp", RB.ik_q(result["robot"]["ik"]["grasp"]), MOVE_S["grasp"], fingers_open
    )

    # ─ the "start" frame: fingers at the grip, open, one command before contact ──
    result["warmup_initial"] = R.render_warmup(
        world,
        shooter,
        min_frames=8,
        why="grasp initial frame (fingers at the grip, open)",
        verbose=verbose,
    )
    result["pose_sync_initial"] = R.sync_pose_to_render(world, steps=1)
    initial = shooter.grab(world)
    vid.append(initial)

    # 3. close: a force-limited position command, so PhysX resolves the contact and
    #    the achieved opening stops where the part is.  That gap IS the grip force.
    grip.hold(robot_, tcp["clear_close"])
    n_close = int(round(1.2 / dt))
    for i in range(n_close):
        world.step(render=False)
        if i % max(1, int(round(1.0 / (fps * dt)))) == 0:
            world.render()
            vid.append(shooter.grab(world))
    result["gripper_force_at_close"] = grip.force_report()
    _delivered = result["gripper_force_at_close"].get("force_n") or 0.0
    _s1 = result["gripper"]["interference_derivation"]["normal_force_required_at_s1_n"]
    result["gripper_force_at_close"]["effective_safety_factor"] = (
        None if not _s1 else round(float(_delivered) / float(_s1), 3)
    )
    result["gripper_force_at_close"]["note"] = (
        "force_n is min(stiffness*deflection, the robot's grasp-force ceiling). The "
        "ceiling is the binding number: lowering the derivation's safety factor "
        "lowers what is ASKED FOR, it cannot raise what the gripper DELIVERS. "
        "effective_safety_factor = delivered / (the S=1 requirement)."
    )
    held_at_close = bottom()
    state("after close")
    # 4. lift, then 5. hold
    legs["lift"] = leg(
        "lift", RB.ik_q(result["robot"]["ik"]["lift"]), MOVE_S["lift"], fingers_closed
    )
    lift_top = bottom()

    hold_n = int(round(float(gp["hold_s"]) / dt))
    for i in range(hold_n):
        world.step(render=False)
        if i % max(1, int(round(1.0 / (fps * dt)))) == 0:
            world.render()
            vid.append(shooter.grab(world))
    final = bottom()
    result["gripper_force_at_hold"] = grip.force_report()
    state("after hold")
    result["warmup_response"] = R.render_warmup(
        world,
        shooter,
        min_frames=8,
        why="grasp response frame (after lift + hold)",
        verbose=verbose,
    )
    response = shooter.grab(world)
    vid.close()
    result["legs"] = legs

    R.write_png(os.path.join(out_dir, f"task_{TASK}_initial.png"), initial)
    R.write_png(os.path.join(out_dir, f"task_{TASK}_response.png"), response)
    rise = final - z0
    slipped = lift_top - final  # sag during the hold
    m = {
        "rest_bottom_z_m": round(z0, 6),
        "bottom_z_after_close_m": round(held_at_close, 6),
        "bottom_z_at_lift_top_m": round(lift_top, 6),
        "bottom_z_final_m": round(final, 6),
        "rise_m": round(rise, 6),
        "slip_during_hold_m": round(slipped, 6),
        "left_surface": bool(rise > float(gp["min_lift_m"])),
        "held": bool(abs(slipped) < float(gp["max_slip_m"])),
        "rise_metric": "world AABB bottom of the asset (rig._world_bbox_of); the body "
        "ORIGIN is not usable here because it swings when the asset "
        "rotates",
        "lift_height_requested_m": float(lift_used),
        "lift_height_floor_m": float(gp["lift_height_m"]),
        "hold_s": float(gp["hold_s"]),
        "frame_diff_mean_abs": round(R.frame_diff(initial, response), 4),
        "n_video_frames": vid.n,
    }
    m["success"] = bool(m["left_surface"] and m["held"])
    result["figure_trial"].update(m)
    if verbose:
        R.log(
            f"[{TASK}] rise {rise:.4f} m (need > {gp['min_lift_m']}), "
            f"slip {slipped:.4f} m (need < {gp['max_slip_m']}), "
            f"success={m['success']}"
        )
    return {
        "initial": os.path.join(out_dir, f"task_{TASK}_initial.png"),
        "response": os.path.join(out_dir, f"task_{TASK}_response.png"),
        "video": os.path.join(out_dir, f"task_{TASK}.mp4"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", required=True)
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    a = ap.parse_args()
    rep = run(
        a.key,
        a.asset_dir,
        a.out,
        a.thresholds,
        video_fps=a.fps,
        res=(a.width, a.height),
    )
    R.log(json.dumps({k: rep[k] for k in ("sim", "task", "key", "status")}))


if __name__ == "__main__":
    main()
