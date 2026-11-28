# SPDX-License-Identifier: Apache-2.0
"""A REAL robot arm for the grasp task -- Franka Emika Panda.

`rig.py` builds test fixtures out of hand-authored rigid bodies (a pusher slab, and
formerly the parallel jaw).  This module is the other kind of thing: an actual
articulated manipulator, loaded from a published robot asset, driven through
inverse kinematics, with the manufacturer's own gripper on the end.

Why the arm replaced the fixture (需求 §8-C asks for a grasp, and a floating jaw on a
rail is not a manipulator): no joint chain, no reach limit, no IK -- none of the
things that make a grasp result say anything about a robot.

The asset itself is NVIDIA's stock Franka USD from the Isaac asset CDN.  Measured on
this box (Isaac Sim 4.5.0-rc.36, headless):

  * 9 DOF: `panda_joint1..7` (revolute) + `panda_finger_joint1/2` (prismatic).
  * finger joint limits are **[0, 0.04] m**, NOT the 0.05 that NVIDIA's own examples
    use -- those examples load `franka_alt_fingers.usd`, a different asset.  Read the
    limit back, never hard-code it.
  * the stock finger drive is `stiffness=10000, damping=1000, maxForce=7.2`.  That
    7.2 N is a pre-grasp force, not a grasp force; see `GRASP_FORCE_N`.
  * Lula inverse kinematics works OFFLINE from the config bundled in
    `/isaac-sim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka`.

Three API facts cost real time to establish and are called out where they bite:

  1. **`compute_inverse_kinematics` wants a wxyz QUATERNION; `compute_forward_kinematics`
     returns a 3x3 MATRIX.**  Lula reorders the input with `quats_to_rot_matrices`,
     which indexes `[1,2,3,0]` -- scalar-first.  So FK's output cannot be fed back
     into IK without `rot_matrices_to_quats`.  This is the quietest way to get a
     wrong-but-plausible arm pose.
  2. **Lula believes the robot base is at the world origin** until told otherwise, and
     `LulaInterfaceHelper.reset()` zeroes it again.  `set_base_pose` must be re-called
     after every `world.reset()` or every target is off by the base offset.
  3. **`Articulation._on_post_reset` re-imposes gains and max forces from the USD on
     EVERY reset**, so drive parameters set through the runtime API are silently
     discarded by the next reset.  They are authored into the USD here, before the
     first reset -- see `author_franka_drives`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


from rfw_w.assets.simulation.tasks import rig as R  # noqa: E402

# The Isaac asset CDN.  Overridable so a pipeline can pin a local mirror later --
# same spirit as ISAAC_PYTHON / PP_EXPORT_ROOT in rig.py.  Note this is a NETWORK
# dependency the drop/push tasks do not have; the run report records the URL used.
FRANKA_USD = os.environ.get(
    "PP_FRANKA_USD",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/4.5/Isaac/Robots/Franka/franka.usd",
)

FRANKA_PRIM = "/World/Franka"
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")
EE_FRAME = "right_gripper"  # the point between the fingers; URDF-only frame
EE_PRIM_NAME = "panda_rightfinger"
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))

# Franka Emika Panda datasheet continuous grasp force.  The stock USD ships 7.2 N
# on the finger drive, which is a PRE-grasp force; the datasheet number is what the
# hardware can actually hold with.  Floored, never lowered -- see
# grasp._close_margin for how the required force is derived, and the run report for
# the force actually delivered.
GRASP_FORCE_N = 70.0

MOVE_SMOOTHSTEPS = 1  # smoothstep exponent is 3-2t; kept for readability


# ── loading ──────────────────────────────────────────────────────────────────
def wait_for_franka(stage, prim: str = FRANKA_PRIM, frames: int = 240) -> dict:
    """Pump the app until the referenced robot's prims actually exist.

    `add_reference_to_stage` with a REMOTE url does not compose synchronously: the
    asset is fetched through omni.client, and asking for `.../panda_finger_joint1`
    immediately afterwards finds nothing.  Returns how many updates it took, so the
    cost is visible rather than hidden in a sleep.
    """
    import omni.kit.app
    from pxr import Usd

    app = omni.kit.app.get_app()
    found = {}
    for i in range(int(frames)):
        root = stage.GetPrimAtPath(prim)
        if root.IsValid() and stage.GetPrimAtPath(f"{prim}/panda_link0").IsValid():
            # The joints are NOT direct children of the mount path (only the links
            # and `rootJoint` are), so search the subtree rather than guessing.
            found = {
                p.GetName(): str(p.GetPath())
                for p in Usd.PrimRange(root)
                if p.GetName() in FINGER_JOINTS
            }
            if len(found) == len(FINGER_JOINTS):
                return {"app_updates": i, "ready": True, "joint_paths": found}
        app.update()
    return {
        "app_updates": int(frames),
        "ready": False,
        "joint_paths": found,
        "children": [c.GetName() for c in stage.GetPrimAtPath(prim).GetChildren()],
    }


def find_franka_joints(stage, prim: str = FRANKA_PRIM) -> dict:
    """Name -> path for the finger joint prims, wherever the asset put them.

    They are NOT at `<mount>/panda_finger_joint1`; that path does not exist and
    assuming it is what made the first version of this module fail.  The links are
    direct children of the mount path, the joints are not.
    """
    from pxr import Usd

    root = stage.GetPrimAtPath(prim)
    if not root.IsValid():
        return {}
    return {
        p.GetName(): str(p.GetPath())
        for p in Usd.PrimRange(root)
        if p.GetName() in FINGER_JOINTS
    }


def author_franka_drives(
    stage,
    max_force_n: float,
    stiffness: float = None,
    damping: float = None,
    prim: str = FRANKA_PRIM,
    joint_paths: dict = None,
) -> dict:
    """Write the finger drive's force limit (and optionally gains) into the USD.

    MUST run after the reference is added and BEFORE the first `world.reset()`.
    `Articulation._on_post_reset` restores gains and max forces from the USD on every
    reset, so the same values set through `Articulation.set_max_efforts()` are
    dropped by the next reset -- silently, leaving a gripper that looks commanded and
    squeezes with 7.2 N.

    Returns what was written, for the report.
    """
    from pxr import UsdPhysics

    paths = dict(joint_paths or find_franka_joints(stage, prim))
    out = {}
    for jn in FINGER_JOINTS:
        if jn not in paths:
            raise RuntimeError(
                f"finger joint {jn} not found under {prim} -- the Franka reference "
                f"must be added and composed BEFORE the drives are authored; found "
                f"{sorted(paths)}"
            )
        p = stage.GetPrimAtPath(paths[jn])
        if not p.IsValid():
            raise RuntimeError(f"{paths[jn]} is not a valid prim")
        drive = (
            UsdPhysics.DriveAPI(p, "linear")
            if p.HasAPI(UsdPhysics.DriveAPI, "linear")
            else UsdPhysics.DriveAPI.Apply(p, "linear")
        )
        drive.GetMaxForceAttr().Set(float(max_force_n))
        if stiffness is not None:
            drive.GetStiffnessAttr().Set(float(stiffness))
        if damping is not None:
            drive.GetDampingAttr().Set(float(damping))
        out[jn] = {
            "max_force_n": float(drive.GetMaxForceAttr().Get()),
            "stiffness": float(drive.GetStiffnessAttr().Get()),
            "damping": float(drive.GetDampingAttr().Get()),
        }
    return out


def read_franka_drives(stage, prim: str = FRANKA_PRIM) -> dict:
    """Read the finger drives back -- so a report can say what PhysX will actually
    impose rather than what the caller believes it wrote."""
    from pxr import UsdPhysics

    paths = find_franka_joints(stage, prim)
    out = {}
    for jn in FINGER_JOINTS:
        p = stage.GetPrimAtPath(paths[jn]) if jn in paths else None
        d = (
            UsdPhysics.DriveAPI.Get(p, "linear")
            if p is not None and p.IsValid()
            else None
        )
        if d is None:
            continue
        out[jn] = {
            "max_force_n": d.GetMaxForceAttr().Get(),
            "stiffness": d.GetStiffnessAttr().Get(),
            "damping": d.GetDampingAttr().Get(),
        }
    return out


def bind_finger_friction(
    stage, mu: float, prim: str = FRANKA_PRIM, links=FINGER_LINKS
) -> dict:
    """Bind a PhysicsMaterial with friction `mu` to the gripper's colliders.

    The stock robot carries the USD default friction, which is NOT the `mu` that
    `grasp._close_margin` derives the required grip force from.  Without this the run
    computes the over-close with one coefficient and grips with another -- the same
    silent mismatch the old hand-built jaw avoided by binding its own material.

    Both purposes, like `rig.make_table`: a surface shader for looks, a
    PhysicsMaterial for contact.  Binding the visual material to "physics" leaves the
    contact on the default friction.
    """
    from pxr import Usd, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{prim}/gripper_phymat")
    mp = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mp.CreateStaticFrictionAttr(float(mu))
    mp.CreateDynamicFrictionAttr(float(mu))
    mp.CreateRestitutionAttr(0.0)
    n = 0
    for link in links:
        for p in Usd.PrimRange(stage.GetPrimAtPath(f"{prim}/{link}")):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(
                    mat, UsdShade.Tokens.weakerThanDescendants, "physics"
                )
                n += 1
    return {"n_colliders_bound": n, "mu": float(mu), "material": str(mat.GetPath())}


def load_franka(
    world,
    stage,
    base_pos,
    base_yaw_deg: float = 180.0,
    mu: float = 1.0,
    max_force_n: float = GRASP_FORCE_N,
    stiffness: float = None,
    damping: float = None,
    usd_path: str = None,
) -> dict:
    """Load the Panda at `base_pos`/`base_yaw_deg` and register it with `world`.

    ORDER IS A HARD REQUIREMENT (see the module docstring):
        add_reference_to_stage -> author drives + friction -> scene.add -> world.reset()
    `Articulation` resolves its root at CONSTRUCTION, and `SingleManipulator.initialize`
    (called by the scene on reset) builds the end-effector path from the prim path --
    both need the reference to already be there.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot.manipulators import SingleManipulator
    from isaacsim.robot.manipulators.grippers import ParallelGripper

    usd_path = usd_path or FRANKA_USD
    add_reference_to_stage(usd_path=usd_path, prim_path=FRANKA_PRIM)
    ready = wait_for_franka(stage)
    if not ready.get("ready"):
        raise RuntimeError(
            f"the Franka reference did not compose from {usd_path!r} after "
            f"{ready.get('app_updates')} app updates; children seen: "
            f"{ready.get('children')}"
        )
    drives = author_franka_drives(
        stage, max_force_n, stiffness, damping, joint_paths=ready.get("joint_paths")
    )
    friction = bind_finger_friction(stage, mu)

    yaw = np.radians(float(base_yaw_deg))
    base_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

    # 0.04 is the STOCK joint limit; read it back rather than trusting this default
    # (NVIDIA's examples use 0.05, which belongs to franka_alt_fingers.usd).
    opened = np.array([0.04, 0.04])
    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PRIM}/{EE_PRIM_NAME}",
        joint_prim_names=list(FINGER_JOINTS),
        joint_opened_positions=opened,
        joint_closed_positions=np.array([0.0, 0.0]),
    )
    robot = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PRIM,
            name="franka",
            end_effector_prim_name=EE_PRIM_NAME,
            position=np.asarray(base_pos, float),
            orientation=base_quat,
            gripper=gripper,
        )
    )
    # before the reset: ParallelGripper.post_reset teleports the fingers to the
    # default state on every reset, and only fills it in if it is still None.
    gripper.set_default_state(opened)
    return {
        "robot": robot,
        "gripper": gripper,
        "usd": usd_path,
        "reference_ready": ready,
        "base_position_m": [round(float(v), 6) for v in base_pos],
        "base_yaw_deg": float(base_yaw_deg),
        "base_quat_wxyz": [round(float(v), 6) for v in base_quat],
        "drives_authored": drives,
        "friction": friction,
    }


def dof_report(robot) -> dict:
    """DOF names and limits as the tensor API reports them.

    Careful with units: this build answers ROTATIONAL limits in DEGREES
    (`panda_joint1` comes back [-166.0, 166.0]) and TRANSLATIONAL limits in metres
    (`panda_finger_joint1` [-0.0, 0.04]).  Do not reuse one as the other.
    """
    out = {"n_dof": int(robot.num_dof), "dof_names": list(robot.dof_names)}
    try:
        lo, hi = robot.get_dof_limits()
        lo = np.asarray(lo[0], float)
        hi = np.asarray(hi[0], float)
        out["limits"] = {
            n: [round(float(a), 6), round(float(b), 6)]
            for n, a, b in zip(out["dof_names"], lo, hi)
        }
        i = out["dof_names"].index(FINGER_JOINTS[0])
        out["finger_limit_m"] = [round(float(lo[i]), 6), round(float(hi[i]), 6)]
    except Exception as e:  # noqa: BLE001
        # SingleManipulator does not expose get_dof_limits; read the joints off the
        # USD instead, which is also what PhysX parses.
        out["limits_error"] = f"{type(e).__name__}: {e}"
        try:
            from pxr import Usd, UsdPhysics

            stage = robot.stage
            lim = {}
            for p in Usd.PrimRange(stage.GetPrimAtPath(robot.prim_path)):
                j = UsdPhysics.PrismaticJoint(p) or UsdPhysics.RevoluteJoint(p)
                if j:
                    lim[p.GetName()] = [
                        j.GetLowerLimitAttr().Get(),
                        j.GetUpperLimitAttr().Get(),
                    ]
            out["limits_from_usd"] = lim
        except Exception as e2:  # noqa: BLE001
            out["limits_usd_error"] = f"{type(e2).__name__}: {e2}"
    return out


# ── kinematics ──────────────────────────────────────────────────────────────
def make_lula_solver(robot, ee_frame: str = EE_FRAME):
    """Lula IK/IK for the Panda, from the config bundled with the extension.

    Loaded from the extension's LOCAL `motion_policy_configs/franka`, so inverse
    kinematics needs no network even though the robot USD does.
    """
    import isaacsim.robot_motion.motion_generation.interface_config_loader as icl
    from isaacsim.robot_motion.motion_generation.lula.kinematics import (
        LulaKinematicsSolver,
    )
    from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import (
        ArticulationKinematicsSolver,
    )

    lula = LulaKinematicsSolver(
        **icl.load_supported_lula_kinematics_solver_config("Franka")
    )
    solver = ArticulationKinematicsSolver(robot, lula, ee_frame)
    return solver, lula


def set_base_pose(solver, robot) -> dict:
    """Tell Lula where the robot actually is.  RE-CALL AFTER EVERY `world.reset()`.

    Lula starts out believing the base is at the world origin with identity rotation,
    and both its FK and IK transform through that belief.  With the base at x=0.44
    every target would be off by 0.44 m.  `ArticulationKinematicsSolver` does not
    wrap this, so reach through to the kinematics solver.
    """
    p, q = robot.get_world_pose()  # articulation root == panda_link0 frame
    p = np.asarray(p, float)
    q = np.asarray(q, float)  # wxyz
    solver.get_kinematics_solver().set_robot_base_pose(
        robot_position=p, robot_orientation=q
    )
    return {
        "base_position_m": [round(float(v), 6) for v in p],
        "base_quat_wxyz": [round(float(v), 6) for v in q],
    }


def tcp_orientation_wxyz(yaw_deg: float, tilt_deg: float = 0.0) -> np.ndarray:
    """Gripper orientation for a top-down grasp closing across `yaw_deg`.

    Hand axes for this gripper (from the Lula URDF: the finger joints slide along
    +/-Y of `panda_hand`, and `right_gripper` sits 0.1 m out along its +Z):

        +Z = approach (out of the palm, between the fingers)  -> must point world-down
        +/-Y = closing                                        -> must be `close_dir`

    `rig.grip_frame` defines `close_dir = (-sin yaw, cos yaw, 0)`, which is exactly
    `Rz(yaw) @ (0,1,0)`, so `R = Rz(yaw) @ Ry(180)` gives it with no sign flip.  Ry
    rather than Rx keeps the pads vertical (square to the part) while flipping X and Z.

    `tilt_deg` leans the approach off vertical -- only useful to escape the wrist
    singularity; it makes the pads contact on an edge, so leave it at 0.
    """
    from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
    from scipy.spatial.transform import Rotation as Rot

    m = (
        Rot.from_euler("z", float(yaw_deg), degrees=True)
        * Rot.from_euler("y", 180.0, degrees=True)
    ).as_matrix()
    if tilt_deg:
        m = m @ Rot.from_euler("x", float(tilt_deg), degrees=True).as_matrix()
    return np.asarray(rot_matrices_to_quats(m), float)


WRIST_SINGULARITY_MIN = 0.15  # rad; |q5| below this is the wrist singularity
BRANCH_JUMP_RAD = 1.0  # a bigger step than this means a different branch


def solve_ik(
    solver,
    lula,
    target_pos,
    target_quat_wxyz,
    position_tol=2e-3,
    orientation_tol=0.02,
    prefer_joint_index: int = 4,
) -> dict:
    """IK with a branch/singularity guard.

    Reading the robot's current joints as the warm start is Lula's default
    (`ArticulationKinematicsSolver` does it for us), and it is what keeps successive
    waypoints on one branch.  It is not sufficient: CCD+BFGS does return a different
    elbow branch, and the naive straight-down solution for a target in front of the
    base lands the wrist on its singularity (q5 ~ 0), where the Jacobian loses rank.

    So: solve, and if the solution jumps away from where the arm is, or sits on the
    singularity, re-seed around the current configuration and solve again.  What
    happened is returned, not swallowed.
    """
    sub = solver.get_joints_subset()
    q_now = np.asarray(sub.get_joint_positions(), float).reshape(-1)
    info = {
        "target_position_m": [round(float(v), 6) for v in target_pos],
        "warm_start": [round(float(v), 4) for v in q_now],
    }
    action, ok = solver.compute_inverse_kinematics(
        np.asarray(target_pos, float),
        np.asarray(target_quat_wxyz, float),
        position_tol,
        orientation_tol,
    )
    q = (
        None
        if action is None or action.joint_positions is None
        else np.asarray(action.joint_positions, float).reshape(-1)
    )
    reseeded = False
    if q is not None and len(q) == len(q_now):
        jump = float(np.max(np.abs(q - q_now)))
        singular = abs(float(q[prefer_joint_index])) < WRIST_SINGULARITY_MIN
        info.update(
            jump_rad=round(jump, 4),
            wrist_q5=round(float(q[prefer_joint_index]), 4),
            wrist_singular=bool(singular),
        )
        if bool(ok) and (jump > BRANCH_JUMP_RAD or singular):
            lula.set_default_cspace_seeds(
                np.stack(
                    [
                        q_now,
                        q_now + 0.2 * np.ones_like(q_now),
                        q_now - 0.2 * np.ones_like(q_now),
                    ]
                )
            )
            action2, ok2 = solver.compute_inverse_kinematics(
                np.asarray(target_pos, float),
                np.asarray(target_quat_wxyz, float),
                position_tol,
                orientation_tol,
            )
            reseeded = True
            q2 = (
                None
                if action2 is None or action2.joint_positions is None
                else np.asarray(action2.joint_positions, float).reshape(-1)
            )
            if q2 is not None and len(q2) == len(q_now):
                jump2 = float(np.max(np.abs(q2 - q_now)))
                sing2 = abs(float(q2[prefer_joint_index])) < WRIST_SINGULARITY_MIN
                # take the re-seeded solution when it is strictly better on both counts
                if (not sing2 and (singular or jump2 < jump)) or (
                    sing2 == singular and jump2 < jump
                ):
                    q, ok = q2, ok2
                    info["jump_rad"] = round(jump2, 4)
                    info["wrist_q5"] = round(float(q2[prefer_joint_index]), 4)
                    info["wrist_singular"] = bool(sing2)
            lula.set_default_cspace_seeds(np.stack([q_now]))
    info["reseeded"] = bool(reseeded)
    info["success"] = bool(ok)
    info["q"] = None if q is None else [round(float(v), 5) for v in q]
    return info


def move_j(
    robot, world, q_goal, seconds: float, dt: float, hold_command=None, on_step=None
) -> dict:
    """Move the arm to `q_goal` over `seconds`, interpolating in JOINT space.

    Applying an IK result directly snaps the arm to the target in one control step --
    which for a grasp means the gripper arrives at the object at whatever speed the
    solver produced, and knocks it over.  Interpolating with a smoothstep (zero
    velocity at both ends) over a fixed duration is the cheap, robust cure, and it is
    what makes the frames readable as a motion.

    `hold_command` is the full-DOF command vector used for the joints that are NOT
    being moved (the fingers); passing it explicitly avoids depending on the
    controller's back-fill of `None` entries.
    """
    from isaacsim.core.utils.types import ArticulationAction

    q_now_full = np.asarray(robot.get_joint_positions(), float).reshape(-1)
    q1 = np.asarray(q_goal, float).reshape(-1)
    # An IK result carries the ARM joints only (7); the robot has 9 with the fingers.
    # The arm joints are DOFs 0..6, so a short goal lands on the first len(q1) slots
    # and the fingers come from `hold_command`.
    if len(q1) > len(q_now_full):
        raise ValueError(
            f"move_j: goal has {len(q1)} joints, robot has " f"{len(q_now_full)}"
        )
    n = max(1, int(round(float(seconds) / dt)))
    base = (
        q_now_full
        if hold_command is None
        else np.asarray(hold_command, float).reshape(-1)
    )
    if len(base) != len(q_now_full):
        raise ValueError("move_j: hold_command must be a full-DOF command vector")
    q_start = q_now_full.copy()
    for i in range(1, n + 1):
        t = i / n
        s = t * t * (3.0 - 2.0 * t)  # smoothstep: zero end velocity
        cmd = base.copy()
        cmd[: len(q1)] = q_start[: len(q1)] + (q1 - q_start[: len(q1)]) * s
        robot.apply_action(ArticulationAction(joint_positions=cmd))
        world.step(render=False)
        if on_step is not None:
            on_step(i, n)
    err = float(
        np.max(
            np.abs(
                np.asarray(robot.get_joint_positions(), float).reshape(-1)[: len(q1)]
                - q1
            )
        )
    )
    return {
        "steps": n,
        "seconds": float(seconds),
        "final_joint_error_rad": round(err, 5),
    }


# ── the gripper ──────────────────────────────────────────────────────────────
def _link_frame_corners(stage, prim_path: str) -> np.ndarray:
    """The 8 corners of a link's geometry, in the LINK's own frame.

    Three frames were tried here and only the third is right.  All of it is measured
    on `panda_leftfinger`, not reasoned about:

    * A WORLD AABB is useless: the hand is tilted, so the two pads' world boxes of a
      rotated bar overlap even though the pads are disjoint.
    * The **link Xform's** `ComputeLocalBound` is NOT the link frame either.  For the
      link prim it returns the box in the robot's AUTHORED BASE FRAME -- measured, z
      in [0.817, 0.871] while the finger's simulated position was z = 0.419, i.e. as
      if the base were at the origin and the arm at its authored pose.  Reading that
      as link-local and applying the measured link transform double-counts the
      geometry; that is what made the first version of this calibration report a pad
      0.8 m from the TCP.
    * The **Mesh prim's** `ComputeLocalBound` is the one that answers the question:
      it gives the finger's box in the finger's own frame (0.021 x 0.026 x 0.054 m),
      and rotating that by the link's measured world pose reproduces the finger's
      known world box exactly (a 45-degree Z rotation of that box gives
      0.0335 x 0.0335 x 0.054, against a measured world box of 0.0343 x 0.0336 x
      0.0543).

    Returns corners in the link frame: the pad sits at Y in [0, 0.0264] and
    Z in [0, 0.0539] relative to the finger's link origin.
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    boxes = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if not p.IsA(UsdGeom.Mesh):
            continue
        rng = cache.ComputeLocalBound(p).ComputeAlignedRange()
        if not rng.IsEmpty():
            boxes.append(
                (np.asarray(rng.GetMin(), float), np.asarray(rng.GetMax(), float))
            )
    if not boxes:
        raise RuntimeError(
            f"{prim_path}: no Mesh prim with a bound -- cannot "
            f"measure the gripper geometry"
        )
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    return _corners(lo, hi)


def _corners(lo, hi):
    return np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )


def _pair_faces(pad_a, pad_b):
    """The two FACING faces of the pads, along the TCP frame's Y.

    Returns `(lo_face, hi_face)` -- the surfaces that actually touch a part -- which
    is not the same as either finger's box, and not the same as the TCP's Y=0.
    Handles either finger being on either side, so it cannot silently return a
    negative gap.
    """
    a_lo, a_hi = float(pad_a[:, 1].min()), float(pad_a[:, 1].max())
    b_lo, b_hi = float(pad_b[:, 1].min()), float(pad_b[:, 1].max())
    if a_hi < b_lo:
        return a_hi, b_lo
    if b_hi < a_lo:
        return b_hi, a_lo
    return float("nan"), float("nan")  # pads overlap -- the assumption is broken


def _pair_gap(pad_a, pad_b) -> float:
    """Clear separation between two pads along the TCP frame's Y."""
    lo, hi = _pair_faces(pad_a, pad_b)
    return hi - lo


def calibrate_gripper(stage, world, robot, solver, tcp_frame: str = EE_FRAME) -> dict:
    """Measure the gripper instead of trusting the URDF.

    Answers, from the SIMULATED geometry:
      * clear opening vs finger joint position -- the URDF says `opening = 2*joint`,
        but that is the URDF's model and this is the asset that will be simulated;
      * where the pads sit along the approach axis relative to the TCP, which is what
        keeps the finger tips out of the tabletop;
      * whether Lula's model and the USD agree at all (`fk_usd_residual_m`).  If that
        is not ~0, every IK result in the run is suspect -- one line, highest value.
    """
    from isaacsim.core.prims import SingleRigidPrim
    from scipy.spatial.transform import Rotation as Rot

    finger_view = {
        n: SingleRigidPrim(prim_path=f"{FRANKA_PRIM}/{n}", name=f"cal_{n}")
        for n in FINGER_LINKS
    }
    corners_in_link = {
        n: _link_frame_corners(stage, f"{FRANKA_PRIM}/{n}") for n in FINGER_LINKS
    }

    sub = solver.get_joints_subset()
    idx = np.asarray(sub.get_joint_subset_indices(), dtype=int)
    j_lo, j_hi = 0.0, 0.04
    try:
        props = robot.dof_properties
        fi = list(robot.dof_names).index(FINGER_JOINTS[0])
        j_lo = float(props["lower"][fi])
        j_hi = float(props["upper"][fi])
    except Exception:  # noqa: BLE001
        # read them off the USD -- `SingleManipulator` has no `dof_properties`
        from pxr import Usd, UsdPhysics

        for p in Usd.PrimRange(stage.GetPrimAtPath(FRANKA_PRIM)):
            if p.GetName() == FINGER_JOINTS[0]:
                jj = UsdPhysics.PrismaticJoint(p)
                if jj:
                    j_lo = float(jj.GetLowerLimitAttr().Get())
                    j_hi = float(jj.GetUpperLimitAttr().Get())

    samples, tcp_pos, tcp_rot = [], None, None
    # Sample two OPEN positions.  At the fully closed end the pads touch, and the
    # clear separation between them is then zero or negative -- which is not a point
    # on the line the fit needs.
    for j in (j_hi, 0.5 * j_hi):
        q_full = np.asarray(robot.get_joint_positions(), float).reshape(-1).copy()
        for k in list(robot.dof_names).index(FINGER_JOINTS[0]), list(
            robot.dof_names
        ).index(FINGER_JOINTS[1]):
            q_full[k] = j
        robot.set_joint_positions(q_full)
        world.step(render=False)
        world.step(render=False)  # give PhysX a step to move the links
        p_tcp, m_tcp = solver.compute_end_effector_pose()
        tcp_pos, tcp_rot = np.asarray(p_tcp, float), np.asarray(m_tcp, float)
        pads, world_pads, link_info = {}, {}, {}
        for n, v in finger_view.items():
            # SingleRigidPrim is the SINGULAR view: get_world_pose() -> (pos, wxyz).
            # RigidPrim (the multi-prim view) is the one with the plural name.
            pw, qw = v.get_world_pose()
            pw = np.asarray(pw, float).reshape(3)
            qw = np.asarray(qw, float).reshape(4)
            rw = Rot.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
            corners_local = corners_in_link[n]  # (8,3), link frame
            corners_world = (rw @ corners_local.T).T + pw  # (8,3), world
            corners_tcp = (tcp_rot.T @ (corners_world - tcp_pos).T).T  # (8,3), TCP
            pads[n], world_pads[n] = corners_tcp, corners_world
            link_info[n] = {
                "world_position_m": [round(float(x), 6) for x in pw],
                "link_frame_bounds": [
                    [
                        round(float(corners_local[:, i].min()), 6),
                        round(float(corners_local[:, i].max()), 6),
                    ]
                    for i in range(3)
                ],
                "world_bounds_z": [
                    round(float(corners_world[:, 2].min()), 6),
                    round(float(corners_world[:, 2].max()), 6),
                ],
            }
        lo_face, hi_face = _pair_faces(pads[FINGER_LINKS[0]], pads[FINGER_LINKS[1]])
        gap = hi_face - lo_face
        allpads = np.vstack(list(pads.values()))
        allworld = np.vstack(list(world_pads.values()))
        samples.append(
            {
                "joint_m": float(j),
                "clear_opening_m": round(gap, 6),
                "faces_tcp_y_m": [round(float(lo_face), 6), round(float(hi_face), 6)],
                "pad_tcp_bounds": [
                    [
                        round(float(allpads[:, i].min()), 6),
                        round(float(allpads[:, i].max()), 6),
                    ]
                    for i in range(3)
                ],
                "pad_world_bounds": [
                    [
                        round(float(allworld[:, i].min()), 6),
                        round(float(allworld[:, i].max()), 6),
                    ]
                    for i in range(3)
                ],
                "tcp_position_m": [round(float(x), 6) for x in tcp_pos],
                "tcp_rot_diag": [round(float(tcp_rot[i, i]), 4) for i in range(3)],
                "links": link_info,
            }
        )

    measured_offset = None
    if (
        len(samples) == 2
        and np.isfinite(samples[0]["clear_opening_m"])
        and np.isfinite(samples[1]["clear_opening_m"])
    ):
        dj = samples[1]["joint_m"] - samples[0]["joint_m"]
        slope = (samples[1]["clear_opening_m"] - samples[0]["clear_opening_m"]) / dj
        measured_offset = samples[0]["clear_opening_m"] - slope * samples[0]["joint_m"]
    else:
        slope = 2.0
    # The OFFSET cannot be trusted from the geometry, and the physics says so.
    #
    # The pad boxes come from the finger MESH's local bound, and the two fingers come
    # back with IDENTICAL boxes -- which a mirror-symmetric pair cannot have.  The
    # resulting offset (2*joint - 26.4 mm) is an artefact of the mesh frame, not of
    # the gripper: it implies the pads are 26.4 mm deep along the closing axis, i.e.
    # they would already be touching when fully open at 0.014 m of joint travel.
    #
    # Where the two disagree, PHYSICS is the arbiter, and it agrees with the URDF
    # (`lula_franka_gen.urdf`: the finger joints slide along +/-Y of the hand from a
    # common origin, so clear opening = 2 * joint).  MEASURED: driving the fingers
    # closed on the claw hammer stalled them at joint = 0.01764 m, an opening of
    # 35.3 mm by `2*joint` -- against a handle whose cross-section is 35.6 mm.  The
    # 26.4 mm-offset map would have called that opening 5.4 mm, which the run
    # contradicted by closing to 16.7 mm through empty air.
    #
    # So: slope from the measurement (it agrees with the URDF at 2.000), offset from
    # the URDF, and the rejected measured value is kept in the report.
    offset = 0.0

    lo_z = min(s["pad_tcp_bounds"][2][0] for s in samples)
    hi_z = max(s["pad_tcp_bounds"][2][1] for s in samples)
    # The OPENING'S CENTRE, which is NOT the TCP.  Measured: the two finger meshes
    # are not mirror-symmetric about the TCP's Y, so the gap between the facing
    # surfaces sits to one side of it.  Aiming the TCP at the part therefore puts the
    # part at the EDGE of the opening -- with a 16.7 mm gap that is the difference
    # between gripping and closing on air, which is what the first runs did.
    cx = float(
        np.mean(
            [
                0.5 * (s["pad_tcp_bounds"][0][0] + s["pad_tcp_bounds"][0][1])
                for s in samples
            ]
        )
    )
    cy = float(
        np.mean(
            [0.5 * (s["faces_tcp_y_m"][0] + s["faces_tcp_y_m"][1]) for s in samples]
        )
    )
    cz = 0.5 * (lo_z + hi_z)

    # Lula vs USD at a frame that exists in BOTH descriptions.
    frame, fk_res = None, None
    names = list(solver.get_kinematics_solver().get_all_frame_names())
    for cand in (EE_PRIM_NAME, "panda_leftfinger", "panda_hand"):
        if cand not in names:
            continue
        prim_path = f"{FRANKA_PRIM}/{cand}"
        if not stage.GetPrimAtPath(prim_path).IsValid():
            continue
        p_lula, _ = solver.get_kinematics_solver().compute_forward_kinematics(
            cand,
            np.asarray(robot.get_joint_positions(), float)[
                np.asarray(sub.get_joint_subset_indices(), dtype=int)
            ],
        )
        p_usd = np.asarray(
            SingleRigidPrim(prim_path=prim_path, name=f"xchk_{cand}").get_world_pose()[
                0
            ],
            float,
        )
        frame = cand
        fk_res = float(np.linalg.norm(np.asarray(p_lula, float) - p_usd))
        break

    return {
        "clear_opening_m_per_joint_m": round(float(slope), 6),
        "clear_opening_offset_m": round(float(offset), 6),
        "clear_opening_offset_measured_m": (
            None if measured_offset is None else round(float(measured_offset), 6)
        ),
        "clear_opening_offset_used": "0.0 from the URDF's `opening = 2 * joint`; "
        "the geometry-measured offset was rejected "
        "because the two finger meshes come back "
        "with identical local boxes, which a "
        "mirror-symmetric pair cannot have",
        "joint_limit_m": [round(j_lo, 6), round(j_hi, 6)],
        "samples": samples,
        "pad_tcp_z_range_m": [round(float(lo_z), 6), round(float(hi_z), 6)],
        "pad_centre_tcp_z_m": round(0.5 * (lo_z + hi_z), 6),
        "opening_centre_tcp_m": [round(cx, 6), round(cy, 6), round(cz, 6)],
        "opening_centre_note": "the point the fingers actually close AROUND, in "
        "the TCP frame; it is not the origin because the "
        "two finger meshes are not mirror-symmetric about "
        "the TCP's Y",
        "fk_usd_frame": frame,
        "fk_usd_residual_m": None if fk_res is None else round(fk_res, 6),
        "note": "measured from the simulated geometry in the TCP frame; the world "
        "AABB is not usable here because the two fingers' rotated boxes "
        "overlap. fk_usd_residual_m must be ~0 -- if it is not, Lula's "
        "model and the USD disagree and every IK result is suspect.",
    }


def ik_q(info: dict) -> np.ndarray:
    """The joint solution out of a `solve_ik` result, or a loud failure."""
    q = info.get("q")
    if q is None:
        raise RuntimeError(f"IK produced no joint solution: {info}")
    return np.asarray(q, float)


def apply_command(robot, command) -> None:
    """Send a full-DOF position command through the articulation controller."""
    from isaacsim.core.utils.types import ArticulationAction

    robot.apply_action(
        ArticulationAction(joint_positions=np.asarray(command, float).reshape(-1))
    )


def tcp_tracking(solver, target_position=None) -> dict:
    """Where the end effector ACTUALLY is, against where it was sent.

    The arm's drives do not compensate gravity, so a commanded TCP pose is not
    necessarily the pose in force -- with a 0.85 kg payload that difference is the
    honest answer to "did the arm get there".  Reported per leg rather than assumed.
    """
    p, _ = solver.compute_end_effector_pose()
    p = np.asarray(p, float)
    out = {"tcp_position_m": [round(float(v), 6) for v in p]}
    if target_position is not None:
        out["tcp_error_m"] = round(
            float(np.linalg.norm(p - np.asarray(target_position, float))), 6
        )
    return out


class FrankaGripper:
    """Finger-joint commands in units of CLEAR OPENING, using a measured map."""

    def __init__(
        self, robot, calibration: dict, drive_stiffness: float, max_force_n: float
    ):
        self.robot = robot
        self.cal = calibration
        self._idx = [list(robot.dof_names).index(j) for j in FINGER_JOINTS]
        self._slope = float(calibration["clear_opening_m_per_joint_m"]) or 2.0
        self._offset = float(calibration["clear_opening_offset_m"])
        self._lo, self._hi = [float(v) for v in calibration["joint_limit_m"]]
        self.stiffness = float(drive_stiffness)
        self.max_force_n = float(max_force_n)
        self.command_m = None

    def joint_for_opening(self, clear_m: float) -> float:
        j = (float(clear_m) - self._offset) / self._slope
        return float(np.clip(j, self._lo, self._hi))

    def clear_opening_now(self) -> float:
        j = np.asarray(self.robot.get_joint_positions(), float).reshape(-1)[
            self._idx[0]
        ]
        return float(self._slope * j + self._offset)

    def command_vector(self, clear_m: float) -> np.ndarray:
        """Full-DOF command with the fingers at `clear_m`.  Does NOT record the
        command -- `command_vector` is also used to build the "hold still" vector
        carried through the arm's motion legs, and treating that as the applied
        finger command made the force report read the wrong number."""
        q = np.asarray(self.robot.get_joint_positions(), float).reshape(-1).copy()
        j = self.joint_for_opening(clear_m)
        for k in self._idx:
            q[k] = j
        return q

    def hold(self, robot, clear_m: float) -> float:
        """Command the fingers to `clear_m` and RECORD it, so the force report is
        reading the command that is actually in force."""
        apply_command(robot, self.command_vector(clear_m))
        self.command_m = float(clear_m)
        return self.command_m

    def force_report(self) -> dict:
        """The normal force each pad is delivering, from the drive model
        `F = min(k * |commanded - achieved|, max_force)`.

        Reported, never assumed: a force-capped gripper delivers what its cap allows
        whatever was asked.  This is the number that decides whether the grip holds,
        and it is the one a lowered safety factor CANNOT raise.
        """
        if self.command_m is None:
            return {"command_m": None}
        got = self.clear_opening_now()
        err = abs(self.command_m - got)
        k = self.stiffness
        return {
            "command_m": round(self.command_m, 6),
            "achieved_m": round(got, 6),
            "deflection_m": round(err, 6),
            "stiffness_n_per_m": k,
            "max_force_n": self.max_force_n,
            "force_uncapped_n": round(err * k, 3),
            "force_n": round(min(err * k, self.max_force_n), 3),
            "saturated": bool(err * k > self.max_force_n),
        }

    def tcp_for_grasp(
        self, point_world, rot_wxyz, tip_clearance_m: float = 0.002
    ) -> dict:
        """TCP position that puts the OPENING around `point_world`.

        An IK target is a TCP target, but a grasp is about where the FINGERS close,
        and the two are not the same point (see `opening_centre_tcp_m`).  This maps
        one to the other with the measured offset, then lifts the TCP if the finger
        tips would otherwise reach the tabletop.

        Sign care, which cost a run: the TCP's +Z is the approach axis, so with the
        hand pointing down a TCP-frame height `zeta` lands at world `z = tcp_z -
        zeta`.  The measured pad span is [-0.0415, +0.0123], i.e. the pad ROOT is
        41.5 mm behind the TCP and the TIPS overhang it by 12.3 mm.  Aiming the TCP
        at the grip height put the pads 30 mm above the part: the fingers closed
        through empty air, hit their commanded opening EXACTLY, and reported zero
        deflection -- a gripper that looks like it is working and holds nothing.
        """
        from scipy.spatial.transform import Rotation as Rot

        q = np.asarray(rot_wxyz, float).reshape(4)
        r = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        centre = np.asarray(self.cal["opening_centre_tcp_m"], float)
        tcp = np.asarray(point_world, float) - r @ centre
        lo, hi = (float(v) for v in self.cal["pad_tcp_z_range_m"])
        tip_world_z = float((tcp + r @ np.array([0.0, 0.0, hi]))[2])
        lifted = 0.0
        if tip_world_z < tip_clearance_m:
            lifted = tip_clearance_m - tip_world_z
            tcp = tcp + np.array([0.0, 0.0, lifted])
        return {
            "tcp_m": [round(float(v), 6) for v in tcp],
            "opening_centre_m": [round(float(v), 6) for v in (tcp + r @ centre)],
            "tip_world_z_m": round(tip_world_z + lifted, 6),
            "lifted_for_tip_clearance_m": round(float(lifted), 6),
            "requested_point_m": [round(float(v), 6) for v in point_world],
        }

    def tip_clearance_z(self, grip_z: float) -> float:
        """TCP height that puts the PADS across the grip height.

        Sign care, which cost a run: the TCP's **+Z is the approach axis**, so with
        the hand pointing down a TCP-frame height `zeta` lands at world `z = tcp_z -
        zeta`.  The measured pad span in the TCP frame is [-0.0415, +0.0123] -- the
        pad ROOT is 41.5 mm behind the TCP and the finger TIPS overhang it by
        12.3 mm -- so in the world the pads occupy

            [tcp_z - 0.0123,  tcp_z + 0.0415]

        and their centre sits at `tcp_z + 0.0146`.  The first version put the TCP at
        `grip_z + 0.0146`, which is 30 mm too high: the fingers then closed through
        empty air, reached their commanded opening EXACTLY, and reported zero
        deflection -- a gripper that looks like it is working and is holding nothing.

        The second term keeps the finger tips off the tabletop: they hang 12.3 mm
        below the TCP.
        """
        lo, hi = (float(v) for v in self.cal["pad_tcp_z_range_m"])
        centre_off = -0.5 * (lo + hi)  # +0.0146 m for this gripper
        return max(float(grip_z) - centre_off, hi + 0.002)


# ── self-test ────────────────────────────────────────────────────────────────
def _selftest() -> None:
    """Boot a throwaway scene, load the arm, and print the measurements.

    This is the check the plan promised: load -> 9 DOF -> calibrate -> Lula/USD
    cross-check.  It touches no asset and writes no report.
    """
    R.require_isaac()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    try:
        import omni.usd
        from isaacsim.core.api import World

        thr = R.load_thresholds()
        dt = float(thr["common"]["physics_dt"])
        world = World(physics_dt=dt, rendering_dt=dt)
        world.reset()
        stage = omni.usd.get_context().get_stage()
        base = [0.44, 0.0, 0.0]
        loaded = load_franka(
            world,
            stage,
            base,
            180.0,
            mu=float(thr["common"]["mu_rig"]),
            max_force_n=GRASP_FORCE_N,
            stiffness=float(thr["grasp"]["jaw_stiffness"]),
            damping=float(thr["grasp"]["jaw_damping"]),
        )
        world.reset()
        robot = loaded["robot"]
        R.log("robot: " + json.dumps(dof_report(robot), ensure_ascii=False))
        R.log("drives read back: " + json.dumps(read_franka_drives(stage)))
        solver, lula = make_lula_solver(robot)
        R.log("base pose set: " + json.dumps(set_base_pose(solver, robot)))
        cal = calibrate_gripper(stage, world, robot, solver)
        R.log("calibration: " + json.dumps(cal, ensure_ascii=False))
        g = FrankaGripper(
            robot, cal, float(thr["grasp"]["jaw_stiffness"]), GRASP_FORCE_N
        )
        R.log(
            f"opening 0.070 m -> joint {g.joint_for_opening(0.070):.5f} m; "
            f"tip_clearance_z(0.018) = {g.tip_clearance_z(0.018):.5f}"
        )
        q = tcp_orientation_wxyz(0.0)
        R.log(f"tcp_orientation_wxyz(0 deg) = {np.round(q, 5).tolist()}")
        info = solve_ik(solver, lula, [0.44 - 0.44, 0.0, 0.06], q)
        R.log("IK to the world origin: " + json.dumps(info, ensure_ascii=False))
    except BaseException:
        R.dump_traceback("/tmp", "robot_selftest")
        raise
    finally:
        app.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
