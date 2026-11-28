# SPDX-License-Identifier: Apache-2.0
"""Rig implementation for RFW-W."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ISAAC_PYTHON = os.environ.get("ISAAC_PYTHON", sys.executable)
EXPORT_ROOT = os.environ.get("PP_EXPORT_ROOT", "")


# ── asset staging ────────────────────────────────────────────────────────────
def asset_key_parts(key: str) -> tuple[str, str]:
    """`0_2_0002` -> ("0", "2").  Domain and class come straight from the key."""
    d, c, _ = key.split("_")
    return d, c


def stage_asset(key: str, work_dir: str, version: str = "v1.3") -> str:
    """Read a configured release leaf and copy into a fresh task directory."""
    from pathlib import Path
    from rfw_w.assets.catalog.release import resolve

    if not EXPORT_ROOT:
        raise ValueError("PP_EXPORT_ROOT must point at a configured release root")
    src = resolve(Path(EXPORT_ROOT), key)
    dst = Path(work_dir) / key
    if dst.exists():
        raise FileExistsError("task staging output must be new")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return str(dst)


def build_usd(key: str, asset_dir: str) -> str:
    """Build `<asset_dir>/<key>.usd` from determined asset physics and collision geometry.

    Run as a SUBPROCESS, not imported: `IsaacUsdBuilder`
    boots its own SimulationApp in `__init__`, and a task script has already booted
    one by this point.  Two SimulationApps in one process is not supported.
    Returns the usd path; raises with the converter's own message on failure.
    """
    usd = os.path.join(asset_dir, f"{key}.usd")
    if os.path.isfile(usd):
        return usd
    cmd = [
        ISAAC_PYTHON,
        "-m",
        "rfw_w.assets.simulation.exporters.isaac_usd",
        "--asset-dir",
        asset_dir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.isfile(usd):
        raise RuntimeError(
            f"build_isaac_usd failed for {key} (exit {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return usd


def read_sidecar(asset_dir: str) -> dict:
    """The export's own physics sidecar -- the numbers the rig and report cite."""
    with open(os.path.join(asset_dir, "part_physics.json")) as fh:
        return json.load(fh)


def load_asset(stage, key: str, usd_path: str, path: str = "/World/asset"):
    """Reference the built `<key>.usd` into the scene.

    The USD's default prim (`obj_<key>`) already carries RigidBodyAPI + MassAPI and
    the per-hull convex colliders (authored by the USD exporter), so the
    reference lands a complete dynamic body at `path` with no further authoring.
    Returns the prim at `path`, which is the prim a RigidPrim view should track.

    Also remembers which FILE this prim came from, which is what `authored_bbox`
    needs -- see there for why the live stage cannot answer that question.
    """
    from pxr import UsdGeom

    p = UsdGeom.Xform.Define(stage, path).GetPrim()
    p.GetReferences().AddReference(usd_path)
    _ASSET_USD[str(p.GetPath())] = usd_path
    _AUTHORED_BBOX.pop(str(p.GetPath()), None)
    return p


def calm_asset_material(prim, emissive_intensity: float = 0.0) -> dict:
    """Zero the asset's `emissive_intensity`, which is otherwise 10000 and renders
    the whole asset BLACK.

    This is THE fix for "the asset renders black in Isaac".  The converted material
    is an `OmniPBR_Opacity` MDL material and is entirely valid -- UVs are present
    (`primvars:st`), `inputs:diffuse_texture` resolves to `./textures/material_0.png`
    (the converter copies it there), the binding is correct and the lighting is
    fine.  The one bad value is:

        inputs:emissive_intensity = 10000.0

    isolated by a three-way comparison on the claw hammer:

        as-is (texture + emissive 10000)       -> object mean RGB 43.7  (black)
        texture unplugged, colour constant red -> (199.8, 39.0, 39.0)   (bright red)
        texture back, emissive_intensity = 0   -> bright

    Rows 1 and 3 differ ONLY in `emissive_intensity`, so that is the cause: with the
    texture connected the huge emissive term overflows the shading result to NaN and
    RTX draws NaN as black.  Note `enable_emission` is already False and does not
    prevent it.  Unplugging the texture also "fixes" it, which is why the bug looked
    like a texture or UV problem and why chasing the shader graph got nowhere.

    DISPLAY ONLY: this writes the visual shader, never the colliders, mass or
    inertia, so 需求 §8's same-collision/mass/inertia requirement is untouched.

    Side note for anyone re-diagnosing this: `UsdShade.Shader.GetShaderId()` returns
    None for this material and that is NORMAL -- MDL materials carry
    `info:mdl:sourceAsset` and have no `info:id`.  It is not evidence of breakage.
    """
    from pxr import Usd, UsdShade

    patched = []
    for p in Usd.PrimRange(prim):
        if not p.IsA(UsdShade.Shader):
            continue
        sh = UsdShade.Shader(p)
        inp = sh.GetInput("emissive_intensity")
        if inp:
            old = inp.Get()
            inp.Set(float(emissive_intensity))
            patched.append(
                {
                    "prim": str(p.GetPath()),
                    "emissive_intensity": [old, float(emissive_intensity)],
                }
            )
    return {
        "n_shaders_patched": len(patched),
        "emissive_intensity": emissive_intensity,
        "detail": patched,
        "display_only": "visual shader only; colliders/mass/inertia untouched",
    }


_ASSET_USD: dict[str, str] = {}  # /World/asset -> the <key>.usd it references
_AUTHORED_BBOX: dict[str, tuple] = {}  # /World/asset -> the authored box, computed once


def authored_bbox(prim) -> tuple[np.ndarray, np.ndarray]:
    """The asset's AABB in its AUTHORED frame, read from the USD FILE.

    **Read from the file on purpose.**  The obvious implementation -- a
    `UsdGeom.BBoxCache` on the live stage -- is wrong the moment the body moves.
    Measured on the claw hammer, on this box:

        after load              world bound  lo=[-0.0738,-0.175,-0.0147] (authored)
        after place_upright     world bound  lo=[-0.0738,-0.175,-0.0147] (authored)
        after a 2.5 s settle    world bound  lo=[-0.2074, 0.1170,-0.0067] (POSED)

    Isaac writes the rigid-body pose back into the prim's `xformOp:translate` /
    `xformOp:orient` / `xformOp:scale`, and the live-stage bbox follows it.  So
    `authored_bbox` silently started returning a *posed* box, which `_rotated_aabb`
    then rotated AGAIN.

    Two real defects came out of that, both invisible until the numbers were checked:
      * `place_upright` mis-posed every trial after the first, because it computes
        `pos` from the box it reads.  The delivered drop "initial" frame showed a
        hammer lying flat instead of upright 0.15 m above the table, and every sweep
        row was measured against a wrong release pose.
      * `_world_bbox_of` double-rotated, so grasp's `grip -> farthest corner`
        distance came back as 0.5034 m for an asset whose bbox diagonal is 0.386 m.

    Reopening the file is cheap (once per asset, then cached) and cannot drift: the
    file's bound is the geometry as authored, no matter what the physics has done
    since.  Falls back to the live-stage bound only for a prim this module did not
    load itself.
    """
    key = str(prim.GetPath())
    hit = _AUTHORED_BBOX.get(key)
    if hit is not None:
        return np.array(hit[0], float), np.array(hit[1], float)
    usd = _ASSET_USD.get(key)
    if not usd:
        log(
            f"! authored_bbox({key}): prim was not loaded by rig.load_asset, so the "
            f"authored file is unknown -- falling back to the LIVE stage bound, which "
            f"is only correct before the body has moved"
        )
        lo, hi = bbox_of(prim)
    else:
        from pxr import Usd, UsdGeom

        st = Usd.Stage.Open(usd)
        root = st.GetDefaultPrim()
        rng = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeWorldBound(root)
            .ComputeAlignedRange()
        )
        if rng.IsEmpty():
            raise RuntimeError(f"{usd}: default prim has no default-purpose geometry")
        lo = np.asarray(rng.GetMin(), float)
        hi = np.asarray(rng.GetMax(), float)
    _AUTHORED_BBOX[key] = (lo.tolist(), hi.tolist())
    return lo, hi


def bbox_of(prim) -> tuple[np.ndarray, np.ndarray]:
    """World-space AABB of a prim's *default* purpose geometry.

    The colliders are authored with `purpose=guide`, so this measures the VISUAL
    mesh -- which is what the figure shows and what "lowest point" should mean to a
    reader looking at the image.  (The physics answer comes from the colliders and
    can differ by a few mm; the report says which one each number came from.)
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        raise RuntimeError(f"no default-purpose geometry under {prim.GetPath()}")
    return (np.asarray(rng.GetMin(), float), np.asarray(rng.GetMax(), float))


def part_boxes(asset_dir: str, key: str) -> dict:
    """Per-part AABB in the AUTHORED frame, straight from the merged collision OBJ.

    The merged file carries one `o .part_<pid>_collision.obj` group per part, so the
    grouping is read rather than guessed -- matching per-hull files back to parts
    needs a nearest-centroid heuristic, which this avoids.

    Written with a hand-rolled OBJ reader because Isaac's python has no trimesh
    (measured; see the module docstring).  Returns {pid: {"lo", "hi", "extents",
    "centre"}} in the export's own frame -- apply `authoring_up_fix` plus the body
    pose to get world coordinates.
    """
    path = os.path.join(asset_dir, "mesh", f"{key}_collision.obj")
    if not os.path.isfile(path):
        return {}
    verts, groups, cur = [], [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("o "):
                groups.append([])
                cur = len(groups) - 1
            elif line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f ") and cur is not None:
                for t in line.split()[1:]:
                    groups[cur].append(int(t.split("/")[0]) - 1)
    if not groups or not verts:
        return {}
    V = np.asarray(verts, float)
    out = {}
    for pid, idx in enumerate(groups):
        if not idx:
            continue
        P = V[np.unique(np.asarray(idx, dtype=np.int64))]
        lo, hi = P.min(axis=0), P.max(axis=0)
        # Points are kept (subsampled) so a caller can measure the part's TRUE
        # cross-section.  The AABB is not enough: a handle lying diagonally has an
        # AABB ~2x its real diameter, and a jaw sized from the AABB closes on air.
        if len(P) > 3000:
            P = P[:: max(1, len(P) // 3000)]
        out[pid] = {
            "lo": lo,
            "hi": hi,
            "extents": hi - lo,
            "centre": (lo + hi) * 0.5,
            "points": P,
        }
    return out


def grasp_target(asset_dir: str, key: str) -> dict:
    """Which part should the gripper close on -- decided by the EXPORT, not by us.

    The physics sidecar carries a per-part `priority_rank` and a
    `Grasped_description` written by the annotation stage.  For the claw hammer that
    is unambiguous: rank 1 is the rubber handle, described as "a robot gripper would
    grasp the handle", while the head, shaft and ferrule each carry an explicit
    statement that they are NOT grasp targets.

    Using it means the task's choice of grasp point is reproducible and citable,
    instead of being a height fraction this code invented.  Falls back to the
    lowest-ranked part, and reports which it used.
    """
    with open(os.path.join(asset_dir, "part_physics.json")) as fh:
        phys = json.load(fh)
    parts = phys.get("parts") or {}
    if not parts:
        raise RuntimeError(f"{key}: part_physics.json has no `parts` block")
    boxes = part_boxes(asset_dir, key)

    def rank(pid_s):
        r = parts[pid_s].get("priority_rank")
        return (0, int(r)) if isinstance(r, (int, float)) else (1, 0)

    ranked = sorted(parts, key=rank)
    pid = int(ranked[0])
    box = boxes.get(pid)
    if box is None:
        raise RuntimeError(f"{key}: no collision geometry for part {pid}")
    return {
        "part_id": pid,
        "priority_rank": parts[ranked[0]].get("priority_rank"),
        "material": parts[ranked[0]].get("material"),
        "grasped_description": parts[ranked[0]].get("Grasped_description"),
        "basic_description": parts[ranked[0]].get("Basic_description"),
        "box_authoring_frame": {
            "lo": [round(float(v), 6) for v in box["lo"]],
            "hi": [round(float(v), 6) for v in box["hi"]],
            "extents": [round(float(v), 6) for v in box["extents"]],
            "centre": [round(float(v), 6) for v in box["centre"]],
        },
        "rejected_parts": [
            {
                "part_id": int(p),
                "priority_rank": parts[p].get("priority_rank"),
                "grasped_description": parts[p].get("Grasped_description"),
            }
            for p in ranked[1:]
        ],
        "provenance": "part_physics.json: lowest `priority_rank`; the Grasped_description "
        "of the rejected parts states why they are not targets",
    }


def settle(
    world,
    view,
    dt: float,
    seconds: float = 3.0,
    speed_tol: float = 0.01,
    quiet_s: float = 0.25,
):
    """Step until the asset is genuinely at rest; return its (position, wxyz).

    A FIXED settle is not enough.  Placed upright, a claw hammer topples -- with a
    flat 0.5 s settle it was still rolling when the task began, which is why the push
    task reported "contact" 0.002 s in: it was measuring the topple, not the push.
    Steps without rendering (this can be 3 s of sim and no frames are wanted).
    """
    n = int(round(seconds / dt))
    need = max(1, int(round(quiet_s / dt)))
    quiet = 0
    for _ in range(n):
        world.step(render=False)
        v = float(np.linalg.norm(np.asarray(view.get_linear_velocities()[0], float)))
        quiet = quiet + 1 if v < speed_tol else 0
        if quiet >= need:
            break
    p, q = view.get_world_poses()
    return np.asarray(p[0], float), np.asarray(q[0], float)


def grip_frame(points_world, close_axis_hint: str = "Y"):
    """Work out HOW to close a parallel jaw on a part from the part's own points.

    Returns (yaw_deg, grip_width_m, long_axis).  The yaw is what makes the jaw's
    built-in closing axis ("Y" in the rig's frame) come out PERPENDICULAR to the
    part's long axis, and the width is the part's extent along that closing
    direction -- its true cross-section.

    Measuring from the AABB instead is what made the first grasp attempt close on
    empty table: a claw hammer lies diagonally, so its handle's AABB came out
    0.079 m wide against a real diameter of 0.037 m, and the jaw opened wider than
    the thing it was supposed to grip.
    """
    P = np.asarray(points_world, float)
    if len(P) < 4:
        return 0.0, 0.0, np.array([1.0, 0.0, 0.0])
    c = P.mean(axis=0)
    Q = P - c
    # principal axis of the part == its long axis (a handle is a long thin blob)
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    u = vt[0]
    uh = np.array([u[0], u[1], 0.0])
    n = np.linalg.norm(uh)
    if n < 1e-6:  # part is vertical: any horizontal axis works
        uh = np.array([1.0, 0.0, 0.0])
    else:
        uh = uh / n
    # yaw so that the rig's +Y (after the yaw) is perpendicular to uh
    yaw = float(np.degrees(np.arctan2(uh[1], uh[0])))
    close_dir = np.array([-np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0.0])
    width = float(np.ptp(Q @ close_dir))
    return yaw, width, u


def grip_pose(
    part_pts_world,
    all_pts_world,
    com_world,
    pad_half_len: float = 0.015,
    clearance: float = 0.02,
    n_candidates: int = 61,
):
    """WHERE on the grasp part to close the jaws, and how wide to open them.

    Returns `(grip_point_world, yaw_deg, width_m, long_axis, com_distance_m, t_ax,
    diagnostics)`.

    **Grip at the part's axial centroid.**  The obvious refinement -- slide the grip
    toward the asset's centre of mass, so the jaws have a shorter lever to resist --
    was implemented and then MEASURED OUT.  On the claw hammer it moves the grip to
    the handle's upper end, which is where the handle TAPERS: the local width drops
    from 0.0373 m to 0.0340 m, the pads close on a cone, and the hammer wedges
    straight out of the jaws.  With the over-close that works at the centroid
    (10.1 mm) the CoM-side grip still lifted nothing (rise 0.0000 m), while the
    centroid grip lifted 0.0975 m.  A parallel jaw wants a PARALLEL section.

    The width is the part's TRUE cross-section inside the pad window -- not the whole
    part's spread, which over-states the width where a part tapers.

    The grip is also walked away from anything wider than the part inside the pad
    window (over ALL parts' points): a jaw is a fixed opening, and on a hammer the
    head next door is wide enough that pads landing there would close on the head
    instead of the handle.
    """
    P = np.asarray(part_pts_world, float)
    A = np.asarray(all_pts_world, float)
    com = np.asarray(com_world, float)
    yaw, _w0, u = grip_frame(P)
    close_dir = np.array([-np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0.0])
    c = P.mean(axis=0)
    t_part = (P - c) @ u
    t_all = (A - c) @ u
    t_lo, t_hi = float(t_part.min()), float(t_part.max())
    # keep the pads inside the part along its axis
    a, b = t_lo + pad_half_len, t_hi - pad_half_len
    if b <= a:
        a = b = 0.5 * (t_lo + t_hi)
    ts = np.linspace(a, b, int(n_candidates))

    def evaluate(t):
        win_p = np.abs(t_part - t) <= pad_half_len
        if int(win_p.sum()) < 8:
            return None
        grip = P[win_p].mean(axis=0)
        w_part = float(np.ptp((P[win_p] - grip) @ close_dir))
        win_a = np.abs(t_all - t) <= pad_half_len
        w_all = (
            float(np.ptp((A[win_a] - grip) @ close_dir))
            if int(win_a.sum()) >= 8
            else w_part
        )
        return grip, w_part, w_all

    # start at the part's axial centroid, then step outward if the window is blocked
    ts = ts[np.argsort(np.abs(ts - 0.5 * (t_lo + t_hi)))]
    chosen, blocked = None, None
    for t in ts:
        r = evaluate(float(t))
        if r is None:
            continue
        if blocked is None:
            blocked = (float(t), r[2], r[1])
        if r[2] <= r[1] + 2.0 * clearance:
            chosen = (float(t), r)
            break
    if chosen is None:  # nothing clears: part's middle
        t = 0.5 * (t_lo + t_hi)
        r = evaluate(t) or evaluate(0.5 * (a + b))
        chosen = (float(t), r if r else (c, 0.0, 0.0))
    t_ax, (grip, w_part, w_all) = chosen
    return (
        grip,
        yaw,
        w_part,
        u,
        float(np.linalg.norm(com - grip)),
        t_ax,
        {
            "grip_axial_t": round(t_ax, 6),
            "part_axial_range": [round(t_lo, 6), round(t_hi, 6)],
            "centre_of_mass_axial_t": round(float((com - c) @ u), 6),
            "width_at_grip_m": round(w_part, 6),
            "width_all_parts_in_window_m": round(w_all, 6),
            "window_blocked_at_centroid": (
                None
                if blocked is None
                else {
                    "t": round(blocked[0], 6),
                    "width_all_m": round(blocked[1], 6),
                    "width_part_m": round(blocked[2], 6),
                }
            ),
            "rule": "axial centroid of the exported grasp part; the width is the "
            "part's cross-section inside one pad length of it",
            "com_rule_rejected": "sliding the grip toward the centre of mass was "
            "measured and rejected: it lands on the tapering "
            "end of the handle and the jaws wedge out",
        },
    )


def settle_centred(world, view, prim, dt: float, seconds: float = 6.0):
    """Settle the asset, then PARK ITS WORLD AABB CENTRE ON THE ORIGIN.

    Returns (pos, wxyz, lo, hi) with the bbox centred in x/y and its bottom on z=0.

    Necessary because an asset that topples also TRAVELS.  The claw hammer placed
    upright ends up with a 0.31 x 0.42 m footprint centred 0.25 m off to the side, so
    a pusher that travels along X through the origin simply missed it -- the sweep
    read "no movement" while the identical figure trial, run before the drift, moved
    0.14 m.  Re-centring after the settle makes the scene predictable for every tool,
    which is worth more than modelling the drift.
    """
    pos, quat = settle(world, view, dt, seconds=seconds)
    lo, hi = _world_bbox_of(prim, pos, quat)
    mid = (lo + hi) * 0.5
    pos = pos - np.array([mid[0], mid[1], 0.0])
    view.set_world_poses(positions=pos.reshape(1, 3), orientations=quat.reshape(1, 4))
    view.set_linear_velocities(np.zeros((1, 3)))
    view.set_angular_velocities(np.zeros((1, 3)))
    pos, quat = settle(world, view, dt, seconds=seconds)
    lo, hi = _world_bbox_of(prim, pos, quat)
    return pos, quat, lo, hi


def _world_bbox_of(prim, pos, quat_wxyz):
    """World AABB of the asset, from its authored box and a measured pose."""
    xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], float)
    lo, hi = _rotated_aabb(*authored_bbox(prim), xyzw)
    pos = np.asarray(pos, float)
    return lo + pos, hi + pos


def mesh_bbox_of(points_auth, pos, quat_wxyz):
    """Exact world AABB of the asset's own mesh points at a measured pose.

    Preferred over `_world_bbox_of` wherever a *height* is judged: that one rotates
    the AUTHORED BOUNDING BOX, whose corners are up to a few cm outside the mesh, so
    for a hammer lying diagonally it reports a bottom ~0.16 m below the table.  The
    bounding-box version is fine for "did the pose change" but not for "did the
    asset leave the surface".
    """
    from scipy.spatial.transform import Rotation as R

    xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], float)
    P = R.from_quat(xyzw).apply(np.asarray(points_auth, float))
    P = P + np.asarray(pos, float)
    return P.min(axis=0), P.max(axis=0)


def authoring_up_fix():
    """Quaternion that takes the EXPORT's authoring frame (Y-up) to world Z-up.

    The exported OBJs are authored Y-up -- the same convention the figure pipeline
    compensates for with `collect_figure_assets._z_up` -- while an Isaac stage is
    Z-up.  Nothing in `isaac_usd` rotates the geometry, and it must not:
    rotating the points would desynchronise the colliders from the visual.  The
    correction therefore belongs on the BODY POSE, as one rigid rotation applied to
    visual and colliders together.

    Measured, not assumed: loaded as-is, the frying pan's world AABB is
    `0.45 x 0.0718 x 0.2735` -- thin along Y, i.e. standing on its edge.  After this
    rotation it is `0.45 x 0.2735 x 0.0718`, lying flat.
    """
    from scipy.spatial.transform import Rotation as R

    return R.from_euler("x", 90.0, degrees=True)


def _rotated_aabb(lo, hi, quat_xyzw):
    """AABB of an axis-aligned box after rotating it -- 8 corners, min/max.

    Used instead of re-reading the prim's world bound, because
    `RigidPrim.set_world_poses` writes the PHYSICS view, not USD: a USD bbox query
    straight afterwards still sees the authored, unrotated transform (this silently
    produced an unrotated frame on the first attempt).
    """
    from scipy.spatial.transform import Rotation as R

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )
    w = R.from_quat(np.asarray(quat_xyzw, float)).apply(corners)
    return w.min(axis=0), w.max(axis=0)


def place_upright(
    view, prim, tilt_xyzw=None, height: float = 0.0, centre_xy: bool = True
):
    """Pose the asset so it rests UPRIGHT with its bbox bottom `height` above z=0.

    The authored bbox (`bbox_of`) is in the export's Y-up frame; the desired world
    AABB is that box rotated by the up-axis correction plus the trial's tilt, which
    `_rotated_aabb` computes directly.  Returns
    (position, wxyz, world bbox lo, world bbox hi) with the bbox in the WORLD frame.
    """
    from scipy.spatial.transform import Rotation as R

    q = authoring_up_fix()
    if tilt_xyzw is not None:
        q = R.from_quat(np.asarray(tilt_xyzw, float)) * q
    xyzw = q.as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    lo0, hi0 = authored_bbox(prim)
    lo, hi = _rotated_aabb(lo0, hi0, xyzw)
    mid = (lo + hi) * 0.5
    pos = np.array(
        [
            -mid[0] if centre_xy else 0.0,
            -mid[1] if centre_xy else 0.0,
            float(height) - lo[2],
        ]
    )
    view.set_world_poses(positions=pos.reshape(1, 3), orientations=wxyz.reshape(1, 4))
    return pos, wxyz, lo + pos, hi + pos


# ── small numeric helpers (reimplemented from the team's earlier drop protocol) ──
def orientations(n: int = 5, seed: int = 42) -> list[list[float]]:
    """Identity plus n-1 small random tilts (5-20 deg), as xyzw quaternions.

    Small tilts on purpose: they perturb which face lands first without turning
    the task into "throw the object".  Same construction as the drop-settle
    protocol the team already ran, so the two result sets stay comparable.
    """
    from scipy.spatial.transform import Rotation as R

    rng = np.random.RandomState(seed)
    quats = [[0.0, 0.0, 0.0, 1.0]]
    while len(quats) < n:
        axis = rng.randn(3)
        axis = axis / np.linalg.norm(axis)
        ang = np.radians(rng.uniform(5.0, 20.0))
        quats.append(R.from_rotvec(ang * axis).as_quat().tolist())
    return quats[:n]


def quat_angle_deg(qa, qb) -> float:
    """Angle between two orientations, in degrees, sign- and order-agnostic."""
    d = abs(float(np.dot(np.asarray(qa, float), np.asarray(qb, float))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Look-at -> wxyz quaternion for an Isaac `Camera`.

    **Isaac's camera does NOT use the OpenGL/USD-camera convention.** It follows the
    robotics one: local **+X is forward**, +Z up, and image-right is -Y.  Measured,
    because assuming otherwise cost a long detour: with the camera at the origin and
    an IDENTITY orientation, a cube at world +X lands dead centre of the frame while
    one at -Z is nowhere in shot.  An earlier version of this function built
    `-Z forward` and so aimed every camera 90 deg away from its subject.

    Columns are (forward, y, up) with `y = up x forward`, which is right-handed and
    gives the standard upright, non-mirrored image.
    """
    from scipy.spatial.transform import Rotation as R

    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    n = np.linalg.norm(fwd)
    if n < 1e-12:
        raise ValueError("look_at_quat: eye and target coincide")
    fwd = fwd / n
    u = np.asarray(up, float)
    u = u - np.dot(u, fwd) * fwd
    if np.linalg.norm(u) < 1e-9:  # looking straight down the up axis
        u = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    u = u / np.linalg.norm(u)
    y = np.cross(u, fwd)
    q = R.from_matrix(np.column_stack([fwd, y, u])).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz, for Isaac


# ── the rig ──────────────────────────────────────────────────────────────────
class Axis:
    """A driven prismatic joint, in world-metres relative to the joint-zero pose.

    `set(v)` commands the child body `v` metres along the axis from `Rig.origin`.
    That 1:1 reading only holds because `Rig` co-locates the body origins -- see
    the module docstring, point 1.
    """

    def __init__(self, drive, name):
        self._drive = drive
        self.name = name

    def set(self, v: float) -> None:
        self._drive.GetTargetPositionAttr().Set(float(v))

    def target(self) -> float:
        return float(self._drive.GetTargetPositionAttr().Get())


class Rig:
    """Kinematic anchor + driven prismatic axes, all sharing one origin.

    Every body added here is authored AT `origin`; geometry is placed with a local
    offset.  Do not "helpfully" give a body a different world position -- that is
    the one thing that breaks the joint semantics (module docstring, point 1).
    """

    def __init__(
        self, stage, origin, name: str = "rig", mu: float = 1.0, yaw_deg: float = 0.0
    ):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        self._Gf, self._UsdGeom = Gf, UsdGeom
        self._UsdPhysics, self._UsdShade = UsdPhysics, UsdShade
        self.stage = stage
        self.origin = tuple(float(x) for x in origin)
        self.name = name
        # Every body is authored at the rig origin, so a yaw here rotates the whole
        # fixture -- and with it the joint axes, which are expressed in the body
        # frame.  That is how the jaw's closing axis gets aimed perpendicular to a
        # diagonally-lying handle without rotating the asset itself.
        self.yaw_deg = float(yaw_deg)

        # one PhysicsMaterial for the whole rig, so the finger / pusher surfaces
        # are the same tribology as the ground
        self.material = UsdShade.Material.Define(stage, f"/World/{name}_phymat")
        mp = UsdPhysics.MaterialAPI.Apply(self.material.GetPrim())
        mp.CreateStaticFrictionAttr(float(mu))
        mp.CreateDynamicFrictionAttr(float(mu))
        mp.CreateRestitutionAttr(0.0)

        # the anchor never moves and never collides: it exists only to give the
        # axes a body0
        self.anchor = self.add_body("anchor", mass=0.01, kinematic=True)

    def add_body(self, name: str, mass: float, kinematic: bool = False):
        p = self._UsdGeom.Xform.Define(
            self.stage, f"/World/{self.name}/{name}"
        ).GetPrim()
        x = self._UsdGeom.Xformable(p)
        x.AddTranslateOp().Set(self._Gf.Vec3d(*self.origin))
        if abs(self.yaw_deg) > 1e-9:
            # rotate about the rig origin, then translate: the two ops compose as
            # T * R, so the body still sits at `origin` but its local axes are yawed
            x.AddRotateZOp().Set(float(self.yaw_deg))
        self._UsdPhysics.RigidBodyAPI.Apply(p)
        if kinematic:
            self._UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr().Set(True)
        self._UsdPhysics.MassAPI.Apply(p).CreateMassAttr(float(mass))
        return p

    def add_box(self, body, name: str, size, offset, collides: bool = True):
        """A box collider inside `body`, positioned by LOCAL offset from `origin`.

        `offset` is relative to the rig origin, not to the body -- bodies all live
        at the origin, so the two are the same thing, and writing it this way keeps
        the geometry's world position readable at the call site.
        """
        p = self._UsdGeom.Cube.Define(self.stage, f"{body.GetPath()}/{name}").GetPrim()
        self._UsdGeom.Cube(p).CreateSizeAttr(1.0)
        x = self._UsdGeom.Xformable(p)
        x.AddTranslateOp().Set(self._Gf.Vec3d(*[float(v) for v in offset]))
        x.AddScaleOp().Set(self._Gf.Vec3f(*[float(v) for v in size]))
        if collides:
            self._UsdPhysics.CollisionAPI.Apply(p)
            self._UsdShade.MaterialBindingAPI.Apply(p).Bind(
                self.material, self._UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        return p

    def add_axis(
        self,
        name: str,
        child,
        axis: str,
        lo: float,
        hi: float,
        stiffness: float = 1e5,
        damping: float = 1e4,
        target: float = 0.0,
        parent=None,
    ) -> Axis:
        """Drive `child` along `axis` of the joint-zero frame, limits [lo, hi] m."""
        parent = parent if parent is not None else self.anchor
        j = self._UsdPhysics.PrismaticJoint.Define(
            self.stage, f"/World/{self.name}/j_{name}"
        )
        j.CreateBody0Rel().SetTargets([parent.GetPath()])
        j.CreateBody1Rel().SetTargets([child.GetPath()])
        j.CreateAxisAttr(axis)
        # both frames at the body origins -- the only thing PhysX honours
        j.CreateLocalPos0Attr(self._Gf.Vec3f(0, 0, 0))
        j.CreateLocalPos1Attr(self._Gf.Vec3f(0, 0, 0))
        j.CreateLowerLimitAttr(float(lo))
        j.CreateUpperLimitAttr(float(hi))
        d = self._UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateStiffnessAttr(float(stiffness))
        d.CreateDampingAttr(float(damping))
        d.CreateTargetPositionAttr(float(target))
        return Axis(d, name)


# `ParallelJaw` used to live here -- a two-finger jaw on a prismatic lift axis.  It
# was honest contact physics (real colliders, force-limited drives, no weld and no
# attachment), but it was not a manipulator: no joint chain, no reach limit, no IK,
# so a grasp result from it said nothing about a robot.  The grasp task now drives a
# real Franka Emika Panda -- see `robot.py`.  Its measured findings live on there:
# the pad height is the lever arm that resists the asset rotating out of the jaws,
# and a force-limited gripper delivers min(k * deflection, its force ceiling),
# so the ceiling is the binding number and not the safety factor.


class Pusher:
    """A driven box that shoves the object -- the push tool.

    Like `ParallelJaw`, the slab sits AT the rig origin (`offset_z` defaults to 0)
    so that the rig origin can be placed at the pusher's start position.  Put it
    behind the asset, not inside it: at `world.reset()` the pusher is at the rig
    origin, and spawning it overlapping the asset makes PhysX eject the asset and
    turns the whole run into an artefact.
    """

    def __init__(
        self,
        rig: Rig,
        size=(0.06, 0.06, 0.05),
        offset_z: float = 0.0,
        axis: str = "X",
        lo: float = -0.05,
        hi: float = 0.8,
        mass: float = 0.4,
        stiffness: float = 1e4,
        damping: float = 3e2,
    ):
        self.body = rig.add_body("pusher", mass=mass)
        self.slab = rig.add_box(self.body, "face", size, (0, 0, offset_z))
        self._offset_z = float(offset_z)
        self.axis = rig.add_axis(
            "push", self.body, axis, lo, hi, stiffness=stiffness, damping=damping
        )

    def set_lateral(self, y: float) -> None:
        """Slide the slab sideways within its body, to set the CONTACT OFFSET.

        The body itself cannot move laterally -- its only degree of freedom is the
        push axis -- so the off-centre case is realised by moving the collider inside
        the body.  Call before `world.reset()` so the physics re-reads the local pose;
        without this the "asymmetric contact" sweep trial is silently identical to
        the centred one.
        """
        from pxr import Gf, UsdGeom

        ops = UsdGeom.Xformable(self.slab).GetOrderedXformOps()
        t = next(
            (o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None
        )
        if t is None:
            UsdGeom.Xformable(self.slab).AddTranslateOp().Set(
                Gf.Vec3d(0.0, float(y), self._offset_z)
            )
        else:
            t.Set(Gf.Vec3d(0.0, float(y), self._offset_z))

    def lateral(self) -> float:
        """The slab's current local Y offset -- read back, so a sweep row can show
        the offset it actually ran with rather than the one it asked for."""
        from pxr import UsdGeom

        ops = UsdGeom.Xformable(self.slab).GetOrderedXformOps()
        t = next(
            (o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None
        )
        if t is None:
            return float("nan")
        return float(t.Get()[1])


def reload_physics() -> bool:
    """Make PhysX re-read the USD after a mid-session geometry edit.

    Needed because `Pusher.set_lateral` moves the slab's collider by editing its
    `xformOp:translate`, and a USD edit does NOT reach PhysX on its own -- not even
    through `world.reset()`.  Measured: with the edit un-forced, the off-centre push
    trial returned 0.0797 m of travel and 1.86 deg of rotation against the centred
    trial's 0.0796 m and 1.85 deg, i.e. the "asymmetric contact" case was silently
    the same experiment run twice.  A duplicate reported as a second trial is worse
    than a failure, because nothing in the numbers says it is a duplicate.

    Returns True if the interface was available.
    """
    try:
        from omni.physx import get_physx_interface

        get_physx_interface().force_load_physics_from_usd()
        return True
    except Exception as e:  # noqa: BLE001
        log(f"! reload_physics failed: {type(e).__name__}: {e}")
        return False


def make_ground(world, mu: float):
    """Static ground with the same friction as the rig."""
    return world.scene.add_default_ground_plane(
        static_friction=float(mu), dynamic_friction=float(mu), restitution=0.0
    )


def make_table(
    world,
    stage,
    mu: float,
    colour=(0.62, 0.63, 0.66),
    size=(3.0, 3.0, 0.04),
    path: str = "/World/table",
):
    """A light-grey static tabletop, top face at z = 0.

    Replaces `add_default_ground_plane` for the figure shots: that plane renders
    nearly black, and 需求 §3 asks for a pale neutral surface so the asset and the
    contact region read against it.  Physics is unaffected -- the table is a plain
    static collider with the same friction, not a rigid body, so it can never move.

    Returns the prim.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    p = UsdGeom.Cube.Define(stage, path).GetPrim()
    UsdGeom.Cube(p).CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(p)
    x.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -size[2] / 2.0))
    x.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    UsdPhysics.CollisionAPI.Apply(p)

    shader = UsdShade.Shader.Define(stage, Sdf.Path(path + "/shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(c) for c in colour])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat = UsdShade.Material.Define(stage, Sdf.Path(path + "/material"))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    phymat = UsdShade.Material.Define(stage, Sdf.Path(path + "/phymat"))
    pm = UsdPhysics.MaterialAPI.Apply(phymat.GetPrim())
    pm.CreateStaticFrictionAttr(float(mu))
    pm.CreateDynamicFrictionAttr(float(mu))
    pm.CreateRestitutionAttr(0.0)

    # two purposes, two materials: the surface shader for looks, the PhysicsMaterial
    # for friction. Binding the visual material to "physics" (an easy slip) leaves
    # the contact with the default friction instead of the value the sidecar states.
    api = UsdShade.MaterialBindingAPI.Apply(p)
    api.Bind(mat, UsdShade.Tokens.weakerThanDescendants)
    api.Bind(phymat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return p


# ── capture ──────────────────────────────────────────────────────────────────
def find_ffmpeg():
    """Locate an ffmpeg binary, or return None.

    Needed because OpenCV inside Isaac's python can encode exactly ONE video codec:
    `mp4v`, i.e. MPEG-4 Part 2 (verified -- `avc1`, `H264` and `h264` all fail to
    open the writer).  MPEG-4 Part 2 is NOT decodable by Chromium/Electron, so files
    written by the obvious `cv2.VideoWriter(..., "mp4v")` route show up as unplayable
    in the VSCode preview, QuickTime and most browsers even though the container is
    perfectly valid.  ffmpeg is used only to transcode the finished file to H.264.
    """
    env = os.environ.get("IMAGEIO_FFMPEG_EXE") or os.environ.get("FFMPEG_BINARY")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    return None


def to_h264(path: str, crf: int = 18) -> bool:
    """Transcode `path` in place to H.264/yuv420p so ordinary players can open it.

    Returns True if the file is H.264 on exit (either it already was, or the
    transcode succeeded).  Failure is non-fatal: the mp4v original is kept, and the
    report notes it, because an unplayable-but-valid video still carries the data.
    """
    ff = find_ffmpeg()
    if not ff:
        return False
    tmp = path + ".h264.mp4"
    cmd = [
        ff,
        "-y",
        "-loglevel",
        "error",
        "-i",
        path,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",  # without this some players refuse it
        "-movflags",
        "+faststart",  # moov up front, so previews start at once
        tmp,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, path)
    return True


def _to_uint8(a: np.ndarray) -> np.ndarray:
    """Camera output is not dtype-stable -- normalise uint8 / float 0..1 / 0..255."""
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    a = a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class Shooter:
    """A FIXED camera.  Both deliverable frames come through this one instance and
    one pose, which is what makes the initial/response pair readable as a pair
    (需求 §3: same projection, same intrinsics, same crop)."""

    def __init__(self, world, eye, target, res=(768, 1024), warmup=4):
        from isaacsim.sensors.camera import Camera

        self.res = (int(res[0]), int(res[1]))
        self.cam = Camera(prim_path="/World/cam", resolution=self.res)
        # Isaac's default near clip is 1.0 m, which silently culls anything closer
        # than a metre -- and the push/grasp cameras sit well inside that. Raise it
        # to something a close-up can still use.
        self.cam.set_clipping_range(0.05, 1000.0)
        self.set_pose(eye, target)
        self.cam.initialize()
        self._warmup = int(warmup)
        self._warmed = False

    def set_pose(self, eye, target) -> None:
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.cam.set_world_pose(
            position=self.eye, orientation=look_at_quat(self.eye, self.target)
        )
        self._warmed = False  # a re-posed camera needs its warm-up again

    def actual_pose(self):
        """What the SENSOR thinks its pose is, read back.

        Worth asking: the pose handed to `set_world_pose` is not necessarily the
        pose in force (the camera is an RTX sensor driven off the USD xform, and a
        re-pose after `initialize()` may not have been picked up).  Framing bugs are
        much easier to see as a position discrepancy than as a picture.
        """
        p, q = self.cam.get_world_pose()
        return np.asarray(p, float), np.asarray(q, float)

    def grab(self, world, force_render: bool = False) -> np.ndarray:
        """Latest rendered frame as uint8 RGB.

        `force_render` re-warms the renderer first.  WITHOUT it this is a pure read
        of the last rendered buffer -- which is what you want when capturing a frame
        at a known sim time, and is actively wrong when you have just changed the
        scene and expect the image to differ (fit_camera's hide/show diff learned
        this the hard way).

        The camera returns an EMPTY array until enough frames have gone through the
        renderer after a pose/scene change, so both paths retry rather than trusting
        a single step.

        Both warm-up paths call `world.render()`, NOT `world.step(render=True)`.
        That matters for an "initial" frame: `step` advances PhysX by a whole
        `physics_dt` per rendered frame, so warming a 4-frame buffer would drop a
        freely-falling asset by 8 ms of fall before the "release" frame was taken.
        `render()` pumps the renderer only (measured: 300 render frames left
        `world.current_time` untouched) -- see `render_warmup`.
        """
        if force_render or not self._warmed:
            for _ in range(self._warmup):
                world.render()
            self._warmed = True
        for _ in range(20):
            a = self.cam.get_rgba()
            if a is not None and np.asarray(a).size:
                return _to_uint8(a)[..., :3]
            world.render()
        raise RuntimeError(
            f"camera produced no frame after 20 renders (res={self.res}); "
            "the renderer may not be running"
        )


def _set_visible(prim, visible: bool) -> None:
    from pxr import UsdGeom

    img = UsdGeom.Imageable(prim)
    (img.MakeVisible if visible else img.MakeInvisible)()


def frame_stats(rgb) -> dict:
    """Brightness summary of a frame -- the evidence a capture was not a black one."""
    g = np.asarray(rgb, float)[..., :3].mean(axis=2)
    return {
        "mean": round(float(g.mean()), 3),
        "dark_px": int((g < 100).sum()),
        "bright_px": int((g > 150).sum()),
    }


def render_warmup(
    world,
    shooter: "Shooter",
    min_frames: int = 45,
    hard_frames: int = 160,
    stable: int = 3,
    tol: float = 1.0,
    verbose: bool = True,
    why: str = "",
) -> dict:
    """Pump RENDER-ONLY frames until the picture has stopped changing.

    Why this is not a `sleep` and not a `world.step`: two things land
    asynchronously, and the first frame of a task was wrong because of both.

    * **The asset's MDL texture is decoded by RTX on a worker thread.**  Measured on
      the claw hammer, on this box: render frames 0-10 come back with the whole
      asset BLACK, the texture lands between frame 10 and frame 15, and every frame
      from 15 on is correct.  The drop task captured its `initial` frame at frame
      0 -- which is exactly why the delivered drop initial frame showed an
      untextured black hammer while the `response` frame, 2.5 s and 70 frames later,
      was perfectly textured.  Waiting a fixed *simulated* time cannot fix this,
      because the decode is driven by render frames, not by physics time.
    * **DLSS accumulates over frames**, so a body that has just been teleported is
      ghosted at its old position for a few frames.

    `world.render()` advances the RENDERER and NOT physics -- measured, 300 render
    frames left `world.current_time` at 0.0120 s with the body's height and velocity
    unchanged.  So the texture can be brought in without eating any task time.

    **It does NOT make the renderer agree with a TELEPORTED pose**, and that limit is
    measured too: after `RigidPrim.set_world_poses` moved the asset to its release
    pose 0.325 m up in clear air, 45 render-only frames still drew it lying on the
    table at its previous pose -- while the physics view reported the new one.  The
    physics -> render transform sync happens inside a physics STEP.  Use
    `sync_pose_to_render` for that; `render_warmup` is for the material only.

    Returns the EVIDENCE, not just a frame count: the report carries
    `dark_px` before and after, so "the texture had loaded when this frame was
    taken" is a measurement in the report rather than a claim.
    """
    first = shooter.grab(world)
    prev = first
    n, run, d = 0, 0, 0.0
    while n < hard_frames:
        world.render()
        n += 1
        cur = shooter.grab(world)
        d = frame_diff(prev, cur)
        prev = cur
        if d < tol:
            run += 1
            if n >= min_frames and run >= stable:
                break
        else:
            run = 0
    a, b = frame_stats(first), frame_stats(prev)
    out = {
        "why": why or "capture",
        "render_frames": n,
        "min_frames": min_frames,
        "hard_frames": hard_frames,
        "stability_tol": tol,
        "last_frame_diff": round(float(d), 4),
        "first": a,
        "last": b,
        "texture_pop_observed": bool(b["dark_px"] < 0.75 * max(1, a["dark_px"])),
        "note": "render-only frames (world.render()): physics time is NOT advanced, "
        "so a frame can be warmed without consuming task time",
    }
    if verbose:
        log(
            f"[warm] {out['why']}: {n} render frames, last diff "
            f"{out['last_frame_diff']}, dark {a['dark_px']} -> {b['dark_px']} "
            f"(pop={out['texture_pop_observed']}), mean {a['mean']} -> {b['mean']}"
        )
    return out


def sync_pose_to_render(world, steps: int = 3) -> dict:
    """Put a TELEPORTED body pose into the renderer, at the cost of `steps` frames.

    `RigidPrim.set_world_poses` writes the physics view.  The renderer picks the new
    transform up during a physics STEP, not during `world.render()`: measured on the
    claw hammer, after teleporting it to its release pose (bbox bottom 0.150 m, body
    0.325 m up, in clear air) 45 render-only frames still drew it lying on the table
    at its previous pose.  The rendered "initial" frame was therefore a picture of
    the wrong pose, and nothing in the frame said so.

    `steps` real steps are the cheapest cure and they are cheap in effect: at
    `physics_dt = 0.002` three steps are 6 ms, in which a freely falling hammer drops
    0.18 mm.  The caller should record `world.current_time` next to the frame so the
    offset is stated rather than implied.
    """
    for _ in range(int(steps)):
        world.step(render=True)
    return {
        "pose_sync_steps": int(steps),
        "sim_time_s": round(float(world.current_time), 6),
        "note": "a teleport needs real steps to reach the renderer; see "
        "rig.sync_pose_to_render",
    }


def fit_camera_points(
    shooter: "Shooter",
    points,
    az_deg: float = 35.0,
    el_deg: float = 18.0,
    frame_frac: float = 0.82,
    min_eye_z: float = 0.05,
    note: str = "",
) -> dict:
    """`fit_camera` on a POINT CLOUD instead of a (centre, radius) pair.

    The grasp task needs this and the other two do not.  For drop and push the
    subject is the asset, and a sphere around the asset bounds the whole task.  For
    grasp the subject is "asset + gripper", and the gripper swings through half a
    metre of Z while the asset itself is only 4 cm tall -- bounding only the asset
    framed the gripper out of shot, which is what made the delivered grasp frames
    show a grasping arm with no arm in them.

    Pass every point that has to stay in shot (the asset where it rests, the asset
    at the top of the lift, the tool's envelope at both ends of its travel); the
    sphere is fitted to all of them.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    if len(P) == 0:
        raise ValueError("fit_camera_points: empty point cloud")
    centre = (P.min(axis=0) + P.max(axis=0)) * 0.5
    radius = float(np.max(np.linalg.norm(P - centre, axis=1)))
    out = fit_camera(
        shooter,
        centre,
        radius,
        az_deg=az_deg,
        el_deg=el_deg,
        frame_frac=frame_frac,
        min_eye_z=min_eye_z,
    )
    out["subject_centre_m"] = [round(float(v), 6) for v in centre]
    out["subject_points"] = int(len(P))
    out["note"] = note or (
        "sphere fitted to a point cloud (asset + tool envelope), not to the asset "
        "alone -- see rig.fit_camera_points"
    )
    return out


def add_lighting(stage, intensity: float = 1.0) -> list[str]:
    """Dome + distant light.

    Isaac's RTX renderer does NOT add a scene light for you in headless: with the
    default ground plane only, everything renders black (measured -- the first drop
    frame came out at mean 10/255 with nothing but the grid visible).  `DomeLight`
    gives the shadowless ambient fill, `DistantLight` gives the form shading.
    """
    from pxr import UsdLux, UsdGeom

    dome = UsdLux.DomeLight.Define(stage, "/World/lights/dome")
    dome.CreateIntensityAttr(600.0 * intensity)
    dist = UsdLux.DistantLight.Define(stage, "/World/lights/key")
    dist.CreateIntensityAttr(1500.0 * intensity)
    dist.CreateAngleAttr(4.0)
    UsdGeom.Xformable(dist.GetPrim()).AddRotateXYZOp().Set(
        (50.0, 0.0, -135.0)
    )  # key from the upper left, camera-ish side
    return ["/World/lights/dome", "/World/lights/key"]


def camera_intrinsics(shooter: "Shooter") -> dict:
    """Pinhole intrinsics the camera ACTUALLY has, read back from the sensor.

    Worth reading rather than assuming: Isaac does not keep the horizontal aperture
    you set.  It holds `vertical_aperture` fixed and re-derives the horizontal one
    from the render aspect -- measured on a 768x1024 camera, `v_ap/h_ap` came back
    as exactly `H/W`, and the resulting `fx == fy` (square pixels), which is the
    consistency check that this mapping is right.  Guessing otherwise is how the
    first framing attempt ended up with the camera buried in the ground plane.
    """
    fl = float(shooter.cam.get_focal_length())
    hap = float(shooter.cam.get_horizontal_aperture())
    vap = float(shooter.cam.get_vertical_aperture())
    w, h = shooter.res
    fx, fy = w * fl / hap, h * fl / vap
    return {
        "focal_length_mm": fl,
        "horizontal_aperture_mm": hap,
        "vertical_aperture_mm": vap,
        "focal_px": [round(fx, 3), round(fy, 3)],
        "hfov_deg": round(float(np.degrees(2 * np.arctan(w / (2 * fx)))), 4),
        "vfov_deg": round(float(np.degrees(2 * np.arctan(h / (2 * fy)))), 4),
    }


def fit_camera(
    shooter: Shooter,
    centre,
    radius,
    az_deg: float = 35.0,
    el_deg: float = 18.0,
    frame_frac: float = 0.82,
    min_eye_z: float = 0.05,
) -> dict:
    """Place the camera so a world-space sphere (centre, radius) fills `frame_frac`.

    Analytic, from the camera's own intrinsics -- no render/measure loop.  An
    earlier version fitted the distance by rendering with the subject hidden and
    diffing, which is self-defeating: it needs a correctly lit scene and a correct
    starting distance to produce a mask, and a bad mask then fed back into the
    distance, walking the camera into the ground.  A closed form has neither failure
    mode.

    `radius` should bound everything that has to stay in shot (for drop, the whole
    fall column), so the framing is stable across the task rather than per-frame.
    """
    intr = camera_intrinsics(shooter)
    centre = np.asarray(centre, float)
    az, el = np.radians(az_deg), np.radians(el_deg)
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    # the tighter axis governs; angular radius of the sphere must be half*frame_frac
    half = min(intr["hfov_deg"], intr["vfov_deg"]) / 2.0
    dist = float(radius) / np.sin(np.radians(half * frame_frac))
    eye = centre + direction * dist
    if eye[2] < min_eye_z:  # never let the camera dip into the floor
        eye = eye.copy()
        eye[2] = min_eye_z
    shooter.set_pose(eye, centre)
    return {
        "eye": [round(float(v), 6) for v in shooter.eye],
        "target": [round(float(v), 6) for v in shooter.target],
        "az_deg": az_deg,
        "el_deg": el_deg,
        "distance_m": round(float(np.linalg.norm(shooter.eye - centre)), 6),
        "subject_radius_m": round(float(radius), 6),
        "frame_frac": frame_frac,
        "eye_z_clamped": bool(eye[2] <= min_eye_z + 1e-9),
        "res": list(shooter.res),
        "intrinsics": intr,
        "note": "analytic fit from the camera's own intrinsics; stable across the "
        "whole task because `radius` bounds the entire working volume",
    }


def write_png(path: str, rgb: np.ndarray) -> str:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.asarray(rgb)[..., :3]).save(path)
    return path


class Video:
    """mp4 writer on cv2.  `imageio` cannot encode mp4 under Isaac's python for
    want of an ffmpeg backend (measured), and cv2 is bundled and works."""

    def __init__(self, path: str, fps: int, size):
        import cv2

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.size = (int(size[0]), int(size[1]))
        self._w = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), self.size
        )
        if not self._w.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {path}")
        self.path = path
        self.n = 0

    def append(self, rgb: np.ndarray) -> None:
        import cv2

        f = np.asarray(rgb)[..., :3]
        if f.dtype != np.uint8:
            f = _to_uint8(f)
        # cv2 wants BGR. Compare against the size captured at construction -- a
        # VideoWriter does NOT answer CAP_PROP_FRAME_WIDTH (it returns 0), so
        # querying the writer for its own size silently resizes to nothing.
        if (f.shape[1], f.shape[0]) != self.size:
            f = cv2.resize(f, self.size)
        self._w.write(f[..., ::-1])
        self.n += 1

    def close(self) -> str:
        """Finish the file and transcode it to H.264.

        cv2 can only give us `mp4v` here, which most players will not open -- see
        `find_ffmpeg`.  Doing the transcode at close keeps the frame path simple and
        costs one pass over a file that is already on disk.
        """
        self._w.release()
        self.h264 = to_h264(self.path)
        if not self.h264:
            log(
                f"    ! {os.path.basename(self.path)} left as mp4v (no ffmpeg found); "
                f"it is valid but Chromium/QuickTime will not open it"
            )
        return self.path


# ── reporting ────────────────────────────────────────────────────────────────
def load_thresholds(path: str = None) -> dict:
    """Thresholds are read from a file that is frozen BEFORE the run (需求 §8).

    Keeping them in a checked-in yaml rather than in the task source is the point:
    the file's git history is the evidence that no threshold was tuned after
    seeing results.
    """
    import yaml

    path = path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "thresholds.yaml"
    )
    with open(path) as fh:
        return yaml.safe_load(fh)


def write_report(out_dir: str, name: str, report: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_report.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return path


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two frames -- the guard that the
    'response' frame is not just the 'initial' frame again."""
    a, b = np.asarray(a, float)[..., :3], np.asarray(b, float)[..., :3]
    if a.shape != b.shape:
        return float("nan")
    return float(np.abs(a - b).mean())


def log(*a) -> None:
    """Progress line that actually survives.

    `SimulationApp` redirects the process's stdout -- and not merely `sys.stdout`,
    it takes the file descriptor, so even `sys.__stdout__` ends up inside Isaac's
    logger and a task's own progress vanishes.  A plain `print()` is worse still.
    So the line goes to a FILE as well, named by `ISAAC_TASK_LOG` (the orchestrator
    sets it per task); stdout is still attempted as a fallback.
    """
    line = " ".join(str(x) for x in a)
    path = os.environ.get("ISAAC_TASK_LOG")
    if path:
        try:
            with open(path, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    try:
        out = getattr(sys, "__stdout__", None) or sys.stdout
        out.write(line + "\n")
        out.flush()
    except Exception:  # noqa: BLE001
        pass


def dump_traceback(out_dir: str, name: str) -> str:
    """Persist the current exception next to the report.

    Needed for the same reason as `log`, plus one more: `SimulationApp.close()`
    terminates the process, so an exception raised inside a `finally`-guarded
    block never reaches the caller's handler.  Write it down before closing.
    """
    import traceback

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_traceback.txt")
    text = traceback.format_exc()
    with open(path, "w") as fh:
        fh.write(text)
    log(text)
    return path


def require_isaac():
    """Fail early and legibly when this is run under the wrong interpreter."""
    try:
        from isaacsim import SimulationApp  # noqa: F401
    except Exception as e:  # noqa: BLE001
        sys.exit(
            "isaac_tasks must run under Isaac Sim's python, e.g.\n"
            f"    {ISAAC_PYTHON} <script> [args]\n"
            f"(import isaacsim failed: {type(e).__name__}: {e})"
        )
