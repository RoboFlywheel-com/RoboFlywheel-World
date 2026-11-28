# SPDX-License-Identifier: Apache-2.0
"""Shared measured geometry and rendering helpers for simulator adapters."""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

# Deterministic prim name for the Isaac URDF importer (see sanitized_urdf).
ROBOT_NAME = "objroot"


def find_urdf(asset_dir):
    """The canonical aligned URDF in an asset dir.

    Prefers <key>.urdf; otherwise the first *.urdf that isn't a sanitized
    Isaac sibling (*_isaac.urdf). Returns a path or None.
    """
    key = os.path.basename(asset_dir)
    exact = os.path.join(asset_dir, f"{key}.urdf")
    if os.path.isfile(exact):
        return exact
    hits = [
        h
        for h in glob.glob(os.path.join(asset_dir, "*.urdf"))
        if not h.endswith("_isaac.urdf")
    ]
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# URDF metadata
# --------------------------------------------------------------------------- #
def parse_urdf_meta(urdf_path):
    """Read the fields the demo needs from an aligned URDF.

    Returns a dict: uuid, mass, aabb_min/max (m, mesh frame == link frame),
    center, extent, real_height, category, description.
    """
    root = ET.parse(urdf_path).getroot()
    info = root.find(".//extra_info")

    def _txt(tag, default=None):
        el = info.find(tag) if info is not None else None
        return el.text.strip() if el is not None and el.text else default

    aabb = info.find("aabb") if info is not None else None
    if aabb is not None:
        amin = np.array([float(x) for x in aabb.find("min").text.split()])
        amax = np.array([float(x) for x in aabb.find("max").text.split()])
    else:  # fall back to the visual mesh bounds
        import trimesh

        visual = root.find(".//visual/geometry/mesh")
        if visual is None:
            raise ValueError("URDF lacks visual geometry")
        mesh_path = os.path.join(
            os.path.dirname(os.path.abspath(urdf_path)), visual.get("filename")
        )
        measured = trimesh.load(mesh_path, force="mesh")
        measured.apply_scale([float(v) for v in visual.get("scale", "1 1 1").split()])
        amin, amax = measured.bounds

    mass_el = root.find(".//inertial/mass")
    mass = float(mass_el.get("value")) if mass_el is not None else 0.0

    return {
        "uuid": _txt("uuid", "asset"),
        "mass": mass,
        "aabb_min": amin,
        "aabb_max": amax,
        "center": (amin + amax) / 2.0,
        "extent": amax - amin,
        "real_height": float(_txt("real_height", str((amax - amin)[2]))),
        "category": _txt("category", ""),
        "description": _txt("description", ""),
    }


def sanitized_urdf(urdf_path, suffix="_isaac"):
    """Write a standard-only URDF sibling for strict parsers (Isaac urdfdom).

    Our aligned URDF carries non-standard tags (<extra_info>, <custom_data>,
    and <gazebo> inside <collision>) that make Isaac's urdfdom return a null
    robot -> "Used null prim". Strip them, keep link/visual/collision/inertial
    with the SAME relative mesh paths (written next to the original so paths
    still resolve). Returns the new path.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    # Give the robot a valid, digit-free name so the Isaac importer maps it to a
    # deterministic prim path (/<ROBOT_NAME>); our uuids start with a digit,
    # which the importer would otherwise rewrite to "a_<uuid>".
    root.set("name", ROBOT_NAME)
    # absolutize mesh paths (Isaac importer resolves relative names against CWD)
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and not os.path.isabs(fn) and not fn.startswith("package://"):
            mesh.set("filename", os.path.join(urdf_dir, fn))
    for link in root.findall("link"):
        for coll in link.findall("collision"):
            for gz in coll.findall("gazebo"):
                coll.remove(gz)
    for tag in ("extra_info", "custom_data"):
        for el in root.findall(f".//{tag}"):
            # ElementTree can't remove by descendant directly; find parent
            for parent in root.iter():
                if el in list(parent):
                    parent.remove(el)
                    break
    out = os.path.splitext(urdf_path)[0] + suffix + ".urdf"
    tree.write(out, xml_declaration=True, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Scene geometry from the AABB (drop pose + camera framing)
# --------------------------------------------------------------------------- #
def rest_pose(meta, table_z=0.0, gap=0.02):
    """Spawn pose so the object's AABB bottom starts `gap` above the table.

    Returns (spawn_xyz, settled_center_xyz) in world coords, where the object's
    local origin is offset from its AABB center by `-center`.
    """
    ext = meta["extent"]
    center = meta["center"]
    # World z of the object *origin* so that aabb_min.z sits at table_z + gap.
    spawn_z = table_z + gap - meta["aabb_min"][2]
    spawn = np.array([0.0, 0.0, spawn_z])
    # After settling the aabb bottom rests on the table; center height ~ half z-extent.
    settled_center = np.array([0.0, 0.0, table_z + ext[2] / 2.0])
    # object origin sits at settled_center - center once rested
    _ = center
    return spawn, settled_center


def orbit_framing(meta, table_z=0.0, radius_k=3.1, height_k=1.35):
    """Camera orbit params sized to the object AABB.

    Returns dict: target (look-at point), radius, height, fov_deg.
    Small/flat objects get pulled in close; tall ones get more distance.
    """
    ext = meta["extent"]
    diag_xy = float(np.hypot(ext[0], ext[1]))
    reach = max(diag_xy, float(ext[2]), 1e-3)
    target = np.array([0.0, 0.0, table_z + ext[2] / 2.0])
    radius = radius_k * reach
    height = table_z + ext[2] / 2.0 + height_k * reach
    return {
        "target": target,
        "radius": float(radius),
        "height": float(height),
        "fov_deg": 40.0,
    }


def vertical_fov(hfov_deg, res):
    """Horizontal fov (deg) -> vertical fov (RADIANS) for a res=(W,H) image.

    orbit_framing returns a horizontal fov (Isaac's aperture convention); the
    other engines take a vertical fov, so every driver converts with the aspect
    ratio to keep the object the same on-screen size. Callers that want degrees
    wrap this in np.degrees().
    """
    h = np.radians(hfov_deg)
    return 2.0 * np.arctan(np.tan(h / 2.0) * res[1] / res[0])


def orbit_positions(framing, n_frames):
    """Yield (x, y, z) camera eye positions around a full 360 circle."""
    r, h = framing["radius"], framing["height"]
    tx, ty, _ = framing["target"]
    for i in range(n_frames):
        a = 2.0 * np.pi * i / n_frames
        yield np.array([tx + r * np.cos(a), ty + r * np.sin(a), h])


def look_at_quat_wxyz(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Look-at rotation as a (w, x, y, z) quaternion.

    Same basis convention as the world Z-up camera basis
    (forward, -right, up), so it matches the repo's camera math.
    """
    from scipy.spatial.transform import Rotation as R

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    rot = np.array([fwd, -right, up2]).T
    q = R.from_matrix(rot).as_quat()  # (x, y, z, w)
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
def write_mp4(frames, out_path, fps=30):
    """Write a list of HxWx3/4 frames (uint8 or float[0,1]) to an mp4."""
    import imageio

    with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as w:
        for f in frames:
            f = np.asarray(f)
            if f.dtype != np.uint8:
                f = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
            w.append_data(f[..., :3])
    return out_path


def tipped_over(quat_wxyz, tol_deg=45.0):
    """True if the object's local +z tilted more than tol from world +z."""
    from scipy.spatial.transform import Rotation as R

    w, x, y, z = quat_wxyz
    zdir = R.from_quat([x, y, z, w]).apply([0.0, 0.0, 1.0])
    ang = np.degrees(np.arccos(np.clip(zdir[2], -1.0, 1.0)))
    return bool(ang > tol_deg), float(ang)
