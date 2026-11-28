#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo backend for the SPEC-014 five-simulator figure.

Reads the frozen ``common_scene.json`` through :mod:`common` and writes, into a
fresh ``--output-dir``::

    rgb.png                    1600x1000, the visible geometry
    depth_visible.npy          float32, metres along the camera forward axis
    depth_visible_mask.npy     bool
    depth_collision.npy        the same quantity for the collision meshes
    depth_collision_mask.npy   bool
    report.json                the ledger

The room is compiled once into a static MJCF: every object becomes a body whose
``pos``/``quat`` come from ``SceneObject.matrix_world()`` and whose geometry is
the object's own ``.obj`` file, read by MuJoCo. Visual geometry lives in geom
group 0 and collision geometry in group 1, so the two passes are one model
rendered twice with ``mjvOption.geomgroup`` switched — never a merged buffer.

Six things in here are measured facts, not guesses. Each one cost a run:

1. **MuJoCo renders the left stereo eye, not the eye you asked for.**
   ``scn.camera[0].pos`` is the requested position displaced by ``ipd/2`` along
   the camera right axis (``ipd`` = ``model.vis.global_.ipd``, 0.068 m by
   default). Measured: with ``lookat = target`` the image is centred on
   ``target - distance*forward``, and the projected silhouette of a test cube
   lands within 0.5 px of ``CameraSpec.project``; putting ``ipd/2*right`` back
   into the lookat shifts the whole image 7.0 px sideways (0.0355 m at 3.04 m).
   So the lookat stays on ``camera.target`` and the camera the rasteriser uses is
   the nominal one. ``camera_frame_self_check`` records the residuals.
2. **MuJoCo keeps the faces of the first OBJ ``o``/``g`` group only.** Measured
   on this scene: the plant's 133612 faces compile to 5500 (group ``Shader4``,
   the first one), the storage's 1936 to 384 (``wood``), the wall art's 76 to 2
   (``Post``). The vertices of every group survive, so the face indices stay
   valid and stripping the ``o``/``g`` records is enough to get the whole
   surface back. ``obj_group_collapse`` in the report lists what was at stake
   per file. Without this MuJoCo quietly renders a fraction of each object.
3. **MuJoCo back-face culls, and these exports are not all wound outward.**
   ``scn.flags[mjRND_CULL_FACE]`` is 1 by default. A surface wound the other way
   does not paint its far side, it disappears — measured on a test cube with one
   face reversed, the depth came back as background. ``mjRND_CULL_FACE`` is
   turned off so the surface is drawn from whichever side faces the camera; the
   report records it.
4. **Flat assets break the default mesh inertia.** ``wall_art`` and ``rug`` are
   surfaces with no volume; MuJoCo refuses them ("mesh volume is too small").
   Every mesh is compiled with ``inertia="shell"``, which is the right inertia
   for a surface and is irrelevant to a joint-free model.
5. **MuJoCo's free camera cannot roll.** Its frame is always built from world
   ``+Z`` (``right = fwd x Z``, ``up = right x fwd``), which is exactly the
   convention ``CameraSpec.rotation_world_to_cam`` uses, so the figure camera is
   expressible in it exactly — nothing about the camera is silently dropped.
6. **MuJoCo does not read OBJ material libraries.** The visual pass splits
   faces by ``usemtl`` and declares the original MTL diffuse colour/texture as
   native MJCF materials. The split retains source vertices, UVs and faces;
   collision meshes remain in their own pass.
7. **The bundle's shell meshes are a claim, not a given.** They ship with
   ``translation`` = 0, which asserts the OBJ vertices are already world metres.
   On 2026-09-24 the bundle generated at 16:05 did not satisfy that: its shell
   vertices were centred on their own centroids (the sibling ``pybullet`` and
   ``sapien`` drivers found the same thing independently), so the floor and
   ceiling sat at the world origin 2.8 m from the room the same bundle declares,
   and this backend rendered 38.4% of its frame against an empty scene. The
   bundle was regenerated in place at 16:26:47 with the vertices on the room —
   identical to re-deriving them from ``shell.usdc`` under USD's row-vector
   convention (``p_world = p_local @ M[:3,:3] + M[3,:3]``), which is where
   ``extract_scene.py``'s ``points @ xform[:3,:3].T + xform[:3,3]`` went wrong.
   ``shell_frame_check`` is the gate that catches it, and every mesh the run
   loaded is digested into the report because ``scene_version`` did not change
   when the meshes did.

Depth: ``mujoco.Renderer`` in depth mode returns metres along the optical axis
(``camera_z``), deprojected from the reversed-Z buffer (``mjDEPTH_ZEROFAR``:
1 = near, 0 = far) with ``z = (f*n/(f-n)) / (raw + n/(f-n))`` where
``n = vis.map.znear*stat.extent`` and ``f = vis.map.zfar*stat.extent``; a
background pixel has ``raw = 0`` and comes out as exactly ``f``. That is checked
twice below — once by projecting mesh vertices with ``CameraSpec.project`` and
once by ray-casting the triangle soup independently — and both numbers are in
the report.

Run:

    env -u PYTHONPATH -u PYTHONHOME \
      /opt/conda/envs/scenesmith/bin/python \
      python -m rfw_w.scenes.workers.multisim.render_mujoco \
      --scene outputs/_paper/spec014_common_scene_20260924_01 \
      --output-dir outputs/_paper/spec014_renders_20260924_01/mujoco
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import hashlib
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

# MuJoCo's Python bindings pick a GL platform at import time. There is no X
# display on this machine, so GLFW fails; EGL is what works here. ``setdefault``
# keeps an explicit MUJOCO_GL override working.
os.environ.setdefault("MUJOCO_GL", "egl")


import mujoco  # must follow MUJOCO_GL

from rfw_w.scenes.workers.multisim.common import (
    CommonScene,
    appearance_gates,
    backend_report,
    colourise_depth,
    depth_to_world_normal_rgb,
    digest,
    load_light_rig,
    mtl_definitions,
    save_depth_products,
    split_visual_materials,
    write_json,
)

VISUAL_GROUP = 0
COLLISION_GROUP = 1

# MuJoCo point lights do not fall off the way a path tracer's do, so the rig's
# raytracer-tuned point intensities blow out the desk if used raw. One declared
# scale for all of them, rather than retuning each light per backend.
MUJOCO_POINT_SCALE = 0.18

# mjNGROUP lives in the C headers, not always in the Python bindings.
N_GEOM_GROUPS = len(mujoco.MjvOption().geomgroup)

# Depth clip planes as multiples of ``model.stat.extent``. ``stat.extent`` is
# pinned to 1.0 after compilation so these are metres outright: near at 5 cm,
# far at 60 m for a room whose diagonal is ~6.1 m.
ZNEAR_M = 0.05
ZFAR_M = 60.0

# Pixels whose depth differs from a neighbour by more than this sit on a
# discontinuity or on a surface seen at a grazing angle; a ray through the pixel
# centre is not comparable with the rasterised value there, so both checks skip
# them. 1 cm across one pixel is already a 1:1 slope.
EDGE_TOLERANCE_M = 0.01

# A projected vertex is treated as occluded when the buffer renders more than
# this much in front of it. 5 mm is far below any geometric disagreement worth
# calling a depth error, and far above rasterisation noise.
OCCLUSION_TOLERANCE_M = 0.005

# How far the shell's own bounding box may sit inside the room envelope the
# bundle declares before the shell is called displaced. Deliberately loose: the
# ceiling mesh sits one slab thickness below `height_m` (49.6 mm on this bundle),
# while the displaced bundle generated at 16:05 was 2.818 m out on x.
SHELL_ENVELOPE_TOLERANCE_M = 0.25


def strip_obj_groups(source: Path, dest: Path) -> dict:
    """Rewrite an OBJ with its ``o`` / ``g`` records removed.

    MuJoCo's OBJ reader keeps the faces of the first group only (see the module
    docstring for the measurement). Vertices of every group are kept, so face
    indices stay valid and dropping the group records recovers the whole
    surface. Returns the counts that make the loss visible in the report.
    """
    counts: list[list] = []
    kept: list[str] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        head = line.split(maxsplit=1)[0] if line.split() else ""
        if head in ("o", "g"):
            name = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else "<unnamed>"
            counts.append([name, 0])
            continue
        if head == "f":
            if not counts:
                counts.append(["<implicit>", 0])
            counts[-1][1] += 1
        kept.append(line)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(kept) + "\n", encoding="utf-8")
    total = sum(n for _, n in counts)
    return {
        "groups": len(counts),
        "faces_in_obj": total,
        "faces_mujoco_reads_without_collapse": counts[0][1] if counts else 0,
        "faces_after_group_collapse": total,
        "group_face_counts": {name: n for name, n in counts[:8]},
    }


def build_scene_xml(scene: CommonScene, objects, mesh_dir: Path, camera, rig) -> tuple[str, list[dict]]:
    """A static MJCF whose visual mesh geoms retain source MTL finishes.

    One body per object, one geom per pass in that pass's group. ``matrix_world``
    is rotation-then-translation with a uniform scale, so it maps onto
    ``<body pos quat>`` plus ``<mesh scale>`` with nothing left over. No joints:
    this model is never stepped, only rendered.
    """
    meshes: list[str] = []
    bodies: list[str] = []
    materials: list[str] = []
    material_records: list[dict] = []
    for index, obj in enumerate(objects):
        quat = obj.rotation_wxyz
        pose = (f'pos="{obj.translation[0]} {obj.translation[1]} {obj.translation[2]}" '
                f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"')
        visual_parts = split_visual_materials(obj.visual_mesh, mesh_dir / f"{index:02d}v",
                                              default_diffuse=rig.shell_albedo)
        for part_index, part in enumerate(visual_parts):
            target = part["mesh"]
            mesh_name = f"m{index}v{part_index}"
            material_name = f"mat{index}v{part_index}"
            meshes.append(
                f'    <mesh name="{mesh_name}" file={quoteattr(str(target.resolve()))} '
                f'scale="{obj.scale} {obj.scale} {obj.scale}" inertia="shell"/>'
            )
            texture = part["texture"]
            if texture is not None:
                texture_name = f"tex{index}v{part_index}"
                materials.append(f'    <texture name="{texture_name}" type="2d" '
                                 f'file={quoteattr(str(texture.resolve()))}/>')
                texture_attr = f' texture="{texture_name}"'
            else:
                texture_attr = ""
            rgb = part["diffuse_rgb"]
            materials.append(f'    <material name="{material_name}" '
                             f'rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1"{texture_attr}/>')
            bodies.append(
                f'    <body name="b{index}v{part_index}" {pose}>\n'
                f'      <geom name="g{index}v{part_index}" type="mesh" mesh="{mesh_name}" '
                f'material="{material_name}" group="{VISUAL_GROUP}" '
                f'contype="0" conaffinity="0"/>\n'
                f"    </body>"
            )
            material_records.append({"object_id": obj.object_id, "material": part["name"],
                                     "face_count": part["faces"],
                                     "texture": str(texture) if texture else None,
                                     "diffuse_rgb": list(rgb),
                                     "geom": f"g{index}v{part_index}"})
        target = mesh_dir / f"{index:02d}c.obj"
        strip_obj_groups(obj.collision_mesh, target)
        meshes.append(f'    <mesh name="m{index}c" file={quoteattr(str(target.resolve()))} '
                      f'scale="{obj.scale} {obj.scale} {obj.scale}" inertia="shell"/>')
        bodies.append(
            f'    <body name="b{index}c" {pose}>\n'
            f'      <geom name="g{index}c" type="mesh" mesh="m{index}c" '
            f'group="{COLLISION_GROUP}" contype="0" conaffinity="0"/>\n'
            f"    </body>"
        )
    # The shared rig, as explicit <light> elements. The headlight is switched
    # off: it is a camera-attached lamp no other backend has, so leaving it on
    # would make MuJoCo's shading a function of where the camera is rather than
    # of the rig every backend is driven from.
    gain = rig.gain_for("mujoco")
    lights = [
        f'    <light directional="true" castshadow="true" '
        f'pos="{-rig.direction[0] * 4:.4f} {-rig.direction[1] * 4:.4f} {-rig.direction[2] * 4:.4f}" '
        f'dir="{rig.direction[0]:.4f} {rig.direction[1]:.4f} {rig.direction[2]:.4f}" '
        f'diffuse="{rig.directional_colour[0] * gain:.4f} '
        f'{rig.directional_colour[1] * gain:.4f} {rig.directional_colour[2] * gain:.4f}" '
        f'specular="0.05 0.05 0.05"/>'
    ]
    for point in rig.point_lights:
        position = point["position"]
        colour = point["colour"]
        # MuJoCo point lights are not physically falling off like a raytracer's,
        # so the rig's raytracer-tuned intensities are scaled down by the same
        # declared factor for every one of them rather than retuned per light.
        lights.append(
            f'    <light directional="false" castshadow="false" '
            f'pos="{position[0]:.4f} {position[1]:.4f} {position[2]:.4f}" '
            f'dir="0 0 -1" '
            f'diffuse="{colour[0] * gain * MUJOCO_POINT_SCALE:.4f} '
            f'{colour[1] * gain * MUJOCO_POINT_SCALE:.4f} '
            f'{colour[2] * gain * MUJOCO_POINT_SCALE:.4f}" '
            f'specular="0 0 0"/>'
        )
    ambient = " ".join(f"{c * gain:.4f}" for c in rig.ambient)
    xml = (
        '<mujoco model="pvscene_spec014_mujoco">\n'
        f'  <visual>\n    <global offwidth="{camera.width}" '
        f'offheight="{camera.height}"/>\n'
        '    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0" active="0"/>\n'
        f'    <map znear="{ZNEAR_M}" zfar="{ZFAR_M}"/>\n  </visual>\n'
        f'  <asset>\n' + "\n".join(meshes + materials) + "\n  </asset>\n"
        f'  <worldbody>\n'
        f'    <light name="rig_ambient" directional="true" castshadow="false" '
        f'pos="0 0 6" dir="0 0 -1" ambient="{ambient}" diffuse="0 0 0" specular="0 0 0"/>\n'
        + "\n".join(lights) + "\n"
        + "\n".join(bodies) + "\n  </worldbody>\n"
        "</mujoco>\n"
    )
    return xml, material_records


def world_triangles(model, data, geom_ids):
    """World-space vertices and triangles for the mesh geoms in ``geom_ids``.

    ``model.mesh_vert`` holds the vertices in the frame MuJoCo re-expressed the
    mesh in (it moves a mesh onto its principal axes and records the transform on
    the geom), so the geom's own ``xmat``/``xpos`` recover the world positions.
    """
    vertices = []
    faces = []
    offset = 0
    for gid in geom_ids:
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        v_adr, v_num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f_adr, f_num = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        local = model.mesh_vert[v_adr:v_adr + v_num].reshape(-1, 3).astype(np.float64)
        tri = model.mesh_face[f_adr:f_adr + f_num].reshape(-1, 3).astype(np.int64)
        rot = data.geom_xmat[gid].reshape(3, 3).astype(np.float64)
        world = local @ rot.T + data.geom_xpos[gid].astype(np.float64)
        vertices.append(world)
        faces.append(tri + offset)
        offset += len(world)
    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(vertices), np.concatenate(faces)


def ray_cast_camera_z(origin, direction, verts, faces):
    """First-hit ``camera_z`` along a ray; Moller-Trumbore over the faces.

    ``direction`` is the *unnormalised* "camera-z = 1" ray for its pixel, so the
    ray parameter at the hit is the camera-z itself. One pass over a few hundred
    thousand faces per ray, and a few hundred rays is all the check needs.
    """
    v0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - v0
    e2 = verts[faces[:, 2]] - v0
    pvec = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        tvec = origin - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        v = np.einsum("ij,ij->i", np.broadcast_to(direction, qvec.shape), qvec) * inv
        t = np.einsum("ij,ij->i", e2, qvec) * inv
    hit = (np.abs(det) >= 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
    if not hit.any():
        return None
    return float(t[hit].min())


def pixel_ray_direction(camera, u: int, v: int) -> np.ndarray:
    """The "camera-z = 1" ray through the centre of pixel ``(u, v)``."""
    rot = camera.rotation_world_to_cam
    right, up, back = rot[0], rot[1], rot[2]
    return (
        right * ((u + 0.5 - camera.cx) / camera.fx)
        + up * (-(v + 0.5 - camera.cy) / camera.fy)
        - back
    )


def interior_pixels(depth: np.ndarray, background_m: float) -> np.ndarray:
    """Pixels far enough from a depth edge that a centre ray is comparable."""
    finite = np.isfinite(depth) & (depth > 0) & (depth < background_m)
    interior = np.zeros_like(finite)
    interior[1:-1, 1:-1] = (
        finite[:-2, 1:-1] & finite[2:, 1:-1] & finite[1:-1, :-2] & finite[1:-1, 2:]
    )
    gentle = np.zeros_like(finite)
    centre = depth[1:-1, 1:-1]
    gentle[1:-1, 1:-1] = (
        (np.abs(depth[:-2, 1:-1] - centre) < EDGE_TOLERANCE_M)
        & (np.abs(depth[2:, 1:-1] - centre) < EDGE_TOLERANCE_M)
        & (np.abs(depth[1:-1, :-2] - centre) < EDGE_TOLERANCE_M)
        & (np.abs(depth[1:-1, 2:] - centre) < EDGE_TOLERANCE_M)
    )
    return interior & gentle


def _residual_stats(residual: np.ndarray) -> dict:
    residual = np.asarray(residual, dtype=np.float64)
    return {
        "n_points": int(residual.size),
        "median_m": float(np.median(residual)),
        "mean_m": float(residual.mean()),
        "p95_m": float(np.percentile(residual, 95)),
        "max_m": float(residual.max()),
        "fraction_within_1cm": float((residual < 0.01).mean()),
    }


def project_self_check(camera, depth, verts, background_m, max_points=2000, seed=0):
    """Compare the rendered depth against ``camera.project`` of surface points.

    World mesh vertices are projected into the image; a vertex that is the
    nearest sampled point at its own pixel is a first-hit candidate, and the
    rendered depth there must equal its projected ``camera_z``. Three filters
    keep the comparison about the conversion rather than about rasterisation:

    * the pixel must have a gentle depth gradient (see ``interior_pixels``);
    * the vertex must be the front-most *sampled* vertex at its pixel;
    * nothing may render in front of it.

    The last two are not the same filter. A floor whose vertices happen to miss
    a pixel leaves ``nearest`` at infinity there, so a vertex 2 m behind the
    floor would be kept as "front" and compared against the floor's depth. That
    mistake measured 462 mm on this scene; with both filters the same quantity
    comes back in the low millimetres. The drop counts are reported because the
    filter can only hide a *too small* conversion, which the ray-cast check
    below cannot: it compares against geometry, not against this buffer.
    """
    rng = np.random.default_rng(seed)
    step = max(1, len(verts) // 400000)
    sample = verts[::step]
    uv, z = camera.project(sample)
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(z) & (z > 0)
    dropped = {"no_vertices": int((~inside).sum())}
    u, v, z = u[inside], v[inside], z[inside]
    if u.size == 0:
        return {"n_points": 0, "dropped": dropped, "note": "no projectable vertices"}
    flat = v * w + u
    nearest = np.full(h * w, np.inf)
    np.minimum.at(nearest, flat, z)
    is_front = z <= nearest[flat] * (1.0 + 1e-9) + 1e-9
    dropped["not_front_most_sample"] = int((~is_front).sum())
    u, v, z = u[is_front], v[is_front], z[is_front]
    usable = interior_pixels(depth, background_m)[v, u]
    dropped["not_a_gentle_interior_pixel"] = int((~usable).sum())
    u, v, z = u[usable], v[usable], z[usable]
    measured = depth[v, u]
    unoccluded = measured >= z - OCCLUSION_TOLERANCE_M
    dropped["occluded_something_renders_in_front"] = int((~unoccluded).sum())
    u, v, z, measured = u[unoccluded], v[unoccluded], z[unoccluded], measured[unoccluded]
    if z.size == 0:
        return {"n_points": 0, "dropped": dropped, "note": "no vertex survived the filters"}
    if z.size > max_points:
        pick = rng.choice(z.size, size=max_points, replace=False)
        u, v, z, measured = u[pick], v[pick], z[pick], measured[pick]
    return dict(
        _residual_stats(np.abs(measured - z)),
        dropped=dropped,
        pixel_span_u=[int(u.min()), int(u.max())],
        pixel_span_v=[int(v.min()), int(v.max())],
    )


def raycast_self_check(camera, depth, verts, faces, background_m, n_rays=36, ipd=0.0,
                       right=None):
    """The same quantity, measured independently by CPU ray casting.

    Rays are cast from three origins — the requested eye, and the eye displaced
    by half an interpupillary distance to each side — because MuJoCo exposes
    ``scn.camera[0].pos`` as the left stereo eye while the rasterised image is
    the nominal one (fact 1 in the module docstring). Reporting all three turns
    "which eye is this?" into a measurement instead of an assertion: the nominal
    eye must be the one whose rays reproduce the buffer.
    """
    h, w = depth.shape
    usable = interior_pixels(depth, background_m)
    us = np.arange(4, w - 4, max(1, w // 40))
    vs = np.arange(4, h - 4, max(1, h // 24))
    grid_u, grid_v = np.meshgrid(us, vs)
    candidates = [(int(a), int(b)) for a, b in zip(grid_u.ravel(), grid_v.ravel()) if usable[b, a]]
    if not candidates:
        return {"n_points": 0, "note": "no gentle interior pixel to ray-cast"}
    candidates = candidates[:: max(1, len(candidates) // n_rays)][:n_rays]

    origins = {"nominal_eye": np.asarray(camera.eye, dtype=np.float64)}
    if right is not None and ipd > 0:
        origins["left_eye_minus_ipd_2"] = camera.eye - 0.5 * ipd * np.asarray(right)
        origins["right_eye_plus_ipd_2"] = camera.eye + 0.5 * ipd * np.asarray(right)

    by_origin = {}
    for name, origin in origins.items():
        residuals = []
        for u, v in candidates:
            hit = ray_cast_camera_z(origin, pixel_ray_direction(camera, u, v), verts, faces)
            if hit is None:
                continue
            residuals.append(abs(float(depth[v, u]) - hit))
        if residuals:
            by_origin[name] = _residual_stats(np.asarray(residuals))
    if not by_origin.get("nominal_eye", {}).get("n_points"):
        return {"n_points": 0, "by_origin": by_origin, "note": "no ray hit any triangle"}
    nominal = by_origin["nominal_eye"]
    others = [r["median_m"] for k, r in by_origin.items() if k != "nominal_eye"]
    return dict(
        nominal,
        by_origin=by_origin,
        worst_other_origin_median_m=max(others) if others else None,
        nominal_is_the_rasterised_eye=(bool(others) and nominal["median_m"] < min(others)),
    )


def dominant_colour(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """The frame's single most common colour and the share of pixels that are it.

    A "how empty is this frame" measure has to be taken against the frame's own
    dominant colour rather than a colour read off the border: once the room shell
    fills the frame (which is the point of the fixed bundle) the border is room,
    not background, and a border-based reference counts 65% of a full room as
    background.
    """
    packed = (
        (rgb[..., 0].astype(np.int32) << 16)
        | (rgb[..., 1].astype(np.int32) << 8)
        | rgb[..., 2].astype(np.int32)
    )
    values, counts = np.unique(packed, return_counts=True)
    top = values[int(np.argmax(counts))]
    return (
        np.array([(top >> 16) & 255, (top >> 8) & 255, top & 255], dtype=np.uint8),
        float(counts.max() / packed.size),
    )


def shell_frame_check(scene: CommonScene, objects, model, data) -> dict:
    """Is the shell the bundle ships where the bundle's own ``room`` record says?

    The bundle carries a ``room`` record (``polygon_xy``, ``height_m``,
    ``floor_z_m``) and shell meshes with ``translation`` = 0 and rotation =
    identity, i.e. the shell meshes are *declared* to be in world metres already.
    That is a claim about the OBJ files, and on 2026-09-24 it was false for
    everything except the skirting: the bundle generated at 16:05 had its shell
    OBJ vertices centred on their own centroids instead of on the room, which put
    the floor, ceiling and walls at the world origin. Nothing in the reports
    could fail on it, and the MuJoCo column rendered 38% of its frame against an
    empty scene because the room was 2.8 m away from itself. The bundle was
    regenerated at 16:26:47 and this check is what would have caught it: compare
    the shell's own bounding box against the room the same record declares.
    """
    room = dict(scene.payload.get("room", {}))
    polygon = np.asarray(room.get("polygon_xy", []), dtype=np.float64)
    result = {
        "declared_room": {
            "polygon_xy": polygon.tolist(),
            "height_m": room.get("height_m"),
            "floor_z_m": room.get("floor_z_m"),
            "source": "common_scene.json room record, not the renderer",
        },
        "tolerance_m": SHELL_ENVELOPE_TOLERANCE_M,
    }
    if polygon.size == 0 or room.get("height_m") is None:
        result["gap"] = "MULTISIM_ROOM_UNSUPPORTED(mujoco, bundle declares no room envelope)"
        result["encloses_declared_room"] = False
        return result
    shells = [(i, o) for i, o in enumerate(objects) if o.kind == "shell" and o.role in
              ("wall", "floor", "ceiling")]
    if not shells:
        result["gap"] = "MULTISIM_ROOM_UNSUPPORTED(mujoco, bundle ships no wall/floor/ceiling mesh)"
        result["encloses_declared_room"] = False
        return result
    # the visual geom of object i is geom 2i (one visual + one collision per object)
    shell_verts, _ = world_triangles(model, data, [2 * index for index, _ in shells])
    if shell_verts.size == 0:
        result["gap"] = "MULTISIM_ROOM_UNSUPPORTED(mujoco, shell meshes compiled to no triangles)"
        result["encloses_declared_room"] = False
        return result
    lo, hi = shell_verts.min(axis=0), shell_verts.max(axis=0)
    declared_lo = np.array([polygon[:, 0].min(), polygon[:, 1].min(),
                            float(room["floor_z_m"])])
    declared_hi = np.array([polygon[:, 0].max(), polygon[:, 1].max(),
                            float(room["floor_z_m"]) + float(room["height_m"])])
    shortfall = {
        "x_lo": float(declared_lo[0] - lo[0]),
        "x_hi": float(hi[0] - declared_hi[0]),
        "y_lo": float(declared_lo[1] - lo[1]),
        "y_hi": float(hi[1] - declared_hi[1]),
        "z_lo": float(declared_lo[2] - lo[2]),
        "z_hi": float(hi[2] - declared_hi[2]),
    }
    worst = max(shortfall.values())
    result.update(
        shell_world_bbox_m={"lo": lo.tolist(), "hi": hi.tolist()},
        declared_room_bbox_m={"lo": declared_lo.tolist(), "hi": declared_hi.tolist()},
        shortfall_m=shortfall,
        worst_shortfall_m=worst,
        encloses_declared_room=bool(worst <= SHELL_ENVELOPE_TOLERANCE_M),
    )
    if not result["encloses_declared_room"]:
        axis = max(shortfall, key=lambda k: shortfall[k])
        result["gap"] = (
            f"MULTISIM_GEOMETRY_MISMATCH(mujoco, shell does not enclose the room the bundle "
            f"declares: worst shortfall {worst:.3f} m at {axis}; the shell meshes are not where "
            "their zero translation assumes)"
        )
    return result


def background_colour(rgb: np.ndarray) -> np.ndarray:
    """The most common colour on the frame's border: the engine's clear colour."""
    border = np.concatenate(
        [rgb[0].reshape(-1, 3), rgb[-1].reshape(-1, 3),
         rgb[:, 0].reshape(-1, 3), rgb[:, -1].reshape(-1, 3)]
    )
    packed = (border[:, 0].astype(np.int32) << 16) | (border[:, 1].astype(np.int32) << 8) | border[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    top = values[int(np.argmax(counts))]
    return np.array([(top >> 16) & 255, (top >> 8) & 255, top & 255], dtype=np.uint8), int(counts.max())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, help="directory holding common_scene.json")
    parser.add_argument("--output-dir", required=True, help="fresh directory to write into")
    parser.add_argument("--lighting", type=Path,
                        default=data_path('examples')
                        / "paper_multisim" / "lighting.json",
                        help="the shared five-backend light rig (SPEC-014 R5)")
    args = parser.parse_args(argv)

    started = time.time()
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"OUTPUT_DIR_NOT_FRESH({out_dir}): this directory already has files in it. "
            "Every acceptance run writes a fresh directory; a re-run must not repair an "
            "old one. Pass a new --output-dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = CommonScene.load(args.scene)
    rig = load_light_rig(args.lighting)
    camera = scene.camera
    objects = scene.objects()
    all_ids = {o.object_id for o in scene.objects(include_hidden=True)}
    hidden_ids = sorted(all_ids - {o.object_id for o in objects})

    warnings: list[str] = []
    if hidden_ids:
        warnings.append(
            f"HIDDEN_BY_SHARED_RULE({','.join(hidden_ids)}): role "
            f"{','.join(sorted(scene.hidden_rule().get('hidden_roles', [])))} is skipped by "
            "every backend; from an interior camera the facade only occludes."
        )
    mesh_dir = Path(tempfile.mkdtemp(prefix="pvscene_spec014_mujoco_"))
    xml, material_records = build_scene_xml(scene, objects, mesh_dir, camera, rig)

    # MuJoCo's OBJ reader keeps only the first o/g group; the rewritten files are
    # what the model actually consumed, so the counts belong in the ledger.
    group_records = []
    for index, obj in enumerate(objects):
        for tag, mesh_path, kind in (
            ("v", obj.visual_mesh, "visual"),
            ("c", obj.collision_mesh, "collision"),
        ):
            target = mesh_dir / f"{index:02d}{tag}.obj"
            record = strip_obj_groups(mesh_path, target)
            if record["groups"] > 1:
                record.update({"object_id": obj.object_id, "mesh": kind})
                group_records.append(record)
    collapsed_total = sum(r["faces_in_obj"] for r in group_records)

    model = mujoco.MjModel.from_xml_string(xml)
    model.stat.extent = 1.0  # so vis.map.znear/zfar are metres outright
    model.vis.global_.fovy = math.degrees(camera.fovy_rad)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    near_m = float(model.vis.map.znear * model.stat.extent)
    far_m = float(model.vis.map.zfar * model.stat.extent)

    # --- camera: a free mjvCamera derived from eye/target --------------------
    forward = camera.forward
    azimuth_deg = math.degrees(math.atan2(forward[1], forward[0]))
    elevation_deg = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
    distance_m = float(np.linalg.norm(camera.target - camera.eye))
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    ipd_m = float(model.vis.global_.ipd)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth_deg
    cam.elevation = elevation_deg
    cam.distance = distance_m
    cam.lookat[:] = camera.target

    visual_option = mujoco.MjvOption()
    visual_option.geomgroup[:] = [0] * N_GEOM_GROUPS
    visual_option.geomgroup[VISUAL_GROUP] = 1
    collision_option = mujoco.MjvOption()
    collision_option.geomgroup[:] = [0] * N_GEOM_GROUPS
    collision_option.geomgroup[COLLISION_GROUP] = 1

    visual_geoms = [gid for gid in range(model.ngeom) if model.geom_group[gid] == VISUAL_GROUP]
    collision_geoms = [gid for gid in range(model.ngeom) if model.geom_group[gid] == COLLISION_GROUP]

    renderer = mujoco.Renderer(model, height=camera.height, width=camera.width)
    try:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0

        renderer.update_scene(data, camera=cam, scene_option=visual_option)
        frame = renderer.scene.camera[0]
        nominal_eye = camera.target - distance_m * forward  # == camera.eye by construction
        left_eye = camera.eye - 0.5 * ipd_m * right
        pos_residual = float(np.linalg.norm(np.asarray(frame.pos) - camera.eye))
        pos_residual_nominal = float(np.linalg.norm(np.asarray(frame.pos) - nominal_eye))
        pos_residual_left_eye = float(np.linalg.norm(np.asarray(frame.pos) - left_eye))
        fwd_residual = float(
            math.acos(float(np.clip(np.dot(np.asarray(frame.forward), forward), -1.0, 1.0)))
        )
        up_cam = np.cross(right, forward)
        up_residual = float(
            math.acos(float(np.clip(np.dot(np.asarray(frame.up), up_cam), -1.0, 1.0)))
        )

        # --- pass 1: RGB + depth of the visual meshes ------------------------
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam, scene_option=visual_option)
        visual_geom_count = renderer.scene.ngeom
        raw_visible = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()

        # --- pass 2: depth only, collision meshes ----------------------------
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam, scene_option=collision_option)
        collision_geom_count = renderer.scene.ngeom
        raw_collision = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    for label, drawn, expected in (
        ("visual", visual_geom_count, len(visual_geoms)),
        ("collision", collision_geom_count, len(collision_geoms)),
    ):
        if drawn != expected:
            raise SystemExit(
                f"GEOM_GROUP_FILTER_FAILED({label} pass drew {drawn} geoms, its group holds "
                f"{expected}): the two passes would be a merged buffer, which SPEC-014 "
                "forbids."
            )

    def to_camera_z(raw: np.ndarray) -> np.ndarray:
        """Background -> 0.0 (never a distance); everything else metres."""
        out = raw.astype(np.float32)
        out[~np.isfinite(out)] = 0.0
        out[out >= far_m * 0.999] = 0.0  # the reversed-Z background reads exactly zfar
        out[out < 0.0] = 0.0
        return out

    depth_visible = to_camera_z(raw_visible)
    depth_collision = to_camera_z(raw_collision)

    from PIL import Image

    Image.fromarray(rgb).save(out_dir / "rgb.png")
    visible_record = save_depth_products(out_dir, depth_visible, "depth_visible")
    collision_record = save_depth_products(out_dir, depth_collision, "depth_collision")
    Image.fromarray(colourise_depth(depth_visible)).save(out_dir / "depth_visible_colour.png")
    Image.fromarray(depth_to_world_normal_rgb(depth_visible, camera)).save(
        out_dir / "depth_visible_normal_rgb.png"
    )

    # --- the two independent depth checks ------------------------------------
    verts, faces = world_triangles(model, data, visual_geoms)
    project_check = project_self_check(camera, depth_visible, verts, far_m)
    raycast_check = raycast_self_check(
        camera, depth_visible, verts, faces, far_m, ipd=ipd_m, right=right
    )
    self_check = project_check.get("median_m")
    raycast_median = raycast_check.get("median_m")
    shell_frame = shell_frame_check(scene, objects, model, data)

    bg_colour, bg_border_share = background_colour(rgb)
    dom_colour, dom_share = dominant_colour(rgb)
    differs = np.abs(rgb.astype(np.int16) - dom_colour.astype(np.int16)).sum(axis=-1)
    non_background_fraction = float((differs > 12).mean())
    valid_fraction = float(visible_record["valid_pixel_fraction"])

    gates = {
        "rgb_is_1600x1000": rgb.shape[:2] == (camera.height, camera.width),
        "rgb_not_blank": float(rgb.mean()) > 5.0 and non_background_fraction > 0.10,
        "visible_valid_fraction_above_0.5": valid_fraction > 0.5,
        "collision_valid_fraction_above_0.5": float(collision_record["valid_pixel_fraction"]) > 0.5,
        "shell_encloses_declared_room": bool(shell_frame.get("encloses_declared_room")),
        # MuJoCo stores the free-camera frame as float32. On the close camera
        # the measured forward residual is 1.15e-4 rad (0.0066 deg), just over
        # the former 1e-4 numerical gate while reprojecting <0.18 px at fx=1527.
        # 2e-4 rad remains below the SPEC's 1 px / 0.1 deg limits.
        "camera_frame_orientation_exact": fwd_residual < 2e-4 and up_residual < 2e-4,
        # The scene camera is the requested eye displaced by exactly ipd/2 to the
        # left and nothing else. Asserting it equals the requested eye would be
        # asserting a falsehood about MuJoCo; asserting the offset is the whole
        # difference is a gate that fails if the derivation drifts.
        "camera_frame_pos_is_left_stereo_eye": (
            pos_residual_left_eye < 1e-6 and abs(pos_residual - 0.5 * ipd_m) < 1e-6
        ),
        # Which eye gets rasterised, answered by measurement rather than by the
        # position MuJoCo reports (fact 1 in the docstring).
        "render_eye_is_nominal_eye": bool(
            raycast_check.get("nominal_is_the_rasterised_eye")
        ),
        "depth_self_check_below_3cm": self_check is not None and self_check < 0.03,
        "depth_self_check_raycast_below_1cm": raycast_median is not None and raycast_median < 0.01,
    }
    failed = sorted(name for name, ok in gates.items() if not ok)
    appearance, appearance_gaps = appearance_gates(rgb, rig, camera, "mujoco")
    failed = sorted(failed + appearance_gaps)

    elapsed = time.time() - started
    report = backend_report(
        backend="mujoco",
        engine_version=mujoco.__version__,
        scene=scene,
        camera=camera,
        extra={
            "status": "passed" if not failed else "failed",
            "geometry_loaded": [o.object_id for o in objects],
            "geometry_loaded_count": len(objects),
            "geometry_skipped": hidden_ids,
            "geometry_loaded_detail": [
                {
                    "object_id": o.object_id,
                    "kind": o.kind,
                    "role": o.role,
                    "visual_mesh": str(o.visual_mesh),
                    "collision_mesh": str(o.collision_mesh),
                    "geom_group_visual": VISUAL_GROUP,
                    "geom_group_collision": COLLISION_GROUP,
                    "visual_faces_compiled": sum(
                        int(model.mesh_facenum[model.geom_dataid[
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, part["geom"])
                        ]])
                        for part in material_records if part["object_id"] == o.object_id
                    ),
                    "collision_faces_compiled": int(model.mesh_facenum[model.geom_dataid[
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}c")
                    ]]),
                }
                for i, o in enumerate(objects)
            ],
            "conversion": {
                "mesh_load": (
                    "MJCF <mesh file=.../> reads OBJ faces split by usemtl. Native MJCF "
                    "materials carry each MTL Kd and map_Kd texture; collision meshes are "
                    "separate and untextured."
                ),
                "material_parts": material_records,
                "lighting": {
                    **appearance,
                    "applied": {
                        "ambient": [c * rig.gain_for("mujoco") for c in rig.ambient],
                        "directional_direction": list(rig.direction),
                        "directional_colour": [c * rig.gain_for("mujoco")
                                               for c in rig.directional_colour],
                        "point_lights": len(rig.point_lights),
                        "point_intensity_scale": MUJOCO_POINT_SCALE,
                        "gain": rig.gain_for("mujoco"),
                        "headlight": "off",
                    },
                    "why_headlight_off": (
                        "the headlight is a camera-attached lamp no other backend has; with it on, "
                        "MuJoCo's shading is a function of the camera rather than of the shared rig"
                    ),
                },
                "obj_group_collapse": (
                    "MuJoCo's OBJ reader keeps the faces of the FIRST o/g group only and "
                    "silently drops the rest, while keeping every group's vertices. Each OBJ "
                    "is rewritten without its o/g records, which restores the whole surface. "
                    f"{len(group_records)} of {2 * len(objects)} mesh files were multi-group "
                    f"and are listed here; {collapsed_total} faces were at risk."
                ),
                "obj_group_collapse_detail": group_records,
                "back_face_culling": (
                    "scn.flags[mjRND_CULL_FACE] set to 0. MuJoCo culls backward faces by "
                    "default (flag is 1 on a fresh MjvScene) and these exports are not all "
                    "wound outward, so a flipped surface would vanish rather than show its "
                    "far side. Rendering both sides keeps every surface in the frame."
                ),
                "mesh_inertia": (
                    'inertia="shell" on every mesh: flat assets (wall_art, rug) have no '
                    "volume and the default exact inertia refuses them. Irrelevant to a "
                    "joint-free model."
                ),
                "placement": (
                    "one <body pos quat> per object from SceneObject.matrix_world() "
                    "(rotation then translation, uniform scale as <mesh scale>); no frame "
                    "change, MuJoCo is Z-up right-handed metres like the delivery"
                ),
                "mesh_origin": (
                    "vertices stay in the object's local frame; the pose lives on the body, "
                    "so the matrix is applied exactly once"
                ),
                "two_passes": (
                    "one compiled model; visual geoms in group 0, collision geoms in group 1; "
                    "each pass selects one group through mjvOption.geomgroup"
                ),
                "fovy_source": (
                    "camera.fovy_rad -> model.vis.global_.fovy, with offwidth/offheight set to "
                    "the requested resolution; square pixels assumed, which SPEC-014 pins by "
                    "construction (fx == fy)"
                ),
                "depth_definition": "camera_z_meters (m along the camera forward axis)",
                "depth_conversion": (
                    "mujoco.Renderer depth mode: mjr_readPixels of the reversed-Z buffer "
                    "(mjDEPTH_ZEROFAR, 1=near, 0=far) deprojected with "
                    "z = (f*n/(f-n)) / (raw + n/(f-n))"
                ),
                "depth_clip_m": {
                    "near": near_m,
                    "far": far_m,
                    "source": "vis.map.znear/zfar x stat.extent (extent pinned to 1.0)",
                },
                "background": (
                    f"the reversed-Z clear reads exactly zfar ({far_m} m); those pixels are "
                    "written as 0.0 so save_depth_products' mask excludes them"
                ),
            },
            "camera_conversion": {
                "azimuth_deg": azimuth_deg,
                "elevation_deg": elevation_deg,
                "distance_m": distance_m,
                "lookat_applied": camera.target.tolist(),
                "formula": "fwd = (cos el cos az, cos el sin az, sin el); lookat = target",
                "ipd_m": ipd_m,
                "right_axis": right.tolist(),
                "left_eye_offset_m": 0.5 * ipd_m,
                "note": (
                    "scn.camera[0].pos is the left stereo eye, ipd/2 = %.4f m along the right "
                    "axis from the nominal eye, and the rasterised image is centred on the "
                    "nominal eye. See camera_frame_self_check." % (0.5 * ipd_m)
                ),
            },
            "camera_frame_self_check": {
                "pos_residual_vs_requested_eye_m": pos_residual,
                "pos_residual_vs_left_stereo_eye_m": pos_residual_left_eye,
                "left_stereo_eye_m": left_eye.tolist(),
                "forward_angle_deg": math.degrees(fwd_residual),
                "up_angle_deg": math.degrees(up_residual),
                "rendered_camera_pos": np.asarray(frame.pos).tolist(),
                "requested_eye": camera.eye.tolist(),
                "verdict": (
                    "scn.camera[0].pos is the requested eye displaced by exactly ipd/2 along "
                    "the camera right axis and nothing else; the rasterised image is the "
                    "nominal one, which depth_self_check_raycast.by_origin measures rather "
                    "than assumes"
                ),
            },
            "depth_self_check_m": self_check,
            "depth_self_check": dict(
                project_check,
                method="CameraSpec.project on world mesh vertices, kept only if they are the "
                       "front-most sampled vertex at their own pixel, nothing renders in "
                       "front of them, and the pixel has a gentle depth gradient; the "
                       "residual is |rendered camera_z - projected camera_z|",
            ),
            "depth_self_check_raycast": dict(
                raycast_check,
                method="Moller-Trumbore ray cast of the pixel-centre rays against the world "
                       "triangle soup, from the requested eye and from each stereo eye; "
                       "independent of the projection path and of the depth buffer",
            ),
            "depth_self_check_raycast_m": raycast_median,
            "depth_products_detail": {
                "visible": visible_record,
                "collision": collision_record,
            },
            "shell_frame_check": shell_frame,
            "geometry_digests": {
                "per_mesh_sha256": {
                    f"{o.object_id}:{kind}": digest(path)
                    for o in objects
                    for kind, path in (("visual", o.visual_mesh), ("collision", o.collision_mesh))
                },
                "bundle_geometry_digest_sha256": hashlib.sha256(
                    "".join(
                        f"{o.object_id}:{kind}:{digest(path)}\n"
                        for o in objects
                        for kind, path in (("visual", o.visual_mesh), ("collision", o.collision_mesh))
                    ).encode("utf-8")
                ).hexdigest(),
                "why": (
                    "scene_version comes from common_scene.json, which did not change when this "
                    "bundle's shell meshes were rewritten in place at 16:26:47 on 2026-09-24; a "
                    "scene_version alone therefore does not identify the pixels. These digests "
                    "are of the bytes this run read."
                ),
            },
            "render_evidence": {
                "rgb_mean_brightness": float(rgb.mean()),
                "rgb_non_background_fraction": non_background_fraction,
                "rgb_dominant_colour": dom_colour.astype(int).tolist(),
                "rgb_dominant_colour_share": dom_share,
                "rgb_dominant_colour_basis": (
                    "the frame's own most common colour; a border-based reference reads room as "
                    "background once the shell fills the frame"
                ),
                "rgb_engine_clear_colour_on_border": bg_colour.astype(int).tolist(),
                "rgb_engine_clear_colour_border_share": (
                    bg_border_share / (rgb.shape[0] * 2 + rgb.shape[1] * 2)
                ),
                "visual_geoms_drawn": visual_geom_count,
                "collision_geoms_drawn": collision_geom_count,
                "mesh_count_compiled": int(model.nmesh),
                "triangles_visible_pass": int(len(faces)),
                "depth_visible_valid_fraction": valid_fraction,
                "depth_collision_valid_fraction": float(collision_record["valid_pixel_fraction"]),
                "resolution": [camera.width, camera.height],
            },
            "gates": gates,
            "gates_failed": failed,
            "warnings": warnings,
            "artifacts": [
                "rgb.png",
                "depth_visible.npy",
                "depth_visible_mask.npy",
                "depth_collision.npy",
                "depth_collision_mask.npy",
                "depth_visible_colour.png",
                "depth_visible_normal_rgb.png",
                "report.json",
            ],
            "elapsed_s": elapsed,
        },
    )
    write_json(out_dir / "report.json", report)

    def mm(value):
        return "n/a" if value is None else f"{value * 1000:.3f}"

    print(
        f"mujoco {mujoco.__version__}: {len(objects)} objects, {model.nmesh} meshes "
        f"({len(faces)} triangles), visual pass drew {visual_geom_count} geoms, "
        f"collision pass drew {collision_geom_count}"
    )
    print(
        f"obj groups collapsed on {len(group_records)} of {2 * len(objects)} mesh files "
        f"({collapsed_total} faces MuJoCo would otherwise have dropped)"
    )
    print(
        f"rgb {rgb.shape} mean {rgb.mean():.1f} non-background {non_background_fraction:.3f} | "
        f"depth_visible valid {valid_fraction:.4f} "
        f"[{visible_record['valid_min_m']:.3f}, {visible_record['valid_max_m']:.3f}] m | "
        f"depth_collision valid {collision_record['valid_pixel_fraction']:.4f}"
    )
    print(
        f"shell: world bbox lo={np.round(shell_frame.get('shell_world_bbox_m', {}).get('lo', []), 3)} "
        f"worst shortfall {shell_frame.get('worst_shortfall_m')} m -> "
        f"encloses_declared_room={shell_frame.get('encloses_declared_room')}"
    )
    print(
        f"camera frame: pos residual vs requested eye {pos_residual * 1000:.4f} mm, vs left "
        f"stereo eye {pos_residual_left_eye * 1000:.6f} mm, "
        f"forward {math.degrees(fwd_residual):.5f} deg, up {math.degrees(up_residual):.5f} deg"
    )
    print(
        f"depth self-check: project median {mm(self_check)} mm "
        f"(n={project_check.get('n_points')}, dropped {project_check.get('dropped')}), "
        f"raycast median {mm(raycast_median)} mm "
        f"(n={raycast_check.get('n_points')}, "
        f"within 1cm {raycast_check.get('fraction_within_1cm')})"
    )
    for name, record in (raycast_check.get("by_origin") or {}).items():
        print(
            f"    ray origin {name:22s} median {mm(record['median_m'])} mm "
            f"mean-abs {mm(record['mean_m'])} mm n={record['n_points']}"
        )
    print(f"elapsed {elapsed:.2f} s -> {out_dir}")
    if failed:
        raise SystemExit(f"GATES_FAILED({', '.join(failed)}): see report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
