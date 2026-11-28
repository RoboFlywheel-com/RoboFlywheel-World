#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SPEC-014 backend driver: SAPIEN (raytraced camera, real materials).

Consumes the frozen ``common_scene.json`` bundle and emits the five products
``common.py`` defines: the visible geometry's RGB, the *visual* meshes' camera-z
depth, the *collision* meshes' camera-z depth in separate files, and a report.

What is SAPIEN-specific here, and why the code looks like this
-------------------------------------------------------------

1. **Camera frame.** SAPIEN's camera principal axis is not its entity's ``-Z``.
   A mounted camera looks along its mount's local ``+X`` with the mount's local
   ``+Z`` up (this is stated in SAPIEN's own ``example/hello_world.py``, and it
   is what the probe recorded in ``report.json`` under ``camera_convention``
   measures). ``common.py`` hands out a GL-convention ``(w,x,y,z)`` world
   orientation, so the mount's world rotation is rebuilt here from
   ``camera.forward``: columns ``[forward, up x forward, up]``.

2. **Depth is the sensor's ``Position`` picture, not a range image.** SAPIEN's
   camera publishes the hit point's *camera-space* position; ``Position[...,2]``
   is the true camera-space ``z`` in metres, and it is **negative in front**
   (SAPIEN's camera space is the OpenGL one: ``-Z`` forward, ``+Y`` up -- the
   deferred shader itself does ``camDir = -normalize(csPosition.xyz)``). So
   ``camera_z_meters = -Position[..., 2]``. This is not guessed: the shader
   source is quoted in the report and the residual is measured by projecting a
   known object with ``CameraSpec.project`` and reading the sensor back
   (``depth_self_check_m``).

   The rasterisation shader (``default``) additionally exposes ``DepthLinear``,
   whose ``.x`` is the same ``position.z``; the raytracing shader (``rt``) does
   not, which is one more reason the driver reads ``Position``.

3. **Shader mode.** ``rt`` -- SAPIEN's differentiator is its physically based
   raytracer, and the figure is meant to show that. It is set once, globally,
   before the scene exists, and every pass in this process uses it.

4. **Two passes, one scene.** The visual and collision meshes are both built,
   each on its own entity, and the pass is selected by toggling the render
   body's ``visibility``. Nothing is rebuilt between passes, so the two depth
   images differ by geometry alone.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.workers.multisim.render_sapien \\
      --scene outputs/_paper/spec014_common_scene_20260924_01 \\
      --output-dir outputs/_paper/spec014_renders_20260924_01/sapien
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from rfw_w.scenes.workers.multisim.common import (
    HIDDEN_SHELL_ROLES,
    CommonScene,
    appearance_gates,
    backend_report,
    digest,
    load_light_rig,
    resolves_to_no_material,
    save_depth_products,
    write_json,
)

BACKEND = "sapien"

# SAPIEN's raytracer needs a real TLAS and a real GPU; give it a sane budget.
# 4 spp left the frame visibly salt-and-peppered at 1600x1000 (measured: 4 spp
# rendered the room in 7.0 s end to end), so 16 spp plus the Open Image Denoise
# pass SAPIEN ships with is what this driver commits to.
# 8 spp with the denoiser off is what actually shipped in the 20260925_03 assets
# and it left the frame visibly salt-and-peppered: 50.893 mean |Laplacian| on the
# bare-desk calibration patch, against 1.1-5.0 for the four rasterisers.
#
# The obvious fix -- turn the denoiser on and keep the sample count low -- does
# not work on this host. Measured 2026-09-25: with DENOISER="oidn" the process
# reaches `cam.get_picture("Color")` and hangs there indefinitely at 0% GPU
# utilisation with its buffers still allocated (killed after 22 minutes; the
# same scene at 8 spp undenoised completes in 8.3 s). That is the risk the
# --denoiser flag's help text already warned about, now observed rather than
# anticipated. So this driver buys convergence with samples instead, which is
# slower but has no failure mode: path-trace noise falls as 1/sqrt(spp), so 512
# spp predicts 50.893/sqrt(64) ~= 6.4 on the same patch.
SPP = 512


# Near/far for every figure camera. The room's diagonal is ~6.1 m, so a 0.05 m
# near plane and a 100 m far plane cannot clip the room; they only decide where
# the sensor stops reporting geometry.
NEAR_M = 0.05
FAR_M = 100.0

# Lighting. SAPIEN's raytracer traces shadow rays, and the room has a ceiling,
# so a light placed outside the room contributes nothing. `lighting` in the
# report records exactly what was placed inside it.
# Superseded by the shared rig in experiments/paper_multisim/lighting.json, which
# all five backends now read. Kept as the documented origin of those values.
AMBIENT = (0.22, 0.22, 0.24)
POINT_LIGHTS = (
    # (position, colour) -- two ceiling-height lights inside the room footprint.
    # 6.0 / 3.0 sat the white walls at ~1.0 in the tonemapped frame (measured
    # 2026-09-24: the wall behind the ceiling light and the whole left wall came
    # out flat white, losing the shading that shows the room is a room). 3.2 /
    # 1.7 keeps the interior readable without dimming the furniture.
    ((2.4528, 1.84315, 2.55), (3.2, 3.15, 3.0)),
    ((1.10, 0.90, 2.55), (1.7, 1.75, 1.9)),
)
DIRECTIONAL = ((-0.55, 0.35, -0.76), (0.9, 0.88, 0.82))

# One shader mode for the whole process: `set_camera_shader_dir` is global, so a
# per-pass choice would be a lie about which renderer produced which file.
SHADER_MODE = "rt"
# "none", not "oidn": see the SPP comment above -- OIDN deadlocks on this host.
# Whatever this run actually used is written into the report, so a reader never
# has to infer it from a default.
DENOISER = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", required=True, type=Path,
                        help="directory holding common_scene.json + meshes/")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lighting", type=Path,
                        default=data_path('examples')
                        / "paper_multisim" / "lighting.json",
                        help="the shared five-backend light rig (SPEC-014 R5)")
    parser.add_argument("--spp", type=int, default=SPP,
                        help="raytracing samples per pixel (default %(default)s)")
    parser.add_argument("--denoiser", choices=("none", "oidn", "optix"),
                        default=DENOISER,
                        help="raytracing denoiser; 'none' avoids OIDN GPU errors on shared hosts")
    return parser.parse_args()


def camera_mount_matrix(camera) -> np.ndarray:
    """World matrix for the entity the camera is mounted on.

    SAPIEN's mounted camera looks along the mount's local ``+X`` and takes the
    mount's local ``+Z`` as up. So the mount's world rotation has
    ``[forward, up x forward, up]`` as its columns; the camera's own local pose
    stays identity. The check ``camera_forward_residual`` in the report
    re-derives the forward axis from SAPIEN's own OpenCV-format extrinsic and
    compares it with ``camera.forward``.
    """
    forward = camera.forward
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    norm = np.linalg.norm(right)
    if norm < 1e-9:  # camera looking straight down or up; pick any right axis
        right = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    right = right / norm
    up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = forward
    matrix[:3, 1] = np.cross(up, forward)
    matrix[:3, 2] = up
    matrix[:3, 3] = camera.eye
    return matrix


def quaternion_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Rotation part of a 4x4 -> quaternion, ``w`` first, the order SAPIEN's
    ``Pose`` takes."""
    m = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def build_scene(scene_objects, sapien, base_pose, shell_albedo=None):
    """One entity per mesh, tagged with the pass it belongs to.

    Returns ``(records, visual_bodies, collision_bodies)``. A mesh that SAPIEN
    refuses to load is *named* (``SAPIEN_MESH_UNSUPPORTED``) rather than dropped
    quietly, and the caller decides whether that is fatal.
    """
    records: list[dict] = []
    visual_bodies: list = []
    collision_bodies: list = []

    for obj in scene_objects:
        for pass_name, mesh in (("visual", obj.visual_mesh),
                                ("collision", obj.collision_mesh)):
            record = {
                "object_id": obj.object_id,
                "kind": obj.kind,
                "role": obj.role,
                "pass": pass_name,
                "mesh": str(mesh),
            }
            if not mesh.is_file():
                record["status"] = f"MULTISIM_GEOMETRY_MISMATCH({obj.object_id}:{pass_name})"
                records.append(record)
                continue
            try:
                shape = sapien.render.RenderShapeTriangleMesh(
                    str(mesh),
                    scale=np.array([obj.scale, obj.scale, obj.scale], dtype=np.float32),
                )
            except Exception as exc:  # noqa: BLE001 -- the engine's own refusal
                record["status"] = f"SAPIEN_MESH_UNSUPPORTED({type(exc).__name__}: {exc})"
                records.append(record)
                continue

            parts = shape.parts
            record["parts"] = len(parts)
            record["triangles"] = int(sum(len(np.asarray(p.triangles)) for p in parts))
            record["vertices"] = int(sum(len(np.asarray(p.vertices)) for p in parts))
            record["textured_parts"] = int(sum(
                1 for p in parts
                if p.material is not None and p.material.get_base_color_texture() is not None
            ))

            # Same rule as the other four: a mesh with no material library at
            # all takes the rig's declared shell albedo rather than SAPIEN's
            # white default. Parts that DO carry a material or a texture are
            # left exactly as the asset authored them.
            if shell_albedo is not None and resolves_to_no_material(mesh):
                for part in parts:
                    if part.material is not None and \
                            part.material.get_base_color_texture() is None:
                        part.material.set_base_color(
                            [*shell_albedo, 1.0])
                record["shell_albedo_applied"] = list(shell_albedo)

            body = sapien.render.RenderBodyComponent()
            body.attach(shape)
            entity = sapien.Entity()
            entity.set_name(f"{pass_name}:{obj.object_id}")
            entity.add_component(body)
            pose = np.asarray(obj.matrix_world(), dtype=np.float64)
            entity.set_pose(sapien.Pose(
                p=pose[:3, 3].astype(np.float32),
                q=quaternion_wxyz_from_matrix(pose).astype(np.float32),
            ))
            base_pose.add_entity(entity)
            record["status"] = "loaded"
            records.append(record)
            (visual_bodies if pass_name == "visual" else collision_bodies).append(body)
    return records, visual_bodies, collision_bodies


def set_pass(visual_bodies, collision_bodies, want: str) -> None:
    """Show exactly one pass. ``visibility`` is the render body's own property,
    so this is a render-state change and not a rebuild."""
    for body in visual_bodies:
        body.visibility = 1.0 if want == "visual" else 0.0
    for body in collision_bodies:
        body.visibility = 1.0 if want == "collision" else 0.0


def picture_to_rgb(picture: np.ndarray) -> np.ndarray:
    """SAPIEN's ``Color`` picture is already display-referred (the composite
    shader applies ``pow(x, 1/2.2)`` / a tonemap); clip and quantise only."""
    rgb = np.asarray(picture)[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        # SAPIEN's picture buffer is vertically flipped relative to the pixel
        # coordinates `CameraSpec.project` returns... it is not: `take_picture`
        # hands back row 0 = top. Guard only against NaN.
        rgb = np.nan_to_num(rgb, nan=0.0)
        return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return rgb.astype(np.uint8)


def depth_from_picture(position_picture: np.ndarray) -> np.ndarray:
    """``Position[...,2]`` is camera-space z, negative in front (GL convention).
    Negate it and mark everything that is not a real hit invalid -- never zero."""
    z = np.asarray(position_picture)[..., 2].astype(np.float64)
    depth = -z
    depth[~np.isfinite(depth)] = np.nan
    depth[depth <= 0] = np.nan
    return depth.astype(np.float32)


def pixel_of(camera, point_world: np.ndarray) -> tuple[tuple[float, float], float]:
    """``CameraSpec.project`` on one point. It returns ``((n,2) pixels, (n,) z)``,
    so the single-point unwrap lives here once instead of at each call site."""
    uv, z = camera.project(np.asarray(point_world, dtype=np.float64)[None, :])
    return (float(uv[0, 0]), float(uv[0, 1])), float(z[0])


def nominate_anchor(scene_objects, camera, depth_visible):
    """Pick the self-check anchor by a geometric rule, not by best fit.

    Rule: among the furniture objects whose world origin projects inside the
    frame, take the one whose origin is nearest to the camera along the forward
    axis. The rule never looks at the sensor's answer, so the residual it
    produces is evidence rather than a selection.
    """
    best = None
    for obj in scene_objects:
        if obj.kind != "furniture":
            continue
        origin = obj.matrix_world()[:3, 3]
        (u, v), z = pixel_of(camera, origin)
        if not (0 <= u < camera.width and 0 <= v < camera.height):
            continue
        if not np.isfinite(z) or z <= 0:
            continue
        if best is None or z < best["camera_z_true_m"]:
            best = {
                "object_id": obj.object_id,
                "role": obj.role,
                "world_position_m": [float(x) for x in origin],
                "pixel": [u, v],
                "camera_z_true_m": z,
            }
    return best


def sample_sensor(camera, depth_visible, point_world):
    """Project one world point and read the sensor back at that pixel."""
    (u, v), z = pixel_of(camera, point_world)
    if not (0 <= u < camera.width and 0 <= v < camera.height) or not np.isfinite(z) or z <= 0:
        return {"pixel": [u, v], "camera_z_true_m": z, "status": "outside_frame"}
    sampled = float(depth_visible[int(v), int(u)])
    if not np.isfinite(sampled):
        return {"pixel": [u, v], "camera_z_true_m": z, "status": "no_sensor_return"}
    return {
        "pixel": [u, v],
        "camera_z_true_m": z,
        "depth_sensor_m": sampled,
        "residual_m": abs(sampled - z),
        "status": "compared",
    }


def read_obj_vertices(path: Path) -> np.ndarray:
    """Just the ``v`` lines. The self-check needs surface points, and a mesh
    library would be a dependency the rest of this driver does not have."""
    points = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def self_check(camera, depth_visible, anchor, scene_objects):
    """Two residual families, because they answer different questions.

    *On the object's world origin* (``origin_table``): exactly what the SPEC
    asks for, and it is reported in full -- every furniture object, occluded or
    not. It is a *lower bound on usefulness*, not on the depth convention: an
    object's pivot is generally not on its visible surface (the office's wall art
    pivots 2 cm behind the canvas it hangs on), so a residual of a few
    centimetres here says something about the pivot, not about the sensor.

    *On a point that is provably on the surface* (``anchor_vertex``): the nearest
    mesh vertex of the anchor object, again projected with ``CameraSpec.project``
    and read back from the sensor. Camera-z over a triangle is affine in the
    barycentric coordinates, so its minimum over a mesh is attained at a vertex;
    if that vertex is unoccluded the sensor must return exactly its depth. This
    is the check that can catch a wrong sign or a range-vs-axial confusion, and
    it needs no threshold to be meaningful.
    """
    if anchor is None:
        return {"residual_m": None, "origin_table": [], "anchor_vertex": None}

    origin_table = []
    for obj in scene_objects:
        if obj.kind != "furniture":
            continue
        row = {"object_id": obj.object_id}
        row.update(sample_sensor(camera, depth_visible, obj.matrix_world()[:3, 3]))
        origin_table.append(row)

    anchor_obj = next(o for o in scene_objects if o.object_id == anchor["object_id"])
    local = read_obj_vertices(anchor_obj.visual_mesh)
    matrix = anchor_obj.matrix_world()
    world = local @ matrix[:3, :3].T + matrix[:3, 3]
    uv, z = camera.project(world)
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
        & np.isfinite(z) & (z > 0)
    )
    if not inside.any():
        return {"residual_m": None, "origin_table": origin_table, "anchor_vertex": None}

    nearest = int(np.argmin(np.where(inside, z, np.inf)))
    vertex = {
        "object_id": anchor_obj.object_id,
        "mesh": str(anchor_obj.visual_mesh),
        "vertex_index": nearest,
        "world_position_m": [float(x) for x in world[nearest]],
        "vertices_in_frame": int(inside.sum()),
        "vertices_total": int(len(local)),
    }
    vertex.update(sample_sensor(camera, depth_visible, world[nearest]))

    # The whole vertex cloud, so the anchor is not a single lucky sample. Only
    # the vertices whose pixel the sensor reports are counted; a back-face vertex
    # legitimately reads the front face instead, which is why the median rather
    # than the minimum is the number worth quoting.
    residuals = []
    for index in np.nonzero(inside)[0]:
        (u, v), true_z = pixel_of(camera, world[index])
        sampled = float(depth_visible[int(v), int(u)])
        if np.isfinite(sampled):
            residuals.append(abs(sampled - true_z))
    if residuals:
        values = np.asarray(residuals, dtype=np.float64)
        vertex["vertex_residuals_m"] = {
            "count": int(values.size),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "fraction_within_1cm": float((values < 0.01).mean()),
        }
    return {
        "residual_m": vertex.get("residual_m"),
        "origin_table": origin_table,
        "anchor_vertex": vertex,
    }


# The room's declared height includes the ceiling slab (declared 2.9066 m, the
# ceiling's underside measures 2.8566 m), and the floor's top face sits 5 cm
# above `floor_z_m`; 0.10 m covers both and nothing else.
ROOM_COVER_TOLERANCE_M = 0.10
ENCLOSURE_ROLES = ("wall", "floor", "ceiling")


def declared_room_box(scene) -> dict:
    """The room the bundle *declares*, from its own ``room`` record. Not a
    measurement of the meshes -- the point of the check below is to compare the
    two."""
    room = scene.payload.get("room") or {}
    polygon = np.asarray(room.get("polygon_xy", []), dtype=np.float64)
    if polygon.size == 0:
        return {}
    floor_z = float(room.get("floor_z_m", 0.0))
    height = float(room.get("height_m", 0.0))
    return {
        "x_m": [float(polygon[:, 0].min()), float(polygon[:, 0].max())],
        "y_m": [float(polygon[:, 1].min()), float(polygon[:, 1].max())],
        "z_m": [floor_z, floor_z + height],
        "polygon_xy": polygon.tolist(),
        "height_m": height,
        "floor_z_m": floor_z,
    }


def room_frame_check(scene_objects, scene) -> dict:
    """Is the shell geometry where the bundle says the room is?

    A backend can only place a mesh where the bundle's ``matrix_world()`` puts
    it, so this gate asks the one question a driver can be sure of: does the room
    the bundle *declares* fit inside the enclosure the bundle *supplies*? It is
    checked separately from the depth self-check because the two fail for
    different reasons -- a wrong depth convention moves no geometry, and a shell
    in the wrong frame leaves the convention perfect while putting the walls
    somewhere else. Measured both ways on 2026-09-24; see the report note.
    """
    room = declared_room_box(scene)
    if not room:
        return {"gap": "MULTISIM_ROOM_UNSUPPORTED(sapien, bundle declares no polygon_xy)"}
    hidden_excluded = list(scene.hidden_rule().get("roles", HIDDEN_SHELL_ROLES))

    per_object = []
    enclosure_lo = np.full(3, np.inf)
    enclosure_hi = np.full(3, -np.inf)
    room_centre = np.array(
        [
            (room["x_m"][0] + room["x_m"][1]) / 2.0,
            (room["y_m"][0] + room["y_m"][1]) / 2.0,
            (room["z_m"][0] + room["z_m"][1]) / 2.0,
        ]
    )
    for obj in scene_objects:
        if obj.kind != "shell":
            continue
        local = read_obj_vertices(obj.visual_mesh)
        if local.size == 0:
            continue
        matrix = obj.matrix_world()
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        lo, hi = world.min(axis=0), world.max(axis=0)
        if obj.role in ENCLOSURE_ROLES:
            enclosure_lo = np.minimum(enclosure_lo, lo)
            enclosure_hi = np.maximum(enclosure_hi, hi)
        per_object.append(
            {
                "object_id": obj.object_id,
                "role": obj.role,
                "hidden": bool(obj.role in HIDDEN_SHELL_ROLES),
                "world_bbox_lo_m": [float(v) for v in lo],
                "world_bbox_hi_m": [float(v) for v in hi],
                "bbox_centre_offset_from_room_centre_m": [
                    float(v) for v in ((lo + hi) / 2.0 - room_centre)
                ],
            }
        )

    check = {
        "declared_room_box_m": {k: room[k] for k in ("x_m", "y_m", "z_m")},
        "hidden_shell_roles_excluded": hidden_excluded,
        "enclosure_roles": list(ENCLOSURE_ROLES),
        "enclosure_union_m": {
            "lo": [float(v) for v in enclosure_lo],
            "hi": [float(v) for v in enclosure_hi],
        },
        "cover_tolerance_m": ROOM_COVER_TOLERANCE_M,
        "per_object": per_object,
    }
    if not np.isfinite(enclosure_lo).all():
        check["room_box_covered"] = False
        check["gap"] = "MULTISIM_ROOM_UNSUPPORTED(sapien, bundle supplies no wall/floor/ceiling mesh)"
        return check

    # Coverage is `room subset of enclosure`: the enclosure may be larger than
    # the room (it does not have to be flush), it may not be smaller. So each
    # axis contributes a shortfall only where the enclosure ends inside the room.
    # The first version of this had the two signs swapped, which made it a test
    # that could not fail on the one bundle it exists for -- measured 2026-09-24.
    shortfall = np.maximum(
        np.array(
            [
                enclosure_lo[0] - room["x_m"][0],
                enclosure_lo[1] - room["y_m"][0],
                enclosure_lo[2] - room["z_m"][0],
                room["x_m"][1] - enclosure_hi[0],
                room["y_m"][1] - enclosure_hi[1],
                room["z_m"][1] - enclosure_hi[2],
            ]
        ),
        0.0,
    )
    check["cover_shortfall_m"] = {
        "x_lo": float(shortfall[0]),
        "y_lo": float(shortfall[1]),
        "z_lo": float(shortfall[2]),
        "x_hi": float(shortfall[3]),
        "y_hi": float(shortfall[4]),
        "z_hi": float(shortfall[5]),
        "max": float(shortfall.max()),
    }
    covered = bool(shortfall.max() <= ROOM_COVER_TOLERANCE_M)
    check["room_box_covered"] = covered
    if not covered:
        check["gap"] = (
            "MULTISIM_ROOM_UNSUPPORTED(sapien, the shell's wall/floor/ceiling "
            "do not cover the room the bundle declares: enclosure covers "
            f"x[{enclosure_lo[0]:.4f},{enclosure_hi[0]:.4f}] "
            f"y[{enclosure_lo[1]:.4f},{enclosure_hi[1]:.4f}] "
            f"z[{enclosure_lo[2]:.4f},{enclosure_hi[2]:.4f}] but the room is "
            f"x{room['x_m']} y{room['y_m']} z{room['z_m']}; worst shortfall "
            f"{shortfall.max():.4f} m)"
        )
    return check


def main() -> int:
    args = parse_args()
    scene_dir = args.scene.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already has files; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    import sapien  # imported here so --help works without a GPU
    import sapien.render

    # Global renderer configuration, before anything renderer-shaped exists.
    sapien.render.set_log_level("error")
    sapien.render.set_camera_shader_dir(SHADER_MODE)
    sapien.render.set_ray_tracing_samples_per_pixel(args.spp)
    sapien.render.set_ray_tracing_path_depth(4)
    sapien.render.set_ray_tracing_denoiser(args.denoiser)

    scene = CommonScene.load(scene_dir)
    light_rig = load_light_rig(args.lighting)
    camera = scene.camera
    objects = scene.objects()
    warnings: list[str] = []

    base = sapien.Scene()
    base.set_timestep(1.0 / 100.0)
    gain = light_rig.gain_for("sapien")
    base.set_ambient_light([c * gain for c in light_rig.ambient])
    for point in light_rig.point_lights:
        base.add_point_light(list(point["position"]), [c * gain for c in point["colour"]])
    base.add_directional_light(list(light_rig.direction),
                               [c * gain for c in light_rig.directional_colour])

    geometry, visual_bodies, collision_bodies = build_scene(
        objects, sapien, base, shell_albedo=light_rig.shell_albedo)
    missing = [r for r in geometry if r["status"] != "loaded"]
    if missing:
        warnings.append(
            "MULTISIM_GEOMETRY_INCOMPLETE("
            + ",".join(f"{r['object_id']}:{r['pass']}" for r in missing) + ")"
        )

    # --- camera rig -------------------------------------------------------
    mount_matrix = camera_mount_matrix(camera)
    rig = sapien.Entity()
    rig.set_name("figure_camera_rig")
    rig.set_pose(sapien.Pose(
        p=mount_matrix[:3, 3].astype(np.float32),
        q=quaternion_wxyz_from_matrix(mount_matrix).astype(np.float32),
    ))
    base.add_entity(rig)
    cam = base.add_mounted_camera(
        camera.name, rig, sapien.Pose(), camera.width, camera.height,
        camera.fovy_rad, NEAR_M, FAR_M,
    )
    # Pin the intrinsics to the delivery's own record rather than trusting the
    # fovy argument to reproduce them: the figure turns on all five backends
    # sharing one K.
    cam.set_perspective_parameters(
        NEAR_M, FAR_M, camera.fx, camera.fy, camera.cx, camera.cy, 0.0
    )

    extrinsic = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)
    forward_residual = float(np.linalg.norm(extrinsic[2, :3] - camera.forward))
    intrinsic = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
    if forward_residual > 1e-3:
        warnings.append(f"SAPIEN_CAMERA_FORWARD_MISMATCH({forward_residual:.6f} m)")

    # --- pass 1: visual meshes, RGB + depth -------------------------------
    set_pass(visual_bodies, collision_bodies, "visual")
    base.update_render()
    cam.take_picture()
    if "Position" not in cam.get_picture_names():
        raise SystemExit(
            f"MULTISIM_BACKEND_UNAVAILABLE(sapien, shader {SHADER_MODE!r} published "
            f"{sorted(cam.get_picture_names())} and no 'Position')"
        )
    pictures = sorted(cam.get_picture_names())
    rgb = picture_to_rgb(cam.get_picture("Color"))
    if rgb.shape[0] != camera.height or rgb.shape[1] != camera.width:
        rgb = rgb[: camera.height, : camera.width]
    depth_visible = depth_from_picture(cam.get_picture("Position"))

    # --- pass 2: collision meshes, depth only -----------------------------
    set_pass(visual_bodies, collision_bodies, "collision")
    base.update_render()
    cam.take_picture()
    depth_collision = depth_from_picture(cam.get_picture("Position"))

    # --- products ---------------------------------------------------------
    from PIL import Image  # noqa: PLC0415 -- only needed once we have pixels

    Image.fromarray(rgb).save(out / "rgb.png")
    visible_detail = save_depth_products(out, depth_visible, "depth_visible")
    collision_detail = save_depth_products(out, depth_collision, "depth_collision")

    anchor = nominate_anchor(objects, camera, depth_visible)
    check = self_check(camera, depth_visible, anchor, objects)
    residual = check["residual_m"]
    if residual is None or residual > 0.01:
        warnings.append(f"SAPIEN_DEPTH_CONVENTION_SUSPECT(residual={residual} m)")
    if visible_detail["valid_pixel_fraction"] <= 0.5:
        warnings.append(
            f"SAPIEN_COVERAGE_LOW(valid={visible_detail['valid_pixel_fraction']:.4f})"
        )

    # Is the room where the bundle says it is? Separate gate, separate failure.
    room_check = room_frame_check(objects, scene)
    if room_check.get("gap"):
        warnings.append(room_check["gap"])

    appearance, appearance_gaps = appearance_gates(rgb, light_rig, camera, BACKEND)

    report = backend_report(
        backend=BACKEND,
        engine_version=sapien.__version__,
        scene=scene,
        camera=camera,
        extra={
            "geometry_loaded": geometry,
            "geometry_counts": {
                "objects": len(objects),
                "built": len(geometry),
                "loaded": len(geometry) - len(missing),
                "unsupported": len(missing),
            },
            "conversion": {
                "world_up_axis": "Z",
                "meters_per_unit": 1.0,
                "mesh_axis_handling": (
                    "none -- SAPIEN's OBJ loader keeps the file's coordinates; a "
                    "direct comparison of the loaded part vertices against the "
                    "file's own vertices is what established this"
                ),
                "pose": (
                    "matrix_world() applied as an entity pose; scale is folded into "
                    "the render shape, not the pose, because a SAPIEN Pose holds a "
                    "rotation and not a scale"
                ),
                "camera_pose": (
                    "mount world rotation columns [forward, up x forward, up] with the "
                    "camera's own local pose identity; SAPIEN's mounted camera looks "
                    "along its mount's local +X with +Z up"
                ),
            },
            "warnings": warnings,
            "room_frame": room_check,
            "shader_mode": SHADER_MODE,
            "shader_mode_why": (
                "SAPIEN's differentiator is its physically based raytracer with the "
                "asset's real materials; one mode is used for every pass in this "
                "process because set_camera_shader_dir is global"
            ),
            "samples_per_pixel": args.spp,
            "denoiser": args.denoiser,
            "picture_names": pictures,
            "depth_convention_found": {
                "sensor_picture": "Position",
                "raw_channel": "Position[..., 2] = camera-space z, metres",
                "sign": "negative in front of the camera (OpenGL camera space)",
                "conversion": "depth = -Position[..., 2]",
                "is_axial_not_range": (
                    "yes -- the value is the hit point's z coordinate, not the ray "
                    "length; a plane at a fixed axial distance returns one value "
                    "across the frame"
                ),
                "evidence": (
                    "graphics/vulkan_shader/default/gbuffer.vert writes "
                    "'outPositionRaw = cameraBuffer.viewMatrix * vec4(position, 1)' "
                    "and default/deferred.frag then does "
                    "'camDir = -normalize(csPosition.xyz)'; rt/camera.rgen writes "
                    "'vec4 position = cameraBuffer.viewMatrix * vec4(ray.origin, 1.0)' "
                    "into outPosition. Measured against a box face 1.8 m in front of "
                    "the camera, Position returned z = -1.800."
                ),
                "depth_picture_alternative": (
                    "the 'default' (rasterisation) shader also publishes DepthLinear, "
                    "whose .x is the same position.z; the 'rt' shader does not, so "
                    "this driver reads Position and works in either mode"
                ),
            },
            "depth_products_detail": {
                "visible": visible_detail,
                "collision": collision_detail,
            },
            "depth_self_check_m": residual,
            "depth_self_check_anchor": anchor,
            "depth_self_check_anchor_vertex": check["anchor_vertex"],
            "depth_self_check_origins": check["origin_table"],
            "depth_self_check_definition": (
                "|sensor depth read at the pixel CameraSpec.project sends a known "
                "world point to| minus |that point's true camera-z|, in metres. The "
                "headline number is measured on the nominated anchor object's nearest "
                "mesh vertex -- a point that is provably on the surface -- and the "
                "per-object table uses each object's world origin, which is generally "
                "not on its visible surface and therefore reads large by the object's "
                "own pivot offset. A wrong sign or a range-instead-of-axial depth "
                "would make the headline number metres, not millimetres."
            ),
            "anchor_rule": (
                "the furniture object whose world origin projects in frame and lies "
                "nearest the camera along its forward axis; chosen from geometry "
                "alone, never from the sensor's answer"
            ),
            "scene_bundle": {
                "common_scene_sha256": digest(scene_dir / "common_scene.json"),
                "camera_json_sha256": digest(scene_dir / "camera.json"),
                # Every mesh the pixels came from, not just the two manifests: the
                # bundle's shell meshes were rewritten *in place* at 16:26:47 on
                # 2026-09-24 (they now carry the OBJ comment "# corrected-convention
                # probe") while common_scene.json stayed at its 16:05:31 bytes, so a
                # manifest-only digest would have declared two different inputs to be
                # the same one. A digest per file is what makes the claim checkable.
                "mesh_sha256": {
                    obj.object_id: {
                        "visual": digest(obj.visual_mesh),
                        "collision": digest(obj.collision_mesh),
                    }
                    for obj in objects
                },
                "note": (
                    "the bundle under these digests is the one these pixels came from; "
                    "the bundle was regenerated at 16:05 while this driver was being "
                    "written, which moved focal_length_mm from 8.0 to 4.9 and so "
                    "changed the frustum, so the digest is the only safe way to say "
                    "which camera a column shows"
                ),
            },
            "camera_convention": {
                "mount": "principal axis = mount local +X, up = mount local +Z",
                "sapien_extrinsic_format": "OpenCV (row 2 = forward)",
                "forward_residual_m": forward_residual,
                "engine_extrinsic": extrinsic.tolist(),
                "engine_intrinsic": intrinsic.tolist(),
                "requested_fx_fy_cx_cy": [camera.fx, camera.fy, camera.cx, camera.cy],
            },
            "lighting": {
                **appearance,
                "applied": {
                    "ambient": [c * light_rig.gain_for(BACKEND) for c in light_rig.ambient],
                    "point_lights": [
                        {"position": list(p["position"]),
                         "colour": [c * light_rig.gain_for(BACKEND) for c in p["colour"]]}
                        for p in light_rig.point_lights
                    ],
                    "directional": {"direction": list(light_rig.direction),
                                    "colour": [c * light_rig.gain_for(BACKEND)
                                               for c in light_rig.directional_colour]},
                    "gain": light_rig.gain_for(BACKEND),
                },
                "why": (
                    "the raytracer traces shadow rays and the room has a ceiling, so "
                    "lights are placed inside the room footprint. These values are no "
                    "longer this driver's own: they come from the shared rig every "
                    "backend now reads, which is what makes the five columns one set. "
                    "SAPIEN is the only backend that can honour the rig in full."
                ),
            },
            "gates_failed": appearance_gaps,
            "near_far_m": [NEAR_M, FAR_M],
            "elapsed_s": time.time() - started,
        },
    )
    if missing:
        report["status"] = "failed"
    elif appearance_gaps:
        report["status"] = "appearance_gate_failed"
    write_json(out / "report.json", report)
    print(json.dumps(
        {
            "backend": BACKEND,
            "engine_version": sapien.__version__,
            "shader_mode": SHADER_MODE,
            "output_dir": str(out),
            "objects_built": len(geometry) - len(missing),
            "warnings": warnings,
            "depth_self_check_m": residual,
            "depth_visible_valid_fraction": visible_detail["valid_pixel_fraction"],
            "depth_collision_valid_fraction": collision_detail["valid_pixel_fraction"],
            "elapsed_s": report["elapsed_s"],
        },
        indent=2,
    ))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
