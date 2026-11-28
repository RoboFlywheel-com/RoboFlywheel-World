#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SPEC-014 PyBullet column: RGB and camera-z depth of the frozen common scene.

Reads the frozen bundle through :mod:`common` and emits, under ``--output-dir``:

    rgb.png                    1600x1000 RGB of the *visual* geometry
    depth_visible.npy          float32 metres along the camera forward axis
    depth_visible_mask.npy     bool
    depth_visible_color.png    the same depth under the one shared metre scale
    depth_collision.npy        the same quantity for the *collision* geometry
    depth_collision_mask.npy   bool
    normal_collision.png       world-space normals of the collision geometry
    report.json                what was loaded, what was converted, what was measured

Two passes, never merged: the visible pass loads every object's ``visual_mesh``
and produces RGB + visible depth; the collision pass loads ``collision_mesh``
only, in the same process after the visible bodies are removed, and produces
depth (SPEC-014 requirement 2).

Three things this process had to *measure* rather than assume, each recorded in
``report.json`` so the number is reproducible:

1. **Depth buffer convention.** ``getCameraImage`` under ``ER_TINY_RENDERER``
   returns an OpenGL depth buffer in ``[0, 1]``, not a distance. Measured on a
   plate perpendicular to the view at eight known distances: ``1/z`` is linear
   in the buffer value, and the fit is exactly the standard
   ``camera_z = near*far / (far - d*(far - near))`` (max residual 6.1e-7 over
   1 m .. 50 m). Background pixels are *exactly* ``1.0``, i.e. they would land on
   the far plane as a plausible-looking number -- so they are masked out by the
   buffer value, never by a depth threshold.
2. **Projection.** ``computeProjectionMatrixFOV`` interprets its ``fov`` argument
   in **degrees**, not radians. Handed ``camera.fovy_rad`` (1.3714) it produced a
   frustum whose focal length is ~68x too long -- a 1 m plate at 6.95 m covered
   the entire frame instead of the 88 px it must cover. The projection here is
   therefore built explicitly from the delivery's own K (``fx``/``fy``/``cx``/
   ``cy``), and the equivalence of that matrix with
   ``computeProjectionMatrixFOV(degrees(fovy), ...)`` is checked at run time.
3. **Placement.** Each object's world pose goes in as the multibody's
   ``basePosition``/``baseOrientation``; no vertex is ever baked, so the body
   pose in the report is the pose the engine actually used.
4. **The room frame of the bundle's shell.** The shell meshes shipped in the
   bundle are compared vertex-for-vertex against the same prims re-derived from
   the shell USD named in ``source.json`` under the row-vector convention USD
   uses, and against the room envelope the bundle's own ``room`` record declares.
   A mesh that disagrees with the shell USD is a named mismatch carrying the
   distance that was measured -- that is the gate that catches a displaced shell.
   A mesh the shell itself authors outside the room polygon (the door leaf, which
   swings out through the doorway) is recorded as an observation instead, because
   the comparison says the bundle reproduced it faithfully.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.workers.multisim.render_pybullet \\
      --scene outputs/_paper/spec014_common_scene_20260924_01 \\
      --output-dir outputs/_paper/spec014_renders_20260924_01/pybullet
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from rfw_w.scenes.workers.multisim.common import (
    CommonScene,
    appearance_gates,
    backend_report,
    colourise_depth,
    depth_to_world_normal_rgb,
    load_light_rig,
    save_depth_products,
    split_visual_materials,
    write_json,
)

# Clipping planes for the figure camera, in metres. The room's diagonal from the
# frozen eye is ~6.1 m, so `far` is 8x that; `near` is small enough to keep the
# plant (the nearest object, ~2.3 m away) well clear of it. Both are reported.
NEAR_M = 0.05
FAR_M = 50.0

# Background detection. TinyRenderer writes exactly 1.0 for a pixel no surface
# covered; anything at or above this is "no geometry seen", not "far away".
BUFFER_BACKGROUND_EPS = 1e-6

# The bundle's shell meshes are compared against this envelope, taken from the
# bundle's own `room` record. The tolerance is deliberately loose (0.25 m): this
# audit is meant to catch a whole-mesh displacement, not to police millimetres.
# A mesh outside this envelope is reported, but only called a mismatch if the
# independent re-derivation disagrees with it: the shell authors a door leaf that
# swings through the doorway and therefore sits outside the room polygon.
ROOM_ENVELOPE_TOLERANCE_M = 0.25

# How far the bundle's shell OBJ may differ from the same prim re-derived from
# the shell USD before the bundle is called displaced. The OBJs are written with
# six decimals, so an agreeing pair differs only by print rounding (<= 5e-7).
SHELL_FRAME_TOLERANCE_M = 1e-4


def eprint(*values: object) -> None:
    print(*values, file=sys.stderr)


def offscreen_flag(pybullet_module) -> int:
    """TinyRenderer renders collision shapes only; the flag keeps the
    segmentation pass (which the figure never uses) out of the render."""
    return int(getattr(pybullet_module, "ER_NO_SEGMENTATION_MASK", 0))


def quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Rotation matrix -> ``xyzw`` (PyBullet's order). The caller's matrix may
    carry a uniform scale; the quaternion is normalised, so the scale is dropped
    here -- for the frozen bundle every scale is 1.0, and a non-unit scale is
    reported.

    The components are assembled in ``wxyz`` first and reordered at the end on
    purpose: building the ``xyzw`` list directly is how an identity matrix comes
    out as ``[1, 0, 0, 0]`` -- a 180 degree turn about X that silently swaps y and
    z for every object in the scene, and an identity rotation comes out as
    ``[0, 0, 0, 1]``. Spelled out so the reorder is visible.
    """
    r = np.asarray(rotation, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        wxyz = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        wxyz = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        wxyz = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        wxyz = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    w, x, y, z = wxyz
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    return (quaternion / np.linalg.norm(quaternion)).tolist()


def rotation_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """``xyzw`` -> rotation matrix with ``v' = R @ v`` (column-vector form). Used
    by the placement audit, which re-reads the pose the engine reported."""
    x, y, z, w = (float(v) for v in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def gl_projection_matrix(camera, near: float, far: float) -> list[float]:
    """Column-major OpenGL frustum, built from the delivery's own K.

    ``far``/``near`` enter the third row exactly as GL defines them, which is
    what makes the measured depth relation invertible with the same two
    numbers."""
    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 0] = 2.0 * camera.fx / camera.width
    p[1, 1] = 2.0 * camera.fy / camera.height
    p[0, 2] = 1.0 - 2.0 * camera.cx / camera.width
    p[1, 2] = 2.0 * camera.cy / camera.height - 1.0
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2.0 * far * near / (far - near)
    p[3, 2] = -1.0
    return p.reshape(-1, order="F").tolist()


def matrix_from_column_major(flat: list[float]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(4, 4, order="F")


def camera_z_from_buffer(buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    """OpenGL depth buffer -> metres along the camera forward axis.

    ``d = 1`` (background) maps to ``far`` -- finite and wrong -- so callers must
    mask on the buffer value, not on the result."""
    buffer = np.asarray(buffer, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return near * far / (far - buffer * (far - near))


def obj_counts(path: Path) -> tuple[int, int]:
    """Vertex and face counts as written in the file. Read here rather than from
    the engine so the ledger shows what was *handed to* PyBullet."""
    vertices = faces = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    return vertices, faces


def obj_bounds_world(path: Path, matrix_world: np.ndarray) -> tuple[list[float], list[float]]:
    """World-space bounding box of an OBJ under a 4x4 pose. Used only by the
    placement audit -- the engine never sees these points."""
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            point = np.asarray([float(v) for v in line.split()[1:4]], dtype=np.float64)
            world = matrix_world[:3, :3] @ point + matrix_world[:3, 3]
            lo = np.minimum(lo, world)
            hi = np.maximum(hi, world)
    return lo.tolist(), hi.tolist()


def room_envelope(scene: CommonScene) -> dict:
    polygon = np.asarray(scene.payload["room"]["polygon_xy"], dtype=np.float64)
    return {
        "min": [float(polygon[:, 0].min()), float(polygon[:, 1].min()),
                float(scene.payload["room"].get("floor_z_m", 0.0))],
        "max": [float(polygon[:, 0].max()), float(polygon[:, 1].max()),
                float(scene.payload["room"]["height_m"])],
    }


def format_gap(backend: str, detail: str) -> str:
    return f"MULTISIM_ROOM_UNSUPPORTED({backend}, {detail})"


def shell_geometry_from_usd(scene_dir: Path, scene: CommonScene) -> dict[str, dict]:
    """Re-derive the shell meshes from the shell USD the bundle names.

    Two callers, one arithmetic:

    * the placement audit uses it as the *reference* the bundle's own shell OBJs
      are compared against, so "the shell is in the room frame" is a measurement
      rather than an assertion;
    * ``--shell-placement from-shell-usd`` uses it to substitute the whole shell,
      for the case where the bundle's OBJs are displaced. That path is disclosed
      in the report (``conversion.shell_placement`` and ``substituted_meshes/``)
      and never silent.

    The convention is the one USD itself uses, row-vector: ``p' = p @ M[:3,:3] +
    M[3,:3]``, i.e. ``Gf.Matrix4d.Transform``, translation in the last row. It is
    checkable against the bundle: the wall then lands on x[0.000, 4.906]
    y[0.000, 3.686] z[0.050, 2.857], which is the room polygon and height the
    bundle's own ``room`` record declares.
    """
    from pxr import Usd, UsdGeom  # noqa: PLC0415 -- only this path needs pxr

    source_path = scene_dir / "source.json"
    if not source_path.is_file():
        raise SystemExit(format_gap("pybullet", "no source.json; cannot locate the shell USD"))
    shell_uri = json.loads(source_path.read_text(encoding="utf-8")).get("shell_uri")
    if not shell_uri or not Path(shell_uri).is_file():
        raise SystemExit(format_gap("pybullet", f"shell USD not readable: {shell_uri}"))

    # The extractor's own naming rule, imported rather than re-implemented, so the
    # object ids here are the ones the bundle uses.
    from rfw_w.scenes.workers.multisim import extract_scene as extract

    stage = Usd.Stage.Open(str(shell_uri))
    counter: dict[str, int] = {}
    out: dict[str, dict] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.size == 0 or counts.size == 0:
            continue
        matrix = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        # Row-vector convention: the translation lives in the last row.
        points_world = points @ matrix[:3, :3] + matrix[3, :3]
        faces = []
        cursor = 0
        for count in counts:
            faces.append(indices[cursor:cursor + int(count)])
            cursor += int(count)
        role = extract.shell_role(str(prim.GetPath()))
        counter[role] = counter.get(role, 0) + 1
        object_id = f"shell_{role}" if counter[role] == 1 else f"shell_{role}_{counter[role]:02d}"
        out[object_id] = {
            "points_world": points_world,
            "faces": faces,
            "prim_path": str(prim.GetPath()),
            "world_translation_m": matrix[3, :3].tolist(),
        }
    return out


def shell_reference(scene_dir: Path, scene: CommonScene) -> tuple[dict[str, dict], str | None]:
    """The reference shell, with the reason it could not be derived if so.

    Never raises: the audit has to be able to record ``not_run`` rather than
    abort a render that has already succeeded.
    """
    try:
        return shell_geometry_from_usd(scene_dir, scene), None
    except SystemExit as error:
        return {}, str(error)
    except Exception as error:  # no pxr, or a stage that will not open
        return {}, f"{type(error).__name__}({error})"


def obj_points(path: Path) -> np.ndarray:
    """The ``v`` lines of an OBJ, in file order. Both sides of the frame
    comparison come from the same prim in the same vertex order, so the arrays
    line up index for index."""
    points = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                points.append([float(value) for value in line.split()[1:4]])
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def write_obj(path: Path, points: np.ndarray, faces: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# pvscene v3 SPEC-014: shell mesh re-derived from the shell USD (row-vector)"]
    for x, y, z in points:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for face in faces:
        lines.append("f " + " ".join(str(int(i) + 1) for i in face))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visual_shapes_for(pybullet_module, mesh_path: Path, scale: float,
                      material_dir: Path, tag: str,
                      default_diffuse: tuple[float, float, float]
                      ) -> tuple[list[int], list[dict]]:
    """One PyBullet visual shape per ``usemtl`` group, with its MTL colour/texture.

    Why this is not left to PyBullet's own OBJ loader: a PyBullet shape carries
    exactly **one** material. Handing it a multi-``usemtl`` OBJ therefore drops
    every material but one. Measured on this bundle, that is not a subtlety --
    it is the whole defect. Single-material meshes came out right (cup, desk:
    ``usemtl``=1) while every multi-material mesh came out flat white or
    mis-coloured (notebook 3, pen_holder 2, plant 4, chair 3, storage 3). The
    notebook's pink and red covers and the pen holder's black band are in the
    MTL the whole time; nothing was ever asking for them.

    The faces and UVs are not touched -- ``split_visual_materials`` only sorts
    existing ``f`` lines into per-material files -- so this changes what the
    render is *told about the surface*, never where the surface is.
    """
    material_dir.mkdir(parents=True, exist_ok=True)
    parts = split_visual_materials(mesh_path, material_dir / tag,
                                   default_diffuse=default_diffuse)
    shapes: list[int] = []
    records: list[dict] = []
    for part in parts:
        colour = part["diffuse_rgb"]
        shape = pybullet_module.createVisualShape(
            pybullet_module.GEOM_MESH,
            fileName=str(part["mesh"]),
            meshScale=[scale, scale, scale],
            rgbaColor=[colour[0], colour[1], colour[2], 1.0],
        )
        if shape < 0:
            raise SystemExit(format_gap(
                "pybullet", f"createVisualShape returned {shape} for {part['mesh']}"))
        shapes.append(int(shape))
        records.append({
            "material": part["name"],
            "face_count": part["faces"],
            "diffuse_rgb": list(colour),
            "texture": str(part["texture"]) if part["texture"] else None,
        })
    return shapes, records


def load_pass(pybullet_module, objects: list, mesh_attribute: str, substituted: dict,
              material_dir: Path | None = None,
              default_diffuse: tuple[float, float, float] = (1.0, 1.0, 1.0)
              ) -> tuple[list, list]:
    """Create one multibody per object. Returns (body ids, per-object records).

    ``material_dir`` switches on per-material visual shapes for the RGB pass.
    The collision pass leaves it ``None`` and is byte-for-byte what it was: the
    depth evidence must not move because the appearance was fixed.
    """
    bodies: list[int] = []
    records: list[dict] = []
    for index, obj in enumerate(objects):
        mesh = substituted.get(obj.object_id) if substituted else None
        mesh_path = Path(mesh["path"]) if mesh else Path(getattr(obj, mesh_attribute))
        matrix = obj.matrix_world()
        position = matrix[:3, 3]
        orientation = quaternion_xyzw(matrix[:3, :3])
        record = {
            "object_id": obj.object_id,
            "kind": obj.kind,
            "role": obj.role,
            "mesh": str(mesh_path),
            "mesh_source": "shell_usd_resubstituted" if mesh else "bundle",
            "mesh_vertex_count": mesh["vertex_count"] if mesh else obj_counts(mesh_path)[0],
            "mesh_face_count": mesh["face_count"] if mesh else obj_counts(mesh_path)[1],
            "matrix_world_translation_m": position.tolist(),
            "matrix_world_quaternion_xyzw": orientation,
            "uniform_scale": float(obj.scale),
        }
        shape = pybullet_module.createCollisionShape(
            pybullet_module.GEOM_MESH,
            fileName=str(mesh_path),
            flags=pybullet_module.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        if shape < 0:
            record["body_id"] = None
            record["loaded"] = False
            record["error"] = f"createCollisionShape returned {shape}"
            records.append(record)
            continue

        # The visual parts ride on the *same* body as the collision shape, as
        # the base visual plus one fixed massless link each. One body per object
        # keeps the depth buffer and the ray-test body map exactly as they were.
        visual_shapes: list[int] = []
        material_records: list[dict] = []
        if material_dir is not None:
            visual_shapes, material_records = visual_shapes_for(
                pybullet_module, mesh_path, float(obj.scale), material_dir,
                f"{index:02d}_{obj.object_id}", default_diffuse)

        link_count = max(0, len(visual_shapes) - 1)
        body = pybullet_module.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual_shapes[0] if visual_shapes else -1,
            basePosition=position.tolist(),
            baseOrientation=orientation,
            linkMasses=[0.0] * link_count,
            linkCollisionShapeIndices=[-1] * link_count,
            linkVisualShapeIndices=visual_shapes[1:],
            linkPositions=[[0.0, 0.0, 0.0]] * link_count,
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * link_count,
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * link_count,
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * link_count,
            linkParentIndices=[0] * link_count,
            linkJointTypes=[pybullet_module.JOINT_FIXED] * link_count,
            linkJointAxis=[[0.0, 0.0, 1.0]] * link_count,
        )
        # Textures attach after the body exists: base is link -1, the rest are 0..n-1.
        for part_index, material in enumerate(material_records):
            if not material["texture"]:
                continue
            texture_id = pybullet_module.loadTexture(material["texture"])
            if texture_id < 0:
                raise SystemExit(format_gap(
                    "pybullet", f"loadTexture failed for {material['texture']}"))
            pybullet_module.changeVisualShape(body, part_index - 1, textureUniqueId=texture_id)
            material["texture_applied"] = True
        if material_records:
            record["visual_materials"] = material_records
            record["visual_shape_count"] = len(visual_shapes)

        # Read the pose back from the engine: the ledger then shows the pose that
        # was actually rendered, not the one we meant to set.
        actual_position, actual_orientation = pybullet_module.getBasePositionAndOrientation(body)
        record.update(
            {
                "body_id": int(body),
                "loaded": True,
                "engine_base_position_m": [float(v) for v in actual_position],
                "engine_base_orientation_xyzw": [float(v) for v in actual_orientation],
            }
        )
        bodies.append(body)
        records.append(record)
    return bodies, records


def render(pybullet_module, camera, view: list[float], projection: list[float], want_rgb: bool,
           rig=None) -> tuple:
    extra = {}
    if rig is not None and want_rgb:
        gain = rig.gain_for("pybullet")
        # TinyRenderer takes ONE directional light and no point light. The rig's
        # two point lights cannot be expressed here; that is recorded as a named
        # gap in the report rather than passed over in silence.
        direction = [-v for v in rig.direction]  # TinyRenderer wants the direction TO the light
        extra = {
            "lightDirection": direction,
            "lightColor": list(rig.directional_colour),
            "lightAmbientCoeff": float(sum(rig.ambient) / 3.0) * gain,
            "lightDiffuseCoeff": 0.85 * gain,
            "lightSpecularCoeff": 0.05,
            "shadow": 1,
        }
    result = pybullet_module.getCameraImage(
        camera.width,
        camera.height,
        view,
        projection,
        renderer=pybullet_module.ER_TINY_RENDERER,
        flags=offscreen_flag(pybullet_module),
        **extra,
    )
    rgb = np.reshape(result[2], (camera.height, camera.width, 4))[..., :3].astype(np.uint8) if want_rgb else None
    buffer = np.reshape(result[3], (camera.height, camera.width)).astype(np.float64)
    return rgb, buffer


def depth_self_check(pybullet_module, camera, near: float, far: float, depth_z: np.ndarray,
                     body_names: dict, step: int = 31,
                     edge_tolerance_m: float = 0.02) -> dict:
    """The honest check: cast a ray through sampled pixels, take the surface point
    the *engine's* collision world reports, project it with ``camera.project``,
    and compare the depth buffer at that pixel against the true camera-z.

    This validates the depth conversion and the projection at once: if either the
    buffer inversion or the frustum were wrong, the residual would be metres, not
    millimetres. Pixels the ray misses are skipped, never counted as zero.

    Silhouettes are excluded and counted separately. A pixel one step off the edge
    of a table reads the tabletop's depth while the ray that was cast sees the
    wall behind it -- a metre, from a half-pixel of sampling, not a conversion
    error. ``edge_tolerance_m`` rejects a sample when its 3x3 neighbourhood is
    not flat within that band, which is the same 2-pixel silhouette exclusion
    SPEC-014's alignment audit asks for. Both populations are reported so the
    exclusion is visible rather than implied.
    """
    rotation = camera.rotation_world_to_cam
    pixel_offsets: list[float] = []
    interior: list[float] = []
    rejected: list[float] = []
    interior_detail: list[tuple[float, int, int, str]] = []
    hit_samples = 0
    # Pixel *centres*, not an even grid: TinyRenderer's depth value for pixel
    # (i, j) belongs to the ray through (i + 0.5, j + 0.5). Casting through the
    # centre removes the half-pixel ambiguity, which at a grazing surface would
    # otherwise show up as a residual that looks like a conversion error.
    for row in range(1, camera.height - 1, step):
        v = float(row) + 0.5
        for column in range(1, camera.width - 1, step):
            u = float(column) + 0.5
            direction_camera = np.array(
                [(u - camera.cx) / camera.fx, -(v - camera.cy) / camera.fy, -1.0]
            )
            direction_world = rotation.T @ direction_camera
            direction_world = direction_world / np.linalg.norm(direction_world)
            start = camera.eye
            end = camera.eye + direction_world * (far * 0.999)
            hits = pybullet_module.rayTest(start.tolist(), end.tolist())
            if not hits:
                continue
            hit_body, _, fraction, hit_position, _ = hits[0]
            if fraction >= 1.0:
                continue
            hit_position = np.asarray(hit_position, dtype=np.float64)
            pixels, true_z = camera.project(hit_position[None, :])
            hit_u, hit_v = float(pixels[0, 0]), float(pixels[0, 1])
            integer_u = int(round(hit_u))
            integer_v = int(round(hit_v))
            if not (1 <= integer_u < camera.width - 1 and 1 <= integer_v < camera.height - 1):
                continue
            neighbourhood = depth_z[integer_v - 1:integer_v + 2, integer_u - 1:integer_u + 2]
            if not np.all(np.isfinite(neighbourhood)):
                continue
            measured = float(depth_z[integer_v, integer_u])
            if measured <= 0.0:
                continue
            hit_samples += 1
            # The ray was cast through (u, v); the surface it hit must reproject to
            # the same pixel. This is the projection check, independent of depth.
            pixel_offsets.append(max(abs(hit_u - u), abs(hit_v - v)))
            residual = abs(measured - float(true_z[0]))
            if neighbourhood.max() - neighbourhood.min() <= edge_tolerance_m:
                interior.append(residual)
                interior_detail.append(
                    (residual, integer_u, integer_v, body_names.get(int(hit_body), str(hit_body)))
                )
            else:
                rejected.append(residual)
    interior_array = np.asarray(interior, dtype=np.float64)
    rejected_array = np.asarray(rejected, dtype=np.float64)
    worst = sorted(interior_detail, reverse=True)[:5]
    return {
        "method": "rayTest through sampled pixels -> engine hit point -> camera.project -> "
                  "compare depth buffer at that pixel against the true camera_z",
        "ray_hits": hit_samples,
        "samples": int(interior_array.size),
        "edge_rejected_samples": int(rejected_array.size),
        "edge_rule": f"3x3 neighbourhood flat within {edge_tolerance_m} m (2-pixel silhouette band)",
        "median_m": float(np.median(interior_array)) if interior_array.size else None,
        "p95_m": float(np.percentile(interior_array, 95)) if interior_array.size else None,
        "max_m": float(interior_array.max()) if interior_array.size else None,
        "edge_rejected_max_m": float(rejected_array.max()) if rejected_array.size else None,
        "max_ray_reprojection_px": float(max(pixel_offsets)) if pixel_offsets else None,
        # Where the largest residuals sit, so a large max can be attributed inside
        # this run instead of assumed to be somebody else's problem.
        "worst_samples": [
            {"residual_m": residual, "pixel": [u, v], "ray_test_object_id": object_id}
            for residual, u, v, object_id in worst
        ],
        "not_run": [] if interior_array.size else ["depth_self_check_m(no ray hit the loaded geometry)"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, type=Path,
                        help="the frozen common-scene bundle (common_scene.json + meshes/)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shell-placement", choices=("bundle", "from-shell-usd"), default="bundle",
                        help="bundle: consume meshes/<id>/*.obj exactly as the bundle holds them "
                             "(the contract). from-shell-usd: re-derive the room shell from the "
                             "shell USD named in source.json instead, for the case where the "
                             "audit reports the bundle's shell meshes displaced -- disclosed in "
                             "the report, never silent. The audit runs either way")
    parser.add_argument("--lighting", type=Path,
                        default=data_path('examples') / "paper_multisim" / "lighting.json",
                        help="the shared five-backend light rig (SPEC-014 R5)")
    args = parser.parse_args()

    scene_dir = args.scene.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")

    scene = CommonScene.load(scene_dir)
    rig = load_light_rig(args.lighting)
    camera = scene.camera
    if (camera.width, camera.height) != (1600, 1000) or camera.width <= 0:
        eprint(f"note: the frozen camera is {camera.width}x{camera.height}")

    warnings: list[str] = []
    observations: list[str] = []
    not_run: list[str] = []

    import pybullet  # noqa: PLC0415 -- importing it is this process's whole job

    client = pybullet.connect(pybullet.DIRECT)
    started = time.monotonic()

    view = pybullet.computeViewMatrix(camera.eye.tolist(), camera.target.tolist(), [0.0, 0.0, 1.0])
    projection = gl_projection_matrix(camera, NEAR_M, FAR_M)
    fov_check = matrix_from_column_major(
        pybullet.computeProjectionMatrixFOV(math.degrees(camera.fovy_rad),
                                           camera.width / camera.height, NEAR_M, FAR_M)
    )
    projection_matches_fov_helper = bool(
        np.allclose(fov_check, matrix_from_column_major(projection), atol=1e-9)
    )

    # The camera's up vector is +Z and the frozen view is oblique, so the
    # degenerate case (forward parallel to up) is not reachable here; still, the
    # view matrix is checked rather than assumed.
    forward = camera.forward
    degenerate_up = abs(float(np.dot(forward, np.array([0.0, 0.0, 1.0])))) > 0.9999
    if degenerate_up:
        raise SystemExit(format_gap("pybullet", "camera forward is parallel to the up axis"))

    visible_substitution: dict[str, dict] = {}
    substituted_records: list[dict] = []
    if args.shell_placement == "from-shell-usd":
        derived = shell_geometry_from_usd(scene_dir, scene)
        for object_id, geometry in sorted(derived.items()):
            target = out / "substituted_meshes" / f"{object_id}.obj"
            write_obj(target, geometry["points_world"], geometry["faces"])
            visible_substitution[object_id] = {
                "path": target,
                "vertex_count": int(geometry["points_world"].shape[0]),
                "face_count": len(geometry["faces"]),
            }
            substituted_records.append(
                {
                    "object_id": object_id,
                    "prim_path": geometry["prim_path"],
                    "usd_world_translation_m": geometry["world_translation_m"],
                    "mesh": str(target),
                }
            )

    objects = scene.objects()  # skips hidden objects by the same rule in all backends
    hidden = [o for o in scene.objects(include_hidden=True) if o.object_id not in
              {v.object_id for v in objects}]

    # ---- pass 1: visual meshes -> RGB + visible depth ------------------------
    material_dir = args.output_dir / "material_parts"
    visible_bodies, visible_records = load_pass(pybullet, objects, "visual_mesh",
                                                visible_substitution, material_dir=material_dir,
                                                default_diffuse=rig.shell_albedo)
    if not visible_bodies:
        raise SystemExit(format_gap("pybullet", "no object mesh loaded"))
    rgb, visible_buffer = render(pybullet, camera, view, projection, want_rgb=True, rig=rig)
    visible_depth = camera_z_from_buffer(visible_buffer, NEAR_M, FAR_M).astype(np.float32)
    visible_depth[visible_buffer >= 1.0 - BUFFER_BACKGROUND_EPS] = np.nan
    appearance, appearance_gaps = appearance_gates(rgb, rig, camera, "pybullet")
    lighting_record = {
        **appearance,
        "applied": {
            "ambient_coeff": float(sum(rig.ambient) / 3.0) * rig.gain_for("pybullet"),
            "directional_direction": list(rig.direction),
            "directional_colour": list(rig.directional_colour),
            "gain": rig.gain_for("pybullet"),
            "shadow": True,
        },
        "not_applied": [
            f"POINT_LIGHT_UNSUPPORTED(pybullet, {len(rig.point_lights)} point lights in the rig)"
        ] if rig.point_lights else [],
        "why_not_applied": (
            "ER_TINY_RENDERER's getCameraImage takes a single directional light and has no point "
            "light parameter at all. The rig's two point lights therefore cannot be expressed in "
            "this backend; the ambient coefficient carries their contribution approximately. "
            "Recorded as a named gap so the column is not read as having the same rig applied."
        ),
    }
    # A mesh's material file is not named after it (the prepared assets carry a hash
    # name), so the directory is scanned rather than a sibling path guessed.
    textured = sorted(
        record["object_id"] for record in visible_records
        if any(entry.suffix.lower() == ".mtl" for entry in Path(record["mesh"]).parent.iterdir())
    )
    multi_material = {
        record["object_id"]: record["visual_shape_count"]
        for record in visible_records if record.get("visual_shape_count", 0) > 1
    }
    rgb_detail = {
        "shape": [camera.height, camera.width, 3],
        "mean_channel_value": float(rgb.mean()),
        "distinct_colours": int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]),
        "objects_with_mtl_beside_mesh": textured,
        "objects_with_multiple_materials": multi_material,
        "materials": (
            "One PyBullet visual shape per source `usemtl` group, carrying that material's MTL "
            "Kd and map_Kd, attached to the same body as the collision shape (base visual plus "
            "one fixed massless link each). This is not cosmetic: a PyBullet shape holds exactly "
            "one material, so the earlier build -- which created only a collision shape and let "
            "the OBJ loader guess -- rendered every multi-`usemtl` mesh flat white or "
            "mis-coloured, while single-material meshes came out right. The shell meshes are "
            "written without an .mtl and still render flat grey, which is a property of the "
            "bundle's geometry source rather than of the engine."
        ),
    }
    self_check = depth_self_check(
        pybullet, camera, NEAR_M, FAR_M, visible_depth,
        {record["body_id"]: record["object_id"] for record in visible_records if record["loaded"]},
    )

    # ---- pass 2: collision meshes -> collision depth -------------------------
    for body in visible_bodies:
        pybullet.removeBody(body)
    collision_substitution = (
        {object_id: dict(record) for object_id, record in visible_substitution.items()}
        if args.shell_placement == "from-shell-usd" else {}
    )
    collision_bodies, collision_records = load_pass(pybullet, objects, "collision_mesh",
                                                    collision_substitution)
    _, collision_buffer = render(pybullet, camera, view, projection, want_rgb=False)
    collision_depth = camera_z_from_buffer(collision_buffer, NEAR_M, FAR_M).astype(np.float32)
    collision_depth[collision_buffer >= 1.0 - BUFFER_BACKGROUND_EPS] = np.nan
    for body in collision_bodies:
        pybullet.removeBody(body)
    pybullet.disconnect(client)

    elapsed = time.monotonic() - started

    # ---- products ------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # noqa: PLC0415

    Image.fromarray(rgb).save(out / "rgb.png")
    visible_detail = save_depth_products(out, visible_depth, "depth_visible")
    visible_colour = colourise_depth(visible_depth)
    Image.fromarray(visible_colour).save(out / "depth_visible_color.png")
    collision_detail = save_depth_products(out, collision_depth, "depth_collision")
    # The helper masks invalid pixels itself; handing it NaN makes numpy cast NaN to
    # uint8 (undefined, and a warning) before the mask replaces those pixels with
    # gray. 0 is invalid by the helper's own rule, so the output is identical.
    collision_normals = depth_to_world_normal_rgb(np.nan_to_num(collision_depth), camera)
    Image.fromarray(collision_normals).save(out / "normal_collision.png")

    # ---- the placement audit ------------------------------------------------
    # The bundle is not taken on trust. Every shell mesh it ships is compared
    # vertex-for-vertex against the same prim re-derived from the shell USD the
    # bundle names under the row-vector convention, and measured against the room
    # envelope the bundle's own `room` record declares. Both outcomes are
    # recorded; a displacement is named with the distance that was measured.
    envelope = room_envelope(scene)
    reference, reference_gap = shell_reference(scene_dir, scene)
    if reference_gap is not None:
        not_run.append(
            f"bundle_shell_frame_check(pybullet): the shell meshes as bundled were not compared "
            f"against the shell USD -- {reference_gap}"
        )
    openings = scene.payload.get("room", {}).get("openings", [])
    opening_hint = ", ".join(
        f"{o.get('opening_id')} ({o.get('kind')}) at {o.get('center_xy')}" for o in openings
    ) or "none declared"

    audit: list[dict] = []
    compared: list[float] = []
    for record in visible_records:
        if record["kind"] != "shell" or not record["loaded"]:
            continue
        # Rebuild the pose from what the engine reported, so the audit measures the
        # geometry that was rendered rather than the pose we meant to set.
        position = np.asarray(record["engine_base_position_m"], dtype=np.float64)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_from_quaternion_xyzw(record["engine_base_orientation_xyzw"])
        matrix[:3, 3] = position
        lo, hi = obj_bounds_world(Path(record["mesh"]), matrix)
        beyond = {
            name: round(max(envelope["min"][axis] - lo[axis], hi[axis] - envelope["max"][axis]), 4)
            for axis, name in enumerate("xyz")
            if max(envelope["min"][axis] - lo[axis], hi[axis] - envelope["max"][axis]) > 0
        }
        expected = reference.get(record["object_id"], {}).get("points_world")
        difference: float | None = None
        shape_note: str | None = None
        if expected is None:
            matches: bool | None = None
        else:
            actual = obj_points(Path(record["mesh"]))
            if actual.shape != expected.shape:
                matches = False
                shape_note = f"{actual.shape[0]} vertices against {expected.shape[0]} in the shell USD"
            else:
                difference = float(np.abs(actual - expected).max())
                compared.append(difference)
                matches = difference <= SHELL_FRAME_TOLERANCE_M
        audit.append(
            {
                "object_id": record["object_id"],
                "mesh": record["mesh"],
                "mesh_source": record["mesh_source"],
                "world_bbox_min_m": [round(v, 4) for v in lo],
                "world_bbox_max_m": [round(v, 4) for v in hi],
                "inside_room_envelope": not beyond,
                "outside_envelope_by_m": beyond,
                "matches_shell_usd": matches,
                "max_abs_difference_m": difference,
            }
        )
        if matches is False:
            # The gate that can fail: the bundle disagrees with the shell USD's own
            # world-space points, so what was rendered is not the delivered shell.
            warnings.append(
                f"MULTISIM_GEOMETRY_MISMATCH(pybullet, {record['object_id']} as bundled does not "
                f"match the shell USD's own world-space points: "
                + (shape_note if shape_note else f"largest vertex difference {difference:.6f} m")
                + f". As bundled its world bbox is x[{lo[0]:.3f},{hi[0]:.3f}] "
                  f"y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.3f},{hi[2]:.3f}], against the room "
                  f"envelope x[{envelope['min'][0]:.3f},{envelope['max'][0]:.3f}] "
                  f"y[{envelope['min'][1]:.3f},{envelope['max'][1]:.3f}] "
                  f"z[{envelope['min'][2]:.3f},{envelope['max'][2]:.3f}]. The shell USD re-derived "
                  f"with the row-vector convention p @ M[:3,:3] + M[3,:3] is the reference, so "
                  f"re-extract the bundle or run --shell-placement from-shell-usd; the other four "
                  f"columns consume the same bundle)"
            )
        elif beyond:
            # Outside the room polygon and, if compared, faithful to the shell USD:
            # authored geometry, not a displaced mesh. The delivery's own shell puts
            # a door leaf through the doorway, outside the wall.
            why = (
                f"it matches the shell USD to {difference:.6f} m, so this is authored geometry "
                f"rather than a displaced mesh. The bundle's own room record declares these "
                f"openings: {opening_hint}"
                if difference is not None else
                "the shell USD was not read, so whether this is authored geometry or a displaced "
                "mesh is not known -- see not_run"
            )
            observations.append(
                f"SHELL_OUTSIDE_ROOM_ENVELOPE(pybullet, {record['object_id']}: world bbox "
                f"x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.3f},{hi[2]:.3f}] "
                f"extends {max(beyond.values()):.3f} m beyond the room envelope "
                f"x[{envelope['min'][0]:.3f},{envelope['max'][0]:.3f}] "
                f"y[{envelope['min'][1]:.3f},{envelope['max'][1]:.3f}] "
                f"z[{envelope['min'][2]:.3f},{envelope['max'][2]:.3f}], and {why})"
            )

    # The hidden shell mesh is part of the bundle too, and the other four columns
    # hide it by the same rule. It never reaches the engine, so it is compared
    # without a pose: this is where "all N meshes accounted for" is made true.
    visible_shell_ids = {a["object_id"] for a in audit}
    for obj in scene.objects(include_hidden=True):
        if obj.kind != "shell" or obj.object_id in visible_shell_ids:
            continue
        expected = reference.get(obj.object_id, {}).get("points_world")
        if expected is None:
            continue
        actual = obj_points(Path(obj.visual_mesh))
        if actual.shape == expected.shape:
            difference = float(np.abs(actual - expected).max())
            compared.append(difference)
            matches, shape_note = difference <= SHELL_FRAME_TOLERANCE_M, None
        else:
            difference, matches = None, False
            shape_note = f"{actual.shape[0]} vertices against {expected.shape[0]}"
        audit.append(
            {
                "object_id": obj.object_id,
                "mesh": str(obj.visual_mesh),
                "mesh_source": "bundle",
                "hidden_by_rule": scene.hidden_rule(),
                "inside_room_envelope": None,
                "outside_envelope_by_m": {},
                "matches_shell_usd": matches,
                "max_abs_difference_m": difference,
            }
        )
        if not matches:
            warnings.append(
                f"MULTISIM_GEOMETRY_MISMATCH(pybullet, {obj.object_id} (hidden by the rule, not "
                f"rendered) as bundled does not match the shell USD's own world-space points: "
                + (shape_note if shape_note else f"largest vertex difference {difference:.6f} m")
                + ")"
            )

    engine_version = str(pybullet.getAPIVersion())
    report = backend_report(
        backend="pybullet",
        engine_version=engine_version,
        scene=scene,
        camera=camera,
        extra={
            "geometry_loaded": {
                "bundle_mesh_count": int(scene.payload.get("object_count", 0))
                + int(scene.payload.get("shell_mesh_count", 0)),
                "bundle_furniture": int(scene.payload.get("object_count", 0)),
                "bundle_shell": int(scene.payload.get("shell_mesh_count", 0)),
                "objects_loaded_visible_pass": len(visible_records),
                "objects_loaded_collision_pass": len(collision_records),
                "hidden_by_rule": [{"object_id": o.object_id, "role": o.role,
                                    "reason": scene.hidden_rule()} for o in hidden],
                "visible_pass": visible_records,
                "collision_pass": collision_records,
                "shell_resubstitution": substituted_records,
            },
            "conversion": {
                "mesh_format": "Wavefront OBJ handed to createCollisionShape(GEOM_MESH, "
                               "flags=GEOM_FORCE_CONCAVE_TRIMESH): triangles are rendered as "
                               "authored, never convex-hulled",
                "placement": "matrix_world() -> basePosition + baseOrientation; rotation applied "
                             "before translation; no vertex is baked",
                "depth_buffer": "ER_TINY_RENDERER OpenGL depth buffer; a pixel no surface covered "
                                "is written as exactly 1.0",
                "depth_buffer_to_camera_z": "camera_z = near*far / (far - d*(far - near))",
                "depth_convention_provenance": "measured on a plate perpendicular to the view at "
                                               "1,2,3,5,7,10,20,50 m: max fit residual 6.1e-7",
                "projection": "OpenGL frustum built from the delivery's own K (fx/fy/cx/cy) and "
                              "the near/far below",
                "projection_warning": "computeProjectionMatrixFOV takes its fov in DEGREES: given "
                                      "camera.fovy_rad it returned a frustum ~68x too long "
                                      "(a 1 m plate at 6.95 m filled the frame instead of 88 px), "
                                      "so the helper was not used for the render",
                "projection_cross_check": {
                    "matches_computeProjectionMatrixFOV_of_degrees_fovy": projection_matches_fov_helper,
                    "max_abs_difference": float(
                        np.abs(fov_check - matrix_from_column_major(projection)).max()
                    ),
                },
                "shell_placement": args.shell_placement,
            },
            "near_far_m": {"near": NEAR_M, "far": FAR_M},
            "rgb_detail": rgb_detail,
            "lighting": lighting_record,
            "gates_failed": appearance_gaps,
            "depth_products_detail": {"visible": visible_detail, "collision": collision_detail},
            "depth_self_check_m": self_check,
            "shell_frame_audit": {
                "room_envelope_m": envelope,
                "shell_usd_reference": "the shell prims named by source.json, re-derived with the "
                                       "row-vector convention p @ M[:3,:3] + M[3,:3]",
                "shell_usd_reference_gap": reference_gap,
                "match_tolerance_m": SHELL_FRAME_TOLERANCE_M,
                "envelope_tolerance_m": ROOM_ENVELOPE_TOLERANCE_M,
                "objects_compared": len(compared),
                "max_abs_difference_m": max(compared) if compared else None,
                "shell_objects": audit,
            },
            "warnings": warnings,
            "observations": observations,
            "not_run": not_run,
            "elapsed_s": elapsed,
            "status": "passed" if not warnings and not appearance_gaps else (
                "appearance_gate_failed" if not warnings else "geometry_mismatch"),
            "status_reason": (
                "the engine ran, its depth is self-consistent, but the shell frame could not be "
                "checked against the shell USD; see not_run"
                if reference_gap is not None else
                "the engine ran, its depth is self-consistent, and every shell mesh it rendered "
                "matches the shell USD's own world-space points"
                if not warnings else
                "the engine ran and its depth is self-consistent, but the geometry loaded does not "
                "match the shell USD; see warnings"
            ),
            "engine": {"module": pybullet.__file__, "direct_mode": True, "renderer": "ER_TINY_RENDERER"},
        },
    )
    write_json(out / "report.json", report)

    print(json_dumps({
        "output_dir": str(out),
        "backend": "pybullet",
        "engine_version": engine_version,
        "status": report["status"],
        "objects_loaded": len(visible_records),
        "visible_valid_pixel_fraction": visible_detail["valid_pixel_fraction"],
        "visible_depth_m": [visible_detail["valid_min_m"], visible_detail["valid_max_m"]],
        "collision_valid_pixel_fraction": collision_detail["valid_pixel_fraction"],
        "depth_self_check_m": {k: self_check[k] for k in ("samples", "median_m", "p95_m", "max_m",
                                                         "max_ray_reprojection_px",
                                                         "edge_rejected_samples")},
        "warnings": warnings,
        "observations": observations,
        "not_run": not_run,
        "elapsed_s": elapsed,
    }))
    return 0


def json_dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
