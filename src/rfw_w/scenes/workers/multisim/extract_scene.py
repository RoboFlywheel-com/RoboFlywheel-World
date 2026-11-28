#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze one v3 rigid delivery into the scene all five backends consume.

SPEC-014 TASK 1. Reads a delivery read-only and writes a self-contained bundle:
the room shell's meshes converted out of USD, every instance's visual and
collision mesh, the world pose of each, and one frozen camera. Nothing here
touches the delivery, and nothing here renders.

The point of the bundle is that the five backends cannot each invent their own
scene: they all read ``common_scene.json`` and the meshes beside it.

Writes, under ``--output-dir``:

    common_scene.json    scene version, objects, poses, camera, hidden rule
    camera.json          the camera on its own, for readers that want only that
    meshes/<id>/         per-object visual (.obj) and collision (_collision.obj)
    source.json          what was read, with a digest per input

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.workers.multisim.extract_scene \\
      --delivery outputs/office_pick_cup/s5_office_full_20260924_02 \\
      --output-dir outputs/_paper/spec014_common_scene_20260924_01
"""

from __future__ import annotations
from rfw_w.scenes.visualization.inputs import resolve_input

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from rfw_w.scenes.workers.multisim.common import HIDDEN_SHELL_ROLES, CameraSpec, digest, write_json

# Shell primitive names carry Infinigen's factory names; map them to the role the
# hidden rule and the caption talk about.
SHELL_ROLE_HINTS = (
    ("exterior", "exterior"),
    ("skirtingboard_ceiling", "ceiling_skirting"),
    ("skirtingboard", "skirtingboard"),
    ("wall", "wall"),
    ("floor", "floor"),
    ("ceiling", "ceiling"),
    ("door", "door"),
    ("window", "window"),
)


def link_or_copy(source: Path, target: Path) -> None:
    """Hardlink when the two live on one filesystem, copy otherwise. The common
    scene is a second reference to bytes the delivery already owns, and the
    JuiceFS quota is tight enough that duplicating them is worth avoiding."""
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def shell_role(prim_path: str) -> str:
    lowered = prim_path.lower()
    for hint, role in SHELL_ROLE_HINTS:
        if hint in lowered:
            return role
    return "shell_other"


def export_shell_meshes(shell_path: Path, out_dir: Path) -> list[dict]:
    """Convert the built shell USD to OBJ, one file per mesh, world-space points."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(shell_path))
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    if str(up_axis) != "Z" or abs(float(meters) - 1.0) > 1e-9:
        raise SystemExit(
            f"MULTISIM_ROOM_UNSUPPORTED(shell axis={up_axis} metersPerUnit={meters})"
        )
    records: list[dict] = []
    counter: dict[str, int] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.size == 0 or counts.size == 0:
            continue
        xform = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        # Gf.Matrix4d is a **row-vector** transform: p_world = p_local @ M, so the
        # translation is the last row and the rotation is not transposed. Reading
        # it as a column-vector matrix drops the translation and inverts the
        # rotation -- measured on the office shell that put the wall at
        # x[-2.714, 2.192] instead of the room's x[0.000, 4.906], i.e. a room
        # centred on the origin, which still renders as a plausible room.
        points_world = points @ xform[:3, :3] + xform[3, :3]
        faces = []
        cursor = 0
        for count in counts:
            faces.append(indices[cursor:cursor + int(count)])
            cursor += int(count)
        role = shell_role(str(prim.GetPath()))
        counter[role] = counter.get(role, 0) + 1
        object_id = f"shell_{role}" if counter[role] == 1 else f"shell_{role}_{counter[role]:02d}"
        target = out_dir / object_id / "visual.obj"
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_obj(target, points_world, faces)
        records.append(
            {
                "object_id": object_id,
                "role": role,
                "prim_path": str(prim.GetPath()),
                "vertex_count": int(points_world.shape[0]),
                "face_count": len(faces),
                "bounds_min": points_world.min(axis=0).tolist(),
                "bounds_max": points_world.max(axis=0).tolist(),
            }
        )
    return records


def assert_shell_in_room_frame(records: list[dict], room: dict, tolerance_m: float = 0.25) -> dict:
    """Fail when the shell did not land on the room the plan declares.

    A gate that can fail, because this exact mistake is invisible: reading USD's
    row-vector matrix as a column-vector one moves every shell mesh to the
    origin, and the result still renders as a perfectly plausible room -- just
    the wrong one, at the wrong place. Measured on the office shell the wall came
    out at x[-2.714, 2.192] instead of x[0.000, 4.906], and nothing in the picture
    said so.

    The test is cheap and independent of the transform code: the shell must
    enclose the floor plan, allowing for the wall solid's thickness.
    """
    if not records:
        raise SystemExit("MULTISIM_ROOM_UNSUPPORTED(shell produced no meshes)")
    low = np.min([np.asarray(r["bounds_min"]) for r in records], axis=0)
    high = np.max([np.asarray(r["bounds_max"]) for r in records], axis=0)
    polygon = np.asarray(room["polygon_xy"], dtype=np.float64)
    plan_low = np.array([polygon[:, 0].min(), polygon[:, 1].min(), float(room["floor_z_m"])])
    plan_high = np.array([polygon[:, 0].max(), polygon[:, 1].max(),
                          float(room["floor_z_m"]) + float(room["height_m"])])
    # The shell must *contain* the plan, so it is allowed to reach past it -- the
    # facade and the wall solid do. Only failing to reach it is a failure.
    shortfall = np.maximum(low - plan_low, plan_high - high)
    worst = float(np.max(shortfall))
    if worst > tolerance_m:
        raise SystemExit(
            f"MULTISIM_ROOM_UNSUPPORTED(shell frame: the shell spans "
            f"{np.round(low, 3).tolist()}..{np.round(high, 3).tolist()} but the plan needs "
            f"{np.round(plan_low, 3).tolist()}..{np.round(plan_high, 3).tolist()}; "
            f"short by {worst:.3f} m)")
    return {
        "shell_bounds_min": low.tolist(),
        "shell_bounds_max": high.tolist(),
        "plan_bounds_min": plan_low.tolist(),
        "plan_bounds_max": plan_high.tolist(),
        "worst_shortfall_m": worst,
        "tolerance_m": tolerance_m,
    }


def _write_obj(path: Path, points: np.ndarray, faces: list[np.ndarray]) -> None:
    """Minimal OBJ writer: the five backends need geometry, not materials, and a
    hand-rolled writer keeps the extraction free of a mesh library."""
    lines = ["# pvscene v3 SPEC-014 common scene"]
    for x, y, z in points:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for face in faces:
        lines.append("f " + " ".join(str(int(i) + 1) for i in face))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of a Wavefront OBJ, read from its vertex lines.

    Read here rather than through a mesh library so the extractor stays free of
    one; the figure only needs the box, and the box is what the fitting rule
    below tests.
    """
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            point = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            lo = np.minimum(lo, point)
            hi = np.maximum(hi, point)
    if not np.isfinite(lo).all():
        raise SystemExit(f"MULTISIM_GEOMETRY_MISMATCH(empty mesh: {path})")
    return lo, hi


def fit_focal_length(instances: list[dict], room: dict, eye: np.ndarray, target: np.ndarray,
                     width: int, height: int, horizontal_aperture_mm: float,
                     margin: float = 0.02) -> tuple[float, dict]:
    """The longest focal length that keeps **every instance's box** inside the frame.

    Chosen by measurement rather than fixed by hand. A camera inside the room
    can never contain the room's own near corners -- the floor reaches the image
    edge in every direction -- so the rule tests the thing the figure actually
    needs to show: that no instance is cropped or missing. Measured on the office
    common scene, the 8 mm lens SPEC-013's preview uses leaves 3 of 11 instances
    outside the frame (the ceiling light half out, the plant entirely out, the
    floor lamp 88% in); 5.0 mm is the longest that fits all of them, and
    continues to fit with a 2% margin from 4.8 mm down.

    Returns the focal length and the evidence for it, so a reader can check the
    choice instead of trusting it.
    """
    boxes = []
    for record in instances:
        lo, hi = obj_bounds(Path(record["visual_uri"]))
        matrix = np.eye(4)
        w, x, y, z = record["rotation_wxyz"]
        matrix[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        matrix[:3, 3] = record["translation"]
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1])
                            for c in (lo[2], hi[2])])
        boxes.append((record["instance_id"], (corners @ matrix[:3, :3].T) + matrix[:3, 3]))

    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    rotation = np.stack([right, np.cross(right, fwd), -fwd], axis=0)
    lo_u, hi_u = margin * width, (1.0 - margin) * width
    lo_v, hi_v = margin * height, (1.0 - margin) * height

    def outside(focal_mm: float) -> list[str]:
        fx = width * focal_mm / horizontal_aperture_mm
        missed = []
        for instance_id, corners in boxes:
            cam = (corners - eye) @ rotation.T
            z = -cam[:, 2]
            if (z <= 0).any():
                missed.append(instance_id)
                continue
            u = width / 2.0 + fx * cam[:, 0] / z
            v = height / 2.0 - fx * cam[:, 1] / z
            if (u < lo_u).any() or (u > hi_u).any() or (v < lo_v).any() or (v > hi_v).any():
                missed.append(instance_id)
        return missed

    chosen, missed_at_choice = None, None
    focal = 20.0
    while focal >= 3.0:
        missed = outside(focal)
        if not missed:
            chosen, missed_at_choice = focal, missed
            break
        focal = round(focal - 0.1, 2)
    if chosen is None:
        raise SystemExit(
            f"MULTISIM_ROOM_UNSUPPORTED(no focal length fits {len(boxes)} instances; "
            f"still failing at {focal + 0.1:.1f} mm: {sorted(outside(focal + 0.1))})")
    return chosen, {
        "chosen_focal_length_mm": chosen,
        "rule": "longest focal length at which every instance's axis-aligned box stays "
                "inside the frame with a 2% margin",
        "instances_required": len(boxes),
        "instances_outside_at_8mm": sorted(outside(8.0)),
        "instances_outside_at_chosen": sorted(missed_at_choice),
        "note": "a camera inside the room cannot contain the room's own near corners; the "
                "rule tests the instances, which is what the figure must show",
    }


def figure_camera(room: dict, aperture_reference: dict, instances: list[dict]) -> tuple:
    """The frozen figure camera: a room-corner oblique aimed at the room centre.

    Uses the same corner inset, height clamp and focal length as SPEC-013's
    room preview, so the two paper figures look at the rooms the same way.

    The vertical aperture is derived from the horizontal one by the resolution
    aspect, which is deliberate and differs from the delivery's own camera
    record (ADR-011: production carries 20.955 / 15.2908, i.e. non-square
    pixels). Only Isaac can honour two apertures at once; MuJoCo, PyBullet,
    Genesis and SAPIEN each expose a single field of view and assume square
    pixels. A camera with non-square pixels therefore cannot be expressed in all
    five engines, and SPEC-014 requires one shared K. Square pixels are the only
    choice under which "the five engines saw the same frustum" is even a
    falsifiable claim; the deviation from the production camera is recorded here
    and in every backend's report.
    """
    polygon = room["polygon_xy"]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    height = float(room["height_m"])
    mid = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    width, height_px = 1600, 1000
    horizontal_aperture = float(aperture_reference["horizontal_aperture_mm"])
    eye = np.array([min_x + 0.34, min_y + 0.34, min(height - 0.22, 2.65)])
    target = np.array([mid[0], mid[1], 0.65])
    focal, evidence = fit_focal_length(instances, room, eye, target, width, height_px,
                                       horizontal_aperture)
    return CameraSpec(
        name="paper_multisim_room",
        eye=eye,
        target=target,
        width=width,
        height=height_px,
        focal_length_mm=focal,
        horizontal_aperture_mm=horizontal_aperture,
        vertical_aperture_mm=horizontal_aperture * height_px / width,
    ), evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-id", default=None,
                        help="reuse a camera from the delivery's own request instead of "
                             "the figure camera")
    parser.add_argument("--camera-config", type=Path, default=None,
                        help="figure-only JSON camera; records its source hash and square-pixel intrinsics")
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")

    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    request = json.loads((delivery / manifest["isaac_request_path"]).read_text())
    scene_ir = json.loads((delivery / "scene_ir.json").read_text())
    version = manifest["scene_version"]
    if request["scene_version"] != version or scene_ir.get("revision") is None:
        raise SystemExit(f"MULTISIM_SCENE_VERSION_MISMATCH(office request={request['scene_version'][:8]})")

    shell_uri = request["room"].get("shell_uri")
    if not shell_uri:
        raise SystemExit("MULTISIM_ROOM_UNSUPPORTED(no shell_uri on the request)")
    shell = (delivery / manifest["isaac_request_path"]).parent / shell_uri
    shell = shell.resolve()
    if not shell.is_file():
        raise SystemExit(f"MULTISIM_ROOM_UNSUPPORTED(shell missing: {shell})")

    if args.camera_id and args.camera_config:
        parser.error("choose only one camera source")
    if args.camera_config:
        record = json.loads(resolve_input(args.camera_config).read_text())
        width, height_px = int(record["width"]), int(record["height"])
        horizontal = float(record["horizontal_aperture_mm"])
        camera = CameraSpec(
            name=str(record["name"]), eye=np.asarray(record["eye"], dtype=np.float64),
            target=np.asarray(record["target"], dtype=np.float64), width=width,
            height=height_px, focal_length_mm=float(record["focal_length_mm"]),
            horizontal_aperture_mm=horizontal,
            vertical_aperture_mm=horizontal * height_px / width)
        framing = {"source": "figure camera config", "config_sha256": digest(args.camera_config),
                   "rule": "explicit task-workspace camera with square pixels"}
    elif args.camera_id:
        record = next((c for c in request["cameras"] if c["camera_id"] == args.camera_id), None)
        if record is None:
            raise SystemExit(f"MULTISIM_CAMERA_MISMATCH(no camera {args.camera_id})")
        camera = CameraSpec(
            name=record["camera_id"],
            eye=np.asarray(record["eye"], dtype=np.float64),
            target=np.asarray(record["target"], dtype=np.float64),
            width=int(record["resolution"][0]),
            height=int(record["resolution"][1]),
            focal_length_mm=float(record["focal_length_mm"]),
            horizontal_aperture_mm=float(record["horizontal_aperture_mm"]),
            vertical_aperture_mm=float(record["vertical_aperture_mm"]),
        )
        framing = {"source": f"delivery camera {args.camera_id}; framing not re-derived"}
    else:
        camera, framing = figure_camera(request["room"], request["cameras"][0],
                                        request["instances"])

    out.mkdir(parents=True, exist_ok=True)
    objects: list[dict] = []

    shell_records = export_shell_meshes(shell, out / "meshes")
    shell_frame = assert_shell_in_room_frame(shell_records, request["room"])
    for rec in shell_records:
        rel = f"meshes/{rec['object_id']}/visual.obj"
        objects.append(
            {
                "object_id": rec["object_id"],
                "kind": "shell",
                "role": rec["role"],
                "visual_mesh": rel,
                "collision_mesh": rel,
                "collision_note": "the room shell uses its visible mesh as its collision mesh",
                "translation": [0.0, 0.0, 0.0],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": 1.0,
                "hidden": rec["role"] in HIDDEN_SHELL_ROLES,
                "vertex_count": rec["vertex_count"],
                "face_count": rec["face_count"],
            }
        )

    for record in request["instances"]:
        object_id = record["instance_id"]
        asset_dir = out / "meshes" / object_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        visual = Path(record["visual_uri"])
        collision = Path(record.get("collision_uri") or record["visual_uri"])
        if not visual.is_file():
            raise SystemExit(f"MULTISIM_GEOMETRY_MISMATCH({object_id}: visual mesh missing)")
        if not collision.is_file():
            raise SystemExit(f"MULTISIM_GEOMETRY_MISMATCH({object_id}: collision mesh missing)")
        link_or_copy(visual, asset_dir / "visual.obj")
        link_or_copy(collision, asset_dir / "collision.obj")
        # The prepared asset keeps its .mtl and texture beside the .obj; without
        # them an engine that reads materials would fall back to its own default
        # and the RGB columns would differ for a reason that is not the engine.
        for extra in visual.parent.iterdir():
            if extra.suffix.lower() in (".mtl", ".png") and extra.is_file():
                link_or_copy(extra, asset_dir / extra.name)
        objects.append(
            {
                "object_id": object_id,
                "kind": "furniture",
                "role": record["entity_id"],
                "visual_mesh": f"meshes/{object_id}/visual.obj",
                "collision_mesh": f"meshes/{object_id}/collision.obj",
                "translation": [float(v) for v in record["translation"]],
                "rotation_wxyz": [float(v) for v in record["rotation_wxyz"]],
                "scale": 1.0,
                "hidden": False,
            }
        )

    payload = {
        "schema_version": "pvscene.multisim-common-scene/v1",
        "scene_version": version,
        "scene_id": manifest["scene_id"],
        "source_delivery": str(delivery),
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "room": request["room"],
        "camera": camera.as_record(),
        "objects": objects,
        "camera_framing": framing,
        "shell_frame": shell_frame,
        "hidden_rule": {
            "hidden_roles": list(HIDDEN_SHELL_ROLES),
            "why": "the outer facade surrounds the room; from an interior camera it only "
                   "occludes, and hiding it changes no interior geometry",
            "applies_to": "all five backends, identically",
        },
        "object_count": sum(1 for o in objects if o["kind"] == "furniture"),
        "shell_mesh_count": len(shell_records),
    }
    write_json(out / "common_scene.json", payload)
    write_json(out / "camera.json", camera.as_record())
    # A per-mesh digest list, written last. This exists because the bundle was
    # once edited in place: a scratch probe symlinked these mesh directories into
    # /tmp and wrote through the links, so eight shell meshes were replaced under
    # a standing bundle without any manifest noticing. A manifest-only digest
    # cannot see that -- it names the bundle, not the bytes. Any consumer can now
    # check the meshes it actually loaded against this list.
    write_json(
        out / "bundle.sha256",
        {
            "scene_version": version,
            "meshes": {
                str(Path(rec["visual_mesh"])): digest(out / rec["visual_mesh"])
                for rec in objects
            },
            "objects": {
                str(Path(rec["collision_mesh"])): digest(out / rec["collision_mesh"])
                for rec in objects
            },
        },
    )
    write_json(
        out / "source.json",
        {
            "delivery": str(delivery),
            "manifest_sha256": digest(delivery / "delivery_manifest.json"),
            "scene_ir_sha256": digest(delivery / "scene_ir.json"),
            "request_sha256": digest(delivery / manifest["isaac_request_path"]),
            "shell_uri": str(shell),
            "shell_sha256": digest(shell),
            "scene_version": version,
        },
    )
    print(json.dumps(
        {
            "output_dir": str(out),
            "scene_version": version,
            "instances": payload["object_count"],
            "shell_meshes": payload["shell_mesh_count"],
            "hidden_roles": list(HIDDEN_SHELL_ROLES),
            "camera": camera.name,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
