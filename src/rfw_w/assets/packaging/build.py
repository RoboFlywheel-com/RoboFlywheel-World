# SPDX-License-Identifier: Apache-2.0
"""Build a portable standalone asset package through the real production modules."""

from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path


def build_package(
    request,
    mesh_path,
    face_ids_path,
    annotations_path,
    out,
    voxel_resolution=64,
    coacd_timeout=600,
    visual_mesh_path=None,
):
    import numpy as np, trimesh
    from rfw_w.assets.annotation.parts import validate
    from rfw_w.assets.geometry.prepare import prepare, prepare_pair
    from rfw_w.assets.geometry.collision import decompose_parts
    from rfw_w.assets.physics.estimate import estimate, anchor
    from rfw_w.assets.contracts.io import write_json
    from . import formats

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    from rfw_w.assets.geometry.quality_metrics import scan_one

    quality = scan_one(Path(mesh_path), None, None, request.asset_id)
    if quality.get("error") or quality.get("degen_face_ratio", 0) > 0.1:
        raise ValueError("mesh geometry quality check failed")
    write_json(out / "geometry_quality.json", quality)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    ids = np.load(face_ids_path)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(ids) != len(mesh.faces)
        or (ids < 0).any()
    ):
        raise ValueError("face_ids must assign every face to an integer part")
    part_ids = [int(v) for v in np.unique(ids)]
    raw = json.loads(Path(annotations_path).read_text())
    dimensions, target, annotations = validate(raw, part_ids)
    if visual_mesh_path is not None:
        source = trimesh.load(visual_mesh_path, force="mesh", process=False)
        visual_metric, metric, frame = prepare_pair(
            source, mesh, ids, dimensions, request.source_up_axis
        )
    else:
        metric, frame = prepare(mesh, ids, dimensions, request.source_up_axis)
        visual_metric = metric
    mesh_dir = out / "mesh"
    mesh_dir.mkdir()
    visual = mesh_dir / ("m_" + request.asset_id + ".obj")
    visual_metric.export(visual)
    collision, hulls = decompose_parts(
        metric,
        ids,
        part_ids,
        mesh_dir / "collision",
        timeout=coacd_timeout,
        seed=request.seed,
    )
    physical, masses = estimate(
        metric, ids, annotations, collision, request.mass_policy, voxel_resolution
    )
    physical = anchor(physical, masses, target, request.mass_policy)
    physical.update(
        key=request.asset_id,
        dimensions_m=frame["dimensions_m"],
        real_bbox=frame["dimensions_m"],
        up_axis="Z",
        frame=frame,
        n_parts=len(part_ids),
        n_collision_hulls=len(hulls),
        coverage={"unassigned_faces": 0, "ok": True},
    )
    write_json(out / "part_physics.json", physical)
    aff = out / "affordance"
    aff.mkdir()
    metric.export(aff / "mesh_part_seg.glb")
    np.save(aff / "face_ids.npy", ids)
    write_json(
        aff / "affordance_annot.json",
        {
            "object": raw["object"],
            "affordances": [
                {**annotations[pid], "id": pid, "physics": physical["parts"][str(pid)]}
                for pid in part_ids
            ],
            "quality_check": {"success": True},
            "functional_verification": "not_run",
            "visual_quality": "not_run",
        },
    )
    relative = [str(path.relative_to(out)) for path in hulls]
    produced = {}
    if "urdf" in request.formats:
        path = out / ("m_" + request.asset_id + ".urdf")
        formats.urdf(
            request.asset_id, str(visual.relative_to(out)), relative, physical, path
        )
        produced["urdf"] = path.name
    if "mjcf" in request.formats:
        directory = out / "mjcf"
        directory.mkdir()
        path = directory / ("m_" + request.asset_id + ".xml")
        formats.mjcf(
            request.asset_id,
            str(visual.relative_to(out)),
            relative,
            physical,
            path,
            visual_metric,
        )
        produced["mjcf"] = str(path.relative_to(out))
    if "usd" in request.formats:
        path = out / (request.asset_id + ".usd")
        formats.usd(request.asset_id, visual_metric, hulls, physical, path)
        produced["usd"] = path.name
    files = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file() and not path.name.endswith(".log")
    }
    report = {
        "schema": "rfw.asset-package/v1",
        "asset_id": request.asset_id,
        "class_id": request.class_id,
        "frame": frame,
        "physics": "part_physics.json",
        "formats": produced,
        "files": files,
        "production_status": "package_complete",
        "simulation_validation": "not_run",
        "geometry_quality": "measured_basic_gate_passed",
        "visual_quality": "not_run",
    }
    write_json(out / "manifest.json", report)
    (out / "_SUCCESS").write_text("package_complete\n")
    return report
