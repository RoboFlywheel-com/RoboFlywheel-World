# SPDX-License-Identifier: Apache-2.0
"""One asset production use case, shared by public single and batch commands."""

from __future__ import annotations
from dataclasses import asdict
import hashlib, json, shutil
from pathlib import Path
from rfw_w.assets.contracts.request import AssetRequest
from rfw_w.assets.contracts.io import write_json


def generate(request, runtime, out):
    if isinstance(request, dict):
        request = AssetRequest.from_dict(request)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("output must be a new directory")
    stages = []
    if request.input_kind == "prompt":
        stages.append("image")
    if request.input_kind in ("prompt", "image"):
        stages.extend(("background", "3d"))
    if request.face_ids is None:
        stages.append("segmentation")
    if request.annotations is None:
        stages.extend(("render", "annotation"))
    stages.append("physical")
    runtime.preflight(stages, request.formats)
    out.mkdir(parents=True)
    report = {
        "schema": "rfw.asset-run/v1",
        "asset_id": request.asset_id,
        "status": "running",
        "stages": {},
        "simulation_validation": "not_run",
        "visual_quality": "not_run",
    }
    write_json(out / "run.json", report)
    raw = {
        "asset_id": request.asset_id,
        "class_id": request.class_id,
        "input": {request.input_kind: request.input_value},
        "category": request.category,
        "seed": request.seed,
        "annotations": request.annotations,
        "face_ids": request.face_ids,
        "formats": list(request.formats),
        "mass_policy": request.mass_policy,
        "source_up_axis": request.source_up_axis,
    }
    write_json(out / "request.json", raw)
    row = {
        "asset_id": request.asset_id,
        "class_id": request.class_id,
        "category": request.category,
        "prompt": request.input_value if request.input_kind == "prompt" else "",
    }
    manifest = out / "worker-input.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    work = out / "intermediate"
    work.mkdir()
    base = ["--manifest", manifest]
    image = work / "images" / request.class_id / (request.asset_id + ".png")
    mesh = work / "meshes" / request.class_id / (request.asset_id + ".glb")
    part = work / "parts" / request.class_id / request.asset_id
    if request.input_kind == "image":
        image.parent.mkdir(parents=True)
        shutil.copy2(request.input_value, image)
    if request.input_kind == "mesh":
        mesh.parent.mkdir(parents=True)
        shutil.copy2(request.input_value, mesh)
    plans = {
        "image": (
            "rfw_w.assets.workers.flux",
            base + ["--out-dir", work / "images", "--seed", request.seed],
        ),
        "background": (
            "rfw_w.assets.workers.background",
            base + ["--in-dir", work / "images", "--out-dir", work / "rgba"],
        ),
        "3d": (
            "rfw_w.assets.workers.trellis",
            base
            + [
                "--in-dir",
                work / "rgba",
                "--out-dir",
                work / "meshes",
                "--seed",
                request.seed,
            ],
        ),
        "segmentation": (
            "rfw_w.assets.workers.segmentation",
            base + ["--in-dir", work / "meshes", "--out-dir", work / "parts"],
        ),
        "render": (
            "rfw_w.assets.workers.render",
            base + ["--in-dir", work / "meshes", "--out-dir", work / "renders"],
        ),
        "annotation": (
            "rfw_w.assets.workers.annotation",
            base + ["--parts-dir", work / "parts", "--renders-dir", work / "renders"],
        ),
    }
    current = "prepare"
    try:
        if request.face_ids:
            part.mkdir(parents=True)
            shutil.copy2(request.face_ids, part / "face_ids.npy")
            shutil.copy2(mesh, part / "mesh_part_seg.glb")
        for current in stages:
            if current == "physical":
                faces = part / "face_ids.npy"
                annotations = (
                    Path(request.annotations)
                    if request.annotations
                    else part / "annotations.json"
                )
                args = [
                    "--request",
                    out / "request.json",
                    "--mesh",
                    part / "mesh_part_seg.glb",
                    "--visual-mesh",
                    mesh,
                    "--face-ids",
                    faces,
                    "--annotations",
                    annotations,
                    "--out",
                    out / "package",
                ]
                module = "rfw_w.assets.workers.physical"
            else:
                module, args = plans[current]
            report["stages"][current] = runtime.run(current, module, args, out)
            write_json(out / "run.json", report)
        report["status"] = "package_complete"
        report["package"] = "package"
        report["formats"] = list(request.formats)
    except Exception as exc:
        report.update(
            status="failed", failed_stage=current, error_type=type(exc).__name__
        )
        write_json(out / "run.json", report)
        raise
    write_json(out / "run.json", report)
    return report
