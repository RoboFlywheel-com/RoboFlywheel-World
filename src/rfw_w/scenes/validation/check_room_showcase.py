#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently verify one-room paper preview provenance and image products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(bundle: Path) -> dict:
    provenance = read(bundle / "provenance.json")
    delivery = Path(provenance["delivery"])
    render_dir = Path(provenance.get("render_dir", delivery))
    files = provenance["files"]
    manifest = read(delivery / "delivery_manifest.json")
    image_dir = render_dir / "isaac" if "render_dir" in provenance else delivery / manifest["isaac_artifact_dir"]
    report = read(image_dir / "report.json")
    captures = {item["camera_id"]: item for item in report["captures"]}
    camera = provenance["camera"]
    capture = captures.get(camera)
    detail_camera = provenance.get("detail_camera")
    detail_capture = captures.get(detail_camera)
    problems: list[str] = []
    if capture is None:
        problems.append(f"SHOWCASE_RENDER_MISSING({camera})")
    if detail_camera and detail_capture is None:
        problems.append(f"SHOWCASE_RENDER_MISSING({detail_camera})")
    if manifest["scene_version"] != provenance["scene_version"] or report["scene_version"] != provenance["scene_version"]:
        problems.append("SHOWCASE_SCENE_VERSION_MISMATCH")
    if report["status"] != "passed":
        problems.append("SHOWCASE_RENDER_MISSING(worker status)")
    root_files = {
        "manifest": delivery / "delivery_manifest.json",
        "scene_ir": delivery / "scene_ir.json",
        "shell_reference": delivery / "shell_reference.json",
        "render_report": image_dir / "report.json",
    }
    plans = sorted(delivery.glob("shell_build/**/room_plan.json"))
    plan = next((p for p in plans if p.with_name("measurement.json").is_file() and
                 sha(p) == files.get("room_plan")), None)
    if plan is None:
        problems.append("SHOWCASE_SHELL_PLAN_MISMATCH(plan missing or changed)")
    else:
        root_files["room_plan"] = plan
        root_files["shell_measurement"] = plan.with_name("measurement.json")
    if capture:
        if capture["depth_convention"] != "camera_z_meters":
            problems.append("SHOWCASE_DEPTH_CONVENTION_MISMATCH")
        root_files["rgb"] = image_dir / capture["rgb_path"]
        root_files["depth"] = image_dir / capture["depth_path"]
    if detail_capture:
        root_files["detail_rgb"] = image_dir / detail_capture["rgb_path"]
    target_id, support_id = provenance.get("target_id"), provenance.get("support_id")
    if target_id and support_id and capture and detail_capture:
        index = {int(key): value for key, value in report.get("instance_id_map", {}).items()}
        for item in (capture, detail_capture):
            visible = {index.get(int(value)) for value in item.get("visible_instance_ids", [])}
            if target_id not in visible or support_id not in visible:
                problems.append(f"SHOWCASE_ENTITY_NOT_VISIBLE({item['camera_id']},target/support)")
    if "render_dir" in provenance:
        root_files["figure_render_provenance"] = render_dir / "provenance.json"
        render_provenance = read(root_files["figure_render_provenance"])
        if (Path(render_provenance["source_delivery"]).resolve() != delivery.resolve() or
                render_provenance["source_scene_version"] != provenance["scene_version"] or
                render_provenance["report_sha256"] != files["render_report"]):
            problems.append("SHOWCASE_PROVENANCE_MISSING(figure render binding)")
    for name, path in root_files.items():
        if not path.is_file() or sha(path) != files.get(name):
            problems.append(f"SHOWCASE_PROVENANCE_MISSING({name})")
    if capture:
        for name, source in (("rgb.png", root_files["rgb"]), ("depth_m.npy", root_files["depth"])):
            copy = bundle / "panels" / name
            if not copy.is_file() or sha(copy) != sha(source):
                problems.append(f"SHOWCASE_PROVENANCE_MISSING(panels/{name})")
    if detail_capture:
        source = root_files["detail_rgb"]
        copy = bundle / "panels" / "detail_rgb.png"
        if not copy.is_file() or sha(copy) != sha(source):
            problems.append("SHOWCASE_PROVENANCE_MISSING(panels/detail_rgb.png)")
    png = bundle / "figure_room_preview.png"
    pdf = bundle / "figure_room_preview.pdf"
    if not png.is_file() or not pdf.is_file() or pdf.stat().st_size == 0:
        problems.append("SHOWCASE_RENDER_MISSING(figure)")
    else:
        with Image.open(png) as picture:
            if picture.width < 2160 or picture.height < 1300:
                problems.append("SHOWCASE_RENDER_MISSING(print resolution)")
    return {"verified": not problems, "problems": problems,
            "scene_version": provenance["scene_version"], "camera": camera,
            "machine_checks": len(root_files) + 4,
            "visual_review": "not_run(user preview pending)",
            "production_readiness": manifest["resolution"]["achieved"]}


def verify_selection(bundle: Path) -> dict:
    """Independent pass over the three-row alignment/diversity bundle.

    Everything here is recomputed from the deliveries and the render reports.
    ``measurements.json`` is the figure's own claim about itself; this reader
    never reads it as evidence, so a figure that mis-measured itself still gets
    caught. The facts SPEC-013's acceptance names -- three rows, two different
    room seeds each, no core drift, every panel traceable -- are each checked
    here directly rather than believed.
    """
    provenance = read(bundle / "provenance.json")
    rows = provenance["rows"]
    problems: list[str] = []
    if len(rows) != 3:
        problems.append(f"SHOWCASE_ROW_COUNT({len(rows)})")
    tasks: dict[str, str] = {}
    seeds_by_row: dict[str, list] = {}
    frozen_core: dict[str, bool] = {}
    traceable = 0
    for row_id, sides in sorted(rows.items()):
        seeds = []
        poses: dict[str, dict] = {}
        for side in ("a", "b"):
            record = sides[side]
            delivery = Path(record["delivery"])
            render_dir = Path(record["render_dir"])
            manifest = read(delivery / "delivery_manifest.json")
            scene = read(delivery / "scene_ir.json")
            report = read(render_dir / "isaac" / "report.json")
            if manifest["scene_version"] != record["scene_version"]:
                problems.append(f"SHOWCASE_SCENE_VERSION_MISMATCH({row_id},{side}:delivery)")
            if report["scene_version"] != record["scene_version"]:
                problems.append(f"SHOWCASE_SCENE_VERSION_MISMATCH({row_id},{side}:report)")
            if report.get("status") != "passed":
                problems.append(f"SHOWCASE_RENDER_MISSING({row_id},{side}:worker)")
            if report.get("capture_state") != "authored":
                problems.append(f"SHOWCASE_SCENE_VERSION_MISMATCH({row_id},{side}:settled)")
            seed = manifest["randomization"].get("room_seed")
            if seed != record["room_seed"]:
                problems.append(f"SHOWCASE_SCENE_VERSION_MISMATCH({row_id},{side}:room_seed)")
            seeds.append(seed)
            image_dir = render_dir / "isaac"
            capture = next((c for c in report["captures"] if c["camera_id"] == provenance["camera"]),
                           None)
            if capture is None:
                problems.append(f"SHOWCASE_RENDER_MISSING({row_id},{side}:{provenance['camera']})")
            elif capture["depth_convention"] != "camera_z_meters":
                problems.append(f"SHOWCASE_DEPTH_CONVENTION_MISMATCH({row_id},{side})")
            files = record["files"]
            # A figure-only rerender carries its own binding to the delivery it
            # rendered: without it a row can name a render dir that was produced
            # from some other delivery and every digest above still matches,
            # because they are all digests of that other delivery's files. The
            # single-room pass already checks this; the row pass has to as well.
            # The file is optional in the layout but not in the evidence, so a
            # missing one is a named gap rather than a crash.
            binding_path = render_dir / "provenance.json"
            binding = read(binding_path) if binding_path.is_file() else None
            if (binding is None
                    or Path(binding.get("source_delivery", "")).resolve() != delivery.resolve()
                    or binding.get("source_scene_version") != record["scene_version"]
                    or binding.get("report_sha256") != files["render_report"]):
                problems.append(f"SHOWCASE_PROVENANCE_MISSING({row_id},{side}:figure render binding)")
            checked = [("manifest", delivery / "delivery_manifest.json"),
                       ("scene_ir", delivery / "scene_ir.json"),
                       ("render_report", image_dir / "report.json")]
            if capture:
                checked += [("rgb", image_dir / capture["rgb_path"]),
                            ("depth", image_dir / capture["depth_path"])]
            for name, path in checked:
                if not path.is_file() or sha(path) != files[name]:
                    problems.append(f"SHOWCASE_PROVENANCE_MISSING({row_id},{side}:{name})")
                    continue
                traceable += 1
                if name in ("rgb", "depth"):
                    suffix = "rgb.png" if name == "rgb" else "depth_m.npy"
                    copy = bundle / "panels" / f"{row_id}_{side}_{suffix}"
                    if not copy.is_file() or sha(copy) != sha(path):
                        problems.append(f"SHOWCASE_PROVENANCE_MISSING({row_id},{side}:panel {name})")
            target_id = manifest["plan"]["target_id"]
            support_id = manifest["plan"]["support_id"]
            present = {i["entity_id"] for i in scene["instances"]}
            if target_id not in present or support_id not in present:
                problems.append(f"SHOWCASE_ENTITY_MISSING({row_id},{side})")
            poses[side] = {
                "target_id": target_id,
                "support_id": support_id,
                "invariant_spec_hash": manifest["randomization"].get("invariant_spec_hash"),
                "invariant_entity_ids": sorted(manifest["randomization"].get("invariant_entity_ids", [])),
                "hard_relation": next(
                    ({"predicate": r["predicate"], "satisfied": r["satisfied"]}
                     for r in scene["relations"]
                     if r["subject"] == target_id and r["object"] == support_id and r["hard"]),
                    None),
            }
        if seeds[0] == seeds[1]:
            problems.append(f"SHOWCASE_NO_VARIATION({row_id}:same room seed)")
        seeds_by_row[row_id] = seeds
        # The core is the frozen closure -- identity, the hard relation, and the
        # invariant hash -- not a world pose. The support is placed against whatever
        # room the seed built, so the pose is expected to move; asserting it constant
        # here would fail every honest row.
        core_ok = (poses["a"]["target_id"] == poses["b"]["target_id"]
                   and poses["a"]["support_id"] == poses["b"]["support_id"]
                   and poses["a"]["invariant_spec_hash"] == poses["b"]["invariant_spec_hash"]
                   and poses["a"]["invariant_entity_ids"] == poses["b"]["invariant_entity_ids"]
                   and poses["a"]["hard_relation"] is not None
                   and poses["a"]["hard_relation"] == poses["b"]["hard_relation"]
                   and poses["a"]["hard_relation"]["satisfied"])
        frozen_core[row_id] = core_ok
        if not core_ok:
            problems.append(f"SHOWCASE_CORE_DRIFT({row_id})")
        tasks[row_id] = Path(sides["a"]["delivery"]).parent.name
    if len(set(tasks.values())) != len(tasks):
        problems.append("SHOWCASE_NO_VARIATION(rows are not distinct tasks)")
    png = bundle / "figure_room_alignment.png"
    pdf = bundle / "figure_room_alignment.pdf"
    depth_png = bundle / "figure_room_alignment_depth.png"
    for path in (png, pdf, depth_png):
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"SHOWCASE_RENDER_MISSING(figure {path.name})")
    if png.is_file():
        with Image.open(png) as picture:
            if picture.width < 2160:
                problems.append("SHOWCASE_RENDER_MISSING(print resolution)")
    return {"verified": not problems, "problems": problems, "mode": "selection",
            "rows": sorted(rows), "tasks": tasks, "room_seeds": seeds_by_row,
            "core_frozen": frozen_core, "panels_traceable": traceable,
            "camera": provenance["camera"],
            "machine_checks": 12 * len(rows) + 3,
            "visual_review": "not_run(user preview pending)"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")
    if (args.bundle.resolve() / "selection.json").is_file():
        result = verify_selection(args.bundle.resolve())
        out.mkdir(parents=True, exist_ok=True)
        (out / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["verified"] else 1
    result = verify(args.bundle.resolve())
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
