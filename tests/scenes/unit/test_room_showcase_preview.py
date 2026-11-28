# SPDX-License-Identifier: Apache-2.0
"""A figure audit must reject changed pixels and a render from another scene."""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = module_path('rfw_w.scenes.validation.check_room_showcase')


audit = importlib.import_module('rfw_w.scenes.validation.check_room_showcase')


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bundle(tmp_path: Path) -> tuple[Path, Path]:
    delivery = tmp_path / "delivery"
    figure = tmp_path / "figure"
    images = delivery / "isaac" / "scene-v1" / "attempt_001"
    shell = delivery / "shell_build" / "one" / "infinigen_build"
    figure.mkdir()
    write_json(delivery / "delivery_manifest.json", {
        "scene_version": "scene-v1", "isaac_artifact_dir": "isaac/scene-v1/attempt_001",
        "resolution": {"achieved": "draft"},
    })
    write_json(delivery / "scene_ir.json", {"schema_version": "test"})
    write_json(delivery / "shell_reference.json", {"scene_version": "scene-v1"})
    write_json(shell / "room_plan.json", {"hash": "plan-v1"})
    write_json(shell / "measurement.json", {"plan_hash": "plan-v1"})
    write_json(images / "report.json", {
        "scene_version": "scene-v1", "status": "passed",
        "instance_id_map": {"1": "cup", "2": "desk"},
        "captures": [{"camera_id": "paper_room_oblique", "rgb_path": "rgb.png",
                      "depth_path": "depth.npy", "depth_convention": "camera_z_meters",
                      "visible_instance_ids": [1, 2]},
                     {"camera_id": "paper_desk_detail", "rgb_path": "detail.png",
                      "depth_path": "detail_depth.npy", "depth_convention": "camera_z_meters",
                      "visible_instance_ids": [1, 2]}],
    })
    Image.new("RGB", (10, 10), "white").save(images / "rgb.png")
    Image.new("RGB", (10, 10), "blue").save(images / "detail.png")
    (images / "depth.npy").write_bytes(b"raw depth fixture")
    panels = figure / "panels"
    panels.mkdir()
    shutil.copy2(images / "rgb.png", panels / "rgb.png")
    shutil.copy2(images / "detail.png", panels / "detail_rgb.png")
    shutil.copy2(images / "depth.npy", panels / "depth_m.npy")
    Image.new("RGB", (2162, 1400), "white").save(figure / "figure_room_preview.png")
    (figure / "figure_room_preview.pdf").write_bytes(b"%PDF-1.4 fixture")
    files = {
        "manifest": delivery / "delivery_manifest.json",
        "scene_ir": delivery / "scene_ir.json",
        "shell_reference": delivery / "shell_reference.json",
        "room_plan": shell / "room_plan.json",
        "shell_measurement": shell / "measurement.json",
        "render_report": images / "report.json",
        "rgb": images / "rgb.png",
        "detail_rgb": images / "detail.png",
        "depth": images / "depth.npy",
    }
    write_json(figure / "provenance.json", {
        "delivery": str(delivery), "scene_version": "scene-v1",
        "camera": "paper_room_oblique", "detail_camera": "paper_desk_detail",
        "target_id": "cup", "support_id": "desk",
        "files": {name: audit.sha(path) for name, path in files.items()},
    })
    return figure, images


def test_rejects_modified_panel_pixels(tmp_path: Path) -> None:
    figure, _ = bundle(tmp_path)
    assert audit.verify(figure)["verified"] is True
    Image.new("RGB", (10, 10), "black").save(figure / "panels" / "rgb.png")
    result = audit.verify(figure)
    assert result["verified"] is False
    assert "SHOWCASE_PROVENANCE_MISSING(panels/rgb.png)" in result["problems"]


def test_rejects_other_scene_report(tmp_path: Path) -> None:
    figure, images = bundle(tmp_path)
    report_path = images / "report.json"
    report = audit.read(report_path)
    report["scene_version"] = "scene-v2"
    write_json(report_path, report)
    result = audit.verify(figure)
    assert result["verified"] is False
    assert "SHOWCASE_SCENE_VERSION_MISMATCH" in result["problems"]


def test_rejects_modified_detail_panel(tmp_path: Path) -> None:
    figure, _ = bundle(tmp_path)
    assert audit.verify(figure)["verified"] is True
    Image.new("RGB", (10, 10), "red").save(figure / "panels" / "detail_rgb.png")
    result = audit.verify(figure)
    assert result["verified"] is False
    assert "SHOWCASE_PROVENANCE_MISSING(panels/detail_rgb.png)" in result["problems"]


def test_rejects_target_hidden_from_detail(tmp_path: Path) -> None:
    figure, images = bundle(tmp_path)
    report_path = images / "report.json"
    report = audit.read(report_path)
    report["captures"][1]["visible_instance_ids"] = [2]
    write_json(report_path, report)
    result = audit.verify(figure)
    assert result["verified"] is False
    assert "SHOWCASE_ENTITY_NOT_VISIBLE(paper_desk_detail,target/support)" in result["problems"]


# ---------------------------------------------------------------------------
# SPEC-013's three-row alignment/diversity bundle. A second figure with a second
# verifier entry point (``verify_selection``), so it needs its own fixture and its
# own negatives. The selection verifier does not read ``provenance.json`` as a
# claim about the rows it describes: it re-reads every delivery, render report and
# raw pixel file behind a panel and compares them itself. The fixture below is
# therefore the smallest bundle in which every one of those comparisons can pass,
# and each negative below moves exactly one of them.
# ---------------------------------------------------------------------------

SELECTION_CAMERA = "paper_room_oblique"
SELECTION_ROWS = {
    "office": {"target": "cup", "support": "desk", "seeds": (32, 33)},
    "kitchen": {"target": "jar", "support": "island", "seeds": (21, 22)},
    "bedroom": {"target": "alarm_clock", "support": "nightstand", "seeds": (44, 45)},
}


def selection_side_paths(tmp_path: Path, row_id: str, side: str) -> tuple[Path, Path]:
    """The fake delivery and render root of one side.

    The delivery sits under a per-row task directory on purpose:
    ``verify_selection`` reads the row's task as ``Path(delivery).parent.name``
    and rejects three rows that share one, exactly as the real layout has one
    task directory per task (``<tasks_root>/<run>``).
    """
    return (tmp_path / "deliveries" / row_id / f"seed_{side}",
            tmp_path / "renders" / f"{row_id}_{side}")


def selection_bundle(tmp_path: Path) -> Path:
    """Three rows of two sides each: a tiny fake delivery plus render per side.

    Every side gets its own task directory (so the rows stay distinct tasks), its
    own room seed, and its own shade of RGB, so a panel swapped between two sides
    changes bytes. The two sides of a row share the frozen closure -- target,
    support, invariant hash, invariant entity set and one satisfied hard relation
    -- because that is the fact the verifier recomputes.
    """
    bundle = tmp_path / "figure"
    (bundle / "panels").mkdir(parents=True)
    rows: dict[str, dict] = {}
    for index, (row_id, row) in enumerate(SELECTION_ROWS.items()):
        sides = {}
        for side_index, side in enumerate(("a", "b")):
            delivery, render_dir = selection_side_paths(tmp_path, row_id, side)
            image_dir = render_dir / "isaac"
            scene_version = f"scene-{row_id}-{side}"
            room_seed = row["seeds"][side_index]
            shade = 40 + 30 * index + 10 * side_index
            write_json(delivery / "delivery_manifest.json", {
                "scene_version": scene_version,
                "randomization": {"room_seed": room_seed,
                                  "invariant_spec_hash": f"core-{row_id}",
                                  "invariant_entity_ids": [row["target"], row["support"]]},
                "plan": {"target_id": row["target"], "support_id": row["support"]},
                "resolution": {"achieved": "draft"},
            })
            write_json(delivery / "scene_ir.json", {
                "schema_version": "test",
                "instances": [{"entity_id": row["target"],
                               "world_transform": {"translation": [0.5, 0.0, 0.9]}},
                              {"entity_id": row["support"],
                               "world_transform": {"translation": [0.5, 0.0, 0.5]}},
                              {"entity_id": "chair",
                               "world_transform": {"translation": [1.2, -0.4, 0.0]}}],
                "relations": [{"subject": row["target"], "object": row["support"],
                               "hard": True, "predicate": "ON", "satisfied": True}],
            })
            write_json(image_dir / "report.json", {
                "scene_version": scene_version, "status": "passed", "capture_state": "authored",
                "captures": [{"camera_id": SELECTION_CAMERA,
                              "depth_convention": "camera_z_meters",
                              "rgb_path": "rgb.png", "depth_path": "depth.npy"}],
            })
            write_json(render_dir / "provenance.json", {
                "source_delivery": str(delivery),
                "source_scene_version": scene_version,
                "report_sha256": audit.sha(image_dir / "report.json"),
            })
            Image.new("RGB", (6, 6), (shade, shade, shade)).save(image_dir / "rgb.png")
            np.save(image_dir / "depth.npy",
                    np.full((6, 6), 1.0 + shade / 100.0, dtype=np.float32))
            shutil.copy2(image_dir / "rgb.png",
                         bundle / "panels" / f"{row_id}_{side}_rgb.png")
            shutil.copy2(image_dir / "depth.npy",
                         bundle / "panels" / f"{row_id}_{side}_depth_m.npy")
            sides[side] = {
                "delivery": str(delivery), "render_dir": str(render_dir),
                "scene_version": scene_version, "room_seed": room_seed,
                "capture_state": "authored",
                "files": {"manifest": audit.sha(delivery / "delivery_manifest.json"),
                          "scene_ir": audit.sha(delivery / "scene_ir.json"),
                          "render_report": audit.sha(image_dir / "report.json"),
                          "rgb": audit.sha(image_dir / "rgb.png"),
                          "depth": audit.sha(image_dir / "depth.npy")},
            }
        rows[row_id] = sides
    write_json(bundle / "provenance.json", {
        "figure": "figure_room_alignment", "camera": SELECTION_CAMERA,
        "depth": {"convention": "camera_z_meters", "display_range_m": [0.0, 6.0],
                  "display_colormap": "turbo_r"},
        "rows": rows,
    })
    write_json(bundle / "selection.json", {
        "schema_version": "pvscene.paper-room-showcase/v1",
        "rows": [{"row_id": row_id} for row_id in SELECTION_ROWS],
    })
    write_json(bundle / "measurements.json",
               {row_id: {"core_closure_frozen": True} for row_id in SELECTION_ROWS})
    Image.new("RGB", (2162, 1400), "white").save(bundle / "figure_room_alignment.png")
    (bundle / "figure_room_alignment.pdf").write_bytes(b"%PDF-1.4 fixture")
    Image.new("RGB", (2162, 1400), "white").save(bundle / "figure_room_alignment_depth.png")
    return bundle


def refresh_selection_digests(bundle: Path) -> None:
    """Re-record every side's digests, so a negative can isolate one gap.

    The point of the negatives below is that the verifier *catches* a specific
    fact, not that it happens to also notice a stale digest. Tests that edit a
    delivery call this first; the one test whose subject is the digest itself
    deliberately does not.
    """
    path = bundle / "provenance.json"
    provenance = audit.read(path)
    for sides in provenance["rows"].values():
        for record in sides.values():
            delivery = Path(record["delivery"])
            image_dir = Path(record["render_dir"]) / "isaac"
            report = audit.read(image_dir / "report.json")
            capture = next(c for c in report["captures"]
                           if c["camera_id"] == provenance["camera"])
            record["files"] = {
                "manifest": audit.sha(delivery / "delivery_manifest.json"),
                "scene_ir": audit.sha(delivery / "scene_ir.json"),
                "render_report": audit.sha(image_dir / "report.json"),
                "rgb": audit.sha(image_dir / capture["rgb_path"]),
                "depth": audit.sha(image_dir / capture["depth_path"]),
            }
            # The render dir's binding records the same report digest, so an edit
            # to a report has to move both or the binding check fires as well and
            # the negative stops isolating its own gap. Only the digest is
            # re-recorded; a binding that names the wrong delivery stays wrong.
            binding_path = Path(record["render_dir"]) / "provenance.json"
            binding = audit.read(binding_path)
            binding["report_sha256"] = record["files"]["render_report"]
            write_json(binding_path, binding)
    write_json(path, provenance)


def test_selection_accepts_a_clean_bundle(tmp_path: Path) -> None:
    figure = selection_bundle(tmp_path)
    result = audit.verify_selection(figure)
    assert result["problems"] == []
    assert result["verified"] is True
    assert result["mode"] == "selection"
    assert result["panels_traceable"] == 5 * 6


def test_selection_rejects_swapped_panel(tmp_path: Path) -> None:
    figure = selection_bundle(tmp_path)
    assert audit.verify_selection(figure)["verified"] is True
    shutil.copy2(figure / "panels" / "kitchen_a_rgb.png",
                 figure / "panels" / "office_a_rgb.png")
    result = audit.verify_selection(figure)
    assert result["verified"] is False
    assert "SHOWCASE_PROVENANCE_MISSING(office,a:panel rgb)" in result["problems"]


def test_selection_rejects_core_drift(tmp_path: Path) -> None:
    figure = selection_bundle(tmp_path)
    assert audit.verify_selection(figure)["verified"] is True
    delivery, _ = selection_side_paths(tmp_path, "office", "b")
    manifest = audit.read(delivery / "delivery_manifest.json")
    manifest["plan"]["target_id"] = "laptop"
    write_json(delivery / "delivery_manifest.json", manifest)
    scene = audit.read(delivery / "scene_ir.json")
    scene["instances"].append({"entity_id": "laptop",
                               "world_transform": {"translation": [0.5, 0.4, 0.75]}})
    scene["relations"].append({"subject": "laptop", "object": "desk", "hard": True,
                               "predicate": "ON", "satisfied": True})
    write_json(delivery / "scene_ir.json", scene)
    refresh_selection_digests(figure)
    result = audit.verify_selection(figure)
    assert result["verified"] is False
    assert "SHOWCASE_CORE_DRIFT(office)" in result["problems"]
    assert result["core_frozen"]["office"] is False


def test_selection_rejects_same_room_seed(tmp_path: Path) -> None:
    figure = selection_bundle(tmp_path)
    assert audit.verify_selection(figure)["verified"] is True
    provenance_path = figure / "provenance.json"
    provenance = audit.read(provenance_path)
    shared = provenance["rows"]["office"]["a"]["room_seed"]
    provenance["rows"]["office"]["b"]["room_seed"] = shared
    write_json(provenance_path, provenance)
    delivery, _ = selection_side_paths(tmp_path, "office", "b")
    manifest_path = delivery / "delivery_manifest.json"
    manifest = audit.read(manifest_path)
    manifest["randomization"]["room_seed"] = shared
    write_json(manifest_path, manifest)
    refresh_selection_digests(figure)
    result = audit.verify_selection(figure)
    assert result["verified"] is False
    assert "SHOWCASE_NO_VARIATION(office:same room seed)" in result["problems"]
    assert result["room_seeds"]["office"] == [shared, shared]


def test_selection_rejects_changed_source(tmp_path: Path) -> None:
    figure = selection_bundle(tmp_path)
    assert audit.verify_selection(figure)["verified"] is True
    # Rewritten after the digests were recorded, and deliberately not re-recorded:
    # this test's subject is the digest, not the field that moved.
    delivery, _ = selection_side_paths(tmp_path, "office", "a")
    manifest_path = delivery / "delivery_manifest.json"
    manifest = audit.read(manifest_path)
    manifest["resolution"]["achieved"] = "meets_request"
    write_json(manifest_path, manifest)
    result = audit.verify_selection(figure)
    assert result["verified"] is False
    assert "SHOWCASE_PROVENANCE_MISSING(office,a:manifest)" in result["problems"]


def test_selection_rejects_settled_capture(tmp_path: Path) -> None:
    figure = selection_bundle(tmp_path)
    assert audit.verify_selection(figure)["verified"] is True
    report_path = selection_side_paths(tmp_path, "bedroom", "b")[1] / "isaac" / "report.json"
    report = audit.read(report_path)
    report["capture_state"] = "settled"
    write_json(report_path, report)
    refresh_selection_digests(figure)
    result = audit.verify_selection(figure)
    assert result["verified"] is False
    assert "SHOWCASE_SCENE_VERSION_MISMATCH(bedroom,b:settled)" in result["problems"]


def test_selection_rejects_render_bound_to_another_delivery(tmp_path: Path) -> None:
    """A render dir that is internally consistent but belongs to another delivery.

    Every digest in the record is a digest of this side's real files, so nothing
    above the binding can tell the difference -- only the render dir's own
    provenance says which delivery it rendered, and it says the wrong one.
    """
    figure = selection_bundle(tmp_path)
    assert audit.verify_selection(figure)["verified"] is True
    other, _ = selection_side_paths(tmp_path, "bedroom", "a")
    binding_path = selection_side_paths(tmp_path, "bedroom", "b")[1] / "provenance.json"
    binding = audit.read(binding_path)
    binding["source_delivery"] = str(other)
    write_json(binding_path, binding)
    result = audit.verify_selection(figure)
    assert result["verified"] is False
    assert "SHOWCASE_PROVENANCE_MISSING(bedroom,b:figure render binding)" in result["problems"]
    assert result["problems"] == ["SHOWCASE_PROVENANCE_MISSING(bedroom,b:figure render binding)"]


def test_selection_rejects_corrupt_render_binding_digest(tmp_path: Path) -> None:
    """Same binding, but its recorded report digest is no longer the report's."""
    figure = selection_bundle(tmp_path)
    assert audit.verify_selection(figure)["verified"] is True
    binding_path = selection_side_paths(tmp_path, "kitchen", "a")[1] / "provenance.json"
    binding = audit.read(binding_path)
    binding["report_sha256"] = "0" * 64
    write_json(binding_path, binding)
    result = audit.verify_selection(figure)
    assert result["verified"] is False
    assert "SHOWCASE_PROVENANCE_MISSING(kitchen,a:figure render binding)" in result["problems"]


def test_selection_rejects_missing_render_binding(tmp_path: Path) -> None:
    """The render dir carries no binding at all: a named gap, not a crash."""
    figure = selection_bundle(tmp_path)
    assert audit.verify_selection(figure)["verified"] is True
    (selection_side_paths(tmp_path, "office", "a")[1] / "provenance.json").unlink()
    result = audit.verify_selection(figure)
    assert result["verified"] is False
    assert "SHOWCASE_PROVENANCE_MISSING(office,a:figure render binding)" in result["problems"]
