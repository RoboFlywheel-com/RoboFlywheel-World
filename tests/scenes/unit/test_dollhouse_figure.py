# SPDX-License-Identifier: Apache-2.0
"""Dollhouse collision companion: request transform, figure assembly, checker gaps."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

REPO = Path(__file__).resolve().parents[3]


preview = importlib.import_module('rfw_w.scenes.visualization.render_room_preview')
figure = importlib.import_module('rfw_w.scenes.visualization.build_dollhouse_figure')
checker = importlib.import_module('rfw_w.scenes.validation.check_dollhouse_figure')


# ---------------------------------------------------------------------------
# collision_request: the --with-collision request transform
# ---------------------------------------------------------------------------

def test_collision_request_substitutes_and_drops_the_field(tmp_path) -> None:
    collision = tmp_path / "cup_collision.obj"
    collision.write_bytes(b"o cup\n")
    request = {"room": {"polygon_xy": [[0, 0], [1, 0], [1, 1]], "height_m": 2.7},
               "cameras": [{"camera_id": "paper_dollhouse"}],
               "instances": [
                   {"instance_id": "cup", "entity_id": "cup",
                    "visual_uri": "/visual/cup.glb", "collision_uri": str(collision)},
                   {"instance_id": "rug", "entity_id": "rug",
                    "visual_uri": "/visual/rug.glb"},
               ]}
    variant, substitutions = preview.collision_request(request)
    # The original request is evidence; the transform works on a copy.
    assert request["instances"][0]["visual_uri"] == "/visual/cup.glb"
    cup, rug = variant["instances"]
    assert cup["visual_uri"] == str(collision)
    assert "collision_uri" not in cup
    assert rug["visual_uri"] == "/visual/rug.glb"
    assert "collision_uri" not in rug
    assert substitutions[0]["substituted"] is True
    assert substitutions[0]["visual_uri_before"] == "/visual/cup.glb"
    assert substitutions[0]["collision_sha256"] == preview.digest(collision)
    assert substitutions[1] == {"instance_id": "rug", "entity_id": "rug",
                                "substituted": False, "reason": "no_collision_uri"}
    # Camera and room ride along byte-identical: the passes differ only in geometry.
    assert variant["cameras"] == request["cameras"]
    assert variant["room"] == request["room"]


def test_collision_request_records_a_missing_mesh_file(tmp_path) -> None:
    request = {"instances": [{"instance_id": "cup", "entity_id": "cup",
                              "visual_uri": "/visual/cup.glb",
                              "collision_uri": str(tmp_path / "absent.obj")}]}
    variant, substitutions = preview.collision_request(request)
    assert variant["instances"][0]["visual_uri"] == str(tmp_path / "absent.obj")
    assert substitutions[0]["substituted"] is True
    assert substitutions[0]["collision_sha256"] is None


# ---------------------------------------------------------------------------
# shared_depth_range
# ---------------------------------------------------------------------------

def test_shared_depth_range_is_one_percentile_band_over_the_union() -> None:
    depths = [np.arange(1.0, 51.0), np.arange(51.0, 101.0)]
    vmin, vmax = figure.shared_depth_range(depths)
    union = np.arange(1.0, 101.0)
    assert vmin == pytest.approx(np.percentile(union, 1.0))
    assert vmax == pytest.approx(np.percentile(union, 99.0))


def test_shared_depth_range_ignores_invalid_pixels() -> None:
    depths = [np.array([1.0, 2.0, 3.0, np.inf, -1.0, 0.0, np.nan])]
    vmin, vmax = figure.shared_depth_range(depths)
    union = np.array([1.0, 2.0, 3.0])
    assert vmin == pytest.approx(np.percentile(union, 1.0))
    assert vmax == pytest.approx(np.percentile(union, 99.0))
    with pytest.raises(ValueError, match="DOLLHOUSE_DEPTH_INVALID"):
        figure.shared_depth_range([np.array([np.inf, 0.0])])


# ---------------------------------------------------------------------------
# Layout guards: the row label must fit its row, the footnote the page
# ---------------------------------------------------------------------------

def test_row_label_drops_the_verb_phrase() -> None:
    assert figure.row_label("office", "pick the cup — cup ON desk") == "office — cup ON desk"
    assert figure.row_label("kitchen", "fetch the jar — jar ON island") == \
        "kitchen — jar ON island"
    assert figure.row_label("office", None) == "office"
    assert figure.row_label("office", "") == "office"


def test_row_label_wraps_a_long_relation() -> None:
    label = figure.row_label("bedroom", "pick the clock — alarm clock ON bedside table")
    lines = label.splitlines()
    assert len(lines) == 2
    assert all(len(line) <= 24 for line in lines)


def test_footnote_wraps_inside_the_page() -> None:
    lines = figure.footnote_text().splitlines()
    # The first figure lost the footnote's tail at the canvas edge; the wrap
    # width is what keeps every line inside 183 mm at the footnote font size.
    assert len(lines) > 1
    assert all(len(line) <= figure.FOOTNOTE_WRAP_CHARS for line in lines)
    assert "provenance.json" in lines[-1]


# ---------------------------------------------------------------------------
# Assembly from synthetic render dirs, then the checker over the bundle
# ---------------------------------------------------------------------------

def make_render_dir(root: Path, room: str, seed: int) -> Path:
    """A minimal render_room_preview --with-collision output: two passes, one
    capture each, plus the provenance and metrics files the bundle cites."""
    render_dir = root / f"render_{room}"
    rng = np.random.default_rng(seed)
    for sub, depth_offset in (("isaac", 0.0), ("isaac_collision", 0.5)):
        image_dir = render_dir / sub
        image_dir.mkdir(parents=True)
        rgb = (rng.random((6, 8, 3)) * 200).astype(np.uint8)
        Image.fromarray(rgb).save(image_dir / "paper_dollhouse.png")
        depth = np.linspace(1.0, 4.0, 48, dtype=np.float32).reshape(6, 8) + depth_offset
        depth[0, 0] = np.inf
        np.save(image_dir / "paper_dollhouse_depth_m.npy", depth)
        (image_dir / "report.json").write_text(json.dumps({
            "status": "passed", "capture_state": "authored",
            "captures": [{"camera_id": "paper_dollhouse",
                          "rgb_path": "paper_dollhouse.png",
                          "depth_path": "paper_dollhouse_depth_m.npy",
                          "depth_convention": "camera_z_meters"}]}) + "\n")
    (render_dir / "provenance.json").write_text(json.dumps({"render": room}) + "\n")
    (render_dir / "render_metrics.json").write_text(json.dumps({
        "cameras": {"paper_dollhouse": {"clipped_fraction": 0.001,
                                        "depth": {"valid_fraction": 0.98}}},
        "gaps": [],
        "collision": {"cameras": {"paper_dollhouse": {"clipped_fraction": 0.002}},
                      "gaps": []}}) + "\n")
    return render_dir


@pytest.fixture
def bundle(tmp_path) -> Path:
    office = make_render_dir(tmp_path, "office", seed=1)
    kitchen = make_render_dir(tmp_path, "kitchen", seed=2)
    selection = {"rows": [{"room": "office", "task": "pick the cup — cup ON desk",
                           "render_dir": str(office)},
                          {"room": "kitchen", "task": "fetch the jar — jar ON island",
                           "render_dir": str(kitchen)}]}
    out = tmp_path / "bundle"
    figure.build(selection, out)
    return out


def problems_of(bundle: Path) -> list[str]:
    return checker.verify(bundle)["problems"]


def test_build_assembles_a_print_bundle(bundle) -> None:
    for name in ("figure_dollhouse.pdf", "figure_dollhouse.png", "provenance.json",
                 "measurements.json", "selection.json"):
        assert (bundle / name).is_file(), name
    for room in ("office", "kitchen"):
        for suffix in ("rgb.png", "depth_m.npy", "collision_rgb.png",
                       "collision_depth_m.npy"):
            assert (bundle / "panels" / f"{room}_{suffix}").is_file()
    provenance = json.loads((bundle / "provenance.json").read_text())
    assert sorted(provenance["rows"]) == ["kitchen", "office"]
    # The row label shortens the task to its relation; the full task string
    # survives here.
    assert provenance["rows"]["office"]["task"] == "pick the cup — cup ON desk"
    assert provenance["depth"]["display_range_m"][0] < \
        provenance["depth"]["display_range_m"][1]
    measurements = json.loads((bundle / "measurements.json").read_text())
    office = measurements["rows"]["office"]
    assert office["depth"]["visible"]["valid_fraction"] == pytest.approx(47 / 48)
    assert office["render_metrics"]["collision"]["clipped_fraction"] == 0.002


def test_figure_files_hit_print_dimensions(bundle) -> None:
    with Image.open(bundle / "figure_dollhouse.png") as picture:
        assert abs(picture.info["dpi"][0] - 300.0) < 1.0
        assert picture.width == pytest.approx(183 * 300 / 25.4, abs=2)
    import re
    match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]",
                      (bundle / "figure_dollhouse.pdf").read_bytes())
    assert match and abs(float(match.group(1)) - 183 * 72 / 25.4) < 2.0


def test_checker_accepts_a_clean_bundle(bundle) -> None:
    result = checker.verify(bundle)
    assert result["problems"] == []
    assert result["verified"] is True


def test_checker_fires_on_a_replaced_panel(bundle) -> None:
    (bundle / "panels" / "office_rgb.png").write_bytes(b"forged")
    assert any(p.startswith("PANEL_SOURCE_MISMATCH(office,visible_rgb)")
               for p in problems_of(bundle))


def test_checker_fires_on_a_missing_companion_panel(bundle) -> None:
    (bundle / "panels" / "kitchen_collision_depth_m.npy").unlink()
    assert any(p.startswith("MISSING_COMPANION(kitchen,collision_depth)")
               for p in problems_of(bundle))


def test_checker_fires_on_a_missing_collision_report(bundle) -> None:
    provenance = json.loads((bundle / "provenance.json").read_text())
    render_dir = Path(provenance["rows"]["office"]["render_dir"])
    (render_dir / "isaac_collision" / "report.json").unlink()
    assert any(p.startswith("MISSING_COMPANION(office,isaac_collision/report.json)")
               for p in problems_of(bundle))


def test_checker_fires_on_a_metrics_gap(bundle) -> None:
    provenance = json.loads((bundle / "provenance.json").read_text())
    metrics_path = Path(provenance["rows"]["office"]["render_dir"]) / "render_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["collision"]["gaps"] = ["OVEREXPOSED(paper_dollhouse, clipped_fraction=0.5)"]
    metrics_path.write_text(json.dumps(metrics) + "\n")
    assert any(p.startswith("METRICS_GAP(office,collision:OVEREXPOSED")
               for p in problems_of(bundle))


def test_checker_fires_on_stale_render_provenance(bundle) -> None:
    provenance = json.loads((bundle / "provenance.json").read_text())
    render_provenance = Path(provenance["rows"]["kitchen"]["render_dir"]) / "provenance.json"
    render_provenance.write_text(render_provenance.read_text() + " ")
    assert any(p.startswith("PROVENANCE_STALE(kitchen)") for p in problems_of(bundle))


def test_checker_fires_on_low_dpi_png(bundle) -> None:
    png = bundle / "figure_dollhouse.png"
    with Image.open(png) as picture:
        picture.save(png, dpi=(72, 72))
    assert any(p.startswith("FIGURE_DIMENSIONS(png") for p in problems_of(bundle))


def test_checker_fires_on_a_wrong_width_pdf(bundle) -> None:
    (bundle / "figure_dollhouse.pdf").write_bytes(
        b"%PDF-1.4\n1 0 obj << /MediaBox [0 0 200 300] >> endobj\n")
    assert any(p.startswith("FIGURE_DIMENSIONS(pdf width") for p in problems_of(bundle))
