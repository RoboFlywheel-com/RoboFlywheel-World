# SPDX-License-Identifier: Apache-2.0
"""Dollhouse camera math, layout/lighting overrides, and post-render metrics."""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = module_path('rfw_w.scenes.visualization.render_room_preview')


preview = importlib.import_module('rfw_w.scenes.visualization.render_room_preview')


# ---------------------------------------------------------------------------
# Dollhouse camera
# ---------------------------------------------------------------------------

def project(eye, target, focal, aperture, width, height, point) -> tuple[float, float]:
    """Pinhole projection into frame fractions, mirroring the worker's intrinsics."""
    forward = preview._unit(preview._sub(target, eye))
    right = preview._unit(preview._cross(forward, [0.0, 0.0, 1.0]))
    up = preview._cross(right, forward)
    v = preview._sub(point, eye)
    depth = preview._dot(v, forward)
    u = focal * preview._dot(v, right) / depth / aperture + 0.5
    vv = focal * preview._dot(v, up) / depth / (aperture * height / width) + 0.5
    return u, vv


def test_dollhouse_camera_geometry() -> None:
    room = {"polygon_xy": [[0, 0], [4, 0], [4, 4], [0, 4]], "height_m": 2.7,
            "floor_z_m": 0.0, "openings": []}
    camera = preview.dollhouse_camera(room, "se")
    distance = (2.2 * 2.7 - 0.4) / math.tan(math.radians(52.0))
    assert camera["eye"] == pytest.approx(
        [2.0 + distance / math.sqrt(2), 2.0 - distance / math.sqrt(2), 2.2 * 2.7])
    assert camera["target"] == pytest.approx([2.0, 2.0, 0.4])
    assert camera["resolution"] == [1600, 1200]
    assert camera["corner"] == "se"


def test_dollhouse_elevation_is_a_parameter() -> None:
    room = {"polygon_xy": [[0, 0], [2.5, 0], [2.5, 2.5], [0, 2.5]], "height_m": 2.6,
            "floor_z_m": 0.0, "openings": []}
    camera = preview.dollhouse_camera(room, "nw")
    assert camera["elevation_deg"] == pytest.approx(52.0)
    assert camera["eye"][2] == pytest.approx(2.2 * 2.6)
    shallow = preview.dollhouse_camera(room, "nw", elevation_deg=35.0)
    assert shallow["elevation_deg"] == pytest.approx(35.0)
    # Same eye height; a shallower view stands farther out, it does not sink.
    assert shallow["eye"][2] == pytest.approx(camera["eye"][2])
    assert shallow["eye_distance_m"] > camera["eye_distance_m"]
    assert shallow["eye_distance_m"] == pytest.approx(
        (2.2 * 2.6 - 0.4) / math.tan(math.radians(35.0)))


def test_dollhouse_room_fits_frame_with_margin() -> None:
    room = {"polygon_xy": [[0, 0], [4.906, 0], [4.906, 3.686], [0, 3.686]],
            "height_m": 2.7, "floor_z_m": 0.0, "openings": []}
    camera = preview.dollhouse_camera(room, "se")
    corners = [[x, y, z] for x, y in room["polygon_xy"] for z in (0.0, 2.7)]
    projected = [project(camera["eye"], camera["target"], camera["focal_length_mm"],
                         camera["horizontal_aperture_mm"], 1600, 1200, p) for p in corners]
    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    assert min(us) >= 0.0 and max(us) <= 1.0
    assert min(vs) >= 0.0 and max(vs) <= 1.0
    # The fitted dimension's extreme sits exactly at the margin band; the span
    # itself is smaller because perspective makes the corner extents asymmetric.
    extreme = max(max(abs(u - 0.5) for u in us), max(abs(v - 0.5) for v in vs))
    assert extreme == pytest.approx(0.5 / preview.DOLLHOUSE_MARGIN, abs=1e-6)
    assert max(max(us) - min(us), max(vs) - min(vs)) >= 0.75


def test_dollhouse_auto_corner_opposes_the_main_door() -> None:
    room = {"polygon_xy": [[0, 0], [4, 0], [4, 4], [0, 4]], "height_m": 2.7,
            "openings": [{"opening_id": "door_main", "kind": "door",
                          "center_xy": [2.0, 0.0], "width_m": 0.9, "height_m": 2.05}]}
    assert preview.dollhouse_corner_for(room) == "ne"
    room["openings"][0]["center_xy"] = [4.0, 1.0]
    assert preview.dollhouse_corner_for(room) == "nw"
    room["openings"] = []
    assert preview.dollhouse_corner_for(room) == "se"
    assert preview.dollhouse_corner_for(room, "sw") == "sw"
    with pytest.raises(ValueError, match="DOLLHOUSE_CORNER_INVALID"):
        preview.dollhouse_corner_for(room, "up")


def test_dollhouse_floor_fit_frames_tighter_than_wall_tops() -> None:
    room = {"polygon_xy": [[0, 0], [4.906, 0], [4.906, 3.686], [0, 3.686]],
            "height_m": 2.7, "floor_z_m": 0.0, "openings": []}
    with_tops = preview.dollhouse_camera(room, "se", elevation_deg=86.0)
    floor_only = preview.dollhouse_camera(room, "se", elevation_deg=86.0,
                                          fit_wall_tops=False)
    # Same vantage; fitting the floor only zooms in, and near a top view the
    # difference is large (the full-height far wall leans far outward).
    assert floor_only["eye"] == pytest.approx(with_tops["eye"])
    assert floor_only["focal_length_mm"] > with_tops["focal_length_mm"] * 1.2
    assert floor_only["fit_wall_tops"] is False
    projected = [project(floor_only["eye"], floor_only["target"],
                         floor_only["focal_length_mm"],
                         floor_only["horizontal_aperture_mm"], 1600, 1200, [x, y, 0.0])
                 for x, y in room["polygon_xy"]]
    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    assert min(us) >= 0.0 and max(us) <= 1.0
    assert min(vs) >= 0.0 and max(vs) <= 1.0


# ---------------------------------------------------------------------------
# Layout overrides
# ---------------------------------------------------------------------------

def request_fixture() -> dict:
    half = math.sqrt(0.5)
    return {
        "room": {"polygon_xy": [[0, 0], [4, 0], [4, 4], [0, 4]], "height_m": 2.7,
                 "floor_z_m": 0.0,
                 "openings": [{"opening_id": "door_main", "kind": "door",
                               "center_xy": [2.0, 0.0], "width_m": 0.9,
                               "height_m": 2.05, "sill_m": 0.0}]},
        "instances": [
            {"instance_id": "desk", "entity_id": "desk",
             "translation": [1.0, 1.0, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            {"instance_id": "cup", "entity_id": "cup",
             "translation": [1.0, 1.0, 0.75], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            {"instance_id": "chair", "entity_id": "chair",
             "translation": [2.0, 2.0, 0.0], "rotation_wxyz": [half, 0.0, 0.0, half]},
        ],
    }


# Bounds are in each instance's LOCAL frame (centred near the origin), matching
# the scene-IR convention the office delivery demonstrates.
BOUNDS = {"desk": ([-0.5, -0.5, 0.0], [0.5, 0.5, 0.75]),
          "cup": ([-0.1, -0.1, 0.0], [0.1, 0.1, 0.1]),
          "chair": ([-0.25, -0.25, 0.0], [0.25, 0.25, 0.5])}


def apply(request, overrides, protected=()):
    return preview.apply_layout_overrides(request, overrides, bounds_by_instance=BOUNDS,
                                          protected_ids=set(protected))


def test_layout_translation_is_applied_and_recorded() -> None:
    request = request_fixture()
    deltas = apply(request, {"desk": {"translation": [2.5, 1.0, 0.0]}})
    desk = next(i for i in request["instances"] if i["instance_id"] == "desk")
    assert desk["translation"] == [2.5, 1.0, 0.0]
    assert deltas == [{"instance_id": "desk", "entity_id": "desk",
                       "translation": {"from": [1.0, 1.0, 0.0], "to": [2.5, 1.0, 0.0]}}]


def test_layout_yaw_replaces_z_rotation() -> None:
    request = request_fixture()
    deltas = apply(request, {"chair": {"yaw_deg": 0.0}})
    chair = next(i for i in request["instances"] if i["instance_id"] == "chair")
    assert chair["rotation_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert deltas[0]["yaw_deg"] == {"from": pytest.approx(90.0), "to": 0.0}


def test_layout_remove_drops_the_instance() -> None:
    request = request_fixture()
    deltas = apply(request, {"chair": {"remove": True}})
    assert [i["instance_id"] for i in request["instances"]] == ["desk", "cup"]
    assert deltas == [{"instance_id": "chair", "entity_id": "chair", "removed": True}]


def test_layout_unknown_instance_is_named() -> None:
    with pytest.raises(ValueError, match="LAYOUT_UNKNOWN_INSTANCE"):
        apply(request_fixture(), {"bookshelf": {"translation": [1.0, 1.0, 0.0]}})


def test_layout_outside_room_is_refused() -> None:
    with pytest.raises(ValueError, match="LAYOUT_OUTSIDE_ROOM"):
        apply(request_fixture(), {"desk": {"translation": [10.0, 10.0, 0.0]}})


def test_layout_overlap_is_refused() -> None:
    with pytest.raises(ValueError, match="LAYOUT_OVERLAP"):
        apply(request_fixture(), {"chair": {"translation": [1.0, 1.0, 0.0]}})


def test_layout_stacked_instances_do_not_overlap() -> None:
    # The cup sits on the desk: XY overlap is total but the z-ranges only touch,
    # so moving it along the tabletop must pass.
    deltas = apply(request_fixture(), {"cup": {"translation": [1.2, 1.2, 0.75]}})
    assert deltas[0]["translation"]["to"] == [1.2, 1.2, 0.75]


def test_layout_doorway_must_stay_clear() -> None:
    with pytest.raises(ValueError, match="LAYOUT_BLOCKS_OPENING"):
        apply(request_fixture(), {"chair": {"translation": [2.0, 0.4, 0.0]}})
    # One metre in from the same door the strip no longer reaches.
    deltas = apply(request_fixture(), {"chair": {"translation": [2.0, 1.0, 0.0]}})
    assert deltas[0]["instance_id"] == "chair"


def test_layout_removing_a_task_target_is_refused() -> None:
    with pytest.raises(ValueError, match="LAYOUT_REMOVES_TARGET"):
        apply(request_fixture(), {"cup": {"remove": True}}, protected=("cup",))


def test_layout_yaw_on_tilted_instance_is_refused() -> None:
    request = request_fixture()
    request["instances"].append(
        {"instance_id": "wall_art", "entity_id": "wall_art",
         "translation": [0.1, 2.0, 1.4], "rotation_wxyz": [0.5, 0.5, 0.5, 0.5]})
    with pytest.raises(ValueError, match="LAYOUT_YAW_NOT_Z_ALIGNED"):
        apply(request, {"wall_art": {"yaw_deg": 10.0}})


def test_layout_moved_instance_needs_bounds() -> None:
    bounds = {k: v for k, v in BOUNDS.items() if k != "desk"}
    with pytest.raises(ValueError, match="LAYOUT_BOUNDS_MISSING"):
        preview.apply_layout_overrides(request_fixture(),
                                       {"desk": {"translation": [2.5, 1.0, 0.0]}},
                                       bounds_by_instance=bounds, protected_ids=set())


def test_layout_refusal_leaves_the_request_untouched() -> None:
    request = request_fixture()
    before = json.dumps(request, sort_keys=True)
    with pytest.raises(ValueError, match="LAYOUT_OUTSIDE_ROOM"):
        apply(request, {"desk": {"translation": [2.5, 1.0, 0.0]},
                        "chair": {"translation": [10.0, 10.0, 0.0]}})
    assert json.dumps(request, sort_keys=True) == before


# ---------------------------------------------------------------------------
# Layout overrides: declared support ("on") exemption
# ---------------------------------------------------------------------------

# The alarm clock rests on the nightstand: its AABB bottom (0.565) grazes the
# nightstand top (0.57), which is a stacked-on relation, not a collision.
STACKED_BOUNDS = {"nightstand": ([-0.25, -0.25, 0.0], [0.25, 0.25, 0.57]),
                  "alarm_clock": ([-0.075, -0.075, -0.075], [0.075, 0.075, 0.075])}


def stacked_fixture() -> tuple[dict, dict]:
    request = request_fixture()
    request["instances"].extend([
        {"instance_id": "nightstand", "entity_id": "nightstand",
         "translation": [2.0, 2.0, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
        {"instance_id": "alarm_clock", "entity_id": "alarm_clock",
         "translation": [1.0, 3.0, 0.64], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
    ])
    return request, {**BOUNDS, **STACKED_BOUNDS}


def test_layout_declared_support_pair_is_exempt() -> None:
    # Support and item move together; the grazed AABBs must not fire the gate.
    request, bounds = stacked_fixture()
    deltas = preview.apply_layout_overrides(
        request,
        {"nightstand": {"translation": [3.0, 2.0, 0.0]},
         "alarm_clock": {"translation": [3.0, 2.0, 0.64], "on": "nightstand"}},
        bounds_by_instance=bounds, protected_ids=set())
    clock = next(d for d in deltas if d["instance_id"] == "alarm_clock")
    assert clock["on"] == "nightstand"
    assert clock["translation"]["to"] == [3.0, 2.0, 0.64]


def test_layout_undeclared_stacked_overlap_still_fires() -> None:
    # The same two moves without the declaration are a collision.
    request, bounds = stacked_fixture()
    with pytest.raises(ValueError, match="LAYOUT_OVERLAP"):
        preview.apply_layout_overrides(
            request,
            {"nightstand": {"translation": [3.0, 2.0, 0.0]},
             "alarm_clock": {"translation": [3.0, 2.0, 0.64]}},
            bounds_by_instance=bounds, protected_ids=set())


def test_layout_exemption_does_not_cover_third_parties() -> None:
    # The exemption is pairwise: the moved nightstand may not land on the desk.
    request, bounds = stacked_fixture()
    with pytest.raises(ValueError, match="LAYOUT_OVERLAP"):
        preview.apply_layout_overrides(
            request,
            {"nightstand": {"translation": [1.0, 1.0, 0.0]},
             "alarm_clock": {"on": "nightstand"}},
            bounds_by_instance=bounds, protected_ids=set())


def test_layout_unknown_support_is_named() -> None:
    request, bounds = stacked_fixture()
    with pytest.raises(ValueError, match="LAYOUT_UNKNOWN_INSTANCE"):
        preview.apply_layout_overrides(
            request, {"alarm_clock": {"translation": [3.0, 2.0, 0.64], "on": "table"}},
            bounds_by_instance=bounds, protected_ids=set())


# ---------------------------------------------------------------------------
# Lighting overrides
# ---------------------------------------------------------------------------

def test_lighting_override_records_before_and_after() -> None:
    request = {"lighting": {"dome_intensity": 3599.0, "sun_intensity": 14940.0}}
    record = preview.apply_lighting_overrides(request, {"dome_intensity": 900.0})
    assert request["lighting"]["dome_intensity"] == 900.0
    assert request["lighting"]["sun_intensity"] == 14940.0
    assert record["applied"]["dome_intensity"] == {"from": 3599.0, "to": 900.0}
    assert record["before"]["dome_intensity"] == 3599.0
    assert record["after"]["dome_intensity"] == 900.0


def test_lighting_unknown_key_is_named() -> None:
    with pytest.raises(ValueError, match="LIGHTING_UNKNOWN_KEY"):
        preview.apply_lighting_overrides({}, {"sun": 1.0})


# ---------------------------------------------------------------------------
# Render metrics
# ---------------------------------------------------------------------------

def test_clipped_fraction_counts_only_fully_saturated_pixels() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = 255          # clipped
    rgb[0, 1] = 250          # clipped: the tonemapper plateaus below 255
    rgb[1, 0] = [249, 255, 255]  # not clipped: one channel below the line
    assert preview.clipped_fraction(rgb) == 0.5
    assert preview.clipped_fraction(np.full((4, 4, 3), 255, dtype=np.uint8)) == 1.0
    assert preview.clipped_fraction(np.zeros((4, 4, 3), dtype=np.uint8)) == 0.0


def test_depth_stats_match_the_worker_validity_definition() -> None:
    depth = np.array([[1.0, np.inf], [0.0, 2.0]], dtype=np.float32)
    stats = preview.depth_stats(depth)
    assert stats["valid_fraction"] == 0.5
    assert stats["min_m"] == 1.0
    assert stats["max_m"] == 2.0


def test_target_pixel_counts_resolve_renderer_ids() -> None:
    ids = np.array([[1, 1, 2], [0, 0, 2]], dtype=np.uint32)
    counts = preview.target_pixel_counts(ids, {"1": "cup", "2": "desk"}, ["cup", "desk"])
    assert counts == {"cup": 2, "desk": 2}
    # A target missing from the id map has zero pixels, not an error.
    assert preview.target_pixel_counts(ids, {}, ["cup"]) == {"cup": 0}


def test_evaluate_render_metrics_clean_render_passes() -> None:
    metrics = {"cameras": {"cam": {"clipped_fraction": 0.01,
                                   "depth": {"valid_fraction": 0.9},
                                   "target_pixels": {"cup": 120}}}}
    assert preview.evaluate_render_metrics(metrics) == []


def test_evaluate_render_metrics_names_each_gap() -> None:
    metrics = {"cameras": {"cam": {"clipped_fraction": 0.5,
                                   "depth": {"valid_fraction": 0.1},
                                   "target_pixels": {"cup": 0}}}}
    gaps = preview.evaluate_render_metrics(metrics)
    assert any(g.startswith("OVEREXPOSED(cam") for g in gaps)
    assert any(g.startswith("DEPTH_INVALID(cam") for g in gaps)
    assert any(g.startswith("TARGET_NOT_VISIBLE(cam, cup") for g in gaps)
    # Thresholds are arguments: a looser exposure ceiling clears the first gap.
    gaps = preview.evaluate_render_metrics(metrics, {"max_clipped_fraction": 0.6})
    assert not any(g.startswith("OVEREXPOSED") for g in gaps)


def test_depth_threshold_splits_by_view() -> None:
    # 0.35 valid depth is the measured dollhouse framing (sky around the room
    # silhouette); it must fail the level-view bar and pass the dollhouse bar.
    metrics = {"cameras": {"cam": {"clipped_fraction": 0.0,
                                   "depth": {"valid_fraction": 0.35}}}}
    assert any(g.startswith("DEPTH_INVALID") for g in preview.evaluate_render_metrics(metrics))
    assert preview.evaluate_render_metrics(metrics, preview.DOLLHOUSE_METRIC_THRESHOLDS) == []


def test_collect_records_the_thresholds_it_judged_against(tmp_path: Path) -> None:
    report = render_dir_fixture(tmp_path, white=False, target_pixels=8)
    metrics = preview.collect_render_metrics(report, ["cup"],
                                             thresholds=preview.DOLLHOUSE_METRIC_THRESHOLDS)
    assert metrics["thresholds"]["min_depth_valid_fraction"] == 0.25
    assert metrics["thresholds"]["max_clipped_fraction"] == 0.02
    assert metrics["gaps"] == []


def render_dir_fixture(tmp_path: Path, *, white: bool, target_pixels: int) -> Path:
    image_dir = tmp_path / "isaac"
    image_dir.mkdir(parents=True)
    shade = 255 if white else 120
    Image.new("RGB", (4, 4), (shade, shade, shade)).save(image_dir / "cam.png")
    np.save(image_dir / "cam_depth_m.npy", np.full((4, 4), 2.0, dtype=np.float32))
    ids = np.zeros((4, 4), dtype=np.uint32)
    ids.flat[:target_pixels] = 1
    np.save(image_dir / "cam_instance_ids.npy", ids)
    (image_dir / "report.json").write_text(json.dumps({
        "status": "passed",
        "captures": [{"camera_id": "paper_dollhouse", "rgb_path": "cam.png",
                      "depth_path": "cam_depth_m.npy",
                      "instance_id_path": "cam_instance_ids.npy",
                      "instance_id_map": {"1": "cup"}}],
    }))
    return image_dir / "report.json"


def test_collect_render_metrics_flags_a_blown_out_render(tmp_path: Path) -> None:
    # The fabricated all-white render is exactly the SPEC-013 failure mode and
    # must not pass the exposure gate.
    metrics = preview.collect_render_metrics(render_dir_fixture(tmp_path, white=True,
                                                                target_pixels=16), ["cup"])
    camera = metrics["cameras"]["paper_dollhouse"]
    assert camera["clipped_fraction"] == 1.0
    assert camera["depth"]["valid_fraction"] == 1.0
    assert camera["target_pixels"] == {"cup": 16}
    assert any(g.startswith("OVEREXPOSED(paper_dollhouse") for g in metrics["gaps"])


def test_collect_render_metrics_flags_a_missing_target(tmp_path: Path) -> None:
    metrics = preview.collect_render_metrics(render_dir_fixture(tmp_path, white=False,
                                                                target_pixels=0), ["cup"])
    assert metrics["cameras"]["paper_dollhouse"]["target_pixels"] == {"cup": 0}
    assert any(g.startswith("TARGET_NOT_VISIBLE(paper_dollhouse, cup") for g in metrics["gaps"])


def test_collect_render_metrics_accepts_a_clean_render(tmp_path: Path) -> None:
    metrics = preview.collect_render_metrics(render_dir_fixture(tmp_path, white=False,
                                                                target_pixels=8), ["cup"])
    assert metrics["gaps"] == []
