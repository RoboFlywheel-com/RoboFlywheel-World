# SPDX-License-Identifier: Apache-2.0
"""SPEC-014's acceptance block names this file, and the criterion it carries is the one
a figure cannot grade for itself: an independent process recomputing requirement 5's
numbers from the raw panel arrays.

So every test here is a bundle with one engineered defect, and each asserts the *named*
gap that has to come out -- not merely that something failed. The fixture builds a whole
five-backend bundle under ``tmp_path``: 64x40 arrays, five fake engines, one column
meeting every threshold and the others breaching one each. The synthetic backends are
small on purpose -- what is under test is which gap is named, not how many pixels it
took. Nothing here reads the real renders, and nothing here reads anything outside
``tmp_path`` and the frozen conventions in ``scripts/figures/multisim/common.py``.
"""

from __future__ import annotations

import hashlib
import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFIER_PATH = module_path('rfw_w.scenes.validation.check_rigid_multisim')
COMMON_PATH = module_path('rfw_w.scenes.workers.multisim.common')

BACKENDS = ("genesis", "isaac", "mujoco", "pybullet", "sapien")
REFERENCE = "mujoco"
HEIGHT, WIDTH = 40, 64
CONTOUR_BAND_PX = 2
SCENE_VERSION = "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809"
INSTANCES = ("instance_a", "instance_b", "instance_c")
ENGINE_VERSIONS = {"genesis": "0.4.3", "isaac": "4.5.0-rc.36", "mujoco": "3.3.7",
                   "pybullet": "3.2.7", "sapien": "3.0.1"}


@pytest.fixture(scope="module")
def verifier():
    return importlib.import_module('rfw_w.scenes.validation.check_rigid_multisim')


@pytest.fixture(scope="module")
def common():
    """The frozen conventions the panels are built with, loaded by path so that nothing
    already on ``sys.path`` can shadow them."""
    return importlib.import_module('rfw_w.scenes.workers.multisim.common')


# ---------------------------------------------------------------------------
# The verifier's CLI, called the way the acceptance command calls it
# ---------------------------------------------------------------------------


def run_verifier(verifier, monkeypatch, bundle: Path, out: Path) -> tuple[int, dict]:
    """Invoke ``main()`` with the documented arguments and return (exit code,
    verification.json). Reading the file rather than main's return value means the
    test sees exactly what a caller of the command sees."""
    monkeypatch.setattr(sys, "argv", ["check_rigid_multisim.py",
                                      "--bundle", str(bundle),
                                      "--output-dir", str(out)])
    code = verifier.main()
    return code, json.loads((out / "verification.json").read_text(encoding="utf-8"))


def only_gap(problems: list[str], prefix: str) -> str:
    """The single problem starting with ``prefix``; asserting there is exactly one is
    part of the test, because a threshold that fires twice is a different defect."""
    matches = [problem for problem in problems if problem.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {prefix!r} gap, got {problems}"
    return matches[0]


def gap_value(gap: str) -> float:
    return float(gap.split(",")[-1].rstrip(")"))


# ---------------------------------------------------------------------------
# The synthetic bundle
# ---------------------------------------------------------------------------


def camera_record(common) -> dict:
    spec = common.CameraSpec(
        name="synthetic_room",
        eye=np.asarray([0.34, 0.34, 2.65], dtype=np.float64),
        target=np.asarray([2.4528, 1.84315, 0.65], dtype=np.float64),
        width=WIDTH,
        height=HEIGHT,
        focal_length_mm=4.9,
        horizontal_aperture_mm=20.955,
        vertical_aperture_mm=13.096875,
    )
    return spec.as_record()


def base_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[6:34, 8:56] = True
    return mask


def ramp() -> np.ndarray:
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    return 2.0 + 0.02 * (rows - 6) + 0.005 * (cols - 8)


def shifted_mask() -> np.ndarray:
    """A one-pixel mask displacement: overlap 1269 px, union 1419 px, IoU 0.894."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[7:35, 9:57] = True
    return mask


def holed_mask(rows: slice, cols: slice) -> np.ndarray:
    mask = base_mask()
    mask[rows, cols] = False
    return mask


def depth_from(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """NaN off the mask, exactly as ``common.save_depth_products`` leaves a product."""
    out = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    out[mask] = values[mask]
    return out


def rgb_image() -> np.ndarray:
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[..., 0] = (cols * 4) % 256
    rgb[..., 1] = (rows * 6) % 256
    rgb[..., 2] = 100
    return rgb


def erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        out = shrunk
    return out


def compare_to_reference(sample_mask, sample_depth, reference_mask, reference_depth) -> dict:
    """The fixture's own implementation of requirement 5's comparison, written here so
    the numbers in ``measurements.json`` are not produced by the code under test. Two
    independent implementations of one frozen definition must agree; that agreement is
    the check."""
    common_px = sample_mask & reference_mask & np.isfinite(sample_depth) & np.isfinite(reference_depth)
    union = sample_mask | reference_mask
    record = {
        "common_valid_pixels": int(common_px.sum()),
        "common_valid_fraction_of_reference": float(
            common_px.sum() / max(int(reference_mask.sum()), 1)),
        "mask_iou": float((sample_mask & reference_mask).sum() / max(int(union.sum()), 1)),
    }
    interior = erode(common_px, CONTOUR_BAND_PX)
    record["pixels_used_after_contour_band"] = int(interior.sum())
    record["pixels_set_aside_by_band"] = int(common_px.sum() - interior.sum())
    record["contour_band_px"] = CONTOUR_BAND_PX
    difference = np.abs(sample_depth[interior].astype(np.float64)
                        - reference_depth[interior].astype(np.float64))
    record["abs_depth_median_m"] = float(np.median(difference))
    record["abs_depth_p95_m"] = float(np.percentile(difference, 95))
    record["abs_depth_max_m"] = float(difference.max())
    return record


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_bundle(root: Path, common, *, mode: str = "good") -> dict:
    """Write a five-backend bundle under ``root``.

    ``good`` is one column per engine with every threshold met (three of the four
    comparisons are not perfect -- IoU 1.0, 0.9933 and 0.9814 -- so the thresholds are
    exercised, not merely satisfied by identical arrays).

    ``bad`` adds, on top of the good arrays, one engineered breach per column: a 50 mm
    median, a one-pixel mask displacement, a 90 mm self-check, a 5.1 mm lens, a stale
    scene version, an rgb.png tampered after provenance was written, a wrong
    measurements.json entry, and sub-print-resolution figures.
    """
    renders = root / "renders"
    bundle = root / "bundle"
    record = camera_record(common)
    base = base_mask()
    heights = ramp()
    rgb = rgb_image()

    masks = {
        "mujoco": base.copy(),
        "isaac": base.copy(),
        "genesis": base.copy(),
        "pybullet": shifted_mask() if mode == "bad" else holed_mask(slice(20, 23), slice(30, 33)),
        "sapien": holed_mask(slice(10, 15), slice(12, 17)),
    }
    depths = {backend: depth_from(masks[backend], heights) for backend in BACKENDS}
    depths["genesis"] = depth_from(base, heights + (0.05 if mode == "bad" else 0.006))
    self_checks = {
        "mujoco": 0.003,
        "isaac": 0.0045,
        "genesis": {"samples": 48, "median_m": 0.005, "p95_m": 0.011, "max_m": 0.02},
        "pybullet": {"samples": 64, "median_m": 0.007, "p95_m": 0.014, "max_m": 0.03},
        "sapien": 0.09 if mode == "bad" else 0.009,
    }
    scene_versions = {backend: SCENE_VERSION for backend in BACKENDS}
    if mode == "bad":
        scene_versions["sapien"] = "deadbeef" * 8

    for backend in BACKENDS:
        directory = renders / backend
        directory.mkdir(parents=True, exist_ok=True)
        mask = masks[backend]
        np.save(directory / "depth_visible.npy", depths[backend])
        np.save(directory / "depth_visible_mask.npy", mask)
        np.save(directory / "depth_collision.npy",
                depth_from(mask, heights * 1.05 + 0.5))
        np.save(directory / "depth_collision_mask.npy", mask)
        Image.fromarray(rgb).save(directory / "rgb.png")

        report_camera = dict(record)
        if mode == "bad" and backend == "genesis":
            report_camera["focal_length_mm"] = 5.1
            report_camera["fx"] = WIDTH * 5.1 / record["horizontal_aperture_mm"]
        report = {
            "backend": backend,
            "engine_version": ENGINE_VERSIONS[backend],
            "scene_version": scene_versions[backend],
            "object_ids": list(INSTANCES),
            "object_count": len(INSTANCES),
            "hidden_rule": {"hidden_roles": ["exterior"]},
            "camera": report_camera,
            "depth_products": {"visible": "depth_visible.npy",
                               "visible_mask": "depth_visible_mask.npy",
                               "collision": "depth_collision.npy",
                               "collision_mask": "depth_collision_mask.npy"},
            "geometry_source": {"rgb_and_visible_depth": "visual_mesh",
                                "collision_depth": "collision_mesh"},
            "status": "passed",
            "depth_self_check_m": self_checks[backend],
        }
        if backend == "isaac":
            report["capture_state"] = "authored"
            report["capture_state_source"] = "worker report"
        write_json(directory / "report.json", report)

    reference_record = compare_to_reference(masks[REFERENCE], depths[REFERENCE],
                                            masks[REFERENCE], depths[REFERENCE])
    backends_record = {}
    for backend in BACKENDS:
        entry = {
            "engine_version": ENGINE_VERSIONS[backend],
            "depth_self_check_m": self_checks[backend],
            "valid_pixel_fraction_visible": float(masks[backend].mean()),
            "valid_pixel_fraction_collision": float(masks[backend].mean()),
            "rgb_sha256": common.digest(renders / backend / "rgb.png"),
            "report_sha256": common.digest(renders / backend / "report.json"),
        }
        if backend != REFERENCE:
            entry["vs_reference"] = compare_to_reference(
                masks[backend], depths[backend], masks[REFERENCE], depths[REFERENCE])
        backends_record[backend] = entry
    if mode == "bad":
        backends_record["genesis"]["vs_reference"]["mask_iou"] = 0.5
    assert reference_record["mask_iou"] == 1.0

    bundle.mkdir(parents=True, exist_ok=True)
    panels = bundle / "panels"
    panels.mkdir(exist_ok=True)
    for backend in BACKENDS:
        Image.fromarray(rgb).save(panels / f"{backend}_rgb.png")
        Image.fromarray(common.colourise_depth(depths[backend])).save(
            panels / f"{backend}_depth_visible.png")
        Image.fromarray(common.depth_to_world_normal_rgb(
            depth_from(masks[backend], heights * 1.05 + 0.5),
            common.CameraSpec(name=record["name"], eye=np.asarray(record["eye"]),
                              target=np.asarray(record["target"]), width=WIDTH, height=HEIGHT,
                              focal_length_mm=record["focal_length_mm"],
                              horizontal_aperture_mm=record["horizontal_aperture_mm"],
                              vertical_aperture_mm=record["vertical_aperture_mm"]))).save(
            panels / f"{backend}_normal_collision.png")

    width_px = 800 if mode == "bad" else 2200
    for stem in ("figure_multisim_depth", "figure_multisim_normals"):
        Image.fromarray(np.zeros((width_px // 2, width_px, 3), dtype=np.uint8)).save(
            bundle / f"{stem}.png")
        (bundle / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n% fixture\n%%EOF\n")

    write_json(bundle / "camera.json", record)
    write_json(bundle / "common_scene.json", {
        "scene_version": SCENE_VERSION,
        "camera": record,
        "hidden_rule": {"hidden_roles": ["exterior"]},
        "objects": [
            {"object_id": instance, "kind": "furniture", "role": instance,
             "visual_mesh": f"meshes/{instance}.obj",
             "collision_mesh": f"meshes/{instance}.obj",
             "translation": [1.0 + index, 1.0, 0.0],
             "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]}
            for index, instance in enumerate(INSTANCES)
        ] + [{"object_id": "shell_floor", "kind": "shell", "role": "floor",
              "visual_mesh": "meshes/floor.obj", "collision_mesh": "meshes/floor.obj",
              "translation": [0.0, 0.0, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]}],
    })
    write_json(bundle / "measurements.json", {
        "reference_backend": REFERENCE,
        "scene_version": SCENE_VERSION,
        "camera": record,
        "depth_convention": "camera_z_meters",
        "backends": backends_record,
    })
    write_json(bundle / "provenance.json", {
        "figure": "figure_multisim_depth",
        "renders": str(renders),
        "scene": str(bundle / "common_scene.json"),
        "scene_version": SCENE_VERSION,
        "backend_order": list(BACKENDS),
        "reference_backend": REFERENCE,
        "panels": {
            backend: {"report": str(renders / backend / "report.json"),
                      "report_sha256": common.digest(renders / backend / "report.json"),
                      "rgb_sha256": common.digest(renders / backend / "rgb.png"),
                      "engine_version": ENGINE_VERSIONS[backend]}
            for backend in BACKENDS
        },
    })

    if mode == "bad":
        # The tamper happens after provenance was written, which is the only way the
        # digest check can be shown to be doing anything.
        tampered = rgb.copy()
        tampered[0, 0] = (255, 0, 0)
        Image.fromarray(tampered).save(renders / "isaac" / "rgb.png")

    return {"renders": renders, "bundle": bundle}


@pytest.fixture
def good(tmp_path, common) -> dict:
    return build_bundle(tmp_path / "good", common)


# ---------------------------------------------------------------------------
# The bundle the figure is supposed to produce
# ---------------------------------------------------------------------------


def test_the_fixture_itself_does_not_miss_the_reference(tmp_path, common, verifier, monkeypatch):
    """The engineered IoU violations are one-pixel displacements, so a broken fixture
    that compared MuJoCo with itself would look identical while proving nothing."""
    paths = build_bundle(tmp_path / "fixture", common)
    payload = json.loads((paths["bundle"] / "measurements.json").read_text())
    ious = {backend: entry.get("vs_reference", {}).get("mask_iou")
            for backend, entry in payload["backends"].items() if backend != REFERENCE}
    assert len(set(ious.values())) == 3, ious


def test_a_good_bundle_is_verified(good, verifier, monkeypatch, tmp_path):
    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "accept")
    assert code == 0
    assert payload["verified"] is True
    assert payload["problems"] == []
    assert payload["visual_review"] == "not_run(user preview pending)"
    table = {row["backend"]: row for row in payload["table"]}
    assert table[REFERENCE]["mask_iou"] is None
    for backend in BACKENDS:
        assert table[backend]["depth_self_check_kind"] in {"float", "dict"}
    assert table["pybullet"]["mask_iou"] == pytest.approx(0.9933, abs=1e-4)
    assert table["sapien"]["mask_iou"] == pytest.approx(0.9814, abs=1e-4)
    # Genesis is offset by 6 mm -- under every threshold, and visible in the number.
    assert table["genesis"]["abs_depth_median_m"] == pytest.approx(0.006, abs=1e-4)
    # The contour band is reported, because a claim about depth that hides how much of
    # the image it set aside is not auditable.
    assert table["genesis"]["pixels_set_aside_by_band"] > 0
    # The accounting has to be printed, not summarised: 28x48 of the 64x40 frame is
    # common to both columns, the 2 px band takes 288 of those, 1056 are compared.
    checks = payload["machine_checks"]
    assert checks["depth_comparison[genesis]"]["detail"] == (
        "1344 common px, 288 set aside by the 2 px contour band, 1056 compared")
    assert checks["depth_comparison[genesis]"]["status"] == "passed"


def test_the_unrun_self_check_of_the_reference_is_recorded_not_passed(good, verifier, monkeypatch,
                                                                     tmp_path):
    """The reference column has no comparison against itself; that is a ``not_run`` with
    a reason, never a pass."""
    _, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "accept")
    reference = [entry for entry in payload["not_run"]
                 if entry["check"] == f"depth_comparison[{REFERENCE}]"]
    assert len(reference) == 1
    assert "reference backend" in reference[0]["reason"]
    assert reference[0]["blocks_verified"] is False


# ---------------------------------------------------------------------------
# Negatives: every one asserts the gap by name, after proving the same fixture passes
# ---------------------------------------------------------------------------


def test_the_engineered_bad_bundle_names_every_violation(tmp_path, common, verifier, monkeypatch):
    good_code, good_payload = run_verifier(
        verifier, monkeypatch, build_bundle(tmp_path / "good", common)["bundle"],
        tmp_path / "accept_good")
    assert good_code == 0 and good_payload["verified"] is True, good_payload["problems"]

    bad = build_bundle(tmp_path / "bad", common, mode="bad")
    code, payload = run_verifier(verifier, monkeypatch, bad["bundle"], tmp_path / "accept_bad")
    assert code == 1
    assert payload["verified"] is False

    problems = payload["problems"]
    thresholds = sorted(p for p in problems
                        if p.startswith("MULTISIM_ALIGNMENT_BELOW_THRESHOLD("))
    others = sorted(p for p in problems
                    if not p.startswith("MULTISIM_ALIGNMENT_BELOW_THRESHOLD("))
    assert others == [
        "MULTISIM_CAMERA_MISMATCH(genesis)",
        "MULTISIM_FIGURE_INCOMPLETE(figure_multisim_depth: png_width=800<2160)",
        "MULTISIM_FIGURE_INCOMPLETE(figure_multisim_normals: png_width=800<2160)",
        "MULTISIM_MEASUREMENT_MISMATCH(genesis:mask_iou)",
        "MULTISIM_PROVENANCE_MISSING(isaac:panel pixels)",
        "MULTISIM_PROVENANCE_MISSING(isaac:rgb/report digest)",
        "MULTISIM_SCENE_VERSION_MISMATCH(sapien)",
    ]
    assert len(thresholds) == 3

    # Each breach is its own gap, arithmetic and all.
    median = only_gap(problems, "MULTISIM_ALIGNMENT_BELOW_THRESHOLD(abs_depth_median_m,")
    assert gap_value(median) == pytest.approx(0.05, abs=1e-4)
    assert gap_value(median) > verifier.ABS_DEPTH_MEDIAN_MAX_M
    self_check = only_gap(problems, "MULTISIM_ALIGNMENT_BELOW_THRESHOLD(depth_self_check_median_m,")
    assert gap_value(self_check) == pytest.approx(0.09, abs=1e-6)
    assert gap_value(self_check) > verifier.SELF_CHECK_TOLERANCE_M
    iou = only_gap(problems, "MULTISIM_ALIGNMENT_BELOW_THRESHOLD(mask_iou,")
    assert gap_value(iou) == pytest.approx(0.8942918, abs=1e-6)
    assert gap_value(iou) < verifier.MASK_IOU_MIN


def test_the_thresholds_are_the_ones_the_spec_prints(verifier):
    """The bar is not a preference: SPEC-014 requirement 5 prints these numbers."""
    assert verifier.MASK_IOU_MIN == 0.95
    assert verifier.ABS_DEPTH_MEDIAN_MAX_M == 0.02
    assert verifier.ABS_DEPTH_P95_MAX_M == 0.10
    assert verifier.CONTOUR_BAND_PX == 2
    assert verifier.FIGURE_PNG_MIN_WIDTH_PX == 2160
    assert verifier.REFERENCE_BACKEND == "mujoco"
    assert verifier.BACKEND_ORDER == ("genesis", "isaac", "mujoco", "pybullet", "sapien")


def test_a_wrong_measurements_json_is_caught_on_its_own(good, verifier, monkeypatch, tmp_path):
    """Nothing but the figure's own arithmetic is touched, so a verifier that read its
    numbers from measurements.json instead of recomputing them would pass this."""
    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "first")
    assert code == 0 and payload["verified"] is True

    measurements = json.loads((good["bundle"] / "measurements.json").read_text())
    measurements["backends"]["pybullet"]["vs_reference"]["abs_depth_p95_m"] = 0.5
    write_json(good["bundle"] / "measurements.json", measurements)

    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "second")
    assert code == 1
    assert payload["problems"] == ["MULTISIM_MEASUREMENT_MISMATCH(pybullet:abs_depth_p95_m)"]
    agreement = payload["measurements"]["backends"]["pybullet"]["agreement_with_figure"]
    assert agreement["abs_depth_p95_m"]["figure"] == 0.5
    assert agreement["abs_depth_p95_m"]["verifier"] == pytest.approx(0.0, abs=1e-9)
    assert agreement["abs_depth_p95_m"]["agrees"] is False
    # The rest of the table is still the verifier's own: the tamper moved one number.
    assert payload["table"][1]["mask_iou"] == pytest.approx(1.0)


def test_centimetres_where_metres_were_claimed_is_caught(good, verifier, monkeypatch, tmp_path):
    """A unit error is the mistake a depth figure is most likely to hide, and it is the
    one the thresholds have to shout about."""
    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "first")
    assert code == 0 and payload["verified"] is True

    depth_path = good["renders"] / "genesis" / "depth_visible.npy"
    np.save(depth_path, (np.load(depth_path) * 100.0).astype(np.float32))

    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "second")
    assert code == 1
    problems = payload["problems"]
    assert gap_value(only_gap(
        problems, "MULTISIM_ALIGNMENT_BELOW_THRESHOLD(abs_depth_median_m,")) > 100.0
    assert gap_value(only_gap(
        problems, "MULTISIM_ALIGNMENT_BELOW_THRESHOLD(abs_depth_p95_m,")) > 100.0
    assert "MULTISIM_MEASUREMENT_MISMATCH(genesis:abs_depth_median_m)" in problems
    assert "MULTISIM_PROVENANCE_MISSING(genesis:panel pixels)" in problems


def test_a_missing_rgb_is_a_blocking_not_run(good, verifier, monkeypatch, tmp_path):
    """A column whose render is gone cannot be recomputed at all, so the verdict is not
    "failed" but "could not be established" -- and that still must not certify."""
    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "first")
    assert code == 0 and payload["verified"] is True

    (good["renders"] / "isaac" / "rgb.png").unlink()

    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "second")
    assert code == 1
    assert payload["verified"] is False
    blocking = [entry for entry in payload["not_run"] if entry["blocks_verified"]]
    assert [entry["check"] for entry in blocking] == ["report[isaac]"]
    assert blocking[0]["gap"] == "MULTISIM_PROVENANCE_MISSING(isaac)"
    assert "MULTISIM_PROVENANCE_MISSING(isaac:rgb/report digest)" in payload["problems"]


def test_a_threshold_that_cannot_be_evaluated_blocks_a_clean_verdict(good, verifier, monkeypatch,
                                                                    tmp_path):
    """Zero problems, and still not verified. A report that does not carry the number
    requirement 5 asks about leaves the threshold unrun, and an unrun threshold may not
    hide behind a green verdict."""
    report_path = good["renders"] / "isaac" / "report.json"
    report = json.loads(report_path.read_text())
    del report["depth_self_check_m"]
    write_json(report_path, report)
    measurements = json.loads((good["bundle"] / "measurements.json").read_text())
    measurements["backends"]["isaac"]["depth_self_check_m"] = None
    write_json(good["bundle"] / "measurements.json", measurements)
    provenance = json.loads((good["bundle"] / "provenance.json").read_text())
    # Provenance is brought up to date deliberately: this is not a digest tamper, it is a
    # bundle honestly built from a report that omits the anchor residual.
    provenance["panels"]["isaac"]["report_sha256"] = hashlib.sha256(
        report_path.read_bytes()).hexdigest()
    write_json(good["bundle"] / "provenance.json", provenance)

    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], tmp_path / "accept")
    assert payload["problems"] == []
    assert code == 1
    assert payload["verified"] is False
    blocking = [entry for entry in payload["not_run"] if entry["blocks_verified"]]
    assert [entry["check"] for entry in blocking] == ["self_check[isaac]"]
    assert blocking[0]["gap"] == "MULTISIM_DEPTH_INVALID(isaac: no depth_self_check_m)"
    assert payload["machine_checks"]["thresholds[isaac]"]["status"] == "passed"


def test_a_non_empty_output_directory_is_refused(good, verifier, monkeypatch, tmp_path):
    """Evidence is never overwritten: a re-run writes a new directory or fails."""
    out = tmp_path / "accept"
    code, payload = run_verifier(verifier, monkeypatch, good["bundle"], out)
    assert code == 0 and payload["verified"] is True
    before = (out / "verification.json").read_text()

    with pytest.raises(SystemExit) as refusal:
        run_verifier(verifier, monkeypatch, good["bundle"], out)
    assert refusal.value.code == 2
    assert (out / "verification.json").read_text() == before


def test_a_bundle_without_provenance_cannot_be_traced(verifier, monkeypatch, tmp_path):
    """No provenance means no panel is traceable to a render, so the whole audit is
    reported as one named gap rather than as a pass with nothing behind it."""
    (tmp_path / "empty").mkdir()
    code, payload = run_verifier(verifier, monkeypatch, tmp_path / "empty", tmp_path / "accept")
    assert code == 1
    assert payload["verified"] is False
    assert payload["problems"] == ["MULTISIM_PROVENANCE_MISSING(provenance.json)"]
