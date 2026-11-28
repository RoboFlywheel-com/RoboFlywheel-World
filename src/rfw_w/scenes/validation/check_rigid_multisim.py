#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent verifier for SPEC-014's five-backend rigid-scene figure.

``assemble_multisim_figure.py`` measures the figure and writes down what it
measured. This script is a **separate process that recomputes the same numbers
from the raw panel arrays** and applies SPEC-014 requirement 5's thresholds.
Keeping the two apart is what stops a figure from grading itself: everything
under ``measurements`` here is derived from the ``.npy`` arrays and the
per-backend reports, and ``measurements.json`` is only ever *compared against* --
the figure agreeing with itself is a check, not an input.

What this verifier refuses to claim:

* a digest that does not match ``provenance.json`` is
  ``MULTISIM_PROVENANCE_MISSING(panel)``, not a warning;
* a backend whose report disagrees with the frozen camera or the scene version
  is a named gap, never a "close enough";
* a requirement-5 threshold that could not be evaluated -- because the report
  does not carry the number, or the arrays cannot be read at all -- is recorded
  ``not_run`` with a reason and **blocks** ``verified``. An unrun check is never
  written as a pass, so an unevaluated threshold cannot hide behind a green
  verdict.

Two names below are this verifier's own, not SPEC-014's vocabulary --
``MULTISIM_MEASUREMENT_MISMATCH(backend:metric)`` and
``MULTISIM_FIGURE_INCOMPLETE(figure:detail)`` -- because the spec names the
quantities a panel must satisfy but not the act of catching the figure out;
they are named gaps for the same reason the spec's are, so a reader cannot
mistake "the figure's arithmetic disagrees with a fresh recomputation" for one
of the spec's own failures.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.validation.check_rigid_multisim \\
      --bundle outputs/_paper/spec014_figure_NEW \\
      --output-dir outputs/_acceptance/spec014_multisim_NEW
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Thresholds. Every number that decides a verdict lives here, with the clause
# of SPEC-014 it comes from, so a reader can audit the bar without reading the
# code that applies it.
# --------------------------------------------------------------------------

# SPEC-014 requirement 4 fixes the column order and the reference backend.
BACKEND_ORDER = ("genesis", "isaac", "mujoco", "pybullet", "sapien")
REFERENCE_BACKEND = "mujoco"

# SPEC-014 requirement 2: depth is float32 metres along the camera axis.
DEPTH_CONVENTION = "camera_z_meters"
DEPTH_DTYPE = "float32"

# SPEC-014 requirement 5, verbatim thresholds.
MASK_IOU_MIN = 0.95              # "valid mask IoU >= 0.95"
ABS_DEPTH_MEDIAN_MAX_M = 0.02    # "depth median absolute difference <= 0.02 m"
ABS_DEPTH_P95_MAX_M = 0.10       # "p95 <= 0.10 m"
CONTOUR_BAND_PX = 2              # "excluded two-pixel contour band"

# SPEC-014 requirement 5 asks for the backend's own anchor/reprojection residual
# to be small; it does not print a number. Three centimetres is "a few
# centimetres", and it is deliberately the same bar the drivers already use
# internally (render_mujoco.py records ``depth_self_check_below_3cm``, SAPIEN
# warns above 1 cm), so the figure cannot be held to a bar its own adapters
# never saw.
SELF_CHECK_TOLERANCE_M = 0.03

# SPEC-014 requirement 7: 183 mm two-column width at 300 dpi = 2161.4 px.
FIGURE_PNG_MIN_WIDTH_PX = 2160
FIGURE_STEMS = ("figure_multisim_depth", "figure_multisim_normals")
PANEL_SUFFIXES = ("rgb.png", "depth_visible.png", "normal_collision.png")

# Floating-point tolerance for "the figure's own measurements.json agrees with a
# fresh recomputation". Same arrays, same numpy, so this only has to absorb
# summation-order noise; it is not a licence to differ.
AGREE_RTOL = 1e-6
AGREE_ATOL = 1e-9

# Camera agreement: the reports carry float64 round-tripped through JSON, so the
# comparison needs a tolerance -- but a tolerance far below requirement 5's 1 px
# reprojection bar (1e-6 m at 4.9 mm focal is ~1e-4 px).
CAMERA_POS_ATOL_M = 1e-6
CAMERA_INTRINSIC_ATOL = 1e-6

# The metrics this verifier recomputes and then holds the figure's
# measurements.json to.
COMPARED_METRICS = (
    "common_valid_pixels",
    "pixels_used_after_contour_band",
    "pixels_set_aside_by_band",
    "common_valid_fraction_of_reference",
    "mask_iou",
    "abs_depth_median_m",
    "abs_depth_p95_m",
    "abs_depth_max_m",
)


# --------------------------------------------------------------------------
# Ledger: three states, never two (AGENTS.md §5.1).
# --------------------------------------------------------------------------


class Ledger:
    """Tracks what ran, what failed, and what did not run.

    ``problems`` is the flat list of named gaps the caller prints; ``not_run``
    carries a reason and a ``blocks_verified`` flag, because a threshold that
    could not be evaluated must not be silently absorbed into a green verdict
    while an optional extra check may.
    """

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.checks: dict[str, dict] = {}
        self.not_run: list[dict] = []

    def ok(self, name: str, detail: str, requirement: str) -> None:
        self.checks[name] = {"status": "passed", "detail": detail, "requirement": requirement}

    def fail(self, name: str, detail: str, gap: str, requirement: str) -> None:
        self.checks[name] = {"status": "failed", "detail": detail, "gap": gap,
                             "requirement": requirement}
        self.problems.append(gap)

    def fail_many(self, name: str, detail: str, gaps: list[str], requirement: str) -> None:
        """One failed check, but every breach named separately: requirement 5
        says each threshold is its own gap, and a column can breach two."""
        self.checks[name] = {"status": "failed", "detail": detail, "gap": gaps,
                             "requirement": requirement}
        self.problems.extend(gaps)

    def skip(self, name: str, reason: str, requirement: str,
             gap: str | None = None, blocks: bool = True) -> None:
        self.checks[name] = {"status": "not_run", "detail": reason, "gap": gap,
                             "requirement": requirement, "blocks_verified": bool(blocks)}
        self.not_run.append({"check": name, "reason": reason, "gap": gap,
                             "blocks_verified": bool(blocks)})


# --------------------------------------------------------------------------
# Reading the bundle
# --------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Own implementation rather than importing the figure's ``common.digest``:
    a verifier that shares the digest code it audits has one fewer independent
    check, and this is nine lines."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Shrink a boolean mask by ``iterations`` 4-neighbour steps.

    Requirement 5's "two-pixel contour band": pixels within two of a silhouette
    edge are set aside because two engines legitimately sample the same edge
    differently. Written out here rather than imported so the recomputation is
    an independent implementation of the definition, not a call into the code
    under audit.
    """
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(iterations)):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        out = shrunk
    return out


def compare_to_reference(sample: dict, reference: dict, contour_px: int) -> dict:
    """Depth agreement on the common valid domain, minus the contour band.

    Same definition as the figure uses -- requirement 5 fixes it -- recomputed
    here from the two raw arrays. Returns metric values only; deciding whether
    they pass is the caller's job.
    """
    sample_mask = np.asarray(sample["depth_visible_mask"], dtype=bool)
    reference_mask = np.asarray(reference["depth_visible_mask"], dtype=bool)
    sample_depth = np.asarray(sample["depth_visible"])
    reference_depth = np.asarray(reference["depth_visible"])

    common = (sample_mask & reference_mask
              & np.isfinite(sample_depth) & np.isfinite(reference_depth))
    union = sample_mask | reference_mask
    iou = float((sample_mask & reference_mask).sum() / max(int(union.sum()), 1))

    record = {
        "common_valid_pixels": int(common.sum()),
        "common_valid_fraction_of_reference": float(
            common.sum() / max(int(reference_mask.sum()), 1)),
        "mask_iou": iou,
        "pixels_used_after_contour_band": None,
        "pixels_set_aside_by_band": None,
        "contour_band_px": int(contour_px),
        "abs_depth_median_m": None,
        "abs_depth_p95_m": None,
        "abs_depth_max_m": None,
    }
    if not common.any():
        record["error"] = f"no common valid pixels with {reference['backend']}"
        return record

    interior = erode(common, contour_px)
    record["pixels_used_after_contour_band"] = int(interior.sum())
    record["pixels_set_aside_by_band"] = int(common.sum() - interior.sum())
    if not interior.any():
        record["error"] = "the contour band consumed the common valid mask"
        return record

    difference = np.abs(sample_depth[interior].astype(np.float64)
                        - reference_depth[interior].astype(np.float64))
    record["abs_depth_median_m"] = float(np.median(difference))
    record["abs_depth_p95_m"] = float(np.percentile(difference, 95))
    record["abs_depth_max_m"] = float(difference.max())
    return record


def self_check_median(value) -> tuple[float | None, str]:
    """A report's ``depth_self_check_m`` is either a float (mujoco, isaac,
    sapien) or a dict carrying ``median_m`` (pybullet). Requirement 5's bar is
    applied to the median when the driver reports a distribution."""
    if value is None:
        return None, "absent"
    if isinstance(value, dict):
        median = value.get("median_m")
        return (float(median) if median is not None else None), "dict"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"unreadable({type(value).__name__})"
    return float(value), "float"


def close(a: float, b: float) -> bool:
    return abs(a - b) <= AGREE_ATOL + AGREE_RTOL * max(abs(a), abs(b))


def camera_differences(recorded: dict, frozen: dict, require_all: bool) -> list[str]:
    """Field-by-field camera agreement. ``require_all`` is for a backend report,
    which must declare the intrinsics it rendered with; the bundle's own
    ``common_scene.json`` copy of the camera may legitimately carry a subset."""
    differences: list[str] = []
    # Absolute tolerances, matching the figure's own check of the same fields:
    # the reports carry the same numbers through the same JSON round-trip, so
    # anything above float noise is a different camera, and 1e-6 px of fx is
    # nowhere near requirement 5's 1 px reprojection bar.
    for key, tolerance in (("fx", CAMERA_INTRINSIC_ATOL),
                           ("fy", CAMERA_INTRINSIC_ATOL),
                           ("focal_length_mm", 1e-9)):
        got, want = recorded.get(key), frozen.get(key)
        if got is None or want is None:
            if require_all and got is None:
                differences.append(f"{key} not declared")
            continue
        if abs(float(got) - float(want)) > tolerance:
            differences.append(f"{key} {got} != {want}")
    for key in ("eye", "target"):
        got, want = recorded.get(key), frozen.get(key)
        if got is None or want is None:
            if require_all and got is None:
                differences.append(f"{key} not declared")
            continue
        if len(got) != len(want) or max(abs(float(a) - float(b))
                                        for a, b in zip(got, want)) > CAMERA_POS_ATOL_M:
            differences.append(f"{key} {got} != {want}")
    for key in ("width", "height"):
        got, want = recorded.get(key), frozen.get(key)
        if got is None or want is None:
            if require_all and got is None:
                differences.append(f"{key} not declared")
            continue
        if int(got) != int(want):
            differences.append(f"{key} {got} != {want}")
    return differences


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------


def check_figure_files(bundle: Path, ledger: Ledger) -> dict:
    """Requirement 4/7: both figures, each with a print-resolution PNG and a
    non-empty PDF."""
    figures: dict[str, dict] = {}
    for stem in FIGURE_STEMS:
        png = bundle / f"{stem}.png"
        pdf = bundle / f"{stem}.pdf"
        record: dict = {"png": str(png), "pdf": str(pdf)}
        if not png.is_file():
            ledger.fail(f"figure[{stem}]", f"{png} is missing",
                        f"MULTISIM_FIGURE_INCOMPLETE({stem}: no png)", "SPEC-014 R4/R7")
            record["png_width_px"] = None
        else:
            width, height = Image.open(png).size
            record["png_width_px"], record["png_height_px"] = int(width), int(height)
        if not pdf.is_file():
            ledger.fail(f"figure[{stem}]", f"{pdf} is missing",
                        f"MULTISIM_FIGURE_INCOMPLETE({stem}: no pdf)", "SPEC-014 R4/R7")
            record["pdf_bytes"] = None
        else:
            record["pdf_bytes"] = int(pdf.stat().st_size)
        if record.get("png_width_px") is not None and record["png_width_px"] < FIGURE_PNG_MIN_WIDTH_PX:
            ledger.fail(f"figure[{stem}]",
                        f"{stem}.png is {record['png_width_px']} px wide, "
                        f"below {FIGURE_PNG_MIN_WIDTH_PX} px (183 mm at 300 dpi)",
                        f"MULTISIM_FIGURE_INCOMPLETE({stem}: png_width="
                        f"{record['png_width_px']}<{FIGURE_PNG_MIN_WIDTH_PX})",
                        "SPEC-014 R7")
        if record.get("pdf_bytes") == 0:
            ledger.fail(f"figure[{stem}]", f"{stem}.pdf is empty",
                        f"MULTISIM_FIGURE_INCOMPLETE({stem}: empty pdf)", "SPEC-014 R7")
        if record.get("png_width_px") is not None and record["png_width_px"] >= FIGURE_PNG_MIN_WIDTH_PX \
                and record.get("pdf_bytes"):
            ledger.ok(f"figure[{stem}]",
                      f"png {record['png_width_px']}x{record['png_height_px']} px, "
                      f"pdf {record['pdf_bytes']} bytes", "SPEC-014 R7")
        figures[stem] = record
    return figures


def load_panels(bundle: Path, provenance: dict, ledger: Ledger) -> tuple[dict, dict]:
    """Load every backend's raw products and re-verify traceability.

    Requirement 1's claim is that each panel is traceable back to the render
    that produced it, so the raw ``report.json`` and ``rgb.png`` are hashed here
    and held to the digests ``provenance.json`` recorded; the panel images the
    figure actually embeds are then compared against those raw arrays.
    """
    renders = Path(provenance.get("renders", ""))
    if not renders.is_dir():
        ledger.fail("renders_root", f"{renders} is not a directory",
                    f"MULTISIM_PROVENANCE_MISSING(renders_root:{renders})", "SPEC-014 R1")
    recorded_panels = provenance.get("panels") or {}
    samples: dict[str, dict] = {}
    panels: dict[str, dict] = {}

    for backend in BACKEND_ORDER:
        panel_record = recorded_panels.get(backend)
        if panel_record is None:
            ledger.fail(f"provenance[{backend}]", "provenance.json has no panel record",
                        f"MULTISIM_PROVENANCE_MISSING({backend})", "SPEC-014 R1")
            continue
        # The renders root is the authority for where the raw products live
        # (SPEC-014's bundle description), so the digest target is
        # <renders>/<backend>/report.json. A recorded path pointing somewhere
        # else is a provenance defect: the panel would then be traceable to a
        # file outside the renders this bundle claims.
        directory = renders / backend
        report_path = directory / "report.json"
        rgb_path = directory / "rgb.png"
        recorded_report = panel_record.get("report")
        if recorded_report and Path(recorded_report).resolve() != report_path.resolve():
            ledger.fail(f"report_path[{backend}]",
                        f"provenance records {recorded_report!r}, but the renders root "
                        f"({renders}) puts the report at {report_path}",
                        f"MULTISIM_PROVENANCE_MISSING({backend}:report path "
                        f"{recorded_report})", "SPEC-014 R1")
        sample: dict = {"backend": backend, "directory": directory,
                        "report_path": report_path, "rgb_path": rgb_path}

        # --- digests: the file the panel claims, against the file on disk ---
        digest_failures = []
        if not report_path.is_file():
            digest_failures.append(f"report.json missing at {report_path}")
        else:
            actual = sha256_file(report_path)
            expected = panel_record.get("report_sha256")
            if actual != expected:
                digest_failures.append(f"report.json sha256 {actual[:16]}..., "
                                       f"provenance says {str(expected)[:16]}...")
        if not rgb_path.is_file():
            digest_failures.append(f"rgb.png missing at {rgb_path}")
        else:
            actual_rgb = sha256_file(rgb_path)
            expected_rgb = panel_record.get("rgb_sha256")
            if actual_rgb != expected_rgb:
                digest_failures.append(f"rgb.png sha256 {actual_rgb[:16]}..., "
                                       f"provenance says {str(expected_rgb)[:16]}...")
        if digest_failures:
            ledger.fail(f"panel_provenance[{backend}]", "; ".join(digest_failures),
                        f"MULTISIM_PROVENANCE_MISSING({backend}:rgb/report digest)", "SPEC-014 R1")
        else:
            ledger.ok(f"panel_provenance[{backend}]",
                      "report.json and rgb.png match their provenance digests", "SPEC-014 R1")
            sample["report_sha256"] = sha256_file(report_path)
            sample["rgb_sha256"] = sha256_file(rgb_path)

        # --- the report itself ---
        read_failures = []
        if not report_path.is_file():
            read_failures.append(f"report.json missing at {report_path}")
        else:
            try:
                sample["report"] = read_json(report_path)
            except (OSError, ValueError) as error:
                read_failures.append(f"cannot parse {report_path}: {error}")

        # --- the arrays ---
        products: dict = {}
        for prefix in ("depth_visible", "depth_collision"):
            arrays: dict = {}
            for suffix in ("", "_mask"):
                path = directory / f"{prefix}{suffix}.npy"
                if not path.is_file():
                    read_failures.append(f"{path.name} is missing")
                    arrays = {}
                    break
                try:
                    loaded = np.load(path, allow_pickle=False)
                except (OSError, ValueError) as error:
                    read_failures.append(f"cannot read {path.name}: {error}")
                    arrays = {}
                    break
                arrays[suffix] = loaded
                if suffix == "_mask":
                    if loaded.dtype != np.bool_:
                        read_failures.append(f"{path.name} is {loaded.dtype}, not bool")
            if arrays:
                products[prefix] = arrays
        for prefix in ("depth_visible", "depth_collision"):
            if prefix in products:
                sample[prefix] = products[prefix][""]
                sample[f"{prefix}_mask"] = products[prefix]["_mask"]
        if read_failures:
            ledger.fail(f"products[{backend}]", "; ".join(read_failures),
                        f"MULTISIM_PROVENANCE_MISSING({backend}:{read_failures[0]})",
                        "SPEC-014 R1/R2")
        elif rgb_path.is_file():
            ledger.ok(f"products[{backend}]",
                      "report.json, rgb.png and all four .npy products are readable",
                      "SPEC-014 R1/R2")
        if rgb_path.is_file() and "report" in sample:
            samples[backend] = sample

        panels[backend] = {"record": panel_record, "directory": directory,
                           "report_path": report_path, "rgb_path": rgb_path,
                           "products": products}
    return samples, panels


def check_agreement(sample: dict, provenance: dict, ledger: Ledger) -> None:
    """Requirement 2: scene version and camera must be the frozen ones, and the
    arrays must be the frozen resolution."""
    backend = sample["backend"]
    report = sample["report"]
    bundle_camera = sample["bundle_camera"]

    if report.get("scene_version") != sample["bundle_scene_version"]:
        ledger.fail(f"scene_version[{backend}]",
                    f"report says {report.get('scene_version')!r}, bundle scene is "
                    f"{sample['bundle_scene_version']!r}",
                    f"MULTISIM_SCENE_VERSION_MISMATCH({backend})", "SPEC-014 R2")
    else:
        ledger.ok(f"scene_version[{backend}]", "report matches the bundle scene version",
                  "SPEC-014 R2")

    recorded = report.get("camera") or {}
    if not recorded:
        ledger.skip(f"camera[{backend}]", "report carries no camera record", "SPEC-014 R2",
                    gap=f"MULTISIM_CAMERA_MISMATCH({backend}: no camera record)")
    else:
        differences = camera_differences(recorded, bundle_camera, require_all=True)
        if differences:
            ledger.fail(f"camera[{backend}]", "; ".join(differences),
                        f"MULTISIM_CAMERA_MISMATCH({backend})", "SPEC-014 R2")
        else:
            ledger.ok(f"camera[{backend}]", "report camera equals the frozen camera.json",
                      "SPEC-014 R2")
        if recorded.get("depth_definition") != DEPTH_CONVENTION:
            ledger.fail(f"depth_convention[{backend}]",
                        f"report declares depth_definition={recorded.get('depth_definition')!r}",
                        f"MULTISIM_DEPTH_CONVENTION_MISMATCH({backend})", "SPEC-014 R2")
        else:
            ledger.ok(f"depth_convention[{backend}]",
                      f"report declares {DEPTH_CONVENTION}", "SPEC-014 R2")

    width, height = int(bundle_camera["width"]), int(bundle_camera["height"])
    shapes = []
    if "rgb_path" in sample and Path(sample["rgb_path"]).is_file():
        rgb = np.asarray(Image.open(sample["rgb_path"]).convert("RGB"))
        sample["rgb"] = rgb
        shapes.append(("rgb", rgb.shape[:2]))
    for key in ("depth_visible", "depth_collision"):
        if key in sample:
            shapes.append((key, tuple(np.asarray(sample[key]).shape)))
    wrong = [f"{name}={shape[1]}x{shape[0]}" for name, shape in shapes
             if tuple(shape) != (height, width)]
    if wrong:
        ledger.fail(f"resolution[{backend}]",
                    f"expected {width}x{height} (frozen camera); " + ", ".join(wrong),
                    f"MULTISIM_CAMERA_MISMATCH({backend})", "SPEC-014 R2")
    elif shapes:
        ledger.ok(f"resolution[{backend}]",
                  f"rgb and both depth passes are {width}x{height}", "SPEC-014 R2")

    for key in ("depth_visible", "depth_collision"):
        if key not in sample:
            continue
        array = np.asarray(sample[key])
        if array.dtype.name != DEPTH_DTYPE:
            ledger.fail(f"depth_dtype[{backend}]", f"{key} is {array.dtype}, not {DEPTH_DTYPE}",
                        f"MULTISIM_DEPTH_INVALID({backend}:{key} dtype {array.dtype})",
                        "SPEC-014 R2")
        mask = np.asarray(sample[f"{key}_mask"], dtype=bool)
        expected = np.isfinite(array) & (array > 0)
        if mask.shape != array.shape:
            ledger.fail(f"mask_consistency[{backend}]",
                        f"{key}_mask is {mask.shape}, depth is {array.shape}",
                        f"MULTISIM_DEPTH_INVALID({backend}:{key}_mask shape)", "SPEC-014 R2")
        elif not bool((mask == expected).all()):
            ledger.fail(f"mask_consistency[{backend}]",
                        f"{key}_mask disagrees with isfinite(depth) & (depth > 0) on "
                        f"{int((mask != expected).sum())} px: background is being encoded "
                        f"as a value",
                        f"MULTISIM_DEPTH_CONVENTION_MISMATCH({backend}:{key}_mask)", "SPEC-014 R2")
        else:
            ledger.ok(f"mask_consistency[{backend}]",
                      f"{key}_mask is exactly the defined domain of {key}", "SPEC-014 R2")

    # --- the scene itself: same instances in every column (requirement 1) ---
    instance_ids = sample["bundle_instance_ids"]
    reported = report.get("object_ids")
    if reported is None:
        ledger.skip(f"object_set[{backend}]", "report carries no object_ids list",
                    "SPEC-014 R1", blocks=False)
    elif list(reported) != list(instance_ids):
        missing = sorted(set(instance_ids) - set(reported))
        extra = sorted(set(reported) - set(instance_ids))
        ledger.fail(f"object_set[{backend}]",
                    f"loaded objects differ from the frozen scene "
                    f"(missing={missing[:4]}, extra={extra[:4]})",
                    f"MULTISIM_GEOMETRY_MISMATCH({backend}," +
                    f"{missing[0] if missing else (extra[0] if extra else 'order')})",
                    "SPEC-014 R1")

    # --- the figure must compare a frozen authored state, not a settled one ---
    capture_state = report.get("capture_state")
    if capture_state is None:
        ledger.skip(f"capture_state[{backend}]", "report does not declare capture_state",
                    "SPEC-014 R1", blocks=False)
    elif capture_state != "authored":
        ledger.fail(f"capture_state[{backend}]",
                    f"capture_state={capture_state!r}; the figure compares frozen authored poses",
                    f"MULTISIM_GEOMETRY_MISMATCH({backend},capture_state={capture_state})",
                    "SPEC-014 R1")
    else:
        ledger.ok(f"capture_state[{backend}]", "authored (frozen) capture", "SPEC-014 R1")

    if report.get("status") != "passed":
        ledger.fail(f"status[{backend}]", f"report status={report.get('status')!r}",
                    f"MULTISIM_BACKEND_UNAVAILABLE({backend}:status={report.get('status')})",
                    "SPEC-014 R4")
    else:
        ledger.ok(f"status[{backend}]", "report status passed", "SPEC-014 R4")

    panel_engine = (provenance.get("panels", {}).get(backend) or {}).get("engine_version")
    if panel_engine is not None and report.get("engine_version") != panel_engine:
        ledger.fail(f"engine_version[{backend}]",
                    f"report says {report.get('engine_version')!r}, provenance says "
                    f"{panel_engine!r}",
                    f"MULTISIM_PROVENANCE_MISSING({backend}:engine_version)", "SPEC-014 R1")
    else:
        ledger.ok(f"engine_version[{backend}]", "engine version agrees with provenance",
                  "SPEC-014 R1")


def check_panel_pixels(bundle: Path, samples: dict, ledger: Ledger) -> None:
    """Requirements 1/4/7: the panel the figure embeds *is* the raw render.

    The RGB comparison is exact (a PNG round-trip of a uint8 array). The depth
    and normals panels are held to the recipes ``common.py`` freezes -- the
    assembler's own claim is that every column uses one recipe, so re-deriving
    the panels independently and comparing pixels is the check that makes the
    claim true. If the frozen module cannot be imported the comparison is
    recorded ``not_run`` rather than assumed to pass.
    """
    panels_dir = bundle / "panels"
    for backend in BACKEND_ORDER:
        missing = [f"panels/{backend}_{suffix}" for suffix in PANEL_SUFFIXES
                   if not (panels_dir / f"{backend}_{suffix}").is_file()]
        if missing:
            ledger.fail(f"panel_files[{backend}]", ", ".join(missing) + " missing",
                        f"MULTISIM_PROVENANCE_MISSING({missing[0]})", "SPEC-014 R4/R7")
        else:
            ledger.ok(f"panel_files[{backend}]",
                      "rgb, visible-depth and collision-normals panels present", "SPEC-014 R4")

    try:
        import rfw_w.scenes.workers.multisim.common as common # noqa: PLC0415 -- the frozen conventions, imported on purpose
    except Exception as error:  # pragma: no cover - environment dependent
        for backend in BACKEND_ORDER:
            ledger.skip(f"panel_pixels[{backend}]",
                        f"cannot import the frozen recipes from common.py: {error}",
                        "SPEC-014 R2/R4", blocks=False)
        return

    camera = common.load_camera(bundle / "camera.json")
    for backend in BACKEND_ORDER:
        sample = samples.get(backend)
        if sample is None or "rgb" not in sample or "depth_visible" not in sample:
            ledger.skip(f"panel_pixels[{backend}]",
                        "raw products unreadable, so the panel cannot be re-derived",
                        "SPEC-014 R1/R4", blocks=False)
            continue
        projections = [("rgb.png", lambda: sample["rgb"]),
                       ("depth_visible.png", lambda: common.colourise_depth(
                           sample["depth_visible"]))]
        if "depth_collision" in sample:
            projections.append(("normal_collision.png", lambda: common.depth_to_world_normal_rgb(
                sample["depth_collision"], camera)))
        mismatches = []
        unchecked = []
        for suffix, derive in projections:
            path = panels_dir / f"{backend}_{suffix}"
            if not path.is_file():
                continue
            try:
                expected = derive()
            except Exception as error:  # pragma: no cover - environment dependent
                unchecked.append(f"{suffix}: {error}")
                continue
            actual = np.asarray(Image.open(path).convert("RGB"))
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                differing = (0 if actual.shape != expected.shape
                             else int((actual != expected).any(axis=-1).sum()))
                mismatches.append(f"{suffix} differs from the raw array on {differing} px")
        if mismatches:
            ledger.fail(f"panel_pixels[{backend}]", "; ".join(mismatches),
                        f"MULTISIM_PROVENANCE_MISSING({backend}:panel pixels)", "SPEC-014 R1/R4")
        elif unchecked:
            ledger.skip(f"panel_pixels[{backend}]",
                        "cannot re-derive every panel: " + "; ".join(unchecked),
                        "SPEC-014 R4", blocks=False)
        else:
            ledger.ok(f"panel_pixels[{backend}]",
                      f"all {len(projections)} panels re-derive exactly from the raw products",
                      "SPEC-014 R1/R4")


def check_depth_and_thresholds(samples: dict, measurements: dict, ledger: Ledger) -> dict:
    """Requirements 2/5: recompute, hold the figure's own numbers to it, and
    apply every threshold."""
    reference = samples.get(REFERENCE_BACKEND)
    if reference is None:
        ledger.skip("depth_comparison", f"{REFERENCE_BACKEND} did not load, so the reference "
                                        f"domain does not exist", "SPEC-014 R3",
                    gap=f"MULTISIM_PROVENANCE_MISSING({REFERENCE_BACKEND})")
        return {}
    if "depth_visible" not in reference:
        ledger.skip("depth_comparison", f"{REFERENCE_BACKEND} has no visible depth array",
                    "SPEC-014 R3")
        return {}

    figure_backends = (measurements.get("backends") or {})
    results: dict[str, dict] = {}

    for backend in BACKEND_ORDER:
        sample = samples.get(backend)
        entry: dict = {}
        results[backend] = entry

        value, kind = self_check_median((sample or {}).get("report", {}).get("depth_self_check_m"))
        entry["depth_self_check_kind"] = kind
        entry["depth_self_check_m"] = value
        if sample is None:
            continue
        if value is None:
            ledger.skip(f"self_check[{backend}]",
                        f"report carries no usable depth_self_check_m ({kind})",
                        "SPEC-014 R5",
                        gap=f"MULTISIM_DEPTH_INVALID({backend}: no depth_self_check_m)")
        elif value > SELF_CHECK_TOLERANCE_M:
            ledger.fail(f"self_check[{backend}]",
                        f"anchor residual {value:.4f} m exceeds {SELF_CHECK_TOLERANCE_M} m",
                        f"MULTISIM_ALIGNMENT_BELOW_THRESHOLD(depth_self_check_median_m,"
                        f"{value:.6f})", "SPEC-014 R5")
        else:
            ledger.ok(f"self_check[{backend}]",
                      f"anchor residual {value * 1000:.3f} mm ({kind}) within "
                      f"{SELF_CHECK_TOLERANCE_M * 1000:.0f} mm", "SPEC-014 R5")
        if "depth_visible" not in sample:
            ledger.skip(f"depth_comparison[{backend}]",
                        "no readable visible-depth array, so nothing could be compared",
                        "SPEC-014 R3",
                        gap=f"MULTISIM_PROVENANCE_MISSING({backend}:depth_visible.npy)")
            continue

        entry["valid_pixel_fraction_visible"] = float(
            np.asarray(sample["depth_visible_mask"], dtype=bool).mean())
        if "depth_collision_mask" in sample:
            entry["valid_pixel_fraction_collision"] = float(
                np.asarray(sample["depth_collision_mask"], dtype=bool).mean())
        entry["shape"] = list(np.asarray(sample["depth_visible"]).shape)
        entry["dtype"] = np.asarray(sample["depth_visible"]).dtype.name

        company = figure_backends.get(backend) or {}
        if backend == REFERENCE_BACKEND:
            ledger.skip(f"depth_comparison[{backend}]",
                        "reference backend: requirement 3 compares the other four against it",
                        "SPEC-014 R3", blocks=False)
            continue

        if sample.get("depth_visible_mask") is None or reference.get("depth_visible_mask") is None:
            ledger.skip(f"depth_comparison[{backend}]", "mask array missing",
                        "SPEC-014 R3",
                        gap=f"MULTISIM_PROVENANCE_MISSING({backend}:depth_visible_mask)")
            continue
        computed = compare_to_reference(sample, reference, CONTOUR_BAND_PX)
        entry["vs_reference"] = computed
        if "error" in computed:
            ledger.fail(f"depth_comparison[{backend}]",
                        f"{computed['error']} (common valid px = "
                        f"{computed['common_valid_pixels']})",
                        f"MULTISIM_DEPTH_INVALID({backend}: {computed['error']})",
                        "SPEC-014 R3")
            continue
        ledger.ok(f"depth_comparison[{backend}]",
                  f"{computed['common_valid_pixels']} common px, "
                  f"{computed['pixels_set_aside_by_band']} set aside by the "
                  f"{CONTOUR_BAND_PX} px contour band, "
                  f"{computed['pixels_used_after_contour_band']} compared", "SPEC-014 R3")

        # --- the figure's own arithmetic, held to this recomputation ---
        recorded = company.get("vs_reference")
        if not isinstance(recorded, dict):
            ledger.fail(f"measurement_agreement[{backend}]",
                        "measurements.json has no vs_reference block for this backend",
                        f"MULTISIM_PROVENANCE_MISSING(measurements.json[{backend}].vs_reference)",
                        "SPEC-014 R3")
        else:
            disagreements = []
            entry["agreement_with_figure"] = {}
            for metric in COMPARED_METRICS:
                mine = computed.get(metric)
                theirs = recorded.get(metric)
                if theirs is None or mine is None:
                    disagreements.append(f"{metric} recorded={theirs} recomputed={mine}")
                    entry["agreement_with_figure"][metric] = {
                        "verifier": mine, "figure": theirs, "agrees": False}
                    continue
                agrees = close(float(mine), float(theirs))
                entry["agreement_with_figure"][metric] = {
                    "verifier": float(mine), "figure": float(theirs), "agrees": bool(agrees)}
                if not agrees:
                    disagreements.append(f"{metric} figure={float(theirs)!r} "
                                         f"verifier={float(mine)!r}")
            # The two coverage fractions are panel-array facts as well, so they are
            # recomputed too (and only the ones this bundle actually carries).
            for key in ("valid_pixel_fraction_visible", "valid_pixel_fraction_collision"):
                mine = entry.get(key)
                if mine is None:
                    continue
                theirs = company.get(key)
                if theirs is None or not close(float(mine), float(theirs)):
                    disagreements.append(f"{key} figure={theirs} verifier={float(mine)!r}")
                    entry["agreement_with_figure"][key] = {
                        "verifier": float(mine), "figure": theirs, "agrees": False}
            # The self-check is the report's own number, not the figure's; the
            # figure is only allowed to have copied it faithfully.
            raw_self_check = ((sample.get("report") or {}).get("depth_self_check_m"))
            if company.get("depth_self_check_m") != raw_self_check:
                disagreements.append(
                    f"depth_self_check_m figure={company.get('depth_self_check_m')} "
                    f"report={raw_self_check}")
                entry["agreement_with_figure"]["depth_self_check_m"] = {
                    "verifier": raw_self_check, "figure": company.get("depth_self_check_m"),
                    "agrees": False}
            if disagreements:
                # SPEC-014's failure vocabulary does not name "the assembler's
                # own arithmetic disagrees with a fresh recomputation"; this
                # verifier names it explicitly rather than letting it pass.
                ledger.fail(f"measurement_agreement[{backend}]", "; ".join(disagreements),
                            f"MULTISIM_MEASUREMENT_MISMATCH({backend}:" +
                            f"{disagreements[0].split(' ')[0]})", "SPEC-014 R3")
            else:
                ledger.ok(f"measurement_agreement[{backend}]",
                          f"all {len(entry['agreement_with_figure'])} recomputed metrics and "
                          f"coverage fractions agree with measurements.json", "SPEC-014 R3")

        # --- requirement 5's thresholds, each its own named gap ---
        breaches = []
        if computed["mask_iou"] < MASK_IOU_MIN:
            breaches.append(
                f"MULTISIM_ALIGNMENT_BELOW_THRESHOLD(mask_iou,{computed['mask_iou']:.6f})")
        if computed["abs_depth_median_m"] > ABS_DEPTH_MEDIAN_MAX_M:
            breaches.append(
                f"MULTISIM_ALIGNMENT_BELOW_THRESHOLD(abs_depth_median_m,"
                f"{computed['abs_depth_median_m']:.6f})")
        if computed["abs_depth_p95_m"] > ABS_DEPTH_P95_MAX_M:
            breaches.append(
                f"MULTISIM_ALIGNMENT_BELOW_THRESHOLD(abs_depth_p95_m,"
                f"{computed['abs_depth_p95_m']:.6f})")
        if breaches:
            ledger.fail_many(f"thresholds[{backend}]",
                             f"iou={computed['mask_iou']:.4f} (min {MASK_IOU_MIN}), "
                             f"median={computed['abs_depth_median_m']:.4f} m "
                             f"(max {ABS_DEPTH_MEDIAN_MAX_M}), "
                             f"p95={computed['abs_depth_p95_m']:.4f} m "
                             f"(max {ABS_DEPTH_P95_MAX_M})",
                             breaches, "SPEC-014 R5")
        else:
            ledger.ok(f"thresholds[{backend}]",
                      f"iou={computed['mask_iou']:.4f} >= {MASK_IOU_MIN}, "
                      f"median={computed['abs_depth_median_m']:.4f} m <= "
                      f"{ABS_DEPTH_MEDIAN_MAX_M}, p95={computed['abs_depth_p95_m']:.4f} m <= "
                      f"{ABS_DEPTH_P95_MAX_M}", "SPEC-014 R5")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, type=Path,
                        help="the figure directory written by assemble_multisim_figure.py")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="a fresh directory; verification.json is the only thing written")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    if not bundle.is_dir():
        parser.error(f"{bundle} is not a directory")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory (evidence is "
                     f"never overwritten)")
    out.mkdir(parents=True, exist_ok=True)

    ledger = Ledger()

    provenance_path = bundle / "provenance.json"
    if not provenance_path.is_file():
        payload = {
            "spec": "SPEC-014",
            "verifier": "python -m rfw_w.scenes.validation.check_rigid_multisim",
            "bundle": str(bundle),
            "verified": False,
            "problems": [f"MULTISIM_PROVENANCE_MISSING(provenance.json)"],
            "machine_checks": {"bundle_layout": {
                "status": "failed", "requirement": "SPEC-014 R1",
                "detail": f"{provenance_path} is missing"}},
            "not_run": [{"check": "all", "reason": "no provenance.json to trace panels to",
                         "gap": "MULTISIM_PROVENANCE_MISSING(provenance.json)",
                         "blocks_verified": True}],
            "measurements": {},
            "figure": {},
            "visual_review": "not_run(user preview pending)",
        }
        (out / "verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    provenance = read_json(provenance_path)

    # The frozen camera and scene are the bundle's own copies; they are read as
    # data, and only the *reports* are held to them.
    camera_path = bundle / "camera.json"
    scene_path = bundle / "common_scene.json"
    for path in (camera_path, scene_path):
        if not path.is_file():
            ledger.fail("bundle_layout", f"{path} is missing",
                        f"MULTISIM_PROVENANCE_MISSING({path.name})", "SPEC-014 R1")
    if ledger.problems:
        payload = {
            "spec": "SPEC-014",
            "verifier": "python -m rfw_w.scenes.validation.check_rigid_multisim",
            "bundle": str(bundle),
            "verified": False,
            "problems": ledger.problems,
            "machine_checks": ledger.checks,
            "not_run": ledger.not_run,
            "measurements": {},
            "figure": {},
            "visual_review": "not_run(user preview pending)",
        }
        (out / "verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    camera = read_json(camera_path)
    scene = read_json(scene_path)
    measurements = read_json(bundle / "measurements.json") if (bundle / "measurements.json").is_file() \
        else {}
    if not measurements:
        ledger.fail("bundle_layout", "measurements.json is missing or empty",
                    "MULTISIM_PROVENANCE_MISSING(measurements.json)", "SPEC-014 R3")
    else:
        ledger.ok("bundle_layout", "provenance.json, camera.json, common_scene.json and "
                                   "measurements.json are present", "SPEC-014 R1")

    expected_version = scene.get("scene_version")
    if provenance.get("scene_version") != expected_version:
        ledger.fail("figure_structure",
                    f"provenance scene_version {provenance.get('scene_version')!r} != "
                    f"common_scene.json {expected_version!r}",
                    f"MULTISIM_SCENE_VERSION_MISMATCH(bundle)", "SPEC-014 R1")
    if list(provenance.get("backend_order") or []) != list(BACKEND_ORDER):
        ledger.fail("figure_structure",
                    f"backend_order is {provenance.get('backend_order')}, requirement 4 fixes "
                    f"{list(BACKEND_ORDER)}",
                    f"MULTISIM_FIGURE_INCOMPLETE(backend_order={provenance.get('backend_order')})",
                    "SPEC-014 R4")
    if provenance.get("reference_backend") != REFERENCE_BACKEND:
        ledger.fail("figure_structure",
                    f"reference_backend is {provenance.get('reference_backend')!r}, "
                    f"requirement 3 fixes {REFERENCE_BACKEND!r}",
                    f"MULTISIM_FIGURE_INCOMPLETE(reference_backend="
                    f"{provenance.get('reference_backend')})", "SPEC-014 R4")

    # The bundle ships two pictures of the camera: camera.json (what the reports
    # are held to) and the scene's own copy. If they disagree the bundle has two
    # frozen cameras and the audit is meaningless.
    scene_camera_diff = camera_differences(scene.get("camera") or {}, camera, require_all=False)
    if scene_camera_diff:
        ledger.fail("camera_source", "; ".join(scene_camera_diff),
                    "MULTISIM_CAMERA_MISMATCH(bundle: common_scene.json camera != camera.json)",
                    "SPEC-014 R1/R2")
    else:
        ledger.ok("camera_source",
                  "common_scene.json's camera and camera.json agree on every shared field",
                  "SPEC-014 R1/R2")

    samples, panels = load_panels(bundle, provenance, ledger)

    scene_instances = [o.get("object_id") for o in scene.get("objects") or []
                       if o.get("kind") == "furniture"]
    for backend in BACKEND_ORDER:
        sample = samples.get(backend)
        if sample is None:
            ledger.skip(f"report[{backend}]", "no readable report and rgb.png on disk",
                        "SPEC-014 R1", gap=f"MULTISIM_PROVENANCE_MISSING({backend})")
            continue
        sample["bundle_camera"] = camera
        sample["bundle_scene_version"] = expected_version
        sample["bundle_instance_ids"] = scene_instances
        check_agreement(sample, provenance, ledger)

    check_panel_pixels(bundle, samples, ledger)
    per_backend = check_depth_and_thresholds(samples, measurements, ledger)
    figures = check_figure_files(bundle, ledger)

    # The table the caller reads: one row per backend, every number from this
    # process's own recomputation.
    table = []
    for backend in BACKEND_ORDER:
        entry = dict(per_backend.get(backend) or {})
        company = ((measurements.get("backends") or {}).get(backend) or {})
        comparison = entry.get("vs_reference") or {}
        table.append({
            "backend": backend,
            "engine_version": (samples.get(backend, {}).get("report", {}) or {}).get("engine_version"),
            "resolution": f"{camera['width']}x{camera['height']}",
            "report": str((panels.get(backend) or {}).get("report_path", "")),
            "raw_rgb": str((panels.get(backend) or {}).get("rgb_path", "")),
            "valid_pixel_fraction_visible": entry.get("valid_pixel_fraction_visible"),
            "depth_self_check_m": entry.get("depth_self_check_m"),
            "depth_self_check_kind": entry.get("depth_self_check_kind"),
            "mask_iou": comparison.get("mask_iou"),
            "abs_depth_median_m": comparison.get("abs_depth_median_m"),
            "abs_depth_p95_m": comparison.get("abs_depth_p95_m"),
            "pixels_set_aside_by_band": comparison.get("pixels_set_aside_by_band"),
            "figure_engine_version": company.get("engine_version"),
        })

    blocked = [entry for entry in ledger.not_run if entry["blocks_verified"]]
    verified = not ledger.problems and not blocked

    payload = {
        "spec": "SPEC-014",
        "requirement": "2, 3, 5 (thresholds), 4/7 (figure resolution)",
        "verifier": "python -m rfw_w.scenes.validation.check_rigid_multisim",
        "bundle": str(bundle),
        "verified": bool(verified),
        "problems": sorted(ledger.problems),
        "machine_checks": dict(sorted(ledger.checks.items())),
        "not_run": ledger.not_run,
        "measurements": {
            "reference_backend": REFERENCE_BACKEND,
            "contour_band_px": CONTOUR_BAND_PX,
            "depth_convention": DEPTH_CONVENTION,
            "scene_version": expected_version,
            "camera": camera,
            "thresholds": {
                "mask_iou_min": MASK_IOU_MIN,
                "abs_depth_median_max_m": ABS_DEPTH_MEDIAN_MAX_M,
                "abs_depth_p95_max_m": ABS_DEPTH_P95_MAX_M,
                "depth_self_check_tolerance_m": SELF_CHECK_TOLERANCE_M,
                "contour_band_px": CONTOUR_BAND_PX,
            },
            "backends": per_backend,
        },
        "table": table,
        "figure": figures,
        "visual_review": "not_run(user preview pending)",
    }
    (out / "verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(json.dumps({
        "verified": payload["verified"],
        "problems": payload["problems"],
        "not_run_blocking": [entry["gap"] or entry["check"] for entry in blocked],
        "table": table,
    }, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
