#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Measure what a delivered view actually covered: the effective aperture, and the footprint.

Why this exists
---------------
`AGENTS.md` records that the renderer's aperture is **not** the number the request declares
("effective vertical aperture ~12.4 mm, every camera ~1.7x narrower than its focal length
implies"), and the worker has since started pinning the declared aperture into the USD camera
(`render_worker.py:730-731`). Nobody has verified which of the two is real -- and SPEC-015's
framing depends on it: aperture and focal multiply into the field of view, so a 1.7x error puts
the far corners of a room outside the frame.

How it measures
---------------
The camera pose is known (the request carries eye/target) and so is the floor's height, so for a
candidate aperture every pixel's depth unprojects to a world z. **A flat floor is flat**: with the
*correct* intrinsics the floor pixels land exactly on `room.floor_z_m` whatever the camera's tilt;
with wrong intrinsics the reconstructed z fans out with image position. So the estimator scans the
aperture and keeps the one that puts the most pixels on the known floor plane.

Two earlier statistics were tried and rejected, and both are why `--self-test` ships first
(AGENTS.md §9: commit the measurement tool, then quote it):

* `z.std()` over the unprojected cloud -- ill-conditioned: it tracks the floor's own tilt, not the
  misfit, so every candidate aperture scored equally badly (measured 2026-09-24, no conclusion).
* RANSAC planarity without the known height -- degenerate at wide apertures, where a wrong scale
  simply collapses the cloud toward the camera and *any* plane fits it.

Scoring against the declared floor height has neither problem, and the score is a fraction rather
than a threshold, so it reads the same on a wall-to-wall floor and on a floor seen past furniture.

    # 1. prove the estimator on synthetic depth whose aperture is known
    python -m rfw_w.scenes.visualization.measure_view_geometry --self-test

    # 2. measure a real delivery
    python -m rfw_w.scenes.visualization.measure_view_geometry \
        --delivery outputs/office_pick_cup/large_office_20260924_01

Read-only: it never writes into a delivery.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

#: The sensor the request declares. A measurement reports the factor this has to be scaled by.
DECLARED_H_MM = 20.955
DECLARED_V_MM = 15.2908
#: Candidate scales (coarse, then refined around the winner) and the ways a renderer can map a
#: film back onto a resolution.
SCALES = tuple(round(0.40 + 0.02 * i, 4) for i in range(0, 81))    # 0.40 .. 2.00
REFINE_STEPS = tuple(round(-0.02 + 0.002 * i, 4) for i in range(0, 21))
FIT_MODES = ("fill", "horizontal", "vertical")
#: How close a pixel's reconstructed z has to sit to the floor to count as *being* the floor.
FLOOR_TOL_M = 0.02
#: Below this fraction of the frame, no aperture hypothesis found a floor worth speaking of.
MIN_FLOOR_FRACTION = 0.02
#: The floor's height is fitted too: measured on four deliveries the rendered floor sits
#: 0.050-0.055 m above `room.floor_z_m`, and a biased floor biases every aperture estimate.
FLOOR_OFFSETS = tuple(round(-0.08 + 0.005 * i, 4) for i in range(0, 33))
#: Depth is subsampled for the scan; the estimate is a shape, not a per-pixel statistic.
SCAN_STRIDE = 4
#: A scale is only quoted when everything within 10% of the best score spans less than this.
IDENTIFIED_SPREAD = 0.10
#: The aperture enters the reconstructed z only through the camera frame's vertical components:
#: z = eye_z + forward_z*d + right_z*(x*d) - up_z*(y*d). A near-nadir view has forward_z ~ -1 and
#: up_z ~ 0.007, so its depth carries almost no aperture information however clean it looks --
#: measured: layout_top reads x0.70 in one room and x1.16 in another. Below this lever the view
#: cannot measure an aperture and the tool says so instead of quoting one.
MIN_APERTURE_LEVER = 0.15
#: The scan scores with a tighter band than the footprint check: a *free* floor offset can absorb
#: part of a scale error over the near part of the frame, so the scan has to be strict to keep the
#: aperture identifiable at all (with a 20 mm band, even synthetic data lost identifiability).
SCAN_TOL_M = 0.004


def camera_basis(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> tuple:
    """The camera frame. `right` comes from the world up, which is what keeps roll well defined."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    if norm < 1e-6:                      # looking straight down the world up axis
        raise ValueError("the view direction is parallel to the world up axis; roll is undefined")
    right = right / norm
    up = np.cross(right, forward)
    return forward, right, up


def intrinsics(focal_mm: float, scale: float, mode: str, width: int, height: int) -> tuple:
    """Focal length in pixels for the two axes, under one aperture hypothesis."""
    aperture_h = DECLARED_H_MM * scale
    aperture_v = DECLARED_V_MM * scale
    fx = focal_mm * width / aperture_h
    if mode == "fill":
        return fx, focal_mm * height / aperture_v
    if mode == "horizontal":
        return fx, fx
    return focal_mm * height / aperture_v, focal_mm * height / aperture_v


def floor_fraction(depth: np.ndarray, camera: dict, scale: float, mode: str, floor_z: float,
                   stride: int = 1, tolerance: float = FLOOR_TOL_M) -> tuple:
    """How much of the frame reconstructs onto the known floor plane, under one hypothesis."""
    height, width = depth.shape
    eye = np.array(camera["eye"], dtype=float)
    forward, right, up = camera_basis(eye, np.array(camera["target"], dtype=float),
                                      np.array([0.0, 0.0, 1.0]))
    fx, fy = intrinsics(camera["focal_length_mm"], scale, mode, width, height)

    depth = depth[::stride, ::stride]
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    d = depth
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return 0.0, 0
    x = (columns - width / 2.0) / fx
    y = (rows - height / 2.0) / fy
    # world z of each pixel, straight from the camera frame -- no point cloud needed
    z = (eye[2] + forward[2] * d + right[2] * (x * d) - up[2] * (y * d))
    on_floor = valid & (np.abs(z - floor_z) <= tolerance)
    return float(on_floor.sum()) / float(valid.sum()), int(valid.sum())


def project_polygon(polygon, camera: dict, scale: float, mode: str, floor_z: float,
                    width: int, height: int) -> np.ndarray:
    """Where the room's own footprint lands in the image, under one aperture hypothesis."""
    eye = np.array(camera["eye"], dtype=float)
    forward, right, up = camera_basis(eye, np.array(camera["target"], dtype=float),
                                      np.array([0.0, 0.0, 1.0]))
    fx, fy = intrinsics(camera["focal_length_mm"], scale, mode, width, height)
    out = []
    for vertex in polygon:
        point = np.array([float(vertex[0]), float(vertex[1]), floor_z]) - eye
        depth = float(point @ forward)
        if depth <= 1e-6:
            return np.zeros((0, 2))
        out.append((width / 2.0 + fx * float(point @ right) / depth,
                    height / 2.0 - fy * float(point @ up) / depth))
    return np.array(out)


def footprint_match(depth: np.ndarray, camera: dict, polygon, floor_z: float, scale: float,
                    mode: str) -> float:
    """IoU between where the room's footprint *should* be and where the floor actually is.

    The first method (`floor_fraction`) asks whether the floor is flat; this one asks whether it is
    in the right *place and size*. A pose convention that is subtly wrong can pass neither, so the
    two are reported together and a scale is only quoted when both agree.
    """
    from PIL import Image, ImageDraw

    height, width = depth.shape
    projected = project_polygon(polygon, camera, scale, mode, floor_z, width, height)
    if len(projected) < 3:
        return 0.0
    mask = Image.new("1", (width, height), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in projected], fill=1)
    expected = np.array(mask, dtype=bool)

    fraction, _ = floor_fraction(depth, camera, scale, mode, floor_z)
    eye = np.array(camera["eye"], dtype=float)
    forward, right, up = camera_basis(eye, np.array(camera["target"], dtype=float),
                                      np.array([0.0, 0.0, 1.0]))
    fx, fy = intrinsics(camera["focal_length_mm"], scale, mode, width, height)
    rows, columns = np.mgrid[0:height, 0:width]
    d = depth
    valid = np.isfinite(d) & (d > 0)
    x = (columns - width / 2.0) / fx
    y = (rows - height / 2.0) / fy
    z = eye[2] + forward[2] * d + right[2] * (x * d) - up[2] * (y * d)
    measured = valid & (np.abs(z - floor_z) <= FLOOR_TOL_M)
    union = (expected | measured).sum()
    return float((expected & measured).sum()) / float(union) if union else 0.0


def mask_coverage(depth: np.ndarray, camera: dict, polygon, floor_z: float, scale: float,
                  mode: str) -> float:
    """How much of the frame the projected footprint covers -- above ~0.95 the check has no power."""
    from PIL import Image, ImageDraw

    height, width = depth.shape
    projected = project_polygon(polygon, camera, scale, mode, floor_z, width, height)
    if len(projected) < 3:
        return 1.0
    mask = Image.new("1", (width, height), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in projected], fill=1)
    return float(np.array(mask, dtype=bool).sum()) / float(width * height)


def footprint_scale(depth: np.ndarray, camera: dict, polygon, floor_z: float,
                    verbose: bool = False) -> dict:
    """The aperture hypothesis whose projected footprint best matches the observed floor."""
    best = None
    for mode in FIT_MODES:
        for scale in SCALES:
            iou = footprint_match(depth, camera, polygon, floor_z, scale, mode)
            if best is None or iou > best["iou"]:
                best = {"mode": mode, "scale": scale, "iou": iou,
                        "mask_coverage": mask_coverage(depth, camera, polygon, floor_z, scale, mode)}
    coarse = best["scale"]
    for mode in FIT_MODES:
        for step in REFINE_STEPS:
            scale = round(coarse + step, 4)
            if scale <= 0:
                continue
            iou = footprint_match(depth, camera, polygon, floor_z, scale, mode)
            if iou > best["iou"]:
                best = {"mode": mode, "scale": scale, "iou": iou,
                        "mask_coverage": mask_coverage(depth, camera, polygon, floor_z, scale, mode)}
    if verbose:
        for mode in FIT_MODES:
            print("   footprint %-11s %s" % (mode, "  ".join(
                "x%.1f %4.2f" % (s, footprint_match(depth, camera, polygon, floor_z, s, mode))
                for s in (0.6, 0.8, 1.0, 1.2, 1.5))))
    best["effective_h_mm"] = DECLARED_H_MM * best["scale"]
    # Two ways this check has no power, both seen on real deliveries:
    #  * the room's footprint is mostly *behind* the camera (a view aimed at the desk), so the
    #    projection is not merely inaccurate -- it is undefined;
    #  * the room projects past every edge of the frame (overview_top's 118-degree lens), so the
    #    expected mask is the whole image and no scale can be told from it -- measured: it pulled
    #    the estimate to x0.78 against the flat floor's x1.00.
    # Without this check the scan returned whatever sat at the end of its range, which reads like a
    # measurement and is not one.
    best["applicable"] = best["iou"] >= 0.5 and best.get("mask_coverage", 0.0) <= 0.95
    return best


def _scan(depth: np.ndarray, camera: dict, floor_z: float, verbose: bool = False) -> dict:
    """Scan (fit mode, aperture scale, floor offset) and report the best plus its identifiability.

    The offset is free because the rendered floor is not where the request says it is -- measured
    on four deliveries it sits 0.050-0.055 m above `room.floor_z_m`. Without it every estimate is
    biased, and a near-nadir camera reads as "no floor in frame at all".

    The *range* matters as much as the best value: a near-nadir camera's depth carries almost no
    aperture information (the aperture only enters through the `up_z` lever arm, ~0.007 there), so
    a confident-looking single number would be a lie. Everything scoring within 10% of the best
    bounds the scale, and a wide bound is reported as such.
    """
    eye = np.array(camera["eye"], dtype=float)
    _, right, up = camera_basis(eye, np.array(camera["target"], dtype=float),
                                np.array([0.0, 0.0, 1.0]))
    lever = float(math.hypot(right[2], up[2]))
    if lever < MIN_APERTURE_LEVER:
        return {"resolved": False, "lever": lever,
                "reason": "this view's optical axis is too close to the floor normal for its depth "
                          "to constrain the aperture (lever %.3f < %.2f); the framing still gets a "
                          "number from the model, but not from this camera" % (lever,
                                                                               MIN_APERTURE_LEVER)}

    ranked: list = []
    best = None
    try:
        for mode in FIT_MODES:
            for offset in FLOOR_OFFSETS:
                for scale in SCALES:
                    fraction, pixels = floor_fraction(depth, camera, scale, mode, floor_z + offset,
                                                      stride=SCAN_STRIDE, tolerance=SCAN_TOL_M)
                    ranked.append((fraction, scale, mode, offset))
                    if best is None or fraction > best[0]:
                        best = (fraction, scale, mode, offset)
    except ValueError as error:
        return {"resolved": False, "reason": str(error)}
    # one refinement pass around the coarse winner
    for mode in (best[2],) + tuple(m for m in FIT_MODES if m != best[2]):
        for step in REFINE_STEPS:
            for offset_step in (-0.005, 0.0, 0.005):
                scale = round(best[1] + step, 4)
                offset = round(best[3] + offset_step, 4)
                if scale <= 0:
                    continue
                fraction, _ = floor_fraction(depth, camera, scale, mode, floor_z + offset,
                                             stride=SCAN_STRIDE, tolerance=SCAN_TOL_M)
                if fraction > best[0]:
                    best = (fraction, scale, mode, offset)
    # the identifiability band is read at the winning floor offset: across offsets the nuisance
    # parameter inflates it, and the question here is "how well does this view pin the aperture".
    within = [s for f, s, mode, offset in ranked
              if f >= 0.9 * best[0] and mode == best[2] and abs(offset - best[3]) <= 0.005] or [best[1]]
    floor_fraction_value = best[0]
    if verbose:
        for mode in FIT_MODES:
            print("   %-11s %s" % (mode, "  ".join(
                "x%.1f %5.1f%%" % (s, floor_fraction(depth, camera, s, mode, floor_z + best[3],
                                                     stride=SCAN_STRIDE)[0] * 100)
                for s in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8))))
    flat_ok = floor_fraction_value >= MIN_FLOOR_FRACTION
    flat = {"scale": best[1], "mode": best[2], "floor_fraction": floor_fraction_value,
            "floor_offset_m": round(best[3], 4), "scale_low": min(within), "scale_high": max(within),
            "lever": lever,
            "identified": (max(within) - min(within)) <= IDENTIFIED_SPREAD * best[1]}

    return flat


def measure(depth: np.ndarray, camera: dict, floor_z: float, polygon=None,
            verbose: bool = False) -> dict:
    """The scan's verdict, plus the footprint cross-check when this view contains the room."""
    flat = _scan(depth, camera, floor_z, verbose)
    if not flat.get("resolved", True) and "scale" not in flat:
        return flat
    flat_ok = flat["floor_fraction"] >= MIN_FLOOR_FRACTION
    if polygon is None:
        if not flat_ok:
            return {"resolved": False, "flat": flat,
                    "reason": "no aperture hypothesis put a floor in the frame"}
        if not flat["identified"]:
            return {"resolved": False, "flat": flat, "method": "flat floor",
                    "reason": "the floor is in frame but the aperture is not identified by this "
                              "view: everything within 10%% of the best score spans x%.2f-x%.2f" % (
                                  flat["scale_low"], flat["scale_high"])}
        return {"resolved": True, "method": "flat floor", "flat": flat, "scale": flat["scale"],
                "mode": flat["mode"], "floor_offset_m": flat["floor_offset_m"],
                "effective_h_mm": DECLARED_H_MM * flat["scale"],
                "effective_v_mm": DECLARED_V_MM * flat["scale"]}

    foot = footprint_scale(depth, camera, polygon, floor_z + flat["floor_offset_m"],
                            verbose=verbose)
    if not foot["applicable"]:
        # The floor is in frame, but the room's footprint is not (a view aimed at the desk has most
        # of the room behind it). Only the flat-floor method applies, and it is only worth quoting
        # when the floor is a real part of the frame rather than a strip at the bottom.
        if not flat_ok or flat["floor_fraction"] < 0.15:
            return {"resolved": False, "flat": flat, "footprint": foot,
                    "reason": "the room's footprint is mostly behind this camera (footprint IoU "
                              "%.2f), so only the flat-floor method applies, and it sees just "
                              "%.0f%% floor -- too little to quote an aperture" % (
                                  foot["iou"], flat["floor_fraction"] * 100)}
        return {"resolved": True, "method": "flat floor only (footprint not in this view)",
                "flat": flat, "footprint": foot, "scale": flat["scale"], "mode": flat["mode"],
                "effective_h_mm": DECLARED_H_MM * flat["scale"],
                "effective_v_mm": DECLARED_V_MM * flat["scale"]}

    result = {"flat": flat, "footprint": foot, "method": "flat floor + footprint",
              "scale": foot["scale"], "mode": foot["mode"],
              "effective_h_mm": foot["effective_h_mm"],
              "effective_v_mm": DECLARED_V_MM * foot["scale"]}
    if flat_ok:
        disagreement = abs(foot["scale"] - flat["scale"]) / flat["scale"]
        result["disagreement"] = disagreement
        if disagreement > 0.10:
            result["resolved"] = False
            result["reason"] = ("the two methods disagree by %.0f%% (flat floor x%.2f vs footprint "
                                "x%.2f), so neither is quoted" % (disagreement * 100, flat["scale"],
                                                                  foot["scale"]))
            return result
    elif foot["iou"] < 0.35:
        result["resolved"] = False
        result["reason"] = ("neither method found the floor (flat %.1f%%, footprint IoU %.2f); "
                            "check the pose convention before reading a number" % (
                                flat["floor_fraction"] * 100, foot["iou"]))
        return result
    result["resolved"] = True
    return result


def synthetic_depth(camera: dict, scale: float, mode: str, floor_z: float, height: int = 720,
                    width: int = 1280) -> np.ndarray:
    """A depth map of a bare floor, computed analytically for a *known* aperture."""
    eye = np.array(camera["eye"], dtype=float)
    forward, right, up = camera_basis(eye, np.array(camera["target"], dtype=float),
                                      np.array([0.0, 0.0, 1.0]))
    fx, fy = intrinsics(camera["focal_length_mm"], scale, mode, width, height)
    rows, columns = np.mgrid[0:height, 0:width]
    x = (columns - width / 2.0) / fx
    y = (rows - height / 2.0) / fy
    direction = (forward[None, None, :] + right[None, None, :] * x[:, :, None]
                 + up[None, None, :] * (-y)[:, :, None])
    origin_z = eye[2] - floor_z
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin_z / direction[:, :, 2]          # where the ray meets the floor plane, in `t`
    depth = t * (direction @ forward)               # ... expressed as camera-z
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)


def self_test() -> int:
    """Prove the estimator recovers an aperture it was not told, on depth it did not render."""
    print("self-test: synthesise depth for a known aperture, then measure it blind\n")
    cases = [
        ("tilted wide view (like overview_top)", dict(
            eye=[2.5, 0.8, 2.7], target=[2.45, 1.8, 0.0], focal_length_mm=6.2), 1.00),
        ("long-edge side view, declared aperture", dict(
            eye=[4.9, 0.15, 1.5], target=[4.9, 7.2, 1.5], focal_length_mm=8.31), 1.00),
        ("same lens, aperture 1.7x narrower", dict(
            eye=[4.9, 0.15, 1.5], target=[4.9, 7.2, 1.5], focal_length_mm=8.31), 0.60),
        ("same lens, aperture 1.4x wider", dict(
            eye=[4.9, 0.15, 1.5], target=[4.9, 7.2, 1.5], focal_length_mm=8.31), 1.40),
        ("eye-level view, narrow lens", dict(
            eye=[1.5, 1.2, 1.55], target=[3.5, 3.2, 1.4], focal_length_mm=22.0), 0.80),
    ]
    failures = 0
    for label, camera, truth in cases:
        depth = synthetic_depth(camera, truth, "fill", floor_z=0.0)
        result = measure(depth, camera, floor_z=0.0)
        if not result.get("resolved"):
            print("  %-42s UNRESOLVED: %s" % (label, result["reason"]))
            failures += 1
            continue
        error = abs(result["scale"] - truth) / truth
        ok = error <= 0.05
        failures += 0 if ok else 1
        flat = result["flat"]
        print("  %-42s truth x%.2f -> recovered x%.2f (%s fit, floor %.0f%%, offset %+.3f m)  %s"
              % (label, truth, result["scale"], result["mode"], flat["floor_fraction"] * 100,
                 flat["floor_offset_m"], "ok" if ok else "FAILED"))
    print("\n%s" % ("self-test passed: a known aperture is recovered within 5%"
                    if failures == 0 else f"self-test FAILED in {failures} case(s)"))
    return 1 if failures else 0


def report(delivery: pathlib.Path, camera_id: str | None, verbose: bool) -> int:
    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    version = manifest["scene_version"]
    attempt = pathlib.Path(manifest["isaac_artifact_dir"]).name
    request_path = delivery / "isaac_requests" / version / f"{attempt}.json"
    if not request_path.is_file():
        print(f"{delivery}: no request at {request_path}")
        return 1
    request = json.loads(request_path.read_text())
    cameras = {c["camera_id"]: c for c in request["cameras"]}
    floor_z = float((request.get("room") or {}).get("floor_z_m", 0.0) or 0.0)
    polygon = (manifest.get("zones") or {}).get("room_polygon_xy") or None
    room = manifest.get("room") or {}
    print("%s\n   version %s  %s  floor_z %.2f  room %.2f m2  footprint %d pts" % (
        delivery, version[:8], attempt, floor_z, room.get("area_m2") or 0.0,
        len(polygon or [])))

    for wanted in ([camera_id] if camera_id else sorted(cameras)):
        camera = cameras.get(wanted)
        if camera is None:
            print("   %-16s (not in this request)" % wanted)
            continue
        depth_path = delivery / "isaac" / version / attempt / f"{wanted}_depth_m.npy"
        if not depth_path.is_file():
            print("   %-16s (no depth saved for this camera)" % wanted)
            continue
        depth = np.load(depth_path)
        result = measure(depth, camera, floor_z, polygon=polygon, verbose=verbose)
        declared = camera.get("horizontal_aperture_mm", DECLARED_H_MM)
        if not result.get("resolved"):
            print("   %-16s UNRESOLVED: %s" % (wanted, result["reason"]))
            continue
        verdict = "matches the declaration" if abs(result["scale"] - 1.0) <= 0.05 else \
            ("NARROWER than declared by %.2fx" % (1.0 / result["scale"]) if result["scale"] < 1.0
             else "WIDER than declared by %.2fx" % result["scale"])
        flat = result["flat"]
        print("   %-16s f=%5.2f mm  declared %.3f -> measured %.3f mm (x%.2f, %s fit)  "
              "flat %.0f%%  footprint IoU %.2f  %s" % (
                  wanted, camera["focal_length_mm"], declared, result["effective_h_mm"],
                  result["scale"], result["mode"], flat["floor_fraction"] * 100,
                  result["footprint"]["iou"], verdict))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--delivery", type=pathlib.Path)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--verbose", action="store_true", help="print the scan at sample scales")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.delivery:
        return report(args.delivery, args.camera, args.verbose)
    parser.error("give --self-test or --delivery")


if __name__ == "__main__":
    sys.exit(main())
