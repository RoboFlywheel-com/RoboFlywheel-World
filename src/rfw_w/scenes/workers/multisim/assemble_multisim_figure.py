#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble the SPEC-014 five-backend figure from five backends' own products.

Reads what the five drivers wrote -- their RGB, their two depth passes, their
reports -- and writes the main figure (RGB over visible-geometry depth over the
absolute depth difference against the reference backend), the companion normals
figure (collision geometry, one world-space recipe for all five), the metrics,
and a provenance table from every panel back to a report digest.

This script measures; it does not decide whether the figure is acceptable.
``python -m rfw_w.scenes.validation.check_rigid_multisim`` recomputes the same numbers in a fresh
process and applies SPEC-014's thresholds. Keeping the two apart is what stops a
figure from grading itself.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      /opt/conda/envs/scenesmith/bin/python \\
      python -m rfw_w.scenes.workers.multisim.assemble_multisim_figure \\
      --renders outputs/_paper/spec014_renders_20260924_01 \\
      --scene outputs/_paper/spec014_common_scene_20260924_01 \\
      --output-dir outputs/_paper/spec014_figure_20260924_01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image

from rfw_w.scenes.workers.multisim.common import (DEPTH_CMAP, DEPTH_VMAX_M, DEPTH_VMIN_M, CommonScene, colourise_depth,
                    depth_to_world_normal_rgb, digest, write_json)

MM = 1 / 25.4

# SPEC-014 requirement 4 fixes this order; the caption names it too.
BACKEND_ORDER = ("genesis", "isaac", "mujoco", "pybullet", "sapien")
BACKEND_LABELS = {
    "genesis": "Genesis",
    "isaac": "Isaac Sim",
    "mujoco": "MuJoCo",
    "pybullet": "PyBullet",
    "sapien": "SAPIEN",
}
REFERENCE_BACKEND = "mujoco"

# ONE display lift for the figure's RGB row, applied identically to all five
# columns. MuJoCo's default headlight is ambient 0.1 against a grey default
# material, so 64.55% of its frame sits at exactly (13,13,13) and the reference
# column reads as an empty cell next to the other four. The lift is a display
# choice, not a measurement: the raw captures are written unchanged to
# `panels/<backend>_rgb.png` and it is only this row that is gamma-corrected.
# Applied equally, disclosed in the caption, and reversible from the panels.
RGB_DISPLAY_GAMMA = 0.55


def display_rgb(rgb: np.ndarray) -> np.ndarray:
    """The shared exposure lift. Equal for every column, by construction."""
    lifted = (np.asarray(rgb, dtype=np.float64) / 255.0) ** RGB_DISPLAY_GAMMA
    return (lifted * 255.0).clip(0, 255).astype(np.uint8)


def erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Shrink a boolean mask by n 4-neighbour steps. Used to drop the pixels
    within n of a silhouette edge, where two engines legitimately disagree
    because they sampled the same silhouette differently."""
    out = mask.copy()
    for _ in range(iterations):
        inner = out.copy()
        inner[1:, :] &= out[:-1, :]
        inner[:-1, :] &= out[1:, :]
        inner[:, 1:] &= out[:, :-1]
        inner[:, :-1] &= out[:, 1:]
        out = inner
    return out


def load_backend(renders: Path, backend: str) -> dict:
    directory = renders / backend
    report_path = directory / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"MULTISIM_BACKEND_UNAVAILABLE({backend}: no report.json)")
    report = json.loads(report_path.read_text())
    if report.get("status") != "passed":
        raise SystemExit(f"MULTISIM_BACKEND_UNAVAILABLE({backend}: status {report.get('status')})")
    depth_visible = np.load(directory / "depth_visible.npy", allow_pickle=False)
    depth_collision = np.load(directory / "depth_collision.npy", allow_pickle=False)
    mask_visible = np.load(directory / "depth_visible_mask.npy", allow_pickle=False)
    mask_collision = np.load(directory / "depth_collision_mask.npy", allow_pickle=False)
    rgb = np.asarray(Image.open(directory / "rgb.png").convert("RGB"))
    return {
        "backend": backend,
        "directory": directory,
        "report": report,
        "report_sha256": digest(report_path),
        "rgb": rgb,
        "rgb_sha256": digest(directory / "rgb.png"),
        "depth_visible": depth_visible,
        "depth_collision": depth_collision,
        "mask_visible": mask_visible,
        "mask_collision": mask_collision,
        "self_check_m": report.get("depth_self_check_m"),
    }


def compare_to_reference(sample: dict, reference: dict, contour_px: int = 2) -> dict:
    """Depth agreement on the common valid domain, minus a contour band.

    The band exists because the two engines rasterise the same silhouette with
    different coverage rules; a pixel on the edge of a chair leg is genuinely a
    different surface in each. Everything the figure claims about depth is
    computed after removing those pixels, and the fraction removed is reported
    so a reader can see how much was set aside.
    """
    common = (sample["mask_visible"] & reference["mask_visible"]
              & np.isfinite(sample["depth_visible"]) & np.isfinite(reference["depth_visible"]))
    if not common.any():
        raise SystemExit(f"MULTISIM_DEPTH_INVALID({sample['backend']}: no common valid pixels)")
    interior = erode(common, contour_px)
    if not interior.any():
        raise SystemExit(f"MULTISIM_DEPTH_INVALID({sample['backend']}: contour band consumed the mask)")
    difference = np.abs(sample["depth_visible"][interior].astype(np.float64)
                        - reference["depth_visible"][interior].astype(np.float64))
    union = (sample["mask_visible"] | reference["mask_visible"]).sum()
    return {
        "common_valid_pixels": int(common.sum()),
        "common_valid_fraction_of_reference": float(common.sum() / max(reference["mask_visible"].sum(), 1)),
        "pixels_used_after_contour_band": int(interior.sum()),
        "contour_band_px": contour_px,
        "pixels_set_aside_by_band": int(common.sum() - interior.sum()),
        "mask_iou": float((sample["mask_visible"] & reference["mask_visible"]).sum() / max(union, 1)),
        "abs_depth_median_m": float(np.median(difference)),
        "abs_depth_p95_m": float(np.percentile(difference, 95)),
        "abs_depth_max_m": float(difference.max()),
        "self_check_m": sample["self_check_m"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--diff-max-m", type=float, default=0.10,
                        help="display ceiling for the difference row, in metres")
    args = parser.parse_args()

    renders = args.renders.resolve()
    scene = CommonScene.load(args.scene.resolve())
    camera = scene.camera
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")

    samples = {backend: load_backend(renders, backend) for backend in BACKEND_ORDER}
    for backend, sample in samples.items():
        if sample["rgb"].shape[:2] != (camera.height, camera.width):
            raise SystemExit(f"MULTISIM_CAMERA_MISMATCH({backend}: rgb is "
                             f"{sample['rgb'].shape[1]}x{sample['rgb'].shape[0]})")
        if sample["depth_visible"].shape != (camera.height, camera.width):
            raise SystemExit(f"MULTISIM_CAMERA_MISMATCH({backend}: depth shape)")
        recorded = sample["report"].get("camera") or {}
        if (abs(float(recorded.get("focal_length_mm", -1)) - camera.focal_length_mm) > 1e-9
                or abs(float(recorded.get("fx", -1)) - camera.fx) > 1e-6):
            raise SystemExit(f"MULTISIM_CAMERA_MISMATCH({backend}: report camera differs from "
                             f"the frozen one)")
        if sample["report"].get("scene_version") != scene.scene_version:
            raise SystemExit(f"MULTISIM_SCENE_VERSION_MISMATCH({backend})")

    reference = samples[REFERENCE_BACKEND]
    measurements = {
        "reference_backend": REFERENCE_BACKEND,
        "scene_version": scene.scene_version,
        "camera": camera.as_record(),
        "depth_convention": "camera_z_meters",
        "backends": {},
    }
    for backend, sample in samples.items():
        entry = {
            "engine_version": sample["report"].get("engine_version"),
            "depth_self_check_m": sample["self_check_m"],
            "valid_pixel_fraction_visible": float(sample["mask_visible"].mean()),
            "valid_pixel_fraction_collision": float(sample["mask_collision"].mean()),
            "rgb_sha256": sample["rgb_sha256"],
            "report_sha256": sample["report_sha256"],
        }
        if backend != REFERENCE_BACKEND:
            entry["vs_reference"] = compare_to_reference(sample, reference)
        measurements["backends"][backend] = entry

    out.mkdir(parents=True, exist_ok=True)
    panels = out / "panels"
    panels.mkdir()
    for backend, sample in samples.items():
        Image.fromarray(sample["rgb"]).save(panels / f"{backend}_rgb.png")
        Image.fromarray(colourise_depth(sample["depth_visible"])).save(
            panels / f"{backend}_depth_visible.png")
        normals = depth_to_world_normal_rgb(sample["depth_collision"], camera)
        Image.fromarray(normals).save(panels / f"{backend}_normal_collision.png")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "axes.titleweight": "bold"})
    depth_cmap = plt.get_cmap(DEPTH_CMAP).copy()
    depth_cmap.set_bad("#202832")

    fig = plt.figure(figsize=(183 * MM, 122 * MM), facecolor="white")
    grid = fig.add_gridspec(3, 5, left=0.012, right=0.988, bottom=0.245, top=0.895,
                            wspace=0.035, hspace=0.20)
    for column, backend in enumerate(BACKEND_ORDER):
        sample = samples[backend]
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(display_rgb(sample["rgb"]))
        ax.axis("off")
        title = BACKEND_LABELS[backend]
        if backend == REFERENCE_BACKEND:
            title += "  (reference)"
        ax.set_title(title, loc="center", fontsize=7.0, pad=3.0, color="#243943")
        ax = fig.add_subplot(grid[1, column])
        ax.imshow(colourise_depth(sample["depth_visible"]))
        ax.axis("off")
        ax = fig.add_subplot(grid[2, column])
        if backend == REFERENCE_BACKEND:
            # The reference has nothing to compare against itself. A flat colour
            # block would read as "zero everywhere"; a neutral field says instead
            # that this cell is the baseline the others are measured from.
            ax.imshow(np.full(sample["depth_visible"].shape, 0.5), cmap=
                      matplotlib.colors.ListedColormap(["#e9edf0"]),
                      vmin=0.0, vmax=1.0)
            ax.set_title("baseline", loc="center", fontsize=6.0, pad=2.5, color="#64737b")
        else:
            delta = np.abs(sample["depth_visible"].astype(np.float64)
                           - reference["depth_visible"].astype(np.float64))
            delta[~(sample["mask_visible"] & reference["mask_visible"])] = np.nan
            ax.imshow(np.ma.masked_invalid(delta), cmap=plt.get_cmap("magma"),
                      vmin=0.0, vmax=args.diff_max_m, interpolation="nearest")
            record = measurements["backends"][backend]["vs_reference"]
            ax.set_title(f"med {record['abs_depth_median_m'] * 1000:.1f} mm · "
                         f"p95 {record['abs_depth_p95_m'] * 1000:.1f} mm",
                         loc="center", fontsize=5.6, pad=2.5, color="#64737b")
        ax.axis("off")
    # Two colourbars, each in its own reserved axis below its own row. Handing
    # matplotlib a group of axes to steal space from put the lower bar on top of
    # the difference panels it belonged under.
    depth_ax = fig.add_axes([0.30, 0.185, 0.40, 0.016])
    bar = fig.colorbar(ScalarMappable(norm=colors.Normalize(DEPTH_VMIN_M, DEPTH_VMAX_M),
                                      cmap=depth_cmap), cax=depth_ax, orientation="horizontal")
    bar.set_label("Visible-geometry camera-z depth (m) — one scale for all five columns",
                  fontsize=6.0, labelpad=1)
    bar.ax.tick_params(labelsize=5.6, length=2)
    diff_ax = fig.add_axes([0.30, 0.090, 0.40, 0.016])
    bar2 = fig.colorbar(ScalarMappable(norm=colors.Normalize(0.0, args.diff_max_m),
                                       cmap=plt.get_cmap("magma")), cax=diff_ax,
                        orientation="horizontal")
    bar2.set_label("|Δ depth| vs MuJoCo (m), contour band included", fontsize=6.0, labelpad=1)
    bar2.ax.tick_params(labelsize=5.6, length=2)
    fig.text(0.012, 0.975, "One frozen rigid scene, five engines, one camera",
             fontsize=9.5, weight="bold", color="#243943", va="top")
    fig.text(0.012, 0.008,
             "Same scene version, meshes, poses and frustum in every column. Depth is metres along the "
             "camera axis of the visible geometry; the difference row is against MuJoCo after a 2 px "
             "contour band.\n"
             "RGB appearance differs by renderer and is not the quantity being compared.\n"
             "The top row is shown under one shared exposure lift (gamma 0.55), applied identically to "
             "all five columns, because MuJoCo's headlight leaves its column near black; raw captures "
             "are in panels/.",
             fontsize=5.2, color="#64737b")
    main_pdf = out / "figure_multisim_depth.pdf"
    main_png = out / "figure_multisim_depth.png"
    fig.savefig(main_pdf, dpi=300)
    fig.savefig(main_png, dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(183 * MM, 48 * MM), facecolor="white")
    grid = fig.add_gridspec(1, 5, left=0.012, right=0.988, bottom=0.14, top=0.80, wspace=0.035)
    for column, backend in enumerate(BACKEND_ORDER):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(depth_to_world_normal_rgb(samples[backend]["depth_collision"], camera))
        ax.axis("off")
        ax.set_title(BACKEND_LABELS[backend], loc="center", fontsize=7.0, pad=3.0, color="#243943")
    fig.text(0.012, 0.93, "Collision geometry, world-space normals — one recipe for all five",
             fontsize=9.0, weight="bold", color="#243943", va="top")
    fig.text(0.012, 0.03,
             "Normals are derived from each engine's collision-geometry depth with the same central-"
             "difference recipe and the same camera; colour is (n+1)/2 in world axes, grey is where the "
             "depth is undefined. Collision meshes are convex exports, so this is not the visible shape.",
             fontsize=5.4, color="#64737b")
    normals_pdf = out / "figure_multisim_normals.pdf"
    normals_png = out / "figure_multisim_normals.png"
    fig.savefig(normals_pdf, dpi=300)
    fig.savefig(normals_png, dpi=300)
    plt.close(fig)

    write_json(out / "common_scene.json", scene.payload)
    write_json(out / "camera.json", camera.as_record())
    write_json(out / "measurements.json", measurements)
    write_json(out / "provenance.json", {
        "figure": "figure_multisim_depth",
        "renders": str(renders),
        "scene": str(args.scene.resolve()),
        "scene_version": scene.scene_version,
        "backend_order": list(BACKEND_ORDER),
        "reference_backend": REFERENCE_BACKEND,
        "panels": {
            backend: {
                "report": str(samples[backend]["directory"] / "report.json"),
                "report_sha256": samples[backend]["report_sha256"],
                "rgb_sha256": samples[backend]["rgb_sha256"],
                "engine_version": samples[backend]["report"].get("engine_version"),
            } for backend in BACKEND_ORDER
        },
    })
    print(json.dumps({
        "main_pdf": str(main_pdf), "normals_pdf": str(normals_pdf),
        "median_abs_depth_mm": {
            b: round(measurements["backends"][b]["vs_reference"]["abs_depth_median_m"] * 1000, 2)
            for b in BACKEND_ORDER if b != REFERENCE_BACKEND},
        "p95_abs_depth_mm": {
            b: round(measurements["backends"][b]["vs_reference"]["abs_depth_p95_m"] * 1000, 2)
            for b in BACKEND_ORDER if b != REFERENCE_BACKEND},
        "mask_iou": {
            b: round(measurements["backends"][b]["vs_reference"]["mask_iou"], 4)
            for b in BACKEND_ORDER if b != REFERENCE_BACKEND},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
