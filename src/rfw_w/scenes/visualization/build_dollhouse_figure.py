#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compose the dollhouse visible-vs-collision comparison figure.

One row per room, four columns: visible RGB | visible depth | collision RGB |
collision depth. Inputs are render dirs produced by ``render_room_preview.py
--view dollhouse --with-collision``: the collision pass draws every instance as
its collision mesh from the same camera, so the two halves of a row differ only
in which geometry the renderer saw.

Raw panel bytes are copied into the bundle untouched; only the displayed depth
colour mapping is applied, and every depth panel shares one metric scale so the
rows are comparable. The output directory must be fresh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image

from rfw_w.scenes.visualization.build_room_showcase import read, sha

MM = 1 / 25.4
CAMERA_ID = "paper_dollhouse"
# Row pitch in mm: a 1600x1200 panel in a quarter of the 183 mm width needs
# about 34 mm of height; the rest is the row label and spacing.
ROW_PITCH_MM = 36


def row_label(room: str, task: str | None) -> str:
    """Short rotated row label: the room plus the relation, not the task sentence.

    The label is the first column's ylabel, rotated 90 degrees, so its length
    must fit inside one row's height; a full task phrase ("pick the cup — cup
    ON desk") is longer than the row and bleeds into the neighbouring rows
    (measured on dollhouse_figure_01). The relation tail is enough at figure
    scale; the full task string stays in the bundle's provenance.json.
    """
    if not task:
        return room
    relation = task
    for separator in ("—", "–", ":"):
        if separator in task:
            relation = task.rsplit(separator, 1)[-1].strip()
    label = f"{room} — {relation}"
    # A long relation wraps into two lines: rotated text spends the row's
    # height, and two half-length lines fit where one full-length one bleeds.
    return textwrap.fill(label, width=24) if len(label) > 24 else label


# fig.text is not wrapped at the figure edge -- an overlong line is simply cut
# off at the canvas (the first figure lost the footnote's tail, "recorded i…").
# Wrap explicitly to a width that fits inside 183 mm at the footnote font size.
FOOTNOTE_WRAP_CHARS = 100


def footnote_text() -> str:
    return textwrap.fill(
        "Left half of a row: the rendered scene. Right half: the same request rendered "
        "with every instance drawn as its collision mesh (instances without one keep "
        "their visual). Depth panels share one metric scale (percentile 1-99 of all "
        "valid pixels); dark grey marks pixels with no valid depth. Layouts are "
        "hand-curated deltas on the cited deliveries, recorded in each render dir's "
        "provenance.json.", width=FOOTNOTE_WRAP_CHARS)


def shared_depth_range(depths, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    """One (vmin, vmax) for every depth panel, from the union of valid pixels.

    Percentiles rather than raw min/max: the extremes of a dollhouse frame are a
    handful of far facade/sky-adjacent pixels, and letting them set the scale
    compresses the room's own depths into one colour. Valid matches the worker's
    own definition (finite and positive).
    """
    valid = []
    for depth in depths:
        arr = np.asarray(depth, dtype=np.float64)
        valid.append(arr[np.isfinite(arr) & (arr > 0)])
    valid = [v for v in valid if v.size]
    if not valid:
        raise ValueError("DOLLHOUSE_DEPTH_INVALID(no valid pixels in any panel)")
    union = np.concatenate(valid)
    return float(np.percentile(union, lo)), float(np.percentile(union, hi))


def depth_stats(depth: np.ndarray) -> dict:
    arr = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0)
    stats = {"valid_fraction": float(valid.mean()) if arr.size else 0.0,
             "min_m": None, "max_m": None}
    if valid.any():
        stats["min_m"] = float(arr[valid].min())
        stats["max_m"] = float(arr[valid].max())
    return stats


def load_pass(render_dir: Path, sub: str, room: str) -> dict:
    """One worker pass (visible or collision) of a dollhouse render dir."""
    report_path = render_dir / sub / "report.json"
    if not report_path.is_file():
        raise ValueError(f"DOLLHOUSE_COMPANION_MISSING({room},{sub}/report.json)")
    report = read(report_path)
    if report.get("status") != "passed":
        raise ValueError(f"DOLLHOUSE_RENDER_NOT_PASSED({room},{sub})")
    capture = next((c for c in report.get("captures", []) if c.get("camera_id") == CAMERA_ID),
                   None)
    if capture is None:
        raise ValueError(f"DOLLHOUSE_COMPANION_MISSING({room},{sub}:{CAMERA_ID})")
    rgb_path = report_path.parent / capture["rgb_path"]
    depth_rel = capture.get("depth_path")
    depth_path = report_path.parent / depth_rel if depth_rel else None
    if not rgb_path.is_file():
        raise ValueError(f"DOLLHOUSE_COMPANION_MISSING({room},{sub}:rgb)")
    if depth_path is None or not depth_path.is_file():
        raise ValueError(f"DOLLHOUSE_COMPANION_MISSING({room},{sub}:depth)")
    return {"report_path": report_path, "capture": capture,
            "rgb_path": rgb_path, "depth_path": depth_path}


def load_row(row: dict) -> dict:
    room = row["room"]
    render_dir = Path(row["render_dir"]).resolve()
    for name in ("provenance.json", "render_metrics.json"):
        if not (render_dir / name).is_file():
            raise ValueError(f"DOLLHOUSE_COMPANION_MISSING({room},{name})")
    return {"room": room, "task": row.get("task"), "render_dir": render_dir,
            "provenance_path": render_dir / "provenance.json",
            "metrics_path": render_dir / "render_metrics.json",
            "passes": {name: load_pass(render_dir, sub, room)
                       for name, sub in (("visible", "isaac"), ("collision", "isaac_collision"))}}


def build(selection: dict, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already has files; choose a fresh output directory")
    rows = selection.get("rows") or []
    if not rows:
        raise SystemExit("DOLLHOUSE_SELECTION_EMPTY")
    rooms = [r["room"] for r in rows]
    if len(set(rooms)) != len(rooms):
        raise SystemExit(f"DOLLHOUSE_DUPLICATE_ROW(room names must be unique: {rooms})")
    loaded = [load_row(row) for row in rows]
    images: dict[str, dict] = {}
    for row in loaded:
        room = row["room"]
        images[room] = {}
        for name in ("visible", "collision"):
            rgb = np.asarray(Image.open(row["passes"][name]["rgb_path"]).convert("RGB"))
            depth = np.load(row["passes"][name]["depth_path"], allow_pickle=False)
            if depth.shape != rgb.shape[:2]:
                raise ValueError(f"DOLLHOUSE_DEPTH_SHAPE_MISMATCH({room},{name})")
            images[room][name] = {"rgb": rgb, "depth": depth}
    vmin, vmax = shared_depth_range(
        [entry["depth"] for row in images.values() for entry in row.values()])

    out.mkdir(parents=True, exist_ok=True)
    panels_dir = out / "panels"
    panels_dir.mkdir()
    panel_sources: dict[str, dict[str, Path]] = {}
    for row in loaded:
        room = row["room"]
        panel_sources[room] = {}
        for name, suffix in (("visible", ""), ("collision", "_collision")):
            rgb_target = panels_dir / f"{room}{suffix}_rgb.png"
            depth_target = panels_dir / f"{room}{suffix}_depth_m.npy"
            shutil.copy2(row["passes"][name]["rgb_path"], rgb_target)
            shutil.copy2(row["passes"][name]["depth_path"], depth_target)
            panel_sources[room][f"{name}_rgb"] = row["passes"][name]["rgb_path"]
            panel_sources[room][f"{name}_depth"] = row["passes"][name]["depth_path"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "axes.titleweight": "bold"})
    count = len(loaded)
    fig = plt.figure(figsize=(183 * MM, (count * ROW_PITCH_MM + 32) * MM),
                     facecolor="white")
    grid = fig.add_gridspec(count, 4, left=0.085, right=0.995, bottom=0.16, top=0.885,
                            wspace=0.05, hspace=0.22)
    cmap = plt.get_cmap("turbo_r").copy()
    cmap.set_bad("#202832")
    titles = ("RGB (visible)", "Depth (visible)", "RGB (collision)", "Depth (collision)")
    for index, row in enumerate(loaded):
        room = row["room"]
        for column, name in ((0, "visible"), (1, "visible"), (2, "collision"), (3, "collision")):
            ax = fig.add_subplot(grid[index, column])
            if column % 2 == 0:
                ax.imshow(images[room][name]["rgb"])
            else:
                depth = images[room][name]["depth"]
                valid = np.isfinite(depth) & (depth > 0)
                ax.imshow(np.ma.masked_where(~valid, depth), cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if index == 0:
                ax.set_title(titles[column], loc="left", fontsize=7.0, pad=3)
            if column == 0:
                ax.set_ylabel(row_label(room, row.get("task")),
                              fontsize=6.8, color="#243943", labelpad=4)
    bar = fig.colorbar(ScalarMappable(norm=colors.Normalize(vmin, vmax), cmap=cmap),
                       ax=fig.axes, orientation="horizontal", fraction=0.03, pad=0.05,
                       aspect=50)
    bar.set_label("Camera-z depth (m) — one scale for every panel", fontsize=6.2, labelpad=2)
    bar.ax.tick_params(labelsize=5.8, length=2)
    fig.text(0.012, 0.982,
             selection.get("title", "Visible geometry vs collision geometry, per room"),
             fontsize=9.5, weight="bold", color="#243943", va="top")
    fig.text(0.012, 0.008, footnote_text(), fontsize=5.6, color="#64737b", va="bottom")
    pdf = out / "figure_dollhouse.pdf"
    png = out / "figure_dollhouse.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    measurements = {"depth_range_m": [vmin, vmax], "depth_percentiles": [1.0, 99.0],
                    "depth_colormap": "turbo_r", "rows": {}}
    provenance = {"figure": "figure_dollhouse", "camera": CAMERA_ID,
                  "selection": selection,
                  "depth": {"convention": "camera_z_meters",
                            "display_range_m": [vmin, vmax],
                            "display_colormap": "turbo_r", "percentiles": [1.0, 99.0]},
                  "rows": {}}
    for row in loaded:
        room = row["room"]
        metrics = read(row["metrics_path"])
        collision_metrics = metrics.get("collision") or {}
        camera_metrics = (metrics.get("cameras") or {}).get(CAMERA_ID, {})
        collision_camera = (collision_metrics.get("cameras") or {}).get(CAMERA_ID, {})
        measurements["rows"][room] = {
            "task": row.get("task"),
            "render_dir": str(row["render_dir"]),
            "panels": {name: {"source": str(path), "sha256": sha(path)}
                       for name, path in sorted(panel_sources[room].items())},
            "depth": {name: depth_stats(images[room][name]["depth"])
                      for name in ("visible", "collision")},
            "render_metrics": {
                "visible": {"clipped_fraction": camera_metrics.get("clipped_fraction"),
                            "gaps": metrics.get("gaps", [])},
                "collision": {"clipped_fraction": collision_camera.get("clipped_fraction"),
                              "gaps": collision_metrics.get("gaps", [])},
            },
        }
        provenance["rows"][room] = {
            "task": row.get("task"),
            "render_dir": str(row["render_dir"]),
            "render_provenance_sha256": sha(row["provenance_path"]),
            "report_sha256": {name: sha(row["passes"][name]["report_path"])
                              for name in ("visible", "collision")},
            "panels": {name: {"source": str(path), "sha256": sha(path)}
                       for name, path in sorted(panel_sources[room].items())},
        }
    (out / "measurements.json").write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n")
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (out / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return {"pdf": str(pdf), "png": str(png), "rows": rooms,
            "depth_range_m": [vmin, vmax]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path,
                        help='JSON {"rows": [{"room", "task", "render_dir"}, ...]}; each render '
                             "dir comes from render_room_preview.py --view dollhouse "
                             "--with-collision")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = build(read(args.selection.resolve()), args.output_dir.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
