#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package overview and task close-ups from new Isaac visible/collision passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image


CAMERAS = ("paper_dollhouse", "paper_workspace_detail")
MM = 1 / 25.4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(render_dir: Path, subdir: str, camera_id: str) -> tuple[Path, Path, Path]:
    report_path = render_dir / subdir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "passed":
        raise ValueError(f"WORKSPACE_RENDER_NOT_PASSED({render_dir.name},{subdir})")
    entry = next((x for x in report["captures"] if x["camera_id"] == camera_id), None)
    if entry is None:
        raise ValueError(f"WORKSPACE_CAMERA_MISSING({render_dir.name},{subdir},{camera_id})")
    return (report_path.parent / entry["rgb_path"],
            report_path.parent / entry["depth_path"], report_path)


def build(selection_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("WORKSPACE_OUTPUT_NOT_FRESH")
    selection = json.loads(selection_path.read_text())
    rows = selection["rows"]
    # Three distinct rooms, but not three particular names. The names used to be pinned to
    # office/kitchen/bedroom, which made the packager the thing that decided what a room was
    # called: when the kitchen turned out to contain no stove, sink or fridge -- only an island,
    # a stool, a cabinet and a plant -- and the label had to stop claiming otherwise, the
    # rename had to pass through here for no reason. What this file is entitled to insist on is
    # that the figure has three rows and that no room is packed twice.
    if len(rows) != 3 or len({x["room"] for x in rows}) != 3:
        raise ValueError("WORKSPACE_SELECTION_EXPECTS_THREE_DISTINCT_ROOMS")
    output.mkdir(parents=True)
    panels = output / "panels"
    panels.mkdir()
    arrays = {}
    provenance = {"selection_sha256": sha(selection_path), "rows": {},
                  "camera_ids": list(CAMERAS), "claim": "figure_only_curated_layout_and_static_geometry"}
    all_depth = []
    for row in rows:
        room = row["room"]
        source = Path(row["render_dir"]).resolve()
        record = json.loads((source / "provenance.json").read_text())
        metrics = json.loads((source / "render_metrics.json").read_text())
        if metrics["gaps"] or metrics["collision"]["gaps"]:
            raise ValueError(f"WORKSPACE_RENDER_METRICS_GAP({room})")
        if not record.get("workspace_detail"):
            raise ValueError(f"WORKSPACE_DETAIL_NOT_RECORDED({room})")
        if (record.get("dollhouse") or {}).get("wall_clip") is not None:
            raise ValueError(f"WORKSPACE_WALLS_CLIPPED({room})")
        if room == "bedroom" and not any(x["instance_id"] == "bed" for x in record.get("asset_replacement", [])):
            raise ValueError("WORKSPACE_BED_REPLACEMENT_MISSING")
        arrays[room] = {}
        panel_records = {}
        for label, subdir, camera_id in (
                ("layout", "isaac", CAMERAS[0]),
                ("visible", "isaac", CAMERAS[1]),
                ("collision", "isaac_collision", CAMERAS[1])):
            rgb, depth, report = capture(source, subdir, camera_id)
            for name, src in (("rgb", rgb), ("depth_m", depth)):
                dst = panels / f"{room}_{label}_{name}{src.suffix}"
                shutil.copy2(src, dst)
                panel_records[f"{label}_{name}"] = {
                    "source": str(src), "source_sha256": sha(src),
                    "copy": str(dst.relative_to(output)), "copy_sha256": sha(dst),
                    "report": str(report), "report_sha256": sha(report)}
            rgb_array = np.asarray(Image.open(rgb).convert("RGB"))
            depth_array = np.load(depth, allow_pickle=False)
            if rgb_array.shape[:2] != depth_array.shape:
                raise ValueError(f"WORKSPACE_DEPTH_SHAPE_MISMATCH({room},{label})")
            arrays[room][label] = (rgb_array, depth_array)
            if label != "layout":
                valid = np.isfinite(depth_array) & (depth_array > 0)
                all_depth.append(depth_array[valid])
        provenance["rows"][room] = {
            "task": row["task"], "render_dir": str(source),
            "render_provenance_sha256": sha(source / "provenance.json"),
            "walls_preserved": True,
            "asset_replacement": record.get("asset_replacement"),
            "panels": panel_records,
            "target_pixels": metrics["cameras"][CAMERAS[1]]["target_pixels"]}
    union = np.concatenate(all_depth)
    vmin, vmax = (float(np.percentile(union, p)) for p in (1, 95))
    provenance["depth"] = {"convention": "camera_z_meters", "range_m": [vmin, vmax],
                           "percentiles": [1, 95], "invalid_colour": "#202832"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#202832")
    for figure, columns, title in (
            ("figure_task_layout", (("layout", "rgb"), ("visible", "rgb")),
             "Room layout and manipulable workspace"),
            ("figure_task_geometry", (("visible", "depth"), ("collision", "rgb"),
                                      ("collision", "depth")),
             "Workspace depth and collision geometry")):
        ncols = len(columns)
        fig, axes = plt.subplots(3, ncols, figsize=(183 * MM, (200 if ncols == 2 else 121) * MM),
                                 facecolor="white", squeeze=False)
        fig.subplots_adjust(left=0.065, right=0.99, top=0.91,
                            bottom=0.065 if ncols == 2 else 0.14,
                            wspace=0.025, hspace=0.09)
        for i, row in enumerate(rows):
            room = row["room"]
            for j, (pass_name, kind) in enumerate(columns):
                ax = axes[i, j]
                rgb, depth = arrays[room][pass_name]
                if kind == "rgb":
                    ax.imshow(rgb, interpolation="nearest")
                else:
                    valid = np.isfinite(depth) & (depth > 0)
                    ax.imshow(np.ma.masked_where(~valid, depth), cmap=cmap,
                              vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.axis("off")
                if i == 0:
                    label = ("Layout (top)" if pass_name == "layout" else
                             "Manipulable objects (close)" if kind == "rgb" and ncols == 2 else
                             "Visible depth (m)" if pass_name == "visible" else
                             "Collision geometry" if kind == "rgb" else "Collision depth (m)")
                    ax.set_title(label, loc="left", fontsize=7, pad=3)
                if j == 0:
                    ax.text(-0.02, 0.5, room.title(), transform=ax.transAxes,
                            rotation=90, va="center", ha="right", fontsize=7)
        fig.text(0.012, 0.985, title, fontsize=9, weight="bold", va="top")
        if ncols == 3:
            bar = fig.colorbar(ScalarMappable(norm=colors.Normalize(vmin, vmax), cmap=cmap),
                               ax=axes.ravel().tolist(), orientation="horizontal",
                               fraction=0.025, pad=0.06, aspect=60)
            bar.set_label("Camera-z distance (m); common scale for visible and collision depth",
                          fontsize=6)
            bar.ax.tick_params(labelsize=6)
        fig.savefig(output / f"{figure}.pdf", dpi=300)
        fig.savefig(output / f"{figure}.png", dpi=300)
        plt.close(fig)
    return {"output": str(output), "depth_range_m": [vmin, vmax],
            "target_pixels": {room: provenance["rows"][room]["target_pixels"] for room in arrays}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.selection, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
