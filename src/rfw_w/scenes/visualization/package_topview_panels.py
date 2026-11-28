#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package the top-view dollhouse renders as a per-room asset bundle.

One render dir per room (from ``render_room_preview.py --view dollhouse
--with-collision``); the bundle carries, per room, exactly three panels:

- ``<room>_rgb.png``           -- the visible render, bytes copied untouched
- ``<room>_depth_m.npy``       -- the raw camera-z depth in metres, untouched
- ``<room>_depth.png``         -- that depth colour-mapped for display
- ``<room>_collision_rgb.png`` -- the collision-mesh render, bytes untouched

Every depth PNG shares one metric scale (percentile 1-99 over the union of all
rooms' valid pixels), so rooms are comparable; invalid pixels are dark grey.
A compact montage (one row per room, 8 px gutters, no titles) is written for a
quick look -- the panels are the deliverable, the montage is only a preview.
The output directory must be fresh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from rfw_w.scenes.visualization.build_dollhouse_figure import depth_stats, load_row, shared_depth_range, sha

CAMERA_ID = "paper_dollhouse"
# Colour used for pixels with no valid depth, matched to build_dollhouse_figure.
INVALID_DEPTH_RGB = (32, 40, 50)
MONTAGE_GUTTER_PX = 8
MONTAGE_PANEL_WIDTH_PX = 640
MONTAGE_BACKGROUND = (255, 255, 255)


def colourise_depth(depth: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    """Full-resolution depth PNG: turbo_r over [vmin, vmax], invalid dark grey."""
    arr = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0)
    normalised = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap("turbo_r")
    rgba = (cmap(normalised)[..., :3] * 255).astype(np.uint8)
    rgba[~valid] = INVALID_DEPTH_RGB
    return Image.fromarray(rgba, mode="RGB")


def build_montage(rows: list[dict], panels: dict[str, dict[str, Path]],
                  depths: dict[str, np.ndarray], vmin: float, vmax: float,
                  out: Path) -> Path:
    """3 rows x 3 cols (RGB | depth | collision RGB), tight gutters, no margins."""
    cell_width = MONTAGE_PANEL_WIDTH_PX
    cells: list[list[Image.Image]] = []
    for row in rows:
        room = row["room"]
        rgb = Image.open(panels[room]["rgb"]).convert("RGB")
        scale = cell_width / rgb.width
        cell_height = int(round(rgb.height * scale))
        rgb = rgb.resize((cell_width, cell_height), Image.LANCZOS)
        depth_img = colourise_depth(depths[room], vmin, vmax).resize(
            (cell_width, cell_height), Image.NEAREST)
        collision = Image.open(panels[room]["collision_rgb"]).convert("RGB").resize(
            (cell_width, cell_height), Image.LANCZOS)
        cells.append([rgb, depth_img, collision])
    cell_height = cells[0][0].height
    gutter = MONTAGE_GUTTER_PX
    width = 3 * cell_width + 2 * gutter
    height = len(cells) * cell_height + (len(cells) - 1) * gutter
    canvas = Image.new("RGB", (width, height), MONTAGE_BACKGROUND)
    for r, row_cells in enumerate(cells):
        for c, cell in enumerate(row_cells):
            canvas.paste(cell, (c * (cell_width + gutter), r * (cell_height + gutter)))
    path = out / "montage_topview.png"
    canvas.save(path)
    return path


def build(selection: dict, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already has files; choose a fresh output directory")
    rows = selection.get("rows") or []
    if not rows:
        raise SystemExit("TOPVIEW_SELECTION_EMPTY")
    rooms = [r["room"] for r in rows]
    if len(set(rooms)) != len(rooms):
        raise SystemExit(f"TOPVIEW_DUPLICATE_ROW(room names must be unique: {rooms})")
    loaded = [load_row(row) for row in rows]

    depths: dict[str, np.ndarray] = {}
    for row in loaded:
        room = row["room"]
        depth = np.load(row["passes"]["visible"]["depth_path"], allow_pickle=False)
        rgb = np.asarray(Image.open(row["passes"]["visible"]["rgb_path"]).convert("RGB"))
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"TOPVIEW_DEPTH_SHAPE_MISMATCH({room})")
        depths[room] = depth
    # 95 rather than 99 at the top: with the walls stubbed, a few percent of the
    # pixels see the far exterior through the window openings (measured on the
    # kitchen: p95 = 6.7 m, p98 = 13.1 m), and letting them set the scale
    # compresses the whole room into one colour.
    vmin, vmax = shared_depth_range(list(depths.values()), lo=1.0, hi=95.0)

    out.mkdir(parents=True, exist_ok=True)
    panels: dict[str, dict[str, Path]] = {}
    panel_sources: dict[str, dict[str, Path]] = {}
    for row in loaded:
        room = row["room"]
        panels[room] = {}
        panel_sources[room] = {}
        rgb_target = out / f"{room}_rgb.png"
        depth_target = out / f"{room}_depth_m.npy"
        collision_target = out / f"{room}_collision_rgb.png"
        shutil.copy2(row["passes"]["visible"]["rgb_path"], rgb_target)
        shutil.copy2(row["passes"]["visible"]["depth_path"], depth_target)
        shutil.copy2(row["passes"]["collision"]["rgb_path"], collision_target)
        colourise_depth(depths[room], vmin, vmax).save(out / f"{room}_depth.png")
        panels[room] = {"rgb": rgb_target, "collision_rgb": collision_target}
        panel_sources[room] = {
            "rgb": row["passes"]["visible"]["rgb_path"],
            "depth": row["passes"]["visible"]["depth_path"],
            "collision_rgb": row["passes"]["collision"]["rgb_path"],
        }
    montage = build_montage(loaded, panels, depths, vmin, vmax, out)

    provenance = {
        "bundle": "topview_panels",
        "camera": CAMERA_ID,
        "depth": {"convention": "camera_z_meters",
                  "display_range_m": [vmin, vmax],
                  "display_colormap": "turbo_r",
                  "display_percentiles": [1.0, 95.0],
                  "invalid_pixel_rgb": list(INVALID_DEPTH_RGB)},
        "montage": {"path": montage.name,
                    "gutter_px": MONTAGE_GUTTER_PX,
                    "panel_width_px": MONTAGE_PANEL_WIDTH_PX,
                    "columns": ["rgb", "depth", "collision_rgb"]},
        "rows": {},
    }
    for row in loaded:
        room = row["room"]
        provenance["rows"][room] = {
            "task": row.get("task"),
            "render_dir": str(row["render_dir"]),
            "render_provenance_sha256": sha(row["provenance_path"]),
            "report_sha256": {name: sha(row["passes"][name]["report_path"])
                              for name in ("visible", "collision")},
            "panels": {name: {"source": str(path), "sha256": sha(path)}
                       for name, path in sorted(panel_sources[room].items())},
            "depth": depth_stats(depths[room]),
        }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (out / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return {"out": str(out), "montage": str(montage), "rows": rooms,
            "depth_range_m": [vmin, vmax]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path,
                        help='JSON {"rows": [{"room", "task", "render_dir"}, ...]}; each render '
                             "dir comes from render_room_preview.py --view dollhouse "
                             "--with-collision")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from rfw_w.scenes.visualization.build_room_showcase import read
    summary = build(read(args.selection.resolve()), args.output_dir.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
