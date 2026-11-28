#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package three room overview/workspace captures as separate image assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rfw_w.scenes.validation.check_workspace_panels import verify as verify_source


#: How many rows the figure has. The NAMES come from the source bundle, not from here -- see
#: package_workspace_panels.py for why a packager naming the rooms was the wrong shape.
ROOM_COUNT = 3
PRODUCTS = {
    "room_overview_rgb.png": "layout_rgb",
    "workspace_rgb.png": "visible_rgb",
    "workspace_depth_m.npy": "visible_depth_m",
    "workspace_collision_rgb.png": "collision_rgb",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def colour_depth(depth: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.zeros(depth.shape, dtype=np.float32)
    scaled[valid] = np.clip((depth[valid] - vmin) / (vmax - vmin), 0, 1)
    rgb = (plt.get_cmap("viridis_r")(scaled)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = (32, 40, 50)  # #202832, the source bundle's invalid colour
    return Image.fromarray(rgb)


def package(source: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"OUTPUT_DIR_NOT_FRESH({output})")
    source_check = verify_source(source.resolve())
    if not source_check["verified"]:
        raise ValueError(f"SOURCE_BUNDLE_INVALID({source_check['problems']})")
    original = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    rooms = tuple(original["rows"])
    if len(rooms) != ROOM_COUNT:
        raise ValueError(f"THREE_SCENE_ROOM_COUNT({len(rooms)} != {ROOM_COUNT})")
    vmin, vmax = original["depth"]["range_m"]
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
        raise ValueError("WORKSPACE_DEPTH_SCALE_INVALID")

    records = {}
    for room in rooms:
        row = original["rows"][room]
        products = {}
        for destination, key in PRODUCTS.items():
            panel = row["panels"][key]
            src = source / panel["copy"]
            if sha(src) != panel["source_sha256"]:
                raise ValueError(f"WORKSPACE_PANEL_HASH_MISMATCH({room},{key})")
            products[destination] = (src, key, panel)
        depth = np.load(products["workspace_depth_m.npy"][0], allow_pickle=False)
        with Image.open(products["workspace_rgb.png"][0]) as close_rgb, \
             Image.open(products["workspace_collision_rgb.png"][0]) as collision_rgb:
            if depth.dtype != np.float32 or depth.shape != (close_rgb.height, close_rgb.width):
                raise ValueError(f"WORKSPACE_DEPTH_SHAPE_MISMATCH({room})")
            if collision_rgb.size != close_rgb.size:
                raise ValueError(f"WORKSPACE_CAMERA_MISMATCH({room}: collision)")
        if not np.any(np.isfinite(depth) & (depth > 0)):
            raise ValueError(f"WORKSPACE_DEPTH_INVALID({room})")
        records[room] = {"source_render": row["render_dir"], "task": row["task"],
                         "walls_preserved": row["walls_preserved"],
                         "target_pixels": row["target_pixels"], "products": products}

    output.mkdir(parents=True, exist_ok=True)
    for room, record in records.items():
        room_out = output / room
        room_out.mkdir()
        packaged = {}
        for destination, (src, key, panel) in record.pop("products").items():
            dst = room_out / destination
            shutil.copyfile(src, dst)
            if sha(dst) != panel["source_sha256"]:
                raise ValueError(f"WORKSPACE_COPY_MISMATCH({room},{key})")
            packaged[destination] = {"source": str(src.resolve()),
                                     "source_report": panel["report"],
                                     "sha256": sha(dst), "processing": "byte-for-byte copy"}
        preview = room_out / "workspace_depth.png"
        colour_depth(np.load(room_out / "workspace_depth_m.npy", allow_pickle=False),
                     vmin, vmax).save(preview)
        packaged[preview.name] = {"source": "workspace_depth_m.npy",
                                  "sha256": sha(preview),
                                  "processing": "viridis_r preview; shared range in metres"}
        for source_name, destination in (("isaac", "render_report.json"),
                                         ("isaac_collision", "collision_report.json")):
            report_src = Path(record["source_render"]) / source_name / "report.json"
            report_dst = room_out / destination
            shutil.copyfile(report_src, report_dst)
            record[destination] = {"sha256": sha(report_dst), "source": str(report_src.resolve())}
        record["files"] = packaged

    ledger = {
        "claim": "three curated rigid rooms: room overview RGB, workspace RGB, visible camera-z depth, workspace collision RGB",
        "source_bundle": str(source.resolve()),
        "source_bundle_provenance_sha256": sha(source / "provenance.json"),
        "source_bundle_verification": {"verified": True, "problems": []},
        "depth_preview": {"colormap": "viridis_r", "range_m": [vmin, vmax],
                          "invalid_colour": "#202832", "raw_definition": "camera_z_meters"},
        "rooms": records,
        "visual_review": "not_run(user review pending)",
    }
    (output / "provenance.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = package(args.source, args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "rooms": list(result["rooms"]),
                      "depth_range_m": result["depth_preview"]["range_m"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
