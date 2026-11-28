#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent source-to-panel audit of a curated task-layout figure bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle: Path) -> dict:
    record = json.loads((bundle / "provenance.json").read_text())
    problems = []
    counts = {}
    for room, row in record["rows"].items():
        render_dir = Path(row["render_dir"])
        if sha(render_dir / "provenance.json") != row["render_provenance_sha256"]:
            problems.append(f"WORKSPACE_PROVENANCE_STALE({room})")
        original = json.loads((render_dir / "provenance.json").read_text())
        if (original.get("dollhouse") or {}).get("wall_clip") is not None:
            problems.append(f"WORKSPACE_WALLS_CLIPPED({room})")
        for name, panel in row["panels"].items():
            src, dst = Path(panel["source"]), bundle / panel["copy"]
            if not src.is_file() or not dst.is_file():
                problems.append(f"WORKSPACE_PANEL_MISSING({room},{name})")
                continue
            if sha(src) != panel["source_sha256"] or sha(dst) != panel["copy_sha256"]:
                problems.append(f"WORKSPACE_PANEL_HASH_MISMATCH({room},{name})")
            if panel["source_sha256"] != panel["copy_sha256"]:
                problems.append(f"WORKSPACE_PANEL_SOURCE_MISMATCH({room},{name})")
            if sha(Path(panel["report"])) != panel["report_sha256"]:
                problems.append(f"WORKSPACE_REPORT_STALE({room},{name})")
            if name.endswith("depth_m"):
                depth = np.load(dst, allow_pickle=False)
                if depth.dtype != np.float32 or not np.any(np.isfinite(depth) & (depth > 0)):
                    problems.append(f"WORKSPACE_DEPTH_INVALID({room},{name})")
        pixels = row["target_pixels"]
        target = next((key for key in pixels if key in ("cup", "jar", "alarm_clock")), None)
        counts[room] = {"target": target, "pixels": pixels.get(target, 0)}
        if target is None or pixels[target] < 3000:
            problems.append(f"WORKSPACE_TARGET_TOO_SMALL({room},{target},{pixels.get(target, 0)})")
        if room == "bedroom":
            asset = next((x for x in row.get("asset_replacement", [])
                          if x.get("instance_id") == "bed"), {})
            for kind in ("visual", "collision"):
                path = Path(asset.get(f"{kind}_path", ""))
                if not path.is_file() or sha(path) != asset.get(f"figure_{kind}_sha256"):
                    problems.append(f"WORKSPACE_BED_PROVENANCE_INVALID({kind})")
    for stem in ("figure_task_layout", "figure_task_geometry"):
        png, pdf = bundle / f"{stem}.png", bundle / f"{stem}.pdf"
        if not png.is_file() or not pdf.is_file():
            problems.append(f"WORKSPACE_FIGURE_MISSING({stem})")
            continue
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))[0]
            if abs(dpi - 300) > 1 or image.width < 2159:
                problems.append(f"WORKSPACE_FIGURE_SIZE_INVALID({stem},{image.width},{dpi})")
    return {"verified": not problems, "problems": problems, "target_pixels": counts,
            "visual_review": "not_run(user review pending)", "bundle": str(bundle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error("fresh output directory required")
    out.mkdir(parents=True)
    report = verify(args.bundle.resolve())
    (out / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
