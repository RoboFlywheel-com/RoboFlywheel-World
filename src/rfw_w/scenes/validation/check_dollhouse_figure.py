#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently verify a dollhouse visible-vs-collision figure bundle.

A separate process that reads only the bundle and the render dirs it cites.
Every problem is a named gap; ``verified`` is the absence of problems, never a
belief in the bundle's own claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

# 183 mm at 300 dpi, in PostScript points (PDF) and pixels (PNG).
PAGE_WIDTH_PT = 183 * 72 / 25.4
PAGE_WIDTH_PX = 183 * 300 / 25.4
MEDIA_BOX = re.compile(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]")

PANELS = ("visible_rgb", "visible_depth", "collision_rgb", "collision_depth")
PANEL_FILES = {"visible_rgb": "{room}_rgb.png",
               "visible_depth": "{room}_depth_m.npy",
               "collision_rgb": "{room}_collision_rgb.png",
               "collision_depth": "{room}_collision_depth_m.npy"}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(bundle: Path) -> dict:
    provenance = read(bundle / "provenance.json")
    rows = provenance["rows"]
    problems: list[str] = []
    per_row: dict[str, dict] = {}
    for room, record in sorted(rows.items()):
        render_dir = Path(record["render_dir"])
        row_problems_before = len(problems)
        # The render dir's own provenance is the bundle's binding to how the
        # renders were produced; if it moved, every hash below belongs to a
        # different render than the figure claims.
        render_provenance = render_dir / "provenance.json"
        if (not render_provenance.is_file()
                or sha(render_provenance) != record["render_provenance_sha256"]):
            problems.append(f"PROVENANCE_STALE({room})")
        for name, sub in (("visible", "isaac"), ("collision", "isaac_collision")):
            report_path = render_dir / sub / "report.json"
            if not report_path.is_file():
                problems.append(f"MISSING_COMPANION({room},{sub}/report.json)")
                continue
            if sha(report_path) != record["report_sha256"][name]:
                problems.append(f"PROVENANCE_STALE({room},{name} report)")
        for panel in PANELS:
            info = record["panels"].get(panel)
            copy = bundle / "panels" / PANEL_FILES[panel].format(room=room)
            source = Path(info["source"]) if info else None
            if info is None or not copy.is_file() or not source.is_file():
                problems.append(f"MISSING_COMPANION({room},{panel})")
            elif sha(source) != info["sha256"]:
                problems.append(f"PROVENANCE_STALE({room},{panel} source)")
            elif sha(copy) != info["sha256"]:
                problems.append(f"PANEL_SOURCE_MISMATCH({room},{panel})")
        metrics_path = render_dir / "render_metrics.json"
        if not metrics_path.is_file():
            problems.append(f"MISSING_COMPANION({room},render_metrics.json)")
        else:
            metrics = read(metrics_path)
            for gap in metrics.get("gaps", []):
                problems.append(f"METRICS_GAP({room},{gap})")
            for gap in (metrics.get("collision") or {}).get("gaps", []):
                problems.append(f"METRICS_GAP({room},collision:{gap})")
        per_row[room] = {"render_dir": str(render_dir),
                         "ok": len(problems) == row_problems_before}
    png = bundle / "figure_dollhouse.png"
    pdf = bundle / "figure_dollhouse.pdf"
    for path in (png, pdf):
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"FIGURE_DIMENSIONS({path.name} missing)")
    if png.is_file() and png.stat().st_size:
        with Image.open(png) as picture:
            dpi = float(picture.info.get("dpi", (0.0, 0.0))[0])
            if abs(dpi - 300.0) > 1.0 or picture.width < PAGE_WIDTH_PX - 2:
                problems.append(
                    f"FIGURE_DIMENSIONS(png {picture.width}x{picture.height} @ {dpi:.1f} dpi, "
                    f"want 183 mm @ 300 dpi)")
    if pdf.is_file() and pdf.stat().st_size:
        match = MEDIA_BOX.search(pdf.read_bytes())
        if not match or abs(float(match.group(1)) - PAGE_WIDTH_PT) > 2.0:
            got = float(match.group(1)) if match else None
            problems.append(f"FIGURE_DIMENSIONS(pdf width {got} pt, want {PAGE_WIDTH_PT:.2f} pt)")
    return {"verified": not problems, "problems": problems,
            "rows": sorted(rows), "per_row": per_row,
            "visual_review": "not_run(user preview pending)"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")
    result = verify(args.bundle.resolve())
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
