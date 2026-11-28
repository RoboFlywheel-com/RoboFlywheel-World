#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rank SimReady library candidates by how colourful their own texture is.

SPEC-016. The paper rooms read as grey boxes with three pale objects on the
desk, and the fix the reference figure suggests is more objects **with colour**.
Choosing them by name alone does not work: "Mug" covers 3667 assets, and most
of a random handful are white ceramic.

So this measures instead of guessing. For each candidate it reads the diffuse
texture the release ships beside the mesh and reports, per asset, the mean
saturation and value in HSV plus the dominant hue. Nothing is prepared, loaded
into a scene or rendered -- it is a read-only pass over the library index and
the PNGs, so it costs seconds rather than minutes.

Saturation is the ranking key because that is the axis the current figures are
short on; value is reported beside it so a reader can see a "colourful" asset
that is actually a dark one.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.visualization.pick_colourful_props \\
      --category "Rubik Cube" --category "Toy Car" --max-dim-m 0.25 \\
      --per-category 6 --output-dir outputs/_probe/props_01
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
from pathlib import Path
import sys

from rfw_w.scenes.visualization.paper_case_common import write


def texture_stats(path: Path) -> dict | None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            small = image.convert("RGB").resize((64, 64))
            pixels = list(small.getdata())
    except Exception:  # noqa: BLE001 - an unreadable texture is simply not a candidate
        return None
    saturation = value = 0.0
    hues: dict[int, int] = {}
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        saturation += s
        value += v
        if s > 0.25:
            bucket = int(h * 12) % 12
            hues[bucket] = hues.get(bucket, 0) + 1
    count = len(pixels)
    dominant = max(hues.items(), key=lambda kv: kv[1])[0] if hues else None
    coloured = sum(hues.values())
    # Hue *variety*, not just saturation. Measured 2026-09-26: ranking on saturation alone put a
    # "Rubik Cube" first whose texture is an abstract orange wash -- 0.947 saturation, one hue, and
    # it renders as a plain orange block. The reference figure's appeal is several hues in frame, so
    # a candidate that holds three or more meaningful hues beats a very saturated single-hue one.
    spread = sorted((c for c in hues.values()), reverse=True)
    significant = sum(1 for c in hues.values() if coloured and c / coloured >= 0.05)
    return {
        "mean_saturation": round(saturation / count, 4),
        "mean_value": round(value / count, 4),
        "dominant_hue_bucket": dominant,
        "hue_buckets_over_5pct": significant,
        "top_hue_shares": [round(c / coloured, 3) for c in spread[:3]] if coloured else [],
        "colourful_pixel_fraction": round(coloured / count, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", action="append", required=True,
                        help="super_category as the index spells it; repeatable")
    parser.add_argument("--max-dim-m", type=float, default=0.30)
    parser.add_argument("--min-dim-m", type=float, default=0.03)
    parser.add_argument("--scan-per-category", type=int, default=60,
                        help="how many index entries per category to measure")
    parser.add_argument("--per-category", type=int, default=5,
                        help="how many to keep in the ranked output")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    index_path = os.environ.get("PVSCENE_SIMREADY_INDEX")
    assets_root = os.environ.get("PVSCENE_SIMREADY_ASSETS_ROOT")
    if not index_path or not assets_root:
        raise SystemExit("PROP_LIBRARY_UNSET(PVSCENE_SIMREADY_INDEX/ASSETS_ROOT)")
    index = json.loads(Path(index_path).read_text())
    root = Path(assets_root)

    wanted = {c.lower() for c in args.category}
    buckets: dict[str, list] = {c: [] for c in args.category}
    for leaf, record in index.items():
        category = str(record.get("super_category", ""))
        if category.lower() not in wanted:
            continue
        target = next(c for c in args.category if c.lower() == category.lower())
        if len(buckets[target]) >= args.scan_per_category:
            continue
        dims = record.get("dimensions_m") or []
        if not dims or not (args.min_dim_m <= max(dims) <= args.max_dim_m):
            continue
        buckets[target].append((leaf, record))

    results: dict[str, list] = {}
    for category, candidates in buckets.items():
        scored = []
        for leaf, record in candidates:
            mesh_dir = root / record["assets_prefix"] / "mesh"
            textures = sorted(mesh_dir.glob("*.png")) + sorted(mesh_dir.glob("*.jpg"))
            if not textures:
                continue
            stats = texture_stats(textures[0])
            if stats is None:
                continue
            scored.append({
                "leaf": leaf,
                "assets_prefix": record["assets_prefix"],
                "super_category": record.get("super_category"),
                "dimensions_m": record.get("dimensions_m"),
                "material": record.get("material"),
                "texture": textures[0].name,
                **stats,
            })
        scored.sort(key=lambda r: (-r["hue_buckets_over_5pct"], -r["mean_saturation"],
                                   -r["colourful_pixel_fraction"]))
        results[category] = scored[:args.per_category]

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write(out / "candidates.json", {
        "categories": results,
        "rule": "ranked by mean HSV saturation of the release's own diffuse texture, within the "
                "declared size band; nothing was prepared or rendered",
        "size_band_m": [args.min_dim_m, args.max_dim_m],
        "scanned_per_category": args.scan_per_category,
    })
    print(json.dumps({c: [(r["leaf"], r["mean_saturation"], r["mean_value"],
                           [round(v, 3) for v in r["dimensions_m"]]) for r in rows]
                      for c, rows in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
