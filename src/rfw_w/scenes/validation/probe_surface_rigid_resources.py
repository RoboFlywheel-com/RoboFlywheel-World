# SPDX-License-Identifier: Apache-2.0
"""Read-only resource census for surface styling and SimReady tool design.

Only environment variable names, material IDs, categories and counts are public.
This report is acceptance evidence; it must never be copied into a model prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time


VARIABLES = (
    "PVSCENE_MATERIALS_ROOT", "PVSCENE_SIMREADY_INDEX",
    "PVSCENE_SIMREADY_ASSETS_ROOT", "PVSCENE_SIMREADY_SEMANTIC_INDEX",
    "PVSCENE_PREPARED_DIR",
)
CLASSES = ("Mug", "Coffee Cup", "Cup", "Book", "Notebook", "Bowl", "Plate")


def census() -> dict:
    started = time.monotonic()
    report = {"schema_version": "pvscene.surface-rigid-resource-probe/v1",
              "environment": {key: "set" if os.environ.get(key) else "unset"
                              for key in VARIABLES},
              "model_requests_used": 0, "retrieval": "not_run",
              "preparation": "not_run", "render": "not_run"}
    root = os.environ.get("PVSCENE_MATERIALS_ROOT")
    materials = []
    if root and Path(root).is_dir():
        for directory in sorted(Path(root).iterdir()):
            if not directory.is_dir():
                continue
            files = [p for p in directory.iterdir()
                     if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            channels = {channel: any(p.stem.lower().endswith("_" + suffix.lower())
                                    for p in files)
                        for channel, suffix in (("base_color", "Color"),
                                                ("normal_gl", "NormalGL"),
                                                ("roughness", "Roughness"))}
            materials.append({"material_id": directory.name, "channels": channels,
                              "complete_pbr": all(channels.values())})
    report["materials"] = {"status": "ran" if root and Path(root).is_dir() else "blocked",
                           "reason_code": None if root and Path(root).is_dir() else "MATERIAL_ENV_MISSING",
                           "candidates": materials, "count": len(materials)}
    index = os.environ.get("PVSCENE_SIMREADY_INDEX")
    assets = os.environ.get("PVSCENE_SIMREADY_ASSETS_ROOT")
    rigid = {"status": "blocked", "reason_code": "SIMREADY_ENV_MISSING"}
    if index and Path(index).is_file() and assets and Path(assets).is_dir():
        rows = json.loads(Path(index).read_bytes())
        counts = Counter()
        samples = {category: [] for category in CLASSES}
        total = rigid_count = 0
        for aid, row in rows.items():
            if aid == "_meta" or not isinstance(row, dict):
                continue
            total += 1
            category = str(row.get("super_category", ""))
            counts[category] += 1
            rigid_count += row.get("physics_type") == "rigid"
            if category in samples and len(samples[category]) < 3:
                prefix = Path(str(row.get("assets_prefix", "")))
                leaf = str(row.get("leaf", aid))
                safe = not prefix.is_absolute() and ".." not in prefix.parts and Path(leaf).name == leaf
                exists = safe and (Path(assets) / prefix / "mesh" / (leaf + ".obj")).is_file()
                samples[category].append({"asset_id": aid, "source_mesh_exists": exists,
                                          "positive_declared_mass": isinstance(row.get("mass"), (int, float))
                                          and row["mass"] > 0})
        rigid = {"status": "ran", "reason_code": None, "catalogue_count": total,
                 "declared_rigid_count": rigid_count,
                 "library_version": rows.get("_meta", {}).get("library_version", "unknown"),
                 "category_counts": {category: counts[category] for category in CLASSES},
                 "related_category_counts": {category: count for category, count in sorted(counts.items())
                                             if any(word in category.lower() for word in
                                                    ("cup", "mug", "bowl", "plate", "book", "jar"))},
                 "source_samples": samples,
                 "coverage_note": "At most three source-file probes per named category; no geometry or physics verification."}
    report["simready"] = rigid
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.exit(2, "RESOURCE_OUTPUT_EXISTS(fresh directory required)\n")
    args.output_dir.mkdir(parents=True)
    try:
        report = census()
    except (OSError, ValueError, TypeError) as error:
        report = {"status": "failed", "reason_code": "RESOURCE_PROBE_FAILED",
                  "error_type": type(error).__name__, "model_requests_used": 0}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
