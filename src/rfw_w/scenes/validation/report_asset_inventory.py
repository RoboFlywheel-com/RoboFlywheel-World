#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare delivered asset diversity and measured readiness, without exposing asset paths."""
import argparse
import json
from pathlib import Path


def inventory(delivery: Path) -> dict:
    data = json.loads((delivery / "delivery_manifest.json").read_text())
    assets = data.get("assets_used", [])
    furnishing = data.get("furnishing", {})
    return {
        "delivery": str(delivery), "scene_version": data["scene_version"],
        "placed": len(assets),
        "unique_assets": len({(a["provider"], a["asset_id"]) for a in assets}),
        "providers": sorted({a["provider"] for a in assets}),
        "assets": [{k: a[k] for k in ("entity_id", "provider", "asset_id", "dimensions_m")}
                   for a in assets],
        "missing": {key: furnishing.get(key, []) for key in
                    ("unplaced_core", "unplaced_task_closure", "unplaced_periphery", "periphery_dropped")},
        "achieved": data["resolution"]["achieved"],
        "failed_gates": data["resolution"]["failed_gates"],
        "not_run_gates": data["resolution"]["not_run_gates"],
        "visual_critique": data.get("visual_critique"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([inventory(path) for path in args.delivery], indent=2, ensure_ascii=False))
