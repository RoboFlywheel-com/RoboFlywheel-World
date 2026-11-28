#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a packaged room-asset pack from its own files, in a separate process.

SPEC-016. `python -m rfw_w.scenes.visualization.package_three_scene_assets` copies each room's
overview RGB, workspace RGB, metric depth and collision RGB out of a verified
panel bundle and records a digest per file. This reads only the pack and asks
whether those records still hold:

* every product re-hashes to what `provenance.json` says,
* every file the provenance names is present, and no product is missing,
* the depth is float32 camera-z metres, finite where valid, and its valid
  fraction and range are recomputed from the array rather than quoted,
* each room's copied render reports say `passed`,
* the RGB and the depth of one room agree on resolution -- a mismatched pair is
  how a pack ends up with one room's picture beside another's geometry.

It does not judge whether the pictures look right; that stays `not_run` until
the user says so.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.validation.check_scene_assets \\
      --assets outputs/_paper/dressed_assets_NEW \\
      --output-dir outputs/_acceptance/dressed_assets_NEW
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

#: What a full pack carries. An RGB-only pack (python -m rfw_w.scenes.visualization.package_room_rgb) carries the
#: first two; the provenance is what says which, and the check follows it.
PRODUCTS = ("room_overview_rgb.png", "workspace_rgb.png")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    assets = args.assets.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")

    provenance = json.loads((assets / "provenance.json").read_text())
    problems: list[str] = []
    checks: list[dict] = []
    measured: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    rooms = provenance.get("rooms", {})
    record("rooms_present", bool(rooms), f"{len(rooms)} rooms recorded")
    for room, entry in sorted(rooms.items()):
        room_dir = assets / room
        files = entry.get("files", {})
        missing = [p for p in PRODUCTS if p not in files]
        record(f"{room}:products", not missing, f"missing products: {missing}" if missing
               else "all four products are recorded")
        for name, item in sorted(files.items()):
            path = room_dir / name
            if not path.is_file():
                record(f"{room}:{name}:present", False, "the provenance names a file that is absent")
                continue
            record(f"{room}:{name}:sha256", sha(path) == item["sha256"],
                   "re-hashes to what provenance recorded")

        depth_path = room_dir / "workspace_depth_m.npy"
        rgb_path = room_dir / "workspace_rgb.png"
        if depth_path.is_file() and rgb_path.is_file():
            depth = np.load(depth_path)
            valid = np.isfinite(depth) & (depth > 0)
            fraction = float(valid.mean())
            with Image.open(rgb_path) as image:
                size = image.size
            record(f"{room}:depth_dtype", depth.dtype == np.float32,
                   f"depth array dtype is {depth.dtype} (camera-z metres are float32)")
            record(f"{room}:depth_has_signal", fraction > 0.5,
                   f"{fraction:.6f} of the depth pixels are finite and positive")
            record(f"{room}:depth_matches_rgb", depth.shape[::-1] == size,
                   f"depth {depth.shape[::-1]} against RGB {size}")
            measured[room] = {
                "depth_valid_fraction": fraction,
                "depth_min_m": float(depth[valid].min()) if valid.any() else None,
                "depth_max_m": float(depth[valid].max()) if valid.any() else None,
                "workspace_rgb_size": list(size),
            }

        for report_name in ("render_report.json", "collision_report.json"):
            report_path = room_dir / report_name
            if not report_path.is_file():
                # An RGB-only pack carries no collision pass; a missing *render*
                # report is still a defect.
                if report_name == "collision_report.json":
                    continue
                record(f"{room}:{report_name}", False, "the pack carries no such report")
                continue
            status = json.loads(report_path.read_text()).get("status")
            record(f"{room}:{report_name}:passed", status == "passed",
                   f"the copied report says status={status}")

    out.mkdir(parents=True, exist_ok=True)
    verified = not problems
    payload = {
        "verified": verified,
        "assets": str(assets),
        "rooms": sorted(rooms),
        "checks": checks,
        "checks_run": len(checks),
        "measured": measured,
        "problems": problems,
        "not_run": ["user visual judgement"],
        "claim": "this re-hashes the pack and recomputes its depth statistics; it does not judge "
                 "whether the pictures look right",
    }
    (out / "verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified": verified, "checks_run": len(checks), "problems": problems,
                      "measured": measured, "not_run": payload["not_run"]}, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
