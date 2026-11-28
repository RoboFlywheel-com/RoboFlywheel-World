#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package the RGB captures of several dressed rooms, and nothing else.

SPEC-016. `package_workspace_panels.py` builds the two-page figure and needs a
collision companion pass for its geometry page. When the ask is just "the RGB
close-ups", that pass is 40 s of GPU and a quarter of a gigabyte per room for a
picture nobody looks at -- so this packages the colour captures on their own.

Byte-for-byte copies: each room's whole-room overview and its workspace
close-up, the render report they came from, and a digest for each. No montage,
no colour work, no resizing.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.visualization.package_room_rgb \\
      --selection experiments/paper_dollhouse/workspace_selection_dressed.json \\
      --output-dir outputs/_paper/room_rgb_NEW
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

CAMERAS = {"room_overview_rgb.png": "paper_dollhouse",
           "workspace_rgb.png": "paper_workspace_detail"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(selection_path: Path, output: Path) -> dict:
    selection = json.loads(selection_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"OUTPUT_DIR_NOT_FRESH({output})")
    output.mkdir(parents=True, exist_ok=True)

    rooms: dict[str, dict] = {}
    for row in selection["rows"]:
        room = row["room"]
        render_dir = Path(row["render_dir"]).resolve()
        report_path = render_dir / "isaac" / "report.json"
        report = json.loads(report_path.read_text())
        if report.get("status") != "passed":
            raise ValueError(f"ROOM_RENDER_NOT_PASSED({room})")
        metrics = json.loads((render_dir / "render_metrics.json").read_text())
        if metrics.get("gaps"):
            raise ValueError(f"ROOM_RENDER_METRICS_GAP({room}, {metrics['gaps']})")
        provenance = json.loads((render_dir / "provenance.json").read_text())

        room_dir = output / room
        room_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict] = {}
        for name, camera_id in CAMERAS.items():
            capture = next((c for c in report["captures"] if c["camera_id"] == camera_id), None)
            if capture is None:
                raise ValueError(f"ROOM_CAPTURE_MISSING({room},{camera_id})")
            source = render_dir / "isaac" / capture["rgb_path"]
            shutil.copy2(source, room_dir / name)
            files[name] = {"source": str(source), "camera_id": camera_id,
                           "resolution": capture["resolution"],
                           "processing": "byte-for-byte copy",
                           "sha256": sha(room_dir / name)}
        shutil.copy2(report_path, room_dir / "render_report.json")
        files["render_report.json"] = {"source": str(report_path), "processing": "copy",
                                       "sha256": sha(room_dir / "render_report.json")}
        additions = provenance.get("additions") or {}
        rooms[room] = {
            "task": row.get("task"),
            "render_dir": str(render_dir),
            "source_delivery": provenance.get("source_delivery"),
            "scene_version": provenance.get("source_scene_version"),
            "robots": additions.get("robots"),
            "props": additions.get("props"),
            "support_table": (additions.get("support_table") or {}).get("size_m"),
            "target_pixels": metrics["cameras"][CAMERAS["workspace_rgb.png"]]["target_pixels"],
            "files": files,
        }

    payload = {
        "claim": "RGB captures only: one whole-room overview and one workspace close-up per room, "
                 "copied byte for byte. No depth, no collision pass, no montage.",
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha(selection_path),
        "rooms": rooms,
        "user_visual_review": "not_run",
        "robot_physics": "not_run",
        "robot_ik": "not_run",
    }
    (output / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), "rooms": sorted(rooms),
            "target_pixels": {r: v["target_pixels"] for r, v in rooms.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.selection, args.output_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
