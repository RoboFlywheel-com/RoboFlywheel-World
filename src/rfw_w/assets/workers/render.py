# SPDX-License-Identifier: Apache-2.0
"""Render generated GLBs from fixed views using a local Blender executable."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rfw_w.assets.contracts.io import asset_path, load_manifest, part_dir, select_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--views", default="front,left,right")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = select_rows(load_manifest(args.manifest), args.start, args.end)
    script = Path(__file__).with_name("blender.py")
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    for index, row in enumerate(rows, 1):
        source = asset_path(args.in_dir, row, ".glb").resolve()
        output_dir = part_dir(args.out_dir, row).resolve()
        expected = [output_dir / f"{row['asset_id']}_{view}.png" for view in views]
        if all(path.exists() for path in expected) and not args.overwrite:
            print(f"[{index}/{len(rows)}] skip {row['asset_id']}")
            continue
        if not source.exists():
            raise FileNotFoundError(f"Missing GLB: {source}")
        command = [
            args.blender,
            "--background",
            "--python",
            str(script),
            "--",
            "--glb",
            str(source),
            "--out-dir",
            str(output_dir),
            "--resolution",
            str(args.resolution),
            "--samples",
            str(args.samples),
            "--views",
            ",".join(views),
        ]
        subprocess.run(command, check=True)
        print(f"[{index}/{len(rows)}] {row['asset_id']} -> {output_dir}")


if __name__ == "__main__":
    main()
