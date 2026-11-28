# SPDX-License-Identifier: Apache-2.0
"""Physical preparation and format assembly in a configured clean environment."""

from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--visual-mesh")
    p.add_argument("--face-ids", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--voxel-resolution", type=int, default=64)
    p.add_argument("--coacd-timeout", type=float, default=600)
    args = p.parse_args()
    from rfw_w.assets.contracts.request import AssetRequest
    from rfw_w.assets.packaging.build import build_package

    request = AssetRequest.from_dict(json.loads(Path(args.request).read_text()))
    build_package(
        request,
        args.mesh,
        args.face_ids,
        args.annotations,
        args.out,
        args.voxel_resolution,
        args.coacd_timeout,
        args.visual_mesh,
    )


if __name__ == "__main__":
    main()
