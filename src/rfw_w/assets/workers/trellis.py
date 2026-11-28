# SPDX-License-Identifier: Apache-2.0
"""Generate textured GLB assets from RGBA images with TRELLIS.2."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from rfw_w.assets.contracts.io import asset_path, load_manifest, select_rows

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", required=True, help="TRELLIS.2 checkpoint")
    parser.add_argument(
        "--trellis-root", required=True, help="TRELLIS.2 source checkout"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation", type=int, default=200_000)
    parser.add_argument(
        "--pipeline-type", choices=["512", "1024", "1024_cascade", "1536_cascade"]
    )
    parser.add_argument("--max-num-tokens", type=int, default=49_152)
    parser.add_argument("--remesh", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    input_root = Path(args.in_dir).resolve()
    output_root = Path(args.out_dir).resolve()
    trellis_root = Path(args.trellis_root).resolve()
    if not trellis_root.is_dir():
        raise FileNotFoundError(f"TRELLIS.2 source directory not found: {trellis_root}")
    sys.path.insert(0, str(trellis_root))
    os.chdir(trellis_root)

    import o_voxel
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    rows = select_rows(load_manifest(manifest), args.start, args.end)
    pending = [
        row
        for row in rows
        if args.overwrite or not asset_path(output_root, row, ".glb").exists()
    ]
    print(
        f"Loading TRELLIS.2 from {args.ckpt} on GPU {os.getenv('CUDA_VISIBLE_DEVICES', '0')}"
    )
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt)
    pipeline.cuda()

    for index, row in enumerate(pending, 1):
        source = asset_path(input_root, row, ".png")
        output = asset_path(output_root, row, ".glb")
        if not source.exists():
            raise FileNotFoundError(f"Missing RGBA image: {source}")
        started = time.perf_counter()
        meshes, latents = pipeline.run(
            Image.open(source),
            seed=args.seed,
            pipeline_type=args.pipeline_type,
            max_num_tokens=args.max_num_tokens,
            return_latent=True,
        )
        mesh = meshes[0]
        mesh.simplify(16_777_216)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation,
            texture_size=args.texture_size,
            remesh=args.remesh,
            verbose=False,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(output), extension_webp=True)
        del meshes, latents, mesh, glb
        torch.cuda.empty_cache()
        print(
            f"[{index}/{len(pending)}] {row['asset_id']} -> {output} ({time.perf_counter()-started:.1f}s)"
        )


if __name__ == "__main__":
    main()
