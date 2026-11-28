# SPDX-License-Identifier: Apache-2.0
"""Generate one seed image per manifest row with FLUX.2-klein."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from rfw_w.assets.contracts.io import (
    asset_path,
    load_manifest,
    select_rows,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", required=True, help="FLUX.2-klein checkpoint")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError(
            "Flux2KleinPipeline is unavailable. Install a FLUX.2-klein-compatible "
            "diffusers build in the image-generation environment."
        ) from exc

    rows = select_rows(load_manifest(args.manifest), args.start, args.end)
    missing_prompt = [row["asset_id"] for row in rows if not row.get("prompt")]
    if missing_prompt:
        raise ValueError(f"Rows missing prompt (first 5): {missing_prompt[:5]}")

    print(
        f"Loading FLUX.2-klein from {args.ckpt} on GPU {os.getenv('CUDA_VISIBLE_DEVICES', '0')}"
    )
    pipeline = Flux2KleinPipeline.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16)
    pipeline.to("cuda")

    for index, row in enumerate(rows, 1):
        output = asset_path(args.out_dir, row, ".png")
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(rows)}] skip {row['asset_id']}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = pipeline(
            prompt=row["prompt"],
            height=args.size,
            width=args.size,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cpu").manual_seed(
                stable_seed(row, args.seed)
            ),
        )
        result.images[0].save(output)
        print(
            f"[{index}/{len(rows)}] {row['asset_id']} -> {output} ({time.perf_counter()-started:.1f}s)"
        )


if __name__ == "__main__":
    main()
