# SPDX-License-Identifier: Apache-2.0
"""Remove image backgrounds with RMBG-2.0 and write RGBA PNG files."""

from __future__ import annotations

import argparse
import os

from rfw_w.assets.contracts.io import asset_path, load_manifest, select_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", required=True, help="RMBG-2.0 checkpoint")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    rows = select_rows(load_manifest(args.manifest), args.start, args.end)
    print(
        f"Loading RMBG-2.0 from {args.ckpt} on GPU {os.getenv('CUDA_VISIBLE_DEVICES', '0')}"
    )
    model = AutoModelForImageSegmentation.from_pretrained(
        args.ckpt, trust_remote_code=True
    )
    model.to("cuda").eval()

    for index, row in enumerate(rows, 1):
        source = asset_path(args.in_dir, row, ".png")
        output = asset_path(args.out_dir, row, ".png")
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(rows)}] skip {row['asset_id']}")
            continue
        if not source.exists():
            raise FileNotFoundError(f"Missing input image: {source}")
        image = Image.open(source).convert("RGB")
        width, height = image.size
        with torch.inference_mode():
            prediction = model(transform(image).unsqueeze(0).to("cuda"))[-1]
            mask = prediction.sigmoid().squeeze().cpu().numpy()
        mask_image = Image.fromarray((mask * 255).astype("uint8")).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        rgba = np.dstack([np.asarray(image), np.asarray(mask_image)])
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, "RGBA").save(output)
        print(f"[{index}/{len(rows)}] {row['asset_id']} -> {output}")


if __name__ == "__main__":
    main()
