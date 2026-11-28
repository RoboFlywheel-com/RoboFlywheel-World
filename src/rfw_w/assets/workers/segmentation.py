# SPDX-License-Identifier: Apache-2.0
"""Create P3-SAM part-wise face annotations for generated GLB assets."""

from __future__ import annotations

import argparse
import os
import sys
import time

for key, value in {
    "CUMM_DISABLE_JIT": "1",
    "SPCONV_DISABLE_JIT": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}.items():
    os.environ.setdefault(key, value)

from rfw_w.assets.contracts.io import (
    asset_path,
    load_manifest,
    part_dir,
    select_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--p3sam-root", required=True, help="Path to Hunyuan3D-Part/P3-SAM"
    )
    parser.add_argument("--ckpt", required=True, help="p3sam.safetensors")
    parser.add_argument("--sonata-ckpt", help="Local Sonata checkpoint; avoids cache downloads")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--prompt-bs", type=int, default=4)
    parser.add_argument(
        "--offline", action="store_true", help="Use cached Hugging Face weights only"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def restore_prediction(source, result, extent):
    """Face labels refer to P3-SAM's cleaned mesh; restore its source frame."""
    import numpy as np

    if len(result) < 3:
        raise ValueError("P3-SAM must return the mesh used for face predictions")
    raw_ids = np.asarray(result[1])
    predicted = result[2].copy()
    if raw_ids.ndim != 1 or len(raw_ids) != len(predicted.faces):
        raise ValueError("P3-SAM face assignments do not match the prediction mesh")
    predicted.apply_scale(extent)
    if not np.allclose(predicted.bounds, source.bounds, rtol=1e-5, atol=1e-7):
        raise ValueError("P3-SAM prediction changed the source coordinate frame")
    return predicted, raw_ids


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import trimesh
    from rfw_w.assets.workers.p3sam_patch import patch_p3sam
    from rfw_w.assets.annotation.palette import COLORS

    p3sam_root = os.path.abspath(args.p3sam_root)
    demo_root = os.path.join(p3sam_root, "demo")
    for path in (demo_root, p3sam_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    patch_p3sam(p3sam_root, sonata_ckpt=args.sonata_ckpt)

    from models import sonata

    original_load = sonata.load

    def load_without_flash(*positional, **keywords):
        custom_config = dict(keywords.get("custom_config") or {})
        custom_config["enable_flash"] = False
        keywords["custom_config"] = custom_config
        return original_load(*positional, **keywords)

    sonata.load = load_without_flash
    from auto_mask import AutoMask

    rows = select_rows(load_manifest(args.manifest), args.start, args.end)
    print(
        f"Loading P3-SAM from {args.ckpt} on GPU {os.getenv('CUDA_VISIBLE_DEVICES', '0')}"
    )
    model = AutoMask(ckpt_path=args.ckpt, threshold=args.threshold, post_process=True)

    for index, row in enumerate(rows, 1):
        source = asset_path(args.in_dir, row, ".glb")
        output_dir = part_dir(args.out_dir, row)
        face_ids_path = output_dir / "face_ids.npy"
        if face_ids_path.exists() and not args.overwrite:
            print(f"[{index}/{len(rows)}] skip {row['asset_id']}")
            continue
        if not source.exists():
            raise FileNotFoundError(f"Missing GLB: {source}")

        started = time.perf_counter()
        mesh = trimesh.load(source, force="mesh")
        extent = float(np.ptp(mesh.vertices, axis=0).max())
        if extent <= 0:
            raise ValueError(f"Mesh has zero extent: {source}")
        normalized = mesh.copy()
        normalized.apply_scale(1.0 / extent)
        result = model.predict_aabb(
            normalized,
            save_path=None,
            threshold=args.threshold,
            post_process=True,
            show_info=False,
            prompt_bs=args.prompt_bs,
        )
        predicted, raw_ids = restore_prediction(mesh, result, extent)
        face_ids = np.full(raw_ids.shape, -1, dtype=np.int64)
        unique_ids = np.unique(raw_ids[raw_ids >= 0])
        for part_index, raw_id in enumerate(unique_ids):
            face_ids[raw_ids == raw_id] = part_index

        face_colors = np.zeros((len(face_ids), 4), dtype=np.uint8)
        face_colors[:, 3] = 255
        for face_index, part_id in enumerate(face_ids):
            if part_id >= 0:
                face_colors[face_index, :3] = COLORS[int(part_id) % len(COLORS)]
        # Predictions use normalized geometry; exported annotations retain source frame.
        colored = predicted.copy()
        colored.visual = trimesh.visual.ColorVisuals(
            mesh=colored, face_colors=face_colors
        )
        colored.metadata["face_ids"] = face_ids.tolist()

        output_dir.mkdir(parents=True, exist_ok=True)
        colored.export(output_dir / "mesh_part_seg.glb")
        np.save(face_ids_path, face_ids)
        elapsed = time.perf_counter() - started
        write_json(
            output_dir / "meta.json",
            {
                **row,
                "source_glb": str(source),
                "n_parts": len(unique_ids),
                "n_faces": len(face_ids),
                "source_n_faces": len(mesh.faces),
                "threshold": args.threshold,
                "segmentation_seconds": round(elapsed, 3),
            },
        )
        print(
            f"[{index}/{len(rows)}] {row['asset_id']}: {len(unique_ids)} parts -> {output_dir} ({elapsed:.1f}s)"
        )


if __name__ == "__main__":
    main()
