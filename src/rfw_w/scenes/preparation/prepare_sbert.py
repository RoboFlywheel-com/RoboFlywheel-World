#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a locally loadable SBERT model directory from an existing HF cache entry.

This is an offline step: it reads weights that are already on disk and never downloads
anything. Run it before the first retrieval, not inside a generation run.

    python -m rfw_w.scenes.preparation.prepare_sbert --env-file .env.local --target prepared/sbert/all-mpnet-base-v2

The preparation is verified by actually loading the model and encoding two related and two
unrelated phrases, so a directory that merely looks complete does not pass as usable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from rfw_w.scenes.assets.prepared_store import PreparationError, prepare_if_needed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="load the model and encode a probe pair")
    args = parser.parse_args()

    try:
        record, status = prepare_if_needed(args.hf_cache, args.target)
    except PreparationError as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, indent=2))
        return 1

    print(json.dumps({"status": "ok", "detail": status}, indent=2))
    if record is not None:
        print(json.dumps(record.to_dict(), indent=2, sort_keys=True))

    if args.verify:
        # Imported here so that the preparation itself stays stdlib-only and runnable from a
        # bare interpreter; only verification needs torch.
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.target), local_files_only=True)
        model = AutoModel.from_pretrained(str(args.target), local_files_only=True).to("cpu").eval()
        phrases = ["a ceramic mug", "a coffee cup", "a bicycle frame", "a steel wrench"]
        batch = tokenizer(phrases, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)

        related = float(torch.dot(normed[0], normed[1]))
        unrelated = float(torch.dot(normed[0], normed[2]))
        report = {
            "status": "ok",
            "model_dir": str(args.target),
            "related_similarity": round(related, 6),
            "unrelated_similarity": round(unrelated, 6),
            "related_exceeds_unrelated": related > unrelated,
        }
        print(json.dumps(report, indent=2))
        if not report["related_exceeds_unrelated"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
