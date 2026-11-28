# SPDX-License-Identifier: Apache-2.0
"""Build a portable SBERT class index from a local catalogue and local weights."""

import argparse
from pathlib import Path
import sys


from rfw_w.scenes.assets.semantic_index import (
    SentenceTransformerEncoder, build_semantic_class_index, encoder_model_identity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="canonical model identity recorded in the index")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("local model directory is required")
    if args.out.exists():
        parser.error("output already exists; use a fresh index filename")
    encoder = SentenceTransformerEncoder(str(args.model_dir))
    if encoder_model_identity(encoder) != args.model_id:
        parser.error("local model identity disagrees with --model-id")
    build_semantic_class_index(args.catalog, args.out, encoder=encoder)
    print("semantic class index built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
