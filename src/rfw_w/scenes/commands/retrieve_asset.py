# SPDX-License-Identifier: Apache-2.0
"""Retrieve and optionally prepare a real asset using only RFW-W source.

The catalogue and asset root are required. SBERT uses a local model and class index;
--scoring lexical explicitly selects a deterministic metadata-only alternative.
"""

import argparse
import json
import os
from pathlib import Path
import sys


from rfw_w.scenes.adapters.simready_retrieval import (
    SimreadyConfig, build_encoder, build_engine, retrieve_entity, to_asset_record,
)
from rfw_w.scenes.contracts.scene_spec import EntitySpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=os.environ.get("PVSCENE_SIMREADY_INDEX"))
    parser.add_argument("--assets-root", type=Path, default=os.environ.get("PVSCENE_SIMREADY_ASSETS_ROOT"))
    parser.add_argument("--semantic-index", type=Path, default=os.environ.get("PVSCENE_SIMREADY_SEMANTIC_INDEX"))
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("PVSCENE_PREPARED_DIR"))
    parser.add_argument("--entity", type=Path, required=True, help="EntitySpec JSON input")
    parser.add_argument("--admit", nargs="+", required=True, help="permitted category names")
    parser.add_argument("--scoring", choices=("semantic", "lexical"), default="semantic")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.index is None or args.assets_root is None:
        parser.error("catalogue and assets root are required")
    if args.out.resolve().is_relative_to(args.assets_root.resolve()):
        parser.error("output must be outside the asset library")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "rfw.retrieval-check/v1", "scoring": args.scoring,
              "status": "failed", "preparation": "not_run", "model_requests_used": 0}
    try:
        meta = json.loads(args.index.read_bytes()).get("_meta", {})
        config = SimreadyConfig(args.index, args.assets_root, args.semantic_index,
                                library_version=meta.get("library_version", "unknown"),
                                prepared_root=args.out / "prepared")
        entity = EntitySpec.from_dict(json.loads(args.entity.read_bytes()))
        encoder = build_encoder(args.model_dir) if args.scoring == "semantic" else None
        engine = build_engine(config, encoder=encoder, scoring=args.scoring)
        outcome = retrieve_entity(engine, entity, config=config, admitted_categories=args.admit)
        report["retrieval"] = outcome.to_dict()
        if outcome.ok:
            if args.prepare:
                record, _ = to_asset_record(outcome.candidates[0], config)
                report.update(preparation="passed", asset_id=record.key.asset_id,
                              dimensions_m=record.dimensions_m.to_list(),
                              canonical_frame=record.canonical_frame)
            report["status"] = "passed"
    except Exception as exc:
        # Paths and catalogue contents remain in local artifacts, not error text.
        report["error_type"] = type(exc).__name__
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
