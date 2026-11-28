#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent contract check for the two-room task graph and named negative fixtures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    graph = TaskWorldGraph.from_dict(json.loads(args.fixture.read_text(encoding="utf-8")))
    roundtrip = TaskWorldGraph.from_dict(graph.to_dict()).content_hash == graph.content_hash
    try:
        graph.check_evidence({"graph_hash": "0" * 64, "subject_ids": ["door"]})
        stale_rejected = False
    except GraphError as error:
        stale_rejected = error.code == "GRAPH_STALE_EVIDENCE"
    negatives = {}
    for path in sorted(args.negative_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            TaskWorldGraph.from_dict(case["graph"])
            got = "accepted"
        except GraphError as error:
            got = error.code
        negatives[path.stem] = {"expected": case["expected_error"], "actual": got,
                                "passed": got == case["expected_error"]}
    report = {"roundtrip": roundtrip, "stable_hash": roundtrip,
              "stale_evidence_rejected": stale_rejected,
              "rooms": sum(n["kind"] == "room" for n in graph.to_dict()["nodes"]),
              "portals": sum(n["kind"] == "portal" for n in graph.to_dict()["nodes"]),
              "steps": len(graph.to_dict()["steps"]), "negatives": negatives,
              "fixture_checks_passed": roundtrip and stale_rejected and bool(negatives) and
                                       all(case["passed"] for case in negatives.values())}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["fixture_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
