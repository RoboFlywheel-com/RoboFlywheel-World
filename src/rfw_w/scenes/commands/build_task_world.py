#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build G2/G3 from a validated task-world graph without an external model request."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.orchestrator.scene_generation_agent import SceneGenerationAgent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    graph = TaskWorldGraph.from_dict(json.loads(args.graph.read_text()))
    report = SceneGenerationAgent().run(graph, args.output_dir)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
