#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently verify a G1 planning directory without model access or raw replies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from rfw_w.scenes.agents.task_world_planner import SENSITIVE_TEXT, request_hash, validate_plan_graph
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.orchestrator.plan_task_world import initial_intent_version


def verify(directory: Path, request_path: Path) -> dict[str, object]:
    problems: list[str] = []
    required = ("planning.json", "tool_request.json", "tool_result.json",
                "task_world_graph.json", "task_world_graph.svg")
    if any(not (directory / name).is_file() for name in required):
        return {"verified": False, "problems": ["REQUIRED_ARTIFACT_MISSING"]}
    if {path.name for path in directory.iterdir()} != set(required):
        problems.append("UNEXPECTED_ARTIFACT_PRESENT")
    if any(SENSITIVE_TEXT.search((directory / name).read_text(encoding="utf-8"))
           for name in required):
        problems.append("SENSITIVE_PATTERN_IN_ARTIFACT")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        summary = json.loads((directory / "planning.json").read_text(encoding="utf-8"))
        tool_request = json.loads((directory / "tool_request.json").read_text(encoding="utf-8"))
        tool_result = json.loads((directory / "tool_result.json").read_text(encoding="utf-8"))
        graph_data = json.loads((directory / "task_world_graph.json").read_text(encoding="utf-8"))
        graph = TaskWorldGraph.from_dict(graph_data)
        validate_plan_graph(graph_data, request)
        if request_hash(request) != summary.get("request_hash"):
            problems.append("REQUEST_HASH_MISMATCH")
        if tool_request.get("scene_version") != initial_intent_version(request):
            problems.append("TOOL_BASE_VERSION_MISMATCH")
        if (tool_request.get("request_id") != tool_result.get("request_id")
                or tool_request.get("request_id") != summary.get("request_id")
                or tool_request.get("tool_version") != tool_result.get("tool_version")):
            problems.append("TOOL_ENVELOPE_MISMATCH")
        if (tool_result.get("status") != "passed" or summary.get("status") != "planned"
                or tool_result.get("base_scene_version") != tool_request.get("scene_version")):
            problems.append("TOOL_STATUS_MISMATCH")
        if (tool_result.get("proposed_delta", {}).get("task_world_graph") != graph_data
                or tool_result.get("metrics", {}).get("graph_hash") != graph.content_hash
                or summary.get("graph_hash") != graph.content_hash):
            problems.append("GRAPH_HASH_MISMATCH")
        expected_requests = 1 if summary.get("source") == "live" else 0
        if (tool_result.get("metrics", {}).get("model_requests_used") != expected_requests
                or summary.get("budget", {}).get("model_requests_used") != expected_requests
                or tool_result.get("metrics", {}).get("candidates_seen") != 1):
            problems.append("BUDGET_MISMATCH")
        if (graph_data["built"] or graph_data["evidence_refs"]
                or summary.get("robot_validation") != "not_run"
                or summary.get("built_geometry") != "not_run"
                or not graph_data["requirements"]
                or any(item["status"] != "not_run" for item in graph_data["requirements"])):
            problems.append("UNVERIFIED_CLAIM")
        if (sum(node["kind"] == "room" for node in graph_data["nodes"]) != 2
                or sum(node["kind"] == "portal" for node in graph_data["nodes"]) != 1
                or [step["verb"] for step in graph_data["steps"]] != ["PICK", "NAVIGATE", "PLACE"]):
            problems.append("TASK_SHAPE_MISMATCH")
    except (OSError, ValueError, TypeError, KeyError, GraphError):
        problems.append("ARTIFACT_INVALID")
        graph = None
    return {"verified": not problems, "problems": problems,
            "graph_hash": None if graph is None else graph.content_hash,
            "rooms": 2 if graph is not None and not problems else None,
            "model_requests_used": summary.get("budget", {}).get("model_requests_used")
            if graph is not None else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.directory, args.request)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
