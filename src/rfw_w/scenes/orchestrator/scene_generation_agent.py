# SPDX-License-Identifier: Apache-2.0
"""Bounded G2/G3 scene generation agent over registered, typed tool transactions.

The success path is a dependency-ordered policy. G1 already extracted task semantics;
there is no reason to ask a model to choose the next obvious stage or invent geometry.
Model-guided repair remains a separate G5 capability, not a hidden retry here.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from rfw_w.scenes.assets.room_tools import RoomAssetError
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.orchestrator.house_tools import run_house_tools
from rfw_w.scenes.orchestrator.room_asset_tools import run_room_asset_tools
from rfw_w.scenes.solvers.house_tools import HouseToolError

TOOL_ORDER = (
    "solve_house_layout", "build_house_shell", "measure_house_access", "commit_house_shell",
    "retrieve_role_candidates", "prepare_asset_instances", "place_core_furniture",
    "place_task_objects", "populate_optional_distractors", "verify_scene_assembly",
    "commit_asset_assembly",
)


class SceneGenerationAgent:
    """Runs one graph through G2 then G3, saving a fresh, version-linked evidence bundle."""

    def run(self, graph: TaskWorldGraph, output_dir: Path) -> dict[str, Any]:
        if output_dir.exists():
            raise GraphError("AGENT_OUTPUT_EXISTS", output_dir.name)
        if graph.to_dict().get("composition_schema") != "pvscene.room-composition/v1":
            raise GraphError("PRECONDITION_UNMET", "room composition graph")
        if graph.to_dict()["built"]:
            raise GraphError("PRECONDITION_UNMET", "fresh intent graph required")
        output_dir.mkdir(parents=True)
        started = time.monotonic()
        phase = "G2"
        try:
            house, house_report = run_house_tools(graph, output_dir / "house")
            phase = "G3"
            final, asset_report = run_room_asset_tools(house, output_dir / "house",
                                                       output_dir / "assets")
            names = [event["request"]["tool_version"].split("/")[0]
                     for event in house_report["events"] + asset_report["events"]]
            if tuple(names) != TOOL_ORDER:
                raise GraphError("AGENT_TOOL_ORDER_INVALID", ",".join(names))
            report = {"status": "passed", "source_graph_hash": graph.content_hash,
                      "house_graph_hash": house.content_hash, "final_graph_hash": final.content_hash,
                      "tool_order": names, "tool_calls": len(names),
                      "rooms": house_report["rooms"], "portals": house_report["portals"],
                      "instances": asset_report["instances"],
                      "core_roles_bound": asset_report["core_roles_bound"],
                      "optional_placed": asset_report["optional_placed"],
                      "model_requests_used": 0, "robot_navigation": "not_run",
                      "robot_ik": "not_run", "robot_rollout": "not_run",
                      "elapsed_s": round(time.monotonic()-started, 3)}
            (output_dir / "workflow.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return report
        except (GraphError, HouseToolError, RoomAssetError) as error:
            report = {"status": "failed", "phase": phase,
                      "failure_code": error.code, "source_graph_hash": graph.content_hash,
                      "model_requests_used": 0,
                      "elapsed_s": round(time.monotonic()-started, 3)}
            (output_dir / "workflow.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            raise
