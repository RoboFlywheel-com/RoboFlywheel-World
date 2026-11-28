#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read only a saved G2/G3 bundle in a separate process and check its evidence chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


from rfw_w.scenes.assets.room_tools import verify_scene_assembly
from rfw_w.scenes.agents.room_asset_visualization import render_room_assets_svg
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.orchestrator.scene_generation_agent import TOOL_ORDER
from rfw_w.scenes.solvers.house_tools import measure_house_access


def verify(directory: Path) -> dict:
    workflow = json.loads((directory / "workflow.json").read_text())
    house_report = json.loads((directory / "house/report.json").read_text())
    asset_report = json.loads((directory / "assets/report.json").read_text())
    house = TaskWorldGraph.from_dict(json.loads((directory / "house/graph.json").read_text()))
    final = TaskWorldGraph.from_dict(json.loads((directory / "assets/graph.json").read_text()))
    plan = json.loads((directory / "assets/geometry/house_plan.json").read_text())
    ledger = json.loads((directory / "assets/asset_ledger.json").read_text())
    measured = measure_house_access(plan, directory / "assets/geometry")
    assets = verify_scene_assembly(house, plan, ledger, directory / "assets")
    names = [event["request"]["tool_version"].split("/")[0]
             for event in house_report["events"] + asset_report["events"]]
    problems = []
    if (workflow["status"] != "passed" or workflow["tool_order"] != names
            or tuple(names) != TOOL_ORDER or workflow["tool_calls"] != len(names)):
        problems.append("TOOL_ORDER_INVALID")
    if (workflow["source_graph_hash"] != house_report["source_graph_hash"]
            or workflow["house_graph_hash"] != house.content_hash
            or house_report["graph_hash"] != house.content_hash
            or workflow["final_graph_hash"] != final.content_hash
            or asset_report["source_graph_hash"] != house.content_hash
            or asset_report["graph_hash"] != final.content_hash
            or final.to_dict()["parent_hash"] != house.content_hash):
        problems.append("GRAPH_VERSION_CONFLICT")
    if (measured["mesh_sha256"] != final.to_dict()["built"]["house_shell"]["mesh_sha256"]
            or measured["measured_clearance_m"] != house_report["measured_clearance_m"]):
        problems.append("HOUSE_SHELL_MISMATCH")
    built = final.to_dict()["built"]["asset_assembly"]
    bindings = [edge for edge in final.to_dict()["edges"] if edge["kind"] == "role_bound_to"]
    expected_bindings = {(item["role_id"], item["instance_id"]) for item in ledger["instances"]}
    actual_bindings = {(edge["source_id"], edge["target_id"]) for edge in bindings}
    instance_nodes = {node["id"]: node for node in final.to_dict()["nodes"]
                      if node["kind"] == "asset_instance"}
    if (built["ledger_hash"] != ledger["content_hash"]
            or actual_bindings != expected_bindings
            or set(built["instance_ids"]) != {item["instance_id"] for item in ledger["instances"]}
            or any(instance_nodes.get(item["instance_id"], {}).get("asset_ref")
                   != "hssd:" + item["asset_id"] for item in ledger["instances"])
            or built["core_roles_bound"] != assets["core_roles_bound"]):
        problems.append("ASSET_BINDING_INVALID")
    if (workflow["model_requests_used"] != 0 or workflow["robot_navigation"] != "not_run"
            or workflow["robot_ik"] != "not_run" or workflow["robot_rollout"] != "not_run"):
        problems.append("UNVERIFIED_CLAIM")
    scene_svg = (directory / "assets/scene.svg").read_text()
    if scene_svg != render_room_assets_svg(plan, ledger):
        problems.append("ASSET_VISUALIZATION_MISMATCH")
    ET.fromstring(scene_svg)
    return {"verified": not problems, "problems": problems,
            "rooms": measured["room_count"], "portals": measured["portal_count"],
            "measured_clearance_m": measured["measured_clearance_m"],
            "instances": assets["instances"], "core_roles_bound": assets["core_roles_bound"],
            "optional_placed": assets["optional_placed"], "tool_calls": len(names),
            "robot_rollout": "not_run"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.directory)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
