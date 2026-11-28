#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh G2+G3 real-library acceptance with independent static replay and named negatives."""
from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.agents.room_asset_visualization import render_room_assets_svg
from rfw_w.scenes.assets.room_tools import RoomAssetError, verify_scene_assembly
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.orchestrator.house_tools import run_house_tools
from rfw_w.scenes.orchestrator.room_asset_tools import run_room_asset_tools
from rfw_w.scenes.solvers.house_tools import measure_house_access


def fixture_graph() -> TaskWorldGraph:
    request = json.loads((data_path('fixtures/task_world/cross_room_request.json')).read_text())
    response = json.loads((data_path('fixtures/task_world/cross_room_slots_response.json')).read_text())
    return compile_pick_place_slots(response["slots"], request)


def _rehash(ledger: dict) -> dict:
    ledger["content_hash"] = content_hash({k:v for k,v in ledger.items() if k != "content_hash"})
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("ASSET_OUTPUT_EXISTS(output directory already exists)")
    source = fixture_graph()
    house, _ = run_house_tools(source, args.output_dir / "house")
    final, report = run_room_asset_tools(house, args.output_dir / "house", args.output_dir / "assets")
    saved_house = TaskWorldGraph.from_dict(json.loads((args.output_dir / "house/graph.json").read_text()))
    saved_final = TaskWorldGraph.from_dict(json.loads((args.output_dir / "assets/graph.json").read_text()))
    plan = json.loads((args.output_dir / "assets/geometry/house_plan.json").read_text())
    ledger = json.loads((args.output_dir / "assets/asset_ledger.json").read_text())
    independent = verify_scene_assembly(saved_house, plan, ledger, args.output_dir / "assets")
    shell = measure_house_access(plan, args.output_dir / "assets/geometry")
    if (saved_final.content_hash != final.content_hash
            or shell["mesh_sha256"] != saved_final.to_dict()["built"]["house_shell"]["mesh_sha256"]
            or independent["instances"] != report["instances"]):
        raise SystemExit("ASSET_LEDGER_MISMATCH(independent replay)")
    bindings = [e for e in saved_final.to_dict()["edges"] if e["kind"] == "role_bound_to"]
    if len(bindings) != independent["instances"]:
        raise SystemExit("ASSET_BINDING_INVALID(graph edges)")
    negatives = {}
    cases = {}
    stale = deepcopy(ledger); stale["source_graph_hash"] = "0"*64
    cases["stale_version"] = stale
    missing = deepcopy(ledger); missing["instances"] = [i for i in missing["instances"] if i["role_id"] != "cup"]
    cases["missing_core"] = _rehash(missing)
    blocked = deepcopy(ledger)
    next(i for i in blocked["instances"] if i["role_id"] == "counter")["pose_xyz_m"][1] = 2.0
    cases["door_blocked"] = _rehash(blocked)
    no_support = deepcopy(ledger)
    next(i for i in no_support["instances"] if i["role_id"] == "counter")["support_surface"] = None
    cases["missing_support"] = _rehash(no_support)
    bad_goal = deepcopy(ledger); bad_goal["goal_placement"]["candidate_pose_xyz_m"][0] = 0.0
    cases["bad_goal"] = _rehash(bad_goal)
    for name, candidate in cases.items():
        try:
            verify_scene_assembly(saved_house, plan, candidate, args.output_dir / "assets")
        except RoomAssetError as error:
            negatives[name] = error.code
    expected = {"stale_version": "GRAPH_VERSION_CONFLICT", "missing_core": "ASSET_UNAVAILABLE",
                "door_blocked": "CLEARANCE_VIOLATION", "missing_support": "SUPPORT_NOT_FOUND",
                "bad_goal": "SUPPORT_NOT_FOUND"}
    if negatives != expected:
        raise SystemExit("ASSET_NEGATIVE_FAILED(" + repr(negatives) + ")")
    # Optional omission remains a valid core task and is explicitly accounted for.
    optional = deepcopy(ledger)
    dropped = [i for i in optional["instances"] if i["priority"] == "optional"]
    optional["instances"] = [i for i in optional["instances"] if i["priority"] != "optional"]
    optional["unplaced_optional"] = [{"role_id": i["role_id"], "code": "OPTIONAL_DROPPED"} for i in dropped]
    optional = _rehash(optional)
    optional_check = verify_scene_assembly(saved_house, plan, optional, args.output_dir / "assets")
    if optional_check["core_roles_bound"] != 3 or optional_check["optional_unplaced"] != 2:
        raise SystemExit("ASSET_OPTIONAL_DROP_FAILED")
    if (args.output_dir / "assets/scene.svg").read_text() != render_room_assets_svg(plan, ledger):
        raise SystemExit("ASSET_VISUALIZATION_MISMATCH")
    acceptance = {**report, "independent_replay": True, "negative_results": negatives,
                  "optional_drop_replay": True, "mesh_remeasured_clearance_m": shell["measured_clearance_m"]}
    (args.output_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: acceptance[k] for k in ("status", "instances", "core_roles_bound",
                                                  "optional_placed", "independent_replay",
                                                  "optional_drop_replay", "negative_results")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
