#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay the first prompt-to-graph tool, including recorded invalid model outputs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path


from rfw_w.scenes.agents.task_world_visualization import render_task_world_svg
from rfw_w.scenes.agents.task_world_planner import request_hash
from rfw_w.scenes.orchestrator.plan_task_world import run_plan_task_world


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("replay",), default="replay")
    args = parser.parse_args(argv)
    req = json.loads(args.fixture.read_text(encoding="utf-8"))
    response_path = args.fixture.parent / "cross_room_response.json"
    response = response_path.read_text(encoding="utf-8")
    slots_response = (args.fixture.parent / "cross_room_slots_response.json").read_text(
        encoding="utf-8")
    first = run_plan_task_world(req, replay_response=response)
    second = run_plan_task_world(req, replay_response=response)
    if first.graph is None or second.graph is None:
        raise RuntimeError("positive fixture was not committed")
    slots_first = run_plan_task_world(req, replay_response=slots_response)
    slots_second = run_plan_task_world(req, replay_response=slots_response)
    if slots_first.graph is None or slots_second.graph is None:
        raise RuntimeError("semantic slots fixture was not committed")
    graph1, graph2 = first.graph, second.graph
    svg = render_task_world_svg(graph1)
    negatives = {}
    for path in sorted(args.negative_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        negative = run_plan_task_world(req, replay_response=json.dumps(case["response"], ensure_ascii=False))
        got = negative.reason_code or "accepted"
        negatives[path.stem] = {"expected": case["expected_error"], "actual": got,
                                "passed": got == case["expected_error"] and negative.graph is None
                                and negative.tool_result.proposed_delta is None}
    # These three task-specific defects must be tested at the tool boundary too. Keep the
    # fixture file single-source so a schema edit cannot update one duplicate but not another.
    base = json.loads(response)
    extra = {}
    fixed_req = deepcopy(req)
    fixed_req["robot"]["mobility"] = "fixed_arm"
    fixed = deepcopy(base)
    fixed["request_hash"] = request_hash(fixed_req)
    next(n for n in fixed["graph"]["nodes"] if n["kind"] == "robot")["mobility"] = "fixed_arm"
    extra["fixed_arm_cross_room"] = (fixed_req, fixed, "ROBOT_MOBILITY_INCOMPATIBLE")
    already = deepcopy(base)
    already["graph"]["initial_state"].append(deepcopy(already["graph"]["goals"][0]))
    extra["goal_already_true"] = (req, already, "TASK_GOAL_ALREADY_TRUE")
    dangling = deepcopy(base)
    dangling["graph"]["steps"][-1]["preconditions"][0]["object"] = "missing_role"
    extra["dangling_model_target"] = (req, dangling, "GRAPH_DANGLING_REF")
    no_reason = deepcopy(base)
    next(n for n in no_reason["graph"]["nodes"] if n["source"] == "inferred").pop("source_reason")
    extra["inferred_reason_missing"] = (req, no_reason, "PLAN_SOURCE_REASON_MISSING")
    no_step_source = deepcopy(base)
    no_step_source["graph"]["steps"][0].pop("source")
    extra["step_source_missing"] = (req, no_step_source, "PLAN_STEP_SOURCE_MISSING")
    ambiguous = deepcopy(base)
    ambiguous["ambiguities"] = ["杯子的具体位置尚未明确"]
    ambiguous["gaps"] = ["PLAN_AMBIGUOUS"]
    extra["ambiguous_task"] = (req, ambiguous, "PLAN_MODEL_REPORTED_GAPS")
    sensitive = deepcopy(base)
    sensitive["graph"]["nodes"][0]["label"] = "API_KEY=synthetic-sensitive-marker"
    extra["model_sensitive_text"] = (req, sensitive, "PLAN_RESPONSE_SENSITIVE_CONTENT")
    for name, (case_request, case_response, expected) in extra.items():
        negative = run_plan_task_world(case_request,
                                       replay_response=json.dumps(case_response, ensure_ascii=False))
        got = negative.reason_code or "accepted"
        negatives[name] = {"expected": expected, "actual": got,
                           "passed": got == expected and negative.graph is None
                           and negative.tool_result.proposed_delta is None}
    slot_base = json.loads(slots_response)
    slot_cases = {}
    missing_slot = deepcopy(slot_base)
    missing_slot["slots"].pop("portal")
    slot_cases["slot_missing_portal"] = (missing_slot, "PLAN_SLOT_INVALID")
    duplicate_id = deepcopy(slot_base)
    duplicate_id["slots"]["target"]["id"] = "counter"
    slot_cases["slot_duplicate_id"] = (duplicate_id, "PLAN_SLOT_ID_CONFLICT")
    closed_door = deepcopy(slot_base)
    closed_door["slots"]["portal"]["initial_state"] = "closed"
    slot_cases["slot_closed_door"] = (closed_door, "TASK_ACTION_UNSUPPORTED")
    no_slot_reason = deepcopy(slot_base)
    no_slot_reason["slots"]["source_room"]["source"] = "inferred"
    slot_cases["slot_inferred_reason_missing"] = (no_slot_reason, "PLAN_SOURCE_REASON_MISSING")
    unresolved_slots = {"request_hash": slot_base["request_hash"], "assumptions": [],
                        "ambiguities": ["目标支撑面未明确"], "gaps": ["PLAN_AMBIGUOUS"]}
    slot_cases["slot_ambiguity_without_slots"] = (unresolved_slots,
                                                    "PLAN_MODEL_REPORTED_GAPS")
    for name, (case_response, expected) in slot_cases.items():
        negative = run_plan_task_world(req,
                                       replay_response=json.dumps(case_response, ensure_ascii=False))
        got = negative.reason_code or "accepted"
        negatives[name] = {"expected": expected, "actual": got,
                           "passed": got == expected and negative.graph is None
                           and negative.tool_result.proposed_delta is None}
    planned = first.to_dict()
    graph_data = graph1.to_dict()
    provenance_complete = (
        all(n["source"] == "user" or bool(n.get("source_reason"))
            for n in graph_data["nodes"])
        and all(s.get("source") in {"user", "inferred"}
                and (s["source"] == "user" or bool(s.get("source_reason")))
                for s in graph_data["steps"])
    )
    ambiguity_result = run_plan_task_world(
        req, replay_response=json.dumps(ambiguous, ensure_ascii=False))
    report = {"mode": "replay", "graph_hash": graph1.content_hash,
              "slots_graph_hash": slots_first.graph.content_hash,
              "slots_stable_hash": slots_first.graph.content_hash == slots_second.graph.content_hash,
              "slots_state_transition_valid": (
                  ("cup", "HELD_BY", "robot") in slots_first.graph.state_after("step_pick")
                  and ("cup", "ON", "desk") in slots_first.graph.state_after("step_place")),
              "stable_hash": graph1.content_hash == graph2.content_hash,
              "rooms": sum(n["kind"] == "room" for n in graph1.to_dict()["nodes"]),
              "portals": sum(n["kind"] == "portal" for n in graph1.to_dict()["nodes"]),
              "steps": len(graph1.to_dict()["steps"]),
              "svg_has_task_timeline": all(s["id"] in svg for s in graph1.to_dict()["steps"]),
              "robot_validation": planned["robot_validation"],
              "tool_link_valid": (first.tool_request.request_id == first.tool_result.request_id
                                  and first.tool_request.scene_version == first.tool_result.base_scene_version),
              "intent_only": graph1.to_dict()["built"] == {} and not graph1.to_dict()["evidence_refs"],
              "asset_queries": len(planned["asset_queries"]),
              "verification_obligations": len(planned["verification_obligations"]),
              "online_model_validation": "not_run",
              "inferred_reason_validation": "passed" if provenance_complete else "failed",
              "ambiguity_preserved": (ambiguity_result.graph is None
                                      and ambiguity_result.unresolved == ("PLAN_AMBIGUOUS",)
                                      and ambiguity_result.ambiguities == ("杯子的具体位置尚未明确",)),
              "first_budget": planned["budget"], "second_budget": second.to_dict()["budget"],
              "negatives": negatives}
    report["replay_checks_passed"] = (report["stable_hash"] and report["slots_stable_hash"]
                                      and report["slots_state_transition_valid"]
                                      and report["rooms"] == 2
                                      and report["portals"] == 1 and report["steps"] >= 3
                                      and report["svg_has_task_timeline"] and report["tool_link_valid"]
                                      and report["intent_only"] and report["asset_queries"] >= 1
                                      and provenance_complete and report["ambiguity_preserved"]
                                      and report["verification_obligations"] >= 5
                                      and report["first_budget"]["candidates_seen"] == 1
                                      and report["first_budget"]["model_requests_used"] == 0
                                      and bool(negatives)
                                      and all(c["passed"] for c in negatives.values()))
    # Replay checks cannot establish that the live provider returns usable semantic slots.
    report["full_spec_accepted"] = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["replay_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
