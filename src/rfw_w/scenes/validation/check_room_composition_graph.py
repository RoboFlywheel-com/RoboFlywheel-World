#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay a task prompt's semantic slots and export a checked room scene graph."""
from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import argparse
import json
import sys
from pathlib import Path


from rfw_w.scenes.agents.task_world_visualization import render_task_world_svg
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.orchestrator.plan_task_world import run_plan_task_world

FIXTURES = data_path('fixtures/task_world')


def _remove_parent(data: dict, child: str) -> None:
    data["edges"] = [e for e in data["edges"]
                     if not (e["kind"] == "contains" and e["target_id"] == child)]


def _parent(data: dict, parent: str, child: str) -> None:
    data["edges"].append({"id": "negative_" + parent + "_" + child,
                          "kind": "contains", "source_id": parent, "target_id": child})


def _negative_cases(graph: TaskWorldGraph) -> dict[str, dict[str, str | bool]]:
    cases: dict[str, tuple[str, object]] = {
        "missing_parent": ("COMPOSITION_PARENT_MISSING",
                           lambda d: _remove_parent(d, "cup")),
        "duplicate_parent": ("COMPOSITION_PARENT_CONFLICT",
                             lambda d: _parent(d, "kitchen", "cup")),
        "cycle": ("COMPOSITION_CYCLE",
                  lambda d: (_remove_parent(d, "counter"),
                             _parent(d, "rc_kitchen_source_surface", "counter"))),
        "wrong_room": ("COMPOSITION_ROOM_MISMATCH",
                       lambda d: (_remove_parent(d, "rc_kitchen_floor"),
                                  _parent(d, "study", "rc_kitchen_floor"))),
        "missing_wall_set": ("COMPOSITION_STRUCTURE_MISSING",
                             lambda d: (d.__setitem__("nodes", [n for n in d["nodes"]
                                                      if n["id"] != "rc_kitchen_walls"]),
                                        _remove_parent(d, "rc_kitchen_walls"))),
        "core_optional": ("COMPOSITION_CORE_OPTIONAL",
                          lambda d: next(n for n in d["nodes"] if n["id"] == "cup")
                          .__setitem__("priority", "optional")),
        "optional_required": ("COMPOSITION_OPTIONAL_IN_TASK",
                              lambda d: d["requirements"].append({"id": "bad_optional",
                                  "kind": "asset_grasp", "subject_ids":
                                  ["rc_kitchen_source_optional_1"], "status": "not_run"})),
        "optional_as_support": ("COMPOSITION_OPTIONAL_IN_TASK",
                                lambda d: d["edges"].append({"id": "negative_optional_support",
                                    "kind": "supports", "source_id":
                                    "rc_kitchen_source_optional_1", "target_id": "cup"})),
        "wrong_support": ("COMPOSITION_SUPPORT_MISMATCH",
                          lambda d: (_remove_parent(d, "cup"),
                                     _parent(d, "rc_kitchen_floor", "cup"))),
    }
    result = {}
    for name, (expected, change) in cases.items():
        data = graph.to_dict()
        data.pop("content_hash")
        change(data)
        try:
            TaskWorldGraph.from_dict(data)
            actual = "accepted"
        except GraphError as error:
            actual = error.code
        result[name] = {"expected": expected, "actual": actual,
                        "passed": actual == expected}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a fresh directory")
    request = json.loads((FIXTURES / "cross_room_request.json").read_text(encoding="utf-8"))
    response = (FIXTURES / "cross_room_slots_response.json").read_text(encoding="utf-8")
    planned = run_plan_task_world(request, replay_response=response)
    if planned.graph is None:
        print(json.dumps({"passed": False, "reason_code": planned.reason_code}))
        return 1
    graph = planned.graph
    data = graph.to_dict()
    roundtrip = TaskWorldGraph.from_dict(data).content_hash == graph.content_hash
    second = run_plan_task_world(request, replay_response=response)
    stable_hash = second.graph is not None and second.graph.content_hash == graph.content_hash
    rooms = [n["id"] for n in data["nodes"] if n["kind"] == "room"]
    portal_count = sum(n["kind"] == "portal" for n in data["nodes"])
    target_role_count = sum(n["id"] == "cup" and n["kind"] == "asset_role"
                            for n in data["nodes"])
    by_room = {room_id: {
        "floor": sum(n["kind"] == "floor" and n.get("room_id") == room_id
                     for n in data["nodes"]),
        "wall_set": sum(n["kind"] == "wall_set" and n.get("room_id") == room_id
                        for n in data["nodes"]),
        "furniture_core": sum(n.get("room_id") == room_id and n["kind"] == "asset_role"
                              and n.get("physical_class") == "furniture"
                              and n.get("priority") == "core" for n in data["nodes"]),
        "optional_roles": sum(n.get("room_id") == room_id and n.get("priority") == "optional"
                              for n in data["nodes"]),
        "distractor_policies": sum(n.get("room_id") == room_id
                                   and n["kind"] == "distractor_policy"
                                   for n in data["nodes"]),
    } for room_id in rooms}
    negatives = _negative_cases(graph)
    svg = render_task_world_svg(graph)
    visible = all(n["id"] in svg for n in data["nodes"]
                  if n["kind"] not in {"house", "robot"})
    passed = (roundtrip and stable_hash and len(rooms) == 2 and portal_count == 1 and
              all(all(values[key] == 1 for key in ("floor", "wall_set", "furniture_core",
                                                 "optional_roles", "distractor_policies"))
                  for values in by_room.values()) and
              target_role_count == 1 and
              data["built"] == {} and all(r["status"] == "not_run" for r in data["requirements"])
              and visible and all(case["passed"] for case in negatives.values()))
    report = {"passed": passed, "source": "offline_semantic_slot_replay",
              "graph_hash": graph.content_hash, "rooms": len(rooms), "portals": portal_count,
              "per_room": by_room, "target_id": "cup", "target_roles_in_graph": target_role_count,
              "composition_nodes_visible": visible, "roundtrip": roundtrip,
              "stable_hash": stable_hash, "built_geometry": "not_run",
              "robot_validation": "not_run", "negatives": negatives}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "graph.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "scene_graph.svg").write_text(svg, encoding="utf-8")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
