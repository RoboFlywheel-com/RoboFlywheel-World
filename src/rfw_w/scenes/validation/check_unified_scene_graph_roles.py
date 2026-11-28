#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure task identity, explicit composition migration and named rejection cases."""
from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import argparse
import json
from pathlib import Path
import sys


from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.agents.task_world_visualization import render_task_world_svg
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph


def fixture_graph() -> TaskWorldGraph:
    root = data_path('fixtures/task_world')
    request = json.loads((root / "cross_room_request.json").read_text())
    slots = json.loads((root / "cross_room_slots_response.json").read_text())["slots"]
    return compile_pick_place_slots(slots, request)


def task_identity(graph: TaskWorldGraph) -> dict:
    raw = graph.to_dict()
    return {**{k: raw[k] for k in ("initial_state", "steps", "goals", "requirements")},
            "core_ids": sorted(n["id"] for n in raw["nodes"]
                               if n.get("priority") == "core")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original = fixture_graph()
    graph = original
    report = {"source_graph_hash": original.content_hash,
              "task_identity": task_identity(original), "built_geometry": "not_run",
              "robot_validation": "not_run"}
    if args.baseline_only:
        report.update(status="passed", migration="not_run", reason="baseline before v2 implementation")
    else:
        from rfw_w.scenes.contracts.scene_composition import migrate_composition, background_role
        graph = migrate_composition(original, expected_hash=original.content_hash)
        raw = graph.to_dict()
        for room in ("kitchen", "study"):
            role, edge = background_role(raw, room, room + "_cabinet", "cabinet", "floor")
            raw["nodes"].append(role)
            raw["edges"].append(edge)
        graph = graph.revise(raw, expected_hash=graph.content_hash)
        negatives = {}
        mutations = {
            "bad_purpose": ("cup", {"purpose": "background"}),
            "background_task": ("cup", {"purpose": "background", "priority": "optional"}),
            "bad_anchor": ("study_cabinet", {"placement_kind": "wall"}),
        }
        for name, (node_id, fields) in mutations.items():
            candidate = graph.to_dict(); candidate.pop("content_hash")
            next(n for n in candidate["nodes"] if n["id"] == node_id).update(fields)
            try:
                TaskWorldGraph.from_dict(candidate)
            except GraphError as error:
                negatives[name] = error.code
        candidate = graph.to_dict(); candidate.pop("content_hash")
        candidate["edges"].append({"id": "conflicting_parent", "kind": "contains",
                                   "source_id": "study", "target_id": "study_cabinet"})
        try:
            TaskWorldGraph.from_dict(candidate)
        except GraphError as error:
            negatives["bad_parent"] = error.code
        try:
            graph.check_evidence({"graph_hash": original.content_hash})
        except GraphError as error:
            negatives["stale_evidence"] = error.code
        expected = {"bad_purpose": "COMPOSITION_PURPOSE_INVALID",
                    "background_task": "COMPOSITION_BACKGROUND_IN_TASK",
                    "bad_anchor": "COMPOSITION_ANCHOR_INVALID",
                    "bad_parent": "COMPOSITION_PARENT_CONFLICT",
                    "stale_evidence": "GRAPH_STALE_EVIDENCE"}
        stable = migrate_composition(original, expected_hash=original.content_hash).content_hash
        migrated = migrate_composition(original, expected_hash=original.content_hash)
        passed = (task_identity(graph) == task_identity(original) and negatives == expected
                  and stable == migrated.content_hash
                  and TaskWorldGraph.from_dict(graph.to_dict()).content_hash == graph.content_hash)
        report.update(status="passed" if passed else "failed", migration="passed",
                      graph_hash=graph.content_hash, negatives=negatives,
                      task_preserved=task_identity(graph) == task_identity(original),
                      stable_migration=stable == migrated.content_hash)
        (args.output_dir / "migrated_graph.json").write_text(json.dumps(migrated.to_dict(), indent=2)+"\n")
    for name, payload in (("source_graph.json", original.to_dict()),
                          ("graph.json", graph.to_dict()), ("report.json", report)):
        (args.output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    (args.output_dir / "scene_graph.svg").write_text(render_task_world_svg(graph))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
