# SPDX-License-Identifier: Apache-2.0
"""Room composition must preserve task identity and reject invalid generation trees."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.agents.task_world_visualization import render_task_world_svg
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.orchestrator.plan_task_world import run_plan_task_world

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/task_world"


def graph() -> TaskWorldGraph:
    slots = json.loads((FIXTURES / "cross_room_slots_response.json").read_text())["slots"]
    request = json.loads((FIXTURES / "cross_room_request.json").read_text())
    return compile_pick_place_slots(slots, request)


def edge(parent: str, child: str) -> dict[str, str]:
    return {"id": "negative_" + parent + "_" + child, "kind": "contains",
            "source_id": parent, "target_id": child}


def test_two_rooms_have_structures_core_roles_and_optional_distractors():
    first = graph()
    second = graph()
    assert first.content_hash == second.content_hash
    assert TaskWorldGraph.from_dict(first.to_dict()).content_hash == first.content_hash
    data = first.to_dict()
    assert data["composition_schema"] == "pvscene.room-composition/v1"
    assert data["built"] == {} and all(r["status"] == "not_run" for r in data["requirements"])
    assert len([n for n in data["nodes"] if n["kind"] == "portal"]) == 1
    for room_id in ("kitchen", "study"):
        members = [n for n in data["nodes"] if n.get("room_id") == room_id]
        assert len([n for n in members if n["kind"] == "floor"]) == 1
        assert len([n for n in members if n["kind"] == "wall_set"]) == 1
        assert len([n for n in members if n["kind"] == "surface"]) == 1
        assert len([n for n in members if n["kind"] == "distractor_policy"]) == 1
        assert len([n for n in members if n.get("priority") == "optional"]) == 1
        assert set(first.room_view(room_id)["node_ids"]) >= {n["id"] for n in members}
    assert first.state_after("step_place") >= {("cup", "ON", "desk")}
    assert len([n for n in data["nodes"] if n["id"] == "cup"]) == 1
    svg = render_task_world_svg(first)
    for text in ("rc_kitchen_floor", "rc_kitchen_walls", "rc_study_floor",
                 "rc_study_walls", "rc_kitchen_source_optional_1",
                 "rc_study_destination_optional_1", "cup", "counter", "desk", "door"):
        assert text in svg
    assert "segments not_run" in svg and "optional" in svg


def test_composition_change_invalidates_old_graph_evidence():
    original = graph()
    data = original.to_dict()
    policy = next(n for n in data["nodes"] if n["kind"] == "distractor_policy")
    policy["max_count"] = 0
    data.pop("content_hash")
    revised = TaskWorldGraph.from_dict(data)
    assert revised.content_hash != original.content_hash
    with pytest.raises(GraphError, match="GRAPH_STALE_EVIDENCE"):
        revised.check_evidence({"graph_hash": original.content_hash, "subject_ids": ["cup"]})


def test_optional_roles_can_be_dropped_without_changing_the_task():
    original = graph()
    data = original.to_dict()
    optional_ids = {n["id"] for n in data["nodes"] if n.get("priority") == "optional"}
    data["nodes"] = [n for n in data["nodes"] if n["id"] not in optional_ids]
    data["edges"] = [e for e in data["edges"] if e["target_id"] not in optional_ids]
    data.pop("content_hash")
    without_distractors = TaskWorldGraph.from_dict(data)
    assert without_distractors.state_after("step_place") == original.state_after("step_place")
    assert without_distractors.to_dict()["goals"] == original.to_dict()["goals"]
    assert without_distractors.content_hash != original.content_hash


def test_asset_queries_keep_optional_priority_for_downstream_selection():
    request = json.loads((FIXTURES / "cross_room_request.json").read_text())
    response = (FIXTURES / "cross_room_slots_response.json").read_text()
    planned = run_plan_task_world(request, replay_response=response)
    assert planned.graph is not None
    queries = {item["role_id"]: item["priority"]
               for item in planned.to_dict()["asset_queries"]}
    assert queries == {"cup": "core", "counter": "core", "desk": "core",
                       "rc_kitchen_source_optional_1": "optional",
                       "rc_study_destination_optional_1": "optional"}


def _missing_parent(data):
    data["edges"] = [e for e in data["edges"] if not (
        e["kind"] == "contains" and e["target_id"] == "cup")]


def _second_parent(data):
    data["edges"].append(edge("kitchen", "cup"))


def _cycle(data):
    data["edges"] = [e for e in data["edges"] if not (
        e["kind"] == "contains" and e["target_id"] == "counter")]
    data["edges"].append(edge("rc_kitchen_source_surface", "counter"))


def _wrong_room(data):
    data["edges"] = [e for e in data["edges"] if not (
        e["kind"] == "contains" and e["target_id"] == "rc_kitchen_floor")]
    data["edges"].append(edge("study", "rc_kitchen_floor"))


def _missing_structure(data):
    data["nodes"] = [n for n in data["nodes"] if n["id"] != "rc_kitchen_walls"]
    data["edges"] = [e for e in data["edges"] if e["target_id"] != "rc_kitchen_walls"]


def _core_made_optional(data):
    next(n for n in data["nodes"] if n["id"] == "cup")["priority"] = "optional"


def _optional_used_by_task(data):
    data["requirements"].append({"id": "bad_optional_required", "kind": "asset_grasp",
                                 "subject_ids": ["rc_kitchen_source_optional_1"],
                                 "status": "not_run"})


def _optional_used_as_support(data):
    data["edges"].append({"id": "negative_optional_support", "kind": "supports",
                          "source_id": "rc_kitchen_source_optional_1", "target_id": "cup"})


def _wrong_initial_support(data):
    data["edges"] = [e for e in data["edges"] if not (
        e["kind"] == "contains" and e["target_id"] == "cup")]
    data["edges"].append(edge("rc_kitchen_floor", "cup"))


@pytest.mark.parametrize("change,code", [
    (_missing_parent, "COMPOSITION_PARENT_MISSING"),
    (_second_parent, "COMPOSITION_PARENT_CONFLICT"),
    (_cycle, "COMPOSITION_CYCLE"),
    (_wrong_room, "COMPOSITION_ROOM_MISMATCH"),
    (_missing_structure, "COMPOSITION_STRUCTURE_MISSING"),
    (_core_made_optional, "COMPOSITION_CORE_OPTIONAL"),
    (_optional_used_by_task, "COMPOSITION_OPTIONAL_IN_TASK"),
    (_optional_used_as_support, "COMPOSITION_OPTIONAL_IN_TASK"),
    (_wrong_initial_support, "COMPOSITION_SUPPORT_MISMATCH"),
])
def test_invalid_room_composition_is_named(change, code):
    data = copy.deepcopy(graph().to_dict())
    data.pop("content_hash")
    change(data)
    with pytest.raises(GraphError) as error:
        TaskWorldGraph.from_dict(data)
    assert error.value.code == code
