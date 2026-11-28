# SPDX-License-Identifier: Apache-2.0
"""Task graph acceptance cases: topology, state transitions, provenance and model boundary."""
import copy
import json
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph, GraphError
from rfw_w.scenes.agents.task_world_planner import (
    parse_plan_response, build_plan_prompt, plan_task_world_graph, request_hash,
)
from rfw_w.scenes.agents.llm_client import LLMConfig
from rfw_w.scenes.agents.task_world_visualization import render_task_world_svg
from rfw_w.scenes.adapters.task_world_single_room import link_single_room, check_single_room_link
from rfw_w.scenes.contracts.room_plan import RoomPlan
from rfw_w.scenes.contracts.state import RoomShell, SceneIR

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/scenes/fixtures/task_world"


def raw_graph():
    return json.loads((FIXTURES / "two_room_pick_place.json").read_text())


def request():
    return json.loads((FIXTURES / "cross_room_request.json").read_text())


def single_room_sources():
    payload = json.loads((ROOT / "tests/scenes/fixtures/office_room_plan.json").read_text())
    payload.pop("hash")
    payload["rooms"][0]["kind"] = "kitchen"
    plan = RoomPlan.from_dict(payload)
    room = plan.rooms[0]
    scene = SceneIR(scene_id="legacy_kitchen", room=RoomShell(polygon_xy=room.polygon_xy),
                    instances=())
    return plan, scene


def test_single_room_link_is_explicit_reference_and_rejects_stale_sources():
    graph = TaskWorldGraph.from_dict(raw_graph())
    plan, scene = single_room_sources()
    link = link_single_room(graph, "kitchen", plan, scene,
                            room_to_world_xy_m=(1.0, 2.0), room_to_world_yaw_rad=0.0)
    assert link.to_dict()["status"] == "reference_only"
    assert link.to_dict()["room_to_world"] == {"translation_xy_m": [1.0, 2.0], "yaw_rad": 0.0}
    assert link.plan_room_id == plan.rooms[0].room_id
    check_single_room_link(link, graph, plan, scene)
    assert graph.to_dict()["built"] == {}
    changed = raw_graph()
    changed["nodes"][3]["requested_clearance_m"] = 0.85
    with pytest.raises(GraphError, match="GRAPH_VERSION_CONFLICT"):
        check_single_room_link(link, TaskWorldGraph.from_dict(changed), plan, scene)
    with pytest.raises(GraphError, match="SINGLE_ROOM_SOURCE_STALE"):
        check_single_room_link(link, graph, plan, SceneIR(scene_id="changed", room=scene.room,
                                                           instances=()))


def test_single_room_link_rejects_missing_id_wrong_kind_or_implicit_frame():
    graph = TaskWorldGraph.from_dict(raw_graph())
    plan, scene = single_room_sources()
    for room_id, xy, code in [("missing", (0.0, 0.0), "GRAPH_DANGLING_REF"),
                               ("kitchen", (float("nan"), 0.0), "GRAPH_FRAME_MISMATCH")]:
        with pytest.raises(GraphError, match=code):
            link_single_room(graph, room_id, plan, scene,
                             room_to_world_xy_m=xy, room_to_world_yaw_rad=0.0)
    wrong = plan.to_dict()
    wrong.pop("hash")
    wrong["rooms"][0]["kind"] = "office"
    with pytest.raises(GraphError, match="SINGLE_ROOM_KIND_MISMATCH"):
        link_single_room(graph, "kitchen", RoomPlan.from_dict(wrong), scene,
                         room_to_world_xy_m=(0.0, 0.0), room_to_world_yaw_rad=0.0)


def test_cross_room_task_graph_has_one_truth_for_space_and_time():
    graph = TaskWorldGraph.from_dict(raw_graph())
    assert graph.to_dict()["content_hash"] == graph.content_hash
    assert TaskWorldGraph.from_dict(graph.to_dict()).content_hash == graph.content_hash
    assert graph.state_after("pick_cup") >= {("cup", "HELD_BY", "robot")}
    assert graph.state_after("cross_door") >= {("robot", "IN_ROOM", "study")}
    assert graph.state_after("place_cup") >= {("cup", "ON", "desk")}
    assert "cup" in graph.room_view("kitchen")["node_ids"]
    assert [s["id"] for s in graph.task_view()["steps"]] == ["pick_cup", "cross_door", "place_cup"]
    assert all(r["status"] == "not_run" for r in graph.to_dict()["requirements"])


@pytest.mark.parametrize("change,code", [
    (lambda d: d["nodes"][3].update(room_ids=["kitchen", "missing"]), "PORTAL_ENDPOINT_INVALID"),
    (lambda d: d["steps"][1].update(route_portal_ids=[]), "TASK_ROUTE_DISCONNECTED"),
    (lambda d: d["steps"][2]["preconditions"][0].update(object="missing"), "GRAPH_DANGLING_REF"),
    (lambda d: d["initial_state"][0].update(object="desk"), "TASK_GOAL_ALREADY_TRUE"),
    (lambda d: d["steps"][1]["preconditions"][0].update(predicate="ON"), "TASK_PRECONDITION_UNSATISFIED"),
    (lambda d: d["nodes"][6].update(mobility="fixed_arm"), "ROBOT_MOBILITY_INCOMPATIBLE"),
    (lambda d: d["steps"][0]["effects_add"][0].update(predicate="ON", object="desk"), "TASK_EFFECT_INVALID"),
    (lambda d: d["nodes"][7].update(robot_verified=True), "GRAPH_SHAPE_INVALID"),
])
def test_real_task_defects_fail_by_name(change, code):
    raw = raw_graph()
    change(raw)
    with pytest.raises(GraphError, match=code):
        TaskWorldGraph.from_dict(raw)


def test_graph_changes_invalidate_hash_and_old_evidence():
    graph = TaskWorldGraph.from_dict(raw_graph())
    raw = raw_graph()
    raw["nodes"][3]["requested_clearance_m"] = 0.9
    changed = TaskWorldGraph.from_dict(raw)
    assert changed.content_hash != graph.content_hash
    with pytest.raises(GraphError, match="GRAPH_STALE_EVIDENCE"):
        changed.check_evidence({"graph_hash": graph.content_hash, "subject_ids": ["door"]})
    revision = graph.revise(raw, expected_hash=graph.content_hash)
    assert revision.to_dict()["revision"] == 1
    assert revision.to_dict()["parent_hash"] == graph.content_hash
    with pytest.raises(GraphError, match="GRAPH_VERSION_CONFLICT"):
        graph.revise(raw, expected_hash="0" * 64)


def test_prompt_boundary_is_semantic_only():
    system, user = build_plan_prompt(request())
    assert request()["prompt"] in user
    assert request()["robot"]["mobility"] in user
    assert "/home/" not in system + user
    assert "outputs/" not in system + user
    assert "TASK_TARGET_UNRESOLVED" in system
    assert "gaps=[]" in system
    assert 'status="not_run"' in system
    assert "deterministic tool builds these" in system


@pytest.mark.parametrize("injection", [
    "API_KEY=" + "synthetic-sensitive-marker", "Bearer " + "synthetic-sensitive-marker",
    "/home/" + "example/private/task.json", ".env.local",
])
def test_sensitive_prompt_is_rejected_before_model_call(injection, monkeypatch):
    req = request()
    req["prompt"] += " " + injection
    monkeypatch.setattr("rfw_w.scenes.agents.task_world_planner.call_vision",
                        lambda *a, **k: pytest.fail("external call"))
    with pytest.raises(GraphError, match="PLAN_REQUEST_SENSITIVE_CONTENT"):
        plan_task_world_graph(req, config=LLMConfig("https://example.invalid", "mock", "UNUSED"))


def test_recorded_model_response_becomes_valid_graph():
    req = request()
    graph = TaskWorldGraph.from_dict(raw_graph())
    response = {"request_hash": request_hash(req),
                "graph": raw_graph(), "assumptions": [], "gaps": []}
    result = parse_plan_response(json.dumps(response), req)
    assert result.content_hash == graph.content_hash


def test_model_cannot_erase_route_or_claim_robot_passed():
    req = request()
    base = {"request_hash": request_hash(req),
            "graph": raw_graph(), "assumptions": [], "gaps": []}
    base["graph"]["steps"][1]["route_portal_ids"] = []
    with pytest.raises(GraphError, match="TASK_ROUTE_DISCONNECTED"):
        parse_plan_response(json.dumps(base), req)
    base["graph"] = raw_graph()
    base["graph"]["requirements"][0]["status"] = "passed"
    with pytest.raises(GraphError, match="GRAPH_UNVERIFIED_CLAIM"):
        parse_plan_response(json.dumps(base), req)
    base["graph"] = raw_graph()
    base["graph"]["requirements"] = []
    with pytest.raises(GraphError, match="PLAN_OBLIGATION_MISSING"):
        parse_plan_response(json.dumps(base), req)


def test_external_call_requires_opt_in(monkeypatch):
    monkeypatch.delenv("PVSCENE_ALLOW_EXTERNAL_VLM", raising=False)
    monkeypatch.setattr("rfw_w.scenes.agents.task_world_planner.call_vision",
                        lambda *a, **k: pytest.fail("external call"))
    with pytest.raises(SystemExit, match="PVSCENE_ALLOW_EXTERNAL_VLM"):
        plan_task_world_graph(request(), config=LLMConfig("https://example.invalid", "mock", "UNUSED"))


def test_visualization_shows_rooms_portal_and_action_order():
    svg = render_task_world_svg(TaskWorldGraph.from_dict(raw_graph()))
    assert "<svg" in svg and "厨房" in svg and "书房" in svg and "内门" in svg
    assert svg.index("pick_cup") < svg.index("cross_door") < svg.index("place_cup")
    assert "not_run" in svg
