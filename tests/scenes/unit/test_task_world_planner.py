# SPDX-License-Identifier: Apache-2.0
"""G1 tool transaction: proposals are bounded, versioned, and never self-committing."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rfw_w.scenes.agents.llm_client import LLMConfig, LLMError, LLMResponse
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.contracts.tool import FailureCode, ToolStatus
from rfw_w.scenes.orchestrator.plan_task_world import (
    commit_plan_result, make_plan_request, run_plan_task_world,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/task_world"


def request():
    return json.loads((FIXTURES / "cross_room_request.json").read_text(encoding="utf-8"))


def response():
    return (FIXTURES / "cross_room_response.json").read_text(encoding="utf-8")


def slots_response():
    return (FIXTURES / "cross_room_slots_response.json").read_text(encoding="utf-8")


def test_semantic_slots_compile_to_stable_checked_state_graph():
    first = run_plan_task_world(request(), replay_response=slots_response())
    second = run_plan_task_world(request(), replay_response=slots_response())
    assert first.graph is not None and second.graph is not None
    assert first.graph.content_hash == second.graph.content_hash
    data = first.graph.to_dict()
    assert [step["verb"] for step in data["steps"]] == ["PICK", "NAVIGATE", "PLACE"]
    assert first.graph.state_after("step_pick") >= {("cup", "HELD_BY", "robot")}
    assert first.graph.state_after("step_place") >= {("cup", "ON", "desk")}
    assert len(data["requirements"]) == 6
    assert all(item["status"] == "not_run" for item in data["requirements"])
    assert data["built"] == {}


@pytest.mark.parametrize("change,code", [
    (lambda d: d["slots"].pop("portal"), "PLAN_SLOT_INVALID"),
    (lambda d: d["slots"]["target"].update(id="counter"), "PLAN_SLOT_ID_CONFLICT"),
    (lambda d: d["slots"]["portal"].update(initial_state="closed"), "TASK_ACTION_UNSUPPORTED"),
    (lambda d: d["slots"]["source_room"].update(source="inferred"),
     "PLAN_SOURCE_REASON_MISSING"),
])
def test_bad_semantic_slots_never_propose_graph(change, code):
    raw = json.loads(slots_response())
    change(raw)
    result = run_plan_task_world(request(), replay_response=json.dumps(raw, ensure_ascii=False))
    assert result.graph is None
    assert result.reason_code == code
    assert result.tool_result.proposed_delta is None


def test_replay_proposes_then_commits_only_intent():
    planned = run_plan_task_world(request(), replay_response=response())
    assert planned.graph is not None
    assert planned.tool_request.max_candidates == 1
    assert planned.tool_request.request_id == planned.tool_result.request_id
    assert planned.tool_request.scene_version == planned.tool_result.base_scene_version
    assert planned.tool_result.status is ToolStatus.PASSED
    assert planned.tool_result.proposed_delta is not None
    assert planned.to_dict()["asset_queries"] and planned.to_dict()["verification_obligations"]
    assert planned.to_dict()["robot_validation"] == "not_run"
    assert planned.graph.to_dict()["built"] == {}
    assert planned.budget["candidates_seen"] == 1
    assert planned.budget["model_requests_used"] == 0
    assert all(n["source"] == "user" or n.get("source_reason")
               for n in planned.graph.to_dict()["nodes"])
    assert all(s.get("source") in {"user", "inferred"}
               for s in planned.graph.to_dict()["steps"])


def test_inferred_item_without_reason_cannot_be_committed():
    raw = json.loads(response())
    next(n for n in raw["graph"]["nodes"] if n["source"] == "inferred").pop("source_reason")
    planned = run_plan_task_world(request(), replay_response=json.dumps(raw))
    assert planned.graph is None
    assert planned.reason_code == "PLAN_SOURCE_REASON_MISSING"
    assert planned.tool_result.proposed_delta is None


def test_failure_area_reports_structure_without_echoing_model_text():
    raw = json.loads(response())
    raw["graph"]["goals"][0]["object"] = "synthetic-sensitive-marker"
    planned = run_plan_task_world(request(), replay_response=json.dumps(raw))
    assert planned.reason_code == "GRAPH_DANGLING_REF"
    assert planned.tool_result.metrics["failure_area"] == "goals"
    assert "synthetic-sensitive-marker" not in json.dumps(planned.tool_result.to_dict())


def test_ambiguity_is_preserved_as_unresolved_without_committing():
    raw = json.loads(response())
    raw["gaps"] = ["PLAN_AMBIGUOUS"]
    raw["ambiguities"] = ["杯子的具体位置尚未明确"]
    planned = run_plan_task_world(request(), replay_response=json.dumps(raw, ensure_ascii=False))
    assert planned.graph is None
    assert planned.unresolved == ("PLAN_AMBIGUOUS",)
    assert planned.ambiguities == ("杯子的具体位置尚未明确",)
    assert planned.to_dict()["ambiguities"] == ["杯子的具体位置尚未明确"]
    assert planned.tool_result.metrics["unresolved"] == ["PLAN_AMBIGUOUS"]


def test_ambiguous_slot_response_can_omit_unresolved_slots():
    raw = {"request_hash": json.loads(slots_response())["request_hash"],
           "assumptions": [], "ambiguities": ["目标支撑面未明确"],
           "gaps": ["PLAN_AMBIGUOUS"]}
    planned = run_plan_task_world(request(), replay_response=json.dumps(raw, ensure_ascii=False))
    assert planned.graph is None
    assert planned.unresolved == ("PLAN_AMBIGUOUS",)
    assert planned.ambiguities == ("目标支撑面未明确",)


def test_model_generated_sensitive_text_is_rejected_before_delta():
    raw = json.loads(response())
    raw["graph"]["nodes"][0]["label"] = "API_KEY=synthetic-sensitive-marker"
    planned = run_plan_task_world(request(), replay_response=json.dumps(raw))
    assert planned.graph is None
    assert planned.reason_code == "PLAN_RESPONSE_SENSITIVE_CONTENT"
    assert planned.tool_result.proposed_delta is None
    assert "synthetic-sensitive-marker" not in json.dumps(planned.tool_result.to_dict())


def test_commit_refuses_stale_version_and_tampered_delta():
    req = request()
    planned = run_plan_task_world(req, replay_response=response())
    with pytest.raises(GraphError, match="GRAPH_VERSION_CONFLICT"):
        commit_plan_result(req, planned.tool_request, planned.tool_result,
                           current_version="0" * 64)
    delta = planned.tool_result.proposed_delta
    assert delta is not None
    graph = json.loads(json.dumps(delta["task_world_graph"]))
    graph["requirements"][0]["status"] = "passed"
    tampered = replace(planned.tool_result, proposed_delta={"task_world_graph": graph})
    with pytest.raises(GraphError, match="GRAPH_UNVERIFIED_CLAIM"):
        commit_plan_result(req, planned.tool_request, tampered,
                           current_version=planned.tool_request.scene_version)


def test_stale_state_discards_a_valid_tool_proposal():
    planned = run_plan_task_world(request(), replay_response=response(),
                                  current_version=lambda: "0" * 64)
    assert planned.graph is None
    assert planned.reason_code == "GRAPH_VERSION_CONFLICT"
    assert planned.tool_result.status is ToolStatus.PASSED


def test_deadline_after_proposal_prevents_commit():
    ticks = iter((0.0, 0.0, 241.0, 241.0))
    planned = run_plan_task_world(request(), replay_response=response(),
                                  deadline_s=240.0, clock=lambda: next(ticks))
    assert planned.graph is None
    assert planned.reason_code == "PLAN_BUDGET_EXCEEDED"
    assert planned.tool_result.status is ToolStatus.TIMEOUT
    assert planned.tool_result.failure_code is FailureCode.BUDGET_EXHAUSTED
    assert planned.tool_result.proposed_delta is None
    assert planned.tool_result.touched_ids == ()


def test_one_live_call_is_charged_and_still_needs_deterministic_validation(monkeypatch):
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.setenv("TEST_KEY", "synthetic-test-only")
    calls = []

    def fake_call(config, *, system_prompt, user_prompt, images):
        calls.append((system_prompt, user_prompt, images))
        return LLMResponse(response(), "offline-model", 123)

    monkeypatch.setattr("rfw_w.scenes.agents.task_world_planner.call_vision", fake_call)
    config = LLMConfig("https://example.invalid/v1", "offline-model", "TEST_KEY")
    planned = run_plan_task_world(request(), config=config)
    assert planned.graph is not None
    assert planned.budget["model_requests_used"] == 1
    assert len(calls) == 1 and calls[0][2] == ()
    assert "/home/" not in calls[0][0] + calls[0][1]
    assert planned.tool_result.metrics["tokens_used"] == 123


def test_model_failure_records_only_safe_code(monkeypatch):
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.setenv("TEST_KEY", "synthetic-test-only")

    def reject(*args, **kwargs):
        raise LLMError("synthetic-sensitive-marker", code="LLM_HTTP_401")

    monkeypatch.setattr("rfw_w.scenes.agents.task_world_planner.call_vision", reject)
    config = LLMConfig("https://example.invalid/v1", "offline-model", "TEST_KEY")
    planned = run_plan_task_world(request(), config=config)
    assert planned.graph is None
    assert planned.reason_code == "LLM_HTTP_401"
    assert planned.tool_result.failure_code is FailureCode.MODEL_CALL_FAILED
    assert planned.budget["model_requests_used"] == 1
    assert planned.budget["candidates_seen"] == 0
    assert "synthetic-sensitive-marker" not in json.dumps(planned.to_dict())
    assert "synthetic-sensitive-marker" not in json.dumps(planned.tool_result.to_dict())


def test_model_timeout_is_not_a_semantic_failure(monkeypatch):
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.setenv("TEST_KEY", "synthetic-test-only")

    def timeout(*args, **kwargs):
        raise LLMError("request timed out", code="LLM_TIMEOUT")

    monkeypatch.setattr("rfw_w.scenes.agents.task_world_planner.call_vision", timeout)
    config = LLMConfig("https://example.invalid/v1", "offline-model", "TEST_KEY")
    planned = run_plan_task_world(request(), config=config)
    assert planned.graph is None
    assert planned.tool_result.status is ToolStatus.TIMEOUT
    assert planned.tool_result.failure_code is FailureCode.BUDGET_EXHAUSTED
    assert planned.budget["model_requests_used"] == 1
    assert planned.budget["candidates_seen"] == 0


def test_external_permission_is_checked_before_model_request(monkeypatch):
    monkeypatch.delenv("PVSCENE_ALLOW_EXTERNAL_VLM", raising=False)
    monkeypatch.setattr("rfw_w.scenes.agents.task_world_planner.call_vision",
                        lambda *a, **k: pytest.fail("external request"))
    config = LLMConfig("https://example.invalid/v1", "offline-model", "TEST_KEY")
    planned = run_plan_task_world(request(), config=config)
    assert planned.graph is None
    assert planned.reason_code == "EXTERNAL_CALL_NOT_ALLOWED"
    assert planned.budget["model_requests_used"] == 0


def test_missing_credential_is_not_counted_as_an_external_request(monkeypatch):
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.delenv("TEST_KEY", raising=False)
    monkeypatch.setattr("rfw_w.scenes.agents.task_world_planner.call_vision",
                        lambda *a, **k: pytest.fail("external request"))
    config = LLMConfig("https://example.invalid/v1", "offline-model", "TEST_KEY")
    planned = run_plan_task_world(request(), config=config)
    assert planned.graph is None
    assert planned.reason_code == "LLM_CREDENTIAL_MISSING"
    assert planned.budget["model_requests_used"] == 0


def test_request_envelope_uses_a_stable_pregraph_version():
    first = make_plan_request(request())
    second = make_plan_request(request())
    assert first.scene_version == second.scene_version
    assert first.idempotency_key == second.idempotency_key


@pytest.mark.parametrize("deadline", [0, float("nan"), float("inf")])
def test_deadline_cannot_disable_the_wall_clock_bound(deadline):
    with pytest.raises(GraphError, match="PLAN_BUDGET_INVALID"):
        make_plan_request(request(), deadline_s=deadline)
