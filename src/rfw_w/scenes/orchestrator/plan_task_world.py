# SPDX-License-Identifier: Apache-2.0
"""One bounded transaction from a task description to a validated intent graph.

This is the first cross-room tool boundary, not a general workflow dispatcher. The request
hash stands for the empty pre-planning intent state; no geometry or robot evidence exists yet.
The model proposes a graph, the tool reports it, and this orchestrator revalidates it before
returning a committed revision-zero graph. Callers never commit a model response directly.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from rfw_w.scenes.agents.llm_client import LLMConfig, LLMError, LLMResponse
from rfw_w.scenes.agents.task_world_planner import (
    PlanModelGap, PlanProposal, parse_plan_proposal, propose_task_world_graph, request_hash,
    validate_plan_graph,
)
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.contracts.tool import FailureCode, ToolRequest, ToolResult, ToolStatus

TOOL_VERSION = "plan_task_world_graph/v1"


def _failure_area(detail: str) -> str:
    """Coarse, fixed vocabulary for model-debug evidence; never persist model-supplied IDs."""
    if detail.startswith("initial_state"):
        return "initial_state"
    if detail.startswith("goals"):
        return "goals"
    if detail.endswith(".target_ids"):
        return "step_targets"
    if detail.endswith(".workspace_id"):
        return "step_workspace"
    if detail.endswith(".route_portal_ids"):
        return "step_route"
    if detail.endswith(".actor"):
        return "step_actor"
    if ".preconditions" in detail or ".effects_" in detail:
        return "step_predicate"
    if detail.endswith(".subject_ids"):
        return "requirement_subjects"
    if detail.endswith(".room_id"):
        return "node_room"
    return "topology_or_other"


def initial_intent_version(request: Mapping[str, Any]) -> str:
    """Version the pre-graph state without claiming a built scene exists."""
    return content_hash({"stage": "unplanned_task_world", "request_hash": request_hash(request)})


def make_plan_request(request: Mapping[str, Any], *, deadline_s: float = 240.0) -> ToolRequest:
    if not isinstance(deadline_s, (int, float)) or not math.isfinite(deadline_s) or deadline_s <= 0:
        raise GraphError("PLAN_BUDGET_INVALID", "deadline_s")
    digest = request_hash(request)
    return ToolRequest(
        request_id="plan_" + digest[:24], tool_version=TOOL_VERSION,
        scene_version=initial_intent_version(request), seed=request["seed"],
        deadline_s=deadline_s, scope_ids=(request["graph_id"],), input_refs=(digest,),
        max_candidates=1, idempotency_key=digest,
    )


def commit_plan_result(
    request: Mapping[str, Any], tool_request: ToolRequest, tool_result: ToolResult,
    *, current_version: str,
) -> TaskWorldGraph:
    """Validate the tool envelope and candidate again at the only G1 commit point."""
    if (tool_result.request_id != tool_request.request_id
            or tool_result.tool_version != tool_request.tool_version
            or tool_result.base_scene_version != tool_request.scene_version
            or current_version != tool_request.scene_version):
        raise GraphError("GRAPH_VERSION_CONFLICT")
    if tool_result.status is not ToolStatus.PASSED or tool_result.read_only:
        raise GraphError("PLAN_PROPOSAL_INVALID", "tool status")
    delta = tool_result.proposed_delta
    if not isinstance(delta, Mapping) or set(delta) != {"task_world_graph"}:
        raise GraphError("PLAN_PROPOSAL_INVALID", "delta shape")
    graph = validate_plan_graph(delta["task_world_graph"], request)
    if graph.content_hash != tool_result.metrics.get("graph_hash"):
        raise GraphError("GRAPH_HASH_MISMATCH")
    return graph


@dataclass(frozen=True, slots=True)
class PlanResult:
    tool_request: ToolRequest
    tool_result: ToolResult
    graph: TaskWorldGraph | None
    source: str
    assumptions: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    reason_code: str | None = None
    budget: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = self.graph.to_dict() if self.graph is not None else None
        asset_queries = [] if data is None else [
            {"role_id": n["id"], "category": n["category"],
             "required_capabilities": n["required_capabilities"], "task_role": n.get("task_role"),
             "priority": n.get("priority", "core"),
             "relations": [
                 {"kind": e["kind"], "source_id": e["source_id"], "target_id": e["target_id"]}
                 for e in data["edges"] if e["kind"] in {"supports", "requires_access"}
                 and n["id"] in {e["source_id"], e["target_id"]}
             ]}
            for n in data["nodes"] if n["kind"] == "asset_role"
        ]
        obligations = [] if data is None else [
            {"id": r["id"], "kind": r["kind"], "subject_ids": r["subject_ids"],
             "status": r["status"]} for r in data["requirements"]
        ]
        return {
            "request_id": self.tool_request.request_id,
            "tool_version": self.tool_request.tool_version,
            "status": "planned" if self.graph is not None else "failed",
            "source": self.source,
            "request_hash": self.tool_request.input_refs[0],
            "graph_hash": None if self.graph is None else self.graph.content_hash,
            "assumptions": list(self.assumptions), "ambiguities": list(self.ambiguities),
            "asset_queries": asset_queries,
            "verification_obligations": obligations,
            "unresolved": list(self.unresolved), "reason_code": self.reason_code,
            "budget": dict(self.budget), "robot_validation": "not_run",
            "built_geometry": "not_run",
        }


def run_plan_task_world(
    request: Mapping[str, Any], *, replay_response: str | None = None,
    config: LLMConfig | None = None, deadline_s: float = 240.0,
    clock: Callable[[], float] = time.monotonic,
    current_version: Callable[[], str] | None = None,
) -> PlanResult:
    """Execute one replay or one external call, never more than one candidate or model call."""
    if (replay_response is None) == (config is None):
        raise GraphError("PLAN_MODE_INVALID")
    tool_request = make_plan_request(request, deadline_s=deadline_s)
    source = "replay" if replay_response is not None else "live"
    start = clock()
    response_metrics: dict[str, Any] = {}
    model_calls = 0
    candidates_seen = 0
    proposal: PlanProposal | None = None
    reason_code: str | None = None
    ambiguities: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    status = ToolStatus.PASSED
    failure_code: FailureCode | None = None

    def on_request() -> None:
        nonlocal model_calls
        model_calls += 1

    def on_response(response: LLMResponse) -> None:
        nonlocal candidates_seen
        candidates_seen = 1
        response_metrics.update(model_id=response.model_id, tokens_used=response.tokens_used,
                                response_hash=content_hash({"text": response.text}),
                                response_chars=len(response.text))

    try:
        if clock() - start >= deadline_s:
            raise TimeoutError("deadline before planning")
        if replay_response is not None:
            candidates_seen = 1
            proposal = parse_plan_proposal(replay_response, request)
            response_metrics.update(response_hash=content_hash({"text": replay_response}),
                                    response_chars=len(replay_response))
        else:
            assert config is not None
            # The existing client makes one request. Its timeout may not outlive this tool's budget.
            bounded_config = replace(config, timeout_s=min(config.timeout_s, deadline_s))
            proposal = propose_task_world_graph(request, config=bounded_config,
                                                on_request=on_request, on_response=on_response)
        if clock() - start >= deadline_s:
            raise TimeoutError("deadline after planning")
    except GraphError as error:
        status = ToolStatus.FAILED
        failure_code = FailureCode.PLAN_PROPOSAL_INVALID
        reason_code = error.code
        response_metrics["failure_area"] = _failure_area(error.detail)
        if isinstance(error, PlanModelGap):
            unresolved = error.gaps
            ambiguities = error.ambiguities
    except LLMError as error:
        status = ToolStatus.TIMEOUT if error.code == "LLM_TIMEOUT" else ToolStatus.FAILED
        failure_code = (FailureCode.BUDGET_EXHAUSTED if status is ToolStatus.TIMEOUT
                        else FailureCode.MODEL_CALL_FAILED)
        reason_code = error.code
    except SystemExit:
        status = ToolStatus.UNSUPPORTED
        failure_code = FailureCode.BACKEND_UNSUPPORTED
        reason_code = "EXTERNAL_CALL_NOT_ALLOWED"
    except TimeoutError:
        status = ToolStatus.TIMEOUT
        failure_code = FailureCode.BUDGET_EXHAUSTED
        reason_code = "PLAN_BUDGET_EXCEEDED"

    elapsed_ms = max(0.0, (clock() - start) * 1000)
    metrics = {**response_metrics, "candidates_seen": candidates_seen,
               "model_requests_used": model_calls}
    if reason_code is not None:
        metrics["reason_code"] = reason_code
    if unresolved:
        metrics["unresolved"] = list(unresolved)
    if ambiguities:
        metrics["ambiguities"] = list(ambiguities)
    delta = None if status is not ToolStatus.PASSED or proposal is None else {
        "task_world_graph": proposal.graph.to_dict()}
    if proposal is not None and status is ToolStatus.PASSED:
        metrics["graph_hash"] = proposal.graph.content_hash
    tool_result = ToolResult(
        request_id=tool_request.request_id, status=status, tool_version=TOOL_VERSION,
        base_scene_version=tool_request.scene_version, read_only=False,
        elapsed_ms=elapsed_ms, touched_ids=() if proposal is None or status is not ToolStatus.PASSED else (
            proposal.graph.to_dict()["graph_id"],), proposed_delta=delta,
        metrics=metrics, failure_code=failure_code,
    )
    graph: TaskWorldGraph | None = None
    if status is ToolStatus.PASSED:
        try:
            version = (current_version or (lambda: tool_request.scene_version))()
            graph = commit_plan_result(request, tool_request, tool_result, current_version=version)
        except GraphError as error:
            reason_code = error.code
    budget = {"deadline_s": deadline_s, "elapsed_s": round(elapsed_ms / 1000, 4),
              "model_requests_used": model_calls, "candidates_seen": candidates_seen,
              "within_deadline": elapsed_ms / 1000 < deadline_s}
    return PlanResult(
        tool_request=tool_request, tool_result=tool_result, graph=graph, source=source,
        assumptions=() if proposal is None else proposal.assumptions,
        ambiguities=ambiguities, unresolved=unresolved,
        reason_code=reason_code, budget=budget,
    )
