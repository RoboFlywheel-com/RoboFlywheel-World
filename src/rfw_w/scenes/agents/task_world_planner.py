# SPDX-License-Identifier: Apache-2.0
"""Prompt to candidate task-world graph, using the existing multimodal model client.

The model proposes only task semantics. A deterministic graph contract rejects unsupported
or inconsistent proposals before an orchestrator can commit them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from rfw_w.scenes.agents.llm_client import (
    CredentialsMissing, LLMConfig, LLMResponse, call_vision, require_external_permission,
)
from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph

MAX_PROMPT_CHARS = 16000
MAX_RESPONSE_CHARS = 200000
# A task prompt may be supplied from a file. Refuse obvious credentials and local paths
# before it reaches the model; neither belongs to task semantics.
SENSITIVE_TEXT = re.compile(
    r"(?i)(?:\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*\S+"
    r"|\bBearer\s+\S+|\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,}|AKIA[A-Z0-9]{16})\b"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|/(?:home|mnt|tmp|root|workspace|opt)/"
    r"|[A-Z]:\\|\.env(?:\.[A-Za-z0-9_-]+)?\b)"
)
MODEL_GAPS = frozenset({"TASK_GOAL_MISSING", "TASK_TARGET_UNRESOLVED",
                        "TASK_ACTION_UNSUPPORTED", "ROBOT_MOBILITY_INCOMPATIBLE",
                        "PLAN_AMBIGUOUS", "TASK_ROUTE_DISCONNECTED"})
SYSTEM_PROMPT = """Extract only task semantics for one rigid-object pick, navigate and place
across exactly two rooms and one already-open portal. Return exactly one JSON object with
request_hash, slots, assumptions, ambiguities, gaps. Echo request_hash exactly.
slots has exactly six objects: source_room, destination_room, portal, target,
source_support, destination_support. Each object has id, label, source (user or
inferred); if source=inferred add a short source_reason. IDs must be distinct,
simple stable identifiers. Both room objects also have room_kind. Portal also
has initial_state="open". Target and support objects also have category.
If essential task semantics are missing, omit slots and report a named gap.
Use the user's stated names and places; do not invent a missing room, support,
door or robot capability. The portal connects the two rooms named by the task.
Do not return graph nodes, edges, action effects, geometry, poses, assets, paths,
verification results or simulator claims. A deterministic tool builds these.
assumptions and ambiguities are short string lists. For a fully specified valid
proposal use gaps=[] and ambiguities=[]. Future reach, width, grasp and rollout
measurements are not gaps; the tool records them as status="not_run" obligations.
For an unplannable or ambiguous task, return only these exact gap codes:
TASK_GOAL_MISSING, TASK_TARGET_UNRESOLVED, TASK_ACTION_UNSUPPORTED,
ROBOT_MOBILITY_INCOMPATIBLE, PLAN_AMBIGUOUS, TASK_ROUTE_DISCONNECTED.
If ambiguous, explain in ambiguities and include PLAN_AMBIGUOUS in gaps.
Never simplify an unsupported task into this three-action pattern.
"""


@dataclass(frozen=True, slots=True)
class PlanProposal:
    """Uncommitted model output. Assumptions are data, never evidence."""

    graph: TaskWorldGraph
    assumptions: tuple[str, ...]
    ambiguities: tuple[str, ...] = ()


class PlanModelGap(GraphError):
    """A model refusal with safe, structured unresolved task semantics."""

    def __init__(self, gaps: tuple[str, ...], ambiguities: tuple[str, ...]) -> None:
        super().__init__("PLAN_MODEL_REPORTED_GAPS", ",".join(gaps))
        self.gaps = gaps
        self.ambiguities = ambiguities


def _safe_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping) or set(request) != {"graph_id", "seed", "prompt", "robot"}:
        raise GraphError("PLAN_REQUEST_INVALID", "fields")
    graph_id = request["graph_id"]
    seed = request["seed"]
    prompt = request["prompt"]
    robot = request["robot"]
    if not isinstance(graph_id, str) or not graph_id or type(seed) is not int:
        raise GraphError("PLAN_REQUEST_INVALID", "identity")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise GraphError("PLAN_REQUEST_INVALID", "prompt")
    if not isinstance(robot, Mapping) or set(robot) != {"id", "name", "mobility", "required_clearance_m"}:
        raise GraphError("PLAN_REQUEST_INVALID", "robot")
    if not isinstance(robot["id"], str) or not robot["id"] or not isinstance(robot["name"], str):
        raise GraphError("PLAN_REQUEST_INVALID", "robot identity")
    if not isinstance(robot["mobility"], str) or robot["mobility"] not in {
            "fixed_arm", "mobile_manipulator"}:
        raise GraphError("ROBOT_CAPABILITY_INCOMPLETE", str(robot["mobility"]))
    clearance = robot["required_clearance_m"]
    if type(clearance) not in (int, float) or not 0 < clearance < 5:
        raise GraphError("PLAN_REQUEST_INVALID", "required_clearance_m")
    for field, value in (("graph_id", graph_id), ("prompt", prompt),
                         ("robot.id", robot["id"]), ("robot.name", robot["name"])):
        if SENSITIVE_TEXT.search(value):
            raise GraphError("PLAN_REQUEST_SENSITIVE_CONTENT", field)
    return {"graph_id": graph_id, "seed": seed, "prompt": prompt,
            "robot": dict(robot)}


def request_hash(request: Mapping[str, Any]) -> str:
    return content_hash(_safe_request(request))


def build_plan_prompt(request: Mapping[str, Any]) -> tuple[str, str]:
    safe = _safe_request(request)
    return SYSTEM_PROMPT, json.dumps({**safe, "request_hash": request_hash(safe)},
                                     ensure_ascii=False, sort_keys=True)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise GraphError("PLAN_DUPLICATE_JSON_KEY", key)
        obj[key] = value
    return obj


def validate_plan_graph(raw: Mapping[str, Any], request: Mapping[str, Any]) -> TaskWorldGraph:
    """Recheck a tool delta at commit time, including identity and unsupported claims."""
    safe = _safe_request(request)
    def contains_sensitive(value: Any) -> bool:
        if isinstance(value, str):
            return bool(SENSITIVE_TEXT.search(value))
        if isinstance(value, Mapping):
            return any(contains_sensitive(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_sensitive(item) for item in value)
        return False

    if contains_sensitive(raw):
        raise GraphError("PLAN_RESPONSE_SENSITIVE_CONTENT")
    if not isinstance(raw, dict) or raw.get("graph_id") != safe["graph_id"]:
        raise GraphError("PLAN_INPUT_IDENTITY_CHANGED")
    if raw.get("revision") != 0 or raw.get("parent_hash") is not None:
        raise GraphError("PLAN_INPUT_IDENTITY_CHANGED", "first revision")
    if raw.get("built") != {} or raw.get("evidence_refs") != []:
        raise GraphError("GRAPH_UNVERIFIED_CLAIM", "model returned built state or evidence")
    robot_nodes = [n for n in raw.get("nodes", []) if isinstance(n, dict) and n.get("kind") == "robot"]
    if len(robot_nodes) != 1 or robot_nodes[0].get("id") != safe["robot"]["id"] or (
            robot_nodes[0].get("mobility") != safe["robot"]["mobility"]):
        raise GraphError("PLAN_INPUT_IDENTITY_CHANGED", "robot")
    graph = TaskWorldGraph.from_dict(raw)
    data = graph.to_dict()
    portals = [n for n in data["nodes"] if n["kind"] == "portal"]
    rooms = [n for n in data["nodes"] if n["kind"] == "room"]
    if len(rooms) != 2 or len(portals) < 1:
        raise GraphError("PLAN_SCOPE_UNSUPPORTED", "first slice needs two connected rooms")
    obligations = {r["kind"] for r in data["requirements"]}
    required = {"portal_clearance", "workspace_reach", "asset_grasp",
                "robot_route", "task_rollout"}
    if not required <= obligations:
        raise GraphError("PLAN_OBLIGATION_MISSING", ",".join(sorted(required - obligations)))
    for portal in portals:
        if portal["requested_clearance_m"] < safe["robot"]["required_clearance_m"]:
            raise GraphError("PORTAL_CLEARANCE_INCOMPATIBLE", portal["id"])
    for node in data["nodes"]:
        if node["source"] == "inferred" and not node.get("source_reason"):
            raise GraphError("PLAN_SOURCE_REASON_MISSING", node["id"])
    for step in data["steps"]:
        if step.get("source") not in {"user", "inferred"}:
            raise GraphError("PLAN_STEP_SOURCE_MISSING", step["id"])
        if step["source"] == "inferred" and not step.get("source_reason"):
            raise GraphError("PLAN_SOURCE_REASON_MISSING", step["id"])
    return graph


def parse_plan_proposal(text: str, request: Mapping[str, Any]) -> PlanProposal:
    safe = _safe_request(request)
    if not isinstance(text, str) or len(text) > MAX_RESPONSE_CHARS:
        raise GraphError("PLAN_RESPONSE_LENGTH")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped).strip()
    try:
        response = json.loads(stripped, object_pairs_hook=_unique,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  GraphError("PLAN_NONFINITE_NUMBER", value)))
    except (json.JSONDecodeError, TypeError) as error:
        raise GraphError("PLAN_UNPARSEABLE") from error
    required = {"request_hash", "assumptions", "gaps"}
    if (not isinstance(response, dict) or not required <= set(response)
            or not set(response) <= required | {"graph", "slots", "ambiguities"}
            or ("graph" in response and "slots" in response)):
        raise GraphError("PLAN_RESPONSE_SHAPE")
    if response["request_hash"] != request_hash(safe):
        raise GraphError("PLAN_REQUEST_HASH_MISMATCH")
    gaps = response["gaps"]
    if not isinstance(gaps, list) or any(not isinstance(g, str) or g not in MODEL_GAPS for g in gaps):
        raise GraphError("PLAN_GAPS_INVALID")
    assumptions = response["assumptions"]
    if (not isinstance(assumptions, list) or len(assumptions) > 30
            or any(not isinstance(a, str) or len(a) > 400 or SENSITIVE_TEXT.search(a)
                   for a in assumptions)):
        raise GraphError("PLAN_ASSUMPTIONS_INVALID")
    ambiguities = response.get("ambiguities", [])
    if (not isinstance(ambiguities, list) or len(ambiguities) > 30
            or any(not isinstance(a, str) or not a.strip() or len(a) > 400
                   or SENSITIVE_TEXT.search(a) for a in ambiguities)):
        raise GraphError("PLAN_AMBIGUITIES_INVALID")
    if gaps or ambiguities:
        raise PlanModelGap(tuple(sorted(set(gaps or ["PLAN_AMBIGUOUS"]))), tuple(ambiguities))
    if "graph" not in response and "slots" not in response:
        raise GraphError("PLAN_RESPONSE_SHAPE")
    raw_graph = (compile_pick_place_slots(response["slots"], safe).to_dict()
                 if "slots" in response else response["graph"])
    return PlanProposal(validate_plan_graph(raw_graph, safe), tuple(assumptions))


def parse_plan_response(text: str, request: Mapping[str, Any]) -> TaskWorldGraph:
    """Compatibility entry for the existing replay fixtures and CLI callers."""
    return parse_plan_proposal(text, request).graph


def propose_task_world_graph(
    request: Mapping[str, Any], *, config: LLMConfig,
    on_request: Callable[[], None] | None = None,
    on_response: Callable[[LLMResponse], None] | None = None,
) -> PlanProposal:
    system, user = build_plan_prompt(request)
    require_external_permission()
    if not config.credential_present():
        raise CredentialsMissing()
    if on_request is not None:
        on_request()
    response = call_vision(config, system_prompt=system, user_prompt=user, images=())
    if on_response is not None:
        on_response(response)
    return parse_plan_proposal(response.text, request)


def plan_task_world_graph(
    request: Mapping[str, Any], *, config: LLMConfig,
    on_request: Callable[[], None] | None = None,
    on_response: Callable[[LLMResponse], None] | None = None,
) -> TaskWorldGraph:
    return propose_task_world_graph(request, config=config, on_request=on_request,
                                    on_response=on_response).graph
