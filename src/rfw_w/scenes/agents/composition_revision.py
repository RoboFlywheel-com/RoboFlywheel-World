# SPDX-License-Identifier: Apache-2.0
"""Restricted semantic patches; models propose, deterministic graph checks decide."""
from __future__ import annotations

import json
from typing import Any, Mapping

from rfw_w.scenes.agents.llm_client import LLMConfig, call_vision, require_external_permission
from rfw_w.scenes.agents.task_world_planner import SENSITIVE_TEXT, _unique
from rfw_w.scenes.contracts.scene_composition import background_role
from rfw_w.scenes.contracts.task_world_graph import COMPOSITION_SCHEMA_V2, GraphError, TaskWorldGraph

CATEGORY_ALTERNATIVES = {"desk": {"desk", "table"}, "counter": {"counter", "kitchen_island"}}
BACKGROUND_CATEGORIES = {"floor": {"cabinet", "bookshelf", "chair", "side_table"},
                         "wall": {"picture", "mirror"}, "surface": {"book", "plate", "bowl"}}


def validate_composition_patch(graph: TaskWorldGraph, patch: Mapping[str, Any], *,
                               expected_hash: str, scope_ids: tuple[str, ...]) -> TaskWorldGraph:
    try:
        return _validate_composition_patch(graph,patch,expected_hash=expected_hash,scope_ids=scope_ids)
    except (TypeError,KeyError,AttributeError):
        raise GraphError("COMPOSITION_PATCH_INVALID", "field types") from None


def _validate_composition_patch(graph: TaskWorldGraph, patch: Mapping[str, Any], *,
                                expected_hash: str, scope_ids: tuple[str, ...]) -> TaskWorldGraph:
    if graph.content_hash != expected_hash:
        raise GraphError("GRAPH_VERSION_CONFLICT", "composition patch")
    raw = graph.to_dict()
    if raw.get("composition_schema") != COMPOSITION_SCHEMA_V2:
        raise GraphError("GRAPH_SCHEMA_UNSUPPORTED", "composition patch requires v2")
    if set(patch) != {"operations"} or not isinstance(patch["operations"], list) or not 1 <= len(patch["operations"]) <= 12:
        raise GraphError("COMPOSITION_PATCH_INVALID", "operations")
    if SENSITIVE_TEXT.search(json.dumps(patch)):
        raise GraphError("COMPOSITION_PATCH_INVALID", "sensitive content")
    # Edits after construction require an explicit rollback transaction first.
    if raw["built"]:
        raise GraphError("PATCH_SCOPE_VIOLATION", "built scene requires checkpoint rollback")
    by_id = {n["id"]: n for n in raw["nodes"]}
    for op in patch["operations"]:
        if not isinstance(op, dict):
            raise GraphError("COMPOSITION_PATCH_INVALID", "operation shape")
        action = op.get("action")
        if action == "replace_inferred_category":
            if set(op) != {"action", "target_id", "category", "reason_code"}:
                raise GraphError("COMPOSITION_PATCH_INVALID", "replace fields")
            node = by_id.get(op["target_id"])
            if not node or node["id"] not in scope_ids:
                raise GraphError("PATCH_SCOPE_VIOLATION", "target")
            if node["kind"] != "asset_role" or node["source"] != "inferred" or node.get("task_role") == "target":
                raise GraphError("CORE_CONSTRAINT_CHANGED", node["id"])
            if op["category"] not in CATEGORY_ALTERNATIVES.get(node["category"], set()):
                raise GraphError("CORE_CONSTRAINT_CHANGED", node["id"])
            node["source_reason"] = "category revision from " + node["category"] + ": " + str(op["reason_code"])
            node["category"] = op["category"]
        elif action == "add_optional_role":
            if set(op) != {"action", "room_id", "role_id", "category", "placement_kind", "purpose"}:
                raise GraphError("COMPOSITION_PATCH_INVALID", "add fields")
            if op["room_id"] not in scope_ids:
                raise GraphError("PATCH_SCOPE_VIOLATION", op["room_id"])
            if op["category"] not in BACKGROUND_CATEGORIES.get(op["placement_kind"], set()):
                raise GraphError("COMPOSITION_PATCH_INVALID", "unsupported category")
            role, edge = background_role(raw, op["room_id"], op["role_id"], op["category"],
                                         op["placement_kind"], purpose=op["purpose"])
            raw["nodes"].append(role); raw["edges"].append(edge)
            by_id[role["id"]] = role
        elif action == "remove_optional_role":
            if set(op) != {"action", "target_id"}:
                raise GraphError("COMPOSITION_PATCH_INVALID", "remove fields")
            node = by_id.get(op["target_id"])
            if not node or node["id"] not in scope_ids:
                raise GraphError("PATCH_SCOPE_VIOLATION", "remove target")
            if node.get("priority") != "optional" or node["kind"] != "asset_role":
                raise GraphError("CORE_CONSTRAINT_CHANGED", node["id"])
            if any(e["source_id"] == node["id"] for e in raw["edges"]):
                raise GraphError("COMPOSITION_PATCH_INVALID", "remove parent with descendants")
            raw["nodes"] = [n for n in raw["nodes"] if n["id"] != node["id"]]
            raw["edges"] = [e for e in raw["edges"] if e["target_id"] != node["id"]]
            del by_id[node["id"]]
        else:
            raise GraphError("COMPOSITION_PATCH_INVALID", "unsupported action")
    raw["evidence_refs"] = []
    candidate = graph.revise(raw, expected_hash=expected_hash)
    before, after = graph.task_view(), candidate.task_view()
    if any(before[k] != after[k] for k in ("initial_state", "steps", "goals")):
        raise GraphError("TASK_GOAL_CHANGED", "task identity")
    return candidate


def propose_composition_patch(task_semantics: Mapping[str, Any], allowed_options: list[dict],
                              config: LLMConfig) -> dict:
    require_external_permission()
    payload = json.dumps({"task": task_semantics, "allowed_options": allowed_options}, ensure_ascii=False)
    if SENSITIVE_TEXT.search(payload):
        raise GraphError("COMPOSITION_PATCH_INVALID", "sensitive prompt")
    response = call_vision(config, system_prompt="Choose one permitted semantic scene patch. Return only its JSON object with operations. Do not invent fields, geometry, assets or verification.",
                           user_prompt=payload)
    try:
        result = json.loads(response.text, object_pairs_hook=_unique)
    except (ValueError, TypeError):
        raise GraphError("COMPOSITION_PATCH_INVALID", "model JSON") from None
    if result not in allowed_options:
        raise GraphError("COMPOSITION_PATCH_INVALID", "model chose an unlisted option")
    return result
