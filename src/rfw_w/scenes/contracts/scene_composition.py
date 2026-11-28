# SPDX-License-Identifier: Apache-2.0
"""Explicit v2 composition migration; loading legacy graphs never changes semantics."""
from __future__ import annotations

from typing import Any

from rfw_w.scenes.contracts.task_world_graph import (
    COMPOSITION_SCHEMA, COMPOSITION_SCHEMA_V2, GraphError, TaskWorldGraph,
)


def migrate_composition(graph: TaskWorldGraph, *, expected_hash: str) -> TaskWorldGraph:
    if graph.content_hash != expected_hash:
        raise GraphError("GRAPH_VERSION_CONFLICT", "composition migration")
    raw = graph.to_dict()
    if raw.get("composition_schema") != COMPOSITION_SCHEMA:
        raise GraphError("GRAPH_SCHEMA_UNSUPPORTED", "migration requires v1")
    raw["composition_schema"] = COMPOSITION_SCHEMA_V2
    by_id = {n["id"]: n for n in raw["nodes"]}
    parents = {e["target_id"]: e["source_id"] for e in raw["edges"] if e["kind"] == "contains"}
    for node in raw["nodes"]:
        if node["kind"] == "asset_role":
            node["purpose"] = "task_core" if node["priority"] == "core" else "distraction"
            node["placement_kind"] = "surface" if by_id[parents[node["id"]]]["kind"] == "surface" else "floor"
    for node in raw["nodes"]:
        if node["kind"] == "asset_instance":
            role = by_id[parents[node["id"]]]
            node["purpose"] = role["purpose"]
            node["placement_anchor"] = {"kind": role["placement_kind"], "ref": parents[role["id"]]}
    # Migration changes graph identity: prior evidence remains in the source graph only.
    raw["evidence_refs"] = []
    return graph.revise(raw, expected_hash=expected_hash)


def background_role(raw: dict[str, Any], room_id: str, role_id: str, category: str,
                    placement_kind: str, *, purpose: str = "background") -> tuple[dict, dict]:
    if purpose not in {"background", "distraction"}:
        raise GraphError("COMPOSITION_PURPOSE_INVALID", role_id)
    kind = {"floor": "floor", "wall": "wall_set", "surface": "surface"}.get(placement_kind)
    anchors = [n["id"] for n in raw["nodes"] if n["kind"] == kind and n.get("room_id") == room_id]
    if len(anchors) != 1:
        raise GraphError("COMPOSITION_ANCHOR_INVALID", role_id)
    role = {"id": role_id, "kind": "asset_role", "label": category, "source": "inferred",
            "source_reason": "optional room furnishing selected within task constraints",
            "room_id": room_id, "category": category,
            "task_role": "distractor" if purpose == "distraction" else "background",
            "required_capabilities": ["rigid"], "priority": "optional",
            "physical_class": "fixture" if placement_kind == "wall" else "furniture",
            "purpose": purpose, "placement_kind": placement_kind}
    return role, {"id": "e_contains_" + role_id, "kind": "contains",
                  "source_id": anchors[0], "target_id": role_id}


def migrate_covering_composition(graph, *, expected_hash):
    from rfw_w.scenes.contracts.task_world_graph import COMPOSITION_SCHEMA_V3
    if graph.content_hash!=expected_hash:raise GraphError('GRAPH_VERSION_CONFLICT','covering migration')
    raw=graph.to_dict()
    if raw.get('composition_schema')!=COMPOSITION_SCHEMA_V2:raise GraphError('GRAPH_SCHEMA_UNSUPPORTED','migration requires v2')
    raw['composition_schema']=COMPOSITION_SCHEMA_V3;raw['evidence_refs']=[]
    return graph.revise(raw,expected_hash=expected_hash)
