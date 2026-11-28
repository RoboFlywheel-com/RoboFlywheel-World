# SPDX-License-Identifier: Apache-2.0
"""Compile a bounded two-room pick/place semantic proposal into graph intent.

The model names task entities. This module owns IDs for derived workspaces, exact
symbolic state transitions and not-run verification obligations.
"""
from __future__ import annotations

from typing import Any, Mapping

from rfw_w.scenes.contracts.task_world_graph import (
    COMPOSITION_SCHEMA, GraphError, TaskWorldGraph, SCHEMA_VERSION,
)

SLOT_KEYS = frozenset({"source_room", "destination_room", "portal", "target",
                       "source_support", "destination_support"})
RESERVED_IDS = frozenset({"house", "work_source", "work_destination", "step_pick",
                          "step_navigate", "step_place", "door_clearance", "source_reach",
                          "target_reach", "grasp_check", "route_check", "rollout_check"})


def _slot(raw: Any, name: str, *, category: bool = False, room: bool = False,
          portal: bool = False) -> dict[str, Any]:
    allowed = {"id", "label", "source", "source_reason"}
    if category:
        allowed.add("category")
    if room:
        allowed.add("room_kind")
    if portal:
        allowed.add("initial_state")
    required = allowed - {"source_reason"}
    if not isinstance(raw, dict) or not required <= set(raw) <= allowed:
        raise GraphError("PLAN_SLOT_INVALID", name)
    for field in required - {"source"}:
        if not isinstance(raw[field], str) or not raw[field].strip() or len(raw[field]) > 160:
            raise GraphError("PLAN_SLOT_INVALID", name + "." + field)
    if raw["source"] not in {"user", "inferred"}:
        raise GraphError("PLAN_SLOT_INVALID", name + ".source")
    if raw["source"] == "inferred" and (not isinstance(raw.get("source_reason"), str)
                                          or not raw["source_reason"].strip()):
        raise GraphError("PLAN_SOURCE_REASON_MISSING", name)
    if "source_reason" in raw and (not isinstance(raw["source_reason"], str)
                                   or not raw["source_reason"].strip()
                                   or len(raw["source_reason"]) > 400):
        raise GraphError("PLAN_SLOT_INVALID", name + ".source_reason")
    if portal and raw["initial_state"] != "open":
        raise GraphError("TASK_ACTION_UNSUPPORTED", "closed_portal")
    return dict(raw)


def _predicate(subject: str, predicate: str, obj: str) -> dict[str, str]:
    return {"subject": subject, "predicate": predicate, "object": obj}


def compile_pick_place_slots(slots: Mapping[str, Any], request: Mapping[str, Any]) -> TaskWorldGraph:
    """Build the only supported G1 action pattern from checked semantic slots."""
    if not isinstance(slots, dict) or set(slots) != SLOT_KEYS:
        raise GraphError("PLAN_SLOT_INVALID", "fields")
    src = _slot(slots["source_room"], "source_room", room=True)
    dst = _slot(slots["destination_room"], "destination_room", room=True)
    door = _slot(slots["portal"], "portal", portal=True)
    target = _slot(slots["target"], "target", category=True)
    source_support = _slot(slots["source_support"], "source_support", category=True)
    destination_support = _slot(slots["destination_support"], "destination_support", category=True)
    values = (src, dst, door, target, source_support, destination_support)
    ids = [item["id"] for item in values]
    robot_id = request["robot"]["id"]
    if len(set(ids)) != len(ids) or set(ids) & (RESERVED_IDS | {robot_id}):
        raise GraphError("PLAN_SLOT_ID_CONFLICT", "duplicate_or_reserved")
    if request["robot"]["mobility"] != "mobile_manipulator":
        raise GraphError("ROBOT_MOBILITY_INCOMPATIBLE", robot_id)

    def node(item: dict[str, Any], kind: str, **extra: Any) -> dict[str, Any]:
        return {"id": item["id"], "kind": kind, "label": item["label"],
                "source": item["source"],
                **({"source_reason": item["source_reason"]} if "source_reason" in item else {}),
                **extra}

    def edge(kind: str, a: str, b: str) -> dict[str, str]:
        return {"id": "e_" + a + "_" + b, "kind": kind, "source_id": a, "target_id": b}

    def requirement(name: str, kind: str, subjects: list[str]) -> dict[str, Any]:
        return {"id": name, "kind": kind, "subject_ids": subjects, "status": "not_run"}

    cup_on_source = _predicate(target["id"], "ON", source_support["id"])
    cup_held = _predicate(target["id"], "HELD_BY", robot_id)
    cup_on_destination = _predicate(target["id"], "ON", destination_support["id"])
    robot_at_source = _predicate(robot_id, "IN_ROOM", src["id"])
    robot_at_destination = _predicate(robot_id, "IN_ROOM", dst["id"])

    data = {
        "schema_version": SCHEMA_VERSION, "graph_id": request["graph_id"],
        "revision": 0, "parent_hash": None, "units": "m", "up_axis": "Z",
        "nodes": [
            {"id": "house", "kind": "house", "label": "task house", "source": "inferred",
             "source_reason": "the two task rooms need one graph container"},
            node(src, "room", room_kind=src["room_kind"]),
            node(dst, "room", room_kind=dst["room_kind"]),
            node(door, "portal", room_ids=[src["id"], dst["id"]],
                 initial_state="open",
                 requested_clearance_m=request["robot"]["required_clearance_m"]),
            {"id": "work_source", "kind": "workspace", "label": "source operation area",
             "source": "inferred", "source_reason": "pick needs a source-side workspace",
             "room_id": src["id"]},
            {"id": "work_destination", "kind": "workspace", "label": "destination operation area",
             "source": "inferred", "source_reason": "place needs a destination-side workspace",
             "room_id": dst["id"]},
            {"id": robot_id, "kind": "robot", "label": request["robot"]["name"],
             "mobility": request["robot"]["mobility"], "source": "user"},
            node(target, "asset_role", room_id=src["id"], category=target["category"],
                 task_role="target", required_capabilities=["rigid", "graspable"]),
            node(source_support, "asset_role", room_id=src["id"],
                 category=source_support["category"], task_role="source_support",
                 required_capabilities=["support_surface"]),
            node(destination_support, "asset_role", room_id=dst["id"],
                 category=destination_support["category"], task_role="destination_support",
                 required_capabilities=["support_surface"]),
        ],
        "edges": [
            edge("contains", "house", src["id"]), edge("contains", "house", dst["id"]),
            edge("connects", door["id"], src["id"]), edge("connects", door["id"], dst["id"]),
            edge("contains", src["id"], target["id"]),
            edge("contains", src["id"], source_support["id"]),
            edge("contains", dst["id"], destination_support["id"]),
            edge("contains", src["id"], "work_source"),
            edge("contains", dst["id"], "work_destination"),
            edge("supports", source_support["id"], target["id"]),
        ],
        "initial_state": [cup_on_source, robot_at_source],
        "steps": [
            {"id": "step_pick", "verb": "PICK", "source": "user", "actor": robot_id,
             "target_ids": [target["id"]], "workspace_id": "work_source", "route_portal_ids": [],
             "preconditions": [cup_on_source, robot_at_source],
             "effects_remove": [cup_on_source], "effects_add": [cup_held]},
            {"id": "step_navigate", "verb": "NAVIGATE", "source": "user", "actor": robot_id,
             "target_ids": [], "workspace_id": "work_destination",
             "route_portal_ids": [door["id"]],
             "preconditions": [cup_held, robot_at_source],
             "effects_remove": [robot_at_source], "effects_add": [robot_at_destination]},
            {"id": "step_place", "verb": "PLACE", "source": "user", "actor": robot_id,
             "target_ids": [target["id"]], "workspace_id": "work_destination",
             "route_portal_ids": [], "preconditions": [cup_held, robot_at_destination],
             "effects_remove": [cup_held], "effects_add": [cup_on_destination]},
        ],
        "goals": [cup_on_destination],
        "requirements": [
            requirement("door_clearance", "portal_clearance", [door["id"], robot_id]),
            requirement("source_reach", "workspace_reach", ["work_source", robot_id, target["id"]]),
            requirement("target_reach", "workspace_reach", ["work_destination", robot_id,
                                                           destination_support["id"]]),
            requirement("grasp_check", "asset_grasp", [target["id"], robot_id]),
            requirement("route_check", "robot_route", [src["id"], door["id"], dst["id"], robot_id]),
            requirement("rollout_check", "task_rollout", [robot_id, target["id"],
                                                           destination_support["id"]]),
        ],
        "built": {}, "evidence_refs": [],
    }
    # One initial placement tree is derived from the task slots. G2 owns actual wall
    # segments and geometry; these nodes are requirements, not built measurements.
    data["composition_schema"] = COMPOSITION_SCHEMA
    extra_nodes: list[dict[str, Any]] = []
    extra_edges: list[dict[str, str]] = []
    existing_ids = {node["id"] for node in data["nodes"]}

    def add(kind: str, identifier: str, label: str, room_id: str,
            parent_id: str, *, reason: str = "deterministic room composition from task roles",
            **fields: Any) -> None:
        if identifier in existing_ids:
            raise GraphError("PLAN_SLOT_ID_CONFLICT", identifier)
        existing_ids.add(identifier)
        extra_nodes.append({"id": identifier, "kind": kind, "label": label,
                            "source": "inferred", "source_reason": reason,
                            "room_id": room_id, **fields})
        extra_edges.append(edge("contains", parent_id, identifier))

    for room in (src, dst):
        rid = room["id"]
        add("floor", "rc_" + rid + "_floor", "Floor · planned", rid, rid)
        add("wall_set", "rc_" + rid + "_walls", "Walls · segments pending", rid, rid)
    for room, support, side in ((src, source_support, "source"),
                                (dst, destination_support, "destination")):
        rid = room["id"]
        surface_id = "rc_" + rid + "_" + side + "_surface"
        add("surface", surface_id, support["label"] + " · placement surface",
            rid, support["id"])
        categories = ({"kitchen": ["plate", "bowl"],
                       "study": ["book", "notebook"],
                       "office": ["book", "notebook"]}.get(room["room_kind"],
                                                         ["small_decor"]))
        categories = [category for category in categories
                      if category.casefold() != target["category"].casefold()]
        if not categories:
            categories = ["neutral_decor"]
        add("distractor_policy", "rc_" + rid + "_" + side + "_distractors",
            "Optional distractors", rid, surface_id,
            reason="optional candidates inferred from room type, not requested by the user",
            candidate_categories=categories, max_count=2)
        # One optional semantic candidate makes the room tree useful to downstream
        # retrieval. Its absence must never invalidate the core pick/place task.
        add("asset_role", "rc_" + rid + "_" + side + "_optional_1",
            categories[0], rid, surface_id,
            reason="optional candidate inferred from room type, not requested by the user",
            category=categories[0],
            task_role="distractor", required_capabilities=["rigid"],
            priority="optional", physical_class="manipuland")
    data["nodes"].extend(extra_nodes)
    # Replace only the target's generation parent. Its task state remains ON the
    # source support until PICK, and its stable ID is used again after PLACE.
    data["edges"] = [e for e in data["edges"]
                     if not (e["kind"] == "contains" and e["target_id"] == target["id"])]
    data["edges"].append(edge("contains", "rc_" + src["id"] + "_source_surface",
                              target["id"]))
    data["edges"].extend(extra_edges)
    for item in data["nodes"]:
        if item["kind"] == "asset_role" and "priority" not in item:
            item["priority"] = "core"
            item["physical_class"] = ("manipuland" if item["id"] == target["id"]
                                      else "furniture")
    return TaskWorldGraph.from_dict(data)
