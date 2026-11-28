# SPDX-License-Identifier: Apache-2.0
"""Task-first house graph. Pure data and deterministic checks; no model or geometry imports.

The graph describes intent. A portal connects rooms, a task step changes symbolic state,
and an obligation says what a later tool must measure. None of these claims that a robot
has actually traversed a door or grasped an asset.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from rfw_w.scenes.contracts.hashing import content_hash, is_digest

SCHEMA_VERSION = "pvscene.task-world-graph/v1"
NODE_KINDS = frozenset({"house", "room", "portal", "zone", "workspace", "robot",
                        "asset_role", "asset_instance", "floor", "wall_set", "surface",
                        "distractor_policy"})
COMPOSITION_SCHEMA = "pvscene.room-composition/v1"
COMPOSITION_SCHEMA_V2 = "pvscene.room-composition/v2"
COMPOSITION_SCHEMA_V3 = "pvscene.room-composition/v3"
EDGE_KINDS = frozenset({"contains", "connects", "supports", "requires_access", "role_bound_to"})
PREDICATES = frozenset({"IN_ROOM", "ON", "INSIDE", "HELD_BY", "DOOR_STATE"})
VERBS = frozenset({"PICK", "NAVIGATE", "PLACE"})
TOP_KEYS = frozenset({"schema_version", "graph_id", "revision", "parent_hash", "units",
                      "up_axis", "nodes", "edges", "initial_state", "steps", "goals",
                      "requirements", "built", "evidence_refs", "content_hash",
                      "composition_schema"})
NODE_FIELDS = {
    "house": {"id", "kind", "label", "source"},
    "room": {"id", "kind", "label", "source", "room_kind"},
    "portal": {"id", "kind", "label", "source", "room_ids", "initial_state",
               "requested_clearance_m"},
    "zone": {"id", "kind", "label", "source", "room_id"},
    "workspace": {"id", "kind", "label", "source", "room_id"},
    "robot": {"id", "kind", "label", "source", "mobility"},
    "asset_role": {"id", "kind", "label", "source", "room_id", "category",
                   "task_role", "required_capabilities", "priority", "physical_class"},
    "asset_instance": {"id", "kind", "label", "source", "room_id", "asset_ref"},
    "floor": {"id", "kind", "label", "source", "room_id"},
    "wall_set": {"id", "kind", "label", "source", "room_id"},
    "surface": {"id", "kind", "label", "source", "room_id"},
    "distractor_policy": {"id", "kind", "label", "source", "room_id",
                          "candidate_categories", "max_count"},
}
EDGE_FIELDS = {"id", "kind", "source_id", "target_id"}
STEP_FIELDS = {"id", "verb", "actor", "target_ids", "workspace_id", "route_portal_ids",
               "preconditions", "effects_remove", "effects_add"}
STEP_PROVENANCE_FIELDS = {"source", "source_reason"}
REQUIREMENT_FIELDS = {"id", "kind", "subject_ids", "status"}


class GraphError(ValueError):
    """A named, attributable graph defect."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}({detail})")


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise GraphError(code, detail)


def _items(value: Any, name: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list), "GRAPH_SHAPE_INVALID", name)
    _require(all(isinstance(item, dict) for item in value), "GRAPH_SHAPE_INVALID", name)
    return value


def _name(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), "GRAPH_SHAPE_INVALID", name)
    return value


def _predicate(raw: Any, ids: set[str], where: str) -> tuple[str, str, str]:
    _require(isinstance(raw, dict) and set(raw) == {"subject", "predicate", "object"},
             "GRAPH_SHAPE_INVALID", where)
    subject = _name(raw["subject"], where + ".subject")
    obj = _name(raw["object"], where + ".object")
    relation = _name(raw["predicate"], where + ".predicate")
    _require(subject in ids and obj in ids, "GRAPH_DANGLING_REF", where)
    _require(relation in PREDICATES, "TASK_PREDICATE_UNSUPPORTED", relation)
    return subject, relation, obj


def _location_conflict(state: set[tuple[str, str, str]]) -> None:
    owners: dict[str, list[tuple[str, str]]] = {}
    for subject, relation, obj in state:
        if relation in {"ON", "INSIDE", "HELD_BY", "IN_ROOM"}:
            owners.setdefault(subject, []).append((relation, obj))
    for subject, places in owners.items():
        _require(len(places) <= 1, "TASK_STATE_CONFLICT", subject)


def _validate_composition(raw: dict[str, Any], by_id: dict[str, dict[str, Any]],
                          edges: list[dict[str, Any]]) -> None:
    """The contains tree is an initial generation order, not the task's mutable state."""
    version = raw.get("composition_schema")
    structural = {"floor", "wall_set", "surface", "distractor_policy"}
    if version is None:
        _require(not any(n["kind"] in structural or "priority" in n for n in by_id.values()),
                 "GRAPH_SCHEMA_UNSUPPORTED", "composition_schema missing")
        return
    _require(version in {COMPOSITION_SCHEMA, COMPOSITION_SCHEMA_V2, COMPOSITION_SCHEMA_V3}, "GRAPH_SCHEMA_UNSUPPORTED", str(version))
    v2 = version in {COMPOSITION_SCHEMA_V2, COMPOSITION_SCHEMA_V3}
    rooms = {key for key, n in by_id.items() if n["kind"] == "room"}
    house = next(key for key, n in by_id.items() if n["kind"] == "house")
    parents: dict[str, str] = {}
    for edge in edges:
        if edge["kind"] != "contains":
            continue
        child, parent = edge["target_id"], edge["source_id"]
        _require(child not in parents, "COMPOSITION_PARENT_CONFLICT", child)
        parents[child] = parent
    _require(house not in parents, "COMPOSITION_PARENT_CONFLICT", house)
    for key, node in by_id.items():
        kind = node["kind"]
        if kind in {"house", "portal", "robot"}:
            _require(key not in parents, "COMPOSITION_PARENT_CONFLICT", key)
            continue
        _require(key in parents, "COMPOSITION_PARENT_MISSING", key)
        parent = parents[key]
        _require(parent in by_id, "COMPOSITION_PARENT_MISSING", key)
        if kind == "room":
            _require(parent == house, "COMPOSITION_ROOM_MISMATCH", key)
        else:
            room_id = node.get("room_id")
            _require(room_id in rooms, "COMPOSITION_ROOM_MISMATCH", key)
            parent_node = by_id[parent]
            _require(parent == room_id or parent_node.get("room_id") == room_id,
                     "COMPOSITION_ROOM_MISMATCH", key)
        seen = {key}
        ancestor = parent
        while ancestor in parents:
            _require(ancestor not in seen, "COMPOSITION_CYCLE", key)
            seen.add(ancestor)
            ancestor = parents[ancestor]
        _require(ancestor == house, "COMPOSITION_PARENT_MISSING", key)
        if kind in {"floor", "wall_set"}:
            _require(parent == node["room_id"], "COMPOSITION_ROOM_MISMATCH", key)
        if kind == "surface":
            _require(by_id[parent]["kind"] == "asset_role", "COMPOSITION_SUPPORT_MISMATCH", key)
        if kind == "distractor_policy":
            _require(by_id[parent]["kind"] in {"surface", "room", "asset_role"},
                     "COMPOSITION_PARENT_CONFLICT", key)
            categories = node.get("candidate_categories")
            _require(isinstance(categories, list) and bool(categories)
                     and all(isinstance(c, str) and c.strip() for c in categories),
                     "GRAPH_SHAPE_INVALID", key + ".candidate_categories")
            count = node.get("max_count")
            _require(type(count) is int and 0 <= count <= 24,
                     "GRAPH_SHAPE_INVALID", key + ".max_count")
        if kind == "asset_role":
            _require(node.get("priority") in {"core", "optional"},
                     "GRAPH_SHAPE_INVALID", key + ".priority")
            _require(node.get("physical_class") in ({"furniture", "manipuland", "fixture", "covering"} if version == COMPOSITION_SCHEMA_V3 else {"furniture", "manipuland", "fixture"} if v2 else {"furniture", "manipuland"}),
                     "GRAPH_SHAPE_INVALID", key + ".physical_class")
            _require(v2 or node["priority"] != "optional" or node.get("task_role") == "distractor",
                     "COMPOSITION_CORE_OPTIONAL", key)
            _require(v2 or node.get("task_role") not in {"target", "source_support",
                                                     "destination_support"}
                     or node["priority"] == "core", "COMPOSITION_CORE_OPTIONAL", key)
            if node.get('physical_class')=='covering':
                _require(node.get('priority')=='optional' and node.get('purpose')!='task_core'
                         and node.get('placement_kind') in ('floor','wall'), 'COVERING_CORE_FORBIDDEN',key)
            if v2:
                purpose = node.get("purpose")
                _require(purpose in {"task_core", "background", "distraction"}
                         and (node["priority"] == "core") == (purpose == "task_core"),
                         "COMPOSITION_PURPOSE_INVALID", key)
                placement = node.get("placement_kind")
                _require(placement in {"floor", "surface", "wall"}, "COMPOSITION_ANCHOR_INVALID", key)
                allowed = {"floor": {"room", "floor"}, "surface": {"surface"}, "wall": {"wall_set"}}
                _require(by_id[parent]["kind"] in allowed[placement], "COMPOSITION_ANCHOR_INVALID", key)
        if kind == "asset_instance" and v2:
            role = by_id[parent]
            bindings = [e for e in edges if e["kind"] == "role_bound_to" and e["target_id"] == key]
            _require(role["kind"] == "asset_role" and len(bindings) == 1
                     and bindings[0]["source_id"] == parent
                     and node.get("purpose") == role.get("purpose"), "COMPOSITION_PURPOSE_INVALID", key)
            anchor = node.get("placement_anchor")
            _require(isinstance(anchor, dict) and set(anchor) == {"kind", "ref"}
                     and anchor["kind"] == role.get("placement_kind")
                     and isinstance(anchor["ref"], str) and bool(anchor["ref"]),
                     "COMPOSITION_ANCHOR_INVALID", key)
            if role.get('physical_class')=='covering':
                surfaces=raw['built'].get('surface_coverings',{}).get('anchors',{})
                _require(anchor['ref'] in surfaces and surfaces[anchor['ref']].get('room_id')==node['room_id'],
                         'COMPOSITION_ANCHOR_INVALID',key)
            elif anchor["kind"] != "wall":
                _require(anchor["ref"] == parents[parent], "COMPOSITION_ANCHOR_INVALID", key)
            else:
                mounts = raw["built"].get("wall_mounts", {}).get("anchors", {})
                _require(anchor["ref"] in mounts
                         and mounts[anchor["ref"]].get("room_id") == node["room_id"],
                         "COMPOSITION_ANCHOR_INVALID", key)
    for room_id in rooms:
        for kind in ("floor", "wall_set"):
            _require(sum(n["kind"] == kind and n.get("room_id") == room_id
                         for n in by_id.values()) == 1,
                     "COMPOSITION_STRUCTURE_MISSING", room_id + "." + kind)
    optional = {key for key, n in by_id.items()
                if n["kind"] == "asset_role" and n.get("priority") == "optional"}
    task_ids = {p[side] for p in raw["initial_state"] + raw["goals"]
                for side in ("subject", "object")}
    for step in raw["steps"]:
        task_ids.update(step["target_ids"])
        for field in ("preconditions", "effects_remove", "effects_add"):
            task_ids.update(p[side] for p in step[field] for side in ("subject", "object"))
    for requirement in raw["requirements"]:
        task_ids.update(requirement["subject_ids"])
    if v2:
        # Structural surfaces inherit task necessity from their generation ancestors.
        for key in tuple(task_ids):
            ancestor = parents.get(key)
            while ancestor is not None:
                task_ids.add(ancestor)
                ancestor = parents.get(ancestor)
    for key in optional & task_ids:
        _require(False, "COMPOSITION_BACKGROUND_IN_TASK" if v2 else "COMPOSITION_OPTIONAL_IN_TASK", key)
    for edge in edges:
        if edge["kind"] in {"supports", "requires_access"}:
            for endpoint in (edge["source_id"], edge["target_id"]):
                _require(endpoint not in optional, "COMPOSITION_BACKGROUND_IN_TASK" if v2 else "COMPOSITION_OPTIONAL_IN_TASK", endpoint)
    for predicate in raw["initial_state"]:
        subject, relation, support = (predicate["subject"], predicate["predicate"],
                                      predicate["object"])
        if relation not in {"ON", "INSIDE"} or by_id[subject]["kind"] != "asset_role":
            continue
        _require(support in by_id and by_id[support]["kind"] == "asset_role",
                 "COMPOSITION_SUPPORT_MISMATCH", subject)
        ancestor = parents.get(subject)
        while ancestor is not None and ancestor != support:
            ancestor = parents.get(ancestor)
        _require(ancestor == support, "COMPOSITION_SUPPORT_MISMATCH", subject)


def _validate(raw: dict[str, Any]) -> None:
    _require(set(raw) <= TOP_KEYS and TOP_KEYS - {"content_hash", "composition_schema"} <= set(raw),
             "GRAPH_SHAPE_INVALID", "top-level fields")
    _require(raw["schema_version"] == SCHEMA_VERSION, "GRAPH_SCHEMA_UNSUPPORTED")
    _name(raw["graph_id"], "graph_id")
    _require(type(raw["revision"]) is int and raw["revision"] >= 0, "GRAPH_SHAPE_INVALID", "revision")
    _require(raw["parent_hash"] is None or is_digest(raw["parent_hash"]),
             "GRAPH_VERSION_CONFLICT", "parent_hash")
    _require(raw["units"] == "m" and raw["up_axis"] == "Z", "GRAPH_FRAME_MISMATCH", "world")
    _require(isinstance(raw["built"], dict), "GRAPH_SHAPE_INVALID", "built")
    _require(isinstance(raw["evidence_refs"], list), "GRAPH_SHAPE_INVALID", "evidence_refs")

    nodes = _items(raw["nodes"], "nodes")
    _require(nodes, "GRAPH_SHAPE_INVALID", "nodes empty")
    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = _name(node.get("id"), "node.id")
        _require(node_id not in by_id, "GRAPH_DUPLICATE_ID", node_id)
        _require(isinstance(node.get("kind"), str) and node["kind"] in NODE_KINDS,
                 "GRAPH_NODE_KIND_UNSUPPORTED", node_id)
        extra = ({"purpose", "placement_kind"} if node["kind"] == "asset_role" else
                 {"purpose", "placement_anchor"} if node["kind"] == "asset_instance" else set())
        _require(set(node) <= NODE_FIELDS[node["kind"]] | {"source_reason"} |
                 (extra if raw.get("composition_schema") in {COMPOSITION_SCHEMA_V2, COMPOSITION_SCHEMA_V3} else set()),
                 "GRAPH_SHAPE_INVALID", node_id + ".fields")
        _name(node.get("label"), node_id + ".label")
        _require(node.get("source") in {"user", "inferred"}, "GRAPH_SOURCE_MISSING", node_id)
        if "source_reason" in node:
            _require(isinstance(node["source_reason"], str) and bool(node["source_reason"].strip())
                     and len(node["source_reason"]) <= 400,
                     "GRAPH_SOURCE_REASON_INVALID", node_id)
        by_id[node_id] = node
    _require(sum(n["kind"] == "house" for n in nodes) == 1, "GRAPH_SHAPE_INVALID", "one house required")
    rooms = {key for key, node in by_id.items() if node["kind"] == "room"}
    _require(rooms, "GRAPH_SHAPE_INVALID", "no rooms")
    robots = {key for key, node in by_id.items() if node["kind"] == "robot"}
    _require(len(robots) == 1, "GRAPH_SHAPE_INVALID", "one robot required")
    portals: dict[str, tuple[str, str]] = {}
    for node_id, node in by_id.items():
        kind = node["kind"]
        if kind == "portal":
            ends = node.get("room_ids")
            _require(isinstance(ends, list) and len(ends) == 2
                     and all(isinstance(end, str) for end in ends) and len(set(ends)) == 2
                     and all(end in rooms or end == "outside" for end in ends)
                     and ends.count("outside") <= 1, "PORTAL_ENDPOINT_INVALID", node_id)
            _require(node.get("initial_state") in {"open", "closed"},
                     "PORTAL_STATE_INVALID", node_id)
            clearance = node.get("requested_clearance_m")
            _require(type(clearance) in (float, int) and clearance > 0,
                     "PORTAL_CLEARANCE_INVALID", node_id)
            portals[node_id] = (ends[0], ends[1])
        if kind in {"workspace", "zone", "asset_role", "asset_instance", "floor",
                    "wall_set", "surface", "distractor_policy"}:
            _require(isinstance(node.get("room_id"), str) and node["room_id"] in rooms,
                     "GRAPH_DANGLING_REF", node_id + ".room_id")
        if kind == "asset_role":
            _name(node.get("category"), node_id + ".category")
            _require(isinstance(node.get("required_capabilities"), list)
                     and bool(node["required_capabilities"])
                     and all(isinstance(c, str) and c for c in node["required_capabilities"]),
                     "ASSET_CAPABILITY_MISSING", node_id)
        if kind == "robot":
            _require(isinstance(node.get("mobility"), str)
                     and node["mobility"] in {"fixed_arm", "mobile_manipulator"},
                     "ROBOT_CAPABILITY_INCOMPLETE", node_id)

    edges = _items(raw["edges"], "edges")
    edge_ids: set[str] = set()
    connections: dict[str, set[str]] = {key: set() for key in portals}
    for edge in edges:
        edge_id = _name(edge.get("id"), "edge.id")
        _require(set(edge) == EDGE_FIELDS, "GRAPH_SHAPE_INVALID", edge_id + ".fields")
        _require(edge_id not in edge_ids and edge_id not in by_id,
                 "GRAPH_DUPLICATE_ID", edge_id)
        edge_ids.add(edge_id)
        kind = edge.get("kind")
        _require(isinstance(kind, str) and kind in EDGE_KINDS,
                 "GRAPH_EDGE_KIND_UNSUPPORTED", edge_id)
        a, b = edge.get("source_id"), edge.get("target_id")
        _require(isinstance(a, str) and isinstance(b, str) and a in by_id and b in by_id,
                 "GRAPH_DANGLING_REF", edge_id)
        if kind == "connects":
            _require(a in portals and b in rooms and b in portals[a],
                     "PORTAL_ENDPOINT_INVALID", edge_id)
            connections[a].add(b)
    for portal_id, ends in portals.items():
        _require(connections[portal_id] == set(ends) - {"outside"},
                 "PORTAL_ENDPOINT_INVALID", portal_id + ".edges")

    ids = set(by_id)
    initial = {_predicate(p, ids, "initial_state") for p in _items(raw["initial_state"], "initial_state")}
    goals = {_predicate(p, ids, "goals") for p in _items(raw["goals"], "goals")}
    _require(goals, "TASK_GOAL_MISSING", raw["graph_id"])
    _require(not goals & initial, "TASK_GOAL_ALREADY_TRUE", raw["graph_id"])
    _location_conflict(initial)

    steps = _items(raw["steps"], "steps")
    _require(steps, "TASK_ACTION_MISSING", raw["graph_id"])
    step_ids: set[str] = set()
    state = set(initial)
    robot_id = next(iter(robots))
    for step in steps:
        sid = _name(step.get("id"), "step.id")
        _require(STEP_FIELDS <= set(step) <= STEP_FIELDS | STEP_PROVENANCE_FIELDS,
                 "GRAPH_SHAPE_INVALID", sid + ".fields")
        if "source" in step:
            _require(step["source"] in {"user", "inferred"}, "GRAPH_SOURCE_MISSING", sid)
        if "source_reason" in step:
            _require(isinstance(step["source_reason"], str) and bool(step["source_reason"].strip())
                     and len(step["source_reason"]) <= 400,
                     "GRAPH_SOURCE_REASON_INVALID", sid)
        _require(sid not in step_ids and sid not in ids, "GRAPH_DUPLICATE_ID", sid)
        step_ids.add(sid)
        verb = step.get("verb")
        _require(isinstance(verb, str) and verb in VERBS,
                 "TASK_ACTION_UNSUPPORTED", str(verb))
        _require(step.get("actor") == robot_id, "GRAPH_DANGLING_REF", sid + ".actor")
        targets = step.get("target_ids")
        _require(isinstance(targets, list) and all(isinstance(t, str) and t in ids for t in targets),
                 "GRAPH_DANGLING_REF", sid + ".target_ids")
        workspace_id = step.get("workspace_id")
        _require(isinstance(workspace_id, str) and workspace_id in by_id
                 and by_id[workspace_id]["kind"] == "workspace",
                 "GRAPH_DANGLING_REF", sid + ".workspace_id")
        routes = step.get("route_portal_ids")
        _require(isinstance(routes, list) and all(isinstance(p, str) and p in portals for p in routes),
                 "GRAPH_DANGLING_REF", sid + ".route_portal_ids")
        before = {_predicate(p, ids, sid + ".preconditions")
                  for p in _items(step.get("preconditions"), sid + ".preconditions")}
        _require(before <= state, "TASK_PRECONDITION_UNSATISFIED", sid)
        remove = {_predicate(p, ids, sid + ".effects_remove")
                  for p in _items(step.get("effects_remove"), sid + ".effects_remove")}
        add = {_predicate(p, ids, sid + ".effects_add")
               for p in _items(step.get("effects_add"), sid + ".effects_add")}
        _require(remove <= state, "TASK_EFFECT_INVALID", sid)
        if verb == "NAVIGATE":
            _require(by_id[robot_id]["mobility"] == "mobile_manipulator",
                     "ROBOT_MOBILITY_INCOMPATIBLE", sid)
            start = [obj for sub, pred, obj in state if sub == robot_id and pred == "IN_ROOM"]
            end = [obj for sub, pred, obj in add if sub == robot_id and pred == "IN_ROOM"]
            _require(len(start) == 1 and len(end) == 1 and start[0] != end[0],
                     "TASK_ROUTE_DISCONNECTED", sid)
            place = start[0]
            for portal_id in routes:
                ends = portals[portal_id]
                _require(place in ends and by_id[portal_id]["initial_state"] == "open",
                         "TASK_ROUTE_DISCONNECTED", sid)
                place = ends[1] if ends[0] == place else ends[0]
            _require(place == end[0] and bool(routes), "TASK_ROUTE_DISCONNECTED", sid)
            _require(all(subject == robot_id and relation == "IN_ROOM"
                         for subject, relation, _ in add | remove),
                     "TASK_EFFECT_INVALID", sid)
        else:
            _require(not routes, "TASK_ROUTE_DISCONNECTED", sid)
            current = [obj for sub, pred, obj in state if sub == robot_id and pred == "IN_ROOM"]
            _require(len(current) == 1 and by_id[workspace_id]["room_id"] == current[0],
                     "TASK_WORKSPACE_WRONG_ROOM", sid)
            _require(len(targets) == 1, "TASK_EFFECT_INVALID", sid + ".target")
            target = targets[0]
            if verb == "PICK":
                _require((target, "HELD_BY", robot_id) in add
                         and any(s == target and rel in {"ON", "INSIDE"} for s, rel, _ in remove),
                         "TASK_EFFECT_INVALID", sid)
            if verb == "PLACE":
                _require((target, "HELD_BY", robot_id) in remove
                         and any(s == target and rel in {"ON", "INSIDE"} for s, rel, _ in add),
                         "TASK_EFFECT_INVALID", sid)
        state.difference_update(remove)
        state.update(add)
        _location_conflict(state)
    _require(goals <= state, "TASK_GOAL_UNSATISFIED", raw["graph_id"])

    requirement_ids: set[str] = set()
    for req in _items(raw["requirements"], "requirements"):
        rid = _name(req.get("id"), "requirement.id")
        _require(set(req) == REQUIREMENT_FIELDS, "GRAPH_SHAPE_INVALID", rid + ".fields")
        _require(rid not in requirement_ids and rid not in ids and rid not in step_ids,
                 "GRAPH_DUPLICATE_ID", rid)
        requirement_ids.add(rid)
        _name(req.get("kind"), rid + ".kind")
        subjects = req.get("subject_ids")
        _require(isinstance(subjects, list) and bool(subjects)
                 and all(isinstance(s, str) and s in ids for s in subjects),
                 "GRAPH_DANGLING_REF", rid + ".subject_ids")
        _require(req.get("status") == "not_run", "GRAPH_UNVERIFIED_CLAIM", rid)

    _validate_composition(raw, by_id, edges)

    if 'surface_appearance' in raw['built']:
        from rfw_w.scenes.contracts.surface_appearance import validate_graph_appearance, SurfaceAppearanceError
        try:
            validate_graph_appearance(raw['built']['surface_appearance'], by_id)
        except SurfaceAppearanceError as error:
            raise GraphError(error.code, error.detail) from error

    supplied = raw.get("content_hash")
    actual = content_hash({k: v for k, v in raw.items() if k not in {"content_hash", "evidence_refs"}})
    _require(supplied is None or supplied == actual, "GRAPH_HASH_MISMATCH", raw["graph_id"])


@dataclass(frozen=True, slots=True)
class TaskWorldGraph:
    _data: dict[str, Any]
    content_hash: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskWorldGraph":
        _require(isinstance(payload, Mapping), "GRAPH_SHAPE_INVALID", "graph")
        raw = deepcopy(dict(payload))
        _validate(raw)
        actual = content_hash({k: v for k, v in raw.items() if k not in {"content_hash", "evidence_refs"}})
        return cls(raw, actual)

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(self._data)
        result["content_hash"] = self.content_hash
        return result

    def state_after(self, step_id: str) -> set[tuple[str, str, str]]:
        state = {(p["subject"], p["predicate"], p["object"])
                 for p in self._data["initial_state"]}
        for step in self._data["steps"]:
            state -= {(p["subject"], p["predicate"], p["object"])
                      for p in step["effects_remove"]}
            state |= {(p["subject"], p["predicate"], p["object"])
                      for p in step["effects_add"]}
            if step["id"] == step_id:
                return state
        raise GraphError("GRAPH_DANGLING_REF", step_id)

    def room_view(self, room_id: str) -> dict[str, Any]:
        rooms = {n["id"] for n in self._data["nodes"] if n["kind"] == "room"}
        _require(room_id in rooms, "GRAPH_DANGLING_REF", room_id)
        ids = [n["id"] for n in self._data["nodes"]
               if n["id"] == room_id or n.get("room_id") == room_id
               or (n["kind"] == "portal" and room_id in n["room_ids"])]
        return {"room_id": room_id, "graph_hash": self.content_hash, "node_ids": ids}

    def task_view(self) -> dict[str, Any]:
        return {"graph_hash": self.content_hash, "initial_state": deepcopy(self._data["initial_state"]),
                "steps": deepcopy(self._data["steps"]), "goals": deepcopy(self._data["goals"])}

    def check_evidence(self, evidence: Mapping[str, Any]) -> None:
        _require(evidence.get("graph_hash") == self.content_hash,
                 "GRAPH_STALE_EVIDENCE", str(evidence.get("graph_hash")))
        ids = {n["id"] for n in self._data["nodes"]}
        _require(all(s in ids for s in evidence.get("subject_ids", [])),
                 "GRAPH_DANGLING_REF", "evidence.subject_ids")

    def revise(self, payload: Mapping[str, Any], *, expected_hash: str) -> "TaskWorldGraph":
        _require(expected_hash == self.content_hash, "GRAPH_VERSION_CONFLICT",
                 f"expected={expected_hash}, actual={self.content_hash}")
        candidate = deepcopy(dict(payload))
        candidate["revision"] = self._data["revision"] + 1
        candidate["parent_hash"] = self.content_hash
        candidate.pop("content_hash", None)
        return TaskWorldGraph.from_dict(candidate)
