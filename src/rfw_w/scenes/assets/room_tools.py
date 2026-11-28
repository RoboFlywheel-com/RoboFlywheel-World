# SPDX-License-Identifier: Apache-2.0
"""Task-role asset retrieval and static placement against a built two-room house.

All absolute library paths stay inside the process and gitignored prepared artifacts.
The graph and the public ledger carry opaque asset IDs and relative prepared mesh refs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from rfw_w.scenes.assets.providers.hssd import (
    HssdConfig, HssdError, load_synset_index, measure_glb_extents,
    prepare_furniture, read_support_surfaces,
)
from rfw_w.scenes.contracts.asset import AffordanceStatus
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.geometry.polygon import Polygon, contains_polygon, overlaps
from rfw_w.scenes.solvers.layout import (
    Footprint, OccupancyMap, SurfacePlacementConfig, place_in_zone, place_on_surface,
)
from rfw_w.scenes.solvers.support import Surface

CATEGORY_SYNSETS: Mapping[str, tuple[str, ...]] = {
    # HSSD's countertop synset has only thin slabs with no annotated support plane.
    # A kitchen island is a complete counter-height support, and the mapping is recorded.
    "counter": ("kitchen_island.n.01",),
    "desk": ("desk.n.01",),
    "cup": ("cup.n.01",),
    "plate": ("plate.n.04",),
    "bowl": ("bowl.n.01",),
    "book": ("book.n.02",),
    "notebook": ("notebook.n.01",),
    "table": ("desk.n.01",),
    "kitchen_island": ("kitchen_island.n.01",),
}
PREFERRED_WIDTH_M = {"counter": 1.5, "desk": 1.4, "cup": 0.11,
                     "plate": 0.22, "bowl": 0.2, "book": 0.22, "notebook": 0.22}


class RoomAssetError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}({detail})")


def asset_config(prepared_root: Path) -> HssdConfig:
    root = os.environ.get("PVSCENE_HSSD_ROOT")
    index = os.environ.get("PVSCENE_HSSD_SYNSET_INDEX")
    if not root or not index:
        raise RoomAssetError("ASSET_ENV_MISSING", "PVSCENE_HSSD_ROOT or PVSCENE_HSSD_SYNSET_INDEX")
    try:
        return HssdConfig(Path(root), Path(index), prepared_root)
    except HssdError as error:
        raise RoomAssetError("ASSET_ENV_MISSING", "HSSD files unavailable") from error


def retrieve_role_candidates(graph: TaskWorldGraph, config: HssdConfig,
                             *, max_candidates: int = 8) -> dict[str, Any]:
    """Return candidates and rejected pool separately, with no local paths."""
    index = load_synset_index(config)
    query: dict[str, Any] = {}
    for role in (n for n in graph.to_dict()["nodes"] if n["kind"] == "asset_role"):
        if role.get("purpose") in {"background", "distraction"} and role.get("placement_kind") in {"floor", "wall"}:
            continue
        category = role["category"]
        synsets = CATEGORY_SYNSETS.get(category, ())
        rows = [row for synset in synsets for row in index.get(synset, [])]
        admitted, rejected = [], []
        for row in rows:
            aid = str(row.get("id", ""))
            if not aid:
                continue
            path = config.glb_path(aid)
            extent = measure_glb_extents(path) if path.is_file() else None
            if extent is None:
                rejected.append({"asset_id": aid, "code": "ASSET_UNAVAILABLE"})
                continue
            dims = extent.to_list()
            furniture = role.get("physical_class") == "furniture"
            limits = (2.5, 1.6, 1.7) if furniture else (0.45, 0.45, 0.35)
            if any(d <= 0 or d > cap for d, cap in zip(dims, limits)):
                rejected.append({"asset_id": aid, "code": "SIZE_UNSUPPORTED"})
                continue
            if furniture and not read_support_surfaces(config, aid):
                rejected.append({"asset_id": aid, "code": "SUPPORT_NOT_FOUND"})
                continue
            target_width = PREFERRED_WIDTH_M.get(category, 0.2)
            score = abs(math.log(max(dims[0], 1e-4) / target_width))
            admitted.append((score, aid, dims))
        admitted.sort(key=lambda item: (item[0], item[1]))
        query[role["id"]] = {
            "role_id": role["id"], "category": category, "priority": role.get("priority", "core"),
            "synsets": list(synsets), "category_mapping": "semantic_alias" if category == "counter" else "direct",
            "candidates": [{"asset_id": aid, "dimensions_m": dims}
                           for _, aid, dims in admitted[:max_candidates]],
            "admitted_total": len(admitted), "rejected": rejected,
            "unexamined_admitted": max(0, len(admitted) - max_candidates),
        }
    return query


def prepare_asset_instances(query: Mapping[str, Any], config: HssdConfig,
                            graph: TaskWorldGraph) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare first usable candidate per role; optional misses remain explicit."""
    roles = {n["id"]: n for n in graph.to_dict()["nodes"] if n["kind"] == "asset_role"}
    prepared: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for rid, entry in query.items():
        role = roles[rid]
        attempts = []
        for candidate in entry["candidates"]:
            aid = candidate["asset_id"]
            try:
                value = prepare_furniture(config, aid)
            except (HssdError, OSError, ValueError) as error:
                attempts.append({"asset_id": aid, "code": "ASSET_PREPARE_FAILED",
                                 "error_type": type(error).__name__})
                continue
            if role["physical_class"] == "furniture":
                usable = [s for s in value.support_polygons
                          if s.plane_z_m > value.dimensions_m.z * 0.5
                          and Polygon.from_sequence(s.vertices_xy).area >= 0.04]
                if not usable:
                    attempts.append({"asset_id": aid, "code": "SUPPORT_NOT_FOUND"})
                    continue
            prepared[rid] = value
            selection[rid] = {"asset_id": aid, "status": "selected", "attempts": attempts}
            break
        if rid not in prepared:
            selection[rid] = {"status": "unplaced" if role["priority"] == "optional" else "failed",
                              "attempts": attempts, "code": "ASSET_UNAVAILABLE"}
            if role["priority"] == "core":
                raise RoomAssetError("ASSET_UNAVAILABLE", rid)
    return prepared, selection


def _room_zone(room: Mapping[str, Any]) -> Polygon:
    x0, _, x1, _ = room["bounds_xy_m"]
    return Polygon.rectangle(x0 + 0.35, 0.25, x1 - 0.35, 1.35)


def _support_surface(role_id: str, prepared, pose: list[float]) -> Surface:
    surfaces = [s for s in prepared.support_polygons
                if s.plane_z_m > prepared.dimensions_m.z * 0.5
                and Polygon.from_sequence(s.vertices_xy).area >= 0.04]
    if not surfaces:
        raise RoomAssetError("SUPPORT_NOT_FOUND", role_id)
    top = max(surfaces, key=lambda s: (Polygon.from_sequence(s.vertices_xy).area, s.plane_z_m))
    polygon = Polygon.from_sequence([(x + pose[0], y + pose[1]) for x, y in top.vertices_xy])
    return Surface(surface_id=role_id + ":top", owner_instance="inst_" + role_id,
                   polygon=polygon, plane_z_m=top.plane_z_m + pose[2],
                   provenance=top.provenance.status)


def _instance(role: Mapping[str, Any], prepared, pose: list[float],
              *, support_role_id: str | None = None, surface: Surface | None = None,
              output_root: Path) -> dict[str, Any]:
    visual = prepared.visual_obj.relative_to(output_root).as_posix()
    collision = prepared.collision_obj.relative_to(output_root).as_posix()
    return {"instance_id": "inst_" + role["id"], "role_id": role["id"],
            "room_id": role["room_id"], "priority": role["priority"],
            "category": role["category"], "asset_id": prepared.asset_key.asset_id,
            "provider": prepared.asset_key.provider,
            "library_version": prepared.asset_key.library_version,
            "dimensions_m": prepared.dimensions_m.to_list(),
            "pose_xyz_m": [round(v, 6) for v in pose], "yaw_rad": 0.0,
            "origin_convention": prepared.origin_convention,
            "visual_ref": visual, "collision_ref": collision,
            "visual_sha256": hashlib.sha256(prepared.visual_obj.read_bytes()).hexdigest(),
            "collision_sha256": hashlib.sha256(prepared.collision_obj.read_bytes()).hexdigest(),
            "support_role_id": support_role_id,
            "support_surface": None if surface is None else surface.to_dict()}


def place_core_furniture(graph: TaskWorldGraph, house_plan: Mapping[str, Any],
                         prepared: Mapping[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Surface]]:
    roles = {n["id"]: n for n in graph.to_dict()["nodes"] if n["kind"] == "asset_role"}
    rooms = {room["id"]: room for room in house_plan["rooms"]}
    corridor = Polygon.rectangle(*_reorder_bounds(house_plan["corridor"]))
    instances = []
    surfaces = {}
    occupancy = OccupancyMap()
    occupancy.add("floor_keepout:corridor", corridor)
    for role in (r for r in roles.values() if r["priority"] == "core" and r["physical_class"] == "furniture"):
        value = prepared[role["id"]]
        dims = value.dimensions_m
        outcome = place_in_zone(
            instance_id="inst_" + role["id"], entity_id=role["id"],
            footprint=Footprint((dims.x/2, dims.y/2)), zone=_room_zone(rooms[role["room_id"]]),
            object_height_m=dims.z, occupancy=occupancy,
            config=SurfacePlacementConfig(step_m=0.05, object_gap_m=0.08, max_candidates=1000),
            origin_convention="base")
        if not outcome.placed:
            raise RoomAssetError("NO_PLACEMENT_CAPACITY", role["id"])
        pose = outcome.placement.position_m.to_list()
        footprint = Footprint((dims.x/2, dims.y/2)).at(pose[0], pose[1])
        if overlaps(footprint, corridor):
            raise RoomAssetError("CLEARANCE_VIOLATION", role["id"])
        occupancy.add(role["id"], footprint)
        surface = _support_surface(role["id"], value, pose)
        surfaces[role["id"]] = surface
        instances.append(_instance(role, value, pose, surface=surface, output_root=output_root))
    return instances, surfaces


def _reorder_bounds(corridor: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (corridor["x_span_m"][0], corridor["y_span_m"][0],
            corridor["x_span_m"][1], corridor["y_span_m"][1])


def _place_on(role: Mapping[str, Any], prepared, surface: Surface, occupancy: OccupancyMap,
              output_root: Path, *, support_role_id: str) -> dict[str, Any]:
    dims = prepared.dimensions_m
    outcome = place_on_surface(
        instance_id="inst_" + role["id"], entity_id=role["id"],
        footprint=Footprint((dims.x/2, dims.y/2)), surface=surface,
        object_height_m=dims.z, occupancy=occupancy,
        config=SurfacePlacementConfig(step_m=0.03, edge_margin_m=0.04,
                                      object_gap_m=0.06, max_candidates=1000),
        origin_convention="base")
    if not outcome.placed:
        raise RoomAssetError("NO_PLACEMENT_CAPACITY", role["id"])
    pose = outcome.placement.position_m.to_list()
    occupancy.add(role["id"], Footprint((dims.x/2,dims.y/2)).at(pose[0],pose[1]))
    return _instance(role, prepared, pose, support_role_id=support_role_id,
                     output_root=output_root)


def place_task_objects(graph: TaskWorldGraph, prepared: Mapping[str, Any],
                       surfaces: Mapping[str, Surface], output_root: Path) -> tuple[list[dict[str, Any]], dict[str, OccupancyMap], dict[str, Any]]:
    roles = {n["id"]: n for n in graph.to_dict()["nodes"] if n["kind"] == "asset_role"}
    targets = [r for r in roles.values() if r.get("task_role") == "target"]
    if len(targets) != 1:
        raise RoomAssetError("ASSET_UNAVAILABLE", "one task target required")
    source = next(p["object"] for p in graph.to_dict()["initial_state"]
                  if p["subject"] == targets[0]["id"] and p["predicate"] == "ON")
    destination = next(p["object"] for p in graph.to_dict()["goals"]
                       if p["subject"] == targets[0]["id"] and p["predicate"] == "ON")
    occupancy = {room["id"]: OccupancyMap() for room in graph.to_dict()["nodes"] if room["kind"] == "room"}
    item = _place_on(targets[0], prepared[targets[0]["id"]], surfaces[source],
                     occupancy[targets[0]["room_id"]], output_root, support_role_id=source)
    # A goal pose is a geometric candidate. The target remains a single initial instance.
    goal_map = OccupancyMap()
    goal_outcome = place_on_surface(
        instance_id="goal_" + targets[0]["id"], entity_id=targets[0]["id"],
        footprint=Footprint((prepared[targets[0]["id"]].dimensions_m.x/2,
                             prepared[targets[0]["id"]].dimensions_m.y/2)),
        surface=surfaces[destination],
        object_height_m=prepared[targets[0]["id"]].dimensions_m.z,
        occupancy=goal_map, config=SurfacePlacementConfig(step_m=0.03, edge_margin_m=0.04,
                                                          max_candidates=1000),
        origin_convention="base")
    if not goal_outcome.placed:
        raise RoomAssetError("NO_PLACEMENT_CAPACITY", destination)
    return [item], occupancy, {"target_role_id": targets[0]["id"],
                               "destination_support_role_id": destination,
                               "candidate_pose_xyz_m": goal_outcome.placement.position_m.to_list(),
                               "status": "candidate_not_executed"}


def populate_optional_distractors(graph: TaskWorldGraph, prepared: Mapping[str, Any],
                                  surfaces: Mapping[str, Surface], occupancy: Mapping[str, OccupancyMap],
                                  output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = graph.to_dict()
    parent = {e["target_id"]: e["source_id"] for e in raw["edges"] if e["kind"] == "contains"}
    roles = {n["id"]: n for n in raw["nodes"] if n["kind"] == "asset_role"}
    placed, unplaced = [], []
    for role in (r for r in roles.values() if r["priority"] == "optional"
                 and r.get("placement_kind", "surface") == "surface"):
        rid = role["id"]
        if rid not in prepared:
            unplaced.append({"role_id": rid, "code": "ASSET_UNAVAILABLE"})
            continue
        surface_node = parent[rid]
        support_role = parent[surface_node]
        try:
            item = _place_on(role, prepared[rid], surfaces[support_role],
                             occupancy[role["room_id"]], output_root,
                             support_role_id=support_role)
            placed.append(item)
        except RoomAssetError as error:
            unplaced.append({"role_id": rid, "code": error.code})
    return placed, unplaced


def build_asset_ledger(graph: TaskWorldGraph, house_plan: Mapping[str, Any],
                       config: HssdConfig, output_root: Path,
                       on_stage: Callable[[str, float, str | None], None] | None = None) -> dict[str, Any]:
    def stage(name: str, fn):
        started = time.monotonic()
        try:
            value = fn()
        except RoomAssetError as error:
            if on_stage is not None:
                on_stage(name, (time.monotonic()-started)*1000, error.code)
            raise
        if on_stage is not None:
            on_stage(name, (time.monotonic()-started)*1000, None)
        return value

    query = stage("retrieve_role_candidates", lambda: retrieve_role_candidates(graph, config))
    prepared, selection = stage("prepare_asset_instances",
                                lambda: prepare_asset_instances(query, config, graph))
    furniture, surfaces = stage("place_core_furniture",
                                lambda: place_core_furniture(graph, house_plan, prepared, output_root))
    task, occupancy, goal = stage("place_task_objects",
                                  lambda: place_task_objects(graph, prepared, surfaces, output_root))
    optional, unplaced = stage("populate_optional_distractors",
                               lambda: populate_optional_distractors(graph, prepared, surfaces,
                                                                     occupancy, output_root))
    ledger = {"schema_version": "pvscene.room-assets/v1", "source_graph_hash": graph.content_hash,
              "house_plan_hash": house_plan["content_hash"],
              "queries": query, "selection": selection,
              "instances": furniture + task + optional,
              "unplaced_optional": unplaced,
              "goal_placement": goal,
              "robot_grasp": "not_run", "robot_ik": "not_run", "robot_rollout": "not_run"}
    ledger["content_hash"] = content_hash(ledger)
    return ledger


def verify_scene_assembly(graph: TaskWorldGraph, house_plan: Mapping[str, Any],
                          ledger: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Recompute static containment/support/clearance from saved ledger and prepared meshes."""
    if ledger.get("source_graph_hash") != graph.content_hash:
        raise RoomAssetError("GRAPH_VERSION_CONFLICT", "asset ledger source")
    if ledger.get("house_plan_hash") != house_plan["content_hash"]:
        raise RoomAssetError("GRAPH_VERSION_CONFLICT", "house plan")
    if content_hash({k:v for k,v in ledger.items() if k != "content_hash"}) != ledger.get("content_hash"):
        raise RoomAssetError("ASSET_LEDGER_MISMATCH", "content hash")
    roles = {n["id"]: n for n in graph.to_dict()["nodes"] if n["kind"] == "asset_role"}
    rooms = {r["id"]: Polygon.rectangle(*r["bounds_xy_m"]) for r in house_plan["rooms"]}
    corridor = Polygon.rectangle(*_reorder_bounds(house_plan["corridor"]))
    found: dict[str, dict[str, Any]] = {}
    floor_footprints: list[tuple[str, Polygon]] = []
    surface_footprints: dict[str, list[tuple[str, Polygon]]] = {}
    for item in ledger["instances"]:
        rid = item["role_id"]
        if rid not in roles or rid in found or item["room_id"] != roles[rid]["room_id"]:
            raise RoomAssetError("ASSET_BINDING_INVALID", rid)
        if item["priority"] != roles[rid]["priority"] or item["instance_id"] != "inst_" + rid:
            raise RoomAssetError("ASSET_BINDING_INVALID", rid)
        if item["provider"] not in {"hssd", "simready"} or not item["asset_id"]:
            raise RoomAssetError("ASSET_UNCERTIFIED", rid)
        query = ledger["queries"].get(rid)
        selected = ledger["selection"].get(rid)
        if (not query or item["asset_id"] not in {c["asset_id"] for c in query["candidates"]}
                or not selected or selected.get("status") != "selected"
                or selected.get("asset_id") != item["asset_id"]):
            raise RoomAssetError("ASSET_BINDING_INVALID", rid)
        if item['provider']=='simready':
            from rfw_w.scenes.geometry.rigid_tool_assets import verify_prepared_rigid
            from rfw_w.scenes.geometry.surface_appearance import checked_path
            metadata_path=checked_path(output_root,item['metadata_ref'])
            expected_meta=f"prepared/simready/{item['library_version']}/{item['asset_id']}/prepared.json"
            if item['metadata_ref']!=expected_meta:raise RoomAssetError('ASSET_LEDGER_MISMATCH',rid)
            metadata=json.loads(metadata_path.read_text())
            verify_prepared_rigid(output_root,metadata)
            if any(metadata.get(k)!=item.get(k) for k in ('provider','asset_id','library_version','dimensions_m','origin_convention','visual_ref','collision_ref','visual_sha256','collision_sha256')):
                raise RoomAssetError('ASSET_LEDGER_MISMATCH',rid)
            if roles[rid]['physical_class']!='manipuland':raise RoomAssetError('ASSET_UNCERTIFIED',rid)
        else:
            metadata_path = output_root / "prepared" / item["asset_id"] / "prepared.json"
            if not metadata_path.is_file():
                raise RoomAssetError("ASSET_UNAVAILABLE", rid)
            metadata = json.loads(metadata_path.read_text())
            if (metadata["asset"]["asset_id"] != item["asset_id"]
                    or metadata["asset"]["provider"] != item["provider"]
                    or metadata["dimensions_m"] != item["dimensions_m"]):
                raise RoomAssetError("ASSET_LEDGER_MISMATCH", rid)
            for kind in ("visual", "collision"):
                ref = item[kind + "_ref"]
                path = output_root / ref
                if Path(ref).is_absolute() or ".." in Path(ref).parts or not path.is_file():
                    raise RoomAssetError("ASSET_UNAVAILABLE", rid)
                if Path(ref).as_posix() != f"prepared/{item['asset_id']}/{metadata[kind + '_obj']}":
                    raise RoomAssetError("ASSET_LEDGER_MISMATCH", rid)
                if hashlib.sha256(path.read_bytes()).hexdigest() != item[kind + "_sha256"]:
                    raise RoomAssetError("ASSET_LEDGER_MISMATCH", rid)
        dims, pose = item["dimensions_m"], item["pose_xyz_m"]
        if any(not math.isfinite(v) or v <= 0 for v in dims):
            raise RoomAssetError("ASSET_LEDGER_MISMATCH", rid)
        yaw = item.get('yaw_rad', 0)
        if not math.isfinite(yaw):
            raise RoomAssetError('ASSET_LEDGER_MISMATCH', 'yaw ' + rid)
        if yaw:
            from rfw_w.scenes.geometry.layout_revision import footprint as rotated_footprint
            footprint = rotated_footprint(item)
        else:
            footprint = Footprint((dims[0]/2,dims[1]/2)).at(pose[0],pose[1])
        if not contains_polygon(rooms[item["room_id"]], footprint):
            raise RoomAssetError("GEOMETRY_OUT_OF_BOUNDS", rid)
        if roles[rid]["physical_class"] == "furniture":
            if overlaps(footprint, corridor):
                raise RoomAssetError("CLEARANCE_VIOLATION", rid)
            if abs(pose[2]) > 1e-5:
                raise RoomAssetError("ASSET_LEDGER_MISMATCH", rid)
            for other_id, other in floor_footprints:
                if overlaps(footprint, other):
                    raise RoomAssetError("GEOMETRY_PENETRATION", rid + ":" + other_id)
            floor_footprints.append((rid, footprint))
            if not item.get("support_surface"):
                raise RoomAssetError("SUPPORT_NOT_FOUND", rid)
            support = item["support_surface"]
            matches = False
            for candidate in metadata["support_polygons"]:
                shifted = [[round(math.cos(yaw)*x-math.sin(yaw)*y + pose[0], 6),
                            round(math.sin(yaw)*x+math.cos(yaw)*y + pose[1], 6)]
                           for x,y in candidate["vertices_xy"]]
                if (all(abs(a-b) <= 1e-5 for point, wanted in zip(support["polygon_xy"], shifted)
                        for a,b in zip(point,wanted))
                        and abs(support["plane_z_m"] - candidate["plane_z_m"] - pose[2]) <= 1e-5):
                    matches = True
                    break
            if not matches:
                raise RoomAssetError("SUPPORT_NOT_FOUND", rid)
        else:
            support_id = item["support_role_id"]
            supporting = next((i for i in ledger["instances"] if i["role_id"] == support_id), None)
            if supporting is None or not supporting.get("support_surface"):
                raise RoomAssetError("SUPPORT_NOT_FOUND", rid)
            support = supporting["support_surface"]
            polygon = Polygon.from_sequence(support["polygon_xy"])
            if not contains_polygon(polygon, footprint) or abs(pose[2] - support["plane_z_m"] - 0.002) > 1e-5:
                raise RoomAssetError("SUPPORT_NOT_FOUND", rid)
            for other_id, other in surface_footprints.get(support_id, []):
                if overlaps(footprint, other):
                    raise RoomAssetError("GEOMETRY_PENETRATION", rid + ":" + other_id)
            surface_footprints.setdefault(support_id, []).append((rid, footprint))
        found[rid] = item
    core = {rid for rid, role in roles.items() if role["priority"] == "core"}
    if not core <= set(found):
        raise RoomAssetError("ASSET_UNAVAILABLE", ",".join(sorted(core-set(found))))
    target = [item for item in found.values() if roles[item["role_id"]].get("task_role") == "target"]
    if len(target) != 1:
        raise RoomAssetError("ASSET_BINDING_INVALID", "target")
    source_support = next(p["object"] for p in graph.to_dict()["initial_state"]
                          if p["subject"] == target[0]["role_id"] and p["predicate"] == "ON")
    destination_support = next(p["object"] for p in graph.to_dict()["goals"]
                               if p["subject"] == target[0]["role_id"] and p["predicate"] == "ON")
    if target[0]["support_role_id"] != source_support:
        raise RoomAssetError("SUPPORT_NOT_FOUND", target[0]["role_id"])
    goal = ledger["goal_placement"]
    if (goal["status"] != "candidate_not_executed"
            or goal["target_role_id"] != target[0]["role_id"]
            or goal["destination_support_role_id"] != destination_support):
        raise RoomAssetError("ASSET_LEDGER_MISMATCH", "goal status")
    destination = found[destination_support]["support_surface"]
    gpose = goal["candidate_pose_xyz_m"]
    gdims = target[0]["dimensions_m"]
    from rfw_w.scenes.geometry.layout_revision import footprint as goal_footprint
    if (not contains_polygon(Polygon.from_sequence(destination["polygon_xy"]),
                             goal_footprint({**target[0], 'pose_xyz_m': gpose}))
            or abs(gpose[2] - destination["plane_z_m"] - 0.002) > 1e-5):
        raise RoomAssetError("SUPPORT_NOT_FOUND", "goal")
    optional = {rid for rid, role in roles.items() if role["priority"] == "optional"
                and role.get("placement_kind", "surface") == "surface"}
    unplaced = [row["role_id"] for row in ledger["unplaced_optional"]]
    if (set(unplaced) & set(found)
            or ((set(found) & optional) | set(unplaced)) != optional):
        raise RoomAssetError("ASSET_BINDING_INVALID", "optional ledger")
    parent = {e["target_id"]: e["source_id"] for e in graph.to_dict()["edges"]
              if e["kind"] == "contains"}
    nodes = {n["id"]: n for n in graph.to_dict()["nodes"]}
    for rid in optional:
        surface_id = parent[rid]
        policies = [n for n in nodes.values()
                    if n["kind"] == "distractor_policy" and parent[n["id"]] == surface_id]
        if len(policies) != 1 or roles[rid]["category"] not in policies[0]["candidate_categories"]:
            raise RoomAssetError("ASSET_BINDING_INVALID", rid)
        placed_on_surface = [item for item in found.values()
                             if item["role_id"] in optional and parent[item["role_id"]] == surface_id]
        if len(placed_on_surface) > policies[0]["max_count"]:
            raise RoomAssetError("ASSET_BINDING_INVALID", policies[0]["id"])
    return {"status": "passed", "core_roles_bound": len(core),
            "optional_placed": len(found)-len(core), "optional_unplaced": len(ledger["unplaced_optional"]),
            "instances": len(found), "target_instances": len(target),
            "robot_grasp": "not_run", "robot_ik": "not_run", "robot_rollout": "not_run"}
