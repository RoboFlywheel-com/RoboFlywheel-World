# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.assets.room_tools import RoomAssetError, asset_config, verify_scene_assembly
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.solvers.house_tools import solve_house_layout

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/task_world"


def graph():
    request = json.loads((FIXTURES / "cross_room_request.json").read_text())
    slots = json.loads((FIXTURES / "cross_room_slots_response.json").read_text())["slots"]
    return compile_pick_place_slots(slots, request)


def _rehash(value):
    value["content_hash"] = content_hash({k:v for k,v in value.items() if k != "content_hash"})
    return value


def fixture_ledger(tmp_path):
    source = graph()
    plan = solve_house_layout(source)
    metadata = {
        "counter": ("kitchen", [-2.0, 0.8, 0.0], [1.0, 0.6, 0.9], 0.9),
        "desk": ("study", [2.0, 0.8, 0.0], [1.0, 0.6, 0.75], 0.75),
        "cup": ("kitchen", [-2.0, 0.8, 0.902], [0.1, 0.1, 0.1], None),
    }
    instances, queries, selection = [], {}, {}
    for role, (room, pose, dims, height) in metadata.items():
        asset_id = role + "_fixture"
        directory = tmp_path / "prepared" / asset_id
        directory.mkdir(parents=True)
        (directory / "visual.obj").write_bytes(b"v 0 0 0\n")
        (directory / "collision.obj").write_bytes(b"v 0 0 0\n")
        surface = None
        polygons = []
        if height is not None:
            local = [[-0.5,-0.3],[0.5,-0.3],[0.5,0.3],[-0.5,0.3]]
            polygons = [{"vertices_xy": local, "plane_z_m": height}]
            surface = {"polygon_xy": [[x+pose[0],y+pose[1]] for x,y in local],
                       "plane_z_m": height}
        sidecar = {"asset": {"asset_id": asset_id, "provider": "hssd"},
                   "dimensions_m": dims, "visual_obj": "visual.obj",
                   "collision_obj": "collision.obj", "support_polygons": polygons}
        (directory / "prepared.json").write_text(json.dumps(sidecar))
        item = {"instance_id": "inst_"+role, "role_id": role, "room_id": room,
                "priority": "core", "category": role, "asset_id": asset_id,
                "provider": "hssd", "dimensions_m": dims, "pose_xyz_m": pose,
                "visual_ref": f"prepared/{asset_id}/visual.obj",
                "collision_ref": f"prepared/{asset_id}/collision.obj",
                "visual_sha256": hashlib.sha256(b"v 0 0 0\n").hexdigest(),
                "collision_sha256": hashlib.sha256(b"v 0 0 0\n").hexdigest(),
                "support_role_id": "counter" if role == "cup" else None,
                "support_surface": surface}
        instances.append(item)
        queries[role] = {"candidates": [{"asset_id": asset_id}]}
        selection[role] = {"status": "selected", "asset_id": asset_id}
    optional = [n["id"] for n in source.to_dict()["nodes"]
                if n["kind"] == "asset_role" and n["priority"] == "optional"]
    ledger = {"source_graph_hash": source.content_hash, "house_plan_hash": plan["content_hash"],
              "instances": instances, "queries": queries, "selection": selection,
              "unplaced_optional": [{"role_id": rid, "code": "OPTIONAL_DROPPED"} for rid in optional],
              "goal_placement": {"status": "candidate_not_executed", "target_role_id": "cup",
                                 "destination_support_role_id": "desk",
                                 "candidate_pose_xyz_m": [2.0,0.8,0.752]}}
    return source, plan, _rehash(ledger)


def test_static_assembly_accepts_explicit_optional_drop(tmp_path):
    source, plan, ledger = fixture_ledger(tmp_path)
    report = verify_scene_assembly(source, plan, ledger, tmp_path)
    assert report["core_roles_bound"] == 3
    assert report["optional_placed"] == 0 and report["optional_unplaced"] == 2
    assert report["robot_ik"] == "not_run"


@pytest.mark.parametrize("mutation,code", [
    ("missing_core", "ASSET_UNAVAILABLE"),
    ("blocked_corridor", "CLEARANCE_VIOLATION"),
    ("missing_support", "SUPPORT_NOT_FOUND"),
    ("bad_goal", "SUPPORT_NOT_FOUND"),
    ("stale", "GRAPH_VERSION_CONFLICT"),
])
def test_static_gate_rejects_named_fault(tmp_path, mutation, code):
    source, plan, ledger = fixture_ledger(tmp_path)
    changed = deepcopy(ledger)
    if mutation == "missing_core":
        changed["instances"] = [i for i in changed["instances"] if i["role_id"] != "cup"]
    elif mutation == "blocked_corridor":
        changed["instances"][0]["pose_xyz_m"][1] = 2.0
    elif mutation == "missing_support":
        changed["instances"][0]["support_surface"] = None
    elif mutation == "bad_goal":
        changed["goal_placement"]["candidate_pose_xyz_m"][0] = 0.0
    else:
        changed["source_graph_hash"] = "0"*64
    with pytest.raises(RoomAssetError, match=code):
        verify_scene_assembly(source, plan, _rehash(changed), tmp_path)


def test_asset_roots_must_be_environment_variables(tmp_path, monkeypatch):
    monkeypatch.delenv("PVSCENE_HSSD_ROOT", raising=False)
    monkeypatch.delenv("PVSCENE_HSSD_SYNSET_INDEX", raising=False)
    with pytest.raises(RoomAssetError, match="ASSET_ENV_MISSING"):
        asset_config(tmp_path)
