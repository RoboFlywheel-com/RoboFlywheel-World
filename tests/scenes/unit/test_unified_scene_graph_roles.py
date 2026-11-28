# SPDX-License-Identifier: Apache-2.0
"""v2 roles retain task identity and reject task contamination and false anchors."""
import json
from pathlib import Path

import pytest

from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.contracts.scene_composition import background_role, migrate_composition
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph


def legacy():
    root = Path(__file__).resolve().parents[1] / "fixtures/task_world"
    return compile_pick_place_slots(json.loads((root / "cross_room_slots_response.json").read_text())["slots"],
                                    json.loads((root / "cross_room_request.json").read_text()))


def scene():
    old = legacy()
    return migrate_composition(old, expected_hash=old.content_hash)


def test_explicit_migration_preserves_legacy_and_task_and_invalidates_evidence():
    old = legacy(); before = old.to_dict(); new = scene()
    assert old.to_dict() == before
    assert old.task_view()["goals"] == new.task_view()["goals"]
    assert old.task_view()["steps"] == new.task_view()["steps"]
    assert new.to_dict()["parent_hash"] == old.content_hash
    assert TaskWorldGraph.from_dict(new.to_dict()).content_hash == new.content_hash
    assert scene().content_hash == new.content_hash
    with pytest.raises(GraphError, match="GRAPH_STALE_EVIDENCE"):
        new.check_evidence({"graph_hash": old.content_hash})
    with pytest.raises(GraphError, match="GRAPH_VERSION_CONFLICT"):
        migrate_composition(old, expected_hash="0" * 64)
    with pytest.raises(GraphError, match="GRAPH_SCHEMA_UNSUPPORTED"):
        migrate_composition(new, expected_hash=new.content_hash)


@pytest.mark.parametrize("fields,code", [
    ({"purpose": "unknown"}, "COMPOSITION_PURPOSE_INVALID"),
    ({"purpose": "background"}, "COMPOSITION_PURPOSE_INVALID"),
    ({"purpose": "background", "priority": "optional"}, "COMPOSITION_BACKGROUND_IN_TASK"),
    ({"placement_kind": "wall"}, "COMPOSITION_ANCHOR_INVALID"),
])
def test_background_cannot_replace_task_ancestor(fields, code):
    raw = scene().to_dict(); raw.pop("content_hash")
    next(n for n in raw["nodes"] if n["id"] == "cup").update(fields)
    with pytest.raises(GraphError, match=code):
        TaskWorldGraph.from_dict(raw)


def test_background_wall_and_floor_roles_share_same_graph():
    graph = scene(); raw = graph.to_dict()
    for placement, category in (("floor", "cabinet"), ("wall", "picture")):
        role, edge = background_role(raw, "study", category, category, placement)
        raw["nodes"].append(role); raw["edges"].append(edge)
    new = graph.revise(raw, expected_hash=graph.content_hash)
    assert {"cabinet", "picture"} <= set(new.room_view("study")["node_ids"])
    assert new.to_dict()["built"] == {}
    raw = new.to_dict(); raw.pop("content_hash")
    raw["requirements"][0]["subject_ids"].append("picture")
    with pytest.raises(GraphError, match="COMPOSITION_BACKGROUND_IN_TASK"):
        TaskWorldGraph.from_dict(raw)


def test_instances_must_inherit_role_purpose_and_binding():
    raw = scene().to_dict(); raw.pop("content_hash")
    raw["nodes"].append({"id": "cup_instance", "kind": "asset_instance", "label": "cup instance",
                         "room_id": "kitchen", "source": "inferred", "asset_ref": "test:cup",
                         "purpose": "task_core", "placement_anchor": {"kind": "surface", "ref": "rc_kitchen_source_surface"}})
    raw["edges"].extend({"id": "test_"+kind, "kind": kind, "source_id": "cup", "target_id": "cup_instance"}
                         for kind in ("contains", "role_bound_to"))
    TaskWorldGraph.from_dict(raw)
    raw["nodes"][-1]["purpose"] = "background"
    with pytest.raises(GraphError, match="COMPOSITION_PURPOSE_INVALID"):
        TaskWorldGraph.from_dict(raw)

def test_legacy_loader_rejects_unversioned_new_fields():
 raw=legacy().to_dict();raw.pop('content_hash');raw['nodes'][0]['purpose']='background'
 with pytest.raises(GraphError,match='GRAPH_SHAPE_INVALID'):TaskWorldGraph.from_dict(raw)

def test_instance_anchor_must_reference_actual_role_parent():
 raw=scene().to_dict();raw.pop('content_hash')
 raw['nodes'].append({'id':'inst','kind':'asset_instance','label':'cup','room_id':'kitchen','source':'inferred','asset_ref':'test:cup',
                     'purpose':'task_core','placement_anchor':{'kind':'surface','ref':'fake'}})
 raw['edges'].extend({'id':'t_'+kind,'kind':kind,'source_id':'cup','target_id':'inst'} for kind in ('contains','role_bound_to'))
 with pytest.raises(GraphError,match='COMPOSITION_ANCHOR_INVALID'):TaskWorldGraph.from_dict(raw)
