# SPDX-License-Identifier: Apache-2.0
"""Room intent must survive compilation/plan; implicit dimensions must stay auditable."""
from rfw_w.scenes.resources import data_path, module_path
from dataclasses import replace
import json
from pathlib import Path
import pytest
from rfw_w.scenes.contracts.scene_spec import RoomSpec, SceneSpec, SpecError, ROOM_SIZE_POLICY_VERSION
from rfw_w.scenes.contracts.room_plan import RoomPlan, floor_plan_payload
from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.room import fit_room_around_cell, RoomError
from rfw_w.scenes.solvers.room_plan import plan_from_fitted

ROOT = Path(__file__).resolve().parents[3]


def test_legacy_room_keeps_its_serialisation():
    raw={'size_bounds_m':[[4,5],[3,4],[2.6,3]]}
    assert RoomSpec.from_dict(raw).to_dict()==raw


@pytest.mark.parametrize('kind', ['office','kitchen','bedroom'])
def test_inferred_large_is_roomy_and_roundtrips(kind):
    room=RoomSpec.from_dict({'room_kind':kind,'scale_class':'large'})
    assert room.size_policy=='inferred'
    assert room.size_bounds_m[0].minimum==8
    assert room.size_policy_version==ROOM_SIZE_POLICY_VERSION
    assert RoomSpec.from_dict(room.to_dict())==room


@pytest.mark.parametrize('payload,code',[
    ({'room_kind':'castle','size_bounds_m':[[4,5],[3,4],[2.7,3]]},'ROOM_KIND_UNSUPPORTED'),
    ({'room_kind':'office','size_policy':'explicit'},'ROOM_SIZE_CONFLICT'),
    ({'room_kind':'office','scale_class':'unlimited'},'ROOM_SCALE_POLICY_MISSING'),
    ({'room_kind':'office','size_policy':'inferred','scale_class':'large','size_bounds_m':[[4,5],[3,4],[2.7,3]]},'ROOM_SIZE_CONFLICT'),
])
def test_unsupported_intent_cannot_silently_fallback(payload,code):
    with pytest.raises(SpecError,match=code):RoomSpec.from_dict(payload)


@pytest.mark.parametrize('kind',['office','kitchen','bedroom'])
def test_kind_and_scale_reach_plan_without_changing_core(kind):
    spec=SceneSpec.from_dict(json.loads((data_path('examples/office_pick_cup/task.json')).read_text()))
    core=solve_cell(spec,seed=spec.seed)
    original=core.core_hash
    room=RoomSpec.from_dict({'room_kind':kind,'scale_class':'large'})
    spec=replace(spec,room=room)
    fitted=fit_room_around_cell(spec,core,room_seed=20)
    plan=plan_from_fitted(spec,core,fitted,room_seed=20)
    assert plan.rooms[0].kind==kind
    assert next(iter(floor_plan_payload(plan)['rooms'])).startswith(kind+'_')
    assert plan.provenance.room_intent==room.to_dict()
    assert core.core_hash==plan.core.core_hash==original
    assert RoomPlan.from_dict(plan.to_dict()).content_hash==plan.content_hash


def test_explicit_room_never_expands_to_fit_core():
    spec=SceneSpec.from_dict(json.loads((data_path('examples/office_pick_cup/task.json')).read_text()))
    core=solve_cell(spec,seed=spec.seed)
    room=RoomSpec.from_dict({'room_kind':'office','size_bounds_m':[[.5,.6],[.5,.6],[2.7,3]]})
    with pytest.raises(RoomError,match='ROOM_SIZE_CONFLICT'):
        fit_room_around_cell(replace(spec,room=room),core,room_seed=20)
