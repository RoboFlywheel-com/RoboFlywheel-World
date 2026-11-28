# SPDX-License-Identifier: Apache-2.0
import copy
import pytest
from rfw_w.scenes.geometry.surface_coverings import covering_vertices,verify_covering
from rfw_w.scenes.contracts.surface_appearance import SurfaceAppearanceError


def setup():
    s={'kind':'wall_set','axis':'y','plane_m':3.95,'normal':[0,-1,0],'room_id':'study','bounds_uv_m':[0,0,4,2.8]}
    i={'instance_id':'poster','surface_id':'wall','room_id':'study','local_position_m':[2,1.4],
       'size_m':[.5,.6],'normal':s['normal'],'collision_policy':'static_clearance_proxy'}
    i['vertices_m']=covering_vertices(s,i['local_position_m'],i['size_m'])
    return s,i


def test_valid_wall_and_front_offset():
    s,i=setup();verify_covering(i,{'wall':s},{},{'instances':[]},{'instances':[]},[])
    assert all(p[1]==pytest.approx(3.949) for p in i['vertices_m'])


@pytest.mark.parametrize('mutation,code',[
    ('normal','COVERING_NORMAL_INVALID'),('outside','COVERING_OUTSIDE_SURFACE'),
    ('missing','COVERING_SURFACE_MISSING'),('overlap','COVERING_OVERLAP')])
def test_negative_coverings(mutation,code):
    s,i=setup();other=[]
    if mutation=='normal':i['normal']=[0,1,0]
    if mutation=='outside':i['local_position_m']=[.05,1.4];i['vertices_m']=covering_vertices(s,i['local_position_m'],i['size_m'])
    if mutation=='missing':i['surface_id']='missing'
    if mutation=='overlap':other=[copy.deepcopy(i)]
    with pytest.raises(SurfaceAppearanceError,match=code):verify_covering(i,{'wall':s},{},{'instances':[]},{'instances':[]},other)


def test_rug_corridor_refused():
    s,i=setup();s.update(kind='floor',axis='z',plane_m=0,normal=[0,0,1],bounds_uv_m=[0,0,4,4])
    i.update(collision_policy='visual_only',normal=s['normal'],local_position_m=[2,2]);i['vertices_m']=covering_vertices(s,i['local_position_m'],i['size_m'])
    plan={'corridor':{'x_span_m':[.5,3.5],'y_span_m':[1.5,2.5]}}
    with pytest.raises(SurfaceAppearanceError,match='COVERING_KEEP_OUT_VIOLATION'):
        verify_covering(i,{'wall':s},plan,{'instances':[]},{'instances':[]},[])


def test_v3_migration_explicit_and_covering_core_rejected():
    import json
    from pathlib import Path
    from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
    from rfw_w.scenes.contracts.scene_composition import migrate_composition,migrate_covering_composition,background_role
    from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph,GraphError
    f=Path(__file__).resolve().parents[1]/'fixtures/task_world'
    g=compile_pick_place_slots(json.loads((f/'cross_room_slots_response.json').read_text())['slots'],json.loads((f/'cross_room_request.json').read_text()))
    g=migrate_composition(g,expected_hash=g.content_hash);old=g.content_hash
    assert g.to_dict()['composition_schema'].endswith('/v2')
    g=migrate_covering_composition(g,expected_hash=old);assert g.content_hash!=old
    raw=g.to_dict();role,edge=background_role(raw,'kitchen','rug','rug','floor');role['physical_class']='covering'
    raw['nodes'].append(role);raw['edges'].append(edge);g=g.revise(raw,expected_hash=g.content_hash)
    raw=g.to_dict();raw.pop('content_hash');next(n for n in raw['nodes'] if n['id']=='rug').update(priority='core',purpose='task_core')
    with pytest.raises(GraphError,match='COVERING_CORE_FORBIDDEN'):TaskWorldGraph.from_dict(raw)
