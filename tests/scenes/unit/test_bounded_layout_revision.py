# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import pytest
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.contracts.layout_revision import permission, verify_pose_difference, dependents


class Graph:
    content_hash = 'a' * 64
    def to_dict(self):
        return {'nodes': [{'id':'kitchen','kind':'room'},
                         {'id':'cup','kind':'asset_role','room_id':'kitchen','source':'user'},
                         {'id':'counter','kind':'asset_role','room_id':'kitchen'}],
                'initial_state':[{'subject':'cup','predicate':'ON','object':'counter'}]}


def case():
    graph = Graph()
    ledger = {'instances':[
        {'role_id':'counter','room_id':'kitchen','pose_xyz_m':[0,0,0],'yaw_rad':0,
         'support_role_id':None,'asset_id':'support'},
        {'role_id':'cup','room_id':'kitchen','pose_xyz_m':[0,0,1],'yaw_rad':0,
         'support_role_id':'counter','asset_id':'real-cup'}],
        'goal_placement': {'destination_support_role_id':'counter','candidate_pose_xyz_m':[0,0,1]},
        'selection': {'cup':'real-cup'}, 'source_graph_hash': 'b'*64}
    grant = permission(graph,'cup',operations=['translate','rotate','change_support'],scope_ids=['kitchen'])
    revised = deepcopy(ledger);revised['instances'][1]['pose_xyz_m'][0]=.1
    return graph,ledger,revised,grant


def test_explicit_inferred_pose_grant_keeps_user_semantic_identity():
    graph,before,after,grant=case()
    result=verify_pose_difference(graph,before,after,grant,'translate')
    assert result['changed_role_ids']==['cup']
    assert 'target_ik' in result['invalidates'] and 'critic' in result['invalidates']


@pytest.mark.parametrize('mutation,code',[
    (lambda a,g:a['instances'][1].update(asset_id='another'), 'LAYOUT_SCOPE_VIOLATION'),
    (lambda a,g:a['instances'][0]['pose_xyz_m'].__setitem__(0,.1), 'LAYOUT_SCOPE_VIOLATION'),
    (lambda a,g:a['instances'].pop(), 'LAYOUT_SCOPE_VIOLATION'),
    (lambda a,g:a['selection'].update(cup='another'), 'LAYOUT_SCOPE_VIOLATION'),
    (lambda a,g:a['instances'][1]['pose_xyz_m'].__setitem__(0,float('nan')), 'LAYOUT_INPUT_INVALID'),
    (lambda a,g:g.update(graph_hash='c'*64), 'LAYOUT_CANDIDATE_STALE'),
])
def test_scope_asset_source_and_pose_negative_cases(mutation,code):
    graph,before,after,grant=case();mutation(after,grant)
    with pytest.raises(GraphError,match=code):verify_pose_difference(graph,before,after,grant,'translate')


def test_no_implicit_permission_and_explicit_pose_lock():
    graph,before,after,grant=case()
    with pytest.raises(GraphError,match='LAYOUT_LOCKED'):verify_pose_difference(graph,before,after,None,'translate')
    grant=permission(graph,'cup',operations=['translate'],scope_ids=['kitchen'],locked_fields=['pose_xyz_m'])
    with pytest.raises(GraphError,match='LAYOUT_LOCKED'):verify_pose_difference(graph,before,after,grant,'translate')


def test_hard_support_relation_cannot_be_overridden_by_grant():
    graph,before,after,grant=case();after['instances'][1]['support_role_id']='another'
    with pytest.raises(GraphError,match='LAYOUT_RELATION_VIOLATION'):
        verify_pose_difference(graph,before,after,grant,'change_support')


def test_support_dependency_closure_prevents_silent_child_deletion():
    graph,before,after,grant=case()
    before['instances'].append({'role_id':'child','support_role_id':'cup'})
    assert dependents(before['instances'],'counter')==['child','cup']
    grant=permission(graph,'counter',operations=['translate'],scope_ids=['kitchen'])
    with pytest.raises(GraphError,match='LAYOUT_DEPENDENTS_UNSUPPORTED'):
        verify_pose_difference(graph,before,deepcopy(before),grant,'translate')


def test_no_delta_rejected():
    graph,before,after,grant=case()
    with pytest.raises(GraphError,match='LAYOUT_NO_PROGRESS'):
        verify_pose_difference(graph,before,before,grant,'translate')


def test_rotated_footprint_measures_orientation_and_conserves_area():
    import math
    from rfw_w.scenes.geometry.layout_revision import footprint
    item={'dimensions_m':[2,1,.1],'pose_xyz_m':[0,0,1],'yaw_rad':math.pi/2}
    poly=footprint(item)
    assert poly.area==pytest.approx(2)
    assert poly.bounds()==pytest.approx((-.5,-1,.5,1))


def test_old_core_gate_rejects_rotated_cup_outside_actual_support(tmp_path):
    import math
    from tests.scenes.unit.test_room_asset_tools import fixture_ledger,_rehash
    from rfw_w.scenes.assets.room_tools import verify_scene_assembly,RoomAssetError
    graph,plan,ledger=fixture_ledger(tmp_path)
    cup=ledger['instances'][2];cup['dimensions_m']=[.1,.4,.1];cup['pose_xyz_m'][0]=-1.56
    prepared=tmp_path/'prepared/cup_fixture/prepared.json'
    import json
    metadata=json.loads(prepared.read_text());metadata['dimensions_m']=cup['dimensions_m'];prepared.write_text(json.dumps(metadata))
    verify_scene_assembly(graph,plan,_rehash(ledger),tmp_path)
    cup['yaw_rad']=math.pi/2
    with pytest.raises(RoomAssetError,match='SUPPORT_NOT_FOUND'):
        verify_scene_assembly(graph,plan,_rehash(ledger),tmp_path)


def test_component_difference_rejects_independent_child_motion():
    graph,before,after,grant=case()
    grant=permission(graph,'counter',operations=['move_component'],scope_ids=['kitchen'])
    after=deepcopy(before);after['instances'][0]['pose_xyz_m'][0]=.1
    with pytest.raises(GraphError,match='LAYOUT_RELATION_VIOLATION'):
        verify_pose_difference(graph,before,after,grant,'move_component',component=True)


def test_legal_support_switch_and_hard_relation_rejection():
    from rfw_w.scenes.geometry.layout_revision import transformed
    graph,before,after,grant=case()
    before['instances'][0]['support_surface']={'plane_z_m':1,'polygon_xy':[[-1,-1],[1,-1],[1,1],[-1,1]]}
    before['instances'].append({'role_id':'another','room_id':'kitchen','pose_xyz_m':[2,0,0],
        'yaw_rad':0,'support_role_id':None,'asset_id':'second-support',
        'support_surface':{'plane_z_m':1,'polygon_xy':[[1,-1],[3,-1],[3,1],[1,1]]}})
    revised,_=transformed(before,{'instances':[]},'cup','change_support',support_id='another')
    class FreeGraph(Graph):
        def to_dict(self):
            raw=super().to_dict();raw['initial_state']=[];return raw
    free=FreeGraph();grant=permission(free,'cup',operations=['change_support'],scope_ids=['kitchen'])
    assert verify_pose_difference(free,before,revised,grant,'change_support')['changed_role_ids']==['cup']
    with pytest.raises(GraphError,match='LAYOUT_RELATION_VIOLATION'):
        verify_pose_difference(graph,before,revised,grant,'change_support')
