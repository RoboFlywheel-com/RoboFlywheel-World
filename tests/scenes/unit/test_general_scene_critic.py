# SPDX-License-Identifier: Apache-2.0
import pytest
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.contracts.scene_critique import validate_request,validate_visual_response
from tests.scenes.unit.test_bounded_layout_revision import Graph


@pytest.mark.parametrize('scope,profiles,code',[
    (['missing'],['static'],'CRITIC_SCOPE_VIOLATION'),
    (['kitchen'],['physics_rollout'],'CRITIC_PROFILE_UNSUPPORTED'),
    (['kitchen'],['static','static'],'CRITIC_PROFILE_UNSUPPORTED'),
])
def test_unsupported_scope_profile_rejected(scope,profiles,code):
    with pytest.raises(GraphError,match=code):validate_request(Graph(),scope,profiles)


def response():
    return {'graph_hash':Graph.content_hash,'quality_pass':False,'issues':[
        {'scope_ids':['cup'],'severity':'error','code':'VISUAL_QUALITY',
         'detail':'cup too close to support edge','suggested_operation':'translate'}]}


def test_visual_issue_is_semantic_read_only_suggestion():
    result=validate_visual_response(Graph(),response(),['kitchen'])
    assert result[0]['source']=='visual' and result[0]['suggested_operation']=='translate'


@pytest.mark.parametrize('field,value',[
    ('scope_ids',['unknown']),('code','PHYSICS_PASSED'),('suggested_operation','delete_core'),
    ('detail',''),('severity','passed'),
])
def test_model_cannot_invent_ids_actions_or_physical_measurements(field,value):
    raw=response();raw['issues'][0][field]=value
    with pytest.raises(GraphError,match='CRITIC_RESPONSE_INVALID'):
        validate_visual_response(Graph(),raw,['kitchen'])


def test_stale_model_version_cannot_approve_current_scene():
    raw=response();raw['graph_hash']='b'*64
    with pytest.raises(GraphError,match='CRITIC_RESPONSE_INVALID'):
        validate_visual_response(Graph(),raw,['kitchen'])


def test_shared_budget_counts_failed_attempts_and_refuses_exhaustion():
    from rfw_w.scenes.orchestrator.scene_critic_tools import ModelBudget
    budget=ModelBudget(1);budget.spend()
    assert budget.used==1
    with pytest.raises(GraphError,match='BUDGET_EXHAUSTED'):budget.spend()


def test_edge_margin_review_is_measured_soft_issue():
    from rfw_w.scenes.orchestrator.scene_critic_tools import support_margin,margin_issues
    core={'instances':[
        {'role_id':'counter','room_id':'kitchen','support_role_id':None,
         'support_surface':{'polygon_xy':[[-1,-1],[1,-1],[1,1],[-1,1]]}},
        {'role_id':'cup','room_id':'kitchen','support_role_id':'counter',
         'pose_xyz_m':[.94,0,1],'dimensions_m':[.1,.1,.1],'yaw_rad':0}]}
    assert support_margin(core,'cup')==pytest.approx(.01)
    issue=margin_issues(Graph(),core,['cup'])[0]
    assert issue['code']=='SUPPORT_EDGE_MARGIN' and issue['severity']=='warning'


def test_cache_key_changes_with_source_digest(tmp_path,monkeypatch):
    import rfw_w.scenes.orchestrator.scene_critic_tools as module
    calls=[]
    monkeypatch.setattr(module,'read_graph',lambda p:Graph())
    marker={'value':'a'}
    monkeypatch.setattr(module,'artifact_digest',lambda p:{'resource':marker['value']})
    def measure(p):
        calls.append(1);raise GraphError('GEOMETRY_OUT_OF_BOUNDS','fixture')
    monkeypatch.setattr(module,'_verify_bundle',measure)
    source=tmp_path/'source';source.mkdir();cache={}
    first=module.critique_scene(source,tmp_path/'first',scope_ids=['cup'],expected_version=Graph.content_hash,cache=cache)
    second=module.critique_scene(source,tmp_path/'second',scope_ids=['cup'],expected_version=Graph.content_hash,cache=cache)
    assert second['cache_hit'] and second['model_requests']==0 and len(calls)==1
    marker['value']='b'
    third=module.critique_scene(source,tmp_path/'third',scope_ids=['cup'],expected_version=Graph.content_hash,cache=cache)
    assert 'cache_hit' not in third and len(calls)==2
