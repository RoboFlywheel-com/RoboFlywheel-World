# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import pytest
from rfw_w.scenes.geometry.optional_rigid_scope import verify_optional_extension
from rfw_w.scenes.contracts.surface_appearance import SurfaceAppearanceError


def values():
    old={'graph_id':'task','nodes':[{'id':'s','kind':'surface','room_id':'r'},
         {'id':'p','kind':'distractor_policy','candidate_categories':['plate'],'max_count':1},
         {'id':'cup','kind':'asset_role','category':'cup'}],
         'edges':[{'id':'ep','kind':'contains','source_id':'s','target_id':'p'}]}
    new=deepcopy(old);new['nodes'][1].update(candidate_categories=['plate','bowl'],max_count=2)
    new['nodes'].append({'id':'b','kind':'asset_role','room_id':'r','category':'bowl','physical_class':'manipuland',
        'placement_kind':'surface','priority':'optional','purpose':'distraction','task_role':'distractor'})
    new['edges'].append({'id':'eb','kind':'contains','source_id':'s','target_id':'b'})
    return old,new


def test_optional_extension_preserves_core():
    old,new=values();verify_optional_extension(old,new,'b','s')


@pytest.mark.parametrize('change',['task','core','policy','surface','role','edge'])
def test_optional_extension_rejects_scope_changes(change):
    old,new=values()
    if change=='task':new['goals']=['changed']
    if change=='core':new['nodes'][2]['category']='jar'
    if change=='policy':new['nodes'][1]['max_count']=5
    if change=='surface':new['nodes'][0]['room_id']='other'
    if change=='role':new['nodes'][-1]['priority']='core'
    if change=='edge':new['edges'].pop(0)
    with pytest.raises(SurfaceAppearanceError):verify_optional_extension(old,new,'b','s')


def test_added_asset_cannot_change_original_goal_or_binding():
    from rfw_w.scenes.geometry.optional_rigid_scope import verify_optional_ledger
    old={'house_plan_hash':'h','goal_placement':{'target_role_id':'cup'},'unplaced_optional':[],
         'instances':[{'role_id':'cup','asset_id':'old'}],'queries':{'cup':{}},'selection':{'cup':{}}}
    new=deepcopy(old);new['instances'].append({'role_id':'b','asset_id':'new'})
    new['queries']['b']={};new['selection']['b']={}
    verify_optional_ledger(old,new,'b')
    new['instances'][0]['asset_id']='wrong'
    with pytest.raises(SurfaceAppearanceError,match='ROLLBACK_CORE_VIOLATION'):
        verify_optional_ledger(old,new,'b')


def test_agent_adds_optional_objects_before_appearance():
    from types import SimpleNamespace
    from rfw_w.scenes.orchestrator.adaptive_scene_agent import AgentState,available_tools
    graph=SimpleNamespace(to_dict=lambda:{'built':{'asset_assembly':{}},'nodes':[{'id':'r','kind':'room'}]})
    state=AgentState(graph,checkpoint_dir='cp',done={'floor','wall'},style_enabled=True,surface_optionals=True)
    assert available_tools(state)==['add_surface_rigid_asset']
    state.optional_added={'bowl','jar'}
    assert 'search_surface_materials' in available_tools(state)
