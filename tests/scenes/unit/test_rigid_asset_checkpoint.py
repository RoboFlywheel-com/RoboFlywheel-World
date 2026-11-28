# SPDX-License-Identifier: Apache-2.0
"""Scope and artifact gates reject changes outside the restored rigid binding."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.surface_appearance import SurfaceAppearanceError
from rfw_w.scenes.geometry.rigid_asset_checkpoint import verify_checkpoint_payload, verify_binding_scope


def data():
    graph = {'graph_id':'task', 'nodes':[{'id':'inst_cup','kind':'asset_instance','asset_ref':'old','source_reason':'old'}],
             'built':{'house_shell':{'mesh':'same'}},'steps':[{'action':'pick'}], 'edges':[]}
    core = {'source_graph_hash':'same','house_plan_hash':'same','unplaced_optional':[],
            'instances':[{'role_id':'cup','asset_id':'old'},{'role_id':'desk','asset_id':'fixed'}],
            'queries':{'cup':{},'desk':{}},'selection':{'cup':{},'desk':{}},
            'goal_placement':{'target_role_id':'cup'}}
    return graph, core


def test_single_binding_change_is_allowed():
    saved, core = data(); current = deepcopy(saved); new = deepcopy(core)
    current['nodes'][0].update(asset_ref='new',source_reason='new');new['instances'][0]['asset_id']='new'
    verify_binding_scope(current,saved,new,core,'cup')


@pytest.mark.parametrize('change', ['task','house','other_instance','other_query','node'])
def test_scope_gate_rejects_unrelated_changes(change):
    saved, core = data();current=deepcopy(saved);new=deepcopy(core)
    if change=='task':current['steps']=[]
    if change=='house':current['built']['house_shell']={}
    if change=='other_instance':new['instances'][1]['asset_id']='changed'
    if change=='other_query':new['queries']['desk']={'changed':True}
    if change=='node':current['nodes'][0]['id']='different'
    with pytest.raises(SurfaceAppearanceError,match='ROLLBACK_CORE_VIOLATION'):
        verify_binding_scope(current,saved,new,core,'cup')


@pytest.mark.parametrize('change',['none','checkpoint','artifact','graph'])
def test_checkpoint_digest(change):
    artifacts={'prepared/cup.obj':'digest'}; graph=SimpleNamespace(content_hash='graph')
    checkpoint={'schema_version':'pvscene.rigid-checkpoint/v1','graph_hash':'graph','artifacts':artifacts.copy()}
    checkpoint['content_hash']=content_hash(checkpoint)
    if change=='checkpoint':checkpoint['graph_hash']='changed'
    if change=='artifact':artifacts={}
    if change=='graph':graph.content_hash='different'
    if change=='none':verify_checkpoint_payload(checkpoint,graph,artifacts)
    else:
        with pytest.raises(SurfaceAppearanceError,match='CHECKPOINT_TAMPERED'):
            verify_checkpoint_payload(checkpoint,graph,artifacts)


def test_goal_support_change_is_forbidden():
    saved,core=data();new=deepcopy(core)
    new['goal_placement']['destination_support_role_id']='other'
    with pytest.raises(SurfaceAppearanceError,match='ROLLBACK_CORE_VIOLATION'):
        verify_binding_scope(saved,saved,new,core,'cup')


def test_agent_requires_checkpoint_then_restore_before_background():
    from rfw_w.scenes.orchestrator.adaptive_scene_agent import AgentState,available_tools
    graph=SimpleNamespace(to_dict=lambda:{'built':{'asset_assembly':{}},'nodes':[]})
    state=AgentState(graph,rigid_enabled=True,rigid_restore=True)
    assert available_tools(state)==['save_rigid_checkpoint']
    state.rigid_checkpoint='saved'
    assert available_tools(state)==['search_rigid_assets']
    state.rigid_selected=True
    assert available_tools(state)==['restore_rigid_asset']
    state.rigid_restored=True
    assert available_tools(state)==['save_checkpoint']


def test_environment_failure_does_not_rebuild_background():
    from rfw_w.scenes.orchestrator.adaptive_scene_agent import AgentState,available_tools
    graph=SimpleNamespace(to_dict=lambda:{'built':{'asset_assembly':{}},'nodes':[]})
    state=AgentState(graph,checkpoint_dir='saved',failure='MATERIAL_ENV_MISSING')
    assert available_tools(state)==['terminate']
