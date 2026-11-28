# SPDX-License-Identifier: Apache-2.0
import pytest
from rfw_w.scenes.orchestrator.adaptive_scene_agent import validate_decision,AgentState,available_tools,ReplayDecisionProvider,public_summary
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.contracts.scene_composition import migrate_composition
import json
from pathlib import Path

def graph():
 root=Path(__file__).resolve().parents[1]/'fixtures/task_world';g=compile_pick_place_slots(json.loads((root/'cross_room_slots_response.json').read_text())['slots'],json.loads((root/'cross_room_request.json').read_text()));return migrate_composition(g,expected_hash=g.content_hash)

def test_initial_tools_are_state_dependent():
 s=AgentState(graph());assert available_tools(s)==['revise_composition'];s.configured=True;assert available_tools(s)==['build_house']

def test_policy_selects_between_simultaneously_legal_background_tools():
 summary={'available_tools':[{'name':n} for n in ('furnish_floor','mount_wall')],'completed_background':[],'current_static_verified':False}
 assert ReplayDecisionProvider('floor')(summary)['tool_name']=='furnish_floor'
 assert ReplayDecisionProvider('wall')(summary)['tool_name']=='mount_wall'

@pytest.mark.parametrize('decision,code',[
 ({'tool_name':'invented','scope_ids':[],'parameters':{},'reason_code':'TEST'},'TOOL_NOT_AVAILABLE'),
 ({'tool_name':'build_house','scope_ids':['cup'],'parameters':{},'reason_code':'TEST'},'PATCH_SCOPE_VIOLATION'),
 ({'tool_name':'build_house','scope_ids':[],'parameters':{'built':True},'reason_code':'TEST'},'AGENT_ACTION_INVALID'),
 ({'tool_name':'build_house','scope_ids':[],'parameters':{},'reason_code':'/home/private'},'AGENT_ACTION_INVALID')])
def test_invalid_agent_proposals_rejected(decision,code):
 with pytest.raises(GraphError,match=code):validate_decision(decision,['build_house'])

def test_summary_excludes_artifact_paths():
 s=AgentState(graph());s.assets_dir=Path('/home/' + 'private/assets');raw=json.dumps(public_summary(s,available_tools(s)));assert '/home/' not in raw

def test_tool_budget_stops_and_preserves_report(tmp_path):
 from rfw_w.scenes.orchestrator.adaptive_scene_agent import AdaptiveSceneAgent
 report=AdaptiveSceneAgent(ReplayDecisionProvider(),max_tools=0).run(graph(),tmp_path/'run')
 assert report['status']=='budget_exhausted'
 assert report['tool_calls']==0
 assert report['model_requests_used']==0
 assert (tmp_path/'run/report.json').is_file()

def test_invalid_decision_terminates_without_tool_execution(tmp_path):
 from rfw_w.scenes.orchestrator.adaptive_scene_agent import AdaptiveSceneAgent
 def bad(summary):return {'tool_name':'finish','scope_ids':[],'parameters':{},'reason_code':'FAKE_PASS'}
 report=AdaptiveSceneAgent(bad).run(graph(),tmp_path/'run')
 assert report['status']=='failed'
 assert report['failure_code']=='TOOL_NOT_AVAILABLE'
 assert report['tool_calls']==0


def test_new_tools_use_real_parameter_schemas_and_pending_state():
 s=AgentState(graph(),rigid_enabled=True)
 raw=s.graph.to_dict();raw['built']['asset_assembly']={'ledger_hash':'0'*64};s.graph=s.graph.revise(raw,expected_hash=s.graph.content_hash)
 assert available_tools(s)==['search_rigid_assets']
 summary=public_summary(s,available_tools(s))
 assert summary['available_tools'][0]['parameter_schema']=={'role_id':'rigid_role_id','description':'task object description'}
 validate_decision({'tool_name':'search_rigid_assets','scope_ids':[],'parameters':{'role_id':'cup','description':'coffee mug'},'reason_code':'TEST'},available_tools(s))


def test_model_code_fence_does_not_trigger_extra_request(monkeypatch):
 import rfw_w.scenes.orchestrator.adaptive_scene_agent as module
 from types import SimpleNamespace
 monkeypatch.setattr(module,'require_external_permission',lambda:None)
 monkeypatch.setattr(module,'call_vision',lambda *a,**k:SimpleNamespace(text='```json\n{"tool_name":"finish","scope_ids":[],"parameters":{},"reason_code":"TEST"}\n```'))
 assert module.OnlineDecisionProvider(None)({})['tool_name']=='finish'


def test_covering_roles_are_excluded_from_furniture_missing_pool():
 # Covering acceptance has its own ledger; it must not be claimed as unplaced HSSD furniture.
 from rfw_w.scenes.contracts.scene_composition import migrate_covering_composition,background_role
 g=migrate_covering_composition(graph(),expected_hash=graph().content_hash);raw=g.to_dict()
 role,edge=background_role(raw,'kitchen','rug','rug','floor');role['physical_class']='covering'
 raw['nodes'].append(role);raw['edges'].append(edge)
 pending=[n['id'] for n in raw['nodes'] if n['kind']=='asset_role' and n.get('purpose')!='task_core'
          and n.get('placement_kind') in ('floor','wall') and n.get('physical_class')!='covering']
 assert 'rug' not in pending


def test_required_style_blocks_final_verification_until_binding():
 s=AgentState(graph(),style_enabled=True)
 raw=s.graph.to_dict();raw['built']['asset_assembly']={'ledger_hash':'0'*64};s.graph=s.graph.revise(raw,expected_hash=s.graph.content_hash)
 s.checkpoint_dir=Path('checkpoint');s.done={'floor','wall'}
 assert available_tools(s)==['search_surface_materials']
 s.style_queries={('kitchen','floor'):{},('kitchen','wall_set'):{}}
 assert available_tools(s)==['set_room_surface_materials']
 s.styled_rooms={'kitchen','study'}
 assert available_tools(s)==['verify_scene']
