# SPDX-License-Identifier: Apache-2.0
"""State-derived tool availability, bounded decisions and immutable attempt history.

Semantic decision providers receive public task intent and legal actions. They never
receive filesystem paths, artifact logs or credentials. Tool executors own measurements.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import json
from pathlib import Path
import time
from typing import Callable
from rfw_w.scenes.agents.composition_revision import validate_composition_patch
from rfw_w.scenes.agents.llm_client import LLMConfig,LLMError,call_vision,require_external_permission
from rfw_w.scenes.agents.task_world_planner import SENSITIVE_TEXT,_unique
from rfw_w.scenes.contracts.task_world_graph import GraphError,TaskWorldGraph
from rfw_w.scenes.contracts.scene_composition import migrate_composition
from rfw_w.scenes.orchestrator.composition_tools import commit_composition_patch
from rfw_w.scenes.orchestrator.house_tools import run_house_tools
from rfw_w.scenes.orchestrator.room_asset_tools import run_room_asset_tools,_event
from rfw_w.scenes.orchestrator.background_tools import run_background_tools,verify_background
from rfw_w.scenes.orchestrator.scene_checkpoints import save_scene_checkpoint,rollback_scene_scope,restore_background_artifacts
from rfw_w.scenes.assets.room_tools import RoomAssetError
from rfw_w.scenes.solvers.house_tools import HouseToolError
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.surface_appearance import SurfaceAppearanceError

from rfw_w.scenes.orchestrator.tool_capabilities import describe_tools, group_tools, legacy_tools

TOOLS = legacy_tools('scene')


def furnishing_options():
    options=[]
    for secondary in ('chair','side_table'):
        operations=[]
        for room in ('kitchen','study'):
            for category in ('cabinet',secondary):
                operations.append({'action':'add_optional_role','room_id':room,'role_id':room+'_'+category,
                    'category':category,'placement_kind':'floor','purpose':'background'})
            operations.append({'action':'add_optional_role','room_id':room,'role_id':room+'_mirror',
                'category':'mirror','placement_kind':'wall','purpose':'background'})
        options.append({'operations':operations})
    return options


@dataclass
class AgentState:
    graph:TaskWorldGraph
    surface_optionals:bool=False
    optional_added:set[str]=field(default_factory=set)
    rigid_restore:bool=False
    rigid_restored:bool=False
    rigid_checkpoint:Path|None=None
    rigid_saved_assets:Path|None=None
    rigid_enabled:bool=False
    rigid_query:dict|None=None
    rigid_selected:bool=False
    coverings_enabled:bool=False
    covering_created:set[str]=field(default_factory=set)
    style_enabled:bool=False
    style_queries:dict=field(default_factory=dict)
    styled_rooms:set[str]=field(default_factory=set)
    configured:bool=False
    house_dir:Path|None=None
    assets_dir:Path|None=None
    checkpoint_dir:Path|None=None
    checkpoint_assets:Path|None=None
    done:set[str]=field(default_factory=set)
    verified_hash:str|None=None
    failure:str|None=None
    failure_scope:tuple[str,...]=()
    calls:int=0
    model_calls:int=0


def available_tools(state):
    raw=state.graph.to_dict();built=raw['built']; legal=[]
    if not built and not state.configured:legal.append('revise_composition')
    if not built and state.configured:legal.append('build_house')
    if 'house_shell' in built and 'asset_assembly' not in built:legal.append('assemble_core')
    if 'asset_assembly' in built:
        if state.rigid_restore and state.rigid_checkpoint is None:return ['save_rigid_checkpoint']
        if state.rigid_restore and state.rigid_selected and not state.rigid_restored:return ['restore_rigid_asset']
        if state.rigid_enabled and not state.rigid_selected:
            return ['select_rigid_asset'] if state.rigid_query else ['search_rigid_assets']
        if state.checkpoint_dir is None:legal.append('save_checkpoint')
        else:
            if state.failure:
                if state.failure.endswith(('ENV_MISSING','INDEX_INVALID')):legal.append('terminate')
                else:legal.extend(('rollback_scope','terminate'))
            else:
                for kind,name in (('floor','furnish_floor'),('wall','mount_wall')):
                    if kind not in state.done:legal.append(name)
                if state.done and not (state.style_enabled and state.done=={'floor','wall'} and len(state.styled_rooms)<len([n for n in raw['nodes'] if n['kind']=='room'])):
                    legal.append('verify_scene')
                if state.done=={'floor','wall'}:
                    if state.surface_optionals and state.optional_added!={'bowl','jar'}:return ['add_surface_rigid_asset']
                    pending=[n['id'] for n in raw['nodes'] if n['kind']=='room' and n['id'] not in state.styled_rooms]
                    if state.style_enabled and pending:
                        room=pending[0]
                        if all((room,k) in state.style_queries for k in ('floor','wall_set')):legal.append('set_room_surface_materials')
                        else:legal.append('search_surface_materials')
                    elif state.coverings_enabled and len(state.covering_created)<2:
                        legal.append('create_surface_covering')
                    else:
                        if state.coverings_enabled and state.covering_created:
                            legal.extend(('move_surface_covering','remove_surface_covering'))
                        if state.verified_hash==state.graph.content_hash:legal.append('finish')
    return legal


def public_summary(state,legal):
    raw=state.graph.to_dict()
    # Only whitelisted semantic fields enter the model payload.
    payload={'task':{k:raw[k] for k in ('initial_state','goals')},
        'graph_hash':state.graph.content_hash,'rooms':[{'id':n['id'],'kind':n['room_kind']} for n in raw['nodes'] if n['kind']=='room'],
        'roles':[{'id':n['id'],'category':n['category'],'purpose':n['purpose'],'placement_kind':n['placement_kind']} for n in raw['nodes'] if n['kind']=='asset_role'],
        'completed_background':sorted(state.done),'current_static_verified':state.verified_hash==state.graph.content_hash,
        'failure_code':state.failure,'tool_calls_used':state.calls,
        'available_tools':describe_tools('scene', legal),
        'patch_options':furnishing_options() if 'revise_composition' in legal else []}
    payload['tool_groups'] = group_tools(payload['available_tools'])
    if SENSITIVE_TEXT.search(json.dumps(payload)):raise GraphError('AGENT_ACTION_INVALID','sensitive summary')
    if state.style_enabled and state.done=={'floor','wall'}:
        pending=[n['id'] for n in raw['nodes'] if n['kind']=='room' and n['id'] not in state.styled_rooms]
        payload['style_room_id']=pending[0] if pending else None
        payload['material_queries']={kind:state.style_queries[(pending[0],kind)] for kind in ('floor','wall_set')
                                    if pending and (pending[0],kind) in state.style_queries}
        payload['material_query_needed']=next((k for k in ('floor','wall_set') if pending and (pending[0],k) not in state.style_queries),None)
    if state.surface_optionals:
        payload['optional_surface_ids']=[n['id'] for n in raw['nodes'] if n['kind']=='surface']
        payload['optional_added']=sorted(state.optional_added)
    if state.rigid_enabled and not state.rigid_selected:
        target=next(n for n in raw['nodes'] if n.get('task_role')=='target')
        payload['rigid_role_id']=target['id']
        payload['rigid_candidates']=state.rigid_query['candidates'] if state.rigid_query else []
    if state.coverings_enabled and 'surface_appearance' in raw['built']:
        appearance=raw['built']['surface_appearance'];from rfw_w.scenes.assets.surface_materials import SOLIDS
        payload['covering_material_ids']=sorted({b['material_id'] for b in appearance['bindings']}|set(SOLIDS))
        if state.assets_dir:
            from rfw_w.scenes.geometry.surface_appearance import surfaces_from_mesh
            plan=json.loads((state.assets_dir/'geometry/house_plan.json').read_text())
            payload['covering_surfaces']=list(surfaces_from_mesh(state.graph,plan,state.assets_dir/'geometry').values())
        payload['covering_instances']=sorted(state.covering_created)
    if SENSITIVE_TEXT.search(json.dumps(payload)):raise GraphError('AGENT_ACTION_INVALID','sensitive material summary')
    return payload


class ReplayDecisionProvider:
    """A local semantic policy recomputing its decision from current legal actions.

    Preferences choose between simultaneously legal furnishing tools. This is an offline
    baseline; its decisions are never reported as model requests.
    """
    def __init__(self,preference='floor',patch_index=0):self.preference=preference;self.patch_index=patch_index
    def __call__(self,summary):
        legal={t['name'] for t in summary['available_tools']}
        if 'add_surface_rigid_asset' in legal:
            category=next(c for c in ('bowl','jar') if c not in summary['optional_added'])
            return {'tool_name':'add_surface_rigid_asset','scope_ids':[],'parameters':{'surface_id':'rc_kitchen_source_surface',
                'role_id':'optional_'+category,'category':category,'description':'a '+category+' on the kitchen counter'},'reason_code':'ADD_OPTIONAL_SURFACE_OBJECT'}
        if 'search_rigid_assets' in legal:
            return {'tool_name':'search_rigid_assets','scope_ids':[],'parameters':{'role_id':summary['rigid_role_id'],
                'description':'a small coffee mug for picking and placing'},'reason_code':'SEARCH_TASK_TARGET'}
        if 'select_rigid_asset' in legal:
            return {'tool_name':'select_rigid_asset','scope_ids':[],'parameters':{'candidate_key':summary['rigid_candidates'][0]['candidate_key']},'reason_code':'SELECT_TASK_TARGET'}
        if 'create_surface_covering' in legal:
            ids=set(summary['covering_instances']);floor=not ids
            material=next(m for m in summary['covering_material_ids'] if m.startswith('Wood'))
            return {'tool_name':'create_surface_covering','scope_ids':[],'parameters':{
                'instance_id':'cover_rug' if floor else 'cover_poster','surface_id':'floor_left:kitchen' if floor else 'wall_top_right:study',
                'material_id':'solid:sage' if floor else material,'local_position_m':[-2,3.25] if floor else [3.2,2.35],
                'size_m':[1.1,.8] if floor else [.5,.4]},'reason_code':'ADD_DECORATIVE_CONTENT'}
        if 'rollback_scope' in legal:name='rollback_scope'
        elif legal=={'terminate'}:name='terminate'
        elif 'finish' in legal:name='finish'
        elif 'verify_scene' in legal and summary['completed_background']==['floor'] and not summary['current_static_verified'] and self.preference=='floor':name='verify_scene'
        elif {'furnish_floor','mount_wall'}<=legal:name='furnish_floor' if self.preference=='floor' else 'mount_wall'
        elif 'verify_scene' in legal and not ({'furnish_floor','mount_wall'}&legal):name='verify_scene'
        else:
            actionable=legal-{'verify_scene','terminate'}
            if len(actionable)!=1:raise GraphError('AGENT_ACTION_INVALID','policy cannot decide')
            name=next(iter(actionable))
        if 'material_query_needed' in summary and summary.get('style_room_id'):
            if 'search_surface_materials' in legal:
                name='search_surface_materials';kind=summary['material_query_needed']
                return {'tool_name':name,'scope_ids':[],'parameters':{'description':'wood oak' if kind=='floor' else 'plaster wall',
                    'surface_kind':kind},'reason_code':'CHOOSE_ROOM_STYLE'}
            if 'set_room_surface_materials' in legal:
                return {'tool_name':'set_room_surface_materials','scope_ids':[],'parameters':{
                    'floor_material_id':summary['material_queries']['floor']['candidates'][0]['material_id'],
                    'wall_material_id':summary['material_queries']['wall_set']['candidates'][0]['material_id'],
                    'meters_per_tile':1},'reason_code':'BIND_ROOM_STYLE'}
        return {'tool_name':name,'scope_ids':[], 'parameters':{'patch_index':self.patch_index} if name=='revise_composition' else {},
                'reason_code':'RECOVER_NAMED_FAILURE' if name=='rollback_scope' else 'SATISFY_TASK_AND_BACKGROUND'}


class OnlineDecisionProvider:
    def __init__(self,config):self.config=config
    def __call__(self,summary):
        require_external_permission()
        response=call_vision(self.config,system_prompt='Choose exactly one currently available scene tool using the task and measured failure summary. Return only JSON with tool_name, scope_ids (empty list), parameters (revise_composition: patch_index 0 or 1; search_surface_materials: description and surface_kind from material_query_needed; set_room_surface_materials: floor_material_id, wall_material_id from returned candidates and meters_per_tile; search_rigid_assets: role_id and task description; select_rigid_asset: candidate_key from rigid_candidates; create_surface_covering: instance_id starting cover_, surface_id from covering_surfaces, material_id, size_m, local_position_m; move_surface_covering: instance_id and local_position_m; remove_surface_covering: instance_id; otherwise empty object), reason_code (one uppercase token). Output a JSON object, never call a function or return a tool invocation. Parameter names and types must match parameter_schema exactly. Prefer finishing when verified; recover named failure using rollback. Do not invent geometry or validation.',
                             user_prompt=json.dumps(summary,ensure_ascii=False))
        text=response.text.strip()
        if text.startswith('```') and text.endswith('```'):
            text=text[text.find('\n')+1:-3].strip()
        try:return json.loads(text,object_pairs_hook=_unique)
        except (ValueError,TypeError):raise GraphError('AGENT_ACTION_INVALID','model JSON format') from None



def validate_decision(decision,legal):
    if not isinstance(decision,dict) or set(decision)!={'tool_name','scope_ids','parameters','reason_code'}:
        raise GraphError('AGENT_ACTION_INVALID','decision fields')
    if decision['tool_name'] not in legal:raise GraphError('TOOL_NOT_AVAILABLE',str(decision['tool_name']))
    if decision['scope_ids']!=[]:raise GraphError('PATCH_SCOPE_VIOLATION','executor derives scope from graph roles')
    params=decision['parameters'];name=decision['tool_name']
    if name=='search_rigid_assets':
        if not isinstance(params,dict) or set(params)!={'role_id','description'}:raise GraphError('AGENT_ACTION_INVALID','rigid query')
    elif name=='add_surface_rigid_asset':
        if not isinstance(params,dict) or set(params)!={'surface_id','role_id','category','description'}:raise GraphError('AGENT_ACTION_INVALID','surface optional parameters')
    elif name=='select_rigid_asset':
        if not isinstance(params,dict) or set(params)!={'candidate_key'}:raise GraphError('AGENT_ACTION_INVALID','rigid selection')
    elif name in ('create_surface_covering','move_surface_covering','remove_surface_covering'):
        fields={'create_surface_covering':{'instance_id','surface_id','material_id','local_position_m','size_m'},
                'move_surface_covering':{'instance_id','local_position_m'},'remove_surface_covering':{'instance_id'}}
        if not isinstance(params,dict) or set(params)!=fields[name]:raise GraphError('AGENT_ACTION_INVALID','covering parameters')
    elif name=='search_surface_materials':
        if not isinstance(params,dict) or set(params)!={'description','surface_kind'} or not isinstance(params['description'],str) or params['surface_kind'] not in ('floor','wall_set'):
            raise GraphError('AGENT_ACTION_INVALID','material query parameters')
    elif name=='set_room_surface_materials':
        if not isinstance(params,dict) or set(params)!={'floor_material_id','wall_material_id','meters_per_tile'}:
            raise GraphError('AGENT_ACTION_INVALID','material binding parameters')
    elif (name=='revise_composition'  and (not isinstance(params,dict) or set(params)!={'patch_index'} or type(params['patch_index']) is not int or params['patch_index'] not in (0,1))) or (name!='revise_composition' and params!={}):
        raise GraphError('AGENT_ACTION_INVALID','parameters')
    reason=decision['reason_code']
    if not isinstance(reason,str) or not reason or len(reason)>80 or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' for c in reason):
        raise GraphError('AGENT_ACTION_INVALID','reason code')
    return name


class AdaptiveSceneAgent:
    def __init__(self,decision_provider, *, max_tools=30,max_model_requests=12,deadline_s=600,inject_failure=False,surface_style=False,rigid_assets=False,coverings=False,rigid_restore=False,surface_optionals=False,robot_layout_executor=None,robot_layout_semantics=None,critic_repair=False):
        if max_tools<0 or max_model_requests<0 or deadline_s<=0:
            raise GraphError('AGENT_ACTION_INVALID','budget limits')
        if robot_layout_executor is not None and isinstance(decision_provider,OnlineDecisionProvider):
            raise GraphError('AGENT_ACTION_INVALID','robot online stage requires separate acceptance')
        self.robot_layout_executor=robot_layout_executor;self.robot_layout_semantics=robot_layout_semantics
        self.critic_repair=critic_repair
        self.provider=decision_provider;self.max_tools=max_tools;self.max_model_requests=max_model_requests
        self.deadline_s=deadline_s;self.inject_failure=inject_failure;self.surface_style=surface_style or coverings;self.rigid_assets=rigid_assets or rigid_restore;self.coverings=coverings;self.rigid_restore=rigid_restore;self.surface_optionals=surface_optionals

    def run_robot_layout(self,graph,assets,output,*,executor,semantics,max_revisions=2):
        from rfw_w.scenes.orchestrator.robot_layout_agent import RobotLayoutAgent
        return RobotLayoutAgent(self.provider,max_tools=self.max_tools,max_revisions=max_revisions,
            deadline_s=self.deadline_s).run(graph,assets,output,executor=executor,semantics=semantics)

    def run(self,graph,output):
        output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
        if graph.to_dict()['built']:raise GraphError('PRECONDITION_UNMET','fresh intent')
        if graph.to_dict()['composition_schema'].endswith('/v1'):graph=migrate_composition(graph,expected_hash=graph.content_hash)
        state=AgentState(graph,surface_optionals=self.surface_optionals,rigid_restore=self.rigid_restore,style_enabled=self.surface_style,rigid_enabled=self.rigid_assets,coverings_enabled=self.coverings);events=[];injected=False;signatures={};terminal=None
        (output/'source_graph.json').write_text(json.dumps(graph.to_dict(),indent=2)+'\n')
        while terminal is None:
            if state.calls>=self.max_tools or time.monotonic()-started>=self.deadline_s:
                terminal='budget_exhausted';state.failure='BUDGET_EXHAUSTED';break
            legal=available_tools(state);summary=public_summary(state,legal)
            if not legal:terminal='failed';state.failure='NO_PROGRESS';break
            online=isinstance(self.provider,OnlineDecisionProvider)
            model_decision=online and (len(legal)>1 or any(t in legal for t in ('add_surface_rigid_asset','revise_composition','search_surface_materials','set_room_surface_materials','search_rigid_assets','select_rigid_asset','create_surface_covering','move_surface_covering','remove_surface_covering')))
            if model_decision and state.model_calls>=self.max_model_requests:
                terminal='budget_exhausted';state.failure='BUDGET_EXHAUSTED';break
            decision_error=None
            if model_decision:state.model_calls+=1
            try:
                if online and not model_decision:
                    decision={'tool_name':legal[0],'scope_ids':[],'parameters':{},'reason_code':'ONLY_LEGAL_DEPENDENCY'}
                else:decision=self.provider(summary)
                name=validate_decision(decision,legal)
            except (GraphError,LLMError,SystemExit) as error:
                state.failure=getattr(error,'code','EXTERNAL_CALL_NOT_ALLOWED');terminal='failed'
                events.append({'available_tools':legal,'source_graph_hash':state.graph.content_hash,'decision_failure':state.failure,'decision_failure_detail':getattr(error,'detail','invalid model response'),'model_requests_used':state.model_calls});break
            signature=(name,state.graph.content_hash,state.failure)
            signatures[signature]=signatures.get(signature,0)+1
            if signatures[signature]>2:state.failure='NO_PROGRESS';terminal='failed';break
            index=state.calls+1;step=output/f'step_{index:03d}_{name}';base=state.graph
            record={'available_tools':legal,'summary':summary,'decision':decision,'source_graph_hash':base.content_hash,
                    'decision_source':'model' if model_decision else 'dependency' if online else 'replay_policy',
                    'tool_calls_before':state.calls,'model_requests_used':state.model_calls,'artifact_dir':step.name}
            tick=time.monotonic();failed=False;details={}
            try:
                if name=='revise_composition':
                    state.graph=commit_composition_patch(base,furnishing_options()[decision['parameters']['patch_index']],step,
                        expected_hash=base.content_hash,scope_ids=('kitchen','study'));state.configured=True
                elif name=='build_house':
                    state.house_dir=step;state.graph,details=run_house_tools(base,step)
                elif name=='assemble_core':
                    state.assets_dir=step;state.graph,details=run_room_asset_tools(base,state.house_dir,step)
                    # Freeze original core verifier input beside the prepared assets.
                    (step/'core_source_graph.json').write_text(json.dumps(base.to_dict(),indent=2)+'\n')
                elif name=='add_surface_rigid_asset':
                    from rfw_w.scenes.orchestrator.surface_optional_rigid_tools import add_surface_rigid_asset
                    params=decision['parameters']
                    if params['category'] in state.optional_added:raise GraphError('FURNISHING_BUDGET_EXCEEDED','duplicate optional category')
                    state.graph,details=add_surface_rigid_asset(base,state.assets_dir,step,expected_hash=base.content_hash,**params)
                    state.optional_added.add(params['category']);state.assets_dir=step;state.verified_hash=None;state.styled_rooms.clear()
                elif name=='save_rigid_checkpoint':
                    from rfw_w.scenes.orchestrator.rigid_asset_tools import save_rigid_checkpoint
                    role=next(n['id'] for n in base.to_dict()['nodes'] if n.get('task_role')=='target')
                    details=save_rigid_checkpoint(base,state.assets_dir,step,role_id=role,expected_hash=base.content_hash)
                    state.rigid_checkpoint=step;state.rigid_saved_assets=state.assets_dir
                elif name=='restore_rigid_asset':
                    from rfw_w.scenes.orchestrator.rigid_asset_tools import restore_rigid_asset
                    state.graph,details=restore_rigid_asset(base,state.assets_dir,state.rigid_checkpoint,
                        state.rigid_saved_assets,step,expected_hash=base.content_hash)
                    state.assets_dir=step;state.rigid_restored=True;state.verified_hash=None
                elif name=='search_rigid_assets':
                    from rfw_w.scenes.assets.rigid_tool_retrieval import search_rigid_assets
                    params=decision['parameters']
                    if params['role_id']!=summary['rigid_role_id']:raise GraphError('RIGID_CATEGORY_UNSUPPORTED','target role')
                    details=search_rigid_assets(base,**params)
                    step.mkdir();(step/'rigid_query.json').write_text(json.dumps(details,indent=2)+'\n')
                    if not details['candidates']:raise GraphError('ASSET_UNAVAILABLE','no admitted rigid candidate')
                    state.rigid_query=details
                elif name=='select_rigid_asset':
                    from rfw_w.scenes.orchestrator.rigid_asset_tools import select_rigid_asset
                    state.graph,details=select_rigid_asset(base,state.assets_dir,step,state.rigid_query,
                        expected_hash=base.content_hash,**decision['parameters'])
                    state.assets_dir=step;state.rigid_selected=True;state.verified_hash=None
                elif name in ('create_surface_covering','move_surface_covering','remove_surface_covering'):
                    from rfw_w.scenes.contracts.scene_composition import migrate_covering_composition
                    from rfw_w.scenes.orchestrator.surface_covering_tools import edit_surface_covering
                    import shutil
                    source=state.assets_dir;current=base
                    if base.to_dict()['composition_schema'].endswith('/v2'):
                        current=migrate_covering_composition(base,expected_hash=base.content_hash)
                        migration=step/'migration';shutil.copytree(source,migration)
                        (migration/'graph.json').write_text(json.dumps(current.to_dict(),indent=2)+'\n');source=migration
                    target=step/'assets'
                    state.graph,details=edit_surface_covering(current,source,target,action=name.split('_')[0],
                        expected_hash=current.content_hash,**decision['parameters'])
                    state.assets_dir=target;state.verified_hash=None
                    if name=='remove_surface_covering':state.covering_created.discard(decision['parameters']['instance_id'])
                    else:state.covering_created.add(decision['parameters']['instance_id'])
                elif name=='search_surface_materials':
                    from rfw_w.scenes.assets.surface_materials import search_surface_materials
                    params=decision['parameters'];room=summary['style_room_id'];kind=summary['material_query_needed']
                    if params['surface_kind']!=kind:raise GraphError('AGENT_ACTION_INVALID','pending surface kind')
                    details=search_surface_materials(**params)
                    if not details['candidates']:raise GraphError('MATERIAL_NOT_FOUND','no matching local material')
                    state.style_queries[(room,kind)]=details;step.mkdir()
                    (step/'material_query.json').write_text(json.dumps(details,indent=2)+'\n')
                elif name=='set_room_surface_materials':
                    from rfw_w.scenes.orchestrator.surface_appearance_tools import set_surface_materials
                    room=summary['style_room_id']
                    state.graph,details=set_surface_materials(base,state.assets_dir,step,room_id=room,
                        expected_hash=base.content_hash,description='Agent-selected local room materials',
                        queries=tuple(state.style_queries[(room,k)] for k in ('floor','wall_set')),**decision['parameters'])
                    state.assets_dir=step;state.styled_rooms.add(room);state.verified_hash=None
                elif name=='save_checkpoint':
                    from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
                    save_scene_checkpoint(base,step,artifact_refs={'assets_ref':state.assets_dir.relative_to(output).as_posix(),
                        'sha256_by_ref':artifact_digest(state.assets_dir)})
                    state.checkpoint_dir=step;state.checkpoint_assets=state.assets_dir
                elif name in ('furnish_floor','mount_wall'):
                    kind='floor' if name=='furnish_floor' else 'wall'
                    state.graph,details=run_background_tools(base,state.assets_dir,step,kind,expected_hash=base.content_hash)
                    state.assets_dir=step;state.done.add(kind);state.verified_hash=None
                elif name=='verify_scene':
                    step.mkdir();assets=state.assets_dir
                    ledger=json.loads((assets/'background_ledger.json').read_text());core=json.loads((assets/'core_asset_ledger.json').read_text());plan=json.loads((assets/'geometry/house_plan.json').read_text())
                    source=TaskWorldGraph.from_dict(json.loads((assets/'input_graph.json').read_text()))
                    if self.inject_failure and not injected and state.done=={'floor'}:
                        # Acceptance-only fault: feed the original geometric verifier an
                        # in-room corridor intrusion. Source output stays immutable.
                        injected=True;candidate=json.loads(json.dumps(ledger));item=next(i for i in candidate['instances'] if i['placement_kind']=='floor')
                        item['pose_xyz_m']=[-2 if item['room_id']=='kitchen' else 2,2,0]
                        candidate.pop('content_hash');candidate['content_hash']=content_hash(candidate)
                        (step/'injected_candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
                        state.failure_scope=tuple(i['role_id'] for i in ledger['instances'] if i['placement_kind']=='floor')
                        verify_background(base,source,plan,core,candidate,assets)
                    accounted={i['role_id'] for i in ledger['instances']}|{u['role_id'] for u in ledger['unplaced_optional']}
                    check=json.loads(json.dumps(ledger))
                    for n in base.to_dict()['nodes']:
                        if n['kind']=='asset_role' and n.get('purpose')!='task_core' and n.get('placement_kind') in ('floor','wall') and n.get('physical_class')!='covering' and n['id'] not in accounted:
                            check['unplaced_optional'].append({'role_id':n['id'],'code':'not_run'})
                    check.pop('content_hash');check['content_hash']=content_hash(check)
                    details=verify_background(base,source,plan,core,check,assets)
                    if 'surface_coverings' in base.to_dict()['built']:
                        from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
                        details['surface_coverings']=verify_surface_coverings(assets)
                    if 'surface_appearance' in base.to_dict()['built']:
                        from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance
                        details['surface_appearance']=verify_surface_appearance(assets)
                    (step/'verification.json').write_text(json.dumps(details,indent=2)+'\n');state.verified_hash=base.content_hash
                elif name=='rollback_scope':
                    from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest,read_checkpoint
                    _,checkpoint=read_checkpoint(state.checkpoint_dir)
                    expected_artifacts=checkpoint['artifact_refs'].get('sha256_by_ref')
                    if expected_artifacts is not None:
                        actual_artifacts=artifact_digest(state.checkpoint_assets)
                        if any(actual_artifacts.get(k)!=v for k,v in expected_artifacts.items()):
                            raise GraphError('CHECKPOINT_TAMPERED','artifact digest')
                    state.graph,details=rollback_scene_scope(base,state.checkpoint_dir,step,expected_hash=base.content_hash,
                        scope_ids=state.failure_scope,reason_code=state.failure)
                    verification=restore_background_artifacts(state.assets_dir,state.checkpoint_assets,
                        step/'assets',state.graph,scope_ids=state.failure_scope)
                    nodes={n['id']:n for n in base.to_dict()['nodes']}
                    affected={nodes[k]['placement_kind'] for k in state.failure_scope if k in nodes}
                    state.assets_dir=step/'assets';state.done-=affected
                    state.failure=None;state.verified_hash=None
                    details['restoration_verification']=verification
                elif name=='finish':step.mkdir();terminal='passed'
                elif name=='terminate':step.mkdir();terminal='failed'
                else:raise GraphError('TOOL_NOT_AVAILABLE',name)
            except (GraphError,RoomAssetError,HouseToolError,SurfaceAppearanceError) as error:
                failed=True;state.graph=base;state.failure=error.code;state.verified_hash=None
                details={'reason_code':error.code}
                if not state.failure_scope:
                    state.failure_scope=tuple(n['id'] for n in base.to_dict()['nodes'] if n['kind']=='asset_role' and n.get('purpose')=='background')
            state.calls+=1
            event=_event(base,name,(),(time.monotonic()-tick)*1000,writer=TOOLS[name]['writes'] and not failed,
                         delta={'graph_hash':state.graph.content_hash} if TOOLS[name]['writes'] and not failed else None,
                         reason_code=state.failure if failed else None)
            record.update(transaction=event,result_graph_hash=state.graph.content_hash,details=details,
                          completed_background=sorted(state.done),failure_code=state.failure,terminal=terminal)
            step.mkdir(parents=True,exist_ok=True)
            (step/'agent_event.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
            (step/'state_graph.json').write_text(json.dumps(state.graph.to_dict(),indent=2)+'\n')
            events.append(record)
            if time.monotonic()-started>=self.deadline_s and terminal=='passed':
                terminal='budget_exhausted';state.failure='BUDGET_EXHAUSTED'
            (output/'events.json').write_text(json.dumps(events,indent=2,sort_keys=True)+'\n')
        report={'status':terminal,'failure_code':state.failure,'final_graph_hash':state.graph.content_hash,
                'tool_calls':state.calls,'model_requests_used':state.model_calls,'elapsed_s':round(time.monotonic()-started,3),
                'assets_ref':state.assets_dir.relative_to(output).as_posix() if state.assets_dir else None,
                'budgets':{'max_tools':self.max_tools,'max_model_requests':self.max_model_requests,'deadline_s':self.deadline_s},
                'surface_optionals':self.surface_optionals,'surface_style':self.surface_style,'rigid_restore':self.rigid_restore,'rigid_assets':self.rigid_assets,'coverings':self.coverings,'tool_path':[e['decision']['tool_name'] for e in events if 'decision' in e],
                'robot_navigation':'not_run','robot_ik':'not_run','robot_rollout':'not_run','injected_failure':injected}
        (output/'graph.json').write_text(json.dumps(state.graph.to_dict(),indent=2)+'\n')
        (output/'events.json').write_text(json.dumps(events,indent=2,sort_keys=True)+'\n')
        if terminal=='passed' and self.critic_repair:
            from rfw_w.scenes.orchestrator.critic_layout_repair import run_repair,RepairBudget
            repair=run_repair(state.assets_dir,output/'critic_repair',target_id='cup',
                budgets=RepairBudget(max_tools=min(12,max(0,self.max_tools-state.calls)),
                    deadline_s=max(.001,min(300,self.deadline_s-(time.monotonic()-started)))),
                robot_executor=self.robot_layout_executor,robot_semantics=self.robot_layout_semantics)
            report['critic_repair']=repair;report['tool_calls']+=repair['tool_calls']
            report['elapsed_s']=round(time.monotonic()-started,3)
            report['status']=repair['status'];report['failure_code']=repair['failure_code']
            if repair['final_scene_ref']:
                report['final_repaired_assets_ref']='critic_repair/'+repair['final_scene_ref']
            report['final_repaired_graph_hash']=repair['final_graph_hash']
        elif terminal=='passed' and self.robot_layout_executor is not None:
            from rfw_w.scenes.orchestrator.robot_layout_agent import RobotLayoutAgent,LayoutReplayProvider
            remaining=max(0,self.max_tools-state.calls)
            remaining_time=max(.001,self.deadline_s-(time.monotonic()-started))
            if hasattr(self.robot_layout_executor,'bind_assets'):
                self.robot_layout_executor.bind_assets(state.assets_dir)
            robot=RobotLayoutAgent(LayoutReplayProvider(),max_tools=remaining,deadline_s=remaining_time).run(
                state.graph,state.assets_dir,output/'robot_layout',executor=self.robot_layout_executor,
                semantics=self.robot_layout_semantics)
            report['robot_layout']=robot
            report['tool_calls']+=robot['tool_calls']
            report['elapsed_s']=round(time.monotonic()-started,3)
            report['robot_ik']='passed' if robot['state']['ik_ref'] else 'not_run'
            report['robot_approach_samples']='passed' if robot['state']['path_ref'] else 'not_run'
            if robot['status']!='passed':
                report['status']=robot['status'];report['failure_code']=robot['failure_code']
        (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        return report
