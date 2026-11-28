# SPDX-License-Identifier: Apache-2.0
"""State-derived local tool loop for reference task production.

Decision providers see semantic roles, legal tools and measured failure summaries.
Executors own paths, geometry, binding and the actual physics/data artifacts.
This explicit reference preset does not expand the existing two-room agent backend.
"""
import json
from copy import deepcopy
from pathlib import Path
from rfw_w.scenes.contracts.hashing import content_hash


def available_tools(state):
    if state['status'] in ('passed','failed','budget_exhausted'):return []
    if not state['world_built']:return ['build_reference_world']
    if not state['world_verified']:return ['verify_reference_world']
    if state['recording_pending']:return ['verify_reference_episode']
    if state['failure']:return ['revise_motion_parameters','terminate']
    if state['passed_resets']>=state['required_resets']:return ['export_reference_delivery']
    return ['execute_reference_episode']


def run_reference_agent(graph, output, *, provider, executor, parameters, required_resets=1,
                        max_calls=18, max_revisions=3):
    if isinstance(required_resets,bool) or not isinstance(required_resets,int) or required_resets<1:
        raise ValueError('REFERENCE_RESET_COUNT_INVALID')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',world_built=False,world_verified=False,recording_pending=False,
               passed_resets=0,required_resets=required_resets,failure=None,revisions=0)
    parameters=deepcopy(parameters);events=[];seen={content_hash(parameters)}
    for call in range(max_calls):
        legal=available_tools(state)
        summary=dict(graph_hash=graph.content_hash,
            roles=[{'id':n['id'],'category':n.get('category')} for n in graph.to_dict()['nodes'] if n['kind']=='asset_role'],
            available_tools=legal,passed_resets=state['passed_resets'],required_resets=required_resets,
            failure=deepcopy(state['failure']),parameters=deepcopy(parameters),revisions=state['revisions'])
        action=provider(summary)
        name=action.get('tool') if isinstance(action,dict) else None
        if name not in legal:raise ValueError('REFERENCE_AGENT_TOOL_NOT_LEGAL')
        if name=='revise_motion_parameters':
            proposed=deepcopy(action.get('parameters'))
            if not isinstance(proposed,dict) or state['revisions']>=max_revisions or content_hash(proposed) in seen:
                state['status']='budget_exhausted';break
            # Revisions change motion only. The source world/task and all criteria stay locked.
            denied={'max_penetration_m','physics_hz','contact_timeconst_s','diagnostic_checkpoint_ref'}
            if any(proposed.get(k)!=parameters.get(k) for k in denied):raise ValueError('REFERENCE_AGENT_PROTECTED_PARAMETER')
            parameters=proposed;seen.add(content_hash(parameters));state['revisions']+=1
            state.update(failure=None,passed_resets=0)
            result={'status':'revised','parameter_hash':content_hash(parameters),'invalidated':['dynamic','data','media']}
        elif name=='terminate':
            state['status']='failed';result={'status':'failed'}
        else:
            try:
                result=executor(name,deepcopy(state),deepcopy(parameters))
            except (OSError,ValueError,RuntimeError) as error:
                state['status']='failed'
                result={'status':'failed','failure_code':type(error).__name__,
                        'errno':getattr(error,'errno',None),'tool':name}
                events.append(dict(call=call,tool=name,result=result,state=deepcopy(state)))
                (output/'tool_events.json').write_text(json.dumps(events,indent=2))
                break
            if name=='build_reference_world':state['world_built']=True
            elif name=='verify_reference_world':
                state['world_verified']=result['status']=='passed'
                if not state['world_verified']:state['status']='failed'
            elif name=='execute_reference_episode':state['recording_pending']=True
            elif name=='verify_reference_episode':
                state['recording_pending']=False
                if result['status']=='passed':state['passed_resets']+=1
                else:state['failure']={'checks':[k for k,v in result['checks'].items() if not v], 'metrics':result.get('metrics',{})}
            elif name=='export_reference_delivery':state['status']='passed'
        events.append(dict(call=call,tool=name,reason=action.get('reason','local_policy'),
            parameter_hash=content_hash(parameters),result=deepcopy(result),state=deepcopy(state)))
        (output/'tool_events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
        if state['status']!='running':break
    else:state['status']='budget_exhausted'
    report=dict(schema_version='pvscene.reference-agent/v1',status=state['status'],graph_hash=graph.content_hash,
        passed_resets=state['passed_resets'],required_resets=required_resets,revisions=state['revisions'],
        tool_calls=len(events),model_requests=0,decision_mode='local_code_provider',
        scene_selection='admitted_workshop_reference_preset; general semantic retrieval unavailable')
    (output/'agent_report.json').write_text(json.dumps(report,indent=2))
    return report


class LocalReferenceProvider:
    """Measured-state tool choices; optional explicit failure-indexed parameter revisions."""
    def __init__(self, revisions=()):self.revisions=list(revisions)
    def __call__(self,summary):
        legal=summary['available_tools']
        if 'revise_motion_parameters' in legal:
            if self.revisions:return {'tool':'revise_motion_parameters','parameters':self.revisions.pop(0),'reason':'measured_failure_revision'}
            return {'tool':'terminate','reason':'no_admitted_revision'}
        return {'tool':legal[0],'reason':'state_precondition_satisfied'}
