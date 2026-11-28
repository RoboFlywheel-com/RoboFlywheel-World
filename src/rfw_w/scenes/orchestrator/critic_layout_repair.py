# SPDX-License-Identifier: Apache-2.0
"""Code-bounded critic/candidate/apply/reverify loop over immutable scene versions."""
from dataclasses import dataclass
import time
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.contracts.layout_revision import permission
from rfw_w.scenes.orchestrator.scene_critic_tools import critique_scene,support_margin,ModelBudget
from rfw_w.scenes.orchestrator.layout_revision_tools import (
    propose_layout_candidates,apply_layout_candidate,read,write,
)
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest


@dataclass
class RepairBudget:
    max_repairs:int=2
    max_tools:int=12
    deadline_s:float=300
    tools:int=0
    repairs:int=0
    def check(self,started):
        if self.tools>=self.max_tools or time.monotonic()-started>=self.deadline_s:
            raise GraphError('BUDGET_EXHAUSTED','repair tools/time')
    def spend(self,started):
        self.check(started);self.tools+=1


def legal_tools(review,*,repairs,max_repairs):
    if review is None:return ['critique_scene']
    measured=review['profile_results'].get('static',{})
    if measured.get('hard_pass') is not True:return ['terminate']
    if review['quality_pass'] is True:return ['finish']
    if repairs>=max_repairs:return ['terminate']
    issues=review['issues']
    if issues and all(i['code']=='SUPPORT_EDGE_MARGIN' and i['suggested_operation']=='translate' for i in issues):
        return ['propose_layout_candidates','terminate']
    return ['terminate']


def available_tools(phase,review,*,repairs,max_repairs):
    if phase=='review':return ['critique_scene']
    if phase=='apply':return ['apply_layout_candidate','terminate']
    return legal_tools(review,repairs=repairs,max_repairs=max_repairs)


def public_summary(graph,names,scope_ids):
    from rfw_w.scenes.orchestrator.tool_capabilities import describe_tools,group_tools
    tools=describe_tools('critic_repair',names)
    return {'graph_hash':graph.content_hash,'scope_ids':list(scope_ids),
            'available_tools':tools,'tool_groups':group_tools(tools)}


def run_repair(source,output,*,target_id='cup',budgets=None,model_budget=None,robot_executor=None,
               robot_semantics=None,provider=None):
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();budget=budgets or RepairBudget()
    models=model_budget or ModelBudget(0);frozen=artifact_digest(source);current=source
    events=[];review=None;seen=set();status='failed';failure=None;robot_report=None
    try:
        while True:
            budget.spend(started);graph=read_graph(current)
            critic_dir=output/f'step_{budget.tools:03d}_critique'
            review=critique_scene(current,critic_dir,scope_ids=[target_id],profiles=['static'],
                                  expected_version=graph.content_hash,budget=models)
            events.append({'tool':'critique_scene','graph_hash':graph.content_hash,
                           'artifact_ref':critic_dir.relative_to(output).as_posix(),'legal_tools':['critique_scene'],
                           'summary':public_summary(graph,['critique_scene'],[target_id])})
            legal=legal_tools(review,repairs=budget.repairs,max_repairs=budget.max_repairs)
            if 'finish' in legal:
                if robot_executor:
                    budget.spend(started);robot_dir=output/f'step_{budget.tools:03d}_robot_layout'
                    from rfw_w.scenes.orchestrator.robot_layout_agent import RobotLayoutAgent,LayoutReplayProvider
                    robot_executor.bind_assets(current)
                    robot_report=RobotLayoutAgent(LayoutReplayProvider(),max_tools=max(0,budget.max_tools-budget.tools-1),
                        deadline_s=max(.001,budget.deadline_s-(time.monotonic()-started))).run(
                            graph,current,robot_dir,executor=robot_executor,semantics=robot_semantics)
                    events.append({'tool':'robot_layout','graph_hash':graph.content_hash,
                                   'artifact_ref':robot_dir.relative_to(output).as_posix()})
                    budget.tools+=robot_report['tool_calls']
                    if robot_report['status']!='passed':
                        failure=robot_report['failure_code'];break
                budget.spend(started);events.append({'tool':'finish','graph_hash':graph.content_hash,'legal_tools':legal})
                status='passed';break
            if legal==['terminate']:
                failure='BUDGET_EXHAUSTED' if budget.repairs>=budget.max_repairs else 'CRITIC_REPAIR_SCOPE_VIOLATION'
                break
            signature=content_hash({'graph_hash':graph.content_hash,'issues':review['issues']})
            if signature in seen:raise GraphError('CRITIC_REPAIR_NO_PROGRESS','repeated state/issues')
            seen.add(signature)
            budget.spend(started)
            node=next(n for n in graph.to_dict()['nodes'] if n['id']==target_id)
            grant=permission(graph,target_id,operations=['translate'],scope_ids=[node['room_id']])
            proposal=propose_layout_candidates(current,grant,'translate',limit=32)
            proposed_dir=output/f'step_{budget.tools:03d}_candidates';proposed_dir.mkdir()
            write(proposed_dir,'candidates.json',proposal)
            events.append({'tool':'propose_layout_candidates','graph_hash':graph.content_hash,
                           'artifact_ref':proposed_dir.relative_to(output).as_posix(),'legal_tools':legal,
                           'summary':public_summary(graph,legal,[target_id])})
            old_margin=support_margin(read(current,'core_asset_ledger.json'),target_id)
            improving=[c for c in proposal['candidates'] if support_margin(c['core'],target_id)>old_margin+1e-6]
            if not improving:raise GraphError('CRITIC_REPAIR_NO_PROGRESS','no improving candidate')
            # Prefer the smallest measured fix meeting the review threshold. Maximizing
            # edge margin can move a valid task target beyond the mounted robot workspace.
            original=next(i for i in read(current,'core_asset_ledger.json')['instances'] if i['role_id']==target_id)
            satisfactory=[c for c in improving if support_margin(c['core'],target_id)>=.06]
            def displacement(candidate):
                item=next(i for i in candidate['core']['instances'] if i['role_id']==target_id)
                return sum((a-b)**2 for a,b in zip(item['pose_xyz_m'],original['pose_xyz_m']))
            selected=min(satisfactory,key=displacement) if satisfactory else max(improving,key=lambda c:support_margin(c['core'],target_id))
            if provider:
                chosen=provider({'graph_hash':graph.content_hash,'issue_ids':[i['id'] for i in review['issues']],
                    'candidates':[{'candidate_id':c['candidate_id'],'edge_margin_m':support_margin(c['core'],target_id)} for c in improving]})
                selected=next((c for c in improving if c['candidate_id']==chosen),None)
                if selected is None:raise GraphError('CRITIC_REPAIR_SCOPE_VIOLATION','illegal selected candidate')
            budget.spend(started);step=output/f'step_{budget.tools:03d}_apply'
            revised,checked=apply_layout_candidate(current,step,proposal,selected['candidate_id'],expected_hash=graph.content_hash)
            events.append({'tool':'apply_layout_candidate','graph_hash':graph.content_hash,'result_graph_hash':revised.content_hash,
                'candidate_id':selected['candidate_id'],'artifact_ref':step.relative_to(output).as_posix(),
                'before_margin_m':old_margin,'after_margin_m':support_margin(selected['core'],target_id)})
            budget.repairs+=1;current=step;review=None
    except (ValueError,OSError,KeyError,TypeError) as error:
        failure=getattr(error,'code','CRITIC_REPAIR_SCOPE_VIOLATION')
    unchanged=artifact_digest(source)==frozen
    if not unchanged:status='failed';failure='CRITIC_SOURCE_CHANGED'
    report={'status':status,'failure_code':failure,'target_id':target_id,'source_artifacts':frozen,
        'source_graph_hash':read_graph(source).content_hash,'final_graph_hash':read_graph(current).content_hash,
        'final_scene_ref':current.relative_to(output).as_posix() if current.is_relative_to(output) else None,
        'events':events,'tool_calls':budget.tools,'repairs':budget.repairs,'model_requests':models.used,
        'elapsed_s':round(time.monotonic()-started,3),'source_unchanged':unchanged,
        'budgets':{'max_repairs':budget.max_repairs,'max_tools':budget.max_tools,'deadline_s':budget.deadline_s},
        'robot_report':robot_report,'grasp':'not_run','rollout':'not_run'}
    report['content_hash']=content_hash(report);write(output,'report.json',report)
    return report
