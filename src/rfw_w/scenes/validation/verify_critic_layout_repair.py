# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_critique import require
from rfw_w.scenes.orchestrator.layout_revision_tools import read,write,verify_layout_revision
from rfw_w.scenes.orchestrator.scene_critic_tools import verify_critique
from rfw_w.scenes.orchestrator.critic_layout_repair import legal_tools
from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph


def verify(source,root):
    from rfw_w.scenes.validation.verify_agent_robot_layout import verify as verify_robot
    report=read(root,'report.json');current=source;review=None;repairs=0;measurements=[]
    require(content_hash({k:v for k,v in report.items() if k!='content_hash'})==report['content_hash'] and
            artifact_digest(source)==report['source_artifacts'],'CRITIC_SOURCE_CHANGED','source/report')
    for event in report['events']:
        require(event['graph_hash']==read_graph(current).content_hash,'STALE_EVIDENCE_REUSED','event graph')
        if 'artifact_ref' in event:
            step=(root/event['artifact_ref']).resolve()
            require(step.is_relative_to(root.resolve()),'CRITIC_REPAIR_SCOPE_VIOLATION','artifact path')
        name=event['tool']
        if name=='critique_scene':
            verify_critique(current,step);review=read(step,'critique.json')
        elif name=='propose_layout_candidates':
            require(name in legal_tools(review,repairs=repairs,max_repairs=report['budgets']['max_repairs']),
                    'CRITIC_REPAIR_SCOPE_VIOLATION','proposal preconditions')
        elif name=='apply_layout_candidate':
            require(review is not None and 'propose_layout_candidates' in legal_tools(review,repairs=repairs,max_repairs=report['budgets']['max_repairs']),
                    'CRITIC_REPAIR_SCOPE_VIOLATION','apply without review')
            checked=verify_layout_revision(step)
            require(read(step,'layout_revision.json')['base_graph_hash']==read_graph(current).content_hash and
                    read(step,'layout_revision.json')['source_artifacts']==artifact_digest(current),
                    'STALE_EVIDENCE_REUSED','apply source chain')
            from rfw_w.scenes.orchestrator.scene_critic_tools import support_margin
            before=support_margin(read(current,'core_asset_ledger.json'),report['target_id'])
            after=support_margin(read(step,'core_asset_ledger.json'),report['target_id'])
            require(event['before_margin_m']==before and event['after_margin_m']==after and after>before,
                    'CRITIC_REPAIR_NO_PROGRESS','measured improvement')
            measurements.append({'before_m':before,'after_m':after});current=step;review=None;repairs+=1
        elif name=='robot_layout':
            require(review is not None and legal_tools(review,repairs=repairs,max_repairs=report['budgets']['max_repairs'])==['finish'],
                    'STALE_EVIDENCE_REUSED','robot before current review')
            require(read(step,'source_artifacts.json')==artifact_digest(current),'STALE_EVIDENCE_REUSED','robot source')
            checked=verify_robot(step)
            if report['status']=='passed':require(checked['verified'],'CRITIC_INPUT_INVALID','robot measurements')
            else:
                raw=read(step,'report.json')
                require(raw['status']!='passed' and raw['failure_code']==report['failure_code'],
                        'CRITIC_INPUT_INVALID','failed robot report binding')
            require(read(step,'report.json')==report['robot_report'],
                    'CRITIC_INPUT_INVALID','nested robot report')
        elif name=='finish':
            require(legal_tools(review,repairs=repairs,max_repairs=report['budgets']['max_repairs'])==['finish'],
                    'STALE_EVIDENCE_REUSED','finish preconditions')
        else:raise ValueError('TOOL_NOT_AVAILABLE')
    require(read_graph(current).content_hash==report['final_graph_hash'] and repairs==report['repairs'],
            'STALE_EVIDENCE_REUSED','final state')
    require(report['tool_calls']<=report['budgets']['max_tools'] and repairs<=report['budgets']['max_repairs'],
            'BUDGET_EXHAUSTED','reported budget')
    require(report['status']!='passed' or report['events'][-1]['tool']=='finish',
            'STALE_EVIDENCE_REUSED','missing finish')
    return {'verified':True,'problems':[],'status':report['status'],'repairs':repairs,
            'improvements':measurements,'source_unchanged':True,'grasp':'not_run','rollout':'not_run'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-bundle',type=Path,required=True)
    p.add_argument('--delivery',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try:report=verify(a.scene_bundle,a.delivery)
    except (ValueError,OSError,KeyError,TypeError) as error:
        report={'verified':False,'failure_code':getattr(error,'code','CRITIC_REPAIR_SCOPE_VIOLATION'),'detail':str(error)}
    write(a.output_dir,'report.json',report);print(json.dumps(report));return 0 if report['verified'] else 1


if __name__=='__main__':raise SystemExit(main())
