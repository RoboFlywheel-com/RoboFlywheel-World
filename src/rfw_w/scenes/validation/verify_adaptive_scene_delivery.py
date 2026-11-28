#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent event legality, version chain, rollback and real mesh evidence replay."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from rfw_w.scenes.orchestrator.adaptive_scene_agent import AgentState,available_tools,validate_decision,furnishing_options
from rfw_w.scenes.orchestrator.background_tools import verify_background
from rfw_w.scenes.agents.composition_revision import validate_composition_patch
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph,GraphError
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.orchestrator.scene_checkpoints import read_checkpoint


def read(path):return json.loads(path.read_text())

def verify_run(directory):
    report=read(directory/'report.json');events=read(directory/'events.json');source=TaskWorldGraph.from_dict(read(directory/'source_graph.json'))
    state=AgentState(source,surface_optionals=report.get('surface_optionals',False),rigid_restore=report.get('rigid_restore',False),style_enabled=report.get('surface_style',False),rigid_enabled=report.get('rigid_assets',False),coverings_enabled=report.get('coverings',False));failures=0;rollbacks=0;checks=0;boundaries=[];model_calls=0
    for i,event in enumerate(events):
        if 'decision_failure' in event:raise GraphError('AGENT_ACTION_INVALID','decision failure in accepted run')
        legal=available_tools(state)
        if legal!=event['available_tools']:raise GraphError('TOOL_NOT_AVAILABLE','recorded availability')
        name=validate_decision(event['decision'],legal);step=directory/event['artifact_dir']
        if event.get('decision_source')=='model':model_calls+=1
        if event.get('decision_source')=='dependency' and len(legal)!=1:
            raise GraphError('AGENT_ACTION_INVALID','multiple choices executed as dependency')
        if event['source_graph_hash']!=state.graph.content_hash:raise GraphError('GRAPH_VERSION_CONFLICT','event chain')
        tx=event['transaction'];req=tx['request'];res=tx['result']
        if req['scene_version']!=state.graph.content_hash or res['base_scene_version']!=req['scene_version'] or res['request_id']!=req['request_id']:
            raise GraphError('GRAPH_VERSION_CONFLICT','typed transaction')
        if req['tool_version'].split('/')[0]!=name or res['tool_version']!=req['tool_version']:
            raise GraphError('TOOL_NOT_AVAILABLE','transaction tool identity')
        if res['read_only'] and res.get('proposed_delta') is not None:
            raise GraphError('AGENT_ACTION_INVALID','read tool proposed mutation')
        actual=TaskWorldGraph.from_dict(read(step/'state_graph.json'))
        if actual.content_hash!=event['result_graph_hash']:raise GraphError('GRAPH_VERSION_CONFLICT','saved state')
        if res['status']=='failed':
            failures+=1
            if actual.content_hash!=state.graph.content_hash:raise GraphError('GRAPH_VERSION_CONFLICT','failed writer changed graph')
            state.failure=event['failure_code'];state.verified_hash=None
            if name=='verify_scene' and (step/'injected_candidate.json').exists():
                root=state.assets_dir;core=read(root/'core_asset_ledger.json');plan=read(root/'geometry/house_plan.json');ledger=read(step/'injected_candidate.json')
                base=TaskWorldGraph.from_dict(read(root/'input_graph.json'))
                try:verify_background(state.graph,base,plan,core,ledger,root)
                except ValueError as e:
                    if getattr(e,'code',None)!=state.failure:raise GraphError('AGENT_ACTION_INVALID','failure replay mismatch')
                else:raise GraphError('AGENT_ACTION_INVALID','injected failure did not fail')
        else:
            if actual.content_hash!=state.graph.content_hash and actual.to_dict()['parent_hash']!=state.graph.content_hash and not (step/'migration/graph.json').exists() and name!='add_surface_rigid_asset':
                raise GraphError('GRAPH_VERSION_CONFLICT','new parent')
            if name=='revise_composition':
                replay=validate_composition_patch(state.graph,furnishing_options()[event['decision']['parameters']['patch_index']],
                    expected_hash=state.graph.content_hash,scope_ids=('kitchen','study'))
                if replay.content_hash!=actual.content_hash:raise GraphError('COMPOSITION_PATCH_INVALID','patch replay')
                state.configured=True
            elif name=='build_house':state.house_dir=step
            elif name=='assemble_core':state.assets_dir=step
            elif name=='add_surface_rigid_asset':
                from rfw_w.scenes.validation.verify_mixed_provider_assembly import verify
                verify(step)
                params=event['decision']['parameters']
                addition=read(step/'optional_rigid_report.json')
                if addition['role_id']!=params['role_id'] or addition['surface_id']!=params['surface_id']:
                    raise GraphError('ROLLBACK_CORE_VIOLATION','optional decision')
                if params['category'] in state.optional_added:raise GraphError('FURNISHING_BUDGET_EXCEEDED','duplicate category')
                state.optional_added.add(params['category']);state.assets_dir=step;state.verified_hash=None;state.styled_rooms.clear()
            elif name=='save_rigid_checkpoint':
                from rfw_w.scenes.geometry.rigid_asset_checkpoint import verify_checkpoint_payload
                from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
                verify_checkpoint_payload(read(step/'checkpoint.json'),state.graph,artifact_digest(state.assets_dir))
                state.rigid_checkpoint=step;state.rigid_saved_assets=state.assets_dir
            elif name=='restore_rigid_asset':
                from rfw_w.scenes.geometry.rigid_asset_checkpoint import verify_checkpoint_payload,verify_binding_scope
                from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
                from rfw_w.scenes.orchestrator.rigid_asset_tools import _verify_bundle
                saved_graph,saved_core,_=_verify_bundle(state.rigid_saved_assets)
                _,current_core,_=_verify_bundle(state.assets_dir)
                cp=read(state.rigid_checkpoint/'checkpoint.json')
                verify_checkpoint_payload(cp,saved_graph,artifact_digest(state.rigid_saved_assets))
                verify_binding_scope(state.graph.to_dict(),saved_graph.to_dict(),current_core,saved_core,cp['role_id'])
                _,restored_core,_=_verify_bundle(step)
                for key in ('instances','queries','selection','goal_placement'):
                    if restored_core[key]!=saved_core[key]:raise GraphError('ROLLBACK_CORE_VIOLATION','restored '+key)
                state.assets_dir=step;state.rigid_restored=True;state.verified_hash=None
            elif name=='search_rigid_assets':
                query=read(step/'rigid_query.json')
                if query['source_graph_hash']!=state.graph.content_hash or content_hash({k:v for k,v in query.items() if k!='query_id'})!=query['query_id']:
                    raise GraphError('RIGID_QUERY_STALE','query chain')
                state.rigid_query=query
            elif name=='select_rigid_asset':
                from rfw_w.scenes.validation.verify_mixed_provider_assembly import verify
                verify(step)
                key=event['decision']['parameters']['candidate_key']
                if key not in {c['candidate_key'] for c in state.rigid_query['candidates']}:raise GraphError('RIGID_CANDIDATE_NOT_ADMITTED','selected candidate')
                state.assets_dir=step;state.rigid_selected=True;state.verified_hash=None
            elif name in ('create_surface_covering','move_surface_covering','remove_surface_covering'):
                from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
                root=step/'assets';verify_surface_coverings(root)
                if (step/'migration/graph.json').exists():
                    migrated=TaskWorldGraph.from_dict(read(step/'migration/graph.json'))
                    if migrated.to_dict()['parent_hash']!=state.graph.content_hash or actual.to_dict()['parent_hash']!=migrated.content_hash:
                        raise GraphError('GRAPH_VERSION_CONFLICT','covering migration chain')
                state.assets_dir=root;state.verified_hash=None
                iid=event['decision']['parameters']['instance_id']
                if name=='remove_surface_covering':state.covering_created.discard(iid)
                else:state.covering_created.add(iid)
            elif name=='search_surface_materials':
                from rfw_w.scenes.orchestrator.adaptive_scene_agent import public_summary
                summary=public_summary(state,legal);query=read(step/'material_query.json')
                if query['description']!=event['decision']['parameters']['description'] or query['surface_kind']!=summary['material_query_needed']:
                    raise GraphError('AGENT_ACTION_INVALID','material query')
                state.style_queries[(summary['style_room_id'],query['surface_kind'])]=query
            elif name=='set_room_surface_materials':
                from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance
                from rfw_w.scenes.orchestrator.adaptive_scene_agent import public_summary
                summary=public_summary(state,legal);verify_surface_appearance(step)
                params=event['decision']['parameters']
                for kind,key in (('floor','floor_material_id'),('wall_set','wall_material_id')):
                    if params[key] not in {c['material_id'] for c in state.style_queries[(summary['style_room_id'],kind)]['candidates']}:
                        raise GraphError('MATERIAL_NOT_FOUND','decision outside candidates')
                state.styled_rooms.add(summary['style_room_id']);state.assets_dir=step;state.verified_hash=None
            elif name=='save_checkpoint':
                from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
                _,cp=read_checkpoint(step)
                hashes=cp['artifact_refs'].get('sha256_by_ref')
                actual_files=artifact_digest(state.assets_dir)
                if hashes and any(actual_files.get(k)!=v for k,v in hashes.items()):
                    raise GraphError('CHECKPOINT_TAMPERED','saved artifacts changed')
                state.checkpoint_dir=step;state.checkpoint_assets=state.assets_dir
            elif name in ('furnish_floor','mount_wall'):
                state.assets_dir=step;state.done.add('floor' if name=='furnish_floor' else 'wall');state.verified_hash=None
            elif name=='verify_scene':
                v=read(step/'verification.json')
                if not v['verified']:raise GraphError('AGENT_ACTION_INVALID','forged validation')
                root=state.assets_dir;ledger=read(root/'background_ledger.json')
                core=read(root/'core_asset_ledger.json');plan=read(root/'geometry/house_plan.json')
                base=TaskWorldGraph.from_dict(read(root/'input_graph.json'))
                accounted={i['role_id'] for i in ledger['instances']}|{u['role_id'] for u in ledger['unplaced_optional']}
                for n in state.graph.to_dict()['nodes']:
                    if n['kind']=='asset_role' and n.get('purpose')!='task_core' and n.get('placement_kind') in ('floor','wall') and n.get('physical_class')!='covering' and n['id'] not in accounted:
                        ledger['unplaced_optional'].append({'role_id':n['id'],'code':'not_run'})
                ledger.pop('content_hash');ledger['content_hash']=content_hash(ledger)
                replay=verify_background(actual,base,plan,core,ledger,root)
                if 'surface_appearance' in actual.to_dict()['built']:
                    from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance
                    replay['surface_appearance']=verify_surface_appearance(root)
                if 'surface_coverings' in actual.to_dict()['built']:
                    from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
                    replay['surface_coverings']=verify_surface_coverings(root)
                if replay!=v:raise GraphError('AGENT_ACTION_INVALID','verification replay')
                state.verified_hash=actual.content_hash;checks+=1
            elif name=='rollback_scope':
                rollback=read(step/'rollback.json');rollbacks+=1
                from rfw_w.scenes.orchestrator.scene_checkpoints import rollback_scene_scope
                import tempfile
                with tempfile.TemporaryDirectory(prefix='pvscene_verify_rollback_') as temp:
                    replay,_=rollback_scene_scope(state.graph,state.checkpoint_dir,Path(temp)/'rollback',
                        expected_hash=state.graph.content_hash,scope_ids=tuple(rollback['scope_ids']),reason_code=rollback['reason_code'])
                if replay.content_hash!=actual.content_hash:raise GraphError('ROLLBACK_SCOPE_INVALID','rollback replay')
                if not rollback['invalidated_evidence'] or not rollback['reverify_tools']:
                    raise GraphError('STALE_EVIDENCE_REUSED','rollback missing invalidation')
                state.failure=None;state.verified_hash=None;state.assets_dir=step/'assets'
                previous={n['id']:n for n in state.graph.to_dict()['nodes']}
                state.done-={previous[k]['placement_kind'] for k in rollback['scope_ids'] if k in previous}
                read(state.assets_dir/'restoration_verification.json')
            elif name=='finish':
                if state.verified_hash!=actual.content_hash:raise GraphError('STALE_EVIDENCE_REUSED','finish')
        state.graph=actual
        state.calls+=1
        if state.done!=set(event['completed_background']):raise GraphError('AGENT_ACTION_INVALID','state transition')
        boundaries.append(state.graph.content_hash)
    if state.graph.content_hash!=report['final_graph_hash'] or report['status']!='passed':raise GraphError('GRAPH_VERSION_CONFLICT','final report')
    robot_tools=0
    repair_tools=0
    if 'critic_repair' in report:
        from rfw_w.scenes.validation.verify_critic_layout_repair import verify as verify_repair
        repair_root=directory/'critic_repair'
        repaired=read(repair_root/'report.json')
        if repaired!=report['critic_repair'] or repaired['source_graph_hash']!=state.graph.content_hash:
            raise GraphError('STALE_EVIDENCE_REUSED','repair phase binding')
        result=verify_repair(directory/report['assets_ref'],repair_root)
        if not result['verified'] or result['status']!='passed':
            raise GraphError('STALE_EVIDENCE_REUSED','repair phase failed')
        if repaired['final_graph_hash']!=report['final_repaired_graph_hash']:
            raise GraphError('STALE_EVIDENCE_REUSED','repaired version')
        repair_tools=repaired['tool_calls']
    if 'robot_layout' in report:
        from rfw_w.scenes.validation.verify_agent_robot_layout import verify as verify_robot
        robot_root=directory/'robot_layout'
        robot_report=read(robot_root/'report.json')
        if robot_report!=report['robot_layout'] or robot_report['source_graph_hash']!=state.graph.content_hash:
            raise GraphError('STALE_EVIDENCE_REUSED','robot phase binding')
        verify_robot(robot_root)
        robot_tools=robot_report['tool_calls']
    if report['tool_calls']!=len(events)+robot_tools+repair_tools:raise GraphError('AGENT_ACTION_INVALID','tool denominator')
    if model_calls!=report['model_requests_used'] and any('decision_source' in e for e in events):
        raise GraphError('AGENT_ACTION_INVALID','model denominator')
    budget=report.get('budgets')
    if budget and (report['tool_calls']>budget['max_tools'] or report['model_requests_used']>budget['max_model_requests'] or report['elapsed_s']>budget['deadline_s']):
        raise GraphError('BUDGET_EXHAUSTED','accepted run exceeded budget')
    final=TaskWorldGraph.from_dict(read(directory/'graph.json'));root=directory/report['assets_ref']
    core=read(root/'core_asset_ledger.json');ledger=read(root/'background_ledger.json');plan=read(root/'geometry/house_plan.json')
    base=TaskWorldGraph.from_dict(read(root/'input_graph.json'))
    v=verify_background(final,base,plan,core,ledger,root)
    if 'surface_appearance' in final.to_dict()['built']:
        from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance
        verify_surface_appearance(root)
    if final.to_dict()['built'].get('background_assembly',{}).get('ledger_hash')!=ledger['content_hash']:
        raise GraphError('ASSET_LEDGER_MISMATCH','final ledger binding')
    if 'surface_coverings' in final.to_dict()['built']:
        from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
        verify_surface_coverings(root)
    bindings={e['target_id']:e['source_id'] for e in final.to_dict()['edges'] if e['kind']=='role_bound_to'}
    for item in core['instances']+ledger['instances']:
        if bindings.get(item['instance_id'])!=item['role_id']:raise GraphError('ASSET_BINDING_INVALID','final graph')
    if any(final.to_dict()[k]!=source.to_dict()[k] for k in ('initial_state','steps','goals','requirements')):
        raise GraphError('TASK_GOAL_CHANGED','end to end task')
    return {'verified':True,'problems':[],'tools':len(events),'failed_tools':failures,'rollbacks':rollbacks,
        'verification_calls':checks,'floor_instances':v['floor'],'wall_instances':v['wall'],'unique_versions':len(set(boundaries)),
        'model_requests_used':report['model_requests_used'],'robot_validation':'layout_only' if robot_tools or
        (report.get('critic_repair',{}).get('robot_report') or {}).get('status')=='passed' else 'not_run'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    a=p.parse_args();runs=[a.directory] if (a.directory/'events.json').exists() else [r for r in a.directory.iterdir() if r.is_dir() and (r/'events.json').exists()]
    results={r.name:verify_run(r) for r in runs};report={'verified':bool(results),'problems':[],'runs':results}
    if a.output_dir:
        a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return 0 if report['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
