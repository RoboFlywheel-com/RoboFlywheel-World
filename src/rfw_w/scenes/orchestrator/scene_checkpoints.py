# SPDX-License-Identifier: Apache-2.0
"""Immutable cross-room graph checkpoints and dependency invalidation.

A rollback is a new revision, never a move of the history pointer to an old hash.
Restoring built background instances schedules their original validator to run again.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError,TaskWorldGraph
from rfw_w.scenes.orchestrator.room_asset_tools import _event
import hashlib
import shutil


def artifact_digest(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def tree_scope(raw,scope_ids):
    ids={n['id'] for n in raw['nodes']}; scope=set(scope_ids)
    if not scope<=ids:raise GraphError('ROLLBACK_SCOPE_INVALID','unknown nodes')
    while True:
        expanded=scope|{e['target_id'] for e in raw['edges'] if e['kind']=='contains' and e['source_id'] in scope}
        if expanded==scope:return scope
        scope=expanded


def save_scene_checkpoint(graph,output, *, artifact_refs=None):
    output.mkdir(parents=True,exist_ok=False)
    checkpoint={'schema_version':'pvscene.scene-checkpoint/v1','graph':graph.to_dict(),
                'graph_hash':graph.content_hash,'artifact_refs':artifact_refs or {}}
    checkpoint['content_hash']=content_hash(checkpoint)
    (output/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2,sort_keys=True)+'\n')
    (output/'transaction.json').write_text(json.dumps(_event(graph,'save_scene_checkpoint',(),0),indent=2)+'\n')
    return checkpoint['content_hash']


def read_checkpoint(directory,expected_hash=None):
    try:checkpoint=json.loads((directory/'checkpoint.json').read_text())
    except (OSError,ValueError):raise GraphError('CHECKPOINT_NOT_FOUND','missing checkpoint') from None
    digest=content_hash({k:v for k,v in checkpoint.items() if k!='content_hash'})
    if digest!=checkpoint.get('content_hash') or (expected_hash is not None and digest!=expected_hash):
        raise GraphError('CHECKPOINT_TAMPERED','digest')
    graph=TaskWorldGraph.from_dict(checkpoint['graph'])
    if graph.content_hash!=checkpoint['graph_hash']:raise GraphError('CHECKPOINT_TAMPERED','graph')
    return graph,checkpoint


def restore_background_artifacts(current_assets,checkpoint_assets,output,graph, *, scope_ids):
    """Restore only selected ledger branches in a fresh bundle and remeasure all survivors."""
    from rfw_w.scenes.orchestrator.background_tools import verify_background
    from rfw_w.scenes.agents.room_asset_visualization import render_room_assets_svg
    shutil.copytree(current_assets,output,ignore=shutil.ignore_patterns('graph.json','report.json',
        'background_report.json','background_ledger.json','input_graph.json','scene.svg','agent_event.json','state_graph.json'))
    def ledger(path):
        p=path/'background_ledger.json'
        return json.loads(p.read_text()) if p.exists() else {'instances':[],'queries':{},'attempts':{},'unplaced_optional':[]}
    current=ledger(current_assets);saved=ledger(checkpoint_assets);scope=set(scope_ids)
    merged={k:v for k,v in current.items() if k!='content_hash'}
    for key in ('instances','unplaced_optional'):
        merged[key]=[i for i in current[key] if i['role_id'] not in scope]+[i for i in saved[key] if i['role_id'] in scope]
    for key in ('queries','attempts'):
        merged[key]={k:v for k,v in current[key].items() if k not in scope}
        merged[key].update({k:v for k,v in saved[key].items() if k in scope})
    for i in saved['instances']:
        if i['role_id'] in scope:
            directory=Path('prepared')/i['asset_id']
            if not (output/directory).exists():shutil.copytree(checkpoint_assets/directory,output/directory)
    core=json.loads((output/'core_asset_ledger.json').read_text());plan=json.loads((output/'geometry/house_plan.json').read_text())
    merged.update(source_graph_hash=graph.content_hash,core_ledger_hash=core['content_hash'],house_plan_hash=plan['content_hash'])
    merged['content_hash']=content_hash(merged)
    check=deepcopy(merged);accounted={i['role_id'] for i in merged['instances']}|{u['role_id'] for u in merged['unplaced_optional']}
    for n in graph.to_dict()['nodes']:
        if n['kind']=='asset_role' and n.get('purpose')!='task_core' and n.get('placement_kind') in ('floor','wall') and n['id'] not in accounted:
            check['unplaced_optional'].append({'role_id':n['id'],'code':'not_run'})
    check.pop('content_hash');check['content_hash']=content_hash(check)
    verification=verify_background(graph,graph,plan,core,check,output)
    for filename,value in (('background_ledger.json',merged),('graph.json',graph.to_dict()),
                           ('input_graph.json',graph.to_dict()),('restoration_verification.json',verification)):
        (output/filename).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    (output/'scene.svg').write_text(render_room_assets_svg(plan,{**core,'instances':core['instances']+merged['instances']}))
    return verification


def rollback_scene_scope(current,checkpoint_dir,output, *, expected_hash,scope_ids,reason_code,checkpoint_hash=None):
    output.mkdir(parents=True,exist_ok=False)
    try:
        if current.content_hash!=expected_hash:raise GraphError('GRAPH_VERSION_CONFLICT','rollback current')
        saved,checkpoint=read_checkpoint(checkpoint_dir,checkpoint_hash)
        raw=current.to_dict();old=saved.to_dict()
        if raw['graph_id']!=old['graph_id']:raise GraphError('ROLLBACK_SCOPE_INVALID','different graph')
        if not isinstance(reason_code,str) or not reason_code.strip():raise GraphError('ROLLBACK_SCOPE_INVALID','reason required')
        # Background rollback may name optional roles or room IDs; room scopes are
        # expanded to optional branches so a room edit cannot erase its task core.
        room_ids={n['id'] for n in raw['nodes'] if n['kind']=='room'}
        roles={n['id']:n for n in raw['nodes'] if n['kind']=='asset_role'}
        scope=set(scope_ids)
        for rid in scope&room_ids:
            scope.remove(rid);scope.update(k for k,n in roles.items() if n['room_id']==rid and n['priority']=='optional')
            scope.update(n['id'] for n in old['nodes'] if n['kind']=='asset_role' and n['room_id']==rid and n['priority']=='optional')
        known={n['id'] for n in raw['nodes']}|{n['id'] for n in old['nodes']}
        if not scope or not scope<=known:raise GraphError('ROLLBACK_SCOPE_INVALID','empty or unknown scope')
        core_scope={k for k in scope if roles.get(k,{}).get('priority')=='core'}
        if core_scope:
            if any(roles[k].get('source')!='inferred' or roles[k].get('task_role')=='target' for k in core_scope):
                raise GraphError('ROLLBACK_CORE_VIOLATION','explicit core identity')
            if old['built']:
                raise GraphError('ROLLBACK_CORE_VIOLATION','core rollback requires prebuild checkpoint')
        for k in scope:
            n=next((n for n in raw['nodes']+old['nodes'] if n['id']==k),None)
            if n['kind']!='asset_role' or (n.get('priority')!='optional' and k not in core_scope):
                raise GraphError('ROLLBACK_SCOPE_INVALID',k)
        current_roots=scope&{n['id'] for n in raw['nodes']};old_roots=scope&{n['id'] for n in old['nodes']}
        now=tree_scope(raw,current_roots) if current_roots else set()
        then=tree_scope(old,old_roots) if old_roots else set()
        current_nodes=[n for n in raw['nodes'] if n['id'] in now]
        saved_nodes=[n for n in old['nodes'] if n['id'] in then]
        if current_nodes==saved_nodes and not core_scope:
            raise GraphError('ROLLBACK_SCOPE_INVALID','scope already equals checkpoint')
        raw['nodes']=[n for n in raw['nodes'] if n['id'] not in now]+[deepcopy(n) for n in old['nodes'] if n['id'] in then]
        raw['edges']=[e for e in raw['edges'] if e['source_id'] not in now and e['target_id'] not in now]
        raw['edges'] += [deepcopy(e) for e in old['edges'] if e['source_id'] in then or e['target_id'] in then]
        invalidated=['background_assembly','render','static_background_validation']
        raw['built'].pop('background_assembly',None)
        if core_scope:
            # Geometric graph hashes cover the whole plan. A core semantic rollback
            # requires rebuilding all downstream artifacts rather than retaining a
            # shell measured against a now incompatible inference.
            instances={n['id'] for n in raw['nodes'] if n['kind']=='asset_instance'}
            raw['nodes']=[n for n in raw['nodes'] if n['id'] not in instances]
            raw['edges']=[e for e in raw['edges'] if e['source_id'] not in instances and e['target_id'] not in instances]
            raw['built']={}
            invalidated=['house_shell','asset_assembly','background_assembly','render','robot_validation']
        raw['evidence_refs']=[]
        new=current.revise(raw,expected_hash=expected_hash)
        # All user/task semantics remain byte identical.
        if any(raw[k]!=old[k] for k in ('initial_state','steps','goals','requirements')):
            raise GraphError('ROLLBACK_CORE_VIOLATION','task semantics')
        report={'source_graph_hash':current.content_hash,'checkpoint_hash':checkpoint['content_hash'],
                'restored_from_graph_hash':saved.content_hash,'graph_hash':new.content_hash,
                'scope_ids':list(scope_ids),'restored_ids':sorted(then),'removed_ids':sorted(now-then),
                'reason_code':reason_code,'invalidated_evidence':invalidated,
                'reverify_tools':(['build_house','assemble_core','verify_floor_background','verify_wall_background']
                                  if core_scope else ['verify_floor_background','verify_wall_background']),
                'robot_validation':'not_run','status':'passed'}
        event=_event(current,'rollback_scene_scope',tuple(sorted(now|then)),0,writer=True,delta=report)
        (output/'graph.json').write_text(json.dumps(new.to_dict(),indent=2)+'\n')
        (output/'rollback.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'transaction.json').write_text(json.dumps(event,indent=2)+'\n')
        return new,report
    except GraphError as error:
        event=_event(current,'rollback_scene_scope',tuple(scope_ids),0,reason_code=error.code)
        (output/'transaction.json').write_text(json.dumps(event,indent=2)+'\n')
        raise
