# SPDX-License-Identifier: Apache-2.0
"""Immutable candidate/apply transactions with a separate saved-difference verifier."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.layout_revision import require, verify_pose_difference
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.geometry.layout_revision import propose, measure_layout
from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
from rfw_w.scenes.orchestrator.rigid_asset_tools import _verify_bundle


def read(root, name):
    return json.loads((root/name).read_text())


def write(root, name, value):
    (root/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def rehash(value):
    value.pop('content_hash',None);value['content_hash']=content_hash(value)
    return value


def propose_layout_candidates(source, grant, operation, *, limit=8):
    graph,core,_=_verify_bundle(source)
    digests=artifact_digest(source)
    background=read(source,'background_ledger.json') if (source/'background_ledger.json').exists() else {'instances':[]}
    plan=read(source,'geometry/house_plan.json')
    report=propose(graph,core,background,plan,grant,operation,source_digest=content_hash(digests),limit=limit)
    require(artifact_digest(source)==digests,'LAYOUT_CANDIDATE_STALE','source changed during proposal')
    return report


def apply_layout_candidate(source, output, proposal, candidate_id, *, expected_hash):
    require(not output.exists(),'LAYOUT_INPUT_INVALID','output exists')
    graph,core,_=_verify_bundle(source);digests=artifact_digest(source)
    require(graph.content_hash==expected_hash==proposal['graph_hash'] and
            content_hash(digests)==proposal['source_digest'], 'LAYOUT_CANDIDATE_STALE','source')
    require(content_hash({k:v for k,v in proposal.items() if k!='content_hash'})==proposal['content_hash'],
            'LAYOUT_CANDIDATE_STALE','proposal hash')
    chosen=next((c for c in proposal['candidates'] if c['candidate_id']==candidate_id),None)
    require(chosen is not None,'LAYOUT_CANDIDATE_STALE','candidate id')
    require(content_hash({k:v for k,v in chosen.items() if k!='candidate_id'})==candidate_id,
            'LAYOUT_CANDIDATE_STALE','candidate digest')
    # Regenerate deterministically; no client-supplied, self-rehashed arbitrary delta.
    regenerated=propose_layout_candidates(source,proposal['permission'],proposal['operation'],limit=32)
    require(any(c==chosen for c in regenerated['candidates']),'LAYOUT_CANDIDATE_STALE','candidate not generated')
    shutil.copytree(source,output)
    shutil.copytree(source,output/'layout_source')
    write(output,'layout_proposal.json',proposal)
    raw=graph.to_dict();newcore=rehash(deepcopy(chosen['core']));bg=rehash(deepcopy(chosen['background']))
    raw['built']['asset_assembly']['ledger_hash']=newcore['content_hash']
    if 'background_assembly' in raw['built']:
        bg['core_ledger_hash']=newcore['content_hash'];rehash(bg)
        raw['built']['background_assembly']['ledger_hash']=bg['content_hash']
        write(output,'background_ledger.json',bg)
    for name in ('asset_ledger.json','core_asset_ledger.json'):write(output,name,newcore)
    for name in ('render','visual_validation','robot_validation','scene_critique','layout_revision'):
        raw['built'].pop(name,None)
    raw['evidence_refs']=[]
    # Appearance bindings survive, but their dependent core/source hashes are newly bound.
    appearance_source=graph.revise(raw,expected_hash=graph.content_hash)
    if 'surface_appearance' in raw['built']:
        appearance=read(output,'surface_appearance.json')
        appearance['source_graph_hash']=appearance_source.content_hash
        appearance['core_ledger_hash']=newcore['content_hash'];rehash(appearance)
        raw['built']['surface_appearance']['ledger_hash']=appearance['content_hash']
        write(output,'appearance_source_graph.json',appearance_source.to_dict())
        write(output,'surface_appearance.json',appearance)
    revised=graph.revise(raw,expected_hash=graph.content_hash)
    write(output,'graph.json',revised.to_dict())
    combined_before={**core,'instances':core['instances']+read(source,'background_ledger.json')['instances']}
    combined_after={**newcore,'instances':newcore['instances']+bg['instances']}
    check=verify_pose_difference(graph,combined_before,combined_after,proposal['permission'],proposal['operation'],
                                 component=proposal['operation']=='move_component')
    report={'schema_version':'pvscene.layout-revision/v1','status':'passed','base_graph_hash':graph.content_hash,
            'result_graph_hash':revised.content_hash,'candidate_id':candidate_id,'permission':proposal['permission'],
            'operation':proposal['operation'],'source_artifacts':digests,'difference':check,
            'model_requests':0,'robot':'not_run','render':'not_run','critic':'not_run'}
    write(output,'layout_revision.json',report)
    verification=verify_layout_revision(output)
    require(artifact_digest(source)==digests,'LAYOUT_CANDIDATE_STALE','source mutated')
    return revised,verification


def verify_layout_revision(root):
    report=read(root,'layout_revision.json');source=root/'layout_source'
    require(artifact_digest(source)==report['source_artifacts'],'LAYOUT_CANDIDATE_STALE','saved source digest')
    old,core,_=_verify_bundle(source);new,newcore,verification=_verify_bundle(root)
    proposal=read(root,'layout_proposal.json')
    require(proposal['permission']==report['permission'] and proposal['operation']==report['operation'] and
            proposal['source_digest']==content_hash(report['source_artifacts']) and
            content_hash({k:v for k,v in proposal.items() if k!='content_hash'})==proposal['content_hash'],
            'LAYOUT_CANDIDATE_STALE','proposal binding')
    regenerated=propose_layout_candidates(source,report['permission'],report['operation'],limit=32)
    candidate=next((c for c in regenerated['candidates'] if c['candidate_id']==report['candidate_id']),None)
    require(candidate is not None and candidate in proposal['candidates'],
            'LAYOUT_CANDIDATE_STALE','regenerated candidate')
    require(newcore==rehash(deepcopy(candidate['core'])), 'LAYOUT_SCOPE_VIOLATION','candidate core')
    require(old.content_hash==report['base_graph_hash'] and new.content_hash==report['result_graph_hash'] and
            new.to_dict()['parent_hash']==old.content_hash,'LAYOUT_CANDIDATE_STALE','version chain')
    bg=read(source,'background_ledger.json');newbg=read(root,'background_ledger.json')
    wanted_bg=deepcopy(candidate['background']);wanted_bg['core_ledger_hash']=newcore['content_hash'];rehash(wanted_bg)
    require(newbg==wanted_bg,'LAYOUT_SCOPE_VIOLATION','candidate background')
    difference=verify_pose_difference(old,{**core,'instances':core['instances']+bg['instances']},
        {**newcore,'instances':newcore['instances']+newbg['instances']},report['permission'],report['operation'],
        component=report['operation']=='move_component')
    require(difference==report['difference'],'LAYOUT_SCOPE_VIOLATION','difference record')
    for key in ('nodes','edges','initial_state','goals','steps','requirements'):
        require(old.to_dict()[key]==new.to_dict()[key],'LAYOUT_SCOPE_VIOLATION','graph '+key)
    for key,value in old.to_dict()['built'].items():
        if key not in ('asset_assembly','background_assembly','surface_appearance','render','visual_validation','robot_validation','scene_critique','layout_revision'):
            require(new.to_dict()['built'].get(key)==value,'LAYOUT_SCOPE_VIOLATION','built '+key)
    for key in ('render','visual_validation','robot_validation','scene_critique'):
        require(key not in new.to_dict()['built'],'LAYOUT_CANDIDATE_STALE','stale '+key)
    require(new.to_dict()['evidence_refs']==[],'LAYOUT_CANDIDATE_STALE','old evidence')
    measured=measure_layout(newcore,newbg,read(root,'geometry/house_plan.json'))
    current=artifact_digest(root)
    for ref,digest in report['source_artifacts'].items():
        if ref.startswith(('prepared/','geometry/','materials/','textures/')) or ref in ('core_source_graph.json','input_graph.json','surface_coverings.json'):
            require(current.get(ref)==digest,'LAYOUT_SCOPE_VIOLATION','source resource '+ref)
    if (root/'surface_appearance.json').exists():
        from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance
        verify_surface_appearance(root)
        original=read(source,'surface_appearance.json');actual=read(root,'surface_appearance.json')
        require({k:v for k,v in original.items() if k not in ('content_hash','core_ledger_hash','source_graph_hash')}==
                {k:v for k,v in actual.items() if k not in ('content_hash','core_ledger_hash','source_graph_hash')},
                'LAYOUT_SCOPE_VIOLATION','appearance')
    if (root/'surface_coverings.json').exists():
        from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
        verify_surface_coverings(root)
    return {'verified':True,'problems':[],'difference':difference,'assembly':verification,'geometry':measured,
            'source_unchanged':True,'robot':'not_run','model_requests':0}
