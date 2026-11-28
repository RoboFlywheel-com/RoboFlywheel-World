# SPDX-License-Identifier: Apache-2.0
"""Add one measured optional rigid object on an existing task support surface."""
from copy import deepcopy
import json,shutil
from pathlib import Path
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.surface_appearance import require
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.contracts.asset import AffordanceStatus
from rfw_w.scenes.geometry.optional_rigid_scope import verify_optional_extension,verify_optional_ledger
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.assets.rigid_tool_retrieval import search_rigid_assets,prepare_selected
from rfw_w.scenes.orchestrator.rigid_asset_tools import _verify_bundle,_read
from rfw_w.scenes.orchestrator.room_asset_tools import _event
from rfw_w.scenes.solvers.layout import Footprint,OccupancyMap,SurfacePlacementConfig,place_on_surface
from rfw_w.scenes.solvers.support import Surface


def _extend(raw,surface_id,role_id,category):
    new=deepcopy(raw);nodes={n['id']:n for n in new['nodes']}
    require(role_id not in nodes and 'inst_'+role_id not in nodes,'COMPOSITION_ID_CONFLICT',role_id)
    require(category in {'bowl','jar'},'RIGID_CATEGORY_UNSUPPORTED',category)
    surface=nodes.get(surface_id)
    require(surface and surface['kind']=='surface','COMPOSITION_ANCHOR_INVALID',surface_id)
    parent={e['target_id']:e['source_id'] for e in new['edges'] if e['kind']=='contains'}
    policies=[n for n in new['nodes'] if n['kind']=='distractor_policy' and parent.get(n['id'])==surface_id]
    require(len(policies)==1,'COMPOSITION_ANCHOR_INVALID','policy')
    policy=policies[0];policy['candidate_categories']=list(dict.fromkeys(policy['candidate_categories']+[category]))
    policy['max_count']+=1
    role={'id':role_id,'kind':'asset_role','label':category,'source':'inferred',
          'source_reason':'explicit optional surface tool request within task constraints',
          'room_id':surface['room_id'],'category':category,'physical_class':'manipuland',
          'placement_kind':'surface','priority':'optional','purpose':'distraction','task_role':'distractor',
          'required_capabilities':['rigid']}
    new['nodes'].append(role)
    new['edges'].append({'id':'e_contains_'+role_id,'kind':'contains','source_id':surface_id,'target_id':role_id})
    verify_optional_extension(raw,new,role_id,surface_id)
    return new,role,parent[surface_id]


def add_surface_rigid_asset(graph,source,output, *, surface_id,role_id,category,description,expected_hash):
    actual,core,_=_verify_bundle(source)
    require(actual.content_hash==graph.content_hash==expected_hash,'GRAPH_VERSION_CONFLICT','optional input')
    require(not output.exists(),'COMPOSITION_ID_CONFLICT','output exists')
    raw,role,support_id=_extend(graph.to_dict(),surface_id,role_id,category)
    # Query a new versioned intent, with stale built evidence removed.
    raw['evidence_refs']=[]
    intent=graph.revise(raw,expected_hash=expected_hash)
    output.mkdir(parents=True)
    (output/'intent_graph.json').write_text(json.dumps(intent.to_dict(),indent=2)+'\n')
    query=search_rigid_assets(intent,role_id,description)
    (output/'optional_rigid_query.json').write_text(json.dumps(query,indent=2)+'\n')
    require(query['candidates'],'ASSET_UNAVAILABLE',role_id)
    for p in source.iterdir():
        if p.is_dir():shutil.copytree(p,output/p.name)
        else:shutil.copy2(p,output/p.name)
    # Carrying a previous addition must not overwrite this transaction's intent/query.
    (output/'intent_graph.json').write_text(json.dumps(intent.to_dict(),indent=2)+'\n')
    (output/'optional_rigid_query.json').write_text(json.dumps(query,indent=2)+'\n')
    metadata=prepare_selected(query,query['candidates'][0]['candidate_key'],output)
    support=next(i for i in core['instances'] if i['role_id']==support_id)
    value=support['support_surface'];occupancy=OccupancyMap()
    for item in core['instances']:
        if item.get('support_role_id')==support_id:
            dims=item['dimensions_m'];pose=item['pose_xyz_m']
            occupancy.add(item['role_id'],Footprint((dims[0]/2,dims[1]/2)).at(pose[0],pose[1]))
    goal=core['goal_placement']
    if goal['destination_support_role_id']==support_id:
        target=next(i for i in core['instances'] if i['role_id']==goal['target_role_id'])
        d=target['dimensions_m'];p=goal['candidate_pose_xyz_m']
        occupancy.add('reserved_goal',Footprint((d[0]/2,d[1]/2)).at(p[0],p[1]))
    dims=metadata['dimensions_m']
    placed=place_on_surface(instance_id='inst_'+role_id,entity_id=role_id,footprint=Footprint((dims[0]/2,dims[1]/2)),
        surface=Surface(surface_id=surface_id,owner_instance=support['instance_id'],
            polygon=Polygon.from_sequence(value['polygon_xy']),plane_z_m=value['plane_z_m'],provenance=AffordanceStatus.MEASURED),
        object_height_m=dims[2],occupancy=occupancy,
        config=SurfacePlacementConfig(step_m=.03,edge_margin_m=.04,object_gap_m=.06,max_candidates=1000),origin_convention='base')
    require(placed.placed,'SUPPORT_NOT_FOUND',role_id)
    item={k:metadata[k] for k in ('asset_id','provider','library_version','dimensions_m','origin_convention',
          'visual_ref','collision_ref','visual_sha256','collision_sha256')}
    item.update(instance_id='inst_'+role_id,role_id=role_id,room_id=role['room_id'],priority='optional',category=category,
                pose_xyz_m=placed.placement.position_m.to_list(),yaw_rad=0.,support_role_id=support_id,support_surface=None,
                metadata_ref=str(Path(metadata['visual_ref']).parent/'prepared.json'))
    frozen=TaskWorldGraph.from_dict(_read(source,'core_source_graph.json'))
    extended,_,_=_extend(frozen.to_dict(),surface_id,role_id,category)
    extended['evidence_refs']=[]
    expanded=frozen.revise(extended,expected_hash=frozen.content_hash)
    (output/'optional_prior_core_source_graph.json').write_text(json.dumps(frozen.to_dict(),indent=2)+'\n')
    (output/'optional_prior_graph.json').write_text(json.dumps(graph.to_dict(),indent=2)+'\n')
    (output/'optional_prior_core_ledger.json').write_text(json.dumps(core,indent=2)+'\n')
    (output/'core_source_graph.json').write_text(json.dumps(expanded.to_dict(),indent=2)+'\n')
    core['source_graph_hash']=expanded.content_hash;core['instances'].append(item)
    core['queries'][role_id]={'provider':'simready','query_id':query['query_id'],'role_id':role_id,'category':category,
        'candidates':[{'asset_id':c['asset_id'],'dimensions_m':c['dimensions_m']} for c in query['candidates']],'rejected':query['rejected']}
    core['selection'][role_id]={'asset_id':item['asset_id'],'status':'selected','provider':'simready','attempts':[]}
    core.pop('content_hash');core['content_hash']=content_hash(core)
    for name in ('core_asset_ledger.json','asset_ledger.json'):(output/name).write_text(json.dumps(core,indent=2)+'\n')
    raw=intent.to_dict()
    raw['nodes'].append({'id':item['instance_id'],'kind':'asset_instance','label':category+' instance','source':'inferred',
        'source_reason':'SimReady source prepared and statically placed','room_id':role['room_id'],
        'asset_ref':query['candidates'][0]['candidate_key'],'purpose':'distraction',
        'placement_anchor':{'kind':'surface','ref':surface_id}})
    raw['edges'].append({'id':'e_instance_'+role_id,'kind':'contains','source_id':role_id,'target_id':item['instance_id']})
    raw['edges'].append({'id':'e_bind_'+role_id,'kind':'role_bound_to','source_id':role_id,'target_id':item['instance_id']})
    raw['built']['asset_assembly']['ledger_hash']=core['content_hash']
    raw['built']['asset_assembly']['instance_ids'].append(item['instance_id'])
    for key in ('surface_appearance','render','visual_validation','robot_validation'):raw['built'].pop(key,None)
    for name in ('surface_appearance.json','appearance_source_graph.json'):
        if (output/name).exists():(output/name).unlink()
    if (output/'background_ledger.json').exists():
        bg=_read(output,'background_ledger.json');bg['core_ledger_hash']=core['content_hash'];bg.pop('content_hash');bg['content_hash']=content_hash(bg)
        (output/'background_ledger.json').write_text(json.dumps(bg,indent=2)+'\n')
        raw['built']['background_assembly']['ledger_hash']=bg['content_hash']
    raw['evidence_refs']=[];revised=intent.revise(raw,expected_hash=intent.content_hash)
    (output/'graph.json').write_text(json.dumps(revised.to_dict(),indent=2)+'\n')
    verify_optional_ledger(_read(output,'optional_prior_core_ledger.json'),core,role_id)
    _,_,verified=_verify_bundle(output)
    if 'surface_coverings' in raw['built']:
        from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
        verify_surface_coverings(output)
    report={'status':'passed','role_id':role_id,'surface_id':surface_id,'graph_hash':revised.content_hash,
            'source_graph_hash':graph.content_hash,'verification':verified,'selected_candidate_key':query['candidates'][0]['candidate_key'],
            'robot':'not_run','dynamics':'not_run','appearance':'not_run'}
    (output/'optional_rigid_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'optional_rigid_transaction.json').write_text(json.dumps(_event(graph,'add_surface_rigid_asset',(role_id,),0,
        writer=True,delta={'graph_hash':revised.content_hash}),indent=2)+'\n')
    return revised,report
