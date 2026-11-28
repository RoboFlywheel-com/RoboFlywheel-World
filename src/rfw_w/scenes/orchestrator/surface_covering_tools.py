# SPDX-License-Identifier: Apache-2.0
"""Immutable thin covering edits and independent remeasurement."""
import json
import shutil
from copy import deepcopy
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.surface_appearance import require
from rfw_w.scenes.contracts.scene_composition import background_role
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph,COMPOSITION_SCHEMA_V3
from rfw_w.scenes.geometry.surface_appearance import surfaces_from_mesh
from rfw_w.scenes.geometry.surface_coverings import covering_vertices,verify_covering
from rfw_w.scenes.orchestrator.room_asset_tools import _event


def load(root,name):return json.loads((root/name).read_text())


def verify_surface_coverings(root):
    graph=TaskWorldGraph.from_dict(load(root,'graph.json'));raw=graph.to_dict()
    ledger=load(root,'surface_coverings.json');built=raw['built']['surface_coverings']
    require(content_hash({k:v for k,v in ledger.items() if k!='content_hash'})==ledger['content_hash']==built['ledger_hash'],
            'COVERING_NORMAL_INVALID','ledger digest')
    plan=load(root,'geometry/house_plan.json');surfaces=surfaces_from_mesh(graph,plan,root/'geometry')
    require(built['anchors']==surfaces,'COVERING_NORMAL_INVALID','anchor geometry')
    core=load(root,'core_asset_ledger.json');background=load(root,'background_ledger.json');accepted=[]
    roles={n['id']:n for n in raw['nodes'] if n['kind']=='asset_role' and n.get('physical_class')=='covering'}
    instances={n['id']:n for n in raw['nodes'] if n['kind']=='asset_instance'}
    from rfw_w.scenes.contracts.surface_appearance import validate_material
    from rfw_w.scenes.geometry.surface_appearance import checked_path,digest
    from PIL import Image
    for mid,material in ledger['materials'].items():
        validate_material(mid,material)
        for c in material['channels'].values():
            path=checked_path(root,c['ref']);require(digest(path)==c['sha256'],'COVERING_MATERIAL_INVALID',mid)
            with Image.open(path) as im:im.load()
    for item in ledger['instances']:
        role=roles.get(item['role_id']);node=instances.get(item['instance_id'])
        require(role and node and role['room_id']==item['room_id'] and role['purpose']==item['purpose']
                and node['placement_anchor']['ref']==item['surface_id'] and item['material_id'] in ledger['materials'],
                'COVERING_CORE_FORBIDDEN','binding')
        verify_covering(item,surfaces,plan,core,background,accepted);accepted.append(item)
    require(set(roles)=={i['role_id'] for i in accepted} and len(accepted)==len(roles),
            'COVERING_CORE_FORBIDDEN','accounting')
    return {'verified':True,'problems':[],'instances':len(accepted),'graph_hash':graph.content_hash,
            'physical':'not_run','user_visual':'not_run'}


def edit_surface_covering(graph,source,output, *, action,expected_hash,instance_id,
                          surface_id=None,material_id=None,size_m=None,local_position_m=None,purpose='background'):
    require(graph.content_hash==expected_hash==TaskWorldGraph.from_dict(load(source,'graph.json')).content_hash,
            'GRAPH_VERSION_CONFLICT','covering input')
    require(graph.to_dict().get('composition_schema')==COMPOSITION_SCHEMA_V3,'GRAPH_SCHEMA_UNSUPPORTED','explicit v3 required')
    require(not output.exists(),'SURFACE_APPEARANCE_OUTPUT_EXISTS',output.name)
    require(action in ('create','move','remove'),'COVERING_CORE_FORBIDDEN','action')
    require(isinstance(instance_id,str) and instance_id.startswith('cover_') and all(c.isalnum() or c=='_' for c in instance_id),
            'COVERING_CORE_FORBIDDEN','covering ID')
    if (source/'surface_coverings.json').exists():verify_surface_coverings(source)
    ledger=load(source,'surface_coverings.json') if (source/'surface_coverings.json').exists() else {'instances':[],'materials':{}}
    ledger=deepcopy(ledger);ledger.pop('content_hash',None);raw=graph.to_dict()
    plan=load(source,'geometry/house_plan.json');surfaces=surfaces_from_mesh(graph,plan,source/'geometry')
    rid=instance_id+'_role';item=next((i for i in ledger['instances'] if i['instance_id']==instance_id),None)
    if action=='create':
        require(item is None and instance_id not in {n['id'] for n in raw['nodes']},'COVERING_CORE_FORBIDDEN','duplicate')
        s=surfaces.get(surface_id);require(s is not None,'COVERING_SURFACE_MISSING',surface_id)
        require(purpose in ('background','distraction'),'COVERING_CORE_FORBIDDEN','purpose')
        materials=load(source,'surface_appearance.json')['materials']
        from rfw_w.scenes.assets.surface_materials import SOLIDS
        require(material_id in materials or material_id in SOLIDS,'COVERING_MATERIAL_INVALID',material_id)
        material=materials[material_id] if material_id in materials else {'provider':'solid','base_color_rgba':SOLIDS[material_id],'roughness':.82,'channels':{}}
        ledger['materials'][material_id]=material
        item={'instance_id':instance_id,'role_id':rid,'room_id':s['room_id'],'surface_id':surface_id,
              'local_position_m':local_position_m,'size_m':size_m,'normal':s['normal'],'material_id':material_id,
              'purpose':purpose,'collision_policy':'visual_only' if s['kind']=='floor' else 'static_clearance_proxy'}
        item['vertices_m']=covering_vertices(s,local_position_m,size_m)
        role,edge=background_role(raw,s['room_id'],rid,'rug' if s['kind']=='floor' else 'wallpaper_patch',
                                  'floor' if s['kind']=='floor' else 'wall',purpose=purpose)
        role['physical_class']='covering';role['required_capabilities']=['visual_covering']
        raw['nodes'].append(role);raw['edges'].append(edge)
        raw['nodes'].append({'id':instance_id,'kind':'asset_instance','label':role['category'],'source':'inferred',
            'source_reason':'explicit thin decorative geometry on measured surface','room_id':s['room_id'],
            'asset_ref':'covering:'+instance_id,'purpose':purpose,'placement_anchor':{'kind':role['placement_kind'],'ref':surface_id}})
        for kind in ('contains','role_bound_to'):raw['edges'].append({'id':'e_'+kind+'_'+instance_id,'kind':kind,'source_id':rid,'target_id':instance_id})
        ledger['instances'].append(item)
    else:
        require(item is not None,'COVERING_SURFACE_MISSING',instance_id)
        if action=='move':
            item['local_position_m']=local_position_m;item['vertices_m']=covering_vertices(surfaces[item['surface_id']],local_position_m,item['size_m'])
        else:
            ledger['instances']=[i for i in ledger['instances'] if i['instance_id']!=instance_id]
            raw['nodes']=[n for n in raw['nodes'] if n['id'] not in (instance_id,rid)]
            raw['edges']=[e for e in raw['edges'] if e['source_id'] not in (instance_id,rid) and e['target_id'] not in (instance_id,rid)]
    accepted=[]
    for i in ledger['instances']:
        verify_covering(i,surfaces,plan,load(source,'core_asset_ledger.json'),load(source,'background_ledger.json'),accepted);accepted.append(i)
    shutil.copytree(source,output,ignore=shutil.ignore_patterns('graph.json','surface_coverings.json','covering_report.json','covering_transaction.json'))
    ledger['content_hash']=content_hash(ledger)
    raw['built']['surface_coverings']={'ledger_ref':'surface_coverings.json','ledger_hash':ledger['content_hash'],'anchors':surfaces}
    raw['evidence_refs']=[];raw['built'].pop('render',None)
    revised=graph.revise(raw,expected_hash=expected_hash)
    for fname,data in (('graph.json',revised.to_dict()),('surface_coverings.json',ledger)):
        (output/fname).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    report=verify_surface_coverings(output)
    (output/'covering_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'covering_transaction.json').write_text(json.dumps(_event(graph,action+'_surface_covering',(rid,),0,
        writer=True,delta={'graph_hash':revised.content_hash}),indent=2)+'\n')
    return revised,report


def rollback_surface_covering(graph,current,saved,output, *, instance_id,expected_hash):
    require(expected_hash==graph.content_hash,'GRAPH_VERSION_CONFLICT','covering rollback')
    verify_surface_coverings(saved);verify_surface_coverings(current)
    item=next((i for i in load(saved,'surface_coverings.json')['instances'] if i['instance_id']==instance_id),None)
    require(item is not None,'COVERING_SURFACE_MISSING','checkpoint item')
    existing=next((i for i in load(current,'surface_coverings.json')['instances'] if i['instance_id']==instance_id),None)
    require(existing is not None and all(existing[k]==item[k] for k in ('material_id','surface_id','size_m','purpose')),
            'COVERING_CORE_FORBIDDEN','only position rollback')
    return edit_surface_covering(graph,current,output,action='move',expected_hash=expected_hash,
                                 instance_id=instance_id,local_position_m=item['local_position_m'])
