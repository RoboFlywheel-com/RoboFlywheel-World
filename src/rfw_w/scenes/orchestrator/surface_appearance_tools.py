# SPDX-License-Identifier: Apache-2.0
"""Versioned surface binding transactions over immutable static bundles."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import time
from copy import deepcopy
from rfw_w.scenes.assets.surface_materials import prepare_material, search_surface_materials
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.surface_appearance import SCHEMA, require, SurfaceAppearanceError
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph, GraphError
from rfw_w.scenes.geometry.surface_appearance import surfaces_from_mesh, verify_surface_appearance, digest
from rfw_w.scenes.orchestrator.room_asset_tools import _event


def read_graph(root):return TaskWorldGraph.from_dict(json.loads((root/'graph.json').read_text()))


def list_room_surface_bindings(graph,room_id):
    require(any(n['id']==room_id and n['kind']=='room' for n in graph.to_dict()['nodes']),
            'SURFACE_BINDING_INVALID',room_id)
    return [b for b in graph.to_dict()['built'].get('surface_appearance',{}).get('bindings',[]) if b['room_id']==room_id]


def _copy(source,output):
    require(not output.exists(),'SURFACE_APPEARANCE_OUTPUT_EXISTS',output.name)
    shutil.copytree(source,output,ignore=shutil.ignore_patterns('graph.json','surface_appearance.json',
        'appearance_source_graph.json','appearance_report.json','appearance_transaction.json'))


def _write(graph,source,output,ledger,bindings,name):
    started=time.monotonic();_copy(source,output)
    plan=json.loads((output/'geometry/house_plan.json').read_text())
    surfaces=surfaces_from_mesh(graph,plan,output/'geometry')
    oldmaterials=ledger.get('materials',{}) if ledger else {}
    materials={}
    for mid in sorted({b['material_id'] for b in bindings}):
        materials[mid]=deepcopy(oldmaterials[mid]) if mid in oldmaterials else prepare_material(mid,output)
    core=json.loads((output/('core_asset_ledger.json' if (output/'core_asset_ledger.json').exists() else 'asset_ledger.json')).read_text())
    ledger={'schema_version':SCHEMA,'source_graph_hash':graph.content_hash,'house_mesh_sha256':digest(output/'geometry/house.obj'),
            'core_ledger_hash':core['content_hash'],'materials':materials,'surfaces':surfaces,'bindings':bindings}
    ledger['content_hash']=content_hash(ledger)
    raw=graph.to_dict();raw['built']['surface_appearance']={'schema_version':SCHEMA,'ledger_ref':'surface_appearance.json',
                                                         'ledger_hash':ledger['content_hash'],'bindings':bindings}
    for key in ('render','visual_validation'):raw['built'].pop(key,None)
    raw['evidence_refs']=[]
    revised=graph.revise(raw,expected_hash=graph.content_hash)
    for fname,data in (('appearance_source_graph.json',graph.to_dict()),('graph.json',revised.to_dict()),('surface_appearance.json',ledger)):
        (output/fname).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    verified=verify_surface_appearance(output)
    before={b['surface_id']:b for b in graph.to_dict()['built'].get('surface_appearance',{}).get('bindings',[])}
    scope=tuple(sorted({b['scope_id'] for b in bindings if before.get(b['surface_id'])!=b}))
    report={'status':'passed','verification':verified,'source_graph_hash':graph.content_hash,'graph_hash':revised.content_hash,
            'invalidated_evidence':['render','visual_validation'],'elapsed_s':round(time.monotonic()-started,3)}
    (output/'appearance_transaction.json').write_text(json.dumps(_event(graph,name,scope,report['elapsed_s']*1000,
        writer=True,delta={'graph_hash':revised.content_hash}),indent=2)+'\n')
    (output/'appearance_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return revised,report


def set_surface_materials(graph,source,output, *, room_id, floor_material_id=None,wall_material_id=None,
                          wall_surface_id=None,meters_per_tile=1,expected_hash,description, intent_source='inferred',queries=()):
    require(expected_hash==graph.content_hash==read_graph(source).content_hash,'GRAPH_VERSION_CONFLICT','appearance input')
    require(floor_material_id or wall_material_id,'SURFACE_BINDING_INVALID','empty edit')
    admitted={c['material_id'] for q in queries for c in q['candidates']}
    require(all(mid in admitted for mid in (floor_material_id,wall_material_id) if mid),
            'MATERIAL_NOT_FOUND','candidate not returned by search')
    plan=json.loads((source/'geometry/house_plan.json').read_text());surfaces=surfaces_from_mesh(graph,plan,source/'geometry')
    old=json.loads((source/'surface_appearance.json').read_text()) if (source/'surface_appearance.json').exists() else None
    if old:verify_surface_appearance(source)
    bindings=deepcopy(old['bindings']) if old else [];chosen=[]
    for sid,s in surfaces.items():
        if s['room_id']!=room_id:continue
        mid=floor_material_id if s['kind']=='floor' else wall_material_id
        if not mid or (wall_surface_id and sid!=wall_surface_id):continue
        chosen.append(sid);bindings=[b for b in bindings if b['surface_id']!=sid]
        bindings.append({'surface_id':sid,'room_id':room_id,'scope_id':s['scope_id'],'material_id':mid,
                         'meters_per_tile':meters_per_tile,'intent':{'source':intent_source,'description':description}})
    require(bool(chosen),'SURFACE_BINDING_INVALID','room or wall side not found')
    # Explicit style is protected until another explicit user edit is supplied.
    for b in old['bindings'] if old else []:
        if b['surface_id'] in chosen and b['intent']['source']=='explicit' and intent_source!='explicit':
            require(next(x for x in bindings if x['surface_id']==b['surface_id'])==b,
                    'SURFACE_BINDING_INVALID','explicit style protected')
    return _write(graph,source,output,old,sorted(bindings,key=lambda b:b['surface_id']),'set_room_surface_materials' if not wall_surface_id else 'set_wall_surface_material')


def rollback_surface_appearance(graph,current,saved,output, *, room_id,expected_hash):
    require(expected_hash==graph.content_hash==read_graph(current).content_hash,'GRAPH_VERSION_CONFLICT','rollback')
    verify_surface_appearance(current);verify_surface_appearance(saved)
    old=json.loads((current/'surface_appearance.json').read_text());prior=json.loads((saved/'surface_appearance.json').read_text())
    require(old['house_mesh_sha256']==prior['house_mesh_sha256'] and old['core_ledger_hash']==prior['core_ledger_hash'],
            'SURFACE_BINDING_INVALID','checkpoint core')
    bindings=[b for b in old['bindings'] if b['room_id']!=room_id]+[b for b in prior['bindings'] if b['room_id']==room_id]
    require(bindings!=old['bindings'],'SURFACE_BINDING_INVALID','rollback no change')
    # Old texture references are portable and copied from the checkpoint as needed.
    merged=deepcopy(old);merged['materials'].update(prior['materials'])
    # Checkpoint material files must already be carried by current bundle.
    for mid in {b['material_id'] for b in bindings}:
        for c in merged['materials'][mid]['channels'].values():
            require((current/c['ref']).is_file(),'SURFACE_BINDING_INVALID','checkpoint texture missing')
    revised,report=_write(graph,current,output,merged,sorted(bindings,key=lambda b:b['surface_id']),'rollback_surface_appearance')
    return revised,report
