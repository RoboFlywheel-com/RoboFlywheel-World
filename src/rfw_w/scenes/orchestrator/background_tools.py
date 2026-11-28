# SPDX-License-Identifier: Apache-2.0
"""Fresh background transactions over the same versioned task-world graph."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import time
from rfw_w.scenes.assets.background_tools import (retrieve_background_candidates,place_floor,place_wall,
    measure_wall_mount_surfaces,verify_prepared_instance,screen_floor,screen_wall)
from rfw_w.scenes.assets.room_tools import asset_config, RoomAssetError, verify_scene_assembly
from rfw_w.scenes.agents.room_asset_visualization import render_room_assets_svg
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError,TaskWorldGraph
from rfw_w.scenes.contracts.scene_composition import migrate_composition
from rfw_w.scenes.orchestrator.room_asset_tools import _event
from rfw_w.scenes.solvers.house_tools import measure_house_access


def source_bundle(source):
    assets=source/'assets' if (source/'assets').is_dir() else source
    graph=TaskWorldGraph.from_dict(json.loads((assets/'graph.json').read_text()))
    if graph.to_dict()['composition_schema'].endswith('/v1'):
        graph=migrate_composition(graph,expected_hash=graph.content_hash)
    return graph,assets


def verify_background(graph,source_graph,plan,core,ledger,root):
    if ledger['source_graph_hash']!=source_graph.content_hash:
        raise RoomAssetError('GRAPH_VERSION_CONFLICT','background source')
    if content_hash({k:v for k,v in ledger.items() if k!='content_hash'})!=ledger['content_hash']:
        raise RoomAssetError('ASSET_LEDGER_MISMATCH','background hash')
    if ledger['core_ledger_hash']!=core['content_hash'] or ledger['house_plan_hash']!=plan['content_hash']:
        raise RoomAssetError('CORE_KEEP_OUT_VIOLATION','changed core')
    if graph.to_dict()['built']['asset_assembly']['ledger_hash']!=core['content_hash']:
        raise RoomAssetError('CORE_KEEP_OUT_VIOLATION','core ledger replaced')
    if content_hash({k:v for k,v in core.items() if k!='content_hash'})!=core['content_hash']:
        raise RoomAssetError('CORE_KEEP_OUT_VIOLATION','core ledger modified')
    if measure_house_access(plan,root/'geometry')['mesh_sha256']!=graph.to_dict()['built']['house_shell']['mesh_sha256']:
        raise RoomAssetError('ASSET_LEDGER_MISMATCH','house mesh')
    # Re-run the original G3 verifier against the original frozen core graph and ledger.
    frozen=TaskWorldGraph.from_dict(json.loads((root/'core_source_graph.json').read_text()))
    verify_scene_assembly(frozen,plan,core,root)
    original=frozen.to_dict();current=graph.to_dict()
    for key in ('initial_state','steps','goals','requirements'):
        if original[key]!=current[key]:raise RoomAssetError('CORE_KEEP_OUT_VIOLATION','task '+key)
    current_nodes={n['id']:n for n in current['nodes']}
    for node in original['nodes']:
        if node.get('priority')=='optional' or node['kind'] in ('asset_instance','distractor_policy'):
            continue
        actual=current_nodes.get(node['id'])
        # Purpose and placement are the explicit v1→v2 annotations; all frozen
        # identity, topology and semantic core fields remain part of this gate.
        if not actual or any(actual.get(k)!=v for k,v in node.items()):
            raise RoomAssetError('CORE_KEEP_OUT_VIOLATION',node['id'])
    roles={n['id']:n for n in graph.to_dict()['nodes'] if n['kind']=='asset_role'}
    anchors=measure_wall_mount_surfaces(plan,root/'geometry')
    accepted=[]; floor_area={r['id']:0. for r in plan['rooms']}
    found=set()
    for item in ledger['instances']:
        rid=item['role_id']; role=roles.get(rid)
        if not role or rid in found or role['purpose']!=item['purpose'] or role['room_id']!=item['room_id']:
            raise RoomAssetError('ASSET_BINDING_INVALID',rid)
        if role.get('placement_kind')!=item['placement_kind']:
            raise RoomAssetError('ASSET_BINDING_INVALID',rid)
        verify_prepared_instance(item,root,ledger['queries'][rid])
        if item['placement_kind']=='floor':
            screen_floor(item,plan,core,accepted)
            floor_area[item['room_id']]+=item['dimensions_m'][0]*item['dimensions_m'][1]
            if floor_area[item['room_id']]>16*.3:
                raise RoomAssetError('FURNISHING_BUDGET_EXCEEDED',item['room_id'])
        else:
            a=anchors.get(item['wall_anchor'])
            if not a: raise RoomAssetError('WALL_SURFACE_MISSING',rid)
            cert=item.get('mount_certificate',{})
            d=item['dimensions_m']
            if (cert.get('method')!='upright_thin_panel_aabb' or cert.get('measured_dimensions_m')!=d
                or not (.001<=d[1]<=.16 and .2<=d[2]<=1.4)):
                raise RoomAssetError('WALL_MOUNT_UNCERTIFIED',rid)
            screen_wall(item,a,plan,core,accepted)
        accepted.append(item);found.add(rid)
    optional={rid for rid,r in roles.items() if r.get('purpose') in ('background','distraction')
              and r.get('placement_kind') in ('floor','wall') and r.get('physical_class')!='covering'}
    missing=[u['role_id'] for u in ledger['unplaced_optional']]
    if len(missing)!=len(set(missing)) or found&set(missing) or found|set(missing)!=optional:
        raise RoomAssetError('ASSET_BINDING_INVALID','optional accounting')
    if any(sum(i['room_id']==r['id'] for i in accepted)>8 for r in plan['rooms']):
        raise RoomAssetError('FURNISHING_BUDGET_EXCEEDED','room count')
    return {'verified':True,'instances':len(accepted),'floor':sum(i['placement_kind']=='floor' for i in accepted),
            'wall':sum(i['placement_kind']=='wall' for i in accepted),'core_preserved':True,'problems':[],
            'robot_validation':'not_run','physical_attachment':'not_run','rear_face_orientation':'not_run'}


def run_background_tools(graph,assets,output,kind, *, expected_hash=None):
    if output.exists():raise GraphError('ASSET_OUTPUT_EXISTS',output.name)
    try:
        return _run_background_tools(graph,assets,output,kind,expected_hash=expected_hash)
    except (GraphError,RoomAssetError) as error:
        if not output.exists():output.mkdir(parents=True)
        failure=output/'background_failure.json'
        if not failure.exists():
            failure.write_text(json.dumps({'status':'failed','reason_code':error.code,
                'event':_event(graph,'background_'+kind,(),0,reason_code=error.code)},indent=2)+'\n')
        raise


def _run_background_tools(graph,assets,output,kind, *, expected_hash=None):
    if expected_hash is not None and expected_hash!=graph.content_hash:
        raise GraphError('GRAPH_VERSION_CONFLICT','background input')
    if kind not in ('floor','wall'): raise GraphError('PRECONDITION_UNMET','placement kind')
    if output.exists(): raise GraphError('ASSET_OUTPUT_EXISTS',output.name)
    raw=graph.to_dict()
    if raw.get('composition_schema')!='pvscene.room-composition/v2' or 'asset_assembly' not in raw['built']:
        raise GraphError('PRECONDITION_UNMET','v2 built core')
    shutil.copytree(assets,output,ignore=shutil.ignore_patterns('graph.json','report.json','scene.svg',
                    'background_ledger.json','background_report.json','input_graph.json','transaction.json',
                    'agent_event.json','state_graph.json','restoration_verification.json'))
    # Copied histories are immutable source artifacts; current outputs have distinct filenames.
    (output/'input_graph.json').write_text(json.dumps(raw,indent=2)+'\n')
    source_core=assets/'core_asset_ledger.json'
    core=json.loads((source_core if source_core.exists() else assets/'asset_ledger.json').read_text())
    (output/'core_asset_ledger.json').write_text(json.dumps(core,indent=2)+'\n')
    if not (output/'core_source_graph.json').exists():
        house_graph=assets.parent/'house/graph.json'
        if not house_graph.exists(): raise GraphError('PRECONDITION_UNMET','frozen core source')
        shutil.copyfile(house_graph,output/'core_source_graph.json')
    plan=json.loads((output/'geometry/house_plan.json').read_text())
    prior=assets/'background_ledger.json'
    ledger=json.loads(prior.read_text()) if prior.exists() else {'instances':[],'queries':{},'unplaced_optional':[],'attempts':{}}
    ledger.pop('content_hash',None)
    ledger.update(source_graph_hash=graph.content_hash,core_ledger_hash=core['content_hash'],house_plan_hash=plan['content_hash'])
    roles=[n for n in raw['nodes'] if n['kind']=='asset_role' and n.get('placement_kind')==kind
           and n['purpose']!='task_core' and n['id'] not in {i['role_id'] for i in ledger['instances']}]
    roles=[r for r in roles if r['id'] not in {u['role_id'] for u in ledger['unplaced_optional']}]
    if not roles: raise GraphError('PRECONDITION_UNMET','no pending '+kind+' roles')
    scope=tuple(r['id'] for r in roles);events=[];config=asset_config(output/'prepared')
    start=time.monotonic()
    for r in roles:ledger['queries'][r['id']]=retrieve_background_candidates(r,config)
    events.append(_event(graph,'retrieve_'+kind+'_candidates',scope,(time.monotonic()-start)*1000))
    anchors=measure_wall_mount_surfaces(plan,output/'geometry') if kind=='wall' else {}
    if kind=='wall':events.append(_event(graph,'measure_wall_mount_surfaces',scope,0))
    start=time.monotonic()
    for role in roles:
        if kind=='floor': item,attempts=place_floor(role,ledger['queries'][role['id']],config,plan,core,ledger['instances'],output)
        else:item,attempts=place_wall(role,ledger['queries'][role['id']],config,plan,core,ledger['instances'],output,anchors)
        ledger['attempts'][role['id']]=attempts
        if item:ledger['instances'].append(item)
        else:ledger['unplaced_optional'].append({'role_id':role['id'],'code':'FURNISHING_CANDIDATES_EXHAUSTED' if kind=='floor' else 'WALL_MOUNT_UNCERTIFIED'})
    events.append(_event(graph,'place_'+kind+'_background',scope,(time.monotonic()-start)*1000))
    ledger['content_hash']=content_hash(ledger)
    raw['built']['wall_mounts']={'anchors':anchors} if kind=='wall' else raw['built'].get('wall_mounts',{'anchors':{}})
    existing={n['id'] for n in raw['nodes']}
    parents={e['target_id']:e['source_id'] for e in raw['edges'] if e['kind']=='contains'}
    for item in ledger['instances']:
        iid=item['instance_id'];rid=item['role_id']
        if iid in existing:continue
        raw['nodes'].append({'id':iid,'kind':'asset_instance','label':item['category'],'source':'inferred',
            'source_reason':'real asset measured and statically placed','room_id':item['room_id'],
            'asset_ref':'hssd:'+item['asset_id'],'purpose':item['purpose'],
            'placement_anchor':{'kind':item['placement_kind'],'ref':item.get('wall_anchor',parents[rid])}})
        for relation in ('contains','role_bound_to'):
            raw['edges'].append({'id':'e_'+relation+'_'+iid,'kind':relation,'source_id':rid,'target_id':iid})
    raw['built']['background_assembly']={'ledger_hash':ledger['content_hash'],'ledger_ref':'background_ledger.json',
                                       'instance_ids':[i['instance_id'] for i in ledger['instances']]}
    raw['evidence_refs']=[]
    revised=graph.revise(raw,expected_hash=graph.content_hash)
    # Other kind's optional roles are pending until its own tool executes.
    accounted={i['role_id'] for i in ledger['instances']}|{u['role_id'] for u in ledger['unplaced_optional']}
    verify_ledger=json.loads(json.dumps(ledger))
    for r in raw['nodes']:
        if r['kind']=='asset_role' and r.get('purpose')!='task_core' and r.get('placement_kind') in ('floor','wall') and r['id'] not in accounted:
            verify_ledger['unplaced_optional'].append({'role_id':r['id'],'code':'not_run'})
    verify_ledger.pop('content_hash');verify_ledger['content_hash']=content_hash(verify_ledger)
    verification=verify_background(revised,graph,plan,core,verify_ledger,output)
    events.append(_event(graph,'verify_'+kind+'_background',scope,0))
    events.append(_event(graph,'commit_'+kind+'_background',scope,0,writer=True,delta={'graph_hash':revised.content_hash}))
    report={'status':'passed','graph_hash':revised.content_hash,'source_graph_hash':graph.content_hash,
            'events':events,'verification':verification,'unplaced_optional':ledger['unplaced_optional']}
    for name,value in (('background_ledger.json',ledger),('graph.json',revised.to_dict()),('background_report.json',report)):
        (output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    combined={**core,'instances':core['instances']+ledger['instances']}
    (output/'scene.svg').write_text(render_room_assets_svg(plan,combined))
    return revised,report
