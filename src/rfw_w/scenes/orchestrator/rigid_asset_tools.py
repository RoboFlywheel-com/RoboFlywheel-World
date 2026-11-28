# SPDX-License-Identifier: Apache-2.0
"""Select/prepare a real SimReady candidate, preserving HSSD supports and task identity."""
import json
import shutil
from pathlib import Path
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.surface_appearance import require
from rfw_w.scenes.assets.rigid_tool_retrieval import prepare_selected
from rfw_w.scenes.assets.room_tools import verify_scene_assembly
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.orchestrator.room_asset_tools import _event
from rfw_w.scenes.geometry.polygon import Polygon,contains_polygon
from rfw_w.scenes.contracts.asset import AffordanceStatus
from rfw_w.scenes.solvers.layout import Footprint,OccupancyMap,SurfacePlacementConfig,place_on_surface
from rfw_w.scenes.solvers.support import Surface


def select_rigid_asset(graph,source,output,query,candidate_key, *, expected_hash):
    require(expected_hash==graph.content_hash==query['source_graph_hash'],'RIGID_QUERY_STALE','graph/query')
    require(content_hash({k:v for k,v in query.items() if k!='query_id'})==query['query_id'],
            'RIGID_CANDIDATE_NOT_ADMITTED','query digest')
    require(not output.exists(),'SURFACE_APPEARANCE_OUTPUT_EXISTS',output.name)
    require(TaskWorldGraph.from_dict(json.loads((source/'graph.json').read_text())).content_hash==graph.content_hash,'GRAPH_VERSION_CONFLICT','source graph')
    role_id=query['role_id'];raw=graph.to_dict();role=next(n for n in raw['nodes'] if n['id']==role_id)
    require(role['category']==query['category'] and role['physical_class']=='manipuland' and role.get('placement_kind')=='surface',
            'RIGID_CATEGORY_UNSUPPORTED',role_id)
    core_path=source/('core_asset_ledger.json' if (source/'core_asset_ledger.json').exists() else 'asset_ledger.json')
    core=json.loads(core_path.read_text());old=next(i for i in core['instances'] if i['role_id']==role_id)
    shutil.copytree(source,output,ignore=shutil.ignore_patterns('graph.json','rigid_report.json','rigid_transaction.json'))
    metadata=prepare_selected(query,candidate_key,output)
    selected=next(c for c in query['candidates'] if c['candidate_key']==candidate_key)
    require(metadata['physics']['mass_kg']==selected['mass_kg'],'RIGID_PHYSICS_UNCERTIFIED','query mass')
    item={**old,**{k:metadata[k] for k in ('provider','asset_id','library_version','dimensions_m','origin_convention',
                                         'visual_ref','collision_ref','visual_sha256','collision_sha256')}}
    item['metadata_ref']=str(Path(item['visual_ref']).parent/'prepared.json')
    # Recompute both poses using real support polygons and current optional occupancy.
    source_support=next(i for i in core['instances'] if i['role_id']==old['support_role_id'])
    def surface(support):
        value=support['support_surface']
        return Surface(surface_id=support['role_id']+':top',owner_instance=support['instance_id'],
            polygon=Polygon.from_sequence(value['polygon_xy']),plane_z_m=value['plane_z_m'],
            provenance=AffordanceStatus.MEASURED)
    d=item['dimensions_m'];occupancy=OccupancyMap()
    for other in core['instances']:
        if other['role_id']!=role_id and other.get('support_role_id')==old['support_role_id']:
            od=other['dimensions_m'];op=other['pose_xyz_m'];occupancy.add(other['role_id'],Footprint((od[0]/2,od[1]/2)).at(op[0],op[1]))
    placed=place_on_surface(instance_id=item['instance_id'],entity_id=role_id,footprint=Footprint((d[0]/2,d[1]/2)),
        surface=surface(source_support),object_height_m=d[2],occupancy=occupancy,
        config=SurfacePlacementConfig(step_m=.03,edge_margin_m=.04,object_gap_m=.06,max_candidates=1000),origin_convention='base')
    require(placed.placed,'SUPPORT_NOT_FOUND','initial candidate');item['pose_xyz_m']=placed.placement.position_m.to_list()
    if role_id==core['goal_placement']['target_role_id']:
        destination=next(i for i in core['instances'] if i['role_id']==core['goal_placement']['destination_support_role_id'])
        goal=place_on_surface(instance_id='goal_'+role_id,entity_id=role_id,footprint=Footprint((d[0]/2,d[1]/2)),
            surface=surface(destination),object_height_m=d[2],occupancy=OccupancyMap(),
            config=SurfacePlacementConfig(step_m=.03,edge_margin_m=.04,max_candidates=1000),origin_convention='base')
        require(goal.placed,'SUPPORT_NOT_FOUND','goal candidate');core['goal_placement']['candidate_pose_xyz_m']=goal.placement.position_m.to_list()
    core['instances']=[item if i['role_id']==role_id else i for i in core['instances']]
    core['queries'][role_id]={'role_id':role_id,'category':role['category'],'candidates':[{'asset_id':c['asset_id'],'dimensions_m':c['dimensions_m']} for c in query['candidates']],
                            'provider':'simready','query_id':query['query_id'],'rejected':query['rejected']}
    core['selection'][role_id]={'asset_id':item['asset_id'],'status':'selected','attempts':[],'provider':'simready'}
    core.pop('content_hash');core['content_hash']=content_hash(core)
    for n in raw['nodes']:
        if n['id']==old['instance_id']:
            n['asset_ref']=candidate_key;n['source_reason']='selected real SimReady candidate and remeasured static support'
    raw['built']['asset_assembly']['ledger_hash']=core['content_hash']
    # Geometry identity survives the target swap. Old appearance and background ledgers
    # must be rebound/reverified against this newly committed core instead of copied as passes.
    for k in ('surface_appearance','render','visual_validation','robot_validation'):raw['built'].pop(k,None)
    for fname in ('surface_appearance.json','appearance_source_graph.json'):
        if (output/fname).exists():(output/fname).unlink()
    for fname in ('asset_ledger.json','core_asset_ledger.json'):(output/fname).write_text(json.dumps(core,indent=2)+'\n')
    frozen=TaskWorldGraph.from_dict(json.loads((source/'core_source_graph.json').read_text()))
    plan=json.loads((source/'geometry/house_plan.json').read_text())
    verified=verify_scene_assembly(frozen,plan,core,output)
    if (output/'background_ledger.json').exists():
        bg=json.loads((output/'background_ledger.json').read_text());bg['core_ledger_hash']=core['content_hash'];bg.pop('content_hash');bg['content_hash']=content_hash(bg)
        raw['built']['background_assembly']['ledger_hash']=bg['content_hash']
        (output/'background_ledger.json').write_text(json.dumps(bg,indent=2)+'\n')
    raw['evidence_refs']=[];revised=graph.revise(raw,expected_hash=expected_hash)
    (output/'graph.json').write_text(json.dumps(revised.to_dict(),indent=2)+'\n')
    (output/'rigid_query.json').write_text(json.dumps(query,indent=2)+'\n')
    report={'status':'passed','source_graph_hash':graph.content_hash,'graph_hash':revised.content_hash,'verification':verified,
            'selected_candidate_key':candidate_key,'metadata':metadata,'robot_grasp':'not_run','dynamics':'not_run'}
    (output/'rigid_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'rigid_transaction.json').write_text(json.dumps(_event(graph,'select_rigid_asset',(role_id,),0,writer=True,
        delta={'graph_hash':revised.content_hash}),indent=2)+'\n')
    return revised,report



def _read(root, name):
    return json.loads((root / name).read_text())


def _verify_bundle(root):
    from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
    graph = read_graph(root)
    core = _read(root, 'core_asset_ledger.json' if (root / 'core_asset_ledger.json').exists() else 'asset_ledger.json')
    frozen = TaskWorldGraph.from_dict(_read(root, 'core_source_graph.json'))
    plan = _read(root, 'geometry/house_plan.json')
    verified = verify_scene_assembly(frozen, plan, core, root)
    require(graph.to_dict()['built']['asset_assembly']['ledger_hash'] == core['content_hash'],
            'ASSET_LEDGER_MISMATCH', 'graph core ledger')
    for item in core['instances']:
        node = next(n for n in graph.to_dict()['nodes'] if n['id'] == item['instance_id'])
        key = ('simready:' + item['library_version'] + ':' if item['provider'] == 'simready' else 'hssd:') + item['asset_id']
        require(node['asset_ref'] == key, 'ASSET_LEDGER_MISMATCH', 'instance asset')
    if (root / 'background_ledger.json').exists():
        from copy import deepcopy
        from rfw_w.scenes.orchestrator.background_tools import verify_background
        background = _read(root, 'background_ledger.json')
        require(content_hash({k:v for k,v in background.items() if k!='content_hash'}) == background['content_hash']
                and graph.to_dict()['built']['background_assembly']['ledger_hash'] == background['content_hash'],
                'ASSET_LEDGER_MISMATCH', 'background digest')
        source = TaskWorldGraph.from_dict(_read(root, 'input_graph.json'))
        checked = deepcopy(background)
        checked['source_graph_hash'] = source.content_hash
        checked.pop('content_hash')
        checked['content_hash'] = content_hash(checked)
        verify_background(graph, source, plan, core, checked, root)
    return graph, core, verified


def save_rigid_checkpoint(graph, source, output, *, role_id, expected_hash):
    from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
    actual, core, verified = _verify_bundle(source)
    require(actual.content_hash == graph.content_hash == expected_hash, 'GRAPH_VERSION_CONFLICT', 'checkpoint')
    require(core['goal_placement']['target_role_id'] == role_id, 'ROLLBACK_SCOPE_INVALID', 'target only')
    require(not output.exists(), 'CHECKPOINT_TAMPERED', 'output exists')
    output.mkdir(parents=True)
    checkpoint = {'schema_version':'pvscene.rigid-checkpoint/v1', 'graph_hash':graph.content_hash,
                  'role_id':role_id, 'artifacts':artifact_digest(source)}
    checkpoint['content_hash'] = content_hash(checkpoint)
    (output / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2) + '\n')
    (output / 'transaction.json').write_text(json.dumps(_event(graph, 'save_rigid_checkpoint', (role_id,), 0), indent=2) + '\n')
    return checkpoint


def restore_rigid_asset(graph, current, checkpoint_dir, saved, output, *, expected_hash):
    from rfw_w.scenes.geometry.rigid_asset_checkpoint import verify_checkpoint_payload, verify_binding_scope
    from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
    current_graph, core, _ = _verify_bundle(current)
    require(current_graph.content_hash == graph.content_hash == expected_hash, 'GRAPH_VERSION_CONFLICT', 'restore')
    saved_graph, prior, _ = _verify_bundle(saved)
    checkpoint = _read(checkpoint_dir, 'checkpoint.json')
    verify_checkpoint_payload(checkpoint, saved_graph, artifact_digest(saved))
    role_id = checkpoint['role_id']
    raw = graph.to_dict(); old = saved_graph.to_dict()
    verify_binding_scope(raw, old, core, prior, role_id)
    require(not output.exists(), 'CHECKPOINT_TAMPERED', 'output exists')
    # Both inputs have been independently measured before restoring a single binding.
    item = next(i for i in prior['instances'] if i['role_id'] == role_id)
    current_item = next(i for i in core['instances'] if i['role_id'] == role_id)
    require(item != current_item, 'ROLLBACK_SCOPE_INVALID', 'no binding change')
    shutil.copytree(current, output)
    directory = Path(item['visual_ref']).parent
    if (output / directory).exists():
        shutil.rmtree(output / directory)
    shutil.copytree(saved / directory, output / directory)
    core['instances'] = [item if i['role_id'] == role_id else i for i in core['instances']]
    for key in ('queries', 'selection'):
        core[key][role_id] = prior[key][role_id]
    core['goal_placement'] = prior['goal_placement']
    core.pop('content_hash'); core['content_hash'] = content_hash(core)
    original_node = next(n for n in old['nodes'] if n['id'] == item['instance_id'])
    raw['nodes'] = [original_node if n['id'] == item['instance_id'] else n for n in raw['nodes']]
    raw['built']['asset_assembly']['ledger_hash'] = core['content_hash']
    for key in ('surface_appearance', 'render', 'visual_validation', 'robot_validation'):
        raw['built'].pop(key, None)
    for name in ('surface_appearance.json', 'appearance_source_graph.json'):
        if (output / name).exists(): (output / name).unlink()
    for name in ('asset_ledger.json', 'core_asset_ledger.json'):
        (output / name).write_text(json.dumps(core, indent=2) + '\n')
    if (output / 'background_ledger.json').exists():
        bg = _read(output, 'background_ledger.json'); bg['core_ledger_hash'] = core['content_hash']
        bg.pop('content_hash'); bg['content_hash'] = content_hash(bg)
        raw['built']['background_assembly']['ledger_hash'] = bg['content_hash']
        (output / 'background_ledger.json').write_text(json.dumps(bg, indent=2) + '\n')
    raw['evidence_refs'] = []
    revised = graph.revise(raw, expected_hash=expected_hash)
    (output / 'graph.json').write_text(json.dumps(revised.to_dict(), indent=2) + '\n')
    _, _, verified = _verify_bundle(output)
    if 'surface_coverings' in raw['built']:
        from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
        verify_surface_coverings(output)
    report = {'status':'passed', 'source_graph_hash':graph.content_hash, 'graph_hash':revised.content_hash,
              'checkpoint_hash':checkpoint['content_hash'], 'role_id':role_id, 'restored_asset_id':item['asset_id'],
              'verification':verified, 'robot':'not_run', 'dynamics':'not_run', 'appearance':'not_run'}
    (output / 'rigid_restore_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'rigid_restore_transaction.json').write_text(json.dumps(_event(graph, 'restore_rigid_asset', (role_id,), 0,
        writer=True, delta={'graph_hash':revised.content_hash}), indent=2) + '\n')
    return revised, report
