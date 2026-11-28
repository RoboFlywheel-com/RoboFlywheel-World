# SPDX-License-Identifier: Apache-2.0
"""Bounded SimReady searches through the existing guarded retrieval adapter."""
import json
import os
from pathlib import Path
import time
from dataclasses import replace
from rfw_w.scenes.contracts.surface_appearance import require
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_spec import EntitySpec,Range1D
from rfw_w.scenes.resources import load_json

_ENGINE=None
_CONFIG=None
_VIEWS={}


def engine():
    global _ENGINE,_CONFIG
    if _ENGINE is not None:return _ENGINE,_CONFIG
    from rfw_w.scenes.adapters.simready_retrieval import SimreadyConfig,build_encoder,build_engine
    keys=('PVSCENE_SIMREADY_INDEX','PVSCENE_SIMREADY_ASSETS_ROOT','PVSCENE_SIMREADY_SEMANTIC_INDEX','PVSCENE_PREPARED_DIR')
    require(all(os.environ.get(k) for k in keys),'SIMREADY_ENV_MISSING','required variables')
    index=json.loads(Path(os.environ[keys[0]]).read_bytes())
    version=index.get('_meta',{}).get('library_version')
    require(version=='v1.3','SIMREADY_INDEX_INVALID','v1.3 required')
    _CONFIG=SimreadyConfig(index_path=Path(os.environ[keys[0]]),assets_root=Path(os.environ[keys[1]]),
        semantic_index_path=Path(os.environ[keys[2]]),library_version=version,
        prepared_root=Path(os.environ.get('PVSCENE_PREPARED_DIR'))/'rigid_tool_cache')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    _ENGINE=build_engine(_CONFIG,encoder=build_encoder(Path(os.environ[keys[3]])))
    return _ENGINE,_CONFIG


def search_rigid_assets(graph,role_id,description, *, limit=8):
    start=time.monotonic();cold=_ENGINE is None
    role=next((n for n in graph.to_dict()['nodes'] if n['id']==role_id),None)
    require(role and role.get('physical_class')=='manipuland' and role.get('placement_kind')=='surface',
            'RIGID_CATEGORY_UNSUPPORTED',role_id)
    require(isinstance(description,str) and 0<len(description)<=400 and type(limit) is int and 1<=limit<=8,
            'RIGID_CATEGORY_UNSUPPORTED','query')
    categories=load_json('configs/rigid_tool_categories_v1.json')['categories']
    admitted=categories.get(role['category']);require(admitted,'RIGID_CATEGORY_UNSUPPORTED',role['category'])
    e,c=engine()
    from rfw_w.scenes.adapters.simready_retrieval import retrieve_entity
    spec=EntitySpec.from_dict({'id':role_id,'category':role['category'],'physics_type':'rigid','mobility':'dynamic',
                              'hard_required':role['priority']=='core','role':'task_target' if role['priority']=='core' else 'context','notes':description})
    if role['category']=='plate':
        spec=replace(spec,size_bounds_m=tuple(Range1D(*bounds)
                    for bounds in ((.18,.26),(.18,.26),(.015,.045))))
    if role['category']=='plate':
        e=type(e)(e.registry,policy=replace(e.policy,dimension_match_mode='permutation_invariant'))
    outcome=retrieve_entity(e,spec,config=c,limit=64 if role['category']=='plate' else limit,
                            admitted_categories=admitted,allow_dimension_permutation=role['category']=='plate')
    if role['category']=='plate':
        # A thin axis is a cheap necessary profile before the three source preparations.
        candidates=sorted(outcome.candidates,key=lambda v:(min(v.dimensions_m.to_list())/max(v.dimensions_m.to_list()),-v.total_score))
    else:candidates=outcome.candidates
    query={'source_graph_hash':graph.content_hash,'role_id':role_id,'category':role['category'],'description':description,
           'provider':'simready','library_version':c.library_version,'mapping_version':'rigid-tool-categories/v1',
           'admitted_categories':admitted,'candidates':[],
           'rejected':[{'asset_id':aid,'reason':reason} for aid,reason in outcome.rejected],
           'cold_engine':cold,'elapsed_s':round(time.monotonic()-start,3),'ranking_method':'SBERT_class_recall_and_measured_admission',
           'failure_code':outcome.failure_code}
    # Normalized vendor collision exports can exchange Y/Z per asset. Measure the
    # source preparation used by delivery; cap expensive preparations at three.
    from rfw_w.scenes.assets.providers.simready import SimreadyPrepareConfig,prepare_rigid
    from rfw_w.scenes.geometry.rigid_tool_assets import verify_task_shape
    from rfw_w.scenes.contracts.surface_appearance import SurfaceAppearanceError
    query['source_preparations_attempted']=0
    for view in candidates[:3]:
        query['source_preparations_attempted']+=1
        try:
            prepared=prepare_rigid(SimreadyPrepareConfig(c.assets_root,c.prepared_root/'source_measurement'),
                prefix=view.library_prefix,leaf=view.asset_key.asset_id)
            dims=prepared.dimensions_m.to_list()
            verify_task_shape(role['category'],dims)
        except (ValueError,OSError) as error:
            query['rejected'].append({'asset_id':view.asset_key.asset_id,
                'reason':getattr(error,'code','RIGID_PREPARE_FAILED')})
            continue
        view=replace(view,dimensions_m=prepared.dimensions_m)
        key='simready:'+c.library_version+':'+view.asset_key.asset_id
        query['candidates'].append({'candidate_key':key,'asset_id':view.asset_key.asset_id,'category':view.super_category,
            'dimensions_m':dims,'mass_kg':view.mass_kg,'score':view.total_score,
            'physics_source':'catalogue_declared:part_physics','grasp':'not_run',
            'dimensions_source':'source_preparation_Z_up'})
        _VIEWS[key]=view
    if not query['candidates']:query['failure_code']='ASSET_UNAVAILABLE'
    query['elapsed_s']=round(time.monotonic()-start,3)
    query['query_id']=content_hash(query)
    return query


def prepare_selected(query,key,root):
    require(key in {c['candidate_key'] for c in query['candidates']} and key in _VIEWS,
            'RIGID_CANDIDATE_NOT_ADMITTED',key)
    view=_VIEWS[key];_,config=engine()
    from rfw_w.scenes.geometry.rigid_tool_assets import verify_task_shape
    verify_task_shape(query['category'],view.dimensions_m.to_list())
    from rfw_w.scenes.assets.providers.simready import SimreadyPrepareConfig,prepare_rigid
    value=prepare_rigid(SimreadyPrepareConfig(config.assets_root,root/'prepared/simready/v1.3'),
                        prefix=view.library_prefix,leaf=view.asset_key.asset_id)
    lo=value.bounds_min_m.to_list();hi=value.bounds_max_m.to_list();shift=[-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2]]
    # Preserve OBJ material/UV statements while transforming vertices in both meshes.
    for path in (value.visual_obj,value.collision_obj):
        lines=[]
        for line in path.read_text().splitlines():
            if line.startswith('v '):
                xyz=[float(x) for x in line.split()[1:4]];line='v '+' '.join(f'{v+d:.9f}' for v,d in zip(xyz,shift))
            lines.append(line)
        path.write_text('\n'.join(lines)+'\n')
    # Decimation can move the hull below the visual base (measured: -9.99 mm).
    # Rebuild an un-decimated single hull from the bridged visual to preserve extrema.
    import trimesh
    visual_mesh=trimesh.load(str(value.visual_obj),force='mesh',process=False,skip_materials=True)
    visual_mesh.convex_hull.export(str(value.collision_obj))
    from rfw_w.scenes.geometry.surface_appearance import digest
    metadata={'provider':'simready','library_version':'v1.3','asset_id':view.asset_key.asset_id,
        'origin_convention':'base','frame':'Z-up','collision_proxy':'single_convex_hull',
        'bounds_min_m':[round(a+d,6) for a,d in zip(lo,shift)],'bounds_max_m':[round(a+d,6) for a,d in zip(hi,shift)],
        'dimensions_m':value.dimensions_m.to_list(),'source_to_base_translation_m':shift,
        'physics':{'mass_kg':view.mass_kg,'source':'catalogue_declared:part_physics',
                   'friction_static':view.friction_static,'friction_dynamic':view.friction_dynamic},
        'preparation_version':'rigid-tool-base/v1'}
    for kind,path in (('visual',value.visual_obj),('collision',value.collision_obj)):
        metadata[kind+'_ref']=path.relative_to(root).as_posix();metadata[kind+'_sha256']=digest(path)
        import trimesh
        mesh=trimesh.load(str(path),force='mesh',process=False,skip_materials=True)
        metadata[kind+'_bounds_min_m']=mesh.bounds[0].tolist();metadata[kind+'_bounds_max_m']=mesh.bounds[1].tolist()
    # The cache metadata contains source paths and source-frame bounds; replace in delivery.
    (value.visual_obj.parent/'prepared.json').write_text(json.dumps(metadata,indent=2)+'\n')
    from rfw_w.scenes.geometry.rigid_tool_assets import verify_prepared_rigid
    verify_prepared_rigid(root,metadata)
    verify_task_shape(query['category'],metadata['dimensions_m'])
    require(all(abs(a-b)<2e-5 for a,b in zip(view.dimensions_m.to_list(),metadata['dimensions_m'])),
            'RIGID_FRAME_INVALID','query source dimensions changed')
    return metadata
