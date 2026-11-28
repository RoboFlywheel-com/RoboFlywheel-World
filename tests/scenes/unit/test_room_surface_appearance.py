# SPDX-License-Identifier: Apache-2.0
"""Adversarial checks for the independent appearance ruler."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image
from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.contracts.scene_composition import migrate_composition
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.contracts.surface_appearance import SCHEMA, CHANNEL_SPACES, SurfaceAppearanceError
from rfw_w.scenes.geometry.surface_appearance import surfaces_from_mesh, verify_surface_appearance
from rfw_w.scenes.solvers.house_tools import solve_house_layout, build_house_shell


def fixture_bundle(root):
    fixtures=Path(__file__).resolve().parents[1]/'fixtures/task_world'
    g=compile_pick_place_slots(json.loads((fixtures/'cross_room_slots_response.json').read_text())['slots'],
                               json.loads((fixtures/'cross_room_request.json').read_text()))
    g=migrate_composition(g,expected_hash=g.content_hash)
    root.mkdir();plan=solve_house_layout(g);built=build_house_shell(plan,root/'geometry')
    core={'instances':[]};core['content_hash']=content_hash(core)
    (root/'core_asset_ledger.json').write_text(json.dumps(core))
    raw=g.to_dict();raw['built']={'house_shell':built,'asset_assembly':{'ledger_hash':core['content_hash']}}
    g=g.revise(raw,expected_hash=g.content_hash)
    (root/'appearance_source_graph.json').write_text(json.dumps(g.to_dict()))
    materials={'plaster':{'provider':'local_pbr','base_color_rgba':[1,1,1,1],'roughness':.8,'channels':{}}}
    for name,space in CHANNEL_SPACES.items():
        p=root/(name+'.png');Image.new('RGB',(4,4),(120,120,255)).save(p)
        materials['plaster']['channels'][name]={'ref':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'color_space':space}
    surfaces=surfaces_from_mesh(g,plan,root/'geometry')
    bindings=[{'surface_id':sid,'room_id':s['room_id'],'scope_id':s['scope_id'],'material_id':'plaster',
               'meters_per_tile':1,'intent':{'source':'inferred','description':'plaster'}} for sid,s in surfaces.items()]
    ledger={'schema_version':SCHEMA,'source_graph_hash':g.content_hash,'house_mesh_sha256':built['mesh_sha256'],
            'core_ledger_hash':core['content_hash'],'materials':materials,'surfaces':surfaces,'bindings':bindings}
    save_fixture(root,g,ledger)
    return g,ledger


def save_fixture(root,source,ledger):
    ledger.pop('content_hash',None);ledger['content_hash']=content_hash(ledger)
    raw=source.to_dict();raw['built']['surface_appearance']={'schema_version':SCHEMA,'ledger_ref':'surface_appearance.json',
        'ledger_hash':ledger['content_hash'],'bindings':ledger['bindings']};raw['evidence_refs']=[]
    g=source.revise(raw,expected_hash=source.content_hash)
    (root/'graph.json').write_text(json.dumps(g.to_dict()));(root/'surface_appearance.json').write_text(json.dumps(ledger))


def test_verifier_reads_real_images_and_shared_sides(tmp_path):
    root=tmp_path/'bundle';_,ledger=fixture_bundle(root);r=verify_surface_appearance(root)
    assert r['verified'] and r['images_decoded']==3
    shared=[s for s in ledger['surfaces'].values() if s['mesh_group']=='wall_shared_lintel']
    assert len(shared)==2 and shared[0]['normal']!=shared[1]['normal']


@pytest.mark.parametrize('mutation,code',[
    ('channel','MATERIAL_CHANNEL_MISSING'),('colorspace','MATERIAL_CHANNEL_MISSING'),
    ('escape','SURFACE_BINDING_INVALID'),('wrong_side','SURFACE_BINDING_INVALID'),
    ('uv','SURFACE_UV_INVALID'),('fake_surface','SURFACE_BINDING_INVALID'),
    ('core','MATERIAL_DIGEST_MISMATCH'),('source','GRAPH_VERSION_CONFLICT')])
def test_tampered_rehashed_ledgers_rejected(tmp_path,mutation,code):
    root=tmp_path/'bundle';g,l=fixture_bundle(root);l=copy.deepcopy(l)
    if mutation=='channel':del l['materials']['plaster']['channels']['roughness']
    if mutation=='colorspace':l['materials']['plaster']['channels']['normal_gl']['color_space']='sRGB'
    if mutation=='escape':l['materials']['plaster']['channels']['base_color']['ref']='../base_color.png'
    if mutation=='wrong_side':l['bindings'][0]['room_id']='study'
    if mutation=='uv':l['bindings'][0]['meters_per_tile']=0
    if mutation=='fake_surface':l['surfaces'][next(iter(l['surfaces']))]['plane_m']=99
    if mutation=='core':l['core_ledger_hash']='0'*64
    if mutation=='source':l['source_graph_hash']='0'*64
    with pytest.raises((SurfaceAppearanceError,GraphError),match=code):
        save_fixture(root,g,l)
        verify_surface_appearance(root)


def test_bad_image_and_digest_are_distinct(tmp_path):
    root=tmp_path/'bundle';g,l=fixture_bundle(root);p=root/'base_color.png';p.write_bytes(b'not an image')
    with pytest.raises(SurfaceAppearanceError,match='MATERIAL_DIGEST_MISMATCH'):verify_surface_appearance(root)
    l['materials']['plaster']['channels']['base_color']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    save_fixture(root,g,l)
    with pytest.raises(SurfaceAppearanceError,match='MATERIAL_IMAGE_INVALID'):verify_surface_appearance(root)


def test_symlink_escape_rejected(tmp_path):
    root=tmp_path/'bundle';_,_=fixture_bundle(root)
    p=root/'base_color.png';data=p.read_bytes();p.unlink();external=tmp_path/'outside.png';external.write_bytes(data);p.symlink_to(external)
    with pytest.raises(SurfaceAppearanceError,match='SURFACE_BINDING_INVALID'):verify_surface_appearance(root)


def test_search_no_unrelated_style_fallback_and_pbr_prepare(tmp_path,monkeypatch):
    from rfw_w.scenes.assets.surface_materials import search_surface_materials,prepare_material
    library=tmp_path/'library';material=library/'Wood094_1K-JPG';material.mkdir(parents=True)
    for suffix in ('Color','NormalGL','Roughness'):Image.new('RGB',(8,8),(120,80,30)).save(material/(suffix+'_'+suffix+'.png'))
    monkeypatch.setenv('PVSCENE_MATERIALS_ROOT',str(library))
    assert search_surface_materials('marble tiles','floor')['candidates']==[]
    q=search_surface_materials('oak wood','floor');assert q['candidates'][0]['material_id']==material.name
    root=tmp_path/'out';root.mkdir();m=prepare_material(material.name,root)
    assert len(m['channels'])==3 and all((root/c['ref']).is_file() for c in m['channels'].values())


def test_transactions_preserve_other_room_and_reject_stale_query(tmp_path,monkeypatch):
    from rfw_w.scenes.orchestrator.surface_appearance_tools import set_surface_materials,rollback_surface_appearance
    from rfw_w.scenes.assets.surface_materials import search_surface_materials
    library=tmp_path/'lib';library.mkdir();monkeypatch.setenv('PVSCENE_MATERIALS_ROOT',str(library))
    source=tmp_path/'source';fixture_bundle(source)
    from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
    g=TaskWorldGraph.from_dict(json.loads((source/'graph.json').read_text()))
    q=search_surface_materials('sage','wall_set')
    first=tmp_path/'first';g1,_=set_surface_materials(g,source,first,room_id='kitchen',wall_material_id='solid:sage',
        expected_hash=g.content_hash,description='sage',queries=(q,))
    second=tmp_path/'second';g2,_=set_surface_materials(g1,first,second,room_id='kitchen',wall_material_id='solid:blue',
        expected_hash=g1.content_hash,description='blue',queries=(search_surface_materials('blue','wall_set'),))
    restored=tmp_path/'restore';gr,_=rollback_surface_appearance(g2,second,first,restored,room_id='kitchen',expected_hash=g2.content_hash)
    b1=g1.to_dict()['built']['surface_appearance']['bindings'];br=gr.to_dict()['built']['surface_appearance']['bindings']
    assert b1==br and verify_surface_appearance(restored)['verified']
    with pytest.raises(SurfaceAppearanceError,match='GRAPH_VERSION_CONFLICT'):
        set_surface_materials(g2,second,tmp_path/'bad',room_id='study',wall_material_id='solid:sage',
            expected_hash=g1.content_hash,description='sage',queries=(q,))
