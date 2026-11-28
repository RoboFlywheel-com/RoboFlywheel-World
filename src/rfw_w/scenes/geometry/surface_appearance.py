# SPDX-License-Identifier: Apache-2.0
"""Independent surface and material ruler, reading actual delivery geometry and files."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.surface_appearance import (
    SCHEMA, SurfaceAppearanceError, require, relative_ref, validate_graph_appearance, validate_material,
)
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.solvers.house_tools import _read_obj_groups, measure_house_access


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surfaces_from_mesh(graph, plan, geometry):
    measured = measure_house_access(plan, geometry)
    groups = _read_obj_groups(geometry / 'house.obj')
    nodes = graph.to_dict()['nodes']
    scopes = {(n['room_id'], n['kind']): n['id'] for n in nodes if n['kind'] in ('floor', 'wall_set')}
    result = {}
    for name, box in groups.items():
        x0, y0, z0, x1, y1, z1 = box
        for room in plan['rooms']:
            rid = room['id']; r = room['bounds_xy_m']
            cx, cy = (r[0]+r[2])/2, (r[1]+r[3])/2
            if name.startswith('floor_'):
                if [x0, y0, x1, y1] != r: continue
                kind, axis, plane, normal, uv_axes = 'floor', 'z', z1, [0, 0, 1], ['x', 'y']
                bounds = [x0, y0, x1, y1]
            elif name.startswith('wall_'):
                kind = 'wall_set'
                axis = 'x' if x1-x0 < y1-y0 else 'y'
                if axis == 'x':
                    if y1 <= r[1] or y0 >= r[3] or x1 < r[0]-.06 or x0 > r[2]+.06: continue
                    normal = [1, 0, 0] if cx > (x0+x1)/2 else [-1, 0, 0]
                    plane = x1 if normal[0] > 0 else x0
                    bounds, uv_axes = [max(y0,r[1]),z0,min(y1,r[3]),z1], ['y','z']
                else:
                    if x1 <= r[0] or x0 >= r[2] or y1 < r[1]-.06 or y0 > r[3]+.06: continue
                    normal = [0,1,0] if cy > (y0+y1)/2 else [0,-1,0]
                    plane = y1 if normal[1] > 0 else y0
                    bounds, uv_axes = [max(x0,r[0]),z0,min(x1,r[2]),z1], ['x','z']
            else: continue
            sid = name + ':' + rid
            result[sid] = {'surface_id':sid, 'room_id':rid, 'scope_id':scopes[(rid,kind)],
                           'kind':kind, 'mesh_group':name, 'axis':axis, 'plane_m':plane,
                           'normal':normal, 'uv_axes':uv_axes, 'bounds_uv_m':bounds,
                           'uv_origin_m':bounds[:2], 'mesh_sha256':measured['mesh_sha256']}
    return result


def checked_path(root, ref):
    relative_ref(ref)
    path = root / ref
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()),
            'SURFACE_BINDING_INVALID', 'missing or escaping file')
    return path


def verify_surface_appearance(root):
    graph = TaskWorldGraph.from_dict(json.loads((root/'graph.json').read_text()))
    raw = graph.to_dict(); value = raw['built'].get('surface_appearance')
    require(value is not None, 'SURFACE_BINDING_INVALID', 'appearance missing')
    validate_graph_appearance(value, {n['id']:n for n in raw['nodes']})
    ledger = json.loads(checked_path(root,value['ledger_ref']).read_text())
    require(set(ledger) == {'schema_version','source_graph_hash','house_mesh_sha256','core_ledger_hash',
                           'materials','surfaces','bindings','content_hash'}, 'SURFACE_BINDING_INVALID','ledger fields')
    require(ledger['schema_version'] == SCHEMA and content_hash({k:v for k,v in ledger.items() if k!='content_hash'})
            == ledger['content_hash'] == value['ledger_hash'], 'MATERIAL_DIGEST_MISMATCH','ledger')
    source = TaskWorldGraph.from_dict(json.loads((root/'appearance_source_graph.json').read_text()))
    require(source.content_hash == ledger['source_graph_hash'] and graph.content_hash != source.content_hash,
            'GRAPH_VERSION_CONFLICT','appearance source')
    source_raw=source.to_dict()
    covering_ids={n['id'] for n in raw['nodes'] if n.get('physical_class')=='covering'}
    covering_ids|={e['target_id'] for e in raw['edges'] if e['source_id'] in covering_ids}
    for key in ('nodes','edges','initial_state','steps','goals','requirements'):
        actual=raw[key]
        if key=='nodes':actual=[n for n in actual if n['id'] not in covering_ids]
        if key=='edges':actual=[e for e in actual if e['source_id'] not in covering_ids and e['target_id'] not in covering_ids]
        expected=source_raw[key]
        if key=='nodes':expected=[n for n in expected if n['id'] not in covering_ids]
        if key=='edges':expected=[e for e in expected if e['source_id'] not in covering_ids and e['target_id'] not in covering_ids]
        require(actual == expected, 'SURFACE_BINDING_INVALID','changed core '+key)
    for key in ('house_shell','asset_assembly','background_assembly'):
        require(raw['built'].get(key) == source.to_dict()['built'].get(key), 'SURFACE_BINDING_INVALID','changed '+key)
    plan = json.loads((root/'geometry/house_plan.json').read_text())
    surfaces = surfaces_from_mesh(graph,plan,root/'geometry')
    require(ledger['surfaces'] == surfaces and ledger['bindings'] == value['bindings'],
            'SURFACE_BINDING_INVALID','actual surfaces/bindings')
    require(digest(root/'geometry/house.obj') == ledger['house_mesh_sha256']
            == raw['built']['house_shell']['mesh_sha256'], 'MATERIAL_DIGEST_MISMATCH','house mesh')
    core_path = root/('core_asset_ledger.json' if (root/'core_asset_ledger.json').exists() else 'asset_ledger.json')
    core = json.loads(core_path.read_text())
    require(content_hash({k:v for k,v in core.items() if k!='content_hash'}) == core['content_hash']
            == ledger['core_ledger_hash'] == raw['built']['asset_assembly']['ledger_hash'],
            'MATERIAL_DIGEST_MISMATCH','core ledger')
    image_count = 0
    from PIL import Image
    for mid,material in ledger['materials'].items():
        validate_material(mid,material)
        for channel in material['channels'].values():
            path = checked_path(root,channel['ref'])
            require(digest(path) == channel['sha256'], 'MATERIAL_DIGEST_MISMATCH',mid)
            try:
                with Image.open(path) as im:
                    im.load(); require(min(im.size)>0,'MATERIAL_IMAGE_INVALID',mid)
            except (OSError,ValueError): raise SurfaceAppearanceError('MATERIAL_IMAGE_INVALID',mid) from None
            image_count += 1
    for b in ledger['bindings']:
        require(b['surface_id'] in surfaces and b['material_id'] in ledger['materials'],
                'SURFACE_BINDING_INVALID',b['surface_id'])
        s = surfaces[b['surface_id']]
        require(s['room_id'] == b['room_id'] and s['scope_id'] == b['scope_id'],
                'SURFACE_BINDING_INVALID',b['surface_id'])
    return {'verified':True,'problems':[],'graph_hash':graph.content_hash,'bindings':len(ledger['bindings']),
            'materials':len(ledger['materials']),'images_decoded':image_count,'core_preserved':True,
            'house_mesh_sha256':ledger['house_mesh_sha256'],'render':'not_run','user_visual':'not_run'}
