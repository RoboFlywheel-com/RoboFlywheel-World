# SPDX-License-Identifier: Apache-2.0
"""Source-bound rigid reference worlds. Task profiles live in the demo client."""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from rfw_w.scenes.adapters.reference_world import freeze_dependencies
from rfw_w.scenes.assets.articulated_urdf import digest, values
from rfw_w.scenes.contracts.reference_episode import EpisodeSpec
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph


def build(config, output, *, graph, template, profile):
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation
    from PIL import Image, ImageDraw
    graph = TaskWorldGraph.from_dict(graph) if isinstance(graph, dict) else graph
    roles = {n['id'] for n in graph.to_dict()['nodes'] if n['kind'] == 'asset_role'}
    object_roles={o['role'] for o in profile['objects']}
    garden=profile['task_id']=='T12' and object_roles=={'pot','saucer','can'}
    bench=profile.get('preset')=='bench_pick_place_v1' and bool(object_roles)
    if not (garden or bench) or not object_roles.union({'source_area','workbench'})<=roles:
        raise ValueError('REFERENCE_RIGID_PRESET_UNSUPPORTED')
    if bench and (len(object_roles)!=len(profile['objects']) or any(o.get('goal_support_role') for o in profile['objects'])):
        raise ValueError('BENCH_PROFILE_UNSUPPORTED')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = ET.parse(template).getroot()
    world, asset = root.find('worldbody'), root.find('asset')
    for item in list(world):
        if item.tag != 'body' or item.get('name') != 'mobile_base':
            world.remove(item)
    for item in list(asset):
        if item.get('name', '').startswith('V13_'):
            asset.remove(item)
    for item in root.iter():
        for k, v in list(item.attrib.items()):
            item.set(k, v.replace('${ROBOT_SOURCES}', config['robot_sources']))
    world.find("body[@name='mobile_base']").set('pos', values(profile.get('dock_position_m', [-.12, 0, .127])))
    root.find('option').set('timestep', '.001')
    bindings = {'robot': 'mobile_base', 'workbench': 'garden_bench', 'source_area': 'garden_bench'}
    physics_records = {}
    for item in profile['objects']:
        leaf, role = item['leaf'], item['role']
        folder = Path(config['scratch']) / 'assets' / leaf
        source = ET.parse(folder / 'asset.xml').getroot()
        prefix = 'V13_' + leaf
        names = {}
        for element in source.find('asset'):
            copied = deepcopy(element)
            name = copied.get('name')
            names[name] = prefix + '_' + name
            copied.set('name', names[name])
            if copied.get('file'):
                copied.set('file', str(folder / copied.get('file')))
            asset.append(copied)
        for element in asset:
            if element.get('name', '').startswith(prefix):
                for key in ('texture',):
                    if element.get(key) in names:
                        element.set(key, names[element.get(key)])
        body = deepcopy(source.find('./worldbody/body'))
        body.set('name', prefix)
        rotation = Rotation.from_euler('xyz', item.get('initial_rpy', [0, 0, 0]))
        vertices = trimesh.load(folder / 'visual.obj', force='mesh', process=False).vertices
        bounds = np.array(vertices @ rotation.as_matrix().T)
        z = profile['surface_z_m'] + item.get('source_raise_m', 0) - bounds[:, 2].min() + .002
        body.set('pos', values([*item['source_xy_m'], z]))
        body.set('quat', values(rotation.as_quat()[[3, 0, 1, 2]]))
        body.find('joint').set('name', prefix + '_free')
        for index, geom in enumerate(body.findall('geom')):
            geom.set('name', prefix + '_' + ('visual' if geom.get('contype') == '0' else 'collision_' + str(index)))
            for key in ('mesh', 'material'):
                if geom.get(key) in names:
                    geom.set(key, names[geom.get(key)])
            geom.set('solref', '.002 1')
            geom.set('solimp', '.98 .99 .001')
        world.append(body)
        bindings[role] = prefix
        raw = json.loads((folder / 'part_physics.json').read_text())
        public = {k: v for k, v in raw.items() if k not in ('urdf', 'timings', 'collision')}
        ref = leaf + '_physics.json'
        (output / ref).write_text(json.dumps(public, indent=2))
        physics_records[leaf] = dict(ref=ref, sha256=digest(output / ref), source_sha256=digest(folder / 'part_physics.json'),
            source_xml_sha256=digest(folder / 'asset.xml'), mass_kg=raw['total_mass'], source_scale=1,
            source_axis_record=raw.get('up_axis_fix'), initial_rpy=item.get('initial_rpy', [0, 0, 0]),
            coordinate_preparation='native release MJCF; explicit task upright rotation applied once to body',
            prepared_bounds_m=[bounds.min(0).tolist(), bounds.max(0).tolist()])
    generated = output / 'scene_resources'
    generated.mkdir()
    for name, color in [('clay', (171, 119, 86)), ('plaster', (225, 216, 194)), ('wood', (147, 109, 66))]:
        rng = np.random.default_rng(47)
        pixels = np.clip(np.array(color)[None, None, :] + rng.normal(0, 2, (256, 256, 1)), 0, 255).astype('uint8')
        image = Image.fromarray(pixels)
        if name == 'clay':
            draw = ImageDraw.Draw(image)
            for i in range(0, 256, 64):
                draw.line((i, 0, i, 255), fill=(118, 91, 71), width=2)
                draw.line((0, i, 255, i), fill=(118, 91, 71), width=2)
        image.save(generated / (name + '.png'))
        ET.SubElement(asset, 'texture', name='garden_' + name + '_tex', type='2d', file=str(generated / (name + '.png')))
        ET.SubElement(asset, 'material', name='garden_' + name, texture='garden_' + name + '_tex', texrepeat='4 4', texuniform='true')
    ET.SubElement(asset, 'material', name='garden_steel', rgba='.10 .17 .14 1', specular='.3')
    fixtures = []
    def box(parent, name, pos, size, material='wood'):
        fixtures.append(name)
        ET.SubElement(parent, 'geom', name=name, type='box', pos=values(pos), size=values(size),
                      material='garden_' + material, friction='.7 .005 .0001', solref='.002 1')
    box(world, 'floor', [0, 0, -.05], [3, 2.5, .05], 'clay')
    box(world, 'rear_wall', [-2.2, 0, 1.4], [.06, 2.5, 1.4], 'plaster')
    box(world, 'side_wall', [0, 2.5, 1.4], [3, .06, 1.4], 'plaster')
    surface = profile['surface_z_m']
    bench = ET.SubElement(world, 'body', name='garden_bench')
    box(bench, 'garden_worktop', [-1.15, 0, surface - .025], [.33, .75, .025])
    for x in (-1.40, -.90):
        for y in (-.63, .63):
            box(bench, 'bench_leg_' + str(x) + '_' + str(y), [x, y, (surface - .05) / 2], [.025, .025, (surface - .05) / 2], 'steel')
    box(bench, 'bench_shelf', [-1.15, 0, .30], [.30, .71, .02])
    for item in profile['objects']:
        height = item.get('source_raise_m', 0)
        if height:
            box(bench, 'source_cradle_' + item['role'], [*item['source_xy_m'], surface + height / 2], [.028, .028, height / 2], 'steel')
    box(world, 'seed_shelf', [-1.98, 1.45, 1.25], [.17, .50, .02])
    for i in range(7 if garden else 0):
        box(world, 'seed_packet_' + str(i), [-1.92, 1.10 + .11 * i, 1.36], [.015, .044, .10], 'plaster')
    for i in range(17 if garden else 0):
        box(world, 'balcony_bar_' + str(i), [-1.8 + .24 * i, 2.20, .60], [.012, .012, .60], 'steel')
    if garden:box(world, 'balcony_rail', [.12, 2.20, 1.21], [1.94, .03, .025], 'steel')
    for z in ((1.35, 2.3) if garden else ()):
        box(world, 'window_frame_' + str(z), [-.2, 2.415, z], [.9, .025, .025])
    for x in ((-1.1, -.2, .7) if garden else ()):
        box(world, 'window_mullion_' + str(x), [x, 2.415, 1.825], [.022, .025, .50])
    ET.SubElement(world, 'light', name='garden_daylight', pos='0 1.2 3', dir='-.5 -.3 -1', diffuse='.9 .87 .76', ambient='.15 .15 .12')
    ET.SubElement(world, 'light', name='garden_fill', pos='0 -2 2.7', dir='-.6 .4 -1', diffuse='.65 .68 .7')
    for name in ('overview', 'detail'):
        ET.SubElement(world, 'camera', name=name, pos='1 -3 2.3')
    render_profile = dict(cameras=profile['cameras'], headlight_ambient=.3, headlight_diffuse=.45)
    (output / 'render_profile.json').write_text(json.dumps(render_profile, indent=2))
    freeze_dependencies(root, output, Path(template).parent)
    ET.ElementTree(root).write(output / 'world.xml', encoding='unicode')
    stages = tuple(dict(name=a['role'] + '_place', command='pick_transport_place_release', targets=[a['role']],
                        requires=['initial_scene_valid', 'robot_ready'], ensures=[a['role'] + '_supported', 'hand_released']) for a in profile['objects'])
    episode = EpisodeSpec(profile['task_id'], graph.content_hash, bindings, stages)
    (output / 'graph.json').write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))
    (output / 'episode.json').write_text(json.dumps(episode.to_dict(), indent=2))
    (output / 'task_profile.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2))
    manifest = dict(schema_version='pvscene.reference-world/v1', task_id=profile['task_id'], graph_hash=graph.content_hash,
                    episode_hash=episode.content_hash, world_sha256=digest(output / 'world.xml'), bindings=bindings,
                    rigid_physics=physics_records, dependencies=json.loads((output / 'dependencies.json').read_text()),
                    profile_sha256=digest(output / 'task_profile.json'), render_profile_sha256=digest(output / 'render_profile.json'),
                    authored_fixtures=fixtures, layout_policy='task_bound_garden_reference_v1' if garden else 'bench_pick_place_v1', status='built_pending_validation',
                    units='m/kg/s/rad', up_axis='Z', task_ready=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
