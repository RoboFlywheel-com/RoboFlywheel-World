# SPDX-License-Identifier: Apache-2.0
"""Build a task-bound source-scale workshop through scene pipeline contracts/adapters."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from rfw_w.scenes.assets.articulated_urdf import import_urdf, values, digest
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.contracts.reference_episode import EpisodeSpec
from rfw_w.scenes.contracts.reference_program import PHASES


def build(config, output, *, graph, template):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    graph = TaskWorldGraph.from_dict(graph) if isinstance(graph, dict) else graph
    raw=graph.to_dict()
    roles={n['id'] for n in raw['nodes'] if n['kind']=='asset_role'}
    if not {'workbench','drawer','caliper','screwdriver'}<=roles or not any(n.get('room_kind')=='workshop' for n in raw['nodes']):
        raise ValueError('REFERENCE_PRESET_TASK_UNSUPPORTED')
    template = Path(template)
    root = ET.parse(template).getroot()
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            for old, new in {'${ROBOT_SOURCES}': config['robot_sources'],
                             '${TASK_ASSETS}': str(Path(config['scratch'])/'assets')}.items():
                value = value.replace(old, new)
            if '${' in value:
                raise ValueError('RESOURCE_UNRESOLVED')
            element.set(key, value)
    world = root.find('worldbody')
    dock = config.get('dock_position_m', [-.12, 0, .127])
    world.find("body[@name='mobile_base']").set('pos', values(dock))
    for element in list(world):
        if element.tag == 'body' and element.get('name') in ('mobile_base', 'V13_11_2_0007', 'V13_9_6_0000'):
            continue
        world.remove(element)
    cabinet_folder = Path(config['pva_root'])/'Other_Cabinet/seed_0000'
    cabinet = import_urdf(root, cabinet_folder, position=(-1.55, 0, 0))
    # Source guide sockets intersect the drawer sides/front by up to 18 mm.
    # Relocate the fixed guide into the measured side clearance, in both meshes.
    guide_repairs = []
    carcass = world.find("body[@name='cabinet_carcass']")
    for index in (12, 13):
        for kind in ('visual', 'collision'):
            geom = carcass.find(f"geom[@name='cabinet_carcass_{kind}_{index}']")
            before = geom.get('pos')
            pos = [float(v) for v in before.split()]
            pos[:2] = [.46, -.725]
            geom.set('pos', values(pos))
            geom.set('size', '.01 .006 .14652')
            guide_repairs.append(dict(geom=geom.get('name'), before_position=before,
                                      after_position=geom.get('pos'), source_half_size='.01 .016 .14652',
                                      after_half_size=geom.get('size')))
    drawer = world.find("body[@name='cabinet_carcass']/body[@name='cabinet_drawer_1']")
    # The URDF omits rail dynamics. Explicit viscous adaptation damps impact
    # motion without locking the slider or driving it to an arbitrary pose.
    drawer.find('joint').set('damping', str(config.get('rail_damping_ns_m', 1.)))
    ET.SubElement(drawer, 'site', name='drawer_handle', pos='.0225 .355125 0', size='.004')
    # This explicit divider moves with the real drawer, preserves its source inertia,
    # and contributes its own measured mass/inertia via a fixed child body.
    fixture = ET.SubElement(drawer, 'body', name='compartment_divider', pos='-.228 .71625 -.03')
    ET.SubElement(fixture, 'geom', name='divider', type='box', size='.198 .006 .125',
                  density='750', friction='.35 .005 .0001', rgba='.45 .29 .15 1')
    ET.SubElement(drawer, 'site', name='left_compartment', pos='-.12 .46 -.153', size='.004')
    ET.SubElement(drawer, 'site', name='right_compartment', pos='-.12 .97 -.153', size='.004')
    # Narrow source pads on the cabinet's existing top: tool cradles, no extra table.
    for leaf, xy, z in [('11_2_0007', (-1.12, -.26), 1.025), ('9_6_0000', (-1.10, .18), 1.025)]:
        xy = config.get('source_positions_xy', {}).get(leaf, xy)
        body = world.find(f"body[@name='V13_{leaf}']")
        body.set('pos', values([*xy, z]))
        ET.SubElement(world, 'geom', name='source_cradle_'+leaf, type='box',
                      pos=values([xy[0], xy[1], .989]), size=values(config.get('source_pad_half_size_m', {}).get(leaf, [.04,.025,.0184])),
                      friction='.8 .005 .0001', rgba='.16 .18 .17 1')
    # Room geometry and peripheral furniture are in the same physical world.
    structures = [('floor', (0, 0, -.05), (3, 2.5, .05)),
                  ('rear_wall', (-2.2, 0, 1.4), (.06, 2.5, 1.4)),
                  ('side_wall', (0, 2.5, 1.4), (3, .06, 1.4)),
                  ('tool_board', (-2.08, 0, 1.5), (.035, .8, .45)),
                  ('shelf_side', (-1.9, 1.65, .7), (.20, .025, .7)),
                  ('shelf_side2', (-1.9, 2.15, .7), (.20, .025, .7))]
    for level in (.15, .6, 1.05, 1.4):
        structures.append(('shelf_'+str(level), (-1.9, 1.9, level), (.2, .275, .018)))
    for name, pos, size in structures:
        ET.SubElement(world, 'geom', name=name, type='box', pos=values(pos), size=values(size),
                      friction='.7 .005 .0001', rgba='.47 .49 .46 1' if 'wall' in name else '.24 .27 .28 1')
    ET.SubElement(world, 'light', pos='0 -2 3', dir='-.3 .3 -1', diffuse='.9 .9 .9')
    ET.SubElement(world, 'camera', name='overview', pos='.8 -3 1.9', xyaxes='1 0.35 0 -.15 .43 .89', fovy='48')
    ET.SubElement(world, 'camera', name='detail', pos='.2 -1.7 1.35', xyaxes='.85 .53 0 -.14 .22 .965', fovy='42')
    dressing = None
    if config.get('scene_dressing'):
        from rfw_w.scenes.adapters.reference_scene import dress
        dressing = dress(root, output, config['scene_dressing'])
    freeze_dependencies(root, output, template.parent)
    root.find('option').set('timestep', '.001')
    ET.ElementTree(root).write(output/'world.xml', encoding='unicode')
    rigid_physics={}
    for leaf in ('11_2_0007','9_6_0000'):
        source=Path(config['scratch'])/'assets'/leaf/'part_physics.json'
        physics=json.loads(source.read_text())
        dest=output/f'{leaf}_physics.json'
        private_fields=('urdf','timings','collision')
        public={k:v for k,v in physics.items() if k not in private_fields}
        dest.write_text(json.dumps(public,indent=2))
        rigid_physics[leaf]={'ref':dest.name,'sha256':digest(dest),'source_sha256':digest(source),'mass_kg':physics['total_mass'],'up_axis_fix':physics.get('up_axis_fix'),'inertia':physics['inertia']}
    bindings = {'robot': 'mobile_base', 'drawer': 'cabinet_drawer_1',
                'caliper': 'V13_11_2_0007', 'screwdriver': 'V13_9_6_0000',
                'workbench': 'cabinet_carcass', 'left_compartment': 'left_compartment',
                'right_compartment': 'right_compartment'}
    stages = tuple(dict(name=p.name, command=p.command, targets=list(p.targets),
                        requires=list(p.requires), ensures=list(p.ensures)) for p in PHASES)
    episode = EpisodeSpec('T03', graph.content_hash, bindings, stages)
    (output/'graph.json').write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))
    (output/'episode.json').write_text(json.dumps(episode.to_dict(), indent=2))
    manifest = dict(schema_version='pvscene.reference-world/v1', graph_hash=graph.content_hash,
                    episode_hash=episode.content_hash, world_sha256=digest(output/'world.xml'),
                    rigid_physics=rigid_physics,
                    robot_template_sha256=digest(template), articulated=cabinet, bindings=bindings,
                    derived_guide_clearance_repairs=guide_repairs,
                    rail_damping_adapter={'value_ns_m':config.get('rail_damping_ns_m',1.),
                                          'status':'estimated','source_dynamics':'absent','joint_drive':False},
                    authored_fixtures=['compartment_divider', 'source_cradle_11_2_0007', 'source_cradle_9_6_0000'],
                    structures=[s[0] for s in structures], units='m/kg/s/rad', up_axis='Z',
                    dependencies=json.loads((output/'dependencies.json').read_text()),
                    layout_policy='admitted_workshop_reference_v1',
                    scene_dressing={'ref':'scene_dressing.json','sha256':digest(output/'scene_dressing.json'),
                        'profile':dressing['profile']} if dressing else None,
                    render_profile_sha256=digest(output/'render_profile.json') if dressing else None,
                    dock_position_m=dock,
                    source_positions_xy=config.get('source_positions_xy', {}),
                    status='built_pending_validation', task_ready=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest



def freeze_dependencies(root, output, template_dir):
    """Copy only selected runtime inputs into the ignored world, with hashes."""
    import shutil
    compiler=root.find('compiler')
    meshdir=Path(compiler.get('meshdir', str(template_dir)))
    records={}
    for element in root.iter():
        if not element.get('file'):
            continue
        source=Path(element.get('file'))
        if not source.is_absolute():
            source=meshdir/source
        key=digest(source)[:16]+'_'+source.name
        dest=output/'resources'/key
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(source,dest)
        if element.tag=='mesh' and not element.get('name'):
            element.set('name',Path(element.get('file')).stem)
        element.set('file','resources/'+key)
        records['resources/'+key]={'sha256':digest(dest),'bytes':dest.stat().st_size,'source_name':source.name}
    compiler.attrib.pop('meshdir',None)
    (output/'dependencies.json').write_text(json.dumps(records,indent=2))
