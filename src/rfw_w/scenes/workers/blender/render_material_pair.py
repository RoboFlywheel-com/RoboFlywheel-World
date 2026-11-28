#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Same-geometry Blender mapped/baked diagnostic; does not claim an Isaac light match.

The camera and the lighting come from `appearance_fixture.py`, the same single source the Isaac
side reads. Until they did, this script hard-coded `sensor_width_mm=22.044444` (117.334 vs
119.883 degrees against what Isaac rendered) and its own area lamp, so the two images in a
"comparison" were a different camera under different light.
"""
from rfw_w.scenes.resources import module_path, data_path
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def _fixture_module():
    return importlib.import_module('rfw_w.scenes.visualization.appearance_fixture')


def main():
    import bpy
    from mathutils import Vector
    from rfw_w.scenes.workers.blender.export_built_shell import architectural_mesh
    fixture_module=_fixture_module()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,
                        default=data_path('fixtures/appearance_fixture_v1.json'))
    parser.add_argument('--mode',default='room_like',
                        help='room_like reproduces the recorded pair; ambient_only is the mode a '
                             'three-way verdict may be written against')
    parser.add_argument('--disable-baked-normal',action='store_true',
                        help='Diagnostic ablation only; never a fidelity acceptance')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output_dir.exists():raise RuntimeError('output exists; use fresh evidence')
    args.output_dir.mkdir(parents=True)
    request=json.loads(args.request.read_text())
    fixture=fixture_module.load(args.fixture)
    rig=fixture_module.blender_rig(fixture,request,args.mode)
    rig['disable_baked_normal']=args.disable_baked_normal
    rows=[]
    for name in ('mapped','baked'):
        source=args.export_dir/(name+'.blend')
        started=time.monotonic()
        bpy.ops.wm.open_mainfile(filepath=str(source))
        scene=bpy.context.scene
        if name == 'baked' and args.disable_baked_normal:
            for mat in bpy.data.materials:
                if not mat.use_nodes:
                    continue
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        for link in list(node.inputs['Normal'].links):
                            mat.node_tree.links.remove(link)
        # The source can carry a compositor exposure pipeline; comparisons use raw renderer
        # colour management so both inputs share the explicit fixture below.
        scene.use_nodes=False
        for obj in scene.objects:obj.hide_render=not architectural_mesh(obj)
        scene.world=bpy.data.worlds.new('MaterialComparisonWorld')
        scene.world.use_nodes=True
        scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value=rig['world_strength']
        if rig['area_enabled']:
            light_data=bpy.data.lights.new('MaterialComparisonArea','AREA')
            light_data.energy=rig['area_energy_W'];light_data.shape='SQUARE';light_data.size=rig['area_size_m']
            light=bpy.data.objects.new('MaterialComparisonArea',light_data);scene.collection.objects.link(light)
            light.location=rig['area_location']
        camera_data=bpy.data.cameras.new('MaterialComparisonCamera')
        obj=bpy.data.objects.new('MaterialComparisonCamera',camera_data);scene.collection.objects.link(obj)
        obj.location=rig['eye'];obj.rotation_euler=(Vector(rig['target'])-obj.location).to_track_quat('-Z','Y').to_euler()
        # `aperture_mm` is the fixture's one horizontal aperture, the same number the worker
        # authors on the Isaac camera; Blender calls it the sensor width under a horizontal fit.
        camera_data.lens=rig['focal_length_mm'];camera_data.sensor_fit=rig['sensor_fit'];camera_data.sensor_width=rig['aperture_mm']
        scene.camera=obj
        scene.render.engine='CYCLES';scene.cycles.device='GPU';scene.cycles.samples=32
        prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='CUDA';prefs.get_devices()
        for device in prefs.devices:device.use=device.type=='CUDA'
        scene.render.resolution_x=rig['resolution'][0];scene.render.resolution_y=rig['resolution'][1]
        scene.render.resolution_percentage=100
        scene.view_settings.view_transform='Standard';scene.view_settings.look='Medium High Contrast'
        scene.view_settings.exposure=0;scene.view_settings.gamma=1
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(args.output_dir/(name+'.png'))
        bpy.ops.render.render(write_still=True)
        rows.append({'source':name+'.blend','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'elapsed_s':round(time.monotonic()-started,3)})
    (args.output_dir/'report.json').write_text(json.dumps({
        'fixture':{'fixture_id':fixture['fixture_id'],'mode':args.mode,'role':rig['role'],
                   'photometric_equivalence':rig['photometric_equivalence']},
        'rig':rig,'renders':rows,
        'role_note':'A diagnostic mode is for looking at, not for a verdict: with sun and fill on, '
                    'a material loss and a lighting difference cannot be told apart.',
        'human_review':'not_run'},indent=2)+'\n')


if __name__=='__main__':main()
