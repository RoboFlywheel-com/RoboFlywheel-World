# SPDX-License-Identifier: Apache-2.0
"""Export accepted measured animations to USD; this script does no rendering."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import shutil
import numpy as np


def prepare(cfg):
    import bpy
    from mathutils import Matrix, Quaternion, Vector
    print('PREPARE_START', cfg['id'], flush=True)
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False)
    trajectory = Path(cfg['trajectory'])
    raw = json.loads(trajectory.read_text())
    validation = json.loads(Path(cfg['validation']).read_text())
    digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    assert validation['status'] == 'passed' and all(validation.get('checks', {}).values())
    assert digest in (validation.get('trajectory_sha256'), validation.get('source_trajectory_sha256'))
    bpy.ops.wm.open_mainfile(filepath=cfg['blend'])
    scene = bpy.context.scene
    source_rows = raw['samples']
    rows = [r for r in source_rows if cfg.get('start_s', -1) <= r['t'] < cfg.get('end_s', 1e9)]
    if cfg.get('body_states'):
        raise ValueError('Use a dedicated body-state adapter')
    if cfg.get('g1_urdf'):
        import xml.etree.ElementTree as ET
        from mathutils import Euler
        joints = {j.get('name'): j for j in ET.parse(cfg['g1_urdf']).getroot().findall('joint')}
        for obj in bpy.data.objects:
            obj.animation_data_clear()
        for frame, sample in enumerate(rows, 1):
            for name, value in zip(raw['joint_names'], sample['joints']):
                joint = joints[name]
                obj = bpy.data.objects[joint.find('child').get('link')]
                origin = joint.find('origin')
                obj.location = Vector(map(float, origin.get('xyz', '0 0 0').split()))
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = Euler(tuple(map(float, origin.get('rpy','0 0 0').split())), 'XYZ').to_quaternion() @ Quaternion(Vector(map(float, joint.find('axis').get('xyz').split())), value)
                obj.keyframe_insert('rotation_quaternion', frame=frame)
            states = {'Mug': {'position':sample['position'], 'quaternion':sample['quaternion']}, **sample['scene_assets']}
            for name, state in states.items():
                obj = next(o for o in bpy.data.objects if o.name.startswith(name+'__'))
                obj.rotation_mode = 'QUATERNION'
                obj.location = state['position']
                obj.rotation_quaternion = Quaternion(state['quaternion'])
                obj.keyframe_insert('location',frame=frame)
                obj.keyframe_insert('rotation_quaternion',frame=frame)
    assert len(rows) == len(source_rows)
    scene.camera = bpy.data.objects.get('Contact') or scene.camera
    # Furniture with shared support moves as a group (books with their shelf,
    # chairs with the dining table). Interactive geometry stays fixed.
    arrangements = {
        'cover_bowl': [('HSSD_table', 'HSSD_chair', 'HSSD_pendant', 'Environment_Replay_runner')],
        'insert_pencil': [('HSSD_sofa', 'HSSD_lamp', 'Environment_Rug', 'Environment_Lounge')],
        'recover_dispenser': [('HSSD_bath', 'HSSD_plant', 'Environment_BathMat')],
        'tea_service': [('HSSD_table', 'HSSD_chair', 'HSSD_pendant')],
        'desk_tidy': [('HSSD_shelf', 'HSSD_books.')],
        'wash_station': [('HSSD_sink',)],
    }
    offsets = {'cover_bowl': (-.25, -.10), 'insert_pencil': (.30, -.25),
        'recover_dispenser': (.35, -.25), 'tea_service': (.30, -.30),
        'desk_tidy': (0, -.30), 'wash_station': (0, -.45)}
    moved = []
    for group in arrangements.get(cfg['id'], []):
        for obj in bpy.data.objects:
            if obj.parent is None and obj.name.startswith(group):
                before = list(obj.location)
                dx, dy = offsets[cfg['id']]
                obj.location.x += dx
                obj.location.y += dy
                moved.append(dict(name=obj.name, before=before, after=list(obj.location)))
    # Distinct camera compositions supplement the furniture arrangement.
    if scene.camera:
        camera = scene.camera
        camera.data.lens *= cfg.get('lens_factor', .88)
    scene.frame_start, scene.frame_end = 1, len(rows)
    scene.render.fps = 30
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for curve in obj.animation_data.action.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = 'LINEAR'
    scene.frame_set(1)
    # Bind the derivative to the exact accepted inputs, and retain per-frame
    # world transforms for an independent USD export check before publishing.
    shutil.copyfile(trajectory, out / 'trajectory.json')
    shutil.copyfile(cfg['validation'], out / 'physics_validation.json')
    moving = [o for o in bpy.data.objects if o.animation_data and o.animation_data.action]
    matrices = np.empty((len(rows), len(moving), 4, 4), dtype=np.float64)
    for frame in range(1, len(rows)+1):
        scene.frame_set(frame)
        for index, obj in enumerate(moving):
            matrices[frame-1, index] = np.array(obj.matrix_world).T
    np.savez_compressed(out/'measured_transforms.npz', names=np.array([o.name for o in moving]), matrices=matrices)
    scene.frame_set(1)
    # Release assets commonly share material_0.png; give every image a unique
    # export filename so one asset cannot silently replace another's texture.
    textures = out / 'source_textures'
    textures.mkdir()
    for index,image in enumerate(bpy.data.images):
        if image.type != 'IMAGE' or (not image.has_data and not image.packed_file):
            continue
        original_name=Path(image.filepath or image.name).name
        if Path(original_name).suffix.lower() not in ('.png','.jpg','.jpeg','.exr','.hdr','.tif','.tiff'):
            original_name += '.png'
        target = textures / (f'{index:04}_' + original_name)
        if image.packed_file:
            target.write_bytes(image.packed_file.data)
        elif Path(bpy.path.abspath(image.filepath)).is_file():
            shutil.copyfile(bpy.path.abspath(image.filepath), target)
        else:
            continue
        image.filepath = str(target)
        image.filepath_raw = str(target)
        image.name = target.name
        if image.packed_file:
            image.unpack(method='REMOVE')
    bpy.ops.wm.usd_export(filepath=str(out / 'scene.usdc'), export_animation=True,
        export_textures=True, relative_paths=True, visible_objects_only=True,
        generate_preview_surface=True, export_lights=True, export_cameras=True,
        author_blender_name=True, allow_unicode=False)
    from pxr import Usd, Sdf, UsdShade
    stage=Usd.Stage.Open(str(out/'scene.usdc'))
    # The Blender exporter can append datablock suffixes after image extensions.
    # Normalize every authored texture asset, not just a shader named Image_Texture.
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName()!=Sdf.ValueTypeNames.Asset or not attr.Get():
                continue
            asset=attr.Get().path
            if 'textures/' not in asset:
                continue
            path=out/'textures'/Path(asset).name
            if path.suffix.lower() not in ('.png','.jpg','.jpeg','.exr','.hdr','.tif','.tiff'):
                from PIL import Image
                target=path.with_name(path.name+'.png')
                Image.open(path).save(target)
                path=target
            assert path.is_file(), path.name
            attr.Set(Sdf.AssetPath('textures/'+path.name))
    stage.GetRootLayer().Save()
    # Large mesh-local offsets amplify quaternion export rounding. Author the
    # measured world matrices in double precision for every animated object.
    from pxr import UsdGeom, Gf
    object_prims={p.GetAttribute('userProperties:blender:object_name').Get():p for p in stage.Traverse()
        if p.GetAttribute('userProperties:blender:object_name')}
    for index,obj in enumerate(moving):
        prim=object_prims.get(obj.name)
        if prim is None:
            continue
        xform=UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        op=xform.AddTransformOp(opSuffix='measuredPrecision')
        for frame in range(len(rows)):
            parent=UsdGeom.XformCache(frame+1).GetLocalToWorldTransform(prim.GetParent())
            op.Set(Gf.Matrix4d(matrices[frame,index].tolist())*parent.GetInverse(),frame+1)
    stage.GetRootLayer().Save()
    (out / 'stage_provenance.json').write_text(json.dumps(dict(id=cfg['id'], title=cfg['title'],
        frames=len(rows), fps=30, trajectory_sha256=digest, source_physics='passed',
        renderer='pending Isaac Sim RTX', camera_name=scene.camera.name,
        layout=cfg['layout'], moved_background=moved, native_measured_animation=True,
        wide_camera='Scene' if bpy.data.objects.get('Scene') else None,
        source_blend_sha256=hashlib.sha256(Path(cfg['blend']).read_bytes()).hexdigest()), indent=2))
    print(cfg['id'], 'USD_READY', len(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    prepare(json.loads(Path(a.config).read_text()))
