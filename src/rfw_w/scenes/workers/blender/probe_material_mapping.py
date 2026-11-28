#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify explicit upstream mappings on a source floor, in a fresh evidence directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output_dir.exists():
        raise RuntimeError('output exists; use fresh evidence')
    args.output_dir.mkdir(parents=True)
    root = Path(os.environ['RFW_INFINIGEN_ROOT'])
    from rfw_w.scenes.workers.blender.infinigen_materials import make_material, mapping
    from infinigen.tools.export import bake_object
    from rfw_w.scenes.solvers.shell_providers import INFINIGEN_PIN
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != INFINIGEN_PIN:
        raise RuntimeError('MATERIAL_BAKE_UNAVAILABLE(source commit mismatch)')
    source = Path(bpy.data.filepath)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    floors = [o for o in bpy.context.scene.objects if o.type == 'MESH' and
              any(c.name == 'unique_assets:room_floor' for c in o.users_collection)]
    if len(floors) != 1:
        raise RuntimeError('MATERIAL_BINDING_MISMATCH(source floor)')
    original = floors[0]
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'CUDA'
    preferences.get_devices()
    devices = [d for d in preferences.devices if d.type == 'CUDA']
    if not devices:
        raise RuntimeError('MATERIAL_BAKE_UNAVAILABLE(no CUDA device)')
    for device in preferences.devices:
        device.use = device in devices
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 8
    for obj in scene.objects:
        obj.hide_render = True
    negative = False
    try:
        mapping('unsupported_shader', 'floor')
    except ValueError as error:
        negative = str(error).startswith('MATERIAL_INTENT_UNSUPPORTED')
    rows = []
    for token, role in [('oak', 'floor'), ('off_white', 'wall'), ('ceramic_tile', 'floor')]:
        for seed in (20, 21):
            out = args.output_dir / f'{token}_{seed}'
            out.mkdir()
            started = time.monotonic()
            mat, summary = make_material(token, role, seed)
            repeat, repeat_summary = make_material(token, role, seed)
            reproducible = summary['sha256'] == repeat_summary['sha256']
            bpy.data.materials.remove(repeat)
            obj = original.copy()
            obj.data = original.data.copy()
            scene.collection.objects.link(obj)
            obj.hide_render = False
            obj.hide_set(False)
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            for face in obj.data.polygons:
                face.material_index = 0
            displacement = any(n.type == 'OUTPUT_MATERIAL' and n.inputs['Displacement'].is_linked
                               for n in mat.node_tree.nodes)
            # Freeze the mapped source for later original/baked visual comparison.
            bpy.ops.wm.save_as_mainfile(filepath=str(out / 'mapped.blend'))
            bake_object(obj, out, args.resolution, True, export_name='surface')
            channels = {}
            for channel in ('DIFFUSE', 'ROUGHNESS', 'NORMAL'):
                path = out / f'surface_{channel}.png'
                image = next(n.image for slot in obj.material_slots if slot.material
                             for n in slot.material.node_tree.nodes
                             if n.type == 'TEX_IMAGE' and n.image and
                             Path(bpy.path.abspath(n.image.filepath)).resolve() == path.resolve())
                channels[channel] = {'uri': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'resolution': list(image.size), 'colour_space': image.colorspace_settings.name}
            bpy.ops.wm.save_as_mainfile(filepath=str(out / 'baked.blend'))
            row = {'token': token, 'role': role, 'seed': seed, 'material': summary,
                   'reproducible': reproducible, 'channels': channels,
                   'source_displacement_linked': displacement,
                   'displacement_fidelity': 'not_run: requires original/baked normal and silhouette comparison',
                   'elapsed_s': round(time.monotonic()-started, 3)}
            (out / 'report.json').write_text(json.dumps(row, indent=2)+'\n')
            rows.append(row)
            obj.hide_render = True
            bpy.data.objects.remove(obj, do_unlink=True)
    unchanged = source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    report = {'scope': 'mapping and bake probe; not full SPEC-007 acceptance',
              'source_commit': commit, 'source_sha256': source_hash, 'source_unchanged': unchanged,
              'blender_version': bpy.app.version_string, 'devices': [d.name for d in devices],
              'cases': rows, 'unknown_token_negative_passed': negative,
              'probe_passed': unchanged and negative and all(r['reproducible'] for r in rows),
              'visual_review': 'not_run', 'spec_accepted': False}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['probe_passed']:
        raise RuntimeError('MATERIAL_MAPPING_PROBE_FAILED')


if __name__ == '__main__':
    main()
