#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export measured architecture in Blender, optionally baking an explicit PBR profile."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


# The pinned builder's semantic collections distinguish the shell from solver placeholders,
# actual coarse furniture, camera helpers and reusable door parts. Observed on the 2026-09-22
# kitchen export: 61 meshes existed but only 10 belonged to the built architecture.
ARCHITECTURAL_COLLECTIONS = frozenset({
    'unique_assets:room_wall', 'unique_assets:room_floor', 'unique_assets:room_ceiling',
    'unique_assets:room_exterior', 'unique_assets:doors', 'unique_assets:windows', 'skirting',
})


def architectural_mesh(obj):
    return (obj.type == 'MESH' and not obj.hide_render
            and any(c.name in ARCHITECTURAL_COLLECTIONS for c in obj.users_collection))


def main():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--material-profile', choices=('native', 'pbr-v1'), default='native')
    parser.add_argument('--room-plan', type=Path)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--bake-device', choices=('CPU', 'CUDA'), default='CPU')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.out.exists():
        raise RuntimeError('EXPORT_BACKEND_UNAVAILABLE(output already exists)')
    if args.material_profile == 'pbr-v1' and (
        any((args.out.parent/name).exists() for name in ('mapped.blend', 'baked.blend', 'material_manifest.json'))
        or any(args.out.parent.glob('surface_*.png'))
    ):
        raise RuntimeError('MATERIAL_BAKE_UNAVAILABLE(output already contains evidence)')
    meshes = [o for o in bpy.context.scene.objects if architectural_mesh(o)]
    excluded = [o.name for o in bpy.context.scene.objects if o.type == 'MESH' and o not in meshes]
    if not meshes:
        raise RuntimeError('SHELL_URI_EMPTY_GEOMETRY(no renderable mesh)')
    source = Path(bpy.data.filepath)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.hide_set(False)
        obj.select_set(True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    low, high = [float('inf')] * 3, [float('-inf')] * 3
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        for vertex in mesh.vertices:
            point = evaluated.matrix_world @ vertex.co
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
        evaluated.to_mesh_clear()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    material_manifest = None
    if args.material_profile == 'pbr-v1':
        if args.room_plan is None:
            raise RuntimeError('MATERIAL_INTENT_UNSUPPORTED(room plan absent)')
        root = Path(os.environ['RFW_INFINIGEN_ROOT'])
        import subprocess
        from rfw_w.scenes.solvers.shell_providers import INFINIGEN_PIN
        from rfw_w.scenes.workers.blender.infinigen_materials import bake_architecture
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != INFINIGEN_PIN:
            raise RuntimeError('MATERIAL_BAKE_UNAVAILABLE(source commit mismatch)')
        material_manifest = bake_architecture(meshes, json.loads(args.room_plan.read_text()),
                                             args.out.parent, args.resolution, args.bake_device)
        material_manifest.update(source_sha256=source_hash, source_commit=commit,
                                 room_plan_sha256=hashlib.sha256(args.room_plan.read_bytes()).hexdigest())
        bpy.ops.object.select_all(action='DESELECT')
        for obj in meshes:
            obj.select_set(True)
    bpy.ops.wm.usd_export(filepath=str(args.out.resolve()), selected_objects_only=True,
                          export_materials=True, export_textures=True,
                          relative_paths=True, root_prim_path='/Shell',
                          # Selection excludes objects, not the World: Blender otherwise adds
                          # an env_light with a texture format Isaac cannot render reliably.
                          convert_world_material=False, export_lights=False)
    if not args.out.is_file():
        raise RuntimeError('EXPORT_BACKEND_UNAVAILABLE(USD exporter produced no file)')
    measurement = json.loads(args.measurement.read_text())
    # The source ruler's actual count, never the request's expected count.
    count = measurement['measurements']['openings.count']
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('MATERIAL_BINDING_MISMATCH(source changed)')
    record = dict(source_file=os.path.relpath(source, args.out.parent),
                  measurement_file=os.path.relpath(args.measurement.resolve(), args.out.parent),
                  measurement_sha256=hashlib.sha256(args.measurement.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  usd_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                  openings_measured=int(count), source_mesh_count=len(meshes),
                  source_bounds_min=low, source_bounds_max=high,
                  exported_objects=[o.name for o in meshes], excluded_objects=excluded,
                  materials='Blender native USD conversion; procedural textures not baked',
                  lighting_policy='worker_only')
    if material_manifest is not None:
        manifest = args.out.parent / 'material_manifest.json'
        manifest.write_text(json.dumps(material_manifest, indent=2) + '\n')
        record.update(materials='pbr-v1; visual fidelity not yet accepted',
                      material_manifest=manifest.name,
                      material_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    args.out.with_suffix('.export.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
