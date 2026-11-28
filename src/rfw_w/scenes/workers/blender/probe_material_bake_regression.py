#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual Blender regressions for intrinsic colour, scalar data and window rejection."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.out.exists():
        raise RuntimeError('output exists; use fresh evidence')
    args.out.mkdir(parents=True)
    root = Path(os.environ['RFW_INFINIGEN_ROOT'])
    from rfw_w.scenes.workers.blender.infinigen_materials import bake_surface, convert_window_material
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add()
    obj = bpy.context.object
    mat = bpy.data.materials.new('probe_intrinsic')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (.8,.8,.8,1)
    bsdf.inputs['Roughness'].default_value = .25
    bsdf.inputs['Metallic'].default_value = .5
    bsdf.inputs['Alpha'].default_value = .1
    bsdf.inputs['Specular IOR Level'].default_value = 0
    obj.data.materials.append(mat)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1
    scene.cycles.device = 'CPU'
    bake_surface(obj,args.out,256,'probe')
    checks=[]
    for channel, expected in [('DIFFUSE',.8),('ROUGHNESS',.25),('METAL',.5)]:
        image=bpy.data.images.load(str(args.out/f'probe_{channel}.png'),check_existing=False)
        image.colorspace_settings.name='sRGB' if channel=='DIFFUSE' else 'Non-Color'
        observed=image.pixels[(128*256+128)*4]
        if channel=='DIFFUSE':
            observed=observed/12.92 if observed<=.04045 else ((observed+.055)/1.055)**2.4
        checks.append({'case':channel,'expected':expected,'observed':observed,
                       'passed':abs(observed-expected)<.015})
    alpha=obj.material_slots[0].material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value
    checks.append({'case':'coverage_restored','observed':alpha,'passed':abs(alpha-.1)<1e-6})
    try:
        convert_window_material(mat)
    except ValueError as error:
        checks.append({'case':'unknown_window_graph','observed':str(error),
                       'passed':str(error).startswith('MATERIAL_CHANNEL_UNSUPPORTED(')})
    else:
        checks.append({'case':'unknown_window_graph','passed':False})
    # A procedural scalar must be sampled, not replaced by the socket default.
    linked = bpy.data.materials.new('probe_linked')
    linked.use_nodes = True
    bsdf = linked.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value = .9
    value = linked.node_tree.nodes.new('ShaderNodeValue')
    value.outputs[0].default_value = .4
    linked.node_tree.links.new(value.outputs[0], bsdf.inputs['Roughness'])
    obj.data.materials.clear()
    obj.data.materials.append(linked)
    for layer in list(obj.data.uv_layers):
        obj.data.uv_layers.remove(layer)
    bake_surface(obj, args.out, 256, 'linked')
    image = bpy.data.images.load(str(args.out/'linked_ROUGHNESS.png'), check_existing=False)
    image.colorspace_settings.name = 'Non-Color'
    observed = image.pixels[(128*256+128)*4]
    checks.append({'case':'linked_roughness','expected':.4,'observed':observed,
                   'passed':abs(observed-.4)<.015})
    # The supported ceramic tile uses a Mix Shader; preserve its weighting.
    second = linked.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    second.inputs['Roughness'].default_value = .8
    mix = linked.node_tree.nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = .25
    linked.node_tree.links.new(bsdf.outputs[0], mix.inputs[1])
    linked.node_tree.links.new(second.outputs[0], mix.inputs[2])
    linked.node_tree.links.new(mix.outputs[0], linked.node_tree.nodes['Material Output'].inputs['Surface'])
    # bake_surface operates on a copy; use the original graph for this probe.
    obj.data.materials[0] = linked
    for link in list(bsdf.inputs['Roughness'].links):
        linked.node_tree.links.remove(link)
    bsdf.inputs['Roughness'].default_value = .4
    from rfw_w.scenes.workers.blender.infinigen_materials import bake_roughness
    bake_roughness(obj, args.out, 256, 'mixed')
    image = bpy.data.images.load(str(args.out/'mixed_ROUGHNESS.png'), check_existing=False)
    image.colorspace_settings.name = 'sRGB'
    encoded = image.pixels[(128*256+128)*4]
    observed = encoded/12.92 if encoded<=.04045 else ((encoded+.055)/1.055)**2.4
    checks.append({'case':'mixed_roughness','expected':.5,'observed':observed,
                   'passed':abs(observed-.5)<.015})
    report={'checks':checks,'passed':all(c['passed'] for c in checks)}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise RuntimeError('MATERIAL_BAKE_REGRESSION')


if __name__=='__main__':
    main()
