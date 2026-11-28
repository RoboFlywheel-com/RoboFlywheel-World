#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove actual Blender material failures reject export before baking."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.out.exists():
        raise RuntimeError('output exists; use fresh evidence')
    root = Path(os.environ['RFW_INFINIGEN_ROOT'])
    from rfw_w.scenes.workers.blender.infinigen_materials import bake_architecture
    plan = {'rooms':[{'floor_material':'oak','wall_material':'off_white'}], 'provenance':{'seed':20}}
    rows = []
    for case, expected in [('emission','MATERIAL_CHANNEL_UNSUPPORTED'),
                           ('true_displacement','MATERIAL_CHANNEL_UNSUPPORTED'),
                           ('unbound','MATERIAL_BINDING_MISMATCH')]:
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.object
        mat = bpy.data.materials.new(case)
        mat.use_nodes = True
        obj.data.materials.append(mat)
        if case == 'emission':
            mat.node_tree.nodes.new('ShaderNodeEmission')
        elif case == 'true_displacement':
            node = mat.node_tree.nodes.new('ShaderNodeDisplacement')
            mat.node_tree.links.new(node.outputs[0],mat.node_tree.nodes.get('Material Output').inputs['Displacement'])
            mat.displacement_method = 'DISPLACEMENT'
        else:
            obj.data.materials.clear()
        observed = 'no failure'
        try:
            bake_architecture([obj],plan,args.out.parent/case,256,'CPU')
        except ValueError as error:
            observed = str(error)
        rows.append({'case':case,'expected':expected,'observed':observed,'passed':observed.startswith(expected+'(')})
        bpy.data.objects.remove(obj,do_unlink=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'cases':rows,'passed':all(r['passed'] for r in rows)},indent=2)+'\n')
    if not all(r['passed'] for r in rows):
        raise RuntimeError('MATERIAL_NEGATIVE_DID_NOT_FAIL')


if __name__=='__main__':main()
