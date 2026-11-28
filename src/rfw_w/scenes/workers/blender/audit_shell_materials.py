#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit source architectural material bindings; run with Blender -b source.blend -P."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


from rfw_w.scenes.workers.blender.export_built_shell import architectural_mesh


def nodes_in(tree, seen=None):
    seen=set() if seen is None else seen
    if tree is None or tree.as_pointer() in seen:return []
    seen.add(tree.as_pointer())
    result=[]
    for node in tree.nodes:
        result.append(node.bl_idname)
        if node.type=='GROUP':result.extend(nodes_in(node.node_tree,seen))
    return result


def audit():
    import bpy
    rows=[]
    materials={}
    for obj in bpy.context.scene.objects:
        if not architectural_mesh(obj):continue
        bindings=[]
        for slot in obj.material_slots:
            mat=slot.material
            if mat is None:
                bindings.append(None);continue
            bindings.append(mat.name)
            if mat.name in materials:continue
            nodes=nodes_in(mat.node_tree)
            materials[mat.name]={'node_types':sorted(set(nodes)),'node_count':len(nodes),
                'native_diffuse_rgba':list(mat.diffuse_color),'use_nodes':mat.use_nodes,
                'procedural_texture_nodes':sorted({n for n in nodes if n.startswith('ShaderNodeTex') and n!='ShaderNodeTexImage'})}
        rows.append({'object':obj.name,'collections':[c.name for c in obj.users_collection],
                     'polygon_count':len(obj.data.polygons),'material_slots':bindings,
                     'used_material_indices':sorted({p.material_index for p in obj.data.polygons})})
    source=Path(bpy.data.filepath)
    return {'schema_version':'pvscene.shell-material-audit/v1',
            'source_file':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'objects':rows,'materials':materials,'baking_status':'not_run'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.out.exists():raise RuntimeError('output exists; use fresh evidence')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report=audit();args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'objects':len(report['objects']),'materials':len(report['materials']),'baking_status':'not_run'}))


if __name__=='__main__':main()
