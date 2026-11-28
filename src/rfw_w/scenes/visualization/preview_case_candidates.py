#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local Blender contact sheet for manual source selection, never a pipeline render."""
import os
import sys
from pathlib import Path
from rfw_w.scenes.visualization.paper_case_common import read,write,fresh,digest

def main(output):
    import bpy
    from mathutils import Vector
    from PIL import Image,ImageDraw
    out=fresh(output)
    catalog=read(os.environ['RFW_SOURCE_CATALOG'])['assets'] if os.environ.get('RFW_SOURCE_CATALOG') else []
    selected=[]
    for label,count in [('Kitchen Sponge',4),('Cleaning Sponge',3),('Plush Toy',4),('Memory Foam Pillow',3)]:
        selected.extend([x for x in catalog if x['semantic_class']==label][:count])
    if '--rigid-only' in sys.argv:selected=[]
    index=read(os.environ['PVSCENE_SIMREADY_INDEX']);root=Path(os.environ['PVSCENE_SIMREADY_ASSETS_ROOT'])
    for label,count in [('Mug',1),('Soup Bowl',3),('Dinner Plate',3),('Dish Soap Bottle',3)]:
        rows=[(k,v) for k,v in index.items() if label.lower()==v.get('super_category','').lower()]
        for key,row in rows[:count]:selected.append({'asset_id':key,'semantic_class':label,'visual_uri':str(root/row['assets_prefix']/'mesh'/f'{key}.obj'),'dimensions_m':row.get('dimensions_m')})
    records=[]
    for row in selected:
        source=Path(row['visual_uri']);id=row['asset_id']
        if not source.is_file():continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if source.suffix=='.glb':bpy.ops.import_scene.gltf(filepath=str(source))
        else:bpy.ops.wm.obj_import(filepath=str(source),forward_axis='NEGATIVE_Z',up_axis='Y')
        meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
        pts=[o.matrix_world@Vector(c) for o in meshes for c in o.bound_box]
        lo=Vector([min(p[k] for p in pts) for k in range(3)]);hi=Vector([max(p[k] for p in pts) for k in range(3)]);centre=(lo+hi)/2;size=(hi-lo).length
        scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=12;scene.cycles.device='CPU'
        scene.render.resolution_x=300;scene.render.resolution_y=280;scene.render.resolution_percentage=100
        scene.world=bpy.data.worlds.new('studio');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.65,.65,.65,1)
        bpy.ops.object.camera_add(location=centre+Vector((1.5,-2,1.4))*size);cam=bpy.context.object;cam.rotation_euler=(centre-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=size*1.15;scene.camera=cam
        bpy.ops.object.light_add(type='AREA',location=centre+Vector((1,-1,2))*size);light=bpy.context.object;light.data.energy=150*size*size;light.data.shape='DISK';light.data.size=size*2;light.rotation_euler=(centre-light.location).to_track_quat('-Z','Y').to_euler()
        scene.render.film_transparent=False;scene.render.filepath=str(out/(id+'.png'));bpy.ops.render.render(write_still=True)
        records.append({'asset_id':id,'semantic_class':row['semantic_class'],'dimensions_m':row['dimensions_m'],'source_sha256':digest(source),'preview':id+'.png'})
    write(out/'candidates.json',{'renderer':'Blender Cycles CPU; candidate selection only','candidates':records})
    board=Image.new('RGB',(1200,330*((len(records)+3)//4)),(240,240,240));d=ImageDraw.Draw(board)
    for i,r in enumerate(records):
        x=(i%4)*300;y=(i//4)*330;board.paste(Image.open(out/r['preview']).convert('RGB'),(x,y));d.text((x+5,y+283),r['semantic_class'],fill='black');d.text((x+5,y+302),r['asset_id'].replace('stage3_100k_qwen2_v1_',''),fill='black')
    board.save(out/'contact_sheet.jpg')
if __name__=='__main__':main(sys.argv[1])
