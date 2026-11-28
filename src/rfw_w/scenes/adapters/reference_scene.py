# SPDX-License-Identifier: Apache-2.0
"""Task-protected workshop furnishing from source-bound local visual assets.

No online model or implicit asset lookup. Preserves source scale and preparation.
Peripheral collision uses one conservative box per asset; core retains source collision.
"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from rfw_w.scenes.assets.articulated_urdf import digest, values
from rfw_w.scenes.assets.prepared_obj import UP_FIX_ROWS, write_scene_obj
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.geometry.clearance import check_clearance


def dress(root, output, config):
    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw
    output=Path(output).resolve()
    asset=root.find('asset');world=root.find('worldbody')
    generated=output/'scene_resources';generated.mkdir()
    palette={'plaster':'.82 .81 .75 1','floor':'.44 .47 .45 1','steel':'.15 .20 .22 1',
             'wood':'.55 .38 .22 1','teal':'.23 .40 .40 1','daylight':'.75 .86 .88 1'}
    for name,rgba in palette.items():
        ET.SubElement(asset,'material',name='workshop_'+name,rgba=rgba,specular='.15',shininess='.1')
    # Deterministic procedural surface maps; explicitly synthetic, no AI textures.
    rng=np.random.default_rng(46)
    for name,color in [('floor',(151,158,151)),('plaster',(207,204,190)),('wood',(168,119,72))]:
        pixels=np.clip(np.array(color)[None,None,:]+rng.normal(0,2,(512,512,1)),0,255).astype('uint8')
        if name=='wood':
            grain=5*np.sin(np.arange(512)[None,:]*.3)+3*np.sin(np.arange(512)[:,None]*.05+np.arange(512)[None,:]*.8)
            pixels=np.clip(pixels.astype(float)+grain[:,:,None],0,255).astype('uint8')
        image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
        if name=='floor':
            for i in range(0,512,128):
                draw.line((i,0,i,511),fill=(112,122,116),width=2);draw.line((0,i,511,i),fill=(112,122,116),width=2)
        image.save(generated/(name+'.png'))
        ET.SubElement(asset,'texture',name='workshop_'+name+'_map',type='2d',file=str(generated/(name+'.png')))
        material=asset.find(f"material[@name='workshop_{name}']")
        material.set('texture','workshop_'+name+'_map');material.set('rgba','1 1 1 1')
        material.set('texrepeat','6 5' if name=='floor' else '2 2');material.set('texuniform','true')
    for name,mat in [('floor','floor'),('rear_wall','plaster'),('side_wall','plaster'),('tool_board','teal')]:
        geom=world.find(f"geom[@name='{name}']");geom.attrib.pop('rgba',None);geom.set('material','workshop_'+mat)
    for geom in list(world.findall('geom')):
        if geom.get('name','').startswith('shelf_'):world.remove(geom)
    for geom in root.findall('.//geom'):
        if geom.get('name','').startswith('cabinet_') and '_visual_' in geom.get('name',''):
            geom.set('material','workshop_wood');geom.set('rgba','1 1 1 1')
    def box(name,pos,size,material='steel'):
        ET.SubElement(world,'geom',name='dressing_'+name,type='box',pos=values(pos),size=values(size),
                      material='workshop_'+material,friction='.7 .005 .0001')
    box('rear_skirt',(-2.125,0,.07),(.012,2.5,.07))
    box('side_skirt',(0,2.425,.07),(3,.012,.07))
    # Pegboard holes are shallow visual details on its front, above the moving core.
    for iy in range(12):
        for iz in range(5):
            ET.SubElement(world,'geom',name=f'dressing_peg_{iy}_{iz}',type='sphere',
                          pos=values((-2.040,-.68+iy*.125,1.24+iz*.13)),size='.008',
                          material='workshop_steel',contype='0',conaffinity='0',mass='0')
    # Banded window/daylight panel and frame, mounted away from the manipulation core.
    box('window_panel',(-.15,2.428,1.75),(.95,.012,.58),'daylight')
    for x in (-1.12,-.15,.82):box('window_vertical_'+str(x),(x,2.403,1.75),(.024,.026,.62),'wood')
    for z in (1.15,2.35):box('window_horizontal_'+str(z),(-.15,2.403,z),(.995,.026,.024),'wood')
    # Source rack attachments mounted on the upper pegboard, no task target substituted.
    for y in (-.48,-.24,0,.24,.48):
        box('upper_hook_'+str(y),(-1.99,y,1.95),(.065,.012,.012),'steel')
    protection=Polygon.rectangle(-1.68,-.90,.75,1.02)
    room=Polygon.rectangle(-2.14,-2.5,3,2.44)
    occupied=[];records=[]
    for index,item in enumerate(config.get('assets',[])):
        source=Path(item['file'])
        if not source.is_file():raise ValueError('REFERENCE_DRESSING_SOURCE_MISSING')
        scene=trimesh.load(source,force='scene',process=False)
        scene.apply_transform(np.array(UP_FIX_ROWS))
        lo,hi=scene.bounds;center=np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]])
        transform=np.eye(4);transform[:3,3]=-center;scene.apply_transform(transform)
        size=hi-lo;yaw=float(item.get('yaw_rad',0));pos=list(item['position_m'])
        c,s=abs(math.cos(yaw)),abs(math.sin(yaw));extent=[c*size[0]+s*size[1],s*size[0]+c*size[1]]
        footprint=Polygon.rectangle(pos[0]-extent[0]/2,pos[1]-extent[1]/2,pos[0]+extent[0]/2,pos[1]+extent[1]/2)
        clearance=check_clearance(footprint,region=room,obstacles=[protection]+occupied,margin_m=.04)
        if not clearance.ok:raise ValueError('REFERENCE_DRESSING_CORE_CLEARANCE')
        occupied.append(footprint)
        name='background_'+str(index);body=ET.SubElement(world,'body',name=name,pos=values(pos),euler=values([0,0,yaw]))
        parts=[]
        for part_index,node in enumerate(scene.graph.nodes_geometry):
            transform,key=scene.graph[node];mesh=scene.geometry[key].copy();mesh.apply_transform(transform)
            one=trimesh.Scene(mesh)
            part_dir=generated/(name+'_'+str(part_index));part_dir.mkdir()
            prefix=part_dir/(name+'_'+str(part_index));write_scene_obj(one,prefix.with_suffix('.obj'))
            mtl=prefix.with_suffix('.mtl').read_text();rgb=[.7,.7,.7];texture=None
            for line in mtl.splitlines():
                if line.startswith('Kd '):rgb=list(map(float,line.split()[1:4]))
                if line.startswith('map_Kd '):texture=part_dir/line[7:]
            material='mat_'+prefix.name;attrs=dict(name=material,rgba=values([*rgb,1]),specular='.15',shininess='.2')
            if texture:
                ET.SubElement(asset,'texture',name='tex_'+prefix.name,type='2d',file=str(texture))
                attrs['texture']='tex_'+prefix.name
            ET.SubElement(asset,'material',**attrs)
            meshname='mesh_'+prefix.name
            ET.SubElement(asset,'mesh',name=meshname,file=str(prefix.with_suffix('.obj')),inertia='shell')
            ET.SubElement(body,'geom',name=prefix.name+'_visual',type='mesh',mesh=meshname,material=material,
                          group='2',contype='0',conaffinity='0',mass='0')
            parts.append({'visual_mesh':meshname,'source_geometry':str(key),'texture':bool(texture)})
        ET.SubElement(body,'geom',name=name+'_collision',type='box',pos=values([0,0,size[2]/2]),
                      size=values(size/2),group='3',rgba='0 0 0 0',friction='.7 .005 .0001')
        records.append(dict(id=item['id'],role=item['role'],source_sha256=digest(source),source_name=source.name,
            license=item['license'],source_scale=1,dimensions_m=size.tolist(),position_m=pos,yaw_rad=yaw,
            coordinate_preparation='GLB Y-up to Z-up + bottom center; pvscene prepared_obj.UP_FIX_ROWS',
            parts=parts,footprint=footprint.to_list(),clearance_m=clearance.measured_margin_m,
            collision='conservative measured AABB; static peripheral furniture, not manipulable'))
    # Soft directional daylight and warm work light, all using the native renderer.
    for light in world.findall('light'):world.remove(light)
    ET.SubElement(world,'light',name='daylight',pos='0 1.5 3',dir='-.5 -.3 -1',diffuse='.85 .87 .9',ambient='.1 .1 .12',castshadow='true')
    ET.SubElement(world,'light',name='task_fill',pos='-.1 -1.8 2.7',dir='-.6 .3 -1',diffuse='.65 .59 .48',castshadow='true')
    profile=dict(cameras={'overview':dict(position=[1.25,-3.0,2.3],target=[-.95,.35,1.0],fovy=46),
                         'detail':dict(position=[-.35,-1.35,2.25],target=[-1.02,0,.88],fovy=44)},
                 headlight_ambient=.3,headlight_diffuse=.45)
    (output/'render_profile.json').write_text(json.dumps(profile,indent=2))
    report=dict(schema_version='pvscene.reference-dressing/v1',profile='workshop_local_assets_v1',
        assets=records,protected_region_xy=protection.to_list(),room_region_xy=room.to_list(),
        authored=['wall/floor maps','pegboard details','window/light fixtures'],online_model_requests=0,
        claim='task-protected local furniture composition; not online autonomous scene generation',
        collision_scope='source core retained; conservative AABB peripheral collision')
    (output/'scene_dressing.json').write_text(json.dumps(report,indent=2))
    return report
