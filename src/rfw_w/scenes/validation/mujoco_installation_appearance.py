# SPDX-License-Identifier: Apache-2.0
"""Portable native MuJoCo visual layer preserving source OBJ faces, UV and materials."""
import json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest


def split_obj(source,prefix):
    definitions={};libraries=[];vertex=[];faces={};active=None
    for line in source.read_text().splitlines():
        tag,_,value=line.partition(' ')
        if tag=='mtllib':libraries.append(source.parent/value.strip())
        elif tag in ('v','vt','vn'):vertex.append(line)
        elif tag=='usemtl':active=value.strip()
        elif tag=='f':faces.setdefault(active,[]).append(line)
    for library in libraries:
        name=None
        for line in library.read_text().splitlines():
            tag,_,value=line.strip().partition(' ')
            if tag=='newmtl':name=value.strip();definitions[name]={'rgb':[1,1,1],'texture':None}
            elif name and tag=='Kd':definitions[name]['rgb']=list(map(float,value.split()[:3]))
            elif name and tag=='map_Kd':definitions[name]['texture']=library.parent/value.strip()
    parts=[]
    for index,(name,lines) in enumerate(faces.items()):
        if name is not None and name not in definitions:raise ValueError('APPEARANCE_MAPPING_MISSING')
        path=prefix.with_name(prefix.name+f'_{index}.obj');path.write_text('\n'.join(vertex+lines)+'\n')
        parts.append({'mesh':path,'faces':len(lines),'name':name,**definitions.get(name,{'rgb':[.7,.7,.7],'texture':None})})
    if not parts:raise ValueError('APPEARANCE_MAPPING_MISSING')
    return parts


def quad(path,vertices,uv):
    path.write_text('\n'.join(['v '+' '.join(map(str,v)) for v in vertices]+['vt '+' '.join(map(str,v)) for v in uv]+['f 1/1 2/2 3/3','f 1/1 3/3 4/4'])+'\n')


def build(source,out):
    shutil.copytree(source,out/'source');(out/'visual_meshes').mkdir();(out/'textures').mkdir()
    tree=ET.parse(source/'scene.xml');mjcf=tree.getroot();assets=mjcf.find('asset');world=mjcf.find('worldbody')
    for index,geom in enumerate(mjcf.findall('.//geom')):geom.set('name','original_'+str(index));geom.set('group','3')
    for mesh in assets.findall('mesh'):mesh.set('file','source/'+mesh.get('file'))
    bodies={b.get('name'):b for b in mjcf.findall('.//body')};ledger={'parts':[],'surfaces':[],'materials':[]}
    def material(name,rgb,channels,roughness=.82):
        mat=ET.SubElement(assets,'material',name=name,rgba=' '.join(map(str,[*rgb[:3],rgb[3] if len(rgb)==4 else 1])),roughness=str(roughness))
        refs={}
        for role,path in channels.items():
            if not path.is_file():raise ValueError('APPEARANCE_TEXTURE_MISSING')
            # Texture IDs use content hashes so same basename from different assets stays distinct.
            target=out/'textures'/(digest(path)+'.png')
            from PIL import Image
            Image.open(path).convert('RGB').save(target)
            ref=target.relative_to(out).as_posix();tex=name+'_'+role
            ET.SubElement(assets,'texture',name=tex,type='2d',file=ref)
            ET.SubElement(mat,'layer',role=role,texture=tex);refs[role]=ref
        ledger['materials'].append({'name':name,'channels':refs,'source_channels':{role:{'ref':path.relative_to(out/'source').as_posix(),'sha256':digest(path)} for role,path in channels.items()},'encoding':'lossless PNG decoded RGB pixels','roughness':roughness})
        return refs
    def add_parts(source_mesh,body,key,kind,default=None,origin=None):
        for index,part in enumerate(split_obj(source_mesh,out/'visual_meshes'/key)):
            name=key+'_'+str(index);channels={'rgb':part['texture']} if part['texture'] else {}
            refs=material('mat_'+name,default if default and part['name'] is None else part['rgb'],channels)
            mesh_ref=part['mesh'].relative_to(out).as_posix()
            ET.SubElement(assets,'mesh',name='mesh_'+name,file=mesh_ref,inertia='shell')
            attrs={'name':'visual_'+name,'type':'mesh','mesh':'mesh_'+name,'material':'mat_'+name,'group':'2','contype':'0','conaffinity':'0','mass':'0'}
            if origin:
                attrs.update(pos=origin.get('xyz','0 0 0'),euler=origin.get('rpy','0 0 0'))
            ET.SubElement(body,'geom',**attrs)
            ledger['parts'].append({'kind':kind,'instance_id':key,'mesh_ref':mesh_ref,'source_mesh':source_mesh.relative_to(out/'source').as_posix(),
                                    'source_sha256':digest(source_mesh),'faces':part['faces'],'source_material':part['name'],'texture_ref':refs.get('rgb')})
    config=json.loads((source/'installation.json').read_text());scene=out/'source/scene'
    for item in config['instances']:
        mesh=scene/item['visual_ref']
        if digest(mesh)!=item['visual_sha256']:raise ValueError('APPEARANCE_SOURCE_MISMATCH')
        add_parts(mesh,bodies['environment_'+item['instance_id']],item['instance_id'],'instance')
    urdf=ET.parse(out/'source/robot/robot.urdf')
    # MJCF stores compiled radian transforms; avoid default-degree interpretation of URDF rpy.
    mjcf.find('compiler').set('angle','radian')
    for link in urdf.findall('link'):
        body=bodies.get(link.get('name'))
        if body is None:continue
        for index,visual in enumerate(link.findall('visual')):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            origin=visual.find('origin');origin=origin.attrib if origin is not None else {}
            color=visual.find('material/color');default=list(map(float,color.get('rgba').split())) if color is not None else [.2,.3,.4,1]
            add_parts(out/'source/robot'/mesh.get('filename'),body,'robot_'+link.get('name')+'_'+str(index),'robot',default,origin)
    appearance=json.loads((scene/'surface_appearance.json').read_text())
    bindings={b['surface_id']:b for b in appearance['bindings']};material_names={}
    def declared(mid,value):
        if mid in material_names:return material_names[mid]
        name='surface_material_'+str(len(material_names));channels={}
        for key,channel in value['channels'].items():
            path=scene/channel['ref']
            if digest(path)!=channel['sha256']:raise ValueError('APPEARANCE_SOURCE_MISMATCH')
            channels[{'base_color':'rgb','normal_gl':'normal','roughness':'roughness'}[key]]=path
        material(name,value['base_color_rgba'],channels,value['roughness']);material_names[mid]=name;return name
    for sid,surface in appearance['surfaces'].items():
        if surface['room_id']!=config['room_id'] or sid not in bindings:continue
        binding=bindings[sid];mat=declared(binding['material_id'],appearance['materials'][binding['material_id']])
        axes=[{'x':0,'y':1,'z':2}[a] for a in surface['uv_axes']];axis={'x':0,'y':1,'z':2}[surface['axis']]
        lo0,lo1,hi0,hi1=surface['bounds_uv_m'];vertices=[];uv=[]
        for u,v in ((lo0,lo1),(hi0,lo1),(hi0,hi1),(lo0,hi1)):
            point=[0,0,0];point[axis]=surface['plane_m'];point[axes[0]]=u;point[axes[1]]=v;vertices.append(point)
            uv.append([(u-surface['uv_origin_m'][0])/binding['meters_per_tile'],(v-surface['uv_origin_m'][1])/binding['meters_per_tile']])
        key='surface_'+str(len(ledger['surfaces']));path=out/'visual_meshes'/(key+'.obj');quad(path,vertices,uv)
        ET.SubElement(assets,'mesh',name=key,file=path.relative_to(out).as_posix(),inertia='shell')
        ET.SubElement(world,'geom',name=key,type='mesh',mesh=key,material=mat,group='2',contype='0',conaffinity='0',mass='0')
        ledger['surfaces'].append({'surface_id':sid,'geom':key,'kind':surface['kind'],'material_id':binding['material_id'],'meters_per_tile':binding['meters_per_tile']})
    coverings=json.loads((scene/'surface_coverings.json').read_text())
    for item in coverings['instances']:
        if item['room_id']!=config['room_id']:continue
        mat=declared(item['material_id'],coverings['materials'][item['material_id']]);key=item['instance_id'];path=out/'visual_meshes'/(key+'.obj');quad(path,item['vertices_m'],[(0,0),(1,0),(1,1),(0,1)])
        ET.SubElement(assets,'mesh',name=key,file=path.relative_to(out).as_posix(),inertia='shell')
        ET.SubElement(world,'geom',name=key,type='mesh',mesh=key,material=mat,group='2',contype='0',conaffinity='0',mass='0')
        ledger['parts'].append({'kind':'covering','instance_id':key,'mesh_ref':path.relative_to(out).as_posix(),'faces':2,'texture_ref':None})
    tree.write(out/'appearance.xml',encoding='utf-8',xml_declaration=True)
    ledger['renderer_limits']={'native_material_layers':['rgb','normal','roughness'],
                               'classic_renderer':'RGB/UV and diffuse color supported; normal/roughness texture layers retained but classic shader effect not claimed',
                               'pbr_equivalence':'not_verified'}
    ledger['artifacts']={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'appearance_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n');return ledger
