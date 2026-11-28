# SPDX-License-Identifier: Apache-2.0
"""Source URDF kinematics and static visual authoring for the dual-Piper paper case."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from rfw_w.scenes.visualization.paper_case_common import read,write,digest


def vector(node,key,default='0 0 0'):
    return [float(x) for x in (node.get(key,default) if node is not None else default).split()]


def kinematics(root):
    joints=[]
    for j in root.findall('joint'):
        limit=j.find('limit')
        joints.append({'name':j.get('name'),'type':j.get('type'),'parent':j.find('parent').get('link'),'child':j.find('child').get('link'),'xyz':vector(j.find('origin'),'xyz'),'rpy':vector(j.find('origin'),'rpy'),'axis':vector(j.find('axis'),'xyz','0 0 1'),'limits':[float(limit.get(k)) for k in ('lower','upper')] if limit is not None else None})
    return {'joints':joints,'tcp_link':'gripper_base','tcp_offset_m':[0,0,.138],'tcp_definition':'midpoint between nominal finger base frames at their source z=0.138 m; pre-operation point, not a calibrated grasp frame'}


def transform(xyz,rpy):
    import numpy as np
    from scipy.spatial.transform import Rotation
    m=np.eye(4);m[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();m[:3,3]=xyz;return m


def fk(model,q):
    import numpy as np
    from scipy.spatial.transform import Rotation
    frames={'world':np.eye(4)}
    for j in model['joints']:
        motion=np.eye(4);value=q.get(j['name'],0)
        if j['type']=='revolute':motion[:3,:3]=Rotation.from_rotvec(np.array(j['axis'])*value).as_matrix()
        elif j['type']=='prismatic':motion[:3,3]=np.array(j['axis'])*value
        frames[j['child']]=frames[j['parent']]@transform(j['xyz'],j['rpy'])@motion
    return frames


def author(row,out,source_root):
    import numpy as np
    import trimesh
    from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,Gf,Sdf
    root=ET.parse(source_root/'piper.urdf').getroot();model=kinematics(root);write(out/'kinematics.json',model)
    # Preserve joint/frame provenance, with no machine-specific mesh URI.
    palette=read(source_root/'palette.json');q={k:math.radians(v) for k,v in row['static_joint_degrees'].items()};q.update(joint7=.035,joint8=-.035);frames=fk(model,q)
    stage=Usd.Stage.CreateNew(str(out/'asset.usda'));UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1);stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim());proof=[]
    for link in root.findall('link'):
        name=link.get('name');x=UsdGeom.Xform.Define(stage,'/World/'+name);x.AddTransformOp().Set(Gf.Matrix4d(frames[name].T.tolist()))
        for i,visual in enumerate(link.findall('visual')):
            filename=visual.find('geometry/mesh').get('filename');source=source_root/'meshes'/filename;mesh=trimesh.load(str(source),force='mesh',process=False)
            m=UsdGeom.Mesh.Define(stage,f'/World/{name}/visual_{i}');m.CreatePointsAttr(mesh.vertices.tolist());m.CreateFaceVertexCountsAttr([3]*len(mesh.faces));m.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist());m.CreateSubdivisionSchemeAttr('none')
            origin=visual.find('origin');m.AddTransformOp().Set(Gf.Matrix4d(transform(vector(origin,'xyz'),vector(origin,'rpy')).T.tolist()))
            colour=palette[name][i][:3];mat=UsdShade.Material.Define(stage,f'/World/Materials/{name}_{i}');sh=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Shader');sh.CreateIdAttr('UsdPreviewSurface');sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour));sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.42);mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
            proof.append({'link':name,'mesh':filename,'sha256':digest(source),'colour':colour})
    joints=[]
    for j in model['joints']:
        path='/World/Joints/'+j['name']
        if j['type']=='revolute':joint=UsdPhysics.RevoluteJoint.Define(stage,path);joint.CreateAxisAttr('Z');joint.CreateLowerLimitAttr(math.degrees(j['limits'][0]));joint.CreateUpperLimitAttr(math.degrees(j['limits'][1]));joints.append(j)
        elif j['type']=='prismatic':joint=UsdPhysics.PrismaticJoint.Define(stage,path);joint.CreateAxisAttr('Z');sign=j['axis'][2];joint.CreateLowerLimitAttr(min(sign*x for x in j['limits']));joint.CreateUpperLimitAttr(max(sign*x for x in j['limits']))
        else:joint=UsdPhysics.FixedJoint.Define(stage,path)
        joint.CreateBody0Rel().SetTargets(['/World/'+j['parent']]);joint.CreateBody1Rel().SetTargets(['/World/'+j['child']]);joint.CreateLocalPos0Attr(Gf.Vec3f(*j['xyz']));rot=Gf.Matrix4d(transform([0,0,0],j['rpy']).T.tolist()).ExtractRotationQuat();joint.CreateLocalRot0Attr(Gf.Quatf(rot))
    stage.GetRootLayer().Save()
    provenance=read(source_root/'provenance.json')
    return stage,[],{'kind':'piper_source_urdf_static_visual','source_sha256':digest(source_root/'piper.urdf'),'manufacturer_repository':provenance['repository'],'manufacturer_commit':provenance['commit'],'joints':joints,'visual_sources':proof,'tcp_definition':model['tcp_definition'],'physics_simulation':'not_run'}
