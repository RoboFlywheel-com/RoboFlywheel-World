# SPDX-License-Identifier: Apache-2.0
"""Independent support footprint, immutable instance pose and MuJoCo contact ruler."""
import argparse,json,math,xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest


def require(condition,code):
    if not condition:raise ValueError(code)


def footprint(root,item):
    import trimesh
    mesh=trimesh.load(str(root/item['collision_ref']),force='mesh',process=False)
    lo,hi=mesh.bounds;theta=item.get('yaw_rad',0);c,s=math.cos(theta),math.sin(theta)
    corners=[(c*x-s*y+item['pose_xyz_m'][0],s*x+c*y+item['pose_xyz_m'][1]) for x in (lo[0],hi[0]) for y in (lo[1],hi[1])]
    return [min(x for x,y in corners),min(y for x,y in corners),max(x for x,y in corners),max(y for x,y in corners)],float(lo[2]+item['pose_xyz_m'][2])


def check_layout(root,config):
    import trimesh
    ledger=json.loads((root/'scene/core_asset_ledger.json').read_text())
    background=json.loads((root/'scene/background_ledger.json').read_text()) if (root/'scene/background_ledger.json').exists() else {'instances':[]}
    items=[i for i in ledger['instances']+background['instances'] if i['room_id']==config['room_id']]
    require(config['instances']==items,'INSTALLATION_INPUT_INVALID')
    support=next(i for i in items if i['role_id']==config['support_role_id']);surface=support['support_surface']
    polygon=surface['polygon_xy'];xs=sorted(set(p[0] for p in polygon));ys=sorted(set(p[1] for p in polygon))
    require(len(xs)==2 and len(ys)==2,'INSTALLATION_INPUT_INVALID')
    bounds=[xs[0],ys[0],xs[1],ys[1]];z=surface['plane_z_m']
    base=trimesh.load(str(root/'robot/meshes/base_link.stl'),force='mesh',process=False)
    proxy={'collision_ref':'robot/meshes/base_link.stl','pose_xyz_m':config['base_position_m'],'yaw_rad':config['base_yaw_rad']}
    fb,bottom=footprint(root,proxy)
    require(all(bounds[k]+.02<=fb[k] for k in (0,1)) and all(fb[k]<=bounds[k]-.02 for k in (2,3)),'INSTALLATION_OUTSIDE_SUPPORT')
    require(abs(bottom-z-.002)<2e-5,'INSTALLATION_OUTSIDE_SUPPORT')
    placed=[]
    for item in items:
        if item.get('support_role_id')!=config['support_role_id']:continue
        box,height=footprint(root/'scene',item)
        require(height>=z-.005 and height<=z+.01 and all(bounds[k]-.001<=box[k] for k in (0,1)) and all(box[k]<=bounds[k]+.001 for k in (2,3)),'OBJECT_SUPPORT_INVALID')
        separate=fb[2]+.02<=box[0] or box[2]+.02<=fb[0] or fb[3]+.02<=box[1] or box[3]+.02<=fb[1]
        require(separate,'INSTALLATION_OCCUPIED')
        placed.append(item['instance_id'])
    target=next(i for i in items if i['role_id']==config['target_role_id'])
    require(math.dist(config['base_position_m'],target['pose_xyz_m'])<=.60,'INSTALLATION_UNAVAILABLE')
    return {'base_footprint_xy_m':fb,'surface_bounds_xy_m':bounds,'surface_z_m':z,'objects_checked':placed,
            'target_distance_m':math.dist(config['base_position_m'],target['pose_xyz_m']),
            'target_distance_method':'screening_proxy_not_ik','physical_fastening':'not_run'}


def contacts(root,config):
    import mujoco,numpy as np
    model=mujoco.MjModel.from_xml_path(str(root/'scene.xml'));data=mujoco.MjData(model)
    base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
    require(np.allclose(model.body_pos[base],config['base_position_m'],atol=2e-6),'INSTALLATION_INPUT_INVALID')
    require(np.allclose(model.body_quat[base],[math.cos(config['base_yaw_rad']/2),0,0,math.sin(config['base_yaw_rad']/2)],atol=2e-6),'INSTALLATION_INPUT_INVALID')
    for item in config['instances']:
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'environment_'+item['instance_id'])
        require(body>=0 and np.allclose(model.body_pos[body],item['pose_xyz_m'],atol=2e-6),'INSTALLATION_INPUT_INVALID')
        require(np.allclose(model.body_quat[body],[math.cos(item.get('yaw_rad',0)/2),0,0,math.sin(item.get('yaw_rad',0)/2)],atol=2e-6),'INSTALLATION_INPUT_INVALID')
    for name,value in config['reference_joints'].items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        require(j>=0 and model.jnt_range[j,0]<=value<=model.jnt_range[j,1],'INSTALLATION_INPUT_INVALID')
        data.qpos[model.jnt_qposadr[j]]=value
    mujoco.mj_forward(model,data)
    penetrations=[]
    for contact in data.contact:
        if contact.dist<-.001:
            penetrations.append({'geom1':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,contact.geom1),
                                 'geom2':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,contact.geom2),'distance_m':float(contact.dist)})
    return {'penetrations':penetrations,'contacts_count':data.ncon,'backend':'mujoco','version':mujoco.__version__}


def verify(root):
    manifest=json.loads((root/'installation_artifacts.json').read_text())
    require({'scene.xml','installation.json'}<=set(manifest),'INSTALLATION_INPUT_INVALID')
    for ref,expected in manifest.items():
        path=(root/ref).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and digest(path)==expected,'INSTALLATION_ARTIFACT_MISMATCH')
    tree=ET.parse(root/'scene.xml')
    for mesh in tree.findall('.//mesh'):
        require(mesh.get('file') in manifest and not Path(mesh.get('file')).is_absolute(),'INSTALLATION_ARTIFACT_MISMATCH')
    config=json.loads((root/'installation.json').read_text());layout=check_layout(root,config);collision=contacts(root,config)
    return {'verified':not collision['penetrations'],'failure_code':'INSTALLATION_COLLISION' if collision['penetrations'] else None,
            'layout':layout,'collision':collision,'mount':'simulated_rigid_attachment','ik':'not_run','grasp':'not_run',
            'settling':'not_run','motion':'not_run','navigation':'unsupported_no_mobile_base','rollout':'not_run'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--delivery',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try:report=verify(a.delivery)
    except (ValueError,OSError,KeyError,StopIteration) as error:report={'verified':False,'failure_code':str(error)}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if report['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
