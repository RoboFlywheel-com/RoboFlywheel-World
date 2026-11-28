# SPDX-License-Identifier: Apache-2.0
"""Independent portable robot references and actual PyBullet motor-control replay."""
import argparse,hashlib,json,math,sys,xml.etree.ElementTree as ET
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root,metadata):
    for ref,expected in metadata['artifacts'].items():
        path=(root/ref).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('ROBOT_REFERENCE_INVALID')
        if digest(path)!=expected:raise ValueError('ROBOT_ARTIFACT_MISMATCH')
    tree=ET.parse(root/'robot.urdf').getroot()
    for mesh in tree.findall('.//mesh'):
        ref=mesh.get('filename');path=(root/ref).resolve()
        if Path(ref).is_absolute() or not path.is_relative_to(root.resolve()) or ref not in metadata['artifacts']:
            raise ValueError('ROBOT_REFERENCE_INVALID')
    limits={j.get('name'):(j.get('type'),j.find('limit')) for j in tree.findall('joint') if j.get('type')!='fixed'}
    for name,value in metadata['commands'].items():
        if name not in limits or not math.isfinite(value):raise ValueError('ROBOT_JOINT_COMMAND_INVALID')
        kind,limit=limits[name]
        if limit is None or not float(limit.get('lower'))<=value<=float(limit.get('upper')):
            raise ValueError('ROBOT_JOINT_COMMAND_INVALID')
    return tree


def execute(root,metadata):
    audit(root,metadata)
    import pybullet as p
    client=p.connect(p.DIRECT)
    try:
        p.setTimeStep(metadata['timestep_s'],physicsClientId=client)
        p.setGravity(0,0,-9.81,physicsClientId=client)
        body=p.loadURDF(str(root/'robot.urdf'),useFixedBase=True,physicsClientId=client)
        names={p.getJointInfo(body,j,physicsClientId=client)[1].decode():j for j in range(p.getNumJoints(body,physicsClientId=client))}
        for name,target in metadata['commands'].items():
            index=names[name];info=p.getJointInfo(body,index,physicsClientId=client)
            if not info[8]<=target<=info[9]:raise ValueError('ROBOT_JOINT_COMMAND_INVALID')
            p.setJointMotorControl2(body,index,p.POSITION_CONTROL,targetPosition=target,force=info[10],
                                   positionGain=.1,velocityGain=1,physicsClientId=client)
        samples=[]
        for step in range(metadata['steps']):
            p.stepSimulation(physicsClientId=client)
            if step%24==0 or step==metadata['steps']-1:
                samples.append({'step':step+1,'joints':{n:p.getJointState(body,names[n],physicsClientId=client)[0] for n in metadata['commands']}})
        final=samples[-1]['joints'];errors={n:abs(final[n]-v) for n,v in metadata['commands'].items()}
        limits={j.get('name'):j.get('type') for j in audit(root,metadata).findall('joint')}
        if any(err>(.003 if limits[n]=='prismatic' else .03) for n,err in errors.items()):raise ValueError('ROBOT_CONTROL_NOT_CONVERGED')
        return {'status':'passed','joint_errors':errors,'samples':samples,'steps':metadata['steps'],
                'simulated_s':metadata['steps']*metadata['timestep_s'],'backend':'pybullet DIRECT',
                'base':'fixed','navigation':'not_run','grasp':'not_run','ik':'not_run','task_rollout':'not_run'}
    finally:p.disconnect(client)


def main():
    p=argparse.ArgumentParser();p.add_argument('--delivery',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        metadata=json.loads((a.delivery/'robot_runtime.json').read_text());r={'verified':True,'execution':execute(a.delivery,metadata)}
    except (ValueError,OSError,KeyError) as e:r={'verified':False,'failure_code':str(e)}
    (a.output_dir/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='execution'}));return 0 if r['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
