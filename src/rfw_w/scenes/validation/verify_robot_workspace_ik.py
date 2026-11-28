# SPDX-License-Identifier: Apache-2.0
"""Independent saved IK candidate FK, limits, and actual mesh collision ruler."""
import argparse,json,math
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import audit,digest


def load(root,metadata):
    if 'robot.urdf' not in metadata['artifacts']:raise ValueError('ROBOT_REFERENCE_INVALID')
    audit(root,metadata)
    import pybullet as p
    client=p.connect(p.DIRECT)
    body=p.loadURDF(str(root/'robot.urdf'),useFixedBase=True,physicsClientId=client)
    info=[p.getJointInfo(body,j,physicsClientId=client) for j in range(p.getNumJoints(body,physicsClientId=client))]
    return p,client,body,info


def measure(p,client,body,info,case):
    names={i[1].decode():j for j,i in enumerate(info)}
    links={i[12].decode():j for j,i in enumerate(info)}
    movable={i[1].decode() for i in info if i[2]!=p.JOINT_FIXED}
    if set(case['candidate'])!=movable:raise ValueError('ROBOT_IK_INPUT_INVALID')
    target=case['target']
    for key,length in (('position',3),('orientation',4)):
        values=target[key]
        if len(values)!=length or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            raise ValueError('ROBOT_IK_INPUT_INVALID')
    if abs(sum(v*v for v in target['orientation'])-1)>1e-5:raise ValueError('ROBOT_IK_INPUT_INVALID')
    for obstacle in case.get('obstacles',[]):
        for key in ('position','half_extents'):
            values=obstacle[key]
            if len(values)!=3 or not all(math.isfinite(v) for v in values):raise ValueError('ROBOT_IK_INPUT_INVALID')
        if any(v<=0 for v in obstacle['half_extents']):raise ValueError('ROBOT_IK_INPUT_INVALID')
    if case['link'] not in links:raise ValueError('ROBOT_LINK_UNKNOWN')
    for name,value in case['candidate'].items():
        if name not in names or not math.isfinite(value):raise ValueError('ROBOT_IK_INPUT_INVALID')
        j=names[name]
        if not info[j][8]-1e-8<=value<=info[j][9]+1e-8:raise ValueError('ROBOT_IK_LIMIT_VIOLATION')
        p.resetJointState(body,j,value,physicsClientId=client)
    state=p.getLinkState(body,links[case['link']],computeForwardKinematics=True,physicsClientId=client)
    pos=list(state[4]);quat=list(state[5]);target=case['target']
    position_error=math.dist(pos,target['position'])
    dot=min(1.,abs(sum(a*b for a,b in zip(quat,target['orientation']))))
    orientation_error=2*math.acos(dot)
    obstacles=[]
    for o in case.get('obstacles',[]):
        shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=o['half_extents'],physicsClientId=client)
        obstacles.append(p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=o['position'],physicsClientId=client))
    p.performCollisionDetection(physicsClientId=client)
    contacts=[]
    for obstacle in obstacles:
        contacts.extend({'link':c[3],'distance_m':c[8]} for c in p.getClosestPoints(body,obstacle,distance=.002,physicsClientId=client) if c[8]<-.0001)
    # Fixed links share one rigid body; exclude that group and directly adjoining groups.
    group={-1:-1}
    for j,i in enumerate(info):group[j]=group[i[16]] if i[2]==p.JOINT_FIXED else j
    adjacent={frozenset((group[j],group[i[16]])) for j,i in enumerate(info)}
    excluded=[];self_contacts=[]
    for a in range(-1,len(info)):
        for b in range(a+1,len(info)):
            if group[a]==group[b] or frozenset((group[a],group[b])) in adjacent:
                excluded.append([a,b]);continue
            for c in p.getClosestPoints(body,body,distance=.002,linkIndexA=a,linkIndexB=b,physicsClientId=client):
                if c[8]<-.0001:self_contacts.append({'links':[a,b],'distance_m':c[8]})
    for o in obstacles:p.removeBody(o,physicsClientId=client)
    failure=('ROBOT_IK_UNREACHABLE' if position_error>.005 or orientation_error>.05 else
             'ROBOT_COLLISION' if contacts else 'ROBOT_SELF_COLLISION' if self_contacts else None)
    return {'status':'failed' if failure else 'passed','failure_code':failure,'position_error_m':position_error,
            'orientation_error_rad':orientation_error,'measured_position':pos,'measured_orientation':quat,
            'obstacle_penetrations':contacts,'self_penetrations':self_contacts,'excluded_link_pairs':excluded,
            'collision_model':'PyBullet URDF mesh convex approximations','motion':'not_run','grasp':'not_run','rollout':'not_run'}


def verify(root):
    integrity=json.loads((root/'ik_integrity.json').read_text())
    if set(integrity)!= {'robot_runtime.json','ik_cases.json'}:raise ValueError('ROBOT_IK_INPUT_INVALID')
    if any(digest(root/ref)!=expected for ref,expected in integrity.items()):raise ValueError('ROBOT_IK_ARTIFACT_MISMATCH')
    metadata=json.loads((root/'robot_runtime.json').read_text());cases=json.loads((root/'ik_cases.json').read_text())
    if set(cases)!= {'reachable','unreachable','blocked'}:raise ValueError('ROBOT_IK_INPUT_INVALID')
    p,c,b,info=load(root,metadata)
    try:
        measured={name:measure(p,c,b,info,case) for name,case in cases.items()}
        if measured['reachable']['status']!='passed':raise ValueError(measured['reachable']['failure_code'])
        if measured['unreachable']['failure_code']!='ROBOT_IK_UNREACHABLE':raise ValueError('ROBOT_IK_NEGATIVE_ACCEPTED')
        if measured['blocked']['failure_code']!='ROBOT_COLLISION':raise ValueError('ROBOT_COLLISION_NEGATIVE_ACCEPTED')
        return {'verified':True,'cases':measured,'robot':'Piper fixed base','task_rollout':'not_run'}
    finally:p.disconnect(c)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--delivery',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try:r=verify(a.delivery)
    except (ValueError,OSError,KeyError) as e:r={'verified':False,'failure_code':str(e)}
    (a.output_dir/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'verified':r['verified'],'failure_code':r.get('failure_code')}));return 0 if r['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
