# SPDX-License-Identifier: Apache-2.0
"""Solve Piper local IK cases; setting a candidate state is not motion execution."""
import argparse,json,shutil,time
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_workspace_ik import load,measure
from rfw_w.scenes.validation.verify_robot_runtime_model import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--robot-bundle',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();shutil.copytree(a.robot_bundle,a.output_dir);start=time.monotonic()
    metadata=json.loads((a.output_dir/'robot_runtime.json').read_text());p,c,b,info=load(a.output_dir,metadata)
    cases={};results={}
    try:
        names={i[1].decode():j for j,i in enumerate(info)};link=next(j for j,i in enumerate(info) if i[12].decode()=='gripper_base')
        movable=[j for j,i in enumerate(info) if i[2]!=p.JOINT_FIXED]
        lower=[info[j][8] for j in movable];upper=[info[j][9] for j in movable]
        reference=metadata['commands']
        for name,value in reference.items():p.resetJointState(b,names[name],value,physicsClientId=c)
        fk=p.getLinkState(b,link,computeForwardKinematics=True,physicsClientId=c)
        target={'position':list(fk[4]),'orientation':list(fk[5])}
        for name in ('reachable','unreachable','blocked'):
            goal=target if name!='unreachable' else {'position':[3,3,3],'orientation':target['orientation']}
            for j in movable:p.resetJointState(b,j,0,physicsClientId=c)
            solution=p.calculateInverseKinematics(b,link,goal['position'],goal['orientation'],lowerLimits=lower,upperLimits=upper,
                jointRanges=[hi-lo for lo,hi in zip(lower,upper)],restPoses=[reference[info[j][1].decode()] for j in movable],
                maxNumIterations=1000,residualThreshold=1e-6,physicsClientId=c)
            candidate={info[j][1].decode():solution[k] for k,j in enumerate(movable)}
            candidate['joint7']=.015;candidate['joint8']=-.015
            cases[name]={'link':'gripper_base','target':goal,'candidate':candidate,'solver':'PyBullet damped least squares',
                'target_source':'FK of declared reference posture' if name!='unreachable' else 'synthetic far target',
                'obstacles':[{'position':target['position'],'half_extents':[.1,.1,.1]}] if name=='blocked' else []}
            try:results[name]=measure(p,c,b,info,cases[name])
            except ValueError as e:results[name]={'status':'failed','failure_code':str(e)}
    finally:p.disconnect(c)
    (a.output_dir/'ik_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (a.output_dir/'ik_integrity.json').write_text(json.dumps({ref:digest(a.output_dir/ref) for ref in ('robot_runtime.json','ik_cases.json')},indent=2)+'\n')
    ok=(results['reachable']['status']=='passed' and results['unreachable']['failure_code']=='ROBOT_IK_UNREACHABLE'
        and results['blocked']['failure_code']=='ROBOT_COLLISION')
    report={'status':'passed' if ok else 'failed','cases':results,'elapsed_s':round(time.monotonic()-start,3),
            'navigation':'not_run','grasp':'not_run','motion':'not_run','task_rollout':'not_run'}
    (a.output_dir/'ik_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':{n:r.get('failure_code') for n,r in results.items()},'elapsed_s':report['elapsed_s']}));return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
