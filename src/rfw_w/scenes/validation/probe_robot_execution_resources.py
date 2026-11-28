# SPDX-License-Identifier: Apache-2.0
"""Read-only robot execution prerequisites. Prints no root paths or credential values."""
import argparse,importlib.util,json,os,time,xml.etree.ElementTree as ET
from pathlib import Path

ROBOT_ROOTS=('PVSCENE_PAPER_PIPER_ROOT','PVSCENE_PAPER_G1_ROOT','PVSCENE_ROBOT_ROOT','PVSCENE_ROBOT_CONTROLLER_ROOT')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--env-file',type=Path,default=Path('.env.local'))
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    # Only named non-secret resource keys are loaded; credential lines are not parsed.
    if a.env_file.is_file():
        for line in a.env_file.read_text().splitlines():
            key,sep,value=line.partition('=')
            if sep and key.strip() in ROBOT_ROOTS:os.environ.setdefault(key.strip(),value.strip().strip('\"\''))
    resources={}
    for key in ROBOT_ROOTS:
        root=Path(os.environ[key]) if os.environ.get(key) else None
        record={'configured':root is not None,'exists':bool(root and root.is_dir()),'models':[],
                'controller_files':0,'controller_execution':'not_run'}
        if root and root.is_dir():
            record['controller_files']=sum(1 for _ in root.rglob('*controller*.py'))
            for file in sorted(root.rglob('*.urdf')):
                try:
                    tree=ET.parse(file).getroot();joints=tree.findall('joint')
                    record['models'].append({'model_name':tree.get('name'),'urdf_name':file.name,
                        'links':len(tree.findall('link')),'joints':len(joints),
                        'movable_joints':sum(j.get('type')!='fixed' for j in joints),
                        'wheel_joints':sum('wheel' in (j.get('name') or '').lower() for j in joints),
                        'gripper_joints':sum(any(s in (j.get('name') or '').lower() for s in ('gripper','finger')) for j in joints)})
                except (ET.ParseError,OSError):record['models'].append({'urdf_name':file.name,'status':'ROBOT_MODEL_INVALID'})
        resources[key]=record
    modules={name:importlib.util.find_spec(name) is not None for name in ('mujoco','pybullet','pinocchio','ikpy','torch','bpy')}
    report={'resources':resources,'runtime_modules_discoverable':modules,'isaac_launcher_exists':Path('/isaac-sim/python.sh').is_file(),
            'robot_load':'not_run','navigation':'not_run','grasp':'not_run','rollout':'not_run','model_requests_used':0,
            'elapsed_s':round(time.monotonic()-start,3)}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'configured_roots':sum(r['configured'] for r in resources.values()),'urdf_models':sum(len(r['models']) for r in resources.values()),
                      'controller_files':sum(r['controller_files'] for r in resources.values()),'modules':modules,'elapsed_s':report['elapsed_s']}))
if __name__=='__main__':main()
