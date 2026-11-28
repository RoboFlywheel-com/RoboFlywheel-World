# SPDX-License-Identifier: Apache-2.0
"""Retain real relocation and negative cases for MuJoCo model and installation."""
import argparse,json,shutil
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_mujoco_piper import execute
from rfw_w.scenes.validation.verify_single_room_installation import verify


def tree(root):return {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--robot-bundle',type=Path,required=True);parser.add_argument('--installation',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);before={'robot':tree(a.robot_bundle),'scene':tree(a.installation)};results={}
    for name,expected in [('robot_relocated',None),('robot_limit','ROBOT_JOINT_COMMAND_INVALID'),('robot_mesh','ROBOT_ARTIFACT_MISMATCH'),
                          ('installation_relocated',None),('installation_outside','INSTALLATION_OUTSIDE_SUPPORT'),
                          ('installation_occupied','INSTALLATION_OCCUPIED'),('installation_mesh','INSTALLATION_ARTIFACT_MISMATCH')]:
        root=a.output_dir/name;robot=name.startswith('robot');shutil.copytree(a.robot_bundle if robot else a.installation,root)
        if name=='robot_limit':
            meta=json.loads((root/'mujoco_runtime.json').read_text());meta['commands']['joint1']=99;(root/'mujoco_runtime.json').write_text(json.dumps(meta))
        elif name=='robot_mesh':
            with next((root/'meshes').glob('*.stl')).open('ab') as stream:stream.write(b'changed')
        elif name=='installation_mesh':
            with next((root/'scene_meshes').glob('*.obj')).open('ab') as stream:stream.write(b'changed')
        elif name in ('installation_outside','installation_occupied'):
            config=json.loads((root/'installation.json').read_text())
            config['base_position_m'][:2]=[10,10] if name.endswith('outside') else next(i['pose_xyz_m'][:2] for i in config['instances'] if i['role_id']==config['target_role_id'])
            (root/'installation.json').write_text(json.dumps(config))
            manifest=json.loads((root/'installation_artifacts.json').read_text());manifest['installation.json']=digest(root/'installation.json');(root/'installation_artifacts.json').write_text(json.dumps(manifest))
        try:
            result=execute(root) if robot else verify(root)
            failure=result.get('failure_code');ok=result.get('status')=='passed' if robot else result['verified']
            if not ok and failure is None:failure='UNEXPECTED_FAILURE'
        except (ValueError,OSError,KeyError) as error:failure=str(error);result={'failure_code':failure}
        (root/'adversarial_report.json').write_text(json.dumps(result,indent=2)+'\n')
        results[name]={'passed':failure==expected,'actual':failure,'expected':expected}
    unchanged=before=={'robot':tree(a.robot_bundle),'scene':tree(a.installation)}
    report={'verified':unchanged and all(r['passed'] for r in results.values()),'cases':results,'source_unchanged':unchanged,'source_artifacts':before}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_artifacts'}));return 0 if report['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
