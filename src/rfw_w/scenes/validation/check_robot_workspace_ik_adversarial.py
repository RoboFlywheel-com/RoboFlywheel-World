# SPDX-License-Identifier: Apache-2.0
"""Retain relocation and saved-candidate rejection evidence without editing a source run."""
import argparse,json,shutil
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_robot_workspace_ik import verify


def tree(root):
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--delivery',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    before=tree(args.delivery);results={}
    cases={'relocated':None,'changed_candidate':'ROBOT_IK_ARTIFACT_MISMATCH',
           'missing_joint':'ROBOT_IK_INPUT_INVALID','limit':'ROBOT_IK_LIMIT_VIOLATION',
           'orientation':'ROBOT_IK_INPUT_INVALID','mesh':'ROBOT_ARTIFACT_MISMATCH'}
    for name,expected in cases.items():
        root=args.output_dir/name;shutil.copytree(args.delivery,root)
        if name=='mesh':
            metadata=json.loads((root/'robot_runtime.json').read_text())
            ref=next(ref for ref in metadata['artifacts'] if ref!='robot.urdf')
            with (root/ref).open('ab') as stream:stream.write(b'changed')
        elif name!='relocated':
            config=json.loads((root/'ik_cases.json').read_text());case=config['reachable']
            if name=='missing_joint':case['candidate'].pop('joint1')
            elif name in ('limit','changed_candidate'):case['candidate']['joint1']=99
            elif name=='orientation':case['target']['orientation']=[0,0,0,0]
            (root/'ik_cases.json').write_text(json.dumps(config,indent=2)+'\n')
            if name!='changed_candidate':
                # Updating the local checksum exercises validation beyond accidental corruption.
                integrity=json.loads((root/'ik_integrity.json').read_text())
                integrity['ik_cases.json']=digest(root/'ik_cases.json')
                (root/'ik_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
        try:
            report=verify(root);failure=None
        except (ValueError,OSError,KeyError) as error:
            failure=str(error);report={'verified':False,'failure_code':failure}
        (root/'adversarial_report.json').write_text(json.dumps(report,indent=2)+'\n')
        results[name]={'passed':failure==expected,'expected':expected,'actual':failure}
    unchanged=before==tree(args.delivery)
    report={'verified':unchanged and all(r['passed'] for r in results.values()),
            'cases':results,'source_unchanged':unchanged,'source_artifacts':before}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'verified':report['verified'],'source_unchanged':unchanged,'cases':results}))
    return 0 if report['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
