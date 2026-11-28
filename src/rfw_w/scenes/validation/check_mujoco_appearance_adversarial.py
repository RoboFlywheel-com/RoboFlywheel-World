# SPDX-License-Identifier: Apache-2.0
"""Actual appearance relocation, texture corruption, and physical mutation rejection."""
import argparse,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_mujoco_installation_appearance import verify


def tree(root):return {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--delivery',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);before=tree(a.delivery);cases={}
    for name,expected in [('relocated',None),('texture','APPEARANCE_ARTIFACT_MISMATCH'),('physics','APPEARANCE_PHYSICS_CHANGED')]:
        root=a.output_dir/name;shutil.copytree(a.delivery,root)
        if name=='texture':
            with next((root/'textures').glob('*.png')).open('ab') as stream:stream.write(b'changed')
        elif name=='physics':
            xml=ET.parse(root/'appearance.xml');geom=next(g for g in xml.findall('.//geom') if g.get('name')=='original_0');geom.set('friction','0.01 0.01 0.01');xml.write(root/'appearance.xml',encoding='utf-8')
            ledger=json.loads((root/'appearance_ledger.json').read_text());ledger['artifacts']['appearance.xml']=digest(root/'appearance.xml');(root/'appearance_ledger.json').write_text(json.dumps(ledger))
        try:report=verify(root);failure=None
        except (ValueError,OSError,KeyError) as e:failure=str(e);report={'verified':False,'failure_code':failure}
        (root/'adversarial_report.json').write_text(json.dumps(report,indent=2)+'\n');cases[name]={'passed':failure==expected,'actual':failure,'expected':expected}
    unchanged=before==tree(a.delivery);report={'verified':unchanged and all(r['passed'] for r in cases.values()),'source_unchanged':unchanged,'cases':cases,'source_artifacts':before}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_artifacts'}));return 0 if report['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
