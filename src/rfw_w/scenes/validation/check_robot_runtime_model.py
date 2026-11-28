# SPDX-License-Identifier: Apache-2.0
"""Prepare a portable Piper URDF and run actual joint motors; no robot task claim."""
import argparse,json,os,shutil,time,xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest,execute,audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--robot',choices=['piper'],required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    for line in Path('.env.local').read_text().splitlines():
        key,sep,value=line.partition('=')
        if sep and key.strip()=='PVSCENE_PAPER_PIPER_ROOT':os.environ.setdefault(key.strip(),value.strip().strip('\"\''))
    if not os.environ.get('PVSCENE_PAPER_PIPER_ROOT'):raise ValueError('ROBOT_SOURCE_UNSET')
    source=Path(os.environ['PVSCENE_PAPER_PIPER_ROOT']);tree=ET.parse(source/'piper.urdf');root=tree.getroot()
    meshes=a.output_dir/'meshes';meshes.mkdir()
    for mesh in root.findall('.//mesh'):
        original=mesh.get('filename');candidate=source/'meshes'/Path(original).name
        if not candidate.is_file():raise ValueError('ROBOT_MESH_MISSING')
        destination=meshes/candidate.name
        if not destination.exists():shutil.copy2(candidate,destination)
        mesh.set('filename','meshes/'+candidate.name)
    tree.write(a.output_dir/'robot.urdf',encoding='utf-8',xml_declaration=True)
    commands={'joint1':.1,'joint2':.3,'joint3':-.3,'joint4':.1,'joint5':.1,'joint6':.1,'joint7':.015,'joint8':-.015}
    metadata={'schema_version':'pvscene.robot-runtime-model/v1','robot':'piper','source_env':'PVSCENE_PAPER_PIPER_ROOT',
              'source_urdf_sha256':digest(source/'piper.urdf'),'commands':commands,'steps':1200,'timestep_s':1/240,
              'artifacts':{p.relative_to(a.output_dir).as_posix():digest(p) for p in a.output_dir.rglob('*') if p.is_file()},
              'base':'fixed','mobile_task':'unsupported'}
    (a.output_dir/'robot_runtime.json').write_text(json.dumps(metadata,indent=2)+'\n')
    try:r=execute(a.output_dir,metadata);r['elapsed_s']=round(time.monotonic()-start,3)
    except Exception as e:r={'status':'failed','failure_code':str(e),'elapsed_s':round(time.monotonic()-start,3)}
    (a.output_dir/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='samples'}))
    return 0 if r['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
