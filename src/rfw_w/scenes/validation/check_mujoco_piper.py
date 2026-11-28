# SPDX-License-Identifier: Apache-2.0
"""Convert a saved Piper URDF into portable MuJoCo MJCF and run real actuators."""
import argparse,json,shutil,time,xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import audit,digest
from rfw_w.scenes.validation.verify_mujoco_piper import execute


def build(source,out):
    import mujoco
    metadata=json.loads((source/'robot_runtime.json').read_text());audit(source,metadata)
    shutil.copytree(source,out);tree=ET.parse(out/'robot.urdf');robot=tree.getroot()
    for mesh in robot.findall('.//mesh'):mesh.set('filename',str((out/mesh.get('filename')).resolve()))
    compiler=ET.SubElement(robot,'mujoco');ET.SubElement(compiler,'compiler',discardvisual='true',fusestatic='false',balanceinertia='true')
    model=mujoco.MjModel.from_xml_string(ET.tostring(robot,encoding='unicode'))
    mujoco.mj_saveLastXML(str(out/'robot.xml'),model)
    tree=ET.parse(out/'robot.xml');mjcf=tree.getroot()
    for mesh in mjcf.findall('.//mesh'):mesh.set('file','meshes/'+Path(mesh.get('file')).name)
    ET.SubElement(mjcf,'option',timestep='0.0005',gravity='0 0 -9.81',integrator='implicitfast')
    for joint in mjcf.findall('.//joint'):joint.set('damping','2' if joint.get('type')!='slide' else '5')
    actuator=ET.SubElement(mjcf,'actuator')
    urdf=ET.parse(out/'robot.urdf')
    for name in metadata['commands']:
        joint=next(j for j in urdf.findall('.//joint') if j.get('name')==name);limit=joint.find('limit')
        force=float(limit.get('effort'));slide=joint.get('type')=='prismatic'
        ET.SubElement(actuator,'position',name='control_'+name,joint=name,kp='2000' if slide else '1500',
                      kv='1' if slide else '10',ctrllimited='true',ctrlrange=limit.get('lower')+' '+limit.get('upper'),
                      forcelimited='true',forcerange=f'{-force} {force}')
    # URDF collision meshes overlap at mechanically connected links; exclude only
    # same rigid group and direct joint neighbours, retaining other self contacts.
    groups={'world':'world'};adjacent=set()
    joints=urdf.findall('.//joint')
    for joint in joints:
        parent=joint.find('parent').get('link');child=joint.find('child').get('link')
        groups[child]=groups[parent] if joint.get('type')=='fixed' else child
        adjacent.add(frozenset((groups[parent],groups[child])))
    contact=ET.SubElement(mjcf,'contact');bodies=[b.get('name') for b in mjcf.findall('.//body')]
    for k,a in enumerate(bodies):
        for b in bodies[k+1:]:
            if groups[a]==groups[b] or frozenset((groups[a],groups[b])) in adjacent:
                ET.SubElement(contact,'exclude',body1=a,body2=b)
    tree.write(out/'robot.xml',encoding='utf-8',xml_declaration=True)
    meta={'schema_version':'pvscene.mujoco-piper/v1','commands':metadata['commands'],'steps':16000,
          'conversion':{'discardvisual':True,'fusestatic':False,'balanceinertia':True,
                        'appearance':'URDF collision meshes','collision':'MuJoCo convex mesh approximation'},
          'artifacts':{ref:digest(out/ref) for ref in ['robot.xml','robot.urdf',*[p.relative_to(out).as_posix() for p in (out/'meshes').iterdir()]]}}
    (out/'mujoco_runtime.json').write_text(json.dumps(meta,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--robot-bundle',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();start=time.monotonic();build(a.robot_bundle,a.output_dir)
    try:report=execute(a.output_dir)
    except (ValueError,OSError,KeyError) as error:report={'status':'failed','failure_code':str(error)}
    report['elapsed_s']=round(time.monotonic()-start,3)
    (a.output_dir/'mujoco_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}));return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
