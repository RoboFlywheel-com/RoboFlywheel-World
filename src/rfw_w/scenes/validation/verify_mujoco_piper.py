# SPDX-License-Identifier: Apache-2.0
"""Independent MuJoCo control replay from portable, hashed robot artifacts."""
import argparse,json,math,xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest


def audit(root):
    meta=json.loads((root/'mujoco_runtime.json').read_text())
    if 'robot.xml' not in meta['artifacts']:raise ValueError('ROBOT_REFERENCE_INVALID')
    for ref,expected in meta['artifacts'].items():
        path=(root/ref).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('ROBOT_REFERENCE_INVALID')
        if digest(path)!=expected:raise ValueError('ROBOT_ARTIFACT_MISMATCH')
    tree=ET.parse(root/'robot.xml')
    for mesh in tree.findall('.//mesh'):
        ref=mesh.get('file')
        if not ref or Path(ref).is_absolute() or ref not in meta['artifacts']:raise ValueError('ROBOT_REFERENCE_INVALID')
    return meta


def execute(root):
    import mujoco,numpy as np
    meta=audit(root);model=mujoco.MjModel.from_xml_path(str(root/'robot.xml'));data=mujoco.MjData(model)
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(model.njnt)]
    if set(names)!=set(meta['commands']) or model.nq!=8 or model.nu!=8:raise ValueError('MUJOCO_MODEL_INVALID')
    for name,value in meta['commands'].items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        actuator=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'control_'+name)
        if actuator<0 or model.actuator_trnid[actuator,0]!=j:raise ValueError('MUJOCO_MODEL_INVALID')
        if not math.isfinite(value) or not model.jnt_range[j,0]<=value<=model.jnt_range[j,1]:raise ValueError('ROBOT_JOINT_COMMAND_INVALID')
        data.ctrl[actuator]=value
    samples=[]
    for step in range(meta['steps']):
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise ValueError('MUJOCO_MODEL_INVALID')
        if step%24==0 or step==meta['steps']-1:
            samples.append({'step':step+1,'joints':{name:float(data.qpos[model.jnt_qposadr[j]]) for j,name in enumerate(names)}})
    errors={name:abs(samples[-1]['joints'][name]-v) for name,v in meta['commands'].items()}
    converged=all(e<=(.003 if name in ('joint7','joint8') else .03) for name,e in errors.items())
    return {'status':'passed' if converged else 'failed','failure_code':None if converged else 'ROBOT_CONTROL_NOT_CONVERGED','backend':'mujoco','version':mujoco.__version__,'joint_errors':errors,'samples':samples,
            'steps':meta['steps'],'simulated_s':float(data.time),'mesh_count':model.nmesh,'base':'fixed',
            'navigation':'unsupported_no_mobile_base','grasp':'not_run','ik':'not_run','rollout':'not_run'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--delivery',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        execution=execute(a.delivery);report={'verified':execution['status']=='passed','execution':execution}
    except (ValueError,OSError,KeyError) as error:report={'verified':False,'failure_code':str(error)}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'verified':report['verified'],'failure_code':report.get('failure_code')}));return 0 if report['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
