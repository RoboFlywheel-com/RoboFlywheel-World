# SPDX-License-Identifier: Apache-2.0
"""Simulation truth exporter. Requires independent passed validation; no hardware label."""
import json
from pathlib import Path
from rfw_w.scenes.assets.articulated_urdf import digest


def export_episode(world, run, output):
    world,run,output=Path(world),Path(run),Path(output)
    report=json.loads((run/'task_validation.json').read_text())
    if report['status']!='passed' or not all(report['checks'].values()):
        raise ValueError('REFERENCE_EXPORT_REQUIRES_PASSED_VALIDATION')
    raw=json.loads((run/'trajectory.json').read_text());rows=raw['samples']
    if digest(run/'trajectory.json')!=report['trajectory_sha256']:
        raise ValueError('REFERENCE_EXPORT_STALE_VALIDATION')
    import mujoco as mj
    import numpy as np
    output.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copyfile(run/'contacts.jsonl.gz',output/'contacts.jsonl.gz')
    shutil.copyfile(run/'controller.py',output/'controller.py')
    shutil.copyfile(run/'parameters.json',output/'parameters.json')
    model=mj.MjModel.from_xml_path(str(world/'world.xml'));data=mj.MjData(model)
    task=json.loads((world/'task_profile.json').read_text()) if (world/'task_profile.json').is_file() else None
    object_names=['mobile_base']+['V13_'+item['leaf'] for item in task['objects']] if task else ['mobile_base','cabinet_drawer_1','V13_11_2_0007','V13_9_6_0000']
    poses=[]
    for row in rows:
        data.qpos[:]=row['qpos'];mj.mj_forward(model,data)
        poses.append([np.r_[data.body(n).xpos,data.body(n).xquat] for n in object_names])
    np.savez_compressed(output/'state_action.npz',time_s=[r['t'] for r in rows],
        qpos=[r['qpos'] for r in rows],qvel=[r['qvel'] for r in rows],
        action_ctrl=[r['ctrl'] for r in rows],tcp_world_m=[r['tcp'] for r in rows],
        object_pose_world=np.array(poses))
    manifest=dict(schema_version='pvscene.simulation-reference-data/v1',label='simulation_reference',
        hardware_recording=False,source_world_sha256=digest(world/'world.xml'),
        source_trajectory_sha256=digest(run/'trajectory.json'),
        validation_sha256=digest(run/'task_validation.json'),data_sha256=digest(output/'state_action.npz'),
        fps=raw['fps'],physics_hz=raw['physics_hz'],samples=len(rows),
        units='m/kg/s/rad',pose_order='xyz,wxyz',up_axis='Z',
        object_names=object_names,joints=[model.joint(i).name or str(i) for i in range(model.njnt)],
        actuators=[model.actuator(i).name for i in range(model.nu)],
        observation_kind='simulation_ground_truth',action_kind='position_and_velocity_actuator_targets',
        contacts_ref='contacts.jsonl.gz',contacts_sha256=digest(output/'contacts.jsonl.gz'),sensors_status='exported_separately_by_media_worker')
    (output/'dataset.json').write_text(json.dumps(manifest,indent=2))
    return manifest
