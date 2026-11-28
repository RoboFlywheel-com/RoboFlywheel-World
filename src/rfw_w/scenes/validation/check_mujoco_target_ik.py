# SPDX-License-Identifier: Apache-2.0
"""Bounded real-cup pregrasp IK; endpoint state only, preserving source appearance."""
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_mujoco_target_ik import geometry, orientation, measure


def render(root, case, rules):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    from PIL import Image
    model = mujoco.MjModel.from_xml_path(str(root / 'source/appearance.xml'))
    data = mujoco.MjData(model)
    for n, v in case['candidate'].items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        data.qpos[model.jnt_qposadr[j]] = v
    model.vis.headlight.ambient[:] = [.4, .4, .4]
    model.vis.headlight.diffuse[:] = [.6, .6, .6]
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    ledger = json.loads((root / 'source/appearance_ledger.json').read_text())
    hidden = []
    for surface in ledger['surfaces']:
        if surface['surface_id'].startswith(('wall_left:', 'wall_bottom_')):
            model.geom_group[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, surface['geom'])] = 4
            hidden.append(surface['surface_id'])
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [rules['position_m'][0] - .15, rules['position_m'][1] - .1, rules['position_m'][2]]
    camera.distance = 1.7
    camera.azimuth = 120
    camera.elevation = -30
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0
        Image.fromarray(renderer.render()).save(root / 'endpoint.png')
    return {'image_sha256': digest(root / 'endpoint.png'), 'pose': 'IK endpoint, not executed motion',
            'hidden_render_only': hidden, 'camera': {'lookat': camera.lookat.tolist(), 'distance': 1.7,
                                                    'azimuth': 120, 'elevation': -30}}


def main():
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from rfw_w.scenes.validation.verify_mujoco_installation_appearance import verify as verify_appearance
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assert verify_appearance(a.delivery)['verified']
    shutil.copytree(a.delivery, a.output_dir / 'source')
    rules = geometry(a.output_dir)
    model = mujoco.MjModel.from_xml_path(str(a.output_dir / 'source/appearance.xml'))
    data = mujoco.MjData(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rules['tcp_body'])
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint' + str(k)) for k in range(1, 7)]
    addresses = [model.jnt_qposadr[j] for j in joint_ids]
    lower, upper = model.jnt_range[joint_ids].T
    config = json.loads((a.output_dir / 'source/source/installation.json').read_text())
    reference = np.array([config['reference_joints']['joint' + str(k)] for k in range(1, 7)])
    rng = np.random.default_rng(38)
    seeds = [reference, (lower + upper) / 2, *rng.uniform(lower, upper, size=(10, 6))]
    attempts = []
    accepted = None
    for yaw_index in range(8):
        yaw = yaw_index * math.pi / 4
        wanted = np.asarray(orientation(yaw))
        for seed_index, seed in enumerate(seeds):
            for n, v in rules['finger_joints_m'].items():
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                data.qpos[model.jnt_qposadr[j]] = v

            def residual(q):
                data.qpos[addresses] = q
                mujoco.mj_kinematics(model, data)
                rotation = data.xmat[body].reshape(3, 3)
                position = data.xpos[body] + rotation @ rules['tcp_local_m']
                # 0.2m lever converts orientation radians into position-sized residuals.
                return np.concatenate((position - rules['position_m'],
                                       .2 * Rotation.from_matrix(wanted @ rotation.T).as_rotvec()))

            solution = least_squares(residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
                                     bounds=(lower, upper), max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            case = {'position_m': rules['position_m'], 'rotation_world': wanted.tolist(), 'yaw_rad': yaw,
                    'candidate': {**{'joint' + str(k + 1): float(v) for k, v in enumerate(solution.x)},
                                  **rules['finger_joints_m']}, 'seed_index': seed_index,
                    'solver': 'bounded scipy least_squares; MuJoCo kinematics; deterministic seeds=38'}
            measurement = measure(model, rules, case)
            attempts.append({'case': case, 'measurement': measurement, 'evaluations': solution.nfev})
            if measurement['status'] == 'passed':
                accepted = case
                break
        if accepted:
            break
    (a.output_dir / 'ik_rules.json').write_text(json.dumps(rules, indent=2) + '\n')
    (a.output_dir / 'ik_attempts.json').write_text(json.dumps(attempts, indent=2) + '\n')
    (a.output_dir / 'ik_case.json').write_text(json.dumps(accepted or attempts[-1]['case'], indent=2) + '\n')
    refs = ['ik_rules.json', 'ik_case.json', 'ik_attempts.json',
            *[p.relative_to(a.output_dir).as_posix() for p in (a.output_dir / 'source').rglob('*') if p.is_file()]]
    (a.output_dir / 'ik_integrity.json').write_text(json.dumps({r: digest(a.output_dir / r) for r in refs}, indent=2) + '\n')
    report = {'status': 'passed' if accepted else 'failed', 'attempts': len(attempts),
              'failure_code': None if accepted else ('TARGET_IK_COLLISION' if any(
                  attempt['measurement']['failure_code'] == 'TARGET_IK_COLLISION' for attempt in attempts)
                  else 'TARGET_IK_UNREACHABLE'),
              'measurement': attempts[-1]['measurement'], 'motion': 'not_run', 'grasp': 'not_run',
              'task_rollout': 'not_run', 'model_requests': 0}
    if accepted:
        try:
            report['render'] = render(a.output_dir, accepted, rules)
        except Exception as error:
            report['status'] = 'failed'
            report['failure_code'] = 'TARGET_IK_RENDER_FAILED'
            report['render_error'] = str(error)
    report['elapsed_s'] = round(time.monotonic() - started, 3)
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
