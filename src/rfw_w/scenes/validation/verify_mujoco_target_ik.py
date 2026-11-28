# SPDX-License-Identifier: Apache-2.0
"""Independent real target geometry, finite joint limits, FK and endpoint contacts."""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest


def require(condition, code):
    if not condition:
        raise ValueError(code)


def geometry(root):
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation
    source = root / 'source'
    config = json.loads((source / 'source/installation.json').read_text())
    item = next(i for i in config['instances'] if i['role_id'] == config['target_role_id'])
    mesh = trimesh.load(source / 'source/scene' / item['collision_ref'], process=False, force='mesh')
    theta = item.get('yaw_rad', 0)
    rotation = Rotation.from_euler('z', theta).as_matrix()
    vertices = np.asarray(mesh.vertices) @ rotation.T + item['pose_xyz_m']
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    tree = ET.parse(source / 'source/robot/robot.urdf')
    tips = []
    for name in ('gripper_link1', 'gripper_link2'):
        link = tree.find(f"link[@name='{name}']")
        joint = next(j for j in tree.findall('joint') if j.find('child').get('link') == name)
        require(joint.find('parent').get('link') == 'gripper_base', 'TARGET_IK_INPUT_INVALID')
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        collision = link.find('collision')
        co = collision.find('origin')
        cr = Rotation.from_euler('xyz', np.fromstring(co.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        cp = np.fromstring(co.get('xyz', '0 0 0'), sep=' ')
        element = collision.find('geometry/mesh')
        ref = element.get('filename')
        path = (source / 'source/robot' / ref).resolve()
        require(path.is_relative_to((source / 'source/robot').resolve()), 'TARGET_IK_INPUT_INVALID')
        finger = trimesh.load(path, process=False, force='mesh')
        scale = np.fromstring(element.get('scale', '1 1 1'), sep=' ')
        v = (np.asarray(finger.vertices) * scale @ cr.T + cp) @ Rotation.from_euler('xyz', rpy).as_matrix().T + xyz
        tips.append(float(v[:, 2].max()))
    tcp = [0., 0., sum(tips) / 2]
    position = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(hi[2] + .08)]
    return {'instance_id': item['instance_id'], 'asset_id': item['asset_id'],
            'tcp_body': 'gripper_base', 'tcp_local_m': tcp,
            'tcp_method': 'closed finger mesh distal Z midpoint, geometric estimate',
            'position_m': position, 'cup_world_bounds_m': [lo.tolist(), hi.tolist()],
            'clearance_m': .08, 'finger_joints_m': {'joint7': .05, 'joint8': -.05}}


def orientation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, s, 0], [s, -c, 0], [0, 0, -1]]


def measure(model, rules, case):
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    candidate = case['candidate']
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    require(set(candidate) == set(names) and len(names) == 8, 'TARGET_IK_JOINT_INVALID')
    data = mujoco.MjData(model)
    for name, value in candidate.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        require(isinstance(value, (int, float)) and math.isfinite(value) and
                model.jnt_range[j, 0] <= value <= model.jnt_range[j, 1], 'TARGET_IK_JOINT_INVALID')
        if name in rules['finger_joints_m']:
            require(abs(value - rules['finger_joints_m'][name]) < 1e-9, 'TARGET_IK_JOINT_INVALID')
        data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rules['tcp_body'])
    require(body >= 0, 'TARGET_IK_INPUT_INVALID')
    rotation = data.xmat[body].reshape(3, 3)
    position = data.xpos[body] + rotation @ np.asarray(rules['tcp_local_m'])
    target = np.asarray(case['position_m'])
    wanted = np.asarray(case['rotation_world'])
    require(target.shape == (3,) and np.isfinite(target).all() and wanted.shape == (3, 3)
            and np.isfinite(wanted).all() and np.allclose(wanted.T @ wanted, np.eye(3), atol=1e-8)
            and abs(np.linalg.det(wanted) - 1) < 1e-8, 'TARGET_IK_INPUT_INVALID')
    distance = float(np.linalg.norm(position - target))
    angle = float(Rotation.from_matrix(wanted @ rotation.T).magnitude())
    penetrations = []
    for contact in data.contact:
        if contact.dist < -.001:
            penetrations.append({'geom1': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                                 'geom2': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
                                 'distance_m': float(contact.dist)})
    result = {'position_error_m': distance, 'orientation_error_rad': angle,
              'actual_tcp_m': position.tolist(), 'contacts_count': data.ncon, 'penetrations': penetrations}
    result['failure_code'] = ('TARGET_IK_UNREACHABLE' if distance > .002 or angle > .02 else
                              'TARGET_IK_COLLISION' if penetrations else None)
    result['status'] = 'failed' if result['failure_code'] else 'passed'
    return result


def verify(root):
    import mujoco
    from rfw_w.scenes.validation.verify_mujoco_installation_appearance import verify as verify_appearance
    hashes = json.loads((root / 'ik_integrity.json').read_text())
    require({'ik_rules.json', 'ik_case.json'} <= set(hashes), 'TARGET_IK_ARTIFACT_MISMATCH')
    for ref, expected in hashes.items():
        path = (root / ref).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and digest(path) == expected,
                'TARGET_IK_ARTIFACT_MISMATCH')
    actual_refs = {p.relative_to(root).as_posix() for p in (root / 'source').rglob('*') if p.is_file()}
    require(actual_refs <= set(hashes), 'TARGET_IK_ARTIFACT_MISMATCH')
    require(verify_appearance(root / 'source')['verified'], 'TARGET_IK_INPUT_INVALID')
    from rfw_w.scenes.validation.verify_single_room_installation import verify as verify_installation
    require(verify_installation(root / 'source/source')['verified'], 'TARGET_IK_INPUT_INVALID')
    rules = json.loads((root / 'ik_rules.json').read_text())
    require(rules == geometry(root), 'TARGET_IK_INPUT_INVALID')
    case = json.loads((root / 'ik_case.json').read_text())
    require(case['position_m'] == rules['position_m'] and case['rotation_world'] == orientation(case['yaw_rad'])
            and any(abs(case['yaw_rad'] - k * math.pi / 4) < 1e-9 for k in range(8)), 'TARGET_IK_INPUT_INVALID')
    urdf = ET.parse(root / 'source/source/robot/robot.urdf')
    for joint in urdf.findall('joint'):
        if joint.get('type') == 'fixed':
            continue
        limit = joint.find('limit')
        value = case['candidate'].get(joint.get('name'))
        require(isinstance(value, (int, float)) and math.isfinite(value) and
                float(limit.get('lower')) <= value <= float(limit.get('upper')), 'TARGET_IK_JOINT_INVALID')
    model = mujoco.MjModel.from_xml_path(str(root / 'source/appearance.xml'))
    result = measure(model, rules, case)
    return {'verified': result['status'] == 'passed', 'measurement': result,
            'failure_code': result['failure_code'], 'backend': 'mujoco', 'version': mujoco.__version__,
            'motion': 'not_run', 'grasp': 'not_run', 'task_rollout': 'not_run', 'model_requests': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = verify(a.delivery)
    except (ValueError, OSError, KeyError, StopIteration, TypeError) as error:
        report = {'verified': False, 'failure_code': str(error)}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
