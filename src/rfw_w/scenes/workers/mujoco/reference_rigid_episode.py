# SPDX-License-Identifier: Apache-2.0
"""Finite-time measured rigid manipulation using the existing mobile FR3 actuators."""
import gzip
import json
from pathlib import Path
import time
from rfw_w.scenes.assets.articulated_urdf import digest
from rfw_w.scenes.workers.mujoco.reference_validation import points


def execute(world, output, parameters):
    import mujoco as mj
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    world, output = Path(world), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'controller.py').write_bytes(Path(__file__).read_bytes())
    (output / 'parameters.json').write_text(json.dumps(parameters, indent=2))
    profile = json.loads((world / 'task_profile.json').read_text())
    model = mj.MjModel.from_xml_path(str(world / 'world.xml'))
    hz = 1000
    model.opt.timestep = 1 / hz
    data, ik = mj.MjData(model), mj.MjData(model)
    joints = [model.joint('fr3_joint' + str(i)).id for i in range(1, 8)]
    addresses = model.jnt_qposadr[joints]
    actuators = np.array([model.actuator('fr3_joint' + str(i)).id for i in range(1, 8)])
    fingers = [model.actuator('finger' + str(i)).id for i in (1, 2)]
    limits = model.jnt_range[joints]
    neutral = np.array([0, -.35, 0, -2, 0, 1.7, .7854])
    data.qpos[addresses] = neutral
    data.ctrl[actuators] = neutral
    for i in (1, 2):
        data.qpos[model.joint('finger_joint' + str(i)).qposadr] = .04
        data.ctrl[model.actuator('finger' + str(i)).id] = .04
    mj.mj_forward(model, data)
    tcp = model.site('tcp').id
    previous = neutral.copy()
    joint_integral = np.zeros(7)
    step = 0
    rows = []
    max_pen = 0.
    failure = None
    stream = gzip.open(output / 'contacts.jsonl.gz', 'wt', compresslevel=1)
    geom_names = [model.geom(g).name or 'geom_' + str(g) for g in range(model.ngeom)]
    body_names = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
    started = time.monotonic()

    def move(name, seconds, target, orientation, opening=.04):
        nonlocal step, previous, max_pen
        mj.mj_forward(model, data)
        start = data.site_xpos[tcp].copy()
        startrot = Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3))
        startfinger = data.ctrl[fingers].copy()
        target = np.array(target)
        targetq = previous.copy()
        count = round(seconds * hz)
        for i in range(count):
            f = (i + 1) / count
            blend = f ** 3 * (10 - 15 * f + 6 * f * f)
            desired = start + (target - start) * blend
            desiredrot = Rotation.from_rotvec((orientation * startrot.inv()).as_rotvec() * blend) * startrot
            if step % 20 == 0:
                ik.qpos[:] = data.qpos
                def residual(q):
                    ik.qpos[addresses] = q
                    mj.mj_kinematics(model, ik)
                    return np.r_[10 * (ik.site_xpos[tcp] - desired),
                                 .9 * Rotation.from_matrix(desiredrot.as_matrix() @ ik.site_xmat[tcp].reshape(3, 3).T).as_rotvec(),
                                 .01 * (q - previous)]
                solution = least_squares(residual, np.clip(previous, limits[:, 0] + .07, limits[:, 1] - .07),
                    bounds=(limits[:, 0] + .07, limits[:, 1] - .07), max_nfev=70)
                targetq = solution.x
                previous = targetq.copy()
            delta = parameters.get('joint_speed_rad_s', 1.2) / hz
            joint_integral[:] = np.clip(joint_integral + parameters.get('joint_integral_gain', 0.) *
                                       (targetq - data.qpos[addresses]) / hz, -.08, .08)
            data.ctrl[actuators] += np.clip(targetq + joint_integral - data.ctrl[actuators], -delta, delta)
            data.ctrl[fingers] = startfinger + (opening - startfinger) * blend
            for side in ('left', 'right'):
                for end in ('front', 'rear'):
                    data.ctrl[model.actuator(end + '_' + side + '_drive').id] = 0
            mj.mj_step(model, data)
            for c in data.contact:
                max_pen = max(max_pen, -float(c.dist))
                if c.dist <= .0005:
                    stream.write(json.dumps(dict(t=float(data.time), bodies=[body_names[g] for g in (c.geom1, c.geom2)],
                        geoms=[geom_names[g] for g in (c.geom1, c.geom2)], distance=float(c.dist)), separators=(',', ':')) + '\n')
            if (step + 1) * 30 // hz != step * 30 // hz:
                mj.mj_forward(model, data)
                rows.append(dict(t=float(data.time), phase=name, qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                                 ctrl=data.ctrl.tolist(), tcp=data.site_xpos[tcp].tolist(), target=desired.tolist()))
            step += 1
            if not np.isfinite(data.qpos).all():
                raise RuntimeError('NONFINITE_STATE')
            if max_pen >= .003:
                raise RuntimeError('COLLISION_PENETRATION')
        print(name, round(float(data.time), 2), 'tcp_error', round(float(np.linalg.norm(data.site_xpos[tcp] - target)), 4), flush=True)

    try:
        rotation = Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3))
        move('initial_settle', 2., data.site_xpos[tcp].copy(), rotation)
        for item in profile['objects']:
            role = item['role']
            if parameters.get('diagnostic_roles') and role not in parameters['diagnostic_roles']:
                continue
            body = 'V13_' + item['leaf']
            opts = {**item, **parameters.get('objects', {}).get(role, {})}
            pos = data.body(body).xpos.copy()
            object_rotation = data.body(body).xmat.reshape(3, 3).copy()
            yaw = float(np.arctan2(object_rotation[1, 0], object_rotation[0, 0]))
            rot = Rotation.from_euler('z', yaw) * Rotation.from_euler('xyz', opts['grasp_rpy'])
            offset = np.array(opts['grasp_offset_m'])
            offset[:2] = Rotation.from_euler('z', yaw).as_matrix()[:2, :2] @ offset[:2]
            grasp = pos + offset
            opening = opts.get('grasp_opening_m', .001)
            current_rot = Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3))
            clearance = data.site_xpos[tcp].copy()
            clearance[2] = max(clearance[2], profile['surface_z_m'] + .43)
            move(role + '_clearance', 2., clearance, current_rot)
            move(role + '_orient', 2., clearance, rot)
            move(role + '_above', 5., grasp + [0, 0, .18], rot)
            move(role + '_descend', 2., grasp, rot)
            move(role + '_close', 2., grasp, rot, opening)
            move(role + '_lift', 3., grasp + [0, 0, .18], rot, opening)
            if data.body(body).xpos[2] - pos[2] < .05:
                raise RuntimeError('OBJECT_NOT_LIFTED:' + role)
            if parameters.get('diagnostic_stop_after_lift'):
                break
            desired_object_rot = Rotation.from_euler('xyz', item.get('initial_rpy', [0, 0, 0]))
            def correct_orientation():
                nonlocal rot
                current_object_rot = Rotation.from_matrix(data.body(body).xmat.reshape(3, 3))
                current_tcp_rot = Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3))
                rot = desired_object_rot * current_object_rot.inv() * current_tcp_rot
            if role == 'can':
                for _ in range(3):
                    correct_orientation()
                    move(role + '_upright', 2., data.site_xpos[tcp].copy(), rot, opening)
            if parameters.get('preserve_object_orientation'):
                for _ in range(3):
                    correct_orientation()
                    move(role + '_upright', 2., data.site_xpos[tcp].copy(), rot, opening)
            # Use the actual object/TCP transform after contact to set the endpoint.
            held = data.site_xpos[tcp].copy() - data.body(body).xpos
            goal_xy = np.array(item['goal_xy_m'])
            if item.get('goal_support_role'):
                support = profile['objects'][0]
                support_points = points(model, data, 'V13_' + support['leaf'])
                # The centre cavity floor, excluding the raised peripheral rim.
                centre = support_points[np.linalg.norm(support_points[:, :2] - goal_xy, axis=1) < .035]
                surface = centre[:, 2].max()
                goal_xy = data.body('V13_' + support['leaf']).xpos[:2].copy()
            else:
                surface = profile['surface_z_m']
            move(role + '_transfer', parameters.get('transfer_seconds',4.), [*(goal_xy + held[:2]), profile['surface_z_m'] + .28 + held[2]], rot, opening)
            if role == 'can':
                for _ in range(3):
                    correct_orientation()
                    move(role + '_upright', 2., data.site_xpos[tcp].copy(), rot, opening)
                held = data.site_xpos[tcp].copy() - data.body(body).xpos
            if parameters.get('preserve_object_orientation'):
                for _ in range(2):
                    correct_orientation()
                    move(role + '_upright', 2., data.site_xpos[tcp].copy(), rot, opening)
                held = data.site_xpos[tcp].copy() - data.body(body).xpos
            for _ in range(20):
                vertices = points(model, data, body)
                gap = vertices[:, 2].min() - surface
                if gap < .0015:
                    break
                target = data.site_xpos[tcp].copy()
                target[:2] = goal_xy + held[:2]
                target[2] -= min(.025, max(0, gap - .0005))
                if role == 'can':
                    correct_orientation()
                move(role + '_lower', .8, target, rot, opening)
            else:
                raise RuntimeError('LOWER_TARGET_NOT_REACHED:' + role)
            for _ in range(4):
                error_xy = goal_xy - data.body(body).xpos[:2]
                if np.linalg.norm(error_xy) < .002:
                    break
                target = data.site_xpos[tcp].copy()
                target[:2] += error_xy
                move(role + '_lower', 1.5, target, rot, opening)
            target = data.site_xpos[tcp].copy()
            actual_rot = Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3))
            move(role + '_supported_hold', 2., target, actual_rot, opening)
            move(role + '_release', 1.5, target, actual_rot)
            move(role + '_withdraw', 3., target + [0, 0, .25], actual_rot)
        move('settle', 2., data.site_xpos[tcp].copy(), Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3)))
    except RuntimeError as error:
        failure = str(error)
    finally:
        stream.close()
    raw = dict(schema_version='pvscene.reference-trajectory/v1', task_id=profile['task_id'], fps=30, physics_hz=hz,
               samples=rows, contacts_ref='contacts.jsonl.gz', contact_sample_hz=hz)
    (output / 'trajectory.json').write_text(json.dumps(raw, separators=(',', ':')))
    record = dict(status='failed' if failure else 'recorded_pending_validation', failure_code=failure,
        world_sha256=digest(world / 'world.xml'), graph_sha256=digest(world / 'graph.json'),
        trajectory_sha256=digest(output / 'trajectory.json'), parameters_sha256=digest(output / 'parameters.json'),
        controller_source_sha256=digest(output / 'controller.py'), contacts_sha256=digest(output / 'contacts.jsonl.gz'),
        contact_sample_hz=hz, max_penetration_m=max_pen, object_attachments=0, root_resets_after_initialization=0,
        mechanism_actuators=0, scope='diagnostic' if any(k.startswith('diagnostic_') for k in parameters) else 'full_task', duration_s=float(data.time), elapsed_s=time.monotonic() - started)
    (output / 'recording.json').write_text(json.dumps(record, indent=2))
    print(record['status'], failure, flush=True)
    return record
