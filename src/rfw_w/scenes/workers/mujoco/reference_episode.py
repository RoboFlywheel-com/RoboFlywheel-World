# SPDX-License-Identifier: Apache-2.0
"""Measured-target code controller for T03. Actuators advance all task motion.

This explicit MuJoCo worker is separate from pure contracts and decision providers.
It records every solver contact and each commanded action; no dynamic pose reset.
"""
import json
import gzip
import time
from pathlib import Path


from rfw_w.scenes.assets.articulated_urdf import digest


def execute(world, output, parameters):
    import mujoco as mj
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    world, output = Path(world), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    controller_bytes = Path(__file__).read_bytes()
    (output/'controller.py').write_bytes(controller_bytes)
    model = mj.MjModel.from_xml_path(str(world/'world.xml'))
    physics_hz = int(parameters.get('physics_hz', 1000))
    if physics_hz not in (500, 1000, 2000):
        raise ValueError('PHYSICS_RATE_UNSUPPORTED')
    model.opt.timestep = 1/physics_hz
    data, ik = mj.MjData(model), mj.MjData(model)
    joint_ids = [model.joint('fr3_joint'+str(i)).id for i in range(1, 8)]
    addresses = np.array([model.jnt_qposadr[j] for j in joint_ids])
    actuators = np.array([model.actuator('fr3_joint'+str(i)).id for i in range(1, 8)])
    limits = model.jnt_range[joint_ids]
    neutral = np.array([0, -.35, 0, -2, 0, 1.7, .7854])
    data.qpos[addresses] = neutral; data.ctrl[actuators] = neutral
    for i in (1, 2):
        data.qpos[model.joint('finger_joint'+str(i)).qposadr] = .04
        data.ctrl[model.actuator('finger'+str(i)).id] = .04
    mj.mj_forward(model, data)
    drawer_joint = model.joint('cabinet_carcass_to_drawer_1').qposadr[0]
    tcp = model.site('tcp').id
    previous = neutral.copy()
    joint_integral = np.zeros(7)
    rows, commands = [], []
    contact_stream = gzip.open(output/'contacts.jsonl.gz', 'wt', encoding='utf-8',compresslevel=1)
    geom_names=[model.geom(g).name or f'geom_{g}' for g in range(model.ngeom)]
    geom_body_names=[model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
    failure = None; max_pen = 0.; step = 0
    handle_opening = parameters.get('handle_opening', .003)
    rot = Rotation.from_euler('y', -np.pi/2)
    top = Rotation.from_euler('xyz', [np.pi, 0, parameters.get('caliper_grasp_yaw', np.pi)])
    screw = Rotation.from_euler('xyz', [np.pi, 0, parameters.get('screwdriver_grasp_yaw', -np.pi/2)])
    if parameters.get('contact_timeconst_s'):
        model.geom_solref[:, 0] = parameters['contact_timeconst_s']

    def motion(name, seconds, target, orientation, opening, joint_target=None):
        nonlocal previous, step, max_pen, failure
        mj.mj_forward(model, data)
        start = data.site_xpos[tcp].copy()
        startrot = Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3))
        finger_actuators = [model.actuator('finger'+str(j)).id for j in (1, 2)]
        finger_start = data.ctrl[finger_actuators].copy()
        target = np.array(target)
        targetq = previous.copy()
        startq = data.qpos[addresses].copy()
        for i in range(round(seconds*physics_hz)):
            f = (i+1)/(seconds*physics_hz); f = np.clip(f, 0, 1); blend = f**3*(10-15*f+6*f*f)
            desired = start+(target-start)*blend
            desiredrot = Rotation.from_rotvec((orientation*startrot.inv()).as_rotvec()*blend)*startrot
            if joint_target is not None:
                targetq = startq+(np.array(joint_target)-startq)*blend
                previous = targetq.copy()
            elif step % (physics_hz//50) == 0:
                ik.qpos[:] = data.qpos
                def residual(q):
                    ik.qpos[addresses] = q; mj.mj_kinematics(model, ik)
                    posture = np.array([.0063,.7186,-.522,-1.1039,.3396,1.7424,.3245])
                    regularizer = parameters.get('screwdriver_posture_weight', 0.) if name.startswith('screwdriver') else 0.
                    return np.r_[10*(ik.site_xpos[tcp]-desired),
                                 .9*Rotation.from_matrix(desiredrot.as_matrix()@ik.site_xmat[tcp].reshape(3, 3).T).as_rotvec(),
                                 regularizer*(q-posture) if regularizer else .01*(q-previous)]
                solution = least_squares(residual, np.clip(previous, limits[:, 0]+.07, limits[:, 1]-.07),
                                        bounds=(limits[:, 0]+.07, limits[:, 1]-.07), max_nfev=100)
                if parameters.get('robust_ik') and name in ('screwdriver_transfer','screwdriver_lower','screwdriver_orient_for_place') and np.linalg.norm(solution.fun[:3])>.015:
                    for seed in (neutral, np.array([.0063,.7186,-.522,-1.1039,.3396,1.7424,.3245]), np.array([-2.28,-.53,2.82,-1.64,2.80,3.89,-1.50])):
                        candidate=least_squares(residual,np.clip(seed,limits[:,0]+.07,limits[:,1]-.07),bounds=(limits[:,0]+.07,limits[:,1]-.07),max_nfev=100)
                        if np.linalg.norm(candidate.fun)<np.linalg.norm(solution.fun):solution=candidate
                targetq=solution.x
                previous = targetq.copy()
            delta = parameters.get('joint_target_speed_rad_s', 1.)/physics_hz
            gain = parameters.get('screwdriver_joint_integral_gain', 0.) if name.startswith('screwdriver') else 0.
            joint_integral[:] = np.clip(joint_integral+gain*(targetq-data.qpos[addresses])/physics_hz, -.08, .08) if gain else 0.
            data.ctrl[actuators] += np.clip(targetq+joint_integral-data.ctrl[actuators], -delta, delta)
            data.ctrl[finger_actuators] = finger_start+(opening-finger_start)*blend
            for side in ('left', 'right'):
                for end in ('front', 'rear'):data.ctrl[model.actuator(end+'_'+side+'_drive').id] = 0
            mj.mj_step(model, data)
            for c in data.contact:
                max_pen = max(max_pen, -float(c.dist))
                if c.dist <= .0005:
                    contact_stream.write(json.dumps(dict(t=float(data.time), bodies=[geom_body_names[g] for g in (c.geom1, c.geom2)],
                                         geoms=[geom_names[g] for g in (c.geom1, c.geom2)], distance=float(c.dist)), separators=(',', ':'))+'\n')
            if (step+1)*30//physics_hz != step*30//physics_hz:
                mj.mj_forward(model, data)
                rows.append(dict(t=float(data.time), phase=name, qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                                 ctrl=data.ctrl.tolist(), tcp=data.site_xpos[tcp].tolist(), target=desired.tolist(),
                                 drawer=float(data.qpos[drawer_joint])))
                commands.append(dict(t=float(data.time), ctrl=data.ctrl.tolist(), phase=name))
            step += 1
            if not np.isfinite(data.qpos).all():
                failure = 'NONFINITE_STATE'; return False
            if max_pen > parameters.get('max_penetration_m', .003):
                failure = 'COLLISION_PENETRATION'; return False
        return True

    def move(name, seconds, target, orientation=rot, opening=.04):
        if not motion(name, seconds*parameters.get('time_scale', 1.), target, orientation, opening):
            raise RuntimeError(failure)

    started = time.monotonic()
    try:
        move('initial_settle', 2, data.site_xpos[tcp].copy(), Rotation.from_matrix(data.site_xmat[tcp].reshape(3, 3)))
        handle_offset = np.array(parameters.get('handle_offset', [.012, 0, .01]))
        handle = data.site('drawer_handle').xpos.copy()+handle_offset
        pull = parameters.get('drawer_open_m', .30)
        handle_opening = parameters.get('handle_opening', .003)
        move('approach', 4, handle+[.12, 0, .06])
        move('reach_handle', 2, handle)
        move('grasp_handle', 1.5, handle, opening=handle_opening)
        move('pull', 4, handle+[pull, 0, 0], opening=handle_opening)
        move('hold_open', 1, handle+[pull, 0, 0], opening=handle_opening)
        if data.qpos[drawer_joint] < .24:raise RuntimeError('DRAWER_NOT_OPEN')
        move('release_handle', 1, data.site('drawer_handle').xpos.copy()+handle_offset)
        move('clear_front', 2, data.site('drawer_handle').xpos+[.15, 0, .30])
        for label, leaf, site, offset, orientation in [('caliper', '11_2_0007', 'left_compartment', [.035, 0, 0], top),
                                                       ('screwdriver', '9_6_0000', 'right_compartment', [0, .055, 0], screw)]:
            if data.qpos[drawer_joint] < .24:
                # Recheck and reopen by contact; do not reset a passive joint.
                measured = data.site('drawer_handle').xpos.copy()+handle_offset
                move('reopen_approach', 3, measured+[.12, 0, .12])
                measured = data.site('drawer_handle').xpos.copy()+handle_offset
                move('reopen_reach', 2, measured)
                measured = data.site('drawer_handle').xpos.copy()+handle_offset
                move('reopen_grasp', 1.5, measured, opening=handle_opening)
                move('reopen_pull', 3, measured+[.325-float(data.qpos[drawer_joint]), 0, 0], opening=handle_opening)
                move('reopen_release', 1, data.site('drawer_handle').xpos.copy()+handle_offset)
                move('reopen_clear', 2, data.site('drawer_handle').xpos+[.15, 0, .30])
                if data.qpos[drawer_joint] < .20:raise RuntimeError('DRAWER_PASSIVE_HOLD_FAILED')
            obj = data.xpos[model.body('V13_'+leaf).id].copy()
            if label=='screwdriver' and parameters.get('screwdriver_neutral_transit', False):
                if not motion('screwdriver_neutral_transit', 5., data.site_xpos[tcp].copy(), orientation, .04, joint_target=parameters.get('screwdriver_transit_joints', neutral.tolist())):
                    raise RuntimeError(failure)
            offset = np.array(parameters.get(label+'_grasp_offset', offset))
            if parameters.get(label+'_grasp_frame') == 'object':
                offset = data.xmat[model.body('V13_'+leaf).id].reshape(3,3)@offset
            g = obj+offset
            release_height = parameters.get(label+'_release_height_m', parameters.get('release_height_m', .12))
            destination = data.site(site).xpos+offset+[0, 0, release_height]
            if label=='screwdriver' and parameters.get('screwdriver_transit_joints'):
                desired_seed=np.array(parameters['screwdriver_transit_joints'])
                if not motion('screwdriver_grasp_posture', 4., data.site_xpos[tcp].copy(), orientation, .04, joint_target=desired_seed):raise RuntimeError(failure)
            move(label+'_above', 6, g+[0, 0, .17], orientation)
            move(label+'_descend', 2, g, orientation)
            move(label+'_close', 2.5, g, orientation, .001)
            move(label+'_lift', 3, g+[0, 0, .19], orientation, .001)
            if data.xpos[model.body('V13_'+leaf).id, 2]-obj[2] < .05:
                raise RuntimeError('OBJECT_NOT_LIFTED:'+label)
            move(label+'_transfer', 4, [destination[0], destination[1], 1.23], orientation, .001)
            # Actual held mesh/TCP transform sets the release height. The grasp
            # can differ from the nominal offset after physical finger closure.
            mj.mj_forward(model, data)
            object_bid = model.body('V13_'+leaf).id
            def object_bottom():
                mj.mj_forward(model, data)
                vertices = []
                for geom in range(model.ngeom):
                    if model.geom_bodyid[geom] == object_bid and model.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH:
                        mesh = model.geom_dataid[geom]
                        v = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
                        vertices.extend(v@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom])
                return np.array(vertices)[:,2].min()
            floor_height = data.site(site).xpos[2]
            lower_xy = destination[:2].copy()
            if label=='screwdriver' and parameters.get('screwdriver_lower_orientation_rpy'):
                orientation=Rotation.from_euler('xyz',parameters['screwdriver_lower_orientation_rpy'])
                move('screwdriver_orient_for_place', 3., [lower_xy[0],lower_xy[1],1.23], orientation,.001)
            for lower_index in range(30):
                gap = object_bottom()-floor_height
                if gap < parameters.get('release_clearance_m', .012):
                    break
                destination = data.site_xpos[tcp].copy()
                if label=='screwdriver' and parameters.get('fixed_lower_xy', True):destination[:2] = lower_xy
                if label=='screwdriver' and parameters.get('screwdriver_lower_orientation_rpy'):
                    orientation=Rotation.from_euler('xyz',parameters['screwdriver_lower_orientation_rpy'])
                destination[2] -= min(.03, max(0., gap-parameters.get('release_clearance_m', .012)*.5))
                # Preserve the actual held orientation during final descent.
                held_orientation = orientation
                if label=='screwdriver' and parameters.get('screwdriver_lower_joint_target'):
                    targetj=np.array(parameters['screwdriver_lower_joint_target'])
                    if not motion(label+'_lower', 6., destination, held_orientation, .001,joint_target=targetj):raise RuntimeError(failure)
                else:move(label+'_lower', .5, destination, held_orientation, .001)
            else:
                raise RuntimeError('LOWER_TARGET_NOT_REACHED')
            destination = data.site_xpos[tcp].copy()
            orientation = Rotation.from_matrix(data.site_xmat[tcp].reshape(3,3))
            move(label+'_release', 1.5, destination, orientation)
            move(label+'_withdraw', 2.5, destination+[0, 0, .45], orientation)
        handle = data.site('drawer_handle').xpos.copy()+handle_offset
        move('return_front', 8, handle+[.15, 0, .20])
        handle = data.site('drawer_handle').xpos.copy()+handle_offset
        move('return_handle', 2, handle)
        handle = data.site('drawer_handle').xpos.copy()+handle_offset
        move('regrasp_handle', 1.5, handle, opening=handle_opening)
        closed_handle = handle-[float(data.qpos[drawer_joint]), 0, 0]
        move('push', 4, closed_handle, opening=handle_opening)
        move('hold_closed', parameters.get('closed_hold_s', 2.), closed_handle, opening=handle_opening)
        # Release at the fixed measured closed target. Following the moving knob
        # while opening the fingers can impart an outward velocity to the drawer.
        move('release', 1.5, closed_handle)
        move('withdraw', 2, closed_handle+[.15, 0, .30])
        move('settle', 2, data.site_xpos[tcp].copy())
    except RuntimeError as error:
        failure = str(error)
    contact_stream.close()
    raw = dict(schema_version='pvscene.reference-trajectory/v1', fps=30, physics_hz=physics_hz,
               samples=rows, contacts_ref='contacts.jsonl.gz', contact_sample_hz=physics_hz)
    (output/'trajectory.json').write_text(json.dumps(raw, separators=(',', ':')))
    (output/'commands.json').write_text(json.dumps(commands, separators=(',', ':')))
    (output/'parameters.json').write_text(json.dumps(parameters, indent=2))
    record = dict(status='failed' if failure else 'recorded_pending_validation', failure_code=failure,
                  world_sha256=digest(world/'world.xml'), graph_sha256=digest(world/'graph.json'),
                  trajectory_sha256=digest(output/'trajectory.json'), controller_source_sha256=digest(output/'controller.py'),
                  parameters_sha256=digest(output/'parameters.json'), max_penetration_m=max_pen,
                  contacts_sha256=digest(output/'contacts.jsonl.gz'),
                  scope='full_task',
                  object_attachments=0, root_resets_after_initialization=0, mechanism_actuators=0,
                  contact_sample_hz=physics_hz, contact_gzip_level=1, duration_s=float(data.time), elapsed_s=time.monotonic()-started)
    (output/'recording.json').write_text(json.dumps(record, indent=2))
    print(record['status'], failure, round(float(data.time), 2), 's; penetration', round(max_pen, 5), flush=True)
    return record
