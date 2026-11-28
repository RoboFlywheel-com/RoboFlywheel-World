# SPDX-License-Identifier: Apache-2.0
"""Independent source, full-step contact and measured rigid task checks."""
import gzip
import json
from pathlib import Path
from rfw_w.scenes.assets.articulated_urdf import digest
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.workers.mujoco.reference_validation import points


def verify(world, run):
    import mujoco as mj
    import numpy as np
    world, run = Path(world), Path(run)
    manifest = json.loads((world / 'manifest.json').read_text())
    profile = json.loads((world / 'task_profile.json').read_text())
    record = json.loads((run / 'recording.json').read_text())
    raw = json.loads((run / 'trajectory.json').read_text())
    rows = raw['samples']
    model = mj.MjModel.from_xml_path(str(world / 'world.xml'))
    data = mj.MjData(model)
    graph = TaskWorldGraph.from_dict(json.loads((world / 'graph.json').read_text()))
    checks = dict(world_hash=digest(world / 'world.xml') == record['world_sha256'] == manifest['world_sha256'],
        graph_binding=graph.content_hash == manifest['graph_hash'] and digest(world / 'graph.json') == record['graph_sha256'],
        profile_binding=digest(world / 'task_profile.json') == manifest['profile_sha256'],
        episode_binding=content_hash(json.loads((world / 'episode.json').read_text())) == manifest['episode_hash'],
        trajectory_hash=digest(run / 'trajectory.json') == record['trajectory_sha256'],
        controller_hash=digest(run / 'controller.py') == record['controller_source_sha256'],
        parameter_hash=digest(run / 'parameters.json') == record['parameters_sha256'],
        contact_hash=digest(run / 'contacts.jsonl.gz') == record['contacts_sha256'],
        dependency_hashes=all(digest(world / p) == v['sha256'] for p, v in manifest['dependencies'].items()),
        render_profile_binding=digest(world / 'render_profile.json') == manifest['render_profile_sha256'],
        complete=record['status'] == 'recorded_pending_validation' and bool(rows) and rows[-1]['phase'] == 'settle',
        full_task_scope=record['scope'] == 'full_task',
        no_assistance=record['object_attachments'] == record['root_resets_after_initialization'] == 0,
        no_object_equality=not any(t in (mj.mjtEq.mjEQ_WELD, mj.mjtEq.mjEQ_CONNECT) for t in model.eq_type),
        full_solver_contact_log=raw['physics_hz'] == raw['contact_sample_hz'] == record['contact_sample_hz'] == 1000,
        rigid_physics_binding=all(digest(world / p['ref']) == p['sha256'] and p['source_scale'] == 1 and
            abs(model.body_mass[model.body('V13_' + leaf).id] - p['mass_kg']) < 1e-6 for leaf, p in manifest['rigid_physics'].items()),
        free_objects=all(model.jnt_type[model.body('V13_' + item['leaf']).jntadr[0]] == mj.mjtJoint.mjJNT_FREE for item in profile['objects']))
    positions, rotations, root = {}, {}, []
    violation = 0.
    for row in rows:
        data.qpos[:] = row['qpos']
        mj.mj_forward(model, data)
        root.append(data.body('mobile_base').xpos.copy())
        for item in profile['objects']:
            role = item['role']
            body = data.body('V13_' + item['leaf'])
            positions.setdefault(role, []).append(body.xpos.copy())
            rotations.setdefault(role, []).append(body.xmat.reshape(3, 3).copy())
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                q = data.qpos[model.jnt_qposadr[j]]
                violation = max(violation, model.jnt_range[j, 0] - q, q - model.jnt_range[j, 1])
    checks['finite_state_action'] = bool(rows) and all(np.isfinite(r[f]).all() for r in rows for f in ('qpos', 'qvel', 'ctrl'))
    checks['joint_limits'] = violation < .002
    checks['base_drift'] = bool(root) and np.linalg.norm(np.array(root) - root[0], axis=1).max() < .01
    checks['uniform_recording'] = len(rows) > 1 and np.allclose(np.diff([r['t'] for r in rows]), 1 / 30, atol=.0011)
    filtered = []
    maxpen = 0.
    bodies = {'V13_' + item['leaf'] for item in profile['objects']}
    with gzip.open(run / 'contacts.jsonl.gz', 'rt') as stream:
        for line in stream:
            c = json.loads(line)
            maxpen = max(maxpen, -c['distance'])
            if bodies.intersection(c['bodies']):
                filtered.append(c)
    checks['penetration_below_3mm'] = maxpen < .003
    objects = {}
    if rows:
        for item in profile['objects']:
            role, body = item['role'], 'V13_' + item['leaf']
            pos = np.array(positions[role])
            final = [c for c in filtered if body in c['bodies'] and c['t'] > rows[-1]['t'] - 1]
            support = 'V13_' + next(o['leaf'] for o in profile['objects'] if o['role'] == item['goal_support_role']) if item.get('goal_support_role') else 'garden_bench'
            checks[role + '_lifted'] = pos[:, 2].max() - pos[0, 2] > .05
            checks[role + '_supported'] = any(support in c['bodies'] for c in final)
            checks[role + '_released'] = bool(final) and not any({'left_finger', 'right_finger', 'hand'}.intersection(c['bodies']) for c in final)
            drift = np.linalg.norm(pos[-30:] - pos[-1], axis=1).max()
            checks[role + '_stable'] = drift < .002
            # Profiles explicitly record inverted source orientation. The task's upright
            # axis is negative source Z for flipped pot/saucer, positive Z for the can.
            sign = -1 if item.get('initial_rpy', [0, 0, 0])[0] > 3 else 1
            tilt = float(np.degrees(np.arccos(np.clip(sign * rotations[role][-1][2, 2], -1, 1))))
            checks[role + '_upright'] = tilt < 10
            goal_xy = np.array(item['goal_xy_m'])
            if item.get('goal_support_role'):
                goal_xy = np.array(positions[item['goal_support_role']][-1][:2])
            centre_error = np.linalg.norm(pos[-1, :2] - goal_xy)
            checks[role + '_goal_position'] = centre_error < (.010 if item.get('goal_support_role') else .015)
            geometry = points(model, data, body)
            checks[role + '_on_worktop'] = geometry[:, 0].min() > -1.48 and geometry[:, 0].max() < -.82 and geometry[:, 1].min() > -.75 and geometry[:, 1].max() < .75
            active = [r for r in rows if r['phase'] in (role + '_lift', role + '_transfer', role + '_lower')]
            missing = []
            if active:
                count = int((active[-1]['t'] - active[0]['t'] - .1) / .1) + 1
                bins = [set() for _ in range(max(0, count))]
                for c in filtered:
                    if body in c['bodies'] and c['distance'] <= .0001:
                        i = int(np.floor((c['t'] - active[0]['t']) / .1))
                        if 0 <= i < len(bins):
                            bins[i].update(c['bodies'])
                missing = [round(active[0]['t'] + i * .1, 4) for i, names in enumerate(bins) if not {'left_finger', 'right_finger'} <= names and support not in names]
                checks[role + '_contact_continuity'] = bool(bins) and not missing
            else:
                checks[role + '_contact_continuity'] = False
            objects[role] = dict(centre_error_m=float(centre_error), drift_m=float(drift), tilt_deg=tilt,
                                 bounds_world_m=[geometry.min(0).tolist(), geometry.max(0).tolist()], missing_contact_windows_s=missing)
        # Direction is frozen by the source spout orientation; no pouring is shown.
        if profile['task_id']=='T12':
            can = rotations['can'][-1]
            checks['can_orientation'] = can[0, 0] > np.cos(np.deg2rad(15))
    return dict(status='passed' if all(checks.values()) else 'failed', checks={k: bool(v) for k, v in checks.items()},
                trajectory_sha256=record['trajectory_sha256'], metrics=dict(max_penetration_m=maxpen, joint_violation_rad=violation),
                objects=objects, scope='source-bound rigid task, full solver contacts and sampled mesh geometry; no continuous certification')
