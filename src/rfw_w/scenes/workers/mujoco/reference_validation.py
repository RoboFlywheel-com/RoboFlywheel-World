# SPDX-License-Identifier: Apache-2.0
"""Independent world/trajectory recomputation; never consumes controller success alone."""
import gzip
import json
from pathlib import Path

from rfw_w.scenes.assets.articulated_urdf import digest
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.contracts.hashing import content_hash


def points(model, data, body):
    import mujoco as mj
    import numpy as np
    result = []
    bid = model.body(body).id
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid and model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            verts = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
            result.extend(verts@data.geom_xmat[g].reshape(3, 3).T+data.geom_xpos[g])
    return np.array(result)


def verify(world, run):
    import mujoco as mj
    import numpy as np
    world, run = Path(world), Path(run)
    manifest = json.loads((world/'manifest.json').read_text())
    record = json.loads((run/'recording.json').read_text())
    raw = json.loads((run/'trajectory.json').read_text())
    model = mj.MjModel.from_xml_path(str(world/'world.xml'));data = mj.MjData(model)
    graph = TaskWorldGraph.from_dict(json.loads((world/'graph.json').read_text()))
    rows = raw['samples']
    if 'contacts_ref' in raw:
        contact_path = run/raw['contacts_ref']
        contacts = (json.loads(line) for line in gzip.open(contact_path, 'rt'))
    else:
        contacts = iter(raw['contacts'])
    checks = dict(world_hash=manifest['world_sha256']==record['world_sha256']==digest(world/'world.xml'),
                  graph_binding=graph.content_hash==manifest['graph_hash'] and record['graph_sha256']==digest(world/'graph.json'),
                  episode_binding=content_hash(json.loads((world/'episode.json').read_text()))==manifest['episode_hash'],
                  no_object_equality=not any(model.eq_type[i] in (mj.mjtEq.mjEQ_WELD,mj.mjtEq.mjEQ_CONNECT) for i in range(model.neq)),
                  trajectory_hash=record['trajectory_sha256']==digest(run/'trajectory.json'),
                  parameter_hash=record['parameters_sha256']==digest(run/'parameters.json'),
                  saved_controller_hash=(run/'controller.py').exists() and record['controller_source_sha256']==digest(run/'controller.py'),
                  passive_mechanism=record['mechanism_actuators']==0 and not any('cabinet' in model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)),
                  source_scale=manifest['articulated']['scale']==1.,
                  source_masses=all(abs(model.body_mass[model.body(v['body']).id]-v['mass_kg'])<1e-6 for v in manifest['articulated']['links'].values()),
                  no_assistance=record['object_attachments']==record['root_resets_after_initialization']==0,
                  complete=record['status']=='recorded_pending_validation' and bool(rows) and rows[-1]['phase']=='settle',
                  full_task_scope=record.get('scope','full_task')=='full_task',
                  full_solver_contact_log=record['contact_sample_hz']==raw['contact_sample_hz']==raw['physics_hz'])
    if 'contacts_ref' in raw:
        checks['contact_hash'] = record['contacts_sha256']==digest(contact_path)
    maxpen = 0.; filtered = []
    for c in contacts:
        maxpen = max(maxpen, -c['distance'])
        if any(b in ('V13_11_2_0007','V13_9_6_0000','left_finger','right_finger') for b in c['bodies']):
            filtered.append(c)
    contacts = filtered
    checks['dependency_hashes']=all((world/k).is_file() and digest(world/k)==v['sha256'] for k,v in manifest.get('dependencies',{}).items())
    if manifest.get('scene_dressing'):
        dressing=manifest['scene_dressing']
        checks['scene_dressing_binding']=digest(world/dressing['ref'])==dressing['sha256']
        checks['render_profile_binding']=digest(world/'render_profile.json')==manifest['render_profile_sha256']
    checks['rigid_physics_binding']=bool(manifest.get('rigid_physics')) and all(digest(world/v['ref'])==v['sha256'] and abs(model.body_mass[model.body('V13_'+leaf).id]-v['mass_kg'])<1e-6 for leaf,v in manifest.get('rigid_physics',{}).items())
    checks['penetration_below_3mm'] = maxpen < .003
    checks['finite_state_action']=all(np.isfinite(row[field]).all() for row in rows for field in ('qpos','qvel','ctrl'))
    positions = {}; root=[];drawer=[];violation=0.
    for row in rows:
        data.qpos[:] = row['qpos'];mj.mj_forward(model,data)
        root.append(data.xpos[model.body('mobile_base').id].copy());drawer.append(row['drawer'])
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                v=data.qpos[model.jnt_qposadr[j]];violation=max(violation,model.jnt_range[j,0]-v,v-model.jnt_range[j,1])
        for leaf in ('11_2_0007','9_6_0000'):
            positions.setdefault(leaf,[]).append(data.xpos[model.body('V13_'+leaf).id].copy())
    checks['drawer_opened'] = max(drawer,default=0)>.24
    checks['drawer_closed'] = bool(drawer) and abs(drawer[-1])<.005
    for phase in ('pull','push'):
        times=[r['t'] for r in rows if r['phase']==phase]
        checks[phase+'_handle_contact']=bool(times) and any(times[0]<=c['t']<=times[-1] and 'cabinet_drawer_1' in c['bodies'] and any('finger' in b for b in c['bodies']) for c in contacts)
    checks['joint_limits'] = violation<.002
    checks['base_drift'] = bool(root) and np.linalg.norm(np.array(root)-root[0],axis=1).max()<.01
    details={}
    if rows:
        for label,leaf,left in [('caliper','11_2_0007',True),('screwdriver','9_6_0000',False)]:
            p=points(model,data,'V13_'+leaf);bid=model.body('cabinet_drawer_1').id
            local=(p-data.xpos[bid])@data.xmat[bid].reshape(3,3)
            minp,maxp=local.min(0),local.max(0)
            contained=(minp[0]>-.426 and maxp[0]<-.028 and minp[2]>-.161 and maxp[2]<.085
                       and minp[1]>.016 and maxp[1]<1.4165
                       and (maxp[1]<.71025 if left else minp[1]>.72225))
            final=[c for c in contacts if c['t']>rows[-1]['t']-1 and 'V13_'+leaf in c['bodies']]
            checks[label+'_contained']=bool(contained)
            checks[label+'_support']=any('cabinet_drawer_1' in c['bodies'] for c in final)
            checks[label+'_released']=not any(any('finger' in b or b=='hand' for b in c['bodies']) for c in final)
            ps=np.array(positions[leaf]);checks[label+'_stable']=np.linalg.norm(ps[-30:]-ps[-1],axis=1).max()<.002
            checks[label+'_lifted']=ps[:,2].max()-ps[0,2]>.05
            active=[s for s in rows if s['phase'] in (label+'_transfer',label+'_lower')]
            bins=[]; missing=[]
            if active:
                count=max(0,int((active[-1]['t']-.1-active[0]['t'])/.1)+1)
                names_by_bin=[set() for _ in range(count)];world_support=[False]*count
                for c in contacts:
                    if 'V13_'+leaf not in c['bodies']:continue
                    index=int(np.floor((c['t']-active[0]['t'])/.1))
                    if 0<=index<count:
                        names_by_bin[index].update(c['bodies'])
                        world_support[index] |= 'world' in c['bodies']
                for index,names in enumerate(names_by_bin):
                    ok={'left_finger','right_finger'}<=names or ('cabinet_drawer_1' in names and not world_support[index])
                    bins.append(ok)
                    if not ok:missing.append(round(float(active[0]['t']+index*.1),4))
            checks[label+'_contact_continuity']=bool(bins) and all(bins)
            details[label]={'bounds_local':[minp.tolist(),maxp.tolist()],'contact_bins':len(bins),'missing_contact_windows_s':missing}
    return dict(status='passed' if all(checks.values()) else 'failed',checks={k:bool(v) for k,v in checks.items()},
                metrics=dict(max_penetration_m=maxpen,joint_violation_rad=violation,drawer_max_m=max(drawer,default=0),drawer_final_m=drawer[-1] if drawer else None),
                objects=details,trajectory_sha256=record['trajectory_sha256'],scope='Independent source-binding, sampled geometry and full solver contact checks; no continuous certification')
