# SPDX-License-Identifier: Apache-2.0
"""G1 contact-driven fixed-base expert rollout. Run with Isaac Sim Python.
Only joint targets are changed during rollout; the object remains a dynamic body.
"""


def main():
    import argparse
    import copy
    import json
    import hashlib
    import xml.etree.ElementTree as ET
    from pathlib import Path

    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--probe',action='store_true')
    a=p.parse_args()
    cp=Path(a.config).resolve();cfg=json.loads(cp.read_text());input_cfg=copy.deepcopy(cfg)
    out=cp.parent/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    if (out/'trajectory.json').exists():raise FileExistsError('Use a new output_dir; trajectory evidence already exists')
    if cfg['physics_hz']%cfg['fps']:raise ValueError('physics_hz must be divisible by fps')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':1280,'height':720,'renderer':'RayTracedLighting','multi_gpu':False,'active_gpu':cfg.get('active_gpu',0)})
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from scipy.optimize import least_squares
    import omni.usd
    from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdLux,UsdShade,Sdf,PhysicsSchemaTools
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation,RigidPrim
    from isaacsim.sensors.camera import Camera

    xml=ET.parse(cfg['g1_urdf']).getroot()
    joints={j.get('name'):j for j in xml.findall('joint')}
    chain=[];link='right_hand_palm_link'
    while link!='pelvis':
     j=next(j for j in joints.values() if j.find('child').get('link')==link)
     chain.insert(0,j);link=j.find('parent').get('link')
    arm=[j.get('name') for j in chain if j.get('type')!='fixed' and j.get('name').startswith('right_')]
    base=np.array(cfg.get('robot_base',[-.34,-.28,.79]))
    def fk(q):
     t=np.eye(4);t[:3,3]=base
     for j in chain:
      ori=j.find('origin');xyz=np.fromstring(ori.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(ori.get('rpy','0 0 0'),sep=' ')
      s=np.eye(4);s[:3,3]=xyz;s[:3,:3]=Rot.from_euler('xyz',rpy).as_matrix();t=t@s
      if j.get('type')!='fixed':
       axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');s=np.eye(4);s[:3,:3]=Rot.from_rotvec(axis*q.get(j.get('name'),0.)).as_matrix();t=t@s
     return t
    lo=np.array([float(joints[n].find('limit').get('lower')) for n in arm]);hi=np.array([float(joints[n].find('limit').get('upper')) for n in arm])
    for name,bounds in cfg.get('ik_limits',{}).items():
     i=arm.index(name);lo[i]=max(lo[i],bounds[0]);hi[i]=min(hi[i],bounds[1])
    def ik(pos,rotation,seed):
     def residual(x):
      t=fk(dict(zip(arm,x)));return np.r_[t[:3,3]-pos,.14*Rot.from_matrix(rotation@t[:3,:3].T).as_rotvec(),.001*(x-seed)]
     sol=least_squares(residual,np.clip(seed,lo+.0001,hi-.0001),bounds=(lo,hi),max_nfev=200)
     for retry in range(12):
      if np.linalg.norm(residual(sol.x)[:3])<.003:break
      candidate=least_squares(residual,np.random.default_rng(retry).uniform(lo+.001,hi-.001),bounds=(lo,hi),max_nfev=300)
      if np.linalg.norm(residual(candidate.x)[:6])<np.linalg.norm(residual(sol.x)[:6]):sol=candidate
     err=residual(sol.x)[:6]
     if np.linalg.norm(err[:3])>cfg.get('ik_position_tolerance',.003) or np.linalg.norm(err[3:])>.14*.06:raise RuntimeError('IK position residual '+str(err))
     return sol.x

    world=World(stage_units_in_meters=1.,physics_dt=1/cfg['physics_hz'],rendering_dt=1/cfg['fps'])
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    scene=UsdPhysics.Scene.Get(stage,world.get_physics_context().prim_path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
    pa=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());pa.CreateEnableCCDAttr(True);pa.CreateEnableStabilizationAttr(True);pa.CreateSolverTypeAttr('TGS')
    pa.CreateEnableGPUDynamicsAttr(False);pa.CreateBroadphaseTypeAttr('MBP')
    def cube(path,position,size,color):
     c=UsdGeom.Cube.Define(stage,path);c.CreateSizeAttr(1)
     x=UsdGeom.Xformable(c);x.AddTranslateOp().Set(Gf.Vec3d(*map(float,position)));x.AddScaleOp().Set(Gf.Vec3f(*map(float,size)))
     c.CreateDisplayColorAttr([Gf.Vec3f(*color)]);UsdPhysics.CollisionAPI.Apply(c.GetPrim());return c.GetPrim()
    cube('/World/Floor',(0,0,-.05),(4,4,.1),(.55,.53,.48))
    cube('/World/Counter',cfg.get('counter_center',[.45,-.25,.79]),cfg.get('counter_size',[.90,1.1,.08]),(.83,.79,.68))
    cube('/World/Cabinet',(.53,-.25,.39),(.66,1.06,.78),(.22,.34,.28))
    cube('/World/BackWall',(1.6,0,1.25),(.08,3.5,2.5),(.78,.79,.73))
    cube('/World/SupportExtension',(-.09,-.22,.79),(.18,.38,.08),(.83,.79,.68))
    if not cfg.get('scene_assets'):
     cube('/World/Tray',(*cfg.get('tray_xy',[.015,-.22]),.837),(.27,.24,.014),(.38,.46,.40))
    from rfw_w.scenes.workers.isaac.episodes.release_assets import add_release_asset
    scene_assets=[add_release_asset(stage,cfg['assets_root'],spec) for spec in cfg.get('scene_assets',[])]
    dome=UsdLux.DomeLight.Define(stage,'/World/Dome');dome.CreateIntensityAttr(700)
    light=UsdLux.DistantLight.Define(stage,'/World/Key');light.CreateIntensityAttr(1800);UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(30,-35,-35))
    root=UsdGeom.Xform.Define(stage,'/World/G1').GetPrim();root.GetReferences().AddReference(str(Path(cfg.get('robot_usd', out/'configuration/g1_import_physics.usd'))),'/g1_29dof_with_hand_rev_1_0')
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*map(float,base)))
    roots=[str(pr.GetPath()) for pr in Usd.PrimRange(root) if pr.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print('ROOTS',roots,flush=True)
    for pr in Usd.PrimRange(root):
     if pr.IsA(UsdPhysics.RevoluteJoint):
      n=pr.GetName()
      if cfg.get('lock_torso') and n.startswith('waist_'):
       UsdPhysics.RevoluteJoint(pr).CreateLowerLimitAttr(0.);UsdPhysics.RevoluteJoint(pr).CreateUpperLimitAttr(0.)
      if n in joints:
       d=UsdPhysics.DriveAPI.Apply(pr,'angular');d.CreateTypeAttr('force')
       hand='hand' in n;d.CreateStiffnessAttr(18. if hand else 300.);d.CreateDampingAttr(.8 if hand else 15.)
       d.CreateMaxForceAttr(float(joints[n].find('limit').get('effort'))*cfg.get('hand_force_scale',1.) if hand else float(joints[n].find('limit').get('effort')))
     if pr.HasAPI(UsdPhysics.RigidBodyAPI):
      rb=PhysxSchema.PhysxRigidBodyAPI.Apply(pr);rb.CreateSolverPositionIterationCountAttr(32);rb.CreateSolverVelocityIterationCountAttr(8);rb.CreateEnableCCDAttr(True)
    leaf=cfg.get('asset_leaf','4_3_0002');domain,cls,_=leaf.split('_');src=Path(cfg['assets_root'])/domain/cls/leaf
    asset=UsdGeom.Xform.Define(stage,'/World/Mug').GetPrim();asset.GetReferences().AddReference(str(src/(leaf+'.usd')))
    print('ASSET ROOT',asset.GetTypeName(),asset.GetAppliedSchemas(),flush=True)
    rbpaths=[str(pr.GetPath()) for pr in Usd.PrimRange(asset) if pr.HasAPI(UsdPhysics.RigidBodyAPI)]
    print('ASSET RBS',rbpaths,flush=True)
    primary_collisions=[pr for pr in Usd.PrimRange(asset) if pr.HasAPI(UsdPhysics.CollisionAPI)]
    if not (src/'_SUCCESS').exists() or len(rbpaths)!=1 or not primary_collisions:raise ValueError('Invalid released task asset')
    asset_source={'leaf':leaf,'release':'v1.3','source_usd_sha256':hashlib.sha256((src/(leaf+'.usd')).read_bytes()).hexdigest(),'source_mesh_sha256':hashlib.sha256((src/'mesh'/('m_'+leaf+'.obj')).read_bytes()).hexdigest(),'source_physics_sha256':hashlib.sha256((src/'part_physics.json').read_bytes()).hexdigest(),'collision_shapes':len(primary_collisions),'scale':1.0,'role':'task_target'}
    # Measure source geometry before placement, with no duplicate up-axis conversion.
    b=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']).ComputeWorldBound(asset).ComputeAlignedRange();bmin=np.array(b.GetMin());bmax=np.array(b.GetMax())
    print('BOUNDS',bmin,bmax,flush=True)
    xy=np.array(cfg.get('asset_xy',[.045,-.47]));asset_rotation=Rot.from_euler('xyz',cfg.get('asset_rpy',[0,0,cfg.get('asset_yaw',0.)]));asset_quat=asset_rotation.as_quat()[[3,0,1,2]]
    import itertools
    rotated_corners=np.array([asset_rotation.apply(v) for v in itertools.product(*zip(bmin,bmax))]);rotmin=rotated_corners.min(axis=0);rotmax=rotated_corners.max(axis=0)
    translation=np.r_[xy-(rotmin[:2]+rotmax[:2])/2,cfg.get('asset_bottom_z',.833)-rotmin[2]]
    UsdGeom.Xformable(asset).AddTranslateOp().Set(Gf.Vec3d(*map(float,translation)))
    UsdGeom.Xformable(asset).AddOrientOp().Set(Gf.Quatf(float(asset_quat[0]),Gf.Vec3f(*map(float,asset_quat[1:]))))
    for pr in Usd.PrimRange(asset):
     if pr.HasAPI(UsdPhysics.RigidBodyAPI):
      rr=PhysxSchema.PhysxRigidBodyAPI.Apply(pr);rr.CreateSolverPositionIterationCountAttr(32);rr.CreateSolverVelocityIterationCountAttr(8);rr.CreateEnableCCDAttr(True)
     if pr.HasAPI(UsdPhysics.CollisionAPI):
      co=PhysxSchema.PhysxCollisionAPI.Apply(pr);co.CreateContactOffsetAttr(.001);co.CreateRestOffsetAttr(0.)
    contacts=[];tick=0
    for rp in rbpaths+[v['path'] for v in scene_assets]:
     PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(rp)).CreateThresholdAttr(0.)
    from omni.physx import get_physx_simulation_interface
    def on_contact(headers,data):
     for h in headers:
      actors=[str(PhysicsSchemaTools.intToSdfPath(h.actor0)),str(PhysicsSchemaTools.intToSdfPath(h.actor1))]
      if not any('/World/Mug' in v for v in actors):continue
      for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
       contacts.append({'t':tick/cfg['physics_hz'],'actors':actors,'separation':float(d.separation),'impulse':list(d.impulse),'position':list(d.position)})
    subscription=get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    robot=world.scene.add(Articulation(prim_paths_expr=roots[0],name='g1'))
    obj=world.scene.add(RigidPrim(prim_paths_expr=rbpaths[0],name='mug'))
    scene_views={v['name']:world.scene.add(RigidPrim(prim_paths_expr=v['path'],name=v['name'])) for v in scene_assets}
    world.reset()
    names=robot.dof_names;print('DOFS',names,flush=True)
    q=np.zeros(len(names))
    ready={'left_shoulder_pitch_joint':-.35,'left_shoulder_roll_joint':.2,'left_elbow_joint':1.1,'right_shoulder_pitch_joint':-.35,'right_shoulder_roll_joint':-.16,'right_elbow_joint':1.1}
    for side in ['left','right']:
     ready.update({side+'_hip_pitch_joint':-.1,side+'_knee_joint':.2,side+'_ankle_pitch_joint':-.1})
    for n,v in ready.items():q[names.index(n)]=v
    initial=ik(np.array(cfg.get('initial_palm_position',np.array(cfg.get('palm_grasp',[-.055,-.51,.89]))+[-.10,0,.12])),Rot.from_euler('xyz',cfg.get('palm_rpy',[0,0,0])).as_matrix(),np.array([ready.get(n,0.) for n in arm]))
    for n,v in zip(arm,initial):q[names.index(n)]=v
    q[names.index('right_hand_thumb_1_joint')]=.5
    robot.set_joint_positions(q[None,:]);robot.set_joint_position_targets(q[None,:])
    for _ in range(round(.5*cfg['physics_hz'])):world.step(render=False)
    obj.set_world_poses(translation[None,:],asset_quat[None,:])
    obj.set_velocities(np.zeros((1,6)))
    for record in scene_assets:
     view=scene_views[record['name']];view.set_world_poses(np.array([record['initial_position']]),np.array([record['initial_quaternion']]));view.set_velocities(np.zeros((1,6)))
    contacts.clear()
    for _ in range(cfg['physics_hz']):world.step(render=False)
    contacts.clear()
    palmview=RigidPrim(prim_paths_expr='/World/G1/right_hand_palm_link',name='palm');palmview.initialize()
    print('PALM ACTUAL',palmview.get_world_poses(),flush=True)
    print('PALM FK',fk(dict(zip(names,q))).tolist(),flush=True)
    print('OBJECT',obj.get_world_poses(),flush=True)
    task_events=[]
    if cfg.get('interaction_task')=='recover_dispenser':
     pp,qq=obj.get_world_poses();up=Rot.from_quat(qq[0][[1,2,3,0]]).apply(cfg.get('upright_axis_local',[0,0,1]));angle=float(np.degrees(np.arccos(np.clip(up[2],-1,1))));task_events.append({'t':0.,'event':'fallen_detected','tilt_deg':angle,'threshold_deg':65.,'method':'Measured rigid pose before any task motion'})
     if angle<=65.:raise RuntimeError('Recovery motion rejected: asset is not fallen')
    print('LINKS',robot.body_names,flush=True)
    stage.GetRootLayer().Export(str(out/'world.local'),args={'format':'usda'})
    if a.probe:
     app.close();raise SystemExit
    # Use the settled asset pose to locate a grasp; only robot joint targets move afterward.
    if cfg.get('grasp_offset_world'):
     settled,_=obj.get_world_poses();grasp=settled[0]+np.array(cfg['grasp_offset_world']);delta=grasp-np.array(cfg['palm_grasp'])
     for pending in cfg.get('motion_phases',[]):
      if pending['phase'] in ('approach','descend','close','lift','hold'):pending['position']=(np.array(pending['position'])+delta).tolist()
     print('SETTLED_GRASP',grasp.tolist(),flush=True)
    # Expert pose and phase schedule, parameterized for contact calibration.
    R=Rot.from_euler('xyz',cfg.get('palm_rpy',[0,0,0])).as_matrix()
    palm=np.array(cfg.get('palm_grasp',[-.055,-.51,.89]))
    seed=np.array([ready.get(n,0.) for n in arm])
    poses=[palm+[-.10,0,.12],palm,palm+[0,0,.18],palm+np.array(cfg.get('transfer_offset',[0,.27,0]))+[0,0,.18],palm+np.array(cfg.get('transfer_offset',[0,.27,0]))+[0,0,cfg.get('place_height_offset',-.006)],palm+np.array(cfg.get('transfer_offset',[0,.27,0]))+np.array(cfg.get('retreat_offset',[-.12,0,.16]))]
    poses.insert(5,palm+np.array(cfg.get('transfer_offset',[0,.27,0]))+np.array(cfg.get('withdraw_offset',[-.12,0,.008])))
    solutions=[]
    custom=cfg.get('motion_phases')
    if custom:
     poses=[np.array(v['position']) for v in custom]
    for i,pos in enumerate(poses):
     orientation=Rot.from_euler('xyz',custom[i]['rpy']).as_matrix() if custom else R
     seed=ik(pos,orientation,seed);solutions.append(seed.copy())
    print('SOLUTIONS',[s.tolist() for s in solutions],flush=True)
    handopen=cfg.get('hand_open',{'right_hand_thumb_0_joint':0.,'right_hand_thumb_1_joint':.5,'right_hand_thumb_2_joint':0.})
    releaseopen=dict(handopen);releaseopen.update(cfg.get('release_open',{}))
    handclosed=cfg.get('hand_closed',{'right_hand_thumb_0_joint':0.,'right_hand_thumb_1_joint':-.8,'right_hand_thumb_2_joint':-1.1,'right_hand_middle_0_joint':.9,'right_hand_middle_1_joint':1.3,'right_hand_index_0_joint':.9,'right_hand_index_1_joint':1.3})
    for n in names:
     if n.startswith('right_hand_'):
      handopen.setdefault(n,0.);releaseopen.setdefault(n,0.)
    def target(sol,hand):
     t=q.copy()
     for n,v in zip(arm,sol):t[names.index(n)]=v
     for n,v in hand.items():t[names.index(n)]=v
     return t
    thumbclear=dict(handclosed);thumbclear['right_hand_thumb_0_joint']=cfg.get('thumb_clear_angle',1.)
    samples=[];phase_ranges=[]
    for record in scene_assets:
     pp,qq=scene_views[record['name']].get_world_poses();record['settled_position']=pp[0].tolist();record['settled_quaternion']=qq[0].tolist()
    camera=None
    if cfg.get('capture'):
     eye=np.array([1.,-2.3,1.75]);look=np.array([-.10,-.29,.82]);f=(look-eye);f/=np.linalg.norm(f);y=np.cross([0,0,1],f);y/=np.linalg.norm(y);z=np.cross(f,y);qt=Rot.from_matrix(np.column_stack([f,y,z])).as_quat()
     camera=Camera('/World/Camera',position=eye,orientation=qt[[3,0,1,2]],resolution=tuple(cfg['resolution']),frequency=cfg['fps']);camera.initialize();camera.set_focal_length(2.0)
     for _ in range(12):world.render()
     import cv2
     writer=cv2.VideoWriter(str(out/'rollout_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),cfg['fps'],tuple(cfg['resolution']))
    last=q.copy();tick=0
    cartesian_seed=solutions[4].copy()
    phases=[('approach',2,solutions[0],handopen),('descend',2,solutions[1],handopen),('close',2,solutions[1],handclosed),('lift',2,solutions[2],handclosed),('hold',1,solutions[2],handclosed),('transfer',2,solutions[3],handclosed),('lower',2,solutions[4],handclosed),('support_wait',1.5,solutions[4],handclosed),('thumb_clear',2,solutions[4],thumbclear),('release',3,solutions[4],releaseopen),('withdraw',2,solutions[5],releaseopen),('retreat',2,solutions[6],releaseopen),('settle',2,solutions[6],releaseopen)]
    if custom:
     phases=[(v['phase'],v['seconds'],sol,{'open':handopen,'closed':handclosed,'release':releaseopen,'thumb_clear':thumbclear}[v['hand']]) for v,sol in zip(custom,solutions)]
     previous_position=np.array(cfg['initial_palm_position']);previous_rotation=Rot.from_euler('xyz',cfg.get('palm_rpy',[0,0,0]));cartesian_seed=initial.copy()
    from scipy.spatial.transform import Slerp
    for phase_index,(phase,seconds,sol,hand) in enumerate(phases):
     if custom:
      destination=np.array(custom[phase_index]['position']);destination_rotation=Rot.from_euler('xyz',custom[phase_index]['rpy']);slerp=Slerp([0,1],Rot.from_quat([previous_rotation.as_quat(),destination_rotation.as_quat()]))
     end=target(sol,hand);count=round(seconds*cfg['physics_hz']);start=tick
     for k in range(count):
      u=(k+1)/count;u=u*u*(3-2*u);cmd=last+(end-last)*u
      if custom:
       if k%(cfg['physics_hz']//cfg['fps'])==0 or k==count-1:cartesian_seed=ik(previous_position+(destination-previous_position)*u,slerp(u).as_matrix(),cartesian_seed)
       for n,v in zip(arm,cartesian_seed):cmd[names.index(n)]=v
      elif phase in ('withdraw','retreat'):
       origin=poses[4] if phase=='withdraw' else poses[5]
       destination=poses[5] if phase=='withdraw' else poses[6]
       if k%(cfg['physics_hz']//cfg['fps'])==0 or k==count-1:
        cartesian_seed=ik(origin+(destination-origin)*u,R,cartesian_seed)
       for n,v in zip(arm,cartesian_seed):cmd[names.index(n)]=v
      robot.set_joint_position_targets(cmd[None,:])
      tick+=1;capture=tick%(cfg['physics_hz']//cfg['fps'])==0
      world.step(render=bool(camera and capture))
      if capture:
       position,orientation=obj.get_world_poses();actual=robot.get_joint_positions()[0]
       samples.append({'t':tick/cfg['physics_hz'],'phase':phase,'position':position[0].tolist(),'quaternion':orientation[0].tolist(),'joints':actual.tolist(),'target':cmd.tolist(),'scene_assets':{name:{'position':view.get_world_poses()[0][0].tolist(),'quaternion':view.get_world_poses()[1][0].tolist()} for name,view in scene_views.items()}})
       if camera:
        rgba=camera.get_rgba()
        if rgba is not None and rgba.size:writer.write(cv2.cvtColor(rgba[:,:,:3].astype(np.uint8),cv2.COLOR_RGB2BGR))
        if k==count-1:cv2.imwrite(str(out/(phase+'.png')),cv2.cvtColor(rgba[:,:,:3].astype(np.uint8),cv2.COLOR_RGB2BGR))
     if custom:previous_position=destination;previous_rotation=destination_rotation
     if custom and phase=='hold' and cfg.get('object_goal_pose') and samples[-1]['position'][2]-samples[0]['position'][2]>.035:
      pp,pq=palmview.get_world_poses();op,oq=obj.get_world_poses();palm_R=Rot.from_quat(pq[0][[1,2,3,0]]);obj_R=Rot.from_quat(oq[0][[1,2,3,0]]);relative_R=palm_R.inv()*obj_R;relative_p=palm_R.inv().apply(op[0]-pp[0]);goal=cfg['object_goal_pose'];goal_R=Rot.from_quat(np.array(goal['quaternion'])[[1,2,3,0]]);desired_R=goal_R*relative_R.inv();desired_p=np.array(goal['position'])-desired_R.apply(relative_p)
      (out/'hold_checkpoint.json').write_text(json.dumps({'palm_position':pp[0].tolist(),'palm_quaternion':pq[0].tolist(),'object_position':op[0].tolist(),'object_quaternion':oq[0].tolist(),'relative_position':relative_p.tolist(),'relative_quaternion_xyzw':relative_R.as_quat().tolist(),'arm_joint_names':arm,'arm_joints':[float(robot.get_joint_positions()[0][names.index(n)]) for n in arm],'samples':samples,'contacts':contacts},indent=2))
      if cfg.get('goal_yaw_candidates_deg'):
       chosen=False
       actual_q=robot.get_joint_positions()[0];hold_seed=np.array([actual_q[names.index(n)] for n in arm])
       for yaw in cfg['goal_yaw_candidates_deg']:
        candidate_R=Rot.from_euler('z',yaw,degrees=True)*goal_R*relative_R.inv();candidate_p=np.array(goal['position'])-candidate_R.apply(relative_p);candidate_above=candidate_p+[0,0,cfg.get('place_approach_height',.16)];candidate_seed=hold_seed.copy();rotation_path=Slerp([0,1],Rot.from_quat([palm_R.as_quat(),candidate_R.as_quat()]))
        try:
         for u in [.25,.5,.75,1.]:candidate_seed=ik(pp[0]+(candidate_above-pp[0])*u,rotation_path(u).as_matrix(),candidate_seed)
         for u in [.25,.5,.75,1.]:candidate_seed=ik(candidate_above+(candidate_p-candidate_above)*u,candidate_R.as_matrix(),candidate_seed)
         candidate_seed=ik(candidate_p+np.array(cfg.get('goal_withdraw_offset',[-.12,0,.10])),candidate_R.as_matrix(),candidate_seed)
        except RuntimeError:continue
        desired_R=candidate_R;desired_p=candidate_p;chosen=True;task_events.append({'t':tick/cfg['physics_hz'],'event':'goal_axial_yaw_selected','yaw_deg':yaw});print('GOAL_YAW_SELECTED',yaw,flush=True);break
       if not chosen:raise RuntimeError('No joint-limit-bounded IK path for axial target candidates')
      print('GRASP_CALIBRATED_GOAL',desired_p.tolist(),desired_R.as_euler('xyz').tolist(),flush=True)
      for pending in custom[phase_index+1:]:
       pending['rpy']=desired_R.as_euler('xyz').tolist();pending['position']=desired_p.tolist()
       if pending['phase']=='release':pending['position']=(desired_p+np.array(cfg.get('release_retract_offset',[0,0,0]))).tolist()
       if pending['phase']=='transfer':pending['position']=(desired_p+[0,0,cfg.get('place_approach_height',.16)]).tolist()
       if pending['phase'] in ('withdraw','retreat','settle'):pending['position']=(desired_p+np.array(cfg.get('goal_withdraw_offset',[-.12,0,.10]))).tolist()
     last=end;phase_ranges.append([phase,start/cfg['physics_hz'],tick/cfg['physics_hz']]);print('PHASE',phase,samples[-1]['position'],flush=True)
    if camera:writer.release()
    (out/'expert_waypoints.json').write_text(json.dumps({'arm_joint_names':arm,'poses_m':[p.tolist() for p in poses],'joint_solutions':[s.tolist() for s in solutions],'hand_open':handopen,'hand_closed':handclosed,'release_open':releaseopen}))
    report={'simulator':'Isaac Sim 4.5 / PhysX','physics_hz':cfg['physics_hz'],'fps':cfg['fps'],'fixed_base':True,'tray_xy':cfg.get('tray_xy',[.015,-.22]),'asset_leaf':leaf,'asset_source':asset_source,'asset_mass_kg':json.loads((src/'part_physics.json').read_text())['total_mass'],'joint_names':names,'phases':phase_ranges,'samples':samples,'contacts':contacts,'asset_bounds_local_m':[bmin.tolist(),bmax.tolist()],'initial_asset_yaw_rad':cfg.get('asset_yaw',0.),'asset_upright_axis_local':asset_rotation.inv().apply([0,0,1]).tolist(),'tray_size_m':cfg.get('tray_size_m',[.27,.24,.014]),'tray_top_m':cfg.get('tray_top_m',.844),'scene_assets':scene_assets,'task':cfg.get('task','Place the mug upright on the tray and withdraw the hand'),'scene_id':cfg.get('scene_id','upright_mug'),'task_events':task_events,'interaction_task':cfg.get('interaction_task'),'upright_axis_local':cfg.get('upright_axis_local',[0,0,1]),'motion_control':'joint_targets_only','torso_mechanically_locked':bool(cfg.get('lock_torso')),'status':'unvalidated'}
    (out/'trajectory.json').write_text(json.dumps(report))
    (out/'run_config.local').write_text(json.dumps(input_cfg,indent=2))
    (out/'executed_motion_phases.json').write_text(json.dumps(cfg.get('motion_phases',[]),indent=2))
    app.close()


if __name__ == "__main__":
    main()
