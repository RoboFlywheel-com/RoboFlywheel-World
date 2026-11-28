# SPDX-License-Identifier: Apache-2.0
"""Reference v1.3 rigid USDs without replacing their collision or mass data."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path


def add_release_asset(stage, assets_root, spec):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    leaf=spec['leaf'];domain,category,_=leaf.split('_');src=Path(assets_root)/domain/category/leaf
    usd=src/(leaf+'.usd');mesh=src/'mesh'/('m_'+leaf+'.obj');metadata=src/'part_physics.json'
    if not all(p.is_file() for p in (usd,mesh,metadata,src/'_SUCCESS')):
        raise ValueError('Incomplete release leaf: '+leaf)
    physics=json.loads(metadata.read_text())
    if physics['total_mass']<=0:raise ValueError('Invalid source mass: '+leaf)
    path='/World/'+spec['name'];root=UsdGeom.Xform.Define(stage,path).GetPrim();root.GetReferences().AddReference(str(usd))
    bodies=[p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    collisions=[p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
    if len(bodies)!=1 or bodies[0]!=root or not collisions:raise ValueError('Expected one root rigid body and release collisions: '+leaf)
    bound=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']).ComputeWorldBound(root).ComputeAlignedRange()
    low,high=np.array(bound.GetMin()),np.array(bound.GetMax())
    rotation=Rotation.from_euler('xyz',spec.get('rpy',[0,0,0]));quat=rotation.as_quat()[[3,0,1,2]]
    pts=rotation.apply(np.array(list(itertools.product(*zip(low,high)))))
    lo,hi=pts.min(axis=0),pts.max(axis=0)
    position=np.r_[np.array(spec['xy'])-(lo[:2]+hi[:2])/2,spec.get('bottom_z',.833)-lo[2]]
    x=UsdGeom.Xformable(root);x.AddTranslateOp().Set(Gf.Vec3d(*map(float,position)));x.AddOrientOp().Set(Gf.Quatf(float(quat[0]),Gf.Vec3f(*map(float,quat[1:]))))
    rb=PhysxSchema.PhysxRigidBodyAPI.Apply(root);rb.CreateSolverPositionIterationCountAttr(32);rb.CreateSolverVelocityIterationCountAttr(8);rb.CreateEnableCCDAttr(True)
    for prim in collisions:
        co=PhysxSchema.PhysxCollisionAPI.Apply(prim);co.CreateContactOffsetAttr(.001);co.CreateRestOffsetAttr(0.)
    return {'name':spec['name'],'leaf':leaf,'path':path,'release':'v1.3','role':spec.get('role','context'),'source_usd_sha256':hashlib.sha256(usd.read_bytes()).hexdigest(),'source_mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),'source_physics_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),'collision_shapes':len(collisions),'mass_kg':physics['total_mass'],'friction':physics['whole_mu1'],'bounds_local_m':[low.tolist(),high.tolist()],'initial_position':position.tolist(),'initial_quaternion':quat.tolist(),'coordinate_preparation':'Whole USD rigid-body root rotated; visual, collision and inertia frame preserved','scale':1.0}
