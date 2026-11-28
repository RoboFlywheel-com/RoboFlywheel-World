# SPDX-License-Identifier: Apache-2.0
"""Small, explicit utilities shared by the standalone SPEC-012 figure entry points."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path





def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fresh(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError('PAPER_OUTPUT_EXISTS(use a fresh directory)')
    path.mkdir(parents=True, exist_ok=True)
    return path


def bounds(stage, prim=None):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False)
    box = cache.ComputeWorldBound(prim or stage.GetDefaultPrim()).ComputeAlignedRange()
    return [[float(x) for x in box.GetMin()], [float(x) for x in box.GetMax()]]


def file_inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def static_joint_pose(stage, degrees):
    """Author a static FK pose from USD joint frames; no IK, controller or dynamics."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage.SetEditTarget(stage.GetSessionLayer())
    joints=[UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    rest_relative={}
    for joint in joints:
        a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
        if a and b:
            pa=UsdGeom.Xformable(stage.GetPrimAtPath(a[0])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pb=UsdGeom.Xformable(stage.GetPrimAtPath(b[0])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            rest_relative[str(joint.GetPath())]=pb*pa.GetInverse()
    def frame(position,rotation):
        matrix=Gf.Matrix4d(1);matrix.SetRotate(Gf.Quatd(rotation));matrix.SetTranslateOnly(Gf.Vec3d(*position));return matrix
    for joint in joints:
        bodies0=joint.GetBody0Rel().GetTargets();bodies1=joint.GetBody1Rel().GetTargets()
        if not bodies0 or not bodies1:continue
        parent=stage.GetPrimAtPath(bodies0[0]);child=stage.GetPrimAtPath(bodies1[0]);prim=joint.GetPrim()
        q=degrees.get(prim.GetName(),0.)
        rotation=Gf.Matrix4d(1)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j=UsdPhysics.RevoluteJoint(prim);lo=j.GetLowerLimitAttr().Get();hi=j.GetUpperLimitAttr().Get()
            if q<lo-1e-4 or q>hi+1e-4:raise ValueError('PAPER_LAYOUT_CONFLICT(joint limit,'+prim.GetName()+')')
            axis={'X':Gf.Vec3d(1,0,0),'Y':Gf.Vec3d(0,1,0),'Z':Gf.Vec3d(0,0,1)}[j.GetAxisAttr().Get()]
            rotation.SetRotate(Gf.Rotation(axis,q))
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            # Preserve the source opening relative to the moved hand.
            pass
        parent_world=UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        j0=frame(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get());j1=frame(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get())
        child_world=(rest_relative[str(joint.GetPath())]*parent_world if prim.IsA(UsdPhysics.PrismaticJoint) else j1.GetInverse()*rotation*j0*parent_world)
        root_world=UsdGeom.Xformable(child.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        x=UsdGeom.Xformable(child);x.ClearXformOpOrder();x.AddTransformOp(opSuffix='paperPose').Set(child_world*root_world.GetInverse())
