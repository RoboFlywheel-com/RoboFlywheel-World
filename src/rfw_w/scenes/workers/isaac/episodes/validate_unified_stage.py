# SPDX-License-Identifier: Apache-2.0
"""Independently compare exported USD animation with measured input transforms."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Sdf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root):
    provenance=json.loads((root/'stage_provenance.json').read_text())
    physics=json.loads((root/'physics_validation.json').read_text())
    stage=Usd.Stage.Open(str(root/'scene.usdc'))
    checks={'physics_passed':physics['status']=='passed' and bool(physics['checks']) and all(physics['checks'].values()),
        'trajectory_hash_matches':digest(root/'trajectory.json')==provenance['trajectory_sha256'],
        'fps_30':stage.GetTimeCodesPerSecond()==30,
        'complete_animation':int(stage.GetEndTimeCode()-stage.GetStartTimeCode()+1)==provenance['frames']}
    missing=[]
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName()==Sdf.ValueTypeNames.Asset and attr.Get():
                asset=attr.Get()
                if not asset.resolvedPath or not Path(asset.resolvedPath).is_file():missing.append(asset.path)
    checks['all_assets_resolve']=not missing
    expected=np.load(root/'measured_transforms.npz')
    names=expected['names'];matrices=expected['matrices'];mapped={}
    for prim in stage.Traverse():
        attr=prim.GetAttribute('userProperties:blender:object_name')
        if attr and attr.Get() in names:mapped[attr.Get()]=prim
    if provenance.get('transform_source')=='rigid_geometries':
        mapped={str(n):stage.GetPrimAtPath('/World/Mesh_Xform_'+str(n)) for n in names}
    checks['all_moving_objects_present']=all(str(n) in mapped and mapped[str(n)].IsValid() for n in names)
    maximum=0.
    if checks['all_moving_objects_present']:
        for frame in range(provenance['frames']):
            cache=UsdGeom.XformCache(frame+provenance.get('time_code_start',1))
            for index,name in enumerate(names):
                actual=np.asarray(cache.GetLocalToWorldTransform(mapped[str(name)]))
                maximum=max(maximum,float(np.max(np.abs(actual-matrices[frame,index]))))
    checks['all_frame_transforms_match']=checks['all_moving_objects_present'] and maximum<2e-5
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,frames_checked=provenance['frames'],
        moving_objects=len(names),max_matrix_error=maximum,missing_assets=missing,
        trajectory_sha256=provenance['trajectory_sha256'],scene_sha256=digest(root/'scene.usdc'),
        scope='Measured animation export consistency; source physics is independently recorded.')
    (root/'stage_validation.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',type=Path);args=parser.parse_args()
    result=validate(args.stage);print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(1)
