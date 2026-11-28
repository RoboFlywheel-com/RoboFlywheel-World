# SPDX-License-Identifier: Apache-2.0
"""Independent prepared SimReady frame and physical provenance checks."""
import math
from rfw_w.scenes.contracts.surface_appearance import require


def verify_rigid_metadata(m):
    require(m.get('provider')=='simready' and m.get('library_version')=='v1.3'
            and m.get('origin_convention')=='base' and m.get('frame')=='Z-up'
            and m.get('collision_proxy')=='single_convex_hull','RIGID_FRAME_INVALID','frame/provider')
    lo=m.get('bounds_min_m');hi=m.get('bounds_max_m');dims=m.get('dimensions_m')
    require(all(isinstance(x,list) and len(x)==3 for x in (lo,hi,dims))
            and all(type(v) in (int,float) and math.isfinite(v) for a in (lo,hi,dims) for v in a),
            'RIGID_FRAME_INVALID','bounds')
    require(abs(lo[2])<1e-5 and abs(lo[0]+hi[0])<1e-5 and abs(lo[1]+hi[1])<1e-5
            and all(d>0 and abs(hi[k]-lo[k]-d)<1e-5 for k,d in enumerate(dims)),
            'RIGID_FRAME_INVALID','base origin')
    p=m.get('physics',{});mass=p.get('mass_kg')
    require(type(mass) in (int,float) and math.isfinite(mass) and mass>0
            and p.get('source')=='catalogue_declared:part_physics','RIGID_PHYSICS_UNCERTIFIED','mass source')
    for k in ('friction_static','friction_dynamic'):
        v=p.get(k);require(v is None or (type(v) in (int,float) and math.isfinite(v) and v>=0),
                          'RIGID_PHYSICS_UNCERTIFIED',k)


def verify_prepared_rigid(root, metadata):
    from rfw_w.scenes.geometry.surface_appearance import checked_path,digest
    verify_rigid_metadata(metadata)
    import trimesh
    for kind in ('visual','collision'):
        path=checked_path(root,metadata[kind+'_ref'])
        require(digest(path)==metadata[kind+'_sha256'],'ASSET_LEDGER_MISMATCH',kind)
        mesh=trimesh.load(str(path),force='mesh',process=False,skip_materials=True)
        lo=mesh.bounds[0].tolist();hi=mesh.bounds[1].tolist()
        require(all(abs(a-b)<2e-5 for a,b in zip(lo,metadata.get(kind+'_bounds_min_m',metadata['bounds_min_m'])))
                and all(abs(a-b)<2e-5 for a,b in zip(hi,metadata.get(kind+'_bounds_max_m',metadata['bounds_max_m']))),
                'RIGID_FRAME_INVALID','measured '+kind)
    require(abs(metadata.get('collision_bounds_min_m',metadata['bounds_min_m'])[2])<2e-5,
            'RIGID_FRAME_INVALID','collision below base')
    return {'verified':True,'grasp':'not_run','dynamics':'not_run'}


def verify_task_shape(category, dimensions):
    """Necessary plate profile only; does not certify vessel function or graspability."""
    if category == 'plate':
        require(dimensions[2] <= min(dimensions[:2]) * .25,
                'RIGID_TASK_SHAPE_UNCERTIFIED', 'plate is not a thin horizontal profile')
