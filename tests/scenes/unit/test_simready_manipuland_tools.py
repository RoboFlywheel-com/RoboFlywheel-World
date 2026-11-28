# SPDX-License-Identifier: Apache-2.0
"""Rigid origin/physics ruler must reject fabricated metadata."""
import pytest
from rfw_w.scenes.contracts.surface_appearance import SurfaceAppearanceError
from rfw_w.scenes.geometry.rigid_tool_assets import verify_rigid_metadata


def metadata():
    return {'provider':'simready','library_version':'v1.3','asset_id':'4_3_0000',
            'origin_convention':'base','bounds_min_m':[-.05,-.04,0],'bounds_max_m':[.05,.04,.1],
            'dimensions_m':[.1,.08,.1],'frame':'Z-up','collision_proxy':'single_convex_hull',
            'physics':{'mass_kg':.2,'source':'catalogue_declared:part_physics','friction_static':None,'friction_dynamic':None}}


def test_rigid_metadata_accepts_declared_mass_without_grasp_claim():verify_rigid_metadata(metadata())
@pytest.mark.parametrize('field,value,code',[
 ('provider','hssd','RIGID_FRAME_INVALID'),('origin_convention','center','RIGID_FRAME_INVALID'),
 ('bounds_min_m',[-.05,-.04,-.02],'RIGID_FRAME_INVALID'),('frame','Y-up','RIGID_FRAME_INVALID'),
 ('physics',{'mass_kg':0,'source':'catalogue_declared:part_physics'},'RIGID_PHYSICS_UNCERTIFIED')])
def test_false_rigid_metadata_refused(field,value,code):
    m=metadata();m[field]=value
    with pytest.raises(SurfaceAppearanceError,match=code):verify_rigid_metadata(m)


def test_collision_proxy_cannot_extend_below_visual_base(tmp_path):
    import trimesh
    import hashlib
    from rfw_w.scenes.geometry.rigid_tool_assets import verify_prepared_rigid
    m=metadata()
    for kind,z in [('visual',.05),('collision',.04)]:
        mesh=trimesh.creation.box(extents=[.1,.08,.1]);mesh.apply_translation([0,0,z]);p=tmp_path/(kind+'.obj');mesh.export(p)
        m[kind+'_ref']=p.name;m[kind+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        m[kind+'_bounds_min_m']=mesh.bounds[0].tolist();m[kind+'_bounds_max_m']=mesh.bounds[1].tolist()
    with pytest.raises(SurfaceAppearanceError,match='collision below base'):verify_prepared_rigid(tmp_path,m)


def test_decorative_plate_profile_is_rejected():
    from rfw_w.scenes.geometry.rigid_tool_assets import verify_task_shape
    verify_task_shape('plate',[.22,.22,.025])
    with pytest.raises(SurfaceAppearanceError,match='RIGID_TASK_SHAPE_UNCERTIFIED'):
        verify_task_shape('plate',[.14,.12,.15])
