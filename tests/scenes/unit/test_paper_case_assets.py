# SPDX-License-Identifier: Apache-2.0
"""The paper audit must reject missing assets, changed cores and invented evidence."""
import copy
import importlib
from rfw_w.scenes.resources import module_path, data_path
from pathlib import Path
import sys
import pytest

path=module_path('rfw_w.scenes.validation.check_paper_case_assets')
audit = importlib.import_module('rfw_w.scenes.validation.check_paper_case_assets');


def fixture():
    config={'targets':['cup','microwave','cushion'],'supports':['desk'],'robot':{'id':'robot'}}
    ids=['cup','microwave','cushion','desk','robot']
    inventory={'assets':[{'id':id} for id in ids],'retrieval_calls':0}
    frozen={'members':[{'id':id,'asset_digest':'source'} for id in ids]};frozen['case_core_hash']=audit.content_hash(frozen)
    placements={'instances':[{'id':id,'asset_digest':'source'} for id in ids]}
    provenance={'external_calls':0,'captures':[{'id':'core','case_core_hash':frozen['case_core_hash']}],'not_run':{'physics':'static'}}
    return config,inventory,frozen,placements,provenance


def test_valid_records():
    assert audit.audit_records(*fixture())==[]


def test_missing_target():
    c,i,f,p,v=fixture();i['assets']=i['assets'][1:]
    assert audit.audit_records(c,i,f,p,v)==['PAPER_ASSET_MISSING(cup)']


def test_changed_core():
    c,i,f,p,v=fixture();f['members'][0]['pose']=[1,0,0]
    assert audit.audit_records(c,i,f,p,v)==['PAPER_CORE_DRIFT(frozen,hash)']


def test_wrong_evidence():
    c,i,f,p,v=fixture();v['captures'][0]['case_core_hash']='wrong'
    assert audit.audit_records(c,i,f,p,v)==['PAPER_EVIDENCE_SCENE_MISMATCH(core)']


def test_wrong_source():
    c,i,f,p,v=fixture();p['instances'][0]['asset_digest']='different texture source'
    assert audit.audit_records(c,i,f,p,v)==['PAPER_CORE_DRIFT(scene,cup)']


def test_invented_physics_pass():
    c,i,f,p,v=fixture();v['physics_passed']=True
    assert audit.audit_records(c,i,f,p,v)==['PAPER_UNSUPPORTED_CLAIM(physics)']


def test_missing_scene_member():
    c,i,f,p,v=fixture();p['instances']=p['instances'][1:]
    assert audit.audit_records(c,i,f,p,v)==['PAPER_ASSET_NOT_IN_SCENE(cup)']


def test_missing_texture(tmp_path):
    row={'id':'cup','package_files':{'textures/albedo.png':'some_hash'}}
    assert audit.check_source_files(tmp_path,row)==['PAPER_DEPENDENCY_MISSING(cup,textures/albedo.png)']


def test_replaced_texture(tmp_path):
    p=tmp_path/'albedo.png';p.write_bytes(b'original');row={'id':'cup','package_files':{'albedo.png':audit.digest(p)}}
    p.write_bytes(b'replacement')
    assert audit.check_source_files(tmp_path,row)==['PAPER_CORE_DRIFT(source,cup)']


def test_static_joint_pose_uses_joint_frames():
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    from rfw_w.scenes.visualization.paper_case_common import static_joint_pose
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World');UsdGeom.Xform.Define(stage,'/World/base');link=UsdGeom.Xform.Define(stage,'/World/link')
    joint=UsdPhysics.RevoluteJoint.Define(stage,'/World/joint');joint.CreateBody0Rel().SetTargets(['/World/base']);joint.CreateBody1Rel().SetTargets(['/World/link']);joint.CreateAxisAttr('Z');joint.CreateLowerLimitAttr(-180);joint.CreateUpperLimitAttr(180);joint.CreateLocalPos0Attr(Gf.Vec3f(0,0,1))
    original=stage.GetRootLayer().ExportToString();static_joint_pose(stage,{'joint':90})
    point=link.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).Transform(Gf.Vec3d(1,0,0))
    assert tuple(point)==pytest.approx((0,1,1))
    assert stage.GetRootLayer().ExportToString()==original


def test_static_joint_pose_rejects_outside_limits():
    from pxr import Usd,UsdGeom,UsdPhysics
    from rfw_w.scenes.visualization.paper_case_common import static_joint_pose
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World');UsdGeom.Xform.Define(stage,'/World/base');UsdGeom.Xform.Define(stage,'/World/link')
    joint=UsdPhysics.RevoluteJoint.Define(stage,'/World/joint');joint.CreateBody0Rel().SetTargets(['/World/base']);joint.CreateBody1Rel().SetTargets(['/World/link']);joint.CreateAxisAttr('Z');joint.CreateLowerLimitAttr(0);joint.CreateUpperLimitAttr(30)
    with pytest.raises(ValueError,match='PAPER_LAYOUT_CONFLICT'):
        static_joint_pose(stage,{'joint':40})


def workstation():
    return {'workbench':[[-1.2,-.6,0],[1.2,.6,.78]],'chair':[[.37,-1.31,0],[.93,-.77,.82]],'mount_plate':[[-.97,.15,.78],[-.63,.49,.8]],'franka':[[-1.03,.19,.8],[-.35,.45,1.68]]}


def test_workstation_contact_and_chair_distance():
    gaps,metrics=audit.check_workstation_layout(workstation(),180)
    assert gaps==[]
    assert metrics['chair_to_desk_gap_m']==pytest.approx(.17)


@pytest.mark.parametrize('change,expected',[
    ('far_chair','PAPER_LAYOUT_CONFLICT(chair,desk_gap)'),
    ('reversed_chair','PAPER_LAYOUT_CONFLICT(chair,facing)'),
    ('floating_robot','PAPER_LAYOUT_CONFLICT(franka,mount_contact)'),
    ('floating_plate','PAPER_LAYOUT_CONFLICT(mount_plate,contact)'),
    ('plate_over_edge','PAPER_LAYOUT_CONFLICT(mount_plate,desk_extent)'),
])
def test_workstation_rejects_bad_placement(change,expected):
    world=workstation();yaw=180
    if change=='far_chair':world['chair'][1][1]-=1
    if change=='reversed_chair':yaw=0
    if change=='floating_robot':world['franka'][0][2]+=.1
    if change=='floating_plate':world['mount_plate'][0][2]+=.1
    if change=='plate_over_edge':world['mount_plate'][0][0]=-1.3
    assert expected in audit.check_workstation_layout(world,yaw)[0]


def test_operating_face_points_to_robot():
    assert audit.check_operating_face((-1,0),(.22,.26),(-.8,.32))[0]==[]
    assert audit.check_operating_face((0,-1),(.22,.26),(-.8,.32))[0]
    assert audit.check_operating_face((1,0),(.22,.26),(-.8,.32))[0]


def test_catalog_soft_visual_needs_source_evidence_and_no_fake_dynamics():
    proof={'kind':'softsim_catalog_visual','catalog_record':{'physics_type':'deformable'},'source_sha256':'mesh','physics_json_sha256':'physics','dynamic_validation':{'status':'not_run'}}
    assert audit.check_deformable_source(proof)
    del proof['physics_json_sha256']
    assert not audit.check_deformable_source(proof)
    proof['physics_json_sha256']='physics';proof['dynamic_validation']['status']='passed'
    assert not audit.check_deformable_source(proof)


def test_door_approach_rejects_objects_in_reserved_gap():
    region=[[-.34,-.05],[-.12,-.05],[-.12,.55],[-.34,.55]]
    assert audit.check_door_approach({'cup':[[-.95,-.36,.78],[-.85,-.26,.9]]},region)==[]
    assert audit.check_door_approach({'cup':[[-.3,0,.78],[-.2,.1,.9]]},region)==['PAPER_LAYOUT_CONFLICT(cup,door_approach)']


def test_table_support_rejects_floating_and_overhang():
    desk=workstation()['workbench']
    assert audit.on_table([[0,0,.78],[.1,.1,.9]],desk)
    assert not audit.on_table([[0,0,.88],[.1,.1,1]],desk)
    assert not audit.on_table([[1.1,0,.78],[1.3,.1,.9]],desk)


def test_texture_coordinates_require_reader_connection():
    from pxr import Usd,UsdShade,Sdf
    stage=Usd.Stage.CreateInMemory();texture=UsdShade.Shader.Define(stage,'/Texture');texture.CreateIdAttr('UsdUVTexture')
    assert audit.check_texture_coordinates(stage)
    reader=UsdShade.Shader.Define(stage,'/Reader');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    assert audit.check_texture_coordinates(stage)==[]


def test_dual_robot_membership_requires_both_arms():
    c,i,f,p,v=fixture();c['robot']['instances']=[{'id':'robot'},{'id':'second_arm'}]
    assert 'PAPER_ASSET_MISSING(second_arm)' in audit.audit_records(c,i,f,p,v)


def test_dual_mount_rejects_mid_table_and_floating_base():
    measured={'workbench':[[-.9,-.45,0],[.9,.45,.78]],'mount':[[-.7,-.45,.78],[-.5,-.27,.8]]}
    placements=[{'id':'arm','translation':[-.6,-.36,.8]}];robots=[{'id':'arm','support':'mount'}]
    assert audit.check_dual_mounts(measured,placements,robots)==[]
    measured['mount'][0][1]-=2e-8
    assert audit.check_dual_mounts(measured,placements,robots)==[]
    measured['mount'][0][1]=-.3
    assert 'PAPER_LAYOUT_CONFLICT(arm,long_edge_mount)' in audit.check_dual_mounts(measured,placements,robots)
    placements[0]['translation'][2]=.9
    assert 'PAPER_LAYOUT_CONFLICT(arm,mount)' in audit.check_dual_mounts(measured,placements,robots)
