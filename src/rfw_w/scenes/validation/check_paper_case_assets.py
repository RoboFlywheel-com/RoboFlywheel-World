#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent SPEC-012 source, membership, pose, image and evidence audit."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from rfw_w.scenes.visualization.paper_case_common import content_hash,digest,fresh,read,write


def audit_records(config,inventory,frozen,placements,provenance):
    gaps=[];rows={a['id']:a for a in inventory['assets']};members={a['id']:a for a in frozen['members']}
    required=set(config['targets']+config['supports']+[config['robot']['id']]+[r['id'] for r in config['robot'].get('instances',[])])
    for id in sorted(required):
        if id not in rows or id not in members:gaps.append(f'PAPER_ASSET_MISSING({id})')
    canonical=copy.deepcopy(frozen);hash_value=canonical.pop('case_core_hash')
    if content_hash(canonical)!=hash_value:gaps.append('PAPER_CORE_DRIFT(frozen,hash)')
    poses={a['id']:a for a in placements['instances']}
    for id,member in members.items():
        if id not in poses:gaps.append(f'PAPER_ASSET_NOT_IN_SCENE({id})')
        elif member['asset_digest']!=poses[id]['asset_digest']:gaps.append(f'PAPER_CORE_DRIFT(scene,{id})')
    for capture in provenance.get('captures',[]):
        if capture['case_core_hash']!=hash_value:gaps.append(f'PAPER_EVIDENCE_SCENE_MISMATCH({capture["id"]})')
    for key,value in provenance.get('not_run',{}).items():
        if not isinstance(value,str) or not value.strip():gaps.append(f'PAPER_UNSUPPORTED_CLAIM({key})')
    if provenance.get('physics_passed') or provenance.get('task_verified'):gaps.append('PAPER_UNSUPPORTED_CLAIM(physics)')
    if inventory.get('retrieval_calls')!=0 or provenance.get('external_calls')!=0:gaps.append('PAPER_UNSUPPORTED_CLAIM(curated provenance)')
    return gaps


def check_source_files(root, row):
    gaps=[]
    for rel,sha in row['package_files'].items():
        p=Path(root)/rel
        if not p.is_file():gaps.append(f'PAPER_DEPENDENCY_MISSING({row["id"]},{rel})')
        elif digest(p)!=sha:gaps.append(f'PAPER_CORE_DRIFT(source,{row["id"]})')
    return gaps


def check_dual_mounts(measured, placements, robots):
    gaps=[];desk=measured['workbench']
    for robot in robots:
        plate=measured[robot['support']];base=next(r['translation'] for r in placements if r['id']==robot['id'])
        if not on_table(plate,desk) or abs(base[2]-plate[1][2])>.003:gaps.append('PAPER_LAYOUT_CONFLICT('+robot['id']+',mount)')
        if abs(plate[0][1]-desk[0][1])>.025:gaps.append('PAPER_LAYOUT_CONFLICT('+robot['id']+',long_edge_mount)')
    return gaps


def check_texture_coordinates(stage):
    from pxr import UsdShade
    gaps=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            shader=UsdShade.Shader(prim)
            if shader.GetIdAttr().Get()=='UsdUVTexture' and (not shader.GetInput('st') or not shader.GetInput('st').HasConnectedSource()):
                gaps.append('PAPER_TEXTURE_COORDINATES_MISSING('+str(prim.GetPath())+')')
    return gaps


def on_table(bounds, desk):
    lo,hi=bounds
    # USD float mesh bounds differed by 1.55e-8 m at a flush table edge.
    return abs(lo[2]-desk[1][2])<=.003 and all(lo[k]>=desk[0][k]-1e-6 and hi[k]<=desk[1][k]+1e-6 for k in (0,1))


def check_deformable_source(proof):
    if proof.get('tet_mesh.json',{}).get('num_elements',0)>0:return True
    record=proof.get('catalog_record',{})
    return (proof.get('kind')=='softsim_catalog_visual' and record.get('physics_type')=='deformable'
            and bool(proof.get('source_sha256')) and bool(proof.get('physics_json_sha256'))
            and proof.get('dynamic_validation',{}).get('status')=='not_run')


def check_door_approach(world, polygon):
    from shapely.geometry import Polygon,box
    area=Polygon(polygon);gaps=[]
    for id,(lo,hi) in world.items():
        if id in ('workbench','microwave'):continue
        if area.intersection(box(lo[0],lo[1],hi[0],hi[1])).area>1e-6:gaps.append(f'PAPER_LAYOUT_CONFLICT({id},door_approach)')
    return gaps


def check_operating_face(normal, position, robot_position):
    """Horizontal face must point toward the robot base within 30 degrees."""
    import math
    delta=[robot_position[i]-position[i] for i in (0,1)];length=math.hypot(*delta)*math.hypot(*normal)
    cosine=sum(normal[i]*delta[i] for i in (0,1))/length if length else -1
    return ([] if cosine>=math.cos(math.radians(30)) else ['PAPER_LAYOUT_CONFLICT(microwave,operating_face)']),cosine


def check_workstation_layout(world, chair_yaw):
    """Measure static contact and desk proximity; this is not a reachability test."""
    gaps=[];desk=world['workbench'];chair=world['chair'];plate=world['mount_plate'];robot=world['franka']
    distance=desk[0][1]-chair[1][1]
    if not .10<=distance<=.25:gaps.append('PAPER_LAYOUT_CONFLICT(chair,desk_gap)')
    if abs((chair_yaw-180+180)%360-180)>1e-4:gaps.append('PAPER_LAYOUT_CONFLICT(chair,facing)')
    if abs(plate[0][2]-desk[1][2])>.003:gaps.append('PAPER_LAYOUT_CONFLICT(mount_plate,contact)')
    if abs(robot[0][2]-plate[1][2])>.003:gaps.append('PAPER_LAYOUT_CONFLICT(franka,mount_contact)')
    if any(plate[0][k]<desk[0][k] or plate[1][k]>desk[1][k] for k in (0,1)):gaps.append('PAPER_LAYOUT_CONFLICT(mount_plate,desk_extent)')
    return gaps,{'chair_to_desk_gap_m':distance,'chair_yaw_deg':chair_yaw,'plate_to_desk_z_gap_m':plate[0][2]-desk[1][2],'robot_to_plate_z_gap_m':robot[0][2]-plate[1][2]}


def audit(bundle,check_relocation=False):
    from PIL import Image
    from pxr import Usd,UsdGeom,UsdUtils
    bundle=Path(bundle).resolve();config=read(bundle/'case_config.json');inventory=read(bundle/'asset_inventory.json');frozen=read(bundle/'frozen_core.json');placements=read(bundle/'placements.json');provenance=read(bundle/'provenance.json')
    gaps=audit_records(config,inventory,frozen,placements,provenance);scene=bundle/'scene'/'case.usda';scene_digest=digest(scene)
    for row in inventory['assets']:
        proof=row.get('type_evidence',{})
        if row['source_physics_type']=='articulated' and not proof.get('joints'):gaps.append(f'PAPER_SOURCE_TYPE_UNPROVEN({row["id"]},joints)')
        if row['source_physics_type']=='deformable' and not check_deformable_source(proof):gaps.append(f'PAPER_SOURCE_TYPE_UNPROVEN({row["id"]},tetrahedra)')
        root=bundle/'scene'/'sources'/row['id']
        gaps.extend(check_source_files(root,row))
        if row.get('source_kind') in ('prepared_obj','simready_obj','softsim_catalog'):
            gaps.extend(check_texture_coordinates(Usd.Stage.Open(str(root/'asset.usda'))))
    for p in (bundle/'scene').rglob('*.usda'):
        if any(token in p.read_text() for token in ['/home/','/data/','/tmp/']):gaps.append(f'PAPER_PRIVATE_PATH_LEAK({p.name})')
    for rel,sha in read(bundle/'artifact_hashes.json').items():
        p=bundle/'assets'/rel
        if not p.is_file() or digest(p)!=sha:gaps.append(f'PAPER_RENDER_MISSING({rel},digest)')
    names=['core_iso','shell_clay_iso','delivery_iso','isaac_view']+[f'material_{i:02d}' for i in range(1,4)]+[f'periphery_{i:02d}' for i in range(1,4)]
    captures={c['id']:c for c in provenance['captures']}
    for name in names:
        p=bundle/'assets'/(name+'.png')
        if not p.is_file() or name not in captures:gaps.append(f'PAPER_RENDER_MISSING({name})');continue
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            if list(im.size)!=captures[name]['resolution']:gaps.append(f'PAPER_PROJECTION_MISMATCH({name})')
            if name!='isaac_view' and (im.mode!='RGBA' or im.getextrema()[3][0]!=0):gaps.append(f'PAPER_RENDER_MISSING({name},transparent background)')
        if captures[name]['scene_sha256']!=scene_digest:gaps.append(f'PAPER_EVIDENCE_SCENE_MISMATCH({name})')
    if not (bundle/'assets'/'room_plan.svg').is_file():gaps.append('PAPER_RENDER_MISSING(room_plan)')
    # Compare vector geometry against the actual plan, not against a screenshot.
    import xml.etree.ElementTree as ET
    svg=bundle/'assets'/'room_plan.svg'
    if svg.is_file():
        polygons=ET.parse(svg).getroot().findall('{http://www.w3.org/2000/svg}polygon')
        plan=read(bundle/'room_plan.json');fit=read(bundle/'room_fit.json');projection=read(bundle/'render_config.json')['projection']['world_to_svg']
        targets=[plan['rooms'][0]['polygon_xy'],[[x+fit['translation_xy'][0],y+fit['translation_xy'][1]] for x,y in frozen['protected_polygon_xy']]]
        for index,points in enumerate(targets):
            expected=[(projection['translate_x']+x*projection['scale'],projection['translate_y']-y*projection['scale']) for x,y in points]
            actual=[tuple(map(float,pair.split(','))) for pair in polygons[index].attrib['points'].split()]
            if len(actual)!=len(expected) or any(abs(a-b)>.002 for pair,other in zip(actual,expected) for a,b in zip(pair,other)):gaps.append('PAPER_PROJECTION_MISMATCH(room_plan)')
    stage=Usd.Stage.Open(str(scene));tx,ty=read(bundle/'room_fit.json')['translation_xy'];world=[]
    for member in frozen['members']:
        path=f'/World/Core/{member["id"]}';p=stage.GetPrimAtPath(path)
        if not p:gaps.append(f'PAPER_ASSET_NOT_IN_SCENE({member["id"]})');continue
        actual=UsdGeom.Xformable(p).GetLocalTransformation().ExtractTranslation();expected=[member['translation'][0]+tx,member['translation'][1]+ty,member['translation'][2]+read(bundle/'room_fit.json').get('translation_z',0)]
        if any(abs(actual[i]-expected[i])>1e-6 for i in range(3)):gaps.append(f'PAPER_CORE_DRIFT(pose,{member["id"]})')
        import math
        direction=UsdGeom.Xformable(p).GetLocalTransformation().TransformDir((1,0,0));yaw=math.degrees(math.atan2(direction[1],direction[0]))
        if abs((yaw-member.get('yaw_deg',0)+180)%360-180)>1e-4:gaps.append(f'PAPER_CORE_DRIFT(rotation,{member["id"]})')
    from rfw_w.scenes.visualization.paper_case_common import bounds
    for row in placements['instances']:
        p=stage.GetPrimAtPath(row['prim'])
        if not p:gaps.append(f'PAPER_ASSET_NOT_IN_SCENE({row["id"]})');continue
        world.append((row['id'],bounds(stage,p)))
    from shapely.geometry import Polygon,box
    plan=read(bundle/'room_plan.json');room_polygon=Polygon(plan['rooms'][0]['polygon_xy'])
    protected=Polygon([(x+tx,y+ty) for x,y in frozen['protected_polygon_xy']])
    if not room_polygon.covers(protected):gaps.append('PAPER_ROOM_NOT_FIT_TO_CORE(protected polygon)')
    periphery={r['id'] for r in inventory['assets'] if r['role']=='periphery'}
    for id,(lo,hi) in world:
        footprint=box(lo[0],lo[1],hi[0],hi[1])
        if not room_polygon.buffer(1e-5).covers(footprint):gaps.append(f'PAPER_LAYOUT_CONFLICT({id},room)')
        if id in periphery and protected.intersection(footprint).area>1e-6:gaps.append(f'PAPER_LAYOUT_CONFLICT({id},protected)')
    for i,(a,(lo,hi)) in enumerate(world):
        for b,(blo,bhi) in world[i+1:]:
            penetration=[min(hi[k],bhi[k])-max(lo[k],blo[k]) for k in range(3)]
            if min(penetration)>.003:
                robot_ids={r['id'] for r in config['robot'].get('instances',[])}
                if a in robot_ids or b in robot_ids:
                    # Whole-arm boxes include large empty spaces between links. Resolve
                    # these pairs with actual visual-mesh boxes, keeping a conservative test.
                    def boxes(id):
                        path=next(r['prim'] for r in placements['instances'] if r['id']==id)
                        return [bounds(stage,p) for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.IsA(UsdGeom.Mesh)] if id in robot_ids else [dict(world)[id]]
                    collision=any(min(min(ahi[k],bhi[k])-max(alo[k],blo[k]) for k in range(3))>.003 for alo,ahi in boxes(a) for blo,bhi in boxes(b))
                    if collision:gaps.append(f'PAPER_LAYOUT_CONFLICT({a},{b},mesh_boxes)')
                else:gaps.append(f'PAPER_LAYOUT_CONFLICT({a},{b})')
    layout_metrics={}
    if 'mount_plate' in dict(world):
        import math
        matrix=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Periphery/chair')).GetLocalTransformation()
        direction=matrix.TransformDir((1,0,0));yaw=math.degrees(math.atan2(direction[1],direction[0]))
        layout_gaps,layout_metrics=check_workstation_layout(dict(world),yaw);gaps.extend(layout_gaps)
    if config['robot'].get('instances'):
        measured=dict(world);desk=measured['workbench'];chair=measured['chair']
        gap=desk[0][1]-chair[1][1];layout_metrics['chair_to_desk_gap_m']=gap
        if not .10<=gap<=.25:gaps.append('PAPER_LAYOUT_CONFLICT(chair,desk_gap)')
        gaps.extend(check_dual_mounts(measured,placements['instances'],config['robot']['instances']))
        layout_metrics['mounts']='two fixed bases on same front long edge; source geometry, static only'
    if config.get('layout',{}).get('operating_face'):
        robot_matrix=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Core/'+config['layout']['operating_face'].get('toward','franka'))).GetLocalTransformation()
        appliance=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Core/microwave')).GetLocalTransformation()
        normal=appliance.TransformDir(tuple(config['layout']['operating_face']['source_normal']))
        face_gaps,cosine=check_operating_face(normal[:2],appliance.ExtractTranslation(),robot_matrix.ExtractTranslation());gaps.extend(face_gaps)
        layout_metrics['operating_face_cosine_to_robot']=cosine
        approach=config['layout']['operating_face']['door_front_keep_clear_local_xy']
        # This corridor is reserved for the robot; other scene objects must leave it empty.
        robots={r['id'] for r in config['robot'].get('instances',[])}
        gaps.extend(check_door_approach({id:box for id,box in world if id not in robots},[(x+tx,y+ty) for x,y in approach]))
        layout_metrics['door_approach_scope']='static XY keep-clear only; not swept-volume or IK'
        desk=dict(world)['workbench']
        for id in config['targets']:
            lo,hi=dict(world)[id]
            if not on_table((lo,hi),desk):gaps.append(f'PAPER_LAYOUT_CONFLICT({id},table_support)')
    # Missing USD dependencies are checked in a separate process after physical relocation.
    relocation={'status':'not_run','reason':'flag not requested'}
    if check_relocation:
        with tempfile.TemporaryDirectory(prefix='paper_case_') as temp:
            target=Path(temp)/'scene';shutil.copytree(bundle/'scene',target)
            code="from pxr import Usd,UsdUtils; import sys; p=sys.argv[1]; s=Usd.Stage.Open(p); layers,assets,missing=UsdUtils.ComputeAllDependencies(p); bad=[x for x in missing if x not in ('OmniPBR.mdl','OmniSurface.mdl')]; assert s.GetPrimAtPath('/World/Core'); assert not bad, 'missing dependencies'; print('relocation passed')"
            result=subprocess.run([sys.executable,'-c',code,str(target/'case.usda')],capture_output=True,text=True)
            relocation={'status':'passed' if result.returncode==0 else 'failed','exit_code':result.returncode,'allowed_renderer_builtins':['OmniPBR.mdl','OmniSurface.mdl']}
            if result.returncode:gaps.append('PAPER_DEPENDENCY_MISSING(relocated scene)')
    return {'schema_version':'pvscene.paper-audit/v1','machine_passed':not gaps,'gaps':gaps,'relocation':relocation,'layout_metrics':layout_metrics,'visual_review':{'status':'not_run','reason':'requires image review'},'spec_accepted':False,'case_core_hash':frozen['case_core_hash'],'dynamic_validation':'not_run','scope':'static figure assets only'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--check-relocation',action='store_true');p.add_argument('--output-dir',required=True);a=p.parse_args();out=fresh(a.output_dir);report=audit(a.bundle,a.check_relocation);write(out/'report.json',report);print('machine_passed',report['machine_passed'],'gaps',report['gaps']);sys.exit(not report['machine_passed'])
