#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze curated local geometry, fit a real RoomPlan, and build its architectural shell."""
from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path
import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import time
import math
import itertools
from rfw_w.scenes.visualization.paper_case_common import bounds, content_hash, digest, fresh, read, write


@dataclass
class FigureCell:
    """Room-fitting adapter, never presented as a production WorkCell solution."""
    protected: object
    core_hash: str
    support_height_m: float = .78
    target_offset: tuple = (0.,0.)
    approach_arc_deg: tuple = (0.,360.)


def build(collection, output, reuse_shell=None, case_config=None):
    from pxr import Usd, UsdGeom, Gf
    from rfw_w.scenes.contracts.scene_spec import SceneSpec
    from rfw_w.scenes.contracts.shell_build import plan_gaps, ShellBuild
    from rfw_w.scenes.geometry.polygon import Polygon
    from rfw_w.scenes.solvers.room import fit_room_around_cell
    from rfw_w.scenes.solvers.room_plan import plan_from_fitted
    from rfw_w.scenes.solvers.shell_providers import InfinigenShellProvider, export_shell
    from rfw_w.scenes.solvers.materials import MaterialExportProfile
    started=time.monotonic();output=fresh(output);collection=Path(collection).resolve()
    inventory=read(collection/'asset_inventory.json');config=read(collection/'case_config.json')
    if case_config:
        revised=read(case_config)
        if revised['assets']!=config['assets']:raise ValueError('PAPER_CORE_DRIFT(layout override changes sources)')
        config=revised
    if inventory['gaps']:raise ValueError('PAPER_ASSET_MISSING(collection failed)')
    shutil.copytree(collection/'sources',output/'scene'/'sources')
    write(output/'asset_inventory.json',inventory);write(output/'case_config.json',config)
    layout=config['layout'];positions=layout['core_positions_m']
    members=[]
    for row in inventory['assets']:
        if row['id'] not in positions:continue
        xyz=list(positions[row['id']])
        if row.get('origin_policy')!='urdf_base':xyz[2]-=row['bounds_m'][0][2]
        yaw=layout.get('core_yaw_deg',{}).get(row['id'],0);angle=math.radians(yaw)
        corners=[(x*math.cos(angle)-y*math.sin(angle)+xyz[0],x*math.sin(angle)+y*math.cos(angle)+xyz[1],z+xyz[2]) for x,y,z in itertools.product(*zip(*row['bounds_m']))]
        box=[[f(p[i] for p in corners) for i in range(3)] for f in (min,max)]
        members.append({'id':row['id'],'asset_digest':row['package_digest'],'translation':xyz,'yaw_deg':yaw,'scale':1,'joint_pose':row.get('static_joint_degrees','source_rest_pose'),'bounds_local_m':box})
    protected=layout['protected_polygon_xy']
    frozen={'members':members,'protected_polygon_xy':protected,'relations':config['relations'],'robot':config['robot'],'units':'m','up_axis':'Z'}
    frozen['case_core_hash']=content_hash(frozen);write(output/'frozen_core.json',frozen)
    cell=FigureCell(Polygon(tuple(map(tuple,protected))),frozen['case_core_hash'])
    specdata=read(data_path('examples/office_pick_cup/task.json'))
    specdata.update(scene_id=config['case_id'],prompt='Room intent adapter for curated paper case; target solver not invoked.')
    specdata['room']['size_bounds_m']=config['room']['size_bounds_m'];specdata['style']['density']='sparse'
    spec=SceneSpec.from_dict(specdata);fitted=fit_room_around_cell(spec,cell,room_seed=config['room']['seed'])
    plan=plan_from_fitted(spec,cell,fitted,room_seed=config['room']['seed'])
    write(output/'room_plan.json',plan.to_dict());write(output/'room_fit.json',{'translation_xy':list(fitted.cell_translation_xy),'method':'fit_room_around_cell(FigureCell aggregate)','input_case_core_hash':frozen['case_core_hash'],'production_task_solver_run':False})
    print('frozen core; fitted room',flush=True)
    if reuse_shell:
        reuse_shell=Path(reuse_shell).resolve()
        old_plan=read(reuse_shell/'room_plan.json')
        if any(old_plan[k]!=plan.to_dict()[k] for k in ['rooms','openings']):
            raise ValueError('PAPER_EVIDENCE_SCENE_MISMATCH(reused shell plan)')
        shutil.copytree(reuse_shell/'scene'/'shell',output/'scene'/'shell')
        from rfw_w.scenes.contracts.state import RoomShell, Opening, OpeningKind
        data=read(reuse_shell/'shell_build.json');shell_data=dict(data.pop('shell'))
        shell_data['openings']=tuple(Opening(**{**o,'kind':OpeningKind(o['kind'])}) for o in shell_data['openings'])
        old_build=ShellBuild(shell=RoomShell(**shell_data),**data)
        rechecked=plan_gaps(plan,old_build)
        write(output/'shell_gate.json',{'status':'passed' if not rechecked else 'failed','gaps':[str(g) for g in rechecked],'method':'plan_gaps against same measured shell; identical rooms and openings','case_core_hash':frozen['case_core_hash']})
        shutil.copyfile(reuse_shell/'shell_build.json',output/'shell_build.json')
        gaps=read(output/'shell_gate.json')['gaps'];usd=output/'scene'/'shell'/'shell.usdc'
        write(output/'shell_reuse.json',{'source_run':reuse_shell.name,'plan_equal':True,'source_shell_sha256':digest(usd)})
    else:
        shell=InfinigenShellProvider().build(plan,output_dir=output/'shell_build')
        write(output/'shell_build.json',shell.to_dict())
        gaps=plan_gaps(plan,shell)
        write(output/'shell_gate.json',{'status':'passed' if not gaps else 'failed','gaps':[str(x) for x in gaps],'method':'plan_gaps','case_core_hash':frozen['case_core_hash']})
        print('shell built; exporting PBR',flush=True)
        usd=export_shell(shell,output_dir=output/'scene'/'shell',material_profile=MaterialExportProfile(resolution=512))
    stage=Usd.Stage.CreateNew(str(output/'scene'/'case.usda'));UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    arch=UsdGeom.Xform.Define(stage,'/World/Shell');arch.GetPrim().GetReferences().AddReference('shell/'+usd.name)
    tx,ty=fitted.cell_translation_xy
    floor_meshes=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and '/Shell/' in str(p.GetPath()) and '_floor/' in str(p.GetPath())]
    floor_z=max(bounds(stage,p)[1][2] for p in floor_meshes)
    write(output/'room_fit.json',{'translation_xy':[tx,ty],'translation_z':floor_z,'method':'fit_room_around_cell(FigureCell aggregate)','floor_alignment':'measured built floor surface; one rigid translation of whole core','input_case_core_hash':frozen['case_core_hash'],'production_task_solver_run':False})
    for group in ['Core','Periphery']:UsdGeom.Xform.Define(stage,'/World/'+group)
    placements=[]
    # Fixed candidate slots around the actual fitted core; later audit measures every box.
    extras={'storage':(.65,4.55,0),'chair':(4.7,3.8,0),'plant':(4.8,4.5,0)}
    for row in inventory['assets']:
        id=row['id'];group='Core' if id in positions else 'Periphery'
        prim=UsdGeom.Xform.Define(stage,f'/World/{group}/{id}')
        UsdGeom.Xform.Define(stage,f'/World/{group}/{id}/Source').GetPrim().GetReferences().AddReference(f'sources/{id}/asset.usda')
        if id in positions:
            item=next(x for x in members if x['id']==id);xyz=list(item['translation']);xyz[0]+=tx;xyz[1]+=ty
        else:xyz=list(extras[id])
        yaw=layout.get('core_yaw_deg',{}).get(id,0)
        if id=='chair':
            xyz=list(layout['chair_relative_to_core_m']);xyz[0]+=tx;xyz[1]+=ty
            xyz[2]-=row['bounds_m'][0][2];yaw=layout['chair_yaw_deg']
        xyz[2]+=floor_z
        prim.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        if yaw:prim.AddRotateZOp().Set(yaw)
        placements.append({'id':id,'group':group,'prim':str(prim.GetPath()),'translation':xyz,'yaw_deg':yaw,'asset_digest':row['package_digest'],'world_bounds_m':bounds(stage,prim.GetPrim())})
    stage.GetRootLayer().Save()
    write(output/'placements.json',{'case_core_hash':frozen['case_core_hash'],'instances':placements})
    write(output/'build_report.json',{'status':'completed','elapsed_s':round(time.monotonic()-started,3),'case_core_hash':frozen['case_core_hash'],'scene_sha256':digest(output/'scene'/'case.usda'),'retrieval_calls':0,'external_calls':0,'settle':{'status':'not_run','reason':'static presentation'},'shell_gate_status':'passed' if not gaps else 'failed'})
    print('case assembled',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--reuse-shell-from');p.add_argument('--case-config');a=p.parse_args();build(a.collection,a.output_dir,a.reuse_shell_from,a.case_config)
