#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect real, curated source assets into a relocatable SPEC-012 local package."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import time

from rfw_w.scenes.visualization.paper_case_common import bounds, content_hash, digest, file_inventory, fresh, read, write, static_joint_pose


def localise_stage(stage, dest):
    """Flatten composition, copy texture dependencies, and keep renderer MDL builtins named."""
    from pxr import UsdUtils
    layer = stage.Flatten()
    layer.documentation = "SPEC-012 local asset; source identity is recorded by digest in the inventory."
    dependencies = []
    def localise(uri):
        if not uri:
            return uri
        if uri == 'OmniPBR.mdl' or uri == 'OmniSurface.mdl':
            dependencies.append({'kind':'renderer_builtin', 'name':uri, 'provided_by':'Isaac'})
            return uri
        p = Path(uri)
        if not p.is_file():
            raise ValueError('PAPER_DEPENDENCY_MISSING(' + p.name + ')')
        name = digest(p)[:16] + '_' + p.name
        target = dest.parent / 'textures' / name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)
        dependencies.append({'kind':'file', 'path':'textures/' + name, 'sha256':digest(p)})
        return 'textures/' + name
    UsdUtils.ModifyAssetPaths(layer, localise)
    layer.Export(str(dest))
    return dependencies


def make_obj(row, out, source):
    from pxr import Usd, UsdGeom
    from rfw_w.scenes.workers.isaac.render_worker import _author_instance
    from rfw_w.scenes.workers.isaac.protocol import MeshInstance
    stage = Usd.Stage.CreateNew(str(out / 'asset.usda'))
    UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
    records = {}
    _author_instance(stage, MeshInstance(row['id'], row['id'], str(source), None,
        (0,0,0), (1,0,0,0), normalisation=row.get('normalisation','none')), index=0,
        out_dir=out, asset_materials=records)
    # The generic OBJ author leaves UsdUVTexture.st unconnected. Bind the source
    # UVs explicitly here; otherwise Isaac samples one texel across the object.
    from pxr import UsdShade, Sdf
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdShade.Shader):
            shader=UsdShade.Shader(prim)
            if shader.GetIdAttr().Get()=='UsdUVTexture':
                reader=UsdShade.Shader.Define(stage,str(prim.GetParent().GetPath())+'/SourceUV')
                reader.CreateIdAttr('UsdPrimvarReader_float2')
                reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
                reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
                shader.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    stage.GetRootLayer().Save()
    deps = localise_stage(stage, out/'asset.usda')
    return stage, deps


def make_support(row, out):
    from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
    stage=Usd.Stage.CreateNew(str(out/'asset.usda'));UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    mat=UsdShade.Material.Define(stage,'/World/Oak');shader=UsdShade.Shader.Define(stage,'/World/Oak/Shader')
    shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.48,.27,.12));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.45);mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    w,d,h=row['dimensions_m'];parts=[('top',(0,0,h-.035),(w,d,.07))]
    for i,(x,y) in enumerate([(-w/2+.12,-d/2+.12),(w/2-.12,-d/2+.12),(-w/2+.12,d/2-.12),(w/2-.12,d/2-.12)]):parts.append((f'leg_{i}',(x,y,(h-.07)/2),(.09,.09,h-.07)))
    if row['source_kind']=='mount_plate':
        parts=[('plate',(0,0,h/2),(w,d,h))]
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(.12,.14,.16))
    for name,loc,size in parts:
        cube=UsdGeom.Cube.Define(stage,'/World/'+name);cube.CreateSizeAttr(1);cube.AddTranslateOp().Set(Gf.Vec3d(*loc));cube.AddScaleOp().Set(Gf.Vec3f(*size));UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save();return stage,[]


def collect(config, output):
    from pxr import Usd, UsdGeom, UsdPhysics
    started=time.monotonic();output=fresh(output);config=read(config);write(output/'case_config.json',config)
    delivery=Path(config.get('source_delivery', os.environ.get('PVSCENE_PAPER_SOURCE_DELIVERY', '.')))
    inventory=[];gaps=[]
    for row in config['assets']:
        out=output/'sources'/row['id'];out.mkdir(parents=True)
        record=dict(row);record.update(license_status='not_verified_for_redistribution',render_representation='static_geometry',status='not_run')
        try:
            kind=row['source_kind'];source=None;proof={}
            if kind=='prepared_obj':
                source=delivery/'prepared_assets'/row['asset_id']/(row['asset_id']+'.obj')
                if not source.is_file():raise ValueError('PAPER_ASSET_MISSING('+row['id']+')')
                stage,deps=make_obj(row,out,source)
                proof={'kind':'prepared_source_mesh', 'source_sha256':digest(source)}
            elif kind=='piper_urdf':
                from rfw_w.scenes.visualization.piper_case_robot import author
                stage,deps,proof=author(row,out,Path(os.environ['PVSCENE_PAPER_PIPER_ROOT']))
            elif kind=='simready_obj':
                parts=row['asset_id'].split('_');source=Path(os.environ['PVSCENE_SIMREADY_ASSETS_ROOT'])/parts[0]/parts[1]/row['asset_id']/'mesh'/(row['asset_id']+'.obj')
                stage,deps=make_obj(row,out,source)
                proof={'kind':'simready_source_mesh','source_sha256':digest(source)}
            elif kind=='source_catalog':
                import trimesh
                catalog=Path(row.get('catalog', os.environ.get('RFW_SOURCE_CATALOG', '')))
                selected=next(x for x in read(catalog)['assets'] if x['asset_id']==row['asset_id'])
                source=Path(selected['visual_uri']);physics=Path(selected['genesis_physics_uri'])
                metadata=read(physics)
                mesh=trimesh.load(str(source),force='scene').to_mesh()
                scale=selected['longest_dimension_m']/max(mesh.extents);mesh.apply_scale(scale)
                mesh.export(str(out/'visual.obj'))
                stage,deps=make_obj(row,out,out/'visual.obj')
                # This is a source-backed static visual, not a newly materialised simulation body.
                from pxr import UsdPhysics
                for prim in stage.Traverse():
                    for api in (UsdPhysics.RigidBodyAPI,UsdPhysics.CollisionAPI,UsdPhysics.MassAPI):
                        if prim.HasAPI(api):prim.RemoveAPI(api)
                stage.GetRootLayer().Save()
                proof={'kind':'source_catalog_visual','catalog_record':{k:selected[k] for k in ('asset_id','semantic_class','physics_type','geometry_family','material_label','dimensions_m','property_provenance')},'source_sha256':digest(source),'physics_json_sha256':digest(physics),'physics_json_keys':list(metadata),'scale_factor':scale,'dynamic_validation':{'status':'not_run','reason':'original softsim visual and physics metadata only; no local tetrahedralisation or simulation'}}
            elif kind in ('workbench','mount_plate'):
                stage,deps=make_support(row,out);proof={'kind':'constructed', 'recipe_sha256':content_hash(row)}
            else:
                if kind != 'usd' or not row.get('source_uri'):
                    raise ValueError('SOURCE_KIND_UNSUPPORTED(require explicit source_uri)')
                source=Path(row['source_uri']).expanduser().resolve()
                if not source.is_file():raise ValueError('PAPER_ASSET_MISSING('+row['id']+')')
                stage=Usd.Stage.Open(str(source))
                if row.get('static_joint_degrees'):static_joint_pose(stage,row['static_joint_degrees'])
                deps=localise_stage(stage,out/'asset.usda')
                record['source_authored_units']=UsdGeom.GetStageMetersPerUnit(stage);record['source_authored_up_axis']=str(UsdGeom.GetStageUpAxis(stage))
                proof={'source_sha256':digest(source),'joints':[]}
                for prim in stage.Traverse():
                    if prim.IsA(UsdPhysics.RevoluteJoint):
                        j=UsdPhysics.RevoluteJoint(prim);proof['joints'].append({'path':str(prim.GetPath()),'axis':str(j.GetAxisAttr().Get()),'limits_deg':[j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()], 'body0':[str(x) for x in j.GetBody0Rel().GetTargets()], 'body1':[str(x) for x in j.GetBody1Rel().GetTargets()]})
                if row['source_physics_type']=='articulated' and not proof['joints']:raise ValueError('PAPER_SOURCE_TYPE_UNPROVEN('+row['id']+',articulated)')
                if row['source_physics_type']=='deformable':
                    for name in ['package_meta.json','tet_mesh.json','preset.json','validation_report.json']:
                        p=source.parent/name
                        if not p.is_file():raise ValueError('PAPER_SOURCE_TYPE_UNPROVEN(cushion,'+name+')')
                        sidecar=read(p)
                        proof[name]={'sha256':digest(p),'keys':list(sidecar)}
                        if name=='tet_mesh.json':
                            if sidecar.get('num_points',0)<=0 or sidecar.get('num_elements',0)<=0:raise ValueError('PAPER_SOURCE_TYPE_UNPROVEN(cushion,empty tetrahedra)')
                            proof[name].update(num_points=sidecar['num_points'],num_elements=sidecar['num_elements'])
                        if name=='package_meta.json':proof[name].update({k:sidecar[k] for k in ['body_type','provider','sim_kind'] if k in sidecar})
                    proof['dynamic_validation']={'status':'not_run','reason':'source sidecars inspected; no simulation rerun'}
                stage=Usd.Stage.Open(str(out/'asset.usda'))
                if UsdGeom.GetStageMetersPerUnit(stage) != 1 or str(UsdGeom.GetStageUpAxis(stage)) != 'Z':
                    raise ValueError('SOURCE_FRAME_UNSUPPORTED(require prepared metre/Z-up USD)')
            record.update(status='collected',usd=str((out/'asset.usda').relative_to(output)),bounds_m=bounds(stage),dependencies=deps,type_evidence=proof)
            record['package_files']=file_inventory(out)
            record['package_digest']=content_hash(record['package_files'])
            print('collected',row['id'],'bounds',record['bounds_m'],flush=True)
        except Exception as exc:
            record['status']='failed';record['gap']=str(exc);gaps.append(str(exc));print('failed',row['id'],type(exc).__name__,flush=True)
        inventory.append(record)
    result={'schema_version':'pvscene.paper-inventory/v1','case_id':config['case_id'],'assets':inventory,'gaps':gaps,'retrieval_calls':0,'external_calls':0,'elapsed_s':round(time.monotonic()-started,3)}
    write(output/'asset_inventory.json',result)
    (output/'MISSING_ASSETS.md').write_text('# Missing assets\n\n'+ ('\n'.join('- '+x for x in gaps) if gaps else 'No missing source assets. Stage renders are not_run.\n'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    result=collect(a.case,a.output_dir);sys.exit(bool(result['gaps']))
