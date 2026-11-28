#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render one immutable paper case using a separate, explicitly static Isaac entry point."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from rfw_w.scenes.visualization.paper_case_common import bounds, digest, file_inventory, fresh, read, write


def room_svg(bundle):
    plan=read(bundle/'room_plan.json');frozen=read(bundle/'frozen_core.json');fit=read(bundle/'room_fit.json')
    room=plan['rooms'][0];poly=room['polygon_xy'];tx,ty=fit['translation_xy'];scale=120;pad=30
    maxy=max(y for x,y in poly);maxx=max(x for x,y in poly)
    def point(x,y):return (pad+x*scale,pad+(maxy-y)*scale)
    def pts(values):return ' '.join(f'{a:.3f},{b:.3f}' for a,b in (point(x,y) for x,y in values))
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{maxx*scale+2*pad}" height="{maxy*scale+2*pad}" viewBox="0 0 {maxx*scale+2*pad} {maxy*scale+2*pad}">',f'<polygon points="{pts(poly)}" fill="#edf5f2" stroke="#42515a" stroke-width="3"/>']
    parts.append(f'<polygon points="{pts([(x+tx,y+ty) for x,y in frozen["protected_polygon_xy"]])}" fill="none" stroke="#ef8735" stroke-width="2" stroke-dasharray="7 5"/>')
    for item in frozen['members']:
        lo,hi=item['bounds_local_m'];rect=[(lo[0]+tx,lo[1]+ty),(hi[0]+tx,lo[1]+ty),(hi[0]+tx,hi[1]+ty),(lo[0]+tx,hi[1]+ty)]
        parts.append(f'<polygon points="{pts(rect)}" fill="#c6b18d" fill-opacity="0.4" stroke="#6a655b"/>')
        x,y=point((lo[0]+hi[0])/2+tx,(lo[1]+hi[1])/2+ty);parts.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="11" text-anchor="middle">{item["id"]}</text>')
    for opening in plan['openings']:
        parts.append(f'<polyline points="{pts(opening["wall_segment_xy"])}" stroke="#4685a3" stroke-width="6"/>')
    parts.append('</svg>');(bundle/'assets'/'room_plan.svg').write_text('\n'.join(parts))
    return {'world_to_svg':{'scale':scale,'translate_x':pad,'translate_y':pad+maxy*scale,'flip_y':True},'source':'room_plan.json + frozen_core.json + room_fit.json'}


def render_isaac(bundle,gpu):
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':gpu,'physics_gpu':gpu,'multi_gpu':False,'sync_loads':True,'renderer':'RayTracedLighting'})
    import carb.settings
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.core.version import get_version
    from PIL import Image
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade
    started=time.monotonic();bundle=Path(bundle).resolve();scene=bundle/'scene'/'case.usda';scene_hash=digest(scene)
    omni.usd.get_context().open_stage(str(scene));stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
    settings=carb.settings.get_settings();settings.set('/rtx/post/aa/op',3)
    rep.orchestrator.set_capture_on_play(False)
    # No world.step/timeline.play: all images show exactly the recorded static pose.
    def visible(path,flag):
        p=stage.GetPrimAtPath(path)
        if p:UsdGeom.Imageable(p).CreateVisibilityAttr().Set('inherited' if flag else 'invisible')
    def aim_light(light,xyz,target):
        m=Gf.Matrix4d(1);m.SetLookAt(Gf.Vec3d(*xyz),Gf.Vec3d(*target),Gf.Vec3d(0,0,1));UsdGeom.Xformable(light).AddTransformOp().Set(m.GetInverse())
    dome=UsdLux.DomeLight.Define(stage,'/World/PaperDome');dome.CreateIntensityAttr(700)
    for name,pos,power,size in [('Key',(1,-2,7),2200,5),('Fill',(6,3,6),1600,4)]:
        l=UsdLux.RectLight.Define(stage,'/World/Paper'+name);l.CreateIntensityAttr(power);l.CreateWidthAttr(size);l.CreateHeightAttr(size);aim_light(l,pos,(2.5,2.5,.7))
    # Cutaway only on the session layer, retaining the original source geometry untouched.
    room=read(bundle/'room_plan.json')['rooms'][0];maxx=max(p[0] for p in room['polygon_xy']);maxy=max(p[1] for p in room['polygon_xy'])
    hidden=[]
    for p in list(stage.GetPrimAtPath('/World/Shell').GetDescendants() if hasattr(stage.GetPrimAtPath('/World/Shell'),'GetDescendants') else Usd.PrimRange(stage.GetPrimAtPath('/World/Shell'))):
        if not p.IsA(UsdGeom.Mesh):continue
        lo,hi=bounds(stage,p);ext=[hi[i]-lo[i] for i in range(3)];cx=(lo[0]+hi[0])/2;cy=(lo[1]+hi[1])/2
        if '_exterior/' in str(p.GetPath()) or lo[2]>2.5 or (cx>maxx-.25 and ext[2]>.4) or (cy<.25 and ext[2]>.4):
            visible(str(p.GetPath()),False);hidden.append(str(p.GetPath()))
    # Infinigen combines all wall directions into one mesh. Cut individual faces
    # in the view session rather than hiding the entire wall object.
    cut_faces={}
    for p in list(Usd.PrimRange(stage.GetPrimAtPath('/World/Shell'))):
        if not p.IsA(UsdGeom.Mesh) or not any(t in str(p.GetPath()) for t in ['_wall/','_exterior/','skirting']):continue
        mesh=UsdGeom.Mesh(p);counts=list(mesh.GetFaceVertexCountsAttr().Get());indices=list(mesh.GetFaceVertexIndicesAttr().Get());points=mesh.GetPointsAttr().Get()
        matrix=UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world=np.array([matrix.Transform(Gf.Vec3d(*pt)) for pt in points])
        st=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st');uvs=np.asarray(st.ComputeFlattened()) if st else None
        newpoints=[];newuvs=[];newcounts=[];remap={};offset=0
        # Clip crossing polygons at a precise plane. Dropping by face centroid left
        # diagonal wall fragments because Infinigen triangulates large wall faces.
        for face,count in enumerate(counts):
            polygon=[]
            for corner in range(offset,offset+count):
                vi=indices[corner];uv=uvs[corner] if uvs is not None else np.array([0.,0.])
                polygon.append((np.asarray(points[vi]),world[vi],uv))
            offset+=count
            for axis,limit,sign in [(0,maxx-.15,-1),(1,.15,1)]:
                clipped=[]
                for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                    da=sign*(a[1][axis]-limit);db=sign*(b[1][axis]-limit)
                    if da>=-1e-8:clipped.append(a)
                    if (da>=0)!=(db>=0):
                        t=da/(da-db);clipped.append(tuple(av+t*(bv-av) for av,bv in zip(a,b)))
                polygon=clipped
                if not polygon:break
            if len(polygon)<3:continue
            remap[face]=len(newcounts);newcounts.append(len(polygon))
            for local,worldpoint,uv in polygon:newpoints.append(tuple(map(float,local)));newuvs.append(tuple(map(float,uv)))
        mesh.GetPointsAttr().Set(newpoints);mesh.GetFaceVertexCountsAttr().Set(newcounts);mesh.GetFaceVertexIndicesAttr().Set(list(range(len(newpoints))))
        mesh.GetNormalsAttr().Block()
        for child in p.GetChildren():
            if child.IsA(UsdGeom.Subset):
                attr=UsdGeom.Subset(child).GetIndicesAttr();attr.Set([remap[i] for i in attr.Get() if i in remap])
        if st:
            st.SetInterpolation('faceVarying');st.Set(newuvs);st.BlockIndices()
        cut_faces[str(p.GetPath())]={'before':len(counts),'after':len(newcounts),'method':'polygon clipping with interpolated UV','planes':{'x_max':maxx-.15,'y_min':.15}}
    UsdGeom.Xform.Define(stage,'/World/Swatches')
    materials=[]
    for p in stage.Traverse():
        if p.IsA(UsdShade.Material) and str(p.GetPath()).startswith('/World/Shell'):
            materials.append(str(p.GetPath()))
    # Prefer three different exported materials. Their real bindings are retained in the manifest.
    materials=[next(m for m in materials if token in m) for token in ['floor_oak','wall_off_white','shader_wood']]
    if len(materials)<3:raise ValueError('PAPER_RENDER_MISSING(three shell materials)')
    swatch_uv_sources=[]
    for i,material in enumerate(materials):
        candidates=[]
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):continue
            bound=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not bound or str(bound.GetPath())!=material:continue
            mesh=UsdGeom.Mesh(prim);pv=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
            if not pv:continue
            values=np.asarray(pv.ComputeFlattened());offset=0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                polygon=values[offset:offset+count];offset+=count
                if len(polygon)<3:continue
                area=abs(float(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1))))/2
                if area>1e-10:candidates.append((area,str(prim.GetPath()),polygon))
        if not candidates:raise ValueError('PAPER_RENDER_MISSING(material UV island)')
        _,source,polygon=max(candidates,key=lambda x:x[0]);uvcentre=polygon.mean(axis=0)
        radius=min(abs(float(np.cross(b-a,uvcentre-a)))/float(np.linalg.norm(b-a)) for a,b in zip(polygon,np.roll(polygon,-1,axis=0)) if np.linalg.norm(b-a)>1e-9)*.45
        swatch_uv_sources.append({'material':material,'mesh':source,'centre_uv':uvcentre.tolist(),'half_width_uv':radius,'method':'square inside largest source face UV island; same PBR textures'})
        sphere=UsdGeom.Mesh.Define(stage,f'/World/Swatches/swatch_{i}')
        points=[];uv=[];faces=[]
        for lat in range(49):
            theta=math.pi*lat/48
            for lon in range(97):
                phi=2*math.pi*lon/96;points.append((.45*math.sin(theta)*math.cos(phi),.45*math.sin(theta)*math.sin(phi),.45*math.cos(theta)));uv.append(tuple(uvcentre+radius*np.array([2*lon/96-1,2*lat/48-1])))
        for lat in range(48):
            for lon in range(96):
                a=lat*97+lon;faces.extend([a,a+97,a+98,a+1])
        sphere.CreatePointsAttr(points);sphere.CreateFaceVertexCountsAttr([4]*(48*96));sphere.CreateFaceVertexIndicesAttr(faces)
        sphere.CreateSubdivisionSchemeAttr('none');pv=UsdGeom.PrimvarsAPI(sphere).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex);pv.Set(uv)
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(material)))
    palette_overrides={}
    for name,path,colour in [('ChairFabric','/World/Periphery/chair',(.20,.30,.28)),('StorageFinish','/World/Periphery/storage',(.40,.28,.17)),('CushionFabric','/World/Core/cushion',(.43,.48,.40))]:
        if not stage.GetPrimAtPath(path):continue
        material=UsdShade.Material.Define(stage,'/World/Paper'+name);shader=UsdShade.Shader.Define(stage,'/World/Paper'+name+'/Shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.72);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material,bindingStrength=UsdShade.Tokens.strongerThanDescendants);palette_overrides[path]={'diffuse':colour,'roughness':.72,'scope':'figure session only'}
    camera=UsdGeom.Camera.Define(stage,'/World/PaperCamera');camera.CreateClippingRangeAttr(Gf.Vec2f(.01,1000))
    jobs=[('core_iso','/World/Core',2048,2048,True),('core_operating_view','/World/Core',2048,1536,True),('core_topdown','/World/Core',2048,1536,True),('shell_clay_iso','/World/Shell',2048,2048,True)]
    jobs += [(f'material_{i+1:02d}',f'/World/Swatches/swatch_{i}',768,768,True) for i in range(3)]
    jobs += [(f'periphery_{i+1:02d}',f'/World/Periphery/{name}',1024,1024,True) for i,name in enumerate(['storage','chair','plant'])]
    config=read(bundle/'case_config.json')
    jobs += [(name,f'/World/Core/{id}',1024,1024,True) for name,id in [('target_rigid','cup'),('target_articulated','microwave'),('target_deformable',config.get('primary_deformable','cushion')),('robot_'+config['robot']['id'],config['robot']['id'])]]
    config=read(bundle/'case_config.json')
    jobs += [('target_'+id,f'/World/Core/{id}',1024,1024,True) for id in config['targets'] if id not in ('cup','microwave','cushion')]
    jobs += [('robot_'+r['id'],'/World/Core/'+r['id'],1024,1024,True) for r in config['robot'].get('instances',[]) if r['id']!=config['robot']['id']]
    jobs += [('delivery_iso','/World',2048,2048,True),('isaac_view','/World',1920,1080,False)]
    captures=[]
    clay=UsdShade.Material.Define(stage,'/World/PaperClay');sh=UsdShade.Shader.Define(stage,'/World/PaperClay/Shader');sh.CreateIdAttr('UsdPreviewSurface');sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.62,.64,.62));sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.8);clay.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
    shellprim=stage.GetPrimAtPath('/World/Shell')
    for name,focus,w,h,alpha in jobs:
        for group in ['Core','Shell','Periphery','Swatches']:
            visible('/World/'+group,focus=='/World' and group!='Swatches' or focus.startswith('/World/'+group))
        for i in range(3):visible(f'/World/Swatches/swatch_{i}',focus==f'/World/Swatches/swatch_{i}')
        for child in stage.GetPrimAtPath('/World/Periphery').GetChildren():visible(str(child.GetPath()),focus in ['/World','/World/Periphery'] or str(child.GetPath())==focus)
        for child in stage.GetPrimAtPath('/World/Core').GetChildren():visible(str(child.GetPath()),focus in ['/World','/World/Core'] or str(child.GetPath())==focus)
        if name=='shell_clay_iso':UsdShade.MaterialBindingAPI.Apply(shellprim).Bind(clay,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        else:UsdShade.MaterialBindingAPI(shellprim).UnbindDirectBinding()
        if focus=='/World':
            lo=[0,0,0];hi=[maxx,maxy,2.8]
        else:lo,hi=bounds(stage,stage.GetPrimAtPath(focus))
        centre=np.array([(lo[i]+hi[i])/2 for i in range(3)]);extent=np.array(hi)-np.array(lo)
        az=math.radians(305 + (config.get('layout',{}).get('core_yaw_deg',{}).get(focus.rsplit('/',1)[-1],0) if focus.startswith('/World/Core/') else 0));el=math.radians(32 if focus=='/World' else 28)
        if name=='core_topdown':az=math.radians(270);el=math.radians(89.9)
        if name=='core_operating_view':az=math.radians(235);el=math.radians(42)
        distance=max(float(np.linalg.norm(extent))*1.8,2.)
        eye=centre+distance*np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
        if name=='isaac_view':
            centre=np.array([maxx*.48,maxy*.55,.95]);eye=np.array([maxx*1.08,-maxy*.18,3.4]);camera.CreateProjectionAttr('perspective');camera.CreateHorizontalApertureAttr(36);camera.CreateVerticalApertureAttr(36*h/w);camera.CreateFocalLengthAttr(26)
        else:
            camera.CreateProjectionAttr('orthographic');span=max(float(np.linalg.norm(extent[:2])),float(extent[2])*1.35)*1.18;camera.CreateHorizontalApertureAttr(span*10);camera.CreateVerticalApertureAttr(span*10*h/w)
        set_camera_view(eye=eye,target=centre,camera_prim_path=str(camera.GetPath()))
        product=rep.create.render_product(str(camera.GetPath()),(w,h));rgb=rep.AnnotatorRegistry.get_annotator('rgb');depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');rgb.attach(product);depth.attach(product)
        for _ in range(25):app.update()
        rep.orchestrator.step(rt_subframes=16)
        arr=np.array(rgb.get_data(),copy=True);z=np.array(depth.get_data(),copy=True).reshape(h,w)
        if not arr.size:raise RuntimeError('PAPER_RENDER_MISSING('+name+')')
        rgba=np.zeros((h,w,4),dtype=np.uint8);rgba[:,:,:3]=arr[:,:,:3];rgba[:,:,3]=np.where(np.isfinite(z)&(z>0)&(z<100),255,0) if alpha else 255
        Image.fromarray(rgba,'RGBA').save(bundle/'assets'/(name+'.png'))
        captures.append({'id':name,'file':'assets/'+name+'.png','scene_sha256':scene_hash,'case_core_hash':read(bundle/'frozen_core.json')['case_core_hash'],'camera':{'eye':eye.tolist(),'target':centre.tolist(),'projection':str(camera.GetProjectionAttr().Get()),'horizontal_aperture':camera.GetHorizontalApertureAttr().Get(),'vertical_aperture':camera.GetVerticalApertureAttr().Get(),'focal_length':camera.GetFocalLengthAttr().Get()},'resolution':[w,h],'alpha_from':'finite camera depth' if alpha else 'opaque','visible_focus':focus,'foreground_pixels':int((rgba[:,:,3]>0).sum())})
        print('rendered',name,flush=True);rgb.detach();depth.detach();product.destroy()
    write(bundle/'isaac'/'render_report.json',{'status':'completed','entry':'python -m rfw_w.scenes.visualization.render_case_assets --isaac-child','renderer':'RayTracedLighting','isaac_version':'.'.join(list(get_version())[:2]),'scene_sha256':scene_hash,'captures':captures,'cutaway_hidden_meshes':hidden,'cutaway_faces':cut_faces,'material_swatch_sources':materials,'swatch_uv_sources':swatch_uv_sources,'palette_overrides':palette_overrides,'settle':{'status':'not_run','reason':'static presentation; timeline never played'},'elapsed_s':round(time.monotonic()-started,3)})
    app.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--case-run');p.add_argument('--active-gpu',type=int,default=4);p.add_argument('--output-dir');p.add_argument('--isaac-child');a=p.parse_args()
    if a.isaac_child:return render_isaac(Path(a.isaac_child),a.active_gpu)
    out=fresh(a.output_dir);src=Path(a.case_run);shutil.copytree(src/'scene',out/'scene')
    for log in (out/'scene'/'shell').glob('*.txt'):log.unlink()
    for name in ['case_config.json','asset_inventory.json','frozen_core.json','room_plan.json','room_fit.json','placements.json','shell_gate.json','build_report.json']:shutil.copyfile(src/name,out/name)
    for run in read(out/'case_config.json').get('selection_evidence',[]):
        destination=out/'selection'/run;destination.mkdir(parents=True)
        # A retained bundle includes its candidates even after old runs are cleaned.
        source=src/'selection'/run
        if not source.is_dir():
            raise FileNotFoundError('CASE_SOURCE_MISSING(supply an existing bundle)')
        for name in ('contact_sheet.jpg','candidates.json'):shutil.copyfile(source/name,destination/name)
    (out/'assets').mkdir();(out/'isaac').mkdir();write(out/'render_config.json',{'projection':room_svg(out),'light_mode':'paper studio session overrides','physics':'static'})
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    with (out/'isaac'/'worker.log').open('w') as log:
        result=subprocess.run([os.environ.get('PVSCENE_ISAAC_PYTHON', '/isaac-sim/python.sh'),'-m','rfw_w.scenes.visualization.render_case_assets','--isaac-child',str(out),'--active-gpu',str(a.active_gpu)],env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode or not (out/'isaac'/'render_report.json').is_file():raise RuntimeError('PAPER_RENDER_MISSING(Isaac failed; see local worker.log)')
    report=read(out/'isaac'/'render_report.json');write(out/'provenance.json',{'case_id':read(out/'case_config.json')['case_id'],'case_core_hash':read(out/'frozen_core.json')['case_core_hash'],'captures':report['captures'],'external_calls':0,'retrieval_calls':0,'not_run':{k:'Static paper presentation, no execution requested' for k in ['settle','visual.critique','robot.ik','robot.nav','robot.motion','task.rollout','deformable.simulation','articulated.task']}})
    # Runtime log may contain local paths; excluded from any public export.
    (out/'MISSING_ASSETS.md').write_text('No missing required renders. Dynamic validation and VLM are not_run.\nLocal source redistribution licenses remain unverified.\n')
    from PIL import Image,ImageOps,ImageDraw
    names=[x['id'] for x in report['captures']];board=Image.new('RGB',(1600,350*math.ceil(len(names)/5)),'#f4f5f6');draw=ImageDraw.Draw(board)
    for i,name in enumerate(names):
        im=Image.open(out/'assets'/(name+'.png')).convert('RGBA');im.thumbnail((300,290));x=(i%5)*320;y=(i//5)*350;board.paste(im,(x+(320-im.width)//2,y+25),im);draw.text((x+12,y+325),name,fill='#17202f')
    board.save(out/'preview.jpg')
    write(out/'artifact_hashes.json',file_inventory(out/'assets'))


if __name__=='__main__':main()
