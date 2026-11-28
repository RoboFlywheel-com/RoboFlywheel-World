# SPDX-License-Identifier: Apache-2.0
"""Search table mounting candidates while preserving original room instance poses."""
import argparse,json,math,shutil,time,xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_mujoco_piper import audit
from rfw_w.scenes.validation.verify_single_room_installation import check_layout,contacts


def xml_scene(root,config):
    import trimesh
    tree=ET.parse(root/'robot/robot.xml');mjcf=tree.getroot();assets=mjcf.find('asset');world=mjcf.find('worldbody')
    for mesh in assets.findall('mesh'):mesh.set('file','robot/'+mesh.get('file'))
    base=next(b for b in world.findall('body') if b.get('name')=='base_link')
    base.set('pos',' '.join(map(str,config['base_position_m'])));base.set('quat',f"{math.cos(config['base_yaw_rad']/2)} 0 0 {math.sin(config['base_yaw_rad']/2)}")
    for geom in base.iter('geom'):geom.set('rgba','.35 .55 .8 1')
    (root/'scene_meshes').mkdir(exist_ok=True)
    for k,item in enumerate(config['instances']):
        mesh=trimesh.load(str(root/'scene'/item['collision_ref']),force='mesh',process=False)
        ref=f'scene_meshes/{item["instance_id"]}.obj';mesh.export(root/ref)
        ET.SubElement(assets,'mesh',name='mesh_'+item['instance_id'],file=ref)
        theta=item.get('yaw_rad',0)
        body=ET.SubElement(world,'body',name='environment_'+item['instance_id'],pos=' '.join(map(str,item['pose_xyz_m'])),quat=f'{math.cos(theta/2)} 0 0 {math.sin(theta/2)}')
        ET.SubElement(body,'geom',name='geom_'+item['instance_id'],type='mesh',mesh='mesh_'+item['instance_id'],rgba='.65 .65 .55 1')
    xmin,ymin,xmax,ymax=config['room_bounds_xy_m'];h=config['wall_height_m']
    for name,pos,size in [('floor',[(xmin+xmax)/2,(ymin+ymax)/2,-.05],[(xmax-xmin)/2,(ymax-ymin)/2,.05]),
        ('wall_left',[xmin-.05,(ymin+ymax)/2,h/2],[.05,(ymax-ymin)/2,h/2]),
        ('wall_right',[xmax+.05,(ymin+ymax)/2,h/2],[.05,(ymax-ymin)/2,h/2]),
        ('wall_bottom',[(xmin+xmax)/2,ymin-.05,h/2],[(xmax-xmin)/2,.05,h/2]),
        ('wall_top',[(xmin+xmax)/2,ymax+.05,h/2],[(xmax-xmin)/2,.05,h/2])]:
        ET.SubElement(world,'geom',name=name,type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),rgba='.9 .9 .9 1')
    tree.write(root/'scene.xml',encoding='utf-8',xml_declaration=True)


def main():
    import numpy as np,trimesh
    parser=argparse.ArgumentParser()
    for name in ('robot-bundle','scene-bundle','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('room','support','target'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--skip-candidates',type=int,default=0)
    a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic();audit(a.robot_bundle)
    from rfw_w.scenes.orchestrator.rigid_asset_tools import _verify_bundle
    _verify_bundle(a.scene_bundle)
    shutil.copytree(a.robot_bundle,a.output_dir/'robot');shutil.copytree(a.scene_bundle,a.output_dir/'scene')
    read=lambda ref:json.loads((a.scene_bundle/ref).read_text())
    items=[i for i in read('core_asset_ledger.json')['instances']+read('background_ledger.json')['instances'] if i['room_id']==a.room]
    support=next(i for i in items if i['role_id']==a.support);surface=support['support_surface'];target=next(i for i in items if i['role_id']==a.target)
    plan=read('geometry/house_plan.json');room=next(r for r in plan['rooms'] if r['id']==a.room)
    z=surface['plane_z_m'];base_mesh=trimesh.load(str(a.output_dir/'robot/meshes/base_link.stl'),force='mesh',process=False)
    polygon=surface['polygon_xy'];xs=[p[0] for p in polygon];ys=[p[1] for p in polygon]
    reference=json.loads((a.robot_bundle/'mujoco_runtime.json').read_text())['commands']
    config={'schema_version':'pvscene.single-room-installation/v1','room_id':a.room,'support_role_id':a.support,'target_role_id':a.target,
            'instances':items,'room_bounds_xy_m':room['bounds_xy_m'],'wall_height_m':plan['wall_height_m'],
            'reference_joints':reference,'original_task':'cross_room_task_not_executed','room_geometry':'closed rectangular wall/floor proxies'}
    candidates=[(float(x),float(y),yaw) for x in np.linspace(min(xs)+.12,max(xs)-.12,15) for y in np.linspace(min(ys)+.12,max(ys)-.12,9) for yaw in (0,math.pi/2,math.pi,-math.pi/2)]
    candidates.sort(key=lambda c:abs(math.dist(c[:2],target['pose_xyz_m'][:2])-.35))
    attempts=[];result=None
    admitted=0
    for x,y,yaw in candidates:
        config.update(base_position_m=[x,y,z+.002-float(base_mesh.bounds[0,2])],base_yaw_rad=yaw)
        try:
            layout=check_layout(a.output_dir,config);xml_scene(a.output_dir,config);collision=contacts(a.output_dir,config)
            if collision['penetrations']:raise ValueError('INSTALLATION_COLLISION')
            if admitted<a.skip_candidates:
                attempts.append({'base_position_m':list(config['base_position_m']),'base_yaw_rad':yaw,'failure_code':'SKIPPED_ADMITTED_INSTALLATION'})
                admitted+=1
                continue
            result={'verified':True,'layout':layout,'collision':collision};break
        except ValueError as error:attempts.append({'base_position_m':config['base_position_m'],'base_yaw_rad':yaw,'failure_code':str(error)})
    report={'status':'passed' if result else 'failed','failure_code':None if result else 'INSTALLATION_UNAVAILABLE',
            'accepted':result,'attempts':attempts,'elapsed_s':round(time.monotonic()-start,3),'ik':'not_run','grasp':'not_run','navigation':'unsupported_no_mobile_base'}
    (a.output_dir/'installation.json').write_text(json.dumps(config,indent=2)+'\n')
    (a.output_dir/'installation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    refs=['scene.xml','installation.json',*[p.relative_to(a.output_dir).as_posix() for directory in ('scene','robot','scene_meshes') for p in (a.output_dir/directory).rglob('*') if p.is_file()]]
    (a.output_dir/'installation_artifacts.json').write_text(json.dumps({r:digest(a.output_dir/r) for r in refs if (a.output_dir/r).is_file()},indent=2)+'\n')
    if result:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from rfw_w.scenes.validation.verify_single_room_installation import footprint
        fig,ax=plt.subplots(figsize=(7,6))
        for item in items:
            box,_=footprint(a.output_dir/'scene',item);ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False));ax.text((box[0]+box[2])/2,(box[1]+box[3])/2,item['category'],ha='center',fontsize=8)
        box=result['layout']['base_footprint_xy_m'];ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],color='steelblue'));ax.text(config['base_position_m'][0],config['base_position_m'][1]-.18,'Piper mount',ha='center')
        xmin,ymin,xmax,ymax=room['bounds_xy_m'];ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='world X (m)',ylabel='world Y (m)',title='Fixed Piper mounting / original object positions');ax.set_aspect('equal');fig.tight_layout();fig.savefig(a.output_dir/'layout.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('attempts','accepted')}));return 0 if result else 1
if __name__=='__main__':raise SystemExit(main())
