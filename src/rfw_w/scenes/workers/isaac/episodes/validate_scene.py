# SPDX-License-Identifier: Apache-2.0
"""Independent source-identity, support-mesh and full-scene rollout checks."""
import argparse,hashlib,json,sys,math
from pathlib import Path
import numpy as np

from rfw_w.scenes.workers.isaac.episodes.validate_rollout import validate,rotate,distance


def read_mesh(path):
    vertices=[];triangles=[]
    for line in path.read_text().splitlines():
        v=line.split()
        if not v:continue
        if v[0]=='v':vertices.append([float(x) for x in v[1:4]])
        elif v[0]=='f':
            ids=[int(x.split('/')[0])-1 for x in v[1:]]
            triangles.extend([ids[0],ids[i],ids[i+1]] for i in range(1,len(ids)-1))
    return np.array(vertices),np.array(triangles,dtype=int)


def transform(vertices,pose):
    matrix=np.array([rotate(pose['quaternion'],v) for v in np.eye(3)]).T
    return vertices@matrix.T+pose['position']


def support_heights(triangles,points):
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    d=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    valid=np.abs(d)>1e-12;safe=np.where(valid,d,1.)
    heights=[]
    for x,y in points:
        u=((b[:,1]-c[:,1])*(x-c[:,0])+(c[:,0]-b[:,0])*(y-c[:,1]))/safe
        v=((c[:,1]-a[:,1])*(x-c[:,0])+(a[:,0]-c[:,0])*(y-c[:,1]))/safe
        mask=valid&(u>=-1e-6)&(v>=-1e-6)&(u+v<=1+1e-6)
        z=u*a[:,2]+v*b[:,2]+(1-u-v)*c[:,2]
        heights.append(float(z[mask].max()) if mask.any() else None)
    return heights


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cp=Path(a.config).resolve();cfg=json.loads(cp.read_text());out=cp.parent/cfg['output_dir'];r=json.loads((out/'trajectory.json').read_text())
    result=validate(r);checks=result['checks'];root=Path(cfg['assets_root'])
    def mesh_path(leaf):
        d,c,_=leaf.split('_');return root/d/c/leaf/'mesh'/('m_'+leaf+'.obj')
    source_checks=[]
    for record in r['scene_assets']+([r['asset_source']] if 'asset_source' in r else []):
        leaf=record['leaf'];d,c,_=leaf.split('_');src=root/d/c/leaf
        source_checks.append((src/'_SUCCESS').exists() and record['scale']==1. and record['collision_shapes']>0 and all(hashlib.sha256((src/file).read_bytes()).hexdigest()==record[key] for file,key in [(leaf+'.usd','source_usd_sha256'),('mesh/m_'+leaf+'.obj','source_mesh_sha256'),('part_physics.json','source_physics_sha256')]))
    checks['scene_assets_match_release_sources']=bool(source_checks) and all(source_checks)
    checks['task_asset_has_frozen_source_identity']='asset_source' in r
    target=next(v for v in r['scene_assets'] if v['name']=='Tray');verts,faces=read_mesh(mesh_path(target['leaf']));object_vertices,_=read_mesh(mesh_path(r['asset_leaf']))
    final=[s for s in r['samples'] if s['phase']=='settle' and s['t']>=r['samples'][-1]['t']-1.]
    gaps=[];coverage=[]
    for sample in final[::5]:
        support=transform(verts,sample['scene_assets']['Tray'])[faces];obj=transform(object_vertices,sample)
        base=obj[obj[:,2]<obj[:,2].min()+.0025];base=base[np.linspace(0,len(base)-1,min(24,len(base)),dtype=int)]
        heights=support_heights(support,base[:,:2]);coverage.append(all(z is not None for z in heights))
        gaps.extend(float(p[2]-z) for p,z in zip(base,heights) if z is not None)
    checks['object_base_covered_by_real_support_mesh']=all(coverage)
    checks['object_base_matches_real_support_surface']=bool(gaps) and max(abs(x) for x in gaps)<.010
    scene_stability={}
    for record in r['scene_assets']:
        name=record['name'];last=final[-1]['scene_assets'][name]
        drift=max(distance(s['scene_assets'][name]['position'],last['position']) for s in final)
        displacement=distance(record['settled_position'],last['position'])
        scene_stability[name]={'final_drift_m':drift,'task_displacement_m':displacement,'leaf':record['leaf']}
    checks['all_scene_assets_stable_at_end']=all(v['final_drift_m']<.003 for v in scene_stability.values())
    checks['context_assets_not_displaced']=all(v['task_displacement_m']<.01 for name,v in scene_stability.items() if name!='Tray')
    tray_final=transform(verts,final[-1]['scene_assets']['Tray']);tray_base=tray_final[tray_final[:,2]<tray_final[:,2].min()+.0025]
    center=cfg.get('counter_center',[.45,0,.79]);size=cfg.get('counter_size',[.9,2.6,.08]);top=center[2]+size[2]/2
    on_counter=(np.abs(tray_base[:,0]-center[0])<=size[0]/2+.002)&(np.abs(tray_base[:,1]-center[1])<=size[1]/2+.002)
    on_extension=(np.abs(tray_base[:,0]+.09)<=.092)&(np.abs(tray_base[:,1]+.22)<=.192)
    checks['real_support_stays_on_worktop']=bool(np.all(on_counter|on_extension)) and abs(float(tray_base[:,2].min())-top)<.010
    result.update(status='passed' if all(checks.values()) else 'failed',scene_id=r['scene_id'],task=r['task'],scene_stability=scene_stability,real_mesh_support_gap_range_m=[min(gaps),max(gaps)] if gaps else None,source_trajectory_sha256=hashlib.sha256((out/'trajectory.json').read_bytes()).hexdigest(),scope='Measured fixed-base contact pick/place with released collision shapes and dynamic scene assets; room/furniture are structural geometry.')
    (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
