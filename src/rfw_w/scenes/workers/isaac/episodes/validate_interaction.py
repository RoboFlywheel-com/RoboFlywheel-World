# SPDX-License-Identifier: Apache-2.0
"""Independent acceptance for cover, insertion and fallen-object recovery."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from rfw_w.scenes.workers.isaac.episodes.validate_scene import read_mesh,transform

def validate(cfg,out):
 r=json.loads((out/'trajectory.json').read_text());samples=r['samples'];contacts=r['contacts'];task=r['interaction_task'];root=Path(cfg['assets_root'])
 def mesh(leaf):
  d,c,_=leaf.split('_');return read_mesh(root/d/c/leaf/'mesh'/('m_'+leaf+'.obj'))[0]
 def axis(q,ax):
  from rfw_w.scenes.workers.isaac.episodes.validate_rollout import rotate
  return np.array(rotate(q,ax))
 def tilt(q,ax):return float(np.degrees(np.arccos(np.clip(axis(q,ax)[2],-1,1))))
 phase=lambda name:[s for s in samples if s['phase']==name]
 initial=phase('approach');hold=phase('hold');end=[s for s in phase('settle') if s['t']>=samples[-1]['t']-1];last=end[-1];tail=[c for c in contacts if c['t']>last['t']-1];hand=lambda c:any('/G1/' in a for a in c['actors'])
 src=r['asset_source'];vs=mesh(src['leaf']);startbottom=min(transform(vs,initial[-1])[:,2]);lift=min(min(transform(vs,s)[:,2])-startbottom for s in hold);slip=max(np.linalg.norm(np.array(s['position'])-hold[0]['position']) for s in hold);drift=max(np.linalg.norm(np.array(s['position'])-last['position']) for s in end);penetration=max([max(0,-c['separation']) for c in contacts]+[0.]);holdcontacts=[c for c in contacts if hold[0]['t']-1/r['fps']<=c['t']<=hold[-1]['t']]
 identities=[]
 for rec in [src]+r['scene_assets']:
  leaf=rec['leaf'];d,c,_=leaf.split('_');base=root/d/c/leaf
  identities.append((base/'_SUCCESS').exists() and rec['scale']==1 and rec['collision_shapes']>0 and all(hashlib.sha256((base/f).read_bytes()).hexdigest()==rec[k] for f,k in [(leaf+'.usd','source_usd_sha256'),('mesh/m_'+leaf+'.obj','source_mesh_sha256'),('part_physics.json','source_physics_sha256')]))
 dt=1/r['fps'];checks={'source_identity_preserved':all(identities),'joint_target_control_only':r.get('motion_control')=='joint_targets_only','physical_lift_over_5cm':lift>.05,'hand_contact_during_hold':any(hand(c) for c in holdcontacts),'hold_drift_below_15mm':slip<.015,'no_hand_contact_after_release':not any(hand(c) for c in tail),'final_drift_below_3mm':drift<.003,'contact_penetration_below_3mm':penetration<.003,'native_sample_cadence':abs(samples[0]['t']-dt)<1e-6 and all(abs(b['t']-a['t']-dt)<1e-6 for a,b in zip(samples,samples[1:])),'at_least_one_second_settled':end[-1]['t']-end[0]['t']>=1-dt-1e-6}
 metrics={};ax=r.get('upright_axis_local',[0,0,1])
 if task=='recover_dispenser':
  angles=[tilt(s['quaternion'],ax) for s in end];checks['initial_fallen_state_detected']=min(tilt(s['quaternion'],ax) for s in initial)>65;checks['final_upright_below_5deg']=max(angles)<5;checks['counter_contact_after_release']=any(any(a in ['/World/Counter','/World/SupportExtension'] for a in c['actors']) for c in tail)
  center=cfg.get('counter_center',[.45,0,.79]);size=cfg.get('counter_size',[.9,2.6,.08]);top=center[2]+size[2]/2;bottoms=[float(min(transform(vs,s)[:,2])) for s in end];checks['base_at_measured_counter_height']=max(abs(z-top) for z in bottoms)<.004;checks['returned_to_counter_work_area']=all(-.04<s['position'][0]<.16 and -.55<s['position'][1]<-.28 for s in end);metrics['initial_tilt_deg']=tilt(initial[-1]['quaternion'],ax);events=r.get('task_events',[]);checks['recovery_gated_by_pre_motion_pose']=any(e.get('event')=='fallen_detected' and e.get('t',1)<=0 and e.get('tilt_deg',0)>65 and abs(e['tilt_deg']-metrics['initial_tilt_deg'])<1 for e in events)
 elif task=='cover_bowl':
  goal='GoalBowl';record=next(v for v in r['scene_assets'] if v['name']==goal);gvs=mesh(record['leaf']);angles=[tilt(s['quaternion'],ax) for s in end];checks['lid_level_below_5deg']=max(angles)<5;checks['lid_supported_by_bowl']=any('/World/'+goal in c['actors'] for c in tail);checks['lid_not_resting_on_counter']=not any('/World/Counter' in c['actors'] for c in tail)
  offsets=[float(np.linalg.norm(np.array(s['position'])[:2]-np.array(s['scene_assets'][goal]['position'])[:2])) for s in end];checks['lid_centered_within_8mm']=max(offsets)<.008;metrics['lid_center_error_m']=max(offsets)
  # All angular directions of the rim must lie under the lid's horizontal footprint.
  from scipy.spatial import ConvexHull
  covered=[]
  for s in end[::5]:
   gv=transform(gvs,s['scene_assets'][goal]);rim=gv[gv[:,2]>gv[:,2].max()-.003];lv=transform(vs,s);h=ConvexHull(lv[:,:2]);covered.append(bool(np.all(rim[:,:2]@h.equations[:,:2].T+h.equations[:,2]<=.001)))
  checks['bowl_rim_covered_by_lid_footprint']=all(covered)
 elif task=='insert_pencil':
  goal='PenHolder';record=next(v for v in r['scene_assets'] if v['name']==goal);gvs=mesh(record['leaf']);depths=[];inside=[];angles=[min(tilt(s['quaternion'],[1,0,0]),180-tilt(s['quaternion'],[1,0,0])) for s in end]
  for s in end:
   gv=transform(gvs,s['scene_assets'][goal]);pv=transform(vs,s);top=gv[:,2].max();depths.append(float(top-pv[:,2].min()));embedded=pv[pv[:,2]<top-.003];center=np.array(s['scene_assets'][goal]['position'])[:2];inside.append(len(embedded)>0 and np.max(np.linalg.norm(embedded[:,:2]-center,axis=1))<.050 and pv[:,2].min()>gv[:,2].min()+.003)
  held=phase('support_wait')[-1];held_pv=transform(vs,held);held_gv=transform(gvs,held['scene_assets'][goal]);held_depth=float(held_gv[:,2].max()-held_pv[:,2].min());metrics['held_insertion_depth_m']=held_depth;checks['held_pen_crosses_opening_before_release']=held_depth>.02 and float(np.linalg.norm(held_pv[np.argmin(held_pv[:,2]),:2]-np.array(held['scene_assets'][goal]['position'])[:2]))<.045
  checks['insertion_depth_over_50mm']=min(depths)>.05;checks['inserted_through_holder_interior']=all(inside);checks['holder_support_contact']=any('/World/'+goal in c['actors'] for c in tail);checks['pen_not_on_counter']=not any('/World/Counter' in c['actors'] for c in tail);metrics['insertion_depth_m']=min(depths)
 else:raise ValueError('Unknown interaction task')
 for rec in r['scene_assets']:
  poses=[s['scene_assets'][rec['name']] for s in end];lastpose=poses[-1];stable=max(np.linalg.norm(np.array(p['position'])-lastpose['position']) for p in poses)<.003;disp=np.linalg.norm(np.array(lastpose['position'])-rec['settled_position']);checks['stable_'+rec['name']]=bool(stable);checks['context_undisturbed_'+rec['name']]=bool(disp<.010)
 result={'status':'passed' if all(checks.values()) else 'failed','checks':{k:bool(v) for k,v in checks.items()},'task':task,'lift_m':float(lift),'hold_slip_m':float(slip),'final_tilt_deg':angles[-1],'maximum_contact_penetration_m':float(penetration),'settled_drift_m':float(drift),'source_trajectory_sha256':hashlib.sha256((out/'trajectory.json').read_bytes()).hexdigest(),'frames':len(samples),'fps':r['fps'],'physics_hz':r['physics_hz'],'fixed_base':True,'torso_mechanically_locked':r.get('torso_mechanically_locked',False),'metrics':metrics,'scope':'Measured contact-driven rigid-body task; no locomotion, whole-body balance or hardware validation'}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cp=Path(a.config).resolve();cfg=json.loads(cp.read_text());out=cp.parent/cfg['output_dir'];report=validate(cfg,out);(out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
