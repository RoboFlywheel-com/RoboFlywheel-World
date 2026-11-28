#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove saved acceptance evidence rejects independent tampering without modifying its source."""
from __future__ import annotations
from copy import deepcopy
import argparse,json,sys,tempfile,shutil
from pathlib import Path
from rfw_w.scenes.validation.verify_adaptive_scene_delivery import verify_run
from rfw_w.scenes.contracts.task_world_graph import GraphError,TaskWorldGraph
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.orchestrator.background_tools import verify_background
from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
from rfw_w.scenes.assets.background_tools import screen_wall,measure_wall_mount_surfaces
from rfw_w.scenes.assets.room_tools import RoomAssetError


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);before=artifact_digest(a.source)
 events=json.loads((a.source/'events.json').read_text());report=json.loads((a.source/'report.json').read_text());results={}
 cases={}
 bad=deepcopy(events);bad[0]['decision']['tool_name']='finish';cases['unavailable_tool']=(bad,report,'TOOL_NOT_AVAILABLE')
 bad=deepcopy(events);bad[1]['source_graph_hash']='0'*64;cases['stale_graph']=(bad,report,'GRAPH_VERSION_CONFLICT')
 bad=deepcopy(events);bad[0]['transaction']['result']['tool_version']='invented/v1';cases['wrong_transaction']=(bad,report,'TOOL_NOT_AVAILABLE')
 bad=deepcopy(events);next(e for e in bad if e['decision']['tool_name']=='finish')['completed_background']=[];cases['false_state']=(bad,report,'AGENT_ACTION_INVALID')
 changed=deepcopy(report);changed['model_requests_used']+=1;cases['false_model_count']=(events,changed,'AGENT_ACTION_INVALID')
 changed=deepcopy(report);changed['budgets']={'max_tools':1,'max_model_requests':20,'deadline_s':1000};cases['over_budget']=(events,changed,'BUDGET_EXHAUSTED')
 for name,(candidate,changed,expected) in cases.items():
  with tempfile.TemporaryDirectory(prefix='pvscene_negative_') as d:
   root=Path(d)/'run';shutil.copytree(a.source,root,copy_function=lambda src,dst:Path(dst).symlink_to(Path(src).resolve()))
   for file,value in [('events.json',candidate),('report.json',changed)]:
    (root/file).unlink();(root/file).write_text(json.dumps(value))
   try:verify_run(root)
   except GraphError as e:results[name]={'code':e.code,'passed':e.code==expected}
   else:results[name]={'passed':False}
 root=a.source/report['assets_ref'];final=TaskWorldGraph.from_dict(json.loads((a.source/'graph.json').read_text()));source=TaskWorldGraph.from_dict(json.loads((root/'input_graph.json').read_text()))
 ledger=json.loads((root/'background_ledger.json').read_text());core=json.loads((root/'core_asset_ledger.json').read_text());plan=json.loads((root/'geometry/house_plan.json').read_text())
 for name,expected in [('core_moved','CORE_KEEP_OUT_VIOLATION'),('stale_ledger','GRAPH_VERSION_CONFLICT'),('wall_uncertified','WALL_MOUNT_UNCERTIFIED'),('fake_asset','ASSET_UNCERTIFIED'),('oversize','ASSET_UNCERTIFIED')]:
  bg=deepcopy(ledger);c=deepcopy(core)
  if name=='core_moved':c['instances'][0]['pose_xyz_m'][0]+=.5
  elif name=='stale_ledger':bg['source_graph_hash']='0'*64
  elif name=='wall_uncertified':next(i for i in bg['instances'] if i['placement_kind']=='wall')['mount_certificate']={}
  elif name=='fake_asset':bg['instances'][0]['provider']='fake'
  elif name=='oversize':bg['instances'][0]['dimensions_m'][0]=10
  bg.pop('content_hash');bg['content_hash']=content_hash(bg)
  try:verify_background(final,source,plan,c,bg,root)
  except RoomAssetError as e:results[name]={'code':e.code,'passed':e.code==expected}
  else:results[name]={'passed':False}
 # Tamper an immutable checkpoint payload while keeping its recorded digest.
 with tempfile.TemporaryDirectory(prefix='pvscene_checkpoint_negative_') as d:
  temp_root=Path(d)/'run';shutil.copytree(a.source,temp_root,copy_function=lambda src,dst:Path(dst).symlink_to(Path(src).resolve()))
  checkpoint=temp_root/'step_004_save_checkpoint/checkpoint.json';value=json.loads(checkpoint.read_text());value['graph_hash']='0'*64
  checkpoint.unlink();checkpoint.write_text(json.dumps(value))
  try:verify_run(temp_root)
  except GraphError as e:results['tampered_checkpoint']={'code':e.code,'passed':e.code=='CHECKPOINT_TAMPERED'}
  else:results['tampered_checkpoint']={'passed':False}
 # A shared wall span at the doorway cannot be repurposed as an installable surface.
 anchors=measure_wall_mount_surfaces(plan,root/'geometry');wall=next(i for i in ledger['instances'] if i['placement_kind']=='wall');candidate=deepcopy(wall)
 anchor=next(v for v in anchors.values() if v['room_id']==wall['room_id'] and v['wall_segment_id']=='wall_shared_low')
 n=anchor['normal'][0];candidate.update(mount_normal=anchor['normal'],yaw_rad=-1.5707963267948966 if n>0 else 1.5707963267948966)
 candidate['pose_xyz_m']=[anchor['plane_m']+n*(candidate['dimensions_m'][1]/2+.002),2.,1.2]
 try:screen_wall(candidate,anchor,plan,core,[])
 except RoomAssetError as e:results['door_opening']={'code':e.code,'passed':e.code=='WALL_OPENING_OVERLAP'}
 else:results['door_opening']={'passed':False}
 passed=all(c['passed'] for c in results.values()) and artifact_digest(a.source)==before
 out={'passed':passed,'negative_results':results,'source_unchanged':artifact_digest(a.source)==before}
 (a.output_dir/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
