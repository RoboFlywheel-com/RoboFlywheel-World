#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-library floor furnishing acceptance preserving the saved core bundle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from rfw_w.scenes.contracts.scene_composition import background_role
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.orchestrator.background_tools import source_bundle,run_background_tools,verify_background
from rfw_w.scenes.assets.background_tools import screen_floor
from rfw_w.scenes.assets.room_tools import RoomAssetError
from rfw_w.scenes.contracts.hashing import content_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    graph,assets=source_bundle(a.source);raw=graph.to_dict()
    for room in ('kitchen','study'):
        for category in ('cabinet','chair'):
            role,edge=background_role(raw,room,room+'_'+category,category,'floor')
            raw['nodes'].append(role);raw['edges'].append(edge)
    graph=graph.revise(raw,expected_hash=graph.content_hash)
    final,report=run_background_tools(graph,assets,a.output_dir/'assets','floor')
    ledger=json.loads((a.output_dir/'assets/background_ledger.json').read_text()); core=json.loads((a.output_dir/'assets/core_asset_ledger.json').read_text())
    plan=json.loads((a.output_dir/'assets/geometry/house_plan.json').read_text())
    saved=TaskWorldGraph.from_dict(json.loads((a.output_dir/'assets/graph.json').read_text()))
    verified=verify_background(saved,graph,plan,core,ledger,a.output_dir/'assets')
    counts={r:sum(i['room_id']==r for i in ledger['instances']) for r in ('kitchen','study')}
    negatives={}
    item=ledger['instances'][0]
    for name,pose in [('out_of_bounds',[20,20,0]),('door',[sum(plan['corridor']['x_span_m'])/2,2,0]),('collision',item['pose_xyz_m'])]:
        candidate=json.loads(json.dumps(item));candidate['pose_xyz_m']=pose
        try:screen_floor(candidate,plan,core,[item] if name=='collision' else [])
        except RoomAssetError as e:negatives[name]=e.code
    expected={'out_of_bounds':'FURNISHING_OUT_OF_BOUNDS','door':'FURNISHING_OUT_OF_BOUNDS','collision':'FURNISHING_COLLISION'}
    # Choose a point inside the correct room's corridor for the obstruction negative.
    candidate=json.loads(json.dumps(item)); candidate['pose_xyz_m']=[-2 if item['room_id']=='kitchen' else 2,2,0]
    try:screen_floor(candidate,plan,core,[])
    except RoomAssetError as e:negatives['door']=e.code
    expected['door']='CORE_KEEP_OUT_VIOLATION'
    tampered=json.loads(json.dumps(ledger));tampered['instances'][0]['asset_id']='fake'
    tampered.pop('content_hash');tampered['content_hash']=content_hash(tampered)
    try:verify_background(saved,graph,plan,core,tampered,a.output_dir/'assets')
    except RoomAssetError as e:negatives['fake_asset']=e.code
    expected['fake_asset']='ASSET_UNCERTIFIED'
    status='passed' if min(counts.values())>=2 and negatives==expected else 'failed'
    acceptance={'status':status,'counts':counts,'verification':verified,'negative_results':negatives,'expected_negatives':expected,
                'robot_validation':'not_run','model_requests_used':0}
    (a.output_dir/'report.json').write_text(json.dumps(acceptance,indent=2)+'\n');print(json.dumps(acceptance));return 0 if status=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
