#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-library wall mounting checks against independently measured source OBJ."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from rfw_w.scenes.contracts.scene_composition import background_role
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.orchestrator.background_tools import source_bundle,run_background_tools,verify_background
from rfw_w.scenes.assets.background_tools import measure_wall_mount_surfaces,screen_wall
from rfw_w.scenes.assets.room_tools import RoomAssetError


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    graph,assets=source_bundle(a.source);raw=graph.to_dict()
    for room in ('kitchen','study'):
        role,edge=background_role(raw,room,room+'_mirror','mirror','wall')
        raw['nodes'].append(role);raw['edges'].append(edge)
    graph=graph.revise(raw,expected_hash=graph.content_hash)
    final,report=run_background_tools(graph,assets,a.output_dir/'assets','wall')
    root=a.output_dir/'assets';ledger=json.loads((root/'background_ledger.json').read_text());core=json.loads((root/'core_asset_ledger.json').read_text());plan=json.loads((root/'geometry/house_plan.json').read_text())
    saved=TaskWorldGraph.from_dict(json.loads((root/'graph.json').read_text()))
    verified=verify_background(saved,graph,plan,core,ledger,root)
    anchors=measure_wall_mount_surfaces(plan,root/'geometry');walls=[i for i in ledger['instances'] if i['placement_kind']=='wall']
    negatives={}
    if walls:
        item=walls[0];anchor=anchors[item['wall_anchor']]
        for name in ('reverse_normal','embedded','outside','collision'):
            value=json.loads(json.dumps(item))
            if name=='reverse_normal':value['mount_normal']=[-v for v in value['mount_normal']]
            if name=='embedded':value['pose_xyz_m'][0 if anchor['axis']=='x' else 1]=anchor['plane_m']
            if name=='outside':value['pose_xyz_m'][2]=3
            try:screen_wall(value,anchor,plan,core,[item] if name=='collision' else [])
            except RoomAssetError as e:negatives[name]=e.code
    expected={'reverse_normal':'WALL_SIDE_MISMATCH','embedded':'WALL_SIDE_MISMATCH','outside':'WALL_OPENING_OVERLAP','collision':'WALL_FIXTURE_COLLISION'}
    status='passed' if len(walls)>=1 and negatives==expected else 'failed'
    result={'status':status,'wall_instances':len(walls),'wall_surfaces':len(anchors),'verification':verified,
            'negative_results':negatives,'mounts':[{'role_id':i['role_id'],'anchor':i['wall_anchor'],'pose_xyz_m':i['pose_xyz_m'],'normal':i['mount_normal']} for i in walls]}
    (a.output_dir/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0 if status=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
