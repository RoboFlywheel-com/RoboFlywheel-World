#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Immutable checkpoint replay on saved real core graph and synthetic failed intent."""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
from rfw_w.scenes.orchestrator.background_tools import source_bundle
from rfw_w.scenes.orchestrator.scene_checkpoints import save_scene_checkpoint,rollback_scene_scope,read_checkpoint
from rfw_w.scenes.contracts.scene_composition import background_role
from rfw_w.scenes.contracts.task_world_graph import GraphError,TaskWorldGraph


def directory_digest(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False); before=directory_digest(a.source)
    old,assets=source_bundle(a.source);cp=save_scene_checkpoint(old,a.output_dir/'checkpoint')
    raw=old.to_dict();role,edge=background_role(raw,'study','failed_cabinet','cabinet','floor');raw['nodes'].append(role);raw['edges'].append(edge)
    current=old.revise(raw,expected_hash=old.content_hash)
    new,report=rollback_scene_scope(current,a.output_dir/'checkpoint',a.output_dir/'rollback',expected_hash=current.content_hash,scope_ids=('failed_cabinet',),reason_code='CORE_KEEP_OUT_VIOLATION',checkpoint_hash=cp)
    saved=TaskWorldGraph.from_dict(json.loads((a.output_dir/'rollback/graph.json').read_text()))
    negatives={}
    for name,scope,expected in [('core',('cup',),current.content_hash),('stale',('failed_cabinet',),'0'*64),('scope',('door',),current.content_hash)]:
        try:rollback_scene_scope(current,a.output_dir/'checkpoint',a.output_dir/name,expected_hash=expected,scope_ids=scope,reason_code='TEST')
        except GraphError as e:negatives[name]=e.code
    try:new.check_evidence({'graph_hash':old.content_hash})
    except GraphError as e:negatives['evidence']=e.code
    report.update(status='passed' if saved.content_hash==new.content_hash and before==directory_digest(a.source) and len({old.content_hash,current.content_hash,new.content_hash})==3 else 'failed',
                  old_directory_unchanged=before==directory_digest(a.source),negative_results=negatives,
                  real_placement_failure='not_run',retry='not_run')
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
