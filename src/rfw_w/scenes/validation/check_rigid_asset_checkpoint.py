# SPDX-License-Identifier: Apache-2.0
"""Real target swap and checkpoint restoration; preserves all prior run evidence."""
import argparse,json,sys,os,shutil
from pathlib import Path
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
from rfw_w.scenes.assets.rigid_tool_retrieval import search_rigid_assets
from rfw_w.scenes.orchestrator.rigid_asset_tools import save_rigid_checkpoint,select_rigid_asset,restore_rigid_asset


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    # Deployment loader does not expose values and uses only existing local configuration.
    for line in Path('.env.local').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key,_,value=line.partition('=');os.environ.setdefault(key.strip(),value.strip().strip('\"\''))
    before=artifact_digest(a.source);g=read_graph(a.source)
    cp=save_rigid_checkpoint(g,a.source,a.output_dir/'checkpoint',role_id='cup',expected_hash=g.content_hash)
    query=search_rigid_assets(g,'cup','a small coffee mug for picking and placing')
    (a.output_dir/'query.json').write_text(json.dumps(query,indent=2)+'\n')
    old=next(i for i in json.loads((a.source/'core_asset_ledger.json').read_text())['instances'] if i['role_id']=='cup')
    candidate=next(c for c in query['candidates'] if c['asset_id']!=old['asset_id'])
    changed,selection=select_rigid_asset(g,a.source,a.output_dir/'changed',query,candidate['candidate_key'],expected_hash=g.content_hash)
    restored,result=restore_rigid_asset(changed,a.output_dir/'changed',a.output_dir/'checkpoint',a.source,a.output_dir/'restored',expected_hash=changed.content_hash)
    assert artifact_digest(a.source)==before
    assert restored.to_dict()['parent_hash']==changed.content_hash
    negatives={}
    for name in ('stale','checkpoint'):
        try:
            if name=='checkpoint':
                shutil.copytree(a.output_dir/'checkpoint',a.output_dir/'tampered_checkpoint')
                bad=dict(cp);bad['role_id']='desk';(a.output_dir/'tampered_checkpoint'/'checkpoint.json').write_text(json.dumps(bad))
            restore_rigid_asset(changed,a.output_dir/'changed',a.output_dir/('tampered_checkpoint' if name=='checkpoint' else 'checkpoint'),a.source,a.output_dir/('rejected_'+name),expected_hash=g.content_hash if name=='stale' else changed.content_hash)
        except ValueError as e:negatives[name]=getattr(e,'code',type(e).__name__)
        else:raise AssertionError('negative accepted: '+name)
    report={'status':'passed','source_unchanged':True,'old_asset_id':old['asset_id'],'changed_asset_id':candidate['asset_id'],
            'restore':result,'negative_rejections':negatives,'model_requests_used':0}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
