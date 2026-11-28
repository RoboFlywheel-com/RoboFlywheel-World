# SPDX-License-Identifier: Apache-2.0
"""Independent mixed asset verifier, without reading source libraries or model calls."""
import argparse,json,sys
from pathlib import Path
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.assets.room_tools import verify_scene_assembly
from rfw_w.scenes.contracts.surface_appearance import require


def verify(root):
    read=lambda n:json.loads((root/n).read_text())
    graph=TaskWorldGraph.from_dict(read('graph.json'));frozen=TaskWorldGraph.from_dict(read('core_source_graph.json'))
    core=read('core_asset_ledger.json');plan=read('geometry/house_plan.json')
    result=verify_scene_assembly(frozen,plan,core,root)
    if (root/'optional_prior_graph.json').exists():
        from rfw_w.scenes.geometry.optional_rigid_scope import verify_optional_extension,verify_optional_ledger
        prior_graph=TaskWorldGraph.from_dict(read('optional_prior_graph.json'))
        prior_frozen=TaskWorldGraph.from_dict(read('optional_prior_core_source_graph.json'))
        intent=TaskWorldGraph.from_dict(read('intent_graph.json'))
        addition=read('optional_rigid_report.json');rid=addition['role_id'];sid=addition['surface_id']
        verify_optional_extension(prior_graph.to_dict(),intent.to_dict(),rid,sid)
        verify_optional_extension(prior_frozen.to_dict(),frozen.to_dict(),rid,sid)
        prior_core=read('optional_prior_core_ledger.json')
        verify_scene_assembly(prior_frozen,plan,prior_core,root)
        verify_optional_ledger(prior_core,core,rid)
        from rfw_w.scenes.contracts.hashing import content_hash
        query=read('optional_rigid_query.json')
        require(query['source_graph_hash']==intent.content_hash
                and content_hash({k:v for k,v in query.items() if k!='query_id'})==query['query_id'],
                'RIGID_QUERY_STALE','optional query provenance')
        require(addition['selected_candidate_key'] in {c['candidate_key'] for c in query['candidates']},
                'RIGID_CANDIDATE_NOT_ADMITTED','optional selected candidate')
        current=graph.to_dict();intent_raw=intent.to_dict()
        require([n for n in current['nodes'] if n['id']!='inst_'+rid]==intent_raw['nodes'],
                'ROLLBACK_CORE_VIOLATION','optional graph nodes')
        require([e for e in current['edges'] if e['target_id']!='inst_'+rid]==intent_raw['edges'],
                'ROLLBACK_CORE_VIOLATION','optional graph edges')
        require(current['parent_hash']==intent.content_hash and intent_raw['parent_hash']==prior_graph.content_hash,
                'GRAPH_VERSION_CONFLICT','optional version chain')


    by_id={n['id']:n for n in graph.to_dict()['nodes']}
    for i in core['instances']:
        expected=('simready:'+i['library_version']+':' if i['provider']=='simready' else 'hssd:')+i['asset_id']
        require(by_id[i['instance_id']]['asset_ref']==expected,'ASSET_LEDGER_MISMATCH','instance provider reference')
    require(graph.to_dict()['built']['asset_assembly']['ledger_hash']==core['content_hash'],
            'ASSET_LEDGER_MISMATCH','graph core')
    return {'verified':True,'problems':[],'providers':sorted({i['provider'] for i in core['instances']}),
            'assembly':result,'robot':'not_run'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--delivery',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try:r=verify(a.delivery)
    except (ValueError,OSError,KeyError,TypeError) as e:r={'verified':False,'problems':[{'code':getattr(e,'code','ASSET_LEDGER_MISMATCH'),'type':type(e).__name__}]}
    (a.output_dir/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));return 0 if r['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
