# SPDX-License-Identifier: Apache-2.0
"""Replace an existing optional plate and probe bowl/jar preparation without claiming placement."""
import argparse,json,os,sys,time
from pathlib import Path
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.assets.rigid_tool_retrieval import search_rigid_assets,prepare_selected
from rfw_w.scenes.orchestrator.rigid_asset_tools import select_rigid_asset,_verify_bundle


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    for line in Path('.env.local').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key,_,value=line.partition('=');os.environ.setdefault(key.strip(),value.strip().strip('\"\''))
    graph=read_graph(a.source);core=json.loads((a.source/'core_asset_ledger.json').read_text())
    role=next(n for n in graph.to_dict()['nodes'] if n['kind']=='asset_role' and n['category']=='plate' and n['priority']=='optional')
    q=search_rigid_assets(graph,role['id'],'a dinner plate on the kitchen counter')
    (a.output_dir/'plate_query.json').write_text(json.dumps(q,indent=2)+'\n')
    if not q['candidates']:
        report={'status':'failed','failure_code':q['failure_code'],'plate_placement':'not_run','model_requests_used':0}
        (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    revised,result=select_rigid_asset(graph,a.source,a.output_dir/'plate_scene',q,q['candidates'][0]['candidate_key'],expected_hash=graph.content_hash)
    actual=json.loads((a.output_dir/'plate_scene'/'core_asset_ledger.json').read_text())
    assert core['goal_placement']==actual['goal_placement']
    assert [i for i in core['instances'] if i['role_id']!=role['id']]==[i for i in actual['instances'] if i['role_id']!=role['id']]
    _verify_bundle(a.output_dir/'plate_scene')
    probes={}
    for category,description in [('bowl','a soup bowl'),('jar','a condiment jar')]:
        # Standalone intent variant for resource measurement; never committed into the delivered scene.
        raw=graph.to_dict()
        for n in raw['nodes']:
            if n['id']==role['id']:n['category']=category
        raw['built']={};raw['evidence_refs']=[]
        raw['nodes']=[n for n in raw['nodes'] if n['kind']!='asset_instance']
        ids={n['id'] for n in raw['nodes']};raw['edges']=[e for e in raw['edges'] if e['source_id'] in ids and e['target_id'] in ids]
        intent=graph.revise(raw,expected_hash=graph.content_hash)
        directory=a.output_dir/category;directory.mkdir()
        (directory/'intent_graph.json').write_text(json.dumps(intent.to_dict(),indent=2)+'\n')
        query=search_rigid_assets(intent,role['id'],description)
        (directory/'query.json').write_text(json.dumps(query,indent=2)+'\n')
        if query['candidates']:
            metadata=prepare_selected(query,query['candidates'][0]['candidate_key'],directory)
            probes[category]={'status':'prepared','metadata':metadata,'candidates':len(query['candidates']),'placement':'not_run'}
        else:probes[category]={'status':'failed','failure_code':query['failure_code'],'placement':'not_run'}
    report={'status':'passed','plate_assembly':result,'other_instances_unchanged':True,'goal_unchanged':True,
            'resource_probes':probes,'model_requests_used':0,'elapsed_s':round(time.monotonic()-started,3)}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
