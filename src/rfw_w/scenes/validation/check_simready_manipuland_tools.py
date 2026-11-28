# SPDX-License-Identifier: Apache-2.0
"""Real v1.3 retrieval and prepared mesh verification (no model calls)."""
import argparse,json,sys
from pathlib import Path
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.assets.rigid_tool_retrieval import search_rigid_assets,prepare_selected
from rfw_w.scenes.geometry.rigid_tool_assets import verify_prepared_rigid


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);g=read_graph(a.source)
    q=search_rigid_assets(g,'cup','a small coffee mug for picking and placing')
    (a.output_dir/'query.json').write_text(json.dumps(q,indent=2)+'\n')
    if not q['candidates']:
        report={'status':'failed','failure_code':q['failure_code'],'model_requests_used':0}
    else:
        m=prepare_selected(q,q['candidates'][0]['candidate_key'],a.output_dir)
        from rfw_w.scenes.orchestrator.rigid_asset_tools import select_rigid_asset
        mixed=a.output_dir/'mixed';g2,assembly=select_rigid_asset(g,a.source,mixed,q,q['candidates'][0]['candidate_key'],expected_hash=g.content_hash)
        warm=search_rigid_assets(g,'cup','a small coffee mug for picking and placing')
        (a.output_dir/'warm_query.json').write_text(json.dumps(warm,indent=2)+'\n')
        report={'status':'passed','verification':verify_prepared_rigid(a.output_dir,m),'metadata':m,
                'cold_s':q['elapsed_s'],'warm_s':warm['elapsed_s'],'candidates':len(q['candidates']),
                'model_requests_used':0,'assembly':assembly}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
