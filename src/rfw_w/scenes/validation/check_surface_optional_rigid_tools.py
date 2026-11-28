# SPDX-License-Identifier: Apache-2.0
"""Real bowl/jar additions on an existing support; no model requests."""
import argparse,json,os,sys,time
from pathlib import Path
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.orchestrator.scene_checkpoints import artifact_digest
from rfw_w.scenes.orchestrator.surface_optional_rigid_tools import add_surface_rigid_asset


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--categories',default='bowl,jar');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    for line in Path('.env.local').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key,_,value=line.partition('=');os.environ.setdefault(key.strip(),value.strip().strip('\"\''))
    source=a.source;before=artifact_digest(source);g=read_graph(source);results={}
    for category in a.categories.split(','):
        target=a.output_dir/category
        g,result=add_surface_rigid_asset(g,source,target,surface_id='rc_kitchen_source_surface',role_id='optional_'+category,
            category=category,description='a '+category+' on the kitchen counter',expected_hash=g.content_hash)
        results[category]=result;source=target
    assert artifact_digest(a.source)==before
    report={'status':'passed','additions':results,'source_unchanged':True,'model_requests_used':0,'elapsed_s':round(time.monotonic()-start,3)}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
