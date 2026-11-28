# SPDX-License-Identifier: Apache-2.0
"""Create and edit real decorative content on a saved styled house."""
import argparse,json,sys,shutil
from pathlib import Path
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.contracts.scene_composition import migrate_covering_composition
from rfw_w.scenes.orchestrator.surface_covering_tools import edit_surface_covering,verify_surface_coverings,rollback_surface_covering
from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);source=a.source;g=read_graph(source)
    migration=a.output_dir/'migration';shutil.copytree(source,migration)
    g=migrate_covering_composition(g,expected_hash=g.content_hash);(migration/'graph.json').write_text(json.dumps(g.to_dict(),indent=2)+'\n');source=migration
    materials=json.loads((source/'surface_appearance.json').read_text())['materials'];mid=next(m for m,v in materials.items() if v['provider']=='local_pbr')
    reports=[]
    for instance,surface,position,size in [('cover_rug','floor_left:kitchen',[-2,3.25],[1.1,.8]),
                                         ('cover_poster','wall_top_right:study',[3.2,2.35],[.5,.4])]:
        out=a.output_dir/instance;g,r=edit_surface_covering(g,source,out,action='create',instance_id=instance,
            surface_id=surface,material_id='solid:sage' if instance=='cover_rug' else mid,size_m=size,local_position_m=position,expected_hash=g.content_hash)
        reports.append(r);source=out
    styled=source
    out=a.output_dir/'moved';g,r=edit_surface_covering(g,source,out,action='move',instance_id='cover_poster',local_position_m=[3.05,2.35],expected_hash=g.content_hash);reports.append(r);source=out
    out=a.output_dir/'restored';g,r=rollback_surface_covering(g,source,styled,out,instance_id='cover_poster',expected_hash=g.content_hash);reports.append(r);source=out
    out=a.output_dir/'removed';g,r=edit_surface_covering(g,source,out,action='remove',instance_id='cover_poster',expected_hash=g.content_hash);reports.append(r)
    relocated=a.output_dir/'relocated';shutil.copytree(styled,relocated)
    report={'status':'passed','transactions':reports,'styled_ref':styled.name,'relocated':verify_surface_coverings(relocated),
            'appearance':verify_surface_appearance(relocated),'render':'not_run','model_requests_used':0}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
