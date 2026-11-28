# SPDX-License-Identifier: Apache-2.0
"""Real local PBR binding, shared-wall override and room-scoped restoration."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph,set_surface_materials,rollback_surface_appearance
from rfw_w.scenes.assets.surface_materials import search_surface_materials
from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    source=a.source/'assets' if (a.source/'assets').is_dir() else a.source
    if (source/'report.json').exists() and not (source/'geometry').exists():
        info=json.loads((source/'report.json').read_text());source=source/info['assets_ref']
    g=read_graph(source);wood=search_surface_materials('wood oak floor','floor');plaster=search_surface_materials('plaster wall','wall_set')
    reports=[];queries=(wood,plaster)
    (a.output_dir/'queries.json').write_text(json.dumps(queries,indent=2)+'\n')
    for room in [n['id'] for n in g.to_dict()['nodes'] if n['kind']=='room']:
        out=a.output_dir/('styled_'+room)
        g,r=set_surface_materials(g,source,out,room_id=room,floor_material_id=wood['candidates'][0]['material_id'],
            wall_material_id=plaster['candidates'][0]['material_id'],meters_per_tile=1,
            expected_hash=g.content_hash,description='wood floor and plaster wall',queries=queries)
        source=out;reports.append(r)
    saved=source
    room=next(n['id'] for n in g.to_dict()['nodes'] if n['kind']=='room')
    surface='wall_shared_high:'+room
    q=search_surface_materials('sage','wall_set');out=a.output_dir/'wall_override'
    g,r=set_surface_materials(g,source,out,room_id=room,wall_material_id='solid:sage',wall_surface_id=surface,
        expected_hash=g.content_hash,description='sage shared wall side',queries=(q,));reports.append(r);source=out
    styled=source
    out=a.output_dir/'restored';g,r=rollback_surface_appearance(g,source,saved,out,room_id=room,expected_hash=g.content_hash);reports.append(r)
    relocated=a.output_dir/'relocated';shutil.copytree(styled,relocated)
    independent=verify_surface_appearance(relocated)
    report={'status':'passed','transactions':reports,'styled_ref':styled.relative_to(a.output_dir).as_posix(),
            'migration_verification':independent,'model_requests_used':0,'render':'not_run','user_visual':'not_run'}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0
if __name__=='__main__':raise SystemExit(main())
