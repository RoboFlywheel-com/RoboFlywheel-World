# SPDX-License-Identifier: Apache-2.0
"""Run actual saved-bundle verifiers against adversarially changed evidence copies."""
import argparse,json,shutil,sys
from pathlib import Path
from rfw_w.scenes.validation.verify_mixed_provider_assembly import verify
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance


def main():
    p=argparse.ArgumentParser();p.add_argument('--mixed',type=Path,required=True);p.add_argument('--coverings',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);results={}
    for name in ('provider','collision','origin','mass','metadata_escape'):
        out=a.output_dir/name;shutil.copytree(a.mixed,out)
        ledger=json.loads((out/'core_asset_ledger.json').read_text());item=next(i for i in ledger['instances'] if i['provider']=='simready')
        if name=='provider':item['provider']='unknown'
        if name=='collision':(out/item['collision_ref']).write_text('v 0 0 0\n')
        if name in ('origin','mass'):
            path=out/item['metadata_ref'];m=json.loads(path.read_text())
            if name=='origin':m['origin_convention']='center'
            else:m['physics']['mass_kg']=0
            path.write_text(json.dumps(m))
        if name=='metadata_escape':item['metadata_ref']='../outside.json'
        ledger.pop('content_hash');ledger['content_hash']=content_hash(ledger)
        (out/'core_asset_ledger.json').write_text(json.dumps(ledger))
        try:verify(out)
        except (ValueError,OSError,KeyError,TypeError) as e:results[name]={'rejected':True,'code':getattr(e,'code','ASSET_LEDGER_MISMATCH')}
        else:results[name]={'rejected':False}
    for name in ('covering_normal','covering_corridor','covering_texture'):
        out=a.output_dir/name;shutil.copytree(a.coverings,out)
        ledger=json.loads((out/'surface_coverings.json').read_text())
        item=ledger['instances'][0]
        if name=='covering_normal':item['normal']=[0,0,-1]
        if name=='covering_corridor':
            from rfw_w.scenes.geometry.surface_coverings import covering_vertices
            graph=json.loads((out/'graph.json').read_text());s=graph['built']['surface_coverings']['anchors'][item['surface_id']]
            item['local_position_m']=[-2,2];item['vertices_m']=covering_vertices(s,item['local_position_m'],item['size_m'])
        if name=='covering_texture':ledger['materials'][item['material_id']]['base_color_rgba']=[-1,1,1,1]
        ledger.pop('content_hash');ledger['content_hash']=content_hash(ledger)
        graph=json.loads((out/'graph.json').read_text());graph.pop('content_hash');graph['built']['surface_coverings']['ledger_hash']=ledger['content_hash']
        (out/'graph.json').write_text(json.dumps(graph));(out/'surface_coverings.json').write_text(json.dumps(ledger))
        try:verify_surface_coverings(out)
        except (ValueError,OSError,KeyError,TypeError) as e:results[name]={'rejected':True,'code':getattr(e,'code','COVERING_MATERIAL_INVALID')}
        else:results[name]={'rejected':False}
    report={'status':'passed' if all(r['rejected'] for r in results.values()) else 'failed','cases':results,'rejected':sum(r['rejected'] for r in results.values()),'model_requests_used':0}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
