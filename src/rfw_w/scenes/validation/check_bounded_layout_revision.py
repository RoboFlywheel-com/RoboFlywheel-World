# SPDX-License-Identifier: Apache-2.0
"""Fresh-output deterministic local layout proposal/apply acceptance."""
import argparse
import json
from pathlib import Path
import time
from rfw_w.scenes.contracts.layout_revision import permission
from rfw_w.scenes.orchestrator.layout_revision_tools import (
    read, write, propose_layout_candidates, apply_layout_candidate,
)
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-bundle',type=Path,required=True)
    parser.add_argument('--target',default='cup')
    parser.add_argument('--operation',choices=['translate','rotate','change_support','move_component'],default='translate')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:
        graph=read_graph(args.scene_bundle)
        role=next(n for n in graph.to_dict()['nodes'] if n['id']==args.target)
        grant=permission(graph,args.target,operations=[args.operation],scope_ids=[role['room_id']])
        proposal=propose_layout_candidates(args.scene_bundle,grant,args.operation)
        write(args.output_dir,'candidates.json',proposal)
        if not proposal['candidates']:
            report={'status':'failed','failure_code':'LAYOUT_CANDIDATES_EXHAUSTED','rejected':proposal['rejected']}
        else:
            revised,checked=apply_layout_candidate(args.scene_bundle,args.output_dir/'delivery',proposal,
                proposal['candidates'][0]['candidate_id'],expected_hash=graph.content_hash)
            report={'status':'passed','graph_hash':revised.content_hash,'verification':checked}
    except (ValueError,OSError,KeyError,TypeError) as error:
        report={'status':'failed','failure_code':getattr(error,'code','LAYOUT_INPUT_INVALID'),'detail':str(error)}
    report.update(elapsed_s=round(time.monotonic()-started,3),model_requests=0)
    write(args.output_dir,'report.json',report);print(json.dumps(report))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
