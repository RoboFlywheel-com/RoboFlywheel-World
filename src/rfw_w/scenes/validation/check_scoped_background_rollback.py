#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real scoped restoration retains the other room and wall instances byte for byte."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.orchestrator.scene_checkpoints import rollback_scene_scope,restore_background_artifacts,artifact_digest
from rfw_w.scenes.orchestrator.background_tools import run_background_tools,verify_background


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);snapshot=artifact_digest(a.source)
 report=json.loads((a.source/'report.json').read_text());assets=a.source/report['assets_ref'];current=TaskWorldGraph.from_dict(json.loads((a.source/'graph.json').read_text()))
 before=json.loads((assets/'background_ledger.json').read_text());scope=('study_cabinet',)
 cp=a.source/'step_004_save_checkpoint';cp_assets=a.source/'step_003_assemble_core'
 new,rollback=rollback_scene_scope(current,cp,a.output_dir/'rollback',expected_hash=current.content_hash,scope_ids=scope,reason_code='FURNISHING_COLLISION')
 restore=restore_background_artifacts(assets,cp_assets,a.output_dir/'restored_assets',new,scope_ids=scope)
 restored=json.loads((a.output_dir/'restored_assets/background_ledger.json').read_text())
 kept=[i for i in before['instances'] if i['role_id'] not in scope]
 unchanged=restored['instances']==kept
 final,rerun=run_background_tools(new,a.output_dir/'restored_assets',a.output_dir/'retry_assets','floor')
 root=a.output_dir/'retry_assets';ledger=json.loads((root/'background_ledger.json').read_text());core=json.loads((root/'core_asset_ledger.json').read_text());plan=json.loads((root/'geometry/house_plan.json').read_text())
 checked=verify_background(final,new,plan,core,ledger,root)
 kept_final=[i for i in ledger['instances'] if i['role_id'] not in scope]
 passed=unchanged and kept_final==kept and checked['verified'] and artifact_digest(a.source)==snapshot
 result={'passed':passed,'scope_ids':scope,'retained_instances':len(kept),'retained_wall_instances':sum(i['placement_kind']=='wall' for i in kept),
         'other_room_and_wall_unchanged':unchanged and kept_final==kept,'source_unchanged':artifact_digest(a.source)==snapshot,
         'restoration_verification':restore,'retry_verification':checked,'invalidated_evidence':rollback['invalidated_evidence']}
 (a.output_dir/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
