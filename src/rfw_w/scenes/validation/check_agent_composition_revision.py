#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh offline/online semantic patch acceptance with independent disk replay."""
from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path
import argparse
import json
import os
from pathlib import Path
import sys
import time
from rfw_w.scenes.validation.check_unified_scene_graph_roles import task_identity
from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.agents.composition_revision import propose_composition_patch, validate_composition_patch
from rfw_w.scenes.agents.llm_client import LLMConfig, LLMError
from rfw_w.scenes.agents.task_world_visualization import render_task_world_svg
from rfw_w.scenes.contracts.scene_composition import migrate_composition
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.orchestrator.composition_tools import commit_composition_patch


def patch_options():
    return [{'operations': [
        {'action': 'replace_inferred_category', 'target_id': 'desk', 'category': category,
         'reason_code': 'DESTINATION_SUPPORT_ALTERNATIVE'},
        {'action': 'add_optional_role', 'room_id': 'study', 'role_id': 'study_cabinet',
         'category': 'cabinet', 'placement_kind': 'floor', 'purpose': 'background'}
    ]} for category in ('table', 'desk')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--online', action='store_true')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    started=time.monotonic()
    # Separate task fixture: this user asks for a destination support without naming its type.
    # The original explicit desk request remains immutable and is tested separately.
    fixture=data_path('fixtures/task_world')
    request=json.loads((fixture/'cross_room_request.json').read_text())
    request['prompt']='Move the cup from the kitchen counter onto a suitable support in the study.'
    slots=json.loads((fixture/'cross_room_slots_response.json').read_text())['slots']
    slots['destination_support'].update(source='inferred',source_reason='user requests a suitable support without specifying category')
    old=compile_pick_place_slots(slots,request)
    graph=migrate_composition(old, expected_hash=old.content_hash)
    options=patch_options(); patch=options[0]; calls=0
    report={'status':'failed', 'model_requests_used':0, 'robot_validation':'not_run', 'source':'online' if args.online else 'replay'}
    try:
        if args.online:
            model=os.environ.get('PVSCENE_AGENT_MODEL') or os.environ.get('PVSCENE_VLM_MODEL')
            if not model:
                raise GraphError('MODEL_ENV_MISSING')
            config=LLMConfig(base_url=os.environ.get('PVSCENE_AGENT_BASE_URL') or os.environ.get('PVSCENE_VLM_BASE_URL') or os.environ.get('SIMFOUNDRY_OPENAI_BASE_URL',''),
                model=model, api_key_env=os.environ.get('PVSCENE_AGENT_API_KEY_ENV') or os.environ.get('PVSCENE_VLM_API_KEY_ENV','DASHSCOPE_API_KEY'),
                wire=os.environ.get('PVSCENE_AGENT_WIRE') or os.environ.get('PVSCENE_VLM_WIRE','chat_completions'), timeout_s=90, max_tokens=4096)
            if not config.credential_present():
                raise GraphError('MODEL_ENV_MISSING')
            if os.environ.get('PVSCENE_ALLOW_EXTERNAL_VLM')!='1':
                raise GraphError('EXTERNAL_CALL_NOT_ALLOWED')
            calls=1
            patch=propose_composition_patch({'goal':'Move cup from kitchen counter to study desk; inferred furniture category may vary while task IDs persist.'}, options, config)
        new=commit_composition_patch(graph,patch,args.output_dir/'accepted',expected_hash=graph.content_hash,scope_ids=('desk','study'))
        saved=TaskWorldGraph.from_dict(json.loads((args.output_dir/'accepted/graph.json').read_text()))
        replay=validate_composition_patch(graph,patch,expected_hash=graph.content_hash,scope_ids=('desk','study'))
        negatives={}
        cases={
            'target_change': {'operations':[{'action':'replace_inferred_category','target_id':'cup','category':'table','reason_code':'INVALID'}]},
            'scope':options[0], 'forged_built':{'operations':[], 'built':{'status':'passed'}},
            'stale_hash':options[0], 'incomplete':{'operations':[{'action':'add_optional_role'}]}}
        for name,candidate in cases.items():
            try:
                commit_composition_patch(graph,candidate,args.output_dir/name,
                    expected_hash='0'*64 if name=='stale_hash' else graph.content_hash,
                    scope_ids=() if name=='scope' else ('desk','study','cup'))
            except GraphError as error:
                negatives[name]=error.code
        expected={'target_change':'CORE_CONSTRAINT_CHANGED','scope':'PATCH_SCOPE_VIOLATION',
            'forged_built':'COMPOSITION_PATCH_INVALID','stale_hash':'GRAPH_VERSION_CONFLICT','incomplete':'COMPOSITION_PATCH_INVALID'}
        passed=task_identity(graph)==task_identity(new) and new.content_hash==saved.content_hash==replay.content_hash and negatives==expected
        report.update(status='passed' if passed else 'failed', graph_hash=new.content_hash, independent_replay=True,
            negatives=negatives,task_preserved=task_identity(graph)==task_identity(new),chosen_patch=patch)
        (args.output_dir/'scene_graph.svg').write_text(render_task_world_svg(new))
    except (GraphError,LLMError,SystemExit) as error:
        report.update(reason_code=getattr(error,'code','EXTERNAL_CALL_NOT_ALLOWED'))
        if report['reason_code'] in ('MODEL_ENV_MISSING','EXTERNAL_CALL_NOT_ALLOWED'):
            report['status']='blocked by environment'
    report.update(model_requests_used=calls,elapsed_s=round(time.monotonic()-started,3))
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report)); return 0 if report['status']=='passed' else 1
if __name__=='__main__':
    raise SystemExit(main())
