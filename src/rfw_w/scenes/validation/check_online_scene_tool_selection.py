# SPDX-License-Identifier: Apache-2.0
"""Budgeted online rigid candidate selection over an existing verified static bundle."""
import argparse,json,os,sys,time
from pathlib import Path
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.assets.rigid_tool_retrieval import search_rigid_assets
from rfw_w.scenes.orchestrator.rigid_asset_tools import select_rigid_asset
from rfw_w.scenes.agents.llm_client import LLMConfig,call_vision,require_external_permission


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);g=read_graph(a.source);start=time.monotonic()
    cfg=LLMConfig(base_url=os.environ.get('PVSCENE_AGENT_BASE_URL') or os.environ.get('PVSCENE_VLM_BASE_URL') or os.environ.get('SIMFOUNDRY_OPENAI_BASE_URL',''),
        model=os.environ.get('PVSCENE_AGENT_MODEL') or os.environ.get('PVSCENE_VLM_MODEL',''),api_key_env=os.environ.get('PVSCENE_AGENT_API_KEY_ENV') or os.environ.get('PVSCENE_VLM_API_KEY_ENV','DASHSCOPE_API_KEY'),
        wire=os.environ.get('PVSCENE_AGENT_WIRE') or os.environ.get('PVSCENE_VLM_WIRE','chat_completions'),timeout_s=90,max_tokens=8192)
    calls=0
    try:
        require_external_permission();calls+=1
        response=call_vision(cfg,system_prompt='Return only JSON {"tool_name":"search_rigid_assets","parameters":{"role_id":"cup","description":"short task object description"}}. Task: pick a coffee mug from kitchen counter and place it on study desk.',user_prompt='Select the rigid asset search tool and formulate its semantic query. No filesystem paths.')
        text=response.text.strip()
        if text.startswith('```'):text=text[text.find('\n')+1:-3].strip()
        decision=json.loads(text)
        if decision.get('tool_name')!='search_rigid_assets' or set(decision.get('parameters',{}))!={'role_id','description'}:raise ValueError('invalid query decision')
        q=search_rigid_assets(g,**decision['parameters']);(a.output_dir/'query.json').write_text(json.dumps(q,indent=2)+'\n')
        (a.output_dir/'search_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        calls+=1
        response=call_vision(cfg,system_prompt='Return only JSON {"tool_name":"select_rigid_asset","candidate_key":"one exact provided candidate_key"}. Choose a suitable small coffee mug. Do not invent a candidate.',user_prompt=json.dumps({'task':'pick coffee mug from kitchen counter and place on study desk','candidates':q['candidates']}))
        text=response.text.strip()
        if text.startswith('```'):text=text[text.find('\n')+1:-3].strip()
        selection=json.loads(text)
        if selection.get('tool_name')!='select_rigid_asset':raise ValueError('invalid selection')
        (a.output_dir/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        g2,assembly=select_rigid_asset(g,a.source,a.output_dir/'mixed',q,selection['candidate_key'],expected_hash=g.content_hash)
        report={'status':'passed','model_requests_used':calls,'assembly':assembly,'elapsed_s':round(time.monotonic()-start,3)}
    except (ValueError,KeyError,OSError,RuntimeError,SystemExit) as e:
        report={'status':'failed','model_requests_used':calls,'failure_code':getattr(e,'code','AGENT_ACTION_INVALID'),'error_type':type(e).__name__,'elapsed_s':round(time.monotonic()-start,3)}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
