# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
from pathlib import Path
from rfw_w.scenes.agents.llm_client import LLMError
from rfw_w.scenes.orchestrator.scene_critic_tools import critique_scene,ModelBudget
from rfw_w.scenes.orchestrator.surface_appearance_tools import read_graph
from rfw_w.scenes.orchestrator.layout_revision_tools import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--robot-delivery',type=Path)
    p.add_argument('--visual',action='store_true');p.add_argument('--render-manifest',type=Path)
    p.add_argument('--model-budget',type=int,default=1);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    graph=read_graph(a.scene_bundle);profiles=['static'];config=None;executor=None
    if a.robot_delivery:
        from rfw_w.scenes.validation.verify_agent_robot_layout import verify
        profiles.append('robot_layout');executor=verify
    if a.visual:
        from rfw_w.scenes.agents.llm_client import LLMConfig
        profiles.append('visual')
        config=LLMConfig(base_url=os.environ.get('PVSCENE_VLM_BASE_URL') or os.environ.get('PVSCENE_AGENT_BASE_URL') or os.environ.get('SIMFOUNDRY_OPENAI_BASE_URL',''),
            model=os.environ.get('PVSCENE_VLM_MODEL') or os.environ.get('PVSCENE_AGENT_MODEL',''),
            api_key_env=os.environ.get('PVSCENE_VLM_API_KEY_ENV') or os.environ.get('PVSCENE_AGENT_API_KEY_ENV','DASHSCOPE_API_KEY'),
            wire=os.environ.get('PVSCENE_VLM_WIRE') or os.environ.get('PVSCENE_AGENT_WIRE','chat_completions'),timeout_s=240,max_tokens=4096)
    budget=ModelBudget(a.model_budget)
    try:
        report=critique_scene(a.scene_bundle,a.output_dir/'critique',scope_ids=['kitchen'],profiles=profiles,
            expected_version=graph.content_hash,robot_delivery=a.robot_delivery,robot_executor=executor,
            render_manifest=a.render_manifest,config=config,budget=budget)
    except (ValueError,OSError,KeyError,TypeError,LLMError) as error:
        report={'status':'failed','failure_code':getattr(error,'code','CRITIC_INPUT_INVALID'),'model_requests':budget.used}
        write(a.output_dir,'report.json',report)
    write(a.output_dir,'report.json',{k:v for k,v in report.items() if k!='input_hashes'})
    print(json.dumps({k:v for k,v in report.items() if k!='input_hashes'}));return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
