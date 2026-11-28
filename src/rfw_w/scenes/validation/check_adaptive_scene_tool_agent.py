#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real asset adaptive tool runs; distinct local decision branches or bounded online run."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from rfw_w.scenes.validation.check_unified_scene_graph_roles import fixture_graph
from rfw_w.scenes.orchestrator.adaptive_scene_agent import AdaptiveSceneAgent,ReplayDecisionProvider,OnlineDecisionProvider
from rfw_w.scenes.agents.llm_client import LLMConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fixture',default='two_room_background');p.add_argument('--online',action='store_true');p.add_argument('--surface-style',action='store_true');p.add_argument('--rigid-assets',action='store_true');p.add_argument('--rigid-restore',action='store_true');p.add_argument('--surface-optionals',action='store_true');p.add_argument('--coverings',action='store_true');p.add_argument('--max-model-requests',type=int,default=12)
    p.add_argument('--robot-bundle',type=Path,help='Opt in to real fixed-Piper MuJoCo layout checks after static generation')
    p.add_argument('--max-tools',type=int,default=30)
    p.add_argument('--deadline-s',type=float,default=600)
    p.add_argument('--critic-repair',action='store_true',help='Opt in to current-version static critic and bounded cup layout repair')
    a=p.parse_args()
    if a.online and (a.robot_bundle or a.critic_repair):p.error('online robot/repair decisions require separate acceptance')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    semantics={'room_id':'kitchen','support_role_id':'counter','target_role_id':'cup',
        'robot':'fixed Piper','objective':'mounting and pregrasp approach checks'}
    def robot_executor():
        if a.robot_bundle is None:return None
        from rfw_w.scenes.validation.robot_layout_executor import RobotLayoutExecutor
        # The generation hook binds the freshly generated final assets before any robot call.
        return RobotLayoutExecutor(None,a.robot_bundle.resolve(),semantics)
    results={}
    if a.online:
        config=LLMConfig(base_url=os.environ.get('PVSCENE_AGENT_BASE_URL') or os.environ.get('PVSCENE_VLM_BASE_URL') or os.environ.get('SIMFOUNDRY_OPENAI_BASE_URL',''),
            model=os.environ.get('PVSCENE_AGENT_MODEL') or os.environ.get('PVSCENE_VLM_MODEL',''),
            api_key_env=os.environ.get('PVSCENE_AGENT_API_KEY_ENV') or os.environ.get('PVSCENE_VLM_API_KEY_ENV','DASHSCOPE_API_KEY'),
            wire=os.environ.get('PVSCENE_AGENT_WIRE') or os.environ.get('PVSCENE_VLM_WIRE','chat_completions'),timeout_s=90,max_tokens=4096)
        results['online']=AdaptiveSceneAgent(OnlineDecisionProvider(config),max_model_requests=a.max_model_requests,deadline_s=900,surface_style=a.surface_style,rigid_assets=a.rigid_assets,rigid_restore=a.rigid_restore,surface_optionals=a.surface_optionals,coverings=a.coverings).run(fixture_graph(),a.output_dir/'online')
    else:
        for name,preference,fault in [('wall_first','wall',False),('floor_failure','floor',True)]:
            results[name]=AdaptiveSceneAgent(ReplayDecisionProvider(preference),max_tools=a.max_tools,deadline_s=a.deadline_s,inject_failure=fault,surface_style=a.surface_style,rigid_assets=a.rigid_assets,rigid_restore=a.rigid_restore,surface_optionals=a.surface_optionals,coverings=a.coverings,robot_layout_executor=robot_executor(),robot_layout_semantics=semantics if a.robot_bundle else None,critic_repair=a.critic_repair).run(fixture_graph(),a.output_dir/name)
    status='passed' if all(r['status']=='passed' for r in results.values()) else 'failed'
    if not a.online and results['wall_first']['tool_path']==results['floor_failure']['tool_path']:status='failed'
    report={'status':status,'runs':results,'robot_validation':'layout_only' if a.robot_bundle else 'not_run'}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if status=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
