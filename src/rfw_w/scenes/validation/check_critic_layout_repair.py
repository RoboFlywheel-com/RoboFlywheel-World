# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
from rfw_w.scenes.orchestrator.critic_layout_repair import run_repair,RepairBudget


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--target',default='cup')
    p.add_argument('--robot-bundle',type=Path);p.add_argument('--max-repairs',type=int,default=2)
    a=p.parse_args();executor=None
    semantics={'room_id':'kitchen','support_role_id':'counter','target_role_id':'cup',
               'robot':'fixed Piper','objective':'post repair mounting/IK/approach checks'}
    if a.robot_bundle:
        from rfw_w.scenes.validation.robot_layout_executor import RobotLayoutExecutor
        executor=RobotLayoutExecutor(a.scene_bundle.resolve(),a.robot_bundle.resolve(),semantics)
    report=run_repair(a.scene_bundle,a.output_dir,target_id=a.target,
        budgets=RepairBudget(max_repairs=a.max_repairs),robot_executor=executor,robot_semantics=semantics)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_artifacts','events')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
