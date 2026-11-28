# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
from rfw_w.scenes.orchestrator.scene_critic_tools import verify_critique
from rfw_w.scenes.orchestrator.layout_revision_tools import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-bundle',type=Path,required=True)
    p.add_argument('--delivery',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--robot-delivery',type=Path);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    executor=None
    if a.robot_delivery:
        from rfw_w.scenes.validation.verify_agent_robot_layout import verify
        executor=verify
    try:report=verify_critique(a.scene_bundle,a.delivery,robot_delivery=a.robot_delivery,robot_executor=executor)
    except (ValueError,OSError,KeyError,TypeError) as error:
        report={'verified':False,'failure_code':getattr(error,'code','CRITIC_INPUT_INVALID')}
    write(a.output_dir,'report.json',report);print(json.dumps(report));return 0 if report['verified'] else 1


if __name__=='__main__':raise SystemExit(main())
