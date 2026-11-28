# SPDX-License-Identifier: Apache-2.0
"""Resume scene Agent with explicit real MuJoCo robot layout tools."""
import argparse
import json
from pathlib import Path
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph
from rfw_w.scenes.orchestrator.adaptive_scene_agent import AdaptiveSceneAgent
from rfw_w.scenes.orchestrator.robot_layout_agent import LayoutReplayProvider
from rfw_w.scenes.validation.robot_layout_executor import RobotLayoutExecutor


def main():
    parser = argparse.ArgumentParser()
    for name in ('scene-bundle', 'robot-bundle', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--skip-initial', type=int, default=0)
    a = parser.parse_args()
    from rfw_w.scenes.orchestrator.rigid_asset_tools import _verify_bundle
    _verify_bundle(a.scene_bundle)
    graph = TaskWorldGraph.from_dict(json.loads((a.scene_bundle / 'graph.json').read_text()))
    semantics = {'room_id': 'kitchen', 'support_role_id': 'counter', 'target_role_id': 'cup',
                 'robot': 'fixed Piper', 'objective': 'mounting and pregrasp approach checks'}
    executor = RobotLayoutExecutor(a.scene_bundle.resolve(), a.robot_bundle.resolve(), semantics, skip_initial=a.skip_initial)
    report = AdaptiveSceneAgent(LayoutReplayProvider(), max_tools=12, deadline_s=600).run_robot_layout(
        graph, a.scene_bundle, a.output_dir, executor=executor, semantics=semantics)
    print(json.dumps(report))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
