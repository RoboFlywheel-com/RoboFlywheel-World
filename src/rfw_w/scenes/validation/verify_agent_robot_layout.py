# SPDX-License-Identifier: Apache-2.0
"""Independent legality, revision invalidation, immutable source and physics replay."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph, GraphError
from rfw_w.scenes.orchestrator.adaptive_scene_agent import validate_decision
from rfw_w.scenes.orchestrator.robot_layout_agent import LayoutState, available, advance, tree_hashes


def guarded(root, ref):
    path = (root / ref).resolve()
    if Path(ref).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('ROBOT_LAYOUT_INPUT_INVALID')
    return path


def verify(root):
    from rfw_w.scenes.validation.verify_single_room_installation import verify as mount
    from rfw_w.scenes.validation.verify_mujoco_installation_appearance import verify as appearance
    from rfw_w.scenes.validation.verify_mujoco_target_ik import verify as ik
    from rfw_w.scenes.validation.verify_mujoco_approach_path import verify as path_check
    report = json.loads((root / 'report.json').read_text())
    graph = TaskWorldGraph.from_dict(json.loads((root / 'source_graph.json').read_text()))
    events = json.loads((root / 'events.json').read_text())
    frozen = json.loads((root / 'source_artifacts.json').read_text())
    state = LayoutState(graph.content_hash, max_revisions=report['budgets']['max_revisions'])
    failed, repairs = 0, 0
    for index, event in enumerate(events):
        legal = available(state)
        if event['available_tools'] != legal or event['before'] != asdict(state) or event['source_graph_hash'] != graph.content_hash:
            raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
        name = validate_decision(event['decision'], legal)
        step = guarded(root, event['artifact_ref'])
        if step.name != f'step_{index + 1:03d}_{name}' or json.loads((step / 'event.json').read_text()) != event:
            raise ValueError('ROBOT_LAYOUT_INPUT_INVALID')
        result = event['result']
        measured = None
        if name in ('install_robot', 'revise_robot_installation'):
            if (step / 'installation/installation_report.json').exists():
                raw = json.loads((step / 'installation/installation_report.json').read_text())
                if raw['status'] == 'passed':
                    measured = mount(step / 'installation')
                    if measured['verified']:
                        appearance(step / 'appearance')
                    instance_root = step / 'installation/scene'
                    if tree_hashes(instance_root) != frozen:
                        raise ValueError('ROBOT_LAYOUT_SOURCE_CHANGED')
                else:
                    # A failed exhaustive search is not a successful installation.
                    measured = {'verified': False, 'failure_code': raw['failure_code']}
            if name == 'revise_robot_installation':
                repairs += 1
            if state.install_ref and measured and measured['verified']:
                old = json.loads((guarded(root, state.install_ref) / 'installation/installation.json').read_text())
                new = json.loads((step / 'installation/installation.json').read_text())
                if old['base_position_m'] == new['base_position_m'] and old['base_yaw_rad'] == new['base_yaw_rad']:
                    raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
        elif name == 'check_target_reachability':
            expected_source = guarded(root, state.install_ref) / 'appearance'
            if tree_hashes(step / 'ik/source') != tree_hashes(expected_source):
                raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
            if event['result']['status']=='passed':
                measured = ik(step / 'ik')
            else:
                raw = json.loads((step / 'ik/report.json').read_text())
                if raw['status']=='passed' or raw['failure_code']!=event['result']['failure_code']:
                    raise ValueError('ROBOT_LAYOUT_INPUT_INVALID')
                measured = {'verified':False,'failure_code':raw['failure_code']}
        elif name == 'check_approach_path':
            meta = json.loads((step / 'path.json').read_text())
            if (step / meta['source_ref']).resolve() != guarded(root, state.ik_ref):
                raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
            measured = path_check(step)
        elif name == 'finish':
            if index != len(events) - 1 or report['status'] != 'passed':
                raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
        elif name == 'terminate':
            if index != len(events) - 1 or report['status'] == 'passed':
                raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
        if measured:
            expected_status = 'passed' if measured['verified'] else 'failed'
            if result['status'] != expected_status or (expected_status == 'failed' and result['failure_code'] != measured['failure_code']):
                raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
        elif name not in ('finish', 'terminate'):
            raise ValueError('ROBOT_LAYOUT_INPUT_INVALID')
        if name not in ('finish', 'terminate'):
            advance(state, name, result)
            failed += result['status'] != 'passed'
        if asdict(state) != event['after']:
            raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
    if report['status']=='budget_exhausted':
        if len(events)<report['budgets']['max_tools'] and report['elapsed_s']<report['budgets']['deadline_s']:
            raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
        state.failure='BUDGET_EXHAUSTED'
    if state.graph_hash != report['source_graph_hash'] or asdict(state) != report['state'] or report['tool_calls'] != len(events):
        raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
    if report['status'] == 'passed' and (not events or events[-1]['decision']['tool_name'] != 'finish' or state.failure):
        raise ValueError('ROBOT_LAYOUT_STALE_EVIDENCE')
    return {'verified': True, 'status': report['status'], 'failures_replayed': failed, 'repairs': repairs,
            'tool_calls': len(events), 'graph_unchanged': True, 'grasp': 'not_run', 'rollout': 'not_run'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = verify(a.delivery)
    except (ValueError, OSError, KeyError, TypeError, GraphError) as error:
        report = {'verified': False, 'failure_code': str(error)}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
