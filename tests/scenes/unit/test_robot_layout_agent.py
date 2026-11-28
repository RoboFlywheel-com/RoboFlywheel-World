# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import pytest
from rfw_w.scenes.orchestrator.robot_layout_agent import LayoutState, available, advance, summary
from rfw_w.scenes.orchestrator.adaptive_scene_agent import validate_decision
from rfw_w.scenes.contracts.task_world_graph import GraphError


def test_tools_follow_evidence_and_invalidate_after_install_revision():
    s = LayoutState('a' * 64)
    assert available(s) == ['install_robot']
    advance(s, 'install_robot', {'status': 'passed', 'artifact_ref': 'install'})
    assert available(s) == ['check_target_reachability']
    advance(s, 'check_target_reachability', {'status': 'passed', 'artifact_ref': 'ik'})
    assert available(s) == ['check_approach_path']
    advance(s, 'check_approach_path', {'status': 'failed', 'failure_code': 'APPROACH_PATH_COLLISION'})
    assert available(s) == ['revise_robot_installation', 'terminate']
    advance(s, 'revise_robot_installation', {'status': 'passed', 'artifact_ref': 'revision'})
    assert s.revision == 1 and s.ik_ref is None and s.path_ref is None
    assert available(s) == ['check_target_reachability']


def test_failure_routing_and_revision_budget():
    for code in ('ROBOT_LAYOUT_SOURCE_CHANGED', 'TARGET_IK_RENDER_FAILED', 'WORKER_ENV_MISSING'):
        s = LayoutState('a' * 64, failure=code)
        assert available(s) == ['terminate']
    s = LayoutState('a' * 64, failure='TARGET_IK_UNREACHABLE', revision=2)
    assert available(s) == ['terminate']


def test_finish_requires_all_current_revision_evidence():
    s = LayoutState('a' * 64, install_ref='mount')
    assert 'finish' not in available(s)
    advance(s, 'check_target_reachability', {'status': 'passed', 'artifact_ref': 'ik'})
    advance(s, 'check_approach_path', {'status': 'passed', 'artifact_ref': 'path'})
    assert available(s) == ['finish']


def test_public_summary_has_no_paths_or_logs():
    s = LayoutState('a' * 64, install_ref='/private/local/install')
    raw = json.dumps(summary(s, {'room_id': 'kitchen', 'target_role_id': 'cup', 'support_role_id': 'counter'}))
    assert '/private/' not in raw and 'install_ref' not in raw


def test_illegal_parameter_or_early_finish_rejected():
    with pytest.raises(GraphError, match='TOOL_NOT_AVAILABLE'):
        validate_decision({'tool_name': 'finish', 'scope_ids': [], 'parameters': {}, 'reason_code': 'TEST'}, ['install_robot'])
    with pytest.raises(GraphError, match='AGENT_ACTION_INVALID'):
        validate_decision({'tool_name': 'install_robot', 'scope_ids': [], 'parameters': {'path': '/private'}, 'reason_code': 'TEST'}, ['install_robot'])


def test_invalid_decision_and_budget_preserve_report(tmp_path):
    from tests.scenes.unit.test_adaptive_scene_tool_agent import graph
    from rfw_w.scenes.orchestrator.robot_layout_agent import RobotLayoutAgent, LayoutReplayProvider
    assets = tmp_path / 'assets'
    assets.mkdir()
    def invalid(_):
        return {'tool_name': 'finish', 'scope_ids': [], 'parameters': {}, 'reason_code': 'TEST'}
    def never(*args):
        raise AssertionError('executor must not be called')
    result = RobotLayoutAgent(invalid).run(graph(), assets, tmp_path / 'invalid', executor=never, semantics={})
    assert result['failure_code'] == 'TOOL_NOT_AVAILABLE'
    assert (tmp_path / 'invalid/report.json').exists()
    result = RobotLayoutAgent(LayoutReplayProvider(), max_tools=0).run(graph(), assets, tmp_path / 'budget', executor=never, semantics={})
    assert result['status'] == 'budget_exhausted'


def test_mutating_source_cannot_finish(tmp_path):
    from tests.scenes.unit.test_adaptive_scene_tool_agent import graph
    from rfw_w.scenes.orchestrator.robot_layout_agent import RobotLayoutAgent, LayoutReplayProvider
    assets = tmp_path / 'assets'
    assets.mkdir()
    def mutate(name, state, step, output):
        (assets / 'changed').write_text('changed')
        return {'status': 'passed', 'artifact_ref': step.name}
    result = RobotLayoutAgent(LayoutReplayProvider()).run(graph(), assets, tmp_path / 'run', executor=mutate, semantics={})
    assert result['status'] == 'failed' and result['failure_code'] == 'ROBOT_LAYOUT_SOURCE_CHANGED'


def test_generation_robot_hook_uses_remaining_budget_and_current_assets(monkeypatch, tmp_path):
    import rfw_w.scenes.orchestrator.adaptive_scene_agent as module
    import rfw_w.scenes.orchestrator.robot_layout_agent as robot
    from tests.scenes.unit.test_adaptive_scene_tool_agent import graph
    g = graph()
    raw = g.to_dict()
    raw['built']['asset_assembly'] = {'ledger_hash': '0' * 64}
    built = g.revise(raw, expected_hash=g.content_hash)
    # Only finish is executed here: the static capabilities already have their own runtime acceptance.
    monkeypatch.setattr(module, 'available_tools', lambda state: ['finish'])
    monkeypatch.setattr(module, 'public_summary', lambda *args: {})
    def provider(_):
        return {'tool_name': 'finish', 'scope_ids': [], 'parameters': {}, 'reason_code': 'TEST'}
    captured = {}
    class Executor:
        def bind_assets(self, assets):
            captured['assets'] = assets
    def fake_run(self, graph_arg, assets, output, **kwargs):
        captured['max_tools'] = self.max_tools
        return {'status': 'passed', 'failure_code': None, 'tool_calls': 4,
                'state': {'ik_ref': 'ik', 'path_ref': 'path'}}
    monkeypatch.setattr(robot.RobotLayoutAgent, 'run', fake_run)
    # Inject the already-built assets into the initial state without rebuilding static geometry.
    original_state = module.AgentState
    def seeded(*args, **kwargs):
        state = original_state(*args, **kwargs)
        state.assets_dir = tmp_path / 'run/assets'
        state.assets_dir.mkdir()
        return state
    monkeypatch.setattr(module, 'AgentState', seeded)
    report = module.AdaptiveSceneAgent(provider, max_tools=8, robot_layout_executor=Executor(),
                                      robot_layout_semantics={}).run(g, tmp_path / 'run')
    assert captured['max_tools'] == 7 and captured['assets'] == tmp_path / 'run/assets'
    assert report['tool_calls'] == 5 and report['robot_ik'] == 'passed'
