# SPDX-License-Identifier: Apache-2.0
"""Bounded robot layout sidecar using the scene Agent decision contract (stdlib only)."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
from rfw_w.scenes.orchestrator.adaptive_scene_agent import validate_decision, OnlineDecisionProvider
from rfw_w.scenes.contracts.task_world_graph import GraphError

from rfw_w.scenes.orchestrator.tool_capabilities import describe_tools, group_tools, legacy_tools

ROBOT_TOOLS = legacy_tools('robot_layout')
REPAIRABLE = {'INSTALLATION_UNAVAILABLE', 'INSTALLATION_OUTSIDE_SUPPORT', 'INSTALLATION_OCCUPIED',
              'INSTALLATION_COLLISION', 'TARGET_IK_UNREACHABLE', 'TARGET_IK_COLLISION', 'APPROACH_PATH_COLLISION'}


@dataclass
class LayoutState:
    graph_hash: str
    revision: int = 0
    max_revisions: int = 2
    install_ref: str | None = None
    ik_ref: str | None = None
    path_ref: str | None = None
    failure: str | None = None


def available(state):
    if state.failure:
        return (['revise_robot_installation', 'terminate'] if state.failure in REPAIRABLE and
                state.revision < state.max_revisions else ['terminate'])
    if state.install_ref is None:
        return ['install_robot']
    if state.ik_ref is None:
        return ['check_target_reachability']
    if state.path_ref is None:
        return ['check_approach_path']
    return ['finish']


def advance(state, name, result):
    if name not in available(state):
        raise GraphError('TOOL_NOT_AVAILABLE', name)
    if name == 'revise_robot_installation':
        state.revision += 1
        state.install_ref = state.ik_ref = state.path_ref = None
    if result['status'] != 'passed':
        state.failure = result['failure_code']
        if name == 'check_target_reachability':
            state.ik_ref = state.path_ref = None
        elif name == 'check_approach_path':
            state.path_ref = None
        return
    state.failure = None
    if name in ('install_robot', 'revise_robot_installation'):
        state.install_ref = result['artifact_ref']
        state.ik_ref = state.path_ref = None
    elif name == 'check_target_reachability':
        state.ik_ref = result['artifact_ref']
        state.path_ref = None
    elif name == 'check_approach_path':
        state.path_ref = result['artifact_ref']


def summary(state, semantics):
    tools = describe_tools('robot_layout', available(state))
    return {'task': semantics, 'graph_hash': state.graph_hash, 'installation_revision': state.revision,
            'installation_checked': state.install_ref is not None, 'target_checked': state.ik_ref is not None,
            'approach_checked': state.path_ref is not None, 'failure_code': state.failure,
            'available_tools': tools,
            'tool_groups': group_tools(tools)}


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class RobotLayoutAgent:
    def __init__(self, provider, *, max_tools=12, max_revisions=2, deadline_s=600):
        if max_tools < 0 or max_revisions < 0 or deadline_s <= 0:
            raise GraphError('AGENT_ACTION_INVALID', 'robot budgets')
        self.provider = provider
        self.max_tools = max_tools
        self.max_revisions = max_revisions
        self.deadline_s = deadline_s

    def run(self, graph, assets, output, *, executor, semantics):
        # Geometry stays behind the explicit executor; online robot decisions are a separate acceptance.
        if isinstance(self.provider, OnlineDecisionProvider):
            raise GraphError('AGENT_ACTION_INVALID', 'robot online phase not yet accepted')
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        state = LayoutState(graph.content_hash, max_revisions=self.max_revisions)
        frozen = tree_hashes(assets)
        (output / 'source_graph.json').write_text(json.dumps(graph.to_dict(), indent=2) + '\n')
        (output / 'source_artifacts.json').write_text(json.dumps(frozen, indent=2) + '\n')
        events = []
        terminal = None
        while terminal is None:
            if len(events) >= self.max_tools or time.monotonic() - started >= self.deadline_s:
                state.failure = 'BUDGET_EXHAUSTED'
                terminal = 'budget_exhausted'
                break
            legal = available(state)
            public = summary(state, semantics)
            try:
                decision = self.provider(public)
                name = validate_decision(decision, legal)
            except GraphError as error:
                state.failure = error.code
                terminal = 'failed'
                break
            step = output / f'step_{len(events) + 1:03d}_{name}'
            before = asdict(state)
            step.mkdir()
            if tree_hashes(assets) != frozen:
                result = {'status': 'failed', 'failure_code': 'ROBOT_LAYOUT_SOURCE_CHANGED'}
                state.failure = 'ROBOT_LAYOUT_SOURCE_CHANGED'
                terminal = 'failed'
            elif name in ('finish', 'terminate'):
                terminal = 'passed' if name == 'finish' else 'failed'
                result = {'status': 'passed', 'artifact_ref': step.name}
            else:
                try:
                    result = executor(name, state, step, output)
                except (OSError, ValueError, GraphError, TimeoutError) as error:
                    result = {'status': 'failed', 'failure_code': getattr(error, 'code', 'ROBOT_TOOL_TIMEOUT' if isinstance(error, TimeoutError) else 'ROBOT_LAYOUT_INPUT_INVALID')}
                if result['status'] == 'passed':
                    ref = result.get('artifact_ref', '')
                    if not ref or Path(ref).is_absolute() or not (output / ref).resolve().is_relative_to(output.resolve()):
                        result = {'status': 'failed', 'failure_code': 'ROBOT_LAYOUT_INPUT_INVALID'}
                advance(state, name, result)
            event = {'available_tools': legal, 'summary': public, 'decision': decision,
                     'before': before, 'after': asdict(state), 'result': result,
                     'artifact_ref': step.name, 'source_graph_hash': graph.content_hash}
            events.append(event)
            (step / 'event.json').write_text(json.dumps(event, indent=2) + '\n')
            (output / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
        if time.monotonic() - started > self.deadline_s and terminal == 'passed':
            terminal = 'budget_exhausted'
            state.failure = 'BUDGET_EXHAUSTED'
        report = {'status': terminal, 'failure_code': state.failure, 'state': asdict(state),
                  'tool_calls': len(events), 'model_requests': 0, 'elapsed_s': round(time.monotonic() - started, 3),
                  'semantics': semantics, 'source_unchanged': tree_hashes(assets) == frozen,
                  'budgets': {'max_tools': self.max_tools, 'max_revisions': self.max_revisions, 'deadline_s': self.deadline_s},
                  'source_graph_hash': graph.content_hash, 'grasp': 'not_run', 'rollout': 'not_run', 'dynamic_motion': 'not_run'}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


class LayoutReplayProvider:
    def __call__(self, public):
        legal = [t['name'] for t in public['available_tools']]
        name = 'revise_robot_installation' if 'revise_robot_installation' in legal else legal[0]
        return {'tool_name': name, 'scope_ids': [], 'parameters': {}, 'reason_code': 'SATISFY_ROBOT_LAYOUT'}
