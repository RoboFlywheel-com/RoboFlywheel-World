# SPDX-License-Identifier: Apache-2.0
"""Capability completeness and exposure cannot broaden the original legal tools."""
import json
from pathlib import Path
import pytest
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.orchestrator.tool_capabilities import (
    catalog, describe_tools, group_tools, legacy_tools, validate_catalog,
)

BASELINE = json.loads((Path(__file__).parents[1] / 'fixtures/tool_capabilities_baseline.json').read_text())


def test_catalog_preserves_original_stage_metadata_and_parameter_schemas():
    records = catalog()
    assert len(records) == 31
    assert len({r['name'] for r in records}) == 27
    for stage in ('scene', 'robot_layout'):
        assert legacy_tools(stage) == BASELINE[stage]
        for tool in describe_tools(stage, list(BASELINE[stage])):
            assert tool['parameter_schema'] == (BASELINE['parameter_schemas'].get(tool['name'], {}) if stage == 'scene' else {})
    assert {r['name'] for r in records if r['stage']=='critic_repair'} == {
        'critique_scene','propose_layout_candidates','apply_layout_candidate','finish','terminate'}
    assert not {'critique_scene', 'apply_layout_candidate'} & {r['name'] for r in records if r['stage']=='scene'}


def test_finish_identity_is_stage_qualified_and_snapshots_are_detached():
    scene = describe_tools('scene', ['finish'])[0]
    robot = describe_tools('robot_layout', ['finish'])[0]
    assert scene['scope_kind'] == 'scene' and robot['scope_kind'] == 'robot_layout'
    scene['parameter_schema']['private'] = 'changed'
    assert describe_tools('scene', ['finish'])[0]['parameter_schema'] == {}
    records = catalog()
    records[0]['reads'].clear()
    assert catalog()[0]['reads']


@pytest.mark.parametrize('change', [
    lambda r: r[0].pop('domain'),
    lambda r: r[0].update(domain='invented'),
    lambda r: r.append(r[0].copy()),
    lambda r: r[0].update(writes='true'),
    lambda r: r[0].update(parameter_schema={'target': None}),
    lambda r: r[0].update(failure_codes=['invalid code']),
    lambda r: r[1].update(invalidates=['target_ik']),
    lambda r: r[0].update(stage=[]),
    lambda r: r[0].update(reverify=['missing_tool']),
])
def test_invalid_catalog_rejected(change):
    records = catalog()
    change(records)
    with pytest.raises(GraphError, match='TOOL_CAPABILITY_INVALID'):
        validate_catalog(records)


@pytest.mark.parametrize('stage,names,code', [
    ('unknown', [], 'TOOL_CAPABILITY_INVALID'),
    ('scene', ['finish', 'finish'], 'TOOL_CAPABILITY_INVALID'),
    ('scene', ['install_robot'], 'TOOL_NOT_AVAILABLE'),
    ('robot_layout', ['verify_scene'], 'TOOL_NOT_AVAILABLE'),
])
def test_illegal_exposure_rejected(stage, names, code):
    with pytest.raises(GraphError, match=code):
        describe_tools(stage, names)


def test_descriptions_and_groups_contain_only_legal_tools_in_order():
    from rfw_w.scenes.orchestrator.adaptive_scene_agent import AgentState, available_tools, public_summary
    from tests.scenes.unit.test_adaptive_scene_tool_agent import graph
    state = AgentState(graph())
    assert available_tools(state) == ['revise_composition']
    raw = state.graph.to_dict()
    raw['built']['asset_assembly'] = {'ledger_hash': '0' * 64}
    state.graph = state.graph.revise(raw, expected_hash=state.graph.content_hash)
    state.checkpoint_dir = Path('checkpoint')
    legal = available_tools(state)
    assert legal == ['furnish_floor', 'mount_wall']
    tools = describe_tools('scene', legal)
    assert [t['name'] for t in tools] == legal
    assert group_tools(tools) == {'layout': legal}
    payload = public_summary(state, legal)
    assert payload['available_tools'] == tools
    assert payload['tool_groups'] == {'layout': legal}
    state.failure = 'CORE_KEEP_OUT_VIOLATION'
    tools = describe_tools('scene', available_tools(state))
    assert group_tools(tools) == {'recovery': ['rollback_scope'], 'delivery': ['terminate']}
    state.failure = 'MATERIAL_ENV_MISSING'
    assert group_tools(describe_tools('scene', available_tools(state))) == {'delivery': ['terminate']}


def test_robot_failure_and_revision_invalidation_metadata():
    from rfw_w.scenes.orchestrator.robot_layout_agent import LayoutState, available, summary
    state = LayoutState('a' * 64, failure='APPROACH_PATH_COLLISION')
    tools = describe_tools('robot_layout', available(state))
    assert group_tools(tools) == {'robot': ['revise_robot_installation'], 'delivery': ['terminate']}
    assert 'target_ik' in tools[0]['invalidates']
    assert tools[0]['reverify'] == ['check_target_reachability', 'check_approach_path']
    payload = summary(state, {'task': 'pick cup'})
    assert payload['available_tools'] == tools
    assert payload['tool_groups'] == group_tools(tools)
    state.revision = state.max_revisions
    assert [t['name'] for t in describe_tools('robot_layout', available(state))] == ['terminate']
