# SPDX-License-Identifier: Apache-2.0
import time
import pytest
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.orchestrator.critic_layout_repair import RepairBudget,legal_tools


def review(hard=True,quality=False,code='SUPPORT_EDGE_MARGIN'):
    return {'profile_results':{'static':{'hard_pass':hard}},'quality_pass':quality,
            'issues':[{'code':code,'suggested_operation':'translate'}]}


def test_finish_requires_current_hard_and_quality_review():
    assert legal_tools(None,repairs=0,max_repairs=2)==['critique_scene']
    assert legal_tools(review(hard=False,quality=True),repairs=0,max_repairs=2)==['terminate']
    assert legal_tools(review(quality=True),repairs=1,max_repairs=2)==['finish']
    assert legal_tools(review(),repairs=2,max_repairs=2)==['terminate']
    assert legal_tools(review(code='ASSET_UNAVAILABLE'),repairs=0,max_repairs=2)==['terminate']


def test_tool_and_wall_clock_budget_are_enforced_in_code():
    budget=RepairBudget(max_tools=1);start=time.monotonic();budget.spend(start)
    with pytest.raises(GraphError,match='BUDGET_EXHAUSTED'):budget.spend(start)
    with pytest.raises(GraphError,match='BUDGET_EXHAUSTED'):RepairBudget(deadline_s=.01).check(start-1)
