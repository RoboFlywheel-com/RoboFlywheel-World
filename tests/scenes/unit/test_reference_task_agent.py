# SPDX-License-Identifier: Apache-2.0
import json
import pytest
from rfw_w.scenes.orchestrator.reference_task_agent import run_reference_agent, LocalReferenceProvider


class Graph:
    content_hash='a'*64
    def to_dict(self):return {'nodes':[{'id':'tool','kind':'asset_role','category':'tool'}]}


@pytest.mark.parametrize("resets",[1,3])
def test_reference_agent_measures_before_export(tmp_path,resets):
    calls=[]
    def executor(tool,state,parameters):
        calls.append(tool)
        if tool=='export_reference_delivery':assert state['passed_resets']==resets
        return {'status':'passed','checks':{'grasp':True}}
    r=run_reference_agent(Graph(),tmp_path/'run',provider=LocalReferenceProvider(),executor=executor,parameters={},required_resets=resets)
    assert r['status']=='passed'
    assert calls.count('execute_reference_episode')==calls.count('verify_reference_episode')==resets


def test_failed_episode_cannot_jump_to_export(tmp_path):
    provider=LocalReferenceProvider()
    def malicious(summary):
        if summary['failure']:return {'tool':'export_reference_delivery'}
        return provider(summary)
    def executor(tool,state,parameters):
        return {'status':'failed' if tool=='verify_reference_episode' else 'passed','checks':{'grasp':False}}
    with pytest.raises(ValueError,match='NOT_LEGAL'):
        run_reference_agent(Graph(),tmp_path/'run',provider=malicious,executor=executor,parameters={})


def test_criteria_cannot_be_relaxed_by_repair(tmp_path):
    provider=LocalReferenceProvider([{'max_penetration_m':.1}])
    def executor(tool,state,parameters):
        return {'status':'failed' if tool=='verify_reference_episode' else 'passed','checks':{'penetration':False}}
    with pytest.raises(ValueError,match='PROTECTED_PARAMETER'):
        run_reference_agent(Graph(),tmp_path/'run',provider=provider,executor=executor,parameters={'max_penetration_m':.003})


def test_runtime_storage_error_is_a_failed_delivery(tmp_path):
    def executor(tool,state,parameters):
        raise OSError(122,'synthetic storage failure')
    r=run_reference_agent(Graph(),tmp_path/'run',provider=LocalReferenceProvider(),executor=executor,parameters={})
    assert r['status']=='failed'
    event=json.loads((tmp_path/'run/tool_events.json').read_text())[0]
    assert event['result']['errno']==122
    assert 'synthetic storage failure' not in json.dumps(event)


def test_default_single_run_keeps_independent_verification(tmp_path):
    calls=[]
    def executor(tool,state,parameters):
        calls.append(tool)
        return {'status':'passed','checks':{'grasp':True}}
    r=run_reference_agent(Graph(),tmp_path/'run',provider=LocalReferenceProvider(),executor=executor,parameters={})
    assert r['required_resets']==r['passed_resets']==1
    assert calls.index('verify_reference_episode')<calls.index('export_reference_delivery')


@pytest.mark.parametrize('resets',[0,-1,True])
def test_invalid_reset_count_rejected(tmp_path,resets):
    with pytest.raises(ValueError,match='RESET_COUNT_INVALID'):
        run_reference_agent(Graph(),tmp_path/'run',provider=LocalReferenceProvider(),executor=lambda *a:None,parameters={},required_resets=resets)
    assert not (tmp_path/'run').exists()
