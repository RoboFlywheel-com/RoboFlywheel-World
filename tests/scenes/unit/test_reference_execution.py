# SPDX-License-Identifier: Apache-2.0
from rfw_w.scenes.orchestrator.reference_execution import run_attempts


def test_repeated_parameters_stop_without_extra_execution(tmp_path):
    calls = []
    def executor(world, out, parameters):
        out.mkdir();calls.append(parameters);return {}
    def verifier(world, out):
        return {'status': 'failed', 'checks': {'grasp': False}}
    report = run_attempts(None, tmp_path/'run', executor=executor, verifier=verifier,
                          provider=lambda summary: summary['parameters'], initial_parameters={'opening': .01})
    assert report['status'] == 'no_progress'
    assert len(calls) == 1


def test_measured_failure_revision_then_verified_result(tmp_path):
    def executor(world, out, parameters):
        out.mkdir();return {}
    def verifier(world, out):
        ok = out.name.endswith('001')
        return {'status': 'passed' if ok else 'failed', 'checks': {'grasp': ok}}
    report = run_attempts(None, tmp_path/'run', executor=executor, verifier=verifier,
                          provider=lambda summary: {'opening': .003}, initial_parameters={'opening': .01})
    assert report['status'] == 'passed'
    assert len(report['attempts']) == 2
    assert report['attempts'][0]['parameter_hash'] != report['attempts'][1]['parameter_hash']
