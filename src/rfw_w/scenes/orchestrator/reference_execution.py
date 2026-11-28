# SPDX-License-Identifier: Apache-2.0
"""Bounded code-controlled attempts with immutable source and parameter revisions.

The provider proposes parameters from public measurement failures. It does not
claim physics success. Runtime and independent verifier own the result.
"""
import json
from copy import deepcopy
import time
from pathlib import Path
from rfw_w.scenes.contracts.hashing import content_hash


def run_attempts(world, output, *, executor, verifier, provider, initial_parameters,
                 max_attempts=5, repeated_failure_limit=2, deadline_s=1800):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parameters = deepcopy(dict(initial_parameters))
    seen = set(); failures = {}; events = []
    status = 'budget_exhausted'
    started=time.monotonic()
    for revision in range(max_attempts):
        if time.monotonic()-started>deadline_s:
            status='budget_exhausted';break
        signature = content_hash(parameters)
        if signature in seen:
            status = 'no_progress'; break
        seen.add(signature)
        attempt = output/f'attempt_{revision:03d}'
        recording = executor(world, attempt, parameters)
        validation = verifier(world, attempt)
        (attempt/'task_validation.json').write_text(json.dumps(validation, indent=2))
        issues = [key for key, value in validation['checks'].items() if not value]
        event = dict(recording_status=recording.get('status'),
                     source_world_hash=recording.get('world_sha256'),
                     controller_hash=recording.get('controller_source_sha256'),
                     revision=revision, parameter_hash=signature, parameters=deepcopy(parameters),
                     result=validation['status'], issues=issues, artifact_ref=attempt.name)
        events.append(event)
        (output/'attempts.json').write_text(json.dumps(events, indent=2))
        if validation['status'] == 'passed':
            status = 'passed'; break
        failure_signature = content_hash(issues)
        failures[failure_signature] = failures.get(failure_signature, 0)+1
        if failures[failure_signature] >= repeated_failure_limit:
            status = 'no_progress'; break
        proposed = provider({'revision': revision, 'issues': issues,
                             'metrics': validation.get('metrics', {}), 'parameters': deepcopy(parameters)})
        if proposed is None:
            status = 'failed'; break
        parameters = deepcopy(dict(proposed))
    report = dict(schema_version='pvscene.reference-attempts/v1', status=status,
                  elapsed_s=time.monotonic()-started,
                  attempts=events, model_requests=0, selection_mode='local_code_provider')
    (output/'report.json').write_text(json.dumps(report, indent=2))
    return report
