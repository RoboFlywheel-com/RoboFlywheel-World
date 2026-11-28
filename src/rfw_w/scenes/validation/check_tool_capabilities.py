#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export executable tool metadata and independently check compatibility/negative cases."""
from rfw_w.scenes.resources import data_path, module_path
import argparse
import json
from pathlib import Path
import sys

from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.orchestrator.tool_capabilities import (
    catalog, describe_tools, group_tools, legacy_tools, validate_catalog,
)


def check():
    records = catalog()
    baseline = json.loads((data_path('fixtures/tool_capabilities_baseline.json')).read_text())
    problems = []
    for stage in ('scene', 'robot_layout'):
        if legacy_tools(stage) != baseline[stage]:
            problems.append('LEGACY_METADATA_CHANGED(' + stage + ')')
        tools = describe_tools(stage, list(baseline[stage]))
        for tool in tools:
            expected = baseline['parameter_schemas'].get(tool['name'], {}) if stage == 'scene' else {}
            if tool['parameter_schema'] != expected:
                problems.append('PARAMETER_SCHEMA_CHANGED(' + stage + ':' + tool['name'] + ')')
    if len(records) != 31 or len({r['name'] for r in records}) != 27:
        problems.append('TOOL_COUNT_CHANGED')
    negatives = {}
    mutations = {
        'missing_field': lambda r: r[0].pop('domain'),
        'unknown_domain': lambda r: r[0].update(domain='invented'),
        'duplicate_identity': lambda r: r.append(r[0].copy()),
        'invalid_identity_type': lambda r: r[0].update(stage=[]),
        'invalid_writes': lambda r: r[0].update(writes='true'),
        'invalid_schema': lambda r: r[0].update(parameter_schema={'target': None}),
        'invalid_failure_code': lambda r: r[0].update(failure_codes=['invalid code']),
        'read_only_invalidation': lambda r: r[1].update(invalidates=['target_ik']),
        'unknown_reverify': lambda r: r[0].update(reverify=['missing_tool']),
    }
    def reject(label, action, expected):
        try:
            action()
        except GraphError as error:
            negatives[label] = {'rejected': error.code == expected, 'failure_code': error.code}
        else:
            negatives[label] = {'rejected': False}
        if not negatives[label]['rejected']:
            problems.append('NEGATIVE_NOT_REJECTED(' + label + ')')
    for label, mutate in mutations.items():
        candidate = catalog()
        mutate(candidate)
        reject(label, lambda: validate_catalog(candidate), 'TOOL_CAPABILITY_INVALID')
    for label, stage, names, expected in (
        ('unknown_stage', 'unknown', [], 'TOOL_CAPABILITY_INVALID'),
        ('duplicate_exposure', 'scene', ['finish', 'finish'], 'TOOL_CAPABILITY_INVALID'),
        ('cross_stage_exposure', 'scene', ['install_robot'], 'TOOL_NOT_AVAILABLE'),
        ('critic_cross_stage_exposure', 'scene', ['critique_scene'], 'TOOL_NOT_AVAILABLE'),
    ):
        reject(label, lambda: describe_tools(stage, names), expected)
    return records, {
        'status': 'passed' if not problems else 'failed', 'problems': problems,
        'catalog_hash': content_hash(records), 'records': len(records),
        'unique_tool_names': len({r['name'] for r in records}),
        'groups': {stage: group_tools(describe_tools(stage, list(baseline[stage])))
                   for stage in ('scene', 'robot_layout')},
        'negative_cases': negatives, 'model_requests': 0,
        'not_run': ['new_scene_generation', 'robot_execution'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records, report = check()
    (args.output_dir / 'catalog.json').write_text(json.dumps(records, indent=2) + '\n')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
