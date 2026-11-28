#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check SPEC-005's three-delivery DoD, without changing the deliveries.

Independent verification only proves a delivery honestly states its result. This command also
requires functional readiness and the referenced shell; an honestly failed delivery must fail it.
"""
from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rfw_w.scenes.workers.isaac.protocol import WorkerReport

EXPECTED = {'office_pick_cup', 'kitchen_fetch_jar', 'bedroom_pick_alarm_clock'}


def check(delivery: Path) -> dict:
    result = {'delivery': str(delivery), 'gaps': []}
    gaps = result['gaps']
    try:
        manifest = json.loads((delivery / 'delivery_manifest.json').read_text())
        result['scene_id'] = manifest['scene_id']
        result['elapsed_s'] = manifest['budget']['elapsed_s']
        resolution = manifest['resolution']
        result['resolution'] = resolution
        verification = subprocess.run(
            [sys.executable, '-m', 'rfw_w.scenes.validation.verify_delivery',
             '--delivery', str(delivery)], capture_output=True, text=True,
        )
        try:
            result['independent_verification'] = json.loads(
                verification.stdout if verification.returncode == 0 else verification.stderr)
        except ValueError:
            result['independent_verification'] = {'verified': False, 'reason': 'verifier did not return JSON'}
        if verification.returncode or not result['independent_verification'].get('verified'):
            gaps.append('DELIVERY_INDEPENDENT_VERIFICATION_FAILED')
        if (resolution.get('achieved') != 'functional_scene_ready'
                or resolution.get('meets_request') is not True
                or resolution.get('failed_gates') != []):
            gaps.append('SPEC005_FUNCTIONAL_READINESS_NOT_MET')
        record = json.loads((delivery / 'shell_reference.json').read_text())
        if record.get('mode') != 'referenced' or record.get('scene_version') != manifest['scene_version']:
            gaps.append('SPEC005_REFERENCED_SHELL_NOT_CURRENT')
        # The independent verifier checks containment and provenance of these paths.
        worker_dir = (delivery / manifest['isaac_artifact_dir']).resolve()
        report = WorkerReport.from_dict(json.loads((worker_dir / 'report.json').read_text()))
        report.assert_self_consistent()
        if report.status != 'passed':
            gaps.append('SPEC005_WORKER_FAILED')
        result['worker_status'] = report.status
        result['rgb_files'] = [str(p.relative_to(delivery)) for p in sorted(worker_dir.glob('*.png'))]
        if not result['rgb_files']:
            gaps.append('SPEC005_RGB_MISSING')
    except (OSError, ValueError, KeyError, AssertionError, TypeError) as error:
        gaps.append(f'SPEC005_EVIDENCE_INVALID({type(error).__name__}: {error})')
    result['passed'] = not gaps
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', action='append', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory is not empty; use a fresh evidence directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [check(p.resolve()) for p in args.delivery]
    found = [row.get('scene_id') for row in rows]
    gaps = [] if len(found) == 3 and set(found) == EXPECTED else ['SPEC005_THREE_EXPERIMENTS_REQUIRED']
    report = {'spec': 'SPEC-005', 'deliveries': rows, 'gaps': gaps,
              'passed': not gaps and all(row['passed'] for row in rows),
              'scope': 'complete delivery acceptance; regression and human visual review reported separately'}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
