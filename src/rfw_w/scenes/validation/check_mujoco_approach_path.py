# SPDX-License-Identifier: Apache-2.0
"""Save deterministic sampled joint approach and its actual collision evidence."""
import argparse
import json
import os
import time
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_mujoco_approach_path import sample, source_hashes, verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = a.delivery.resolve()
    if not source.is_relative_to(a.output_dir.resolve().parent):
        raise ValueError('APPROACH_SOURCE_MISMATCH')
    start = json.loads((source / 'source/source/installation.json').read_text())['reference_joints']
    end = json.loads((source / 'ik_case.json').read_text())['candidate']
    meta = {'source_ref': os.path.relpath(source, a.output_dir), 'source_hashes': source_hashes(source),
            'method': 'linear joint samples; max .01rad/.001m', 'frames': sample(start, end)}
    (a.output_dir / 'path.json').write_text(json.dumps(meta, indent=2) + '\n')
    (a.output_dir / 'path.sha256').write_text(digest(a.output_dir / 'path.json') + '\n')
    try:
        report = verify(a.output_dir)
    except (ValueError, OSError, KeyError) as error:
        report = {'verified': False, 'failure_code': str(error)}
    report['elapsed_s'] = round(time.monotonic() - started, 3)
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
