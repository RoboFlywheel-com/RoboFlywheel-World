#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure whether a capture's task target is lit, in linear Rec.709 luminance.

Two modes, and the first is the one to start with:

    # measure captures that already exist, no rendering and no GPU
    python -m rfw_w.scenes.validation.check_lighting --captures <delivery isaac dir> --output-dir <new dir>

    # render and measure a matrix of cases (TASK 2 of the SPEC-008 plan)
    python -m rfw_w.scenes.validation.check_lighting --matrix tests/fixtures/lighting_matrix.json --output-dir <new dir>

The readings are SPEC-008's proposed thresholds: among the pixels of the task target, at most a
fifth may be darker than Y=0.01 and at most a twentieth brighter than Y=0.98. A capture whose
target the camera cannot see at all is refused rather than measured as zero.

Every run also drives a black frame through the same check and requires it to fail. A measurement
that cannot fail measures nothing, and this is the cheapest way to keep saying so.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()


def _metrics():
    return importlib.import_module('rfw_w.scenes.visualization.lighting_metrics')


def _linear_decode(numpy, encoded):
    """The module's sRGB decode, applied to an array. Its constants, its formula, one place."""
    metrics = _metrics()
    low = encoded <= 0.04045
    return numpy.where(low, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4).astype('float64')


def _luminance(numpy, rgb):
    metrics = _metrics()
    linear = _linear_decode(numpy, rgb[:, :, :3].astype('float64') / 255.0)
    return (metrics.REC709[0] * linear[:, :, 0] + metrics.REC709[1] * linear[:, :, 1]
            + metrics.REC709[2] * linear[:, :, 2])


def _mask_for(numpy, capture, delivery, id_map):
    """The task target's pixels, from the instance-id channel the worker already saved."""
    ids = numpy.load(delivery / capture['instance_id_path'])
    wanted = [int(key) for key, name in id_map.items()
              if name in capture['target_instance_ids'] and key.isdigit()]
    if not wanted:
        raise ValueError(f'CAMERA_TARGET_NOT_VISIBLE({capture["camera_id"]}: no id for target)')
    return numpy.isin(ids, wanted)


def measure(delivery: Path) -> list[dict]:
    import numpy as np
    from PIL import Image

    metrics = _metrics()
    report = json.loads((delivery / 'report.json').read_text())
    id_map = report['instance_id_map']
    rows = []
    for capture in report['captures']:
        row = {'camera_id': capture['camera_id'],
               'target_instance_ids': list(capture['target_instance_ids'])}
        if not capture['target_instance_ids']:
            row['status'] = 'not_run'
            row['reason'] = 'camera declares no task target, so it has nothing to measure'
            rows.append(row)
            continue
        if not capture.get('instance_id_path'):
            row['status'] = 'blocked'
            row['failure'] = f'CAMERA_TARGET_NOT_VISIBLE({capture["camera_id"]}: no id channel)'
            rows.append(row)
            continue
        try:
            mask = _mask_for(np, capture, delivery, id_map)
            if not mask.any():
                raise ValueError(f'CAMERA_TARGET_NOT_VISIBLE({capture["camera_id"]}: mask empty)')
            rgb = np.asarray(Image.open(delivery / f'{capture["camera_id"]}.png').convert('RGB'))
            values = _luminance(np, rgb)[mask]
            stats = metrics.statistics(values.tolist())
            row['status'] = 'ran'
            row['statistics'] = {key: (round(value, 6) if isinstance(value, float) else value)
                                 for key, value in stats.items()}
            row['summary'] = metrics.exposure_summary(stats)
            metrics.check_camera(capture['camera_id'], stats)
            row['passed'] = True
        except ValueError as error:
            row.setdefault('status', 'ran')
            row['passed'] = False
            row['failure'] = str(error)
        rows.append(row)
    return rows


def black_frame_negative(rows: list[dict]) -> dict:
    """The same check, driven with a black frame; it has to fail or the reading means nothing."""
    import numpy as np

    metrics = _metrics()
    measured = [row for row in rows if row['status'] == 'ran']
    if not measured:
        return {'passed': False, 'reason': 'no camera produced a reading to test the check against'}
    black = metrics.statistics(np.zeros(measured[0]['statistics']['target_pixels']).tolist())
    try:
        metrics.check_camera('black_frame', black)
    except ValueError as error:
        return {'passed': str(error).startswith('TARGET_UNDEREXPOSED'), 'failure': str(error)}
    return {'passed': False, 'reason': 'a black frame passed the exposure check'}


MATRIX_SCHEMA_VERSION = 'pvscene.lighting-matrix/v1'


def _relative(path: Path, base: Path) -> str:
    """A path as written in the report: relative when it is under ``base``, else absolute.

    A rejudged row's render lives under the *previous* matrix's directory, so assuming it is under
    this run's output would raise -- and it would raise inside the per-case `try`, turning every
    rejudged case into a `blocked` with a message about the wrong thing.
    """
    resolved = path.resolve()
    return str(resolved.relative_to(base.resolve())) if base.resolve() in resolved.parents \
        else str(resolved)


def _room_extent(delivery: Path) -> dict:
    """The delivered room's width, depth and height -- what a profile is sized to.

    Read from the delivery's own manifest rather than from the fixture: the matrix's cases declare a
    *room*, and if the built room does not match the declaration the render would be measured
    against the wrong profile. The manifest is where the delivery publishes what it built.
    """
    manifest = json.loads((delivery / 'delivery_manifest.json').read_text())
    room = manifest.get('room') or {}
    # `room.vertices` is the vertex *count* in this manifest, not the polygon; the room the zones
    # were planned against is published under `zones.room_polygon_xy`, and it is the same polygon the
    # SPEC-009 audit replays and compares against.
    vertices = (manifest.get('zones') or {}).get('room_polygon_xy') or []
    if not vertices or not room.get('height_m'):
        raise ValueError(f'LIGHT_ROOM_INVALID({delivery}: the manifest publishes no room extent)')
    xs = [float(point[0]) for point in vertices]
    ys = [float(point[1]) for point in vertices]
    return {'room_width_m': max(xs) - min(xs), 'room_depth_m': max(ys) - min(ys),
            'room_height_m': float(room['height_m'])}


def _load_lighting():
    """The lighting contract, imported from `src` so a case renders what the run would render."""
    from rfw_w.scenes.contracts import lighting  # noqa: PLC0415
    return lighting


def render_case(case: dict, delivery: Path, output_dir: Path, *, gpu: int) -> Path:
    """Render one case's lighting over a delivery that already exists, and say where it landed.

    A *render*, not a rebuild: the profile is a property of the light, and the room, the retrieval
    and the target are already in the parent delivery. That is what makes 18 groups affordable, and
    it is also the honest experiment -- two cases then differ in exactly one thing.

    The parent's own request is patched rather than rebuilt, so everything else about the render
    (cameras, instances, the shell) is byte-for-byte what the delivery was rendered with.
    """
    requests = sorted((delivery / 'isaac_requests').glob('*/attempt_001.json'))
    if not requests:
        raise ValueError(f'LIGHT_RENDER_UNAVAILABLE({delivery}: no recorded worker request)')
    payload = json.loads(requests[-1].read_text())
    room = _room_extent(delivery)
    lighting = _load_lighting()
    plan = lighting.profile(case['profile'], **room, seed=int(case.get('seed', 0)))
    payload['lighting'] = plan.to_request_fields()
    # The shell URI is relative to the request's own directory, so it is made absolute before the
    # patch is written somewhere else: a copied request with a relative path would resolve against
    # the copy's directory and the worker would refuse a shell that is not there.
    if payload.get('room', {}).get('shell_uri'):
        payload['room']['shell_uri'] = str(
            (requests[-1].parent / payload['room']['shell_uri']).resolve())
    request_dir = output_dir / 'requests'
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{case['id']}.json"
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    variant = output_dir / 'renders' / str(case['id'])
    # The render directory is left for the *worker* to create: it refuses an output directory that
    # already exists ("a stale report must not look current"), which is the same rule this
    # repository's runs follow -- so the case's own records live beside the render, not inside it.
    variants = output_dir / 'variants'
    variants.mkdir(parents=True, exist_ok=True)
    (variants / f"{case['id']}.json").write_text(json.dumps({
        'case': case['id'], 'profile': plan.profile, 'calibration': plan.calibration,
        'room_m': room, 'request': plan.to_request_fields(),
        'named_gaps': list(plan.named_gaps()),
        'parent_delivery': str(delivery),
        'patched_from': str(requests[-1]),
    }, indent=2, sort_keys=True) + '\n')
    completed = subprocess.run(
        [sys.executable, "-m", 'rfw_w.scenes.workers.launch',
         '--request', str(request_path), '--output-dir', str(variant),
         '--active-gpu', str(gpu), '--timeout', '1200'],
        capture_output=True, text=True)
    (variant / 'worker.log').write_text((completed.stdout or '') + '\n' + (completed.stderr or ''))
    if not (variant / 'report.json').is_file():
        raise ValueError(f'LIGHT_RENDER_UNAVAILABLE({case["id"]}: the worker wrote no report; '
                         f'see {variant / "worker.log"})')
    return variant


def run_matrix(fixture_path: Path, output_dir: Path, *, gpu: int,
               rejudge_from: Path | None = None) -> int:
    """SPEC-008's 18 groups: three room types x small/large x three profiles, rendered and measured.

    A case whose parent delivery does not exist is `not_run` with the command that produces it --
    never a pass, and never silently skipped. The matrix measures what is on disk and says what is
    missing.

    **What the acceptance is, and what is only measured.** Only the calibrated profile is held to
    the thresholds: `mixed` is the profile the default path renders with, and "the light in every
    room is inside the thresholds" is a claim about *it*. `daylight` and `warm_artificial` carry
    `calibration: not_run` in the contract and in every delivery block; their cases are rendered and
    measured so a reader can see what they do, and their failures are reported beside that word
    rather than counted as a failure of the calibrated profile -- or hidden by counting them as
    passes.

    ``rejudge_from`` recomputes the verdict from a previous matrix run's renders: measuring costs
    seconds and rendering costs half an hour, so a change to *how* the rows are judged must not mean
    re-rendering the same eighteen pictures.
    """
    fixture = json.loads(fixture_path.read_text())
    version = fixture.get('schema_version')
    if version != MATRIX_SCHEMA_VERSION:
        raise SystemExit(f'LIGHT_MATRIX_SCHEMA_UNSUPPORTED({version})')
    cases = list(fixture.get('cases', ()))
    if not cases:
        raise SystemExit('LIGHT_MATRIX_EMPTY(no cases)')
    calibrated_profile = str(fixture.get('calibrated_profile', 'mixed'))
    previous: dict[str, dict] = {}
    if rejudge_from is not None:
        prior = json.loads((rejudge_from / 'report.json').read_text())
        previous = {str(row['id']): row for row in prior.get('cases', ())}
    rows = []
    for case in cases:
        row = {'id': case['id'], 'room_type': case.get('room_type'), 'scale': case.get('scale'),
               'profile': case['profile'], 'seed': case.get('seed')}
        delivery = ROOT / str(case['delivery'])
        if not (delivery / 'delivery_manifest.json').is_file():
            row.update(status='not_run', not_run_reason=str(case.get('command', 'no command')),
                       note=str(case.get('case_note', '')))
            rows.append(row)
            print(f"{case['id']:24s} not_run")
            continue
        try:
            if rejudge_from is not None:
                recorded = previous.get(str(case['id']), {}).get('rendered')
                if not recorded:
                    raise ValueError(f'LIGHT_REJUDGE_INPUT_MISSING({case["id"]}: no render in '
                                     f'{rejudge_from})')
                rendered = rejudge_from / str(recorded)
                cameras = measure(rendered)
            else:
                rendered = render_case(case, delivery, output_dir, gpu=gpu)
                cameras = measure(rendered)
            negative = black_frame_negative(cameras)
            row.update(status='ran', rendered=_relative(rendered, output_dir),
                       cameras=cameras, black_frame_negative=negative,
                       profile_calibration=_load_lighting().CALIBRATION[case['profile']])
            row['holds_to_the_thresholds'] = case['profile'] == calibrated_profile
            row['measured_passed'] = (all(camera.get('passed', False) for camera in cameras
                                          if camera['status'] == 'ran')
                                      and negative['passed'])
        except (ValueError, SystemExit) as error:
            row.update(status='blocked', failure=str(error), measured_passed=False)
        rows.append(row)
        print(f"{case['id']:24s} {row['status']:8s} "
              f"{'+' if row.get('measured_passed') else '-'} "
              + '; '.join(f"{c['camera_id']}={c.get('summary') or c.get('failure') or c.get('reason')}"
                          for c in row.get('cameras', ()))[:110])
    ran = [row for row in rows if row['status'] == 'ran']
    calibrated_rows = [row for row in ran if row.get('holds_to_the_thresholds')]
    by_profile = {
        name: {'ran': len([row for row in ran if row['profile'] == name]),
               'passed': sum(1 for row in ran if row['profile'] == name
                             and row['measured_passed']),
               'calibration': _load_lighting().CALIBRATION[name],
               'holds_to_the_thresholds': name == calibrated_profile}
        for name in _load_lighting().PROFILES}
    report = {
        'scope': 'SPEC-008 TASK 3: 3 room types x small/large x 3 profiles, rendered from the '
                 'parent delivery of each room and measured on the task target',
        'fixture': (str(fixture_path.resolve().relative_to(ROOT))
                    if ROOT in fixture_path.resolve().parents else str(fixture_path.resolve())),
        'matrix_schema_version': MATRIX_SCHEMA_VERSION,
        'rejudged_from': None if rejudge_from is None else str(rejudge_from),
        'acceptance': (
            f'the calibrated profile (`{calibrated_profile}`) is held to the thresholds in every '
            f'room; the other profiles carry `calibration: not_run` and are rendered and measured '
            f'so a reader can see what they do, not as passes and not as failures of this one'),
        'thresholds': {'underexposed_limit': _metrics().UNDEREXPOSED_LIMIT,
                       'overexposed_limit': _metrics().OVEREXPOSED_LIMIT,
                       'dark_y': _metrics().DARK_Y, 'bright_y': _metrics().BRIGHT_Y,
                       'space': 'linear Rec.709 relative luminance decoded from sRGB',
                       'frozen_by': 'this run, for the calibrated profile'},
        'profiles': {name: {'levels': _load_lighting().PROFILE_LEVELS[name],
                            'calibration': _load_lighting().CALIBRATION[name]}
                     for name in _load_lighting().PROFILES},
        'cases': rows,
        'cases_ran': len(ran),
        'cases_not_run': [row['id'] for row in rows if row['status'] == 'not_run'],
        'cases_blocked': [row['id'] for row in rows if row['status'] == 'blocked'],
        'by_profile': by_profile,
        'calibrated_profile': calibrated_profile,
        # What the acceptance is: every case ran, the black-frame selftest passed everywhere (a
        # measurement that cannot fail measures nothing), and the calibrated profile is inside the
        # thresholds in all six rooms.
        'measured_passed': bool(calibrated_rows) and len(calibrated_rows) == len(
            [row for row in rows if row['profile'] == calibrated_profile])
        and all(row['measured_passed'] for row in calibrated_rows)
        and all(row['black_frame_negative']['passed'] for row in ran),
        'measured_passed_all_profiles': bool(ran) and all(row['measured_passed'] for row in ran),
        'human_review': 'not_run',
        'spec_accepted': False,
    }
    report['spec_accepted'] = bool(report['measured_passed'] and not report['cases_blocked']
                                   and not report['cases_not_run'])
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, summary in by_profile.items():
        print(f"{name:16s} {summary['passed']}/{summary['ran']} passed, "
              f"calibration={summary['calibration']}"
              + ('' if summary['holds_to_the_thresholds'] else ' (measured, not held to it)'))
    return 0 if report['spec_accepted'] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, help='a worker run to measure without rendering')
    parser.add_argument('--matrix', type=Path, help='cases to render, measured as they land')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--active-gpu', type=int, default=0)
    parser.add_argument('--rejudge-from', type=Path, default=None,
                        help='a previous matrix output directory: re-measure its renders and '
                             're-judge them, rendering nothing (a verdict change must not cost '
                             'eighteen renders)')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output exists; use fresh evidence')
    if bool(args.captures) == bool(args.matrix):
        parser.error('exactly one of --captures or --matrix')
    if args.rejudge_from is not None and args.matrix is None:
        parser.error('--rejudge-from re-judges a matrix; give --matrix as well')
    args.output_dir.mkdir(parents=True)
    if args.matrix:
        return run_matrix(args.matrix, args.output_dir, gpu=args.active_gpu,
                          rejudge_from=args.rejudge_from)
    rows = measure(args.captures)
    negative = black_frame_negative(rows)
    report = {'scope': 'target-mask luminance on captures that already exist; nothing rendered',
              'captures_dir': str(args.captures),
              'thresholds': {'underexposed_limit': _metrics().UNDEREXPOSED_LIMIT,
                             'overexposed_limit': _metrics().OVEREXPOSED_LIMIT,
                             'dark_y': _metrics().DARK_Y, 'bright_y': _metrics().BRIGHT_Y,
                             'space': 'linear Rec.709 relative luminance decoded from sRGB'},
              'cameras': rows, 'black_frame_negative': negative,
              'measured_passed': all(row.get('passed', False) for row in rows
                                     if row['status'] == 'ran') and negative['passed'],
              'human_review': 'not_run',
              'spec_accepted': False}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        print(f"{row['camera_id']:16s} {row['status']:8s} "
              f"{row.get('summary') or row.get('failure') or row.get('reason')}")
    print(f"black frame negative: {'passed' if negative['passed'] else 'DID NOT FAIL'}")
    return 0 if report['measured_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
