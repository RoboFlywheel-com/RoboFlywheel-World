#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Attribute SPEC-011's two measured numbers by comparing runs of the *same* room.

TASK 2 measured, for one bedroom, both a zero intrusion and a much emptier room (32 -> 13 generated
objects). Neither number is attributable from the injected run alone: three things changed at once
between it and SPEC-009's probe -- the entry script, the machinery in the solver's state, and the
costs that decide the annealing -- and the solver's sampling depends on the costs, so the object
count is not a fixed-seed constant across variants. TASK 3's first job is therefore a *control*:
`probe_constraint_injection.py --reservations degenerate` sends the same objects, tags, relations
and terms over 5 cm squares, so the machinery is identical and only the geometry fails to bite. The
ruler measures every run against the *real* regions (it re-derives them from the delivery, never
from a bundle), so what the degenerate run shows is what generation would have placed inside the
task area had the constraint not been sent.

This tool reads those measurements and writes one table. It measures and compares; it does not
judge. The verdicts (`BACKGROUND_PROTECTED_OVERLAP`, the per-case green/red) stay in
`check_background_constraints.py` and the contract, which is also why nothing here is called
"passed":

    env -u PYTHONPATH -u PYTHONHOME /opt/conda/envs/scenesmith/bin/python \\
      python -m rfw_w.scenes.validation.compare_constraint_runs \\
        --run control=outputs/_probe/spec011_probe_regions_20260923_01/generated_furniture_regions.json \\
        --run ablation=outputs/_probe/spec011_ablation_20260923_01/generated_furniture.json \\
        --run injected=outputs/_probe/spec011_injection_20260923_03/generated_furniture.json \\
        --effect ablation>injected \\
        --output-dir outputs/_acceptance/spec011_task3_20260923_01

What it refuses rather than compares: runs of different rooms. A delta between two rooms is not a
constraint's effect, and this tool exists because a number that cannot be attributed gets quoted
anyway.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path.cwd()

from rfw_w.scenes.contracts.background_constraints import (
    INJECTION_FILENAME,
    REGION_CIRCULATION,
    REGION_DOOR_SWEEP,
    REGION_PROTECTED,
    SOLVE_STAGES,
)

AUDIT = 'python -m rfw_w.scenes.validation.check_background_constraints'
ENTRY_OURS = 'run_infinigen_build.py'
ENTRY_WRAPPER = 'run_generate_indoors.py'


class ComparisonInputError(RuntimeError):
    """The comparison cannot be made from the input given, named rather than reported as a delta."""


def _load_audit():
    return importlib.import_module('rfw_w.scenes.validation.check_background_constraints')


def _relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if ROOT in resolved.parents else str(resolved)


def _rings(polygons: Sequence[Sequence[Sequence[float]]], *, places: int = 6):
    return sorted(tuple(sorted((round(float(x), places), round(float(y), places)) for x, y in ring))
                  for ring in polygons)


def _read_json(path: Path, what: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ComparisonInputError(f"CALIBRATION_INPUT_MISSING({what}): {path}")
    return json.loads(path.read_text())


def _entry_of(command: str) -> str:
    """Which entry script the build ran, read from its own `command.txt`.

    Not from the record: the record is written by the patch, and a run's entry is a fact about the
    command line. `unknown` is a real answer -- a wrapper invocation that says nothing about which
    script it is must not be filed as ours.
    """
    for token in command.split():
        if token.endswith(ENTRY_OURS):
            return 'ours'
        if token.endswith(ENTRY_WRAPPER):
            return 'wrapper'
    return 'unknown'


def describe_run(label: str, measurement_path: Path) -> dict[str, Any]:
    """Everything one run measured, in the order a reader needs it to attribute a number."""
    audit = _load_audit()
    measurement = _read_json(measurement_path, f'run:{label}')
    build_dir = Path(str(measurement.get('build', '')))
    objects = [entry for entry in measurement.get('objects', ()) if entry.get('generated')]

    intrusion = {region: round(sum(float(entry[f'{region}_overlap_m2']) for entry in objects), 4)
                 for region in (REGION_PROTECTED, REGION_CIRCULATION, REGION_DOOR_SWEEP)}
    intruders = [{'name': str(entry['name']), 'factory': str(entry['factory']),
                  **{f'{region}_overlap_m2': float(entry[f'{region}_overlap_m2'])
                     for region in (REGION_PROTECTED, REGION_CIRCULATION, REGION_DOOR_SWEEP)}}
                 for entry in objects
                 if any(float(entry[f'{region}_overlap_m2']) > 0.0
                        for region in (REGION_PROTECTED, REGION_CIRCULATION, REGION_DOOR_SWEEP))]

    row: dict[str, Any] = {
        'label': label,
        'measurement': _relative(measurement_path),
        'measurement_tool_version': measurement.get('tool_version'),
        'build': _relative(build_dir) if str(build_dir) else '',
        'delivery': str(measurement.get('delivery', '')),
        'room_polygon_xy': measurement.get('reservations', {}).get('room_polygon_xy'),
        'bands': measurement.get('bands'),
        'counts': measurement.get('counts', {}),
        'generated_objects': len(objects),
        'factories': {factory: sum(1 for entry in objects if entry['factory'] == factory)
                      for factory in sorted({str(entry['factory']) for entry in objects})},
        # The three regions stay separate: a sum would answer "did anything intrude" and lose the
        # question SPEC-011 asks -- *which* region, because the repairs differ.
        'intrusion_m2': intrusion,
        'intruding_objects': intruders,
        'keepout_note': 'keepout_overlap_m2 is SPEC-009\'s aggregate on the single old band, not a '
                        'sum of the three regions, so it is not reported as a total here',
    }

    command_path, log_path = build_dir / 'command.txt', build_dir / 'stderr.txt'
    if command_path.is_file():
        command = command_path.read_text()
        overrides = audit._compose_overrides(command)
        row['entry'] = _entry_of(command)
        row['generation'] = any(overrides.get(f'{stage}_enabled')
                                for stage in audit.GENERATING_STAGES)
        row['abort_switches'] = {stage: bool(overrides.get(
            audit.abort_param(stage).removeprefix('compose_indoors.'))) for stage in SOLVE_STAGES}
    else:
        row['entry'] = 'unknown'
        row['generation'] = None
        row['abort_switches'] = None
    # A probe that failed once and then succeeded keeps both logs side by side (`attempt1_stderr.txt`
    # is the interpreter that could not import; `attempt2_stderr.txt` is the build), which is exactly
    # the layout of SPEC-009's control. Reading only `stderr.txt` would report that control as "0
    # unsatisfied groups" -- an absence of measurement printed as a measurement, which is the one
    # thing this column must not do.
    if not log_path.is_file():
        attempts = sorted(build_dir.glob('attempt*_stderr.txt'))
        log_path = attempts[-1] if attempts else log_path
    row['log'] = _relative(log_path) if log_path.is_file() else ''
    if log_path.is_file():
        log = log_path.read_text()
        counts, violations = audit._solve_outcomes(log)
        row['unsatisfied_groups'] = {stage: counts[stage] if audit._stage_finished(log, stage)
                                     else None for stage in SOLVE_STAGES}
        row['unsatisfied_violations'] = violations
    else:
        row['unsatisfied_groups'] = None
        row['unsatisfied_violations'] = None

    record_path = build_dir / INJECTION_FILENAME
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        row['record'] = {
            'injection': record.get('injection'),
            'injection_reason': record.get('injection_reason'),
            'entry_script': record.get('entry_script'),
            'patch_module': record.get('patch_module'),
            'digests_recorded': all(record.get(key) for key in
                                    ('upstream_pin', 'entry_script_sha256', 'patch_sha256')),
            'reservation_objects': [item.get('object') for item in
                                    record.get('reservation_objects', ())],
            'objects_removed_before_export': record.get('objects_removed_before_export'),
            'abort_unsatisfied': record.get('abort_unsatisfied'),
        }
    else:
        row['record'] = None
    probe_path = build_dir.parent / 'probe.json'
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text())
        row['probe'] = {'ok': probe.get('ok'), 'error': probe.get('error'),
                        'elapsed_s': probe.get('elapsed_s'),
                        'reservation_mode': probe.get('reservation_mode'),
                        'policy': probe.get('policy')}
    else:
        row['probe'] = None
    return row


def effects(rows: Sequence[Mapping[str, Any]], pairs: Sequence[str]) -> list[dict[str, Any]]:
    """The arithmetic between two runs, labelled with what differs between them.

    It is arithmetic on two measurements, not a significance test: each side is one build, and the
    solver's acceptance depends on the cost values, so the two runs do not consume randomness
    identically. The note says both things in the record rather than in a comment here.
    """
    by_label = {row['label']: row for row in rows}
    out: list[dict[str, Any]] = []
    for pair in pairs:
        if '>' not in pair:
            raise ComparisonInputError(f"CALIBRATION_EFFECT_MALFORMED({pair}): use base>other")
        base_label, _, other_label = pair.partition('>')
        for label in (base_label, other_label):
            if label not in by_label:
                raise ComparisonInputError(f"CALIBRATION_EFFECT_UNKNOWN_RUN({pair}:{label})")
        base, other = by_label[base_label], by_label[other_label]
        if _rings([base['room_polygon_xy'] or []]) != _rings([other['room_polygon_xy'] or []]):
            raise ComparisonInputError(
                f"CALIBRATION_RUNS_NOT_THE_SAME_ROOM({base_label},{other_label}): a delta between "
                f"two rooms is not the effect of anything")
        out.append({
            'base': base_label, 'other': other_label,
            'delta': {
                'generated_objects': other['generated_objects'] - base['generated_objects'],
                **{f'{region}_m2': round(other['intrusion_m2'][region]
                                         - base['intrusion_m2'][region], 4)
                   for region in (REGION_PROTECTED, REGION_CIRCULATION, REGION_DOOR_SWEEP)},
            },
            'note': 'arithmetic on two measurements of the same room: each side is one build, and '
                    'acceptance depends on the cost values, so the two runs do not consume the '
                    'solver\'s randomness identically. A delta is a paired reading at these seeds, '
                    'not a variance estimate',
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', default=[], metavar='LABEL=PATH',
                        help='a run\'s ruler output (generated_furniture*.json)')
    parser.add_argument('--effect', action='append', default=[], metavar='BASE>OTHER',
                        help='a pair whose difference is the thing being attributed')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--note', default='', help='what this comparison is for')
    args = parser.parse_args()
    if not args.run:
        parser.error('at least one --run LABEL=PATH is needed')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('use a fresh output directory: a comparison is evidence and is not overwritten')

    rows = []
    for item in args.run:
        label, _, path = item.partition('=')
        if not label or not path:
            parser.error(f'--run needs LABEL=PATH, got {item!r}')
        rows.append(describe_run(label, Path(path)))
    attributed = effects(rows, args.effect)

    report = {
        'tool': 'python -m rfw_w.scenes.validation.compare_constraint_runs',
        'spec': 'SPEC-011-background-constraint-injection',
        'task': 'TASK 3 first step: attribute the injected build\'s zero intrusion and its lower '
                'object count by comparing runs of the same room',
        'note': args.note,
        'runs': rows,
        'effects': attributed,
        'not_measured_here': [
            'rejections: the build log carries no rejection records for these runs, so "how many '
            'candidates were rejected" is TASK 4\'s (the audit manifest reports rejection_bounds '
            'empty for a real build rather than a fabricated 0)',
            'room types other than this one: the same comparison on kitchen and office is what '
            'turns a single-room reading into a claim',
        ],
        'reused_from': f'{AUDIT} for `_solve_outcomes`, `_stage_finished`, `_compose_overrides`',
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'calibration.json').write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'run':<12}{'entry':<10}{'gen':<6}{'objs':>5}  "
          f"{'protected':>10}{'circulation':>12}{'door_sweep':>12}  unsatisfied")
    for row in rows:
        groups = row['unsatisfied_groups']
        # `n/a` rather than a sum of Nones: a stage whose `finished in` line is absent, or a build
        # whose log was not found, has *not* been observed to satisfy anything.
        total = 'n/a' if groups is None else str(sum(v for v in groups.values() if v is not None))
        print(f"{row['label']:<12}{row['entry']:<10}{str(row['generation']):<6}"
              f"{row['generated_objects']:>5}  {row['intrusion_m2'][REGION_PROTECTED]:>10.4f}"
              f"{row['intrusion_m2'][REGION_CIRCULATION]:>12.4f}"
              f"{row['intrusion_m2'][REGION_DOOR_SWEEP]:>12.4f}  {total}")
    for item in attributed:
        print(f"effect {item['base']} > {item['other']}: {item['delta']}")
    print(f"-> {args.output_dir / 'calibration.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
