#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one opt-in build with the task's reservations injected, and report what the build recorded.

SPEC-011 TASK 2's measurement entry. It exists because the constraint must be *sent into* a real
build to be worth anything (requirement 1), and because the weights are not calibrated yet (TASK 3):
this tool is how a candidate weight set is tried without editing code.

The room is the one a delivery already defines: its effective spec and recorded `room_seed` are
replayed exactly the way `check_background_constraints.py` replays them, so the reservations this
build is asked to respect are the same polygons SPEC-009's probe intruded into -- the对照 only works
if the room is the same room. The delivery's own recorded zones supply the corridor and the door
sweep, so nothing here is a second derivation of them.

**`--reservations degenerate` is TASK 3's control, not a product mode.** The injected build measured
both a zero intrusion and a much emptier room (32 -> 13 objects), and neither number can be
attributed from the injected build alone: the entry changed, the machinery was present, and the
solver consumes randomness in an order that depends on the costs. A degenerate run sends the *same
number of objects, tags, relations and terms*, with each polygon shrunk to a 5 cm square at its own
centroid, so the machinery is identical and only the geometry fails to bite. What the ruler then
measures against the *real* regions (it re-derives them from the delivery, never from this bundle)
is the counterfactual: what generation would have put inside the task area had the constraint not
been sent. The difference between that run and the injected one is the constraint's own effect.

**`--gaps` is what makes a build's generation bounded** (SPEC-011 TASK 4): the categories this room
allows generation to fill travel into the build, where the rejection pool refuses anything outside
them -- a covered category, a category this room never asked for, or a piece standing in a reserved
region, each with its own named reason in the record. A run that asks for generation without
declaring gaps is refused by the provider (`BACKGROUND_BUNDLE_EMPTY(no gaps to fill)`), because the
pool would otherwise refuse everything the build realises.

What it does *not* do: judge. The build's record, the intrusion measurement
(`python -m rfw_w.scenes.workers.blender.measure_generated_furniture`, run separately under Blender) and the audit
(`python -m rfw_w.scenes.validation.check_background_constraints`) each say one thing, and this tool is only the first.

    env -u PYTHONPATH -u PYTHONHOME python -m rfw_w.scenes.preparation.probe_constraint_injection \\
        --delivery outputs/bedroom_pick_alarm_clock/s005_bedroom_full_20260922_02 \\
        --output-dir outputs/_probe/spec011_injection_20260923_01 --no-abort
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path.cwd()

from rfw_w.scenes.contracts.background_constraints import (
    UNCALIBRATED_POLICY_NOTE,
    ReservationPolicy,
)
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.room import fit_room_around_cell
from rfw_w.scenes.solvers.room_plan import (
    plan_from_fitted,
    reservations_from_plan,
)
from rfw_w.scenes.solvers.shell_providers import (
    BackgroundConstraints,
    InfinigenShellProvider,
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"PROBE_INPUT_MISSING({path})")
    return json.loads(path.read_text())


def load_local_env() -> None:
    """`.env.local` holds the machine-bound roots, and a bare interpreter does not read it.

    `scripts/dev/test.sh` exports it for the test tiers; this tool runs outside them, and the one
    thing it must not do is fall back to "an infinigen somewhere on `sys.path`" -- the provider
    refuses that by name (`RFW_INFINIGEN_ROOT` unset), which is what happened the first time
    this probe was run.
    """
    path = ROOT / '.env.local'
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip().strip('\'"'))


def degenerate(polygon: Polygon, half_m: float) -> Polygon:
    """The polygon shrunk to a square of `2*half_m` a side around its own centroid.

    A square rather than the original ring scaled down: a scaled ring keeps its own vertex count,
    and the point is that nothing about the *request* changes except the area it covers. The
    centroid is inside the region, so the reservation object is still a room object -- which is
    what keeps the ablation honest, because an object that fails `_is_active_room_object` would
    take the constraints out of the graph entirely and measure something else.
    """
    cx = sum(x for x, _ in polygon.vertices) / len(polygon.vertices)
    cy = sum(y for _, y in polygon.vertices) / len(polygon.vertices)
    return Polygon([(cx - half_m, cy - half_m), (cx + half_m, cy - half_m),
                    (cx + half_m, cy + half_m), (cx - half_m, cy + half_m)])


def ablate(reservations, half_m: float):
    """The same requests with every polygon replaced by a square that cannot bite.

    Region, index, band and the number of polygons all survive: what the ablation changes is the
    area the constraint is about. The source keeps the original and says what was done to it, so
    the bundle on disk cannot be read as "the plan asked for these".
    """
    return tuple(
        replace(item, source=f'ablation(degenerate):{half_m}m-square of {item.source}',
                polygon_xy=degenerate(Polygon(item.polygon_xy), half_m).vertices)
        for item in reservations)


def replay(delivery: Path):
    """The delivery's own room, rebuilt from its effective spec and its recorded room seed."""
    spec = SceneSpec.from_dict(_read_json(delivery / 'effective_spec.json'))
    room_seed = int(_read_json(delivery / 'generation_attempt_001.json')['room_seed'])
    core = solve_cell(spec, seed=spec.seed)
    fitted = fit_room_around_cell(spec, core, room_seed=room_seed)
    plan = plan_from_fitted(spec, core, fitted, room_seed=room_seed)
    zones = _read_json(delivery / 'delivery_manifest.json').get('zones', {})
    return plan, zones


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--delivery', type=Path, required=True,
                        help='the delivery whose spec, seed and zones define the room')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--abort', dest='abort', action='store_true', default=None,
                        help='pass abort_unsatisfied_*=True (an unsatisfied solve fails the build)')
    parser.add_argument('--no-abort', dest='abort', action='store_false',
                        help='leave the switches at upstream defaults (the build continues)')
    parser.add_argument('--band-weight', type=float, default=60.0)
    parser.add_argument('--distance-weight', type=float, default=25.0)
    parser.add_argument('--distance-min-m', type=float, default=0.15)
    parser.add_argument('--hard-penetration-m', type=float, default=0.02)
    parser.add_argument('--reservations', choices=('real', 'degenerate'), default='real',
                        help='degenerate = the same objects and terms over 5 cm squares; TASK 3\'s '
                             'control for "what would generation have placed without the constraint"')
    parser.add_argument('--degenerate-half-m', type=float, default=0.05)
    parser.add_argument('--note', default=UNCALIBRATED_POLICY_NOTE,
                        help='the sentence that stops these weights being read as measurements')
    parser.add_argument('--gaps', default='',
                        help='comma-separated categories this room allows generation to fill '
                             '(SPEC-011 requirement 4, ADR-009 decision 4). Required for a build '
                             'that realises background content: without it the rejection pool would '
                             'refuse every object the build makes, and the provider refuses the '
                             'request by name rather than spending three minutes on it.')
    args = parser.parse_args()
    load_local_env()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('use a fresh output directory: a build is evidence and a re-run writes a new one')
    args.output_dir.mkdir(parents=True)

    plan, zones = replay(args.delivery)
    reservations = reservations_from_plan(
        plan,
        circulation=[Polygon(ring) for ring in zones.get('circulation', ())],
        door_sweep=[Polygon(ring) for ring in zones.get('door_sweep', ())])
    if args.reservations == 'degenerate':
        reservations = ablate(reservations, args.degenerate_half_m)
    # An unset --abort/--no-abort means "upstream's default", which is the honest reading of a run
    # that says nothing about the switches: record it as not applied rather than as a choice.
    abort = bool(args.abort) if args.abort is not None else False
    policy = ReservationPolicy(
        abort_unsatisfied=abort, band_penetration_weight=args.band_weight,
        distance_weight=args.distance_weight, distance_min_m=args.distance_min_m,
        hard_penetration_m=args.hard_penetration_m,
        note=args.note)
    gaps = tuple(item.strip() for item in args.gaps.split(',') if item.strip())

    started = time.monotonic()
    result: dict = {'delivery': str(args.delivery), 'plan_id': plan.plan_id,
                    'core_hash': plan.core.core_hash, 'policy': policy.to_dict(),
                    'reservation_mode': args.reservations,
                    'gaps': list(gaps),
                    'reservations': [item.to_dict() for item in reservations],
                    'output_dir': str(args.output_dir)}
    try:
        # Inside the try: a request the provider refuses (no gaps declared, a covered category, an
        # unknown one) is a *recorded* failure with a name, not a traceback -- the same rule this
        # tool applies to a build that fails.
        constraints = BackgroundConstraints(reservations=reservations, policy=policy, gaps=gaps)
        provider = InfinigenShellProvider(background_constraints=constraints)
        build = provider.build(plan, output_dir=args.output_dir)
        result['ok'] = True
        result['shell'] = {'provider': build.provider,
                           'measurements': {key: value for key, value
                                            in build.measurements.items()
                                            if key.startswith('constraint.')}}
    except Exception as error:  # a failed build is the measurement, not a crash to hide
        result['ok'] = False
        result['error'] = f'{type(error).__name__}: {error}'
    result['elapsed_s'] = round(time.monotonic() - started, 3)
    record_path = args.output_dir / 'infinigen_build' / 'constraint_injection.json'
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        result['injection'] = {key: record.get(key) for key in
                              ('injection', 'injection_reason', 'error', 'patch_sha256',
                               'entry_script_sha256', 'upstream_pin', 'abort_unsatisfied',
                               'terms', 'objects_removed_before_export')}
        pool = record.get('rejection_pool') or {}
        # The pool's own verdict, summarised: who was kept, who was refused, by which rule. The full
        # ledger stays in the record (one copy, and it is the copy the audit reads).
        result['rejection_pool'] = {
            'ran': pool.get('ran'), 'reason': pool.get('reason'), 'kept': pool.get('kept'),
            'rejected': pool.get('rejected'),
            'rejected_categories': pool.get('rejected_categories'),
            'bounds': pool.get('bounds'),
            'refusals': [{'name': item['name'], 'category': item['category'],
                          'reasons': item['reasons']}
                         for item in pool.get('candidates', ()) if item['outcome'] == 'rejected']}
    (args.output_dir / 'probe.json').write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != 'reservations'}, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
