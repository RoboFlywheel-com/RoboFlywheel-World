#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit whether a generated periphery was kept out of the task's reservations, and whether in
generation it only produced the categories retrieval cannot cover.

SPEC-011 TASK 1 built this tool and its negatives before anything changed behaviour; TASK 2 gave it
a build to read (entry script and patch, credentials in the record), TASK 3 calibrated the weights
against it, and TASK 4 added the three rules the earlier tasks could only state in prose: the
rejection pool must have run (`BACKGROUND_POOL_UNRECORDED`), its ledger and the ruler's measurement
must agree about a named object (`BACKGROUND_POOL_LEDGER_INCOMPLETE`), and an unsatisfied constraint
carrying our prefix is our term failing (`BACKGROUND_RESERVATION_UNSATISFIED`). Everything it judges
is read off evidence that already exists:

* **The contract** is `contracts/background_constraints.py`, which holds the region semantics, the
  factory -> category table, the accounting rules and the named failures. This tool supplies the
  measurements and the vocabulary it can import (`solvers/furnishing_plan.CATEGORY_PROFILES`) and
  nothing else, so the rules and the negative fixtures read one list.
* **The rooms** come from the matrix fixture (`tests/fixtures/background_constraint_matrix.json`):
  SPEC-009's probe as the control -- the same floor plan and seed as
  `outputs/bedroom_pick_alarm_clock/s005_bedroom_full_20260922_02`, with only
  `compose_indoors.populate_assets_enabled` flipped to True -- plus one case per other room type.
* **The geometry** comes from `python -m rfw_w.scenes.workers.blender.measure_generated_furniture`, which re-derives the
  reservations inside Blender and measures each generated object's rotated footprint against them,
  per region. This tool does not re-measure and does not touch the mesh.
* **The ownership/stage/plate half** is SPEC-009's own tool, imported and reused: requirement 8 says
  that audit stays the judge for the frozen core, and re-implementing its stage parsing here would
  be a second reader of one evidence format.

**Why the control case is still red, and why that is the point.** The probe build's own numbers
stand: the shelf `CellShelfFactory(699309).spawn_asset(3222513)` is 0.1265 m2 inside the protected
region and 0.0315 m2 inside the corridor, the build recorded nothing about its patch or pin, and
upstream's `abort_unsatisfied_*` default is `False` -- the probe's log shows unsatisfied groups in
`solve_medium` that the build carried on from. The matrix fixture declares which failures it expects
to see (`expect_failures`), so "red" is a checkable statement rather than an impression. The two
pooled cases are the same rooms with the pool on, and they are red on requirement 5 alone -- those
groups are upstream's own (`lighting`, `desk`, `office_desk_chairs`), which TASK 4 made checkable
rather than a reading.

The negatives are run against a **repaired base**: the same room's manifest with every requirement
satisfied by construction (credentials recorded, regions built and invisible, switches on, gaps
filled, a pool that kept the room's objects, allowances declared), after which one fact is broken
per negative. Mutating the red control manifest instead would make every negative fire for the same
reasons and prove nothing about the rule it names. The base is a fixture and says so:
`negatives_base.synthetic` is true, and its two digests are placeholders.

`--matrix` is the SPEC's acceptance entry point. Nothing here is reported as a pass that was not
measured, and a case with no evidence is reported `not_run` with its reason rather than skipped.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path.cwd()

from rfw_w.scenes.contracts.background import (
    EVIDENCE_OVERRIDE,
    STAGES,
    OwnedEntity,
    StageRecord,
)
from rfw_w.scenes.contracts.background_constraints import (
    ATTEMPTS_PER_CANDIDATE_MAX,
    GAP_UNFILLED_KINDS,
    GENERATED_OBJECTS_MAX_PER_STAGE,
    INJECTION_FILENAME,
    CIRCULATION_BAND_TOP_M,
    CONSTRAINTS_VERSION,
    DOOR_SWEEP_BAND_TOP_M,
    INJECTION_APPLIED,
    INJECTION_NOT_APPLIED,
    POOL_VERSION,
    NOT_A_GAP_NOT_DECLARED,
    REGION_CIRCULATION,
    REGION_DOOR_SWEEP,
    REGION_PROTECTED,
    RESERVATIONS,
    SOLVE_STAGES,
    BuildProvenance,
    CandidateAttempt,
    CategoryGap,
    ConstraintManifest,
    ConstraintStage,
    FACTORY_CATEGORIES,
    GeneratedObject,
    PoolCandidate,
    RejectionPool,
    Reservation,
    ReservationObject,
    abort_param,
    category_of,
)
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_spec import SceneScope
from rfw_w.scenes.solvers.furnishing_plan import CATEGORY_PROFILES

SCHEMA_VERSION = "pvscene.background-constraint-matrix/v1"

#: The upstream stages that *realise* assets. `solve_*` runs either way (SPEC-009 TASK 1 measured
#: it at 97.8-98.7% of a build), so the mode a case is in is decided by these three.
GENERATING_STAGES = ("populate_intermediate_pholders", "populate_assets", "floating_objs")

#: The stage `spawn_asset` objects come from: `populate_assets` is the only stage that calls
#: `factory.spawn_asset` (`generate_indoors.py:307`), and the measurement tool identifies realised
#: assets by exactly that name.
REALISING_STAGE = "populate_assets"

#: `compose_indoors.<name>=<True|False>` -- every override, not just the `<stage>_enabled` ones,
#: because the abort switches are overrides of the same block.
_OVERRIDE = re.compile(r"compose_indoors\.(?P<key>\w+)=(?P<value>True|False)")

#: Upstream's own constraint-failure line, which is the only place a solve stage's outcome exists:
#: ``Solver has failed to satisfy constraints for stage ('on_wall_0', 'office_0/0').
#: violations={'lighting': 1}. Continuing anyway, override `solve_objects.abort_unsatisfied=True```.
_UNSATISFIED = re.compile(
    r"Solver has failed to satisfy constraints for stage \('(?P<stage>\w+?)_\d+'(?P<rest>[^)]*)\)"
    r".*?violations=(?P<violations>\{[^}]*\})")

#: Inner stage name -> the outer stage that runs it (`generate_indoors.py:206-303` at the pinned
#: commit). Needed because the failure line names the inner stage, and the SPEC's failure is per
#: outer stage.
SOLVE_STAGE_OF = {
    "on_floor_and_wall": "solve_large",
    "on_floor_freestanding": "solve_large",
    "on_wall": "solve_medium",
    "on_ceiling": "solve_medium",
    "side_obj": "solve_medium",
    "obj_ontop_obj": "solve_small",
    "obj_on_support": "solve_small",
}

RESERVATION_SOURCE = ("python -m rfw_w.scenes.workers.blender.measure_generated_furniture:reservations "
                      "(replay of solve_cell -> fit_room_around_cell -> plan_from_fitted)")

#: What the negatives' repaired base declares about the patch. The paths are the real ones
#: (`solvers/shell_providers.py:BACKGROUND_ENTRY_SCRIPT` / `BACKGROUND_PATCH_MODULE`); the digests
#: are placeholders because the base is a fixture, and `negatives_base.synthetic` says so.
PLANNED_ENTRY_SCRIPT = "python -m rfw_w.scenes.workers.blender.run_infinigen_build"
PLANNED_PATCH_MODULE = "python -m rfw_w.scenes.workers.blender.infinigen_constraint_patch"
PLACEHOLDER_DIGEST = "0" * 64
UPSTREAM_PIN = "892947be71c4ff102c6a558d088ffe149561c11f"


class ConstraintInputError(RuntimeError):
    """The audit cannot be run on the input it was given, named rather than reported as a pass."""




from rfw_w.scenes.validation import check_generated_background as BACKGROUND_AUDIT


def _read_json(path: Path, what: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConstraintInputError(f"BACKGROUND_CONSTRAINT_INPUT_MISSING({what}): {path}")
    return json.loads(path.read_text())


def _relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if ROOT in resolved.parents else str(resolved)


def _rings(polygons: Sequence[Sequence[Sequence[float]]], *, places: int = 6):
    """Polygons rounded for comparison, so two spellings of one region are not two regions."""
    return sorted(tuple(sorted((round(float(x), places), round(float(y), places)) for x, y in ring))
                  for ring in polygons)


def _compose_overrides(command: str) -> dict[str, bool]:
    return {match.group('key'): match.group('value') == 'True'
            for match in _OVERRIDE.finditer(command)}


def _run_with_build(run, build_dir: Path, log_name: str = 'stderr.txt'):
    """Point an already-loaded delivery at the build directory a probe rebuilt the room in.

    `log_name` is the case's own evidence file rather than always `stderr.txt`, because a probe
    that failed once and then succeeded keeps both logs side by side (`attempt1_stderr.txt` is the
    interpreter that could not import; `attempt2_stderr.txt` is the build). Pointing at the wrong
    one silently reports every stage as unreported, which is how this parameter came to exist.

    The four lines that read the placeholder census are `check_generated_background.load_run`'s
    own; they are repeated rather than factored out because that tool reads the build directory a
    *delivery* recorded, and a probe build lives somewhere else on purpose.
    """
    run.build_dir = build_dir
    run.stage_log = ((build_dir / log_name).read_text(errors='replace')
                     if (build_dir / log_name).is_file() else '')
    run.command = ((build_dir / 'command.txt').read_text(errors='replace')
                   if (build_dir / 'command.txt').is_file() else '')
    state = _read_json(build_dir / 'scene' / 'solve_state.json', 'solve_state.json')
    objs = list(state['objs'].values())
    generated = [obj for obj in objs if obj.get('generator')]
    run.placeholders = len(generated)
    run.real_placeholders = sum(1 for obj in objs
                                if 'Semantics(real-placeholder)' in obj.get('tags', ()))
    run.placeholder_generators = tuple(sorted({
        str(obj['generator']).split('(')[0] for obj in generated}))
    return run


def _stage_finished(stage_log: str, stage: str) -> bool:
    """Whether the build's own log says this stage ran, so a zero is a measurement and not silence."""
    return re.search(rf"\[{re.escape(stage)}\] finished in", stage_log) is not None


def _solve_outcomes(stage_log: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Unsatisfied constraint groups per solve stage, and the violations they named.

    Measured from the build's own log, which is the only place this outcome exists -- and
    `{"lighting": 1}` is worth keeping: it says the unsatisfied groups in the probe were upstream's
    own (lighting, desk, office_desk_chairs), not ours, which is a different repair.
    """
    counts = {stage: 0 for stage in SOLVE_STAGES}
    violations: dict[str, list[str]] = {stage: [] for stage in SOLVE_STAGES}
    for line in stage_log.splitlines():
        match = _UNSATISFIED.search(line)
        if not match:
            continue
        outer = SOLVE_STAGE_OF.get(match.group('stage'))
        if outer is None:
            continue
        counts[outer] += 1
        violations[outer].append(match.group('violations'))
    return counts, violations


def _violation_names(violations: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """The constraint names upstream blamed, out of the `{name: count}` reprs it logged.

    The keys matter and the counts do not: upstream's report is keyed by the *constraint's own name*
    (`evaluator/evaluate.py:262-275`), so `lighting` is upstream's constraint and
    `pvscene_reservation_protected_penetration` would be ours. Parsing the repr with
    `ast.literal_eval` rather than regexing for quotes is deliberate -- the string is a Python
    literal produced by `f"{violations=}"`, so it is parsed, not guessed at.
    """
    names: set[str] = set()
    for value in violations.values():
        for item in value:
            try:
                parsed = ast.literal_eval(item)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, Mapping):
                names.update(str(name) for name in parsed)
    return tuple(sorted(names))


def _reservations(where: Mapping[str, Any], case_id: str) -> tuple[Reservation, ...]:
    for key, expected in (('circulation_band_top_m', CIRCULATION_BAND_TOP_M),
                          ('door_sweep_band_top_m', DOOR_SWEEP_BAND_TOP_M)):
        if round(float(where[key]), 6) != round(float(expected), 6):
            # The measurement and the contract must not disagree about how high a reservation
            # reaches: the number decides which objects are measured at all.
            raise ConstraintInputError(
                f"BACKGROUND_CONSTRAINT_BAND_MISMATCH({case_id}:{key}={where[key]}!={expected})")
    records = [Reservation(region=REGION_PROTECTED,
                           polygon_xy=tuple((float(x), float(y))
                                            for x, y in where['protected_polygon_xy']),
                           band_top_m=float(where['protection_band_top_m']),
                           source=RESERVATION_SOURCE)]
    for region, key, band in ((REGION_CIRCULATION, 'circulation_polygons_xy',
                               CIRCULATION_BAND_TOP_M),
                              (REGION_DOOR_SWEEP, 'door_sweep_polygons_xy',
                               DOOR_SWEEP_BAND_TOP_M)):
        records.extend(Reservation(region=region,
                                   polygon_xy=tuple((float(x), float(y)) for x, y in ring),
                                   band_top_m=band, source=RESERVATION_SOURCE)
                       for ring in where[key])
    return tuple(records)


def _unfilled_reason(category: str, filled: Mapping[str, Sequence[str]],
                     pool: RejectionPool | None) -> str:
    """Why a declared gap is empty, in one of the kinds the record allows.

    Requirement 4 asks every gap to be accounted for, and the accounting has to be a *renamed* fact
    rather than a sentence someone wrote: either a candidate of this category was proposed and the
    pool refused it (the refusal is in the ledger, by name), or no candidate of this category was
    proposed at all (the ledger is complete, so its absence is evidence), or -- when the build
    recorded no pool -- only "the room does not contain one" could be measured, and saying more than
    that would be inventing the evidence. The three kinds are told apart on purpose: the first is a
    rule working, the second is the sampler's limit (ADR-009's Consequences: upstream's category
    choice is a hardcoded `uniform` and we cannot ask it for a category), and the third is an
    un-audited build.
    """
    if category in filled:
        return ""
    if pool is not None and pool.ran:
        refused = pool.rejected_by_category().get(category, ())
        if refused:
            reasons = sorted({reason for item in refused for reason in item.reasons})
            return f"{GAP_UNFILLED_KINDS[0]}({','.join(reasons)})"
        return (f"{GAP_UNFILLED_KINDS[1]}(the build's pool judged every realised object and none "
                f"was a {category})")
    return (f"{GAP_UNFILLED_KINDS[2]}(no generated object of this category is in the room; the "
            "build recorded no rejection pool, so nothing is known about what it proposed)")


def constraint_manifest(case: Mapping[str, Any], *, started: float) -> ConstraintManifest:
    """One case as a `ConstraintManifest`, with every cross-check that makes its inputs evidence."""
    case_id = str(case['id'])
    delivery = ROOT / case['delivery']
    measurement_path = ROOT / case['measurement']
    build_dir = ROOT / case['build'] if case.get('build') else None
    measurement = _read_json(measurement_path, f'{case_id}:measurement')
    if Path(str(measurement.get('delivery', ''))).resolve() != delivery.resolve():
        raise ConstraintInputError(
            f"BACKGROUND_CONSTRAINT_INPUT_MISMATCH({case_id}): the measurement is of "
            f"{measurement.get('delivery')}, not of {case['delivery']}")
    if build_dir is not None and Path(str(measurement.get('build', ''))).resolve() != build_dir.resolve():
        raise ConstraintInputError(
            f"BACKGROUND_CONSTRAINT_INPUT_MISMATCH({case_id}): the measurement was taken in "
            f"{measurement.get('build')}, not in {case['build']}")
    where = measurement['reservations']
    zones = _read_json(delivery / 'delivery_manifest.json', 'delivery_manifest').get('zones', {})
    keepouts = list(zones.get('circulation', ())) + list(zones.get('door_sweep', ()))
    if _rings(where['keepout_polygons_xy']) != _rings(keepouts):
        raise ConstraintInputError(
            f"BACKGROUND_CONSTRAINT_INPUT_MISMATCH({case_id}:keepout): the measurement's keepout "
            f"polygons are not the delivered ones")
    if 'room_polygon_xy' in zones and (_rings([where['room_polygon_xy']])
                                      != _rings([zones['room_polygon_xy']])):
        raise ConstraintInputError(
            f"BACKGROUND_ROOM_REPLAY_MISMATCH({case_id}): the measurement's room is not the "
            f"delivered room, so its reservations are not evidence either")

    run = BACKGROUND_AUDIT.load_run(delivery)
    if build_dir is not None:
        run = _run_with_build(run, build_dir, str(case.get('log', 'stderr.txt')))
    overrides = _compose_overrides(run.command)
    mode = ('generated' if any(overrides.get(f'{stage}_enabled') for stage in GENERATING_STAGES)
            else 'retrieved')
    counts, violations = _solve_outcomes(run.stage_log)
    violation_names = _violation_names(violations)

    # The build's own record, when there is one: SPEC-011 TASK 2's builds write
    # `constraint_injection.json` beside `command.txt`, and a case whose build has one is audited
    # against the credentials the build actually wrote rather than against the fixture's claim
    # about them. A build without the file is the default path, and says so.
    record_path = None if build_dir is None else build_dir / INJECTION_FILENAME
    record = json.loads(record_path.read_text()) if record_path is not None \
        and record_path.is_file() else None
    if record is not None and record.get('reservations'):
        # The pool judged the objects against the polygons the *request* carried, and the ruler
        # measures them against the polygons it re-derives from the delivery. If those are not the
        # same polygons, the ledger and the measurement describe two different rooms -- and the
        # pool's claim would be about geometry the room never had.
        delivered = {REGION_PROTECTED: ([where['protected_polygon_xy']]
                                       if where['protected_polygon_xy'] else []),
                     REGION_CIRCULATION: where['circulation_polygons_xy'],
                     REGION_DOOR_SWEEP: where['door_sweep_polygons_xy']}
        recorded: dict[str, list[Any]] = {}
        for item in record['reservations']:
            recorded.setdefault(str(item['region']), []).append(item['polygon_xy'])
        for region, rings in delivered.items():
            if _rings(recorded.get(region, ())) != _rings(rings):
                raise ConstraintInputError(
                    f"BACKGROUND_CONSTRAINT_INPUT_MISMATCH({case_id}:{region}): the build was asked "
                    f"to keep clear the polygons in {case['build']}/reservations.json, and the "
                    f"measurement re-derived different ones from {case['delivery']}")
    generated = tuple(
        GeneratedObject(name=str(entry['name']), factory=str(entry['factory']),
                        stage=REALISING_STAGE, category=category_of(str(entry['factory'])),
                        base_z_m=None if entry.get('base_z_m') is None else float(entry['base_z_m']))
        for entry in measurement['objects'] if entry['generated'])
    overlap_by_entity = {
        str(entry['name']): {REGION_PROTECTED: float(entry['protected_overlap_m2']),
                             REGION_CIRCULATION: float(entry['circulation_overlap_m2']),
                             REGION_DOOR_SWEEP: float(entry['door_sweep_overlap_m2'])}
        for entry in measurement['objects'] if entry['generated']}
    overlap_by_entity = {name: measured for name, measured in overlap_by_entity.items()
                         if any(value > 0.0 for value in measured.values())}

    filled: dict[str, list[str]] = {}
    for item in generated:
        filled.setdefault(item.category, []).append(item.name)
    # The build's own pool, and the two `gaps` declarations compared: the fixture's (which is what
    # the matrix judges) and the request's (which is what the build enforced). A case whose two
    # declarations disagree is judged against a rule the build was never given -- the same class of
    # input mismatch this tool already refuses for polygons and rooms.
    pool = None if record is None or not record.get('rejection_pool') else \
        RejectionPool.from_dict(record['rejection_pool'])
    if record is not None and case.get('gap_categories') is not None:
        requested = tuple(sorted(str(item) for item in record.get('gaps', ())))
        if requested and requested != tuple(sorted(str(item) for item in case['gap_categories'])):
            raise ConstraintInputError(
                f"BACKGROUND_CONSTRAINT_INPUT_MISMATCH({case_id}:gaps): the build was asked to fill "
                f"{list(requested)} and the matrix declares {sorted(case['gap_categories'])}")
    gaps = tuple(CategoryGap(category=str(category),
                             why='declared by the matrix fixture as a category the retrieval '
                                 'vocabulary does not cover',
                             filled_by=tuple(sorted(filled.get(str(category), ()))),
                             unfilled_reason=_unfilled_reason(str(category), filled, pool))
                 for category in case['gap_categories'])

    # The switch's authority is the command the build ran (`command.txt`), not the record's copy of
    # it: the record is written by the patch, and a record read as evidence for the run that produced
    # it is a claim checking itself. Measured the other way round, too -- the patch's first attempt
    # at this evidence used gin and reported "unbound" for a switch that was demonstrably in force.
    constraint_stages = tuple(
        ConstraintStage(
            stage=stage,
            abort_unsatisfied=bool(overrides.get(abort_param(stage)
                                                 .removeprefix('compose_indoors.'))),
            # An unmeasured outcome is `None`, never 0: a stage the log does not show running has
            # not been observed to satisfy anything, and reporting "0 unsatisfied" for it would be
            # this tool's own version of the failure it exists to catch.
            unsatisfied=(counts[stage] if _stage_finished(run.stage_log, stage) else None),
            reason=('' if _stage_finished(run.stage_log, stage) else
                    'the build log has no `finished in` line for this stage, so its constraint '
                    'outcome is not measured'))
        for stage in SOLVE_STAGES)

    background = BACKGROUND_AUDIT.manifest_for(run, run_started=started)
    generated_entities = tuple(
        OwnedEntity(entity_id=item.name, owner='generated', scope=SceneScope.PERIPHERY.value,
                    stage=item.stage, provider='infinigen', asset_ref=item.name)
        for item in generated)
    background = background.with_changes(
        mode=mode,
        # The allowance is the calibrated constant unless the case declares its own: ADR-009
        # decision 5 freezes N in code, and a case that narrowed it would be measuring a different
        # allowance than the builds were held to (the negatives narrow it on purpose, and say so).
        budget={**dict(GENERATED_OBJECTS_MAX_PER_STAGE), **dict(case.get('budget', {}))},
        entities=background.entities + generated_entities)

    if record is not None:
        build_provenance = BuildProvenance(
            upstream_pin=str(record.get('upstream_pin', '')),
            entry_script=str(record.get('entry_script', '')),
            entry_script_sha256=str(record.get('entry_script_sha256', '')),
            patch_module=str(record.get('patch_module', '')),
            patch_sha256=str(record.get('patch_sha256', '')),
            injection=str(record.get('injection', 'unknown')),
            injection_reason=str(record.get('injection_reason', '')))
        reservation_objects = tuple(
            ReservationObject(name=str(item['object']), region=str(item['region']),
                              visible=bool(item.get('visible', False)),
                              exported=bool(item.get('exported', False)),
                              listed_as_instance=bool(item.get('listed_as_instance', False)))
            for item in record.get('reservation_objects', ()))
    else:
        build_provenance = BuildProvenance(
            upstream_pin='', injection=INJECTION_NOT_APPLIED,
            injection_reason='this build left no constraint record: it ran without the patch '
                             '(the default path, or a build from before SPEC-011 TASK 2)')
        reservation_objects = ()

    manifest = ConstraintManifest(
        scene_id=str(case['id']), seed=int(case['seed']), mode=mode, background=background,
        reservations=_reservations(where, case_id),
        reservation_objects=reservation_objects,
        build=build_provenance,
        constraint_stages=constraint_stages, gaps=gaps, generated=generated,
        # The attempts and the bounds are the *pool's* own record, converted rather than re-derived:
        # `RejectionPool.attempts()` is the one conversion (a candidate tried once, kept or refused),
        # and a second counting here would let the ledger and the record disagree about how many
        # times a candidate was tried. A build with no pool keeps the calibrated bounds, because the
        # allowance is a property of the run, not of the record's completeness.
        attempts=() if pool is None else pool.attempts(),
        rejection_bounds=({} if pool is None else dict(pool.bounds))
        or dict(GENERATED_OBJECTS_MAX_PER_STAGE),
        pool=pool,
        retrieval_categories=tuple(sorted(CATEGORY_PROFILES)),
        overlap_by_entity=overlap_by_entity,
        measurements={
            'generated_objects': len(generated),
            'realised_stage': REALISING_STAGE,
            'spawned_and_left_as_placeholder': measurement.get('counts', {}),
            'unsatisfied_groups': counts,
            'unsatisfied_violations': violations,
            'unsatisfied_group_names': list(violation_names),
            'pool': None if pool is None else pool.to_dict(),
            'pool_note': ('the rejection pool is the build\'s own record of what it refused; a '
                          'build that ran generation without it is BACKGROUND_POOL_UNRECORDED, and '
                          'a pool that ran is checked against this run\'s own measurement '
                          '(BACKGROUND_POOL_LEDGER_INCOMPLETE) rather than believed'),
            'unsatisfied_note': 'the solver logs its failure unconditionally when a stage has '
                                'violations (solve.py:206-217) and abort_unsatisfied only decides '
                                'whether it raises or warns, so zero is an absence of that line '
                                'for a stage whose own `finished in` line proves it ran',
            'abort_overrides': {abort_param(stage): overrides.get(
                abort_param(stage).removeprefix('compose_indoors.')) for stage in SOLVE_STAGES},
            'abort_default_note': 'upstream reads an absent abort_unsatisfied_{group} as False '
                                  '(generate_indoors.py:202), so an override that is not there is '
                                  'a measured unguarded build, not an unknown one',
            'min_generated_categories': int(case.get('min_generated_categories', 0)),
            'generated_categories': list(sorted({item.category for item in generated
                                                 if item.category})),
            'gap_eligible_categories': list(sorted(set(category_of(f) for f in
                                                       measurement.get('background_factories', ()))
                                                   - set(CATEGORY_PROFILES))),
        },
        provenance={'audited_by': 'python -m rfw_w.scenes.validation.check_background_constraints',
                    'delivery': _relative(delivery),
                    'measurement': _relative(measurement_path),
                    'measurement_tool': measurement.get('tool'),
                    'measurement_tool_version': measurement.get('tool_version'),
                    'build': None if build_dir is None else _relative(build_dir),
                    'injection_record': None if record is None else _relative(record_path),
                    'reuses': 'python -m rfw_w.scenes.validation.check_generated_background for ownership/stages',
                    'elapsed_s': round(time.monotonic() - started, 3)})
    return manifest


# ---------------------------------------------------------------------------------------------
# The negatives: one repaired base, one broken fact per negative
# ---------------------------------------------------------------------------------------------

def with_pool(manifest: ConstraintManifest, pool: RejectionPool | None) -> ConstraintManifest:
    """A manifest whose attempts and bounds are the pool's -- one conversion, so the two agree.

    The manifest reads attempts for its boundedness and its rejection counts and the pool for its
    ledger; a fixture that set one and forgot the other would be testing a room no build can
    produce. Every place that changes a pool goes through here.
    """
    return manifest.with_changes(
        pool=pool, attempts=() if pool is None else pool.attempts(),
        rejection_bounds=(dict(pool.bounds) if pool is not None
                          else dict(GENERATED_OBJECTS_MAX_PER_STAGE)))


def _factory_for(category: str) -> str:
    """One upstream factory that maps to a category, for the fixtures' stub objects."""
    for factory, name in FACTORY_CATEGORIES.items():
        if name == category:
            return factory
    return 'UnknownFactory'


def repaired_base(manifest: ConstraintManifest, case: Mapping[str, Any]) -> ConstraintManifest:
    """The same room with every SPEC-011 requirement satisfied, as a fixture.

    Each of the six negatives has to be caught *by the rule it names*, and the control manifest is
    red for a dozen reasons at once, so mutating it would produce a dozen identical rows. This base
    keeps the room's real geometry, the real generated objects of the declared gap categories and
    the real solve outcomes' source, and repairs everything else: credentials recorded, one
    invisible reservation object per region, the three switches on, a rejection pool that kept every
    object in the room and refused one candidate per unfilled gap, allowances declared, no
    intrusion. `negatives_base.synthetic` marks it, and the run reports whether it is clean -- a
    base that is not clean would make every negative meaningless.
    """
    gaps = set(manifest.declared_gap_categories())
    generated = tuple(item for item in manifest.generated if item.category in gaps)
    filled: dict[str, list[str]] = {}
    for item in generated:
        filled.setdefault(item.category, []).append(item.name)
    # Every declared gap is accounted for in the base, and the two ways are the two the tool
    # derives: a kept object filled it, or the pool refused a candidate of that category. A gap with
    # neither would red the base on requirement 4 before any negative ran.
    refused = tuple(
        PoolCandidate(name=f'{_factory_for(category)}(0).spawn_asset(0)',
                      factory=_factory_for(category), category=category, outcome='rejected',
                      reasons=(f'not-a-gap({category}:{NOT_A_GAP_NOT_DECLARED})',))
        for category in sorted(gaps) if not filled.get(category))
    pool = RejectionPool(
        stage=REALISING_STAGE, ran=True,
        rule='the repaired base: a pool that judged every realised object and kept the ones the '
             'room has',
        bounds={stage: 32 for stage in STAGES},
        candidates=tuple(PoolCandidate(name=item.name, factory=item.factory, category=item.category,
                                       outcome='kept') for item in generated) + refused)
    background = manifest.background
    assert background is not None  # a case with evidence always has the ownership half
    stages = tuple(record for record in background.stages if record.stage != 'floating_objs') + (
        StageRecord(stage='floating_objs', state='not_run', evidence=EVIDENCE_OVERRIDE,
                    reason='the repaired base declares the explicit override SPEC-009 TASK 2 adds'),
    )
    # Every generated entity is dropped and the surviving ones re-added, rather than filtered:
    # a filter that kept the overlap between the old and the new set would count each object twice
    # and report an exceeded budget in the base that no negative asked for.
    entities = tuple(entity for entity in background.entities if entity.owner != 'generated')
    return with_pool(manifest.with_changes(
        mode='generated',
        background=background.with_changes(
            mode='generated', stages=stages, budget={stage: 32 for stage in STAGES},
            entities=entities + tuple(
                OwnedEntity(entity_id=item.name, owner='generated',
                            scope=SceneScope.PERIPHERY.value, stage=item.stage,
                            provider='infinigen', asset_ref=item.name)
                for item in generated),
            overlap_by_entity={}),
        reservation_objects=tuple(ReservationObject(name=f'pvscene_reservation_{region}',
                                                    region=region) for region in RESERVATIONS),
        build=BuildProvenance(upstream_pin=UPSTREAM_PIN, entry_script=PLANNED_ENTRY_SCRIPT,
                              entry_script_sha256=PLACEHOLDER_DIGEST,
                              patch_module=PLANNED_PATCH_MODULE,
                              patch_sha256=PLACEHOLDER_DIGEST, injection=INJECTION_APPLIED),
        constraint_stages=tuple(ConstraintStage(stage=stage, abort_unsatisfied=True, unsatisfied=0)
                                for stage in SOLVE_STAGES),
        gaps=tuple(CategoryGap(category=gap.category, why=gap.why,
                               filled_by=tuple(sorted(filled.get(gap.category, ()))))
                   for gap in manifest.gaps),
        generated=generated,
        overlap_by_entity={}), pool)


def negatives(base: ConstraintManifest, *, real_overlap: Mapping[str, Mapping[str, float]],
              unsatisfied_medium: int) -> tuple[list[dict[str, Any]], str, tuple[str, ...]]:
    """One broken fact per negative, each announced by name.

    `real_overlap` and `unsatisfied_medium` are the control case's own measurements, so the two
    negatives whose whole point is "the failure comes back" carry the probe's numbers rather than
    invented ones.
    """
    assert base.build is not None and base.background is not None
    declared = base.declared_gap_categories()
    fillable = next((gap for gap in base.gaps if gap.filled_by), base.gaps[0])
    intruder = sorted(real_overlap)[0] if real_overlap else ''

    def stages_medium(abort: bool, unsatisfied: int) -> tuple[ConstraintStage, ...]:
        return tuple(replace(item, abort_unsatisfied=abort, unsatisfied=unsatisfied)
                     if item.stage == 'solve_medium' else item for item in base.constraint_stages)

    return [
        {'name': 'no_separation_intrusion_returns',
         'expect': ['BACKGROUND_PROTECTED_OVERLAP(', 'BACKGROUND_KEEPOUT_OVERLAP(',
                    'BACKGROUND_KEEPOUT_OBJECT_MISSING('],
         'manifest': base.with_changes(
             reservations=base.reservations, reservation_objects=(),
             build=replace(base.build, injection=INJECTION_NOT_APPLIED,
                           injection_reason='negative 1: the injection was removed'),
             overlap_by_entity={name: dict(measured) for name, measured in real_overlap.items()})},
        {'name': 'abort_switch_off',
         'expect': ['BACKGROUND_SOLVER_UNGUARDED(solve_medium)'],
         'manifest': base.with_changes(constraint_stages=stages_medium(False, 0))},
        {'name': 'unsatisfied_solve_continued',
         'expect': ['BACKGROUND_SOLVER_ABORTED(solve_medium)',
                    'BACKGROUND_SOLVER_UNGUARDED(solve_medium)'],
         'manifest': base.with_changes(
             constraint_stages=stages_medium(False, unsatisfied_medium))},
        {'name': 'covered_category_generated',
         'expect': ['BACKGROUND_NOT_A_GAP(desk,covered-by-retrieval'],
         'manifest': base.with_changes(generated=base.generated + (
             GeneratedObject(name='SimpleDeskFactory(1).spawn_asset(2)',
                             factory='SimpleDeskFactory', stage=REALISING_STAGE,
                             category=category_of('SimpleDeskFactory')),))},
        {'name': 'gap_declared_but_covered',
         'expect': ['BACKGROUND_NOT_A_GAP(plant,covered-by-retrieval'],
         'manifest': base.with_changes(gaps=base.gaps + (
             CategoryGap(category='plant', why='negative: declared a gap by mistake',
                         unfilled_reason='no plant was generated'),))},
        {'name': 'budget_exceeded',
         'expect': ['BACKGROUND_BUDGET_EXCEEDED(populate_assets)'],
         # One less than what the base generated, rather than a fixed 1: the count is what the
         # rule is about, and a constant would pass or fail on how many objects the case happens
         # to have rather than on the rule.
         'manifest': base.with_changes(
             background=base.background.with_changes(
                 budget={**dict(base.background.budget), REALISING_STAGE: max(0, base.background
                          .generated_by_stage().get(REALISING_STAGE, 0) - 1)}))},
        {'name': 'credentials_missing',
         'expect': ['BACKGROUND_PATCH_UNRECORDED(patch_sha256)'],
         'manifest': base.with_changes(build=replace(base.build, patch_sha256=''))},
        {'name': 'rejections_unbounded',
         'expect': ['BACKGROUND_REJECTIONS_UNBOUNDED('],
         # The same candidate tried three times, with the reason a real refusal would carry: what is
         # broken is the repetition, not the reason. The pool carries the tries (through `with_pool`,
         # so the ledger and the attempt list cannot disagree about how many there were).
         'manifest': with_pool(base, replace(
             base.pool, candidates=base.pool.candidates + tuple(
                 PoolCandidate(name='CellShelfFactory(699309)', factory='CellShelfFactory',
                               category='shelf', outcome='rejected', attempt=attempt,
                               reasons=('not-a-gap(shelf:not-declared-in-this-case)',))
                 for attempt in range(1, ATTEMPTS_PER_CANDIDATE_MAX + 2))))},
        {'name': 'reservation_object_leaked',
         'expect': ['BACKGROUND_RESERVATION_OBJECT_LEAKED('],
         'manifest': base.with_changes(reservation_objects=tuple(
             replace(item, exported=True) if item.region == REGION_CIRCULATION else item
             for item in base.reservation_objects))},
        {'name': 'factory_unmapped',
         'expect': ['BACKGROUND_FACTORY_UNMAPPED('],
         'manifest': base.with_changes(generated=base.generated + (
             GeneratedObject(name='WobbleFactory(3).spawn_asset(4)', factory='WobbleFactory',
                             stage=REALISING_STAGE, category=category_of('WobbleFactory')),))},
        {'name': 'gap_unfilled_unaccounted',
         'expect': ['BACKGROUND_GAP_UNFILLED('],
         'manifest': base.with_changes(gaps=tuple(
             replace(gap, filled_by=(), unfilled_reason='')
             if gap.category == fillable.category else gap for gap in base.gaps))},
        # -- TASK 4's own rules, which the six negatives above never reached ----------------------
        {'name': 'pool_not_recorded',
         'expect': ['BACKGROUND_POOL_UNRECORDED(populate_assets'],
         # Generation happened (the base's stage record says so) and the room has no pool: the
         # objects stood in the room unjudged, which is what the probe build did.
         'manifest': with_pool(base, None)},
        {'name': 'pool_ledger_incomplete',
         'expect': ['BACKGROUND_PROTECTED_OVERLAP(', 'BACKGROUND_KEEPOUT_OVERLAP(',
                    'BACKGROUND_POOL_LEDGER_INCOMPLETE('],
         # The realistic shape of this bug: the pool's ledger says it refused this piece and the
         # ruler measures it still standing in the region (a deletion that did not happen). The
         # intruder is in the ledger *on purpose* -- an intruder the ledger never mentions is the
         # overlap failure's own business, and this negative is about the two accounts disagreeing
         # about one named object.
         'manifest': with_pool(
             base.with_changes(
                 overlap_by_entity={intruder: {REGION_PROTECTED: 0.1265,
                                               REGION_CIRCULATION: 0.0315}}),
             replace(base.pool, candidates=base.pool.candidates + (
                 PoolCandidate(name=intruder, factory=intruder.split('(')[0],
                               category=category_of(intruder.split('(')[0]), outcome='rejected',
                               reasons=('protected-overlap(0.1265m2)',)),)))},
        {'name': 'reservation_term_unsatisfied',
         'expect': ['BACKGROUND_RESERVATION_UNSATISFIED('],
         # Upstream's report is keyed by the constraint's own name, so this is our term failing.
         # The name is the patch's own spelling (`infinigen_constraint_patch.py:269-276`).
         'manifest': base.with_changes(measurements={
             **dict(base.measurements),
             'unsatisfied_group_names': ['pvscene_reservation_protected_penetration']})},
    ], intruder, declared


def run_negatives(base: ConstraintManifest, *, real_overlap, unsatisfied_medium: int,
                  declared: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """Run each negative and require it to be caught by every name it announces."""
    items, intruder, declared_gaps = negatives(base, real_overlap=real_overlap,
                                               unsatisfied_medium=unsatisfied_medium)
    by_name = {str(item['name']): item for item in declared}
    rows: list[dict[str, Any]] = []
    for case in items:
        faults = list(case['manifest'].audit())
        expects = list(case['expect'])
        found = [expect for expect in expects if any(gap.startswith(expect) for gap in faults)]
        declared = by_name.get(case['name'], {})
        rows.append({'name': case['name'],
                     'spec_negative': declared.get('spec_negative'),
                     'expected': expects,
                     'declared_expected': list(declared.get('expect', ())),
                     'expectation_matches_declaration': list(declared.get('expect', ())) == expects,
                     'observed': faults,
                     'missed': [expect for expect in expects if expect not in found],
                     'extra': [gap for gap in faults
                               if not any(gap.startswith(expect) for expect in expects)],
                     'caught': len(found) == len(expects)})
    base_faults = list(base.audit())
    meta = {'synthetic': True,
            'what': 'the control case with every SPEC-011 requirement satisfied by construction, '
                    'so each negative breaks exactly one fact',
            'clean': not base_faults, 'faults': base_faults,
            'intruder': intruder, 'declared_gaps': list(declared_gaps),
            'declared': sorted(by_name), 'produced': sorted(case['name'] for case in items),
            'expectations_agree': all(row['expectation_matches_declaration'] for row in rows)}
    return rows, meta


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def audit_case(case: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {'id': str(case['id']), 'role': case.get('role'),
                           'room_type': case.get('room_type'), 'seed': case.get('seed')}
    if not case.get('measurement') or not case.get('delivery'):
        row['status'] = 'not_run'
        row['not_run_reason'] = case.get('not_run_reason', 'no reason recorded')
        row['passed'] = False
        row['elapsed_s'] = round(time.monotonic() - started, 3)
        return row
    try:
        manifest = constraint_manifest(case, started=started)
    except Exception as error:  # a case the tool cannot read is reported, never skipped
        row['status'] = 'blocked'
        row['gaps_own'] = [f'{type(error).__name__}({error})']
        row['gaps_background'] = []
        row['passed'] = False
        row['elapsed_s'] = round(time.monotonic() - started, 3)
        return row
    own = list(manifest.own_gaps())
    background = list(manifest.background.audit()) if manifest.background else []
    required = int(case.get('min_generated_categories', 0))
    observed = len(manifest.generated_categories())
    row.update({
        'status': 'ran',
        'mode': manifest.mode,
        'manifest': manifest.to_dict(),
        'gaps_own': own,
        'gaps_background': background,
        'expect_failures': list(case.get('expect_failures', [])),
        'expected_failures_present': [
            expect for expect in case.get('expect_failures', [])
            if any(gap.startswith(expect) for gap in own + background)],
        'min_generated_categories': {'required': required, 'observed': observed,
                                     'met': observed >= required},
        'passed': not own and not background and observed >= required,
        'elapsed_s': round(time.monotonic() - started, 3),
    })
    if not row['min_generated_categories']['met']:
        row['gaps_own'].append(
            f'BACKGROUND_MIN_GENERATED_CATEGORIES({observed}<{required})')
        row['passed'] = False
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--matrix', type=Path, required=True,
                        help='the SPEC-011 acceptance matrix fixture')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('use a fresh output directory: an audit never overwrites evidence')
    matrix = json.loads(args.matrix.read_text())
    if matrix.get('schema_version') != SCHEMA_VERSION:
        parser.error(f"BACKGROUND_MATRIX_SCHEMA_UNSUPPORTED({matrix.get('schema_version')}): "
                     f"this tool reads {SCHEMA_VERSION}")
    cases = matrix.get('cases', [])
    if not cases:
        parser.error('the matrix has no cases')

    args.output_dir.mkdir(parents=True)
    rows = [audit_case(case) for case in cases]
    control = next((row for row in rows if row.get('role') == 'control' and row['status'] == 'ran'),
                   None)
    negative_rows: list[dict[str, Any]] = []
    base_meta: dict[str, Any] = {'synthetic': False, 'what': 'no control case ran', 'clean': False}
    if control is not None:
        case = next(item for item in cases if str(item['id']) == control['id'])
        manifest = constraint_manifest(case, started=time.monotonic())
        real_overlap = dict(manifest.overlap_by_entity)
        unsatisfied = int(manifest.measurements['unsatisfied_groups']['solve_medium'])
        base = repaired_base(manifest, case)
        negative_rows, base_meta = run_negatives(base, real_overlap=real_overlap,
                                                 unsatisfied_medium=unsatisfied,
                                                 declared=matrix.get('negatives', []))
        base_meta['overlap_source'] = 'the control case\'s own measurement'
        base_meta['unsatisfied_source'] = f'solve_medium={unsatisfied} in the control build log'
    declared_names = sorted(str(item['name']) for item in matrix.get('negatives', []))
    produced_names = sorted(row['name'] for row in negative_rows)
    set_matches = declared_names == produced_names

    ran = [row for row in rows if row['status'] == 'ran']
    report = {
        'tool': 'python -m rfw_w.scenes.validation.check_background_constraints',
        'spec': 'SPEC-011-background-constraint-injection',
        'stage': 'TASK 5 (a full delivery through the opt-in path); the cases are red exactly where '
                 'the fixture declares they are, and every pooled or delivery room is red on '
                 'upstream\'s own unsatisfied constraints alone',
        'constraints_version': CONSTRAINTS_VERSION,
        'matrix': _relative(args.matrix),
        'matrix_sha256': content_hash(matrix),
        'retrieval_vocabulary': sorted(CATEGORY_PROFILES),
        'cases': rows,
        'negative_set_matches_fixture': set_matches,
        'declared_negatives': declared_names,
        'produced_negatives': produced_names,
        'negatives_base': base_meta,
        'negatives': negative_rows,
        'ran': len(ran), 'not_run': len(rows) - len(ran),
        'matrix_run': len(ran) == len(rows),
        'human_review': {'status': 'not_run',
                         'items': ['whether a room with generation on still reads as a room, '
                                   'not as a furniture warehouse',
                                   'whether the reservation objects are invisible in Blender '
                                   'rather than merely flagged so']},
        'measured_passed': bool(rows) and all(row['passed'] for row in rows),
        'negatives_passed': bool(negative_rows) and base_meta['clean'] and set_matches
                            and base_meta['expectations_agree']
                            and all(row['caught'] for row in negative_rows),
        # A pass here would say every case is green. None can be: requirement 5's switch cannot be turned
        # on in this environment without the room failing on upstream's own unsatisfied constraints
        # (TASK 3 measured 20.2 s), so every case carries that red by design. TASK 5's delivery and
        # the pool's refusals are measured and recorded; the reading is in the PLAN's TASK 5 record.
        'spec_accepted': False,
    }
    report['expected_red_confirmed'] = all(
        row.get('expected_failures_present') == row.get('expect_failures')
        for row in rows if row.get('expect_failures') is not None and row['status'] == 'ran')
    reasons: list[str] = []
    if not ran:
        reasons.append('BACKGROUND_NO_CASE_RAN')
    if not report['negatives_passed']:
        reasons.append('BACKGROUND_NEGATIVES_NOT_CAUGHT')
    # Red and not_run are told apart: one is a case that ran and failed a criterion, the other is a
    # case with no evidence yet, and a reader who mixed them would think the kitchen had been tried.
    red = [row['id'] for row in ran if not row['passed']]
    if red:
        reasons.append('BACKGROUND_CASE_RED(' + ','.join(red) + ')')
    missing = [row['id'] for row in rows if row['status'] != 'ran']
    if missing:
        reasons.append('BACKGROUND_CASE_NOT_RUN(' + ','.join(missing) + ')')
    if not report['expected_red_confirmed']:
        # Deliberately a reason and not a warning: a declared red that is gone means the fixture
        # still says TASK 1's state is current. The declaration has to be revised and the revision
        # recorded -- deleting the check instead would be exactly the silent pass this tool exists
        # to prevent.
        reasons.append('BACKGROUND_DECLARED_RED_GONE(revise expect_failures and record why)')
    report['not_ready_reasons'] = reasons
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))
    for row in rows:
        if row['status'] != 'ran':
            print(f"{row['id']}: {row['status'].upper()} ({row.get('not_run_reason', '')[:80]})")
            continue
        print(f"{row['id']}: {'PASS' if row['passed'] else 'RED'} in {row['elapsed_s']} s "
              f"({len(row['gaps_own'])} constraint gaps, {len(row['gaps_background'])} reused, "
              f"expected present "
              f"{len(row['expected_failures_present'])}/{len(row['expect_failures'])})")
        for gap in row['gaps_own']:
            print(f'    {gap}')
    print(f"negatives: {sum(row['caught'] for row in negative_rows)}/{len(negative_rows)} caught by "
          f"name, repaired base clean={base_meta['clean']}, set matches={set_matches}")
    print(f"measured={report['measured_passed']} negatives={report['negatives_passed']} "
          f"expected_red_confirmed={report['expected_red_confirmed']} "
          f"spec_accepted={report['spec_accepted']} -> {args.output_dir / 'report.json'}")
    if reasons:
        print('not ready: ' + ', '.join(reasons))
    return 0 if not reasons else 1


if __name__ == '__main__':
    sys.exit(main())
