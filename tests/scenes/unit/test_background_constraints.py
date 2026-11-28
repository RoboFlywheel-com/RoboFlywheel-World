# SPDX-License-Identifier: Apache-2.0
"""The task's reservations, the credentials, the gaps and the pool -- and the negatives that prove
each rule can fail.

SPEC-011's whole job is to stop a *silent* violation: a generated shelf standing in the task's
protected region, a solve that could not be satisfied and carried on anyway, a patch nobody
recorded. Every one of those has a named failure, and the names are only worth having if a run can
produce them. So the tests below are in two halves:

**The contract**, built as data: each test starts from a manifest in which every requirement holds
and breaks exactly one fact, so the assertion is "this rule caught this failure" rather than "the
audit is non-empty". That includes the two rules whose spelling is easy to get subtly wrong --
`protected` and `corridor` are separate failures, and the corridor's number is circulation plus door
sweep -- and the factory->category table, which is checked against the *real* retrieval vocabulary:
if somebody adds `shelf.n.01` to `CATEGORY_PROFILES`, the entry that calls a wall shelf a gap has
to be revisited, and these tests fail until it is.

**The tool's negatives**: one repaired base, fourteen broken facts, each caught by the name it
announces and by nothing else. The fourteen are the SPEC's six (with negative 2 split into "the
switch is off" and "an unsatisfied solve continued") plus eight the SPEC's requirements imply -- the
leaked reservation object (requirement 9, which ADR-009's Consequences asks a negative for), the
unmapped factory, the unaccounted gap, the fixture declaring a covered category as a gap, and TASK
4's three: a build that generated with no pool in its record, a ledger and a ruler disagreeing about
a named object, and our own constraint being the unsatisfied one.

Nothing here reads `outputs/`: the evidence those tools audit is not in the repository, and a test
that needed it would be a test that only runs on one machine.
"""
from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.background import (
    EVIDENCE_LOG,
    STAGES,
    BackgroundManifest,
    OwnedEntity,
    StageRecord,
)
from rfw_w.scenes.contracts.background_constraints import (
    ATTEMPTS_PER_CANDIDATE_MAX,
    CIRCULATION_BAND_TOP_M,
    CONSTRAINTS_VERSION,
    DOOR_SWEEP_BAND_TOP_M,
    FACTORY_CATEGORIES,
    GAP_UNFILLED_KINDS,
    GENERATED_OBJECTS_MAX_PER_STAGE,
    INJECTION_APPLIED,
    INJECTION_NOT_APPLIED,
    POOL_REASONS,
    REALISING_STAGE,
    REGION_CIRCULATION,
    REGION_DOOR_SWEEP,
    REGION_PROTECTED,
    RESERVATIONS,
    SOLVE_STAGES,
    BackgroundConstraintError,
    BuildProvenance,
    CandidateAttempt,
    CategoryGap,
    ConstraintManifest,
    ConstraintStage,
    GeneratedObject,
    PoolCandidate,
    RejectionPool,
    Reservation,
    ReservationBundle,
    ReservationPolicy,
    ReservationObject,
    abort_param,
    category_of,
    overlap_reason,
)
from rfw_w.scenes.solvers.furnishing_plan import CATEGORY_PROFILES

REPO_ROOT = Path(__file__).resolve().parents[3]
_PATH = module_path('rfw_w.scenes.validation.check_background_constraints')

TOOL = importlib.import_module('rfw_w.scenes.validation.check_background_constraints')
# Registered before it runs, the same way the SPEC-009 tool's tests load theirs: the module defines
# dataclasses, and `@dataclass` looks its own module up in `sys.modules` to resolve annotations.
# It also imports its sibling by path, which is what mounts the 009 audit inside this one.


MATRIX = json.loads((REPO_ROOT / 'tests/scenes/fixtures/background_constraint_matrix.json').read_text())
DIGEST = 'a' * 64
CORE = ('table', 'thing')
#: The categories the factory table calls gaps, each reviewed against the retrieval vocabulary. A
#: factory the table maps *outside* `CATEGORY_PROFILES` has to be declared here: mapping something
#: to a category nobody can retrieve is a judgement, and this list is where the judgement is
#: written down.
GAP_KINDS = ('book_stack', 'book_column', 'shelf', 'shelf_trinkets', 'mirror_frame', 'monitor')

SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def reservation(region, *, band=CIRCULATION_BAND_TOP_M):
    return Reservation(region=region, polygon_xy=SQUARE, band_top_m=band, source='synthetic')


def generated(name, factory, *, stage='populate_assets'):
    return GeneratedObject(name=name, factory=factory, stage=stage, category=category_of(factory),
                           base_z_m=0.05)


def clean_manifest(**changes) -> ConstraintManifest:
    """A manifest in which every SPEC-011 requirement holds, so a test can break exactly one."""
    background = BackgroundManifest(
        scene_id='synthetic', seed=3, mode='generated',
        entities=(OwnedEntity(entity_id='table', owner='retrieved', scope='core'),
                  OwnedEntity(entity_id='thing', owner='retrieved', scope='core'),
                  OwnedEntity(entity_id='BookStackFactory(1).spawn_asset(2)', owner='generated',
                              scope='periphery', stage='populate_assets')),
        stages=tuple(StageRecord(stage=stage, state='executed', evidence=EVIDENCE_LOG)
                     for stage in STAGES),
        required_entity_ids=CORE, core_hash=DIGEST, core_hash_requested=DIGEST,
        budget={'populate_assets': 4})
    defaults = dict(
        scene_id='synthetic', seed=3, mode='generated', background=background,
        reservations=tuple(reservation(region) for region in RESERVATIONS),
        reservation_objects=tuple(ReservationObject(name=f'pvscene_reservation_{region}',
                                                    region=region) for region in RESERVATIONS),
        build=BuildProvenance(upstream_pin='892947be', entry_script='scripts/ops/build.py',
                              entry_script_sha256=DIGEST, patch_module='patch.py',
                              patch_sha256=DIGEST, injection=INJECTION_APPLIED),
        constraint_stages=tuple(ConstraintStage(stage=stage, abort_unsatisfied=True, unsatisfied=0)
                                for stage in SOLVE_STAGES),
        gaps=(CategoryGap(category='book_stack', why='d',
                          filled_by=('BookStackFactory(1).spawn_asset(2)',)),
              # A gap may stay empty, but only with a reason: one filled, one explained away, so
              # the clean fixture covers both ways a gap is accounted for.
              CategoryGap(category='shelf', why='d', unfilled_reason='no wall run left')),
        generated=(generated('BookStackFactory(1).spawn_asset(2)', 'BookStackFactory'),),
        # A clean manifest has a rejection pool too (TASK 4): generation happened, so an absent or
        # un-run pool is a failure in its own right, and a fixture without one would make every
        # test below assert against a red base.
        pool=RejectionPool(
            stage='populate_assets', ran=True, rule='synthetic: one kept object',
            bounds={'populate_assets': 4},
            candidates=(PoolCandidate(name='BookStackFactory(1).spawn_asset(2)',
                                      factory='BookStackFactory', category='book_stack',
                                      outcome='kept'),)),
        attempts=(CandidateAttempt(candidate_id='BookStackFactory(1).spawn_asset(2)',
                                   stage='populate_assets', attempt=1, outcome='placed'),),
        rejection_bounds={'populate_assets': 4},
        # Upstream's own unsatisfied groups: they must not be read as ours (TASK 3's attribution,
        # in the fixture, so `BACKGROUND_RESERVATION_UNSATISFIED` is tested against a real shape).
        measurements={'unsatisfied_group_names': ['desk', 'lighting', 'office_desk_chairs']},
        retrieval_categories=tuple(sorted(CATEGORY_PROFILES)), overlap_by_entity={})
    defaults.update(changes)
    return ConstraintManifest(**defaults)


@pytest.fixture(scope='module')
def clean() -> ConstraintManifest:
    manifest = clean_manifest()
    assert manifest.own_gaps() == ()  # the fixture is only useful if it starts clean
    return manifest


def broken(manifest: ConstraintManifest, expect: str, **changes) -> tuple[str, ...]:
    """Apply one change and return the gaps, asserting the named failure is among them."""
    gaps = manifest.with_changes(**changes).own_gaps()
    assert any(gap.startswith(expect) for gap in gaps), f'{expect} not in {gaps}'
    return gaps


# ---------------------------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------------------------

def test_regions_are_three_and_the_bands_come_from_the_adr():
    assert RESERVATIONS == (REGION_PROTECTED, REGION_CIRCULATION, REGION_DOOR_SWEEP)
    # ADR-009 decision 3 numbers the corridor at 1.8 m and says the door sweep is included; the
    # protected region's top is the plan's own support height and is never a constant here.
    assert CIRCULATION_BAND_TOP_M == 1.8
    assert DOOR_SWEEP_BAND_TOP_M == 1.8


def test_this_module_keeps_the_protected_region_name_the_ownership_contract_uses():
    """One region, one name across both contracts: a rename here would stop the two matching."""
    from rfw_w.scenes.contracts.background import RESERVATION_PROTECTED
    assert REGION_PROTECTED == RESERVATION_PROTECTED


def test_abort_param_is_the_upstream_spelling():
    assert abort_param('solve_medium') == 'compose_indoors.abort_unsatisfied_medium'
    assert abort_param('solve_large') == 'compose_indoors.abort_unsatisfied_large'


@pytest.mark.parametrize('factory,category', sorted(FACTORY_CATEGORIES.items()))
def test_the_factory_table_agrees_with_the_retrieval_vocabulary(factory, category):
    """Every entry is one of two claims, and both are checkable against the real vocabulary.

    A covered entry is one whose category is in `CATEGORY_PROFILES` -- that is what makes generating
    it `BACKGROUND_NOT_A_GAP`. Everything else is a gap, and a gap is a *judgement* (a wall shelf is
    not the `bookcase.n.01` retrieval provides), so it has to be listed in `GAP_KINDS` by hand.
    """
    if category in CATEGORY_PROFILES:
        assert category not in GAP_KINDS, f'{category} is retrievable and declared a gap'
    else:
        assert category in GAP_KINDS, (
            f'{factory} -> {category} is neither in the retrieval vocabulary nor declared a gap')


def test_every_declared_gap_kind_is_really_unreachable_by_retrieval():
    for category in GAP_KINDS:
        assert category not in CATEGORY_PROFILES, f'{category} is retrievable after all'


def test_adr_009s_own_gap_examples_are_gaps():
    # "book stacks, decorative shelves, mirror frames" -- the ADR named them, so the table cannot
    # quietly reclassify them as covered.
    assert category_of('BookStackFactory') in GAP_KINDS
    assert category_of('CellShelfFactory') in GAP_KINDS
    assert category_of('MirrorFactory') in GAP_KINDS
    assert category_of('WobbleFactory') == ''  # not in the table, so not judged


# ---------------------------------------------------------------------------------------------
# Record validation: a record that cannot be read is refused, not audited
# ---------------------------------------------------------------------------------------------

def test_a_reservation_needs_a_region_a_polygon_a_band_and_a_source():
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_REGION_UNSUPPORTED'):
        reservation('somewhere')
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_REGION_DEGENERATE'):
        Reservation(region=REGION_PROTECTED, polygon_xy=((0.0, 0.0), (1.0, 1.0)),
                    band_top_m=0.6, source='s')
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_REGION_BAND_INVALID'):
        Reservation(region=REGION_PROTECTED, polygon_xy=SQUARE, band_top_m=0.0, source='s')
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_REGION_SOURCE_MISSING'):
        Reservation(region=REGION_PROTECTED, polygon_xy=SQUARE, band_top_m=0.6, source='')


def test_an_unrecorded_credential_is_listed_but_a_reason_is_required_to_be_unpatched():
    empty = BuildProvenance(upstream_pin='', injection=INJECTION_NOT_APPLIED, injection_reason='r')
    assert empty.unrecorded() == ('upstream_pin', 'entry_script', 'entry_script_sha256',
                                  'patch_module', 'patch_sha256')
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_INJECTION_REASON_MISSING'):
        BuildProvenance(upstream_pin='p', injection=INJECTION_NOT_APPLIED)
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_INJECTION_UNSUPPORTED'):
        BuildProvenance(upstream_pin='p', injection='maybe', injection_reason='r')
    # A digest that is not one is unrecorded, so a truncated or hand-written string cannot pass.
    assert BuildProvenance(upstream_pin='p', entry_script='e', entry_script_sha256='abc',
                           patch_module='m', patch_sha256=DIGEST,
                           injection=INJECTION_APPLIED).unrecorded() == ('entry_script_sha256',)


def test_a_solve_stage_record_says_which_stage_and_why_it_has_no_outcome():
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_SOLVE_STAGE_UNSUPPORTED'):
        ConstraintStage(stage='populate_assets', abort_unsatisfied=True, unsatisfied=0)
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_SOLVER_OUTCOME_REASON_MISSING'):
        ConstraintStage(stage='solve_small', abort_unsatisfied=True, unsatisfied=None)
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_SOLVER_OUTCOME_INVALID'):
        ConstraintStage(stage='solve_small', abort_unsatisfied=True, unsatisfied=-1)


def test_a_gap_cannot_be_both_filled_and_explained_away():
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_GAP_OVERACCOUNTED'):
        CategoryGap(category='shelf', filled_by=('CellShelfFactory(1).spawn_asset(2)',),
                    unfilled_reason='nothing fitted')
    assert CategoryGap(category='shelf', unfilled_reason='nothing fitted').accounted_for
    assert not CategoryGap(category='shelf').accounted_for


def test_a_rejected_attempt_carries_its_reason_and_the_stage_it_belongs_to():
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_REJECTION_REASON_MISSING'):
        CandidateAttempt(candidate_id='c', stage='populate_assets', attempt=1, outcome='rejected')
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_ATTEMPT_INDEX_INVALID'):
        CandidateAttempt(candidate_id='c', stage='populate_assets', attempt=0, outcome='placed')
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_ATTEMPT_OUTCOME_INVALID'):
        CandidateAttempt(candidate_id='c', stage='populate_assets', attempt=1, outcome='maybe')
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_STAGE_UNSUPPORTED'):
        CandidateAttempt(candidate_id='c', stage='solve_teapot', attempt=1, outcome='placed')


# ---------------------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------------------

def test_the_default_path_is_not_judged_by_these_rules(clean):
    """`retrieved` generates nothing, so there is no constraint to have been applied.

    The retrieved baseline's own failures are SPEC-009's audit, and reporting them here as
    constraint gaps would make every existing delivery look broken.
    """
    assert clean.with_changes(mode='retrieved', build=None, reservation_objects=()).own_gaps() == ()


def test_a_version_this_module_does_not_know_is_a_failure(clean):
    assert broken(clean, 'BACKGROUND_VERSION_UNSUPPORTED', version='pvscene.other/v9')


def test_every_missing_credential_is_named_separately(clean):
    gaps = broken(clean, 'BACKGROUND_PATCH_UNRECORDED(patch_module)',
                  build=replace(clean.build, patch_module=''))
    assert sorted(gap for gap in gaps if gap.startswith('BACKGROUND_PATCH_UNRECORDED')) == [
        'BACKGROUND_PATCH_UNRECORDED(patch_module)']


def test_no_injection_means_every_region_object_is_missing(clean):
    gaps = broken(clean, 'BACKGROUND_KEEPOUT_OBJECT_MISSING(protected)',
                  build=replace(clean.build, injection=INJECTION_NOT_APPLIED,
                                injection_reason='turned off'),
                  reservation_objects=())
    assert [gap for gap in gaps if gap.startswith('BACKGROUND_KEEPOUT_OBJECT_MISSING')] == [
        f'BACKGROUND_KEEPOUT_OBJECT_MISSING({region})' for region in RESERVATIONS]


def test_a_region_object_that_reaches_the_delivery_is_named_with_what_leaked(clean):
    gaps = broken(clean, 'BACKGROUND_RESERVATION_OBJECT_LEAKED', reservation_objects=tuple(
        replace(item, exported=True, listed_as_instance=True) if item.region == REGION_CIRCULATION
        else item for item in clean.reservation_objects))
    assert [gap for gap in gaps if gap.startswith('BACKGROUND_RESERVATION_OBJECT_LEAKED')] == [
        'BACKGROUND_RESERVATION_OBJECT_LEAKED(pvscene_reservation_circulation:'
        'exported,listed_as_instance)']


def test_an_unsatisfied_solve_is_a_failure_and_an_off_switch_is_a_second_one(clean):
    off = tuple(replace(item, abort_unsatisfied=False) if item.stage == 'solve_medium' else item
                for item in clean.constraint_stages)
    assert broken(clean, 'BACKGROUND_SOLVER_UNGUARDED(solve_medium)',
                  constraint_stages=off) == ('BACKGROUND_SOLVER_UNGUARDED(solve_medium)',)
    continued = tuple(replace(item, abort_unsatisfied=False, unsatisfied=3)
                      if item.stage == 'solve_medium' else item for item in clean.constraint_stages)
    assert broken(clean, 'BACKGROUND_SOLVER_ABORTED(solve_medium)',
                  constraint_stages=continued) == ('BACKGROUND_SOLVER_UNGUARDED(solve_medium)',
                                                   'BACKGROUND_SOLVER_ABORTED(solve_medium)')


def test_a_solve_stage_with_no_record_is_not_read_as_a_pass(clean):
    gaps = broken(clean, 'BACKGROUND_SOLVER_UNGUARDED(solve_small:unrecorded)',
                  constraint_stages=tuple(item for item in clean.constraint_stages
                                          if item.stage != 'solve_small'))
    assert gaps == ('BACKGROUND_SOLVER_UNGUARDED(solve_small:unrecorded)',)


def test_generation_that_retrieval_already_covers_is_not_a_gap(clean):
    gaps = broken(clean, 'BACKGROUND_NOT_A_GAP(desk,covered-by-retrieval', generated=(
        clean.generated + (generated('SimpleDeskFactory(3).spawn_asset(4)', 'SimpleDeskFactory'),)))
    assert [gap for gap in gaps if gap.startswith('BACKGROUND_NOT_A_GAP')] == [
        'BACKGROUND_NOT_A_GAP(desk,covered-by-retrieval,1)']


def test_a_gap_eligible_category_the_case_did_not_declare_is_also_not_a_gap(clean):
    """`monitor` is a gap by the table's test, but this case never declared one.

    The two reasons are spelled apart because the repairs are different: one edits the patch, the
    other edits the fixture.
    """
    gaps = broken(clean, 'BACKGROUND_NOT_A_GAP(monitor,not-declared-in-this-case',
                  generated=clean.generated + (generated('MonitorFactory(5).spawn_asset(6)',
                                                         'MonitorFactory'),))
    assert [gap for gap in gaps if gap.startswith('BACKGROUND_NOT_A_GAP')] == [
        'BACKGROUND_NOT_A_GAP(monitor,not-declared-in-this-case,1)']


def test_a_fixture_that_declares_a_covered_category_is_caught_too(clean):
    gaps = broken(clean, 'BACKGROUND_NOT_A_GAP(plant,covered-by-retrieval',
                  gaps=clean.gaps + (CategoryGap(category='plant', why='mistake',
                                                 unfilled_reason='nothing fitted'),))
    # The declared gap is the failure; the unfilled reason keeps GAP_UNFILLED out of the row. The
    # count is zero because nothing generated a plant: the two directions of the rule are spelled
    # the same way, so a reader sees which side of it fired.
    assert gaps == ('BACKGROUND_NOT_A_GAP(plant,covered-by-retrieval,0)',)


def test_a_gap_with_neither_a_fill_nor_a_reason_is_a_failure(clean):
    gaps = broken(clean, 'BACKGROUND_GAP_UNFILLED(shelf,no-fill-and-no-reason-recorded',
                  gaps=(clean.gaps[0], CategoryGap(category='shelf', why='d')))
    assert gaps == ('BACKGROUND_GAP_UNFILLED(shelf,no-fill-and-no-reason-recorded)',)
    # ...and either one accounts for it.
    assert clean.with_changes(gaps=(clean.gaps[0],
                                    CategoryGap(category='shelf', why='d',
                                                unfilled_reason='no wall run left'))).own_gaps() == ()


def test_an_undeclared_allowance_is_not_an_allowance(clean):
    gaps = broken(clean, 'BACKGROUND_BUDGET_EXCEEDED(populate_assets)',
                  background=clean.background.with_changes(budget={}))
    assert gaps == ('BACKGROUND_BUDGET_EXCEEDED(populate_assets)',)
    # A declared cap the run stayed inside of is clean, and the reused SPEC-009 rule catches the
    # other half (more generated than the cap allows).
    assert clean.with_changes(background=clean.background.with_changes(
        budget={'populate_assets': 1})).own_gaps() == ()


def test_the_rejection_pool_is_bounded_by_attempts_and_by_a_declared_bound(clean):
    retries = tuple(CandidateAttempt(candidate_id='CellShelfFactory(7)', stage='populate_assets',
                                     attempt=attempt, outcome='rejected', reason='no wall run')
                    for attempt in range(1, ATTEMPTS_PER_CANDIDATE_MAX + 2))
    gaps = broken(clean, 'BACKGROUND_REJECTIONS_UNBOUNDED(populate_assets:CellShelfFactory(7),3)',
                  attempts=retries, rejection_bounds={'populate_assets': 4})
    assert gaps == ('BACKGROUND_REJECTIONS_UNBOUNDED(populate_assets:CellShelfFactory(7),3)',)
    inside = clean.with_changes(attempts=retries[:2],
                                rejection_bounds={'populate_assets': 4})
    assert inside.own_gaps() == ()
    over = clean.with_changes(attempts=retries[:2], rejection_bounds={'populate_assets': 1})
    assert [gap for gap in over.own_gaps()
            if gap.startswith('BACKGROUND_REJECTIONS_UNBOUNDED')] == [
        'BACKGROUND_REJECTIONS_UNBOUNDED(populate_assets:2>1)']
    # An undeclared bound is its own failure, and it takes an explicitly empty one: a manifest
    # without the key gets the calibrated allowance (GENERATED_OBJECTS_MAX_PER_STAGE).
    unbounded = clean.with_changes(attempts=retries[:1], rejection_bounds={})
    assert [gap for gap in unbounded.own_gaps()] == [
        'BACKGROUND_REJECTIONS_UNBOUNDED(populate_assets:no-bound,1)']


# ---------------------------------------------------------------------------------------------
# TASK 4: the pool that refuses, the frozen allowance, and the two accounts of one room
# ---------------------------------------------------------------------------------------------

#: A pool that judged one object and kept it, as a value the tests can mutate: the shape a real
#: build writes (one row per candidate, one attempt each).
def pool_that_ran(*, kept=('BookStackFactory(1).spawn_asset(2)',), rejected=(),
                  bounds=({'populate_assets': 4})) -> RejectionPool:
    return RejectionPool(
        stage=REALISING_STAGE, ran=True, rule='synthetic', bounds=bounds,
        candidates=tuple(
            [PoolCandidate(name=name, factory=name.split('(')[0], category=category_of(
                name.split('(')[0]), outcome='kept') for name in kept]
            + [PoolCandidate(name=item, factory=item.split('(')[0],
                             category=category_of(item.split('(')[0]), outcome='rejected',
                             reasons=('not-a-gap(shelf:not-declared-in-this-case)',))
               for item in rejected]))


def test_the_allowance_is_frozen_between_the_measured_rooms_and_the_runaway():
    """The cap's *calibration*, not just its value: 19 is the largest measured injected room (the
    kitchen at the first weights), 32 the same bedroom without the constraint, and a cap outside
    that window would either never bite or fail the rooms the TASK 3 sweep measured."""
    assert GENERATED_OBJECTS_MAX_PER_STAGE == {REALISING_STAGE: 24}
    assert 19 < GENERATED_OBJECTS_MAX_PER_STAGE[REALISING_STAGE] < 32


def test_generation_without_a_pool_record_is_a_failure_and_a_pool_that_ran_clears_it(clean):
    assert clean.own_gaps() == ()
    absent = clean.with_changes(pool=None, attempts=(), rejection_bounds={})
    assert absent.own_gaps() == ('BACKGROUND_POOL_UNRECORDED(populate_assets:no-pool-recorded)',)
    unrun = clean.with_changes(pool=RejectionPool(stage=REALISING_STAGE, ran=False,
                                                  reason='the realising stage is switched off',
                                                  bounds={'populate_assets': 4}))
    assert unrun.own_gaps() == (
        'BACKGROUND_POOL_UNRECORDED(populate_assets:not-run:the realising stage is switched off)',)
    # ... and a build that never ran the realising stage is not asked for a pool at all: the stage's
    # own record is where that is established, and it has to be, or "no pool" and "no generation"
    # would be one fact.
    quiet = clean.with_changes(
        pool=None, attempts=(), rejection_bounds={},
        background=clean.background.with_changes(stages=tuple(
            replace(record, state='not_run', evidence=EVIDENCE_LOG,
                    reason='the default path switches this stage off')
            if record.stage == REALISING_STAGE else record
            for record in clean.background.stages)))
    assert quiet.own_gaps() == ()


def test_the_pool_ledger_and_the_ruler_must_agree_about_an_intruder(clean):
    intruder = 'CellShelfFactory(699309).spawn_asset(3222513)'
    measured = {intruder: {REGION_PROTECTED: 0.1265, REGION_CIRCULATION: 0.0315}}
    # The bug this rule is for: the ledger says the piece was refused and the ruler measures it
    # standing in the region -- a deletion that did not happen.
    disagreeing = clean.with_changes(
        overlap_by_entity=measured, pool=pool_that_ran(rejected=(intruder,))).own_gaps()
    assert 'BACKGROUND_PROTECTED_OVERLAP(%s,0.1265)' % intruder in disagreeing
    assert 'BACKGROUND_POOL_LEDGER_INCOMPLETE(%s,0.1265)' % intruder in disagreeing
    # A pool that kept it is the same disagreement from the other side.
    kept = clean.with_changes(
        overlap_by_entity=measured, pool=pool_that_ran(kept=(intruder,))).own_gaps()
    assert 'BACKGROUND_POOL_LEDGER_INCOMPLETE(%s,0.1265)' % intruder in kept
    # An intruder the ledger never mentions is the overlap failure's own business, and reporting it
    # here too would name one object twice and make every intrusion negative prove two rules.
    unjudged = clean.with_changes(overlap_by_entity=measured).own_gaps()
    assert [gap for gap in unjudged if 'LEDGER_INCOMPLETE' in gap] == []
    # And a build with no pool at all is a third failure (`BACKGROUND_POOL_UNRECORDED`), not this
    # one: "nothing judged the room" and "the ledger and the ruler disagree" are different repairs.
    unpooled = clean.with_changes(overlap_by_entity=measured, pool=None, attempts=(),
                                  rejection_bounds={}).own_gaps()
    assert [gap for gap in unpooled if 'LEDGER_INCOMPLETE' in gap] == []


def test_only_a_constraint_wearing_our_name_is_our_constraint_failing(clean):
    upstream = clean.own_gaps()
    assert [gap for gap in upstream if 'RESERVATION_UNSATISFIED' in gap] == []
    ours = clean.with_changes(measurements={
        **dict(clean.measurements),
        'unsatisfied_group_names': ['lighting', 'pvscene_reservation_protected_penetration']})
    assert [gap for gap in ours.own_gaps() if 'RESERVATION_UNSATISFIED' in gap] == [
        'BACKGROUND_RESERVATION_UNSATISFIED(pvscene_reservation_protected_penetration)']


def test_the_pool_record_refuses_a_shape_it_could_not_be_read_back_in():
    good = PoolCandidate(name='MirrorFactory(1).spawn_asset(0)', factory='MirrorFactory',
                         category='mirror_frame', outcome='rejected',
                         reasons=(overlap_reason(REGION_PROTECTED, 0.0245),))
    assert RejectionPool.from_dict(pool_that_ran(rejected=(good.name,)).to_dict()) is not None
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_POOL_REASON_MISSING'):
        PoolCandidate(name='X', factory='F', category='shelf', outcome='rejected')
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_POOL_KEPT_WITH_REASON'):
        PoolCandidate(name='X', factory='F', category='shelf', outcome='kept', reasons=('why',))
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_POOL_REASON_UNKNOWN'):
        PoolCandidate(name='X', factory='F', category='shelf', outcome='rejected',
                      reasons=('it looked wrong',))
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_POOL_REASON_MISSING'):
        # An absent pool carries the reason it is absent, or a room with generation and no judgement
        # in it reads as a room where nothing needed judging.
        RejectionPool(stage=REALISING_STAGE, ran=False)


def test_the_pool_hands_its_ledger_to_the_rules_that_already_count_attempts():
    pool = pool_that_ran(kept=('BookStackFactory(1).spawn_asset(2)',),
                         rejected=('CellShelfFactory(7).spawn_asset(0)',))
    assert [(item.candidate_id, item.outcome, item.attempt) for item in pool.attempts()] == [
        ('BookStackFactory(1).spawn_asset(2)', 'placed', 1),
        ('CellShelfFactory(7).spawn_asset(0)', 'rejected', 1)]
    assert [item.reason for item in pool.attempts()][1].startswith('not-a-gap(')
    assert pool.attempts_per_candidate() == {'BookStackFactory(1).spawn_asset(2)': 1,
                                             'CellShelfFactory(7).spawn_asset(0)': 1}
    assert sorted(pool.rejected_by_category()) == ['shelf']
    assert pool.rejection_reasons() == {'not-a-gap': ('CellShelfFactory(7).spawn_asset(0)',)}
    assert [item.category for item in pool.kept()] == ['book_stack']


def test_the_refusal_reason_spells_the_region_and_the_number():
    assert overlap_reason(REGION_PROTECTED, 0.0245) == 'protected-overlap(0.0245m2)'
    # The door sweep is reported with the corridor, which is how the contract's own `corridor_overlap`
    # counts it and how SPEC-009's second number was measured.
    assert overlap_reason(REGION_DOOR_SWEEP, 0.0) == 'corridor-overlap(0.0000m2)'
    assert overlap_reason(REGION_CIRCULATION, 0.0315) == 'corridor-overlap(0.0315m2)'
    assert set(POOL_REASONS) == {'protected-overlap', 'corridor-overlap', 'not-a-gap',
                                 'factory-unmapped'}


def test_a_gap_reason_says_which_of_the_measurements_it_is(clean):
    """The kinds, in the order the evidence supports them: a refused candidate is the pool working,
    a category nothing proposed is the sampler's limit, a build with no pool can only report what is
    in the room, and the last two are the delivery's own -- the assembly refused what the pool kept,
    or the assembly did not run and the kept objects reached no delivery at all."""
    assert GAP_UNFILLED_KINDS == ('proposed-but-rejected', 'not-proposed-by-this-run',
                                  'not-in-the-room', 'refused-at-assembly',
                                  'kept-but-not-delivered')
    filled = {'shelf': ('CellShelfFactory(7).spawn_asset(0)',)}
    assert TOOL._unfilled_reason('shelf', filled, pool_that_ran()) == ''
    refused = TOOL._unfilled_reason('shelf', {}, pool_that_ran(rejected=('CellShelfFactory(7)',)))
    assert refused.startswith('proposed-but-rejected(not-a-gap(shelf:not-declared-in-this-case)')
    never = TOOL._unfilled_reason('shelf', {}, pool_that_ran())
    assert never.startswith('not-proposed-by-this-run(')
    unpooled = TOOL._unfilled_reason('shelf', {}, None)
    assert unpooled.startswith('not-in-the-room(')
    assert all(reason.split('(')[0] in GAP_UNFILLED_KINDS
               for reason in (refused, never, unpooled))


def test_the_bundle_the_build_is_handed_carries_the_gaps_and_the_allowance():
    bundle = ReservationBundle(
        scene_id='synthetic', core_hash=DIGEST, upstream_pin='892947be',
        entry_script='scripts/ops/run_infinigen_build.py', entry_script_sha256=DIGEST,
        patch_module='scripts/ops/infinigen_constraint_patch.py', patch_sha256=DIGEST,
        reservations=(reservation(REGION_PROTECTED, band=0.575),),
        policy=ReservationPolicy(note='test'), gaps=('book_stack',),
        retrieval_categories=('desk', 'shelf'))
    assert bundle.budget == GENERATED_OBJECTS_MAX_PER_STAGE
    assert ReservationBundle.from_dict(bundle.to_dict()).gaps == ('book_stack',)
    assert ReservationBundle.from_dict(bundle.to_dict()).budget == GENERATED_OBJECTS_MAX_PER_STAGE
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_BUNDLE_VOCABULARY_MISSING'):
        # Without the vocabulary the pool's refusal reason cannot tell "retrieval covers it" from
        # "this room did not ask for it", and the record would be guessing.
        ReservationBundle.from_dict({**bundle.to_dict(), 'retrieval_categories': []})


def test_the_two_reservations_are_reported_separately_and_the_corridor_includes_the_sweep(clean):
    measured = {REGION_PROTECTED: 0.1265, REGION_CIRCULATION: 0.02, REGION_DOOR_SWEEP: 0.0115}
    gaps = broken(clean, 'BACKGROUND_KEEPOUT_OVERLAP(', overlap_by_entity={'CellShelfFactory(7)': measured})
    assert gaps == ('BACKGROUND_PROTECTED_OVERLAP(CellShelfFactory(7),0.1265)',
                    'BACKGROUND_KEEPOUT_OVERLAP(CellShelfFactory(7),0.0315)')
    # Within the tolerance is the sliver two coincident edges share, not an intrusion.
    assert clean.with_changes(overlap_by_entity={
        'CellShelfFactory(7)': {REGION_PROTECTED: 1e-9, REGION_CIRCULATION: 0.0,
                                REGION_DOOR_SWEEP: 0.0}}).own_gaps() == ()


def test_the_audit_is_the_reused_ownership_half_plus_this_specs_rules(clean):
    """Requirement 8: SPEC-009's audit stays the judge for the core, and is not re-implemented."""
    # A generated entity claiming a requested entity's id is the 009 duplicate-owner failure, and
    # it must appear in this manifest's verdict too.
    background = clean.background.with_changes(
        entities=clean.background.entities + (
            OwnedEntity(entity_id='table', owner='generated', scope='periphery',
                        stage='populate_assets'),))
    both = clean.with_changes(background=background,
                              overlap_by_entity={'CellShelfFactory(7)': {
                                  REGION_PROTECTED: 0.125, REGION_CIRCULATION: 0.0,
                                  REGION_DOOR_SWEEP: 0.0}})
    verdict = both.audit()
    assert 'BACKGROUND_DUPLICATE_OWNER(table)' in verdict       # the reused half
    assert 'BACKGROUND_DUPLICATE_OWNER(table)' not in both.own_gaps()
    assert 'BACKGROUND_PROTECTED_OVERLAP(CellShelfFactory(7),0.1250)' in verdict  # this SPEC's
    assert set(both.own_gaps()) < set(verdict)


def test_the_report_payload_carries_both_halves_and_the_derived_facts(clean):
    changed = clean.with_changes(attempts=(CandidateAttempt(
        candidate_id='c', stage='populate_assets', attempt=1, outcome='rejected',
        reason='r'),))
    payload = changed.to_dict()
    assert payload['version'] == CONSTRAINTS_VERSION
    assert payload['named_gaps'] == list(changed.own_gaps())
    assert payload['own_named_gaps'] == list(changed.own_gaps())
    assert payload['generated_by_stage'] == {'populate_assets': 1}
    assert payload['declared_gap_categories'] == ['book_stack', 'shelf']
    assert payload['rejected_candidates'] == [
        {'entity_id': 'c', 'stage': 'populate_assets', 'reason': 'r'}]
    assert payload['background'] is not None  # the ownership half is in the same report


# ---------------------------------------------------------------------------------------------
# The tool: the negatives, and the shape of the fixture
# ---------------------------------------------------------------------------------------------

def tool_base() -> ConstraintManifest:
    """A base the tool's negatives can be built from, with the two live measurements supplied."""
    return clean_manifest(
        gaps=(CategoryGap(category='book_stack', why='d', filled_by=(
            'BookStackFactory(1).spawn_asset(2)',)),),
        rejection_bounds={stage: 4 for stage in STAGES})


def test_the_repaired_base_the_negatives_start_from_is_clean():
    base = tool_base()
    assert base.audit() == ()


def test_every_negative_is_caught_by_the_name_it_announces_and_isolates_one_fact():
    base = tool_base()
    rows, meta = TOOL.run_negatives(
        base, real_overlap={'CellShelfFactory(699309).spawn_asset(3222513)': {
            REGION_PROTECTED: 0.1265, REGION_CIRCULATION: 0.0315, REGION_DOOR_SWEEP: 0.0}},
        unsatisfied_medium=3, declared=MATRIX['negatives'])
    assert meta['clean'] and meta['expectations_agree']
    assert meta['declared'] == meta['produced'] == sorted(row['name'] for row in MATRIX['negatives'])
    for row in rows:
        assert row['caught'], row
        # One broken fact each: a negative that tripped other rules would not say which rule it
        # proved, and the SPEC's six would be six rows proving the same thing.
        assert row['extra'] == [], row


def test_a_solve_outcome_is_only_reported_for_a_stage_the_log_shows_running():
    """The tool's own version of the rule it exists to enforce: silence is not zero.

    The solver logs its failure unconditionally (`solve.py:206-217`), so an absence of that line is
    a measurement of a stage whose `finished in` line proves it ran -- and for one that did not, the
    outcome is unmeasured, not satisfied.
    """
    counts, violations = TOOL._solve_outcomes('')
    assert counts == {stage: 0 for stage in SOLVE_STAGES}
    assert not TOOL._stage_finished('', 'solve_large')
    assert TOOL._stage_finished('[solve_large] finished in 0:00:06.388297', 'solve_large')
    line = ("| Solver has failed to satisfy constraints for stage ('side_obj_0', 'office_0/0'). "
            "violations={'desk': 2, 'office_desk_chairs': 2}. Continuing anyway, override")
    counts, violations = TOOL._solve_outcomes(line)
    assert counts['solve_medium'] == 1 and counts['solve_large'] == 0
    assert violations['solve_medium'] == ["{'desk': 2, 'office_desk_chairs': 2}"]
    # An inner stage name nothing maps is not attributed to a guessed outer stage.
    assert TOOL._solve_outcomes(
        "Solver has failed to satisfy constraints for stage ('what_is_this_0', 'r'). "
        "violations={'x': 1}.")[0] == {stage: 0 for stage in SOLVE_STAGES}


def test_every_inner_solve_stage_maps_to_one_of_the_three_outer_stages():
    assert set(TOOL.SOLVE_STAGE_OF.values()) == set(SOLVE_STAGES)
    for outer in SOLVE_STAGES:
        assert [name for name, group in TOOL.SOLVE_STAGE_OF.items() if group == outer]


def test_the_six_spec_negatives_are_all_present_and_the_extras_are_declared_as_extras():
    by_spec: dict[object, list[str]] = {}
    for item in MATRIX['negatives']:
        by_spec.setdefault(item['spec_negative'], []).append(item['name'])
    assert sorted(name for key, names in by_spec.items() if key is not None for name in names) == [
        'abort_switch_off', 'budget_exceeded', 'covered_category_generated', 'credentials_missing',
        'no_separation_intrusion_returns', 'rejections_unbounded', 'unsatisfied_solve_continued']
    # Negative 2 is split, so seven rows cover the SPEC's six; the rest are extras and say so.
    # TASK 4 added three more extras, each for a rule the SPEC's six never reached: the pool that
    # must have run (requirement 1's fallback), the two accounts of one room disagreeing about an
    # object, and our own constraint being the unsatisfied one (which TASK 3 could only attribute
    # in prose until a rule named it).
    assert len(by_spec[None]) == 7


def test_the_two_negatives_that_replay_a_failure_carry_the_probes_own_numbers():
    base = tool_base()
    rows, meta = TOOL.run_negatives(
        base, real_overlap={'CellShelfFactory(699309).spawn_asset(3222513)': {
            REGION_PROTECTED: 0.1265, REGION_CIRCULATION: 0.0315, REGION_DOOR_SWEEP: 0.0}},
        unsatisfied_medium=3, declared=MATRIX['negatives'])
    intrusion = next(row for row in rows if row['name'] == 'no_separation_intrusion_returns')
    assert 'BACKGROUND_PROTECTED_OVERLAP(CellShelfFactory(699309).spawn_asset(3222513),0.1265)' \
        in intrusion['observed']
    assert 'BACKGROUND_KEEPOUT_OVERLAP(CellShelfFactory(699309).spawn_asset(3222513),0.0315)' \
        in intrusion['observed']
    continued = next(row for row in rows if row['name'] == 'unsatisfied_solve_continued')
    assert 'BACKGROUND_SOLVER_ABORTED(solve_medium)' in continued['observed']
    assert meta['intruder'] == 'CellShelfFactory(699309).spawn_asset(3222513)'


def test_the_negatives_refuse_to_edit_the_base_they_were_derived_from():
    base = tool_base()
    before = base.to_dict()
    TOOL.run_negatives(base, real_overlap={'e': {REGION_PROTECTED: 0.125, REGION_CIRCULATION: 0.0,
                                                 REGION_DOOR_SWEEP: 0.0}},
                       unsatisfied_medium=1, declared=MATRIX['negatives'])
    assert base.to_dict() == before


def test_the_fixture_declares_the_control_case_the_room_types_and_the_pooled_pairs():
    cases = MATRIX['cases']
    assert [case['id'] for case in cases] == ['bedroom_control_37', 'bedroom_injected_37',
                                              'kitchen_medium_20', 'office_medium_20',
                                              'office_pooled_20', 'bedroom_pooled_37',
                                              'kitchen_pooled_23', 'office_delivery_gen_20']
    control = cases[0]
    assert control['role'] == 'control'
    # The control is SPEC-009's probe room: same delivery, same seed, and the probe build beside it.
    assert control['seed'] == 37 and 'spec009_populate_probe' in control['build']
    assert set(control['gap_categories']) < set(GAP_KINDS)
    assert {'book_stack', 'shelf', 'mirror_frame'} <= set(control['gap_categories'])
    assert control['min_generated_categories'] >= 2  # ADR-009 decision 11's lower bound
    injected = cases[1]
    # The injected case is the control's own room through our entry (SPEC-011 TASK 2), so the two
    # rows are one bedroom before and after the constraint -- which is what makes the comparison a
    # measurement rather than a story about two different rooms.
    assert injected['delivery'] == control['delivery']
    assert injected['seed'] == control['seed'] == 37
    assert 'spec011_injection' in injected['build']
    assert injected['gap_categories'] == control['gap_categories']
    # TASK 3 built the two room types, so these rows stopped being declarations and became
    # measurements: each has a build, a ruler output, and a note saying what the row claims. The
    # `not_run_reason` is gone *because* it is no longer true -- a reason left behind on a case with
    # evidence would be the fixture lying in the direction nobody checks.
    for case in cases[2:4]:
        assert case['room_type'] in ('kitchen', 'office')
        assert case['measurement'] and 'spec011_' in case['build']
        assert 'not_run_reason' not in case
        assert case['case_note']
    kitchen, office = cases[2], cases[3]
    # The kitchen's own seed declaration was 20 while the case never ran; the room's seed is 23
    # (delivery and build command agree). Corrected here so the row cannot be read as "seed 20 was
    # tried and worked".
    assert kitchen['seed'] == 23 and office['seed'] == 20
    # TASK 4's three pooled rows: each is one of the rooms above, same delivery and same seed, built
    # through our entry with the rejection pool on -- so each pair is one room with and without the
    # refusal, which is what makes the pool's effect a measurement rather than an argument.
    pooled = {case['id']: case for case in cases if case['role'] == 'pooled'}
    assert sorted(pooled) == ['bedroom_pooled_37', 'kitchen_pooled_23', 'office_pooled_20']
    for pair, unpooled in (('office_pooled_20', office), ('bedroom_pooled_37', injected),
                           ('kitchen_pooled_23', kitchen)):
        case = pooled[pair]
        assert case['delivery'] == unpooled['delivery'] and case['seed'] == unpooled['seed']
        assert 'spec011_' in case['build'] and case['measurement'] and case['case_note']
        assert case['gap_categories'] == unpooled['gap_categories']
        assert 'not_run_reason' not in case
    # TASK 5's row points at a *delivery* rather than a probe: the same room as office_medium_20,
    # built through the pipeline's own run with the opt-in on, so the delivery's `background` block
    # and its build's ledger are what the row is judged against.
    delivery = next(case for case in cases if case['role'] == 'delivery')
    assert delivery['delivery'] == 'outputs/office_pick_cup/spec011_gen_20260923_01'
    assert 'shell_build' in delivery['build'] and delivery['measurement'] and delivery['case_note']
    assert delivery['gap_categories'] == office['gap_categories']
    assert delivery['seed'] == office['seed'] == 20


#: Every named failure this SPEC can produce, the reused SPEC-009 names included. Written out here
#: on purpose: the fixture's expectations are only meaningful if they name failures that exist, and
#: renaming one should force an edit to this list rather than quietly matching nothing.
NAMED_FAILURES = frozenset({
    'BACKGROUND_VERSION_UNSUPPORTED', 'BACKGROUND_DUPLICATE_OWNER',
    'BACKGROUND_CORE_OWNER_GENERATED', 'BACKGROUND_REQUIRED_ENTITY_MISSING',
    'BACKGROUND_CORE_MUTATED', 'BACKGROUND_PROTECTED_OVERLAP', 'BACKGROUND_KEEPOUT_OVERLAP',
    'BACKGROUND_BUDGET_EXCEEDED', 'BACKGROUND_STAGE_NOT_REPORTED', 'BACKGROUND_STAGE_UNSUPPORTED',
    'BACKGROUND_PATCH_UNRECORDED', 'BACKGROUND_KEEPOUT_OBJECT_MISSING',
    'BACKGROUND_RESERVATION_OBJECT_LEAKED', 'BACKGROUND_SOLVER_ABORTED',
    'BACKGROUND_SOLVER_UNGUARDED', 'BACKGROUND_NOT_A_GAP', 'BACKGROUND_GAP_UNFILLED',
    'BACKGROUND_REJECTIONS_UNBOUNDED', 'BACKGROUND_FACTORY_UNMAPPED',
    'BACKGROUND_RESERVATION_UNSATISFIED', 'BACKGROUND_POOL_UNRECORDED',
    'BACKGROUND_POOL_LEDGER_INCOMPLETE',
})


def test_every_failure_name_the_fixture_expects_is_one_this_spec_can_produce():
    expectations = [expect for case in MATRIX['cases']
                    for expect in case.get('expect_failures', [])]
    expectations += [expect for item in MATRIX['negatives'] for expect in item['expect']]
    # The count was 25 while the fixture had two cases, 32 after TASK 3 built the kitchen and the
    # office, 57 after TASK 4 revised the three declarations the calibration invalidated (recorded
    # in each case's note), added the three negatives for its own rules, and added the three pooled
    # cases; 62 after TASK 5 added the delivery row. It stays pinned because a silently dropped
    # expectation is exactly the kind of drift this test exists to catch -- and every *ran* case must
    # declare what its red means, or its row would be a red nobody can check.
    assert len(expectations) == 62
    for prefix in expectations:
        assert prefix.startswith('BACKGROUND_'), prefix
        assert prefix.split('(')[0] in NAMED_FAILURES, prefix
    ran = [case for case in MATRIX['cases'] if case.get('measurement')]
    assert ran and all(case.get('expect_failures') for case in ran)
