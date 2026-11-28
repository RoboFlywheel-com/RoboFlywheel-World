# SPDX-License-Identifier: Apache-2.0
"""The constraint injection's wiring: the request, the two commands, and the record's refusals.

SPEC-011 TASK 2 builds a pipe between our provider and a build running inside Blender. The build
itself needs Blender and is not a unit test's business, but everything *around* it is: which
reservations the plan yields, what the two commands look like, whether a bundle survives a round
trip, and whether a build's record can be refused for the four ways it can be one we did not ask
for. Those are the parts where a mistake is silent -- a command that quietly enables generation in
the default path, or a record nobody checks -- so they are pinned here.

The most important assertion in this file is a *negative* one: **the default command does not
change.** Its bytes are what produced the deliveries SPEC-004/005/009 quote, and an opt-in feature
that rewrote the default path's entry script would make every recorded command unreproducible.
"""
from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.background_constraints import (
    BUNDLE_FILENAME,
    GENERATED_OBJECTS_MAX_PER_STAGE,
    INJECTION_APPLIED,
    INJECTION_FAILED,
    INJECTION_FILENAME,
    RESERVATION_OBJECT_PREFIX,
    UNCALIBRATED_POLICY_NOTE,
    BackgroundConstraintError,
    Reservation,
    ReservationBundle,
    ReservationPolicy,
)
from rfw_w.scenes.contracts.room_plan import RoomPlan
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.furnishing_plan import CATEGORY_PROFILES
from rfw_w.scenes.solvers.room_plan import reservations_from_plan
from rfw_w.scenes.solvers.shell_providers import (
    BACKGROUND_ENTRY_SCRIPT,
    background_block,
    generation_provider,
    INFINIGEN_PIN,
    INFINIGEN_STAGE_OVERRIDES_OFF,
    BackgroundConstraints,
    InfinigenShellProvider,
    ShellProviderError,
    provider_named,
    reservation_bundle_for,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_ENTRY = module_path('rfw_w.scenes.workers.blender.run_infinigen_build')

ENTRY = importlib.import_module('rfw_w.scenes.workers.blender.run_infinigen_build')


_PROBE = module_path('rfw_w.scenes.preparation.probe_constraint_injection')

PROBE = importlib.import_module('rfw_w.scenes.preparation.probe_constraint_injection')


_PATCH = module_path('rfw_w.scenes.workers.blender.infinigen_constraint_patch')

PATCH = importlib.import_module('rfw_w.scenes.workers.blender.infinigen_constraint_patch')


SQUARE = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
RUN_DIR = Path('/tmp/pvscene-run')


def plan_of(path: Path) -> RoomPlan:
    return RoomPlan.from_dict(json.loads(path.read_text()))


@pytest.fixture(scope='module')
def plan() -> RoomPlan:
    return plan_of(REPO_ROOT / 'tests/scenes/fixtures/office_room_plan.json')


# ---------------------------------------------------------------------------------------------
# The reservations a plan yields
# ---------------------------------------------------------------------------------------------

def test_every_region_gets_a_reservation_with_its_own_band_and_source(plan):
    records = reservations_from_plan(plan, circulation=[SQUARE, SQUARE], door_sweep=[SQUARE])
    by_region = {region: [r for r in records if r.region == region] for region in
                 ('protected', 'circulation', 'door_sweep')}
    assert len(by_region['protected']) == 1
    assert len(by_region['circulation']) == 2 and len(by_region['door_sweep']) == 1
    # The protected band is the plan's own support height; the corridor's and the sweep's come from
    # the contract, because ADR-009 decision 3 fixed them and a second copy would drift.
    assert by_region['protected'][0].band_top_m == plan.core.support_top_z_m
    assert by_region['circulation'][0].band_top_m == 1.8
    assert by_region['door_sweep'][0].band_top_m == 1.8
    assert by_region['protected'][0].source == 'plan.core.protected_in_room_xy'
    assert by_region['circulation'][0].source == 'zones.circulation'
    assert {item.index for item in by_region['circulation']} == {0, 1}
    # Names are per polygon: five objects sharing one name overwrite each other in the solver's
    # state, which is what the first injected build measured.
    assert {item.object_name for item in records} == {
        f'{RESERVATION_OBJECT_PREFIX}protected_0', f'{RESERVATION_OBJECT_PREFIX}circulation_0',
        f'{RESERVATION_OBJECT_PREFIX}circulation_1', f'{RESERVATION_OBJECT_PREFIX}door_sweep_0'}


# ---------------------------------------------------------------------------------------------
# TASK 3's control: the same request, inert geometry
# ---------------------------------------------------------------------------------------------

def test_the_ablation_sends_the_same_requests_over_tiny_squares(plan):
    """The control that makes the injected build's two numbers attributable.

    "0 m² intrusion but 32 -> 13 objects" cannot be read from the injected run alone. The ablation
    keeps the region, the index, the band and the polygon count -- so the solver sees the same
    objects, tags, relations and terms -- and shrinks only the area. What the ruler then measures
    against the real regions (it re-derives them from the delivery) is what generation would have
    placed inside them had the constraint not been sent.
    """
    reservations = reservations_from_plan(plan, circulation=[SQUARE, SQUARE], door_sweep=[SQUARE])
    ablated = PROBE.ablate(reservations, 0.05)
    assert [(item.region, item.index, item.band_top_m) for item in ablated] == \
           [(item.region, item.index, item.band_top_m) for item in reservations]
    assert [item.object_name for item in ablated] == [item.object_name for item in reservations]
    for item in ablated:
        polygon = Polygon(item.polygon_xy)
        assert polygon.area == pytest.approx(0.01)
        # Not the plan's polygon, and it says so: the bundle on disk must not be miscounted as one
        # the plan asked for.
        assert item.source.startswith('ablation(degenerate):0.05m-square of ')
        assert item.polygon_xy != next(r.polygon_xy for r in reservations
                                       if (r.region, r.index) == (item.region, item.index))


# ---------------------------------------------------------------------------------------------
# The two commands
# ---------------------------------------------------------------------------------------------

def test_the_default_command_is_the_one_the_recorded_deliveries_were_built_with(plan, tmp_path):
    """No opt-in, no change: same entry script, generation still switched off, same overrides."""
    # `embodiedgen_root` is injected rather than taken from `.env.local`: this test is about the
    # argv, and a machine-bound path is not part of that claim (the real provider refuses an unset
    # root by name, which is its own test elsewhere).
    provider = replace(provider_named('infinigen'), infinigen_root=tmp_path)
    argv = provider.command(plan, run_dir=RUN_DIR)
    overrides = argv[argv.index('-p') + 1:]
    assert argv[1:3] == ['-m', BACKGROUND_ENTRY_SCRIPT]
    assert 'compose_indoors.populate_assets_enabled=False' in overrides
    assert not any('populate_assets_enabled=True' in item for item in overrides)
    assert not any('abort_unsatisfied' in item for item in overrides)
    # SPEC-009 TASK 2's small piece: `floating_objs` says so in the command rather than leaving the
    # stage's state to be inferred from silence.
    assert 'compose_indoors.floating_objs_enabled=False' in overrides
    assert set(INFINIGEN_STAGE_OVERRIDES_OFF) == {
        'compose_indoors.populate_assets_enabled=False',
        'compose_indoors.floating_objs_enabled=False'}


def test_the_opt_in_command_runs_our_entry_and_names_the_bundle_and_the_record(plan, tmp_path):
    provider = InfinigenShellProvider(
        infinigen_root=tmp_path,
        background_constraints=BackgroundConstraints(
            reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),),
            gaps=('book_stack',)))
    argv = provider.command(plan, run_dir=RUN_DIR)
    overrides = argv[argv.index('-p') + 1:]
    assert argv[1:3] == ['-m', BACKGROUND_ENTRY_SCRIPT]
    assert '--pvscene-reservations' in argv and str(RUN_DIR / BUNDLE_FILENAME) in argv
    assert '--pvscene-provenance' in argv and str(RUN_DIR / INJECTION_FILENAME) in argv
    # Generation on, said explicitly rather than by leaving an override out: the first injected
    # build left it off and measured a zero intrusion over a nearly empty room.
    assert 'compose_indoors.populate_assets_enabled=True' in overrides
    assert 'compose_indoors.populate_assets_enabled=False' not in overrides
    # Requirement 5's spelling: the switch that makes an unsatisfied solve loud.
    assert [item for item in overrides if 'abort_unsatisfied' in item] == [
        f'compose_indoors.abort_unsatisfied_{group}=True' for group in ('large', 'medium', 'small')]


def test_a_policy_that_asks_for_no_loud_failure_sends_no_abort_switch(plan, tmp_path):
    provider = InfinigenShellProvider(
        infinigen_root=tmp_path,
        background_constraints=BackgroundConstraints(
            reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),),
            gaps=('book_stack',),
            policy=ReservationPolicy(abort_unsatisfied=False, note='probing')))
    argv = provider.command(plan, run_dir=RUN_DIR)
    assert not any('abort_unsatisfied' in item for item in argv[argv.index('-p') + 1:])


def test_a_run_can_ask_for_the_constraints_without_asking_generation_on(plan, tmp_path):
    """The knob exists so SPEC-009's baseline can be reproduced through our entry as well."""
    provider = InfinigenShellProvider(
        infinigen_root=tmp_path,
        background_constraints=BackgroundConstraints(
            reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),),
            generation=False))
    overrides = provider.command(plan, run_dir=RUN_DIR)
    overrides = overrides[overrides.index('-p') + 1:]
    assert 'compose_indoors.populate_assets_enabled=False' in overrides
    # And the stage whose state nobody could otherwise see still says so either way.
    assert 'compose_indoors.floating_objs_enabled=False' in overrides


def test_our_arguments_are_taken_out_of_the_builds_argv_without_disturbing_the_rest():
    """The generator parses `sys.argv` itself, so ours must leave nothing behind in it."""
    argv = ['build.py', '--output_folder', '/tmp/scene', '--pvscene-reservations', '/tmp/b.json',
            '--seed', '37', '--pvscene-provenance', '/tmp/r.json', '-p', 'a=True']
    ours, rest = ENTRY.split_our_arguments(argv)
    assert ours == {'reservations': '/tmp/b.json', 'provenance': '/tmp/r.json'}
    assert rest == ['build.py', '--output_folder', '/tmp/scene', '--seed', '37', '-p', 'a=True']


# ---------------------------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------------------------

def test_a_bundle_carries_the_credentials_and_survives_a_round_trip(plan):
    constraints = BackgroundConstraints(
        reservations=reservations_from_plan(plan, circulation=[SQUARE], door_sweep=[SQUARE]),
        gaps=('book_stack', 'mirror_frame'))
    bundle = reservation_bundle_for(
        plan, constraints, entry_script=module_path(BACKGROUND_ENTRY_SCRIPT),
        patch_module=module_path('rfw_w.scenes.workers.blender.infinigen_constraint_patch'))
    payload = json.loads(json.dumps(bundle.to_dict()))
    again = ReservationBundle.from_dict(payload)
    assert again == bundle
    assert again.upstream_pin == INFINIGEN_PIN
    assert again.reservations == bundle.reservations
    assert len(again.entry_script_sha256) == 64 and len(again.patch_sha256) == 64
    # The first weights travel as a request, and the note is what stops a reader taking them for
    # measurements: TASK 3 calibrates them.
    assert again.policy.calibrated is False
    assert again.policy.note == UNCALIBRATED_POLICY_NOTE
    # TASK 4: the request carries what the pool enforces (which categories may be added), the
    # vocabulary that reason is judged against, and the allowance the count is judged by.
    assert again.gaps == ('book_stack', 'mirror_frame')
    assert again.budget == GENERATED_OBJECTS_MAX_PER_STAGE
    assert again.retrieval_categories == tuple(sorted(CATEGORY_PROFILES))


def test_a_request_that_asks_for_generation_without_declaring_gaps_is_refused():
    """ADR-009 decision 4, refused at request time: with no gap declared the pool would refuse
    every object the build realises, so the three minutes would buy a slower way to build the same
    room. The message says which emptiness it is."""
    with pytest.raises(ShellProviderError, match=r'BACKGROUND_BUNDLE_EMPTY\(no gaps to fill\)'):
        BackgroundConstraints(reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),))
    # ... and a request that asks for no generation needs no gaps.
    quiet = BackgroundConstraints(
        reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),), generation=False)
    assert quiet.gaps == ()


def test_a_declared_gap_retrieval_already_covers_is_refused_before_the_build():
    """The vocabulary check runs at request time, not only in the audit: a gap the retrieval
    already provides is a request the pipeline cannot honour (requirement 4)."""
    with pytest.raises(ShellProviderError, match=r'BACKGROUND_NOT_A_GAP\(desk,covered-by-retrieval'):
        BackgroundConstraints(reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),),
                              gaps=('desk',))
    with pytest.raises(ShellProviderError, match='BACKGROUND_GAP_CATEGORY_UNKNOWN'):
        BackgroundConstraints(reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),),
                              gaps=('wobble',))


def test_an_uncalibrated_policy_has_to_say_so():
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_POLICY_NOTE_MISSING'):
        ReservationPolicy(band_penetration_weight=10.0)
    assert ReservationPolicy(calibrated=True).note == ''


def test_a_bundle_with_no_reservations_a_bad_digest_or_no_path_is_refused():
    # The provider's own type refuses an empty request, and the contract's refuses a malformed one.
    with pytest.raises(ShellProviderError, match='BACKGROUND_BUNDLE_EMPTY'):
        BackgroundConstraints(reservations=())
    reserve = Reservation('protected', SQUARE.vertices, 0.6, 's')
    base = dict(scene_id='p', core_hash='a' * 64, upstream_pin='pin', entry_script='e.py',
                entry_script_sha256='a' * 64, patch_module='p.py', patch_sha256='b' * 64,
                reservations=(reserve,), policy=ReservationPolicy(note='n'))
    assert ReservationBundle(**base).patch_sha256 == 'b' * 64
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_BUNDLE_DIGEST_INVALID'):
        ReservationBundle(**{**base, 'patch_sha256': 'nonsense'})
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_BUNDLE_PATH_MISSING'):
        ReservationBundle(**{**base, 'entry_script': ''})
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_BUNDLE_EMPTY'):
        ReservationBundle(**{**base, 'reservations': ()})


def test_a_bundle_this_module_does_not_know_is_refused_rather_than_half_read(plan):
    constraints = BackgroundConstraints(
        reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),),
        gaps=('book_stack',))
    payload = reservation_bundle_for(
        plan, constraints, entry_script=module_path(BACKGROUND_ENTRY_SCRIPT),
        patch_module=module_path('rfw_w.scenes.workers.blender.infinigen_constraint_patch')).to_dict()
    with pytest.raises(BackgroundConstraintError, match='BACKGROUND_BUNDLE_VERSION_UNSUPPORTED'):
        ReservationBundle.from_dict({**payload, 'version': 'pvscene.reservation-bundle/v9'})


# ---------------------------------------------------------------------------------------------
# The opt-in delivery path (TASK 5): the same reservations a probe sends, from the run's own plan
# ---------------------------------------------------------------------------------------------

class _Zones:
    """A zone graph as `generation_provider` reads it: two tuples of polygons, nothing else."""

    def __init__(self, circulation=(), door_sweep=()):
        self.circulation = tuple(circulation)
        self.door_sweep = tuple(door_sweep)


def test_the_opt_in_provider_derives_the_reservations_from_the_plan_and_its_zones(plan):
    """A delivery run and a probe must ask the build for the same thing: this is the one call that
    turns a plan plus the zones this run drew into the request, so the two cannot drift apart."""
    provider = generation_provider(plan, zones=_Zones(circulation=[SQUARE], door_sweep=[]),
                                  gaps=['book_stack', 'mirror_frame'])
    constraints = provider.background_constraints
    regions = [item.region for item in constraints.reservations]
    assert regions == ['protected', 'circulation']
    protected = constraints.reservations[0]
    assert protected.polygon_xy == tuple(plan.core.protected_in_room_xy)
    assert protected.band_top_m == plan.core.support_top_z_m
    assert constraints.gaps == ('book_stack', 'mirror_frame')
    assert constraints.generation is True
    # The default path is a different value, not a flag: `provider_named` asks for no reservations
    # and records nothing about them (requirement 7).
    assert provider_named('infinigen').background_constraints is None


def test_the_deliverys_opt_in_block_records_the_request_and_the_builds_own_records(tmp_path):
    """The delivery's `background` block is the reader's way in: what was asked, and where the
    build's own evidence is. A run that asked for generation and left no record says *that* rather
    than carrying an empty block."""
    assert background_block(generation=False, gaps=[], out=tmp_path) == {}
    missing = background_block(generation=True, gaps=['book_stack'], out=tmp_path)
    assert missing['injection'] == 'not-recorded'
    assert 'BACKGROUND_PATCH_UNRECORDED' in missing['injection_reason']
    assert missing['mode'] == 'generated' and missing['gaps'] == ['book_stack']
    build = tmp_path / 'shell_build' / 'abc123' / 'infinigen_build'
    build.mkdir(parents=True)
    (build / BUNDLE_FILENAME).write_text('{}')
    (build / INJECTION_FILENAME).write_text(json.dumps({
        'injection': INJECTION_APPLIED, 'injection_reason': 'reservations removed after solve_small',
        'upstream_pin': INFINIGEN_PIN, 'patch_sha256': 'a' * 64, 'entry_script_sha256': 'b' * 64,
        'budget': {'populate_assets': 24},
        'reservations': [{'region': 'protected'}],
        'rejection_pool': {'ran': True, 'kept': 3, 'rejected': 5, 'bounds': {'populate_assets': 24},
                           'rejected_categories': ['desk'], 'rejection_reasons': {'not-a-gap': ['x']},
                           'rejected_state_keys': ['desk.1'], 'rule': 'one rule'},
    }))
    block = background_block(generation=True, gaps=['book_stack'], out=tmp_path, injection_record=build / INJECTION_FILENAME)
    assert block['injection'] == INJECTION_APPLIED
    assert block['rejection_pool'] == {'ran': True, 'rule': 'one rule',
                                      'bounds': {'populate_assets': 24}, 'kept': 3, 'rejected': 5,
                                      'rejected_categories': ['desk'],
                                      'rejection_reasons': {'not-a-gap': ['x']}}
    assert block['rejected_state_keys'] == ['desk.1']
    assert block['record_path'].endswith('constraint_injection.json')
    assert block['bundle_path'].endswith(BUNDLE_FILENAME)
    assert block['why_abort_is_off'].startswith('abort_unsatisfied_* is off')
    # No assembly step ran for this delivery, and the block says so instead of leaving a reader to
    # infer it from a missing key (SPEC-009 TASK 2's remaining half).
    assert 'assembly' not in block
    assert 'did not run or did not finish' in block['delivery_note']
    # With the assembly on disk the same block says the opposite, and carries what the step wrote.
    directory = tmp_path / 'generated_furniture'
    directory.mkdir()
    (directory / 'assembly.json').write_text(json.dumps({
        'status': 'assembled', 'instances': [{'instance_id': 'generated_monitor_1',
                                              'category': 'monitor'}],
        'categories': ['monitor'], 'kept_in_ledger': 1, 'exported': 1, 'skipped': [],
    }))
    assembled = background_block(generation=True, gaps=['book_stack'], out=tmp_path, assembly=json.loads((directory / 'assembly.json').read_text()), injection_record=build / INJECTION_FILENAME)
    assert assembled['assembly']['instances'][0]['instance_id'] == 'generated_monitor_1'
    assert 'instance of this delivery' in assembled['delivery_note']


def test_the_opt_in_provider_refuses_a_request_the_build_could_not_enforce(plan):
    zones = _Zones(circulation=[SQUARE])
    with pytest.raises(ShellProviderError, match=r'BACKGROUND_BUNDLE_EMPTY\(no gaps to fill\)'):
        generation_provider(plan, zones=zones, gaps=[])
    with pytest.raises(ShellProviderError, match=r'BACKGROUND_NOT_A_GAP\(desk,covered-by-retrieval'):
        generation_provider(plan, zones=zones, gaps=['desk'])


# ---------------------------------------------------------------------------------------------
# The pool's judgement (TASK 4): pure logic, so it is testable without Blender
# ---------------------------------------------------------------------------------------------

def _injection(plan: RoomPlan) -> object:
    bundle = reservation_bundle_for(
        plan, BackgroundConstraints(
            reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),),
            gaps=('book_stack', 'shelf')),
        entry_script=module_path(BACKGROUND_ENTRY_SCRIPT), patch_module=_PATCH)
    return PATCH.Injection(bundle, provenance_path=Path('/tmp/pvscene-record.json'),
                           patch_path=_PATCH, entry_path=module_path(BACKGROUND_ENTRY_SCRIPT))


def test_the_pool_refuses_by_where_a_piece_stands_and_by_what_category_it_would_add(plan):
    """Both halves of requirement 4/3 in one place, with the reasons the audit reads back.

    The pool's judgement runs inside Blender, but the *decision* is this method: given a name and
    the two numbers the shared geometry rule produced, which named reasons apply. That is what makes
    it testable here rather than only in a three-minute build.
    """
    injection = _injection(plan)
    # Standing in the protected region *and* being a category this room never asked for: two
    # reasons, and the second one says which kind of not-a-gap it is (the monitor is gap-eligible
    # but this request declared only book_stack and shelf).
    assert injection._refusal_reasons('MonitorFactory(1).spawn_asset(2)', 0.0245, 0.0) == (
        'protected-overlap(0.0245m2)', 'not-a-gap(monitor:not-declared-in-this-case)')
    # A *declared gap* that stands in a region is refused for the region alone: the room wanted the
    # category, and the reservation is what it may not occupy.
    assert injection._refusal_reasons('CellShelfFactory(1).spawn_asset(2)', 0.0245, 0.0) == (
        'protected-overlap(0.0245m2)',)
    # A covered category that stands nowhere: refused, and the reason says why it is not a gap.
    assert injection._refusal_reasons('SimpleDeskFactory(1).spawn_asset(0)', 0.0, 0.0) == (
        'not-a-gap(desk:covered-by-retrieval)',)
    # A gap category standing clear of every region: the pool keeps it.
    assert injection._refusal_reasons('BookStackFactory(1).spawn_asset(0)', 0.0, 0.0) == ()
    # The corridor's number is the corridor's (door sweep included), and a factory the table cannot
    # place is reported rather than defaulted into either column.
    assert injection._refusal_reasons('WobbleFactory(1).spawn_asset(0)', 0.0, 0.0315) == (
        'corridor-overlap(0.0315m2)', 'factory-unmapped(WobbleFactory)')


def test_the_pool_record_the_patch_writes_is_a_contract_type(plan):
    """A pool that cannot be read back cannot be written either: the record is built through
    `RejectionPool`/`PoolCandidate`, the same schema the audit parses, and the contract refuses a
    refusal with no reason, a kept candidate with one, and an unknown reason kind."""
    source = _PATCH.read_text()
    assert 'contract.RejectionPool(' in source and 'contract.PoolCandidate(' in source
    assert 'geometry.reservations' in source  # the shared rule, named in the record
    bundle = _injection(plan).bundle
    record = PATCH._contract().RejectionPool(
        stage='populate_assets', ran=True, rule='test',
        bounds=dict(bundle.budget),
        candidates=(PATCH._contract().PoolCandidate(
            name='BookStackFactory(1).spawn_asset(0)', factory='BookStackFactory',
            category='book_stack', outcome='kept'),))
    assert PATCH._contract().RejectionPool.from_dict(record.to_dict()) == record
    # And the patch's own record shape (what a build writes) reads back as a pool.
    written = {**record.to_dict(), 'rejected_state_keys': [], 'attempts_per_candidate_max': 1}
    assert PATCH._contract().RejectionPool.from_dict(written).ran is True
    assert PATCH._contract().POOL_VERSION == PATCH._contract().RejectionPool.from_dict(
        written).version


# ---------------------------------------------------------------------------------------------
# The record's refusals
# ---------------------------------------------------------------------------------------------

def _bundle(plan: RoomPlan) -> ReservationBundle:
    return reservation_bundle_for(
        plan, BackgroundConstraints(
            reservations=(Reservation('protected', SQUARE.vertices, 0.6, 's'),),
            gaps=('book_stack',)),
        entry_script=module_path(BACKGROUND_ENTRY_SCRIPT),
        patch_module=module_path('rfw_w.scenes.workers.blender.infinigen_constraint_patch'))


def _record(bundle: ReservationBundle, tmp_path: Path, **changes) -> Path:
    body = {'injection': INJECTION_APPLIED, 'injection_reason': 'installed',
            'patch_sha256': bundle.patch_sha256,
            'entry_script_sha256': bundle.entry_script_sha256,
            'objects_removed_before_export': {'removed': True, 'still_in_bpy': [],
                                              'still_in_state': []}}
    body.update(changes)
    (tmp_path / INJECTION_FILENAME).write_text(json.dumps(body))
    return tmp_path


def test_a_build_that_left_no_record_is_refused(plan, tmp_path):
    with pytest.raises(ShellProviderError, match='BACKGROUND_INJECTION_FAILED'):
        InfinigenShellProvider().read_injection(_bundle(plan), run_dir=tmp_path)


def test_a_patch_that_could_not_install_is_refused_with_its_reason(plan, tmp_path):
    run_dir = _record(_bundle(plan), tmp_path, injection=INJECTION_FAILED,
                      injection_reason='ImportError: no util')
    with pytest.raises(ShellProviderError, match='BACKGROUND_INJECTION_FAILED'):
        InfinigenShellProvider().read_injection(_bundle(plan), run_dir=run_dir)


def test_a_record_whose_digest_is_not_the_requested_one_is_refused(plan, tmp_path):
    """The record is what makes the constraint auditable, so a record that disagrees with the
    request means the build ran something other than what was asked for."""
    run_dir = _record(_bundle(plan), tmp_path, patch_sha256='c' * 64)
    with pytest.raises(ShellProviderError, match=r'BACKGROUND_PATCH_UNRECORDED\(patch_sha256\)'):
        InfinigenShellProvider().read_injection(_bundle(plan), run_dir=run_dir)


def test_reservation_objects_that_stayed_in_the_scene_are_refused(plan, tmp_path):
    """Requirement 9: an object still in the scene when assets are realised can reach the export."""
    run_dir = _record(_bundle(plan), tmp_path,
                      objects_removed_before_export={'removed': False})
    with pytest.raises(ShellProviderError, match='BACKGROUND_RESERVATION_OBJECT_LEAKED'):
        InfinigenShellProvider().read_injection(_bundle(plan), run_dir=run_dir)


def test_a_record_that_agrees_is_returned(plan, tmp_path):
    bundle = _bundle(plan)
    record = InfinigenShellProvider().read_injection(bundle, run_dir=_record(bundle, tmp_path))
    assert record['injection'] == INJECTION_APPLIED


# ---------------------------------------------------------------------------------------------
# The audit tool reads the record instead of a claim about it
# ---------------------------------------------------------------------------------------------

def test_the_audit_tool_takes_credentials_and_objects_from_the_builds_own_record():
    """SPEC-011 requirement 2 reaches the audit through the record, not through the fixture.

    Checked on the tool's source rather than by running it, because running it needs a build on
    disk; what matters here is that the reader exists and that the old "nothing recorded" path is
    still what a build without a record gets.
    """
    source = (module_path('rfw_w.scenes.validation.check_background_constraints')).read_text()
    assert 'INJECTION_FILENAME' in source and 'reservation_objects' in source
    assert 'BACKGROUND_PATCH_UNRECORDED' in source  # the name the record's absence produces
    assert 'PLANNED_PATCH_MODULE = "python -m rfw_w.scenes.workers.blender.infinigen_constraint_patch"' in source
