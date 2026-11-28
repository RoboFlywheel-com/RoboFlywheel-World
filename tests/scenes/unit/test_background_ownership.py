# SPDX-License-Identifier: Apache-2.0
"""Ownership, the frozen core, and the stage evidence a background run has to produce.

Two things are under test and both are load-bearing.

**The ownership line.** SPEC-009 rests on one question -- is this piece the task's or the
generator's -- and the answer is not the entity's role. A book the task rests on the nightstand has
role `decoration` and is core, because the hard ON relation has no subject without it. If that rule
is wrong, generation is allowed to replace the thing the task is defined by, and every other check
in the SPEC would still pass.

**The stage report.** The baseline this tool measures is not "Infinigen supports background
furniture"; it is that `compose_indoors` already *solves* furniture placeholders and the export
throws them away. A stage report that reads upstream's silence as `not_run` would make that
invisible, so the tests below pin the three kinds of evidence apart: the command we ran, a line in
the build's own log, and nothing at all.
"""
from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
import sys
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.background import (
    BACKGROUND_VERSION,
    EVIDENCE_LOG,
    EVIDENCE_NONE,
    EVIDENCE_OVERRIDE,
    RESERVATION_KEEPOUT,
    RESERVATION_PROTECTED,
    STAGES,
    BackgroundError,
    BackgroundManifest,
    BackgroundRequest,
    OwnedEntity,
    RejectedCandidate,
    StageRecord,
    core_entity_ids,
    enabled_param,
    periphery_entity_ids,
    retrieved_ownership,
)
from rfw_w.scenes.contracts.scene_spec import SceneSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
_PATH = module_path('rfw_w.scenes.validation.check_generated_background')

TOOL = importlib.import_module('rfw_w.scenes.validation.check_generated_background')
# Registered before it runs: the module defines dataclasses, and `@dataclass` looks its own
# module up in `sys.modules` to resolve annotations.


ROOM = {"size_bounds_m": [[4.0, 5.0], [3.0, 4.0], [2.6, 3.0]]}
STYLE = {"theme": "t", "palette": ["oak"], "density": "medium"}


def entity(eid, category, role, *, hard=False, mobility="fixed"):
    return {"id": eid, "category": category, "physics_type": "rigid", "mobility": mobility,
            "hard_required": hard, "role": role}


def spec_of(entities, relations, targets):
    return SceneSpec.from_dict({
        "schema_version": "pvscene.scene-spec/v3", "scene_id": "synthetic", "seed": 3,
        "prompt": "synthetic", "room": ROOM, "style": STYLE, "requested_backends": ["isaac"],
        "requested_readiness": "functional_scene_ready", "entities": entities,
        "initial_relations": relations,
        "task": {"action": "pick_place", "robot_profile": "franka_panda", "targets": list(targets)},
    })


@pytest.fixture(scope='module')
def spec() -> SceneSpec:
    """The bedroom's shape in miniature: a support, a target, and a book the task rests on it."""
    return spec_of(
        [entity("table", "desk", "support", hard=True),
         entity("thing", "cup", "task_target", hard=True, mobility="dynamic"),
         entity("book", "book", "decoration"),
         entity("rug", "rug", "decoration")],
        [{"id": "r1", "subject": "thing", "predicate": "ON", "object": "table", "hard": True},
         {"id": "r2", "subject": "book", "predicate": "ON", "object": "table", "hard": True}],
        ["thing"])


class TestTheOwnershipLine:
    def test_a_decoration_the_task_rests_on_is_core(self, spec) -> None:
        """The rule the whole split rests on, and the one role alone gets wrong."""
        assert 'book' in core_entity_ids(spec)
        assert 'book' not in periphery_entity_ids(spec)

    def test_the_support_and_the_target_are_core_by_role(self, spec) -> None:
        assert set(core_entity_ids(spec)) == {'table', 'thing', 'book'}

    def test_an_entity_the_task_does_not_name_is_periphery(self, spec) -> None:
        assert periphery_entity_ids(spec) == ('rug',)

    def test_retrieval_owns_the_core_always_and_the_periphery_today(self, spec) -> None:
        owned = {item.entity_id: item for item in retrieved_ownership(spec)}
        assert owned['book'].owner == 'retrieved' and owned['book'].scope == 'core'
        assert owned['rug'].owner == 'retrieved' and owned['rug'].scope == 'periphery'

    def test_a_core_entity_may_not_be_handed_to_the_generator(self, spec) -> None:
        entities = tuple(
            OwnedEntity(entity_id=item.entity_id, owner='generated', scope=item.scope,
                        stage='solve_medium') if item.entity_id == 'book' else item
            for item in retrieved_ownership(spec))
        assert ('BACKGROUND_CORE_OWNER_GENERATED(book)'
                in _manifest(spec, entities=entities, mode='generated').audit())


class TestTheVocabulary:
    def test_an_unknown_stage_is_named_by_its_upstream_name(self) -> None:
        with pytest.raises(BackgroundError, match=r'BACKGROUND_STAGE_UNSUPPORTED\(decorate\)'):
            StageRecord(stage='decorate', state='executed')

    def test_a_stage_that_did_not_run_must_say_why(self) -> None:
        with pytest.raises(BackgroundError, match=r'BACKGROUND_STAGE_REASON_MISSING\(solve_medium\)'):
            StageRecord(stage='solve_medium', state='not_run')

    def test_an_executed_stage_needs_no_reason(self) -> None:
        assert StageRecord(stage='solve_medium', state='executed').state == 'executed'

    def test_a_generated_entity_must_name_the_stage_that_made_it(self) -> None:
        with pytest.raises(BackgroundError, match=r'BACKGROUND_GENERATED_STAGE_MISSING\(rug\)'):
            OwnedEntity(entity_id='rug', owner='generated', scope='periphery')

    def test_an_unknown_owner_is_named(self) -> None:
        with pytest.raises(BackgroundError, match=r'BACKGROUND_OWNER_UNSUPPORTED\(rug:hssd\)'):
            OwnedEntity(entity_id='rug', owner='hssd', scope='periphery')

    def test_an_unknown_mode_is_named(self) -> None:
        with pytest.raises(BackgroundError, match=r'BACKGROUND_MODE_UNSUPPORTED\(hybrid\)'):
            BackgroundRequest(scene_id='s', mode='hybrid')

    def test_a_rejection_carries_its_reason(self) -> None:
        with pytest.raises(BackgroundError, match=r'BACKGROUND_REJECTION_REASON_MISSING\(sofa\)'):
            RejectedCandidate(entity_id='sofa', stage='solve_large', reason='')

    def test_the_enabled_parameter_is_the_one_the_stage_runner_reads(self) -> None:
        assert enabled_param('populate_assets') == 'compose_indoors.populate_assets_enabled'

    def test_a_version_this_contract_does_not_describe_is_refused(self) -> None:
        with pytest.raises(BackgroundError, match=r'BACKGROUND_VERSION_UNSUPPORTED\(v0\)'):
            BackgroundManifest.from_dict({'scene_id': 's', 'version': 'v0'})


def _manifest(spec: SceneSpec, **changes) -> BackgroundManifest:
    """A clean manifest for the fixture spec: every stage reported, nothing generated."""
    base = BackgroundManifest(
        scene_id='synthetic', mode='retrieved', entities=retrieved_ownership(spec),
        stages=tuple(StageRecord(stage=stage, state='executed', evidence=EVIDENCE_LOG)
                     for stage in STAGES),
        required_entity_ids=core_entity_ids(spec), core_hash='a' * 64, core_hash_requested='a' * 64)
    return base.with_changes(**changes)


class TestTheAuditCanPass:
    def test_a_clean_manifest_has_no_gaps(self, spec) -> None:
        """A gate that cannot pass is not a gate. This is the case that proves it can."""
        assert _manifest(spec).audit() == ()

    def test_mode_retrieved_allows_no_generated_entity_anywhere(self, spec) -> None:
        generated = tuple(
            OwnedEntity(entity_id=item.entity_id, owner='generated', scope=item.scope,
                        stage='solve_medium') if item.entity_id == 'rug' else item
            for item in retrieved_ownership(spec))
        gaps = _manifest(spec, entities=generated).audit()
        assert 'BACKGROUND_BUDGET_EXCEEDED(solve_medium)' in gaps


class TestTheAuditNamesWhatIsWrong:
    def test_two_owners_for_one_entity_are_named(self, spec) -> None:
        entities = retrieved_ownership(spec) + (
            OwnedEntity(entity_id='rug', owner='generated', scope='periphery',
                        stage='solve_medium'),)
        gaps = _manifest(spec, entities=entities, mode='generated',
                         budget={stage: 2 for stage in STAGES}).audit()
        assert 'BACKGROUND_DUPLICATE_OWNER(rug)' in gaps

    def test_a_required_entity_that_was_not_placed_is_named(self, spec) -> None:
        entities = tuple(item for item in retrieved_ownership(spec) if item.entity_id != 'book')
        assert ('BACKGROUND_REQUIRED_ENTITY_MISSING(book)'
                in _manifest(spec, entities=entities).audit())

    def test_a_core_hash_that_moved_is_named(self, spec) -> None:
        gaps = _manifest(spec, core_hash='b' * 64).audit()
        assert f'BACKGROUND_CORE_MUTATED({"b" * 16})' in gaps

    def test_a_footprint_in_the_protected_region_is_named_with_its_area(self, spec) -> None:
        gaps = _manifest(spec, overlap_by_entity={
            'rug': {RESERVATION_PROTECTED: 0.125, RESERVATION_KEEPOUT: 0.0}}).audit()
        assert 'BACKGROUND_PROTECTED_OVERLAP(rug,0.1250)' in gaps

    def test_a_footprint_in_a_keepout_is_named_by_its_own_line(self, spec) -> None:
        """The corridor and the task cell are different reservations, so they read differently."""
        gaps = _manifest(spec, overlap_by_entity={
            'rug': {RESERVATION_PROTECTED: 0.0, RESERVATION_KEEPOUT: 0.0625}}).audit()
        assert 'BACKGROUND_KEEPOUT_OVERLAP(rug,0.0625)' in gaps

    def test_a_square_millimetre_of_shared_edge_is_not_an_intrusion(self, spec) -> None:
        gaps = _manifest(spec, overlap_by_entity={
            'rug': {RESERVATION_PROTECTED: 1e-7, RESERVATION_KEEPOUT: 0.0}}).audit()
        assert gaps == ()

    def test_a_stage_over_its_allowance_is_named_by_stage(self, spec) -> None:
        entities = tuple(
            OwnedEntity(entity_id=item.entity_id, owner='generated', scope=item.scope,
                        stage='solve_small') if item.entity_id == 'rug' else item
            for item in retrieved_ownership(spec))
        gaps = _manifest(spec, entities=entities, mode='generated',
                         budget={'solve_small': 0}).audit()
        assert 'BACKGROUND_BUDGET_EXCEEDED(solve_small)' in gaps

    def test_a_stage_nobody_reported_is_named(self, spec) -> None:
        stages = tuple(record for record in _manifest(spec).stages if record.stage != 'solve_small')
        assert 'BACKGROUND_STAGE_NOT_REPORTED(solve_small)' in _manifest(spec, stages=stages).audit()

    def test_a_stage_whose_state_rests_on_nothing_is_named_too(self, spec) -> None:
        """Silence in the build log is not a measurement, even when a record exists."""
        stages = tuple(
            StageRecord(stage=stage, state='not_run', evidence=EVIDENCE_NONE, reason='unlogged')
            if stage == 'floating_objs' else record
            for stage, record in ((r.stage, r) for r in _manifest(spec).stages))
        assert ('BACKGROUND_STAGE_NOT_REPORTED(floating_objs)'
                in _manifest(spec, stages=stages).audit())

    def test_a_stage_upstream_cannot_run_is_named_unsupported(self, spec) -> None:
        stages = tuple(record for record in _manifest(spec).stages if record.stage != 'solve_small')
        stages += (StageRecord(stage='solve_small', state='unsupported', evidence=EVIDENCE_LOG,
                               reason='this build has no small-furniture solver'),)
        assert 'BACKGROUND_STAGE_UNSUPPORTED(solve_small)' in _manifest(spec, stages=stages).audit()


class TestTheManifestRoundTrips:
    def test_a_manifest_survives_a_dict_round_trip(self, spec) -> None:
        original = _manifest(spec, overlap_by_entity={'rug': {RESERVATION_PROTECTED: 0.0,
                                                             RESERVATION_KEEPOUT: 0.25}})
        restored = BackgroundManifest.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()
        assert restored.audit() == original.audit()

    def test_the_report_carries_its_own_verdict(self, spec) -> None:
        payload = _manifest(spec, core_hash='b' * 64).to_dict()
        assert f'BACKGROUND_CORE_MUTATED({"b" * 16})' in payload['named_gaps']
        assert payload['version'] == BACKGROUND_VERSION
        assert payload['stage_table']['solve_large'] == 'executed'


# ---------------------------------------------------------------------------------------------
# The tool: the geometry it measures with, and the evidence it reads
# ---------------------------------------------------------------------------------------------

class TestTheFootprintIsTheBoxThatWasPlaced:
    def test_an_unrotated_footprint_is_the_rectangle_and_turns_with_its_yaw(self) -> None:
        square = TOOL.footprint_polygon((0.0, 0.0, 0.0), (2.0, 1.0, 0.5), 0.0)
        assert square.area == pytest.approx(2.0)
        turned = TOOL.footprint_polygon((0.0, 0.0, 0.0), (2.0, 1.0, 0.5), 3.141592653589793 / 2)
        assert turned.area == pytest.approx(2.0)
        assert max(x for x, _ in turned.vertices) == pytest.approx(0.5)
        assert max(y for _, y in turned.vertices) == pytest.approx(1.0)

    def test_the_overlap_is_the_shared_area_not_a_boolean(self) -> None:
        # Two 2 x 1 footprints 1.5 m apart share a 0.5 x 1.0 strip: a boolean would say "they
        # overlap" and a bounding-box check would say the same thing about a piece two metres away.
        quarter = TOOL.footprint_polygon((1.5, 0.0, 0.0), (2.0, 1.0, 0.5), 0.0)
        region = TOOL.footprint_polygon((0.0, 0.0, 0.0), (2.0, 1.0, 0.5), 0.0)
        assert TOOL.overlap_area_m2(quarter, [region]) == pytest.approx(0.5)

    def test_a_footprint_clear_of_the_region_shares_nothing(self) -> None:
        away = TOOL.footprint_polygon((9.0, 9.0, 0.0), (2.0, 1.0, 0.5), 0.0)
        region = TOOL.footprint_polygon((0.0, 0.0, 0.0), (2.0, 1.0, 0.5), 0.0)
        assert TOOL.overlap_area_m2(away, [region]) == 0.0


class TestTheStageEvidence:
    """Three kinds of evidence, kept apart: our command, the build's log, and silence."""

    _LOG = (
        '[15:08:32.518] [logging] [INFO] | [solve_large] finished in 0:00:06.416313\n'
        'x | [pipeline] [INFO] | Skipping run_stage(nature_backdrop...) due to unmet prereq=\'t\'\n'
        '[15:10:26.901] [logging] [INFO] | [solve_medium] finished in 0:01:54.033239\n'
    )
    _COMMAND = ('python run_generate_indoors.py -g base_indoors custom_solve '
                'compose_indoors.populate_assets_enabled=False')

    def _run(self):
        class _Run:
            stage_log = self._LOG
            command = self._COMMAND
            # A delivery with no build directory: `stage_records` reads the build's own record for a
            # stage that *failed*, and this fixture is about the three sources that need no record.
            build_dir = None
        return _Run()

    def test_a_stage_the_build_recorded_as_failed_is_failed_not_unreported(self, tmp_path) -> None:
        """The fourth source, and the one that only exists when a build dies part-way.

        Measured on bedroom seed 11: `populate_assets` crashed on a missing Blender addon and the
        injection record says `<stage> failed` plus why. Reporting that stage as `not_run` with no
        evidence would read as "we did not look", which is the opposite of what happened.
        """
        build = tmp_path / 'infinigen_build'
        build.mkdir()
        (build / TOOL.INJECTION_FILENAME).write_text(json.dumps({
            'injection_reason': 'reservations removed after solve_small; populate_assets failed',
            'rejection_pool': {'ran': False, 'reason': "ValueError: addon='x' not found"}}))

        class _Run:
            stage_log = self._LOG
            command = self._COMMAND
            build_dir = build

        records = {record.stage: record for record in TOOL.stage_records(_Run())}
        assert records['populate_assets'].state == 'failed'
        assert records['populate_assets'].evidence == EVIDENCE_LOG
        assert 'addon' in records['populate_assets'].reason

    def test_a_stage_in_the_build_log_is_executed_with_its_measured_duration(self) -> None:
        records = {record.stage: record for record in TOOL.stage_records(self._run())}
        assert records['solve_medium'].state == 'executed'
        assert records['solve_medium'].duration_s == pytest.approx(round(114.033239, 3))
        assert records['solve_medium'].evidence == EVIDENCE_LOG

    def test_the_stage_our_own_command_switched_off_is_evidence_about_our_command(self) -> None:
        records = {record.stage: record for record in TOOL.stage_records(self._run())}
        assert records['populate_assets'].state == 'not_run'
        assert records['populate_assets'].evidence == EVIDENCE_OVERRIDE
        assert 'populate_assets_enabled=False' in records['populate_assets'].reason

    def test_a_stage_the_build_never_mentions_is_not_measured_at_all(self) -> None:
        records = {record.stage: record for record in TOOL.stage_records(self._run())}
        assert records['floating_objs'].state == 'not_run'
        assert records['floating_objs'].evidence == EVIDENCE_NONE

    def test_a_skipped_stage_quotes_the_line_that_skipped_it(self) -> None:
        log = ('x | [pipeline] [INFO] | Skipping run_stage(solve_small...) '
               'due to unmet prereq=\'solve_medium\'\n')
        run = self._run()
        run.stage_log = log
        records = {record.stage: record for record in TOOL.stage_records(run)}
        assert records['solve_small'].state == 'not_run'
        assert "prereq='solve_medium'" in records['solve_small'].reason


class TestTheNegatives:
    """Each negative is a mutation of a real manifest and must be caught by the name it claims."""

    @pytest.fixture()
    def run(self, spec, tmp_path):
        """A minimal delivery on disk: one periphery piece standing on the floor, one hanging."""
        class _Run:
            run_dir = tmp_path
            spec = None
            delivery = None
            support_top_z_m = 0.75
        run = _Run()
        run.spec = spec
        run.delivery = {
            'scene_id': 'synthetic',
            'assets_used': [{'entity_id': 'rug', 'provider': 'hssd', 'asset_id': 'a',
                             'dimensions_m': [2.0, 1.5, 0.02]},
                            {'entity_id': 'lamp', 'provider': 'hssd', 'asset_id': 'b',
                             'dimensions_m': [0.35, 0.35, 0.3]}],
            'placement': {'instances': {
                'rug': {'position_m': [1.0, 1.0, 0.006], 'yaw_rad': 0.0},
                'lamp': {'position_m': [1.0, 1.0, 2.6], 'yaw_rad': 0.0}}},
        }
        return run

    def test_the_intrusion_negative_is_a_piece_that_stands_on_the_floor(self, run) -> None:
        """A lamp hanging 2.6 m up is not the intruder, and picking it would prove nothing."""
        assert TOOL._intruding_entity(run, core=set(core_entity_ids(run.spec)))[0] == 'rug'

    def test_every_negative_is_caught_by_the_name_it_announces(self, run) -> None:
        manifest = BackgroundManifest(
            scene_id='synthetic', entities=retrieved_ownership(run.spec),
            stages=tuple(StageRecord(stage=stage, state='executed', evidence=EVIDENCE_LOG)
                         for stage in STAGES),
            required_entity_ids=core_entity_ids(run.spec),
            core_hash='a' * 64, core_hash_requested='a' * 64)
        rows = TOOL.run_negatives(run, manifest)
        assert [row['name'] for row in rows] == ['protected_intrusion', 'keepout_intrusion',
                                                 'core_forced_generated', 'duplicate_owner',
                                                 'required_entity_dropped', 'tiny_budget',
                                                 'stage_left_unreported']
        for row in rows:
            assert row['caught'], f"{row['name']} was not caught: {row['observed']}"

    def test_a_negative_never_edits_the_evidence_it_came_from(self, run) -> None:
        manifest = BackgroundManifest(
            scene_id='synthetic', entities=retrieved_ownership(run.spec),
            stages=tuple(StageRecord(stage=stage, state='executed', evidence=EVIDENCE_LOG)
                         for stage in STAGES),
            required_entity_ids=core_entity_ids(run.spec))
        before = manifest.to_dict()
        TOOL.run_negatives(run, manifest)
        assert manifest.to_dict() == before

class TestTheGeneratedHalf:
    """What a delivery carries of the generated branch, and whether it matches the declaration.

    The pool (SPEC-011) judges candidates inside the build; these are the delivery-side rules
    (ADR-009 decision 11) and they are not a second opinion about the same thing -- they are about
    what actually reached the delivery. A category the room never declared, a gap nobody filled and
    nobody explained, or a manifest row with no instance behind it are all failures that a build
    record cannot show and a delivery can.
    """

    @pytest.fixture()
    def run(self, spec, tmp_path):
        """A delivery with two generated instances and a build directory that explains them."""
        build = tmp_path / 'shell_build' / 'abc123' / 'infinigen_build'
        build.mkdir(parents=True)
        (build / TOOL.BUNDLE_FILENAME).write_text(json.dumps({
            'gaps': ['monitor', 'shelf', 'shelf_trinkets']}))
        (build / TOOL.INJECTION_FILENAME).write_text(json.dumps({'rejection_pool': {
            'candidates': [
                {'name': 'MonitorFactory(1).spawn_asset(2)', 'category': 'monitor',
                 'outcome': 'kept', 'reasons': []},
                {'name': 'MonitorFactory(3).spawn_asset(4)', 'category': 'monitor',
                 'outcome': 'kept', 'reasons': []},
                {'name': 'CellShelfFactory(5).spawn_asset(6)', 'category': 'shelf',
                 'outcome': 'rejected', 'reasons': ['protected-overlap(0.02m2)']},
            ]}}))

        class _Run:
            run_dir = tmp_path
            spec = None
            scene = None
            delivery = None
            support_top_z_m = 0.75
            build_dir = build

        run = _Run()
        run.spec = spec
        run.delivery = {
            'scene_id': 'synthetic',
            'background': {'gaps': ['monitor', 'shelf', 'shelf_trinkets'],
                           'assembly': {'status': 'assembled'}},
            'assets_used': [
                {'entity_id': 'rug', 'provider': 'hssd', 'asset_id': 'a',
                 'dimensions_m': [2.0, 1.5, 0.02]},
                {'entity_id': 'generated_monitor_1', 'provider': 'generated',
                 'asset_id': 'm1', 'dimensions_m': [0.6, 0.2, 0.4]},
                {'entity_id': 'generated_monitor_2', 'provider': 'generated',
                 'asset_id': 'm2', 'dimensions_m': [0.6, 0.2, 0.4]},
            ],
            'placement': {'instances': {
                'rug': {'position_m': [1.0, 1.0, 0.006], 'yaw_rad': 0.0},
                'generated_monitor_1': {'position_m': [2.0, 1.0, 0.0], 'yaw_rad': 0.0},
                'generated_monitor_2': {'position_m': [2.4, 1.0, 0.0], 'yaw_rad': 0.0}}},
        }
        run.scene = {'instances': [
            {'entity_id': 'rug'}, {'entity_id': 'generated_monitor_1'},
            {'entity_id': 'generated_monitor_2'}]}
        prepared = tmp_path / 'prepared_assets'
        for asset_id in ('m1', 'm2'):
            directory = prepared / asset_id
            directory.mkdir(parents=True)
            for name in (f'{asset_id}.obj', f'{asset_id}_collision.obj'):
                (directory / name).write_text('')
        return run

    def test_only_the_generated_entity_ids_are_the_generated_rows(self, run) -> None:
        assert [row['entity_id'] for row in TOOL.generated_rows(run)] == [
            'generated_monitor_1', 'generated_monitor_2']
        assert [row['category'] for row in TOOL.generated_rows(run)] == ['monitor', 'monitor']

    def test_the_declaration_is_read_from_the_build_and_compared_with_the_block(self, run) -> None:
        gaps, source = TOOL.declared_gaps(run)
        assert gaps == ('monitor', 'shelf', 'shelf_trinkets')
        assert 'reservations.json' in source
        # A delivery block that declared something else is a fixture passing itself, and the audit
        # refuses to average the two: it names the disagreement instead.
        run.delivery['background'] = {'gaps': ['monitor']}
        with pytest.raises(TOOL.AuditError, match='BACKGROUND_GAP_DECLARATION_MISMATCH'):
            TOOL.declared_gaps(run)

    def test_a_gap_is_explained_by_what_the_pool_did_with_it(self, run) -> None:
        reasons = TOOL.gap_reasons(run, ('monitor', 'shelf', 'shelf_trinkets'))
        # `monitor` is filled, so it is not a gap any more and carries no reason.
        assert 'monitor' not in reasons
        assert reasons['shelf'] == 'proposed-but-rejected'
        assert reasons['shelf_trinkets'] == 'not-proposed-by-this-run'
        # A build with no ledger can only say the room does not contain one -- and it says that
        # rather than nothing, which is the difference between a named reason and a shrug.
        run.build_dir = None
        assert TOOL.gap_reasons(run, ('mirror_frame',)) == {'mirror_frame': 'not-in-the-room'}

    def _manifest(self, run, **changes):
        manifest = BackgroundManifest(
            scene_id='synthetic', mode='generated', entities=retrieved_ownership(run.spec),
            stages=tuple(StageRecord(stage=stage, state='executed', evidence=EVIDENCE_LOG)
                         for stage in STAGES),
            required_entity_ids=core_entity_ids(run.spec),
            gaps=('monitor', 'shelf', 'shelf_trinkets'),
            gap_reasons={'shelf': 'proposed-but-rejected',
                         'shelf_trinkets': 'not-proposed-by-this-run'})
        return manifest.with_changes(**changes)

    def test_aligned_gaps_are_not_a_failure(self, run) -> None:
        rows = TOOL.generated_rows(run)
        entities = tuple(OwnedEntity(entity_id=row['entity_id'], owner='generated',
                                     scope='periphery', stage=TOOL.REALISING_STAGE)
                         for row in rows)
        assert TOOL.alignment_gaps(self._manifest(run, entities=entities)) == []

    def test_a_generated_category_the_room_never_declared_is_named(self, run) -> None:
        entities = (OwnedEntity(entity_id='generated_desk_1', owner='generated',
                                scope='periphery', stage=TOOL.REALISING_STAGE),)
        gaps = TOOL.alignment_gaps(self._manifest(run, entities=entities))
        assert any(gap.startswith('BACKGROUND_GENERATED_CATEGORY_NOT_A_GAP(generated_desk_1,desk)')
                   for gap in gaps)

    def test_a_gap_with_no_fill_and_no_reason_is_named(self, run) -> None:
        gaps = TOOL.alignment_gaps(self._manifest(run, gap_reasons={}))
        assert any(gap == 'BACKGROUND_GAP_UNFILLED(monitor,no-fill-and-no-reason-recorded)'
                   for gap in gaps)
        # A reason of a kind the contract does not know is reported with the text, not hidden: a
        # reader has to be able to see what was claimed.
        unknown = TOOL.alignment_gaps(self._manifest(run, gap_reasons={
            'monitor': 'because', 'shelf': 'proposed-but-rejected',
            'shelf_trinkets': 'not-proposed-by-this-run'}))
        assert any(gap.startswith('BACKGROUND_GAP_UNFILLED(monitor,because)') for gap in unknown)

    def test_a_delivery_that_declares_no_gaps_makes_no_alignment_claim(self, run) -> None:
        # The rule is about the declaration, so a room that declared nothing cannot violate it --
        # and saying so is what keeps "no gaps" from reading as "aligned".
        assert TOOL.alignment_gaps(self._manifest(run, gaps=())) == []

    def test_every_way_a_delivery_can_claim_an_object_it_does_not_carry(self, run) -> None:
        assert TOOL.assembly_gaps(run) == []
        # The inventory keeps the row and the scene loses the instance: the manifest claims an
        # object nothing places, which is the direction that reads as a scene that has it.
        dropped_instance = replace_run(run, scene={'instances': [
            item for item in run.scene['instances']
            if item['entity_id'] != 'generated_monitor_1']})
        assert TOOL.assembly_gaps(dropped_instance) == [
            'BACKGROUND_ASSEMBLY_MISMATCH(generated_monitor_1)']
        # And the reverse: the instance is in the scene, the inventory never mentions it.
        dropped_row = {**run.delivery}
        dropped_row['assets_used'] = [asset for asset in run.delivery['assets_used']
                                      if asset['entity_id'] != 'generated_monitor_2']
        assert TOOL.assembly_gaps(replace_run(run, delivery=dropped_row)) == [
            'BACKGROUND_INSTANCE_UNRECORDED(generated_monitor_2)']
        elsewhere = replace_run(run, run_dir=run.run_dir / 'not-a-delivery')
        assert TOOL.assembly_gaps(elsewhere) == [
            'BACKGROUND_GENERATED_ASSET_MISSING(generated_monitor_1,m1.obj)',
            'BACKGROUND_GENERATED_ASSET_MISSING(generated_monitor_1,m1_collision.obj)',
            'BACKGROUND_GENERATED_ASSET_MISSING(generated_monitor_2,m2.obj)',
            'BACKGROUND_GENERATED_ASSET_MISSING(generated_monitor_2,m2_collision.obj)']

    def test_the_delivery_negatives_are_caught_and_leave_the_evidence_alone(self, run) -> None:
        manifest = self._manifest(run)
        before = (dict(run.delivery), dict(run.scene))
        rows = TOOL.run_negatives(run, manifest)
        names = [row['name'] for row in rows]
        for expected in ('assembly_row_without_instance', 'instance_without_assembly_row',
                         'generated_asset_not_on_disk'):
            assert expected in names
        for row in rows:
            assert row['caught'], f"{row['name']} was not caught: {row['observed']}"
        assert (run.delivery, run.scene) == before


def replace_run(run, **changes):
    """A copy of the synthetic run with one field changed, so a negative cannot edit the original.

    Class attributes are carried too: the fixture declares `run_dir` and `build_dir` on the fake
    class and the rest on the instance, and a copy that dropped either would look like a delivery
    whose files are missing -- a different failure than the one under test.
    """
    values = {**{name: value for name, value in vars(type(run)).items()
                 if not name.startswith('__')},
              **vars(run), **changes}
    return type('_Run', (), values)()


class TestTheMatrixEntryPoint:
    """The acceptance entry point: a case with no delivery is `not_run` with a reason, never a pass."""

    def test_a_case_whose_delivery_is_absent_is_reported_not_run(self, tmp_path) -> None:
        fixture = tmp_path / 'matrix.json'
        fixture.write_text(json.dumps({
            'schema_version': TOOL.MATRIX_SCHEMA_VERSION,
            'cases': [{'id': 'bedroom_37', 'delivery': 'outputs/does_not_exist'}],
            'negatives': ['protected_intrusion']}))
        out = tmp_path / 'report'
        assert TOOL.run_matrix(fixture, out) == 1
        report = json.loads((out / 'report.json').read_text())
        assert report['matrix_run'] is True
        assert report['cases_ran'] == 0 and report['cases_not_run'] == ['bedroom_37']
        assert report['spec_accepted'] is False
        assert report['cases'][0]['not_run_reason']

    def test_a_fixture_of_another_schema_is_refused_by_name(self, tmp_path) -> None:
        fixture = tmp_path / 'matrix.json'
        fixture.write_text(json.dumps({'schema_version': 'pvscene.background-matrix/v0',
                                       'cases': []}))
        with pytest.raises(TOOL.AuditError, match='BACKGROUND_MATRIX_SCHEMA_UNSUPPORTED'):
            TOOL.run_matrix(fixture, tmp_path / 'report')


class TestTheAssemblysOwnAccounting:
    """The two reasons only a delivery can know: refused here, or never delivered at all."""

    @pytest.fixture()
    def run(self, spec, tmp_path):
        build = tmp_path / 'shell_build' / 'abc123' / 'infinigen_build'
        build.mkdir(parents=True)
        (build / TOOL.BUNDLE_FILENAME).write_text(json.dumps({'gaps': ['book_stack', 'monitor']}))
        (build / TOOL.INJECTION_FILENAME).write_text(json.dumps({'rejection_pool': {
            'candidates': [
                {'name': 'BookStackFactory(1).spawn_asset(2)', 'category': 'book_stack',
                 'outcome': 'kept', 'reasons': []}]}}))

        class _Run:
            run_dir = tmp_path
            spec = None
            scene = None
            delivery = None
            support_top_z_m = 0.75
            build_dir = build

        run = _Run()
        run.spec = spec
        run.scene = {'instances': []}
        run.delivery = {'scene_id': 'synthetic', 'assets_used': [],
                        'background': {'generation': True,
                                       'gaps': ['book_stack', 'monitor'],
                                       'assembly': {'status': 'all-refused-at-assembly',
                                                    'refused_at_assembly': [
                                                        {'category': 'book_stack',
                                                         'reason': 'retrieved-overlap(0.0301m2)',
                                                         'generated_name': 'BookStackFactory(1)'
                                                                           '.spawn_asset(2)'}]}}}
        return run

    def test_a_gap_the_pool_kept_and_the_assembly_refused_says_so(self, run) -> None:
        reasons = TOOL.gap_reasons(run, ('book_stack', 'monitor'))
        # `book_stack` was kept by the pool and refused by the delivery; `monitor` was never
        # proposed. Both are named, and neither is "no reason recorded".
        assert reasons == {'book_stack': 'refused-at-assembly',
                           'monitor': 'not-proposed-by-this-run'}
        assert set(reasons.values()) <= set(TOOL.GAP_UNFILLED_KINDS)

    def test_a_kept_gap_the_delivery_simply_does_not_carry_says_so(self, run) -> None:
        run.delivery['background']['assembly'] = {'status': 'not_run', 'reason': 'x'}
        assert TOOL.gap_reasons(run, ('book_stack',)) == {'book_stack': 'kept-but-not-delivered'}

    def test_a_delivery_that_asked_for_generation_and_carries_none_is_named(self, run) -> None:
        # Every kept object was refused at assembly: carrying nothing is the recorded outcome, not
        # a defect, so the delivery is clean.
        assert run.delivery['background']['assembly']['status'] == 'all-refused-at-assembly'
        assert TOOL.assembly_gaps(run) == []
        # The assembly did not run, and the ledger says objects were kept: they are in no delivery.
        run.delivery['background']['assembly'] = {'status': 'not_run', 'reason': 'the shell failed'}
        gaps = TOOL.assembly_gaps(run)
        assert gaps == ['BACKGROUND_ASSEMBLY_MISSING(1 kept in the pool ledger, none delivered, '
                        'none refused)']
        # With no ledger (nothing kept) the same delivery is innocent: generation ran and the
        # build kept nothing, which is a result and not a defect.
        (run.build_dir / TOOL.INJECTION_FILENAME).unlink()
        assert TOOL.assembly_gaps(run) == []

    def test_the_alignment_rule_judges_a_delivery_that_asked_even_with_no_instances(self, run) -> None:
        manifest = BackgroundManifest(
            scene_id='synthetic', mode='generated', entities=retrieved_ownership(run.spec),
            stages=tuple(StageRecord(stage=stage, state='executed', evidence=EVIDENCE_LOG)
                         for stage in STAGES),
            required_entity_ids=core_entity_ids(run.spec),
            gaps=('book_stack',), gap_reasons={'book_stack': 'refused-at-assembly'})
        assert TOOL.alignment_gaps(manifest) == []
