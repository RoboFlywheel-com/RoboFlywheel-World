# SPDX-License-Identifier: Apache-2.0
"""Deriving the plan from the spec, and the gaps it refuses to guess at.

Two things are under test, and the second is the one that matters.

**Equivalence.** `plan_from_spec` replaces a nine-tuple table that had been the furniture set since
the pipeline was written. The guard is a frozen golden captured from that table *before* it was
deleted (`tests/fixtures/m1_office_plan_golden.json`): for the office spec, every piece the table
named must come back with every field identical. Without it the refactor is unfalsifiable, because
"the office still builds" is also true of a derivation that silently drops the rug.

**No silent defaults.** Every entity the vocabulary does not know, every support it cannot resolve,
every relation naming something the spec never declared, must produce a *named* gap or a refusal.
The failure this replaces was not an exception -- it was a table that quietly disagreed with the
spec, and the one field (`spec.entities`) that would have caught it being read by nobody.
"""

from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import json
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.solvers.furnish import Orientation, PlacementKind
from rfw_w.scenes.solvers.furnishing_plan import (
    CATEGORY_PROFILES,
    PlanDerivationError,
    plan_from_spec,
    profile_for,
    resolve_support,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = REPO_ROOT / "tests/scenes" / "fixtures" / "m1_office_plan_golden.json"
OFFICE_SPEC = data_path('examples') / 'office_pick_cup' / 'task.json'

ROOM = {"size_bounds_m": [[4.0, 5.0], [3.0, 4.0], [2.6, 3.0]]}
STYLE = {"theme": "t", "palette": ["oak"], "density": "medium"}


def entity(eid, category, role, *, hard=False, bounds=None, mobility="fixed",
           physics="rigid", quantity=1):
    doc = {"id": eid, "category": category, "physics_type": physics, "mobility": mobility,
           "hard_required": hard, "role": role}
    if bounds is not None:
        doc["size_bounds_m"] = bounds
    if quantity != 1:
        doc["quantity"] = quantity
    return doc


BOX = [[0.3, 0.5], [0.3, 0.5], [0.3, 0.5]]


def spec_of(entities, relations, targets, *, room=None):
    return SceneSpec.from_dict({
        "schema_version": "pvscene.scene-spec/v3",
        "scene_id": "synthetic",
        "seed": 3,
        "prompt": "synthetic",
        "room": room or ROOM,
        "style": STYLE,
        "requested_backends": ["isaac"],
        "requested_readiness": "functional_scene_ready",
        "entities": entities,
        "initial_relations": relations,
        "task": {"action": "pick_place", "robot_profile": "franka_panda", "targets": list(targets)},
    })


def minimal(*, support_category="desk", support_role="support", target_category="cup",
            extra_entities=(), extra_relations=(), targets=None, support_bounds=BOX):
    """A spec with one support and one target, plus whatever the test adds."""
    return spec_of(
        [entity("table", support_category, support_role, hard=True, bounds=support_bounds),
         entity("thing", target_category, "task_target", hard=True, mobility="dynamic", bounds=BOX),
         *extra_entities],
        [{"id": "r1", "subject": "thing", "predicate": "ON", "object": "table", "hard": True},
         *extra_relations],
        targets or ["thing"],
    )


@pytest.fixture(scope="module")
def golden() -> dict:
    """The plan the deleted table produced, frozen before it was deleted."""
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def derived() -> dict:
    """The plan the derivation produces for the same spec."""
    spec = SceneSpec.from_dict(json.loads(OFFICE_SPEC.read_text()))
    return plan_from_spec(spec).to_dict()


@pytest.fixture(scope="module")
def hssd_index() -> dict:
    """The real HSSD synset index, skipped when it is not configured."""
    import os

    path = os.environ.get("PVSCENE_HSSD_SYNSET_INDEX")
    if not path or not Path(path).is_file():
        pytest.skip("PVSCENE_HSSD_SYNSET_INDEX is not configured")
    return json.loads(Path(path).read_text())


class TestOfficeEquivalence:
    """The derivation reproduces the table it replaces, for the spec that table was written for."""

    def test_every_piece_is_present_with_every_field_identical(self, golden, derived) -> None:
        """Compared by entity id, not by position.

        The order differs, and deliberately: `FURNISHING_ROWS` listed the rug last while the spec
        declares it before the wall art and the pendant, and the spec is the source now. The three
        pieces that move are the rug, the picture and the pendant -- all three are non-floor-competing
        kinds (`FLOOR_COVERING`, `WALL_HUNG`, `CEILING`), so their order cannot change which floor a
        competitor gets. A positional comparison would be pinning an accident of the old table.
        `supports` is compared here too, and its provenance is different: the golden's furniture and
        surfaces rows were captured from the deleted table, while the desk's support row was written
        when the support stopped being retrieved. It says what the derivation produces now -- a form
        and a size, where the table had a synset and a candidate index -- and comparing it is what
        keeps that row from drifting unnoticed.
        The pen-holder synset was subsequently corrected to the catalogue's exact pen pot
        category after rendering showed the old organizer category selecting drawer boxes.
        """
        for section in ("furniture", "surfaces", "supports"):
            want = {row["entity_id"]: row for row in golden[section]}
            got = {row["entity_id"]: row for row in derived[section]}
            assert set(want) == set(got), f"{section} names different pieces"
            for name, row in want.items():
                assert got[name] == row, f"{section}.{name} drifted"

    def test_only_the_order_moved(self, golden, derived) -> None:
        """The claim above, stated as a test of its own so a reordering cannot hide a drift."""
        assert {r["entity_id"] for r in golden["furniture"]} == {
            r["entity_id"] for r in derived["furniture"]
        }
        assert [r["entity_id"] for r in derived["furniture"]] != [
            r["entity_id"] for r in golden["furniture"]
        ], "the order was expected to differ; if it no longer does, this test is stale, not wrong"

    def test_the_plan_constants_are_unchanged(self, golden, derived) -> None:
        for key in ("mount_height_m", "ceiling_clearance_m", "wall_hang_gap_m",
                    "wall_face_offset_m", "object_gap_m", "appearance"):
            assert derived[key] == golden[key], f"{key} drifted"

    def test_the_support_and_target_come_from_the_spec_not_from_a_literal(self, derived) -> None:
        assert derived["support_id"] == "desk"
        assert derived["target_id"] == "cup"
        assert derived["derivation_gaps"] == []

    def test_every_new_spec_derives_without_a_gap(self) -> None:
        """The three experiments are the point of the change; each must build a plan silently."""
        for task in sorted((data_path('examples')).glob("*/task.json")):
            spec = SceneSpec.from_dict(json.loads(task.read_text()))
            plan = plan_from_spec(spec)
            assert plan.derivation_gaps == (), f"{task.parent.name}: {plan.derivation_gaps}"
            assert plan.support_id in plan.entity_ids
            assert plan.target_id not in plan.entity_ids

    def test_room_specific_support_and_bed_placement_are_not_office_wall_defaults(self) -> None:
        specs = {
            task.parent.name: SceneSpec.from_dict(json.loads(task.read_text()))
            for task in sorted((data_path('examples')).glob("*/task.json"))
        }
        kitchen = plan_from_spec(specs["kitchen_fetch_jar"])
        # The island is a *support* row now (it is built, not retrieved), so its placement kind
        # lives there. The intent is unchanged: an island in the seating zone would be wall
        # furniture, and treating it as such made every kitchen candidate fail its own work zone.
        assert kitchen.support_for("island").kind is PlacementKind.ANYWHERE
        assert kitchen.furniture_for("island") is None
        bedroom = plan_from_spec(specs["bedroom_pick_alarm_clock"])
        assert bedroom.furniture_for("bed").zone_kind is None


class TestSupportAndTarget:
    def test_the_support_is_the_object_of_the_hard_on_relation(self) -> None:
        plan = plan_from_spec(minimal())
        assert plan.support_id == "table"
        assert plan.target_id == "thing"

    def test_a_soft_on_relation_still_names_the_support(self) -> None:
        spec = spec_of(
            [entity("table", "desk", "support", hard=True, bounds=BOX),
             entity("thing", "cup", "task_target", hard=True, mobility="dynamic", bounds=BOX)],
            [{"id": "r1", "subject": "thing", "predicate": "ON", "object": "table", "hard": False}],
            ["thing"],
        )
        assert plan_from_spec(spec).support_id == "table"

    def test_two_supports_for_one_target_are_refused(self) -> None:
        """Ambiguity is refused, not resolved by picking one: the two produce different scenes."""
        spec = spec_of(
            [entity("a", "desk", "support", hard=True, bounds=BOX),
             entity("b", "desk", "support", hard=True, bounds=BOX),
             entity("thing", "cup", "task_target", hard=True, mobility="dynamic", bounds=BOX)],
            [{"id": "r1", "subject": "thing", "predicate": "ON", "object": "a", "hard": True},
             {"id": "r2", "subject": "thing", "predicate": "ON", "object": "b", "hard": True}],
            ["thing"],
        )
        with pytest.raises(PlanDerivationError, match="SUPPORT_UNRESOLVED"):
            plan_from_spec(spec)

    def test_a_target_with_no_on_relation_falls_back_and_says_so(self) -> None:
        spec = spec_of(
            [entity("table", "desk", "support", hard=True, bounds=BOX),
             entity("thing", "cup", "task_target", hard=True, mobility="dynamic", bounds=BOX)],
            [],
            ["thing"],
        )
        plan = plan_from_spec(spec)
        assert plan.support_id == "table"
        assert any("TARGET_NOT_ON_SUPPORT(thing)" in gap for gap in plan.derivation_gaps)

    def test_a_target_with_no_support_at_all_is_refused(self) -> None:
        spec = spec_of(
            [entity("thing", "cup", "task_target", hard=True, mobility="dynamic", bounds=BOX)],
            [], ["thing"],
        )
        with pytest.raises(PlanDerivationError, match="SUPPORT_UNRESOLVED"):
            plan_from_spec(spec)

    def test_no_target_at_all_is_refused_by_the_contract_not_by_the_derivation(self) -> None:
        """The guarantee lives upstream, so the derivation has no branch for it.

        Three unreachable raises were removed from `furnishing_plan` once these tests showed the
        contract already refuses the inputs: `task.targets` must be non-empty, and every relation
        must reference a declared entity. A check that cannot fire is the defect this repository
        keeps finding, so the tests assert *where* the refusal happens rather than deleting the
        cases.
        """
        from rfw_w.scenes.contracts.scene_spec import SpecError

        with pytest.raises(SpecError, match="non-empty list of entity ids"):
            spec_of([entity("table", "desk", "support", hard=True, bounds=BOX)], [], [])

    def test_more_than_one_target_is_a_gap_and_the_first_is_used(self) -> None:
        spec = spec_of(
            [entity("table", "desk", "support", hard=True, bounds=BOX),
             entity("a", "cup", "task_target", hard=True, mobility="dynamic", bounds=BOX),
             entity("b", "jar", "task_target", hard=True, mobility="dynamic", bounds=BOX)],
            [{"id": "r1", "subject": "a", "predicate": "ON", "object": "table", "hard": True},
             {"id": "r2", "subject": "b", "predicate": "ON", "object": "table", "hard": True}],
            ["a", "b"],
        )
        plan = plan_from_spec(spec)
        assert plan.target_id == "a"
        assert any("MULTIPLE_TASK_TARGETS(2)" in gap for gap in plan.derivation_gaps)

    def test_a_support_the_spec_never_declares_is_refused_by_the_contract(self) -> None:
        """A relation onto a ghost cannot be written down, so the derivation never sees one."""
        from rfw_w.scenes.contracts.scene_spec import SpecError

        with pytest.raises(SpecError, match="references unknown entity 'ghost'"):
            spec_of(
                [entity("thing", "cup", "task_target", hard=True, mobility="dynamic", bounds=BOX)],
                [{"id": "r1", "subject": "thing", "predicate": "ON", "object": "ghost",
                  "hard": True}],
                ["thing"],
            )


class TestSurfaceObjects:
    def test_a_soft_on_relation_makes_a_surface_object(self) -> None:
        """Rule 3 beats rule 4: `notebook` and `chair` share a role, and only the relation differs."""
        spec = minimal(extra_entities=[entity("book", "notebook", "context", bounds=BOX)],
                       extra_relations=[{"id": "r2", "subject": "book", "predicate": "ON",
                                         "object": "table", "hard": False}])
        plan = plan_from_spec(spec)
        assert plan.surface_for("book") is not None
        assert plan.surface_for("book").support_entity == "table"
        assert plan.furniture_for("book") is None

    def test_a_context_entity_with_no_relation_stays_on_the_floor(self) -> None:
        spec = minimal(extra_entities=[entity("seat", "chair", "context", bounds=BOX)])
        plan = plan_from_spec(spec)
        assert plan.furniture_for("seat") is not None
        assert plan.surface_for("seat") is None

    def test_the_target_is_never_given_a_plan_row(self) -> None:
        """It is a SimReady asset; a surface row would claim an HSSD synset for it."""
        plan = plan_from_spec(minimal())
        assert plan.furniture_for("thing") is None
        assert plan.surface_for("thing") is None
        assert plan.entity_ids == ("table",)

    def test_a_relation_naming_an_undeclared_entity_is_refused_by_the_contract(self) -> None:
        from rfw_w.scenes.contracts.scene_spec import SpecError

        with pytest.raises(SpecError, match="references unknown entity 'ghost'"):
            minimal(extra_relations=[{"id": "r2", "subject": "ghost", "predicate": "ON",
                                      "object": "table", "hard": False}])


class TestNamedGaps:
    def test_an_optional_entity_with_an_unknown_category_is_dropped_and_named(self) -> None:
        spec = minimal(extra_entities=[entity("mystery", "unobtainium", "decoration", bounds=BOX)])
        plan = plan_from_spec(spec)
        assert plan.furniture_for("mystery") is None
        assert "CATEGORY_UNKNOWN(unobtainium)" in plan.derivation_gaps[0]

    def test_a_hard_entity_with_an_unknown_category_refuses_the_spec(self) -> None:
        """A delivery that silently omits what the task requires must not be produced at all."""
        spec = minimal(extra_entities=[
            entity("mystery", "unobtainium", "context", hard=True, bounds=BOX)])
        with pytest.raises(PlanDerivationError, match="CATEGORY_UNKNOWN\\(unobtainium\\)"):
            plan_from_spec(spec)

    def test_a_non_rigid_entity_is_dropped_by_physics_and_named(self) -> None:
        spec = minimal(extra_entities=[
            entity("drawer", "cabinet", "context", bounds=BOX, physics="articulated")])
        plan = plan_from_spec(spec)
        assert plan.furniture_for("drawer") is None
        assert any("PHYSICS_UNSUPPORTED(drawer,articulated)" in gap for gap in plan.derivation_gaps)

    def test_an_entity_asked_for_more_than_once_is_dropped_and_named(self) -> None:
        spec = minimal(extra_entities=[
            entity("stool", "stool", "context", bounds=BOX, quantity=3)])
        plan = plan_from_spec(spec)
        assert plan.furniture_for("stool") is None
        assert any("QUANTITY_UNSUPPORTED(stool,3)" in gap for gap in plan.derivation_gaps)

    def test_a_second_support_is_placed_but_recorded_as_unused(self) -> None:
        spec = minimal(extra_entities=[entity("other", "nightstand", "support", bounds=BOX)])
        plan = plan_from_spec(spec)
        assert plan.furniture_for("other") is not None
        assert any("SUPPORT_UNUSED(other)" in gap for gap in plan.derivation_gaps)

    def test_gaps_survive_into_the_serialised_plan(self) -> None:
        """A gap left out of `to_dict` is a gap a repair can change without moving the version."""
        spec = minimal(extra_entities=[entity("mystery", "unobtainium", "decoration", bounds=BOX)])
        rendered = plan_from_spec(spec).to_dict()
        assert "derivation_gaps" in rendered
        assert rendered["derivation_gaps"], "the gap did not reach the serialised plan"


class TestVocabulary:
    def test_an_unknown_category_has_no_profile_and_no_convention_fallback(self) -> None:
        """`f"{category}.n.01"` is not an HSSD key for four of the office's twelve categories."""
        assert profile_for("pen_holder") is not None
        assert profile_for("pen_holder.n.01") is None
        assert profile_for("") is None

    def test_the_four_categories_that_a_convention_would_get_wrong_are_explicit(self) -> None:
        for category, synset in (("pen_holder", "pen_holder"),
                                 ("side_table", "end_table.n.01"),
                                 ("plant", "potted_plant.n.01"),
                                 ("ceiling_light", "ceiling_lamp.n.01")):
            assert CATEGORY_PROFILES[category].synsets[0] == synset

    def test_faces_toward_needs_a_referent(self) -> None:
        from rfw_w.scenes.solvers.furnishing_plan import CategoryProfile

        with pytest.raises(PlanDerivationError, match="faces_support"):
            CategoryProfile(("chair.n.01",), PlacementKind.BESIDE, Orientation.FACES_TOWARD,
                            CATEGORY_PROFILES["chair"].zone)

    def test_a_profile_with_no_synset_is_refused(self) -> None:
        from rfw_w.scenes.solvers.furnishing_plan import CategoryProfile

        with pytest.raises(PlanDerivationError, match="at least one synset"):
            CategoryProfile((), PlacementKind.ANYWHERE, Orientation.SYMMETRIC,
                            CATEGORY_PROFILES["chair"].zone)

    def test_the_specs_only_use_categories_the_vocabulary_knows(self) -> None:
        for task in sorted((data_path('examples')).glob("*/task.json")):
            spec = SceneSpec.from_dict(json.loads(task.read_text()))
            unknown = [e.category for e in spec.entities if profile_for(e.category) is None]
            assert not unknown, f"{task.parent.name} uses {unknown}"


class TestRealLibrary:
    """Cross-checks against the real HSSD index, skipped when it is not configured."""

    def test_every_first_choice_synset_is_a_real_index_key(self, hssd_index) -> None:
        """The drift this stops: `required_categories_v1.json` already carries two dead keys."""
        missing = {
            category: profile.synsets[0]
            for category, profile in CATEGORY_PROFILES.items()
            if profile.synsets[0] not in hssd_index
        }
        assert not missing, f"not HSSD synset keys: {missing}"

    def test_no_profile_names_a_synset_with_no_assets(self, hssd_index) -> None:
        empty = {
            category: synset
            for category, profile in CATEGORY_PROFILES.items()
            for synset in profile.synsets
            if not hssd_index.get(synset)
        }
        assert not empty, f"keys with no assets behind them: {empty}"
