# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import json
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.scene_spec import (
    SceneScope,
    SceneSpec,
    relation_closure_ids,
    scope_of,
)
from rfw_w.scenes.solvers.furnishing_plan import plan_from_spec
from rfw_w.scenes.solvers.randomization import randomize_plan, task_invariant_ids
from rfw_w.scenes.solvers.scene_ir import requested_entity_coverage

ROOT = Path(__file__).resolve().parents[3]


def office() -> SceneSpec:
    return SceneSpec.from_dict(json.loads(
        (data_path('examples/office_pick_cup/task.json')).read_text()
    ))


def test_same_seed_is_exactly_replayable() -> None:
    spec = office()
    first = randomize_plan(spec, plan_from_spec(spec), variant_seed=41, enabled=True)
    second = randomize_plan(spec, plan_from_spec(spec), variant_seed=41, enabled=True)
    assert first[0].to_dict() == second[0].to_dict()
    assert first[1].to_dict() == second[1].to_dict()


def test_different_seeds_change_recorded_domain() -> None:
    spec = office()
    first_plan, first_record = randomize_plan(spec, plan_from_spec(spec), variant_seed=41, enabled=True)
    second_plan, second_record = randomize_plan(spec, plan_from_spec(spec), variant_seed=42, enabled=True)
    assert first_record.room_seed != second_record.room_seed
    assert first_plan.appearance != second_plan.appearance


def test_relation_closure_is_never_an_asset_randomization_target() -> None:
    spec = office()
    _plan, record = randomize_plan(spec, plan_from_spec(spec), variant_seed=7, enabled=True)
    invariant_entities, invariant_relations = task_invariant_ids(spec)
    assert record.invariant_entity_ids == invariant_entities
    assert record.invariant_relation_ids == invariant_relations
    assert not set(invariant_entities) & {choice.entity_id for choice in record.entity_choices}
    # cup, desk, notebook, pen_holder and chair are the full relation closure for this task.
    assert set(invariant_entities) == {"cup", "desk", "notebook", "pen_holder", "chair"}


def test_disabled_randomization_is_still_explicit_and_recorded() -> None:
    spec = office()
    base = plan_from_spec(spec)
    plan, record = randomize_plan(spec, base, variant_seed=spec.seed, enabled=False)
    assert plan == base
    assert record.enabled is False
    assert record.entity_choices == ()
    assert record.room_seed is None


# --- S0 characterisation: the three sets that are NOT the same set -------------------------------
#
# The core-first plan originally proposed collapsing these into one. Measured, they are three
# different sets, and replacing the relation closure with the narrower role-based core would shrink
# the randomiser's protected set and shrink `semantics.task_structure_covered`'s required set --
# the second of which is a loosened acceptance gate. These tests pin the difference so the S1
# de-duplication cannot quietly move it.

EXPERIMENTS = ("office_pick_cup", "kitchen_fetch_jar", "bedroom_pick_alarm_clock")

# A: the relation closure, including soft-relation endpoints. Pinned per experiment.
RELATION_CLOSURE = {
    "office_pick_cup": {"cup", "desk", "notebook", "pen_holder", "chair"},
    "kitchen_fetch_jar": {"jar", "island", "bowl", "stool"},
    "bedroom_pick_alarm_clock": {"alarm_clock", "nightstand", "bed", "book"},
}

# CORE: role says task, not room. Strictly narrower.
CORE = {
    "office_pick_cup": {"cup", "desk"},
    "kitchen_fetch_jar": {"jar", "island"},
    "bedroom_pick_alarm_clock": {"alarm_clock", "nightstand"},
}


def spec_for(name: str) -> SceneSpec:
    return SceneSpec.from_dict(json.loads((data_path('examples') / name / 'task.json').read_text()))


def closure_of(spec: SceneSpec) -> set[str]:
    """A, as the randomiser defines it (`randomization.py:18-22`)."""
    return set(task_invariant_ids(spec)[0])


def coverage_closure_of(spec: SceneSpec) -> set[str]:
    """A, as `scene_ir.py:126-128` recomputes it.

    Nothing is placed, so the missing-task set is the whole required set.
    """
    _fraction, missing_task, _total, _missing_all = requested_entity_coverage(
        spec, {"instances": ()}
    )
    return set(missing_task)


def core_of(spec: SceneSpec) -> set[str]:
    """CORE, from the production scope classifier introduced in S1."""
    return {entity.id for entity in spec.entities if scope_of(entity) is SceneScope.CORE}


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_shared_relation_closure_matches_the_legacy_randomization_name(name: str) -> None:
    spec = spec_for(name)
    assert relation_closure_ids(spec) == task_invariant_ids(spec)


def test_a_soft_relation_does_not_promote_periphery_into_the_core() -> None:
    spec = office()
    chair = spec.entity("chair")
    assert "chair" in relation_closure_ids(spec)[0]
    assert scope_of(chair) is SceneScope.PERIPHERY


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_two_copies_of_the_relation_closure_agree(name: str) -> None:
    """The only thing S1 may de-duplicate: these two implementations are word-for-word identical.

    If this ever fails, the two copies have already diverged and the de-duplication is a behaviour
    change wearing a refactor's clothes.
    """
    spec = spec_for(name)
    assert closure_of(spec) == coverage_closure_of(spec)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_relation_closure_is_pinned(name: str) -> None:
    assert closure_of(spec_for(name)) == RELATION_CLOSURE[name]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_core_is_pinned_and_strictly_narrower(name: str) -> None:
    """CORE ⊊ A, on every experiment. The gap is real and must stay visible."""
    spec = spec_for(name)
    core, closure = core_of(spec), closure_of(spec)
    assert core == CORE[name]
    assert core < closure, "CORE is not strictly narrower; the two sets have collapsed"
    # The entities in the gap are periphery named by a *soft* relation -- droppable per the spec,
    # yet protected by both the randomiser and the coverage gate today.
    assert closure - core == RELATION_CLOSURE[name] - CORE[name]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_role_and_hard_required_agree_on_these_specs(name: str) -> None:
    """Why the closure was easy to mistake for the core: on these three they *do* coincide with
    `hard_required`. The divergence is A vs both of them, not role vs hard_required."""
    spec = spec_for(name)
    assert core_of(spec) == {entity.id for entity in spec.hard_entities}
