# SPDX-License-Identifier: Apache-2.0
"""SceneSpec parsing and the protected goal set.

The central rule: `initial_relations` (true when generated) and `goal_predicates` (true
after the robot acts) must not be conflated. A goal that already holds at generation time
cannot demonstrate the task was performed, so it is rejected.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.scene_spec import (
    EntityRole,
    Mobility,
    PhysicsType,
    Predicate,
    Range1D,
    SceneSpec,
    SpecError,
)

MINIMAL = {
    "schema_version": "pvscene.scene-spec/v3",
    "scene_id": "office_pick_cup_m1_001",
    "seed": 11,
    "prompt": "warm wood office with a cup on the desk",
    "room": {"size_bounds_m": [[4, 5], [3, 4], [2.6, 3.0]]},
    "style": {"theme": "warm_minimal", "palette": ["wood", "off_white"], "density": "medium"},
    "requested_backends": ["isaac"],
    "requested_readiness": "functional_scene_ready",
    "entities": [
        {"id": "desk", "category": "desk", "physics_type": "rigid", "mobility": "fixed",
         "hard_required": True},
        {"id": "cup", "category": "cup", "physics_type": "rigid", "mobility": "dynamic",
         "hard_required": True, "role": "task_target"},
    ],
    "initial_relations": [
        {"id": "r1", "subject": "cup", "predicate": "ON", "object": "desk", "hard": True}
    ],
    "task": {
        "action": "pick_place",
        "robot_profile": "franka_panda",
        "targets": ["cup"],
        "goal_predicates": [],
    },
}


def spec_with(**overrides: object) -> SceneSpec:
    return SceneSpec.from_dict({**MINIMAL, **overrides})


class TestParsing:
    def test_parses_the_minimal_spec(self) -> None:
        spec = spec_with()
        assert spec.scene_id == "office_pick_cup_m1_001"
        assert spec.seed == 11
        assert [entity.id for entity in spec.entities] == ["desk", "cup"]
        assert spec.task.action == "pick_place"

    def test_round_trips_through_dict(self) -> None:
        assert spec_with().to_dict() == spec_with().to_dict()
        reparsed = SceneSpec.from_dict(spec_with().to_dict())
        assert reparsed.to_dict() == spec_with().to_dict()

    def test_entity_asset_requirements_round_trip(self) -> None:
        entities = [
            {
                **MINIMAL["entities"][0],
                "material_tokens": ["warm_wood", "matte"],
                "requires_support_surface": True,
            },
            MINIMAL["entities"][1],
        ]
        spec = spec_with(entities=entities)
        desk = spec.entity("desk")
        assert desk.material_tokens == ("warm_wood", "matte")
        assert desk.requires_support_surface is True
        assert SceneSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()

    def test_omitted_asset_requirements_do_not_change_the_serialised_shape(self) -> None:
        desk = spec_with().entity("desk")
        assert desk.material_tokens == ()
        assert desk.requires_support_surface is False
        assert "material_tokens" not in desk.to_dict()
        assert "requires_support_surface" not in desk.to_dict()

    @pytest.mark.parametrize("value", ["wood", 1, None])
    def test_material_tokens_must_be_a_list(self, value: object) -> None:
        entities = [{**MINIMAL["entities"][0], "material_tokens": value}]
        with pytest.raises(SpecError, match="material_tokens must be a list"):
            spec_with(entities=entities)

    def test_requires_support_surface_must_be_boolean(self) -> None:
        entities = [{**MINIMAL["entities"][0], "requires_support_surface": "yes"}]
        with pytest.raises(SpecError, match="requires_support_surface must be a boolean"):
            spec_with(entities=entities)

    def test_rejects_unknown_schema_version(self) -> None:
        with pytest.raises(SpecError, match="schema_version"):
            spec_with(schema_version="pvscene.scene-spec/v2")

    def test_rejects_unknown_keys(self) -> None:
        with pytest.raises(SpecError, match="unknown keys"):
            spec_with(unexpected_field=1)

    def test_removed_room_kind_is_rejected_instead_of_silently_ignored(self) -> None:
        room = {**MINIMAL["room"], "kind": "office"}
        with pytest.raises(SpecError, match="room has unknown keys.*kind"):
            spec_with(room=room)

    def test_rejects_duplicate_entity_ids(self) -> None:
        entities = [*MINIMAL["entities"], MINIMAL["entities"][1]]
        with pytest.raises(SpecError, match="duplicate entity ids"):
            spec_with(entities=entities)

    def test_rejects_relation_to_unknown_entity(self) -> None:
        relations = [{"id": "r1", "subject": "cup", "predicate": "ON", "object": "shelf"}]
        with pytest.raises(SpecError, match="unknown entity"):
            spec_with(initial_relations=relations)

    def test_rejects_task_target_that_does_not_exist(self) -> None:
        task = {**MINIMAL["task"], "targets": ["teapot"]}
        with pytest.raises(SpecError, match="task.targets"):
            spec_with(task=task)

    def test_rejects_duplicate_backends(self) -> None:
        with pytest.raises(SpecError, match="duplicates"):
            spec_with(requested_backends=["isaac", "isaac"])

    def test_rejects_unknown_physics_type(self) -> None:
        entities = [{**MINIMAL["entities"][0], "physics_type": "plasma"}]
        with pytest.raises(SpecError, match="physics_type"):
            spec_with(entities=entities)

    def test_rejects_inverted_size_bounds(self) -> None:
        with pytest.raises(SpecError, match="exceeds maximum"):
            Range1D(2.0, 1.0)


class TestGoalSeparation:
    def test_goal_matching_an_initial_relation_is_rejected(self) -> None:
        """The cup cannot start where the goal says it should end up.

        Otherwise the pick-and-place task is "satisfied" before anything moves, which is the
        failure mode architecture §5.1 names explicitly.
        """
        task = {
            **MINIMAL["task"],
            "goal_predicates": [{"subject": "cup", "predicate": "ON", "object": "desk"}],
        }
        with pytest.raises(SpecError, match="also an initial relation"):
            spec_with(task=task)

    def test_a_genuine_goal_is_accepted(self) -> None:
        task = {
            **MINIMAL["task"],
            "goal_predicates": [{"subject": "cup", "predicate": "INSIDE", "object": "cupboard"}],
        }
        entities = [*MINIMAL["entities"],
                    {"id": "cupboard", "category": "cupboard", "physics_type": "rigid",
                     "mobility": "fixed", "hard_required": False}]
        spec = spec_with(task=task, entities=entities)
        assert len(spec.task.goal_predicates) == 1
        assert spec.task.goal_predicates[0].predicate is Predicate.INSIDE

    def test_goal_predicates_are_not_part_of_the_initial_relations(self) -> None:
        task = {
            **MINIMAL["task"],
            "goal_predicates": [{"subject": "cup", "predicate": "INSIDE", "object": "cupboard"}],
        }
        entities = [*MINIMAL["entities"],
                    {"id": "cupboard", "category": "cupboard", "physics_type": "rigid",
                     "mobility": "fixed", "hard_required": False}]
        spec = spec_with(task=task, entities=entities)
        assert [relation.predicate for relation in spec.initial_relations] == [Predicate.ON]


class TestProtectedGoals:
    def test_hard_entities_and_relations_are_listed(self) -> None:
        summary = spec_with().protected_summary()
        assert summary["hard_entities"] == ["cup", "desk"]
        assert summary["task_targets"] == ["cup"]
        assert summary["hard_relations"] == ["cup:ON:desk"]
        assert summary["task_action"] == "pick_place"

    def test_optional_entities_are_not_protected(self) -> None:
        entities = [*MINIMAL["entities"],
                    {"id": "poster", "category": "poster", "physics_type": "rigid",
                     "mobility": "fixed", "hard_required": False}]
        summary = spec_with(entities=entities).protected_summary()
        assert "poster" not in summary["hard_entities"]

    def test_hard_required_defaults_to_true(self) -> None:
        # Defaulting to optional would make deleting a failing object the cheapest repair,
        # which the architecture forbids for hard goals.
        entities = [{"id": "vase", "category": "vase", "physics_type": "rigid", "mobility": "fixed"}]
        spec = spec_with(entities=entities, initial_relations=[], task={**MINIMAL["task"], "targets": ["vase"]})
        assert spec.hard_entities[0].id == "vase"

    def test_summary_is_stable_across_key_order(self) -> None:
        assert spec_with().protected_summary() == spec_with().protected_summary()

    def test_summary_captures_goal_predicates_too(self) -> None:
        # A patch must preserve the post-execution goals as well as the initial ones.
        task = {
            **MINIMAL["task"],
            "goal_predicates": [{"subject": "cup", "predicate": "INSIDE", "object": "cupboard"}],
        }
        entities = [*MINIMAL["entities"],
                    {"id": "cupboard", "category": "cupboard", "physics_type": "rigid",
                     "mobility": "fixed", "hard_required": False}]
        summary = spec_with(task=task, entities=entities).protected_summary()
        assert summary["goal_predicates"] == ["cup:INSIDE:cupboard"]


class TestEntitySemantics:
    def test_role_defaults_to_context(self) -> None:
        entities = [{"id": "vase", "category": "vase", "physics_type": "rigid", "mobility": "fixed"}]
        spec = spec_with(entities=entities, initial_relations=[], task={**MINIMAL["task"], "targets": ["vase"]})
        assert spec.entities[0].role is EntityRole.CONTEXT
        assert not spec.entities[0].is_task_target

    def test_task_target_role_round_trips(self) -> None:
        assert spec_with().entity("cup").role is EntityRole.TASK_TARGET

    def test_rejects_non_identifier_entity_id(self) -> None:
        entities = [{**MINIMAL["entities"][0], "id": "not an id"}]
        with pytest.raises(SpecError, match="identifier"):
            spec_with(entities=entities)

    def test_joint_task_needs_an_articulated_entity(self) -> None:
        task = {
            **MINIMAL["task"],
            "joint_task": {"joint_id": "door_hinge", "initial_q": 0.0, "goal_q": 1.4},
        }
        with pytest.raises(SpecError, match="no articulated entity"):
            spec_with(task=task)

    def test_joint_task_keeps_initial_goal_and_schedule_distinct(self) -> None:
        entities = [*MINIMAL["entities"],
                    {"id": "cabinet", "category": "cabinet", "physics_type": "articulated",
                     "mobility": "fixed", "hard_required": True}]
        task = {
            **MINIMAL["task"],
            "joint_task": {
                "joint_id": "cabinet_door",
                "initial_q": 0.0,
                "goal_q": 1.4,
                "test_q_schedule": [0.0, 0.7, 1.4],
            },
        }
        joint = spec_with(task=task, entities=entities).task.joint_task
        assert joint is not None
        assert joint.initial_q == 0.0
        assert joint.goal_q == 1.4
        assert joint.test_q_schedule == (0.0, 0.7, 1.4)
