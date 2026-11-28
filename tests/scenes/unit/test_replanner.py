# SPDX-License-Identifier: Apache-2.0
"""The L5 replanner: what a patch may do, and what it is refused.

The design gives this role three prohibitions, and each is a way an L5 call becomes dangerous rather
than useful: it *"may not delete hard entities, change the task, or relax hard relations"*. So every
operation is checked against the plan and against the request's hard constraints, and a refused
operation is recorded with its reason -- a replan whose operations were all refused has to read as
"the model had nothing usable to say", not as "the model agreed".

The vocabulary is closed and the unimplemented design actions are named in their refusal, so a
reader can tell whether the vocabulary or the code is short.
"""

from __future__ import annotations

import json

import pytest

from rfw_w.scenes.agents.replanner import (
    IMPLEMENTED_ACTIONS,
    PatchAction,
    UNIMPLEMENTED_DESIGN_ACTIONS,
    application_requests,
    failure_code_for_patch,
    parse_patch,
    scene_summary,
)
from rfw_w.scenes.contracts.tool import FailureCode
from rfw_w.scenes.solvers.furnish import (
    FurnitureRequest,
    Orientation,
    PlacementKind,
    ScenePlan,
    SurfaceRequest,
)

VOCABULARY = ("cabinet.n.01", "chair.n.01", "desk.n.01", "notebook.n.01")


def plan(*, desk_required: bool = True) -> ScenePlan:
    return ScenePlan(
        furniture=(
            FurnitureRequest(
                entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
                orientation=Orientation.FRONT_TO_ROOM, hard_required=desk_required,
                optional=not desk_required,
            ),
            FurnitureRequest(
                entity_id="chair", synset="chair.n.01", kind=PlacementKind.BESIDE,
                orientation=Orientation.FACES_TOWARD, faces_entity="desk",
            ),
        ),
        surfaces=(SurfaceRequest(
            entity_id="notebook", synset="notebook.n.01", target_width_m=0.2, support_entity="desk"
        ),),
    )


def parse(operations) -> "object":
    return parse_patch(
        json.dumps({"diagnosis": "the chair blocks the passage", "operations": operations}),
        plan=plan(), synset_vocabulary=VOCABULARY,
    )


class TestAccepted:
    def test_a_valid_patch_survives_intact(self) -> None:
        patch = parse([
            {"action": "swap", "target_id": "chair"},
        ])
        assert [operation.action for operation in patch.operations] == [PatchAction.SWAP]
        assert patch.rejected == ()
        assert patch.usable

    def test_terminating_is_a_usable_answer(self) -> None:
        """An agent that has established the task cannot be met should say so rather than thrash."""
        patch = parse([{"action": "terminate"}])
        assert patch.terminates
        assert patch.operations[0].target_id == "", "a termination is addressed at the task"

    def test_adding_is_refused_until_the_application_can_add_to_the_spec(self) -> None:
        patch = parse([{"action": "add", "target_id": "shelf", "synset": "cabinet.n.01"}])
        assert not patch.usable
        assert patch.rejected[0].reason == "action-not-implemented"


class TestRefused:
    def test_removing_a_required_entity_is_refused(self) -> None:
        patch = parse([{"action": "remove", "target_id": "desk"}])
        assert patch.operations == ()
        assert patch.rejected[0].reason == "hard-entity-removal"

    def test_removing_a_non_optional_entity_is_refused(self) -> None:
        """Every piece being optional would make "delete the furniture" the cheapest repair."""
        strict = parse_patch(
            json.dumps({"operations": [{"action": "remove", "target_id": "chair"}]}),
            plan=ScenePlan(furniture=(
                FurnitureRequest(
                    entity_id="chair", synset="chair.n.01", kind=PlacementKind.BESIDE,
                    orientation=Orientation.FACES_TOWARD, faces_entity="desk", optional=False,
                ),
            )),
        )
        assert strict.rejected[0].reason == "non-optional-removal"

    def test_removing_an_optional_entity_is_allowed(self) -> None:
        patch = parse([{"action": "remove", "target_id": "chair"}])
        assert patch.usable
        assert patch.operations[0].action is PatchAction.REMOVE

    def test_an_unimplemented_design_action_is_refused_and_named(self) -> None:
        """A closed vocabulary that lies about what it can do is worse than a small one."""
        for action in UNIMPLEMENTED_DESIGN_ACTIONS:
            patch = parse([{"action": action, "target_id": "chair"}])
            assert patch.operations == ()
            assert patch.rejected[0].reason == "action-not-implemented"
            assert action in patch.rejected[0].detail

    def test_an_unknown_action_is_refused(self) -> None:
        patch = parse([{"action": "make_scene_good", "target_id": "chair"}])
        assert patch.rejected[0].reason == "unknown-action"

    def test_an_unknown_target_is_refused(self) -> None:
        patch = parse([{"action": "swap", "target_id": "sofa"}])
        assert patch.rejected[0].reason == "unknown-target"

    def test_a_synset_outside_the_vocabulary_is_refused(self) -> None:
        """An L5 patch names what the request already permits; it does not choose what the scene
        contains."""
        patch = parse([{"action": "add", "target_id": "bin", "synset": "helicopter.n.01"}])
        assert patch.rejected[0].reason == "action-not-implemented"

    def test_adding_without_a_synset_is_refused(self) -> None:
        patch = parse([{"action": "add", "target_id": "bin"}])
        assert patch.rejected[0].reason == "action-not-implemented"

    def test_a_patch_of_only_refusals_is_not_usable(self) -> None:
        patch = parse([{"action": "remove", "target_id": "desk"}])
        assert not patch.usable
        assert not patch.terminates

    def test_one_bad_operation_does_not_discard_a_good_one(self) -> None:
        patch = parse([
            {"action": "remove", "target_id": "desk"},
            {"action": "swap", "target_id": "chair"},
        ])
        assert len(patch.operations) == 1
        assert len(patch.rejected) == 1


class TestParsing:
    def test_an_unparseable_response_is_an_outcome_rather_than_a_crash(self) -> None:
        """The run has already spent the request; crashing would lose the record that it spent it."""
        patch = parse_patch("I could not decide.", plan=plan())
        assert not patch.usable
        assert patch.rejected[0].reason == "unparseable"

    def test_an_empty_response_is_recorded_as_empty(self) -> None:
        assert parse_patch("", plan=plan()).rejected[0].reason == "empty-response"

    def test_the_diagnosis_is_capped(self) -> None:
        """A replanner that returns an essay has not returned a patch, and the diagnosis lands in
        the run's own artefacts."""
        patch = parse_patch(
            json.dumps({"diagnosis": "x" * 5000, "operations": []}), plan=plan()
        )
        assert len(patch.diagnosis) <= 400


class TestApplication:
    def test_a_removal_becomes_a_plan_without_that_entity(self) -> None:
        edits = application_requests(parse([{"action": "remove", "target_id": "chair"}]), plan())
        assert edits == (("chair", None),)

    def test_a_swap_advances_the_candidate_index(self) -> None:
        edits = application_requests(parse([{"action": "swap", "target_id": "chair"}]), plan())
        assert len(edits) == 1
        entity_id, request = edits[0]
        assert entity_id == "chair" and request.asset_index == 1

    def test_an_unimplemented_add_produces_no_application(self) -> None:
        edits = application_requests(
            parse([{"action": "add", "target_id": "shelf", "synset": "cabinet.n.01"}]), plan()
        )
        assert edits == ()


class TestSummary:
    def test_the_summary_carries_the_failure_and_the_tried_repairs(self) -> None:
        """The tried repairs are what stop the model proposing the repair the local search just
        exhausted."""
        summary = scene_summary(
            plan(), goal="pick up the cup", stage="gate.geometry",
            active_failure={"kind": "TASK_UNREACHABLE"},
            suggested_repairs=["resample_pose", "optimise_pose"],
            known_targets=("desk",),
        )
        assert summary["goal"] == "pick up the cup"
        assert summary["active_failure"]["kind"] == "TASK_UNREACHABLE"
        assert summary["suggested_repair"] == ["resample_pose", "optimise_pose"]
        assert summary["task_critical"]["desk"]["required"] is True

    def test_the_summary_carries_no_paths(self) -> None:
        """A replanner that can see the repository can be asked questions it should not answer."""
        text = json.dumps(scene_summary(
            plan(), goal="pick up the cup", stage="gate", active_failure={}, known_targets=("desk",)
        ))
        assert "/home/" not in text and "/mnt/" not in text and ".obj" not in text


class TestFailureCodes:
    def test_a_refusal_is_distinct_from_a_termination(self) -> None:
        """The first says the model had nothing to offer; the second says it judged the task
        unmeetable. Recording both as "failed" would lose which happened."""
        assert failure_code_for_patch(parse([{"action": "terminate"}])) is FailureCode.HARD_GOAL_DROPPED
        assert failure_code_for_patch(
            parse([{"action": "remove", "target_id": "desk"}])
        ) is FailureCode.PATCH_INVALID

    def test_a_usable_patch_that_did_not_help_is_no_progress(self) -> None:
        assert failure_code_for_patch(
            parse([{"action": "swap", "target_id": "chair"}])
        ) is FailureCode.NO_PROGRESS

    def test_the_implemented_vocabulary_covers_every_patch_action(self) -> None:
        assert set(IMPLEMENTED_ACTIONS) == {"swap", "remove", "terminate"}

    def test_an_advertised_action_always_has_an_application(self) -> None:
        patch = parse([
            {"action": "swap", "target_id": "chair"},
            {"action": "remove", "target_id": "chair"},
        ])
        assert len(application_requests(patch, plan())) == 2

    def test_a_declared_but_unimplemented_action_is_refused(self) -> None:
        patch = parse([{"action": "move", "target_id": "chair"}])
        assert not patch.usable
        assert patch.rejected[0].reason == "action-not-implemented"
