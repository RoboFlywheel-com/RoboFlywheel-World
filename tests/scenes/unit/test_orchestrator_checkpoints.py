# SPDX-License-Identifier: Apache-2.0
"""Hierarchical checkpoints: what a rollback restores, and how much of it.

Two claims from the executable design are pinned here. A rollback must be **scoped** -- restoring
the subgraph that failed rather than the whole scene is the entire reason the checkpoints are
hierarchical. And it must restore the **candidate cursor** along with the geometry (§10), so a
repaired region does not immediately re-exhaust the candidates it exhausted before. The cursor lives
inside the plan, so the two cannot come apart; the test says so rather than leaving it implicit.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.orchestrator.checkpoints import (
    CheckpointError,
    CheckpointName,
    CheckpointStack,
    RollbackKind,
    plan_closure,
    rollback_kind,
)
from rfw_w.scenes.solvers.furnish import (
    FurnitureRequest,
    Orientation,
    PlacementKind,
    ScenePlan,
    SurfaceRequest,
)

VERSION_A = "a" * 64
VERSION_B = "b" * 64


def plan(*, desk_skip: int = 0, extra: bool = False) -> ScenePlan:
    """A desk, the chair that faces it, a cabinet, and a notebook standing on the desk."""
    furniture = [
        FurnitureRequest(
            entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
            orientation=Orientation.FRONT_TO_ROOM, hard_required=True, optional=False,
            skip_candidates=desk_skip,
        ),
        FurnitureRequest(
            entity_id="chair", synset="chair.n.01", kind=PlacementKind.BESIDE,
            orientation=Orientation.FACES_TOWARD, faces_entity="desk",
        ),
        FurnitureRequest(
            entity_id="storage", synset="cabinet.n.01", kind=PlacementKind.AGAINST_WALL,
            orientation=Orientation.FRONT_TO_ROOM,
        ),
    ]
    if extra:
        furniture.append(FurnitureRequest(
            entity_id="plant", synset="potted_plant.n.01", kind=PlacementKind.CORNER,
            orientation=Orientation.SYMMETRIC,
        ))
    return ScenePlan(
        furniture=tuple(furniture),
        surfaces=(SurfaceRequest(
            entity_id="notebook", synset="notebook.n.01", target_width_m=0.2, support_entity="desk"
        ),),
    )


class TestClosure:
    def test_the_closure_includes_what_depends_on_the_target(self) -> None:
        """The chair faces the desk and the notebook stands on it, so both are in the closure."""
        assert plan_closure(plan(), ("desk",)) == frozenset({"desk", "chair", "notebook"})

    def test_the_closure_includes_the_target_itself(self) -> None:
        assert "storage" in plan_closure(plan(), ("storage",))

    def test_a_cycle_does_not_hang_the_traversal(self) -> None:
        """A mutual facing pair is legitimate, so a visited set is required rather than optional."""
        mutual = ScenePlan(furniture=(
            FurnitureRequest(entity_id="a", synset="chair.n.01", kind=PlacementKind.BESIDE,
                             orientation=Orientation.FACES_TOWARD, faces_entity="b"),
            FurnitureRequest(entity_id="b", synset="chair.n.01", kind=PlacementKind.BESIDE,
                             orientation=Orientation.FACES_TOWARD, faces_entity="a"),
        ))
        assert plan_closure(mutual, ("a",)) == frozenset({"a", "b"})

    def test_an_unknown_target_is_not_invented(self) -> None:
        assert plan_closure(plan(), ("ghost",)) == frozenset()


class TestStack:
    def test_a_named_checkpoint_can_be_restored(self) -> None:
        stack = CheckpointStack()
        stack.push(CheckpointName.FURNITURE, plan(), scene_version=VERSION_A)
        restored = stack.restore(CheckpointName.FURNITURE)
        assert restored.scene_version == VERSION_A

    def test_a_checkpoint_that_was_never_taken_is_refused(self) -> None:
        """Refusing rather than returning the nearest: a caller that asked for the room layout and
        silently received the furniture one would roll back less than it believed."""
        stack = CheckpointStack()
        stack.push(CheckpointName.FURNITURE, plan(), scene_version=VERSION_A)
        with pytest.raises(CheckpointError, match="cannot restore a state the run never recorded"):
            stack.restore(CheckpointName.ROOM_LAYOUT)

    def test_pushing_the_same_level_twice_replaces_it(self) -> None:
        """The five names are levels, not attempts: two furniture checkpoints would leave "restore
        to the furniture stage" ambiguous."""
        stack = CheckpointStack()
        stack.push(CheckpointName.FURNITURE, plan(), scene_version=VERSION_A)
        stack.push(CheckpointName.FURNITURE, plan(desk_skip=2), scene_version=VERSION_B)
        assert len(stack.checkpoints) == 1
        assert stack.restore(CheckpointName.FURNITURE).scene_version == VERSION_B

    def test_the_deepest_covering_checkpoint_is_chosen(self) -> None:
        """Restoring the newest valid state undoes the least work."""
        stack = CheckpointStack()
        stack.push(CheckpointName.ROOM_LAYOUT, plan(), scene_version=VERSION_A)
        stack.push(CheckpointName.FURNITURE, plan(desk_skip=1), scene_version=VERSION_B)
        found = stack.deepest_covering(("desk", "chair"))
        assert found is not None and found.scene_version == VERSION_B

    def test_a_closure_nothing_covers_yields_no_rollback(self) -> None:
        stack = CheckpointStack()
        stack.push(
            CheckpointName.FURNITURE, plan(), scene_version=VERSION_A, scope_ids=("desk",),
        )
        assert stack.deepest_covering(("plant",)) is None

    def test_an_empty_closure_is_not_a_rollback(self) -> None:
        stack = CheckpointStack()
        stack.push(CheckpointName.FURNITURE, plan(), scene_version=VERSION_A)
        assert stack.deepest_covering(()) is None

    def test_the_candidate_cursor_travels_with_the_plan(self) -> None:
        """§10 requires a rollback to restore the cursor, so that a repaired region does not
        immediately re-exhaust the candidates it exhausted before.

        It is stored inside the plan rather than beside it on purpose: two things restored together
        can be restored out of step, and one thing cannot.
        """
        stack = CheckpointStack()
        stack.push(CheckpointName.FURNITURE, plan(desk_skip=0), scene_version=VERSION_A)
        stack.push(CheckpointName.MANIPULANDS, plan(desk_skip=3), scene_version=VERSION_B)
        restored = stack.restore(CheckpointName.FURNITURE)
        assert restored.plan.furniture_for("desk").skip_candidates == 0


class TestRollbackKind:
    def test_a_room_rollback_is_named_as_one(self) -> None:
        stack = CheckpointStack()
        checkpoint = stack.push(
            CheckpointName.ROOM_LAYOUT, plan(), scene_version=VERSION_A
        )
        assert rollback_kind(checkpoint, ("desk",)) is RollbackKind.ROOM

    def test_a_rollback_of_one_required_piece_is_an_asset_rollback(self) -> None:
        stack = CheckpointStack()
        checkpoint = stack.push(
            CheckpointName.TASK_CRITICAL_ASSETS, plan(), scene_version=VERSION_A
        )
        assert rollback_kind(checkpoint, ("desk",)) is RollbackKind.ASSET

    def test_a_rollback_of_a_neighbourhood_is_a_subgraph_rollback(self) -> None:
        stack = CheckpointStack()
        checkpoint = stack.push(
            CheckpointName.FURNITURE, plan(), scene_version=VERSION_A, scope_ids=("chair", "desk"),
        )
        assert rollback_kind(checkpoint, ("chair",)) is RollbackKind.SUBGRAPH

    def test_rollback_for_reports_the_closure_it_would_restore(self) -> None:
        stack = CheckpointStack()
        stack.push(CheckpointName.FURNITURE, plan(), scene_version=VERSION_A)
        found = stack.rollback_for(("desk",), plan=plan())
        assert found is not None
        checkpoint, kind, closure = found
        assert checkpoint.name is CheckpointName.FURNITURE
        assert kind is RollbackKind.ROOM or kind is RollbackKind.SUBGRAPH
        assert closure == frozenset({"desk", "chair", "notebook"})

    def test_no_checkpoint_means_no_rollback_rather_than_a_guess(self) -> None:
        assert CheckpointStack().rollback_for(("desk",), plan=plan()) is None
