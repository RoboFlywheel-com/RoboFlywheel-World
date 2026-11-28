# SPDX-License-Identifier: Apache-2.0
"""The repair atoms: what each one edits, and what it refuses to pretend it edited.

The rule these tests exist for is the one an atom is easiest to get wrong about: an edit the build
does not read changes nothing, and the ladder would then record a rung as *tried* on a scene that is
byte-identical. So each claim here is of the form "this atom moved a field the scene version
hashes", not "this atom ran".
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.orchestrator.escalation import Atom
from rfw_w.scenes.orchestrator.repair import RepairAction, apply
from rfw_w.scenes.solvers.furnish import (
    FurnitureRequest,
    LayoutError,
    Orientation,
    PlacementKind,
    ScenePlan,
    SurfaceRequest,
)


def plan(**overrides: object) -> ScenePlan:
    """A plan with a task target, which is the field most of these tests are about."""
    payload: dict[str, object] = {
        "furniture": (
            FurnitureRequest(
                entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
                orientation=Orientation.FRONT_TO_ROOM,
            ),
            FurnitureRequest(
                entity_id="plant", synset="plant.n.01", kind=PlacementKind.CORNER,
                orientation=Orientation.FRONT_TO_ROOM,
            ),
        ),
        "surfaces": (
            SurfaceRequest(
                entity_id="notebook", synset="notebook.n.01", target_width_m=0.2,
                support_entity="desk",
            ),
        ),
        "support_id": "desk",
        "target_id": "cup",
    }
    payload.update(overrides)
    return ScenePlan(**payload)  # type: ignore[arg-type]


def swap(target: str) -> object:
    return apply(plan(), RepairAction(atom=Atom.SWAP_ASSET, targets=(target,), source="critic"))


class TestTaskTarget:
    """The task target is not a plan row, so every atom that edits rows used to pass it by.

    It is still the entity `prompt_adherence` is measured against, and `swap_asset` is one of the
    two repairs that criterion admits -- so a swap aimed at it has to reach something the build
    reads, or the criterion has nothing that can answer it.
    """

    def test_the_target_is_not_dropped_as_an_unknown_target(self) -> None:
        outcome = swap("cup")
        assert "ignored unknown targets" not in " ".join(outcome.notes)
        assert outcome.changed is True

    def test_a_swap_on_the_target_advances_its_candidate(self) -> None:
        outcome = swap("cup")
        assert outcome.plan.target_asset_index == 1

    def test_that_change_moves_the_scene_version(self) -> None:
        """An edit the version cannot see is committed as if nothing happened."""
        outcome = swap("cup")
        assert content_hash(outcome.plan.to_dict()) != content_hash(plan().to_dict())

    def test_the_candidate_is_part_of_the_canonical_form(self) -> None:
        assert plan().to_dict()["target_asset_index"] == 0
        assert plan(target_asset_index=2).to_dict()["target_asset_index"] == 2

    def test_a_negative_candidate_is_refused(self) -> None:
        with pytest.raises(LayoutError, match="cannot use candidate"):
            plan(target_asset_index=-1)

    def test_a_swap_on_a_plan_row_still_advances_that_row(self) -> None:
        outcome = swap("plant")
        request = outcome.plan.furniture_for("plant")
        assert request is not None and request.asset_index == 1
        # And it did not touch the target on the way past.
        assert outcome.plan.target_asset_index == 0

    def test_an_atom_that_edits_rows_passes_the_target_by_rather_than_crashing(self) -> None:
        """`resample_pose` has nothing to resample for the target, and says so."""
        outcome = apply(plan(), RepairAction(atom=Atom.RESAMPLE_POSE, targets=("cup",)))
        assert outcome.changed is False
        assert any("sweep to resample" in note for note in outcome.notes)
