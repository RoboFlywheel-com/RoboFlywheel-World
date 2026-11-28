# SPDX-License-Identifier: Apache-2.0
"""The escalation ladder's ordering rule, which is the whole content of §17.

Three rules are pinned here, and all three are ways the ladder degrades into something else:

* a rung cannot be reached while a cheaper admissible rung is untried — otherwise "local recovery
  first" is a comment rather than a policy, and the loop asks the model whenever anything is wrong;
* the two tracks do not cross — a lighting adjustment cannot fix a penetration, and a penetration
  cannot be fixed by one, so the appearance track is inadmissible for a geometric failure and the
  geometric track is inadmissible for a lighting complaint;
* only L5 may remove an entity — otherwise "delete the furniture" is the cheapest way to clear any
  deadlock and the room empties while every gate passes.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.tool import FailureCode
from rfw_w.scenes.orchestrator.escalation import (
    ATOMS,
    Atom,
    EscalationError,
    Ladder,
    RepairLevel,
    Track,
    admissible_atoms,
    atom_for_stage,
    stage_names,
)
from rfw_w.scenes.orchestrator.events import EventKind, FailureEvent


def event(
    kind: EventKind = EventKind.LAYOUT_DEADLOCK,
    *,
    code: FailureCode = FailureCode.GEOMETRY_OUT_OF_BOUNDS,
    targets: tuple[str, ...] = ("desk",),
    origin: str = "gate",
    gate: str = "geometry.in_bounds",
) -> FailureEvent:
    return FailureEvent(
        kind=kind, source_gate=gate, targets=targets, failure_code=code, origin=origin
    )


class TestOrdering:
    def test_the_first_untried_rung_is_the_cheapest(self) -> None:
        ladder = Ladder()
        failure = event()
        assert ladder.next_untried(failure) is Atom.RESAMPLE_POSE
        ladder.record(failure, Atom.RESAMPLE_POSE, resolved=False, improved=False)
        assert ladder.next_untried(failure) is Atom.OPTIMISE_POSE
        ladder.record(failure, Atom.OPTIMISE_POSE, resolved=False, improved=False)
        assert ladder.next_untried(failure) is Atom.SWAP_ASSET

    def test_a_later_rung_is_refused_while_a_cheaper_one_is_untried(self) -> None:
        """The rule itself, stated as a check rather than as a comment."""
        ladder = Ladder()
        with pytest.raises(EscalationError, match="local recovery comes first"):
            ladder.assert_order(event(), Atom.SWAP_ASSET)

    def test_the_same_rung_again_is_allowed_once_nothing_cheaper_remains(self) -> None:
        """What the check actually forbids is *skipping*, not repeating."""
        ladder = Ladder()
        failure = event()
        ladder.record(failure, Atom.RESAMPLE_POSE, resolved=False, improved=False)
        ladder.assert_order(failure, Atom.OPTIMISE_POSE)

    def test_the_ladder_is_per_signature(self) -> None:
        """A second failure starts at L0 rather than inheriting the first one's progress."""
        ladder = Ladder()
        first = event(code=FailureCode.GEOMETRY_OUT_OF_BOUNDS)
        second = event(code=FailureCode.GEOMETRY_PENETRATION)
        ladder.record(first, Atom.RESAMPLE_POSE, resolved=False, improved=False)
        assert ladder.next_untried(second) is Atom.RESAMPLE_POSE
        assert ladder.next_untried(first) is Atom.OPTIMISE_POSE

    def test_the_ladder_runs_out_and_says_so(self) -> None:
        ladder = Ladder()
        failure = event()
        for atom in admissible_atoms(failure):
            ladder.record(failure, atom, resolved=False, improved=False)
        assert ladder.next_untried(failure) is None

    def test_reaching_the_last_rung_requires_every_earlier_one(self) -> None:
        ladder = Ladder()
        failure = event()
        seen: list[Atom] = []
        while (atom := ladder.next_untried(failure)) is not None:
            seen.append(atom)
            ladder.record(failure, atom, resolved=False, improved=False)
        assert seen == list(admissible_atoms(failure))
        assert seen[-1] is Atom.REPLAN
        assert [ATOMS[atom].level for atom in seen] == sorted(
            ATOMS[atom].level for atom in seen
        ), "the ladder must be walked in level order"


class TestTracks:
    def test_the_appearance_track_is_one_rung_and_then_the_agent(self) -> None:
        complaint = event(
            kind=EventKind.SEMANTIC_CONFLICT, code=FailureCode.VISUAL_GATE_FAILED,
            targets=(), origin="critic", gate="visual.critique",
        )
        assert admissible_atoms(complaint, track=Track.APPEARANCE) == (
            Atom.ADJUST_LIGHTING_MATERIAL, Atom.REPLAN,
        )

    def test_a_geometric_failure_has_no_appearance_rungs(self) -> None:
        """A wall embedding is not answered by moving the lighting."""
        assert Atom.ADJUST_LIGHTING_MATERIAL not in admissible_atoms(event())

    def test_a_lighting_complaint_has_no_pose_rungs(self) -> None:
        """The other direction: a lighting complaint is not answered by re-solving the floor plan."""
        complaint = event(
            kind=EventKind.SEMANTIC_CONFLICT, code=FailureCode.VISUAL_GATE_FAILED,
            targets=(), origin="critic", gate="visual.critique",
        )
        appearance = admissible_atoms(complaint, track=Track.APPEARANCE)
        assert Atom.RESAMPLE_POSE not in appearance
        assert Atom.OPTIMISE_POSE not in appearance
        assert Atom.SWAP_ASSET not in appearance

    def test_an_atom_from_the_wrong_track_is_refused(self) -> None:
        ladder = Ladder()
        complaint = event(
            kind=EventKind.SEMANTIC_CONFLICT, code=FailureCode.VISUAL_GATE_FAILED,
            targets=(), origin="critic", gate="visual.critique",
        )
        with pytest.raises(EscalationError, match="cannot be used on the"):
            ladder.assert_order(complaint, Atom.ADJUST_LIGHTING_MATERIAL, track=Track.GEOMETRIC)

    def test_a_track_with_no_rungs_for_an_event_returns_nothing(self) -> None:
        """An empty tuple means "this track cannot help", not "anything goes"."""
        assert admissible_atoms(event(), track=Track.APPEARANCE) == ()
        assert admissible_atoms(
            event(kind=EventKind.ASSET_EXHAUSTED, code=FailureCode.CANDIDATES_EXHAUSTED),
            track=Track.APPEARANCE,
        ) == ()


class TestRemoval:
    def test_only_the_agent_rung_may_remove_an_entity(self) -> None:
        """Otherwise the cheapest way to clear any deadlock is to delete the furniture."""
        removable = [atom for atom, info in ATOMS.items() if info.may_remove]
        assert removable == [Atom.REPLAN]

    def test_only_the_agent_rung_costs_a_model_request(self) -> None:
        spending = {atom for atom, info in ATOMS.items() if info.needs_model}
        assert spending == {Atom.REPLAN}


class TestStages:
    def test_every_atom_has_a_distinct_stage_name(self) -> None:
        names = stage_names()
        assert len(names) == len(set(names)) == len(ATOMS)

    def test_a_stage_name_maps_back_to_its_atom(self) -> None:
        for atom, info in ATOMS.items():
            assert atom_for_stage(info.stage) is atom

    def test_a_stage_that_is_not_a_repair_maps_to_nothing(self) -> None:
        assert atom_for_stage("geometry.in_bounds") is None


class TestMetrics:
    def test_a_locally_resolved_failure_counts_as_a_local_recovery(self) -> None:
        ladder = Ladder()
        failure = event()
        ladder.record(failure, Atom.RESAMPLE_POSE, resolved=True, improved=True)
        assert ladder.failures_seen == 1
        assert ladder.resolved_without_model == 1
        assert ladder.local_recovery_rate == 1.0

    def test_a_failure_only_the_model_resolved_is_not_a_local_recovery(self) -> None:
        ladder = Ladder()
        failure = event()
        ladder.record(failure, Atom.RESAMPLE_POSE, resolved=False, improved=False)
        ladder.record(failure, Atom.REPLAN, resolved=True, improved=True)
        assert ladder.resolved_without_model == 0
        assert ladder.local_recovery_rate == 0.0
        assert ladder.model_requests_spent() == 1

    def test_a_run_with_no_failures_reports_a_rate_of_one_and_no_failures(self) -> None:
        """``1.0`` beside ``failures_seen == 0`` is the honest reading, and must not be averaged."""
        ladder = Ladder()
        assert ladder.local_recovery_rate == 1.0
        assert ladder.failures_seen == 0
        assert ladder.model_requests_spent() == 0

    def test_the_levels_are_ordered(self) -> None:
        assert RepairLevel.L0_POSE_RESAMPLE < RepairLevel.L1_LOCAL_OPTIMISATION
        assert RepairLevel.L4_SUBGRAPH_ROLLBACK < RepairLevel.L5_AGENT_REPLAN
