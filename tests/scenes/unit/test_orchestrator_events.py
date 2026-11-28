# SPDX-License-Identifier: Apache-2.0
"""Typed events: what raises one, what does not, and what it names.

The two claims worth pinning are negative. A **stale** failure must raise nothing, because a failure
measured on a scene that no longer exists would send the ladder after something the current scene
does not have. And an **unsupported** capability must raise nothing either, because a declared
boundary is not a failure to escalate on -- raising `ARTICULATION_FAILURE` for every run that
declares articulated assets unsupported would put a permanent, unfixable entry in the ladder.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.tool import Evidence, EvidenceLedger, EvidenceStatus, FailureCode
from rfw_w.scenes.orchestrator.events import (
    EventKind,
    EventError,
    FailureEvent,
    code_for_gate,
    events_from_evidence,
    events_from_ledger,
    kind_for_gate,
    stage_complete,
    targets_from_witnesses,
)

CURRENT = "c" * 64
OTHER = "d" * 64


def gate(
    gate_id: str,
    status: EvidenceStatus,
    *,
    state_hash: str = CURRENT,
    witnesses: tuple[str, ...] = (),
    detail: str = "",
) -> Evidence:
    return Evidence(
        gate_id=gate_id, status=status, state_hash=state_hash, backend="test",
        tool_version="test/v1", method="test", detail=detail, witness_refs=witnesses,
        # The contract refuses a pass that does not name the profile it passed against, and these
        # tests exercise the event rules rather than that one.
        threshold_profile="test_profile",
    )


class TestWhatRaisesAnEvent:
    def test_a_current_failure_raises_its_gate_s_event(self) -> None:
        raised = events_from_evidence(
            gate("geometry.in_bounds", EvidenceStatus.FAIL, witnesses=("instance:desk:inside-wall",)),
            scene_version=CURRENT,
        )
        assert len(raised) == 1
        assert raised[0].kind is EventKind.LAYOUT_DEADLOCK
        assert raised[0].failure_code is FailureCode.GEOMETRY_OUT_OF_BOUNDS
        assert raised[0].targets == ("desk",)

    def test_a_stale_failure_raises_nothing(self) -> None:
        """The trap: a failure from an older version describes a scene that is gone."""
        raised = events_from_evidence(
            gate("geometry.in_bounds", EvidenceStatus.FAIL, state_hash=OTHER), scene_version=CURRENT
        )
        assert raised == ()

    def test_a_pass_raises_nothing(self) -> None:
        assert events_from_evidence(
            gate("geometry.in_bounds", EvidenceStatus.PASS), scene_version=CURRENT
        ) == ()

    def test_a_not_run_gate_raises_nothing(self) -> None:
        """Not-run is an unfinished job, not a failure."""
        assert events_from_evidence(
            gate("visual.critique", EvidenceStatus.NOT_RUN), scene_version=CURRENT
        ) == ()

    def test_an_unsupported_capability_raises_nothing(self) -> None:
        """A declared boundary is reported through the readiness outcome, not escalated."""
        assert events_from_evidence(
            gate("capability.articulated.*", EvidenceStatus.UNSUPPORTED), scene_version=CURRENT
        ) == ()

    def test_a_backend_failure_raises_nothing(self) -> None:
        """A backend that could not load says nothing about the scene; the loop retries it instead."""
        assert kind_for_gate("backend.isaac.load") is None
        assert events_from_evidence(
            gate("backend.isaac.load", EvidenceStatus.FAIL), scene_version=CURRENT
        ) == ()

    def test_an_unclassified_gate_raises_nothing(self) -> None:
        """Visible as unclassified rather than filed under a guess."""
        assert kind_for_gate("something.new") is None

    def test_every_gate_maps_to_a_code_the_repair_policy_can_switch_on(self) -> None:
        for gate_id in (
            "geometry.no_penetration", "geometry.in_bounds", "geometry.supports",
            "function.initial_relations", "function.space_conditions", "visual.critique",
        ):
            assert isinstance(code_for_gate(gate_id), FailureCode)


class TestTargets:
    def test_the_three_witness_namespaces_all_yield_targets(self) -> None:
        assert targets_from_witnesses(
            ("instance:desk:inside-wall", "pair:cup|desk:overlap", "region:passage")
        ) == ("cup", "desk", "passage")

    def test_targets_are_sorted_and_deduplicated(self) -> None:
        """The order a gate happened to list them in is not information."""
        assert targets_from_witnesses(
            ("instance:desk:x", "instance:chair:y", "instance:desk:z")
        ) == ("chair", "desk")

    def test_an_undefined_witness_format_yields_nothing(self) -> None:
        """Not parsed leniently: a target that names nothing is worse than no target."""
        assert targets_from_witnesses(("desk is in the wall", "instance", ":", "x:y:z")) == ()

    def test_a_pair_witness_names_both_ends(self) -> None:
        assert targets_from_witnesses(("pair:cup|desk:overlap",)) == ("cup", "desk")


class TestSignatures:
    def test_the_signature_ignores_the_detail_text(self) -> None:
        """Two runs that fail the same way must produce the same signature.

        A message that changed a number would otherwise make an identical failure look new, and the
        no-progress rule would never fire.
        """
        first = FailureEvent(
            kind=EventKind.LAYOUT_DEADLOCK, source_gate="geometry.in_bounds",
            failure_code=FailureCode.GEOMETRY_OUT_OF_BOUNDS, detail="0.1 m",
        )
        second = FailureEvent(
            kind=EventKind.LAYOUT_DEADLOCK, source_gate="geometry.in_bounds",
            failure_code=FailureCode.GEOMETRY_OUT_OF_BOUNDS, detail="0.2 m",
        )
        assert first.signature == second.signature

    def test_the_signature_ignores_the_targets(self) -> None:
        """Repairing one of two colliding pieces leaves the same problem present."""
        one = FailureEvent(
            kind=EventKind.LAYOUT_DEADLOCK, source_gate="geometry.no_penetration",
            targets=("a",), failure_code=FailureCode.GEOMETRY_PENETRATION,
        )
        two = FailureEvent(
            kind=EventKind.LAYOUT_DEADLOCK, source_gate="geometry.no_penetration",
            targets=("b",), failure_code=FailureCode.GEOMETRY_PENETRATION,
        )
        assert one.signature == two.signature

    def test_a_failure_without_a_typed_code_is_refused(self) -> None:
        """The policy switches on the code; free text would make identical failures look new."""
        with pytest.raises(EventError):
            FailureEvent(kind=EventKind.LAYOUT_DEADLOCK, source_gate="geometry.in_bounds")

    def test_stage_complete_carries_no_code_and_is_not_a_failure(self) -> None:
        event = stage_complete("backend.render")
        assert event.kind is EventKind.STAGE_COMPLETE
        assert not event.kind.is_failure
        assert event.failure_code is None


class TestLedgerSweep:
    def test_only_the_failing_gates_of_the_current_version_are_raised(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(gate("geometry.in_bounds", EvidenceStatus.FAIL,
                           witnesses=("instance:desk:inside-wall",)))
        ledger.record(gate("geometry.no_penetration", EvidenceStatus.PASS))
        ledger.record(gate("function.space_conditions", EvidenceStatus.FAIL, state_hash=OTHER))
        raised = events_from_ledger(ledger, scene_version=CURRENT)
        assert [event.source_gate for event in raised] == ["geometry.in_bounds"]

    def test_the_sweep_can_be_restricted_to_named_gates(self) -> None:
        """A stage is responsible for its own family, and must not escalate another's failure."""
        ledger = EvidenceLedger()
        ledger.record(gate("geometry.in_bounds", EvidenceStatus.FAIL))
        ledger.record(gate("function.space_conditions", EvidenceStatus.FAIL))
        raised = events_from_ledger(
            ledger, scene_version=CURRENT, gates=("function.space_conditions",)
        )
        assert [event.source_gate for event in raised] == ["function.space_conditions"]

    def test_the_sweep_order_is_stable(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(gate("geometry.supports", EvidenceStatus.FAIL))
        ledger.record(gate("geometry.in_bounds", EvidenceStatus.FAIL))
        first = events_from_ledger(ledger, scene_version=CURRENT)
        second = events_from_ledger(ledger, scene_version=CURRENT)
        assert [event.source_gate for event in first] == [event.source_gate for event in second]
