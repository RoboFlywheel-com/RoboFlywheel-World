# SPDX-License-Identifier: Apache-2.0
"""Readiness algebra.

The rule under test is `ready = ALL(pass AND hash matches)`. Two failure modes matter more
than the happy path:

* a conjunction degrading into a disjunction, so one backend passing covers for another
  failing;
* `not_run` or `unsupported` counting as satisfied, so absent evidence reads as success.

Both are the difference between a system that reports its own gaps and one that ships them.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from rfw_w.scenes.contracts.readiness import (
    GateId,
    Outcome,
    ReadinessError,
    ReadinessLevel,
    backend_gates,
    required_gates,
    resolve_readiness,
)
from rfw_w.scenes.contracts.tool import Evidence, EvidenceLedger, EvidenceStatus
from rfw_w.scenes.contracts.hashing import content_hash

VERSION_A = content_hash({"scene": "a"})
VERSION_B = content_hash({"scene": "b"})

BACKENDS = ("isaac",)


def _evidence(
    gate_id: str,
    status: EvidenceStatus,
    *,
    state_hash: str = VERSION_A,
    detail: str = "",
) -> Evidence:
    measured = 0.001 if status is EvidenceStatus.PASS else None
    return Evidence(
        gate_id=gate_id,
        status=status,
        state_hash=state_hash,
        backend="isaac",
        tool_version="test/1",
        method="test fixture",
        measured_value=measured,
        unit="m" if measured is not None else None,
        threshold_profile="engineering_v1" if status is EvidenceStatus.PASS else None,
        detail=detail,
    )


def _ledger_passing(
    level: ReadinessLevel,
    backends: Iterable[str] = BACKENDS,
    *,
    stale: Iterable[str] = (),
) -> EvidenceLedger:
    """A ledger that passes ``level``, with the named gates measured on another scene instead.

    ``stale`` is how a test says "this gate was measured, but not on the scene being resolved".
    Recording the gate a second time no longer says it: the ledger keeps every version, so a second
    entry sits beside the first rather than replacing it.
    """
    stale = set(stale)
    ledger = EvidenceLedger()
    for gate in required_gates(level, tuple(backends)):
        ledger.record(
            _evidence(gate, EvidenceStatus.PASS, state_hash=VERSION_B if gate in stale else VERSION_A)
        )
    return ledger


class TestRequiredGates:
    def test_levels_are_cumulative(self) -> None:
        previous: set[str] = set()
        for level in ReadinessLevel:
            gates = set(required_gates(level, BACKENDS))
            assert previous <= gates, f"{level.value} dropped gates required by a lower level"
            previous = gates

    def test_backend_gates_are_namespaced_per_backend(self) -> None:
        # One shared backend.run would let Isaac's success stand in for MuJoCo's.
        assert set(backend_gates("isaac")).isdisjoint(backend_gates("mujoco"))

    def test_requires_at_least_one_backend(self) -> None:
        with pytest.raises(ReadinessError):
            required_gates(ReadinessLevel.DRAFT, ())

    def test_draft_requires_nothing(self) -> None:
        assert required_gates(ReadinessLevel.DRAFT, BACKENDS) == tuple(backend_gates("isaac"))


class TestConjunction:
    def test_all_gates_passing_reaches_the_requested_level(self) -> None:
        resolution = resolve_readiness(
            _ledger_passing(ReadinessLevel.FUNCTIONAL_SCENE_READY),
            scene_version=VERSION_A,
            requested=ReadinessLevel.FUNCTIONAL_SCENE_READY,
            requested_backends=BACKENDS,
        )
        assert resolution.achieved is ReadinessLevel.FUNCTIONAL_SCENE_READY
        assert resolution.meets_request
        assert resolution.outcome is None

    def test_one_failing_gate_blocks_the_level(self) -> None:
        ledger = _ledger_passing(ReadinessLevel.FUNCTIONAL_SCENE_READY)
        gate = GateId.FUNCTION_INITIAL_RELATIONS.value
        ledger.record(_evidence(gate, EvidenceStatus.FAIL, detail="cup ended on the floor"))
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.FUNCTIONAL_SCENE_READY,
            requested_backends=BACKENDS,
        )
        assert resolution.achieved is ReadinessLevel.SIMREADY_SCENE_READY
        assert not resolution.meets_request
        assert resolution.outcome is Outcome.FAILED
        assert gate in resolution.failed_gates

    def test_one_backend_passing_does_not_cover_another(self) -> None:
        # The manual's requirement: a single-engine pass does not offset a required engine
        # failing. Isaac passes everything; MuJoCo has no evidence at all.
        ledger = _ledger_passing(ReadinessLevel.SIMREADY_SCENE_READY, ("isaac",))
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.SIMREADY_SCENE_READY,
            requested_backends=("isaac", "mujoco"),
        )
        assert resolution.achieved is ReadinessLevel.DRAFT
        assert not resolution.meets_request
        assert any("mujoco" in gate for gate in resolution.not_run_gates)

    def test_beauty_does_not_offset_a_penetration(self) -> None:
        # An aesthetic score carries no gate, so no visual score can rescue a geometry gate.
        ledger = _ledger_passing(ReadinessLevel.SIMREADY_SCENE_READY)
        ledger.record(_evidence(GateId.GEOMETRY_NO_PENETRATION.value, EvidenceStatus.FAIL))
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.SIMREADY_SCENE_READY,
            requested_backends=BACKENDS,
        )
        assert resolution.achieved is ReadinessLevel.DRAFT


class TestNonSatisfyingStatuses:
    @pytest.mark.parametrize(
        "status", [EvidenceStatus.FAIL, EvidenceStatus.NOT_RUN, EvidenceStatus.UNSUPPORTED]
    )
    def test_only_pass_satisfies(self, status: EvidenceStatus) -> None:
        ledger = _ledger_passing(ReadinessLevel.SIMREADY_SCENE_READY)
        gate = GateId.PHYSICS_SETTLE.value
        ledger.record(_evidence(gate, status))
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.SIMREADY_SCENE_READY,
            requested_backends=BACKENDS,
        )
        assert resolution.achieved is ReadinessLevel.DRAFT
        assert not resolution.meets_request

    def test_absent_evidence_is_not_run(self) -> None:
        ledger = EvidenceLedger()
        assert ledger.status_for(GateId.PHYSICS_SETTLE.value) is EvidenceStatus.NOT_RUN

    def test_unsupported_is_reported_as_unsupported_not_failed(self) -> None:
        ledger = _ledger_passing(ReadinessLevel.SIMREADY_SCENE_READY)
        gate = GateId.VISUAL_CRITIQUE.value
        ledger.record(_evidence(gate, EvidenceStatus.UNSUPPORTED))
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.SIMREADY_SCENE_READY,
            requested_backends=BACKENDS,
        )
        assert resolution.outcome is Outcome.UNSUPPORTED
        assert gate in resolution.unsupported_gates
        assert gate not in resolution.failed_gates


class TestStaleness:
    def test_evidence_from_another_scene_version_does_not_count(self) -> None:
        gate = GateId.VISUAL_CRITIQUE.value
        # A pass measured on version B, presented while the scene is version A.
        ledger = _ledger_passing(ReadinessLevel.SIMREADY_SCENE_READY, stale=(gate,))
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.SIMREADY_SCENE_READY,
            requested_backends=BACKENDS,
        )
        assert resolution.stale_gates == (gate,)
        assert not resolution.meets_request
        assert resolution.achieved is ReadinessLevel.DRAFT

    def test_a_stale_pass_is_not_reported_as_a_failure(self) -> None:
        # Stale and failed need different repairs: one re-runs, the other re-solves.
        gate = GateId.VISUAL_CRITIQUE.value
        ledger = _ledger_passing(ReadinessLevel.SIMREADY_SCENE_READY, stale=(gate,))
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.SIMREADY_SCENE_READY,
            requested_backends=BACKENDS,
        )
        assert gate not in resolution.failed_gates
        assert "was measured on scene" in resolution.blockers[0].reason

    def test_a_rollback_re_resolves_from_the_version_it_returns_to(self) -> None:
        """Why the ledger keeps every version.

        The ladder advances the scene, measures it, does not improve it, and rolls back to the
        scene it started from. Under one-entry-per-gate that rollback landed on a version whose
        evidence the measurement had already destroyed, so every gate read as stale and the run
        reported `draft` with nothing measured -- whatever the restored scene had measured.
        """
        gate = GateId.GEOMETRY_IN_BOUNDS.value
        ledger = _ledger_passing(ReadinessLevel.SIMREADY_SCENE_READY)
        ledger.record(_evidence(gate, EvidenceStatus.FAIL, state_hash=VERSION_B))

        def at(version: str) -> "object":
            return resolve_readiness(
                ledger,
                scene_version=version,
                requested=ReadinessLevel.SIMREADY_SCENE_READY,
                requested_backends=BACKENDS,
            )

        # The repaired scene failed its gate, and only the gate it failed is blocking.
        assert at(VERSION_B).achieved is ReadinessLevel.DRAFT
        assert at(VERSION_B).failed_gates == (gate,)
        # Rolled back, the pass recorded before the repair is still the pass for this scene: the
        # level comes back without re-measuring anything.
        assert at(VERSION_A).meets_request
        assert at(VERSION_A).achieved is ReadinessLevel.SIMREADY_SCENE_READY

    def test_ledger_reports_stale_gates(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(_evidence(GateId.PHYSICS_SETTLE.value, EvidenceStatus.PASS, state_hash=VERSION_B))
        assert ledger.stale_gate_ids(VERSION_A) == (GateId.PHYSICS_SETTLE.value,)
        assert not ledger.satisfies(GateId.PHYSICS_SETTLE.value, VERSION_A)


class TestOutcomeSeparation:
    def test_outcome_is_not_a_readiness_level(self) -> None:
        # Kept as separate types so a caller cannot compare "failed" against a level.
        assert not hasattr(Outcome, "rank")
        assert isinstance(ReadinessLevel.DRAFT.rank, int)

    def test_a_lower_achievable_level_does_not_mask_a_failure(self) -> None:
        """Simready passes fully, but the requested functional level fails.

        Reporting this as "simready_scene_ready, partially functional" is exactly the
        overclaim the outcome field exists to prevent.
        """
        ledger = _ledger_passing(ReadinessLevel.SIMREADY_SCENE_READY)
        gate = GateId.FUNCTION_SPACE.value
        ledger.record(_evidence(gate, EvidenceStatus.FAIL, detail="corridor below 0.9 m"))
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.FUNCTIONAL_SCENE_READY,
            requested_backends=BACKENDS,
        )
        assert resolution.achieved is ReadinessLevel.SIMREADY_SCENE_READY
        assert resolution.outcome is Outcome.FAILED
        assert not resolution.meets_request

    def test_resolution_serialises_its_reasoning(self) -> None:
        ledger = EvidenceLedger()
        resolution = resolve_readiness(
            ledger,
            scene_version=VERSION_A,
            requested=ReadinessLevel.SIMREADY_SCENE_READY,
            requested_backends=BACKENDS,
        )
        payload = resolution.to_dict()
        assert payload["meets_request"] is False
        assert payload["blockers"]
        assert payload["scene_version"] == VERSION_A

    def test_rejects_a_malformed_scene_version(self) -> None:
        with pytest.raises(ReadinessError):
            resolve_readiness(
                EvidenceLedger(),
                scene_version="not-a-hash",
                requested=ReadinessLevel.DRAFT,
                requested_backends=BACKENDS,
            )


def test_task_cell_intrusion_blocks_functional_readiness():
    ledger = _ledger_passing(ReadinessLevel.FUNCTIONAL_SCENE_READY)
    ledger.record(_evidence("geometry.task_cell_clear", EvidenceStatus.FAIL))
    resolution = resolve_readiness(ledger, scene_version=VERSION_A,
                                  requested=ReadinessLevel.FUNCTIONAL_SCENE_READY,
                                  requested_backends=BACKENDS)
    assert not resolution.meets_request
    assert "geometry.task_cell_clear" in resolution.failed_gates


@pytest.mark.parametrize('status', [EvidenceStatus.NOT_RUN, EvidenceStatus.FAIL, EvidenceStatus.UNSUPPORTED])
def test_draft_request_with_unsatisfied_backend_gates_never_claims_success(status):
    ledger = _ledger_passing(ReadinessLevel.DRAFT)
    ledger.record(_evidence('backend.isaac.load', status))
    resolution = resolve_readiness(ledger, scene_version=VERSION_A, requested=ReadinessLevel.DRAFT, requested_backends=BACKENDS)
    assert resolution.achieved is ReadinessLevel.DRAFT
    assert resolution.blockers
    assert not resolution.meets_request
    assert not resolution.to_dict()['meets_request']
