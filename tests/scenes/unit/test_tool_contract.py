# SPDX-License-Identifier: Apache-2.0
"""Tool exchange rules.

The rule with teeth here is that a read-only tool cannot propose a scene change. Validators
observe; only the orchestrator commits, from a writer's delta validated on an isolated state.
Making that a constructor error rather than a convention means a validator cannot quietly
become a writer.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.tool import (
    Evidence,
    EvidenceLedger,
    EvidenceStatus,
    FailureCode,
    ToolError,
    ToolRequest,
    ToolResult,
    ToolStatus,
)

VERSION = content_hash({"scene": "a"})


class TestToolRequest:
    def test_requires_a_tool_version(self) -> None:
        # A result without the tool version is not interpretable, so the version is required
        # at request time rather than left to the caller's discipline.
        with pytest.raises(ToolError, match="tool_version"):
            ToolRequest(
                request_id="r1", tool_version="", scene_version=VERSION, seed=1, deadline_s=10.0
            )

    def test_requires_a_scene_hash(self) -> None:
        with pytest.raises(ToolError, match="scene_version"):
            ToolRequest(
                request_id="r1", tool_version="t/1", scene_version="v3", seed=1, deadline_s=10.0
            )

    def test_rejects_non_positive_deadline(self) -> None:
        with pytest.raises(ToolError, match="deadline_s"):
            ToolRequest(
                request_id="r1", tool_version="t/1", scene_version=VERSION, seed=1, deadline_s=0.0
            )

    def test_rejects_non_positive_candidate_budget(self) -> None:
        with pytest.raises(ToolError, match="max_candidates"):
            ToolRequest(
                request_id="r1", tool_version="t/1", scene_version=VERSION, seed=1,
                deadline_s=10.0, max_candidates=0,
            )


def result(**overrides: object) -> ToolResult:
    payload: dict[str, object] = {
        "request_id": "r1",
        "status": ToolStatus.PASSED,
        "tool_version": "geometry/1",
        "base_scene_version": VERSION,
        "read_only": False,
        "elapsed_ms": 12.5,
        "touched_ids": ("desk",),
    }
    payload.update(overrides)
    return ToolResult(**payload)  # type: ignore[arg-type]


class TestReadOnlyRule:
    def test_read_only_tool_may_not_propose_a_delta(self) -> None:
        with pytest.raises(ToolError, match="read-only tool"):
            result(read_only=True, proposed_delta={"instances": {"desk": {"pose": [0, 0, 0]}}})

    def test_read_only_tool_may_return_metrics(self) -> None:
        observation = result(read_only=True, metrics={"penetration_m": 0.0})
        assert observation.proposed_delta is None

    def test_writer_may_propose_a_delta(self) -> None:
        writer = result(proposed_delta={"instances": {"desk": {"pose": [1, 0, 0]}}})
        assert writer.proposed_delta is not None


class TestFailureTyping:
    @pytest.mark.parametrize(
        "status", [ToolStatus.FAILED, ToolStatus.UNSUPPORTED, ToolStatus.TIMEOUT]
    )
    def test_non_passed_status_needs_a_failure_code(self, status: ToolStatus) -> None:
        with pytest.raises(ToolError, match="failure_code"):
            result(status=status)

    def test_passed_must_not_carry_a_failure_code(self) -> None:
        with pytest.raises(ToolError, match="must not carry failure_code"):
            result(failure_code=FailureCode.GEOMETRY_PENETRATION)

    def test_a_passed_result_is_not_retryable(self) -> None:
        with pytest.raises(ToolError, match="not retryable"):
            result(retryable=True)

    def test_timeout_is_an_infrastructure_failure(self) -> None:
        # A timeout says nothing about the scene, so it must not be routed to the model as a
        # semantic problem -- that would spend a model request on a transient fault.
        timed_out = result(status=ToolStatus.TIMEOUT, failure_code=FailureCode.BUDGET_EXHAUSTED)
        assert timed_out.is_infrastructure_failure

    def test_asset_io_failure_is_an_infrastructure_failure(self) -> None:
        io_failure = result(status=ToolStatus.FAILED, failure_code=FailureCode.ASSET_IO_FAILURE)
        assert io_failure.is_infrastructure_failure

    def test_a_geometry_failure_is_not_an_infrastructure_failure(self) -> None:
        geometric = result(status=ToolStatus.FAILED, failure_code=FailureCode.GEOMETRY_PENETRATION)
        assert not geometric.is_infrastructure_failure

    def test_unsupported_is_not_an_infrastructure_failure(self) -> None:
        # Unsupported is a capability answer, not a transient fault: retrying cannot help.
        unsupported = result(status=ToolStatus.UNSUPPORTED, failure_code=FailureCode.BACKEND_UNSUPPORTED)
        assert not unsupported.is_infrastructure_failure


class TestEvidence:
    @staticmethod
    def passing(**overrides: object) -> Evidence:
        payload: dict[str, object] = {
            "gate_id": "geometry.supports",
            "status": EvidenceStatus.PASS,
            "state_hash": VERSION,
            "backend": "isaac",
            "tool_version": "geometry/1",
            "method": "support_polygon_coverage",
            "measured_value": 0.0031,
            "unit": "m",
            "threshold_profile": "engineering_v1",
        }
        payload.update(overrides)
        return Evidence(**payload)  # type: ignore[arg-type]

    def test_evidence_must_carry_the_scene_hash(self) -> None:
        """Without the hash, a stale approval is indistinguishable from a current one."""
        with pytest.raises(ToolError, match="scene hash it was measured on"):
            self.passing(state_hash="latest")

    def test_a_pass_reports_a_unit(self) -> None:
        # A bare 0.0031 cannot be compared against a threshold.
        with pytest.raises(ToolError, match="without a unit"):
            self.passing(unit=None)

    def test_a_pass_names_its_threshold_profile(self) -> None:
        # Thresholds are configured and calibrated, not universal, so a pass must say which
        # profile it passed against.
        with pytest.raises(ToolError, match="threshold profile"):
            self.passing(threshold_profile=None)

    def test_a_failure_needs_neither_unit_nor_profile(self) -> None:
        Evidence(
            gate_id="geometry.supports", status=EvidenceStatus.FAIL, state_hash=VERSION,
            backend="isaac", tool_version="geometry/1", method="support_polygon_coverage",
            detail="support coverage 12% of the object footprint",
        )

    def test_satisfies_requires_pass_and_matching_hash(self) -> None:
        evidence = self.passing()
        assert evidence.satisfies(VERSION)
        assert not evidence.satisfies(content_hash({"scene": "b"}))
        assert not self.passing(status=EvidenceStatus.FAIL, unit=None, threshold_profile=None) \
            .satisfies(VERSION)

    def test_round_trips_through_dict(self) -> None:
        evidence = self.passing(witness_refs=("support_polygon.json",), timestamp="2026-09-15T00:00:00Z")
        assert Evidence.from_dict(evidence.to_dict()).to_dict() == evidence.to_dict()

    def test_rejects_unknown_status_on_parse(self) -> None:
        payload = self.passing().to_dict()
        payload["status"] = "probably_fine"
        with pytest.raises(ToolError, match="unknown evidence status"):
            Evidence.from_dict(payload)

    def test_parse_requires_the_identifying_fields(self) -> None:
        payload = self.passing().to_dict()
        del payload["state_hash"]
        with pytest.raises(ToolError, match="missing"):
            Evidence.from_dict(payload)


class TestLedger:
    def test_absent_evidence_reports_not_run(self) -> None:
        # A gate nobody executed must not read as satisfied or as failed.
        ledger = EvidenceLedger()
        assert ledger.status_for("physics.settle_stability") is EvidenceStatus.NOT_RUN
        assert not ledger.satisfies("physics.settle_stability", VERSION)

    def test_recorded_pass_satisfies(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(TestEvidence.passing())
        assert ledger.satisfies("geometry.supports", VERSION)

    def test_recording_replaces_earlier_evidence_for_the_gate(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(TestEvidence.passing())
        ledger.record(TestEvidence.passing(measured_value=0.02))
        assert ledger.get("geometry.supports").measured_value == 0.02

    def test_latest_evidence_from_an_old_version_is_stale(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(TestEvidence.passing(state_hash=content_hash({"scene": "old"})))
        assert ledger.stale_gate_ids(VERSION) == ("geometry.supports",)

    def test_serialises_sorted_by_gate(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(TestEvidence.passing())
        ledger.record(TestEvidence.passing(gate_id="physics.clock_advanced"))
        assert [entry["gate_id"] for entry in ledger.to_list()] == [
            "geometry.supports", "physics.clock_advanced",
        ]
