# SPDX-License-Identifier: Apache-2.0
"""The evidence ledger: what it keeps, and what a lookup is allowed to mean.

The property under test is that a piece of evidence belongs to the scene it was measured on. The
ledger used to hold one entry per gate with the latest write winning, which made a repair that
advanced the scene *destroy* the evidence for the version it advanced from -- so a rollback landed
on a scene with nothing recorded for it, every gate read as stale, and the run reported `draft`
however well the restored scene had measured.

Both directions are checked here: evidence for another scene must not satisfy this one, and evidence
for this one must not be thrown away because a later scene was measured.
"""

from __future__ import annotations

from rfw_w.scenes.contracts.tool import Evidence, EvidenceLedger, EvidenceStatus

VERSION_A = "a" * 64
VERSION_B = "b" * 64
GATE = "geometry.in_bounds"


def evidence(
    status: EvidenceStatus, *, version: str = VERSION_A, gate: str = GATE, detail: str = ""
) -> Evidence:
    measured = 0.001 if status is EvidenceStatus.PASS else None
    return Evidence(
        gate_id=gate, status=status, state_hash=version,
        backend="isaac", tool_version="test/1", method="test fixture",
        measured_value=measured, unit="m" if measured is not None else None,
        threshold_profile="engineering_v1" if status is EvidenceStatus.PASS else None,
        detail=detail,
    )


class TestVersionedLookup:
    def test_measuring_a_later_scene_does_not_discard_the_earlier_one(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(evidence(EvidenceStatus.PASS))
        ledger.record(evidence(EvidenceStatus.FAIL, version=VERSION_B))

        assert ledger.get(GATE, VERSION_A).status is EvidenceStatus.PASS
        assert ledger.get(GATE, VERSION_B).status is EvidenceStatus.FAIL

    def test_a_pass_belonging_to_another_scene_does_not_satisfy_this_one(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(evidence(EvidenceStatus.PASS, version=VERSION_B))

        assert ledger.satisfies(GATE, VERSION_B)
        assert not ledger.satisfies(GATE, VERSION_A)

    def test_the_latest_entry_is_readable_without_naming_a_version(self) -> None:
        # For questions about the history -- "has this been measured", "what did the last critique
        # say" -- not for questions about the current scene.
        ledger = EvidenceLedger()
        ledger.record(evidence(EvidenceStatus.PASS))
        ledger.record(evidence(EvidenceStatus.FAIL, version=VERSION_B))
        assert ledger.latest_for(GATE).status is EvidenceStatus.FAIL

    def test_an_unmeasured_gate_has_no_latest(self) -> None:
        assert EvidenceLedger().latest_for(GATE) is None

    def test_a_gate_measured_on_two_scenes_reports_both(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(evidence(EvidenceStatus.PASS))
        ledger.record(evidence(EvidenceStatus.FAIL, version=VERSION_B))
        assert ledger.versions_for(GATE) == (VERSION_A, VERSION_B)
        assert ledger.gate_ids() == (GATE,)

    def test_stale_means_measured_elsewhere_and_not_here(self) -> None:
        ledger = EvidenceLedger()
        ledger.record(evidence(EvidenceStatus.PASS, version=VERSION_B))
        ledger.record(evidence(EvidenceStatus.PASS, gate="geometry.supports"))

        assert ledger.stale_gate_ids(VERSION_A) == (GATE,)
        # Measured on this scene, so not stale, even though another scene has an entry too.
        ledger.record(evidence(EvidenceStatus.PASS, version=VERSION_A))
        assert ledger.stale_gate_ids(VERSION_A) == ()

    def test_a_gate_nobody_measured_is_not_run_rather_than_stale(self) -> None:
        ledger = EvidenceLedger()
        assert ledger.status_for("geometry.supports") is EvidenceStatus.NOT_RUN
        assert ledger.stale_gate_ids(VERSION_A) == ()


class TestRecordOfTheRun:
    def test_every_version_survives_into_the_serialised_form(self) -> None:
        """The manifest carries this list, so keeping every version is also what lets a reviewer
        read the run's history instead of its last state."""
        ledger = EvidenceLedger()
        ledger.record(evidence(EvidenceStatus.PASS))
        ledger.record(evidence(EvidenceStatus.FAIL, version=VERSION_B))

        payloads = ledger.to_list()
        assert len(payloads) == 2
        assert {item["state_hash"] for item in payloads} == {VERSION_A, VERSION_B}

    def test_a_round_trip_through_the_serialised_form_keeps_the_versions_apart(self) -> None:
        from rfw_w.scenes.contracts.tool import Evidence as EvidenceType

        ledger = EvidenceLedger()
        ledger.record(evidence(EvidenceStatus.PASS))
        ledger.record(evidence(EvidenceStatus.FAIL, version=VERSION_B))

        rebuilt = EvidenceLedger()
        for payload in ledger.to_list():
            rebuilt.record(EvidenceType.from_dict(payload))

        assert rebuilt.entries == ledger.entries
        assert rebuilt.satisfies(GATE, VERSION_A)
        assert not rebuilt.satisfies(GATE, VERSION_B)
