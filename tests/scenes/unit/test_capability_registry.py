# SPDX-License-Identifier: Apache-2.0
"""Capability honesty.

The registry exists to stop two specific overclaims: a catalogue declaring a physics type and
the system reporting it as verified, and an untested backend being treated as working. Both
are tested here, along with the rule that approximation has to be opted into at request time
rather than discovered in the result.
"""

from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

from pathlib import Path

import pytest

from rfw_w.scenes.assets.capabilities import (
    CapabilityEntry,
    CapabilityError,
    CapabilityRegistry,
    Subject,
    SupportLevel,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = data_path('configs') / 'capabilities_v1.json'


def entry(subject: Subject = Subject.RIGID, **overrides: object) -> CapabilityEntry:
    payload: dict[str, object] = {
        "backend": "isaac",
        "subject": subject,
        "level": SupportLevel.SUPPORTED,
        "tested_scope": "probe fixture",
        "evidence_ref": "outputs/_smoke/runtime_supported_verified/report.json",
    }
    payload.update(overrides)
    return CapabilityEntry(**payload)  # type: ignore[arg-type]


class TestEntryRules:
    def test_supported_requires_evidence(self) -> None:
        """A supported claim must point at a run that demonstrated it."""
        with pytest.raises(CapabilityError, match="evidence_ref"):
            entry(evidence_ref=None)

    def test_tested_scope_is_always_required(self) -> None:
        # Even an unsupported entry records what was examined, so a later reviewer can tell
        # "we tried and it failed" from "we never looked".
        with pytest.raises(CapabilityError, match="tested_scope"):
            entry(level=SupportLevel.UNSUPPORTED, evidence_ref=None, tested_scope="")

    def test_approximate_must_name_its_approximation(self) -> None:
        with pytest.raises(CapabilityError, match="approximation"):
            entry(level=SupportLevel.APPROXIMATE, evidence_ref=None, approximations=())

    def test_approximate_entries_are_accepted_with_a_reason(self) -> None:
        approximate = entry(
            level=SupportLevel.APPROXIMATE,
            evidence_ref=None,
            approximations=("convex hull per mesh island instead of an exact mesh",),
        )
        assert not approximate.level.is_exact

    def test_supported_must_not_also_be_approximate(self) -> None:
        # Otherwise a lossy implementation could be labelled exact.
        with pytest.raises(CapabilityError, match="both supported and approximate"):
            entry(level=SupportLevel.SUPPORTED, approximations=("hull proxy",))

    def test_unsupported_needs_no_evidence(self) -> None:
        entry(level=SupportLevel.UNSUPPORTED, evidence_ref=None)


class TestResolution:
    def registry(self, **entries: SupportLevel) -> CapabilityRegistry:
        built: dict[tuple[str, Subject], CapabilityEntry] = {}
        for key, level in entries.items():
            backend, subject_name = key.split("__")
            subject = Subject(subject_name)
            built[(backend, subject)] = entry(
                subject=subject,
                backend=backend,
                level=level,
                evidence_ref="evidence.json" if level is SupportLevel.SUPPORTED else None,
                approximations=("hull proxy instead of exact mesh",)
                if level is SupportLevel.APPROXIMATE
                else (),
            )
        return CapabilityRegistry(entries=built)

    def test_unlisted_combination_is_untested_not_supported(self) -> None:
        registry = self.registry(isaac__rigid=SupportLevel.SUPPORTED)
        assert registry.level_for("isaac", Subject.ARTICULATED) is SupportLevel.UNTESTED
        assert not registry.usable("isaac", Subject.ARTICULATED)

    def test_exact_support_is_usable(self) -> None:
        registry = self.registry(isaac__rigid=SupportLevel.SUPPORTED)
        assert registry.usable("isaac", Subject.RIGID)

    def test_approximate_is_not_usable_by_default(self) -> None:
        registry = self.registry(isaac__CAPPING=SupportLevel.APPROXIMATE)
        assert not registry.usable("isaac", Subject.CAPPING)

    def test_approximate_is_usable_once_opted_in(self) -> None:
        registry = self.registry(isaac__CAPPING=SupportLevel.APPROXIMATE)
        opted_in = registry.with_approximate_allowed(True)
        assert opted_in.usable("isaac", Subject.CAPPING)

    def test_unsupported_stays_unsupported_even_when_approximation_is_allowed(self) -> None:
        # Approximation is a fidelity trade, not a way to enable a missing implementation.
        registry = self.registry(isaac__deformable=SupportLevel.UNSUPPORTED)
        assert not registry.with_approximate_allowed(True).usable("isaac", Subject.DEFORMABLE)


class TestPreflight:
    def registry(self) -> CapabilityRegistry:
        return CapabilityRegistry(
            entries={
                ("isaac", Subject.RIGID): entry(),
                ("isaac", Subject.ON): entry(subject=Subject.ON),
                ("mujoco", Subject.RIGID): entry(
                    backend="mujoco", level=SupportLevel.UNTESTED, evidence_ref=None
                ),
            }
        )

    def test_satisfied_when_every_required_pair_is_exact(self) -> None:
        report = self.registry().preflight(requested_backends=["isaac"], requested_subjects=[Subject.RIGID, Subject.ON])
        assert report.satisfied
        assert report.gaps == ()

    def test_one_untested_required_backend_fails_the_request(self) -> None:
        """Exact semantics: a dual-backend request does not silently degrade to one."""
        report = self.registry().preflight(
            requested_backends=["isaac", "mujoco"], requested_subjects=[Subject.RIGID]
        )
        assert not report.satisfied
        assert [gap.backend for gap in report.gaps] == ["mujoco"]
        assert report.gaps[0].level is SupportLevel.UNTESTED

    def test_gap_explains_itself(self) -> None:
        report = self.registry().preflight(
            requested_backends=["isaac"], requested_subjects=[Subject.DEFORMABLE]
        )
        assert "never been exercised" in report.gaps[0].reason

    def test_requires_a_backend(self) -> None:
        with pytest.raises(CapabilityError):
            self.registry().preflight(requested_backends=[], requested_subjects=[Subject.RIGID])

    def test_report_serialises(self) -> None:
        payload = self.registry().preflight(
            requested_backends=["isaac", "mujoco"], requested_subjects=[Subject.RIGID]
        ).to_dict()
        assert payload["satisfied"] is False
        assert payload["unsupported"] is True
        assert payload["resolved"]["isaac/rigid"] == "supported"


class TestShippedConfig:
    def test_config_loads(self) -> None:
        assert CapabilityRegistry.from_path(CONFIG).backends == ("isaac", "mujoco")

    def test_every_supported_entry_has_evidence(self) -> None:
        for (backend, subject), ship_entry in CapabilityRegistry.from_path(CONFIG).entries.items():
            if ship_entry.level is SupportLevel.SUPPORTED:
                assert ship_entry.evidence_ref, f"{backend}/{subject.value}"

    def test_config_rejects_a_wrong_schema_version(self) -> None:
        with pytest.raises(CapabilityError, match="schema_version"):
            CapabilityRegistry.from_dict({"schema_version": "pvscene.capabilities/v9", "backends": {}})

    def test_articulated_is_not_claimed_supported(self) -> None:
        # No verified articulated asset exists locally, so claiming support would be a
        # declaration dressed up as a result.
        registry = CapabilityRegistry.from_path(CONFIG)
        assert registry.level_for("isaac", Subject.ARTICULATED) is SupportLevel.UNSUPPORTED

    def test_single_backend_rigid_request_is_servable(self) -> None:
        report = CapabilityRegistry.from_path(CONFIG).preflight(
            requested_backends=["isaac"], requested_subjects=[Subject.RIGID]
        )
        assert report.satisfied
