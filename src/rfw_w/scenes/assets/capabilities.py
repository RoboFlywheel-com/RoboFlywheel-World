# SPDX-License-Identifier: Apache-2.0
"""Capability registry: what each backend is actually known to do.

The registry exists to keep two claims apart that are easy to merge and expensive to merge:

* a catalogue **declares** a physics type for an asset;
* a backend has been **verified** to handle that physics type or relation.

The acceptance manual is explicit that the first does not imply the second. So an entry is
never ``supported`` without an ``evidence_ref`` pointing at something that was actually run,
an untested combination is reported as ``untested`` rather than assumed to work, and the
default resolution is *exact*: if any requested backend does not support a required subject,
the whole request is unsupported rather than silently degraded.

Approximation is available but must be asked for. A backend that can only approximate a
contact relation is not the same as one that handles it, and the difference has to appear in
the request rather than in the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "pvscene.capabilities/v1"


class CapabilityError(ValueError):
    """The capability registry or a capability query contradicts the contract."""


class SupportLevel(str, Enum):
    """How well a backend handles a subject."""

    SUPPORTED = "supported"
    APPROXIMATE = "approximate"
    UNSUPPORTED = "unsupported"
    UNTESTED = "untested"

    @property
    def is_exact(self) -> bool:
        return self is SupportLevel.SUPPORTED


class Subject(str, Enum):
    """The physics types and relations a capability entry can be about."""

    RIGID = "rigid"
    ARTICULATED = "articulated"
    DEFORMABLE = "deformable"
    ON = "ON"
    INSIDE = "INSIDE"
    AGAINST_WALL = "AGAINST_WALL"
    FACING = "FACING"
    NEAR = "NEAR"
    OPENABLE = "OPENABLE"
    CLOSABLE = "CLOSABLE"
    CAPPING = "CAPPING"
    SLOTTING = "SLOTTING"
    HOOKING = "HOOKING"
    PEGGING = "PEGGING"
    SOFT_INSIDE = "SOFT_INSIDE"
    DRAPED_ON = "DRAPED_ON"


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    """One (backend, subject) capability claim.

    ``tested_scope`` is required even for a supported entry: "works" without a scope is the
    claim this registry exists to prevent, and the scope is what a reviewer checks when a
    result is contradicted.
    """

    backend: str
    subject: Subject
    level: SupportLevel
    tested_scope: str
    approximations: tuple[str, ...] = ()
    evidence_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.backend:
            raise CapabilityError("capability entry must name a backend")
        if not self.tested_scope:
            raise CapabilityError(
                f"{self.backend}/{self.subject.value}: tested_scope is required even when "
                "unsupported, so the claim records what was actually examined"
            )
        if self.level is SupportLevel.SUPPORTED and not self.evidence_ref:
            raise CapabilityError(
                f"{self.backend}/{self.subject.value} is claimed supported without an "
                "evidence_ref; a supported claim must point at a run that demonstrated it"
            )
        if self.level is SupportLevel.APPROXIMATE and not self.approximations:
            raise CapabilityError(
                f"{self.backend}/{self.subject.value} is approximate but names no approximation; "
                "the loss of fidelity has to be recorded"
            )
        if self.level is SupportLevel.SUPPORTED and self.approximations:
            raise CapabilityError(
                f"{self.backend}/{self.subject.value} is both supported and approximate; "
                "use the approximate level so the request has to opt in"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "level": self.level.value,
            "tested_scope": self.tested_scope,
        }
        if self.approximations:
            payload["approximations"] = list(self.approximations)
        if self.evidence_ref:
            payload["evidence_ref"] = self.evidence_ref
        return payload

    @classmethod
    def from_dict(cls, backend: str, subject: Subject, payload: Mapping[str, Any]) -> CapabilityEntry:
        try:
            level = SupportLevel(payload["level"])
        except KeyError as error:
            raise CapabilityError(f"{backend}/{subject.value}: missing level") from error
        except ValueError as error:
            raise CapabilityError(f"{backend}/{subject.value}: unknown level {payload['level']!r}") from error
        approximations = payload.get("approximations", [])
        if isinstance(approximations, (str, bytes)) or not isinstance(approximations, Sequence):
            raise CapabilityError(f"{backend}/{subject.value}: approximations must be a list")
        return cls(
            backend=backend,
            subject=subject,
            level=level,
            tested_scope=str(payload.get("tested_scope", "")),
            approximations=tuple(str(item) for item in approximations),
            evidence_ref=payload.get("evidence_ref"),
        )


@dataclass(frozen=True, slots=True)
class CapabilityGap:
    """A requested capability that cannot be met, in a form the preflight can print."""

    backend: str
    subject: Subject
    level: SupportLevel
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "subject": self.subject.value,
            "level": self.level.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """The answer to "can this request be served at all", before any work is done.

    Reported before room generation rather than discovered afterwards, which is what the
    architecture's ``preflight`` stage is for: an unsupported request must fail cheaply and
    name its gap, not produce a scene that quietly omits the unsupported part.
    """

    requested_backends: tuple[str, ...]
    requested_subjects: tuple[Subject, ...]
    allow_approximate: bool
    satisfied: bool
    gaps: tuple[CapabilityGap, ...]
    resolved: Mapping[str, SupportLevel]

    @property
    def unsupported(self) -> bool:
        return not self.satisfied

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_backends": list(self.requested_backends),
            "requested_subjects": [subject.value for subject in self.requested_subjects],
            "allow_approximate": self.allow_approximate,
            "satisfied": self.satisfied,
            "unsupported": self.unsupported,
            "gaps": [gap.to_dict() for gap in self.gaps],
            "resolved": {key: level.value for key, level in sorted(self.resolved.items())},
        }


@dataclass(frozen=True, slots=True)
class CapabilityRegistry:
    """Every capability claim the system is willing to make.

    ``allow_approximate`` is the request-level opt-in from §11: an approximate capability is
    only usable when the request says so, because the request is where the loss of fidelity
    becomes the caller's informed decision rather than a surprise in the result.
    """

    entries: Mapping[tuple[str, Subject], CapabilityEntry]
    version: str = SCHEMA_VERSION
    allow_approximate: bool = False

    def __post_init__(self) -> None:
        if not self.entries:
            raise CapabilityError("capability registry must contain at least one entry")
        for (backend, subject), entry in self.entries.items():
            if entry.backend != backend or entry.subject is not subject:
                raise CapabilityError(
                    f"registry key {backend}/{subject.value} disagrees with entry "
                    f"{entry.backend}/{entry.subject.value}"
                )

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(sorted({backend for backend, _ in self.entries}))

    def entry(self, backend: str, subject: Subject) -> CapabilityEntry | None:
        return self.entries.get((backend, subject))

    def level_for(self, backend: str, subject: Subject) -> SupportLevel:
        """The support level, defaulting to ``untested``.

        An unlisted combination is unknown, not working. Defaulting to supported is how a
        backend nobody has exercised gets reported as passing.
        """
        found = self.entry(backend, subject)
        return SupportLevel.UNTESTED if found is None else found.level

    def usable(self, backend: str, subject: Subject) -> bool:
        """Whether this combination may be relied on for the current request."""
        level = self.level_for(backend, subject)
        if level.is_exact:
            return True
        return level is SupportLevel.APPROXIMATE and self.allow_approximate

    def preflight(
        self,
        *,
        requested_backends: Sequence[str],
        requested_subjects: Iterable[Subject],
    ) -> CapabilityReport:
        """Resolve a request's required capabilities into a satisfied/gap verdict.

        Exact semantics: one required backend failing a required subject fails the request.
        Returning a partial scene and calling it a success is the outcome this avoids.
        """
        backends = tuple(dict.fromkeys(requested_backends))
        if not backends:
            raise CapabilityError("preflight requires at least one requested backend")
        subjects = tuple(dict.fromkeys(requested_subjects))

        gaps: list[CapabilityGap] = []
        resolved: dict[str, SupportLevel] = {}
        for backend in backends:
            for subject in subjects:
                level = self.level_for(backend, subject)
                resolved[f"{backend}/{subject.value}"] = level
                if self.usable(backend, subject):
                    continue
                if level is SupportLevel.APPROXIMATE:
                    reason = (
                        "only an approximate implementation exists and the request did not opt "
                        f"in (approximations: {', '.join(self.entry(backend, subject).approximations)})"
                    )
                elif level is SupportLevel.UNTESTED:
                    reason = "no evidence of this combination; it has never been exercised"
                else:
                    reason = "declared unsupported"
                gaps.append(CapabilityGap(backend=backend, subject=subject, level=level, reason=reason))

        return CapabilityReport(
            requested_backends=backends,
            requested_subjects=subjects,
            allow_approximate=self.allow_approximate,
            satisfied=not gaps,
            gaps=tuple(gaps),
            resolved=resolved,
        )

    def to_dict(self) -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}
        for (backend, subject), entry in sorted(self.entries.items(), key=lambda item: (item[0][0], item[0][1].value)):
            grouped.setdefault(backend, {})[subject.value] = entry.to_dict()
        return {
            "schema_version": self.version,
            "allow_approximate": self.allow_approximate,
            "backends": grouped,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CapabilityRegistry:
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise CapabilityError(f"unsupported capabilities schema_version {version!r}")
        backends = payload.get("backends")
        if not isinstance(backends, Mapping):
            raise CapabilityError("capabilities payload must contain a backends mapping")
        entries: dict[tuple[str, Subject], CapabilityEntry] = {}
        for backend, subjects in backends.items():
            if not isinstance(subjects, Mapping):
                raise CapabilityError(f"backend {backend!r} must map subjects to entries")
            for raw_subject, entry_payload in subjects.items():
                try:
                    subject = Subject(raw_subject)
                except ValueError as error:
                    raise CapabilityError(f"{backend}: unknown subject {raw_subject!r}") from error
                entries[(str(backend), subject)] = CapabilityEntry.from_dict(
                    str(backend), subject, entry_payload
                )
        return cls(
            entries=entries,
            version=version,
            allow_approximate=bool(payload.get("allow_approximate", False)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> CapabilityRegistry:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def with_approximate_allowed(self, allowed: bool) -> CapabilityRegistry:
        return CapabilityRegistry(entries=self.entries, version=self.version, allow_approximate=allowed)
