# SPDX-License-Identifier: Apache-2.0
"""ToolRequest, ToolResult and Evidence: how every tool talks to the orchestrator.

The types encode three rules from architecture §5.4 and §10.

**Read-only tools do not propose deltas.** ``ToolResult.read_only`` exists so that a
validator returning a proposed scene change is a contract error rather than a convention
someone forgets. Only the orchestrator commits, and only after validating on an isolated
state.

**Failure codes are typed, and a timeout is not a failure.** ``timeout``, ``unsupported``
and ``failed`` lead to different orchestrator decisions: a timeout may be retried within an
infrastructure budget, ``unsupported`` may end the request at its requested readiness, and
``failed`` feeds the repair policy. Collapsing them into a bool loses what the scheduler
needs.

**Evidence is only pass/fail/not_run/unsupported, and it carries the state hash it was
measured against.** ``not_run`` is a first-class outcome: it is what an unexecuted check
reports, and it never satisfies a gate. A missing run and a passing run must not be
representable by the same value, because that is how "no evidence" becomes "verified".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from rfw_w.scenes.contracts.hashing import is_digest

SCHEMA_VERSION = "pvscene.tool/v3"


class ToolError(ValueError):
    """The tool exchange contradicts the contract."""


class ToolStatus(str, Enum):
    """Outcome of running a tool."""

    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"


class EvidenceStatus(str, Enum):
    """Outcome of a single gate or predicate check.

    ``NOT_RUN`` is deliberately distinct from ``FAIL``. A gate that was never executed has
    not failed, and must not be silently upgraded to a pass either.
    """

    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"
    UNSUPPORTED = "unsupported"

    @property
    def satisfies_gate(self) -> bool:
        """Only ``PASS`` satisfies a gate. Everything else leaves it unsatisfied."""
        return self is EvidenceStatus.PASS


class FailureCode(str, Enum):
    """Typed failure reasons. The repair policy switches on these, not on free text."""

    # Retrieval and assets
    ASSET_IO_FAILURE = "ASSET_IO_FAILURE"
    ASSET_UNAVAILABLE = "ASSET_UNAVAILABLE"
    PAIR_INCOMPATIBLE = "PAIR_INCOMPATIBLE"
    ASSET_UNCERTIFIED = "ASSET_UNCERTIFIED"
    # Geometry and function
    GEOMETRY_PENETRATION = "GEOMETRY_PENETRATION"
    GEOMETRY_OUT_OF_BOUNDS = "GEOMETRY_OUT_OF_BOUNDS"
    SUPPORT_NOT_FOUND = "SUPPORT_NOT_FOUND"
    NO_PLACEMENT_CAPACITY = "NO_PLACEMENT_CAPACITY"
    FUNCTION_PREDICATE_FAILED = "FUNCTION_PREDICATE_FAILED"
    CLEARANCE_VIOLATION = "CLEARANCE_VIOLATION"
    # Physics
    PHYSICS_OBJECT_DROPPED = "PHYSICS_OBJECT_DROPPED"
    PHYSICS_UNSTABLE = "PHYSICS_UNSTABLE"
    PHYSICS_CLOCK_STALLED = "PHYSICS_CLOCK_STALLED"
    PHYSICS_NONFINITE_STATE = "PHYSICS_NONFINITE_STATE"
    # Rendering and visual
    RENDER_FAILED = "RENDER_FAILED"
    RENDER_STALE = "RENDER_STALE"
    VISUAL_CRITIQUE_INVALID = "VISUAL_CRITIQUE_INVALID"
    VISUAL_GATE_FAILED = "VISUAL_GATE_FAILED"
    # Orchestration
    HARD_GOAL_DROPPED = "HARD_GOAL_DROPPED"
    PATCH_INVALID = "PATCH_INVALID"
    NO_PROGRESS = "NO_PROGRESS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CANDIDATES_EXHAUSTED = "CANDIDATES_EXHAUSTED"
    MODEL_CALL_FAILED = "MODEL_CALL_FAILED"
    PLAN_PROPOSAL_INVALID = "PLAN_PROPOSAL_INVALID"
    # Export and backend
    EXPORT_MISMATCH = "EXPORT_MISMATCH"
    BACKEND_UNSUPPORTED = "BACKEND_UNSUPPORTED"
    # Contract
    SCHEMA_INVALID = "SCHEMA_INVALID"
    PRECONDITION_UNMET = "PRECONDITION_UNMET"


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """A unit of work handed to a tool service."""

    request_id: str
    tool_version: str
    scene_version: str
    seed: int
    deadline_s: float
    scope_ids: tuple[str, ...] = ()
    input_refs: tuple[str, ...] = ()
    constraints_ref: str | None = None
    max_candidates: int | None = None
    idempotency_key: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ToolError("request_id must be non-empty")
        if not self.tool_version:
            raise ToolError("tool_version must be non-empty; a result is only interpretable with it")
        if not is_digest(self.scene_version):
            raise ToolError(f"scene_version must be a scene hash, got {self.scene_version!r}")
        if self.deadline_s <= 0:
            raise ToolError(f"deadline_s must be positive, got {self.deadline_s}")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ToolError(f"max_candidates must be at least 1, got {self.max_candidates}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "tool_version": self.tool_version,
            "scene_version": self.scene_version,
            "seed": self.seed,
            "deadline_s": self.deadline_s,
            "scope_ids": list(self.scope_ids),
            "input_refs": list(self.input_refs),
        }
        if self.constraints_ref is not None:
            payload["constraints_ref"] = self.constraints_ref
        if self.max_candidates is not None:
            payload["max_candidates"] = self.max_candidates
        if self.idempotency_key is not None:
            payload["idempotency_key"] = self.idempotency_key
        return payload


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool service returns.

    ``base_scene_version`` records the state the tool ran against. The orchestrator compares
    it before committing: a delta computed against a state that has since moved must not be
    applied, which is the optimistic-concurrency check from §10.
    """

    request_id: str
    status: ToolStatus
    tool_version: str
    base_scene_version: str
    read_only: bool
    elapsed_ms: float
    touched_ids: tuple[str, ...] = ()
    invalidated_evidence: tuple[str, ...] = ()
    proposed_delta: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    failure_code: FailureCode | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ToolError("result must carry the request_id it answers")
        if not is_digest(self.base_scene_version):
            raise ToolError(f"base_scene_version must be a scene hash, got {self.base_scene_version!r}")
        if self.elapsed_ms < 0:
            raise ToolError(f"elapsed_ms must be non-negative, got {self.elapsed_ms}")
        if self.read_only and self.proposed_delta is not None:
            raise ToolError(
                f"read-only tool {self.tool_version} proposed a delta; only the orchestrator "
                "commits, and it does so from a writer's proposed delta on an isolated state"
            )
        if self.status is not ToolStatus.PASSED and self.failure_code is None:
            raise ToolError(
                f"status {self.status.value} requires a typed failure_code; a bare failure "
                "cannot be routed by the repair policy"
            )
        if self.status is ToolStatus.PASSED and self.failure_code is not None:
            raise ToolError(f"status passed must not carry failure_code {self.failure_code.value}")
        if self.retryable and self.status is ToolStatus.PASSED:
            raise ToolError("a passed result is not retryable")

    @property
    def is_infrastructure_failure(self) -> bool:
        """Failures the scheduler retries on its own budget instead of asking for a repair.

        A timeout or an I/O failure says nothing about the scene, so sending it to the model
        as a semantic problem wastes a model request and can lead to an unnecessary rewrite.
        """
        return self.status is ToolStatus.TIMEOUT or self.failure_code in (
            FailureCode.ASSET_IO_FAILURE,
            FailureCode.RENDER_FAILED,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "status": self.status.value,
            "tool_version": self.tool_version,
            "base_scene_version": self.base_scene_version,
            "read_only": self.read_only,
            "elapsed_ms": self.elapsed_ms,
            "touched_ids": list(self.touched_ids),
            "invalidated_evidence": list(self.invalidated_evidence),
            "metrics": dict(self.metrics),
            "evidence_refs": list(self.evidence_refs),
            "retryable": self.retryable,
        }
        if self.proposed_delta is not None:
            payload["proposed_delta"] = dict(self.proposed_delta)
        if self.failure_code is not None:
            payload["failure_code"] = self.failure_code.value
        return payload


@dataclass(frozen=True, slots=True)
class Evidence:
    """A measured gate outcome, bound to the exact scene state it was measured on.

    The hash binding is what makes stale evidence detectable. A render approved for version
    A must not approve version B, which is why ``state_hash`` is required and why the
    readiness resolver compares it rather than trusting the status alone.
    """

    gate_id: str
    status: EvidenceStatus
    state_hash: str
    backend: str
    tool_version: str
    method: str
    measured_value: float | None = None
    unit: str | None = None
    threshold_profile: str | None = None
    witness_refs: tuple[str, ...] = ()
    timestamp: str = ""
    backend_version: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.gate_id:
            raise ToolError("evidence must name the gate it belongs to")
        if not is_digest(self.state_hash):
            raise ToolError(
                f"evidence {self.gate_id} must carry the scene hash it was measured on; "
                "without it a stale approval is indistinguishable from a current one"
            )
        if not self.backend:
            raise ToolError(f"evidence {self.gate_id} must name the backend")
        if not self.tool_version:
            raise ToolError(f"evidence {self.gate_id} must record the tool version")
        if not self.method:
            raise ToolError(f"evidence {self.gate_id} must record how it was measured")
        # A pass that reports a number must say what unit that number is in: a bare 0.002
        # cannot be compared against a threshold.
        if self.status is EvidenceStatus.PASS and self.measured_value is not None and not self.unit:
            raise ToolError(f"evidence {self.gate_id} reports a measured value without a unit")
        if self.status is EvidenceStatus.PASS and self.threshold_profile is None:
            raise ToolError(
                f"evidence {self.gate_id} passed without naming the threshold profile it "
                "passed against; thresholds are configured and calibrated, not universal"
            )

    def is_current_for(self, scene_version: str) -> bool:
        """Whether this evidence belongs to the given scene version."""
        return self.state_hash == scene_version

    def satisfies(self, scene_version: str) -> bool:
        """A gate is satisfied only by a current pass."""
        return self.is_current_for(scene_version) and self.status.satisfies_gate

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "state_hash": self.state_hash,
            "backend": self.backend,
            "tool_version": self.tool_version,
            "method": self.method,
            "witness_refs": list(self.witness_refs),
        }
        if self.measured_value is not None:
            payload["measured_value"] = self.measured_value
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.threshold_profile is not None:
            payload["threshold_profile"] = self.threshold_profile
        if self.timestamp:
            payload["timestamp"] = self.timestamp
        if self.backend_version:
            payload["backend_version"] = self.backend_version
        if self.detail:
            payload["detail"] = self.detail
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Evidence:
        required = ("gate_id", "status", "state_hash", "backend", "tool_version", "method")
        for key in required:
            if key not in payload:
                raise ToolError(f"evidence is missing {key!r}: {dict(payload)}")
        try:
            status = EvidenceStatus(payload["status"])
        except ValueError as error:
            raise ToolError(f"unknown evidence status {payload['status']!r}") from error
        witnesses = payload.get("witness_refs", [])
        if isinstance(witnesses, (str, bytes)) or not isinstance(witnesses, Sequence):
            raise ToolError("witness_refs must be a list")
        return cls(
            gate_id=str(payload["gate_id"]),
            status=status,
            state_hash=str(payload["state_hash"]),
            backend=str(payload["backend"]),
            tool_version=str(payload["tool_version"]),
            method=str(payload["method"]),
            measured_value=None if payload.get("measured_value") is None else float(payload["measured_value"]),
            unit=payload.get("unit"),
            threshold_profile=payload.get("threshold_profile"),
            witness_refs=tuple(str(item) for item in witnesses),
            timestamp=str(payload.get("timestamp", "")),
            backend_version=str(payload.get("backend_version", "")),
            detail=str(payload.get("detail", "")),
        )


@dataclass(slots=True)
class EvidenceLedger:
    """All evidence collected so far, keyed by gate **and scene version**.

    Keyed by gate alone the ledger held one entry per gate with the latest write winning, so a
    repair that advanced the scene and re-measured *destroyed* the evidence for the version it
    advanced from. A rollback to that version then had nothing to resolve against: every measured
    gate came back "measured on X but the scene is now Y", and a run whose restored scene had
    measured clean still reported `draft` with nothing measured at all. The version is part of what
    a piece of evidence *is*, not a label on it, so it is part of the key.

    Keeping every version is also what makes a delivery a record of the run rather than a snapshot
    of its last state: every critique, every render's verdict and every geometry gate survives for
    the scene it described, and a reviewer can see the ladder's history instead of its endpoint.

    A missing key and a ``not_run`` entry remain different situations, and both are representable.
    """

    entries: dict[tuple[str, str], Evidence] = field(default_factory=dict)

    def record(self, evidence: Evidence) -> None:
        self.entries[(evidence.gate_id, evidence.state_hash)] = evidence

    def get(self, gate_id: str, scene_version: str | None = None) -> Evidence | None:
        """The evidence for one gate.

        With ``scene_version``: that gate on that scene, and nothing else. A question about whether
        a gate holds is a question about a version and has to say which one.

        Without it: whichever version was recorded last, which is only right for a question about
        the *history* -- "has this been measured at all", "what did the last critique say" -- and
        never for one about the current scene.
        """
        if scene_version is not None:
            return self.entries.get((gate_id, scene_version))
        return self.latest_for(gate_id)

    def latest_for(self, gate_id: str) -> Evidence | None:
        """The most recently recorded entry for a gate, whatever scene it describes."""
        for (name, _version), entry in reversed(self.entries.items()):
            if name == gate_id:
                return entry
        return None

    def versions_for(self, gate_id: str) -> tuple[str, ...]:
        """Every scene this gate was measured on, oldest first."""
        return tuple(version for (name, version) in self.entries if name == gate_id)

    def gate_ids(self) -> tuple[str, ...]:
        """Every gate the ledger holds anything for, in a stable order."""
        return tuple(sorted({name for name, _version in self.entries}))

    def status_for(self, gate_id: str, scene_version: str | None = None) -> EvidenceStatus:
        """``NOT_RUN`` when a gate has no evidence at all.

        Absent evidence is reported as not-run rather than as a pass or a failure, so that
        a gate nobody executed cannot be mistaken for one that succeeded.
        """
        entry = self.get(gate_id, scene_version)
        return EvidenceStatus.NOT_RUN if entry is None else entry.status

    def satisfies(self, gate_id: str, scene_version: str) -> bool:
        entry = self.entries.get((gate_id, scene_version))
        # The version is the key, so it cannot disagree with the entry: only the status is left to
        # check, and a pass recorded for another scene can no longer be read as a pass for this one.
        return entry is not None and entry.status.satisfies_gate

    def stale_gate_ids(self, scene_version: str) -> tuple[str, ...]:
        """Gates measured on some other scene and not on this one."""
        return tuple(
            sorted(
                gate_id
                for gate_id in self.gate_ids()
                if (gate_id, scene_version) not in self.entries
            )
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [
            self.entries[key].to_dict()
            for key in sorted(self.entries, key=lambda item: (item[0], item[1]))
        ]
