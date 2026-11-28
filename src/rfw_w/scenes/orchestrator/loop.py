# SPDX-License-Identifier: Apache-2.0
"""The bounded orchestrator loop (architecture §10).

This is the only place that commits. Everything else observes or proposes, which is what makes the
loop's rules enforceable rather than advisory.

Four rules are implemented here because each one is a way a generation run goes wrong quietly:

* **the deadline and the no-progress limit are checked before any work is started.** A run that
  cannot finish stops and says why, instead of spending its budget on retries that cannot succeed.
* **commit is optimistic-concurrency checked.** A result computed against a scene version that has
  moved is discarded rather than applied, and the moved object's dependants have their evidence
  invalidated so nothing stale is reused.
* **the same failure signature twice with no progress escalates.** Retrying an identical failure is
  the cheapest way to burn a budget while looking busy.
* **an infrastructure failure is not a semantic one.** A timeout or an I/O error retries on its own
  allowance instead of spending a model request to "replan" a scene that is fine.

The loop is deterministic given its stages: no wall-clock decisions other than the deadline, and no
randomness. That is what lets a run be replayed from its checkpoint and produce the same result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

from rfw_w.scenes.contracts.tool import FailureCode, ToolStatus
from rfw_w.scenes.orchestrator.budgets import Budget

SCHEMA_VERSION = "pvscene.orchestrator/v1"


class LoopError(RuntimeError):
    """The loop was configured in a way that cannot terminate."""


class TerminalReason(str, Enum):
    """Why the loop stopped. Every one of these is a distinct delivery outcome."""

    READY = "ready"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"
    REPAIR_EXHAUSTED = "repair_exhausted"
    AWAITING_MODEL = "awaiting_model"


@dataclass(frozen=True, slots=True)
class Job:
    """One unit of work. Read-only jobs may not propose a delta."""

    job_id: str
    stage: str
    read_only: bool
    scope_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.job_id or not self.stage:
            raise LoopError("a job needs an id and a stage name")


@dataclass(slots=True)
class LoopState:
    """The mutable state a stage may read and propose changes against."""

    scene_version: str
    revision: int = 0
    invalidated_gates: set[str] = field(default_factory=set)

    def advance(self, new_version: str) -> None:
        if new_version == self.scene_version:
            raise LoopError("a committed change must produce a new scene version")
        self.scene_version = new_version
        self.revision += 1


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What a stage returned, in the terms the loop reasons about."""

    job_id: str
    status: ToolStatus
    base_scene_version: str
    detail: str = ""
    failure_code: FailureCode | None = None
    read_only: bool = True
    proposed_delta: Mapping[str, Any] | None = None
    touched_ids: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status is ToolStatus.PASSED

    @property
    def infrastructure_failure(self) -> bool:
        """A timeout or an I/O fault says nothing about the scene."""
        return self.status is ToolStatus.TIMEOUT or self.failure_code in (
            FailureCode.ASSET_IO_FAILURE,
            FailureCode.RENDER_FAILED,
        )

    @property
    def signature(self) -> str:
        """The identity of a failure, for detecting repeats.

        Composed from the stage and the typed code, not from the message: two runs that fail the
        same way must produce the same signature, and free text would let an identical failure
        look new because a number in a message changed.
        """
        code = self.failure_code.value if self.failure_code else self.status.value
        return f"{self.stage_signature()}:{code}"

    def stage_signature(self) -> str:
        return self.job_id.split("@", 1)[0]


class StageFn(Protocol):
    """A stage: takes a job and the current state, returns an outcome."""

    def __call__(self, job: Job, state: LoopState) -> StageOutcome: ...


@dataclass(frozen=True, slots=True)
class LoopEvent:
    """One recorded step, so a run can be read back rather than re-derived."""

    index: int
    job_id: str
    status: str
    signature: str
    scene_version: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "job_id": self.job_id, "status": self.status,
            "signature": self.signature, "scene_version": self.scene_version, "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class OrchestratorResult:
    """The loop's verdict, with everything needed to explain it."""

    terminal: TerminalReason
    scene_version: str
    iterations: int
    committed: tuple[str, ...]
    discarded_for_version_mismatch: tuple[str, ...]
    invalidated_gates: tuple[str, ...]
    failed_signatures: tuple[str, ...]
    budget: Mapping[str, Any]
    events: tuple[LoopEvent, ...] = ()

    @property
    def ready(self) -> bool:
        return self.terminal is TerminalReason.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "terminal": self.terminal.value,
            "ready": self.ready,
            "scene_version": self.scene_version,
            "iterations": self.iterations,
            "committed": list(self.committed),
            "discarded_for_version_mismatch": list(self.discarded_for_version_mismatch),
            "invalidated_gates": sorted(self.invalidated_gates),
            "failed_signatures": list(self.failed_signatures),
            "budget": dict(self.budget),
            "events": [event.to_dict() for event in self.events],
        }


def _plan(stages: Sequence[tuple[str, bool]]) -> list[Job]:
    """Turn a stage sequence into jobs, numbered so repeats of a stage are distinguishable."""
    jobs: list[Job] = []
    counts: dict[str, int] = {}
    for stage, read_only in stages:
        index = counts.get(stage, 0)
        counts[stage] = index + 1
        jobs.append(Job(job_id=f"{stage}@{index}", stage=stage, read_only=read_only))
    return jobs


class Orchestrator:
    """Runs the stages in order, committing what passes and explaining what does not.

    ``stages`` is the plan; ``handlers`` maps a stage name to its implementation. ``on_commit`` is
    called with ``(job, outcome)`` when a delta is accepted, so the caller can rebuild the scene
    and return the new version. Everything else about the loop is decided here, which is what keeps
    the rules in one readable place.
    """

    def __init__(
        self,
        *,
        budget: Budget,
        stages: Sequence[tuple[str, bool]],
        handlers: Mapping[str, StageFn],
        initial_scene_version: str,
    ) -> None:
        if not stages:
            raise LoopError("an orchestrator with no stages can never terminate usefully")
        unknown = sorted({stage for stage, _ in stages} - set(handlers))
        if unknown:
            raise LoopError(f"no handler for stage(s): {unknown}")
        if len(initial_scene_version) != 64:
            raise LoopError(
                f"initial_scene_version must be a scene hash, got {initial_scene_version!r}"
            )
        self.budget = budget
        self.jobs = _plan(stages)
        self.handlers = handlers
        self.state = LoopState(scene_version=initial_scene_version)

    def run(
        self,
        *,
        all_evidence_current: Callable[[str], bool],
        on_commit: Callable[[Job, StageOutcome], str] | None = None,
        repair_for: Callable[[str], Job | None] | None = None,
        needs_semantic_change: Callable[[StageOutcome], bool] | None = None,
    ) -> OrchestratorResult:
        """Run to a terminal reason.

        ``all_evidence_current`` is consulted first on every iteration: a run whose gates are
        already satisfied finishes, rather than doing more work because the plan had more stages.
        """
        committed: list[str] = []
        discarded: list[str] = []
        signatures: list[str] = []
        events: list[LoopEvent] = []
        pending = list(self.jobs)
        repairs: list[Job] = []
        repeat_count: dict[str, int] = {}
        iterations = 0

        while True:
            # Terminal checks come first, in this order. A run that cannot finish must not start
            # more work, and an already-complete run must not do any.
            if self.budget.expired():
                return self._result(
                    TerminalReason.BUDGET_EXHAUSTED, committed, discarded, signatures, events, iterations
                )
            if all_evidence_current(self.state.scene_version):
                return self._result(TerminalReason.READY, committed, discarded, signatures, events, iterations)

            if repairs:
                job = repairs.pop(0)
            elif pending:
                job = pending.pop(0)
            else:
                # The plan is exhausted and the gates are not satisfied. That is a failure with a
                # reason, not a reason to loop.
                reason = (
                    TerminalReason.REPAIR_EXHAUSTED if signatures else TerminalReason.FAILED
                )
                return self._result(reason, committed, discarded, signatures, events, iterations)

            iterations += 1
            self.budget.stages_run += 1
            handler = self.handlers[job.stage]
            try:
                outcome = handler(job, self.state)
            except Exception as error:  # noqa: BLE001 - a stage that raises is a typed failure
                outcome = StageOutcome(
                    job_id=job.job_id,
                    status=ToolStatus.FAILED,
                    base_scene_version=self.state.scene_version,
                    failure_code=FailureCode.PRECONDITION_UNMET,
                    detail=f"{type(error).__name__}: {error}",
                    read_only=job.read_only,
                )
            events.append(
                LoopEvent(
                    index=len(events),
                    job_id=job.job_id,
                    status=outcome.status.value,
                    signature=outcome.signature,
                    scene_version=self.state.scene_version,
                    detail=outcome.detail[:400],
                )
            )

            if outcome.passed:
                if on_commit is None or outcome.proposed_delta is None:
                    # Nothing to commit: an observation, or a stage that changed nothing. Its
                    # repeat count is cleared because progress is not the same as change.
                    continue
                if outcome.base_scene_version != self.state.scene_version:
                    # Optimistic concurrency: the delta was computed against a scene that has since
                    # moved. Applying it would mix two versions.
                    discarded.append(job.job_id)
                    continue
                new_version = on_commit(job, outcome)
                self.state.advance(new_version)
                committed.append(job.job_id)
                repeat_count.clear()
                continue

            signature = outcome.signature
            signatures.append(signature)
            repeat_count[signature] = repeat_count.get(signature, 0) + 1

            if outcome.infrastructure_failure:
                # Not a semantic problem, so it does not consume a model request.
                if self.budget.permits_infrastructure_retry():
                    self.budget.charge_infrastructure_retry()
                    repairs.insert(0, job)
                    continue
                return self._result(
                    TerminalReason.BUDGET_EXHAUSTED, committed, discarded, signatures, events, iterations
                )

            if repeat_count[signature] >= self.budget.no_progress_limit:
                # The same failure twice with nothing committed in between. Retrying it again is
                # the cheapest way to look busy while making no progress.
                return self._result(
                    TerminalReason.NO_PROGRESS, committed, discarded, signatures, events, iterations
                )

            local = repair_for(signature) if repair_for else None
            if local is not None and self.budget.permits_rollback():
                self.budget.charge_rollback()
                repairs.append(local)
                continue

            if needs_semantic_change and needs_semantic_change(outcome):
                if not self.budget.permits_model_request():
                    return self._result(
                        TerminalReason.BUDGET_EXHAUSTED, committed, discarded, signatures, events, iterations
                    )
                # The loop stops here rather than calling a model itself. Deciding to spend a model
                # request is a policy the caller owns, and the budget has already been consulted.
                return self._result(
                    TerminalReason.AWAITING_MODEL, committed, discarded, signatures, events, iterations
                )

            if pending or repairs:
                continue
            return self._result(
                TerminalReason.FAILED, committed, discarded, signatures, events, iterations
            )

    def _result(
        self,
        terminal: TerminalReason,
        committed: list[str],
        discarded: list[str],
        signatures: list[str],
        events: list[LoopEvent],
        iterations: int,
    ) -> OrchestratorResult:
        return OrchestratorResult(
            terminal=terminal,
            scene_version=self.state.scene_version,
            iterations=iterations,
            committed=tuple(committed),
            discarded_for_version_mismatch=tuple(discarded),
            invalidated_gates=tuple(sorted(self.state.invalidated_gates)),
            failed_signatures=tuple(signatures),
            budget=self.budget.report().to_dict(),
            events=tuple(events),
        )

    def invalidate(self, gates: Sequence[str]) -> None:
        """Record that gates no longer hold.

        Called by the caller when a committed change invalidates dependent evidence. The loop keeps
        the set so the result can report which gates a run has invalidated, which is how a
        reviewer sees that a later stage was correctly re-run rather than reused.
        """
        self.state.invalidated_gates.update(gates)
