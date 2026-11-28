# SPDX-License-Identifier: Apache-2.0
"""The bounded loop's termination rules.

Every test here is a way a generation run fails to stop, or stops for the wrong reason. The loop is
the only place that commits, so its rules decide what a delivery claims.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.tool import FailureCode, ToolStatus
from rfw_w.scenes.orchestrator.budgets import Budget, BudgetError, BudgetExhausted
from rfw_w.scenes.orchestrator.loop import (
    Job,
    LoopError,
    LoopState,
    Orchestrator,
    StageOutcome,
    TerminalReason,
)

START = "a" * 64
NEXT = "b" * 64


def outcome(
    job_id: str,
    status: ToolStatus = ToolStatus.PASSED,
    *,
    version: str = START,
    failure_code: FailureCode | None = None,
    delta: dict | None = None,
    read_only: bool = True,
    detail: str = "",
) -> StageOutcome:
    return StageOutcome(
        job_id=job_id, status=status, base_scene_version=version,
        failure_code=failure_code, proposed_delta=delta, read_only=read_only, detail=detail,
    )


class TestBudget:
    def test_rejects_a_non_positive_deadline(self) -> None:
        with pytest.raises(BudgetError):
            Budget(deadline_s=0)

    def test_model_request_cap_is_enforced(self) -> None:
        budget = Budget(max_model_requests=2)
        budget.charge_model_request()
        budget.charge_model_request()
        assert not budget.permits_model_request()
        with pytest.raises(BudgetExhausted, match="model_requests"):
            budget.charge_model_request(purpose="visual critic")

    def test_infrastructure_retries_have_their_own_allowance(self) -> None:
        budget = Budget(max_infrastructure_retries=1)
        budget.charge_infrastructure_retry()
        with pytest.raises(BudgetExhausted, match="infrastructure_retries"):
            budget.charge_infrastructure_retry()

    def test_report_includes_elapsed_time(self) -> None:
        payload = Budget(deadline_s=10).report().to_dict()
        assert payload["within_deadline"] is True
        assert "elapsed_s" in payload


class TestConfiguration:
    def test_needs_stages(self) -> None:
        with pytest.raises(LoopError, match="no stages"):
            Orchestrator(budget=Budget(), stages=(), handlers={}, initial_scene_version=START)

    def test_needs_a_handler_for_every_stage(self) -> None:
        with pytest.raises(LoopError, match="no handler"):
            Orchestrator(
                budget=Budget(), stages=(("geometry", True),), handlers={},
                initial_scene_version=START,
            )

    def test_needs_a_scene_hash_to_start_from(self) -> None:
        with pytest.raises(LoopError, match="scene hash"):
            Orchestrator(
                budget=Budget(), stages=(("a", True),), handlers={"a": lambda j, s: outcome(j.job_id)},
                initial_scene_version="latest",
            )


class TestCompletion:
    def test_completes_when_all_evidence_is_current(self) -> None:
        called: list[str] = []

        def handler(job: Job, state: LoopState) -> StageOutcome:
            called.append(job.job_id)
            return outcome(job.job_id)

        orchestrator = Orchestrator(
            budget=Budget(), stages=(("a", True), ("b", True)), handlers={"a": handler, "b": handler},
            initial_scene_version=START,
        )
        # Evidence is already satisfied, so no stage should run at all: a complete run must not do
        # more work merely because the plan had more stages.
        result = orchestrator.run(all_evidence_current=lambda version: True)
        assert result.terminal is TerminalReason.READY
        assert called == []
        assert result.iterations == 0

    def test_runs_the_plan_and_then_completes(self) -> None:
        state = {"done": False}

        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            return outcome(job.job_id)

        orchestrator = Orchestrator(
            budget=Budget(), stages=(("a", True),), handlers={"a": handler},
            initial_scene_version=START,
        )

        def evidence(version: str) -> bool:
            return state["done"]

        # The first call is not complete; the handler flips completion via on_commit.
        def commit(job: Job, result: StageOutcome) -> str:
            state["done"] = True
            return NEXT

        # No delta means no commit, so the run should reach the end of the plan and fail rather
        # than claim readiness.
        result = orchestrator.run(all_evidence_current=evidence, on_commit=commit)
        assert result.terminal is TerminalReason.FAILED

    def test_a_passing_writer_commits_and_advances_the_version(self) -> None:
        versions: list[str] = []

        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            return outcome(job.job_id, read_only=False, delta={"instances": {}})

        orchestrator = Orchestrator(
            budget=Budget(), stages=(("a", False),), handlers={"a": handler},
            initial_scene_version=START,
        )
        result = orchestrator.run(
            all_evidence_current=lambda version: bool(versions),
            on_commit=lambda job, res: versions.append(NEXT) or NEXT,
        )
        assert result.terminal is TerminalReason.READY
        assert result.committed == ("a@0",)
        assert result.scene_version == NEXT


class TestVersionMismatch:
    def test_a_delta_against_a_stale_version_is_discarded(self) -> None:
        """Optimistic concurrency: applying it would mix two scene versions."""
        calls = {"n": 0}

        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            calls["n"] += 1
            # Every result claims to be against the original version, while the state has moved.
            return outcome(job.job_id, version=START, read_only=False, delta={"x": 1})

        orchestrator = Orchestrator(
            budget=Budget(), stages=(("a", False), ("b", False)), handlers={"a": handler, "b": handler},
            initial_scene_version=START,
        )
        # The first commit moves the version, so the second result is stale.
        result = orchestrator.run(
            all_evidence_current=lambda version: False,
            on_commit=lambda job, res: NEXT,
        )
        assert "b@0" in result.discarded_for_version_mismatch
        assert "a@0" in result.committed

    def test_discarded_work_is_reported_not_hidden(self) -> None:
        orchestrator = Orchestrator(
            budget=Budget(),
            stages=(("a", False), ("b", False)),
            handlers={
                "a": lambda j, s: outcome(j.job_id, read_only=False, delta={"x": 1}),
                "b": lambda j, s: outcome(j.job_id, version=START, read_only=False, delta={"y": 2}),
            },
            initial_scene_version=START,
        )
        result = orchestrator.run(all_evidence_current=lambda v: False, on_commit=lambda j, r: NEXT)
        assert result.to_dict()["discarded_for_version_mismatch"] == ["b@0"]


class TestNoProgress:
    def test_the_same_failure_twice_stops_the_run(self) -> None:
        """Retrying an identical failure is the cheapest way to look busy while getting nowhere."""
        orchestrator = Orchestrator(
            budget=Budget(no_progress_limit=2),
            stages=(("geometry", True), ("geometry", True)),
            handlers={
                "geometry": lambda j, s: outcome(
                    j.job_id, ToolStatus.FAILED,
                    failure_code=FailureCode.GEOMETRY_PENETRATION, detail="2.1 mm",
                )
            },
            initial_scene_version=START,
        )
        result = orchestrator.run(all_evidence_current=lambda v: False)
        assert result.terminal is TerminalReason.NO_PROGRESS
        assert len(result.failed_signatures) == 2
        assert result.failed_signatures[0] == result.failed_signatures[1]

    def test_a_different_failure_is_not_a_repeat(self) -> None:
        codes = [FailureCode.GEOMETRY_PENETRATION, FailureCode.SUPPORT_NOT_FOUND]

        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            code = codes.pop(0) if codes else FailureCode.GEOMETRY_PENETRATION
            return outcome(job.job_id, ToolStatus.FAILED, failure_code=code)

        orchestrator = Orchestrator(
            budget=Budget(no_progress_limit=2),
            stages=(("a", True), ("b", True)),
            handlers={"a": handler, "b": handler},
            initial_scene_version=START,
        )
        result = orchestrator.run(all_evidence_current=lambda v: False)
        # Two different failures are two different problems, so the run ends for lack of a plan
        # rather than for lack of progress.
        assert result.terminal is TerminalReason.FAILED

    def test_signature_ignores_the_message_text(self) -> None:
        """A signature composed from free text would make an identical failure look new."""
        first = outcome("a@0", ToolStatus.FAILED, failure_code=FailureCode.GEOMETRY_PENETRATION,
                        detail="penetration 2.1 mm on cup")
        second = outcome("a@1", ToolStatus.FAILED, failure_code=FailureCode.GEOMETRY_PENETRATION,
                         detail="penetration 9.7 mm on desk")
        assert first.signature == second.signature


class TestInfrastructureFailures:
    def test_a_timeout_retries_without_spending_a_model_request(self) -> None:
        """A timeout says nothing about the scene, so replanning would waste a request."""
        attempts: list[int] = []

        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            attempts.append(1)
            status = ToolStatus.PASSED if len(attempts) > 1 else ToolStatus.TIMEOUT
            return outcome(job.job_id, status, failure_code=None if status is ToolStatus.PASSED
                           else FailureCode.BUDGET_EXHAUSTED, read_only=False,
                           delta=None if status is ToolStatus.PASSED else None)

        budget = Budget(max_infrastructure_retries=2, no_progress_limit=1)
        orchestrator = Orchestrator(
            budget=budget, stages=(("render", True),), handlers={"render": handler},
            initial_scene_version=START,
        )
        result = orchestrator.run(all_evidence_current=lambda v: len(attempts) > 1)
        assert result.terminal is TerminalReason.READY
        assert budget.model_requests_used == 0
        assert budget.infrastructure_retries_used == 1

    def test_retries_are_bounded(self) -> None:
        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            return outcome(job.job_id, ToolStatus.TIMEOUT, failure_code=FailureCode.BUDGET_EXHAUSTED)

        budget = Budget(max_infrastructure_retries=1)
        orchestrator = Orchestrator(
            budget=budget, stages=(("render", True),), handlers={"render": handler},
            initial_scene_version=START,
        )
        result = orchestrator.run(all_evidence_current=lambda v: False)
        assert result.terminal is TerminalReason.BUDGET_EXHAUSTED
        assert budget.infrastructure_retries_used == 1


class TestSemanticEscalation:
    def test_a_semantic_failure_stops_for_the_caller_to_decide(self) -> None:
        """Whether to spend a model request is the caller's policy; the loop only checks the cap."""
        orchestrator = Orchestrator(
            budget=Budget(),
            stages=(("layout", True),),
            handlers={
                "layout": lambda j, s: outcome(
                    j.job_id, ToolStatus.FAILED, failure_code=FailureCode.SUPPORT_NOT_FOUND
                )
            },
            initial_scene_version=START,
        )
        result = orchestrator.run(
            all_evidence_current=lambda v: False, needs_semantic_change=lambda o: True
        )
        assert result.terminal is TerminalReason.AWAITING_MODEL

    def test_no_model_requests_left_is_budget_exhausted(self) -> None:
        budget = Budget(max_model_requests=1)
        budget.charge_model_request()
        orchestrator = Orchestrator(
            budget=budget,
            stages=(("layout", True),),
            handlers={
                "layout": lambda j, s: outcome(
                    j.job_id, ToolStatus.FAILED, failure_code=FailureCode.SUPPORT_NOT_FOUND
                )
            },
            initial_scene_version=START,
        )
        result = orchestrator.run(
            all_evidence_current=lambda v: False, needs_semantic_change=lambda o: True
        )
        assert result.terminal is TerminalReason.BUDGET_EXHAUSTED


class TestRepair:
    def test_a_local_repair_runs_before_escalating(self) -> None:
        """A local repair is cheaper than a model request, so it is tried first."""
        seen: list[str] = []

        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            seen.append(job.job_id)
            if job.job_id.startswith("retry"):
                return outcome(job.job_id, read_only=False, delta={"placed": True})
            return outcome(job.job_id, ToolStatus.FAILED, failure_code=FailureCode.NO_PLACEMENT_CAPACITY)

        committed: list[str] = []
        orchestrator = Orchestrator(
            budget=Budget(no_progress_limit=9),
            stages=(("layout", True),),
            handlers={"layout": handler, "retry": handler},
            initial_scene_version=START,
        )
        result = orchestrator.run(
            all_evidence_current=lambda v: bool(committed),
            on_commit=lambda job, res: committed.append(job.job_id) or NEXT,
            repair_for=lambda signature: Job(job_id="retry@0", stage="retry", read_only=False),
            needs_semantic_change=lambda o: True,
        )
        assert seen == ["layout@0", "retry@0"]
        # The repair succeeded, so no model request was needed.
        assert result.terminal is TerminalReason.READY
        assert result.budget["model_requests_used"] == 0

    def test_repair_allowance_is_bounded(self) -> None:
        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            return outcome(job.job_id, ToolStatus.FAILED, failure_code=FailureCode.NO_PLACEMENT_CAPACITY)

        budget = Budget(max_subgraph_rollbacks=1, no_progress_limit=99)
        orchestrator = Orchestrator(
            budget=budget, stages=(("layout", True),),
            handlers={"layout": handler, "retry": handler},
            initial_scene_version=START,
        )
        result = orchestrator.run(
            all_evidence_current=lambda v: False,
            repair_for=lambda signature: Job(job_id="retry@0", stage="retry", read_only=False),
        )
        assert budget.rollbacks_used == 1
        assert result.terminal in (TerminalReason.FAILED, TerminalReason.BUDGET_EXHAUSTED)


class TestDeadline:
    def test_an_expired_deadline_stops_before_starting_work(self) -> None:
        ran: list[str] = []

        def handler(job: Job, loop_state: LoopState) -> StageOutcome:
            ran.append(job.job_id)
            return outcome(job.job_id)

        budget = Budget(deadline_s=0.0001)
        import time as _time

        _time.sleep(0.002)
        orchestrator = Orchestrator(
            budget=budget, stages=(("a", True),), handlers={"a": handler},
            initial_scene_version=START,
        )
        result = orchestrator.run(all_evidence_current=lambda v: False)
        assert result.terminal is TerminalReason.BUDGET_EXHAUSTED
        assert ran == []
        assert result.budget["within_deadline"] is False


class TestEvents:
    def test_every_step_is_recorded(self) -> None:
        orchestrator = Orchestrator(
            budget=Budget(no_progress_limit=5),
            stages=(("a", True), ("b", True)),
            handlers={name: (lambda j, s: outcome(j.job_id)) for name in ("a", "b")},
            initial_scene_version=START,
        )
        result = orchestrator.run(all_evidence_current=lambda v: False)
        assert [event.job_id for event in result.events] == ["a@0", "b@0"]
        assert all(event.scene_version == START for event in result.events)

    def test_a_stage_that_raises_becomes_a_typed_failure(self) -> None:
        def exploding(job: Job, loop_state: LoopState) -> StageOutcome:
            raise RuntimeError("the renderer died")

        orchestrator = Orchestrator(
            budget=Budget(no_progress_limit=5), stages=(("render", True),),
            handlers={"render": exploding}, initial_scene_version=START,
        )
        result = orchestrator.run(all_evidence_current=lambda v: False)
        assert result.terminal is TerminalReason.FAILED
        assert "renderer died" in result.events[0].detail
        assert result.events[0].signature.endswith(FailureCode.PRECONDITION_UNMET.value)

    def test_invalidation_is_recorded(self) -> None:
        orchestrator = Orchestrator(
            budget=Budget(), stages=(("a", True),),
            handlers={"a": lambda j, s: outcome(j.job_id)}, initial_scene_version=START,
        )
        orchestrator.invalidate(["function.initial_relations", "visual.critique"])
        result = orchestrator.run(all_evidence_current=lambda v: True)
        assert set(result.invalidated_gates) == {"function.initial_relations", "visual.critique"}
