# SPDX-License-Identifier: Apache-2.0
"""Budgets: time, model requests, retries and per-entity candidate limits.

The architecture's local limits live in one object so that they compose. Each limit is meaningful
alone and dangerous in combination: eight candidates per entity, three materialised, thirty-two
poses each, eight physics probes per relation, two rollbacks -- multiplied out those are unbounded.
The global deadline is therefore checked first, and the sub-limits only decide what happens within
it.

Model requests are counted here too, including schema repairs and extra critic calls. The manual is
explicit that those count towards the cap rather than being hidden inside a tool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "pvscene.budget/v1"


class BudgetError(ValueError):
    """The budget is not well formed."""


class BudgetExhausted(RuntimeError):
    """A budget was reached and the work cannot continue within it."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass(slots=True)
class Budget:
    """Every limit that bounds a single scene request.

    ``deadline_s`` is wall-clock from the moment the request was accepted, not from the first
    stage: the manual requires queueing and recovery to stay inside the end-to-end latency rather
    than being quietly excluded from it.
    """

    deadline_s: float = 300.0
    max_model_requests: int = 5
    max_candidates_per_entity: int = 8
    max_materialised_per_entity: int = 3
    max_poses_per_candidate: int = 32
    max_physics_probes_per_relation: int = 8
    max_subgraph_rollbacks: int = 2
    max_schema_repairs: int = 1
    no_progress_limit: int = 2
    max_infrastructure_retries: int = 2

    started_at: float = field(default_factory=time.monotonic)
    model_requests_used: int = 0
    infrastructure_retries_used: int = 0
    rollbacks_used: int = 0
    schema_repairs_used: int = 0
    stages_run: int = 0

    def __post_init__(self) -> None:
        if self.deadline_s <= 0:
            raise BudgetError(f"deadline_s must be positive, got {self.deadline_s}")
        for name in (
            "max_model_requests", "max_candidates_per_entity", "max_materialised_per_entity",
            "max_poses_per_candidate", "max_physics_probes_per_relation", "max_subgraph_rollbacks",
            "no_progress_limit",
        ):
            if getattr(self, name) < 1:
                raise BudgetError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.max_schema_repairs < 0 or self.max_infrastructure_retries < 0:
            raise BudgetError("repair and retry allowances must not be negative")

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_s(self) -> float:
        return self.deadline_s - self.elapsed_s

    def expired(self) -> bool:
        return self.remaining_s <= 0.0

    def permits_model_request(self) -> bool:
        """Whether one more model call fits. The cap includes repairs and extra critic passes."""
        return self.model_requests_used < self.max_model_requests

    def charge_model_request(self, *, purpose: str = "") -> None:
        if not self.permits_model_request():
            raise BudgetExhausted(
                "model_requests",
                f"the {self.max_model_requests}-request cap is reached"
                + (f" (last attempted for {purpose})" if purpose else ""),
            )
        self.model_requests_used += 1

    def permits_rollback(self) -> bool:
        return self.rollbacks_used < self.max_subgraph_rollbacks

    def permits_infrastructure_retry(self) -> bool:
        return self.infrastructure_retries_used < self.max_infrastructure_retries

    def charge_infrastructure_retry(self) -> None:
        if not self.permits_infrastructure_retry():
            raise BudgetExhausted(
                "infrastructure_retries",
                f"the {self.max_infrastructure_retries}-retry allowance is spent",
            )
        self.infrastructure_retries_used += 1

    def charge_rollback(self) -> None:
        if not self.permits_rollback():
            raise BudgetExhausted(
                "rollbacks", f"the {self.max_subgraph_rollbacks}-rollback allowance is spent"
            )
        self.rollbacks_used += 1

    def assert_time_remains(self, *, stage: str) -> None:
        if self.expired():
            raise BudgetExhausted(
                "deadline",
                f"the {self.deadline_s}s deadline expired before {stage}; a stage that cannot "
                "finish inside the budget must not start",
            )

    def report(self) -> BudgetReport:
        return BudgetReport(
            deadline_s=self.deadline_s,
            elapsed_s=round(self.elapsed_s, 4),
            remaining_s=round(self.remaining_s, 4),
            model_requests_used=self.model_requests_used,
            max_model_requests=self.max_model_requests,
            infrastructure_retries_used=self.infrastructure_retries_used,
            rollbacks_used=self.rollbacks_used,
            schema_repairs_used=self.schema_repairs_used,
            stages_run=self.stages_run,
        )


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """What was spent, for the delivery manifest."""

    deadline_s: float
    elapsed_s: float
    remaining_s: float
    model_requests_used: int
    max_model_requests: int
    infrastructure_retries_used: int
    rollbacks_used: int
    schema_repairs_used: int
    stages_run: int

    @property
    def within_deadline(self) -> bool:
        return self.elapsed_s <= self.deadline_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "deadline_s": self.deadline_s,
            "elapsed_s": self.elapsed_s,
            "remaining_s": self.remaining_s,
            "within_deadline": self.within_deadline,
            "model_requests_used": self.model_requests_used,
            "max_model_requests": self.max_model_requests,
            "infrastructure_retries_used": self.infrastructure_retries_used,
            "rollbacks_used": self.rollbacks_used,
            "schema_repairs_used": self.schema_repairs_used,
            "stages_run": self.stages_run,
        }
