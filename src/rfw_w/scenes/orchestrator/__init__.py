# SPDX-License-Identifier: Apache-2.0
"""The bounded orchestrator: one writer, a job DAG, and evidence-bound completion.

Architecture §10 defines the loop. Two of its properties are what this package exists to make
structural rather than conventional:

* **a global deadline and a no-progress limit are checked before anything else**, so a scene that
  cannot be finished stops and says why instead of consuming its budget on retries;
* **commit is optimistic-concurrency checked**. A result computed against a scene version that has
  since moved is discarded, not applied, and the dependent evidence is invalidated rather than
  reused.
"""

from __future__ import annotations

from rfw_w.scenes.orchestrator.budgets import (
    Budget,
    BudgetError,
    BudgetExhausted,
    BudgetReport,
)
from rfw_w.scenes.orchestrator.loop import (
    LoopError,
    Orchestrator,
    OrchestratorResult,
    TerminalReason,
)

__all__ = [
    "Budget",
    "BudgetError",
    "BudgetExhausted",
    "BudgetReport",
    "LoopError",
    "Orchestrator",
    "OrchestratorResult",
    "TerminalReason",
]
