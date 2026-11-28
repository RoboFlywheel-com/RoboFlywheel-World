# SPDX-License-Identifier: Apache-2.0
"""Final entity losses, distinct from failed candidate attempts. Stdlib-only."""
from dataclasses import dataclass
from typing import Iterable

from .scene_spec import SceneSpec, SceneScope, relation_closure_ids, scope_of

UNPLACED_REASONS = frozenset({
    "NO_ASSET", "NO_CANDIDATE", "PREPARE_FAILED", "NO_POSITION", "NO_SURFACE_SPACE",
    "SUPPORT_UNAVAILABLE", "SIZE_BOUND_RELAXED", "SIZE_UNSUPPORTED", "NOT_IN_PLAN",
})


@dataclass(frozen=True, slots=True)
class UnplacedItem:
    entity_id: str
    kind: str
    reason: str
    arena: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in UNPLACED_REASONS:
            raise ValueError(f"unknown unplaced reason: {self.reason}")

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in ("entity_id", "kind", "reason", "arena", "detail")}


def split_unplaced(spec: SceneSpec, placed_ids: Iterable[str], planned_ids: Iterable[str],
                   attempts: Iterable[UnplacedItem]) -> dict:
    """One final loss per missing entity; a successful retry is never a loss.

    Core and task closure take precedence over plan removal: deleting a required context entity
    cannot make it an optional drop. The last attempt supplies the final failure's diagnosis.
    """
    placed, planned = set(placed_ids), set(planned_ids)
    last = {item.entity_id: item for item in attempts}
    closure = set(relation_closure_ids(spec)[0]) | {e.id for e in spec.hard_entities}
    result = {key: [] for key in (
        "unplaced_core", "unplaced_task_closure", "unplaced_periphery", "periphery_dropped",
    )}
    for entity in spec.entities:
        if entity.id in placed:
            continue
        item = last.get(entity.id, UnplacedItem(
            entity.id, entity.category, "NOT_IN_PLAN" if entity.id not in planned else "NO_CANDIDATE",
            "plan" if entity.id not in planned else "room", "no placed instance was produced",
        ))
        if scope_of(entity) is SceneScope.CORE:
            key = "unplaced_core"
        elif entity.id in closure:
            key = "unplaced_task_closure"
        elif entity.id not in planned:
            key = "periphery_dropped"
        else:
            key = "unplaced_periphery"
        result[key].append(item.to_dict())
    return result
