# SPDX-License-Identifier: Apache-2.0
"""Hierarchical checkpoints and rollback (architecture §18).

    checkpoint_room_layout -> checkpoint_furniture -> checkpoint_task_critical_assets
      -> checkpoint_manipulands -> checkpoint_final_scene

    allowed: pose rollback, asset rollback, subgraph rollback, room rollback
    the point: avoid regenerating the whole scene

Three rules are implemented here rather than left to discipline:

* **a rollback is scoped, not global.** Restoring the subgraph that failed rather than the whole
  scene is the entire reason checkpoints are hierarchical; a stack that could only restore
  everything would be a restart with extra bookkeeping.
* **the candidate cursor travels with the plan.** The executable design §10 requires a rollback to
  restore the candidate cursor *as well as* the geometry, so that a repaired subgraph does not
  immediately re-exhaust the candidates it exhausted before. Rather than restoring two things
  together and hoping they stay in step, the cursor lives inside the plan
  (:attr:`~rfw_w.scenes.solvers.furnish.FurnitureRequest.skip_candidates`), so a checkpoint cannot
  restore one without the other.
* **a rollback restores the invalidation set too.** §10 again: what the region's evidence was
  invalidated *by* is part of the state. Dropping it would leave a gate recorded as invalidated for
  a reason that no longer exists, and the run's report would name a cause it cannot point at.

The scope of a checkpoint is the set of entities whose placement it is authoritative for, and the
closure of a rollback is computed through the plan's own dependency edges -- what faces what, what
stands on what. This is the plan-level counterpart of
:meth:`~rfw_w.scenes.contracts.state.SceneIR.dependency_closure`; M1 has no ``SceneIR`` yet, and
inventing one to serve the checkpoint stack would be building the container before the contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from rfw_w.scenes.solvers.furnish import ScenePlan

SCHEMA_VERSION = "pvscene.checkpoints/v1"


class CheckpointError(RuntimeError):
    """The checkpoint stack was asked for something it cannot supply."""


class CheckpointName(str, Enum):
    """The five levels from §18, in the order they are reached."""

    ROOM_LAYOUT = "checkpoint_room_layout"
    FURNITURE = "checkpoint_furniture"
    TASK_CRITICAL_ASSETS = "checkpoint_task_critical_assets"
    MANIPULANDS = "checkpoint_manipulands"
    FINAL_SCENE = "checkpoint_final_scene"

    @property
    def rank(self) -> int:
        return _ORDER.index(self)


_ORDER: tuple[CheckpointName, ...] = (
    CheckpointName.ROOM_LAYOUT,
    CheckpointName.FURNITURE,
    CheckpointName.TASK_CRITICAL_ASSETS,
    CheckpointName.MANIPULANDS,
    CheckpointName.FINAL_SCENE,
)


class RollbackKind(str, Enum):
    """How much a rollback undoes. Named because they are different repairs with different costs."""

    POSE = "pose_rollback"
    ASSET = "asset_rollback"
    SUBGRAPH = "subgraph_rollback"
    ROOM = "room_rollback"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One restore point: the plan, the version it produced, and what it is authoritative for."""

    name: CheckpointName
    plan: ScenePlan
    scene_version: str
    scope_ids: frozenset[str]
    revision: int = 0
    invalidated_gates: frozenset[str] = frozenset()
    note: str = ""

    @property
    def entity_ids(self) -> frozenset[str]:
        return frozenset(self.plan.entity_ids)

    def covers(self, closure: Iterable[str]) -> bool:
        """Whether this checkpoint is authoritative for every entity a rollback would touch.

        Strictly by scope: every name in the closure must be one this checkpoint is authoritative
        for. An earlier checkpoint's plan may well contain a name outside its scope, but that copy
        predates whatever was decided about that name afterwards, and restoring it would quietly
        revert a decision this checkpoint was never about.
        """
        return all(name in self.scope_ids for name in closure)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "scene_version": self.scene_version,
            "revision": self.revision,
            "scope_ids": sorted(self.scope_ids),
            "invalidated_gates": sorted(self.invalidated_gates),
            "note": self.note,
        }


def plan_closure(plan: ScenePlan, targets: Sequence[str]) -> frozenset[str]:
    """The targets plus everything in the plan that depends on them.

    Dependency edges are the plan's own: a piece that *faces* a target and a surface object that
    *stands on* one both have their placement justified by it, so moving the target invalidates
    their evidence even though they did not move. Traversal is cycle-safe -- a plan can legitimately
    name a mutual facing pair -- which is why a visited set is required rather than optional.
    """
    known = set(plan.entity_ids)
    frontier = [name for name in targets if name in known]
    seen: set[str] = set()
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        for request in plan.furniture:
            if request.faces_entity == node and request.entity_id not in seen:
                frontier.append(request.entity_id)
        for surface in plan.surfaces:
            if surface.support_entity == node and surface.entity_id not in seen:
                frontier.append(surface.entity_id)
    return frozenset(seen)


def rollback_kind(checkpoint: Checkpoint, closure: Iterable[str]) -> RollbackKind:
    """What the restore amounts to, from how much of the plan it covers.

    The names come from §18's list and are reported rather than inferred by a reader, because "we
    rolled back the room" and "we moved one chair back" are very different claims about a run.
    """
    touched = set(closure)
    scope = set(checkpoint.scope_ids) or set(checkpoint.plan.entity_ids)
    if not touched:
        return RollbackKind.POSE
    if checkpoint.name is CheckpointName.ROOM_LAYOUT:
        return RollbackKind.ROOM
    if len(touched) == 1 and touched <= _task_critical(checkpoint.plan):
        return RollbackKind.ASSET
    if scope >= touched and len(scope) > len(touched):
        return RollbackKind.SUBGRAPH
    return RollbackKind.SUBGRAPH


def _task_critical(plan: ScenePlan) -> set[str]:
    """The pieces the task cannot lose, so that rolling one back can be recognised as such.

    The support is in here for a reason worth stating: it is not in `plan.furniture` any more (it is
    built, not retrieved), and before this it was task-critical *because it was a furniture row with
    `hard_required=True`*. Leaving it out would have silently changed how a rollback of the desk is
    classified -- from "an asset rollback" to "a subgraph rollback" -- without a single test or gate
    reporting anything.
    """
    return {
        request.entity_id for request in plan.furniture if request.hard_required
    } | {request.entity_id for request in plan.supports}


@dataclass(slots=True)
class CheckpointStack:
    """The checkpoints a run has taken, oldest first."""

    checkpoints: list[Checkpoint] = field(default_factory=list)

    def push(
        self,
        name: CheckpointName,
        plan: ScenePlan,
        *,
        scene_version: str,
        scope_ids: Iterable[str] | None = None,
        revision: int = 0,
        invalidated_gates: Iterable[str] = (),
        note: str = "",
    ) -> Checkpoint:
        """Record a restore point.

        Pushing a name that is already on the stack replaces the earlier entry rather than adding a
        second: the five names are levels, not attempts, and two ``checkpoint_furniture`` entries
        would leave "restore to the furniture stage" ambiguous.
        """
        checkpoint = Checkpoint(
            name=name, plan=plan, scene_version=scene_version,
            scope_ids=frozenset(scope_ids) if scope_ids is not None else frozenset(plan.entity_ids),
            revision=revision, invalidated_gates=frozenset(invalidated_gates), note=note,
        )
        for index, existing in enumerate(self.checkpoints):
            if existing.name is name:
                self.checkpoints[index] = checkpoint
                break
        else:
            self.checkpoints.append(checkpoint)
        return checkpoint

    def by_name(self, name: CheckpointName) -> Checkpoint | None:
        for checkpoint in self.checkpoints:
            if checkpoint.name is name:
                return checkpoint
        return None

    def restore(self, name: CheckpointName) -> Checkpoint:
        """The named checkpoint, or a refusal.

        Refusing rather than returning the nearest one: a caller that asked for the room layout
        checkpoint and silently received the furniture one would roll back less than it believed,
        and the run would report a rollback that did not happen.
        """
        checkpoint = self.by_name(name)
        if checkpoint is None:
            raise CheckpointError(
                f"no {name.value} has been taken; a rollback cannot restore a state the run never "
                f"recorded"
            )
        return checkpoint

    def deepest_covering(self, closure: Iterable[str]) -> Checkpoint | None:
        """The most recent checkpoint authoritative for the whole closure, or ``None``.

        Most recent rather than most specific: restoring the newest state that is still valid undoes
        the least work, which is what "avoid regenerating the whole scene" asks for.
        """
        needed = set(closure)
        if not needed:
            return None
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.covers(needed):
                return checkpoint
        return None

    def rollback_for(
        self, targets: Sequence[str], *, plan: ScenePlan
    ) -> tuple[Checkpoint, RollbackKind, frozenset[str]] | None:
        """The checkpoint a subgraph rollback would restore, with its closure and its kind."""
        closure = plan_closure(plan, targets)
        checkpoint = self.deepest_covering(closure)
        if checkpoint is None:
            return None
        return checkpoint, rollback_kind(checkpoint, closure), closure

    def to_list(self) -> list[dict[str, Any]]:
        return [checkpoint.to_dict() for checkpoint in self.checkpoints]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "checkpoints": self.to_list(),
            "depth": len(self.checkpoints),
        }
