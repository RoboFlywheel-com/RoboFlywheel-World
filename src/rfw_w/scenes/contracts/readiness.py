# SPDX-License-Identifier: Apache-2.0
"""The readiness ladder and its resolution rule.

Architecture §11 defines the levels and, more importantly, the rule:

    ready = ALL(required evidence.status == pass AND evidence.state_hash matches)

Three consequences are implemented here rather than left to discipline:

* The rule is a conjunction over *required* gates. It never degrades to "any", so a pass in
  one backend cannot compensate for a failure in another, and a beauty score cannot offset a
  penetration.
* ``not_run`` and ``unsupported`` never satisfy a gate. Absent or unexecuted evidence leaves
  a level unreachable; it does not silently read as success.
* Evidence measured against a different scene hash does not count, so an approval cannot
  survive the scene changing underneath it.

``resolve_readiness`` is a pure function of the ledger, the scene version, the requested
level and the requested backends. That is deliberate: the independent delivery verifier
re-runs this same function on the delivered artefacts to check that the claimed level is
the recomputed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from rfw_w.scenes.contracts.hashing import is_digest
from rfw_w.scenes.contracts.tool import EvidenceLedger, EvidenceStatus

SCHEMA_VERSION = "pvscene.readiness/v3"


class ReadinessError(ValueError):
    """The readiness request or ledger contradicts the contract."""


class ReadinessLevel(str, Enum):
    """Ordered success levels. Order matters; ``DRAFT`` is the floor."""

    DRAFT = "draft"
    SIMREADY_SCENE_READY = "simready_scene_ready"
    FUNCTIONAL_SCENE_READY = "functional_scene_ready"
    ROBOT_FEASIBLE_SCENE_READY = "robot_feasible_scene_ready"
    TASK_VERIFIED = "task_verified"

    @property
    def rank(self) -> int:
        return _LEVEL_ORDER.index(self)


_LEVEL_ORDER: tuple[ReadinessLevel, ...] = (
    ReadinessLevel.DRAFT,
    ReadinessLevel.SIMREADY_SCENE_READY,
    ReadinessLevel.FUNCTIONAL_SCENE_READY,
    ReadinessLevel.ROBOT_FEASIBLE_SCENE_READY,
    ReadinessLevel.TASK_VERIFIED,
)


class Outcome(str, Enum):
    """Terminal outcomes that are not success levels and must never be reported as one.

    Kept separate from :class:`ReadinessLevel` so that a caller cannot compare them, and so
    that ``failed`` cannot be mistaken for a low-but-acceptable level.
    """

    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    BUDGET_EXHAUSTED = "budget_exhausted"


class GateId(str, Enum):
    """Named gates. Evidence keys must be members of this set."""

    SEMANTICS_HARD_ENTITIES = "semantics.hard_entities_preserved"
    SEMANTICS_TASK_STRUCTURE = "semantics.task_structure_covered"
    SEMANTICS_RANDOMIZATION_INVARIANTS = "semantics.randomization_invariants"
    GEOMETRY_NO_PENETRATION = "geometry.no_penetration"
    GEOMETRY_TASK_CELL_CLEAR = "geometry.task_cell_clear"
    GEOMETRY_SUPPORTS = "geometry.supports"
    GEOMETRY_IN_BOUNDS = "geometry.in_bounds"
    PHYSICS_SETTLE = "physics.settle_stability"
    PHYSICS_CLOCK = "physics.clock_advanced"
    VISUAL_VIEWS = "visual.required_views"
    VISUAL_CRITIQUE = "visual.critique"
    FUNCTION_INITIAL_RELATIONS = "function.initial_relations"
    FUNCTION_SPACE = "function.space_conditions"
    # Whether the robot can get to the piece the task is about. A *reach proxy*: a base position
    # exists, the target is inside the arm's reach from it, and the lane between is clear. It is
    # not `robot.ik` and must not be read as one -- no grasp is claimed.
    FUNCTION_TASK_REACHABLE = "function.task_reachable"
    # Whether the shell that was built is the shell the plan asked for. It exists because a room
    # generator that accepts a floor plan and does not hand its geometry back (ADR-002) leaves no
    # other evidence: without this gate, "we gave Infinigen a plan" and "we built a room" are two
    # facts with nothing connecting them. See SPEC-004 for the comparison set, which is only the
    # quantities the plan can actually constrain.
    SHELL_MATCHES_PLAN = "shell.matches_plan"
    ROBOT_NAV = "robot.nav"
    ROBOT_IK = "robot.ik"
    ROBOT_MOTION = "robot.motion"
    TASK_ROLLOUT = "task.rollout"


def backend_gates(backend: str) -> tuple[str, ...]:
    """The per-backend load, run and render gates.

    Namespaced by backend so that "isaac ran" and "mujoco ran" are separate evidence. A
    single shared ``backend.run`` gate would let one backend's success stand in for both,
    which is exactly what the manual means by "single-engine pass does not offset a
    required engine failing".
    """
    return (
        f"backend.{backend}.load",
        f"backend.{backend}.run",
        f"backend.{backend}.render",
    )


# Gates required per level, excluding the per-backend ones which depend on the request.
_LEVEL_GATES: Mapping[ReadinessLevel, tuple[str, ...]] = {
    ReadinessLevel.DRAFT: (),
    ReadinessLevel.SIMREADY_SCENE_READY: (
        GateId.SEMANTICS_HARD_ENTITIES.value,
        GateId.SEMANTICS_TASK_STRUCTURE.value,
        GateId.SEMANTICS_RANDOMIZATION_INVARIANTS.value,
        GateId.GEOMETRY_NO_PENETRATION.value,
        GateId.GEOMETRY_SUPPORTS.value,
        GateId.GEOMETRY_IN_BOUNDS.value,
        GateId.PHYSICS_SETTLE.value,
        GateId.PHYSICS_CLOCK.value,
        GateId.VISUAL_VIEWS.value,
        GateId.VISUAL_CRITIQUE.value,
    ),
    ReadinessLevel.FUNCTIONAL_SCENE_READY: (
        GateId.SEMANTICS_HARD_ENTITIES.value,
        GateId.SEMANTICS_TASK_STRUCTURE.value,
        GateId.SEMANTICS_RANDOMIZATION_INVARIANTS.value,
        GateId.GEOMETRY_NO_PENETRATION.value,
        GateId.GEOMETRY_SUPPORTS.value,
        GateId.GEOMETRY_IN_BOUNDS.value,
        GateId.PHYSICS_SETTLE.value,
        GateId.PHYSICS_CLOCK.value,
        GateId.VISUAL_VIEWS.value,
        GateId.VISUAL_CRITIQUE.value,
        GateId.GEOMETRY_TASK_CELL_CLEAR.value,
        GateId.FUNCTION_INITIAL_RELATIONS.value,
        GateId.FUNCTION_SPACE.value,
        GateId.FUNCTION_TASK_REACHABLE.value,
        GateId.SHELL_MATCHES_PLAN.value,
    ),
    ReadinessLevel.ROBOT_FEASIBLE_SCENE_READY: (
        GateId.SEMANTICS_HARD_ENTITIES.value,
        GateId.SEMANTICS_TASK_STRUCTURE.value,
        GateId.SEMANTICS_RANDOMIZATION_INVARIANTS.value,
        GateId.GEOMETRY_NO_PENETRATION.value,
        GateId.GEOMETRY_SUPPORTS.value,
        GateId.GEOMETRY_IN_BOUNDS.value,
        GateId.PHYSICS_SETTLE.value,
        GateId.PHYSICS_CLOCK.value,
        GateId.VISUAL_VIEWS.value,
        GateId.VISUAL_CRITIQUE.value,
        GateId.GEOMETRY_TASK_CELL_CLEAR.value,
        GateId.FUNCTION_INITIAL_RELATIONS.value,
        GateId.FUNCTION_SPACE.value,
        GateId.FUNCTION_TASK_REACHABLE.value,
        GateId.SHELL_MATCHES_PLAN.value,
        GateId.ROBOT_NAV.value,
        GateId.ROBOT_IK.value,
        GateId.ROBOT_MOTION.value,
    ),
    ReadinessLevel.TASK_VERIFIED: (
        GateId.SEMANTICS_HARD_ENTITIES.value,
        GateId.SEMANTICS_TASK_STRUCTURE.value,
        GateId.SEMANTICS_RANDOMIZATION_INVARIANTS.value,
        GateId.GEOMETRY_NO_PENETRATION.value,
        GateId.GEOMETRY_SUPPORTS.value,
        GateId.GEOMETRY_IN_BOUNDS.value,
        GateId.PHYSICS_SETTLE.value,
        GateId.PHYSICS_CLOCK.value,
        GateId.VISUAL_VIEWS.value,
        GateId.VISUAL_CRITIQUE.value,
        GateId.GEOMETRY_TASK_CELL_CLEAR.value,
        GateId.FUNCTION_INITIAL_RELATIONS.value,
        GateId.FUNCTION_SPACE.value,
        GateId.FUNCTION_TASK_REACHABLE.value,
        GateId.SHELL_MATCHES_PLAN.value,
        GateId.ROBOT_NAV.value,
        GateId.ROBOT_IK.value,
        GateId.ROBOT_MOTION.value,
        GateId.TASK_ROLLOUT.value,
    ),
}


def required_gates(level: ReadinessLevel, requested_backends: Sequence[str]) -> tuple[str, ...]:
    """Every gate that must pass for ``level``, including per-backend gates.

    Duplicates are removed but order is preserved, so reports read in a stable order.
    """
    if not requested_backends:
        raise ReadinessError("at least one requested backend is required to resolve readiness")
    gates: list[str] = list(_LEVEL_GATES[level])
    for backend in requested_backends:
        if not backend:
            raise ReadinessError("requested backend names must be non-empty")
        gates.extend(backend_gates(backend))
    seen: set[str] = set()
    ordered: list[str] = []
    for gate in gates:
        if gate not in seen:
            seen.add(gate)
            ordered.append(gate)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class GateBlocker:
    """Why a gate does not count, in a form a report can print verbatim."""

    gate_id: str
    status: EvidenceStatus
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"gate_id": self.gate_id, "status": self.status.value, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ReadinessResolution:
    """The outcome of resolving a ledger against a requested level.

    ``achieved`` is the highest level whose gates all pass. It may be lower than requested,
    and ``achieved`` is never a non-success :class:`Outcome`; those live in ``outcome`` so
    that a caller cannot accidentally treat "failed" as a level.
    """

    requested: ReadinessLevel
    achieved: ReadinessLevel
    outcome: Outcome | None
    scene_version: str
    requested_backends: tuple[str, ...]
    blockers: tuple[GateBlocker, ...]
    stale_gates: tuple[str, ...] = ()
    unsupported_gates: tuple[str, ...] = ()
    failed_gates: tuple[str, ...] = ()
    not_run_gates: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    # Keys `to_dict` emits that are not stored fields, with the JSON Schema fragment for each. The
    # emitter reads this so the document it produces describes the document that is actually
    # written: without it, `meets_request` was in every manifest and in no schema, and
    # `tests/contract/test_manifest_matches_schema.py` rejected a real delivery over it. Derived
    # rather than stored because storing it would let it disagree with `outcome` and `achieved`.
    SERIALISED_DERIVED: ClassVar[Mapping[str, Any]] = {"meets_request": {"type": "boolean"}}

    def __post_init__(self) -> None:
        if not is_digest(self.scene_version):
            raise ReadinessError(f"scene_version must be a scene hash, got {self.scene_version!r}")

    @property
    def meets_request(self) -> bool:
        """Whether the requested level was reached with no disqualifying outcome."""
        return not self.blockers and self.outcome is None and self.achieved.rank >= self.requested.rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested": self.requested.value,
            "achieved": self.achieved.value,
            "outcome": None if self.outcome is None else self.outcome.value,
            "meets_request": self.meets_request,
            "scene_version": self.scene_version,
            "requested_backends": list(self.requested_backends),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "stale_gates": list(self.stale_gates),
            "unsupported_gates": list(self.unsupported_gates),
            "failed_gates": list(self.failed_gates),
            "not_run_gates": list(self.not_run_gates),
        }


def _classify(ledger: EvidenceLedger, gate_ids: Iterable[str], scene_version: str) -> tuple[
    tuple[GateBlocker, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    blockers: list[GateBlocker] = []
    stale: list[str] = []
    unsupported: list[str] = []
    failed: list[str] = []
    not_run: list[str] = []

    for gate_id in gate_ids:
        entry = ledger.get(gate_id, scene_version)
        if entry is None:
            # Nothing at *this* version. Whether that is "never measured" or "measured on a scene
            # that is gone" is a difference a run's report has to make: the first is an unfinished
            # job, the second is evidence that has been superseded -- and a pass from an older scene
            # version is the more dangerous of the two, because it looks like a success.
            older = ledger.latest_for(gate_id)
            if older is not None:
                stale.append(gate_id)
                blockers.append(
                    GateBlocker(
                        gate_id,
                        older.status,
                        f"evidence was measured on scene {older.state_hash[:12]} but not on the "
                        f"current scene {scene_version[:12]}",
                    )
                )
            else:
                not_run.append(gate_id)
                blockers.append(
                    GateBlocker(
                        gate_id, EvidenceStatus.NOT_RUN, "no evidence was recorded for this gate"
                    )
                )
            continue
        if entry.status is EvidenceStatus.PASS:
            continue
        if entry.status is EvidenceStatus.UNSUPPORTED:
            unsupported.append(gate_id)
        elif entry.status is EvidenceStatus.FAIL:
            failed.append(gate_id)
        else:
            not_run.append(gate_id)
        blockers.append(
            GateBlocker(
                gate_id,
                entry.status,
                entry.detail or f"gate reported {entry.status.value}",
            )
        )

    return tuple(blockers), tuple(stale), tuple(unsupported), tuple(failed), tuple(not_run)


def resolve_readiness(
    ledger: EvidenceLedger,
    *,
    scene_version: str,
    requested: ReadinessLevel,
    requested_backends: Sequence[str],
) -> ReadinessResolution:
    """Resolve the achieved level for a ledger.

    The level is the highest one whose *every* required gate is a current pass. A gate that is
    unsupported, failed, stale or simply absent blocks the level it belongs to.

    ``outcome`` is set when the blocked gates include an ``unsupported`` or a ``fail`` for the
    *requested* level: an unsupported required capability means the request cannot be met at
    all, and a failure means it was attempted and did not succeed. Neither is reported as a
    lower success level, because that is how a failed scene gets shipped as a partial one.
    """
    if not is_digest(scene_version):
        raise ReadinessError(f"scene_version must be a scene hash, got {scene_version!r}")

    backends = tuple(requested_backends)
    achieved = ReadinessLevel.DRAFT
    requested_blockers: tuple[GateBlocker, ...] = ()
    requested_stale: tuple[str, ...] = ()
    requested_unsupported: tuple[str, ...] = ()
    requested_failed: tuple[str, ...] = ()
    requested_not_run: tuple[str, ...] = ()

    for level in _LEVEL_ORDER:
        gates = required_gates(level, backends)
        blockers, stale, unsupported, failed, not_run = _classify(ledger, gates, scene_version)
        if level is requested:
            requested_blockers = blockers
            requested_stale = stale
            requested_unsupported = unsupported
            requested_failed = failed
            requested_not_run = not_run
        if not blockers:
            achieved = level

    # Evaluate the requested level on its own so that a lower level passing cannot mask an
    # unsupported or failed required capability.
    if not requested_blockers:
        blockers, stale, unsupported, failed, not_run = _classify(
            ledger, required_gates(requested, backends), scene_version
        )
    else:
        blockers = requested_blockers
        stale, unsupported, failed, not_run = (
            requested_stale,
            requested_unsupported,
            requested_failed,
            requested_not_run,
        )

    outcome: Outcome | None = None
    if unsupported and requested.rank > achieved.rank:
        outcome = Outcome.UNSUPPORTED
    elif failed and requested.rank > achieved.rank:
        outcome = Outcome.FAILED

    return ReadinessResolution(
        requested=requested,
        achieved=achieved,
        outcome=outcome,
        scene_version=scene_version,
        requested_backends=backends,
        blockers=blockers,
        stale_gates=stale,
        unsupported_gates=unsupported,
        failed_gates=failed,
        not_run_gates=not_run,
    )
