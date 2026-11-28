# SPDX-License-Identifier: Apache-2.0
"""Typed failure events: the only thing that wakes the agent (architecture §16).

The research draft is explicit that the agent loop is not ``LLM -> tool -> LLM -> tool`` (§15).
The executor runs the geometry, and the agent is woken by a named event (§16):

    ASSET_EXHAUSTED      LAYOUT_DEADLOCK      TASK_UNREACHABLE
    RELATION_UNSATISFIABLE  ARTICULATION_FAILURE  DEFORMABLE_SIM_FAILURE
    SEMANTIC_CONFLICT    STAGE_COMPLETE

Four rules are implemented here rather than left to discipline, because each one is a way the
escalation ladder starts repairing the wrong thing:

* **an event comes from a gate, never from prose.** The kind is derived from the gate id and the
  typed :class:`FailureCode`, so two runs that fail the same way produce the same event. An event
  built from a message would let a run look novel because a number in a sentence changed.
* **stale evidence raises nothing.** A ``FAIL`` measured on an older scene version describes a
  scene that no longer exists. Escalating on it would spend the ladder repairing something the
  current scene does not have, so staleness is checked before status -- the same ordering the
  readiness resolver uses, and for the same reason.
* **``unsupported`` is not an event.** An unimplemented capability is a declared boundary, reported
  through :class:`~rfw_w.scenes.contracts.readiness.Outcome`, not a failure to escalate on. Emitting
  ``ARTICULATION_FAILURE`` for every run that declares articulated assets unsupported would put a
  permanent, unfixable entry in the ladder and make the escalation history unreadable.
* **the targets are the objects the gate named.** They are read from the evidence's witness
  references, which are namespaced for exactly this purpose (``instance:<id>:outside-room``,
  ``pair:<a>|<b>:overlap``). A gate that cannot name the object it failed on cannot drive a repair
  of it, and no free-text parsing is allowed to guess.

``STAGE_COMPLETE`` is the one kind not derived from a failure. Per the executable design §10 it
advances state and does not wake a model, so it is constructed explicitly rather than inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from rfw_w.scenes.contracts.tool import Evidence, EvidenceLedger, EvidenceStatus, FailureCode

SCHEMA_VERSION = "pvscene.events/v1"


class EventError(ValueError):
    """An event was constructed in a way the ladder cannot interpret."""


class EventKind(str, Enum):
    """The eight events from architecture §16. Closed: a ninth is a design decision."""

    ASSET_EXHAUSTED = "ASSET_EXHAUSTED"
    LAYOUT_DEADLOCK = "LAYOUT_DEADLOCK"
    TASK_UNREACHABLE = "TASK_UNREACHABLE"
    RELATION_UNSATISFIABLE = "RELATION_UNSATISFIABLE"
    ARTICULATION_FAILURE = "ARTICULATION_FAILURE"
    DEFORMABLE_SIM_FAILURE = "DEFORMABLE_SIM_FAILURE"
    SEMANTIC_CONFLICT = "SEMANTIC_CONFLICT"
    STAGE_COMPLETE = "STAGE_COMPLETE"

    @property
    def is_failure(self) -> bool:
        """``STAGE_COMPLETE`` is progress, not a failure, and must not escalate."""
        return self is not EventKind.STAGE_COMPLETE


# Gate id to event kind. Matching is by exact gate id first, then by prefix, so a gate this
# module has never heard of is still classified by its family rather than silently dropped.
_GATE_KINDS: Mapping[str, EventKind] = {
    "semantics.hard_entities_preserved": EventKind.SEMANTIC_CONFLICT,
    "geometry.task_cell_clear": EventKind.LAYOUT_DEADLOCK,
    "geometry.no_penetration": EventKind.LAYOUT_DEADLOCK,
    "geometry.in_bounds": EventKind.LAYOUT_DEADLOCK,
    "geometry.wall_clearance": EventKind.LAYOUT_DEADLOCK,
    "geometry.supports": EventKind.LAYOUT_DEADLOCK,
    "function.initial_relations": EventKind.RELATION_UNSATISFIABLE,
    "function.space_conditions": EventKind.TASK_UNREACHABLE,
    "visual.required_views": EventKind.SEMANTIC_CONFLICT,
    "visual.critique": EventKind.SEMANTIC_CONFLICT,
}

_PREFIX_KINDS: tuple[tuple[str, EventKind], ...] = (
    ("geometry.", EventKind.LAYOUT_DEADLOCK),
    ("physics.", EventKind.LAYOUT_DEADLOCK),
    ("robot.", EventKind.TASK_UNREACHABLE),
    ("task.", EventKind.TASK_UNREACHABLE),
    ("articulated.", EventKind.ARTICULATION_FAILURE),
    ("deformable.", EventKind.DEFORMABLE_SIM_FAILURE),
    ("visual.", EventKind.SEMANTIC_CONFLICT),
    ("function.", EventKind.SEMANTIC_CONFLICT),
    ("semantics.", EventKind.SEMANTIC_CONFLICT),
    # A shell that does not match its plan is not a layout problem, a clearance problem or a
    # rendering problem -- there is no atom that moves a wall. It is classified `SEMANTIC_CONFLICT`
    # because that is the kind whose repair is "the request and the artefact disagree", and the
    # only honest answer here is to re-read the request. The prefix exists at all because without
    # it `kind_for_gate` returns `None`, the gate raises no event, and a failure that should stop
    # a stage is filed as unclassified and noticed by nobody.
    ("shell.", EventKind.SEMANTIC_CONFLICT),
)

# The typed code the repair policy switches on. Keyed by gate id, because the same code means
# different things in different gates -- a clearance violation in the passage gate is not one in
# the wall gate.
_GATE_CODES: Mapping[str, FailureCode] = {
    "geometry.task_cell_clear": FailureCode.CLEARANCE_VIOLATION,
    "geometry.no_penetration": FailureCode.GEOMETRY_PENETRATION,
    "geometry.in_bounds": FailureCode.GEOMETRY_OUT_OF_BOUNDS,
    "geometry.wall_clearance": FailureCode.GEOMETRY_OUT_OF_BOUNDS,
    "geometry.supports": FailureCode.SUPPORT_NOT_FOUND,
    "function.initial_relations": FailureCode.FUNCTION_PREDICATE_FAILED,
    "function.space_conditions": FailureCode.CLEARANCE_VIOLATION,
    "physics.clock_advanced": FailureCode.PHYSICS_CLOCK_STALLED,
    "visual.required_views": FailureCode.RENDER_FAILED,
    "visual.critique": FailureCode.VISUAL_GATE_FAILED,
    "semantics.hard_entities_preserved": FailureCode.HARD_GOAL_DROPPED,
    # Explicit rather than left to the `PRECONDITION_UNMET` fallback so the mapping is reviewable
    # in one place. It is the same value, and it is the right one: a shell that is not the planned
    # room breaks the precondition every gate below it is standing on, and no repair atom moves a
    # wall -- the ladder's honest move here is to stop, not to re-place furniture in a room that
    # was never the requested one.
    "shell.matches_plan": FailureCode.PRECONDITION_UNMET,
}

# Gates whose failure says nothing about the scene, so retrying is the right response and
# escalating is not. The loop already routes these through the infrastructure allowance
# (``StageOutcome.infrastructure_failure``); this keeps an event from being raised for them too.
_INFRASTRUCTURE_PREFIXES = ("backend.",)


def kind_for_gate(gate_id: str) -> EventKind | None:
    """The event a gate's failure raises, or ``None`` if it raises none.

    ``None`` is a real answer: backend gates fail for reasons that are not about the scene, and a
    gate nobody has classified must be visible as unclassified rather than filed under a guess.
    """
    if gate_id.startswith(_INFRASTRUCTURE_PREFIXES):
        return None
    exact = _GATE_KINDS.get(gate_id)
    if exact is not None:
        return exact
    for prefix, kind in _PREFIX_KINDS:
        if gate_id.startswith(prefix):
            return kind
    return None


def code_for_gate(gate_id: str, *, status: EvidenceStatus = EvidenceStatus.FAIL) -> FailureCode:
    """The typed failure code for a gate. An invalid critique is its own code.

    A critique that could not be parsed is not the same failure as one that scored the scene badly:
    the first is a request or schema problem and the second is a scene problem, and repairing the
    scene for the first wastes the ladder on a scene that may be fine.
    """
    if gate_id == "visual.critique" and status is not EvidenceStatus.FAIL:
        return FailureCode.VISUAL_CRITIQUE_INVALID
    return _GATE_CODES.get(gate_id, FailureCode.PRECONDITION_UNMET)


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """One event, with everything the ladder needs to choose an atom.

    ``targets`` is the set of instance or region ids the source gate named. An event with no
    targets is legal and means "this gate fails, but not about one object" -- a passage failure is
    a property of the room, not of a single piece -- and the ladder is expected to treat it as
    broader rather than to guess a target.
    """

    kind: EventKind
    source_gate: str
    targets: tuple[str, ...] = ()
    failure_code: FailureCode | None = None
    detail: str = ""
    witnesses: tuple[str, ...] = ()
    # Where the event came from: a measured gate, a critique, or the agent's own patch. Recorded so
    # the escalation history says who asked for the repair, not only what was repaired.
    origin: str = "gate"

    def __post_init__(self) -> None:
        if not self.source_gate:
            raise EventError("an event must name the gate or stage that raised it")
        if self.kind.is_failure and self.failure_code is None:
            raise EventError(
                f"{self.kind.value} must carry a typed failure code; the repair policy switches on "
                "the code, and free text would make two identical failures look different"
            )

    @property
    def signature(self) -> str:
        """The identity of this event for repeat detection.

        Composed from the kind and the typed code, and **not** from the targets or the detail. A
        repair that fixes one of two colliding pieces produces a new failing pair with the same
        signature, which is correct: the same problem is still there and the ladder should keep
        escalating rather than restart at L0.
        """
        code = self.failure_code.value if self.failure_code else self.kind.value
        return f"{self.kind.value}:{code}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source_gate": self.source_gate,
            "targets": list(self.targets),
            "failure_code": None if self.failure_code is None else self.failure_code.value,
            "detail": self.detail,
            "witnesses": list(self.witnesses),
            "origin": self.origin,
            "signature": self.signature,
        }


def targets_from_witnesses(witnesses: Sequence[str]) -> tuple[str, ...]:
    """The instance and region ids named by a gate's witness references.

    Recognised forms, which are the ones the gates actually emit:

    * ``instance:<id>:<what>`` -- one instance, e.g. ``instance:desk:inside-wall``
    * ``pair:<a>|<b>:<what>``    -- a relation between two, e.g. ``pair:cup|desk:overlap``
    * ``region:<id>``            -- a zone or region, e.g. ``region:passage``

    Anything else is ignored for target extraction and still carried through in ``witnesses``. It
    is deliberately *not* parsed leniently: a witness format nobody defined would produce targets
    that name nothing, and a repair aimed at an id that does not exist is worse than no repair.
    """
    found: list[str] = []
    for witness in witnesses:
        parts = str(witness).split(":")
        if len(parts) < 2:
            continue
        head, body = parts[0], parts[1]
        if head == "instance" and body:
            found.append(body)
        elif head == "pair" and body:
            found.extend(part for part in body.split("|") if part)
        elif head == "region" and body:
            found.append(body)
    # Sorted and de-duplicated: the event's targets are a set of objects, and the order the gate
    # happened to list them in is not information.
    return tuple(sorted(dict.fromkeys(found)))


def events_from_evidence(evidence: Evidence, *, scene_version: str) -> tuple[FailureEvent, ...]:
    """The events one piece of evidence raises. Usually zero or one.

    A current ``FAIL`` raises its gate's event. Everything else raises nothing: a pass is not an
    event, a ``NOT_RUN`` gate is an unfinished job rather than a failure, ``unsupported`` is a
    declared boundary, and a failure from a different scene version describes a scene that is gone.
    """
    if evidence.status is not EvidenceStatus.FAIL:
        return ()
    if not evidence.is_current_for(scene_version):
        return ()
    kind = kind_for_gate(evidence.gate_id)
    if kind is None:
        return ()
    return (
        FailureEvent(
            kind=kind,
            source_gate=evidence.gate_id,
            targets=targets_from_witnesses(evidence.witness_refs),
            failure_code=code_for_gate(evidence.gate_id, status=evidence.status),
            detail=evidence.detail,
            witnesses=evidence.witness_refs,
            origin="gate",
        ),
    )


def events_from_ledger(
    ledger: EvidenceLedger, *, scene_version: str, gates: Sequence[str] | None = None
) -> tuple[FailureEvent, ...]:
    """Every event the ledger currently raises, in a stable order.

    ``gates`` restricts the sweep to the gates the caller is at this stage responsible for. The
    default is the whole ledger, which is what a driver that has just finished a segment wants.
    """
    names = tuple(gates) if gates is not None else ledger.gate_ids()
    raised: list[FailureEvent] = []
    for gate_id in names:
        # For this scene, not the last one recorded: a gate that failed on a scene the run has
        # already moved past is not a failure to work on, and raising it would put the ladder back
        # on a problem the repair just answered.
        entry = ledger.get(gate_id, scene_version)
        if entry is None:
            continue
        raised.extend(events_from_evidence(entry, scene_version=scene_version))
    return tuple(raised)


def stage_complete(stage: str, *, detail: str = "") -> FailureEvent:
    """``STAGE_COMPLETE``: a stage finished.

    It carries no failure code because it is not a failure, and per the executable design §10 it
    advances state without waking a model. It exists as an event so that the run's event log shows
    the stages that completed, not only the ones that broke.
    """
    return FailureEvent(kind=EventKind.STAGE_COMPLETE, source_gate=stage, detail=detail)
