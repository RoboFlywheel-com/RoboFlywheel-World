# SPDX-License-Identifier: Apache-2.0
"""The failure escalation ladder (architecture §17) and its ordering rule.

    L0 pose resample -> L1 local optimisation -> L2 alternative asset
      -> L3 alternative support / local layout -> L4 subgraph rollback -> L5 agent replan

    local recovery first  ->  semantic reasoning last

The ordering is the whole content of the rule, so it is enforced here rather than described. A
caller that asks for a level while a cheaper one is untried is refused, which is what stops the
ladder degrading into "ask the model whenever anything is wrong" -- the failure mode the design's
§29 ablation ("full ReAct / sparse agent") exists to measure.

Two tracks, and the split is load-bearing rather than cosmetic:

* the **geometric** track repairs the scene's arrangement. Every rung changes geometry, so every
  rung invalidates the physics evidence and the render.
* the **appearance** track repairs lighting and material. Per the executable design §9 a
  material/lighting change does **not** re-pass the physics gates but **does** invalidate the
  render and the critique, so it re-renders and re-critiques and nothing else.

They meet only at L5. A lighting adjustment cannot fix a penetration and a penetration cannot be
fixed by one, so an atom from one track is refused for a failure on the other. Without that rule
the appearance atom becomes a fifth way to move a chair, and a lighting complaint becomes a
reason to re-solve the floor plan.

L0 and L1 are both pose work and are genuinely different operations, which is why the design lists
them separately: **L0 re-runs the same solver over the same search space with a different
candidate offset** (the answer may simply have been the first one tried), while **L1 changes the
constraint being satisfied** (stand further off the wall, search a wider span, honour a margin that
was not being honoured). L0 is a resample and L1 is a re-solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping

from rfw_w.scenes.orchestrator.events import EventKind, FailureEvent

SCHEMA_VERSION = "pvscene.escalation/v1"


class EscalationError(RuntimeError):
    """The ladder was asked to do something the ordering forbids."""


class Track(str, Enum):
    """Which evidence an atom invalidates. The two tracks do not share a ladder."""

    GEOMETRIC = "geometric"
    APPEARANCE = "appearance"


class RepairLevel(IntEnum):
    """The six levels of §17, in order. The value is the rank."""

    L0_POSE_RESAMPLE = 0
    L1_LOCAL_OPTIMISATION = 1
    L2_ALTERNATIVE_ASSET = 2
    L3_ALTERNATIVE_SUPPORT = 3
    L4_SUBGRAPH_ROLLBACK = 4
    L5_AGENT_REPLAN = 5


class Atom(str, Enum):
    """The repair atoms. Each is a semantic state transition, not a parameter write.

    §25 of the research draft rules out ``set_x`` / ``set_y`` / ``set_yaw`` (a parameter write with
    no verifiable state transition) and rules out ``make_scene_good`` (no defined transition at
    all). What remains is a named operation that can be re-measured afterwards and undone.
    """

    RESAMPLE_POSE = "resample_pose"
    OPTIMISE_POSE = "optimise_pose"
    SWAP_ASSET = "swap_asset"
    CHANGE_SUPPORT = "change_support"
    ROLLBACK_SUBGRAPH = "rollback_subgraph"
    REPLAN = "replan"
    ADJUST_LIGHTING_MATERIAL = "adjust_lighting_material"


@dataclass(frozen=True, slots=True)
class AtomInfo:
    """What an atom is, what it costs, and what it invalidates."""

    atom: Atom
    track: Track
    level: RepairLevel
    stage: str
    description: str
    # Whether the atom spends one of the scene's model requests. Only L5 does.
    needs_model: bool = False
    # Whether the atom may remove an optional entity. Only L5 may, and only optional ones --
    # otherwise "delete the furniture" becomes the cheapest way to clear any deadlock and the room
    # empties while every gate passes.
    may_remove: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom": self.atom.value, "track": self.track.value, "level": self.level.name,
            "stage": self.stage, "needs_model": self.needs_model, "may_remove": self.may_remove,
        }


ATOMS: Mapping[Atom, AtomInfo] = {
    Atom.RESAMPLE_POSE: AtomInfo(
        atom=Atom.RESAMPLE_POSE, track=Track.GEOMETRIC, level=RepairLevel.L0_POSE_RESAMPLE,
        stage="repair.resample_pose",
        description="re-run the placement sweep for the target with a different candidate offset",
    ),
    Atom.OPTIMISE_POSE: AtomInfo(
        atom=Atom.OPTIMISE_POSE, track=Track.GEOMETRIC, level=RepairLevel.L1_LOCAL_OPTIMISATION,
        stage="repair.optimise_pose",
        description="re-place the target against the constraint that was violated",
    ),
    Atom.SWAP_ASSET: AtomInfo(
        atom=Atom.SWAP_ASSET, track=Track.GEOMETRIC, level=RepairLevel.L2_ALTERNATIVE_ASSET,
        stage="repair.swap_asset",
        description="take the next retrieval candidate for the same entity and query",
    ),
    Atom.CHANGE_SUPPORT: AtomInfo(
        atom=Atom.CHANGE_SUPPORT, track=Track.GEOMETRIC, level=RepairLevel.L3_ALTERNATIVE_SUPPORT,
        stage="repair.change_support",
        description="put the target on a different support surface, wall or zone",
    ),
    Atom.ROLLBACK_SUBGRAPH: AtomInfo(
        atom=Atom.ROLLBACK_SUBGRAPH, track=Track.GEOMETRIC, level=RepairLevel.L4_SUBGRAPH_ROLLBACK,
        stage="repair.rollback_subgraph",
        description="restore the nearest checkpoint covering the dependency closure",
    ),
    Atom.REPLAN: AtomInfo(
        atom=Atom.REPLAN, track=Track.GEOMETRIC, level=RepairLevel.L5_AGENT_REPLAN,
        stage="repair.replan",
        description="one model request: the replanner returns a schema'd patch",
        needs_model=True, may_remove=True,
    ),
    Atom.ADJUST_LIGHTING_MATERIAL: AtomInfo(
        atom=Atom.ADJUST_LIGHTING_MATERIAL, track=Track.APPEARANCE,
        level=RepairLevel.L1_LOCAL_OPTIMISATION, stage="repair.appearance",
        description="change lighting or material: no geometry moves, and the render is redone",
    ),
}

# The ordered ladder per event kind, on each track. Order within a tuple *is* the escalation rule:
# a caller takes the first entry it has not tried, so it cannot reach a rung without passing the
# ones before it.
_GEOMETRIC_LADDERS: Mapping[EventKind, tuple[Atom, ...]] = {
    EventKind.LAYOUT_DEADLOCK: (
        Atom.RESAMPLE_POSE, Atom.OPTIMISE_POSE, Atom.SWAP_ASSET, Atom.CHANGE_SUPPORT,
        Atom.ROLLBACK_SUBGRAPH, Atom.REPLAN,
    ),
    EventKind.RELATION_UNSATISFIABLE: (
        Atom.RESAMPLE_POSE, Atom.OPTIMISE_POSE, Atom.SWAP_ASSET, Atom.CHANGE_SUPPORT,
        Atom.ROLLBACK_SUBGRAPH, Atom.REPLAN,
    ),
    # A passage is a property of the room, so there is no single pose to resample: the piece that
    # is in the way has to be moved or moved elsewhere, which is L1 and L3.
    EventKind.TASK_UNREACHABLE: (
        Atom.OPTIMISE_POSE, Atom.CHANGE_SUPPORT, Atom.ROLLBACK_SUBGRAPH, Atom.REPLAN,
    ),
    # The agent is the last resort, and for an exhausted retrieval it is the first resort: there
    # is no candidate left to swap to, so re-asking the catalogue under the same query is pointless.
    EventKind.ASSET_EXHAUSTED: (Atom.REPLAN,),
    # Unreachable today: no articulated or deformable asset exists in the library, so no gate can
    # fail on one. They are here because §16 names them and because the capability list already
    # declares them unsupported -- and they are marked unreachable rather than presented as working.
    EventKind.ARTICULATION_FAILURE: (Atom.REPLAN,),
    EventKind.DEFORMABLE_SIM_FAILURE: (Atom.REPLAN,),
    # A semantic complaint from the critic is not a geometric failure. The geometric rungs are
    # admissible only where the complaint names an object the geometry can answer for; the critic's
    # own suggestion is consulted by the repair policy, not here.
    EventKind.SEMANTIC_CONFLICT: (Atom.SWAP_ASSET, Atom.CHANGE_SUPPORT, Atom.REPLAN),
    EventKind.STAGE_COMPLETE: (),
}

_APPEARANCE_LADDERS: Mapping[EventKind, tuple[Atom, ...]] = {
    EventKind.SEMANTIC_CONFLICT: (Atom.ADJUST_LIGHTING_MATERIAL, Atom.REPLAN),
    EventKind.STAGE_COMPLETE: (),
}


def admissible_atoms(event: FailureEvent, *, track: Track = Track.GEOMETRIC) -> tuple[Atom, ...]:
    """The ordered atoms an event may use on one track.

    An event kind with no ladder on a track returns an empty tuple, which the caller must read as
    "this track cannot help here" rather than "anything goes".
    """
    table = _GEOMETRIC_LADDERS if track is Track.GEOMETRIC else _APPEARANCE_LADDERS
    return table.get(event.kind, ())


@dataclass(frozen=True, slots=True)
class Attempt:
    """One rung, tried, with what happened."""

    signature: str
    atom: Atom
    resolved: bool
    improved: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature, "atom": self.atom.value, "resolved": self.resolved,
            "improved": self.improved, "level": ATOMS[self.atom].level.name, "detail": self.detail,
        }


@dataclass(slots=True)
class Ladder:
    """Per-signature attempt history, and the ordering rule that reads it.

    ``next_untried`` is the executable design §10's ``repair_policy.next_untried(failure_signature)``:
    it returns the cheapest atom admissible for this event that has not been tried, or ``None`` when
    the ladder is exhausted and the failure has to become a semantic replan event.
    """

    attempts: list[Attempt] = field(default_factory=list)
    _tried: dict[str, set[Atom]] = field(default_factory=dict, repr=False)

    def tried(self, signature: str) -> frozenset[Atom]:
        return frozenset(self._tried.get(signature, ()))

    def next_untried(
        self, event: FailureEvent, *, track: Track = Track.GEOMETRIC
    ) -> Atom | None:
        """The next rung for this event, or ``None`` when every rung has been tried.

        The ladder is per *signature*, not per target: a repair that fixes one of two colliding
        pieces leaves the same failure signature present, and the right response is to continue up
        the ladder rather than to start again at L0 for the remaining pair.
        """
        already = self._tried.get(event.signature, set())
        for atom in admissible_atoms(event, track=track):
            if atom not in already:
                return atom
        return None

    def assert_order(self, event: FailureEvent, atom: Atom, *, track: Track = Track.GEOMETRIC) -> None:
        """Refuse an atom while a cheaper rung on the same track is untried.

        This is the rule the whole module exists for, stated as a check rather than as a comment. A
        caller -- including one acting on a critic's suggestion -- cannot skip to a more expensive
        atom because it believes it will work.
        """
        allowed = admissible_atoms(event, track=track)
        info = ATOMS.get(atom)
        if info is None:
            raise EscalationError(f"{atom} is not a known atom")
        if info.track is not track:
            raise EscalationError(
                f"{atom.value} is a {info.track.value} repair and cannot be used on the "
                f"{track.value} track; the two tracks invalidate different evidence"
            )
        if atom not in allowed:
            raise EscalationError(
                f"{atom.value} is not admissible for {event.kind.value}; admissible on this track: "
                f"{[candidate.value for candidate in allowed]}"
            )
        untried = self.next_untried(event, track=track)
        if untried is not None and ATOMS[untried].level < info.level:
            raise EscalationError(
                f"{atom.value} is {info.level.name} but {untried.value} at "
                f"{ATOMS[untried].level.name} has not been tried; local recovery comes first"
            )

    def record(
        self, event: FailureEvent, atom: Atom, *, resolved: bool, improved: bool, detail: str = ""
    ) -> Attempt:
        """Record a rung's outcome. ``improved`` is judged by the failure set, not by opinion.

        The executable design is explicit that progress is decided by the hard failure set shrinking
        or a residual falling, and not by a model saying it is fixed. ``improved=False`` on a rung
        that committed a change is the signal that the loop is thrashing rather than repairing.
        """
        attempt = Attempt(
            signature=event.signature, atom=atom, resolved=resolved, improved=improved, detail=detail
        )
        self.attempts.append(attempt)
        self._tried.setdefault(event.signature, set()).add(atom)
        return attempt

    # -- the two metrics from §28, computed from the history ------------------------------------

    @property
    def failures_seen(self) -> int:
        """Distinct failure signatures this run raised."""
        return len({attempt.signature for attempt in self.attempts})

    @property
    def resolved_without_model(self) -> int:
        """Signatures resolved by an atom that spends no model request (§28.3's numerator)."""
        done: set[str] = set()
        for attempt in self.attempts:
            if attempt.resolved and not ATOMS[attempt.atom].needs_model:
                done.add(attempt.signature)
        return len(done)

    @property
    def local_recovery_rate(self) -> float:
        """§28.3: failures solved without an LLM, over all failures.

        ``1.0`` when nothing failed, which is the honest reading: there were no failures to solve
        locally. It is not a perfect score and must not be averaged as one, so it is reported
        beside ``failures_seen``.
        """
        if not self.attempts:
            return 1.0
        return self.resolved_without_model / max(1, self.failures_seen)

    def model_requests_spent(self) -> int:
        return sum(1 for attempt in self.attempts if ATOMS[attempt.atom].needs_model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "failures_seen": self.failures_seen,
            "resolved_without_model": self.resolved_without_model,
            "local_recovery_rate": round(self.local_recovery_rate, 4),
            "model_requests_spent_on_repair": self.model_requests_spent(),
        }


def stage_names() -> tuple[str, ...]:
    """Every atom's orchestrator stage name, so a driver can register handlers for all of them."""
    return tuple(sorted(info.stage for info in ATOMS.values()))


def atom_for_stage(stage: str) -> Atom | None:
    """The atom a stage implements, or ``None`` if the stage is not a repair atom."""
    for atom, info in ATOMS.items():
        if info.stage == stage:
            return atom
    return None
