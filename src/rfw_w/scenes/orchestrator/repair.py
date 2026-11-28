# SPDX-License-Identifier: Apache-2.0
"""The repair atoms: what each one changes, and what that change invalidates.

Each atom edits the :class:`~rfw_w.scenes.solvers.furnish.ScenePlan` and declares the evidence its
edit destroys. The declaration is the load-bearing part, and it comes from the executable design:

* §9 -- *"a geometric change must re-pass the physics gate"*; *"a material/lighting change also
  invalidates the render and appearance evidence -- do not reuse the old critic"*.
* §10 -- the loop commits a delta and *"invalidates dependent evidence"*; a rollback must restore
  the evidence invalidation set as well as the pose.

What a moved box invalidates is therefore read from
:func:`~rfw_w.scenes.contracts.state.gates_for_evidence` rather than restated here. Two lists of the
same fact drift, and the drift shows up as a gate re-run that was not needed -- or, worse, as one
that was needed and was not done, which leaves a stale pass standing for a scene it never measured.

**An atom that cannot act must say so.** Every atom returns a
:class:`RepairOutcome` whose ``changed`` flag is honest: an atom that finds nothing to do reports
``changed=False`` and a reason, and the ladder records the attempt and moves up. An atom that
reports success while changing nothing is how a loop spends its budget looking busy, and the
no-progress rule would then be firing on a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Mapping

from rfw_w.scenes.contracts.state import (
    APPEARANCE_INVALIDATION,
    POSE_INVALIDATION,
    gates_for_evidence,
)
from rfw_w.scenes.orchestrator.escalation import ATOMS, Atom, EscalationError, Track
from rfw_w.scenes.solvers.furnish import AppearanceSettings, ScenePlan, SurfaceRequest

SCHEMA_VERSION = "pvscene.repair/v1"


class RepairError(RuntimeError):
    """A repair was asked for something the plan cannot express."""


@dataclass(frozen=True, slots=True)
class RepairAction:
    """One atom, aimed at something, for a recorded reason.

    ``source`` records who asked: a measured gate, the visual critic's suggestion, or the agent's
    own patch. The escalation history then reads as "the critic asked for this and the ladder
    allowed it", rather than as an anonymous list of edits.
    """

    atom: Atom
    targets: tuple[str, ...] = ()
    parameters: Mapping[str, Any] | None = None
    rationale: str = ""
    source: str = "gate"

    def __post_init__(self) -> None:
        if self.source not in ("gate", "critic", "agent"):
            raise RepairError(f"unknown repair source {self.source!r}")
        if self.atom is Atom.REPLAN and self.source != "agent":
            raise RepairError(
                "a replan is an agent action; recording it as anything else would hide the model "
                "request in the history that exists to count them"
            )

    @property
    def info(self):
        return ATOMS[self.atom]

    @property
    def stage(self) -> str:
        return ATOMS[self.atom].stage

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom": self.atom.value, "stage": self.stage, "targets": list(self.targets),
            "parameters": dict(self.parameters or {}), "rationale": self.rationale,
            "source": self.source, "level": self.info.level.name,
            "track": self.info.track.value, "needs_model": self.info.needs_model,
        }


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """What an atom did to the plan, and what that costs in evidence.

    ``invalidated_gates`` is what the caller hands to the ledger and to
    :meth:`Orchestrator.invalidate`. ``revalidates_physics`` is separate from it and deliberately so:
    it is the one-line answer to "does this repair have to re-pass the physics gate?", which the
    design makes a hard rule for geometry and a hard prohibition for lighting.
    """

    action: RepairAction
    plan: ScenePlan
    changed: bool
    notes: tuple[str, ...] = ()
    invalidated_gates: frozenset[str] = frozenset()
    revalidates_physics: bool = False
    removed: tuple[str, ...] = ()

    @property
    def atom(self) -> Atom:
        return self.action.atom

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "action": self.action.to_dict(),
            "changed": self.changed,
            "notes": list(self.notes),
            "invalidated_gates": sorted(self.invalidated_gates),
            "revalidates_physics": self.revalidates_physics,
            "removed": list(self.removed),
        }


def invalidated_gates_for(atom: Atom) -> frozenset[str]:
    """Every gate an atom's edit invalidates.

    Appearance and geometry are separate sets, not one set with an exception: a lighting change
    cannot invalidate a penetration measurement, because nothing moved, and pretending otherwise
    would re-run the physics for no reason and make the two loops indistinguishable.
    """
    track = ATOMS[atom].track
    kinds = POSE_INVALIDATION if track is Track.GEOMETRIC else APPEARANCE_INVALIDATION
    return gates_for_evidence(kinds)


def revalidates_physics_for(atom: Atom) -> bool:
    """Whether the atom's edit obliges a fresh physics measurement."""
    return ATOMS[atom].track is Track.GEOMETRIC


# The L1 strategies, as a closed set. Each one names the *constraint* being changed, because that is
# what separates L1 from L0: L0 resamples the same search space, L1 changes what the search is
# asked to satisfy.
OPTIMISE_STRATEGIES = (
    # A piece must stand against the wall's real face, not the outline it was drawn from.
    "honour-wall-face",
    # Pieces must keep their declared gap, so an overlap is answered by asking for more room.
    "widen-gap",
    # The same search, started further along: for a piece that a first-fit rule parked badly.
    "search-wider",
)

# Which strategy answers which typed failure. A failure code with no strategy is not an error: it
# means L1 has nothing specific to offer and the ladder should move up, which is what the atom
# reports by returning ``changed=False``.
_OPTIMISE_BY_CODE: Mapping[str, str] = {
    "GEOMETRY_OUT_OF_BOUNDS": "honour-wall-face",
    "GEOMETRY_PENETRATION": "widen-gap",
    "CLEARANCE_VIOLATION": "widen-gap",
}


def _targets_present(plan: ScenePlan, targets: tuple[str, ...]) -> tuple[str, ...]:
    """Which of the named targets the plan actually contains.

    Unknown names are dropped rather than raising: an event can name a region (``region:passage``)
    or a pair that includes a piece the plan has already removed, and refusing the whole repair
    over one stale name would strand the failure. What was dropped is returned to the caller so it
    lands in the notes instead of disappearing.

    The task target counts as present even though the plan places no row for it. It is the entity
    `prompt_adherence` is measured against, `swap_asset` is one of only two repairs that criterion
    admits, and the candidate it uses is a plan fact -- so dropping it here made the one repair the
    criterion could be answered by report "no target in the plan to swap" for its own target.
    """
    return tuple(
        name for name in targets
        if plan.furniture_for(name) or plan.surface_for(name) or name == plan.target_id
    )


def apply(
    plan: ScenePlan,
    action: RepairAction,
    *,
    wall_thickness_m: float = 0.1,
    failure_code: str = "",
) -> RepairOutcome:
    """Apply one atom to the plan and report what it invalidated.

    ``wall_thickness_m`` is passed in rather than guessed: it is a property of the room shell, and
    the atom that stands pieces off the wall face needs the real number.
    """
    invalidated = invalidated_gates_for(action.atom)
    revalidates = revalidates_physics_for(action.atom)
    known = _targets_present(plan, action.targets)
    dropped = tuple(name for name in action.targets if name not in known)

    def outcome(
        new_plan: ScenePlan, *, changed: bool, notes: tuple[str, ...], removed: tuple[str, ...] = ()
    ) -> RepairOutcome:
        return RepairOutcome(
            action=action, plan=new_plan, changed=changed,
            notes=notes + (("ignored unknown targets: " + ", ".join(dropped),) if dropped else ()),
            invalidated_gates=invalidated if changed else frozenset(),
            revalidates_physics=revalidates and changed,
            removed=removed,
        )

    if action.atom is Atom.RESAMPLE_POSE:
        return _resample(plan, known, outcome)
    if action.atom is Atom.OPTIMISE_POSE:
        return _optimise(plan, known, failure_code, wall_thickness_m, action, outcome)
    if action.atom is Atom.SWAP_ASSET:
        return _swap_asset(plan, known, outcome)
    if action.atom is Atom.CHANGE_SUPPORT:
        return _change_support(plan, known, outcome)
    if action.atom is Atom.ADJUST_LIGHTING_MATERIAL:
        return _adjust_appearance(plan, action, outcome)
    if action.atom is Atom.ROLLBACK_SUBGRAPH:
        # Restoring a checkpoint is the checkpoint stack's job, not the plan's: the atom reports
        # the invalidation and the caller supplies the restored plan. Doing it here would need the
        # stack, and giving every atom the stack is how an atom ends up able to undo its neighbours.
        return outcome(
            plan, changed=False,
            notes=(
                "a subgraph rollback is applied by the checkpoint stack; the caller restores the "
                "plan and this atom contributes its invalidation set",
            ),
        )
    if action.atom is Atom.REPLAN:
        return outcome(
            plan, changed=False,
            notes=(
                "a replan is applied by the replanner's patch; this atom contributes its "
                "invalidation set",
            ),
        )
    raise RepairError(f"no implementation for {action.atom.value}")


def _resample(plan: ScenePlan, known: tuple[str, ...], outcome) -> RepairOutcome:
    """L0: re-run the same solver over the same space with a different candidate offset."""
    if not known:
        return outcome(plan, changed=False,
                       notes=("no target in the plan, so there is nothing to resample",))
    updated = plan
    moved: list[str] = []
    for entity_id in known:
        request = updated.furniture_for(entity_id)
        if request is None:
            continue
        updated = updated.replaced_furniture(
            dataclass_replace(request, skip_candidates=request.skip_candidates + 1)
        )
        moved.append(entity_id)
    if not moved:
        return outcome(plan, changed=False,
                       notes=("only floor pieces have a sweep to resample; nothing changed",))
    return outcome(
        updated, changed=True,
        notes=(f"resampled the placement sweep for {', '.join(moved)}",),
    )


def _optimise(
    plan: ScenePlan, known: tuple[str, ...], failure_code: str, wall_thickness_m: float,
    action: RepairAction, outcome,
) -> RepairOutcome:
    """L1: change the constraint the placement is satisfying."""
    strategy = str((action.parameters or {}).get("strategy") or _OPTIMISE_BY_CODE.get(failure_code, ""))
    if strategy not in OPTIMISE_STRATEGIES:
        return outcome(
            plan, changed=False,
            notes=(
                f"no local optimisation is defined for {failure_code or 'this failure'}; the "
                f"ladder has nothing to try at this level",
            ),
        )
    if strategy == "honour-wall-face":
        if plan.wall_face_offset_m >= wall_thickness_m - 1e-9:
            return outcome(
                plan, changed=False,
                notes=(
                    f"pieces already stand {plan.wall_face_offset_m:.4f} m off the wall line, which "
                    f"is at or beyond the {wall_thickness_m:.4f} m wall face; this constraint is "
                    "already satisfied",
                ),
            )
        return outcome(
            dataclass_replace(plan, wall_face_offset_m=wall_thickness_m), changed=True,
            notes=(
                f"stand pieces {wall_thickness_m:.4f} m off the wall line so their backs meet the "
                f"wall face rather than entering the wall solid",
            ),
        )
    if strategy == "widen-gap":
        widened = round(plan.object_gap_m + 0.02, 4)
        return outcome(
            dataclass_replace(plan, object_gap_m=widened), changed=True,
            notes=(f"ask the placement solver for a {widened:.4f} m gap between pieces",),
        )
    # search-wider: the same search, entered further along for the pieces that failed.
    if not known:
        return outcome(plan, changed=False, notes=("nothing to re-search",))
    updated = plan
    for entity_id in known:
        request = updated.furniture_for(entity_id)
        if request is not None:
            updated = updated.replaced_furniture(
                dataclass_replace(request, skip_candidates=request.skip_candidates + 1)
            )
    return outcome(updated, changed=True,
                   notes=(f"re-searched past the first solution for {', '.join(known)}",))


def _swap_asset(plan: ScenePlan, known: tuple[str, ...], outcome) -> RepairOutcome:
    """L2: take the next retrieval candidate for the same entity and query."""
    if not known:
        return outcome(plan, changed=False, notes=("no target to swap",))
    updated = plan
    swapped: list[str] = []
    for entity_id in known:
        if entity_id == plan.target_id and plan.target_id:
            # The task target is not a plan row -- it comes from the rigid library -- so the loop
            # below cannot see it. It is still the entity `prompt_adherence` is about, and the
            # candidate it uses is a plan fact, so advancing it is the same edit made in the place
            # the build reads from.
            updated = dataclass_replace(
                updated, target_asset_index=updated.target_asset_index + 1
            )
            swapped.append(f"{entity_id}->candidate {updated.target_asset_index}")
            continue
        request = updated.furniture_for(entity_id)
        if request is not None:
            updated = updated.replaced_furniture(
                dataclass_replace(request, asset_index=request.asset_index + 1)
            )
            swapped.append(f"{entity_id}->candidate {request.asset_index + 1}")
            continue
        surface = updated.surface_for(entity_id)
        if surface is not None:
            updated = dataclass_replace(
                updated,
                surfaces=tuple(
                    dataclass_replace(item, asset_index=item.asset_index + 1)
                    if item.entity_id == entity_id else item
                    for item in updated.surfaces
                ),
            )
            swapped.append(f"{entity_id}->candidate {surface.asset_index + 1}")
    if not swapped:
        return outcome(plan, changed=False, notes=("no target in the plan to swap",))
    return outcome(updated, changed=True,
                   notes=(f"advanced the retrieval candidate for {', '.join(swapped)}",))


def _change_support(plan: ScenePlan, known: tuple[str, ...], outcome) -> RepairOutcome:
    """L3: put the piece on a different support surface, wall or zone.

    Only a surface object has a support to change, and only when the plan names an alternative that
    is genuinely materialised. The M1 build materialises exactly one support surface -- the desk's
    top, from the desk asset's own recorded support polygon -- so there is no second surface for a
    notebook to move onto, and this reports that rather than editing the plan's ``support_entity``
    and leaving the scene unchanged.

    That distinction matters more than it looks. An atom that changes a plan field the build does
    not read commits nothing (the scene version does not move), so the ladder would record a rung
    that "ran" and changed nothing -- and a run whose every L3 attempt looked like a repair would
    report a repair rate it never earned. Refusing with a reason is the honest answer, and it lets
    the ladder move up instead of stalling.
    """
    if not known:
        return outcome(plan, changed=False, notes=("no target to re-support",))
    surface_objects = [item for item in plan.surfaces if item.entity_id in known]
    if not surface_objects:
        return outcome(
            plan, changed=False,
            notes=(f"{', '.join(known)} stand on the floor; re-supporting is for surface objects",),
        )
    return outcome(
        plan, changed=False,
        notes=(
            "this build materialises one support surface (the desk's top), so there is no "
            "alternative support for "
            + ", ".join(item.entity_id for item in surface_objects)
            + " to move onto; this level has nothing to offer",
        ),
    )


def _adjust_appearance(plan: ScenePlan, action: RepairAction, outcome) -> RepairOutcome:
    """The appearance track: move lighting numbers, move no geometry.

    The parameters come from the critic's own suggestion where it gave one -- a critic that says the
    room is too dark has just told us which way to move the number -- and fall back to a fixed step
    otherwise. Both directions are bounded, so repeated complaints cannot drive the lighting to a
    limit that makes the render useless and then be reported as "repaired".
    """
    parameters = action.parameters or {}
    direction = str(parameters.get("direction", "brighter"))
    settings: AppearanceSettings = plan.appearance
    scale = 1.35 if direction == "brighter" else 1.0 / 1.35

    def bounded(value: float, *, low: float, high: float) -> float:
        return round(min(high, max(low, value)), 3)

    updated = dataclass_replace(
        settings,
        dome_intensity=bounded(settings.dome_intensity * scale, low=120.0, high=2400.0),
        sun_intensity=bounded(settings.sun_intensity * scale, low=400.0, high=7000.0),
        ceiling_fill_intensity=bounded(
            settings.ceiling_fill_intensity * scale, low=120.0, high=2400.0
        ),
    )
    if updated == settings:
        return outcome(
            plan, changed=False,
            notes=(f"the lighting is already at its {direction} limit; nothing changed",),
        )
    return outcome(
        dataclass_replace(plan, appearance=updated), changed=True,
        notes=(
            f"moved the lighting {direction}: dome {settings.dome_intensity}->"
            f"{updated.dome_intensity}, sun {settings.sun_intensity}->{updated.sun_intensity}, "
            f"fill {settings.ceiling_fill_intensity}->{updated.ceiling_fill_intensity}",
        ),
    )


def action_for_event(
    event,
    *,
    atom: Atom,
    source: str = "gate",
    target: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    rationale: str = "",
) -> RepairAction:
    """Build the action for one rung of an event's ladder.

    ``target`` narrows the event's targets to the one the caller is repairing, which matters when a
    gate names several objects: repairing the first and re-measuring is cheaper than repairing all
    of them and finding that only one was the problem.
    """
    targets = (target,) if target else tuple(event.targets)
    if atom is Atom.REPLAN and source != "agent":
        source = "agent"
    if not targets and ATOMS[atom].track is Track.GEOMETRIC and source != "agent":
        # A geometric atom with no target would rebuild the whole scene and still be recorded as a
        # local repair. An appearance atom is different: the lighting is a property of the room, not
        # of a piece, so an untargeted lighting change is exactly what is meant -- and a critique
        # about lighting cites view labels rather than object ids, so requiring a target would make
        # the appearance track unreachable for the complaints it exists to answer.
        raise EscalationError(
            f"{event.kind.value} names no target, so there is nothing for {atom.value} to repair; "
            "an untargeted geometric atom would change the whole scene and be reported as a local "
            "repair"
        )
    return RepairAction(
        atom=atom, targets=targets, parameters=parameters or {}, rationale=rationale or event.detail,
        source=source,
    )
