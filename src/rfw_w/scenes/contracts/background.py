# SPDX-License-Identifier: Apache-2.0
"""Who owns each object when a room is generated around a frozen task core.

Infinigen's `compose_indoors` does not stop when the walls are up. `solve_large`, `solve_medium`
and `solve_small` place furniture placeholders, `populate_intermediate_pholders` and
`populate_assets` turn them into real generated assets, and `floating_objs` scatters loose things.
ADR-002 gave the furniture to retrieval, so RFW-W switches the *last* of those off
(`compose_indoors.populate_assets_enabled=False` and `populate_assets` is the one stage that
override names). SPEC-009 opens an optional mode in which some of that content is kept -- and the
whole point of the split is that the task's own pieces stay retrieved, frozen, and unreplaced.

That needs a vocabulary before it needs code:

* **Owner.** Every explicit entity has exactly one, `retrieved` or `generated`. An entity in both
  is the failure this contract exists to name: a piece that the solver and the generator each
  believe they own is a piece whose provenance no report can state.
* **Scope.** The *task dependency closure* is core -- the target, its supports, and both endpoints
  of every initial relation. The bedroom's `book` has role `decoration` and is still core, because
  the task's ON relation has no subject without it. Everything else is periphery, and periphery is
  the only place generation is allowed.
* **Stage.** Reported under its upstream name with the gin parameter that decides it, in the five
  states SPEC-009 requires. A stage that did not execute carries a reason, and the reason says
  *which evidence* established that: "our own override switched it off" and "it left no trace in
  the build log" are different claims, and only one of them is a measurement.

The audit below is **data-only on purpose** -- contracts import no geometry (`room_plan.py:32`) --
so the geometric intrusion measurement lives in the tool that has polygons. The rules that decide
*which* measurements are failures live here, so the tool and its negative fixtures read one list.

The two overlap names are deliberately not one. SPEC-009 §Failure lists
`BACKGROUND_PROTECTED_OVERLAP(entity, m2)` for the task cell's protected region; the circulation
corridor and the door sweep are separate reservations with the same "must not be entered" meaning
and a different owner, and a report that merged them would not say which one a piece was standing
in. `BACKGROUND_KEEPOUT_OVERLAP` is that second line.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from rfw_w.scenes.contracts.hashing import is_digest
from rfw_w.scenes.contracts.scene_spec import SceneScope, relation_closure_ids, scope_of

BACKGROUND_VERSION = "pvscene.background/v1"
#: Which content the run is allowed to take from generation. `retrieved` is today's behaviour and
#: the only one that has been measured; `generated` is the optional branch SPEC-009 opens.
MODES = ("retrieved", "generated")
OWNERS = ("retrieved", "generated")

#: The upstream stages that produce background content, in the order `compose_indoors` runs them
#: (`infinigen_examples/generate_indoors.py:189-335` at the pinned commit `892947be`). Named after
#: upstream rather than after our own vocabulary, because the stage report has to be checkable
#: against the build log the stage itself wrote.
STAGES = ("solve_large", "populate_intermediate_pholders", "solve_medium", "solve_small",
          "populate_assets", "floating_objs")
STAGE_STATES = ("requested", "executed", "unsupported", "failed", "not_run")

#: How a stage's state is established. An override we passed is evidence about *our* command; a
#: line in the build's own log is evidence about the build; neither is a guess about the other.
EVIDENCE_OVERRIDE = "override"
EVIDENCE_LOG = "build-log"
EVIDENCE_NONE = "unlogged"

#: A footprint overlapping a reservation by less than this is the sliver two coincident edges
#: share, not an intrusion: 1e-6 m2 is one square millimetre. Same order as the geometry
#: module's own invariant tolerance (`solvers/zones.py:INVARIANT_TOLERANCE_M2`).
OVERLAP_TOLERANCE_M2 = 1e-6

#: The reservations a placed object may not stand in. Both are floor regions (`solvers/zones.py`),
#: which is why the check above them is about what reaches down into them rather than about what
#: passes over them: a pendant whose base is 2.53 m up hangs over the workspace, it does not stand
#: in it, and calling that an intrusion would fail every room with a ceiling light in the middle.
RESERVATION_PROTECTED = "protected"
RESERVATION_KEEPOUT = "keepout"


class BackgroundError(ValueError):
    """A background plan or report that cannot be read as what it claims to be."""


def enabled_param(stage: str) -> str:
    """The gin parameter that decides whether upstream runs ``stage``.

    `RandomStageExecutor._should_run_stage` (`pipeline.py:42`) reads
    ``params[f"{name}_enabled"]``, and ``params`` is the `compose_indoors` override dict -- which
    is why this is the name the run's own command line carries.
    """
    return f"compose_indoors.{stage}_enabled"


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One background stage: what upstream did, and how that was established.

    ``evidence`` is not decoration. A stage nobody logged -- because upstream's stage runner is
    silent for a stage it does not run -- can be reported as `not_run` for two very different
    reasons, and only one of them is a measurement of the build.
    """

    stage: str
    state: str
    reason: str = ""
    evidence: str = EVIDENCE_NONE
    duration_s: float | None = None
    entities: tuple[str, ...] = ()
    generated: int = 0

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise BackgroundError(f"BACKGROUND_STAGE_UNSUPPORTED({self.stage})")
        if self.state not in STAGE_STATES:
            raise BackgroundError(f"BACKGROUND_STAGE_STATE_INVALID({self.stage}:{self.state})")
        if self.state != "executed" and not self.reason:
            # The house rule for every other report in this repository: an absent measurement
            # carries the reason it is absent, or it reads as a pass.
            raise BackgroundError(f"BACKGROUND_STAGE_REASON_MISSING({self.stage})")
        if self.evidence not in (EVIDENCE_OVERRIDE, EVIDENCE_LOG, EVIDENCE_NONE):
            raise BackgroundError(f"BACKGROUND_STAGE_EVIDENCE_INVALID({self.stage}:{self.evidence})")
        if self.duration_s is not None and self.duration_s < 0:
            raise BackgroundError(f"BACKGROUND_STAGE_DURATION_INVALID({self.stage})")

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "state": self.state, "reason": self.reason,
                "evidence": self.evidence, "enabled_param": enabled_param(self.stage),
                "duration_s": self.duration_s, "entities": list(self.entities),
                "generated": self.generated}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageRecord":
        return cls(stage=str(payload["stage"]), state=str(payload["state"]),
                   reason=str(payload.get("reason", "")),
                   evidence=str(payload.get("evidence", EVIDENCE_NONE)),
                   duration_s=None if payload.get("duration_s") is None
                   else float(payload["duration_s"]),
                   entities=tuple(str(v) for v in payload.get("entities", ())),
                   generated=int(payload.get("generated", 0)))


@dataclass(frozen=True, slots=True)
class OwnedEntity:
    """One explicit entity and the single source that owns it."""

    entity_id: str
    owner: str
    scope: str
    stage: str = ""
    provider: str = ""
    asset_ref: str = ""

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise BackgroundError("BACKGROUND_ENTITY_ID_MISSING: an owned entity needs an id")
        if self.owner not in OWNERS:
            raise BackgroundError(f"BACKGROUND_OWNER_UNSUPPORTED({self.entity_id}:{self.owner})")
        if self.scope not in (SceneScope.CORE.value, SceneScope.PERIPHERY.value):
            raise BackgroundError(f"BACKGROUND_SCOPE_UNSUPPORTED({self.entity_id}:{self.scope})")
        if self.owner == "generated" and self.stage not in STAGES:
            # A generated entity that cannot say which stage produced it cannot be audited against
            # that stage's budget, and a budget nobody can attribute is not a budget.
            raise BackgroundError(f"BACKGROUND_GENERATED_STAGE_MISSING({self.entity_id})")

    @property
    def is_core(self) -> bool:
        return self.scope == SceneScope.CORE.value

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "owner": self.owner, "scope": self.scope,
                "stage": self.stage, "provider": self.provider, "asset_ref": self.asset_ref}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OwnedEntity":
        return cls(entity_id=str(payload["entity_id"]), owner=str(payload["owner"]),
                   scope=str(payload["scope"]), stage=str(payload.get("stage", "")),
                   provider=str(payload.get("provider", "")),
                   asset_ref=str(payload.get("asset_ref", "")))


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    """A generated candidate that was refused, and why. Its own record, not a missing entity."""

    entity_id: str
    stage: str
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise BackgroundError(f"BACKGROUND_REJECTION_REASON_MISSING({self.entity_id})")
        if self.stage not in STAGES:
            raise BackgroundError(f"BACKGROUND_STAGE_UNSUPPORTED({self.stage})")

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "stage": self.stage, "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RejectedCandidate":
        return cls(entity_id=str(payload["entity_id"]), stage=str(payload["stage"]),
                   reason=str(payload["reason"]))


@dataclass(frozen=True, slots=True)
class BackgroundRequest:
    """What a run asked the background branch for, before anything was built.

    ``budget`` is per stage and counts *generated entities*, because
    `BACKGROUND_BUDGET_EXCEEDED(stage)` names a stage: a run that generates more than its allowance
    somewhere has to say where. An empty budget is not "no limit" -- it is the `retrieved` mode's
    statement that generation nowhere has an allowance.
    """

    scene_id: str
    seed: int = 0
    mode: str = "retrieved"
    stages: tuple[str, ...] = STAGES
    budget: Mapping[str, int] = field(default_factory=dict)
    core_hash: str = ""
    required_entity_ids: tuple[str, ...] = ()
    protected_polygon_xy: tuple[tuple[float, float], ...] = ()
    keepout_polygons_xy: tuple[tuple[tuple[float, float], ...], ...] = ()
    version: str = BACKGROUND_VERSION

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise BackgroundError(f"BACKGROUND_MODE_UNSUPPORTED({self.mode})")
        for stage in self.stages:
            if stage not in STAGES:
                raise BackgroundError(f"BACKGROUND_STAGE_UNSUPPORTED({stage})")
        for stage, cap in self.budget.items():
            if stage not in STAGES:
                raise BackgroundError(f"BACKGROUND_STAGE_UNSUPPORTED({stage})")
            if cap < 0:
                raise BackgroundError(f"BACKGROUND_BUDGET_INVALID({stage}:{cap})")
        if self.core_hash and not is_digest(self.core_hash):
            raise BackgroundError("BACKGROUND_CORE_HASH_INVALID: expected a sha256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "scene_id": self.scene_id, "seed": self.seed,
                "mode": self.mode, "stages": list(self.stages), "budget": dict(self.budget),
                "core_hash": self.core_hash,
                "required_entity_ids": list(self.required_entity_ids),
                "protected_polygon_xy": [list(v) for v in self.protected_polygon_xy],
                "keepout_polygons_xy": [[list(v) for v in ring]
                                        for ring in self.keepout_polygons_xy]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackgroundRequest":
        version = str(payload.get("version", BACKGROUND_VERSION))
        if version != BACKGROUND_VERSION:
            raise BackgroundError(f"BACKGROUND_VERSION_UNSUPPORTED({version})")
        return cls(
            scene_id=str(payload["scene_id"]), seed=int(payload.get("seed", 0)),
            mode=str(payload.get("mode", "retrieved")),
            stages=tuple(str(v) for v in payload.get("stages", STAGES)),
            budget={str(k): int(v) for k, v in payload.get("budget", {}).items()},
            core_hash=str(payload.get("core_hash", "")),
            required_entity_ids=tuple(str(v) for v in payload.get("required_entity_ids", ())),
            protected_polygon_xy=tuple(tuple(float(v) for v in ring)  # type: ignore[misc]
                                       for ring in payload.get("protected_polygon_xy", ())),
            keepout_polygons_xy=tuple(
                tuple(tuple(float(v) for v in ring) for ring in rings)
                for rings in payload.get("keepout_polygons_xy", ())),
            version=version)


@dataclass(frozen=True, slots=True)
class BackgroundManifest:
    """What a run did with the background branch, and what it measured about it.

    ``overlap_by_entity`` maps an entity to its overlap in square metres with each reservation --
    a measurement, not a verdict. Which of those are failures is decided by :meth:`audit`, so the
    tool that has polygons and the test that has a hand-built case both ask the same question.
    """

    scene_id: str
    seed: int = 0
    version: str = BACKGROUND_VERSION
    mode: str = "retrieved"
    entities: tuple[OwnedEntity, ...] = ()
    stages: tuple[StageRecord, ...] = ()
    rejected: tuple[RejectedCandidate, ...] = ()
    required_entity_ids: tuple[str, ...] = ()
    core_hash: str = ""
    core_hash_requested: str = ""
    budget: Mapping[str, int] = field(default_factory=dict)
    overlap_by_entity: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    measurements: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    #: The categories this room declared as gaps -- the only set generation was allowed to produce
    #: (ADR-009 decision 11). Empty is the `retrieved` statement that nothing was declared, which is
    #: why an empty tuple never means "unconstrained" here.
    gaps: tuple[str, ...] = ()
    #: For each declared gap that no generated entity fills, the *named* reason it is empty
    #: (`background_constraints.GAP_UNFILLED_KINDS`). A gap with neither a fill nor a reason is
    #: `BACKGROUND_GAP_UNFILLED(category,no-fill-and-no-reason-recorded)`, which is the difference
    #: between a room that accounted for its gaps and one that simply has fewer pieces.
    gap_reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise BackgroundError(f"BACKGROUND_MODE_UNSUPPORTED({self.mode})")
        for stage, cap in self.budget.items():
            if stage not in STAGES:
                raise BackgroundError(f"BACKGROUND_STAGE_UNSUPPORTED({stage})")
            if cap < 0:
                raise BackgroundError(f"BACKGROUND_BUDGET_INVALID({stage}:{cap})")

    # -- what the run did ---------------------------------------------------------------------
    def entities_owned_by(self, owner: str) -> tuple[OwnedEntity, ...]:
        return tuple(entity for entity in self.entities if entity.owner == owner)

    def owners_of(self, entity_id: str) -> tuple[str, ...]:
        return tuple(sorted({entity.owner for entity in self.entities
                             if entity.entity_id == entity_id}))

    def duplicate_owners(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Entities claimed by more than one owner, in a deterministic order."""
        found = ((entity_id, self.owners_of(entity_id))
                 for entity_id in sorted({entity.entity_id for entity in self.entities}))
        return tuple((entity_id, owners) for entity_id, owners in found if len(owners) > 1)

    def generated_by_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entity in self.entities_owned_by("generated"):
            counts[entity.stage] = counts.get(entity.stage, 0) + 1
        return counts

    def missing_required(self) -> tuple[str, ...]:
        placed = {entity.entity_id for entity in self.entities}
        return tuple(sorted(set(self.required_entity_ids) - placed))

    def stage_table(self) -> dict[str, str]:
        """Stage -> state, filled with `requested` for a stage nobody reported.

        Not filled with `not_run`: a stage that left no record is exactly what
        `BACKGROUND_STAGE_NOT_REPORTED` is for, and defaulting it to `not_run` here would hide it.
        """
        reported = {record.stage: record.state for record in self.stages}
        return {stage: reported.get(stage, "requested") for stage in STAGES}

    # -- the verdict --------------------------------------------------------------------------
    def audit(self) -> tuple[str, ...]:
        """Every named failure this manifest shows, in one deterministic tuple."""
        gaps: list[str] = []
        if self.version != BACKGROUND_VERSION:
            gaps.append(f"BACKGROUND_VERSION_UNSUPPORTED({self.version})")
        for entity_id, owners in self.duplicate_owners():
            gaps.append(f"BACKGROUND_DUPLICATE_OWNER({entity_id})")
        for entity in self.entities:
            if entity.owner == "generated" and entity.is_core:
                # SPEC's `BACKGROUND_CORE_MUTATED(hash)` names a hash. Handing a core entity to
                # the generator is a different failure with a different repair, so it gets its
                # own line rather than borrowing the hash one.
                gaps.append(f"BACKGROUND_CORE_OWNER_GENERATED({entity.entity_id})")
        for entity_id in self.missing_required():
            gaps.append(f"BACKGROUND_REQUIRED_ENTITY_MISSING({entity_id})")
        if self.core_hash and self.core_hash_requested and self.core_hash != self.core_hash_requested:
            gaps.append(f"BACKGROUND_CORE_MUTATED({self.core_hash[:16]})")
        for entity_id, overlaps in sorted(self.overlap_by_entity.items()):
            protected = float(overlaps.get(RESERVATION_PROTECTED, 0.0))
            keepout = float(overlaps.get(RESERVATION_KEEPOUT, 0.0))
            if protected > OVERLAP_TOLERANCE_M2:
                gaps.append(f"BACKGROUND_PROTECTED_OVERLAP({entity_id},{protected:.4f})")
            if keepout > OVERLAP_TOLERANCE_M2:
                gaps.append(f"BACKGROUND_KEEPOUT_OVERLAP({entity_id},{keepout:.4f})")
        for stage, used in sorted(self.generated_by_stage().items()):
            # The `retrieved` mode's allowance is zero everywhere by definition rather than by
            # omission: a manifest that carried an empty budget and generated something anyway
            # would otherwise pass, which is the one case this rule exists to catch.
            cap = self.budget.get(stage, 0 if self.mode == "retrieved" else None)
            if cap is not None and used > cap:
                gaps.append(f"BACKGROUND_BUDGET_EXCEEDED({stage})")
        reported = {record.stage: record for record in self.stages}
        for stage in STAGES:
            record = reported.get(stage)
            # A missing record and a record that establishes nothing are the same failure to a
            # reader: upstream's stage runner is silent for a stage it does not run, so "no line
            # in the build log" is not a measurement of what the build did.
            if record is None or record.evidence == EVIDENCE_NONE:
                gaps.append(f"BACKGROUND_STAGE_NOT_REPORTED({stage})")
        for record in self.stages:
            if record.state == "unsupported":
                gaps.append(f"BACKGROUND_STAGE_UNSUPPORTED({record.stage})")
        return tuple(gaps)

    def with_changes(self, **changes: Any) -> "BackgroundManifest":
        """A changed manifest as a new value, so a negative fixture cannot edit the original."""
        return replace(self, **changes)

    # -- serialisation ------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "scene_id": self.scene_id, "seed": self.seed,
            "mode": self.mode,
            "entities": [entity.to_dict() for entity in self.entities],
            "stages": [record.to_dict() for record in self.stages],
            "stage_table": self.stage_table(),
            "rejected": [item.to_dict() for item in self.rejected],
            "required_entity_ids": list(self.required_entity_ids),
            "missing_required": list(self.missing_required()),
            "core_hash": self.core_hash, "core_hash_requested": self.core_hash_requested,
            "budget": dict(self.budget), "generated_by_stage": self.generated_by_stage(),
            "overlap_by_entity": {k: dict(v) for k, v in self.overlap_by_entity.items()},
            "measurements": dict(self.measurements), "provenance": dict(self.provenance),
            "named_gaps": list(self.audit()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackgroundManifest":
        version = str(payload.get("version", BACKGROUND_VERSION))
        if version != BACKGROUND_VERSION:
            raise BackgroundError(f"BACKGROUND_VERSION_UNSUPPORTED({version})")
        return cls(
            scene_id=str(payload["scene_id"]), seed=int(payload.get("seed", 0)), version=version,
            mode=str(payload.get("mode", "retrieved")),
            entities=tuple(OwnedEntity.from_dict(item) for item in payload.get("entities", ())),
            stages=tuple(StageRecord.from_dict(item) for item in payload.get("stages", ())),
            rejected=tuple(RejectedCandidate.from_dict(item)
                           for item in payload.get("rejected", ())),
            required_entity_ids=tuple(str(v) for v in payload.get("required_entity_ids", ())),
            core_hash=str(payload.get("core_hash", "")),
            core_hash_requested=str(payload.get("core_hash_requested", "")),
            budget={str(k): int(v) for k, v in payload.get("budget", {}).items()},
            overlap_by_entity={str(k): {str(name): float(value) for name, value in overlaps.items()}
                               for k, overlaps in payload.get("overlap_by_entity", {}).items()},
            measurements=dict(payload.get("measurements", {})),
            provenance=dict(payload.get("provenance", {})))


# ---------------------------------------------------------------------------------------------
# The ownership line, read from the spec rather than from a table.
# ---------------------------------------------------------------------------------------------

def core_entity_ids(spec: Any) -> tuple[str, ...]:
    """The task's own entities: its scope-core plus the closure every relation endpoints.

    Two rules, unioned because neither is enough. ``scope_of`` is authoritative for role -- the
    target and its supports. ``relation_closure_ids`` is authoritative for structure: a book the
    task rests on the nightstand is core even though its role is `decoration`, and dropping it
    removes the subject of a hard ON relation.

    Takes the spec structurally (``entities``, ``task``, ``initial_relations``) so the rule can be
    read by a tool holding a plain dict as well as by one holding a `SceneSpec`.
    """
    by_role = {entity.id for entity in spec.entities if scope_of(entity) is SceneScope.CORE}
    return tuple(sorted(by_role | set(relation_closure_ids(spec)[0])))


def periphery_entity_ids(spec: Any) -> tuple[str, ...]:
    """Every requested entity that is not core. The only place generation may act."""
    return tuple(sorted({entity.id for entity in spec.entities} - set(core_entity_ids(spec))))


def retrieved_ownership(spec: Any) -> tuple[OwnedEntity, ...]:
    """Who retrieval owns for this spec: the core always, the periphery in today's mode.

    This is the `retrieved` baseline the SPEC-009 matrix is measured against -- every requested
    entity placed from a library, nothing generated.

    Under `generated` the periphery is only *allowed* to be generated: which entity a stage really
    produced, and which candidates were refused, are facts about a run. They are read from the
    manifest, and a spec that claimed them here would be a plan asserting its own outcome.
    """
    core = set(core_entity_ids(spec))
    return tuple(
        OwnedEntity(entity_id=entity.id, owner="retrieved",
                    scope=(SceneScope.CORE.value if entity.id in core
                           else SceneScope.PERIPHERY.value))
        for entity in sorted(spec.entities, key=lambda item: item.id))
