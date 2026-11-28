# SPDX-License-Identifier: Apache-2.0
"""What the task's reservations are, and whether generated periphery stayed out of them.

`contracts/background.py` says *who owns* each object and *what each upstream stage did*. It does
not say **where a generated object is not allowed to stand**. This module is that half, and it
exists because of one measured fact: with `populate_assets` switched on, Infinigen's own furniture
solver put a 2.01 m shelf **0.1265 m2 inside the task's protected region and 0.0315 m2 inside the
circulation corridor** (`outputs/_probe/spec009_populate_probe_20260922_01/`). The solver has no
notion of a task reservation -- ADR-009 counted that as decided, `accessibility_cost` is the
upstream expression to use for it, and this module is where the resulting claims are written down.

**The vocabulary, and why each part of it is not derivable.**

* **Region.** Three of them, and their *band* is what makes an intrusion measurable at all:
  `protected` reaches from the floor to the plan's own `core.support_top_z_m`; `circulation`
  reaches to 1.8 m; `door_sweep` is the leaf's swing and is reserved alongside circulation
  (ADR-009 decision 3). The protected band is the *plan's* number rather than a constant because
  the band is what stops a pendant hanging over the desk being reported as standing in it, and
  the two cases that decided that are measured, not assumed.
* **Factory -> category.** A generated object arrives named `BookStackFactory(2469050)`. Whether
  that is allowed depends on whether its *category* is one the retrieval vocabulary cannot cover
  (ADR-009 decision 4: "generation only fills gaps"), and there is **no derivation** from an
  upstream factory name to our category -- the same reason `furnishing_plan.py` keeps an explicit
  `CATEGORY_PROFILES` table with no `f"{category}.n.01"` fallback. A factory this table does not
  name is reported, never quietly allowed: `BACKGROUND_FACTORY_UNMAPPED(factory)`.
* **Credentials.** The patch that injects the constraint lives in *our* repository (ADR-009
  decision 1) and is discovered by the build only if the build records it. A build that cannot
  name the entry script, the patch digest and the upstream pin is not a build whose constraints
  can be audited, so it fails as `BACKGROUND_PATCH_UNRECORDED(...)` rather than passing quietly.

**The failures this module adds beyond the SPEC's list**, each with the reason it could not borrow
an existing name:

* `BACKGROUND_RESERVATION_OBJECT_LEAKED(name)` -- ADR-009's Consequences require a negative for
  "the reservation object is invisible geometry": it must not be exported, must not be in the
  instance list, must not be visible. `BACKGROUND_KEEPOUT_OBJECT_MISSING` is the *opposite*
  failure (the object was not built), and a leak filed under it would read as the fix.
* `BACKGROUND_SOLVER_UNGUARDED(stage)` -- `abort_unsatisfied_{large,medium,small}` is what turns
  "the constraints were not solved" into a loud failure. The outcome (`BACKGROUND_SOLVER_ABORTED`)
  is a different claim from the switch that makes it possible, and the switch can be off in a run
  that happened to solve everything -- which is exactly the run whose *next* build is silent.
* `BACKGROUND_FACTORY_UNMAPPED(factory)` -- see above; an unjudgeable factory is not a pass.
* `BACKGROUND_RESERVATION_UNSATISFIED(group)` (TASK 4) -- upstream's unsatisfied-constraint report is
  keyed by the *constraint's own name* (`evaluator/evaluate.py:262-275`), so a key carrying our
  prefix is our term failing. TASK 3 established that the unsatisfied groups in every build are
  upstream's `lighting`/`desk`/`office_desk_chairs`; this rule is what keeps that a measurement
  rather than a reading, and it fires the day one of ours is the one that fails.
* `BACKGROUND_POOL_UNRECORDED(stage)` (TASK 4) -- requirement 1's fallback has to *have run*:
  generation that happened with no pool in the record is a stage whose objects stood in the room
  unjudged (`spec009_populate_probe_20260922_01` is exactly that build, and its 0.1265 m^2 shelf is
  the consequence).
* `BACKGROUND_POOL_LEDGER_INCOMPLETE(entity, m2)` (TASK 4) -- the pool's ledger and the ruler's
  measurement are two independent accounts of one room, and this is the rule for when they disagree
  about an object that stands in a region: the pool either never saw it or judged it legal. A reader
  must not have to choose which account to believe.

**Reused from `background.py`, unchanged:** `BACKGROUND_PROTECTED_OVERLAP(entity, m2)`,
`BACKGROUND_KEEPOUT_OVERLAP(entity, m2)` and `BACKGROUND_BUDGET_EXCEEDED(stage)` as failures, the
`RejectedCandidate` record for each refused candidate (through
:meth:`ConstraintManifest.rejected_candidates`, so a rejection is one object type in one place),
and every rule in the embedded `BackgroundManifest.audit()`, which this manifest includes as its
own first half.

**How the two overlap names are spelled here.** SPEC-009's lines are kept verbatim: `protected` is
the protected region, and `circulation + door_sweep` is the corridor, because SPEC-011 requirement 3
asks for those two to be reported separately and says the door sweep is *included* with the
corridor. The per-region breakdown is not lost -- it is the manifest's own `overlap_by_entity`,
which is keyed by region, so a reader can still ask which of the two a piece stood in.

Like every contract, this module is stdlib-only and free of geometry: the polygons arrive as
coordinates, the intrusion areas arrive as measurements taken by whichever tool had the mesh
(`rfw_w.scenes.workers.blender.measure_generated_furniture`), and the rules that turn them into failures live
here so the tool and its negatives read one list.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from rfw_w.scenes.contracts.background import (
    OVERLAP_TOLERANCE_M2,
    STAGES,
    BackgroundManifest,
    RejectedCandidate,
)
from rfw_w.scenes.contracts.hashing import is_digest

CONSTRAINTS_VERSION = "pvscene.background-constraints/v1"

#: The two files the constraint injection moves between our provider and the build: the request
#: (written before the build, read by the patch inside Blender) and the record (written by the
#: patch, read by whoever audits the build). Both live beside `command.txt` in the build's own
#: directory, so a run's constraints are evidence next to the run rather than in a side channel.
RESERVATION_BUNDLE_VERSION = "pvscene.reservation-bundle/v1"
INJECTION_VERSION = "pvscene.constraint-injection/v1"
BUNDLE_FILENAME = "reservations.json"
INJECTION_FILENAME = "constraint_injection.json"

#: The rejection pool's own record, inside the injection record. Its own version because it is
#: written by the same patch but read by a different rule set, and the two can be revised apart.
POOL_VERSION = "pvscene.rejection-pool/v1"

#: A third injection state, because "the patch could not install" is not "we chose not to inject":
#: the first is a broken build that must be fixed, the second is the default path working as
#: designed. Recording both as `not-applied` would erase that difference.
INJECTION_FAILED = "failed"

#: The policy the patch is asked to express, as data. Every number here is a knob TASK 3
#: calibrates, so they travel in the request and land in the build's record instead of living as
#: literals in the patch -- a weight that exists only in code makes the calibration a code edit
#: rather than a measured number.
POLICY_DEFAULTS: dict[str, Any] = {
    "abort_unsatisfied": True,
    "band_penetration_weight": 60.0,
    "distance_weight": 25.0,
    "distance_min_m": 0.15,
    "hard_penetration_m": 0.02,
}

#: The note an uncalibrated policy carries, so the first weights cannot travel as if they were
#: measured (SPEC-011 TASK 3 measures them).
UNCALIBRATED_POLICY_NOTE = "first weights, uncalibrated: SPEC-011 TASK 3 measures them"

#: The reservations. `protected` is the task cell's own region; `circulation` is the corridor the
#: robot needs; `door_sweep` is the floor the leaf sweeps. They are separate names because a report
#: that merged them could not say which one a piece was standing in -- the argument
#: `background.py` makes for keeping the two *failure* names apart applies to the regions too.
#: `REGION_PROTECTED` is spelled the same string as `background.RESERVATION_PROTECTED` on purpose
#: (one region, one name across both contracts), and a unit test pins that they agree.
REGION_PROTECTED = "protected"
REGION_CIRCULATION = "circulation"
REGION_DOOR_SWEEP = "door_sweep"
RESERVATIONS = (REGION_PROTECTED, REGION_CIRCULATION, REGION_DOOR_SWEEP)
#: The name the build gives each region's object. Written here rather than only in the patch
#: because the audit looks the object up by name in the build's own record, and two spellings of
#: one name would make a built object read as a missing one.
RESERVATION_OBJECT_PREFIX = "pvscene_reservation_"
#: The prefix every name we add to the build carries -- the reservation *objects*
#: (`pvscene_reservation_<region>_<i>`) and the terms the patch appends to the constraint graph
#: (`pvscene_reservation_<region>_penetration`). The shorter prefix is the one to test a constraint
#: name against, because upstream's unsatisfied-constraint report is keyed by the constraint's own
#: name (`evaluator/evaluate.py:262-275`) and our terms are not spelled the way our objects are.
RESERVATION_TERM_PREFIX = "pvscene_"

#: How high each reservation reaches. The protected region's top is *not* here: it is the plan's
#: own `core.support_top_z_m`, passed in as `Reservation.band_top_m`, because the office wall art
#: at 1.450 m over a 0.750 m desk and the bedroom pendant at 2.528 m over a 0.575 m nightstand are
#: hanging over the reservation rather than standing in it, and only the plan knows that height.
CIRCULATION_BAND_TOP_M = 1.8
#: ADR-009 decision 3 says the door sweep is "included" and numbers only the corridor, so the
#: sweep inherits the corridor's band. Recorded as its own constant so that the day the sweep
#: needs a different clearance, the number it changes is not shared with circulation.
DOOR_SWEEP_BAND_TOP_M = 1.8

#: The three stages that carry upstream's `abort_unsatisfied_{group}` switch
#: (`generate_indoors.py:202`, keyed by group: large/medium/small).
SOLVE_STAGES = ("solve_large", "solve_medium", "solve_small")

#: Whether the build carried the constraint patch. `unknown` is not `not-applied`: the first is a
#: build whose provenance was not recorded, the second is a build that says it ran unconstrained.
INJECTION_APPLIED = "applied"
INJECTION_NOT_APPLIED = "not-applied"
INJECTION_UNKNOWN = "unknown"
INJECTIONS = (INJECTION_APPLIED, INJECTION_NOT_APPLIED, INJECTION_UNKNOWN)

#: How many times one candidate may be attempted before the pool is unbounded. ADR-009 decision 8:
#: "bounded, and no retrying until a better picture comes out". Two attempts is "try once more"; a
#: third is churn, and three is the number SPEC-011's sixth negative is written against.
ATTEMPTS_PER_CANDIDATE_MAX = 2


class BackgroundConstraintError(ValueError):
    """A constraint record that cannot be read as what it claims to be."""


def abort_param(stage: str) -> str:
    """The gin parameter that makes an unsatisfied solve loud, for a solve stage.

    ``compose_indoors.abort_unsatisfied_large`` -- the override block is `compose_indoors`'s own
    (`generate_indoors.py:136` takes ``**overrides``, and ``:202`` reads
    ``overrides.get(f"abort_unsatisfied_{group}", False)``), which is also why the *default* is
    `False` and why a build with no override is an unguarded build rather than an unknown one.
    """
    group = stage.split("_")[-1]
    return f"compose_indoors.abort_unsatisfied_{group}"


# ---------------------------------------------------------------------------------------------
# Factory -> category: the table that decides what generation was allowed to produce
# ---------------------------------------------------------------------------------------------

#: Upstream factory name -> our category. Total where it matters and explicit everywhere: the
#: entries below are the factories that have actually appeared in a probe build, each with the
#: judgement that put it in its column. A factory that is *not* here is reported by name
#: (`BACKGROUND_FACTORY_UNMAPPED`) rather than defaulted to either column, because the whole rule
#: is "generation only fills gaps" and an unjudged factory is not evidence of a gap.
FACTORY_CATEGORIES: Mapping[str, str] = {
    # -- gaps: categories the retrieval vocabulary does not name --------------------------------
    # ADR-009 decision 4 lists these four kinds itself (book stacks, decorative shelves, mirror
    # frames, wall art), and these five factories are those kinds. They are *not* the spec's own
    # `book` / `bookcase` / `wall_art` entities: a stack of books as decor is not the notebook the
    # task rests on the nightstand, and calling it `book` would make the repair "turn generation
    # off", which is the one thing the split exists to avoid.
    "BookStackFactory": "book_stack",
    "BookColumnFactory": "book_column",
    # A wall-mounted compartment shelf. `bookcase.n.01` in the retrieval vocabulary is a
    # freestanding storage piece, so this is a decoration the vocabulary does not name.
    "CellShelfFactory": "shelf",
    "NatureShelfTrinketsFactory": "shelf_trinkets",
    "MirrorFactory": "mirror_frame",
    # No `monitor` category exists in the retrieval vocabulary, so a generated monitor is a gap by
    # the same test the others are -- whether a *bedroom* should have one is the fixture's call,
    # not this table's.
    "MonitorFactory": "monitor",
    # -- covered: categories retrieval does provide, so generating them is BACKGROUND_NOT_A_GAP --
    "SimpleDeskFactory": "desk",
    "SimpleBookcaseFactory": "bookcase",
    "SimpleChairFactory": "chair",
    "SimpleStoolFactory": "stool",
    "SimpleCabinetFactory": "cabinet",
    "SimpleWardrobeFactory": "wardrobe",
    "CeilingLightFactory": "ceiling_light",
    "DeskLampFactory": "table_lamp",
    "FloorLampFactory": "floor_lamp",
    "LargePlantContainerFactory": "plant",
    "SmallPlantContainerFactory": "plant",
    "RugFactory": "rug",
    "WallArtFactory": "wall_art",
    # -- the three that only a real build could name (SPEC-011 TASK 4) --------------------------
    # The office probe generated five objects from factories this table did not name, and the audit
    # reported them by name (`BACKGROUND_FACTORY_UNMAPPED`) rather than guessing -- which is the
    # rule working. Each judgement below rests on the upstream class, not on the object's looks:
    #   * `PlantContainerFactory` is the *base* of `Large/SmallPlantContainerFactory`
    #     (`assets/objects/tableware/plant_container.py:47,128`): a pot plus a plant, so `plant`.
    #   * `SingleCabinetFactory` (`assets/objects/shelves/single_cabinet.py:301`) samples a
    #     0.25-0.35 x 0.3-0.7 x 0.9-1.8 m shelved cabinet: a storage cabinet, so `cabinet`.
    #   * `KitchenCabinetFactory` (`assets/objects/shelves/kitchen_cabinet.py:358`) is the kitchen
    #     cabinet unit; `cabinet.n.01` covers it, so `cabinet`.
    # All three land in the *covered* column, which is the consequential half: retrieval provides
    # cabinets and plants, so generation producing them violates "generation only fills gaps"
    # (requirement 4) -- and the pool refuses them with that reason. Mapping them instead to new
    # gap categories would have made the table say "a kitchen cabinet is something retrieval cannot
    # provide", which is how a wrong entry stops being a mistake and becomes a policy.
    "PlantContainerFactory": "plant",
    "SingleCabinetFactory": "cabinet",
    "KitchenCabinetFactory": "cabinet",
}


def category_of(factory: str) -> str:
    """The category a generated factory belongs to, or ``""`` when it is not in the table."""
    return FACTORY_CATEGORIES.get(factory, "")


# ---------------------------------------------------------------------------------------------
# How a kept generated object is named in a delivery
# ---------------------------------------------------------------------------------------------
#
# The pool judges in *categories* -- that is what the gap vocabulary, the refusal reasons and the
# budget are keyed by -- while a delivery names *entities*. One naming rule, here beside the
# category table, so the ledger and the assembly cannot disagree about what a kept object is: the
# category is inside the id rather than in a parallel field that could drift from it.

#: The leading token of a generated entity id: `generated_<category>_<ordinal>`.
GENERATED_ENTITY_PREFIX = "generated"

#: The provider a generated asset is delivered under. It is the image of the ownership contract's
#: `generated` owner (`contracts/background.py`): the manifest row, the instance's `asset_ref` and
#: the audit's ownership table all say one word for one thing.
GENERATED_PROVIDER = "generated"


def generated_entity_id(category: str, ordinal: int) -> str:
    """The delivery id of the ``ordinal``-th kept object of ``category``.

    An ordinal rather than one entity per category: two monitors are two instances, and
    `placement.instances` is keyed by entity id -- collapsing them would silently drop one.

    **Underscores, not colons**, and that is not cosmetic: an instance id becomes part of a USD
    prim path (`/World/Instances/<id>`, `workers/isaac/render_worker.py:420`), where a colon makes
    the path a *property* path -- measured on the first assembled delivery, Isaac's load failed
    with `_IsValidPathForCreatingPrim ... Path must be an absolute path: <>` and the whole run
    ended without a settle. The character class is checked rather than assumed, because the same
    string ends up in prim names, texture prefixes and the annotator's labels.
    """
    if not category:
        raise ValueError("GENERATED_CATEGORY_MISSING: a generated entity is named by its category")
    if not all(character.isalnum() or character == "_" for character in str(category)):
        raise ValueError(f"GENERATED_CATEGORY_NOT_AN_IDENTIFIER({category}): a category becomes part "
                         f"of an instance id and a USD prim path")
    return f"{GENERATED_ENTITY_PREFIX}_{category}_{int(ordinal)}"


def generated_category(entity_id: str) -> str | None:
    """The category a generated entity id names, or ``None`` for an id this contract did not mint.

    The split is on the *last* underscore because a category may contain one (`shelf_trinkets`),
    and only the trailing ordinal is numeric -- so the two parts cannot be confused for each other.
    """
    text = str(entity_id)
    prefix = f"{GENERATED_ENTITY_PREFIX}_"
    if not text.startswith(prefix):
        return None
    category, _, ordinal = text[len(prefix):].rpartition("_")
    if not category or not ordinal.isdigit():
        return None
    return category


# ---------------------------------------------------------------------------------------------
# The budget, and the rejection pool that enforces the two rules above
# ---------------------------------------------------------------------------------------------

#: How many generated objects one stage may leave in the room. ADR-009 decision 5: a count, not a
#: time, because a time limit makes the same input pass or fail with the machine's load.
#:
#: **Where 24 comes from** (the measurement is the bound, the number inside it is a choice):
#: the reference policy's own rooms left 13 (bedroom full_02 seed 37), 19 (kitchen medium seed 23)
#: and 18 (office medium seed 20) objects, while the *same bedroom without the constraint* left 32
#: (`spec009_populate_probe_20260922_01`) -- and that runaway is the case the cap exists to catch.
#: 24 is a deliberate number strictly between 19 and 32, not a measurement; it is frozen here so the
#: matrix judges the same allowance the builds were held to (SPEC-011 TASK 4).
GENERATED_OBJECTS_MAX_PER_STAGE: Mapping[str, int] = {"populate_assets": 24}

#: The stage a generated object is realised in, as a name the *record* can be read against: the
#: budget above is keyed by it, and `background.py`'s `OwnedEntity` requires a generated entity to
#: name a stage (STAGES) rather than this literal.
REALISING_STAGE = "populate_assets"

#: The reason kinds the pool writes. They travel as the *leading token* of a rejection reason, so a
#: reader (and a test) can count refusals by kind without parsing prose -- and so the two sides of
#: requirement 4 have one spelling each: what the room may not gain, and what the sampler proposed
#: that this table does not recognise.
POOL_REASON_PROTECTED = "protected-overlap"
POOL_REASON_CORRIDOR = "corridor-overlap"
POOL_REASON_NOT_A_GAP = "not-a-gap"
POOL_REASON_FACTORY_UNMAPPED = "factory-unmapped"
POOL_REASONS = (POOL_REASON_PROTECTED, POOL_REASON_CORRIDOR, POOL_REASON_NOT_A_GAP,
                POOL_REASON_FACTORY_UNMAPPED)

#: The names a rejection reason can carry for a region, and the two sub-reasons of `not-a-gap`.
#: `covered-by-retrieval` is the one ADR-009 decision 4 is about; `not-declared-in-this-case` is the
#: same failure seen from the room's own declaration (a category this room never asked for), and
#: they are told apart because the repairs differ: one is a table judgement, the other a fixture
#: declaration.
NOT_A_GAP_COVERED = "covered-by-retrieval"
NOT_A_GAP_NOT_DECLARED = "not-declared-in-this-case"
#: The three kinds of named reason an empty gap can carry (`CategoryGap.unfilled_reason`), in
#: descending order of how much the evidence supports. They are separate kinds because a reader
#: deciding what to do next needs to tell them apart: a *refused* candidate is the pool working, a
#: *not proposed* category is the sampler's own limit (ADR-009's Consequences: upstream's category
#: choice is a hardcoded `uniform` and we cannot ask it for a category), and *not in the room* is
#: all a build without a pool can say -- the room does not contain one, and nothing is known about
#: what it proposed.
GAP_UNFILLED_KINDS = ("proposed-but-rejected", "not-proposed-by-this-run", "not-in-the-room",
                      # The fourth kind exists because the delivery has an accounting the build
                      # does not: the pool kept a candidate, and the *assembly* refused it (it would
                      # have shared space with a retrieved piece or reached outside the room). The
                      # ledger says "kept" and the delivery carries nothing, so without this kind
                      # the category would read as unfilled-and-unexplained -- measured on office
                      # seed 11, where every kept `book_stack` was refused at assembly.
                      "refused-at-assembly",
                      # And the fifth, which is the assembly not happening at all: the pool kept
                      # candidates of this category and the delivery carries none of them and
                      # refused none of them. `BACKGROUND_ASSEMBLY_MISSING` names the same absence
                      # from the delivery's side; this is the gap's side of it.
                      "kept-but-not-delivered")


def overlap_reason(region: str, m2: float) -> str:
    """The reason a piece standing in a region was refused, spelled with the number that showed it."""
    kind = POOL_REASON_PROTECTED if region == REGION_PROTECTED else POOL_REASON_CORRIDOR
    return f"{kind}({m2:.4f}m2)"


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Reservation:
    """One reservation as geometry: which region, which polygon, how high it reaches, from where.

    ``source`` is not decoration. Every number here is re-derived from a delivery's spec and seed
    by whichever tool took the measurement, and a reservation that cannot say where its polygon
    came from is a second copy of a number whose whole job is to be the same one the core was
    solved against.
    """

    region: str
    polygon_xy: tuple[tuple[float, float], ...]
    band_top_m: float
    source: str
    #: Which polygon of this region it is. A region is often several polygons (the bedroom probe's
    #: corridor is four), and without this the objects would share one name and overwrite each other
    #: in the solver's state -- measured on the first injected build, where five objects became one.
    index: int = 0

    def __post_init__(self) -> None:
        if self.region not in RESERVATIONS:
            raise BackgroundConstraintError(f"BACKGROUND_REGION_UNSUPPORTED({self.region})")
        if len(self.polygon_xy) < 3:
            raise BackgroundConstraintError(f"BACKGROUND_REGION_DEGENERATE({self.region})")
        if not self.band_top_m > 0:
            raise BackgroundConstraintError(f"BACKGROUND_REGION_BAND_INVALID({self.region})")
        if not self.source:
            raise BackgroundConstraintError(f"BACKGROUND_REGION_SOURCE_MISSING({self.region})")

    @property
    def object_name(self) -> str:
        """The name the patch gives this polygon's object, and the build's record carries it."""
        return f"{RESERVATION_OBJECT_PREFIX}{self.region}_{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {"region": self.region, "polygon_xy": [list(v) for v in self.polygon_xy],
                "band_top_m": self.band_top_m, "source": self.source, "index": self.index}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Reservation":
        return cls(region=str(payload["region"]),
                   polygon_xy=tuple((float(x), float(y)) for x, y in payload["polygon_xy"]),
                   band_top_m=float(payload["band_top_m"]),
                   source=str(payload.get("source", "")), index=int(payload.get("index", 0)))


@dataclass(frozen=True, slots=True)
class ReservationObject:
    """The invisible object the patch builds so upstream's solver can see the reservation.

    Upstream's constraint language has geometric relations between *objects* and no polygon
    predicate (ADR-009 Evidence), so the region has to exist in the scene as an object before it
    can be named by a constraint. That object is machinery, not furniture: it must stay out of the
    export, out of the instance list and out of sight, which is what the three flags record and
    `BACKGROUND_RESERVATION_OBJECT_LEAKED` checks.
    """

    name: str
    region: str
    visible: bool = False
    exported: bool = False
    listed_as_instance: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise BackgroundConstraintError("BACKGROUND_RESERVATION_OBJECT_UNNAMED")
        if self.region not in RESERVATIONS:
            raise BackgroundConstraintError(f"BACKGROUND_REGION_UNSUPPORTED({self.region})")

    @property
    def leaked(self) -> bool:
        return bool(self.visible or self.exported or self.listed_as_instance)

    def leak_fields(self) -> tuple[str, ...]:
        """Which of the three claims is broken -- the fix differs, so the report names it."""
        return tuple(name for name, value in (("visible", self.visible),
                                              ("exported", self.exported),
                                              ("listed_as_instance", self.listed_as_instance))
                     if value)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "region": self.region, "visible": self.visible,
                "exported": self.exported, "listed_as_instance": self.listed_as_instance}


@dataclass(frozen=True, slots=True)
class BuildProvenance:
    """Which code built this shell, and whether it carried the constraint patch.

    SPEC-011 requirement 2: the entry script and the patch digests plus the upstream pin. The
    patch lives in our repository (ADR-009 decision 1) and is reached through our own entry
    script, so these three are what make the *constraint* auditable rather than merely claimed.
    """

    upstream_pin: str
    entry_script: str = ""
    entry_script_sha256: str = ""
    patch_module: str = ""
    patch_sha256: str = ""
    injection: str = INJECTION_UNKNOWN
    injection_reason: str = ""

    def __post_init__(self) -> None:
        if self.injection not in INJECTIONS:
            raise BackgroundConstraintError(f"BACKGROUND_INJECTION_UNSUPPORTED({self.injection})")
        if self.injection != INJECTION_APPLIED and not self.injection_reason:
            # An absent thing carries the reason it is absent, or it reads as a pass.
            raise BackgroundConstraintError(
                f"BACKGROUND_INJECTION_REASON_MISSING({self.injection})")

    def unrecorded(self) -> tuple[str, ...]:
        """Which credentials are missing or malformed, named in the order the SPEC lists them."""
        missing = []
        if not self.upstream_pin:
            missing.append("upstream_pin")
        if not self.entry_script:
            missing.append("entry_script")
        if not is_digest(self.entry_script_sha256):
            missing.append("entry_script_sha256")
        if not self.patch_module:
            missing.append("patch_module")
        if not is_digest(self.patch_sha256):
            missing.append("patch_sha256")
        return tuple(missing)

    def to_dict(self) -> dict[str, Any]:
        return {"upstream_pin": self.upstream_pin, "entry_script": self.entry_script,
                "entry_script_sha256": self.entry_script_sha256,
                "patch_module": self.patch_module, "patch_sha256": self.patch_sha256,
                "injection": self.injection, "injection_reason": self.injection_reason,
                "unrecorded": list(self.unrecorded()),
                "abort_params": {stage: abort_param(stage) for stage in SOLVE_STAGES}}


@dataclass(frozen=True, slots=True)
class ConstraintStage:
    """What one solve stage's constraint outcome was, and whether it was allowed to be quiet.

    ``unsatisfied is None`` means *not measured*, which is a different claim from zero: the probe
    build's log has no constraint-outcome line, and recording "0 unsatisfied" there would be
    reporting an unrun check as a pass.
    """

    stage: str
    abort_unsatisfied: bool
    unsatisfied: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.stage not in SOLVE_STAGES:
            raise BackgroundConstraintError(f"BACKGROUND_SOLVE_STAGE_UNSUPPORTED({self.stage})")
        if self.unsatisfied is None and not self.reason:
            raise BackgroundConstraintError(
                f"BACKGROUND_SOLVER_OUTCOME_REASON_MISSING({self.stage})")
        if self.unsatisfied is not None and self.unsatisfied < 0:
            raise BackgroundConstraintError(f"BACKGROUND_SOLVER_OUTCOME_INVALID({self.stage})")

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "abort_unsatisfied": self.abort_unsatisfied,
                "abort_param": abort_param(self.stage), "unsatisfied": self.unsatisfied,
                "reason": self.reason}


@dataclass(frozen=True, slots=True)
class CategoryGap:
    """A category retrieval cannot cover for this room, and its accounting.

    A gap is accounted for exactly one way: it was filled by a named generated object, or it has a
    named reason for staying empty. Neither is `BACKGROUND_GAP_UNFILLED(category, reason)`.
    """

    category: str
    why: str = ""
    filled_by: tuple[str, ...] = ()
    unfilled_reason: str = ""

    def __post_init__(self) -> None:
        if not self.category:
            raise BackgroundConstraintError("BACKGROUND_GAP_CATEGORY_MISSING")
        if self.filled_by and self.unfilled_reason:
            raise BackgroundConstraintError(f"BACKGROUND_GAP_OVERACCOUNTED({self.category})")

    @property
    def accounted_for(self) -> bool:
        return bool(self.filled_by or self.unfilled_reason)

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "why": self.why, "filled_by": list(self.filled_by),
                "unfilled_reason": self.unfilled_reason,
                "accounted_for": self.accounted_for}


@dataclass(frozen=True, slots=True)
class GeneratedObject:
    """One object a background stage realised, with the category that decides whether it may exist."""

    name: str
    factory: str
    stage: str
    category: str = ""
    base_z_m: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.factory:
            raise BackgroundConstraintError("BACKGROUND_GENERATED_OBJECT_UNNAMED")
        if self.stage not in STAGES:
            raise BackgroundConstraintError(f"BACKGROUND_STAGE_UNSUPPORTED({self.stage})")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "factory": self.factory, "stage": self.stage,
                "category": self.category, "base_z_m": self.base_z_m}


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    """One attempt at one candidate. The pool's boundedness is a property of these counts."""

    candidate_id: str
    stage: str
    attempt: int
    outcome: str = "rejected"
    reason: str = ""

    OUTCOMES = ("rejected", "placed")

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise BackgroundConstraintError("BACKGROUND_ATTEMPT_CANDIDATE_MISSING")
        if self.stage not in STAGES:
            raise BackgroundConstraintError(f"BACKGROUND_STAGE_UNSUPPORTED({self.stage})")
        if self.attempt < 1:
            raise BackgroundConstraintError(f"BACKGROUND_ATTEMPT_INDEX_INVALID({self.attempt})")
        if self.outcome not in self.OUTCOMES:
            raise BackgroundConstraintError(f"BACKGROUND_ATTEMPT_OUTCOME_INVALID({self.outcome})")
        if self.outcome == "rejected" and not self.reason:
            # ADR-009 decision 8: every rejection carries its own reason, or the pool is a count
            # with no ledger.
            raise BackgroundConstraintError(f"BACKGROUND_REJECTION_REASON_MISSING({self.candidate_id})")

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "stage": self.stage, "attempt": self.attempt,
                "outcome": self.outcome, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class PoolCandidate:
    """One object the build's own rejection pool looked at, and what it did with it.

    Every realised object is a row -- the refusals *and* the keepers. A ledger of refusals alone
    could not tell "the pool judged this object and let it stand" from "the pool never saw it", and
    that difference is exactly what the ruler's measurement is checked against
    (`BACKGROUND_POOL_LEDGER_INCOMPLETE`).
    """

    name: str
    factory: str
    category: str
    outcome: str
    attempt: int = 1
    parts: int = 1
    overlaps_m2: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    OUTCOMES = ("kept", "rejected")

    def __post_init__(self) -> None:
        if not self.name or not self.factory:
            raise BackgroundConstraintError("BACKGROUND_POOL_CANDIDATE_UNNAMED")
        if self.outcome not in self.OUTCOMES:
            raise BackgroundConstraintError(
                f"BACKGROUND_POOL_OUTCOME_INVALID({self.name}:{self.outcome})")
        if self.attempt < 1:
            raise BackgroundConstraintError(f"BACKGROUND_POOL_ATTEMPT_INVALID({self.name})")
        if self.outcome == "rejected" and not self.reasons:
            # A refusal without a reason is a deletion, and this SPEC's whole point is that a
            # deletion a reader cannot explain is indistinguishable from a piece that fell off.
            raise BackgroundConstraintError(f"BACKGROUND_POOL_REASON_MISSING({self.name})")
        if self.outcome == "kept" and self.reasons:
            raise BackgroundConstraintError(
                f"BACKGROUND_POOL_KEPT_WITH_REASON({self.name}:{','.join(self.reasons)})")
        if any(reason.split("(")[0] not in POOL_REASONS for reason in self.reasons):
            raise BackgroundConstraintError(
                f"BACKGROUND_POOL_REASON_UNKNOWN({self.name}:{','.join(self.reasons)})")

    @property
    def rejected(self) -> bool:
        return self.outcome == "rejected"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "factory": self.factory, "category": self.category,
                "outcome": self.outcome, "attempt": self.attempt, "parts": self.parts,
                "overlaps_m2": {name: float(value) for name, value in self.overlaps_m2.items()},
                "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoolCandidate":
        return cls(name=str(payload["name"]), factory=str(payload["factory"]),
                   category=str(payload.get("category", "")), outcome=str(payload["outcome"]),
                   attempt=int(payload.get("attempt", 1)), parts=int(payload.get("parts", 1)),
                   overlaps_m2={str(k): float(v)
                                for k, v in payload.get("overlaps_m2", {}).items()},
                   reasons=tuple(str(item) for item in payload.get("reasons", ())))


@dataclass(frozen=True, slots=True)
class RejectionPool:
    """What the build's own pool did, as the build recorded it (ADR-009 decision 8).

    The pool is *inside* the build rather than a filter over its output, and the reason is measured:
    the office room kept two cabinets 0.0400 m^2 inside the task's protected region under the best
    soft weights this pin offers (SPEC-011 TASK 3), and no weight setting was found that protects
    the region without emptying the room. Refusing the piece is the only form of "do not stand
    there" the solver cannot outvote -- and it is what makes requirement 3's zero true of the scene
    rather than of a filtered copy of it.

    `ran` is not `is not None`: a build that switched the realising stage off has a pool with
    nothing to judge, and recording that as "0 refusals" would read as a clean room.
    """

    stage: str
    ran: bool
    reason: str = ""
    rule: str = ""
    bounds: Mapping[str, int] = field(default_factory=dict)
    candidates: tuple[PoolCandidate, ...] = ()
    version: str = POOL_VERSION

    def __post_init__(self) -> None:
        if self.version != POOL_VERSION:
            raise BackgroundConstraintError(f"BACKGROUND_POOL_VERSION_UNSUPPORTED({self.version})")
        if self.stage not in STAGES:
            raise BackgroundConstraintError(f"BACKGROUND_STAGE_UNSUPPORTED({self.stage})")
        if not self.ran and not self.reason:
            # An absent pool carries the reason it is absent, or a room with generation on and no
            # judgement in it reads as a room where nothing needed judging.
            raise BackgroundConstraintError(f"BACKGROUND_POOL_REASON_MISSING({self.stage})")
        for stage, cap in self.bounds.items():
            if stage not in STAGES or cap < 0:
                raise BackgroundConstraintError(f"BACKGROUND_BUDGET_INVALID({stage}:{cap})")

    # -- facts about the pool, before any verdict ---------------------------------------------
    def kept(self) -> tuple[PoolCandidate, ...]:
        return tuple(item for item in self.candidates if not item.rejected)

    def rejected(self) -> tuple[PoolCandidate, ...]:
        return tuple(item for item in self.candidates if item.rejected)

    def rejected_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.rejected())

    def kept_categories(self) -> tuple[str, ...]:
        return tuple(sorted({item.category for item in self.kept() if item.category}))

    def rejected_by_category(self) -> dict[str, tuple[PoolCandidate, ...]]:
        """The refusals grouped by the category they would have added -- the gap's own ledger."""
        grouped: dict[str, list[PoolCandidate]] = {}
        for item in self.rejected():
            grouped.setdefault(item.category, []).append(item)
        return {category: tuple(items) for category, items in grouped.items()}

    def attempts_per_candidate(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.candidates:
            counts[item.name] = max(counts.get(item.name, 0), item.attempt)
        return counts

    def rejection_reasons(self) -> dict[str, tuple[str, ...]]:
        """Which kind of refusal happened, counted -- the reading ADR-009 decision 8 asks for."""
        counts: dict[str, list[str]] = {}
        for item in self.rejected():
            for reason in item.reasons:
                counts.setdefault(reason.split("(")[0], []).append(item.name)
        return {kind: tuple(names) for kind, names in counts.items()}

    def attempts(self) -> tuple[CandidateAttempt, ...]:
        """The pool in `background.py`'s attempt vocabulary: one row per candidate, one attempt.

        The conversion is not decoration. `CandidateAttempt` is what the manifest's boundedness and
        rejection-count rules already read, and a second pool-specific counting would let the two
        disagree about how many times a candidate was tried.
        """
        return tuple(CandidateAttempt(candidate_id=item.name, stage=self.stage,
                                      attempt=item.attempt, outcome=(
                                          "rejected" if item.rejected else "placed"),
                                      reason=";".join(item.reasons))
                     for item in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "stage": self.stage, "ran": self.ran,
                "reason": self.reason, "rule": self.rule, "bounds": dict(self.bounds),
                "candidates": [item.to_dict() for item in self.candidates],
                "kept": len(self.kept()), "rejected": len(self.rejected()),
                "rejected_categories": sorted(self.rejected_by_category()),
                "rejection_reasons": {kind: list(names)
                                      for kind, names in self.rejection_reasons().items()}}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RejectionPool":
        return cls(stage=str(payload["stage"]), ran=bool(payload.get("ran", False)),
                   reason=str(payload.get("reason", "")), rule=str(payload.get("rule", "")),
                   bounds={str(k): int(v) for k, v in payload.get("bounds", {}).items()},
                   candidates=tuple(PoolCandidate.from_dict(item)
                                    for item in payload.get("candidates", ())),
                   version=str(payload.get("version", POOL_VERSION)))


# ---------------------------------------------------------------------------------------------
# The manifest and its verdict
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ConstraintManifest:
    """What a run did with the task's reservations, and what it measured about them.

    The ownership/stage/budget half is the embedded `BackgroundManifest` -- one record, not two, so
    a generated object is claimed by exactly one owner in exactly one place. The constraint half is
    everything specific to *this* SPEC: the regions and their bands, the invisible reservation
    objects, the build's credentials, the solve stages' constraint outcomes, the gaps and what
    filled them, and the rejection pool's bounds.

    `retrieval_categories` is the retrieval vocabulary as it stood for this run, recorded by the
    tool that could import it (`contracts/` cannot) so that "generation only fills gaps" is judged
    against the vocabulary of the run rather than against today's table.
    """

    scene_id: str
    seed: int = 0
    version: str = CONSTRAINTS_VERSION
    mode: str = "retrieved"
    background: BackgroundManifest | None = None
    reservations: tuple[Reservation, ...] = ()
    reservation_objects: tuple[ReservationObject, ...] = ()
    build: BuildProvenance | None = None
    constraint_stages: tuple[ConstraintStage, ...] = ()
    gaps: tuple[CategoryGap, ...] = ()
    generated: tuple[GeneratedObject, ...] = ()
    attempts: tuple[CandidateAttempt, ...] = ()
    rejection_bounds: Mapping[str, int] = field(default_factory=dict)
    #: What the build's own rejection pool did (`None` = the build recorded none, which is a
    #: different fact from a pool that ran and refused nothing -- `RejectionPool.ran`).
    pool: RejectionPool | None = None
    retrieval_categories: tuple[str, ...] = ()
    overlap_by_entity: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    measurements: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    # -- facts about the record, before any verdict ------------------------------------------
    def regions(self) -> dict[str, Reservation]:
        return {reservation.region: reservation for reservation in self.reservations}

    def generated_by_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.generated:
            counts[item.stage] = counts.get(item.stage, 0) + 1
        return counts

    def generated_categories(self) -> tuple[str, ...]:
        return tuple(sorted({item.category for item in self.generated if item.category}))

    def declared_gap_categories(self) -> tuple[str, ...]:
        return tuple(sorted({gap.category for gap in self.gaps}))

    def gap_eligible_categories(self) -> tuple[str, ...]:
        """Every category this module's table calls a gap, whether or not it was generated.

        The comparison target for the fixture's declaration: a declared gap that is *covered* by
        retrieval is the matrix cheating itself (ADR-009 decision 11), and the report has to be
        able to show both sides.
        """
        return tuple(sorted(set(FACTORY_CATEGORIES.values()) - set(self.retrieval_categories)))

    def region_overlap(self, entity_id: str) -> dict[str, float]:
        measured = self.overlap_by_entity.get(entity_id, {})
        return {region: float(measured.get(region, 0.0)) for region in RESERVATIONS}

    def corridor_overlap(self, entity_id: str) -> float:
        """The corridor's number, door sweep included -- SPEC-011 requirement 3's second report."""
        measured = self.region_overlap(entity_id)
        return measured[REGION_CIRCULATION] + measured[REGION_DOOR_SWEEP]

    def attempts_per_candidate(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for item in self.attempts:
            key = (item.stage, item.candidate_id)
            counts[key] = max(counts.get(key, 0), item.attempt)
        return counts

    def rejections_per_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.attempts:
            if item.outcome == "rejected":
                counts[item.stage] = counts.get(item.stage, 0) + 1
        return counts

    def rejected_candidates(self) -> tuple[RejectedCandidate, ...]:
        """The pool as `background.py`'s own record type: one entry per refused attempt.

        Not a second field that could disagree with `attempts` -- the rejection pool has exactly
        one source, and this reads it in the vocabulary the delivery report already speaks.
        """
        return tuple(RejectedCandidate(entity_id=item.candidate_id, stage=item.stage,
                                       reason=item.reason)
                     for item in self.attempts if item.outcome == "rejected")

    # -- the verdict --------------------------------------------------------------------------
    def own_gaps(self) -> tuple[str, ...]:
        """The failures this SPEC's rules find, without the ownership half the SPEC reuses.

        Kept apart so a report can say which half a red run is red in -- the reused half is
        SPEC-009's judgement of the same build, and a reader who conflated them would read a
        stage-evidence gap as an intrusion.
        """
        gaps: list[str] = []
        if self.version != CONSTRAINTS_VERSION:
            gaps.append(f"BACKGROUND_VERSION_UNSUPPORTED({self.version})")
        if self.mode != "generated":
            # The default path generates nothing, so there is no constraint to have been applied
            # and nothing here to judge. SPEC-011 requirement 7 keeps it that way on purpose.
            return tuple(gaps)
        gaps.extend(self._build_gaps())
        gaps.extend(self._reservation_gaps())
        gaps.extend(self._solver_gaps())
        gaps.extend(self._gap_gaps())
        gaps.extend(self._pool_gaps())
        for entity_id in sorted(self.overlap_by_entity):
            protected = self.region_overlap(entity_id)[REGION_PROTECTED]
            corridor = self.corridor_overlap(entity_id)
            if protected > OVERLAP_TOLERANCE_M2:
                gaps.append(f"BACKGROUND_PROTECTED_OVERLAP({entity_id},{protected:.4f})")
            if corridor > OVERLAP_TOLERANCE_M2:
                gaps.append(f"BACKGROUND_KEEPOUT_OVERLAP({entity_id},{corridor:.4f})")
        return tuple(gaps)

    def _build_gaps(self) -> list[str]:
        """SPEC-011 requirement 2: a build that cannot name its patch is not an audited build."""
        if self.build is None:
            return ["BACKGROUND_PATCH_UNRECORDED(nothing-recorded)"]
        return [f"BACKGROUND_PATCH_UNRECORDED({name})" for name in self.build.unrecorded()]

    def _reservation_gaps(self) -> list[str]:
        """Requirements 1 and 9: the regions exist upstream, and their objects are machinery."""
        gaps: list[str] = []
        if self.build is None or self.build.injection != INJECTION_APPLIED:
            # Requirement 1's substance: without the injection the regions were never in the
            # scene, so no region object can have been built. Reported per region rather than as
            # one line, because the repair is per region.
            return [f"BACKGROUND_KEEPOUT_OBJECT_MISSING({region})" for region in RESERVATIONS]
        built = {item.region for item in self.reservation_objects}
        for region in RESERVATIONS:
            if region not in built:
                gaps.append(f"BACKGROUND_KEEPOUT_OBJECT_MISSING({region})")
        for item in self.reservation_objects:
            if item.leaked:
                gaps.append(f"BACKGROUND_RESERVATION_OBJECT_LEAKED({item.name}:"
                            f"{','.join(item.leak_fields())})")
        return gaps

    def _solver_gaps(self) -> list[str]:
        """Requirement 5: an unsatisfied solve is a failure, and the switch is what makes it loud."""
        gaps: list[str] = []
        recorded = {item.stage: item for item in self.constraint_stages}
        for stage in SOLVE_STAGES:
            record = recorded.get(stage)
            if record is None:
                gaps.append(f"BACKGROUND_SOLVER_UNGUARDED({stage}:unrecorded)")
                continue
            if not record.abort_unsatisfied:
                gaps.append(f"BACKGROUND_SOLVER_UNGUARDED({stage})")
            if record.unsatisfied:
                gaps.append(f"BACKGROUND_SOLVER_ABORTED({stage})")
        # Whose constraint failed is a different fact from whether one did, and upstream names the
        # culprit: `violations` is keyed by the *constraint's own name*
        # (`evaluator/evaluate.py:262-275`), so a key carrying our prefix is our term failing rather
        # than upstream's `lighting`/`desk`. That distinction is the whole of TASK 3's attribution in
        # one line, and it is why this rule reads the names instead of the count.
        ours = sorted({str(name) for name in self.measurements.get('unsatisfied_group_names', ())
                       if str(name).startswith(RESERVATION_TERM_PREFIX)})
        gaps.extend(f"BACKGROUND_RESERVATION_UNSATISFIED({name})" for name in ours)
        return gaps

    def _gap_gaps(self) -> list[str]:
        """Requirement 4: generation only fills gaps, and every gap is accounted for.

        Both directions of the same rule: a *declared* gap that retrieval covers is not a gap
        (the matrix declaring itself work to do), and a generated category outside the declared
        gaps is not one either.
        """
        gaps: list[str] = []
        covered = set(self.retrieval_categories)
        declared = set(self.declared_gap_categories())
        for gap in self.gaps:
            if not gap.accounted_for:
                # The two ways a gap is accounted for are "something filled it" and "nothing did,
                # and here is what happened instead" -- the second being the pool's ledger (a
                # candidate of this category was proposed and refused) or the measurement itself (no
                # candidate of this category was ever proposed). A gap with neither is the failure:
                # the reader cannot tell a sampler that never offered it from a step that lost it.
                gaps.append(f"BACKGROUND_GAP_UNFILLED({gap.category},no-fill-and-no-reason-recorded)")
        # One line per category, with how many generated objects it covered -- the probe generated
        # three ceiling lights, and three identical lines would push the two lines that matter out
        # of the reader's view. The objects themselves are in `generated`.
        offenders: dict[tuple[str, str], int] = {}
        for category in self.declared_gap_categories():
            if category in covered:
                offenders[(category, "covered-by-retrieval")] = 0
        for item in self.generated:
            if not item.category:
                key = (item.factory, "")
            elif item.category not in declared:
                key = (item.category, ("covered-by-retrieval" if item.category in covered
                                       else "not-declared-in-this-case"))
            else:
                continue
            offenders[key] = offenders.get(key, 0) + 1
        for (name, reason), count in sorted(offenders.items()):
            if not reason:
                gaps.append(f"BACKGROUND_FACTORY_UNMAPPED({name},{count})")
            else:
                gaps.append(f"BACKGROUND_NOT_A_GAP({name},{reason},{count})")
        for stage, used in sorted(self.generated_by_stage().items()):
            budget = self.background.budget if self.background is not None else {}
            if stage not in budget and used:
                # An allowance nobody declared is not an allowance: `background.py` reads a
                # missing cap as zero in the retrieved mode, and the generated mode cannot read
                # it as "unlimited" without making the budget unjudgeable.
                gaps.append(f"BACKGROUND_BUDGET_EXCEEDED({stage})")
        return gaps

    def _pool_gaps(self) -> list[str]:
        """Requirement 6: the pool is bounded, is applied, and agrees with the ruler.

        Three separate claims, and each has its own repair: the pool must not retry a candidate
        until something good happens (boundedness), it must have *run* whenever generation did (or
        the pieces stood in the room unjudged), and the room it left must contain nothing the ruler
        measures as an intrusion that the ledger cannot account for (parity).
        """
        gaps: list[str] = []
        for (stage, candidate_id), attempt in sorted(self.attempts_per_candidate().items()):
            if attempt > ATTEMPTS_PER_CANDIDATE_MAX:
                gaps.append(f"BACKGROUND_REJECTIONS_UNBOUNDED({stage}:"
                            f"{candidate_id},{attempt})")
        for stage, rejected in sorted(self.rejections_per_stage().items()):
            bound = self.rejection_bounds.get(stage)
            if bound is None:
                gaps.append(f"BACKGROUND_REJECTIONS_UNBOUNDED({stage}:no-bound,{rejected})")
            elif rejected > bound:
                gaps.append(f"BACKGROUND_REJECTIONS_UNBOUNDED({stage}:{rejected}>{bound})")
        # The realising stage's own record says whether generation happened at all: upstream's
        # stage runner is silent for a stage it does not run, and this SPEC's rule is that the
        # *build's* statement is the measurement, not the audit's assumption.
        ran = any(record.stage == REALISING_STAGE and record.state == "executed"
                  for record in (self.background.stages if self.background is not None else ()))
        if ran and (self.pool is None or not self.pool.ran):
            detail = "no-pool-recorded" if self.pool is None else f"not-run:{self.pool.reason}"
            gaps.append(f"BACKGROUND_POOL_UNRECORDED({REALISING_STAGE}:{detail})")
        if self.pool is not None and self.pool.ran:
            # The ledger and the ruler are two accounts of one room, and this is the rule for when
            # they disagree about a *named object*: the pool says it refused this piece and the
            # ruler measures it standing in a region (a deletion that did not happen), or the pool
            # kept it and the ruler measures it intruding (two applications of one rule, with two
            # answers). An intruder the ledger does not name at all is already carried by the two
            # overlap failures above -- naming it here too would report one object under two names.
            judged = {item.name for item in self.pool.candidates}
            for entity_id in sorted(self.overlap_by_entity):
                measured = max(self.region_overlap(entity_id)[REGION_PROTECTED],
                               self.corridor_overlap(entity_id))
                if measured <= OVERLAP_TOLERANCE_M2 or entity_id not in judged:
                    continue
                gaps.append(f"BACKGROUND_POOL_LEDGER_INCOMPLETE({entity_id},{measured:.4f})")
        return gaps

    def audit(self) -> tuple[str, ...]:
        """Every named failure this manifest shows: the reused ownership rules first, then ours."""
        gaps: list[str] = []
        if self.background is not None:
            gaps.extend(self.background.audit())
        gaps.extend(self.own_gaps())
        return tuple(gaps)

    def with_changes(self, **changes: Any) -> "ConstraintManifest":
        """A changed manifest as a new value, so a negative fixture cannot edit the original."""
        return replace(self, **changes)

    # -- serialisation ------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "scene_id": self.scene_id, "seed": self.seed,
            "mode": self.mode,
            "background": None if self.background is None else self.background.to_dict(),
            "reservations": [item.to_dict() for item in self.reservations],
            "reservation_objects": [item.to_dict() for item in self.reservation_objects],
            "build": None if self.build is None else self.build.to_dict(),
            "constraint_stages": [item.to_dict() for item in self.constraint_stages],
            "gaps": [item.to_dict() for item in self.gaps],
            "generated": [item.to_dict() for item in self.generated],
            "generated_by_stage": self.generated_by_stage(),
            "generated_categories": list(self.generated_categories()),
            "declared_gap_categories": list(self.declared_gap_categories()),
            "gap_eligible_categories": list(self.gap_eligible_categories()),
            "attempts": [item.to_dict() for item in self.attempts],
            "attempts_per_candidate": {"|".join(key): value
                                       for key, value in sorted(self.attempts_per_candidate().items())},
            "rejections_per_stage": self.rejections_per_stage(),
            "rejected_candidates": [item.to_dict() for item in self.rejected_candidates()],
            "rejection_bounds": dict(self.rejection_bounds),
            "pool": None if self.pool is None else self.pool.to_dict(),
            "retrieval_categories": list(self.retrieval_categories),
            "overlap_by_entity": {k: {name: float(value) for name, value in measured.items()}
                                  for k, measured in self.overlap_by_entity.items()},
            "measurements": dict(self.measurements),
            "provenance": dict(self.provenance),
            "own_named_gaps": list(self.own_gaps()),
            "named_gaps": list(self.audit()),
        }


# ---------------------------------------------------------------------------------------------
# The request the build is handed, and the policy inside it
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReservationPolicy:
    """How the patch was asked to express the reservations, and whether those numbers are measured.

    ``calibrated`` is the field that keeps SPEC-011 TASK 2 honest. The first weights are guesses, and
    a guess that travels without saying so becomes a number a later reader trusts -- so an
    uncalibrated policy has to carry the note saying what it is, the same way every unmeasured
    quantity in this repository carries its reason.
    """

    abort_unsatisfied: bool = True
    band_penetration_weight: float = 60.0
    distance_weight: float = 25.0
    distance_min_m: float = 0.15
    hard_penetration_m: float = 0.02
    calibrated: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("band_penetration_weight", "distance_weight", "distance_min_m",
                     "hard_penetration_m"):
            if not float(getattr(self, name)) > 0:
                raise BackgroundConstraintError(f"BACKGROUND_POLICY_WEIGHT_INVALID({name})")
        if not self.calibrated and not self.note:
            raise BackgroundConstraintError(
                "BACKGROUND_POLICY_NOTE_MISSING: uncalibrated weights have to say so, or a later "
                "reader takes a guess for a measurement")

    def to_dict(self) -> dict[str, Any]:
        return {"abort_unsatisfied": self.abort_unsatisfied,
                "band_penetration_weight": self.band_penetration_weight,
                "distance_weight": self.distance_weight, "distance_min_m": self.distance_min_m,
                "hard_penetration_m": self.hard_penetration_m,
                "calibrated": self.calibrated, "note": self.note}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReservationPolicy":
        defaults = dict(POLICY_DEFAULTS)
        defaults.update({k: v for k, v in payload.items() if k in defaults})
        return cls(**defaults, calibrated=bool(payload.get("calibrated", False)),
                   note=str(payload.get("note", "")))


@dataclass(frozen=True, slots=True)
class ReservationBundle:
    """The request our provider hands the build: the reservations, the policy, and the credentials.

    The digests travel *in* the request as well as in the build's own record, and the two are
    compared after the build -- a record that agrees with itself proves nothing, but a record that
    disagrees with the request that was sent is a build that ran something else.
    """

    scene_id: str
    core_hash: str
    upstream_pin: str
    entry_script: str
    entry_script_sha256: str
    patch_module: str
    patch_sha256: str
    reservations: tuple[Reservation, ...]
    policy: ReservationPolicy
    #: The categories this room's retrieval vocabulary cannot cover, which is the only set the
    #: build's pool may let a generated object belong to (requirement 4). It travels in the *request*
    #: rather than only being declared in the matrix, because a rule the build cannot see is a rule
    #: the build cannot enforce -- and because the two declarations can then be compared
    #: (`BACKGROUND_CONSTRAINT_INPUT_MISMATCH(...:gaps)`).
    gaps: tuple[str, ...] = ()
    #: The retrieval vocabulary as it stood when the request was written, so "covered by retrieval"
    #: is judged against the vocabulary of the run rather than against today's table.
    retrieval_categories: tuple[str, ...] = ()
    #: The per-stage allowance the pool refuses against and the audit judges the count by. It
    #: defaults to the calibrated constant rather than to empty: an allowance nobody declared is not
    #: an allowance (ADR-009 decision 5).
    budget: Mapping[str, int] = field(default_factory=lambda: dict(GENERATED_OBJECTS_MAX_PER_STAGE))
    version: str = RESERVATION_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.version != RESERVATION_BUNDLE_VERSION:
            raise BackgroundConstraintError(f"BACKGROUND_BUNDLE_VERSION_UNSUPPORTED({self.version})")
        if not self.reservations:
            raise BackgroundConstraintError("BACKGROUND_BUNDLE_EMPTY(no reservations to inject)")
        if self.gaps and not self.retrieval_categories:
            # The pool's refusal reason says *why* a category is not a gap, and the two halves are
            # "retrieval covers it" and "this room did not ask for it". Without the vocabulary the
            # first half cannot be told from the second, and the reason would be a guess.
            raise BackgroundConstraintError(
                "BACKGROUND_BUNDLE_VOCABULARY_MISSING(gaps without retrieval_categories)")
        for name, digest in (("entry_script_sha256", self.entry_script_sha256),
                             ("patch_sha256", self.patch_sha256)):
            if not is_digest(digest):
                raise BackgroundConstraintError(f"BACKGROUND_BUNDLE_DIGEST_INVALID({name})")
        for name, path in (("entry_script", self.entry_script), ("patch_module", self.patch_module)):
            if not path:
                raise BackgroundConstraintError(f"BACKGROUND_BUNDLE_PATH_MISSING({name})")
        for stage, cap in self.budget.items():
            if cap < 0:
                raise BackgroundConstraintError(f"BACKGROUND_BUDGET_INVALID({stage}:{cap})")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "scene_id": self.scene_id, "core_hash": self.core_hash,
                "upstream_pin": self.upstream_pin, "entry_script": self.entry_script,
                "entry_script_sha256": self.entry_script_sha256,
                "patch_module": self.patch_module, "patch_sha256": self.patch_sha256,
                "reservations": [item.to_dict() for item in self.reservations],
                "policy": self.policy.to_dict(), "gaps": list(self.gaps),
                "retrieval_categories": list(self.retrieval_categories),
                "budget": dict(self.budget)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReservationBundle":
        return cls(
            scene_id=str(payload["scene_id"]), core_hash=str(payload.get("core_hash", "")),
            upstream_pin=str(payload.get("upstream_pin", "")),
            entry_script=str(payload.get("entry_script", "")),
            entry_script_sha256=str(payload.get("entry_script_sha256", "")),
            patch_module=str(payload.get("patch_module", "")),
            patch_sha256=str(payload.get("patch_sha256", "")),
            reservations=tuple(Reservation.from_dict(item)
                               for item in payload.get("reservations", ())),
            policy=ReservationPolicy.from_dict(payload.get("policy", {})),
            gaps=tuple(str(item) for item in payload.get("gaps", ())),
            retrieval_categories=tuple(str(item) for item in payload.get("retrieval_categories", ())),
            budget={str(k): int(v) for k, v in payload.get("budget", {}).items()}
            or dict(GENERATED_OBJECTS_MAX_PER_STAGE),
            version=str(payload.get("version", "")))
