# SPDX-License-Identifier: Apache-2.0
"""Derive the furniture plan from the scene spec, instead of from a table in the script.

Until this module existed the furniture set was ``FURNISHING_ROWS``, a nine-tuple table at the top of
``rfw_w.scenes.commands.generate``, and ``SceneSpec.entities`` had **no reader at all**. The consequences
were not theoretical:

* Three prompts naming a wood office, a white laboratory and a reading nook produced byte-identical
  rooms, because nothing in the pipeline read the prompt either.
* ``EntitySpec.size_bounds_m`` was declared for every entity and enforced for none. The HSSD path
  ranked candidates by width alone -- ``rank_candidates_by_size`` scores ``max(extent_x, extent_z)``
  and never looks at height -- so a spec asking for a 0.12-0.35 m ceiling light was handed a 1.67 m
  pendant, hung it from the ceiling, and left its shade 1.25 m above the floor in the walking path.
  ``geometry.in_bounds`` cannot see that: the box is inside the room, it is simply too low.

**The two rules this module exists to keep.**

*Category is not derivable from the id.* ``f"{category}.n.01"`` is not an HSSD synset key for
``pen_holder``, ``side_table``, ``plant`` or ``ceiling_light`` -- four of the twelve office
categories. So the vocabulary is an explicit table with no convention fallback, and a category it
does not know is a **named gap**, never a guess.

*Role is not enough to place a piece.* ``context`` spans three placement kinds and two plan
sections: a chair is floor-standing and faces the support, while a notebook is a surface object that
stands *on* it. What separates them is the ``ON`` relation, not the role. So the classification below
reads both, and the relation wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rfw_w.scenes.contracts.background_constraints import FACTORY_CATEGORIES
from rfw_w.scenes.contracts.scene_spec import EntityRole, EntitySpec, SceneSpec, ZoneKind
from rfw_w.scenes.solvers.materials import materials_for
from rfw_w.scenes.solvers.furnish import (
    AppearanceSettings,
    FurnitureRequest,
    Orientation,
    PlacementKind,
    ScenePlan,
    SupportRequest,
    SurfaceRequest,
)
from rfw_w.scenes.solvers.support_shape import SupportShape, shape_from


class PlanDerivationError(ValueError):
    """A hard-required entity cannot be derived from the spec.

    Raised rather than guessed at. An optional entity that cannot be derived is dropped and its gap
    recorded; a hard one is a spec the pipeline cannot honour, and building a scene that quietly
    omits the task support would produce a delivery whose manifest claims a task it cannot support.
    """


@dataclass(frozen=True, slots=True)
class CategoryProfile:
    """How a category is placed, and which library synsets stand for it.

    ``synsets`` is priority-ordered because one word is often several HSSD keys. The exact
    ``pen_holder`` key contains a pen pot; broader desk organizers include drawer boxes and
    remain fallback candidates. ``simready_categories`` is only consulted for the task target, which comes
    from the rigid library rather than from HSSD.

    ``target_extent_m`` overrides the size the retrieval aims for. It is **not** derivable from
    ``size_bounds_m``: the office plan's 1.35 m desk, 2.20 m rug and 0.10 m pen holder were chosen,
    not computed, and a desk declares no bounds at all. Keeping them here puts each one next to the
    reason it is that number rather than hiding it in a script.
    """

    synsets: tuple[str, ...]
    kind: PlacementKind
    orientation: Orientation
    zone: ZoneKind | None
    simready_categories: tuple[str, ...] = ()
    faces_support: bool = False
    target_extent_m: float | None = None
    # The form a *constructed* support of this category is built to. Only supports carry one: the
    # periphery is still retrieved, so a form here would name something nothing reads.
    support_shape: str | None = None

    def __post_init__(self) -> None:
        if not self.synsets:
            raise PlanDerivationError("a category profile needs at least one synset")
        if self.orientation is Orientation.FACES_TOWARD and not self.faces_support:
            raise PlanDerivationError(
                "an orientation of faces_toward needs faces_support, or the request it produces "
                "names no referent and FurnitureRequest rejects it"
            )


# The vocabulary. Every synset here is a key in the real HSSD index (checked by a test that reads
# the index when PVSCENE_HSSD_SYNSET_INDEX is configured), and the counts in the comments are why
# the order is what it is.
CATEGORY_PROFILES: Mapping[str, CategoryProfile] = {
    # -- supports -------------------------------------------------------------------------------
    "desk": CategoryProfile(("desk.n.01",), PlacementKind.AGAINST_WALL, Orientation.FRONT_TO_ROOM,
                            ZoneKind.WORK, ("Desk",), target_extent_m=1.35,
                            support_shape="slab_legs"),
    # An island is a freestanding work surface. Treating it as wall furniture made every kitchen
    # candidate fail its own work zone before the task target could be placed.
    "kitchen_island": CategoryProfile(("kitchen_island.n.01",), PlacementKind.ANYWHERE,
                                      Orientation.FRONT_TO_ROOM, None, ("Island",),
                                      target_extent_m=1.55, support_shape="plinth"),
    "nightstand": CategoryProfile(("nightstand.n.01",), PlacementKind.AGAINST_WALL,
                                  Orientation.FRONT_TO_ROOM, ZoneKind.WORK, ("Nightstand",),
                                  target_extent_m=0.60, support_shape="pedestal"),
    # -- task targets (retrieved from SimReady, never given a plan row) --------------------------
    "cup": CategoryProfile(("cup.n.01",), PlacementKind.ANYWHERE, Orientation.SYMMETRIC,
                           ZoneKind.WORK, ("Mug", "Coffee Cup")),
    "jar": CategoryProfile(("jar.n.01",), PlacementKind.ANYWHERE, Orientation.SYMMETRIC,
                           ZoneKind.WORK, ("Condiment Jar", "Jar")),
    "alarm_clock": CategoryProfile(("alarm_clock.n.01",), PlacementKind.ANYWHERE,
                                   Orientation.SYMMETRIC, ZoneKind.WORK,
                                   ("Alarm Clock", "Clock")),
    # -- seating and storage --------------------------------------------------------------------
    "chair": CategoryProfile(("chair.n.01", "armchair.n.01", "swivel_chair.n.01"),
                             PlacementKind.BESIDE, Orientation.FACES_TOWARD, ZoneKind.SEATING,
                             ("Chair",), faces_support=True, target_extent_m=0.55),
    "stool": CategoryProfile(("stool.n.01",), PlacementKind.BESIDE, Orientation.FACES_TOWARD,
                             ZoneKind.SEATING, ("Stool",), faces_support=True,
                             target_extent_m=0.42),
    "cabinet": CategoryProfile(("cabinet.n.01", "chest_of_drawers.n.01"),
                               PlacementKind.AGAINST_WALL, Orientation.FRONT_TO_ROOM,
                               ZoneKind.STORAGE, ("Cabinet",), target_extent_m=0.95),
    "bookcase": CategoryProfile(("bookcase.n.01",), PlacementKind.AGAINST_WALL,
                                Orientation.FRONT_TO_ROOM, ZoneKind.STORAGE, ("Bookshelf",),
                                target_extent_m=1.10),
    "wardrobe": CategoryProfile(("wardrobe.n.01",), PlacementKind.AGAINST_WALL,
                                Orientation.FRONT_TO_ROOM, ZoneKind.STORAGE, ("Wardrobe",),
                                target_extent_m=1.40),
    "side_table": CategoryProfile(("end_table.n.01", "coffee_table.n.01"),
                                  PlacementKind.AGAINST_WALL, Orientation.FRONT_TO_ROOM,
                                  ZoneKind.STORAGE, ("Table",), target_extent_m=0.55),
    # ZoneKind has no sleeping zone. Calling a bed "storage" confines it to a 0.65 m strip that no
    # bed can fit. Until the vocabulary grows a sleeping zone, use the whole legal wall sweep and
    # record no false semantic assignment.
    "bed": CategoryProfile(("bed.n.01",), PlacementKind.AGAINST_WALL, Orientation.FRONT_TO_ROOM,
                           None, ("Bed",), target_extent_m=1.60),
    # -- surface objects ------------------------------------------------------------------------
    # kind is ANYWHERE because a surface object's position comes from the support it stands on, not
    # from a wall; the kind here is only what the plan records if the entity is ever re-derived as a
    # floor piece.
    "notebook": CategoryProfile(("notebook.n.01", "book.n.02"), PlacementKind.ANYWHERE,
                                Orientation.SYMMETRIC, ZoneKind.WORK, target_extent_m=0.20),
    "pen_holder": CategoryProfile(("pen_holder", "desk_organizer.n.01", "paper_organizer.n.01"),
                                  PlacementKind.ANYWHERE, Orientation.SYMMETRIC, ZoneKind.WORK,
                                  target_extent_m=0.10),
    "bowl": CategoryProfile(("bowl.n.01",), PlacementKind.ANYWHERE, Orientation.SYMMETRIC,
                            ZoneKind.WORK,
                            # Widened 2026-09-26 after a bowl *target* failed retrieval with
                            # CATEGORY_NOT_ADMITTED: v1.3 splits bowls across six super-categories
                            # and the two named here are not the populous ones (Dipping Bowl 2237,
                            # Serving Bowl 1707, Mixing Bowl 1438, Sugar Bowl 882, Soup Bowl 720,
                            # Batter Bowl 675, and a bare "Bowl" that has none at all). Semantic
                            # recall was returning real bowls and the admitted set was refusing
                            # them, which reads as "there are no bowls" and is not true.
                            ("Dipping Bowl", "Serving Bowl", "Mixing Bowl", "Soup Bowl",
                             "Sugar Bowl", "Batter Bowl", "Bowl"),
                            target_extent_m=0.26),
    "book": CategoryProfile(("book.n.02", "notebook.n.01"), PlacementKind.ANYWHERE,
                            Orientation.SYMMETRIC, ZoneKind.WORK, target_extent_m=0.20),
    # -- SPEC-017: the scene matrix's own surface vocabulary --------------------------------------
    # Added because the vocabulary admitted exactly three task targets -- cup, jar, alarm_clock --
    # and ten scenes that differ only in which of three objects is picked up are not ten scenes.
    #
    # Nothing here is guessed. Every synset is a key this machine's HSSD index actually carries
    # (it has 481), and every SimReady super-category was counted in the v1.3 catalogue before it
    # was written down (2026-09-26): Sports Bottle 602, Dressing Bottle 423, Beverage Can 434,
    # Ring-Pull Beverage Can 388, Dinner Plate 2100, Salad Plate 930, Stackable Storage Box 1098,
    # Rigid Gift Box 905, Jewelry Box 3140, Toy Car 655, Toy Dinosaur 505, Toy Robot 549,
    # Table Clock 668, Fruit Basket 427, Spice Jar 1298. A name with nothing behind it retrieves
    # nothing and surfaces as an unplaced item, which reads as a defect in the solver rather than
    # as a typo in a table -- `Jar` and `Bowl` above are exactly that, 0 assets each, carried
    # along by names that do have some.
    "bottle": CategoryProfile(("bottle.n.01", "wine_bottle.n.01"), PlacementKind.ANYWHERE,
                              Orientation.SYMMETRIC, ZoneKind.WORK,
                              ("Sports Bottle", "Dressing Bottle", "Squeeze Sauce Bottle"),
                              target_extent_m=0.10),
    # Same widening, same reason: recall returns food and drink cans across six super-categories
    # (Beverage Can 434, Ring-Pull Beverage Can 388, Tall Food Can 291, Pull Tab Food Can 262,
    # Flat Food Can 241, Ring Pull Can 167) and a set of two refused most of them --
    # SEMANTIC_BELOW_THRESHOLD was what a can target actually came back with.
    "can": CategoryProfile(("soda_can.n.01",), PlacementKind.ANYWHERE, Orientation.SYMMETRIC,
                           ZoneKind.WORK,
                           ("Beverage Can", "Ring-Pull Beverage Can", "Tall Food Can",
                            "Pull Tab Food Can", "Flat Food Can", "Ring Pull Can"),
                           target_extent_m=0.07),
    "plate": CategoryProfile(("plate.n.04",), PlacementKind.ANYWHERE, Orientation.SYMMETRIC,
                             ZoneKind.WORK, ("Dinner Plate", "Salad Plate"), target_extent_m=0.26),
    "storage_box": CategoryProfile(("storage_box.n.01", "gift_box.n.01", "toy_box.n.01"),
                                   PlacementKind.ANYWHERE, Orientation.SYMMETRIC, ZoneKind.WORK,
                                   ("Stackable Storage Box", "Rigid Gift Box"),
                                   target_extent_m=0.25),
    "jewelry_box": CategoryProfile(("jewelry_box.n.01",), PlacementKind.ANYWHERE,
                                   Orientation.SYMMETRIC, ZoneKind.WORK, ("Jewelry Box",),
                                   target_extent_m=0.14),
    "toy_car": CategoryProfile(("plush_toy.n.01", "toy_box.n.01"), PlacementKind.ANYWHERE,
                               Orientation.SYMMETRIC, ZoneKind.WORK, ("Toy Car",),
                               target_extent_m=0.14),
    "plush_toy": CategoryProfile(("plush_toy.n.01",), PlacementKind.ANYWHERE,
                                 Orientation.SYMMETRIC, ZoneKind.WORK,
                                 # v1.3 has no "plush toy" at all; it has toys by subject
                                 # (Toy Car 655, Toy Camera 622, Toy Robot 549, Toy Farm Animal
                                 # Figure 541, Toy Dinosaur 505). Admitting only two of them made
                                 # a toy target unretrievable while the library holds thousands.
                                 ("Toy Dinosaur", "Toy Robot", "Toy Farm Animal Figure",
                                  "Toy Camera", "Toy Car"), target_extent_m=0.20),
    "table_clock": CategoryProfile(("clock.n.01", "mantel_clock.n.01", "alarm_clock.n.01"),
                                   PlacementKind.ANYWHERE, Orientation.SYMMETRIC, ZoneKind.WORK,
                                   ("Table Clock", "Digital Clock"), target_extent_m=0.18),
    "basket": CategoryProfile(("basket.n.01",), PlacementKind.ANYWHERE, Orientation.SYMMETRIC,
                              ZoneKind.WORK, ("Fruit Basket", "Bread Basket"),
                              target_extent_m=0.28),
    "spice_jar": CategoryProfile(("jar.n.01",), PlacementKind.ANYWHERE, Orientation.SYMMETRIC,
                                 ZoneKind.WORK, ("Spice Jar", "Condiment Jar"),
                                 target_extent_m=0.09),
    # -- decoration -----------------------------------------------------------------------------
    "floor_lamp": CategoryProfile(("floor_lamp.n.01",), PlacementKind.CORNER,
                                  Orientation.SYMMETRIC, ZoneKind.DISPLAY, ("Lamp",),
                                  target_extent_m=0.40),
    "table_lamp": CategoryProfile(("table_lamp.n.01",), PlacementKind.AGAINST_WALL,
                                  Orientation.SYMMETRIC, ZoneKind.DISPLAY, ("Lamp",),
                                  target_extent_m=0.30),
    "plant": CategoryProfile(("potted_plant.n.01",), PlacementKind.CORNER, Orientation.SYMMETRIC,
                             ZoneKind.DISPLAY, ("Plant Pot",), target_extent_m=0.45),
    "rug": CategoryProfile(("rug.n.01",), PlacementKind.FLOOR_COVERING, Orientation.SYMMETRIC,
                           ZoneKind.AMBIENT, target_extent_m=2.20),
    "wall_art": CategoryProfile(("wall_art.n.01", "picture_frame.n.01", "wall_mirror.n.01"),
                                PlacementKind.WALL_HUNG, Orientation.FRONT_TO_ROOM,
                                ZoneKind.DISPLAY, target_extent_m=0.70),
    "ceiling_light": CategoryProfile(("ceiling_lamp.n.01", "pendant_lamp.n.01"),
                                     PlacementKind.CEILING, Orientation.SYMMETRIC,
                                     ZoneKind.AMBIENT, target_extent_m=0.45),
}


def profile_for(category: str, *, profiles: Mapping[str, CategoryProfile] = CATEGORY_PROFILES):
    """The profile for a category, or None. No ``f"{category}.n.01"`` fallback, deliberately."""
    return profiles.get(category)


def background_gaps(declared: Sequence[str], *,
                    profiles: Mapping[str, CategoryProfile] = CATEGORY_PROFILES,
                    factory_categories: Mapping[str, str] = FACTORY_CATEGORIES) -> tuple[str, ...]:
    """The categories generation may fill for this room, checked at *request* time.

    "Generation only fills gaps" (ADR-009 decision 4, SPEC-011 requirement 4) has two halves, and
    this function is the one that runs before the build: a declared category the retrieval
    vocabulary **does** cover is refused here (`BACKGROUND_NOT_A_GAP`) rather than reported after a
    build has already produced it, and a category no factory maps to is refused too
    (`BACKGROUND_GAP_CATEGORY_UNKNOWN`) because a gap nothing can fill is a declaration about the
    table rather than about the room.

    The check is the same rule the audit applies (this repository's habit: the tool that can refuse
    earliest refuses), but its *effect* differs by design -- here it stops a request the pipeline
    could not honour, and there it reports a build whose room gained a covered category anyway. Both
    are needed: this one cannot see what the solver sampled, and that one cannot stop a bad request
    before it costs three minutes of Blender.
    """
    gaps: list[str] = []
    for category in declared:
        name = str(category)
        if not name:
            raise PlanDerivationError("BACKGROUND_GAP_CATEGORY_MISSING")
        if name in profiles:
            raise PlanDerivationError(
                f"BACKGROUND_NOT_A_GAP({name},covered-by-retrieval): the retrieval vocabulary has a "
                "profile for this category, so generation producing it would duplicate retrieval "
                "rather than fill a gap")
        if name not in set(factory_categories.values()):
            raise PlanDerivationError(
                f"BACKGROUND_GAP_CATEGORY_UNKNOWN({name},no-generated-factory-maps-to-it): the gap "
                "table has no factory in this category, so nothing generation makes could fill it")
        if name not in gaps:
            gaps.append(name)
    return tuple(sorted(gaps))


def _midpoint(axis) -> float:
    return (axis.minimum + axis.maximum) / 2.0


def support_shape_for(entity: EntitySpec, profile: CategoryProfile) -> SupportShape:
    """The form this entity's support is built to.

    The spec's own word wins; the category's profile supplies one when the spec is silent. A
    category that offers neither is refused by name rather than given a default table -- a spec
    asking for a work surface this release cannot build is a fact about the request, and building a
    different one is how a manifest ends up describing a scene nobody asked for.
    """
    named = entity.support_shape or profile.support_shape
    if not named:
        raise PlanDerivationError(
            f"NO_SUPPORT_SHAPE({entity.id},{entity.category}): the entity and its category profile "
            "both name no form, so there is nothing to build the work surface as; set "
            "entity.support_shape or the category's support_shape"
        )
    return shape_from(named)


def _target_extent(entity: EntitySpec, profile: CategoryProfile) -> float:
    """The size the retrieval aims for: the profile's stated choice, else the spec's own bounds.

    The midpoint of the declared range rather than its maximum, following
    ``retrieval.target_size_for``: aiming at the largest value in a range systematically selects the
    biggest asset in it, which for a cup means picking a bowl.
    """
    if profile.target_extent_m is not None:
        return profile.target_extent_m
    if entity.size_bounds_m is None:
        raise PlanDerivationError(
            f"NO_TARGET_EXTENT({entity.id}): {entity.category!r} has no target_extent_m in the "
            f"vocabulary and {entity.id} declares no size_bounds_m, so there is no size to aim at"
        )
    x, y, _z = entity.size_bounds_m
    return round(max(_midpoint(x), _midpoint(y)), 4)


def _on_relations(spec: SceneSpec) -> Sequence[Any]:
    return tuple(r for r in spec.initial_relations if r.predicate.value == "ON")


def resolve_support(spec: SceneSpec, gaps: list[str]) -> str:
    """The piece the task target stands on, derived from the hard ``ON`` relation.

    Not a literal ``"desk"``: that literal was in the script in a dozen places, and a spec with no
    desk in it would have produced a scene built around a piece that was not there.

    Two things are *not* checked here because ``TaskSpec.__post_init__`` and
    ``SceneSpec.__post_init__`` already guarantee them: ``task.targets`` is non-empty, and every
    relation references a declared entity. This module used to raise for both -- unreachable
    branches, and a check that cannot fire is the defect this repository keeps finding.
    """
    targets = tuple(spec.task.targets)
    if len(targets) > 1:
        gaps.append(
            f"MULTIPLE_TASK_TARGETS({len(targets)}): this build places one grasp target; "
            f"using {targets[0]!r} and ignoring the rest"
        )
    target = targets[0]
    over = [r for r in _on_relations(spec) if r.subject == target]
    hard_over = [r for r in over if r.hard]
    if hard_over:
        supports = {r.object for r in hard_over}
    elif over:
        supports = {r.object for r in over}
    else:
        supports = set()
    if len(supports) > 1:
        raise PlanDerivationError(
            f"SUPPORT_UNRESOLVED(ambiguous): {target!r} stands on {sorted(supports)}"
        )
    if supports:
        return next(iter(supports))
    # No ON relation at all: fall back to the unique entity the spec calls a support.
    declared = [e.id for e in spec.entities if e.role is EntityRole.SUPPORT]
    if len(declared) == 1:
        gaps.append(
            f"TARGET_NOT_ON_SUPPORT({target}): no ON relation names {target}, so the support is "
            f"the spec's own role:support entity {declared[0]!r}"
        )
        return declared[0]
    raise PlanDerivationError(
        f"SUPPORT_UNRESOLVED(none): {target!r} has no ON relation and the spec declares "
        f"{len(declared)} role:support entit{'y' if len(declared) == 1 else 'ies'}"
    )


def derive_support_and_target(spec: SceneSpec, gaps: list[str]) -> tuple[str, str]:
    """``(support_id, target_id)`` for this spec.

    Both are always declared entities: ``resolve_support`` returns either the object of a relation
    (which the contract requires to exist) or a ``role: support`` entity from the spec itself.
    """
    return resolve_support(spec, gaps), spec.task.targets[0]


def plan_from_spec(
    spec: SceneSpec,
    *,
    wall_face_offset_m: float = 0.0,
    object_gap_m: float = 0.12,
    appearance: AppearanceSettings | None = None,
    mount_height_m: float = 1.45,
    ceiling_clearance_m: float = 0.05,
    wall_hang_gap_m: float = 0.02,
    profiles: Mapping[str, CategoryProfile] = CATEGORY_PROFILES,
) -> ScenePlan:
    """The plan this spec describes, with every derivation gap named.

    Four rules, first match wins, and the order is the design:

    1. ``role is SUPPORT`` -> a floor request in the WORK zone.
    2. the object of the hard ``ON`` relation whose subject is the task target -> **not** a plan row.
       It is retrieved from SimReady with its own contract (dynamic, centre origin, ``y_up_to_z_up``)
       and giving it a surface row would claim an HSSD synset for an asset that is not from HSSD.
    3. the subject of any ``ON`` relation whose object is the support -> a surface request.
    4. everything else -> a floor request from its category profile.

    Rule 3 must beat rule 4: ``notebook`` and ``chair`` both carry ``role: context``, and only the
    relation distinguishes a piece that stands on the support from one that stands beside it.
    """
    gaps: list[str] = []
    support, target = derive_support_and_target(spec, gaps)

    # Subjects of an ON relation onto the support, excluding the target itself (rule 2).
    surface_ids = {
        r.subject for r in _on_relations(spec) if r.object == support and r.subject != target
    }

    furniture: list[FurnitureRequest] = []
    surfaces: list[SurfaceRequest] = []
    supports: list[SupportRequest] = []

    def drop(entity: EntitySpec, gap: str) -> None:
        """Record a gap, or refuse the spec when the entity is one the task depends on."""
        if entity.hard_required:
            raise PlanDerivationError(
                f"{gap}; {entity.id!r} is hard_required, so this spec cannot be honoured"
            )
        gaps.append(gap)

    for entity in spec.entities:
        if entity.id == target:
            continue  # rule 2
        if entity.physics_type.value != "rigid":
            drop(entity, f"PHYSICS_UNSUPPORTED({entity.id},{entity.physics_type.value})")
            continue
        profile = profile_for(entity.category, profiles=profiles)
        if profile is None:
            drop(entity, f"CATEGORY_UNKNOWN({entity.category})")
            continue
        if entity.quantity != 1:
            drop(entity, f"QUANTITY_UNSUPPORTED({entity.id},{entity.quantity})")
            continue

        if entity.role is EntityRole.SUPPORT and entity.id != support:
            # A second support the task does not use. Placed, but not as the task support.
            gaps.append(f"SUPPORT_UNUSED({entity.id}): the task support is {support!r}")

        if entity.id == support:
            # The task's own support is built, not fetched -- see `SupportRequest`. It still gets a
            # row, because `ScenePlan` requires the named support to be something the plan places,
            # and because the form it is built to is a plan decision a repair could change.
            supports.append(
                SupportRequest(
                    entity_id=entity.id,
                    shape=support_shape_for(entity, profile).value,
                    target_extent_m=_target_extent(entity, profile),
                    kind=profile.kind,
                    orientation=profile.orientation,
                )
            )
            continue

        if entity.id in surface_ids:
            surfaces.append(
                SurfaceRequest(
                    entity_id=entity.id, synset=profile.synsets[0],
                    target_width_m=_target_extent(entity, profile),
                    support_entity=support,
                )
            )
            continue

        furniture.append(
            FurnitureRequest(
                entity_id=entity.id, synset=profile.synsets[0], kind=profile.kind,
                orientation=profile.orientation,
                hard_required=entity.hard_required, optional=not entity.hard_required,
                faces_entity=support if profile.faces_support else None,
                target_extent_m=_target_extent(entity, profile),
                zone_kind=profile.zone,
            )
        )

    return ScenePlan(
        furniture=tuple(furniture), surfaces=tuple(surfaces), supports=tuple(supports),
        appearance=appearance or AppearanceSettings(),
        # From the spec's own palette and seed, so the room's surfaces are part of what the spec
        # asked for rather than four constants in the render worker.
        room_materials=materials_for(spec.style, seed=spec.seed),
        mount_height_m=mount_height_m, ceiling_clearance_m=ceiling_clearance_m,
        wall_hang_gap_m=wall_hang_gap_m, wall_face_offset_m=wall_face_offset_m,
        object_gap_m=object_gap_m, support_id=support, target_id=target,
        derivation_gaps=tuple(gaps),
    )
