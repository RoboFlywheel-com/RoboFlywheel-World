# SPDX-License-Identifier: Apache-2.0
"""Placing things: furniture in zones, objects on support surfaces.

Deterministic and bounded. The architecture's split is that a model supplies semantics and the
solvers own the geometry, so nothing here asks a model where something should go: candidates are
enumerated on a lattice, rejected against real geometry, and ranked by an explicit score.

Two rules from the acceptance manual shape the search:

* occupied space is tracked as polygons, not as bounding boxes, because a bounding box for a
  notched desk reserves its notch and the circulation corridor through it;
* a candidate is rejected rather than nudged. Silently moving an object a few centimetres to make
  it fit would hide that the requested layout does not work, and the failure would surface later
  as an unexplained collision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from rfw_w.scenes.contracts.units import Quat, Transform, Vec3
from rfw_w.scenes.geometry.clearance import check_clearance
from rfw_w.scenes.geometry.polygon import (
    Polygon,
    PolygonError,
    contains_polygon,
    convex_hull,
    overlaps,
    validate_simple,
)
from rfw_w.scenes.geometry.robot import RobotProfile, standoffs_for
from rfw_w.scenes.solvers.support import Surface

SCHEMA_VERSION = "pvscene.layout/v1"

# Rejected candidates recorded per object before a placement is abandoned, so the search is
# bounded and the report can say how close it came.
DEFAULT_MAX_CANDIDATES = 512

# Reservations that hold *floor* and nothing above it. A door sweep and a circulation corridor
# reserve where a robot stands; a cup resting on a desk whose plan view crosses the corridor is
# not obstructed by it, but planar occupancy cannot see the height difference, so the exclusion
# is by owner name. Entries added with this prefix are floor-only.
FLOOR_KEEPOUT_PREFIX = "keepout:"


class LayoutError(ValueError):
    """The layout request is not well formed."""


@dataclass(frozen=True, slots=True)
class Footprint:
    """An object's horizontal extent, centred on its own origin."""

    half_extents: tuple[float, float]

    def __post_init__(self) -> None:
        for index, value in enumerate(self.half_extents):
            if value <= 0 or not math.isfinite(value):
                raise LayoutError(f"footprint half-extent {index} must be finite and positive")

    @property
    def width_m(self) -> float:
        return self.half_extents[0] * 2

    @property
    def depth_m(self) -> float:
        return self.half_extents[1] * 2

    @property
    def area_m2(self) -> float:
        return self.width_m * self.depth_m

    def at(self, x: float, y: float) -> Polygon:
        """The footprint as a polygon centred at ``(x, y)``."""
        hx, hy = self.half_extents
        return Polygon.from_sequence(
            [(x - hx, y - hy), (x + hx, y - hy), (x + hx, y + hy), (x - hx, y + hy)]
        )

    @classmethod
    def from_extents(cls, extents: Vec3) -> Footprint:
        return cls((extents.x / 2.0, extents.y / 2.0))

    def rotated(self, yaw_rad: float) -> Footprint:
        """The axis-aligned footprint that encloses this one after a yaw rotation.

        Placement reasons in world-axis-aligned rectangles, but a piece can face any direction. A
        desk yawed 90 degrees and placed with its unrotated footprint sticks half its length through
        the wall behind it: the occupancy check, the wall stand-off and the containment test all
        used the wrong shape, and the result rendered as a desk hanging outside the room with the
        cup positioned against the old, unrotated support outline.

        Exact for multiples of 90 degrees; conservative otherwise, which is the safe direction for a
        clearance test.
        """
        cos = abs(math.cos(yaw_rad))
        sin = abs(math.sin(yaw_rad))
        half_x, half_y = self.half_extents
        return Footprint((half_x * cos + half_y * sin, half_x * sin + half_y * cos))

    def half_extent_along(self, direction: tuple[float, float]) -> float:
        """This footprint's half-extent measured along a unit direction.

        The footprint is axis-aligned, so its extent along an arbitrary axis is the projection of
        the two half-extents onto it. Reading ``half_extents[1]`` as "the depth" is only correct
        when that axis happens to be the y axis: against a vertical wall it measured the piece's
        *width*, so the piece stood off from the wall by its own width and a desk ended up a third
        of a metre into the room with nothing reporting it.
        """
        return abs(direction[0]) * self.half_extents[0] + abs(direction[1]) * self.half_extents[1]

    def placed_extent(
        self, height_m: float, *, rotation: Quat, position: Vec3,
        origin_convention: str = "centre",
    ) -> "PlacedExtent":
        """Where this piece actually is in the world: its plan outline and its height range.

        :meth:`rotated` answers a narrower question -- the enclosing axis-aligned rectangle for a
        *yaw* -- and returns a footprint rather than a place. This one takes the whole box, turns
        all eight corners by a full rotation, and reports what the piece occupies: the hull of the
        projections onto the floor, and the range of heights it reaches.

        The distinction matters as soon as a piece leaves the plane. A wall-hung panel is 0.70 m
        wide and 1.00 m tall but 0.005 m deep, so its plan outline is a thin strip while it reaches
        a metre up; and a plan-only reading calls a cup resting on a desk a collision because the
        two overlap in x and y and nowhere else.
        """
        if origin_convention not in ("centre", "base"):
            raise LayoutError(f"unknown origin convention {origin_convention!r}")
        half_x, half_y = self.half_extents
        # The mesh origin is not always the middle: a base-origin mesh's own z runs 0..height, so
        # its two z faces are at 0 and +height rather than at -height/2 and +height/2.
        at_base = origin_convention == "base"
        corners = [
            rotation.rotate(Vec3(
                sign_x * half_x,
                sign_y * half_y,
                (sign_z + 1.0) * height_m / 2.0 if at_base else sign_z * height_m / 2.0,
            ))
            for sign_x in (-1.0, 1.0)
            for sign_y in (-1.0, 1.0)
            for sign_z in (-1.0, 1.0)
        ]
        heights = [position.z + corner.z for corner in corners]
        return PlacedExtent(
            outline=convex_hull([(position.x + corner.x, position.y + corner.y) for corner in corners]),
            z_low_m=min(heights),
            z_high_m=max(heights),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"half_extents_m": list(self.half_extents), "area_m2": round(self.area_m2, 6)}


@dataclass(frozen=True, slots=True)
class PlacedExtent:
    """A placed piece's world extent: what it covers on the floor and how far up it reaches.

    Both readings are needed to judge a room. Containment is a plan question until a piece stands
    through a ceiling, and interpenetration is a plan question until a pendant hangs over a rug.
    """

    outline: Polygon
    z_low_m: float
    z_high_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "outline_xy": self.outline.to_list(),
            "z_low_m": round(self.z_low_m, 6),
            "z_high_m": round(self.z_high_m, 6),
        }


@dataclass(slots=True)
class OccupancyMap:
    """What is already occupying space, as polygons.

    Deliberately not a grid: a 2 cm lattice over a room is millions of cells and still cannot
    represent a notch exactly, while a few dozen polygons can. The trade is that each query is
    polygon work rather than an array lookup, which is the right way round at this scale.
    """

    entries: list[tuple[str, Polygon]] = field(default_factory=list)

    def add(self, owner_id: str, polygon: Polygon) -> None:
        if not owner_id:
            raise LayoutError("occupancy entry must name its owner")
        self.entries.append((owner_id, polygon))

    def blockers_for(self, *excluded: str, include_floor_only: bool = True) -> tuple[Polygon, ...]:
        """Every occupied polygon except those of the named owners.

        Occupancy here is planar: it records where things stand, not how tall they are. That is
        enough for placing objects side by side, and it is *wrong* for placing one thing on top of
        another -- the desk's floor footprint overlaps the cup resting on its top, so a naive query
        rejects every position. The support's owner is therefore excluded explicitly, which is the
        statement "the thing supporting you does not obstruct you".

        The limitation is real and worth stating: two objects stacked at different heights in the
        same plan position would still read as overlapping. Nothing in this release stacks objects
        off a support surface, and a height-aware occupancy is the fix when something does.

        ``include_floor_only=False`` drops the floor reservations named with
        :data:`FLOOR_KEEPOUT_PREFIX`. Callers placing something *on a surface* want that: the
        object is above the floor the reservation refers to.
        """
        skip = set(excluded)
        return tuple(
            polygon
            for entry_id, polygon in self.entries
            if entry_id not in skip
            and (include_floor_only or not entry_id.startswith(FLOOR_KEEPOUT_PREFIX))
        )

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class Placement:
    """One object's solved pose, with what it rests on."""

    instance_id: str
    entity_id: str
    footprint: Footprint
    position_m: Vec3
    yaw_rad: float
    surface_id: str | None

    @property
    def transform(self) -> Transform:
        return Transform(
            translation=self.position_m,
            rotation=Quat.from_yaw(self.yaw_rad),
            parent_frame="world",
            child_frame=self.instance_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "entity_id": self.entity_id,
            "position_m": [round(value, 6) for value in self.position_m.to_list()],
            "yaw_rad": round(self.yaw_rad, 6),
            "surface_id": self.surface_id,
            "footprint": self.footprint.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PlacementRejection:
    """Why a candidate position was refused, and by how much it missed."""

    x: float
    y: float
    reason: str
    measured_margin_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "x": round(self.x, 4), "y": round(self.y, 4), "reason": self.reason,
        }
        if self.measured_margin_m is not None:
            payload["measured_margin_m"] = round(self.measured_margin_m, 6)
        return payload


@dataclass(frozen=True, slots=True)
class LayoutOutcome:
    """The result for one object: a placement, or the reason there is none."""

    instance_id: str
    entity_id: str
    placement: Placement | None
    rejections: tuple[PlacementRejection, ...] = ()
    failure_code: str | None = None
    detail: str = ""

    @property
    def placed(self) -> bool:
        return self.placement is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "entity_id": self.entity_id,
            "placed": self.placed,
            "placement": None if self.placement is None else self.placement.to_dict(),
            "failure_code": self.failure_code,
            "detail": self.detail,
            "rejected_count": len(self.rejections),
            "rejections": [rejection.to_dict() for rejection in self.rejections[:8]],
        }


def _lattice(bounds: tuple[float, float, float, float], step_m: float) -> list[tuple[float, float]]:
    """Candidate centres on a lattice covering the bounds, outermost first.

    Outermost first because objects against a wall or a surface edge are the common intent, and
    the first accepted candidate is the one that wins.
    """
    min_x, min_y, max_x, max_y = bounds
    if step_m <= 0:
        raise LayoutError(f"lattice step must be positive, got {step_m}")
    xs: list[float] = []
    value = min_x
    while value <= max_x + 1e-12:
        xs.append(round(value, 6))
        value += step_m
    # The far edge is included even when the step does not land on it, so a surface whose width
    # is not a multiple of the step still offers its edge.
    if xs and xs[-1] < max_x - 1e-9:
        xs.append(round(max_x, 6))

    ys: list[float] = []
    value = min_y
    while value <= max_y + 1e-12:
        ys.append(round(value, 6))
        value += step_m
    if ys and ys[-1] < max_y - 1e-9:
        ys.append(round(max_y, 6))

    centres = [(x, y) for x in xs for y in ys]
    if not centres:
        return []
    mid_x, mid_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    centres.sort(key=lambda point: (-_ring(point, (mid_x, mid_y)), point))
    return centres


def _ring(point: tuple[float, float], centre: tuple[float, float]) -> float:
    return math.dist(point, centre)


# Dynamic objects are spawned this far above their support rather than exactly on it.
#
# A body placed with its base exactly on the support plane lands a fraction of a millimetre *inside*
# it once floating-point has had its say, and a solver resolves that penetration by pushing the body
# out -- violently. Measured here: a cup 0.3 mm into a desk top was ejected half a metre in one step
# and ended on the floor, while a notebook 0.5 mm clear of the same top never moved. The settle then
# reported both as failures for reasons that had nothing to do with physics. Dropping them a couple
# of millimetres is also what a real scene does, and it gives the settle something to actually settle.
SPAWN_CLEARANCE_M = 0.002


@dataclass(frozen=True, slots=True)
class SurfacePlacementConfig:
    """Search parameters for placing an object on a support surface."""

    step_m: float = 0.03
    edge_margin_m: float = 0.01
    object_gap_m: float = 0.02
    yaw_samples: int = 4
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    spawn_clearance_m: float = SPAWN_CLEARANCE_M

    def __post_init__(self) -> None:
        if self.step_m <= 0:
            raise LayoutError("step_m must be positive")
        if self.edge_margin_m < 0 or self.object_gap_m < 0:
            raise LayoutError("margins must not be negative")
        if self.yaw_samples < 1:
            raise LayoutError("yaw_samples must be at least 1")
        if self.max_candidates < 1:
            raise LayoutError("max_candidates must be at least 1")


@dataclass(frozen=True, slots=True)
class ApproachPreference:
    """Robot geometry used to rank otherwise valid surface candidates.

    Omitted by default so existing callers retain exact first-fit behaviour. ``region`` is the
    floor available to the robot base in the coordinate frame of the support; it need not be a
    room and is therefore usable while solving a room-independent task cell.
    """

    region: Polygon
    profile: RobotProfile
    obstacles: tuple[Polygon, ...] = ()
    minimum_directions: int = 1
    direction_samples: int = 24

    def __post_init__(self) -> None:
        if self.minimum_directions < 1:
            raise LayoutError("minimum_directions must be at least 1")
        if self.direction_samples < 1:
            raise LayoutError("direction_samples must be at least 1")
        if self.minimum_directions > self.direction_samples:
            raise LayoutError("minimum_directions cannot exceed direction_samples")


def _approach_direction_count(
    point: tuple[float, float], *, surface: Surface, preference: ApproachPreference
) -> int:
    standoffs = standoffs_for(
        point,
        region=preference.region,
        obstacles=(*preference.obstacles, surface.polygon),
        profile=preference.profile,
        support=surface.polygon,
        directions=preference.direction_samples,
    )
    return len({
        round(math.atan2(item.centre[1] - point[1], item.centre[0] - point[0]), 9)
        for item in standoffs
    })


def place_on_surface(
    *,
    instance_id: str,
    entity_id: str,
    footprint: Footprint,
    surface: Surface,
    object_height_m: float,
    occupancy: OccupancyMap,
    config: SurfacePlacementConfig | None = None,
    origin_convention: str = "centre",
    approach: ApproachPreference | None = None,
) -> LayoutOutcome:
    """Find a position for an object on a support surface.

    The object's base is put on the surface plane, plus the spawn clearance. Where that is in the
    recorded ``z`` depends on the mesh's own origin, which is not the same across libraries:
    ``"centre"`` for a mesh centred on its own middle, ``"base"`` for one whose origin sits under
    it. Assuming the wrong one floats the object by half its height and nothing reports it --
    measured on a real delivery, the pen holder hung 6.7 cm above its desk because a centre-origin
    formula met a base-origin mesh.

    Candidate centres are searched from the outside in, and each is tested against the surface
    polygon inset by the edge margin and against everything already placed, inflated by the
    object gap. Both checks are exact polygon work, so a candidate in a desk's notch is rejected
    for the right reason rather than because a box happened to overlap.
    """
    if origin_convention not in ("centre", "base"):
        raise LayoutError(f"unknown origin convention {origin_convention!r}")
    settings = config or SurfacePlacementConfig()
    if object_height_m <= 0:
        raise LayoutError(f"{instance_id}: object height must be positive")
    try:
        validate_simple(surface.polygon)
    except PolygonError as error:
        raise LayoutError(f"support surface {surface.surface_id} is not usable: {error}") from error

    # Inset by the edge margin so an object is not balanced on the very rim.
    try:
        usable = (
            surface.polygon
            if settings.edge_margin_m == 0
            else _inset(surface.polygon, settings.edge_margin_m)
        )
    except PolygonError as error:
        return LayoutOutcome(
            instance_id=instance_id,
            entity_id=entity_id,
            placement=None,
            failure_code="NO_PLACEMENT_CAPACITY",
            detail=(
                f"surface {surface.surface_id} is smaller than the {settings.edge_margin_m} m "
                f"edge margin: {error}"
            ),
        )

    # The support's owner is excluded: an object on a desk is not obstructed by the desk, whose
    # floor footprint overlaps it in plan view. The floor reservations are dropped for the same
    # reason -- this object rests on a surface, and the corridor below it reserves floor.
    blockers = occupancy.blockers_for(
        instance_id,
        *([surface.owner_instance] if surface.owner_instance else []),
        include_floor_only=False,
    )
    rejections: list[PlacementRejection] = []
    min_x, min_y, max_x, max_y = usable.bounds()
    # Centres must keep the whole footprint inside the usable region, so the lattice is shrunk by
    # the footprint's own half-extents.
    hx, hy = footprint.half_extents
    inner = (min_x + hx, min_y + hy, max_x - hx, max_y - hy)
    if inner[0] > inner[2] or inner[1] > inner[3]:
        return LayoutOutcome(
            instance_id=instance_id,
            entity_id=entity_id,
            placement=None,
            failure_code="NO_PLACEMENT_CAPACITY",
            detail=(
                f"footprint {footprint.width_m:.3f} x {footprint.depth_m:.3f} m does not fit on "
                f"surface {surface.surface_id} after the {settings.edge_margin_m} m edge margin"
            ),
        )

    best: tuple[int, Placement] | None = None
    for attempt, (x, y) in enumerate(_lattice(inner, settings.step_m)):
        if attempt >= settings.max_candidates:
            rejections.append(
                PlacementRejection(x, y, "candidate budget exhausted", None)
            )
            break
        candidate = footprint.at(x, y)
        if not contains_polygon(usable, candidate):
            rejections.append(PlacementRejection(x, y, "footprint leaves the usable surface", None))
            continue
        clearance = check_clearance(
            candidate, obstacles=blockers, margin_m=settings.object_gap_m
        )
        if not clearance.ok:
            rejections.append(
                PlacementRejection(x, y, clearance.reason, clearance.measured_margin_m)
            )
            continue
        placement = Placement(
                instance_id=instance_id,
                entity_id=entity_id,
                footprint=footprint,
                position_m=Vec3(
                    x, y,
                    surface.plane_z_m + settings.spawn_clearance_m
                    + (0.0 if origin_convention == "base" else object_height_m / 2.0),
                ),
                yaw_rad=0.0,
                surface_id=surface.surface_id,
            )
        if approach is None:
            return LayoutOutcome(
                instance_id=instance_id, entity_id=entity_id, placement=placement,
                rejections=tuple(rejections),
            )
        directions = _approach_direction_count(
            (x, y), surface=surface, preference=approach
        )
        if directions < approach.minimum_directions:
            rejections.append(PlacementRejection(
                x, y,
                f"robot has {directions} approach direction(s), fewer than "
                f"the required {approach.minimum_directions}",
            ))
            continue
        if best is None or directions > best[0]:
            best = (directions, placement)

    if best is not None:
        return LayoutOutcome(
            instance_id=instance_id,
            entity_id=entity_id,
            placement=best[1],
            rejections=tuple(rejections),
            detail=f"selected the candidate with {best[0]} robot approach direction(s)",
        )

    return LayoutOutcome(
        instance_id=instance_id,
        entity_id=entity_id,
        placement=None,
        rejections=tuple(rejections),
        failure_code="NO_PLACEMENT_CAPACITY",
        detail=(
            f"no position on {surface.surface_id} accepted {len(rejections)} candidate(s); "
            "see the rejection reasons for how close each came"
        ),
    )


def _inset(polygon: Polygon, distance: float) -> Polygon:
    from rfw_w.scenes.geometry.polygon import inset

    return inset(polygon, distance)


def place_in_zone(
    *,
    instance_id: str,
    entity_id: str,
    footprint: Footprint,
    zone: Polygon,
    object_height_m: float,
    occupancy: OccupancyMap,
    config: SurfacePlacementConfig | None = None,
    against_wall: Polygon | None = None,
    origin_convention: str = "centre",
) -> LayoutOutcome:
    """Find a position for a piece of furniture on the floor within a zone.

    ``origin_convention`` says where the mesh's own origin sits: ``"centre"`` for a mesh centred on
    its bounding box, ``"base"`` for one whose origin is the footprint centre at its underside.
    Assuming the wrong one lifts the object by half its height or buries it, and the result is not
    an error -- it is a scene that renders, settles and looks plausible with the object in the
    wrong place.
    """
    if origin_convention not in ("centre", "base"):
        raise LayoutError(f"unknown origin convention {origin_convention!r}")

    # ``against_wall`` biases the search towards the wall: candidates that touch it come first, so a
    # desk ends up against a wall without the caller having to say exactly where. The zone is the
    # hard constraint; the wall is a preference, and a candidate that satisfies the zone but not the
    # wall is still used rather than failing the layout.

    settings = config or SurfacePlacementConfig()
    if object_height_m <= 0:
        raise LayoutError(f"{instance_id}: object height must be positive")
    try:
        validate_simple(zone)
    except PolygonError as error:
        raise LayoutError(f"zone for {instance_id} is not a usable polygon: {error}") from error

    blockers = occupancy.blockers_for(instance_id)
    hx, hy = footprint.half_extents
    min_x, min_y, max_x, max_y = zone.bounds()
    inner = (min_x + hx, min_y + hy, max_x - hx, max_y - hy)
    if inner[0] > inner[2] or inner[1] > inner[3]:
        return LayoutOutcome(
            instance_id=instance_id,
            entity_id=entity_id,
            placement=None,
            failure_code="NO_PLACEMENT_CAPACITY",
            detail=f"footprint {footprint.width_m:.3f} m is wider than the zone",
        )

    candidates = _lattice(inner, settings.step_m)
    if against_wall is not None:
        wall_box = against_wall.bounding_box()
        # Sorting by distance to the wall puts wall-adjacent candidates first without excluding
        # the rest of the zone, which is what "against the wall" should mean for a preference.
        candidates.sort(key=lambda point: (min(
            abs(point[0] - wall_box.bounds()[0]), abs(point[0] - wall_box.bounds()[2]),
            abs(point[1] - wall_box.bounds()[1]), abs(point[1] - wall_box.bounds()[3]),
        ), point))

    rejections: list[PlacementRejection] = []
    for attempt, (x, y) in enumerate(candidates):
        if attempt >= settings.max_candidates:
            break
        candidate = footprint.at(x, y)
        clearance = check_clearance(candidate, obstacles=blockers, margin_m=settings.object_gap_m)
        if not clearance.ok:
            rejections.append(PlacementRejection(x, y, clearance.reason, clearance.measured_margin_m))
            continue
        if not contains_polygon(zone, candidate):
            # Rejected, not placed-and-flagged. A zone is not decoration: the zone planner keeps
            # furniture out of door sweeps and task space, so an object *outside* its zone may
            # block a door. Choosing to overhang would trade a visible layout failure for an
            # invisible functional one, which is the trade the manual forbids.
            rejections.append(PlacementRejection(x, y, "footprint leaves the zone", None))
            continue

        placed = Placement(
            instance_id=instance_id,
            entity_id=entity_id,
            footprint=footprint,
            position_m=Vec3(
                x, y, 0.0 if origin_convention == "base" else object_height_m / 2.0
            ),
            yaw_rad=0.0,
            surface_id=None,
        )
        return LayoutOutcome(
            instance_id=instance_id, entity_id=entity_id, placement=placed,
            rejections=tuple(rejections),
        )

    return LayoutOutcome(
        instance_id=instance_id,
        entity_id=entity_id,
        placement=None,
        rejections=tuple(rejections),
        failure_code="NO_PLACEMENT_CAPACITY",
        detail=(
            f"no position in the zone accepted {len(rejections)} candidate(s); the zone cannot "
            "hold this footprint, so the zone or the asset must change rather than the placement"
        ),
    )
