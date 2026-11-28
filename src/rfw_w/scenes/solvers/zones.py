# SPDX-License-Identifier: Apache-2.0
"""Zone planning: which part of the room is for what (architecture §7 ``plan_zones``).

This stage exists for one stated reason -- "avoid conflicts between door swing, drawer sweep and
the task workspace" -- and for one that follows from it: without zones, every piece of furniture is
placed against whichever wall is longest, because nothing says where anything belongs. A room whose
desk, chair, cabinet and shelf all crowd one corner is not a solver failure; it is a solver doing
exactly what it was told, with no plan.

So the zones come first, and placement happens inside them:

* **circulation** is the path from the door to the work area, sized to the robot, and it is keepout;
* **work** is the strip along the best non-door wall, which the task's support piece occupies;
* **seating**, **storage**, **display** and **ambient** take the remaining walls and the centre;
* the **door sweep** is reserved outright, because a door that cannot open is a passage that does
  not exist.

Every zone is a polygon, so a notched room keeps its notch and a door on the short wall does not
reserve the wrong strip. And every published polygon is *clipped*: a candidate strip is intersected
with the room outline and then has every reservation subtracted from it, so placeable zones are
mutually disjoint and clear of the keepouts by construction rather than by hope. That matters
because ``placeable_area_m2`` is the denominator a coverage target is measured against -- a zone
reaching outside the room, or on top of a keepout, makes the room look bigger than it is and the
target look met when it is not.

What is left is checked rather than asserted: :meth:`ZoneGraph.check_invariants` recomputes all of
it from the published geometry and names every gap, and :func:`plan_zones` refuses to return a
graph that fails. This module is stdlib-only (``rfw_w.scenes.validation.check_stdlib_only`` walks it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

from rfw_w.scenes.contracts.scene_spec import Density, StylePlan, ZoneKind
from rfw_w.scenes.contracts.state import OpeningKind, RoomShell
from rfw_w.scenes.geometry.clearance import DEFAULT_MARGIN_M, gap_between
from rfw_w.scenes.geometry.polygon import (
    Polygon,
    PolygonError,
    contains_polygon,
    inset,
    interior_point,
    intersect_polygons,
    subtract_polygons,
)
from rfw_w.scenes.solvers.furnish import Wall, openings_on_wall, room_walls

# The strip along a wall that furniture may occupy, measured inward from the wall face. Deeper than
# most furniture by design: a zone is a region a piece is allowed to be in, not a tight fit.
WALL_DEPTH_M = 1.25
# A door sweeps a quarter circle of roughly its own width; reserved as a square of that size.
DOOR_SWEEP_DEPTH_M = 0.95
# Below this, a piece is the sliver a subtraction leaves along a shared edge rather than a region
# anything can stand on. Keeping those would inflate the zone count and the placeable area.
MIN_ZONE_AREA_M2 = 0.05
# Slack for the "is this polygon really inside the room" comparisons, in square metres.
INVARIANT_TOLERANCE_M2 = 1e-6
# The floor the ambient margin may shrink to. A margin shallower than this is not a margin: the
# centre zone would press against the same walls the strips already occupy. If a room cannot take
# even this, the centre is dropped and `ambient_inset_m` records that it was.
MIN_AMBIENT_INSET_M = 0.1


class ZoneError(ValueError):
    """The zone plan cannot be built for this room."""


@dataclass(frozen=True, slots=True)
class Zone:
    """A region of floor with a purpose and a density.

    Held as a set of pieces rather than a single outline, because subtracting a keepout legitimately
    splits a wall strip in two -- and publishing only the larger half would understate the floor the
    zone really offers, which is the denominator the coverage target is read against.
    """

    kind: ZoneKind
    polygons: tuple[Polygon, ...]
    density: Density
    keepout: bool = False
    # Which wall it was derived from, so placement can orient a piece against that wall.
    wall_index: int | None = None

    def __post_init__(self) -> None:
        if not self.polygons:
            raise ZoneError(f"a {self.kind.value} zone needs at least one piece of floor")

    @property
    def polygon(self) -> Polygon:
        """The largest piece, for callers that need one interior point or one orientation."""
        return max(self.polygons, key=lambda piece: piece.area)

    @property
    def area_m2(self) -> float:
        return sum(piece.area for piece in self.polygons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "density": self.density.value,
            "keepout": self.keepout,
            "wall_index": self.wall_index,
            "area_m2": round(self.area_m2, 4),
            "polygons_xy": [piece.to_list() for piece in self.polygons],
        }


@dataclass(frozen=True, slots=True)
class ZoneGraph:
    """Every zone, plus the keepout geometry the layout must respect."""

    room: Polygon
    zones: tuple[Zone, ...]
    door_sweep: tuple[Polygon, ...]
    circulation: tuple[Polygon, ...]
    coverage_target: float
    # The margin the centre zone actually got, or None when even the floor was too deep. Recorded
    # because it is a place the plan adapted to the room's shape, and an adaptation nobody can see
    # is indistinguishable from the zone having been forgotten.
    ambient_inset_m: float | None = None
    # Zone kinds the corridor left no floor for. Same reason: a zone missing because the room
    # could not hold it must not read the same as one that was never asked for.
    dropped_zones: tuple[str, ...] = ()
    # The task cell's protected region, in room coordinates, when the plan was drawn around one.
    # Every zone here was carved around it and the corridor was aimed at it, so it belongs in the
    # plan's own description of itself -- and `check_invariants` reads it to ask whether the way in
    # actually arrives at the task.
    workspace: "Polygon | None" = None

    def of_kind(self, kind: ZoneKind) -> tuple[Zone, ...]:
        return tuple(zone for zone in self.zones if zone.kind is kind)

    def zone_for(self, kind: ZoneKind) -> Zone | None:
        """The largest zone of ``kind``, or ``None`` when the plan has no such zone.

        ``kind`` is *checked*, not compared loosely, and that is the whole point. ``of_kind``
        compares with ``is`` against the ``ZoneKind`` members, so any other value returns an empty
        tuple and this returns ``None`` -- which reads as "the room has no such zone". A caller
        passing a ``PlacementKind`` asked for a zone that could never exist and was handed that
        silence; measured, the support's intended work-zone confinement had therefore never been in
        effect. The build no longer makes that call, and this is the check that would have named it.

        Where a caller wants *all* the floor of a kind rather than its largest part, that is
        :meth:`region_for` -- and for the periphery that is the one it wants.
        """
        self._require_zone_kind(kind)
        found = self.of_kind(kind)
        return max(found, key=lambda zone: zone.area_m2) if found else None

    def region_for(self, kind: ZoneKind) -> Zone | None:
        """Every zone of ``kind`` carried as one region, or ``None`` when there is none.

        ``zone_for`` returns the largest zone of a kind, which is a restriction rather than a
        convenience: measured on the office, three ``DISPLAY`` strips are published (1.02, 0.20 and
        0.06 m2), so a plan row confined to that kind can only ever reach the first and the other
        two carry floor no piece can be placed on.

        Carrying the pieces together is what the permission actually means, and :class:`Zone`
        already says so in its own docstring -- a keepout legitimately splits a strip in two, and
        "publishing only the larger half would understate the floor the zone really offers". The
        density is the largest piece's, and so is ``wall_index``, because that one is a *preference*
        for orientation and not a restriction (see ``_walls_named_by``).
        """
        self._require_zone_kind(kind)
        found = self.of_kind(kind)
        if not found:
            return None
        largest = max(found, key=lambda zone: zone.area_m2)
        return Zone(
            kind=kind,
            polygons=tuple(piece for zone in found for piece in zone.polygons),
            density=largest.density,
            keepout=largest.keepout,
            wall_index=largest.wall_index,
        )

    @staticmethod
    def _require_zone_kind(kind: ZoneKind) -> None:
        if not isinstance(kind, ZoneKind):
            raise ZoneError(
                f"ZONE_KIND_EXPECTED({kind!r}): this takes a ZoneKind, not a "
                f"{type(kind).__name__}; a value that is not one can only ever come back as None, "
                "which is indistinguishable from the room genuinely having no zone of that kind"
            )

    @property
    def keepouts(self) -> tuple[Polygon, ...]:
        """Every polygon placement must avoid: the door sweep, the corridor, and keepout zones."""
        blocked: list[Polygon] = list(self.door_sweep)
        blocked.extend(self.circulation)
        blocked.extend(
            piece
            for zone in self.zones
            if zone.keepout and zone.kind is not ZoneKind.CIRCULATION
            for piece in zone.polygons
        )
        return tuple(blocked)

    @property
    def placeable_area_m2(self) -> float:
        """Floor that furniture may occupy, with the keepouts already removed.

        Already removed by construction -- every placeable zone was clipped before it was
        published -- and :meth:`check_invariants` re-derives the same number from the polygons
        rather than taking this property's word for it.
        """
        return sum(zone.area_m2 for zone in self.zones if not zone.keepout)

    @property
    def periphery_area_m2(self) -> float:
        """Floor the periphery may sample: the room less the task's protected region.

        This is the denominator the claim "the periphery samples in ``room - protected``" rests on,
        and nothing reported it: a reader could see ``placeable_area_m2`` shrink after the room was
        fitted around a cell and not see why. Measured from the two published areas rather than from
        the reservation list, for the same reason ``placeable_area_m2`` is.

        It is *not* what the periphery is confined to -- the zones are, and they are carved with the
        keepouts on top of this. It is the ceiling on what any of them could offer.
        """
        if self.workspace is None:
            return round(self.room.area, 6)
        return round(self.room.area - self.workspace.area, 6)

    def check_invariants(self) -> tuple[str, ...]:
        """Named violations of what this graph claims about the room; empty when it is true.

        Named rather than boolean, so a failure says which zone and which way it is wrong -- the
        convention the rest of the repo uses (``validate_shell``'s gaps, the coverage audit's
        ``UNINDEXED_DIRS(n)``). The numbers are recomputed from the published polygons, so this
        checks the graph rather than restating how it was built.
        """
        gaps: list[str] = []

        for zone in self.zones:
            escaped = _outside_area(zone.polygons, self.room)
            if escaped > INVARIANT_TOLERANCE_M2:
                gaps.append(f"ZONE_OUTSIDE_ROOM({zone.kind.value}, {escaped:.4f} m2)")
        escaped = _outside_area(self.circulation, self.room)
        if escaped > INVARIANT_TOLERANCE_M2:
            gaps.append(f"CIRCULATION_OUTSIDE_ROOM({escaped:.4f} m2)")
        escaped = _outside_area(self.door_sweep, self.room)
        if escaped > INVARIANT_TOLERANCE_M2:
            gaps.append(f"DOOR_SWEEP_OUTSIDE_ROOM({escaped:.4f} m2)")

        placeable = [zone for zone in self.zones if not zone.keepout]
        for index, first in enumerate(placeable):
            for second in placeable[index + 1:]:
                overlap = _shared_area(first.polygons, second.polygons)
                if overlap > INVARIANT_TOLERANCE_M2:
                    gaps.append(
                        f"ZONES_OVERLAP({first.kind.value}, {second.kind.value}, {overlap:.4f} m2)"
                    )

        keepouts = self.keepouts
        for zone in placeable:
            overlap = _shared_area(zone.polygons, keepouts)
            if overlap > INVARIANT_TOLERANCE_M2:
                gaps.append(f"ZONE_OVERLAPS_KEEPOUT({zone.kind.value}, {overlap:.4f} m2)")

        # The task's floor is a *reservation*, not a keepout, so nothing above this line looks at
        # it: `keepouts` carries the door sweep, the corridor and keepout zones, and the protected
        # region is in none of them. It is enforced by construction -- every zone was carved from
        # `reserved`, which holds the protected region first -- and until now only a test watched it
        # (`test_no_zone_and_no_corridor_claims_floor_the_task_reserved`). Recomputing it here is the
        # convention the rest of this method follows: from the published polygons, so the graph is
        # checked rather than the way it was built being restated.
        if self.workspace is not None:
            for zone in placeable:
                overlap = _shared_area(zone.polygons, (self.workspace,))
                if overlap > INVARIANT_TOLERANCE_M2:
                    gaps.append(
                        f"ZONE_OVERLAPS_THE_TASK_FLOOR({zone.kind.value}, {overlap:.4f} m2)"
                    )

        # The passage has to arrive somewhere. It cannot be checked as an overlap, because the
        # corridor is subtracted from its destination by construction, so the two are disjoint by the
        # time this runs; meeting at all is the strongest statement available here, and it is a
        # weaker one than it looks. What it does catch is a corridor aimed at the wrong place and
        # stopping short -- a passage that was reserved but leads nowhere.
        #
        # Its destination is the task cell when there is one. The work strip is a strip on whichever
        # wall looked longest, and the corridor was only ever aimed at it as a guess at where the
        # task would be; given a cell, "arrives somewhere" has to mean "arrives at the task".
        if self.circulation:
            workspace = self.workspace
            if workspace is not None:
                arrives = any(
                    gap_between(piece, workspace) <= INVARIANT_TOLERANCE_M2
                    for piece in self.circulation
                )
                if not arrives:
                    gaps.append(
                        "CIRCULATION_MISSES_TASK_CELL(passage reserved but does not reach the "
                        "task's protected region)"
                    )
            else:
                work = self.zone_for(ZoneKind.WORK)
                if work is not None:
                    arrives = any(
                        gap_between(piece, region) <= INVARIANT_TOLERANCE_M2
                        for piece in self.circulation
                        for region in work.polygons
                    )
                    if not arrives:
                        gaps.append(
                            "CIRCULATION_MISSES_WORK_ZONE(passage reserved but leads nowhere)"
                        )

        return tuple(gaps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_polygon_xy": self.room.to_list(),
            "coverage_target": self.coverage_target,
            "placeable_area_m2": round(self.placeable_area_m2, 4),
            "room_area_m2": round(self.room.area, 4),
            "ambient_inset_m": self.ambient_inset_m,
            "dropped_zones": list(self.dropped_zones),
            "door_sweep": [piece.to_list() for piece in self.door_sweep],
            "circulation": [piece.to_list() for piece in self.circulation],
            # None when the plan was drawn without a task cell. A reader can then tell a plan that
            # had no workspace from one whose workspace happened to be empty.
            "workspace_xy": None if self.workspace is None else self.workspace.to_list(),
            "zones": [zone.to_dict() for zone in self.zones],
        }


def _shared_area(left: Sequence[Polygon], right: Sequence[Polygon]) -> float:
    """Area two groups of polygons have in common. Exact, so it can be compared to a threshold."""
    return sum(
        piece.area
        for first in left
        for second in right
        for piece in intersect_polygons(first, second)
    )


def _outside_area(pieces: Sequence[Polygon], reference: Polygon) -> float:
    """How much of ``pieces`` is not inside ``reference``."""
    return sum(
        max(0.0, piece.area - sum(part.area for part in intersect_polygons(piece, reference)))
        for piece in pieces
    )


def _strip(wall: Wall, depth_m: float, *, start_m: float, end_m: float) -> Polygon:
    """A quadrilateral running along a wall and ``depth_m`` into the room."""
    if end_m - start_m <= 0.05:
        raise ZoneError(f"strip along wall {wall.index} is too short ({end_m - start_m:.3f} m)")
    a = wall.point_along(start_m)
    b = wall.point_along(end_m)
    inward = wall.inward
    return Polygon((
        a, b,
        (b[0] + inward[0] * depth_m, b[1] + inward[1] * depth_m),
        (a[0] + inward[0] * depth_m, a[1] + inward[1] * depth_m),
    ))


def _band(wall: Wall, *, from_wall_m: float, to_wall_m: float,
          start_m: float, end_m: float) -> Polygon:
    """A quadrilateral between two depths from a wall, along a span of it."""
    if to_wall_m <= from_wall_m:
        raise ZoneError(f"band depth must increase, got {from_wall_m} -> {to_wall_m}")
    if end_m - start_m <= 0.05:
        raise ZoneError("band span is too short")
    a, b = wall.point_along(start_m), wall.point_along(end_m)
    inward = wall.inward
    return Polygon((
        (a[0] + inward[0] * from_wall_m, a[1] + inward[1] * from_wall_m),
        (b[0] + inward[0] * from_wall_m, b[1] + inward[1] * from_wall_m),
        (b[0] + inward[0] * to_wall_m, b[1] + inward[1] * to_wall_m),
        (a[0] + inward[0] * to_wall_m, a[1] + inward[1] * to_wall_m),
    ))


def _corridor(start: tuple[float, float], end: tuple[float, float], width_m: float) -> Polygon:
    """A rectangle of ``width_m`` running from ``start`` to ``end``.

    Built as a rotated rectangle rather than an axis-aligned box: a corridor from a door on a short
    wall to a desk in a corner runs diagonally, and its bounding box would reserve floor the path
    never crosses -- blocking placements that are in fact clear.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        raise ZoneError("a corridor needs two distinct ends")
    ux, uy = dx / length, dy / length
    nx, ny = -uy * width_m / 2.0, ux * width_m / 2.0
    return Polygon((
        (start[0] + nx, start[1] + ny),
        (end[0] + nx, end[1] + ny),
        (end[0] - nx, end[1] - ny),
        (start[0] - nx, start[1] - ny),
    ))


def _closest_point_on_segment(point: tuple[float, float], start: tuple[float, float],
                              end: tuple[float, float]) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-18:
        return start
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    return (start[0] + t * dx, start[1] + t * dy)


def _nearest_boundary_point(
    pieces: Sequence[Polygon], point: tuple[float, float],
) -> tuple[tuple[float, float], Polygon]:
    """The closest point on any piece's boundary to ``point``, and the piece it is on."""
    best: tuple[float, float] | None = None
    best_piece: Polygon | None = None
    best_distance = math.inf
    for piece in pieces:
        for start, end in piece.edges:
            candidate = _closest_point_on_segment(point, start, end)
            distance = math.dist(point, candidate)
            if distance < best_distance - 1e-12:
                best_distance, best, best_piece = distance, candidate, piece
    if best is None or best_piece is None:
        raise ZoneError("no zone geometry to aim the corridor at")
    return best, best_piece


def _toward(start: tuple[float, float], end: tuple[float, float],
            distance: float) -> tuple[float, float]:
    """The point ``distance`` from ``start`` along the line to ``end``, never overshooting it."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return start
    step = min(distance, length * 0.5)
    return (start[0] + dx / length * step, start[1] + dy / length * step)


def _door_wall(walls: Sequence[Wall], shell: RoomShell) -> tuple[Wall | None, tuple[float, float, float] | None]:
    """The wall carrying the door, and that door's ``(centre_x, centre_y, width)``.

    Refuses a shell with more than one door rather than planning the first and ignoring the rest.
    One sweep is reserved and one corridor is drawn, so a second door would be planned as if it did
    not exist -- a quiet lie of exactly the kind the named gaps in this module exist to avoid. The
    shell produces one door today; this is the flag that must be removed when it produces more.
    """
    doors = [o for o in shell.openings if o.kind is OpeningKind.DOOR]
    if not doors:
        return None, None
    if len(doors) > 1:
        raise ZoneError(
            f"MULTIPLE_DOORS_NOT_PLANNED({len(doors)}): the zone plan reserves one door sweep and "
            "one corridor; planning the others is a change to this function, not to the shell"
        )
    door = doors[0]
    for wall in walls:
        spans = openings_on_wall(wall, [(door.opening_id, door.center_xy[0], door.center_xy[1], door.width_m)])
        if spans:
            return wall, (door.center_xy[0], door.center_xy[1], door.width_m)
    return None, (door.center_xy[0], door.center_xy[1], door.width_m)


def door_sweep_for(shell: RoomShell) -> tuple[Polygon, ...]:
    """The part of the door's inward swing that occupies real room floor.

    Shared by zone planning and core-first room fitting so the two stages cannot disagree about
    whether the task cell blocks the door.
    """
    room = Polygon(shell.polygon_xy)
    walls = room_walls(room)
    door_wall, door = _door_wall(walls, shell)
    if door is None or door_wall is None:
        return ()
    along = math.dist(door_wall.start, (door[0], door[1]))
    along = max(0.0, min(door_wall.length_m, along))
    half = door[2] / 2.0
    a = door_wall.point_along(max(0.0, along - half))
    b = door_wall.point_along(min(door_wall.length_m, along + half))
    inward = door_wall.inward
    depth = max(DOOR_SWEEP_DEPTH_M, door[2])
    sweep = Polygon((
        a, b,
        (b[0] + inward[0] * depth, b[1] + inward[1] * depth),
        (a[0] + inward[0] * depth, a[1] + inward[1] * depth),
    ))
    if contains_polygon(room, sweep):
        return (sweep,)
    try:
        return tuple(
            piece for piece in intersect_polygons(sweep, room)
            if piece.area >= MIN_ZONE_AREA_M2
        )
    except PolygonError as error:
        raise ZoneError(f"cannot clip the door sweep to the room outline: {error}") from error


def plan_zones(
    shell: RoomShell,
    style: StylePlan,
    *,
    robot_footprint_m: float = 0.6,
    margin_m: float = DEFAULT_MARGIN_M,
    work_wall_depth_m: float = WALL_DEPTH_M,
    cell_protected: Polygon | None = None,
) -> ZoneGraph:
    """Divide the room into functional zones from its shell and its style plan.

    The door decides everything downstream: it is where passage enters, so the wall it sits on is
    never the work wall, the strip in front of it is reserved, and the path from it to the work area
    is keepout. Without that, a desk placed against the door wall blocks the only way in -- and a
    layout check that only measured door *width* would not notice.

    ``cell_protected`` is the room-free task cell's protected region, already translated into the
    room by ``fit_room_around_cell``. Where the ``WORK`` strip is a *proxy* for the task's work area
    -- a strip on whichever wall looked longest -- this is the work area itself: the support, the
    places the robot may stand, and the approach between them. Given it, the plan

    * reserves it, so the existing subtraction machinery keeps every zone and every piece out of it,
      and
    * ends the door corridor **on its boundary** instead of aiming at the strip.

    ``None`` -- the default -- leaves the plan byte-identical to what it was before this parameter
    existed, which is what the room-only tools and the zone tests exercise.

    Raises :class:`ZoneError` rather than returning a graph that misdescribes the room: the caller
    gets a named gap it can report, not a plan that quietly understates the keepouts.
    """
    if robot_footprint_m <= 0:
        raise ZoneError(f"robot footprint must be positive, got {robot_footprint_m}")

    room = Polygon(shell.polygon_xy)
    walls = room_walls(room)
    if not walls:
        raise ZoneError("the shell has no usable walls")

    if cell_protected is not None and not contains_polygon(room, cell_protected):
        # The fit is supposed to guarantee this, so reaching here means the fit and the shell have
        # come apart. Zoning "around" a region that is partly outside the room would produce a
        # plausible-looking plan over floor that does not exist.
        c_min_x, c_min_y, c_max_x, c_max_y = cell_protected.bounds()
        r_min_x, r_min_y, r_max_x, r_max_y = room.bounds()
        raise ZoneError(
            f"CELL_PROTECTED_OUTSIDE_ROOM(cell={[round(v, 4) for v in (c_min_x, c_min_y, c_max_x, c_max_y)]}, "
            f"room={[round(v, 4) for v in (r_min_x, r_min_y, r_max_x, r_max_y)]}): the task cell's "
            "protected region is not inside the shell, so the plan would reserve floor the room "
            "does not have"
        )

    door_wall, door = _door_wall(walls, shell)
    corridor_width = robot_footprint_m + 2 * margin_m

    def clip_to_room(candidate: Polygon) -> tuple[Polygon, ...]:
        """The part of a candidate region that is real floor, split into usable pieces."""
        if contains_polygon(room, candidate):
            # Kept whole rather than run through the intersection, which would hand back the same
            # area as a set of triangles and make every zone harder to reason about than it is.
            return (candidate,) if candidate.area >= MIN_ZONE_AREA_M2 else ()
        try:
            pieces = intersect_polygons(candidate, room)
        except PolygonError as error:
            raise ZoneError(f"cannot clip a zone to the room outline: {error}") from error
        return tuple(piece for piece in pieces if piece.area >= MIN_ZONE_AREA_M2)

    def free_of(pieces: Sequence[Polygon], blockers: Sequence[Polygon]) -> tuple[Polygon, ...]:
        """``pieces`` with every blocker removed."""
        remaining = list(pieces)
        for blocker in blockers:
            if not remaining:
                break
            carved: list[Polygon] = []
            for piece in remaining:
                carved.extend(subtract_polygons(piece, blocker))
            remaining = [piece for piece in carved if piece.area >= MIN_ZONE_AREA_M2]
        return tuple(remaining)

    reserved: list[Polygon] = []
    # The task cell first. It is the one reservation that is not a choice about this room: the
    # support, the floor the robot may stand on and the approach between them were solved before the
    # room existed, and every zone below is carved around them rather than the other way round.
    if cell_protected is not None:
        reserved.append(cell_protected)
    door_sweep = door_sweep_for(shell)
    if door_sweep:
        reserved.extend(door_sweep)

    # Candidate walls for furniture, longest first, excluding the door wall.
    usable_walls = [w for w in walls if door_wall is None or w.index != door_wall.index]
    usable_walls.sort(key=lambda w: -w.length_m)
    if not usable_walls:
        usable_walls = list(walls)

    zones: list[Zone] = []

    def add(kind: ZoneKind, candidate: Polygon, *, density: Density,
            wall_index: int | None = None) -> Zone | None:
        """Publish a candidate as a zone, keeping only floor that is still free.

        Anything a previously accepted zone or the door sweep already claims is subtracted first,
        which is what makes the published zones disjoint and clear of the keepouts by construction.
        """
        pieces = free_of(clip_to_room(candidate), reserved)
        if not pieces:
            return None
        zone = Zone(kind=kind, polygons=pieces, density=density, wall_index=wall_index)
        zones.append(zone)
        reserved.extend(pieces)
        return zone

    def add_strip(kind: ZoneKind, wall: Wall, *, depth: float, fraction: float) -> Zone | None:
        span = wall.length_m - 0.3
        if span <= 0.4:
            return None
        try:
            candidate = _strip(wall, depth, start_m=0.15, end_m=0.15 + span * fraction)
        except (ZoneError, PolygonError):
            return None
        return add(kind, candidate, density=style.density_for(kind), wall_index=wall.index)

    if usable_walls:
        add_strip(ZoneKind.WORK, usable_walls[0], depth=work_wall_depth_m, fraction=0.75)
    if len(usable_walls) > 1:
        add_strip(ZoneKind.STORAGE, usable_walls[1], depth=0.65, fraction=0.9)
    for wall in usable_walls[2:]:
        add_strip(ZoneKind.DISPLAY, wall, depth=0.45, fraction=0.9)

    # Seating sits *beyond* the work strip, not on top of it: a chair is between the desk and the
    # room, and a strip built from the wall inward would simply be subtracted away by the work
    # zone's own claim. A band between two depths is the shape that describes it.
    #
    # Unless the task cell is what the room was fitted around. The band is anchored to a wall
    # because the support's position was a wall's business; now it is the cell's, and the cell's
    # work area is 1.8 m deep -- support and the aisle the arm needs in front of it. Measured on
    # `office_pick_cup` seed 11, the band landed on the wall the workspace itself occupies and the
    # seating zone came out at 0.11-0.72 m2, which no chair fits in, while the chair was the one
    # piece the randomization closure would not let the run drop.
    #
    # What the chair actually needs is floor beside the support it faces, and its placement rule
    # already knows that: `BESIDE` with `faces_support` ranks candidates by distance to the support
    # it looks up in `placed_by_entity`. A wall strip was the best available proxy for "beside the
    # desk" while the desk's position was unknown. It is known now, so the proxy is not published,
    # and the chair searches the floor the workspace left rather than the band a wall guessed at.
    if usable_walls and cell_protected is None:
        wall = usable_walls[0]
        span = wall.length_m - 0.3
        if span > 0.5:
            try:
                band = _band(wall, from_wall_m=work_wall_depth_m * 0.85,
                             to_wall_m=work_wall_depth_m * 1.7,
                             start_m=0.15, end_m=0.15 + span * 0.5)
            except (ZoneError, PolygonError):
                band = None
            if band is not None:
                add(ZoneKind.SEATING, band, density=style.density_for(ZoneKind.SEATING),
                    wall_index=wall.index)

    # The centre is available for floor coverings and floor-standing pieces, minus a wide margin so
    # they do not press against the walls the strips already occupy.
    #
    # The margin is walked down when the room cannot take it, and the value used is recorded. A T-
    # or U-shaped room has a neck, and an inset deeper than half that neck's width crosses its own
    # offset -- measured on a room whose middle is 1.079 m wide, the configured 0.75 m inset is a
    # bow tie. Refusing the inset and dropping the zone, which is what this did, leaves the room
    # with no floor for its rug and nothing anywhere to say why; shrinking it keeps the room usable
    # and puts the number in the manifest.
    ambient_inset_m = min(0.9, max(0.3, work_wall_depth_m * 0.6))
    while ambient_inset_m > MIN_AMBIENT_INSET_M:
        try:
            centre = inset(room, ambient_inset_m)
            break
        except PolygonError:
            ambient_inset_m = round(ambient_inset_m / 2, 6)
    else:
        centre = None
    if centre is not None:
        add(ZoneKind.AMBIENT, centre, density=style.density_for(ZoneKind.AMBIENT))

    # The corridor is drawn from the door centre to the work zone, and ends one margin *inside* it
    # rather than on its boundary. Landing on the boundary makes the passage touch the zone at a
    # single point -- and at a corner of the strip, the continuation of the line from the door
    # leaves the zone altogether, so the passage reached nothing and the overlap measured exactly
    # zero. The push is one margin and no more: the corridor is a keepout subtracted from the work
    # strip, so pushing a whole robot footprint in would cut the strip in two and leave no room for
    # the desk the strip exists for. Clipping to the room is what stops the corridor reserving floor
    # outside the building, which is where half its width starts when the door is on the wall line.
    circulation: list[Polygon] = []
    if door is not None:
        work = next((zone for zone in zones if zone.kind is ZoneKind.WORK), None)
        door_point = (door[0], door[1])
        if cell_protected is not None:
            # The way in ends where the task begins. This is the whole of what the work-zone branch
            # below was approximating: the strip it aims at is a strip on whichever wall looked
            # longest, and the work area the passage is supposed to lead to is the cell.
            destination, _ = _nearest_boundary_point((cell_protected,), door_point)
        elif work is not None:
            nearest, piece = _nearest_boundary_point(work.polygons, door_point)
            try:
                inside = interior_point(piece, margin=margin_m * 2.0)
            except PolygonError as error:
                raise ZoneError(
                    f"the work strip is too thin to aim a passage into: {error}"
                ) from error
            destination = _toward(nearest, inside, margin_m)
        else:
            destination = interior_point(room, margin=corridor_width)
        try:
            corridor = _corridor(door_point, destination, corridor_width)
        except (ZoneError, PolygonError):
            corridor = None
        if corridor is not None:
            pieces = clip_to_room(corridor)
            if cell_protected is not None:
                # The destination lies *on* the protected boundary, but the corridor is a rectangle
                # ending there, and a rectangle whose end edge is not parallel to that boundary still
                # has corners past it. Subtracting makes "the way in never enters the workspace" a
                # property of the construction rather than a piece of luck about the angles.
                pieces = free_of(pieces, (cell_protected,))
            circulation.extend(pieces)

    # The corridor is keepout, so it comes out of every placeable zone that was built before it.
    if circulation:
        trimmed: list[Zone] = []
        dropped: list[str] = []
        for zone in zones:
            pieces = free_of(zone.polygons, circulation)
            if not pieces:
                if zone.kind is ZoneKind.WORK and cell_protected is None:
                    # The one zone the task cannot do without, and the one the space gate asks
                    # for. A room whose way in crosses all of it cannot hold the task.
                    #
                    # Only while the work strip is standing in for the task's work area. With a cell
                    # present the task's floor is the cell's protected region -- reserved before any
                    # zone was drawn -- and the strip is one wall among several, so a corridor that
                    # crosses all of it is a smaller periphery, not an unusable room.
                    raise ZoneError(
                        f"the passage from the door leaves no floor for the {zone.kind.value} "
                        "zone; this room cannot hold the task and the way in at once"
                    )
                # Every other zone is floor for something the task does not need -- a rug, a
                # plant, a display wall. A room whose corridor crosses all of it has no floor left
                # for that thing, which is a smaller room than was hoped for, not an unusable one.
                # Refusing the whole room for it was the behaviour, and it rejected 3 of 40 seeds
                # of the office spec once the shell gained shape families; the zone is recorded as
                # dropped instead, so the absence is visible rather than silent.
                dropped.append(zone.kind.value)
                continue
            trimmed.append(replace(zone, polygons=pieces) if pieces != zone.polygons else zone)
        zones = trimmed
        zones.append(Zone(kind=ZoneKind.CIRCULATION, polygons=tuple(circulation),
                          density=Density.SPARSE, keepout=True))
    else:
        dropped = []

    graph = ZoneGraph(
        room=room, zones=tuple(zones), door_sweep=door_sweep, circulation=tuple(circulation),
        coverage_target=style.effective_coverage_target,
        ambient_inset_m=ambient_inset_m if centre is not None else None,
        dropped_zones=tuple(dropped),
        workspace=cell_protected,
    )
    gaps = graph.check_invariants()
    if gaps:
        raise ZoneError("the zone plan does not describe its own room: " + "; ".join(gaps))
    return graph
