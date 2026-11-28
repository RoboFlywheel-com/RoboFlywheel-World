# SPDX-License-Identifier: Apache-2.0
"""The room plan: core's conclusion, expressed as a metric floor plan.

The task cell is solved in a frame that has no room in it (``solvers/cell.py``), and the room is
then drawn *around* it rather than the cell being fitted into a room drawn first. That reversal is
worth a measured amount -- it moved the office cup's usable approach directions from 2 of 24 to 19
-- and it only keeps that value if **the core stops moving once the room exists**.

This module is the thing that makes the handover explicit. It is a *contract*: a frozen, hashed
description of a room that a provider can build from, so "what we asked for" is a document that can
be compared against "what got built" instead of being whatever the builder happened to do.

Two rules shape every field here.

**One truth per fact.** ``workspace_polygon_xy`` was in the first draft of this contract and is
deliberately absent: the zone planner's workspace *is* the cell's protected region
(``solvers/zones.py:760`` passes ``cell_protected`` straight into it), so a second copy would be
the same polygon written twice -- and two representations of one fact are free to disagree. The
same standard refuses ``RoomShell.holes_xy``, and it applies here.

**Expectations are labelled as expectations.** ``width_m`` / ``height_m`` / ``sill_z_m`` on an
opening are what we would *like*; they are not a promise about built geometry, because the one
provider that builds rooms from an external generator does not honour them (SPEC-001 §5 has the
measurement: a 0.9 m door line and a 2.0 m door line produce the same 0.929 m door). What the plan
does promise is that *some* opening of that kind is on that wall. Keeping the two apart is what
lets ``shell.matches_plan`` compare honestly instead of failing on a quantity nobody controls.

The freedom that distinction buys is spent in exactly one place: `floor_plan_payload` hands a
window a **host span** wider than the width requested, because a provider that never gets a
decision right is a provider that has to be given room to make it. Everything else in this module
treats the plan's own geometry as the thing that was asked for.

Stdlib only, no I/O, no geometry import: a contract has to be checkable in a unit test with no
asset root and no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rfw_w.scenes.contracts.hashing import content_hash, is_digest
from rfw_w.scenes.contracts.state import Opening, OpeningKind, RoomShell

SCHEMA_VERSION = "pvscene.room-plan/v1"

#: Explicit, not inferred. An external plan in feet or a Y-up export would otherwise be read as
#: metres and Z-up and be wrong in a way that only shows up as geometry.
UNITS = "m"
UP_AXIS = "Z"

#: Sources a plan can come from. ``bank`` is reserved for pre-generated rooms; only ``core`` is
#: produced today.
SOURCES = ("core", "external", "bank")

#: The tolerance R7 needs, derived rather than picked.
#:
#: A plan's geometry is written with 6 decimals (``solvers/room_plan.py`` rounds the segment's two
#: endpoints the way ``_materialise`` rounds an opening's centre), so each endpoint moves by up to
#: ``5e-7`` in x and y, and the length of a *diagonal* segment therefore moves by up to
#: ``2 * sqrt(2) * 5e-7 = 1.42e-6``. The tolerance has to clear that or a slanted wall would be
#: reported as a disagreement with itself. It is still three orders below anything geometric: the
#: R7 test moves a width by 5 cm.
SEGMENT_LENGTH_TOLERANCE_M = 1e-5

#: `RoomShell.source` for a shell derived from a plan. The suffix carries the plan's own source,
#: so a reader can tell a core-produced room from an external one without opening the plan
#: (ADR-001 asks for exactly this).
PLAN_SHELL_SOURCE = "plan"

#: Infinigen's own constants, from the version this repository pins (`INFINIGEN_PIN`). They are not
#: ours to choose and they are why an opening's `width_m` is a request rather than a promise.
#:
#: `RoomConstants.global_params(segment_margin=1.4)` (`room/constants.py:54`) and
#: `RoomConstants.door_params(door_width_ratio=("uniform", 0.7, 0.8))` (`:68`).
INFINIGEN_SEGMENT_MARGIN_M = 1.4
INFINIGEN_DOOR_WIDTH_RATIO_MAX = 0.8


def buildable_window_span_m(wall_thickness_m: float) -> float:
    """The longest stretch Infinigen can still refuse to build a window in.

    `door_width = (segment_margin - wall_thickness) * rg(door_width_ratio)`
    (`room/constants.py:70`) with the ratio drawn from ``uniform(0.7, 0.8)``, so this is the top of
    the range the draw can land on -- the value a span has to clear whatever seed the build got.
    """
    return (INFINIGEN_SEGMENT_MARGIN_M - wall_thickness_m) * INFINIGEN_DOOR_WIDTH_RATIO_MAX


#: The stretch of wall a window is given to sit in, and why it is not the width the plan asked for.
#:
#: Infinigen decides whether a window exists at all from the length of the line it is handed, and it
#: decides it by **dropping** it: `make_window_cutter` (`room/solidifier.py:592`) walks
#: `split_mls(mls, door_width)`, and `split_mls` (`:160`) yields only the straight pieces *longer*
#: than `door_width`. A piece at or below it yields nothing -- the loop body never runs, no cutter is
#: made, and the built room has neither the hole nor any statement that one was asked for. Measured
#: on the 2026-09-21 office build: pieces of 1.283 / 1.386 m were built, a piece of 0.926 m against
#: a `door_width` of 0.9296 was not (SPEC-004 §8).
#:
#: At the 0.1 m thickness this repository hands Infinigen (`INFINIGEN_WALL_THICKNESS_M`) the ceiling
#: is `buildable_window_span_m(0.1) = 1.04 m`, and 1.10 clears it by 6 cm. The margin is for the
#: thickness, which is the one number the plan cannot supply: 1.10 still clears every thickness from
#: 0.025 m to 0.15 m, so a future provider asking for other walls does not lose its windows on a
#: difference nobody wrote down. `tests/unit/test_room_plan.py` pins that relationship.
WINDOW_HOST_SPAN_M = 1.10


class RoomPlanError(ValueError):
    """A plan could not be constructed or serialised. The message leads with the failure code."""


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _points(raw: Any, *, what: str) -> tuple[tuple[float, float], ...]:
    """Read ``[[x, y], ...]`` into a tuple of float pairs, refusing anything else.

    Refusing rather than coercing is the point: a plan quietly built from strings or from
    three-element rows would produce geometry that is subtly somewhere else.
    """
    if not isinstance(raw, (list, tuple)):
        raise RoomPlanError(f"PLAN_NOT_SERIALISABLE({what}): expected a list of [x, y], got {type(raw).__name__}")
    points: list[tuple[float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RoomPlanError(f"PLAN_NOT_SERIALISABLE({what}[{index}]): expected [x, y]")
        x, y = item
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise RoomPlanError(f"PLAN_NOT_SERIALISABLE({what}[{index}]): non-numeric coordinate")
        if not _finite(float(x), float(y)):
            raise RoomPlanError(f"PLAN_NOT_SERIALISABLE({what}[{index}]): non-finite coordinate")
        points.append((float(x), float(y)))
    return tuple(points)


def _segment_length(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.dist(start, end)


def _midpoint(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)


#: How far off a wall a point may be and still be on it. The same millimetre the shell gate uses for
#: the same question (`contracts/shell_build.py`): a plan writes six decimals, so an endpoint sits on
#: the line it was computed from to within a rounding, and nothing off by more than a millimetre is
#: on that wall.
WALL_TOLERANCE_M = 1e-3


def _on_wall(
    wall_start: tuple[float, float],
    wall_end: tuple[float, float],
    point: tuple[float, float],
    *,
    tolerance_m: float = WALL_TOLERANCE_M,
) -> bool:
    """Whether a point lies on a wall, ends included."""
    dx, dy = wall_end[0] - wall_start[0], wall_end[1] - wall_start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 0:
        return False
    t = ((point[0] - wall_start[0]) * dx + (point[1] - wall_start[1]) * dy) / length_squared
    # The slack along the wall is the tolerance in metres, not in `t`: a millimetre past the end of
    # a four-metre wall must not be the same slack as a millimetre past the end of a half-metre one.
    slack = tolerance_m / math.sqrt(length_squared)
    if not (-slack <= t <= 1 + slack):
        return False
    projection = (wall_start[0] + t * dx, wall_start[1] + t * dy)
    return math.dist(projection, point) <= tolerance_m


def _edges(
    polygon_xy: tuple[tuple[float, float], ...],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The outline's edges, cyclically. The same list `_place_openings` indexes into."""
    return [(polygon_xy[i], polygon_xy[(i + 1) % len(polygon_xy)]) for i in range(len(polygon_xy))]


def _wall_containing(
    polygon_xy: tuple[tuple[float, float], ...],
    segment: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The wall both of a segment's endpoints lie on, or ``None``.

    Geometric rather than by index, because the plan does not carry one: `PlanOpening` names a
    stretch of wall, not a wall, and `wall_index` is gone by the time an opening reaches a shell
    (`solvers/room_plan.py` says why). Re-deriving it here from the outline keeps that, at the cost
    of asking a question whose answer is exact -- an endpoint is either on the line or it is not.
    """
    for start, end in _edges(polygon_xy):
        if _on_wall(start, end, segment[0]) and _on_wall(start, end, segment[1]):
            return start, end
    return None


def _widen_about_centre(
    segment: tuple[tuple[float, float], tuple[float, float]],
    wall: tuple[tuple[float, float], tuple[float, float]],
    minimum_length_m: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """A segment grown about its midpoint to `minimum_length_m`, then held inside its wall.

    Grown symmetrically first, so the provider is asked for the place the plan named rather than a
    place beside it; clamped second, so a stretch the wall cannot hold slides back inside it instead
    of running through the corner. A wall shorter than the minimum yields the whole wall, which is
    the most it can give -- the provider that cannot build in that is expected to say so by name.
    """
    (ax, ay), (bx, by) = wall
    wall_length = math.dist((ax, ay), (bx, by))
    if wall_length <= 0:
        return segment
    ux, uy = (bx - ax) / wall_length, (by - ay) / wall_length
    (sx, sy), (ex, ey) = segment
    centre = _midpoint((sx, sy), (ex, ey))
    offset = (centre[0] - ax) * ux + (centre[1] - ay) * uy
    half = max(math.dist((sx, sy), (ex, ey)), minimum_length_m) / 2.0
    low, high = offset - half, offset + half
    if low < 0.0:
        high, low = high - low, 0.0
    if high > wall_length:
        low, high = low - (high - wall_length), wall_length
    low, high = max(low, 0.0), min(high, wall_length)
    near = (ax + ux * low, ay + uy * low)
    far = (ax + ux * high, ay + uy * high)
    # Returned in the *segment's* direction rather than the wall's, so a caller can compare the
    # span against the segment it came from without knowing which way the outline is wound. The
    # two are the same stretch of wall either way -- this is about which end is written first.
    if (ex - sx) * ux + (ey - sy) * uy < 0:
        return far, near
    return near, far


def opening_host_span(
    plan: RoomPlan, opening: PlanOpening
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The stretch of wall an opening is given to sit in.

    **A door gets its own segment, widened by nothing.** The door cutter (`solidifier.py:543`) takes
    `max_mls` of whatever it is handed -- the longest piece, with no length test at all -- so a door
    is built whatever length it is given. What that cutter does do is size the door from `door_width`
    and place it anywhere within the piece, which SPEC-004 §4 R2 records as outside the comparison
    set. Widening would change nothing about whether a door exists.

    **A window gets its segment grown to `WINDOW_HOST_SPAN_M`**, because a window is the one opening
    Infinigen will silently not build (see that constant). Grown about the segment's own centre and
    clamped to the wall it lies on.

    Both directions of the handoff call this one function -- `floor_plan_payload` to write what the
    provider is given, and the shell gate to decide which planned opening a measured hole is -- so
    the span a hole is held to is the span it was handed. Widening on the serialisation side alone
    would hold a hole to a stretch its builder never saw, and widening on the comparison side alone
    would do the same in the other direction.

    **The plan's `wall_segment_xy` is not changed by this** and stays exactly what the drawer asked
    for. Widening it in the plan would move every plan hash and every scene version, and would make
    the document claim a width nobody drew -- the opposite of what `width_m` is for.

    Raises `PLAN_OPENING_NOT_ON_A_WALL` when the segment lies on none of the room's walls. That is a
    malformed plan rather than a provider problem, and `check_plan` answers the same question as a
    gap, so neither has to know whether the other noticed.
    """
    if opening.kind != OpeningKind.WINDOW.value:
        return opening.wall_segment_xy
    room = next((room for room in plan.rooms if room.room_id == opening.room_id), None)
    if room is None:
        raise RoomPlanError(
            f"PLAN_OPENING_NOT_ON_A_WALL({opening.opening_id}): room {opening.room_id!r} is not in "
            "this plan, so the wall behind the segment cannot be found"
        )
    wall = _wall_containing(room.polygon_xy, opening.wall_segment_xy)
    if wall is None:
        raise RoomPlanError(
            f"PLAN_OPENING_NOT_ON_A_WALL({opening.opening_id}, {opening.wall_segment_xy}): the plan "
            "puts an opening on a stretch that is not part of any wall it declares"
        )
    return _widen_about_centre(opening.wall_segment_xy, wall, WINDOW_HOST_SPAN_M)


@dataclass(frozen=True, slots=True)
class PlanRoom:
    """One room's outline. Single-room by construction, but keyed so multi-room is an addition."""

    room_id: str
    kind: str
    polygon_xy: tuple[tuple[float, float], ...]
    floor_z_m: float = 0.0
    ceiling_height_m: float = 2.8
    floor_material: str = ""
    wall_material: str = ""

    def __post_init__(self) -> None:
        if not self.room_id:
            raise RoomPlanError("PLAN_NOT_SERIALISABLE(room_id): a room needs an id")
        if len(self.polygon_xy) < 3:
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE({self.room_id}): a room outline needs at least 3 vertices, "
                f"got {len(self.polygon_xy)}"
            )
        if self.ceiling_height_m <= 0:
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE({self.room_id}): ceiling height must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "kind": self.kind,
            "polygon_xy": [list(vertex) for vertex in self.polygon_xy],
            "floor_z_m": self.floor_z_m,
            "ceiling_height_m": self.ceiling_height_m,
            "floor_material": self.floor_material,
            "wall_material": self.wall_material,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanRoom":
        return cls(
            room_id=str(payload.get("room_id", "")),
            kind=str(payload.get("kind", "")),
            polygon_xy=_points(payload.get("polygon_xy"), what="rooms.polygon_xy"),
            floor_z_m=float(payload.get("floor_z_m", 0.0)),
            ceiling_height_m=float(payload.get("ceiling_height_m", 0.0)),
            floor_material=str(payload.get("floor_material", "")),
            wall_material=str(payload.get("wall_material", "")),
        )


@dataclass(frozen=True, slots=True)
class PlanOpening:
    """A hole in a wall, described as the wall segment it may sit in.

    ``wall_segment_xy`` is the load-bearing field: it is the one thing every provider can honour,
    and it is also Infinigen's own representation (a ``LineString`` along the wall). The three
    dimension fields beside it are the *request*, and R7 keeps them honest against the segment
    rather than against the built room.
    """

    opening_id: str
    kind: str
    room_id: str
    wall_segment_xy: tuple[tuple[float, float], tuple[float, float]]
    width_m: float
    height_m: float
    sill_z_m: float = 0.0

    def __post_init__(self) -> None:
        if not self.opening_id:
            raise RoomPlanError("PLAN_NOT_SERIALISABLE(opening_id): an opening needs an id")
        if self.kind not in (OpeningKind.DOOR.value, OpeningKind.WINDOW.value):
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE({self.opening_id}): kind must be 'door' or 'window', "
                f"got {self.kind!r}"
            )
        if self.width_m <= 0 or self.height_m <= 0:
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE({self.opening_id}): width and height must be positive"
            )
        if self.sill_z_m < 0:
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE({self.opening_id}): sill height must not be negative"
            )

    @property
    def segment_length_m(self) -> float:
        return _segment_length(*self.wall_segment_xy)

    @property
    def centre_m(self) -> tuple[float, float]:
        """The request path's only conversion: the segment midpoint becomes the opening centre."""
        return _midpoint(*self.wall_segment_xy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opening_id": self.opening_id,
            "kind": self.kind,
            "room_id": self.room_id,
            "wall_segment_xy": [list(point) for point in self.wall_segment_xy],
            "width_m": self.width_m,
            "height_m": self.height_m,
            "sill_z_m": self.sill_z_m,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanOpening":
        segment = _points(payload.get("wall_segment_xy"), what="openings.wall_segment_xy")
        if len(segment) != 2:
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE(wall_segment_xy): a wall segment is two points, "
                f"got {len(segment)}"
            )
        return cls(
            opening_id=str(payload.get("opening_id", "")),
            kind=str(payload.get("kind", "")),
            room_id=str(payload.get("room_id", "")),
            wall_segment_xy=(segment[0], segment[1]),
            width_m=float(payload.get("width_m", 0.0)),
            height_m=float(payload.get("height_m", 0.0)),
            sill_z_m=float(payload.get("sill_z_m", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class PlanCore:
    """The frozen cell, as evidence rather than as a copy.

    ``protected_polygon_xy`` is in the **cell frame**; apply ``translation_xy`` to get the region
    in room coordinates. Keeping the translation separate is what makes "the room solve may only
    move the core, never reshape it" checkable -- everything else in this block must survive the
    room unchanged, and :func:`core_gaps` is how that is proven.
    """

    core_hash: str
    translation_xy: tuple[float, float]
    rotation_yaw_rad: float = 0.0
    protected_polygon_xy: tuple[tuple[float, float], ...] = ()
    support_top_z_m: float = 0.0
    target_offset_xy: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not is_digest(self.core_hash):
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE(core.core_hash): expected a {64}-character hex digest, "
                f"got {self.core_hash!r}"
            )
        if not _finite(*self.translation_xy, self.rotation_yaw_rad, self.support_top_z_m,
                       *self.target_offset_xy):
            raise RoomPlanError("PLAN_NOT_SERIALISABLE(core): non-finite value")

    @property
    def protected_in_room_xy(self) -> tuple[tuple[float, float], ...]:
        """The protected region once the translation is applied."""
        dx, dy = self.translation_xy
        return tuple((x + dx, y + dy) for x, y in self.protected_polygon_xy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_hash": self.core_hash,
            "translation_xy": list(self.translation_xy),
            "rotation_yaw_rad": self.rotation_yaw_rad,
            "protected_polygon_xy": [list(vertex) for vertex in self.protected_polygon_xy],
            "support_top_z_m": self.support_top_z_m,
            "target_offset_xy": list(self.target_offset_xy),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCore":
        return cls(
            core_hash=str(payload.get("core_hash", "")),
            translation_xy=_pair(payload.get("translation_xy"), what="core.translation_xy"),
            rotation_yaw_rad=float(payload.get("rotation_yaw_rad", 0.0)),
            protected_polygon_xy=_points(
                payload.get("protected_polygon_xy"), what="core.protected_polygon_xy"
            ),
            support_top_z_m=float(payload.get("support_top_z_m", 0.0)),
            target_offset_xy=_pair(payload.get("target_offset_xy"), what="core.target_offset_xy"),
        )


def _pair(raw: Any, *, what: str) -> tuple[float, float]:
    """Read ``[x, y]`` into a float pair. A missing value is refused, not defaulted to zero."""
    return _points([raw], what=what)[0]


@dataclass(frozen=True, slots=True)
class PlanProvenance:
    """Where this plan came from. A plan that cannot answer this is not a fixture, it is a guess."""

    spec_hash: str
    seed: int
    shell_options_digest: str = ""
    room_intent: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.room_intent:
            from rfw_w.scenes.contracts.scene_spec import RoomSpec, SpecError
            try:
                RoomSpec.from_dict(self.room_intent)
            except (SpecError, TypeError) as error:
                raise RoomPlanError(f"PLAN_NOT_SERIALISABLE(room_intent): {error}") from error
        if not is_digest(self.spec_hash):
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE(provenance.spec_hash): expected a 64-character hex digest, "
                f"got {self.spec_hash!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_hash": self.spec_hash,
            "seed": self.seed,
            "shell_options_digest": self.shell_options_digest,
            **({"room_intent": dict(self.room_intent)} if self.room_intent else {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanProvenance":
        return cls(
            spec_hash=str(payload.get("spec_hash", "")),
            seed=int(payload.get("seed", 0)),
            shell_options_digest=str(payload.get("shell_options_digest", "")),
            room_intent=dict(payload.get("room_intent", {})),
        )


@dataclass(frozen=True, slots=True)
class RoomPlan:
    """A frozen, hashed floor plan.

    ``hash`` covers every field *except itself* -- a hash that included its own value could never
    be computed. Reading a plan back recomputes it, so a plan edited on disk is detected rather
    than silently accepted under its old identity.

    ``units`` and ``up_axis`` have **no defaults**. A plan that could omit them would be relying on
    the convention that this repository is metric and Z-up, which is exactly the inference R1
    forbids -- an external plan in feet would then be read as metres without anyone deciding that.
    """

    plan_id: str
    source: str
    rooms: tuple[PlanRoom, ...]
    core: PlanCore
    provenance: PlanProvenance
    units: str
    up_axis: str
    openings: tuple[PlanOpening, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.units != UNITS:
            raise RoomPlanError(
                f"PLAN_UNITS_UNDECLARED(units): a plan is metric; got {self.units!r}"
            )
        if self.up_axis != UP_AXIS:
            raise RoomPlanError(
                f"PLAN_UNITS_UNDECLARED(up_axis): a plan is Z-up; got {self.up_axis!r}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE(schema_version): expected {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
        if self.source not in SOURCES:
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE(source): expected one of {SOURCES}, got {self.source!r}"
            )
        if not self.rooms:
            raise RoomPlanError("PLAN_NOT_SERIALISABLE(rooms): a plan needs at least one room")
        room_ids = {room.room_id for room in self.rooms}
        for opening in self.openings:
            if opening.room_id not in room_ids:
                raise RoomPlanError(
                    f"PLAN_NOT_SERIALISABLE({opening.opening_id}): room_id "
                    f"{opening.room_id!r} is not in this plan"
                )
        ids = [opening.opening_id for opening in self.openings]
        if len(set(ids)) != len(ids):
            raise RoomPlanError("PLAN_NOT_SERIALISABLE(openings): duplicate opening ids")

    @property
    def content_hash(self) -> str:
        """Identity of everything this plan says, excluding ``hash`` itself."""
        return content_hash(self.payload())

    def payload(self) -> dict[str, Any]:
        """Everything the hash covers."""
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source": self.source,
            "units": self.units,
            "up_axis": self.up_axis,
            "rooms": [room.to_dict() for room in self.rooms],
            "openings": [opening.to_dict() for opening in self.openings],
            "core": self.core.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["hash"] = self.content_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoomPlan":
        plan = cls(
            plan_id=str(payload.get("plan_id", "")),
            source=str(payload.get("source", "")),
            rooms=tuple(PlanRoom.from_dict(item) for item in payload.get("rooms") or ()),
            openings=tuple(PlanOpening.from_dict(item) for item in payload.get("openings") or ()),
            core=PlanCore.from_dict(payload.get("core") or {}),
            provenance=PlanProvenance.from_dict(payload.get("provenance") or {}),
            # ``get`` with an empty default rather than a missing key: an undeclared unit has to
            # surface as PLAN_UNITS_UNDECLARED, not as a TypeError from the constructor.
            units=payload.get("units", ""),
            up_axis=payload.get("up_axis", ""),
            schema_version=payload.get("schema_version", ""),
        )
        declared = payload.get("hash")
        if declared is not None and declared != plan.content_hash:
            raise RoomPlanError(
                f"PLAN_NOT_SERIALISABLE(hash): the plan says {declared!r} but its content hashes to "
                f"{plan.content_hash!r}; the plan was edited after it was frozen"
            )
        return plan


def check_plan(plan: RoomPlan) -> tuple[str, ...]:
    """Named gaps in a plan's internal consistency. Empty means self-consistent.

    R7 is two claims, and it is worth keeping them apart:

    * the two representations of an opening's width agree -- the segment the plan names is as long
      as the width the plan asks for;
    * the segment lies **on a wall the plan declares**.

    The second was implicit while the only plans in existence came from the drawer, which derives
    the segment from a wall it already knows. It stops being implicit the moment a plan can arrive
    from anywhere else (`RoomPlan.source` is ``core`` / ``external`` / ``bank``), because the
    serialisation has to clamp a window's host span to that wall and cannot clamp to a wall it
    cannot find.

    Deliberately **not** checked here: anything against built geometry. That is
    ``shell.matches_plan``'s job, and a plan that failed because a builder chose its own door width
    would be blaming the wrong document (SPEC-001 §5).
    """
    gaps: list[str] = []
    for opening in plan.openings:
        length = opening.segment_length_m
        if abs(opening.width_m - length) > SEGMENT_LENGTH_TOLERANCE_M:
            gaps.append(
                f"PLAN_OPENING_WIDTH_DISAGREES_WITH_SEGMENT({opening.opening_id}, "
                f"{opening.width_m:.6f}, {length:.6f})"
            )
        room = next((room for room in plan.rooms if room.room_id == opening.room_id), None)
        if room is not None and _wall_containing(room.polygon_xy, opening.wall_segment_xy) is None:
            gaps.append(
                f"PLAN_OPENING_NOT_ON_A_WALL({opening.opening_id}, {opening.wall_segment_xy}): the "
                "opening is not on any wall of the room it belongs to"
            )
    return tuple(gaps)


def core_gaps(before: RoomPlan, after: RoomPlan) -> tuple[str, ...]:
    """Named gaps where the frozen core moved between two plans (R5).

    Only ``translation_xy`` and ``rotation_yaw_rad`` may differ. Both are permitted because they
    are the room solve's one degree of freedom; everything else is the task's own conclusion and
    must survive the room untouched, byte for byte.
    """
    gaps: list[str] = []
    fields = ("core_hash", "protected_polygon_xy", "support_top_z_m", "target_offset_xy")
    for name in fields:
        left, right = getattr(before.core, name), getattr(after.core, name)
        if left != right:
            gaps.append(f"PLAN_CORE_MUTATED(core.{name}, {left!r}, {right!r})")
    return tuple(gaps)


def core_identity(plan: RoomPlan) -> str:
    """A digest of the core fields that must not move. Pinned in tests as the frozen value."""
    return content_hash(
        {
            name: getattr(plan.core, name)
            for name in ("core_hash", "protected_polygon_xy", "support_top_z_m", "target_offset_xy")
        }
    )


def plan_to_shell(plan: RoomPlan) -> RoomShell:
    """The request path: a plan becomes the shell that will be asked for.

    The only conversion allowed in this direction is the segment midpoint becoming the opening
    centre, and the result is a *request*, not a report of anything built. A provider that
    measures what it built produces its own ``RoomShell`` -- see SPEC-003.
    """
    if len(plan.rooms) != 1:
        raise RoomPlanError(
            f"PLAN_NOT_SERIALISABLE(rooms): shell conversion is single-room, got {len(plan.rooms)}"
        )
    room = plan.rooms[0]
    openings = tuple(
        Opening(
            opening_id=opening.opening_id,
            kind=OpeningKind(opening.kind),
            center_xy=opening.centre_m,
            width_m=opening.width_m,
            height_m=opening.height_m,
            sill_z_m=opening.sill_z_m,
        )
        for opening in plan.openings
        if opening.room_id == room.room_id
    )
    return RoomShell(
        polygon_xy=room.polygon_xy,
        holes_xy=(),
        height_m=room.ceiling_height_m,
        floor_z_m=room.floor_z_m,
        openings=openings,
        source=f"{PLAN_SHELL_SOURCE}:{plan.source}",
    )


def shape_literal(vertices: Sequence[tuple[float, float]]) -> str:
    """A polygon as the string form Infinigen's predefined floor plan loader expects.

    ``room/predefined.py:141`` runs ``eval`` on any shape string it finds, which is why this is the
    only place such a string may be built and why it takes a sequence of numbers rather than text.
    Nothing external is ever interpolated into it.
    """
    if len(vertices) < 3:
        raise RoomPlanError(
            f"PLAN_NOT_SERIALISABLE(polygon): a shape literal needs at least 3 vertices, "
            f"got {len(vertices)}"
        )
    for index, (x, y) in enumerate(vertices):
        if not _finite(float(x), float(y)):
            raise RoomPlanError(f"PLAN_NOT_SERIALISABLE(polygon[{index}]): non-finite coordinate")
    body = ",".join(f"({float(x)!r},{float(y)!r})" for x, y in vertices)
    return f"shapely.Polygon([{body}])"


def segment_literal(segment: Sequence[tuple[float, float]]) -> str:
    """A wall segment as a ``LineString`` literal. Doors and windows are segments in the plan."""
    if len(segment) != 2:
        raise RoomPlanError(
            f"PLAN_NOT_SERIALISABLE(segment): a wall segment is two points, got {len(segment)}"
        )
    for index, (x, y) in enumerate(segment):
        if not _finite(float(x), float(y)):
            raise RoomPlanError(f"PLAN_NOT_SERIALISABLE(segment[{index}]): non-finite coordinate")
    body = ",".join(f"({float(x)!r},{float(y)!r})" for x, y in segment)
    return f"shapely.LineString([{body}])"


def floor_plan_payload(plan: RoomPlan) -> dict[str, dict[str, dict[str, Any]]]:
    """The plan in Infinigen's predefined-floor-plan shape.

    Keys are Infinigen's own vocabulary -- ``rooms`` / ``doors`` / ``windows`` / ``opens`` /
    ``interiors`` -- and the room key has to be ``"<roomtype>_<storey>/<idx>"`` because
    ``predefined.py`` parses the room type and storey back out of it. ``opens`` and ``interiors``
    are emitted empty: this SPEC is single-room, so there is no interior wall for either to sit in.

    An opening's shape is its **host span** (`opening_host_span`), not its `wall_segment_xy`. For a
    window the two differ, and this is the whole reason: handed the plan's own 0.926 m segment,
    Infinigen builds no window at all and says nothing about it. The span it is handed here is the
    same span the gate later holds the measured hole to, so "the plan promised a window on that
    wall" is a claim both ends can be checked against.

    Producing this is the plan contract's job; deciding *when* to hand it to a generator is
    SPEC-004's.
    """
    rooms: dict[str, dict[str, Any]] = {}
    for room in plan.rooms:
        rooms[f"{room.kind}_0/0"] = {"shape": shape_literal(room.polygon_xy)}
    doors: dict[str, dict[str, Any]] = {}
    windows: dict[str, dict[str, Any]] = {}
    for opening in plan.openings:
        target = doors if opening.kind == OpeningKind.DOOR.value else windows
        entry: dict[str, Any] = {"shape": segment_literal(opening_host_span(plan, opening))}
        if opening.kind == OpeningKind.WINDOW.value:
            # Present because the loader reads it; we have no opinion on panoramic windows yet,
            # and 0 is the non-panoramic branch.
            entry["is_panoramic"] = 0
        target[opening.opening_id] = entry
    return {"rooms": rooms, "doors": doors, "interiors": {}, "opens": {}, "windows": windows}
