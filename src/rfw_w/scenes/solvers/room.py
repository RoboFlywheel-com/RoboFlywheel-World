# SPDX-License-Identifier: Apache-2.0
"""The procedural room shell: an exact outline, not a bounding box.

This is the low-latency default backend from architecture §7: a deterministic polygon with
walls, openings and a floor, generated from the spec and the seed. Furniture still comes from
real retrieval; only the shell is synthesised.

The reason the outline is an explicit polygon is the reason the acceptance manual gives: an
L-shaped room or a notched wall cannot be described by a rectangle, and a rectangle stand-in
passes containment checks the real room would fail. So the shape is generated as vertices,
its area is checked against the requested size bounds *as a polygon*, and a door is validated
against the wall it actually sits in rather than against the room's extent.

A door is not decoration. Whether it admits the robot is a functional question, so its width
is checked against the robot's footprint plus the configured margin, and a door that is too
narrow is reported as a named gap rather than allowed to fail later at the navigation gate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace as dataclass_replace
from enum import Enum
from typing import Any, Mapping, Sequence

from rfw_w.scenes.contracts.scene_spec import Range1D, SceneSpec
from rfw_w.scenes.contracts.state import Opening, OpeningKind, RoomShell
from rfw_w.scenes.contracts.units import UnitsError
from rfw_w.scenes.geometry.clearance import DEFAULT_MARGIN_M
from rfw_w.scenes.geometry.polygon import (
    Polygon,
    PolygonError,
    contains_polygon,
    distance_to_boundary,
    inset,
    overlaps,
    validate_simple,
)

SHELL_SOURCE = "procedural_shell"

# How far a notch must stay from the far end of the dimension it cuts into. A notch running the
# whole width would leave two disconnected rooms, which is a floor plan and not a shell.
#
# `MIN_WALL_BESIDE_OPENING_M = 0.3` used to sit above this with the same value as
# `DOOR_WALL_MARGIN_M`, described as "wall that must remain either side of an opening". It was read
# by nothing -- the door uses `DOOR_WALL_MARGIN_M` and the window uses `WINDOW_EDGE_INSET_M`, and
# both were already there. A constant that describes a rule the code does not follow is worse than
# no constant, because the next reader takes it for the rule.
MIN_MATERIAL_BESIDE_NOTCH_M = 0.05

# The narrowest a room may be across its neck -- the middle a T keeps, or the stub a U leaves.
#
# Not a taste threshold: it is what the *rest of the pipeline* can still plan. The zone planner
# insets the room by up to 0.75 m from every side to find the centre zone, and an inset deeper
# than half a neck collapses the offset onto itself -- measured on a room whose middle is 1.079 m
# wide, inset by 0.75 m, the result is a polygon with two collinear edges overlapping along the
# neck line. That polygon is not simple, and it is not caught until much later inside
# triangulation.
#
# 1.8, and the number this replaced was wrong in the direction that matters. It used to be 1.5,
# with a docstring claiming that was "twice that margin, so a neck this wide always has a centre".
# Twice the margin is exactly the width at which the inset leaves *nothing*: 2 x 0.75 = 1.5, so a
# 1.5 m neck insets to zero width and raises. Verified directly -- a T with a 1.5 m neck raises
# `PolygonError: zero-length edge` at inset 0.75 and succeeds at 0.375, and 5 of 16 U-shaped
# office rooms were quietly reading `ambient_inset_m = 0.375` instead of 0.75 because
# `_edge_notch`'s clamp had left flanks of exactly this width. The planner's walk-down hides it,
# so the cost was a silent centre-zone loss and one number nobody compared to anything.
MIN_ROOM_NECK_M = 1.8

# The smallest extent, in metres, that a notch may be drawn at.
#
# Below this a bite is not a small notch, it is a rounding artefact: `_clamp` rounds to three
# places and `Polygon.from_sequence` drops near-collinear vertices, so the replacement vertices
# collapse, the outline comes back a rectangle, and `validate_shell` reports
# OUTLINE_DOES_NOT_REALISE_THE_REQUESTED_SHAPE -- a named failure that blames the bounding-box
# reduction for a notch that was merely too small to see.
MIN_NOTCH_EXTENT_M = 0.30

# How much of a dimension a single notch may take, as a band the drawer picks from.
#
# It used to be the point 0.25, and that is the whole of why every notched room was the same room:
# the bite was a fixed fraction of the axes it was cut into, so an L was fully determined by its
# width, its depth and which corner was missing. Measured over 20 seeds an experiment, the bite's
# share of the bounding box took exactly two values across all 60 shells -- nothing, and 0.0625 --
# and the vocabulary's only real degree of freedom was the corner.
#
# Modest on purpose, at both ends. A room that is mostly its bounding box is what the downstream --
# strips along the walls, a centre, a corridor -- was written against, and the shape is meant to
# vary within that, not to replace it; so 0.34 leaves an L's arms at 0.66 of each dimension, well
# clear of the neck floor. And 0.18 is the floor of `_notch_range`'s own `share`, so the band and
# the predicate that judges the result cannot disagree about what the smallest legal bite is.
NOTCH_FRACTION_BAND = (0.18, 0.34)

# The smallest *share* of a dimension a notch may take, as a floor under the drawn extent.
#
# Doing double duty, deliberately. It keeps a bite legible -- an extent that is a small fraction
# of the wall it is cut into reads as a nick rather than as a room shape -- and it is what bounds
# the neck from below on a T without a second constant: with each of two bites at least this
# share, the neck they leave is at most `1 - 2 * NOTCH_SHARE = 0.64` of the edge, so "a neck wider
# than three quarters of the wall is not a T" holds by construction rather than by a separate cap
# that would never fire.
NOTCH_SHARE = 0.18

# Openings. The wall a door or window sits in is divided into thirds and one is drawn, so an
# opening lands where a person would put it rather than always at the midpoint -- which is what
# the shell did while there was exactly one window and it was centred on the longest wall.
SEGMENT_BOUNDARIES = (0.0, 1 / 3, 2 / 3, 1.0)
# Clearance from the end of a wall, for a door. A door at a corner is not a doorway.
DOOR_WALL_MARGIN_M = 0.3
# The same for a window, and smaller: a window may sit closer to a corner than a door.
WINDOW_EDGE_INSET_M = 0.1
# Openings on one wall must not touch. Two holes a hand's width apart are one hole.
MIN_OPENING_SEPARATION_M = 0.5
# A window wider than this is a wall of glass, so a wall longer than this takes more than one and
# the span is divided. The value is SceneSmith's `max_window_length` lower bound; the upper one
# produced a single full-height panel here, which is a curtain wall.
#
# `MAX_WINDOWS_PER_WALL = 3` used to sit below this. It was read by nothing, and it contradicted
# the code next to it: `_windows_on_wall` caps a wall at two (`fractions = (0.25, 0.75) if pairs
# else (None,)`) and says so in its own docstring, because "past that the wall is a curtain wall".
MAX_WINDOW_SPAN_M = 2.4


class RoomError(ValueError):
    """The room could not be generated or is not a valid shell."""


class Family(str, Enum):
    """The outline's topology, named rather than described by booleans.

    A room is not "notched or not": it is a rectangle, an L, a T, or an alcove. Naming the family
    is what lets a failure read as "the outline was asked to be an L and came out a rectangle" --
    the bounding-box reduction this module exists to catch -- instead of a bare vertex count.

    The four are the shapes a single rectangular bite or two can make:

    ``RECT``  no bite.
    ``L``     one corner removed.
    ``T``     both corners of one edge removed, so that edge keeps only its middle.
    ``U``     a rectangular recess in the middle of one edge, corners intact -- an alcove.
    """

    RECT = "rect"
    L = "l"
    T = "t"
    U = "u"

    @property
    def expected_vertices(self) -> int:
        """How many outline vertices this family produces, for the realisation check."""
        return {"rect": 4, "l": 6, "t": 8, "u": 8}[self.value]


class Edge(str, Enum):
    """A side of the rectangle, in counter-clockwise order from the bottom."""

    BOTTOM = "bottom"
    RIGHT = "right"
    TOP = "top"
    LEFT = "left"


class Corner(str, Enum):
    """A corner of the rectangle, in counter-clockwise order from the bottom-left."""

    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP_RIGHT = "top_right"
    TOP_LEFT = "top_left"


@dataclass(frozen=True, slots=True)
class Notch:
    """A rectangular bite out of the outline.

    Two kinds, because a corner and an edge make different rooms and the distinction is not
    expressible as one of the other: a corner notch (``corner`` set) removes a corner, and an edge
    notch (``edge`` set) leaves the corners and removes a span from the middle of a side. The
    second is the only way to make an alcove -- ``U`` -- out of a rectangle; corner notches alone
    cannot, because they can only ever leave an edge's middle.
    """

    width_m: float
    depth_m: float
    corner: Corner | None = None
    edge: Edge | None = None

    def __post_init__(self) -> None:
        if (self.corner is None) == (self.edge is None):
            raise RoomError(
                "a notch is either a corner notch or an edge notch: exactly one of `corner` and "
                f"`edge` must be set, got corner={self.corner} edge={self.edge}"
            )
        if self.width_m <= 0 or self.depth_m <= 0:
            raise RoomError(
                f"notch must have positive extent, got {self.width_m} x {self.depth_m}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_m": self.width_m,
            "depth_m": self.depth_m,
            "corner": None if self.corner is None else self.corner.value,
            "edge": None if self.edge is None else self.edge.value,
        }


@dataclass(frozen=True, slots=True)
class OpeningPlan:
    """A resolved opening: which wall it is in, how far along that wall, and how big.

    Resolved here rather than at build time so the shell can be reviewed as a value. The wall index
    and the offset along it are what a reader needs to see where an opening went -- a centre in
    world coordinates does not say which wall it belongs to, and a wall is what the worker segments
    around it.
    """

    opening_id: str
    kind: str
    wall_index: int
    offset_along_m: float
    width_m: float
    height_m: float
    sill_m: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in (OpeningKind.DOOR.value, OpeningKind.WINDOW.value):
            raise RoomError(f"opening {self.opening_id}: unknown kind {self.kind!r}")
        if self.wall_index < 0:
            raise RoomError(f"opening {self.opening_id}: wall index must not be negative")
        if self.width_m <= 0 or self.height_m <= 0:
            raise RoomError(f"opening {self.opening_id}: width and height must be positive")
        if self.sill_m < 0:
            raise RoomError(f"opening {self.opening_id}: sill must not be negative")
        if self.sill_m > 0 and self.kind == OpeningKind.DOOR.value:
            raise RoomError(f"opening {self.opening_id}: a door has no sill")

    @property
    def span_m(self) -> tuple[float, float]:
        """Where the opening starts and ends along its wall."""
        half = self.width_m / 2
        return (self.offset_along_m - half, self.offset_along_m + half)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opening_id": self.opening_id,
            "kind": self.kind,
            "wall_index": self.wall_index,
            "offset_along_m": round(self.offset_along_m, 6),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "sill_m": self.sill_m,
        }


@dataclass(frozen=True, slots=True)
class ShellOptions:
    """The resolved shell program: everything that shapes the room, already decided.

    ``generate_options`` is the only function that draws random numbers, and this dataclass is
    what it hands back. ``build_shell`` is then a pure function of a spec and one of these, which
    is what makes "the same seed gives the same room" checkable by reading a value rather than
    hoped for by auditing a call graph.

    Anything that changes the outline is an explicit field here rather than a hidden draw. That
    promise used to be in this docstring without being true -- ``build_shell`` opened a second
    ``random.Random(spec.seed)`` of its own and took the height from it -- so a new draw added
    anywhere would silently renumber the others. There is one stream, and it is here.
    """

    family: Family = Family.RECT
    notches: tuple[Notch, ...] = ()
    width_m: float | None = None
    depth_m: float | None = None
    height_m: float = 2.8
    door_width_m: float = 0.9
    door_height_m: float = 2.05
    door_center: tuple[float, float] | None = None
    window: bool = True
    window_width_m: float = 1.2
    window_height_m: float = 1.2
    window_sill_m: float = 0.9
    # The shell policy's floor on windows. Zero means "as many as the room affords",
    # which is what every room did before there was anywhere to say otherwise.
    min_windows: int = 0
    # The resolved openings, in wall order. Empty means "derive them", which is what every caller
    # that does not care about the layout leaves it as; `generate_options` fills it.
    openings: tuple[OpeningPlan, ...] = ()
    wall_thickness_m: float = 0.1
    robot_footprint_m: float = 0.6
    clearance_margin_m: float = DEFAULT_MARGIN_M

    def __post_init__(self) -> None:
        if (self.width_m is None) != (self.depth_m is None):
            raise RoomError("explicit shell width and depth must be supplied together")
        if self.width_m is not None and (self.width_m <= 0 or self.depth_m <= 0):
            raise RoomError("explicit shell width and depth must be positive")
        if self.door_width_m <= 0:
            raise RoomError(f"door width must be positive, got {self.door_width_m}")
        if self.door_height_m <= 0:
            raise RoomError(f"door height must be positive, got {self.door_height_m}")
        if self.wall_thickness_m < 0:
            raise RoomError("wall thickness must not be negative")
        if self.height_m <= 0:
            raise RoomError(f"room height must be positive, got {self.height_m}")
        # The door width is both a field and a fact about the resolved plan, and the two must agree. They
# silently did not: `replace(options, door_width_m=0.5)` left a plan whose door was still 0.9 m
# wide, `build_shell` honoured the plan, and the override did nothing -- which is how
# `test_a_door_narrower_than_the_robot_fails` stopped failing. Windows carry no such check because
# each one's width is drawn separately and `window_width_m` is only the drawer's starting point.
        doors = [item for item in self.openings if item.kind == OpeningKind.DOOR.value]
        if doors and abs(doors[0].width_m - self.door_width_m) > 1e-9:
            raise RoomError(
                f"DOOR_WIDTH_CONFLICTS_WITH_THE_RESOLVED_PLAN(field={self.door_width_m}, "
                f"plan={doors[0].width_m}); use `with_door_width` so both move together"
            )
        if self.family is Family.RECT and self.notches:
            raise RoomError(
                f"family is rect but {len(self.notches)} notch(es) were requested; a rectangle "
                "with a bite out of it is an L, a T or a U, and saying rect would make the "
                "realisation check pass on the wrong shape"
            )
        if self.family is not Family.RECT and not self.notches:
            raise RoomError(
                f"family is {self.family.value} but no notch was requested; the outline would be "
                "a rectangle and the shape would have been reduced to its bounding box"
            )

    def with_door_width(self, width_m: float) -> ShellOptions:
        """The same room with a different door, applied to the resolved plan and the field together.

        The field is what the *drawer* used and the plan is what the shell is built from, so
        changing one alone is a change that does not happen. This exists so a caller who wants a
        narrow door to exercise the passage gate does not have to know that.
        """
        return dataclass_replace(
            self,
            door_width_m=width_m,
            openings=tuple(
                dataclass_replace(item, width_m=width_m)
                if item.kind == OpeningKind.DOOR.value else item
                for item in self.openings
            ),
        )

    @property
    def notch_size_m(self) -> tuple[float, float] | None:
        """The single corner notch's size, or ``None``.

        A read-only view for callers that only ever dealt with one notch -- the room-only viewer's
        ``--no-notch`` and the older tests. The field is gone because "notch" stopped meaning one
        rectangle; this keeps those callers reading something rather than reaching into a field
        that no longer describes the shape.
        """
        corner_notches = [item for item in self.notches if item.corner is not None]
        if len(self.notches) != 1 or len(corner_notches) != 1:
            return None
        return (self.notches[0].width_m, self.notches[0].depth_m)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "family": self.family.value,
            "notches": [notch.to_dict() for notch in self.notches],
            "height_m": self.height_m,
            "door_width_m": self.door_width_m,
            "door_height_m": self.door_height_m,
            "door_center": None if self.door_center is None else list(self.door_center),
            "window": self.window,
            "window_width_m": self.window_width_m,
            "window_height_m": self.window_height_m,
            "window_sill_m": self.window_sill_m,
            "min_windows": self.min_windows,
            "openings": [opening.to_dict() for opening in self.openings],
            "wall_thickness_m": self.wall_thickness_m,
            "robot_footprint_m": self.robot_footprint_m,
            "clearance_margin_m": self.clearance_margin_m,
        }
        if self.width_m is not None:
            payload["width_m"] = self.width_m
            payload["depth_m"] = self.depth_m
        return payload


def _pick(rng: random.Random, bounds: Range1D) -> float:
    """Draw uniformly within an inclusive range, consuming one draw whatever the range's width.

    The draw is taken even when the range is degenerate -- ``uniform(a, a)`` returns ``a`` -- because
    the *number* of draws is what keeps a stream aligned. Returning early on a zero-width range, as
    this did, skipped a draw: with ``need_w == x_bounds.maximum`` the width's slot consumed nothing
    and the height draw slid into it, so one seed in a blue moon produced a room carrying another
    room's height. The value of the degenerate draw is unchanged, so nothing that is not degenerate
    moves.
    """
    return rng.uniform(bounds.minimum, bounds.maximum)


@dataclass(frozen=True, slots=True)
class RoomProfile:
    """The shape families and opening floor the shell solver supports.

    Grounded in SceneSmith's designer guidance, which states preferred footprints per room type
    ("kitchen: 3-4 m x 3-4 m", "primary bedroom: 4-5 m x 3.5-4 m") and hands them to a model as
    prose. Here they are code, and they **modulate within the spec's bounds rather than override
    them**: ``validate_shell`` requires the footprint to lie inside the requested bounds, so a
    profile that disagreed with the spec would be rejected by the shell's own validator. The spec
    is the size envelope; the solver chooses a shape inside it.

    A separate size preference is deliberately not here yet. Biasing the width and depth draws would
    move every existing room's size in the same change that moves its shape, and the two would
    then be impossible to review apart -- and size already varies by seed within the spec's
    bounds. It is the next step, not a field that quietly does nothing.
    """

    families: tuple[Family, ...]
    min_windows: int = 0


# The alcove is published; the T is not. Both are built, validated and reachable, and the reasoning
# for each is a measurement rather than a preference.
#
# **U is published because the measurement that withheld it no longer holds.** It was withheld
# because an alcove's flank leaves the work strip too short for the task's own support: the strip
# has a fixed 1.25 m depth (`zones.WALL_DEPTH_M`) and a length taken from the wall it runs along,
# and at the time an alcove left a 1.25 m strip against a 1.370 m desk, so the build failed outright
# (`'desk' could not be placed, so nothing can support the task target`) in 9 of 900 rooms. Two
# things have moved since. `MIN_ROOM_NECK_M` went from 1.5 to 1.8, which is the width a flank is
# held to, and the core-first build no longer places the support through the zone planner at all --
# the support stands where the cell solved it. Re-measured over 40 seeds of each experiment, the
# longest work strip is 2.125 m on an alcove against the desk's 1.370 m, and the sweep finds no
# refusal for that reason.
#
# Publishing it cost something, and that is recorded rather than argued: with the alcove in the
# vocabulary, 7 of 180 fitted rooms were refused by `plan_zones` -- every one notched, none
# pre-existing. That is what the escalating fit is for, and with it the count is 0.
#
# T is published with explicit periphery loss reporting. The committed capacity tool's 20-seed
# medians (2026-09-20) are office rect 8.77 / T 6.94 m2 placeable, a 21% cost; periphery costs
# 5%. Kitchen costs 25% / 10%, bedroom 20% / 4%. The former 4.85 / 46% claim was not reproduced.
# All four families still use the fit's recorded fallback when the protected cell cannot fit.
#
# The former office/kitchen/bedroom profiles all admitted the same two observable outcomes and
# required one window. Room semantics therefore did not select a distinct shell policy. Keep the
# actual policy in the solver instead of asking SceneSpec to carry a dead kind label.
ROOM_PROFILE = RoomProfile(families=(Family.L, Family.U, Family.T, Family.RECT), min_windows=1)

# Which corners a given edge runs between, in the counter-clockwise order the outline walks.
_CORNERS_OF_EDGE: Mapping[Edge, tuple[Corner, Corner]] = {
    Edge.BOTTOM: (Corner.BOTTOM_LEFT, Corner.BOTTOM_RIGHT),
    Edge.RIGHT: (Corner.BOTTOM_RIGHT, Corner.TOP_RIGHT),
    Edge.TOP: (Corner.TOP_RIGHT, Corner.TOP_LEFT),
    Edge.LEFT: (Corner.TOP_LEFT, Corner.BOTTOM_LEFT),
}

# The corners in the counter-clockwise order the outline walks them.
_CORNERS_CCW = (Corner.BOTTOM_LEFT, Corner.BOTTOM_RIGHT, Corner.TOP_RIGHT, Corner.TOP_LEFT)
# The edge leaving each corner, by the same index.
_EDGES_CCW = (Edge.BOTTOM, Edge.RIGHT, Edge.TOP, Edge.LEFT)


def _clamp(value: float, low: float, high: float) -> float:
    """``value`` held inside ``[low, high]``, which the caller has already shown to be non-empty."""
    if high < low - 1e-9:
        raise RoomError(f"NOTCH_HAS_NO_ROOM_TO_SIT_IN(low={low:.3f}, high={high:.3f})")
    return round(min(max(value, low), high), 3)


def _notch_range(
    dim: float, *, surviving: float, share: float = NOTCH_SHARE, bites: int = 1
) -> tuple[float, float] | None:
    """The admissible extent of a notch on one axis, or ``None`` if the axis cannot take one.

    One function because three callers have to agree exactly and used not to: ``_t_edges`` and
    ``_u_edges`` answer "can this family be built here", ``_draw_notches`` answers "between what
    may I draw", and ``_notch_gaps`` answers "was the outline it produced legal". While those were
    separate expressions of the same predicate they disagreed -- the drawer could be handed an
    empty range and raise over a number, and the flanks of a U could be left sitting exactly on
    the width at which the zone planner's inset collapses.

    ``surviving`` is what the rest of the axis keeps: ``MIN_ROOM_NECK_M`` for an arm, a stem or
    one flank, twice that for the two flanks an alcove leaves. ``bites`` is how many notches share
    this axis -- two for a T's edge, where the drawer picks each independently and so each may
    only claim half the budget.

    The ceiling is **floored** to the drawn resolution, and that is not tidiness. ``_clamp``
    rounds its result to three places, and ``round`` rounds up: a bite bounded by 1.3045 m came out
    of the drawer as 1.305 m, so two of them left a T's neck 0.001 m short of the floor its own
    bound promised -- measured, office seeds 34 and 49, necks of 1.800 and 1.799 against a floor
    of 1.800. The same round-up put a window 0.5 mm past the end of its wall.
    """
    low = max(MIN_NOTCH_EXTENT_M, share * dim)
    high = math.floor(((dim - surviving) / bites) * 1000) / 1000
    return (low, high) if high >= low else None


def _draw_notch_extent(
    rng: random.Random, dim: float, bounds: tuple[float, float] | None, what: str,
    band: tuple[float, float] = NOTCH_FRACTION_BAND,
) -> float:
    """One extent of one notch, drawn from the band and held inside ``bounds``.

    ``bounds`` is ``None`` when the axis cannot take a notch at all -- which the family filter has
    already refused, but a hand-built ``ShellOptions`` can reach the drawer without passing through
    it. Named rather than unpacked into a ``TypeError``: the caller's mistake is that the room is
    too small for the shape, and that is what the message should say.

    ``band`` is a parameter rather than the module constant so the fit can retry with a smaller
    bite without drawing a different room: scaling the band leaves every draw in place and in
    order, so the retry is the same room with a shallower notch.
    """
    if bounds is None:
        raise RoomError(
            f"NOTCH_HAS_NO_ROOM_TO_SIT_IN({what} of {dim:.3f} m): the axis cannot take a notch and "
            "one was asked for"
        )
    low, high = bounds
    return _clamp(dim * rng.uniform(*band), low, high)


# The band a retry uses when the room as drawn leaves the zone planner with nothing it can use.
# Halved, and measured to be enough: of the five seeds an office/kitchen/bedroom sweep refused with
# an alcove published, four plan cleanly at half the bite.
NARROWER_NOTCH_FRACTION_BAND = (NOTCH_FRACTION_BAND[0] / 2, NOTCH_FRACTION_BAND[1] / 2)


def _corner_notch(
    rng: random.Random, corner: Corner, width: float, depth: float, *,
    bounds_x: tuple[float, float] | None, bounds_y: tuple[float, float] | None,
    band: tuple[float, float] = NOTCH_FRACTION_BAND,
) -> Notch:
    """A corner bite of a drawn band of each dimension, held inside the family's budget.

    A range rather than a point, because a bite can be too *small* as well as too large. A T leaves
    a stem, and a stem narrower than ``MIN_ROOM_NECK_M`` is a niche the zone planner cannot inset a
    centre into -- measured on a 4.39 m wide room, a 1.098 m stem, which is what the old fixed
    fraction produced and which the planner then refused. Too large is the older constraint: two
    bites that consume the dimension cut the room in two.

    The two axes are drawn independently, so an L's bite is not forced to be square. That is a
    degree of freedom the fixed fraction never had, and it is the cheapest one in the vocabulary:
    no new family, no new geometry, just a shape that is not a scaled copy of the last one.
    """
    return Notch(
        corner=corner,
        width_m=_draw_notch_extent(rng, width, bounds_x, "width", band),
        depth_m=_draw_notch_extent(rng, depth, bounds_y, "depth", band),
    )


def _edge_notch(
    rng: random.Random, edge: Edge, width: float, depth: float,
    band: tuple[float, float] = NOTCH_FRACTION_BAND,
) -> Notch:
    """A recess in the middle of an edge: the only way a rectangle becomes an alcove.

    Centred rather than offset. An offset recess would need its own fit check on both sides, and
    the diversity this family buys comes from *which* edge is bitten and by how much, not from
    where along the edge the bite sits. Centred is also what keeps the two flanks equal -- and the
    flanks are held to ``MIN_ROOM_NECK_M`` for the same reason a T's stem is.
    """
    if edge in (Edge.BOTTOM, Edge.TOP):
        along, inward = width, depth
    else:
        along, inward = depth, width
    return Notch(
        edge=edge,
        width_m=_draw_notch_extent(
            rng, along, _notch_range(along, surviving=2 * MIN_ROOM_NECK_M), "width", band),
        depth_m=_draw_notch_extent(
            rng, inward, _notch_range(inward, surviving=MIN_ROOM_NECK_M), "depth", band),
    )


def _t_edges(width: float, depth: float) -> tuple[Edge, ...]:
    """The edges a T can be built on: those leaving both a neck and a stem of usable width.

    The neck is what the two bites leave between them; the stem is what is left of the strip they
    cut into. Both are floors rather than ceilings -- a stem 1.1 m wide is a niche, not a room --
    and both are checked, because a T measured only against its neck produced rooms the zone
    planner then refused (bedroom seeds 45 and 182).

    The stem is the one that used to be checked at the wrong end: the drawer's bound floored the
    bite's *depth* at the neck width rather than capping it below the room's own depth, so the
    strip left behind could be 0.05 m -- the minimum material beside a notch -- and still be drawn
    as a T. The range below is the same one the drawer uses, so that cannot happen any more.
    """
    edges: list[Edge] = []
    # Bites on a horizontal edge leave a neck measured in x, and a stem as deep as the bite in y.
    if (_notch_range(width, surviving=MIN_ROOM_NECK_M, bites=2)
            and _notch_range(depth, surviving=MIN_ROOM_NECK_M)):
        edges.extend((Edge.BOTTOM, Edge.TOP))
    if (_notch_range(depth, surviving=MIN_ROOM_NECK_M, bites=2)
            and _notch_range(width, surviving=MIN_ROOM_NECK_M)):
        edges.extend((Edge.RIGHT, Edge.LEFT))
    return tuple(edges)


def _u_edges(width: float, depth: float) -> tuple[Edge, ...]:
    """The edges an alcove can be cut into, with both flanks and the floor behind it wide enough.

    The recess leaves *two* flanks along the edge it is cut into and one floor behind it, so the
    along axis has to find twice the neck and the inward axis once. Getting that wrong is what put
    the flanks at exactly ``MIN_ROOM_NECK_M`` and, therefore, exactly on the width where the zone
    planner's inset collapses.
    """
    edges: list[Edge] = []
    if (_notch_range(width, surviving=2 * MIN_ROOM_NECK_M)
            and _notch_range(depth, surviving=MIN_ROOM_NECK_M)):
        edges.extend((Edge.BOTTOM, Edge.TOP))
    if (_notch_range(depth, surviving=2 * MIN_ROOM_NECK_M)
            and _notch_range(width, surviving=MIN_ROOM_NECK_M)):
        edges.extend((Edge.RIGHT, Edge.LEFT))
    return tuple(edges)


def _edges_for(width: float, depth: float, family: Family) -> tuple[Edge, ...]:
    """The edges a family can be built on -- empty for the ones that need no edge at all."""
    if family is Family.T:
        return _t_edges(width, depth)
    if family is Family.U:
        return _u_edges(width, depth)
    return ()


def _families_that_fit(
    profile: RoomProfile, width: float, depth: float
) -> tuple[Family, ...]:
    """The profile's families this room can actually be built as, in the profile's order.

    A narrow room cannot hold a T, an alcove, or even an L, and drawing one anyway would fail in
    the notch drawer over an empty range -- a crash whose message would be about a number rather
    than about the room being too small for the shape. Filtering first makes the choice a choice.

    An L is filtered on the same predicate as the other two. It was not, on the reasoning that a
    corner bite always fits; but a bite takes a neck's width out of *both* axes, so a room narrower
    than roughly 2.4 m cannot hold one, and the drawer would have raised
    ``NOTCH_HAS_NO_ROOM_TO_SIT_IN`` -- the number-shaped crash this filter exists to prevent. No
    shipped experiment is near that width, which is why the gap was never reached.
    """
    if not profile.families:
        raise RoomError("a room profile with no families cannot produce a shell")

    def fits(family: Family) -> bool:
        if family is Family.RECT:
            return True
        if family is Family.L:
            # An L spends a neck's width on both axes: one arm each way.
            return bool(
                _notch_range(width, surviving=MIN_ROOM_NECK_M)
                and _notch_range(depth, surviving=MIN_ROOM_NECK_M)
            )
        return bool(_edges_for(width, depth, family))

    usable = tuple(family for family in profile.families if fits(family))
    # A rectangle is always buildable, so a room is never left without a shape at all.
    return usable or (Family.RECT,)


def _choose_family(
    seed: int, rng: random.Random, profile: RoomProfile, width: float, depth: float
) -> Family:
    """Whether this seed gets a bite, and which family gives it.

    The notched-or-not dice is the one the shell has always used (``seed % 3``), kept rather than
    replaced by a straight draw from the profile. Drawing the family outright was the first thing
    tried, and it changed *which* seeds are notched: that alone moved the bedroom from an L to a
    rectangle and cost it the picture on its wall, and it moved the office onto a room whose desk
    had nowhere to stand. Those two are the same failure -- the stages after the shell are tuned
    against a population of rooms that is two-thirds notched, and swapping which two-thirds is a
    larger change than it looks. So the dice still decides *whether*, and what changed is only
    *which*: the profile lists what is acceptable and the notched ones are drawn from.

    A **draw** rather than an index, and this is the subtle one. `RoomProfile.families` reads like a
    set and is not: the previous version returned `notched[0]`, so everything after the first
    non-rectangle entry was unreachable. Measured with `(T, L, RECT)` it produced 40 T and 20 rect
    over 60 seeds and **zero L** -- so "publishing T and U" by adding them to the tuple would have
    published nothing at all, and the only way to find that out is to measure the population rather
    than read the tuple.

    ``rng`` is a separate substream (`SEED:family`) and not the shape one, because the shape
    stream's first act is to pick a corner: prepending the family draw there would renumber that
    for every seed, and the seeds that stay L would change their corner as well as their bite.
    """
    usable = _families_that_fit(profile, width, depth)
    plain = tuple(family for family in usable if family is Family.RECT)
    notched = tuple(family for family in usable if family is not Family.RECT)
    if seed % 3 == 0:
        return plain[0] if plain else usable[0]
    return rng.choice(notched) if notched else usable[0]


def _draw_notches(
    rng: random.Random, family: Family, width: float, depth: float,
    band: tuple[float, float] = NOTCH_FRACTION_BAND,
) -> tuple[Notch, ...]:
    """The notches a family implies, sized and placed from the shape substream.

    The bounds come from :func:`_notch_range`, the same function ``_*_edges`` filters on and
    :func:`_notch_gaps` judges against. They used to be worked out again here, and the three
    answers disagreed: this one could be handed a range the edge filter had already refused, and
    a T's bite depth was bounded so that the strip it left could be 0.05 m wide.
    """
    if family is Family.RECT:
        return ()
    if family is Family.L:
        # An L's bite leaves no stem, so there is no floor to keep -- only the ceiling that stops
        # the two ends of a dimension being consumed.
        return (
            _corner_notch(rng, rng.choice(_CORNERS_CCW), width, depth,
                          bounds_x=_notch_range(width, surviving=MIN_ROOM_NECK_M),
                          bounds_y=_notch_range(depth, surviving=MIN_ROOM_NECK_M),
                          band=band),
        )
    if family is Family.T:
        edge = rng.choice(_t_edges(width, depth))
        # The two corners of one edge, so that edge keeps only its middle. Both bites draw from the
        # same range, so each may claim only half of what the neck leaves over.
        if edge in (Edge.RIGHT, Edge.LEFT):
            bounds_x = _notch_range(width, surviving=MIN_ROOM_NECK_M)
            bounds_y = _notch_range(depth, surviving=MIN_ROOM_NECK_M, bites=2)
        else:
            bounds_x = _notch_range(width, surviving=MIN_ROOM_NECK_M, bites=2)
            bounds_y = _notch_range(depth, surviving=MIN_ROOM_NECK_M)
        return tuple(
            _corner_notch(rng, corner, width, depth, bounds_x=bounds_x, bounds_y=bounds_y, band=band)
            for corner in _CORNERS_OF_EDGE[edge]
        )
    if family is Family.U:
        return (_edge_notch(rng, rng.choice(_u_edges(width, depth)), width, depth, band),)
    raise RoomError(f"no notch recipe for family {family.value!r}")


def _sample_shell_program(
    spec: SceneSpec,
    *,
    seed: int,
    need_w_m: float | None = None,
    need_d_m: float | None = None,
    notch_band: tuple[float, float] = NOTCH_FRACTION_BAND,
) -> tuple[ShellOptions, float, float]:
    """The shell program one seed draws, with an optional floor need as the size's lower bound.

    Shared by the two room paths so that they cannot drift apart. ``generate_options`` is the
    room-only derivation -- a spec on its own, which is what ``run_room_only.py`` and the frozen
    baseline record -- and ``fit_room_around_cell`` is the build's, where the task cell's protected
    region raises the floor on how small the room may be. Both must draw from the same substreams in
    the same order, or one seed would give two different rooms depending on which path asked.

    The width and depth come back beside the options rather than written into them, because whether
    they are *pinned* is exactly what differs: a room-only shell leaves them ``None`` so that
    ``build_shell`` re-derives them from ``spec.seed``, and a fitted one pins the size it fitted.
    """
    # The three draws that existed before the shape vocabulary did, from the same integer seed they
    # were always taken from. Moving them onto a named substream would have been tidier and would
    # have changed every room's size as a side effect -- which is what happened first, and it
    # silently shrank the office from 14.54 m2 to 13.73 m2 and left its desk with nowhere to stand.
    # Kept here deliberately: a shape change should change the shape.
    size_rng = random.Random(seed)
    x_bounds, y_bounds, z_bounds = spec.room.size_bounds_m
    if need_w_m is not None:
        x_bounds = Range1D(max(x_bounds.minimum, need_w_m), x_bounds.maximum)
    if need_d_m is not None:
        y_bounds = Range1D(max(y_bounds.minimum, need_d_m), y_bounds.maximum)
    # Rounded to the same four places `build_shell` rounds them to, so that both build the *same*
    # outline. They were left unrounded here while `build_shell` rounded, which was harmless while
    # nothing derived from the dimensions -- and is not now that the openings are placed against
    # a wall's length: a span that just fitted the unrounded wall would overrun the rounded one.
    width = round(_pick(size_rng, x_bounds), 4)
    depth = round(_pick(size_rng, y_bounds), 4)
    # The third draw of this stream is what `build_shell` used to take from its own generator, so
    # folding the two into one leaves every seed's height unchanged.
    height = _pick(size_rng, z_bounds)

    # The decisions the shape vocabulary introduced, each on its own substream so that adding
    # another one later cannot renumber the draws above.
    shape_rng = random.Random(f"{seed}:shape")
    family_rng = random.Random(f"{seed}:family")
    profile = ROOM_PROFILE
    family = _choose_family(seed, family_rng, profile, width, depth)
    notches = _draw_notches(shape_rng, family, width, depth, band=notch_band)

    partial = ShellOptions(
        family=family,
        notches=notches,
        height_m=round(height, 4),
        window=z_bounds.maximum > 0,
        min_windows=profile.min_windows,
    )
    return partial, width, depth


def generate_options(spec: SceneSpec) -> ShellOptions:
    """Derive the whole shell program from the spec and its seed.

    The room-only derivation: ``build_shell`` is a pure function of the spec and what this returns,
    and that is what makes "the same seed gives the same room" checkable by reading a value instead
    of auditing a call graph.

    Randomness comes from named substreams rather than one linear stream. A single stream means a
    draw added by a later phase renumbers every draw after it, so a frozen baseline stops meaning
    anything; a named substream can be added to without moving what is already there.
    ``random.Random(str)`` seeds through a digest of the string, so a substream is stable across
    processes -- unlike ``hash()``, which is salted per interpreter.

    The build does not call this: it fits the room around the task cell
    (:func:`fit_room_around_cell`), which draws from the same substreams through
    :func:`_sample_shell_program`. What this still serves is the room-only path -- looking at a
    shell on its own -- and the frozen baseline that pins it.
    """
    partial, width, depth = _sample_shell_program(spec, seed=spec.seed)
    # The door used to be pinned to `(width / 2, 0.0)`, which is only on the boundary while the
    # bottom edge is whole -- and a room may now have a bite taken out of that edge, at which point
    # the pinned centre is a point in the middle of the floor and the build refuses it. Placing it
    # on the longest wall keeps it where it was for a rectangle, whose longest wall is the bottom.
    polygon = _outline(width, depth, partial.family, partial.notches)
    placed = _place_openings(polygon, partial, random.Random(f"{spec.seed}:openings"))
    return dataclass_replace(partial, openings=placed)


@dataclass(frozen=True, slots=True)
class FittedRoom:
    """A shell fitted around a task cell, plus the cell's local-to-room translation."""

    shell: RoomShell
    options: ShellOptions
    cell_translation_xy: tuple[float, float]
    protected_in_room: Polygon
    opening_attempts: int
    # The margin the fit kept between the protected region and the wall's face. Recorded rather than
    # remembered by the caller: it is what turns "the room held the cell" into "the room held the
    # cell with this much to spare", and a manifest that cannot say that cannot show a room that had
    # to grow for the task.
    wall_standoff_m: float = 0.0
    # Which shape was asked for and which one the room got, when they differ. A seed that had to
    # give its shape up must not look like a seed that never asked for one, and this is the only
    # place that distinction can be recorded.
    shape_fallback: str | None = None
    # What the room had left over once the task's protected region and its standoff were
    # accommodated -- `(width - need_w, depth - need_d)`. This is the number the claim that
    # diversity comes from the leftover space rests on, and `wall_standoff_m` is not it: that
    # records the design margin, not the realised one.
    size_margin_m: tuple[float, float] | None = None
    # Where the way in is, from the task's target, and whether that is a bearing the arm can work
    # from. A diagnostic and not a gate -- see `_door_bearing` for why the naive form of this is
    # either vacuous or refuses 40 % of seeds.
    door_bearing_deg: float | None = None
    door_arc_margin_deg: float | None = None
    door_in_approach_arc: bool | None = None


# Why a fitted room is not the shape its seed drew. Both are outcomes of the retry below, and both
# are named rather than described, because a fallback that reads as a shape the seed never asked
# for is a diversity loss nobody can see.
BITE_HALVED = "BITE_HALVED_BECAUSE_THE_ZONE_PLAN_REFUSED"
SHAPE_DROPPED = "RECT_BECAUSE_THE_ZONE_PLAN_REFUSED"


def _door_bearing(
    shell: RoomShell, target_xy: tuple[float, float], arc_deg: tuple[float, float] | None
) -> tuple[float | None, float | None, bool | None]:
    """Where the door is from the task's target, against the arm's approach arc.

    The computable half of a requirement whose stated form turned out to be vacuous. The design
    doc asks for a gap named DOOR_CORRIDOR_CROSSES_APPROACH_ARC, and that gap cannot fire: the
    corridor is built to *end* on the protected region's boundary and is then subtracted from it,
    and every standoff lies inside the protected region, so the two are disjoint by construction.
    The requirement was written before `cell_protected` was wired into `plan_zones`.

    What is left of the requirement is whether the way in arrives from a direction the arm can work
    from. Measured over 20 fitted seeds, that would refuse **40 % of office seeds** -- the door's
    bearing falls inside the arc on 8 of 20, 8 of 20 and 0 of 20 for the three experiments -- which
    is the shape of mistake this file has already reverted twice. So it is a diagnostic, recorded
    where a reader can see it, and not a gate.

    The arc is usable as a wedge: all three experiments give contiguous bearings (office -135 to
    -30 degrees with every 15-degree step present, kitchen -150 to -30, bedroom -105 to -45), and no
    transform is needed because the cell's frame is a pure translation.
    """
    if arc_deg is None:
        return None, None, None
    door = next((item for item in shell.openings if item.kind == OpeningKind.DOOR.value), None)
    if door is None:
        return None, None, None
    bearing = math.degrees(math.atan2(
        door.center_xy[1] - target_xy[1], door.center_xy[0] - target_xy[0]
    ))
    low, high = arc_deg
    inside = low - 1e-9 <= bearing <= high + 1e-9
    margin = min(bearing - low, high - bearing)
    return round(bearing, 3), round(margin, 3), inside


@dataclass(frozen=True, slots=True)
class _FittedAttempt:
    """One sampled shell that held the cell, and whether the zone plan accepted it."""

    shell: RoomShell
    options: ShellOptions
    translation: tuple[float, float]
    protected_in_room: Polygon
    opening_attempts: int
    size_margin_m: tuple[float, float]
    plan_refused: bool
    door_bearing_deg: float | None = None
    door_arc_margin_deg: float | None = None
    door_in_approach_arc: bool | None = None


def _fit_one_shell(
    spec: SceneSpec,
    *,
    room_seed: int,
    need_w: float,
    need_d: float,
    protected: Polygon,
    margin_m: float,
    notch_band: tuple[float, float] = NOTCH_FRACTION_BAND,
    family_override: Family | None = None,
    target_offset_xy: tuple[float, float] | None = None,
    approach_arc_deg: tuple[float, float] | None = None,
) -> _FittedAttempt:
    """Draw one shell at these settings and fit the cell into it, or fail by name.

    Raises ``ROOM_CANNOT_CONTAIN_TASK_CELL`` when the outline leaves no legal position -- which the
    retry below catches, because that is a reason to try a different shape rather than to give up.
    """
    partial, width, depth = _sample_shell_program(
        spec, seed=room_seed, need_w_m=need_w, need_d_m=need_d, notch_band=notch_band,
    )
    if family_override is not None:
        partial = dataclass_replace(partial, family=family_override, notches=())
    # Pinned, because a fitted room must be the one that was fitted: ``build_shell`` would otherwise
    # re-derive the size from the spec and hand back a room the cell was never fitted into.
    partial = dataclass_replace(partial, width_m=width, depth_m=depth)
    family, notches = partial.family, partial.notches
    outline = _outline(width, depth, family, notches)
    available = inset(outline, partial.wall_thickness_m)
    fitted = _fit_candidates(protected, available, margin_m=margin_m)
    if not fitted:
        raise RoomError(
            f"ROOM_CANNOT_CONTAIN_TASK_CELL(sampled={width:.3f}x{depth:.3f} m, "
            f"protected={need_w:.3f}x{need_d:.3f} m including standoff)"
        )
    # Openings are cheaper to resample than the shell. Keep the cell and outline frozen while
    # trying bounded opening substreams and the cell's allowed rigid translation together. The
    # cell's internal geometry stays frozen; only its local frame moves inside the room.
    from rfw_w.scenes.solvers.zones import door_sweep_for

    options = partial
    shell = None
    opening_attempts = 0
    plan_refused = False
    conflicting_sweep: tuple[Polygon, ...] = ()
    translation, protected_in_room = fitted[0]
    for attempt in range(16):
        openings = _place_openings(
            outline, partial, random.Random(f"{room_seed}:openings:{attempt}")
        )
        candidate_options = dataclass_replace(partial, openings=openings)
        candidate_shell = build_shell(spec, options=candidate_options)
        sweep = door_sweep_for(candidate_shell)
        candidate_fit = _best_fit_candidate(
            spec, candidate_shell,
            _fit_candidates(protected, available, margin_m=margin_m, blockers=sweep),
        )
        if candidate_fit is not None:
            translation, protected_in_room, plan_refused = candidate_fit
            options, shell = candidate_options, candidate_shell
            opening_attempts = attempt + 1
            break
        conflicting_sweep = sweep
    if shell is None:
        raise RoomError(
            f"DOOR_SWEEP_CROSSES_TASK_CELL(room_seed={room_seed}, attempts=16, "
            f"conflicting_pieces={len(conflicting_sweep)})"
        )
    # The cell's frame is a pure translation, so its target offset carries over as it stands.
    bearing = margin = None
    inside = None
    if target_offset_xy is not None:
        bearing, margin, inside = _door_bearing(
            shell,
            (translation[0] + target_offset_xy[0], translation[1] + target_offset_xy[1]),
            approach_arc_deg,
        )
    return _FittedAttempt(
        shell=shell,
        options=options,
        translation=translation,
        protected_in_room=protected_in_room,
        opening_attempts=opening_attempts,
        size_margin_m=(round(width - need_w, 4), round(depth - need_d, 4)),
        plan_refused=plan_refused,
        door_bearing_deg=bearing,
        door_arc_margin_deg=margin,
        door_in_approach_arc=inside,
    )


def _translated(polygon: Polygon, offset: tuple[float, float]) -> Polygon:
    return Polygon(tuple((x + offset[0], y + offset[1]) for x, y in polygon.vertices))


def _fit_candidates(
    protected: Polygon,
    available: Polygon,
    *,
    margin_m: float,
    blockers: Sequence[Polygon] = (),
) -> tuple[tuple[tuple[float, float], Polygon], ...]:
    """Every deterministic translation that holds ``protected`` inside ``available``.

    Five samples an axis rather than the whole feasible set: the extremes, the middle and the two
    quarter points, in the order the fit used to take them. The extreme is what an earlier version
    returned immediately, and taking the *first* admissible one is what put the office's work cell
    in the middle of the room -- where it split the floor the periphery needs into pieces no piece
    of furniture fits in. All of them come back now so the caller can choose
    (:func:`_best_fit_candidate`).
    """
    p_min_x, p_min_y, p_max_x, p_max_y = protected.bounds()
    a_min_x, a_min_y, a_max_x, a_max_y = available.bounds()
    low_x = a_min_x + margin_m - p_min_x
    high_x = a_max_x - margin_m - p_max_x
    low_y = a_min_y + margin_m - p_min_y
    high_y = a_max_y - margin_m - p_max_y
    if low_x > high_x or low_y > high_y:
        return ()

    def samples(low: float, high: float) -> tuple[float, ...]:
        if abs(high - low) <= 1e-9:
            return (low,)
        middle = (low + high) / 2.0
        return (middle, low, high, (low + middle) / 2.0, (middle + high) / 2.0)

    found: list[tuple[tuple[float, float], Polygon]] = []
    for x in samples(low_x, high_x):
        for y in samples(low_y, high_y):
            moved = _translated(protected, (x, y))
            if contains_polygon(available, moved) and not any(
                overlaps(moved, blocker) for blocker in blockers
            ):
                found.append(((round(x, 6), round(y, 6)), moved))
    return tuple(found)


def _best_fit_candidate(
    spec: SceneSpec, shell: RoomShell, candidates: Sequence[tuple[tuple[float, float], Polygon]],
) -> tuple[tuple[float, float], Polygon, bool] | None:
    """The placement that leaves the most usable floor, or the first one if that cannot be said.

    Holding the cell is necessary and not sufficient. Where the cell *is* decides whether the rest
    of the room is one region or a scatter of slivers: measured on `office_pick_cup` seed 11, the
    admissible placements span **5.93 to 7.47 m2** of placeable floor, and the seating zone ranges
    from 0.11 m2 -- no chair fits in it -- to 0.72 m2. Taking the first admissible candidate, which
    is what this did, picked the 0.11 m2 end and the run then failed its own randomization closure
    because a chair named by a soft relation could not be placed.

    The zone plan is the definition of "usable floor" here rather than a proxy for it, so this asks
    it directly. A placement whose zone plan refuses (a corridor that reaches nothing, a zone
    outside the room) is not a placement, and is ranked last rather than crashing the fit: the fit's
    job is to find somewhere the cell fits, and the plan's own invariants are the authority on
    whether the room still works.

    Ties keep the earlier candidate, so the choice is a function of the geometry and not of
    iteration order.

    The third element of the result says whether **every** candidate was refused by the plan. The
    caller needs to know: a fit whose plan refuses is not a fit, and returning one silently is how
    the shape's cost arrives as `CIRCULATION_MISSES_TASK_CELL` from `plan_zones` in a later stage
    that has no idea a notch was involved. Measured on a sweep with an alcove published, 7 of 180
    fitted rooms were refused, every one of them with a notch and none of them pre-existing.
    """
    from rfw_w.scenes.solvers.zones import ZoneError, plan_zones

    best: tuple[tuple[float, float], Polygon] | None = None
    best_area = -1.0
    for placement, moved in candidates:
        try:
            graph = plan_zones(shell, spec.style, cell_protected=moved)
        except ZoneError:
            continue
        if graph.placeable_area_m2 > best_area + 1e-9:
            best, best_area = (placement, moved), graph.placeable_area_m2
    if best is None:
        # Every admissible placement was refused by the zone plan. Returning the first keeps the
        # failure where it belongs -- `plan_zones` raises in the caller with its own named gap --
        # rather than replacing it with "the room cannot hold the cell", which would be false.
        return (candidates[0][0], candidates[0][1], True) if candidates else None
    return best[0], best[1], False


def fit_room_around_cell(
    spec: SceneSpec,
    cell: Any,
    *,
    room_seed: int,
    wall_standoff_m: float = 0.20,
) -> FittedRoom:
    """Fit a varied shell around ``cell.protected`` without changing the cell.

    The build's room path, and the point of the reversal: the room is drawn *to hold* the task
    rather than the task being fitted into the room that was drawn. The cell's own geometry --
    its support footprint, target offset, standoffs and approach arc -- is read for its protected
    region and nothing else, and is not reshaped here; only its local frame moves, and the fit
    reports that translation so the build can put the cell where the room was fitted around it.

    Two things fail by name before anything is placed: a cell that cannot fit the requested bounds
    (``ROOM_TOO_SMALL_FOR_TASK_CELL``), and a room whose notches leave no legal position for it
    (``ROOM_CANNOT_CONTAIN_TASK_CELL``).
    """
    if wall_standoff_m < 0:
        raise RoomError("wall_standoff_m must not be negative")
    protected = cell.protected
    p_min_x, p_min_y, p_max_x, p_max_y = protected.bounds()
    need_w = (p_max_x - p_min_x) + 2.0 * wall_standoff_m
    need_d = (p_max_y - p_min_y) + 2.0 * wall_standoff_m
    x_bounds, y_bounds, _z_bounds = spec.room.size_bounds_m
    if need_w > x_bounds.maximum + 1e-9 or need_d > y_bounds.maximum + 1e-9:
        raise RoomError(
            f"ROOM_SIZE_CONFLICT(ROOM_TOO_SMALL_FOR_TASK_CELL; need={need_w:.3f}x{need_d:.3f} m, "
            f"bound={x_bounds.maximum:.3f}x{y_bounds.maximum:.3f} m)"
        )

    # The shape the seed drew, then two ways of giving ground -- and only as far as it takes.
    #
    # A notch is not free. It shortens the walls the zone planner draws its strips along, and it
    # takes floor the cell could have stood in, so a room whose *bounding box* holds the cell can
    # still leave the plan with nothing it can use. Measured with an alcove published: 7 of 180
    # fitted rooms were refused by `plan_zones`, every one of them notched, and 4 of the 5 in one
    # experiment sweep planned cleanly at half the bite.
    #
    # So the escalation is: the drawn shape, then the same room with a shallower bite, then the
    # rectangle. Each step keeps the sampled width and depth and every other draw in place, so the
    # room is recognisably the same room -- and the step that was needed is recorded rather than
    # silently taken, because a seed that gave its shape up must not look like one that never asked
    # for a shape. The rectangle stays the last resort, not the first: it used to be reached for
    # whenever the fit found nothing, and it is now only reached when nothing else plans.
    attempts: list[tuple[str | None, _FittedAttempt]] = []
    last_error: RoomError | None = None
    for label, band, override in (
        (None, NOTCH_FRACTION_BAND, None),
        (BITE_HALVED, NARROWER_NOTCH_FRACTION_BAND, None),
        (SHAPE_DROPPED, NOTCH_FRACTION_BAND, Family.RECT),
    ):
        try:
            attempt = _fit_one_shell(
                spec, room_seed=room_seed, need_w=need_w, need_d=need_d,
                protected=protected, margin_m=wall_standoff_m,
                notch_band=band, family_override=override,
                target_offset_xy=getattr(cell, "target_offset", None),
                approach_arc_deg=getattr(cell, "approach_arc_deg", None),
            )
        except RoomError as error:
            last_error = error
            continue
        attempts.append((label, attempt))
        if not attempt.plan_refused:
            break
    if not attempts:
        raise last_error or RoomError(
            f"ROOM_SEARCH_EXHAUSTED(ROOM_CANNOT_CONTAIN_TASK_CELL; protected={need_w:.3f}x{need_d:.3f} m including "
            "standoff)"
        )

    # Prefer an attempt the plan accepted; fall back to the first one that fitted at all, so the
    # refusal surfaces from `plan_zones` in the caller with its own named gap rather than being
    # replaced by "the room cannot hold the cell", which would be false.
    fallback, chosen = next(
        ((label, item) for label, item in attempts if not item.plan_refused), attempts[0]
    )
    return FittedRoom(
        chosen.shell, chosen.options, chosen.translation, chosen.protected_in_room,
        chosen.opening_attempts,
        wall_standoff_m=wall_standoff_m,
        shape_fallback=fallback,
        size_margin_m=chosen.size_margin_m,
        door_bearing_deg=chosen.door_bearing_deg,
        door_arc_margin_deg=chosen.door_arc_margin_deg,
        door_in_approach_arc=chosen.door_in_approach_arc,
    )


def _rectangle_outline(width: float, depth: float) -> list[tuple[float, float]]:
    """The four corners, counter-clockwise from the bottom-left."""
    return [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]


def _corner_cut(
    corner: Corner, width: float, depth: float, notch: Notch
) -> list[tuple[float, float]]:
    """The three vertices that replace a corner when it is bitten off, in counter-clockwise order.

    ``width_m`` is the notch's extent along x and ``depth_m`` its extent along y, for every
    corner -- stated here because the alternative (measuring each along the edge you arrive on)
    would make the same two numbers mean different things at different corners.
    """
    bite_x, bite_y = notch.width_m, notch.depth_m
    if corner is Corner.BOTTOM_LEFT:
        return [(0.0, bite_y), (bite_x, bite_y), (bite_x, 0.0)]
    if corner is Corner.BOTTOM_RIGHT:
        return [(width - bite_x, 0.0), (width - bite_x, bite_y), (width, bite_y)]
    if corner is Corner.TOP_RIGHT:
        return [(width, depth - bite_y), (width - bite_x, depth - bite_y), (width - bite_x, depth)]
    return [(bite_x, depth), (bite_x, depth - bite_y), (0.0, depth - bite_y)]


def _edge_recess(
    start: tuple[float, float], end: tuple[float, float], notch: Notch
) -> list[tuple[float, float]]:
    """The four vertices of a recess in the middle of an edge, in counter-clockwise order.

    The inward direction is the left normal of the edge, which for a counter-clockwise outline is
    the room's interior -- so the recess cuts into the room rather than bulging out of it.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        raise RoomError(f"degenerate edge from {start} to {end}")
    along = (dx / length, dy / length)
    inward = (-along[1], along[0])
    centre = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    half = notch.width_m / 2

    def offset(along_m: float, inward_m: float) -> tuple[float, float]:
        return (
            round(centre[0] + along[0] * along_m + inward[0] * inward_m, 9),
            round(centre[1] + along[1] * along_m + inward[1] * inward_m, 9),
        )

    return [
        offset(-half, 0.0),
        offset(-half, notch.depth_m),
        offset(half, notch.depth_m),
        offset(half, 0.0),
    ]


def _check_family_matches_notches(family: Family, notches: Sequence[Notch]) -> None:
    """Refuse a family and a notch set that describe different shapes.

    ``ShellOptions`` can be built by hand -- the room-only viewer and the tests both do -- so the
    family and its notches must be checked against each other somewhere, and a named refusal is
    what stops a hand-built ``family=L`` with no notch from silently drawing a rectangle and
    passing the realisation check.
    """
    at_corners = [item for item in notches if item.corner is not None]
    at_edges = [item for item in notches if item.edge is not None]
    if family is Family.L:
        if len(at_corners) != 1 or at_edges:
            raise RoomError(
                f"FAMILY_L_NEEDS_ONE_CORNER_NOTCH(corner={len(at_corners)}, edge={len(at_edges)})"
            )
    elif family is Family.T:
        if len(at_corners) != 2 or at_edges:
            raise RoomError(
                f"FAMILY_T_NEEDS_TWO_CORNER_NOTCHES(corner={len(at_corners)}, edge={len(at_edges)})"
            )
        pair = {item.corner for item in at_corners}
        if not any(pair == set(_CORNERS_OF_EDGE[edge]) for edge in Edge):
            raise RoomError(
                f"FAMILY_T_CORNERS_ARE_NOT_ONE_EDGE({sorted(c.value for c in pair)}); a T is one "
                "edge with both its corners taken, so that edge keeps only its middle"
            )
    elif family is Family.U:
        if len(at_edges) != 1 or at_corners:
            raise RoomError(
                f"FAMILY_U_NEEDS_ONE_EDGE_NOTCH(corner={len(at_corners)}, edge={len(at_edges)})"
            )


def _surviving_parts(width: float, depth: float, notches: Sequence[Notch]) -> list[tuple[str, float]]:
    """Every part of the room a notch narrows, with the name a gap should call it by.

    Computed from the request, not read back off the outline. What a zone plan has to inset into is
    what was asked for minus what was bitten, and measuring it from the cleaned polygon would
    report the vertex-cleaning tolerance instead of the shape -- which is how a neck of exactly
    ``MIN_ROOM_NECK_M`` sat on the inset's collapse boundary without anything saying so.
    """
    parts: list[tuple[str, float]] = []
    corners = [item for item in notches if item.corner is not None]
    present = {item.corner for item in corners}
    paired: set[Corner] = set()
    for edge, (first, second) in _CORNERS_OF_EDGE.items():
        if first in present and second in present:
            one = next(item for item in corners if item.corner is first)
            two = next(item for item in corners if item.corner is second)
            if edge in (Edge.BOTTOM, Edge.TOP):
                parts.append(("neck", width - one.width_m - two.width_m))
                parts.append(("stem", depth - max(one.depth_m, two.depth_m)))
            else:
                parts.append(("neck", depth - one.depth_m - two.depth_m))
                parts.append(("stem", width - max(one.width_m, two.width_m)))
            paired |= {first, second}
    for item in corners:
        if item.corner not in paired:  # alone, so it spends an arm on each axis
            parts.append(("arm_x", width - item.width_m))
            parts.append(("arm_y", depth - item.depth_m))
    for item in notches:
        if item.edge is None:
            continue
        along, inward = ((width, depth) if item.edge in (Edge.BOTTOM, Edge.TOP)
                         else (depth, width))
        parts.append(("flank", along - item.width_m))
        parts.append(("floor_behind", inward - item.depth_m))
    return parts


def _notch_gaps(
    width: float, depth: float, family: Family, notches: Sequence[Notch]
) -> tuple[str, ...]:
    """What is wrong with these notches, as named gaps -- empty when nothing is.

    One predicate with two callers, for two different reasons. :func:`_outline` raises its first
    gap, so a shell that cannot be built explains itself in the room's terms rather than failing
    later over a number. ``validate_shell`` appends all of them, so a shell handed in from outside
    is judged by exactly the rule the drawer obeys.

    It used to live inside ``_outline`` alone, and ``validate_shell`` -- which builds its polygon
    straight from ``polygon_xy`` and never calls ``_outline`` -- therefore never applied it.
    Measured: a hand-built T with a 0.1 m neck passed with ``ok=True, gaps=()``.

    Three kinds of violation, ordered so the most specific diagnosis wins:

    * corner notches that **add up** past the room sever it into disconnected pieces -- a floor
      plan, not a shell -- and that is reported first because it is the extreme of the next one;
    * an edge recess that eats its own edge, or is deeper than the room;
    * any surviving part narrower than a neck the rest of the pipeline can still plan.
    """
    gaps: list[str] = []
    spare = MIN_MATERIAL_BESIDE_NOTCH_M
    # Two corner bites on one edge share the axis they *differ* in, and only that one: a T on the
    # bottom edge takes x from both ends and leaves the depth alone. Adding both axes for every
    # notch would refuse that T for consuming 3.0 m of a 3.04 m depth it never touched.
    corners = [item for item in notches if item.corner is not None]
    present = {item.corner for item in corners}
    corner_x = corner_y = 0.0
    paired: set[Corner] = set()
    for edge, (first, second) in _CORNERS_OF_EDGE.items():
        if first in present and second in present:
            one = next(item for item in corners if item.corner is first)
            two = next(item for item in corners if item.corner is second)
            if edge in (Edge.BOTTOM, Edge.TOP):
                corner_x += one.width_m + two.width_m
            else:
                corner_y += one.depth_m + two.depth_m
            paired |= {first, second}
    for item in corners:
        if item.corner not in paired:  # alone on its own: it spends both axes
            corner_x += item.width_m
            corner_y += item.depth_m
    if corner_x > width - spare + 1e-9:
        gaps.append(
            f"CORNER_NOTCHES_CONSUME_THE_WIDTH({corner_x:.3f} m of {width:.3f} m): the outline "
            "would be cut into disconnected pieces"
        )
    if corner_y > depth - spare + 1e-9:
        gaps.append(
            f"CORNER_NOTCHES_CONSUME_THE_DEPTH({corner_y:.3f} m of {depth:.3f} m): the outline "
            "would be cut into disconnected pieces"
        )
    for item in notches:
        if item.edge is None:
            continue
        horizontal = item.edge in (Edge.BOTTOM, Edge.TOP)
        along, inward = (width, depth) if horizontal else (depth, width)
        if item.width_m > along - 2 * spare + 1e-9:
            gaps.append(
                f"EDGE_NOTCH_LEAVES_NO_EDGE(edge={item.edge.value}, notch={item.width_m:.3f} m of "
                f"{along:.3f} m): {2 * spare:.3f} m must remain, {spare:.3f} m at each end"
            )
        if item.depth_m > inward - spare + 1e-9:
            gaps.append(
                f"EDGE_NOTCH_IS_DEEPER_THAN_THE_ROOM(edge={item.edge.value}, "
                f"notch={item.depth_m:.3f} m of {inward:.3f} m)"
            )

    # The one rule the drawer and the validator both have to obey, and neither did: every part a
    # notch leaves must still be something the zone planner can inset a centre into. Without it a
    # hand-built T with a 0.1 m neck validated clean, and a drawn one could leave a stem of 0.05 m
    # -- the minimum material beside a notch -- while still calling itself a T.
    for part, extent in _surviving_parts(width, depth, notches):
        if extent < MIN_ROOM_NECK_M - 1e-9:
            gaps.append(
                f"NOTCH_LEAVES_NO_USABLE_NECK(family={family.value}, part={part}, "
                f"measured={max(extent, 0.0):.3f}, floor={MIN_ROOM_NECK_M:.3f}): the {part.replace('_', ' ')} "
                "is too narrow for the zone planner to inset a centre into, and an inset wider "
                "than half of it collapses the offset onto itself"
            )
    return tuple(gaps)


def _outline(width: float, depth: float, family: Family, notches: Sequence[Notch]) -> Polygon:
    """The floor outline, counter-clockwise, for a family and its notches.

    A rectangle with bites taken out of it, walked in one pass: each corner emits either itself or
    the three vertices that replace it, and then the edge leaving that corner emits either nothing
    or a recess. Doing it in one walk rather than cutting a polygon repeatedly keeps the winding
    and the vertex order a direct consequence of the family, so a change to the shape is legible
    in the vertex list.
    """
    if family is Family.RECT:
        if notches:
            raise RoomError("FAMILY_RECT_HAS_NO_NOTCHES")
        return Polygon.rectangle(0.0, 0.0, width, depth)

    _check_family_matches_notches(family, notches)
    problems = _notch_gaps(width, depth, family, notches)
    if problems:
        raise RoomError(problems[0])

    corners = _rectangle_outline(width, depth)
    at_corner = {item.corner: item for item in notches if item.corner is not None}
    at_edge = [item for item in notches if item.edge is not None]

    points: list[tuple[float, float]] = []
    for index, corner in enumerate(_CORNERS_CCW):
        if corner in at_corner:
            points.extend(_corner_cut(corner, width, depth, at_corner[corner]))
        else:
            points.append(corners[index])
        leaving = _EDGES_CCW[index]
        for item in at_edge:
            if item.edge is leaving:
                points.extend(
                    _edge_recess(corners[index], corners[(index + 1) % 4], item)
                )

    polygon = Polygon.from_sequence(points)
    validate_simple(polygon)
    return polygon


def _wall_of(polygon: Polygon, point: tuple[float, float]) -> int | None:
    """Which edge a boundary point lies on, or ``None`` if it is on none of them."""
    for index, (start, end) in enumerate(polygon.edges):
        if _point_on_segment(start, end, point):
            return index
    return None


def _point_on_segment(start, end, point, *, tolerance: float = 1e-6) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return False
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    if not (0.0 <= t <= 1.0):
        return False
    projection = (start[0] + t * dx, start[1] + t * dy)
    return abs(projection[0] - point[0]) < tolerance and abs(projection[1] - point[1]) < tolerance


def _point_along(start, end, offset_m: float) -> tuple[float, float]:
    """The point ``offset_m`` from ``start`` toward ``end``."""
    length = math.dist(start, end)
    if length <= 0:
        raise RoomError(f"degenerate wall from {start} to {end}")
    t = offset_m / length
    return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)


def _segment_centre(
    rng: random.Random, length: float, width_m: float, margin_m: float
) -> float:
    """An offset along a wall, drawn from one of three bands and held clear of both ends.

    The wall is divided into thirds and one is drawn, so an opening lands where a person would put
    it rather than always at the midpoint -- which is what the shell did while it had exactly one
    window, always centred on the longest wall. The band is a *preference* and the margin is a
    requirement: when the drawn third lies wholly inside the margin the opening goes in the middle
    of what is left, rather than redrawing, because a redraw would make the number of draws depend
    on the room's shape.
    """
    # Tolerant by a nanometre, not exact. ``length`` is a distance between two polygon vertices
    # and ``width_m`` has just been rounded to three places, so a wall that fits exactly still
    # compares as short: measured, a 1.501 m wall with a 1.301 m window fails ``1.501 <
    # 1.301 + 0.2`` because the sum is 1.5010000000000003. A shell that refuses to build over
    # 3e-16 m is not enforcing a clearance, it is enforcing floating-point noise.
    if length < width_m + 2 * margin_m - 1e-9:
        raise RoomError(
            f"WALL_TOO_SHORT_FOR_OPENING(length={length:.3f} m, width={width_m:.3f} m, "
            f"margin={margin_m:.3f} m)"
        )
    half = width_m / 2
    low = margin_m + half
    high = length - margin_m - half
    band = rng.randrange(len(SEGMENT_BOUNDARIES) - 1)
    band_low = max(low, SEGMENT_BOUNDARIES[band] * length)
    band_high = min(high, SEGMENT_BOUNDARIES[band + 1] * length)
    if band_high <= band_low:
        return (low + high) / 2
    return rng.uniform(band_low, band_high)


def _windows_on_wall(
    rng: random.Random, wall_index: int, length: float, options: ShellOptions, first_id: int
) -> list[OpeningPlan]:
    """The windows one wall takes, and where.

    One, or two when the wall is long enough for two windows with ``MIN_OPENING_SEPARATION_M``
    between them -- a 5 m wall is two windows, not one picture window. Two is the cap: past that
    the wall is a curtain wall and the room stops reading as a room.

    Two windows go at the quarter points rather than by the band rule, because the band rule can
    put two drawn centres close enough to touch and there is then nothing left to do but drop one
    -- and a window dropped for a reason nobody can see is worse than one placed simply.

    The drawn width is held to the span that is left rather than to ``window_width_m``. That field
    is the *drawer's starting point* -- 1.2 m -- and the guard below tests against it, but the draw
    goes up to 1.4 m, so a wall between 1.4 m and 1.6 m long passed the guard and then handed
    ``_segment_centre`` a width it refused, raising ``WALL_TOO_SHORT_FOR_OPENING`` out of a shell
    that had every reason to build. Measured before the fix, on a T whose neck and a U whose flank
    are exactly ``MIN_ROOM_NECK_M`` wide: 12 to 14 of 40 notched office seeds died there.

    Clamping rather than redrawing, and the distinction matters: the number of draws is unchanged,
    so the opening substreams stay aligned and the sixteen-attempt door loop still replays them.
    """
    low, high = WINDOW_EDGE_INSET_M, length - WINDOW_EDGE_INSET_M
    span = high - low
    if span < options.window_width_m:
        return []
    # Floored before it is used as a clamp, because ``round`` rounds *up*: a span of 1.3045 m
    # clamped and then rounded to three places becomes 1.305 m, which is wider than the wall.
    # The error message prints three places too, so the two disagree by an amount neither shows.
    usable = math.floor(span * 1000) / 1000
    pairs = span >= 2 * MAX_WINDOW_SPAN_M
    fractions: tuple[float | None, ...] = (0.25, 0.75) if pairs else (None,)
    plans: list[OpeningPlan] = []
    for index, fraction in enumerate(fractions):
        width = round(min(rng.uniform(0.8, 1.4), MAX_WINDOW_SPAN_M, usable), 3)
        sill = round(rng.uniform(0.8, 1.1), 3)
        offset = (_segment_centre(rng, length, width, WINDOW_EDGE_INSET_M)
                  if fraction is None else low + (high - low) * fraction)
        plans.append(OpeningPlan(
            opening_id=f"window_{first_id + index:02d}",
            kind=OpeningKind.WINDOW.value,
            wall_index=wall_index,
            offset_along_m=round(offset, 6),
            width_m=width,
            height_m=options.window_height_m,
            sill_m=sill,
        ))
    return plans


def _place_openings(
    polygon: Polygon, options: ShellOptions, rng: random.Random
) -> tuple[OpeningPlan, ...]:
    """Where every opening goes, as one resolved set.

    The door is placed first because the wall it takes is the wall the windows cannot use -- which
    was the whole of the window rule while there was exactly one window, and is still the rule
    that keeps a window out of a doorway.
    """
    walls = list(polygon.edges)
    if not walls:
        raise RoomError("the outline has no walls to put an opening in")
    lengths = [math.dist(start, end) for start, end in walls]

    if options.door_center is not None:
        door_wall = _wall_of(polygon, options.door_center)
        if door_wall is None:
            raise RoomError(
                f"door center {options.door_center} does not lie on the room boundary; an opening "
                "in the middle of the floor is not a door"
            )
        door_offset = math.dist(walls[door_wall][0], options.door_center)
    else:
        door_wall = max(range(len(walls)), key=lambda index: lengths[index])
        door_offset = _segment_centre(
            rng, lengths[door_wall], options.door_width_m, DOOR_WALL_MARGIN_M
        )

    plans: list[OpeningPlan] = [OpeningPlan(
        opening_id="door_main",
        kind=OpeningKind.DOOR.value,
        wall_index=door_wall,
        offset_along_m=round(door_offset, 6),
        width_m=options.door_width_m,
        height_m=options.door_height_m,
    )]

    if not options.window:
        return tuple(plans)

    # Longest wall first, so a room with more walls than it has windows puts them where they are
    # most likely to be seen and least likely to be blocked.
    order = sorted((index for index in range(len(walls)) if index != door_wall),
                   key=lambda index: -lengths[index])
    windows = 0
    for index in order:
        placed = _windows_on_wall(rng, index, lengths[index], options, windows + 1)
        plans.extend(placed)
        windows += len(placed)

    if windows < options.min_windows:
        raise RoomError(
            f"ROOM_CANNOT_HOLD_THE_REQUESTED_WINDOWS(asked={options.min_windows}, got={windows}): "
            "no wall the door is not in is long enough"
        )
    return tuple(plans)


def _materialise(polygon: Polygon, plans: tuple[OpeningPlan, ...]) -> tuple[Opening, ...]:
    """Turn resolved plans into openings with a place in the room."""
    walls = list(polygon.edges)
    openings: list[Opening] = []
    for plan in plans:
        if not 0 <= plan.wall_index < len(walls):
            raise RoomError(
                f"{plan.opening_id}: wall {plan.wall_index} is outside the outline's "
                f"{len(walls)} wall(s)"
            )
        start, end = walls[plan.wall_index]
        length = math.dist(start, end)
        if plan.span_m[0] < -1e-9 or plan.span_m[1] > length + 1e-9:
            raise RoomError(
                f"OPENING_OVERRUNS_ITS_WALL({plan.opening_id}, span={plan.span_m[0]:.3f}.."
                f"{plan.span_m[1]:.3f} m of {length:.3f} m)"
            )
        centre = _point_along(start, end, plan.offset_along_m)
        openings.append(Opening(
            opening_id=plan.opening_id,
            kind=OpeningKind(plan.kind),
            center_xy=(round(centre[0], 6), round(centre[1], 6)),
            width_m=plan.width_m,
            height_m=plan.height_m,
            sill_z_m=plan.sill_m,
        ))
    return tuple(openings)


@dataclass(frozen=True, slots=True)
class ShellValidation:
    """Outcome of validating a shell, with a named reason when it fails."""

    ok: bool
    gaps: tuple[str, ...] = ()
    measured: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "gaps": list(self.gaps), "measured": dict(self.measured)}


def build_shell(spec: SceneSpec, *, options: ShellOptions | None = None) -> RoomShell:
    """Generate the room shell for a spec.

    Deterministic in ``(spec, options)``: the same inputs give the same polygon, so a shell can
    be diffed between runs and a regression in the outline is visible rather than a matter of
    luck.

    It draws no random number of its own. It used to open a second ``random.Random(spec.seed)``
    and take the height from it, which worked only because that stream happened to start where
    ``generate_options``'s had stopped -- so a draw added to either function would have silently
    renumbered the other. The width, depth and openings are re-derived from the same named
    substreams ``generate_options`` drew them from, which keeps them a function of ``spec.seed``
    alone: a hand-built ``ShellOptions`` still gets the room the spec asks for, rather than being
    able to pin a size the spec's own bounds forbid.
    """
    resolved = options or generate_options(spec)
    if resolved.width_m is None:
        size_rng = random.Random(spec.seed)
        x_bounds, y_bounds, _ = spec.room.size_bounds_m
        width = round(_pick(size_rng, x_bounds), 4)
        depth = round(_pick(size_rng, y_bounds), 4)
    else:
        width, depth = resolved.width_m, resolved.depth_m

    polygon = _outline(width, depth, resolved.family, resolved.notches)

    plans = resolved.openings or _place_openings(
        polygon, resolved, random.Random(f"{spec.seed}:openings")
    )
    openings = _materialise(polygon, plans)

    return RoomShell(
        polygon_xy=polygon.vertices,
        holes_xy=(),
        height_m=resolved.height_m,
        floor_z_m=0.0,
        openings=openings,
        source=SHELL_SOURCE,
    )


def check_door_connectivity(
    shell: RoomShell, *, minimum_width_m: float, opening_id: str = "door_main"
) -> tuple[str, ...]:
    """Whether the named opening admits the robot.

    Returns a named gap rather than a boolean, so a failure reads as what is wrong and by how
    much. A door narrower than the footprint plus margin is a hard failure: the passage
    requirement cannot be met by any layout behind it.
    """
    if minimum_width_m <= 0:
        raise RoomError(f"minimum width must be positive, got {minimum_width_m}")
    opening = next((item for item in shell.openings if item.opening_id == opening_id), None)
    if opening is None:
        return (f"no opening named {opening_id!r} in the shell",)
    if opening.width_m < minimum_width_m:
        return (
            f"{opening_id} is {opening.width_m:.3f} m wide but {minimum_width_m:.3f} m is "
            f"required (robot footprint plus margin); the passage cannot be established",
        )
    return ()


def validate_shell(
    shell: RoomShell,
    spec: SceneSpec,
    *,
    options: ShellOptions | None = None,
    protected_in_room: Polygon | None = None,
) -> ShellValidation:
    """Check a shell against the request, fail-closed and with named gaps.

    The bounding-box check is deliberate and load-bearing: if the outline is a rectangle while
    the request carried a notch, the shape was almost certainly reduced to its box somewhere,
    and that reduction is exactly the error the manual warns produces a passing containment
    test for an object that is floating in the notch.
    """
    gaps: list[str] = []
    measured: dict[str, Any] = {}

    try:
        polygon = Polygon(shell.polygon_xy)
        validate_simple(polygon)
    except (PolygonError, UnitsError) as error:
        return ShellValidation(ok=False, gaps=(f"outline is not a valid polygon: {error}",))

    # A shell carrying holes is checked on its outline alone, which is a false pass: the outline
    # is what every consumer reads, and each of them treats a hole as floor. Named rather than
    # ignored, because a field that validates cleanly while meaning nothing is worse than one
    # that is refused -- see the note on `RoomShell.holes_xy`.
    measured["holes"] = len(shell.holes_xy)
    if shell.holes_xy:
        gaps.append(
            f"ROOM_HOLES_NOT_SUPPORTED(holes={len(shell.holes_xy)}): the shell carries interior "
            "holes, and this validator checks the outline only -- so the floor, the zone plan and "
            "the door sweep would all treat a courtyard as somewhere to stand"
        )

    # The task's protected region has to be inside the *air*, not merely inside the outline, because
    # the outline includes the wall body. Containment rather than overlap, deliberately: a region
    # that pokes out through a notch is not intersecting a wall, it is outside the room, and
    # reporting that as an intersection would name the wrong fault.
    #
    # Honest value: on the build path this can never fire, because `_fit_candidates` already puts
    # the region inside `inset(outline, wall_thickness)`. Its value is for a shell from anywhere
    # else -- the same value class as OUTLINE_DOES_NOT_REALISE_THE_REQUESTED_SHAPE -- and
    # `protected_checked` says whether it was looked at at all, so "no protected region was given"
    # is distinguishable from "the region was found to be inside".
    measured["protected_checked"] = protected_in_room is not None
    if protected_in_room is not None:
        thickness = (options or ShellOptions()).wall_thickness_m
        try:
            inner = inset(polygon, thickness)
        except PolygonError as error:
            gaps.append(
                f"WALL_INTRUDES_PROTECTED_REGION(the room is too narrow to inset by the wall "
                f"thickness): {error}"
            )
        else:
            if not contains_polygon(inner, protected_in_room):
                gap = "WALL_INTRUDES_PROTECTED_REGION"
                gaps.append(
                    f"{gap}: the task's protected region is not inside the room's inner face, so "
                    "part of the task cell would stand in a wall or outside the room"
                )

    measured["vertex_count"] = len(polygon.vertices)
    measured["floor_area_m2"] = round(polygon.area, 6)
    measured["area_ratio_to_bounding_box"] = round(polygon.area_ratio_to_bounding_box(), 6)
    measured["height_m"] = shell.height_m
    measured["openings"] = [opening.opening_id for opening in shell.openings]

    x_bounds, y_bounds, z_bounds = spec.room.size_bounds_m
    min_x, min_y, max_x, max_y = polygon.bounds()
    measured["footprint_m"] = [round(max_x - min_x, 6), round(max_y - min_y, 6)]

    # The footprint must be within the requested size bounds, and the floor area must be within
    # what those bounds could enclose. A bounding-box reduction would pass the first and fail
    # the second.
    if not (x_bounds.contains(max_x - min_x) and y_bounds.contains(max_y - min_y)):
        gaps.append(
            f"room footprint {measured['footprint_m']} is outside the requested bounds "
            f"x={x_bounds.to_list()} y={y_bounds.to_list()}"
        )
    max_area = x_bounds.maximum * y_bounds.maximum
    if polygon.area > max_area + 1e-9:
        gaps.append(
            f"floor area {polygon.area:.3f} m2 exceeds the maximum {max_area:.3f} m2 implied by "
            "the requested bounds"
        )
    if not z_bounds.contains(shell.height_m):
        gaps.append(
            f"height {shell.height_m} m is outside the requested bounds {z_bounds.to_list()}"
        )

    # The shape the program asked for, against the shape that came out. `options=None` means no
# shape was declared, so there is nothing to compare against -- and that is recorded rather than
# quietly treated as a pass. This used to default to `ShellOptions(notch_size_m=None)`, which
# meant the check silently weakened to nothing whenever a caller omitted the options.
    resolved = options
    if resolved is None:
        measured["shape_checked"] = False
    else:
        measured["shape_checked"] = True
        measured["family"] = resolved.family.value
        if resolved.family is not Family.RECT and polygon.is_rectangle():
            gaps.append(
                f"OUTLINE_DOES_NOT_REALISE_THE_REQUESTED_SHAPE(family={resolved.family.value}, "
                f"vertices={len(polygon.vertices)}, "
                f"expected_vertices={resolved.family.expected_vertices}, area_ratio=1.0): the "
                "outline is a rectangle, so the shape was reduced to its bounding box, which "
                "will pass containment checks the real room would fail"
            )
        elif resolved.family is not Family.RECT and (
            len(polygon.vertices) != resolved.family.expected_vertices
        ):
            gaps.append(
                f"OUTLINE_DOES_NOT_REALISE_THE_REQUESTED_SHAPE(family={resolved.family.value}, "
                f"vertices={len(polygon.vertices)}, "
                f"expected_vertices={resolved.family.expected_vertices}, "
                f"area_ratio={polygon.area_ratio_to_bounding_box():.4f})"
            )
        # The same notch rules the drawer obeys, applied here as well. This validator builds its
        # polygon straight from `polygon_xy`, so the checks inside `_outline` never ran for it and
        # a shell from anywhere else was judged on its vertices alone. Measured: a hand-built T
        # with a 0.1 m neck passed with `ok=True, gaps=()`.
        gaps.extend(_notch_gaps(
            resolved.width_m if resolved.width_m is not None else max_x - min_x,
            resolved.depth_m if resolved.depth_m is not None else max_y - min_y,
            resolved.family,
            resolved.notches,
        ))

    for opening in shell.openings:
        if distance_to_boundary(polygon, opening.center_xy) > 1e-6:
            gaps.append(f"opening {opening.opening_id} is not on the room boundary")

    limits = resolved or ShellOptions()
    minimum = limits.robot_footprint_m + limits.clearance_margin_m
    gaps.extend(check_door_connectivity(shell, minimum_width_m=minimum))
    measured["required_door_width_m"] = round(minimum, 4)

    return ShellValidation(ok=not gaps, gaps=tuple(gaps), measured=measured)
