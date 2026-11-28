# SPDX-License-Identifier: Apache-2.0
"""Furnishing a room: which pieces, where, and which way they face.

The count and the orientation rules follow SceneSmith's designer procedure, which is the part of
its pipeline worth porting: it states furniture counts per room area (<10 m2: 3-6, 10-20: 7-12,
>20: 12+) and sorts each piece into one of three orientation categories --

* **A, faces toward something** -- a chair turns to face the desk it belongs to;
* **B, functional front** -- a wardrobe or cabinet presents its front to the room, which means its
  back goes to the wall;
* **C, symmetric** -- a side table or a stool has no meaningful front and is left unrotated.

Everything here is deterministic: the plan is a declared list, walls are chosen by length and
occupancy, and positions come from the same lattice the layout solver already uses. No model decides
positions, which keeps a run reproducible and the placement reviewable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace as dataclass_replace
from enum import Enum
from typing import Any, Sequence

from rfw_w.scenes.contracts.scene_spec import ZoneKind
from rfw_w.scenes.contracts.units import Quat, Vec3
from rfw_w.scenes.solvers.materials import RoomMaterials
from rfw_w.scenes.geometry.polygon import (
    Polygon,
    contains_point,
    contains_polygon,
    interior_point,
    subtract_polygons,
)
from rfw_w.scenes.solvers.layout import (
    Footprint,
    LayoutError,
    OccupancyMap,
    SurfacePlacementConfig,
)

# SceneSmith's furniture counts, keyed to floor area. Ported as a table rather than paraphrased so
# the numbers can be compared against its designer prompt.
# A piece is set this far inside the wall it stands against. Rotating a footprint back to world
# axes leaves the trigonometric residue of cos(pi/2), which put a flush desk 0.05 mm outside the
# room. That is physically meaningless but not *numerically* meaningless: containment is tested at
# 1e-9, so a flush object reads as intruding. Nudging it in by a tenth of a millimetre keeps the
# placement strictly inside and leaves every check at its real tolerance -- the alternative,
# loosening containment, would hide a genuine centimetre of intrusion too.
WALL_INSET_M = 1e-4


FURNITURE_COUNT_BY_AREA = (
    (10.0, 3, 6),
    (20.0, 7, 12),
    (float("inf"), 12, 18),
)


class Orientation(str, Enum):
    """Which way a piece faces. The names match SceneSmith's A/B/C categories."""

    FACES_TOWARD = "faces_toward"   # category A
    FRONT_TO_ROOM = "front_to_room"  # category B: back against a wall
    SYMMETRIC = "symmetric"          # category C: no meaningful front


class PlacementKind(str, Enum):
    """Where a piece goes, before orientation is applied.

    WALL_HUNG, CEILING and FLOOR_COVERING are not variations on AGAINST_WALL: each ignores the
    floor occupancy map entirely, because a picture occupies wall, a lamp occupies ceiling, and a
    rug lies *under* the furniture rather than competing with it for floor space.
    """

    AGAINST_WALL = "against_wall"
    BESIDE = "beside"
    CORNER = "corner"
    ANYWHERE = "anywhere"
    WALL_HUNG = "wall_hung"
    CEILING = "ceiling"
    FLOOR_COVERING = "floor_covering"


@dataclass(frozen=True, slots=True)
class FurnitureRequest:
    """One piece in the plan."""

    entity_id: str
    synset: str
    kind: PlacementKind
    orientation: Orientation
    hard_required: bool = False
    faces_entity: str | None = None
    target_extent_m: float = 1.2
    optional: bool = True
    # Which retrieval candidate to materialise, and how many of the solver's own candidates to skip
    # before it starts looking. Both are plan decisions rather than solver internals -- they are
    # what an L0 resample and an L2 asset swap actually change -- and both are recorded so a run
    # can be read back to see which candidate was in use when a gate failed.
    asset_index: int = 0
    skip_candidates: int = 0
    # Which zone constrains where this piece may stand. It lives on the request rather than in a
    # table beside the script, because the table was keyed by entity id and only held the nine rows
    # the M1 plan named -- so an entity that a replan *added* had no row and the build raised
    # KeyError on it. A plan that cannot say where a piece belongs cannot be repaired either.
    # None means "any zone", which is what the placement already does when it is not told.
    zone_kind: ZoneKind | None = None

    def __post_init__(self) -> None:
        if not self.entity_id or not self.synset:
            raise LayoutError("furniture request needs an entity id and a synset")
        if self.orientation is Orientation.FACES_TOWARD and not self.faces_entity:
            raise LayoutError(
                f"{self.entity_id} faces toward something but names nothing; category A without a "
                "referent is category C with extra steps"
            )
        # The two search parameters, validated here rather than in a second __post_init__ further
        # down: a later definition of the same method replaces the earlier one, so a split pair
        # leaves the first silently unenforced. That happened once in this file.
        if self.asset_index < 0:
            raise LayoutError(f"{self.entity_id} cannot use candidate {self.asset_index}")
        if self.skip_candidates < 0:
            raise LayoutError(f"{self.entity_id} cannot skip {self.skip_candidates} candidates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id, "synset": self.synset, "kind": self.kind.value,
            "orientation": self.orientation.value, "hard_required": self.hard_required,
            "faces_entity": self.faces_entity, "target_extent_m": self.target_extent_m,
            "optional": self.optional, "asset_index": self.asset_index,
            "skip_candidates": self.skip_candidates,
            "zone_kind": None if self.zone_kind is None else self.zone_kind.value,
        }


def furniture_target_count(floor_area_m2: float) -> tuple[int, int]:
    """The scene's furniture target for a floor area, as ``(minimum, ideal)``.

    The upper bound is the ideal rather than another minimum: a room with more wall than the ideal
    count can fill does not need to be padded, and padding a room to hit a number is how scenes end
    up with furniture nobody asked for.
    """
    for threshold, low, high in FURNITURE_COUNT_BY_AREA:
        if floor_area_m2 < threshold:
            return low, high
    return 12, 18


@dataclass(frozen=True, slots=True)
class SurfaceRequest:
    """One object in the plan that goes on a support surface rather than the floor."""

    entity_id: str
    synset: str
    target_width_m: float
    # Which piece it stands on, and which retrieval candidate it uses. The support is named rather
    # than implied because "put it somewhere else" is exactly what an L3 repair does, and a plan
    # that cannot say where a thing goes cannot be repaired to put it elsewhere.
    #
    # Required, with no default. It defaulted to "desk" -- the last hardcode inside the solver --
    # which meant a spec with no desk in it produced a plan that silently stood the notebook on a
    # piece that was not in the scene. The docstring above already said the support must be named;
    # the default was the one place that did not.
    support_entity: str
    asset_index: int = 0

    def __post_init__(self) -> None:
        if not self.entity_id or not self.synset:
            raise LayoutError("a surface request needs an entity id and a synset")
        if self.target_width_m <= 0:
            raise LayoutError(f"{self.entity_id} must have a positive width")
        if not self.support_entity:
            raise LayoutError(f"{self.entity_id} must name the piece it stands on")
        if self.asset_index < 0:
            raise LayoutError(f"{self.entity_id} cannot use candidate {self.asset_index}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id, "synset": self.synset,
            "target_width_m": self.target_width_m, "support_entity": self.support_entity,
            "asset_index": self.asset_index,
        }


@dataclass(frozen=True, slots=True)
class SupportRequest:
    """The surface the task is performed on, which is **built** rather than retrieved.

    A plan row of its own because the support is no longer a piece of furniture somebody else made.
    It has no synset and no candidate index, so it cannot honestly be a ``FurnitureRequest``; it has
    a *form*, which the build turns into geometry at the dimensions the room-free core solves.

    ``target_extent_m`` is carried for the same reason the furniture rows carry one -- it is the
    plan's record of the size the support is meant to be, so a plan changed by a repair is a plan
    whose record of the support changed with it. The exact width, depth and height are the core's
    decision, not the plan's.
    """

    entity_id: str
    shape: str
    target_extent_m: float
    # Where it goes before it is oriented, carried here rather than looked up at build time so that
    # the placement solver can treat a built support exactly like a retrieved one: it reads `kind`,
    # `entity_id` and `faces_entity` off whatever it is handed, and nothing else.
    kind: PlacementKind = PlacementKind.ANYWHERE
    orientation: Orientation = Orientation.FRONT_TO_ROOM
    # A support is never placed to face another piece -- it is what the others are placed against.
    faces_entity: str | None = None

    def __post_init__(self) -> None:
        if not self.entity_id or not self.shape:
            raise LayoutError("a support request needs an entity id and a shape")
        if self.target_extent_m <= 0:
            raise LayoutError(f"{self.entity_id} must have a positive target extent")
        if self.faces_entity is not None:
            raise LayoutError(
                f"{self.entity_id} is a built support and cannot face another piece; it is what "
                "the others are placed against"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "shape": self.shape,
            "target_extent_m": self.target_extent_m,
            "kind": self.kind.value,
            "orientation": self.orientation.value,
        }


@dataclass(frozen=True, slots=True)
class AppearanceSettings:
    """Lighting the renderer is asked for. Material changes ride the same repair.

    Kept as plain numbers rather than a light rig, because the appearance repair's job is to move
    them and the renderer's job is to honour them. A critic that says the room reads flat is asking
    for a different *number*, not for a different scene, and keeping that difference visible is what
    stops an appearance complaint being answered by moving furniture.
    """

    # Three-room reviews found clipped sun patches and deep shadows with 900/2600/700.
    # Retain diffuse fill while reducing the direct component; measured again by the critic.
    dome_intensity: float = 650.0
    sun_intensity: float = 650.0
    ceiling_fill_intensity: float = 500.0
    dome_color: tuple[float, float, float] = (0.92, 0.94, 1.0)
    sun_color: tuple[float, float, float] = (1.0, 0.96, 0.88)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dome_intensity": self.dome_intensity,
            "sun_intensity": self.sun_intensity,
            "ceiling_fill_intensity": self.ceiling_fill_intensity,
            "dome_color": list(self.dome_color),
            "sun_color": list(self.sun_color),
        }

@dataclass(frozen=True, slots=True)
class ScenePlan:
    """The whole placement plan as a value, so a repair can hand back a changed one.

    Before this existed the plan was three module-level literals read directly by the furnishing
    loop, which meant no repair could change a decision -- only re-run the same decisions and hope
    the answer differed. A loop whose atoms cannot change the input is not a loop.
    """

    furniture: tuple[FurnitureRequest, ...] = ()
    surfaces: tuple[SurfaceRequest, ...] = ()
    # The support the task is performed on. A third row kind rather than a field, because it is
    # placed like a row -- it occupies floor, it is in the scene version, and a repair that changed
    # which form it is would have to change this. It is *not* a `FurnitureRequest` because there is
    # no library behind it.
    supports: tuple[SupportRequest, ...] = ()
    appearance: AppearanceSettings = AppearanceSettings()
    # The shell's surfaces, from the spec's palette. A sibling of `appearance` rather than a field
    # inside it: `AppearanceSettings` is documented as "lighting the renderer is asked for", and a
    # material is not a light. It rides the same `repair.appearance` rung, which is the only repair
    # that can move a surface without moving furniture -- and until this existed that rung changed
    # the plan, changed the scene version, invalidated the render and the critique, and produced an
    # identical image, because nothing read what it changed.
    room_materials: RoomMaterials = RoomMaterials()
    # The three placement constants, which are plan decisions rather than solver internals: they are
    # exactly what an L1 repair changes when a piece is not where it should be.
    mount_height_m: float = 1.45
    ceiling_clearance_m: float = 0.05
    wall_hang_gap_m: float = 0.02
    # How far off the line a wall was built from a piece stands. The room polygon is the wall's
    # *outer* face and the wall body extends inward, so this is the wall thickness for a piece to
    # sit against the surface a person can see. It is a plan value and it defaults to zero, which
    # is what every delivery before this change effectively used: the solver stands a piece off by
    # its own half-depth alone, and the piece's back then sits one wall thickness inside the wall
    # solid. From inside the room that is indistinguishable from standing against the wall, which
    # is why it survived until a gate was written that measures the inner face.
    wall_face_offset_m: float = 0.0
    # The gap the placement solver keeps between pieces.
    object_gap_m: float = 0.12
    # The piece the task target stands on, and the target itself. Both are plan facts rather than
    # build arguments: they come from the spec's own task and relations, an L3 re-support repair
    # would change the first, and the build reads them a dozen times -- as `scene["desk"]` and
    # `scene["cup_pos"]` before this. Empty means a plan built by hand rather than by
    # `plan_from_spec`, which is what the tests and the repair atoms construct.
    support_id: str = ""
    target_id: str = ""
    # Which retrieved candidate the task target uses. It is a plan fact for the same reason
    # `target_id` is one: the target is the entity a critic is most likely to complain about --
    # `prompt_adherence` is measured against it and admits exactly two repairs, one of which is
    # `swap_asset` -- and before this the candidate was a literal `[0]` in the build. A repair that
    # names the target therefore advanced nothing, which read as "the swap had no effect" rather
    # than as "the swap never happened".
    target_asset_index: int = 0
    # Every derivation decision that could not be honoured, named. A gap here is not a warning: it
    # is the record of a spec the pipeline could only partly obey, and it travels into the manifest
    # so a delivery cannot silently omit a piece the request asked for.
    derivation_gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = [request.entity_id for request in self.furniture]
        ids.extend(request.entity_id for request in self.surfaces)
        ids.extend(request.entity_id for request in self.supports)
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise LayoutError(f"the plan names {duplicates} more than once")
        for name in ("wall_face_offset_m", "object_gap_m", "ceiling_clearance_m",
                     "wall_hang_gap_m", "mount_height_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise LayoutError(f"{name} must be a non-negative finite distance, got {value}")
        # A named support the plan cannot place is a plan that fails later with a worse message, so it is
        # refused here. The target is the opposite case and is deliberately *not* required to be a
        # plan row: it arrives from the SimReady library with its own contract (dynamic, centre
        # origin, y_up_to_z_up), never through the HSSD furniture loop.
        if self.support_id and self.support_id not in ids:
            raise LayoutError(
                f"support_id is {self.support_id!r}, which the plan does not place; a plan that "
                f"names a support it cannot build is a plan that fails at build time instead of here"
            )
        if self.target_id and self.target_id in ids:
            raise LayoutError(
                f"target_id is {self.target_id!r}, which the plan places as furniture; the task "
                f"target comes from the rigid library, not from the plan, and a row for it would "
                f"claim an HSSD synset for an asset that is not from HSSD"
            )
        if self.target_id and not self.support_id:
            raise LayoutError("a plan cannot name a target without naming what it stands on")
        if self.target_asset_index < 0:
            raise LayoutError(
                f"the task target cannot use candidate {self.target_asset_index}; a negative index "
                f"is not a candidate, and the build would read it as one"
            )

    def to_dict(self) -> dict[str, Any]:
        """The plan's canonical form, which is what the scene version hashes.

        Complete on purpose: a field left out here is a field a repair can change without changing
        the scene version, and the loop would then commit it and leave every gate's evidence looking
        current for a scene it never measured.
        """
        return {
            "furniture": [request.to_dict() for request in self.furniture],
            "surfaces": [request.to_dict() for request in self.surfaces],
            "supports": [request.to_dict() for request in self.supports],
            "appearance": self.appearance.to_dict(),
            "room_materials": self.room_materials.to_dict(),
            "mount_height_m": self.mount_height_m,
            "ceiling_clearance_m": self.ceiling_clearance_m,
            "wall_hang_gap_m": self.wall_hang_gap_m,
            "wall_face_offset_m": self.wall_face_offset_m,
            "object_gap_m": self.object_gap_m,
            "support_id": self.support_id,
            "target_id": self.target_id,
            "target_asset_index": self.target_asset_index,
            "derivation_gaps": list(self.derivation_gaps),
        }

    def furniture_for(self, entity_id: str) -> FurnitureRequest | None:
        for request in self.furniture:
            if request.entity_id == entity_id:
                return request
        return None

    def surface_for(self, entity_id: str) -> SurfaceRequest | None:
        for request in self.surfaces:
            if request.entity_id == entity_id:
                return request
        return None

    def support_for(self, entity_id: str) -> SupportRequest | None:
        for request in self.supports:
            if request.entity_id == entity_id:
                return request
        return None

    @property
    def entity_ids(self) -> tuple[str, ...]:
        return tuple(
            [request.entity_id for request in self.furniture]
            + [request.entity_id for request in self.surfaces]
            + [request.entity_id for request in self.supports]
        )

    def replaced_furniture(self, request: FurnitureRequest) -> "ScenePlan":
        """A copy with one piece's request replaced, or added when it is not there yet."""
        found = self.furniture_for(request.entity_id)
        if found is None:
            return dataclass_replace(self, furniture=self.furniture + (request,))
        return dataclass_replace(
            self,
            furniture=tuple(
                request if existing.entity_id == request.entity_id else existing
                for existing in self.furniture
            ),
        )

    def without(self, entity_id: str) -> "ScenePlan":
        """A copy with one entity removed.

        Only an L5 replan may call this, and only for an entity the request marked optional. The
        check is in the replanner, not here: this is the mechanism, and the policy for using it
        belongs with the policy for everything else.
        """
        if self.furniture_for(entity_id) is None and self.surface_for(entity_id) is None:
            raise LayoutError(f"{entity_id} is not in the plan, so it cannot be removed")
        return dataclass_replace(
            self,
            furniture=tuple(item for item in self.furniture if item.entity_id != entity_id),
            surfaces=tuple(item for item in self.surfaces if item.entity_id != entity_id),
        )


@dataclass(frozen=True, slots=True)
class Wall:
    """One room edge, with the inward direction worked out once."""

    index: int
    start: tuple[float, float]
    end: tuple[float, float]
    length_m: float
    inward: tuple[float, float]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)

    @property
    def tangent(self) -> tuple[float, float]:
        """Unit vector along the wall, from its start vertex to its end."""
        return ((self.end[0] - self.start[0]) / self.length_m,
                (self.end[1] - self.start[1]) / self.length_m)

    def point_along(self, distance_m: float) -> tuple[float, float]:
        """A point on the wall's line, measured from its start vertex."""
        ux, uy = self.tangent
        return (self.start[0] + ux * distance_m, self.start[1] + uy * distance_m)

    def yaw_facing_into_room(self) -> float:
        """Yaw that turns a piece's front away from this wall, into the room.

        A piece's front is its local **+y** axis. That is the convention the whole furnishing
        solver already uses -- a footprint is ``(width, depth)``, so the axis that faces anything
        is the depth axis -- and this returned ``atan2(inward)`` instead, which aims the piece's
        local +x, its *width*, into the room. A 1.37 m desk was therefore laid 1.37 m out from its
        wall, standing lengthwise into the room and facing along the wall. Nothing caught it: with
        no containment check to fail, the desk simply stood across the circulation corridor and
        every gate passed.

        ``atan2(-nx, ny)`` is the rotation that carries local +y onto the inward normal.
        """
        return math.atan2(-self.inward[0], self.inward[1])

    def mount_rotation(self) -> Quat:
        """The rotation that stands a flat, wall-hung piece up on this wall.

        A yaw is enough for a piece that already stands on its own base, because its height is
        already along world z. A wall-hung piece does not: the prepared mesh for the picture used
        here measures 0.70 x 1.00 x 0.005, so its 5 mm axis is vertical and a yaw alone leaves it
        a horizontal slab hanging in the air. Measured on a real delivery, that slab also reached
        0.13 m through the wall.

        Afterwards local +x lies along the wall, local +y is world up, and local +z points into the
        room -- so the panel's extent *across* the room is its thickness, and its origin sits on
        the back face that goes against the wall.

        Built by composing two rotations rather than from a matrix, because :class:`Quat` has no
        ``from_matrix3``: a quarter turn about X stands the piece up (local +y becomes world +Z),
        and a yaw about Z then turns it onto this wall. The determinant is ``cos^2 + sin^2 = 1``
        for every wall, so no sign special case is needed.

        The along-wall direction is ``e_z x inward`` and deliberately **not** ``self.tangent``.
        The stored tangent follows the outline's winding, so it reverses when the same wall is
        written the other way round -- and with it the rotation, mirroring the piece about the
        wall plane. ``e_z x inward`` is fixed by the inward normal alone, which is the thing that
        actually decides how a piece on this wall faces, so the same wall gives the same answer
        whichever way its edge was walked.
        """
        along = (-self.inward[1], self.inward[0])
        psi = math.atan2(along[1], along[0])
        return Quat.from_yaw(psi) * Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), math.pi / 2.0)




def openings_on_wall(
    wall: "Wall", openings: Sequence[tuple[float, float, float]], *, tolerance_m: float = 1e-6
) -> tuple[tuple[float, float, str], ...]:
    """The spans of this wall covered by an opening, as ``(start_along, end_along, opening_id)``.

    ``openings`` are ``(opening_id, centre_x, centre_y, width)``. The opening's projection onto the
    wall is used rather than its distance to it, because a door and a picture frame both sit on the
    same wall and the question is whether their spans collide along it.
    """
    found: list[tuple[float, float, str]] = []
    ux = (wall.end[0] - wall.start[0]) / wall.length_m
    uy = (wall.end[1] - wall.start[1]) / wall.length_m
    for opening_id, cx, cy, width in openings:
        dx, dy = cx - wall.start[0], cy - wall.start[1]
        along = dx * ux + dy * uy
        # Perpendicular distance to the wall's line; anything off the wall is not on it.
        perpendicular = abs(dx * (-uy) + dy * ux)
        if perpendicular > tolerance_m or not (0.0 <= along <= wall.length_m):
            continue
        found.append((along - width / 2, along + width / 2, opening_id))
    return tuple(sorted(found))


def wall_span_is_clear(
    wall: Wall, start_m: float, end_m: float, openings: Sequence[tuple[float, float, str]]
) -> bool:
    """Whether a stretch of wall is free of every opening, with a margin at each end."""
    for opening_start, opening_end, _ in openings:
        if end_m <= opening_start or start_m >= opening_end:
            continue
        return False
    return True


def room_walls(polygon: Polygon) -> tuple[Wall, ...]:
    """Every edge, with the inward normal resolved against the actual polygon.

    The normal is checked by stepping a little way in and testing containment rather than inferred
    from winding order: a reversed outline would otherwise put every piece of furniture outside the
    room, pressed against the far side of the wall.
    """
    walls: list[Wall] = []
    count = len(polygon.vertices)
    for index in range(count):
        start = polygon.vertices[index]
        end = polygon.vertices[(index + 1) % count]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        # Candidate normal; if stepping along it leaves the room, use the other one.
        nx, ny = -dy / length, dx / length
        probe = (
            (start[0] + end[0]) / 2 + nx * 0.15,
            (start[1] + end[1]) / 2 + ny * 0.15,
        )
        if not contains_point(polygon, probe):
            nx, ny = -nx, -ny
        walls.append(Wall(index, start, end, length, (nx, ny)))
    return tuple(walls)


@dataclass(frozen=True, slots=True)
class PlacementRejection:
    """One candidate the wall sweep refused, and why.

    The sweep has always discarded these: it collected ``best_rejections`` and then returned
    ``None``, so a piece that could not be placed said nothing about where it was tried or how
    close each candidate came. That is the difference between a failure a repair can act on and a
    failure a repair can only retry, and it is why the escalation ladder had nothing to escalate.
    """

    centre: tuple[float, float]
    travelled_m: float
    reason: str
    required_margin_m: float = 0.0
    measured_margin_m: float = 0.0

    @property
    def shortfall_m(self) -> float:
        """How much clearance the best candidate was short by. The residual §10 measures progress
        with, and zero for a candidate refused for leaving its zone rather than for colliding."""
        return max(0.0, self.required_margin_m - self.measured_margin_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "centre": list(self.centre), "travelled_m": self.travelled_m, "reason": self.reason,
            "required_margin_m": self.required_margin_m,
            "measured_margin_m": self.measured_margin_m,
            "shortfall_m": self.shortfall_m,
        }


@dataclass(frozen=True, slots=True)
class WallPlacement:
    """The sweep's result: a placement, or the record of every candidate that failed."""

    placement: Any | None
    rejections: tuple[PlacementRejection, ...] = ()

    @property
    def placed(self) -> bool:
        return self.placement is not None


def within_union(
    candidate: Polygon, regions: Sequence[Polygon], *, tolerance: float = 1e-6
) -> bool:
    """Whether ``candidate`` lies inside the *union* of ``regions``.

    A published zone is a set of pieces, not one outline. Every reservation is subtracted with
    :func:`~rfw_w.scenes.geometry.polygon.subtract_polygons`, which returns disjoint convex parts that
    **share edges** -- measured on the office at 7 pieces for a 1.37 m2 storage strip and 14 for a
    2.37 m2 ambient one, once the task's floor is cut out of the room. Testing
    ``contains_polygon(piece, candidate)`` piece by piece therefore refuses a footprint that
    straddles two touching parts of the region it was assigned, and that is most of them: the office
    storage wall sweep has 11 positions a 0.95 x 0.50 cabinet could legally occupy by union and
    **0** by single piece.

    The single-piece test is tried first because it is exact and cheap, and it is the whole answer
    whenever a piece happens to hold the footprint. The fallback subtracts each region from what is
    left of the candidate and measures the remainder, rather than summing intersections: regions can
    in principle overlap each other, and a sum would then count the same square metre twice and
    accept a footprint that is not in the union at all.
    """
    if not regions:
        return True
    if any(contains_polygon(region, candidate) for region in regions):
        return True

    remaining: list[Polygon] = [candidate]
    for region in regions:
        carved: list[Polygon] = []
        for piece in remaining:
            carved.extend(subtract_polygons(piece, region))
        remaining = carved
        if not remaining:
            break
    return sum(piece.area for piece in remaining) <= tolerance


def place_against_wall(
    *,
    instance_id: str,
    entity_id: str,
    footprint: Footprint,
    wall: Wall,
    occupancy: OccupancyMap,
    object_height_m: float,
    wall_face_offset_m: float,
    config: SurfacePlacementConfig | None = None,
    yaw: float | None = None,
    allowed: Sequence[Polygon] = (),
    preferred_xy: tuple[float, float] | None = None,
) -> WallPlacement:
    """Slide a piece along a wall until its footprint fits.

    The piece is pushed out from the wall by half its own depth, so its back sits flush and its
    whole footprint stays inside the room. Sweeping along the wall is what makes a second piece land
    beside the first rather than on top of it.

    ``wall_face_offset_m`` is how far the wall's *usable* face is from the line this wall was built
    from, and it is a required argument rather than a defaulted one because getting it wrong is
    invisible from inside the room. The room polygon is the wall's outer face; its body extends
    inward, so the surface a person sees is one wall thickness inside the line. Standing a piece off
    by its own half-depth alone therefore buries its back in the wall solid by exactly the wall
    thickness -- and since a piece flush to the outline looks identical, from every camera inside,
    to a piece flush to the wall, nothing caught it. The wall-hung branch had always compensated;
    the floor branches had not.

    ``allowed`` confines the sweep to the region the zone plan assigned this piece. Empty means no
    zone constraint, which is what a caller that has no plan passes; the zone is a *permission*,
    not an obstacle, so it is tested as containment rather than added to the clearance obstacles --
    otherwise the boundary of the zone a piece belongs in would read as something to keep away from.
    """
    settings = config or SurfacePlacementConfig(step_m=0.05, object_gap_m=0.12)
    # Both extents are measured *along the wall's own axes*, not read off the footprint's fields:
    # which half-extent is the depth depends on whether the wall is horizontal, and which is the
    # along-wall length depends on its direction.
    depth = footprint.half_extent_along(wall.inward)
    along_half = footprint.half_extent_along(wall.tangent)
    stand_off = wall_face_offset_m + depth + WALL_INSET_M

    rejections: list[PlacementRejection] = []
    step = settings.step_m
    # Inset from the wall's end as well as from its face: starting exactly one half-extent along
    # leaves the piece's corner on the wall junction, where the same trigonometric residue makes it
    # read as outside the room in the other axis. The wall's *ends* are its own face too -- the
    # junction with the neighbouring wall is at the outline -- so the same offset applies lengthwise.
    start = along_half + wall_face_offset_m + WALL_INSET_M
    count = max(0, math.floor((wall.length_m - along_half - wall_face_offset_m - start + 1e-9) / step) + 1)
    positions = [start + index * step for index in range(count)]
    if preferred_xy is not None:
        positions.sort(key=lambda distance: math.dist(wall.point_along(distance), preferred_xy))
    for travelled in positions:
        along = wall.point_along(travelled)
        centre = (along[0] + wall.inward[0] * stand_off, along[1] + wall.inward[1] * stand_off)
        candidate = footprint.at(centre[0], centre[1])
        if not within_union(candidate, allowed):
            rejections.append(PlacementRejection(
                centre=centre, travelled_m=travelled, reason="outside-assigned-zone",
            ))
            continue
        blockers = occupancy.blockers_for(instance_id)
        from rfw_w.scenes.geometry.clearance import check_clearance

        clearance = check_clearance(candidate, obstacles=blockers, margin_m=settings.object_gap_m)
        if clearance.ok:
            from rfw_w.scenes.solvers.layout import Placement

            return WallPlacement(
                placement=Placement(
                    instance_id=instance_id, entity_id=entity_id, footprint=footprint,
                    position_m=Vec3(
                        centre[0], centre[1], object_height_m / 2.0 if object_height_m else 0.0
                    ),
                    yaw_rad=yaw if yaw is not None else wall.yaw_facing_into_room(),
                    surface_id=None,
                ),
                rejections=tuple(rejections),
            )
        rejections.append(PlacementRejection(
            centre=centre, travelled_m=travelled, reason=clearance.reason,
            required_margin_m=clearance.required_margin_m,
            measured_margin_m=clearance.measured_margin_m,
        ))
    return WallPlacement(placement=None, rejections=tuple(rejections))


def yaw_toward(source: tuple[float, float], target: tuple[float, float]) -> float:
    """Yaw that turns a piece's front -- its local +y -- toward a point. Category A, made concrete.

    The same front-axis convention as :meth:`Wall.yaw_facing_into_room`: the piece faces the
    target with its depth axis, not with the axis it is widest along. A chair turned by
    ``atan2(dy, dx)`` had its side to the desk it was placed for.
    """
    return math.atan2(-(target[0] - source[0]), target[1] - source[1])


def yaw_into_room_from_corner(polygon: Polygon, position: tuple[float, float]) -> float:
    """Yaw pointing from a corner toward the room's interior. Used for lamps and tall pieces."""
    centre = interior_point(polygon, margin=0.5)
    return yaw_toward(position, centre)
