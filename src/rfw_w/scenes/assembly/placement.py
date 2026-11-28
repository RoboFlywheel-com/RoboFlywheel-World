# SPDX-License-Identifier: Apache-2.0
"""Task-first asset binding and measured placement."""

from __future__ import annotations
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.unplaced import UnplacedItem, split_unplaced
from rfw_w.scenes.contracts.scene_spec import SceneSpec, ZoneKind
from rfw_w.scenes.contracts.units import Quat
from rfw_w.scenes.contracts.units import Vec3
from rfw_w.scenes.geometry.polygon import Polygon, PolygonError, contains_polygon, inset, interior_point
from rfw_w.scenes.solvers.layout import FLOOR_KEEPOUT_PREFIX, Footprint, OccupancyMap, SurfacePlacementConfig, place_on_surface
from rfw_w.scenes.geometry.clearance import check_clearance
from rfw_w.scenes.solvers.furnishing_plan import plan_from_spec, profile_for
from rfw_w.scenes.solvers.furnish import AppearanceSettings, SupportRequest, Wall, FurnitureRequest, PlacementKind, ScenePlan, furniture_target_count, openings_on_wall, place_against_wall, room_walls, wall_span_is_clear, within_union
from rfw_w.scenes.orchestrator.driver import RepairUnavailable
from rfw_w.scenes.solvers.room import RoomError, fit_room_around_cell, validate_shell
from rfw_w.scenes.solvers.room_plan import plan_from_fitted
from rfw_w.scenes.solvers.support import PlacedInstance, Surface
from rfw_w.scenes.solvers.zones import Zone, plan_zones
from rfw_w.scenes.contracts.asset import AffordanceStatus
from rfw_w.scenes.runtime import _cache_file_token, step
from dataclasses import field

FLOOR_COMPETING_KINDS = (PlacementKind.AGAINST_WALL, PlacementKind.BESIDE, PlacementKind.CORNER, PlacementKind.ANYWHERE)
MOUNT_HEIGHT_M = 1.45
CEILING_CLEARANCE_M = 0.05
WALL_HANG_GAP_M = 0.02

@dataclass
class AssemblyAudit:
    placement_rejections: list[dict[str, Any]] = field(default_factory=list)
    size_rejections: list[dict[str, Any]] = field(default_factory=list)
    zone_fallbacks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildState:
    retrieval_engines: dict[tuple[str, ...], Any] = field(default_factory=dict)
    retrieval_results: dict[Any, Any] = field(default_factory=dict)

class CoreUnplaced(SystemExit):
    """A core placement failure; remains outside the loop's generic exception handler."""

    def __init__(self, item: UnplacedItem):
        self.item = item
        super().__init__(f"{item.entity_id}: {item.reason}: {item.detail}")


def build_plan(
    spec: SceneSpec,
    *,
    wall_face_offset_m: float = 0.0,
    object_gap_m: float = 0.12,
    appearance: AppearanceSettings | None = None,
) -> ScenePlan:
    """The plan for a spec, with the repair-facing knobs left open.

    ``wall_face_offset_m`` defaults to **zero**, which is the value every delivery before the loop
    existed effectively used. The room polygon is the wall's *outer* face and the wall body extends
    inward, so a piece standing off by its own half-depth alone has its back one wall thickness
    inside the wall solid -- the desk, the storage, the side table, the lamp and the plant all did.
    That was invisible from every camera inside the room, because a piece flush to the outline looks
    exactly like a piece flush to the wall. It is left at zero here on purpose: it is a real defect,
    it is what the ``geometry.in_bounds`` gate now measures, and repairing it is what the escalation
    ladder's L1 rung does. Setting it to the wall thickness here would hide the defect and the loop
    would have nothing to demonstrate.

    ``appearance`` is the lighting the renderer is asked for. It rides the plan because the
    appearance track's repair changes it, and a repair that cannot change its input is not a repair.

    ``hard_required`` now comes from the spec's own flag. It used to come from a hardcoded
    ``{"desk", "cup"}`` that deliberately disagreed with the spec, because the spec marked the chair
    and the side table as required context while the loop must be free to drop them -- with
    everything hard, the cheapest way to clear any deadlock is to delete the furniture. That
    disagreement is now expressed where it belongs, in the spec itself.
    """
    return plan_from_spec(
        spec,
        wall_face_offset_m=wall_face_offset_m,
        object_gap_m=object_gap_m,
        appearance=appearance,
        mount_height_m=MOUNT_HEIGHT_M,
        ceiling_clearance_m=CEILING_CLEARANCE_M,
        wall_hang_gap_m=WALL_HANG_GAP_M,
    )


def _record_rejections(entity_id: str, wall_index: int, rejections: Sequence[Any], audit: AssemblyAudit) -> None:
    for rejection in rejections:
        audit.placement_rejections.append({
            "entity_id": entity_id, "wall_index": wall_index, **rejection.to_dict(),
        })


def _most_constrained_first(
    requests: Sequence[FurnitureRequest],
) -> list[FurnitureRequest]:
    """The plan's furniture, the largest piece first.

    The plan's order is the spec's entity order, which is a statement about the *request* and not
    about the room -- and the loop consumed it as a placement priority. Measured on the office, that
    put the rug -- the largest and most constrained piece in the scene, 2.2 m -- fifth of eight, so
    every smaller and freer piece had already taken the floor it needed by the time it was tried.
    Same on the bedroom, where it is fourth of six.

    ``target_extent_m`` is the plan's own declared width for the piece, so this is a statement the
    plan already makes rather than a new one. Ties break on ``entity_id`` so the order is a function
    of the plan and not of the spec's file order.

    "Most constrained first" is the rule; size is how it is read here. A piece confined to a zone is
    *also* more constrained than one that may stand anywhere, and that is deliberately not part of
    the key: measured on the office, the storage zone can hold exactly one of the two pieces that
    want it, so ordering by zone changes which of the two loses and nothing else.
    """
    return sorted(requests, key=lambda request: (-request.target_extent_m, request.entity_id))


def _arena_name(kind: PlacementKind, zone_kind: ZoneKind | None) -> str:
    """The arena a placement actually searched, named from what was *applied*.

    `FLOOR_COVERING` and `CEILING` ignore the zone entirely (`FLOOR_COMPETING_KINDS` is what decides
    `allow`), so a rug that found nowhere to lie was reported as having "no position in the ambient
    zone" -- naming a constraint that was never in effect. A message pointing at the wrong arena
    sends the next reader to the wrong place, which is the same class of defect as a gate that
    cannot fail.

    Derived from the same set that decides the permission, so the two cannot drift apart.
    """
    if kind in FLOOR_COMPETING_KINDS and zone_kind is not None:
        return f"{zone_kind.value} zone"
    return "the room"


def _zone_could_hold(zone: Any, footprint: Footprint, *, step_m: float = 0.1) -> bool:
    """Whether any position in ``zone`` holds this footprint, ignoring everything else.

    A *capacity* test, not a fit test, and the distinction is load-bearing: it asks only whether the
    zone was ever able to hold the piece, so that a piece refused by a zone that could have held it
    stays refused. No clearance, no other pieces, and a coarse grid on purpose -- the answer needed
    is "could this zone ever hold it", not "where".
    """
    for region in zone.polygons:
        min_x, min_y, max_x, max_y = region.bounds()
        steps_x = int((max_x - min_x) / step_m) + 1
        steps_y = int((max_y - min_y) / step_m) + 1
        for index_x in range(steps_x + 1):
            for index_y in range(steps_y + 1):
                centre = (min_x + index_x * step_m, min_y + index_y * step_m)
                if within_union(footprint.at(centre[0], centre[1]), zone.polygons):
                    return True
    return False


@dataclass
class Furnished:
    """One placed instance, in the form both the worker request and the manifest need."""

    instance_id: str
    entity_id: str
    visual_uri: str
    collision_uri: str
    position_m: Any
    yaw_rad: float
    mass_kg: float | None
    dynamic: bool
    normalisation: str
    dimensions_m: Any
    provider: str
    library_version: str
    asset_id: str
    source_hash: str = ""
    license: str = ""
    mass_provenance: str = ""
    # The full orientation, and authoritative over ``yaw_rad`` when it is set. A wall-hung piece
    # is not merely turned about Z -- it has to be stood up -- so a yaw cannot express where it is,
    # and the worker request, the manifest and the geometry gates all need the real rotation.
    rotation_wxyz: tuple[float, float, float, float] | None = None
    # Where the mesh's own origin sits. Not the same across libraries: the prepared HSSD meshes
    # are centred in x and y with their base at z=0, while the retrieved SimReady cup is centred in
    # all three. A gate that assumes one convention reads a floating object as resting on its
    # support, so the convention is recorded next to the pose rather than assumed at the gate.
    origin_at_base: bool = True
    local_bounds_min_m: Vec3 | None = None
    local_bounds_max_m: Vec3 | None = None
    rigid_physics: dict | None = None
    collision_realization: str = ""
    source_collision_hull_count: int = 0

    def geometry_centre_m(self) -> Vec3:
        """World centre of the measured mesh bounds, respecting the recorded asset origin."""
        if self.local_bounds_min_m is None or self.local_bounds_max_m is None:
            return self.position_m
        local = (self.local_bounds_min_m + self.local_bounds_max_m).scale(0.5)
        rotation = Quat(*(self.rotation_wxyz or tuple(Quat.from_yaw(self.yaw_rad).to_list())))
        return self.position_m + rotation.rotate(local)

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "entity_id": self.entity_id, "provider": self.provider,
            "library_version": self.library_version, "asset_id": self.asset_id,
        }
        if self.mass_kg is not None:
            payload["mass_kg"] = self.mass_kg if self.rigid_physics is not None else round(self.mass_kg, 3)
        if self.mass_provenance:
            payload["mass_provenance"] = {"status": self.mass_provenance}
        if self.source_hash:
            payload["source_hash"] = self.source_hash
        if self.license:
            payload["license"] = self.license
        if self.rigid_physics is not None:
            payload["rigid_physics"] = self.rigid_physics
            payload["physics_realization"] = "mass_only_worker_inertia_estimated"
        if self.collision_realization:
            payload["collision_realization"] = self.collision_realization
            payload["source_collision_hull_count"] = self.source_collision_hull_count
        if self.local_bounds_min_m is not None:
            payload["local_bounds_min_m"] = self.local_bounds_min_m.to_list()
            payload["local_bounds_max_m"] = self.local_bounds_max_m.to_list()
        # Dimensions are recorded because a placement cannot be checked against an assumed size:
        # verifying a desk against 1.42 x 0.76 when the asset is 1.37 x 0.69 reports a wall
        # intrusion that is not there, and would hide one that is.
        payload["dimensions_m"] = [round(v, 4) for v in self.dimensions_m.to_list()]
        return payload


def _walls_named_by(zone: Zone | None, walls: Sequence["Wall"]) -> Sequence["Wall"]:
    """The single wall a zone was derived from, or every wall when there is no plan to follow.

    A zone that names a wall is a statement about *which* wall the piece belongs on -- the work
    strip was chosen for being the longest wall clear of the door, and letting the desk drift to
    another wall afterwards would throw that choice away.
    """
    if zone is None or zone.wall_index is None:
        return walls
    return [wall for wall in walls if wall.index == zone.wall_index] or list(walls)


def _corner_positions(wall_length_m: float, *, step_m: float = 0.2) -> tuple[float, ...]:
    """Where to try a corner piece along a wall: both ends first, then inward in steps.

    The two ends come first so a room with free corners is furnished exactly as it was before the
    zone plan existed; the sweep is the fallback for a zone whose region does not reach the corner
    it names.
    """
    near, far = 0.35, wall_length_m - 0.35
    positions = [near]
    if far > near + 1e-9:
        positions.append(far)
    travelled = near + step_m
    while travelled < far - 1e-9:
        positions.append(round(travelled, 6))
        travelled += step_m
    return tuple(positions)


def _floor_covering_sites(
    polygon: Polygon, footprint: "Footprint",
) -> tuple[tuple[float, float], ...]:
    """Candidate centres for a floor covering, on a grid set by the piece's own size.

    Coarse on purpose: a covering is the largest thing on the floor, so a fine lattice would spend
    its budget on positions that differ by less than the piece itself. Ordered from the shape's
    lower-left, so which site wins is a function of the geometry and not of iteration order.
    """
    min_x, min_y, max_x, max_y = polygon.bounds()
    half_x, half_y = footprint.half_extents
    step_x = max(0.2, half_x / 3.0)
    step_y = max(0.2, half_y / 3.0)
    steps_x = int((max_x - min_x) / step_x) + 1
    steps_y = int((max_y - min_y) / step_y) + 1
    return tuple(
        (min_x + index_x * step_x, min_y + index_y * step_y)
        for index_x in range(steps_x + 1)
        for index_y in range(steps_y + 1)
    )


def _furniture_placement(
    # Either a `FurnitureRequest` or the `SupportRequest` a built support arrives as; the body reads
    # only `kind`, `entity_id` and `faces_entity`, which both carry, so a constructed support is
    # placed by exactly the code that places retrieved furniture rather than by a second sweep that
    # would have to be kept in step with it.
    request: "FurnitureRequest | SupportRequest",
    footprint: "Footprint",
    height_m: float,
    *,
    polygon: Polygon,
    walls: Sequence["Wall"],
    occupancy: OccupancyMap,
    placed_by_entity: dict[str, Any],
    opening_spans: dict[int, tuple[tuple[float, float, str], ...]],
    room_height_m: float,
    wall_thickness_m: float,
    plan: ScenePlan,
    zone: Zone | None = None,
    preferred_xy: tuple[float, float] | None = None,
    audit: AssemblyAudit | None = None,
) -> tuple[Any, float, float, Quat] | None:
    """Place one piece, returning ``(polygon_position, z_m, yaw, rotation)`` or None.

    The rotation is authoritative and the yaw is kept alongside it because a yaw is still the
    right description for the pieces that only turn about Z -- which is all of them except a
    wall-hung one, whose stored yaw says which way it faces in plan and cannot say that it is
    upright.

    Wall-hung, ceiling and floor-covering pieces each bypass the floor occupancy map: a picture
    occupies wall, a pendant occupies ceiling, and a rug lies *under* the furniture. Testing them
    against floor space would reject every rug and every picture, since the floor is precisely the
    region they are allowed to share.

    ``zone`` is the region the zone plan assigned this piece, and it confines where the piece may
    stand. Returning None because the piece does not fit its zone is a real result: the caller
    reports it by name rather than falling back to a wall sweep, which would place the same piece
    somewhere the plan never allowed and say nothing about it.
    """
    audit = audit if audit is not None else AssemblyAudit()
    from rfw_w.scenes.solvers.furnish import yaw_into_room_from_corner, yaw_toward

    # Empty for a piece that does not compete for floor, and for a caller with no plan at all.
    allow = tuple(zone.polygons) if zone is not None and request.kind in FLOOR_COMPETING_KINDS else ()

    def permitted(candidate: Polygon) -> bool:
        """Whether a candidate footprint stands wholly inside the region the plan allows."""
        return within_union(candidate, allow)

    if request.kind is PlacementKind.WALL_HUNG:
        # A picture goes on the wall its zone names, when the plan named one: the display zone is
        # derived from a specific wall, and hanging the art anywhere else ignores that.
        # The zone's wall first, then the rest -- a preference, not a restriction. It was a
# restriction, and `zoned or walls` only fell back when the zone named *no* wall, so a
# picture whose wall had no clear span left was dropped rather than hung elsewhere. That
# was invisible while the shell had one window; with three, the office's display wall
# gained a window and all three art candidates went unplaced.
        candidates = walls
        if zone is not None and zone.wall_index is not None:
            zoned = [wall for wall in walls if wall.index == zone.wall_index]
            candidates = zoned + [wall for wall in walls if wall.index != zone.wall_index]
        for wall in candidates:
            spans = opening_spans.get(wall.index, ())
            rotation = wall.mount_rotation()
            span = footprint.width_m   # local x, which the mount lays along the wall
            thickness = height_m       # local z: what the piece measures across the room once hung
            tangent = wall.tangent
            travelled = 0.4
            while travelled + span <= wall.length_m - 0.4 + 1e-9:
                along = wall.point_along(travelled + span / 2)
                # What the hung panel covers on the floor: its width along the wall by its
                # thickness across it. Checked against the furniture already standing there,
                # because a picture hung behind a floor lamp is a real intersection -- measured
                # once the mount was fixed, the lamp's top (1.12 m) met the panel's lower edge
                # (0.95 m) with only 2 cm between the lamp and the wall.
                back = wall_thickness_m + plan.wall_hang_gap_m
                plan_centre = (
                    along[0] + wall.inward[0] * (back + thickness / 2.0),
                    along[1] + wall.inward[1] * (back + thickness / 2.0),
                )
                projection = Polygon.from_sequence([
                    (plan_centre[0] + tangent[0] * along_half + wall.inward[0] * inward_half,
                     plan_centre[1] + tangent[1] * along_half + wall.inward[1] * inward_half)
                    for along_half, inward_half in (
                        (span / 2.0, thickness / 2.0), (-span / 2.0, thickness / 2.0),
                        (-span / 2.0, -thickness / 2.0), (span / 2.0, -thickness / 2.0),
                    )
                ])
                if (
                    wall_span_is_clear(wall, travelled, travelled + span, spans)
                    and check_clearance(
                        projection, obstacles=occupancy.blockers_for(request.entity_id),
                        margin_m=0.05,
                    ).ok
                ):
                    # Four things the old form got wrong at once, all of them from reading a
                    # footprint that was still lying down and a yaw that could not stand it up.
                    #
                    # The rotation turns local +z onto +inward, so the panel grows *into* the room
                    # from wherever its origin is put, and its local z=0 face -- the one the
                    # prepared mesh's origin sits on -- is its back. The origin therefore belongs
                    # on the back face, and the thickness needs no allowance: it falls the right
                    # way on its own.
                    #
                    # The back has to clear the wall's inner face, not the room outline.
                    # `build_wall` offsets the wall body inward from the outline (`geometry.py`,
                    # "so an outline walked counter-clockwise yields walls inside the room"), so
                    # the outline is the wall's *outer* face and the surface a picture hangs on is
                    # one wall-thickness inside it. Hanging flush to the outline buries the panel
                    # in the wall solid, where nothing renders it and no plan-view gate can tell
                    # the difference.
                    #
                    # And the old code stood the piece off by half the *width* it had while lying
                    # down, which put its far edge 0.13 m through the wall.
                    #
                    # Local +y is the axis that stands up and the origin is that axis's centre, so
                    # the recorded z is the picture's vertical centre rather than its base -- which
                    # is why the mount height is used as it stands.
                    stand_off = wall_thickness_m + plan.wall_hang_gap_m
                    centre = (
                        along[0] + wall.inward[0] * stand_off,
                        along[1] + wall.inward[1] * stand_off,
                    )
                    return centre, plan.mount_height_m, wall.yaw_facing_into_room(), rotation
                travelled += 0.25
        return None

    if request.kind is PlacementKind.CEILING:
        centre = interior_point(polygon, margin=0.8)
        # Base-origin mesh, so the recorded z is the pendant's *lowest* point and the whole
        # hanging length comes off the ceiling. The centre-origin form halved it, and a 1.67 m
        # pendant under a 2.97 m ceiling then spanned 2.09..3.75 m: 0.78 m of it above the roof.
        # Nothing reported it -- the ceiling mesh is authored invisible, so no render showed the
        # violation either, and no gate measured a height at all.
        return centre, room_height_m - height_m - plan.ceiling_clearance_m, 0.0, Quat.from_yaw(0.0)

    if request.kind is PlacementKind.FLOOR_COVERING:
        # A rug is floor, so it has to respect the floor reservations -- which this branch did not.
        # It took the room's interior point and consulted no occupancy at all, so a rug went down
        # wherever the middle of the room happened to be. That was invisible while everything stood
        # against the walls. It is not invisible now: with the task's workspace reserved as a floor
        # keepout, a 2.44 x 1.52 m rug was laid across the desk *and* the floor the arm stands on,
        # and `function.task_reachable` then found no base position at all -- on a scene where every
        # other gate passed.
        #
        # The interior point is still tried first, so a room whose centre is free places its rug
        # exactly where it always did; the search is what happens when it is not.
        try:
            preferred = [interior_point(polygon, margin=max(footprint.half_extents) + 0.2)]
        except PolygonError:
            preferred = []
        for centre in (*preferred, *_floor_covering_sites(polygon, footprint)):
            candidate = footprint.at(*centre)
            if not contains_polygon(polygon, candidate):
                continue
            if check_clearance(
                candidate,
                obstacles=occupancy.blockers_for(request.entity_id),
                margin_m=plan.object_gap_m,
            ).ok:
                # 6 mm up: enough to keep a 15 mm rug out of the floor plane's z-fighting, small
                # enough that the rug still reads as lying on the floor.
                return centre, 0.006, 0.0, Quat.from_yaw(0.0)
        return None

    if request.kind is PlacementKind.ANYWHERE:
        # Freestanding supports (notably a kitchen island) need an interior search. ANYWHERE used
        # to fall through to the AGAINST_WALL branch, so its name was false and a valid island was
        # rejected because the wall sweep could not fit inside the post-corridor work polygons.
        # Search every allowed polygon deterministically and try both principal orientations.
        permission = allow or (polygon,)
        for region in permission:
            min_x, min_y, max_x, max_y = region.bounds()
            for yaw in (0.0, math.pi / 2.0):
                rotated = footprint.rotated(yaw)
                step = 0.1
                steps_x = int((max_x - min_x) / step) + 1
                steps_y = int((max_y - min_y) / step) + 1
                for index_x in range(steps_x + 1):
                    for index_y in range(steps_y + 1):
                        centre = (min_x + index_x * step, min_y + index_y * step)
                        candidate = rotated.at(*centre)
                        if not within_union(candidate, permission):
                            continue
                        if check_clearance(
                            candidate,
                            obstacles=occupancy.blockers_for(request.entity_id),
                            margin_m=plan.object_gap_m,
                        ).ok:
                            return centre, 0.0, yaw, Quat.from_yaw(yaw)
        return None

    if request.kind is PlacementKind.BESIDE:
        anchor = placed_by_entity.get(request.faces_entity or "")
        if anchor is None:
            return None
        target = (anchor.position_m.x, anchor.position_m.y)
        # Candidates are enumerated across the zone the chair was assigned, ranked by distance to
        # the piece it faces, so it sits as close to the desk as its own zone permits. A ring
        # drawn around the desk instead is the wrong search here: the seating band begins where the
        # work strip ends, and a ring sized for the desk's front lands inside the strip the desk
        # already owns, where nothing is permitted to stand.
        best: tuple[float, tuple[float, float], float] | None = None
        permission = allow or (polygon,)
        for region in permission:
            min_x, min_y, max_x, max_y = region.bounds()
            step = 0.1
            steps_x = int((max_x - min_x) / step) + 1
            steps_y = int((max_y - min_y) / step) + 1
            for index_x in range(steps_x + 1):
                for index_y in range(steps_y + 1):
                    centre = (min_x + index_x * step, min_y + index_y * step)
                    facing = yaw_toward(centre, target)
                    candidate = footprint.rotated(facing).at(centre[0], centre[1])
                    if not within_union(candidate, permission):
                        continue
                    clearance = check_clearance(
                        candidate, obstacles=occupancy.blockers_for(request.entity_id),
                        margin_m=0.10,
                    )
                    if not clearance.ok:
                        continue
                    distance = math.dist(centre, target)
                    if best is None or distance < best[0] - 1e-9:
                        best = (distance, centre, facing)
        if best is None:
            return None
        _, centre, facing = best
        return centre, 0.0, facing, Quat.from_yaw(facing)

    if request.kind is PlacementKind.CORNER:
        # Corners first, then anywhere along a wall: a lamp or plant in a corner is the intent, but
        # a room whose corners are taken should still get the light. The two true corners are tried
        # in their original order and the wall is then swept inward, which is what the sentence above
        # always claimed and the two fixed offsets never did -- and which a zone that stops short of
        # a corner (an ambient inset, a strip clipped by the corridor) needs in order to place at all.
        corner_walls = _walls_named_by(zone, walls)
        for wall in sorted(corner_walls, key=lambda w: -w.length_m):
            for travelled in _corner_positions(wall.length_m):
                corner_yaw = yaw_into_room_from_corner(polygon, wall.point_along(travelled))
                corner_footprint = footprint.rotated(corner_yaw)
                along = wall.point_along(travelled)
                stand_off = (
                    plan.wall_face_offset_m
                    + corner_footprint.half_extent_along(wall.inward) + 0.05 + 1e-4
                )
                centre = (
                    along[0] + wall.inward[0] * stand_off,
                    along[1] + wall.inward[1] * stand_off,
                )
                candidate = corner_footprint.at(centre[0], centre[1])
                if not contains_polygon(polygon, candidate):
                    continue
                if not permitted(candidate):
                    continue
                clearance = check_clearance(
                    candidate, obstacles=occupancy.blockers_for(request.entity_id), margin_m=0.08
                )
                if clearance.ok:
                    return centre, 0.0, corner_yaw, Quat.from_yaw(corner_yaw)
        return None

    # AGAINST_WALL: sweep the longest walls first so the desk gets the best wall, or the one wall
    # the zone plan named when there is a plan.
    wall_candidates = _walls_named_by(zone, walls)
    best_wall_placement = None
    for wall in sorted(wall_candidates, key=lambda w: -w.length_m):
        yaw = wall.yaw_facing_into_room()
        # The yaw is known before the position, so the footprint is rotated first. Placing a yawed
        # piece with its unrotated footprint is what put a desk half its length through the wall.
        rotated = footprint.rotated(yaw)
        sweep = place_against_wall(
            instance_id=request.entity_id, entity_id=request.entity_id, footprint=rotated,
            wall=wall, occupancy=occupancy, object_height_m=0.0, yaw=yaw, allowed=allow,
            # The wall's usable face is one wall thickness inside the line this wall was built
            # from, and the plan is what says how far off that line a piece stands. Passing zero --
            # which is what the plan says until the ladder repairs it -- puts the piece's back
            # inside the wall solid.
            wall_face_offset_m=plan.wall_face_offset_m,
            config=SurfacePlacementConfig(step_m=0.05, object_gap_m=plan.object_gap_m),
            preferred_xy=preferred_xy,
        )
        _record_rejections(request.entity_id, wall.index, sweep.rejections, audit)
        placement = sweep.placement
        if placement is not None:
            result = (
                (placement.position_m.x, placement.position_m.y), 0.0, yaw, Quat.from_yaw(yaw),
            )
            if preferred_xy is None:
                return result
            if best_wall_placement is None or math.dist(result[0], preferred_xy) < math.dist(best_wall_placement[0], preferred_xy):
                best_wall_placement = result
    return best_wall_placement


def build_scene(
    spec: SceneSpec, *, output_dir: Path, seed: int, plan: ScenePlan | None = None,
    state: BuildState | None = None,
    asset_bindings: dict[str, Path] | None = None,
    asset_license: str | None = None,
) -> dict:
    """Room, assets, layout and the initial relations, from a plan.

    ``plan`` is a value rather than a set of module literals because the escalation ladder repairs
    *the plan*: a rung that cannot hand back a changed plan is a rung that can only re-run the same
    decisions and hope the answer differs.
    """
    from rfw_w.scenes.adapters.simready_retrieval import (
        SimreadyConfig, build_encoder, build_engine, retrieve_entity, to_asset_record,
        worker_normalisation,
    )
    from rfw_w.scenes.assets.providers.hssd import (
        HssdConfig, prepare_furniture,
        screen_candidates_by_size,
    )
    from rfw_w.scenes.assets.procedural_support import materialise
    from rfw_w.scenes.assets.selection import rank_for_entity
    from rfw_w.scenes.solvers.support_shape import SupportShape

    plan = plan or build_plan(spec)
    if plan.target_id and plan.target_id != spec.task.targets[0]:
        raise SystemExit(
            f"the plan is for target {plan.target_id!r} but the spec asks for "
            f"{spec.task.targets[0]!r}; the scene would be built around the wrong task"
        )
    # Who the task is about, from the spec rather than from a literal. `"cup"` and `"desk"` used to
    # appear at roughly fifty places in this file, so a spec naming anything else built a scene
    # around a piece that was not in it. The plan already resolved both -- it fails to build at all
    # if it cannot -- so this is a read, not a second derivation.
    support_id, target_id = plan.support_id, plan.target_id
    if not support_id or not target_id:
        raise SystemExit(
            "the plan names no support and no task target; it was built by hand rather than from a "
            "spec, and this build cannot tell what the task is about"
        )
    bindings = asset_bindings or {}
    unknown_bindings = set(bindings) - {entity.id for entity in spec.entities}
    if unknown_bindings:
        raise ValueError("asset binding names an unknown entity: " + ", ".join(sorted(unknown_bindings)))
    unsupported_bindings = set(bindings) - {target_id}
    if unsupported_bindings:
        raise ValueError("GENERATED_BINDING_UNSUPPORTED: direct packages currently bind the task target; support/context packages need certified support affordances and placement adapters: " + ", ".join(sorted(unsupported_bindings)))
    prepared_target = None
    if target_id in bindings:
        from rfw_w.scenes.assets.providers.generated_package import prepare_package
        if plan.target_asset_index != 0:
            raise RepairUnavailable("EXPLICIT_BINDING_FIXED: the task target is explicitly bound; provide another package to change it")
        prepared_target = prepare_package(bindings[target_id], prepared_root=output_dir / "prepared_assets", license=asset_license, semantic=spec.entity(target_id).category)
    audit = AssemblyAudit()
    state = state if state is not None else BuildState()
    # The request remains immutable; a variant seed changes only the generated domain inside the
    # room bounds. Keeping the original spec is what lets the invariant hash remain comparable.
    generation_spec = spec if seed == spec.seed else replace(spec, seed=seed)

    # --- the task cell, before there is a room -----------------------------------------------
    #
    # The order of the two blocks that follow is the whole change. The task's own geometry -- the
    # work surface, where the target stands on it, how many ways the arm can get at it, and the
    # floor all of that must keep free -- is solved in a frame that has no room in it, and the room
    # is then drawn *around* that protected region rather than the cell being fitted into a room
    # that was drawn first. Before this, the room's width seed decided how many ways the arm could
    # approach the target: measured on office_pick_cup, 2 of 24.
    from rfw_w.scenes.solvers.cell import solve_cell

    cell = solve_cell(generation_spec, seed=generation_spec.seed)

    # The margin the cell keeps from the wall's *face* must be at least what the plan asks pieces
    # to keep from it, or the support can be inside the room and still outside `placeable`.
    wall_standoff_m = max(0.20, plan.wall_face_offset_m)
    try:
        fitted = fit_room_around_cell(
            generation_spec, cell,
            room_seed=generation_spec.seed, wall_standoff_m=wall_standoff_m,
        )
    except RoomError as error:
        raise SystemExit(str(error)) from error
    shell = fitted.shell
    options = fitted.options
    validation = validate_shell(
        shell, generation_spec, options=options, protected_in_room=fitted.protected_in_room,
    )
    step("fit_room_around_cell",
         f"{len(shell.polygon_xy)} vertices, openings={len(shell.openings)}, valid={validation.ok}, "
         f"cell at ({fitted.cell_translation_xy[0]:.3f}, {fitted.cell_translation_xy[1]:.3f}) "
         f"in {fitted.opening_attempts} opening attempt(s), "
         f"shape_fallback={fitted.shape_fallback or 'none'}, "
         f"size_margin={fitted.size_margin_m}, "
         f"door_bearing={fitted.door_bearing_deg} "
         f"(in_approach_arc={fitted.door_in_approach_arc})")
    if not validation.ok:
        raise SystemExit("the room shell did not validate: " + "; ".join(validation.gaps))

    room_polygon = Polygon(shell.polygon_xy)
    # The task's workspace, in room coordinates: the support, the floor the robot may stand on, and
    # the approach between them. This is what the room was fitted around, what the zone plan
    # reserves before any zone is drawn, and what `function.space_conditions` measures the way in
    # against -- the `WORK` strip it used to be measured against is a strip on whichever wall
    # looked longest, which is a proxy for this and not this.
    workspace = fitted.protected_in_room

    # The zone plan comes before anything is placed, and raises with a named gap rather than
    # returning a graph that misdescribes the room. Without it every piece is placed against
    # whichever wall is longest and nothing reserves the way in from the door. The cell's protected
    # region goes in with it, so the periphery is carved around the task rather than the task
    # competing with it for floor.
    zones = plan_zones(shell, spec.style, cell_protected=workspace)
    step("plan_zones", f"{len(zones.zones)} zones, {zones.placeable_area_m2:.2f} m2 placeable "
                       f"of {room_polygon.area:.2f} m2 of floor, "
                       f"{workspace.area:.2f} m2 reserved for the task cell")

    floor = Surface(
        surface_id="floor", owner_instance=None, polygon=room_polygon,
        plane_z_m=shell.floor_z_m, provenance=AffordanceStatus.MEASURED,
    )
    walls = room_walls(room_polygon)
    opening_tuples = [
        (o.opening_id, o.center_xy[0], o.center_xy[1], o.width_m) for o in shell.openings
    ]
    opening_spans = {w.index: openings_on_wall(w, opening_tuples) for w in walls}

    env = __import__("os").environ
    config = None
    engine = None
    engine_key = ()
    if prepared_target is None:
        config = SimreadyConfig(
            index_path=(Path(env["PVSCENE_SIMREADY_INDEX"]) if env.get("PVSCENE_SIMREADY_INDEX")
                        else Path(env["PVSCENE_SIMREADY_LIB"]) / "index.json"),
            assets_root=Path(env["PVSCENE_SIMREADY_ASSETS_ROOT"]),
            semantic_index_path=Path(env["PVSCENE_SIMREADY_SEMANTIC_INDEX"]),
            # From the environment rather than hardcoded: the version decides which meshes get opened
            # and which frame they are in, and a run that reads one library while claiming another is
            # a measurement of nothing. Defaulted to the version the shipped experiments were built
            # against, so a deployment that has not been migrated keeps behaving as it did.
            library_version=env.get("PVSCENE_SIMREADY_LIBRARY_VERSION", "v1.3"),
            # Where a version whose meshes have to be prepared (v1.3's source frame plus a proxy built
            # from it) writes them. Under the run's own prepared directory, beside the furniture's, so
            # a delivery carries everything it was built from.
            prepared_root=output_dir / "prepared_assets",
        )
        engine_key = (
            _cache_file_token(config.index_path), _cache_file_token(config.semantic_index_path),
            str(Path(env["PVSCENE_PREPARED_DIR"])),
        )
        engine = state.retrieval_engines.get(engine_key)
        if engine is None:
            engine = build_engine(
                config,
                encoder=build_encoder(Path(env["PVSCENE_PREPARED_DIR"])),
            )
            state.retrieval_engines[engine_key] = engine
    hssd = None
    if plan.furniture or plan.surfaces:
        hssd = HssdConfig(
            models_root=Path(env["PVSCENE_HSSD_ROOT"]),
            synset_index_path=Path(env["PVSCENE_HSSD_SYNSET_INDEX"]),
            prepared_root=output_dir / "prepared_assets",
        )

    occupancy = OccupancyMap()
    # The keepouts go in first, so the door sweep and the corridor block placement from the very
    # first piece. Nothing did this before, which is how a desk could be seated across its own
    # doorway with every gate still passing. They are named with the floor-only prefix because they
    # reserve *floor*: the cup resting on a desk above the corridor is not obstructed by it.
    for index, keepout in enumerate(zones.keepouts):
        occupancy.add(f"{FLOOR_KEEPOUT_PREFIX}zone{index}", keepout)
    # The task's own workspace goes in with them. The zone plan already carved every zone around it,
    # so nothing the zones place can land in it -- but a piece whose plan row names no zone falls
    # back to the whole room outline, and that path has never heard of the cell. Floor-only, like
    # the rest: the target and the surface objects stand *above* this region on the support, and
    # must not be obstructed by the floor it reserves.
    occupancy.add(f"{FLOOR_KEEPOUT_PREFIX}task_workspace", workspace)
    step("reserve_keepouts",
         f"{len(zones.keepouts)} zone keepout polygon(s) plus the "
         f"{workspace.area:.2f} m2 task workspace reserved before placement")

    def zone_for(kind: ZoneKind) -> Zone | None:
        """The region the plan assigned a kind, or None when the plan has no such zone."""
        return zones.zone_for(kind)

    furnished: list[Furnished] = []
    placed_by_entity: dict[str, Any] = {}
    failures: list[UnplacedItem] = []

    def failed(entity_id: str, reason: str, detail: str, arena: str = "room") -> None:
        failures.append(UnplacedItem(entity_id, spec.entity(entity_id).category, reason, arena, detail))
    # Which instances stand on the floor, as opposed to on a wall, a ceiling or another surface.
    # The passage gate subtracts exactly these: a rug is floor, but it is flat and beneath the
    # furniture, and a picture is not on the floor at all.
    floor_occupants: set[str] = set()

    # --- the hard task target: rigid, dynamic, from the rigid library -------------
    if prepared_target is not None:
        target_record, target_handle = prepared_target.record, prepared_target.handle
        step("bind_assets", f"{target_id} <- verified {target_record.key.provider}")
    else:
        target_spec = spec.entity(target_id)
        # Which SimReady super-categories stand for this target. From the vocabulary rather than from a
        # literal tuple: a spec whose target is a jar cannot be retrieved by asking for a Mug, and an
        # empty admitted set admits everything -- the retrieval's own comment records a "Desk Bell"
        # arriving for a query for a desk.
        target_profile = profile_for(target_spec.category)
        if target_profile is None or not target_profile.simready_categories:
            raise SystemExit(
                f"the vocabulary has no SimReady categories for {target_spec.category!r}, so the task "
                f"target {target_id!r} cannot be retrieved"
            )
        retrieval_key = (
            engine_key,
            content_hash(target_spec.to_dict()),
            tuple(target_profile.simready_categories),
            8,
        )
        target_outcome = state.retrieval_results.get(retrieval_key)
        if target_outcome is None:
            target_outcome = retrieve_entity(
                engine, target_spec, config=config, limit=8,
                admitted_categories=target_profile.simready_categories,
            )
            state.retrieval_results[retrieval_key] = target_outcome
        if not target_outcome.ok:
            raise SystemExit(f"retrieval failed for the task target: {target_outcome.detail}")
        # Which candidate, from the plan rather than from a literal. The target is what a critic is
        # most likely to complain about, and `prompt_adherence` admits exactly one repair that can
        # answer it -- `swap_asset` -- so an L2 swap on the target has to be able to move this index or
        # it edits nothing the build reads.
        candidate_index = plan.target_asset_index
        if candidate_index >= len(target_outcome.candidates):
            # Named rather than clamped. Re-using rank 0 would rebuild the scene the repair was asked
            # to change and record it as "the swap had no effect", which is a different claim from
            # "there was no further candidate to swap to".
            raise RepairUnavailable(
                f"TARGET_CANDIDATES_EXHAUSTED: the plan asks for candidate {candidate_index} of the task target but retrieval "
                f"returned {len(target_outcome.candidates)} for it; the asset swap has run off the end"
            )
        target_record, target_handle = to_asset_record(
            target_outcome.candidates[candidate_index], config
        )
        step("retrieve_assets",
             f"{target_id} <- {target_record.key.provider} rank {candidate_index} "
             f"of {len(target_outcome.candidates)}")

    # The floor a piece may occupy: the outline less the walls standing on it. Passed to the
    # placement solver as its containment reference, so that a piece stood off the wall is stood off
    # the wall's *face* -- corner, beside and floor-covering placements all measured from the
    # outline before, which is the wall's outer face, and all of them buried the piece's edge in the
    # wall solid by exactly the wall thickness. Only the against-wall branch had the notion of a
    # wall face at all, which is why fixing that one alone left the lamp, the plant and the rug
    # still inside the wall.
    placeable = inset(room_polygon, plan.wall_face_offset_m)

    # The support is BUILT, not retrieved, and it is built *first*.
    #
    # It used to be a `FurnitureRequest` like any other, placed somewhere in the middle of the
    # furniture loop. Now the loop never sees it -- there is no library behind it -- and its position
    # in the order is a decision rather than an accident: pieces that face the support (the office
    # chair is `BESIDE` with `faces_support`) look it up in `placed_by_entity` to find their anchor,
    # so a support placed after them leaves them with nothing to face. Placing it first is also what
    # core-first means: the task's surface exists before the room's contents do.
    support_request = plan.support_for(support_id)
    if support_request is None:
        raise SystemExit(
            f"{support_id!r} is named as the task support but the plan has no support row for it, "
            "so there is neither a form to build nor a size to build it to"
        )
    # The room-free core decides the work surface -- width, depth, height and form -- before any
    # room is fitted around it. It was solved at the top of this function; this is the build
    # honouring its answer.
    cell_bounds = cell.support_footprint.bounds()
    support_width_m = round(cell_bounds[2] - cell_bounds[0], 6)
    support_depth_m = round(cell_bounds[3] - cell_bounds[1], 6)
    support_prepared = materialise(
        SupportShape(cell.support_shape),
        width_m=support_width_m, depth_m=support_depth_m, height_m=cell.support_height_m,
        out_dir=output_dir / "prepared_assets",
        # The finish comes from the plan's palette, through the same `furniture` role the room's
        # surfaces come through -- one vocabulary, and the support is a surface it names. Without
        # this the mesh carries no material and the renderer invents a colour for the piece the
        # prompt is most specific about.
        colour=plan.room_materials.furniture.colour,
        roughness=plan.room_materials.furniture.roughness,
        wood_grain=bool(set(spec.entity(support_id).material_tokens or spec.style.palette)
                        & {"oak", "wood", "walnut", "warm_wood"}),
    )
    # Placed where the cell put it, not where a wall sweep would like it.
    #
    # This is the one place the room stopped deciding the task. The support used to go through
    # `_furniture_placement` like any retrieved piece -- `AGAINST_WALL`, longest wall first -- so
    # the room's shape chose the desk's position, the desk's position chose the cup's, and the
    # cup's position chose how many ways the arm could reach it. `cell.protected` is the region the
    # room was fitted around, so the support's own footprint is already inside it and its pose is
    # simply the cell's, translated.
    cell_offset = cell.support_centre_offset_m
    centre = (
        round(fitted.cell_translation_xy[0] + cell_offset[0], 6),
        round(fitted.cell_translation_xy[1] + cell_offset[1], 6),
    )
    yaw = cell.local_frame.yaw_rad
    base_z = 0.0
    rotation = Quat.from_yaw(yaw)
    support_footprint = Footprint((support_width_m / 2.0, support_depth_m / 2.0))
    occupancy.add(support_id, support_footprint.rotated(yaw).at(centre[0], centre[1]))
    floor_occupants.add(support_id)
    support_entry = Furnished(
        instance_id=support_id, entity_id=support_id,
        visual_uri=str(support_prepared.visual_obj),
        collision_uri=str(support_prepared.collision_obj),
        position_m=Vec3(centre[0], centre[1], base_z), yaw_rad=yaw,
        mass_kg=support_prepared.mass_kg, dynamic=False, normalisation="none",
        dimensions_m=support_prepared.dimensions_m,
        provider=support_prepared.asset_key.provider,
        library_version=support_prepared.asset_key.library_version,
        asset_id=support_prepared.asset_key.asset_id,
        mass_provenance=support_prepared.provenance.status.value,
        rotation_wxyz=tuple(rotation.to_list()),
        origin_at_base=support_prepared.origin_convention.endswith("base_at_zero"),
    )
    furnished.append(support_entry)
    placed_by_entity[support_id] = support_entry

    if not support_prepared.support_polygons:
        raise SystemExit(
            f"the built {support_id!r} produced no support surface, so {target_id!r} "
            f"has nothing to stand on"
        )
    top = support_prepared.support_polygons[0]
    # The support outline is in the desk's own frame, so it rotates with the desk before it is
    # translated. Moving it without rotating put the cup's landing area where the desk *would* have
    # been had it not been turned -- which is how a cup ends up on the floor beside the desk.
    support_yaw = support_entry.yaw_rad
    cos_yaw, sin_yaw = math.cos(support_yaw), math.sin(support_yaw)
    support_top = Surface(
        surface_id=f"{support_id}:top", owner_instance=support_id,
        polygon=Polygon.from_sequence(
            [
                (
                    x * cos_yaw - y * sin_yaw + support_entry.position_m.x,
                    x * sin_yaw + y * cos_yaw + support_entry.position_m.y,
                )
                for x, y in top.vertices_xy
            ]
        ),
        plane_z_m=top.plane_z_m, provenance=top.provenance.status,
    )

    # --- furniture and decoration --------------------------------------------------
    min_x, min_y, max_x, max_y = room_polygon.bounds()
    for request in _most_constrained_first(plan.furniture):
        entity_id, synset, kind = request.entity_id, request.synset, request.kind
        target_extent, zone_kind = request.target_extent_m, request.zone_kind
        # A piece with no zone is placed against whichever wall suits the solver. That is the same
        # behaviour as before, minus the crash: the zone used to come from a table keyed by entity
        # id that only held the nine rows the plan named, so an entity a replan *added* raised
        # KeyError here rather than being placed.
        zone = zone_for(zone_kind) if zone_kind is not None else None
        # Honor NEAR in either direction without moving the frozen task core: the bed is
        # placed near its already-built nightstand, rather than moving the nightstand later.
        neighbours = [relation.object if relation.subject == entity_id else relation.subject
                      for relation in spec.initial_relations
                      if relation.predicate.value == "NEAR" and entity_id in (relation.subject, relation.object)]
        anchor = next((placed_by_entity[name] for name in neighbours if name in placed_by_entity), None)
        preferred_xy = (anchor.position_m.x, anchor.position_m.y) if anchor is not None else None
        entity_spec = spec.entity(entity_id)
        profile = profile_for(entity_spec.category)
        synsets = tuple(dict.fromkeys((synset, *(profile.synsets if profile else ()))))
        candidates = rank_for_entity(hssd, entity_spec, synsets=synsets,
                                    target_width_m=target_extent, kind=kind.value,
                                    palette=spec.style.palette)
        if not candidates:
            failed(entity_id, "NO_ASSET", f"no {synset} on disk")
            continue
        ranked = candidates
        # The spec's own size bound, enforced. This is the check the SimReady path has always had
        # and this one never did: `rank_candidates_by_size` scores width alone and rejects nothing,
        # so a spec asking for a 0.12-0.35 m ceiling light was handed a 1.67 m pendant, which the
        # placement hung from the ceiling and left 1.25 m above the floor in the walking path.
        # `geometry.in_bounds` cannot see that -- the box is inside the room, it is simply too low.
        entity_spec = spec.entity(entity_id)
        screen = screen_candidates_by_size(
            hssd, ranked, bounds=entity_spec.size_bounds_m, kind=kind.value,
        )
        # A rejected candidate is not an unplaced item, and reporting it as one floods the log with
# eighteen lines for a single entity -- the ceiling light alone rejects seventeen. They go to
# `furnishing.size_bound_rejections` instead, where a reader can see how much of the pool the
# bound removed without mistaking it for furniture that could not be placed.
        for asset_id, reason in screen.relaxed:
            audit.size_rejections.append(
                {"entity_id": entity_id, "asset_id": asset_id, "reason": reason,
                 "declared_bounds_m": [
                     [axis.minimum, axis.maximum] for axis in (entity_spec.size_bounds_m or ())
                 ]}
            )
        if not screen.admitted and screen.relaxed:
            # Nothing fits. Take the least-bad and say so, rather than failing the item or --
            # which is what used to happen -- silently taking whichever candidate ranked first.
            # A named relaxation is repairable (L2 swaps the asset for the next candidate); a silent
            # one is not visible at all, which is how the 1.67 m pendant survived a spec that asked
            # for 0.12-0.35 m.
            worst_first = tuple(asset for asset, _ in screen.relaxed)
            failed(entity_id, "SIZE_BOUND_RELAXED",
                f"{entity_id}: SIZE_BOUND_RELAXED - none of {len(ranked)} candidates fits "
                f"{[[axis.minimum, axis.maximum] for axis in (entity_spec.size_bounds_m or ())]}; "
                f"taking {worst_first[0][:10]}, the least-bad"
            )
            ranked = worst_first + screen.unmeasurable
        else:
            ranked = list(screen.admitted)
        # Top-3 materialised, next on failure: the architecture's rule, and the reason a single
        # unsuitable asset does not lose the whole item. The plan's own candidate index starts the
        # list, which is what makes an L2 asset swap change anything: without it every rebuild
        # ranked the same way and picked the same asset.
        ranked = ranked[request.asset_index:]
        if not ranked:
            failed(entity_id, "NO_CANDIDATE", f"{entity_id}: the plan asks for candidate {request.asset_index} but "
                            f"only {request.asset_index} exist for {synset}")
            continue
        for asset_id in ranked[:3]:
            try:
                prepared = prepare_furniture(hssd, asset_id)
            except Exception as error:  # noqa: BLE001 - a bad asset must not abort the room
                failed(entity_id, "PREPARE_FAILED", f"{asset_id[:10]} did not prepare ({type(error).__name__})")
                continue
            footprint = Footprint((prepared.dimensions_m.x / 2, prepared.dimensions_m.y / 2))
            placed = _furniture_placement(
                request, footprint, prepared.dimensions_m.z,
                polygon=placeable, walls=walls, occupancy=occupancy,
                placed_by_entity=placed_by_entity, opening_spans=opening_spans,
                room_height_m=shell.height_m, wall_thickness_m=options.wall_thickness_m,
                plan=plan, zone=zone, preferred_xy=preferred_xy,
            audit=audit)
            if placed is None and zone is not None and not _zone_could_hold(zone, footprint):
                # The zone was never a constraint this piece could satisfy, so this is a capacity
                # failure rather than a refusal: retry on the floor the room left, and record that
                # the arena changed. A piece the zone *could* have held is not retried -- that stays
                # a refusal, which is what `_furniture_placement` documents.
                placed = _furniture_placement(
                    request, footprint, prepared.dimensions_m.z,
                    polygon=placeable, walls=walls, occupancy=occupancy,
                    placed_by_entity=placed_by_entity, opening_spans=opening_spans,
                    room_height_m=shell.height_m, wall_thickness_m=options.wall_thickness_m,
                    plan=plan, zone=None, preferred_xy=preferred_xy,
                audit=audit)
                if placed is not None:
                    audit.zone_fallbacks.append({
                        "entity_id": entity_id,
                        "zone": zone_kind.value,
                        "zone_area_m2": round(zone.area_m2, 4),
                        "arena": "room",
                        "reason": "zone_could_not_hold_it",
                    })
            if placed is None:
                arena = _arena_name(kind, zone_kind)
                where = "anywhere legal" if arena == "the room" else "nowhere the plan allows"
                failed(entity_id, "NO_POSITION",
                       f"{asset_id[:10]} has no position ({where})", arena)
                continue
            centre, base_z, yaw, rotation = placed
            if kind in (PlacementKind.AGAINST_WALL, PlacementKind.BESIDE, PlacementKind.CORNER,
                        PlacementKind.ANYWHERE):
                # The rotated shape, so the next piece sees what is really there. Registering the
                # unrotated footprint let a cabinet be placed where the yawed desk already was.
                occupancy.add(entity_id, footprint.rotated(yaw).at(centre[0], centre[1]))
                floor_occupants.add(entity_id)
            entry = Furnished(
                instance_id=entity_id, entity_id=entity_id,
                visual_uri=str(prepared.visual_obj), collision_uri=str(prepared.collision_obj),
                position_m=Vec3(centre[0], centre[1], base_z), yaw_rad=yaw,
                mass_kg=prepared.mass_kg, dynamic=False, normalisation="none",
                dimensions_m=prepared.dimensions_m, provider="hssd",
                library_version="hssd-models", asset_id=asset_id,
                mass_provenance=prepared.provenance.status.value,
                rotation_wxyz=tuple(rotation.to_list()),
                # prepare_furniture centres x and y and leaves the base at z=0, and records that
                # in origin_convention; the placement branches above rely on it, so it is carried
                # here too rather than re-derived by whoever reads the pose later.
                origin_at_base=prepared.origin_convention.endswith("base_at_zero"),
            )
            # prepared objects were centred base-at-origin, so the z is the base; the worker wants
            # the origin, which is exactly z=0 for that convention.
            entry.position_m = Vec3(centre[0], centre[1], base_z)
            furnished.append(entry)
            placed_by_entity[entity_id] = entry
            break

    # --- the cup, on the desk ------------------------------------------------------
    #
    # Where it stands is the cell's answer, not a search. `place_on_surface` solved exactly this
    # problem a second time -- and scored nothing: the cell's candidate scoring is what found the
    # positions the arm can actually work from, so re-placing the target here would discard the one
    # piece of reasoning this whole reordering exists to keep.
    target_footprint = Footprint((target_record.dimensions_m.x / 2, target_record.dimensions_m.y / 2))
    target_config = SurfacePlacementConfig(step_m=0.02, edge_margin_m=0.06, object_gap_m=0.06)
    target_xy = (
        round(fitted.cell_translation_xy[0] + cell.target_offset[0], 6),
        round(fitted.cell_translation_xy[1] + cell.target_offset[1], 6),
    )
    # The cell solved the target against the support polygon it declared, for the size the spec
    # *declared*. The asset that arrives is a different size, and nothing screens it: the SimReady
    # path aims at the middle of the spec's bounds (so a cup is not retrieved as a bowl) and rejects
    # nothing, so the jar that lands can be wider than the offset was solved for. Measured on
    # `kitchen_fetch_jar`: the cell's offset put the jar 0.04 m over the island's front edge.
    #
    # So the cell's answer is taken when it is realisable and the surface's own search is used when
    # it is not, and which one happened is recorded rather than left to be inferred. The alternative
    # -- refusing the scene -- would fail a run over a 4 cm mismatch between a declared size and a
    # retrieved one, which is a fact about retrieval and not about the task's geometry.
    if contains_polygon(support_top.polygon, target_footprint.at(target_xy[0], target_xy[1])):
        target_source = "task_cell"
        # Retrieved SimReady geometry, centred on its own midpoint -- the opposite convention to
        # the prepared HSSD meshes above. Its base is half its height below the recorded z, which is
        # the reading the ON predicate already takes at the bottom of this function.
        target_pos = Vec3(
            target_xy[0], target_xy[1],
            support_top.plane_z_m + target_config.spawn_clearance_m
            + target_record.dimensions_m.z / 2.0,
        )
    else:
        target_source = "surface_search"
        outcome = place_on_surface(
            instance_id=target_id, entity_id=target_id, footprint=target_footprint,
            surface=support_top, object_height_m=target_record.dimensions_m.z,
            occupancy=occupancy, config=target_config, origin_convention="centre",
        )
        if not outcome.placed:
            raise CoreUnplaced(UnplacedItem(
                target_id, spec.entity(target_id).category, "NO_SURFACE_SPACE",
                f"surface:{support_id}", outcome.detail,
            ))
        target_pos = outcome.placement.position_m
    target_geometry_centre = target_pos
    if prepared_target is not None:
        target_pos = target_pos - (target_record.bounds_min_m + target_record.bounds_max_m).scale(0.5)
    furnished.append(Furnished(
        instance_id=target_id, entity_id=target_id,
        visual_uri=target_handle.visual_uri, collision_uri=(prepared_target.worker_collision_uri if prepared_target is not None else target_handle.collision_uris[0]),
        position_m=target_pos, yaw_rad=0.0, mass_kg=target_record.rigid_physics.mass_kg,
        dynamic=True, normalisation=("none" if prepared_target is not None else worker_normalisation(config.library_version)),
        dimensions_m=target_record.dimensions_m,
        provider=target_record.key.provider, library_version=target_record.key.library_version,
        asset_id=target_record.key.asset_id, source_hash=target_record.source_hash,
        license=target_record.license,
        rotation_wxyz=tuple(Quat.from_yaw(0.0).to_list()),
        # SimReady, centred: see the ON predicate below, which reads its base as z - height/2.
        origin_at_base=False,
        local_bounds_min_m=target_record.bounds_min_m if prepared_target is not None else None,
        local_bounds_max_m=target_record.bounds_max_m if prepared_target is not None else None,
        rigid_physics=target_record.rigid_physics.to_dict() if prepared_target is not None else None,
        collision_realization=prepared_target.collision_realization if prepared_target is not None else "",
        source_collision_hull_count=len(target_handle.collision_uris) if prepared_target is not None else 0,
    ))
    placed_by_entity[target_id] = furnished[-1]
    # Register the task target before anything else is placed on the same surface.
    #
    # Without this the cup is invisible to the occupancy map, and the next surface object is placed
    # straight through it: two dynamic bodies spawned overlapping, which a solver resolves by
    # shoving them apart. Measured here, a pen holder 5 cm from a cup with a 14 cm combined radius
    # threw the cup 0.44 m sideways and off the desk -- and the settle then reported it as a physics
    # failure, when the physics had been correct and the *placement* had been impossible.
    occupancy.add(target_id, target_footprint.at(target_geometry_centre.x, target_geometry_centre.y))

    # --- surface objects, on the desk ----------------------------------------------
    for surface_request in plan.surfaces:
        entity_id, synset = surface_request.entity_id, surface_request.synset
        target = surface_request.target_width_m
        # Named rather than silently ignored: the build materialises one support surface, the desk's
        # top, and a plan that asks for another would otherwise place the object on the desk anyway
        # while reporting that it had been moved.
        if surface_request.support_entity != support_id:
            failed(entity_id, "SUPPORT_UNAVAILABLE",
                f"{entity_id}: the plan asks for support {surface_request.support_entity!r} but "
                "only the desk's top surface is materialised"
            )
            continue
        # Raises for an entity the spec does not name, which is the point: a plan row with no
        # matching request is a plan the request cannot justify.
        entity_spec = spec.entity(entity_id)
        profile = profile_for(entity_spec.category)
        synsets = tuple(dict.fromkeys((synset, *(profile.synsets if profile else ()))))
        candidates = rank_for_entity(hssd, entity_spec, synsets=synsets,
                                    target_width_m=target, palette=spec.style.palette)
        if not candidates:
            failed(entity_id, "NO_ASSET", f"no {synset} on disk")
            continue
        surface_ranked = candidates
        for asset_id in surface_ranked[surface_request.asset_index:][:3]:
            try:
                prepared = prepare_furniture(hssd, asset_id)
            except Exception as error:  # noqa: BLE001
                failed(entity_id, "PREPARE_FAILED", f"{asset_id[:10]} did not prepare ({type(error).__name__})")
                continue
            if prepared.dimensions_m.z > 0.35:
                failed(entity_id, "SIZE_UNSUPPORTED", "surface object exceeds 0.35 m height", f"surface:{support_id}")
                continue  # too tall to sit on a desk in a way that reads as a desk object
            placement = place_on_surface(
                instance_id=entity_id, entity_id=entity_id,
                footprint=Footprint((prepared.dimensions_m.x / 2, prepared.dimensions_m.y / 2)),
                surface=support_top, object_height_m=prepared.dimensions_m.z, occupancy=occupancy,
                config=SurfacePlacementConfig(step_m=0.03, edge_margin_m=0.06, object_gap_m=0.08),
                # These are prepared HSSD meshes, base at z=0. The default centre convention put
                # the pen holder 6.7 cm above the desk and the notebook 1.5 cm above it, with
                # nothing measuring either gap.
                origin_convention="base",
            )
            if not placement.placed:
                failed(entity_id, "NO_SURFACE_SPACE", placement.detail, f"surface:{support_id}")
                continue
            entry = Furnished(
                instance_id=entity_id, entity_id=entity_id,
                visual_uri=str(prepared.visual_obj), collision_uri=str(prepared.collision_obj),
                position_m=placement.placement.position_m, yaw_rad=0.0,
                mass_kg=prepared.mass_kg, dynamic=True, normalisation="none",
                dimensions_m=prepared.dimensions_m, provider="hssd",
                library_version="hssd-models", asset_id=asset_id,
                mass_provenance=prepared.provenance.status.value,
                rotation_wxyz=tuple(Quat.from_yaw(0.0).to_list()),
                origin_at_base=True,
            )
            occupancy.add(entity_id, Footprint((prepared.dimensions_m.x / 2,
                                                prepared.dimensions_m.y / 2)
                                               ).at(entry.position_m.x, entry.position_m.y))
            furnished.append(entry)
            placed_by_entity[entity_id] = entry
            break

    # --- the initial relation, from the real support polygons -----------------------
    # One `subjects` entry per hard ON relation whose subject was actually placed. It is a mapping
    # rather than the single `subject` this used to hold, because which relation is the task
    # relation is now the spec's decision and not "index 0" -- and a spec may state more than one.
    surfaces = [floor, support_top]
    subjects: dict[str, PlacedInstance] = {}
    for relation in spec.initial_relations:
        if not relation.hard or relation.predicate.value != "ON":
            continue
        if relation.subject in subjects:
            continue
        posed = placed_by_entity.get(relation.subject)
        if posed is None:
            continue  # unplaced; the relation gate reports it as unsatisfied rather than crashing
        # The base depends on the instance's own origin convention, which the ON predicate reads
        # the same way. Recording it here rather than assuming keeps a centred SimReady asset and a
        # base-at-zero HSSD mesh from being measured identically when they are not.
        half = posed.dimensions_m.z / 2.0
        base_z = (posed.position_m.z + posed.local_bounds_min_m.z if posed.local_bounds_min_m is not None else posed.position_m.z if posed.origin_at_base else posed.position_m.z - half)
        geometric_centre = posed.geometry_centre_m()
        subjects[relation.subject] = PlacedInstance(
            instance_id=relation.subject,
            footprint=Footprint(
                (posed.dimensions_m.x / 2.0, posed.dimensions_m.y / 2.0)
            ).at(geometric_centre.x, geometric_centre.y),
            base_z_m=base_z, is_dynamic=posed.dynamic,
        )

    losses = split_unplaced(spec, placed_by_entity, plan.entity_ids, failures)
    budget_low, budget_ideal = furniture_target_count(room_polygon.area)
    step("furnish", f"{len(furnished)} instances placed (target {budget_low}-{budget_ideal}); "
                    f"{sum(len(items) for items in losses.values())} entity(s) unplaced")
    for items in losses.values():
        for failure in items:
            step("  unplaced", str(failure))

    return {
        "plan": plan,
        # The room as a plan, *beside* the room that was drawn -- never in place of it. The
        # translation is pure (`plan_from_fitted` reads the `FittedRoom` that already exists rather
        # than fitting again), so adding it cannot move the room, and `scene["shell"]` stays the
        # thing every existing gate and the worker request are built from. What it buys is a room
        # description a *different* builder can be handed: that is what `shell.matches_plan`
        # compares against, and without it the gate would have nothing to be about.
        "room_plan": plan_from_fitted(
            generation_spec, cell, fitted, room_seed=generation_spec.seed
        ),
        "wall_thickness_m": options.wall_thickness_m,
        "shell": shell, "validation": validation, "room_polygon": room_polygon,
        "floor": floor, "walls": walls, "opening_spans": opening_spans,
        "zones": zones, "scene_id": spec.scene_id, "floor_occupants": frozenset(floor_occupants),
        # The task's workspace in room coordinates -- the region the room was fitted around and the
        # zone plan reserved. `function.space_conditions` measures the way in against this. It used
        # to measure it against the `WORK` zone, which is a strip on whichever wall looked longest:
        # a proxy that was only ever a guess at where the task would end up.
        "workspace": workspace, "cell": cell, "room_fit": fitted,
        "instances": furnished, "placed_by_entity": placed_by_entity, "failures": failures, **losses,
        # The two ids everything downstream needs. They are on the plan too, but the scene dict is
        # what `build_worker_request` and the gates are handed -- and reading them from the plan in
        # one place and from the scene in another is how a bare `target_id` survived the rename in
        # two functions that never defined it. That NameError is wrapped by the orchestrator into
        # PRECONDITION_UNMET, which the driver treats as a semantic failure and spends a model
        # request on: it looked like a scene problem for two whole runs.
        "support_id": support_id, "target_id": target_id,
        "assembly_audit": audit,
        "support": support_prepared, "support_pos": support_entry.position_m, "support_entry": support_entry,
        "support_top": support_top, "target_record": target_record, "target_handle": target_handle,
        "target_pos": target_geometry_centre, "target_footprint": target_footprint, "subjects": subjects,
        # Which rule put the target where it is: the task cell's own answer, or the support
        # surface's search, when the retrieved asset turned out bigger than the size the cell
        # solved for. Recorded rather than inferred, because it is the one place the core's answer
        # could not be followed and a reader should not have to guess whether it was.
        "target_source": target_source,
        "surfaces": surfaces,
        "furniture_target": (budget_low, budget_ideal),
    }
