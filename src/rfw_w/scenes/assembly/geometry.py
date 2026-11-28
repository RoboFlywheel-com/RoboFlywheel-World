# SPDX-License-Identifier: Apache-2.0
"""Independent geometry and reach measurements on assembled instances."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.contracts.state import OpeningKind
from rfw_w.scenes.contracts.units import Quat
from rfw_w.scenes.geometry.polygon import Polygon, contains_point, distance_to_boundary, inset, intersect_polygons
from rfw_w.scenes.solvers.layout import Footprint, PlacedExtent
from rfw_w.scenes.solvers.furnish import PlacementKind
from rfw_w.scenes.solvers.room import generate_options
from rfw_w.scenes.solvers.zones import ZoneGraph

if TYPE_CHECKING:
    from rfw_w.scenes.assembly.placement import Furnished

CONTACT_TOLERANCE_M = 0.02

def _placed_extents(scene: dict) -> list[tuple[str, "PlacedExtent"]]:
    """Every placed instance as the box it really is: plan outline plus height range.

    Taken from the recorded pose -- position, the full rotation and the mesh's own origin
    convention -- and not from the shape the solver searched with. The distinction is the whole
    point: a solver that reasons about the wrong shape gets it wrong in its own filter too, so a
    check that reuses that shape inherits the mistake. It did, twice: a 5 mm picture panel
    searched as a 0.70 x 1.00 footprint passed every in-loop containment test and still reached
    0.13 m through a wall, and a 1.67 m pendant whose origin was read as its centre passed while
    0.78 m of it stood above the roof.
    """
    placed: list[tuple[str, PlacedExtent]] = []
    for entry in scene["instances"]:
        dimensions = entry.dimensions_m
        rotation = Quat(*(
            entry.rotation_wxyz
            or (math.cos(entry.yaw_rad / 2), 0.0, 0.0, math.sin(entry.yaw_rad / 2))
        ))
        placed.append((entry.instance_id, Footprint(
            (dimensions.x / 2.0, dimensions.y / 2.0)
        ).placed_extent(
            dimensions.z, rotation=rotation, position=entry.geometry_centre_m() if getattr(entry, "local_bounds_min_m", None) is not None else entry.position_m,
            origin_convention=("centre" if getattr(entry, "local_bounds_min_m", None) is not None else "base" if entry.origin_at_base else "centre"),
        )))
    return placed


def _inner_floor(scene: dict, wall_thickness_m: float) -> Polygon:
    """The floor a piece may actually occupy: the outline less the walls standing on it.

    The room polygon is the wall's **outer** face. ``build_wall`` offsets the wall body inward from
    it, so the usable floor is the outline inset by the wall thickness and anything inside the
    difference is inside a wall. Every gate and every placement that used the outline as "the room"
    was therefore measuring the wrong region -- and could not tell, because a piece flush to the
    outline looks, from any camera inside the room, exactly like a piece flush to the wall.
    """
    return inset(scene["room_polygon"], wall_thickness_m)


def _escape_of(extent: "PlacedExtent", *, room: Polygon, outline: Polygon, floor_z: float,
               ceiling_z: float) -> tuple[float, str]:
    """How far one placed box reaches out of the room, and which kind of escape it is.

    One rule, two readers: `geometry.in_bounds` judges the finished room with it, and the
    generated-furniture assembly asks it of one candidate before admitting the candidate to the
    delivery. Measured on kitchen seed 11, a kept mirror frame stood **0.0400 m inside the wall
    solid** in the final version -- a red the loop's pose repairs cannot fix, because they move the
    retrieval furniture and the frame's pose comes from the build.

    The two escapes are named differently because they are different defects. A piece inside the
    wall is a placement error invisible from inside the room; a piece past the outline leaves the
    building, and nothing could have missed it.
    """
    escape = 0.0
    inside_outline_only = True
    for vertex in extent.outline.vertices:
        if not contains_point(room, vertex):
            escape = max(escape, distance_to_boundary(room, vertex))
        if not contains_point(outline, vertex):
            inside_outline_only = False
    if extent.z_low_m < floor_z - 1e-6:
        escape = max(escape, floor_z - extent.z_low_m)
    if extent.z_high_m > ceiling_z + 1e-6:
        escape = max(escape, extent.z_high_m - ceiling_z)
    return escape, ("inside-wall" if inside_outline_only else "outside-room")


def _geometry_in_bounds(
    scene: dict, *, wall_thickness_m: float
) -> tuple[bool, float, str, tuple[str, ...]]:
    """Whether every placed instance's own box is inside the walls and between floor and ceiling.

    Computed, not asserted, and over **every** instance rather than over the two the task names.
    It used to be an unconditional PASS whose detail read "both footprints lie inside the
    6-vertex room outline" while twelve instances were placed -- so it could not have failed for
    ten of them, and one of those ten, the wall art, was 0.13 m through a wall at the time. The
    same reasoning the ``function.space_conditions`` docstring gives applies here: a gate that
    cannot fail is not a gate.

    It is measured against the walls' **inner** face rather than against the outline. The outline
    is the wall's outer face, so testing it accepted every piece whose back was buried in the wall
    solid -- which was all five wall-adjacent pieces, by one wall thickness each.

    Returns ``(ok, worst_intrusion_m, detail, witnesses)``. The measurement is how far outside the
    nearest offending vertex reaches, so a pass reads 0.0 and a failure reports the distance.
    """
    room = _inner_floor(scene, wall_thickness_m)
    floor_z = scene["shell"].floor_z_m
    ceiling_z = floor_z + scene["shell"].height_m

    worst = 0.0
    offenders: list[str] = []
    witnesses: list[str] = []
    outline = scene["room_polygon"]
    for instance_id, extent in _placed_extents(scene):
        escape, kind = _escape_of(extent, room=room, outline=outline, floor_z=floor_z,
                                  ceiling_z=ceiling_z)
        if escape > 1e-6:
            offenders.append(
                f"{instance_id} ({escape:.4f} m: z {extent.z_low_m:.3f}..{extent.z_high_m:.3f} "
                f"of {floor_z:.3f}..{ceiling_z:.3f})"
            )
            witnesses.append(f"instance:{instance_id}:{kind}")
        worst = max(worst, escape)

    if offenders:
        return (
            False, worst,
            f"{len(offenders)} of {len(scene['instances'])} placed instances leave the room: "
            + "; ".join(offenders),
            tuple(witnesses),
        )
    return (
        True, 0.0,
        f"all {len(scene['instances'])} placed instances lie inside the walls' inner face "
        f"(the outline inset by {wall_thickness_m:.4f} m), between the floor at {floor_z:.3f} m "
        f"and the ceiling at {ceiling_z:.3f} m",
        tuple(f"instance:{instance_id}:inside-walls" for instance_id, _ in _placed_extents(scene)),
    )


def _geometry_task_cell_clear(scene: dict) -> tuple[bool, float, str, tuple[str, ...]]:
    """Measure intrusion into the frozen task floor, including thin floor coverings."""
    workspace = scene.get("workspace")
    if workspace is None:
        return False, 0.0, "task workspace is missing", ()
    kinds = {r.entity_id: r.kind for r in scene["plan"].furniture}
    floor_ids = set(scene["floor_occupants"]) | {
        name for name, kind in kinds.items() if kind is PlacementKind.FLOOR_COVERING
    }
    worst, witnesses, diagnostic = 0.0, [], []
    for name, extent in _placed_extents(scene):
        shared = sum(p.area for p in intersect_polygons(extent.outline, workspace))
        if name == scene["support_id"]:
            diagnostic.append(f"{name}: core support belongs to the protected cell")
            continue
        if kinds.get(name) in (PlacementKind.CEILING, PlacementKind.WALL_HUNG):
            diagnostic.append(f"{name}: {kinds[name].value} excluded from floor occupancy; air projection={shared:.6f} m2")
            continue
        if name not in floor_ids:
            continue
        worst = max(worst, shared)
        if shared > 1e-8:
            witnesses.append(f"instance:{name}:task-cell-overlap")
    return not witnesses, worst, (
        f"worst floor intrusion={worst:.8f} m2; " + "; ".join(diagnostic)
    ), tuple(witnesses)


def _pair_shared_area(first_id: str, first: "PlacedExtent", second_id: str, second: "PlacedExtent",
                      floor_coverings: set[str]) -> float:
    """The area two placed instances really share, or 0.0 when they legitimately share it.

    One rule, two callers: `geometry.no_penetration` judges the whole room with it, and the
    generated-furniture assembly asks the same question of one candidate before it lets the
    candidate into the delivery. Two spellings of "occupies the same space" would let a delivered
    object pass the assembly and fail the gate -- which would be a delivery whose own gate is red,
    from a rule written to keep it green.

    Plan overlap alone is the wrong test and would fail a correct room: measured on a real delivery
    it flags seven pairs, of which five are legitimate -- a cup, a notebook and a pen holder resting
    on the desk whose footprint contains them, and a rug lying under both the chair and the pendant.
    So three things have to hold at once: the outlines overlap, the height ranges overlap, and
    neither piece is a floor covering. ``OccupancyMap.blockers_for`` already encodes the support
    case for placement -- the thing holding you up does not obstruct you -- and it is re-derived
    here from the heights rather than from a list, so it stays true if the plan changes.
    """
    if first_id in floor_coverings or second_id in floor_coverings:
        return 0.0
    if (abs(first.z_low_m - second.z_high_m) <= CONTACT_TOLERANCE_M
            or abs(second.z_low_m - first.z_high_m) <= CONTACT_TOLERANCE_M):
        return 0.0
    if min(first.z_high_m, second.z_high_m) - max(first.z_low_m, second.z_low_m) <= 1e-6:
        return 0.0
    shared = sum(piece.area for piece in intersect_polygons(first.outline, second.outline))
    return shared if shared > 1e-9 else 0.0


def _assembly_refusal(scene: dict, *, dimensions: Any, position: Any,
                      rotation_wxyz: tuple[float, float, float, float],
                      wall_thickness_m: float) -> str:
    """Why a proposed instance may not enter this delivery, or ``""`` when it may.

    Two questions, both asked with the *gate's* own rule so the assembly can neither refuse what
    the gate would accept nor admit what the gate would reject:

    * `geometry.no_penetration` (`_pair_shared_area`): does it share space with a piece the
      delivery already places?
    * `geometry.in_bounds` (`_escape_of`): does its box stay inside the walls' inner face?

    Both were measured, not imagined. The first assembled delivery had a monitor sharing 0.0301 m2
    with the retrieved floor lamp; kitchen seed 11 kept a mirror frame 0.0400 m inside the wall
    solid. Neither is repairable later: the loop's pose repairs move the *retrieval* furniture, and
    a generated object's pose comes from the build.
    """
    candidate = Footprint((dimensions.x / 2.0, dimensions.y / 2.0)).placed_extent(
        dimensions.z, rotation=Quat(*rotation_wxyz), position=position,
        origin_convention="base")
    floor_z = scene["shell"].floor_z_m
    escape, kind = _escape_of(candidate, room=_inner_floor(scene, wall_thickness_m),
                              outline=scene["room_polygon"], floor_z=floor_z,
                              ceiling_z=floor_z + scene["shell"].height_m)
    if escape > 1e-6:
        return f"{kind}({escape:.4f}m)"
    floor_coverings = _floor_coverings(scene)
    worst, who = 0.0, ""
    for other_id, other in _placed_extents(scene):
        shared = _pair_shared_area("candidate", candidate, other_id, other, floor_coverings)
        if shared > worst:
            worst, who = shared, other_id
    if worst > 0.0:
        return f"retrieved-overlap({worst:.4f}m2,{who})"
    return ""


def _floor_coverings(scene: dict) -> set[str]:
    """The plan's floor coverings, read from the *plan* rather than from the module's rows.

    A plan a repair changed is the one judged: exempting a piece the current plan no longer marks
    as a floor covering would let a rug-shaped hole hide a real overlap.
    """
    return {request.entity_id for request in scene["plan"].furniture
            if request.kind is PlacementKind.FLOOR_COVERING}


def _geometry_no_penetration(scene: dict) -> tuple[bool, float, str, tuple[str, ...]]:
    """Whether any two placed instances occupy the same space at the same time.

    Returns ``(ok, worst_shared_area_m2, detail, witnesses)``.
    """
    floor_coverings = _floor_coverings(scene)
    placed = _placed_extents(scene)
    worst = 0.0
    offenders: list[str] = []
    witnesses: list[str] = []
    for index, (first_id, first) in enumerate(placed):
        for second_id, second in placed[index + 1:]:
            shared = _pair_shared_area(first_id, first, second_id, second, floor_coverings)
            if shared <= 0.0:
                continue
            offenders.append(
                f"{first_id} and {second_id} share {shared:.6f} m2 with overlapping heights "
                f"({first.z_low_m:.3f}..{first.z_high_m:.3f} vs "
                f"{second.z_low_m:.3f}..{second.z_high_m:.3f})"
            )
            witnesses.append(f"pair:{first_id}|{second_id}:overlap")
            worst = max(worst, shared)

    if offenders:
        return (
            False, worst,
            f"{len(offenders)} pair(s) of placed instances interpenetrate: " + "; ".join(offenders),
            tuple(witnesses),
        )
    return (
        True, 0.0,
        f"no two of the {len(placed)} placed instances occupy the same space: plan outlines that "
        f"overlap are separated in height, or one rests on the other",
        (),
    )


def _space_conditions(
    scene: dict, spec: SceneSpec,
) -> tuple[bool, float, str, tuple[str, ...]]:
    """Whether the robot has a wide enough way from the door to the work zone once furnished.

    Computed, not asserted. It used to be an unconditional PASS whose detail claimed "the desk zone
    leaves the approach from the door clear" -- a sentence nothing in the repository calculated, on
    a gate that ``FUNCTIONAL_SCENE_READY`` and above require. A desk seated across its own doorway
    passed every check in the run.

    The measurement is the robot's own centre-space: the room with each placed floor footprint
    grown by the robot's radius plus the configured margin, flood-filled from the door. Growing the
    obstacles makes it a *width* test rather than a connectivity test -- a slot narrower than the
    robot closes outright instead of reading as a way through -- and it is taken from the placed
    geometry with the zone plan deliberately not consulted, so the answer is about the room rather
    than about the plan's own claim.

    Returns ``(ok, measured_clearance_m, detail, witnesses)``.
    """
    from rfw_w.scenes.geometry.clearance import gap_between
    from rfw_w.scenes.geometry.polygon import (
        distance_to_boundary,
        offset_outward,
        subtract_polygons,
    )

    options = generate_options(spec)
    robot_m = options.robot_footprint_m
    margin_m = options.clearance_margin_m
    corridor_width_m = robot_m + 2 * margin_m

    shell = scene["shell"]
    doors = [opening for opening in shell.openings if opening.kind is OpeningKind.DOOR]
    if not doors:
        return False, 0.0, "the room has no door, so there is no requested entrance to reach", ()
    door_width_m = min(opening.width_m for opening in doors)
    if door_width_m < corridor_width_m:
        return (
            False, door_width_m,
            f"the {door_width_m:.3f} m door is narrower than the {corridor_width_m:.3f} m the "
            f"robot needs ({robot_m:.2f} m footprint plus {margin_m:.2f} m margin each side)",
            (),
        )

    zones: ZoneGraph = scene["zones"]
    # What the robot has to be able to reach. This is the task's own workspace -- the support, the
    # floor the robot may stand on, and the approach between them -- solved before the room existed
    # and carried through here in room coordinates.
    #
    # It used to be `zones.zone_for(ZoneKind.WORK)`: a strip on whichever wall happened to be
    # longest, which is a guess at where the task would end up rather than the place it did. The
    # docstring above says the measurement is taken "with the zone plan deliberately not consulted";
    # with the workspace in hand, that is now true of the goal as well as of the route.
    workspace = scene.get("workspace")
    if workspace is None:
        return (
            False, 0.0,
            "the scene records no task workspace, so there is nothing to measure an approach to", (),
        )
    if not zones.circulation:
        return (
            False, 0.0,
            "the plan reserved no passage from the door; the approach was never laid out", (),
        )

    # Every instance that stands on the floor, at the footprint it was actually placed with.
    occupants = [
        (entry.instance_id, Footprint((entry.dimensions_m.x / 2.0, entry.dimensions_m.y / 2.0))
         .rotated(entry.yaw_rad).at(entry.position_m.x, entry.position_m.y))
        for entry in scene["instances"]
        if entry.instance_id in scene["floor_occupants"]
    ]

    # The robot's centre-space, recomputed from the placed geometry: the room with every placed
    # floor footprint *grown* by the robot's radius plus the margin. Growing the obstacles rather
    # than shrinking the floor is what makes this a width test -- a passage narrower than the robot
    # plus its margin is closed outright -- and it keeps neighbouring free pieces sharing their
    # boundaries, so connectivity still means something. Shrinking each free piece instead recesses
    # it from the very edges that join it to its neighbours, and the floor comes apart everywhere.
    #
    # The zone plan is deliberately not consulted. The plan's own corridor is robot-width by
    # construction, so testing it would restate the construction rather than check the room; and it
    # is subtracted from the work zone by design, so it could never overlap the zone it is drawn to.
    probe_radius_m = robot_m / 2.0 + margin_m
    barriers = [offset_outward(footprint, probe_radius_m) for _, footprint in occupants]
    centre_space: list[Polygon] = [scene["room_polygon"]]
    for barrier in barriers:
        centre_space = [
            piece for region in centre_space for piece in subtract_polygons(region, barrier)
        ]
    witnesses = tuple(
        f"region:{scene['scene_id']}:centre-space/{index}"
        for index in range(len(centre_space))
    )
    if not centre_space:
        return False, 0.0, "the placed furniture fills the room; no floor is left to stand on", witnesses

    door_xy = doors[0].center_xy
    # The door's own point lies on the room boundary, so a free piece whose boundary carries it is
    # the one immediately inside the entrance; the workspace is reached when a free piece meets it.
    starts = [
        index for index, piece in enumerate(centre_space)
        if distance_to_boundary(piece, door_xy) <= 1e-6
    ]
    goals = [
        index for index, piece in enumerate(centre_space)
        if gap_between(piece, workspace) <= 1e-6
    ]
    if not starts:
        return (
            False, 0.0,
            "no floor survives inside the door once the robot's own width is accounted for: the "
            "entrance is obstructed",
            witnesses,
        )
    if not goals:
        return (
            False, 0.0,
            "the robot cannot stand in the task workspace: the furniture there leaves no floor wide "
            "enough for it",
            witnesses,
        )

    # Flood fill across pieces that meet, from what is inside the door.
    reached = set(starts)
    queue = list(starts)
    while queue:
        current = queue.pop()
        for index in range(len(centre_space)):
            if index in reached:
                continue
            if gap_between(centre_space[current], centre_space[index]) <= 1e-6:
                reached.add(index)
                queue.append(index)

    if not any(index in reached for index in goals):
        sealed = ", ".join(sorted(name for name, _ in occupants)) or "nothing placed"
        return (
            False, 0.0,
            f"the floor the robot can occupy does not reach the task workspace: the placed furniture "
            f"({sealed}) cuts the room in two between the entrance and the task surface",
            witnesses,
        )

    # The plan's own passage, cut down by what was placed, as a second and independent reading:
    # the zones are built so nothing stands in the corridor, so any intrusion is the plan and the
    # placement disagreeing. Reported either way rather than trusted.
    planned = list(zones.circulation)
    planned_area = sum(piece.area for piece in planned)
    passage = list(planned)
    for _, footprint in occupants:
        passage = [
            piece for corridor in passage for piece in subtract_polygons(corridor, footprint)
        ]
    intrusion_m2 = max(0.0, planned_area - sum(piece.area for piece in passage))

    return (
        True, corridor_width_m,
        f"the {door_width_m:.3f} m door admits the {corridor_width_m:.3f} m the robot needs "
        f"({robot_m:.2f} m footprint plus {margin_m:.2f} m margin each side), and the floor left "
        f"by all {len(occupants)} placed floor instances still connects the door to the task "
        f"workspace with the robot's width to spare; the reserved corridor lost {intrusion_m2:.4f} m2 of "
        f"{planned_area:.4f} m2 to it",
        witnesses,
    )


def _instance_polygon(entry: "Furnished"):
    """An instance's floor footprint, from the pose the build recorded for it."""
    from rfw_w.scenes.geometry.polygon import Polygon

    half_x = entry.dimensions_m.x / 2.0
    half_y = entry.dimensions_m.y / 2.0
    cos_yaw, sin_yaw = math.cos(entry.yaw_rad), math.sin(entry.yaw_rad)
    corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    return Polygon(tuple(
        (entry.position_m.x + x * cos_yaw - y * sin_yaw,
         entry.position_m.y + x * sin_yaw + y * cos_yaw)
        for x, y in corners
    ))


def _task_reachable(scene: dict, spec: SceneSpec):
    """Whether the robot can get to the piece the task is about.

    A **reach proxy**, and it says so in its method and its detail. It answers three geometric
    questions and no kinematic ones: is there a footprint-sized patch of floor a base could
    occupy, is the target inside the arm's reach from there, and is the lane between the two clear
    at the width a gripper needs. ``robot.ik`` stays ``not_run`` -- this is not a grasp, and
    reporting it as one would be the overclaim the manual forbids.

    It exists because none of the twenty-six gates had a *robot* to measure against, so the
    question "can this be picked up" was not merely unchecked but unrepresentable. A visual critic
    found a real instance of it -- an alarm clock at the rear corner of a nightstand, with the
    gripper's approach directions restricted -- and every gate passed, because not one of them
    knew how far an arm reaches.

    Returns ``(ok, measured, detail, witnesses)``.
    """
    from rfw_w.scenes.geometry.polygon import inset
    from rfw_w.scenes.geometry.robot import RobotError, profile_for, standoffs_for

    try:
        profile = profile_for(spec.task.robot_profile)
    except RobotError as error:
        # A spec naming a robot this release has no numbers for. Not a pass: the reach of an
        # unknown arm is unknown, and saying "reachable" would be asserting something unmeasured.
        return False, 0.0, str(error), ()

    target_pos = scene.get("target_pos")
    if target_pos is None:
        return False, 0.0, "the plan placed no task target, so there is nothing to reach", ()

    inner = inset(scene["room_polygon"], scene["wall_thickness_m"])
    support = getattr(scene.get("support_top"), "polygon", None)
    # Everything placed except the target and the surface it stands on. The support is passed
    # separately because the arm reaches *over* it: counted as an obstruction it stands between
    # the target and every base position there is, and the gate would fail on a mug in the middle
    # of a desk.
    obstacles = tuple(
        _instance_polygon(entry) for entry in scene["instances"]
        if entry.instance_id not in (scene.get("target_id"), scene.get("support_id"))
    )
    target_xy = (target_pos.x, target_pos.y)
    standoffs = standoffs_for(
        target_xy, region=inner, obstacles=obstacles, profile=profile, support=support
    )
    directions = len({
        round(math.atan2(item.centre[1] - target_xy[1], item.centre[0] - target_xy[0]), 3)
        for item in standoffs
    })
    witnesses = tuple(
        f"{scene.get('target_id', 'target')} standoff ({item.centre[0]:.3f}, {item.centre[1]:.3f})"
        for item in standoffs[:6]
    )
    measured = float(directions)
    if not standoffs:
        return (
            False, 0.0,
            f"TASK_UNREACHABLE: no base position for the {profile.name} satisfies all three of "
            f"fitting in the room, clearing what is placed, and leaving a "
            f"{profile.approach_clearance_m:.2f} m lane to {scene.get('target_id', 'the target')} "
            f"at ({target_xy[0]:.3f}, {target_xy[1]:.3f}); the arm reaches {profile.reach_m:.3f} m",
            (),
        )
    return (
        True, measured,
        f"the {profile.name} reaches {scene.get('target_id', 'the target')} from "
        f"{len(standoffs)} base position(s) covering {directions} approach direction(s), each "
        f"inside the room, clear of the {len(obstacles)} other placed instance(s), and with a "
        f"{profile.approach_clearance_m:.2f} m lane; measured as a reach proxy, not a grasp",
        witnesses,
    )
