# SPDX-License-Identifier: Apache-2.0
"""Camera fitting and Isaac request construction from assembled geometry."""

from __future__ import annotations
import math
from dataclasses import replace
from typing import Any
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.geometry.polygon import Polygon, contains_point, distance_to_boundary, interior_point, point_inside_toward
from rfw_w.scenes.assembly.geometry import _inner_floor
from rfw_w.scenes.runtime import step


def _vantage(polygon: Polygon, target: tuple[float, float], desired: tuple[float, float],
             *, margin: float = 0.35, minimum_clearance_m: float = 0.0) -> tuple[float, float] | None:
    """A camera position looking at ``target``, constrained to be inside ``polygon``.

    ``None`` when ``polygon`` has nowhere that clears ``minimum_clearance_m``. The caller owns that
    case rather than this function inventing a point: see the note in `point_inside_toward`.
    """
    return point_inside_toward(
        polygon, target, desired, margin=margin, minimum_clearance_m=minimum_clearance_m
    )


def fitted_focal(eye, target, points, *, aperture_h_mm: float, resolution: tuple, margin: float = 1.06,
                 minimum_forward_m: float = 0.5) -> tuple:
    """The longest lens from ``eye`` (looking at ``target``) that still keeps every point in frame.

    SPEC-015's rule, and the reason the camera set could fail a room it never photographed: the
    delivered set held fixed focals (14 / 14 / 22 / 18 / 35 / 20 mm), so a 67.62 m2 room got the
    same 4.06 m of coverage a 18.08 m2 room got, and two views framed nothing but floor and wall.

    The aperture model is measured, not assumed (measured view geometry,
    2026-09-24): the renderer reproduces the declared **horizontal** aperture (four independent
    views agree within 3% of 20.955 mm) and derives the vertical one from the image aspect -- pixels
    are square, so `fy == fx == f * width / aperture_h`. That is what `AGENTS.md` recorded as
    "~12.4 mm effective vertical aperture"; the declared `vertical_aperture_mm` changes nothing.

    Returns ``(focal_mm, tan_half_horizontal, tan_half_vertical)``. A point behind or beside the
    camera is not this view's obligation and is skipped: a wall-hugging side view cannot contain the
    corner it stands in, and pretending otherwise would make every close shot "fail".
    """
    eye = (float(eye[0]), float(eye[1]), float(eye[2]))
    target = (float(target[0]), float(target[1]), float(target[2]))
    forward = [target[index] - eye[index] for index in range(3)]
    length = math.sqrt(sum(component * component for component in forward))
    if length <= 1e-9:
        raise ValueError("fitted_focal got an eye and a target in the same place")
    forward = [component / length for component in forward]
    right = [forward[1], -forward[0], 0.0]
    right_length = math.hypot(*right[:2])
    if right_length <= 1e-9:
        raise ValueError("fitted_focal cannot define roll for a camera looking straight down")
    right = [component / right_length for component in right]
    up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]

    width_px, height_px = int(resolution[0]), int(resolution[1])
    tan_half_h = tan_half_v = 0.0
    for point in points:
        offset = [float(point[index]) - eye[index] for index in range(3)]
        forward_m = sum(offset[i] * forward[i] for i in range(3))
        if forward_m <= minimum_forward_m:
            continue
        tan_half_h = max(tan_half_h, abs(sum(offset[i] * right[i] for i in range(3)) / forward_m))
        tan_half_v = max(tan_half_v, abs(sum(offset[i] * up[i] for i in range(3)) / forward_m))
    # horizontal: fx = f * width / aperture_h, so tan(half_h) = width * aperture_h / (2 * f * width)
    # ... which simplifies to tan(half_h) = aperture_h / (2 * f), i.e. f = aperture_h / (2 tan).
    focal_h = aperture_h_mm / (2.0 * tan_half_h * margin) if tan_half_h > 0 else math.inf
    # vertical: fy = fx, so tan(half_v) = (height_px/2) / fx = height_px * aperture_h / (2 * f * width)
    focal_v = (aperture_h_mm * height_px) / (
        2.0 * width_px * tan_half_v * margin) if tan_half_v > 0 else math.inf
    return min(focal_h, focal_v), tan_half_h, tan_half_v


def layout_obligation(zones, room_polygon, target_xy, *, floor_z_m: float,
                      support_top_z: float) -> list:
    """What the layout view owes an image of: the reserved task cell and the task target.

    Not the whole footprint: fitted to that, the lens became 4.93 mm in an 18 m2 office and the
    desk it exists to show became a speck (`workspace_and_passage` 4 -> 2 on the same scene).
    ``zones`` is the scene's `ZoneGraph`, not its dict form -- reading it as a mapping raised
    AttributeError inside the request build, which the orchestrator turns into a failed stage with
    no traceback in the log (measured; see SPEC-015's PLAN).
    """
    workspace = getattr(zones, "workspace", None)
    cell = [(float(vertex[0]), float(vertex[1])) for vertex in (workspace.to_list() if workspace else [])]
    if not cell:
        # A plan that reserves no cell falls back to the footprint rather than to a lens with
        # nothing to contain, which `fitted_focal` would answer with infinity.
        cell = [(float(vertex[0]), float(vertex[1])) for vertex in room_polygon.vertices]
    return ([(x, y, floor_z_m) for x, y in cell]
            + [(float(target_xy[0]), float(target_xy[1]), support_top_z)])


def long_edge_side_view(polygon, *, floor_z_m: float, height_m: float, eye_height_m: float = 1.50,
                        inset_m: float = 0.15, clearance_m: float = 0.25,
                        aperture_h_mm: float = 20.955,
                        resolution: tuple = (1280, 720), margin: float = 1.06) -> dict:
    """ADR-011's overview: stand at the long edge's centre, look across, fit the lens to the room.

    The obligation is geometric and checkable before anything renders: every footprint vertex in the
    room's forward half has to land inside the frame, and so does the room's height. Returns the
    eye, the target, the fitted focal and the coverage, and raises when the footprint has no legal
    vantage -- the caller turns that into a named gap instead of spending a render on it.
    """
    vertices = [(float(vertex[0]), float(vertex[1])) for vertex in polygon.vertices]
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    long_axis_is_x = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    mid = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    # Stand on the long edge's centre line, as close to that edge as the walls allow. Solved with
    # containment and clearance rather than `_vantage`: that helper aims at a target, and for a
    # U-shaped room the target (the bounding box's centre) is inside the notch -- it answered by
    # sliding the eye 2.9 m sideways, off the centre line the rule is about.
    eye_xy = None
    for sample_index in range(80):
        offset = inset_m + sample_index * 0.05
        candidate = ((mid[0], min(ys) + offset) if long_axis_is_x else (min(xs) + offset, mid[1]))
        if (contains_point(polygon, candidate)
                and distance_to_boundary(polygon, candidate) >= clearance_m):
            eye_xy, moved_m = candidate, offset
            break
    if eye_xy is None:
        raise ValueError("VIEW_NO_VANTAGE(long_edge_side_view): the footprint has no point on the "
                         "long edge's centre that clears the walls")
    # The target is where the ray leaving the eye meets the footprint again -- *not* the bounding
    # box's far side, which for a U- or L-shaped room is not a wall at all: measured on a U whose
    # notch opens onto its far edge, the bbox's centre point sits in the notch, outside the room.
    direction = (0.0, 1.0) if long_axis_is_x else (1.0, 0.0)
    farthest = None
    for index in range(len(vertices)):
        p1, p2 = vertices[index], vertices[(index + 1) % len(vertices)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        denominator = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(denominator) < 1e-12:
            continue                       # parallel to the ray
        to_edge = (p1[0] - eye_xy[0], p1[1] - eye_xy[1])
        along_ray = (to_edge[0] * edge[1] - to_edge[1] * edge[0]) / denominator
        along_edge = (to_edge[0] * direction[1] - to_edge[1] * direction[0]) / denominator
        if along_ray > 0 and 0.0 <= along_edge <= 1.0:
            if farthest is None or along_ray > farthest:
                farthest = along_ray
    if farthest is None:
        raise ValueError("VIEW_NO_VANTAGE(long_edge_side_view): the eye cannot see any wall "
                         "across the room")
    back_off_m = 0.05
    target_xy = (eye_xy[0] + direction[0] * max(farthest - back_off_m, back_off_m),
                 eye_xy[1] + direction[1] * max(farthest - back_off_m, back_off_m))
    eye = (eye_xy[0], eye_xy[1], floor_z_m + eye_height_m)
    target = (target_xy[0], target_xy[1], floor_z_m + eye_height_m)
    distance = math.dist(eye_xy, target_xy)

    forward_half = [vertex for vertex in vertices
                    if math.dist(vertex, eye_xy) >= 0.5 * distance] or vertices
    points = ([(vertex[0], vertex[1], floor_z_m) for vertex in forward_half]
              + [(vertex[0], vertex[1], floor_z_m + height_m) for vertex in forward_half])
    focal_mm, tan_h, tan_v = fitted_focal(eye, target, points, aperture_h_mm=aperture_h_mm,
                                         resolution=resolution, margin=margin)
    return {"eye": eye, "target": target, "focal_length_mm": round(focal_mm, 3),
            "distance_m": distance, "eye_offset_from_edge_m": moved_m,
            "tan_half_horizontal": tan_h, "tan_half_vertical": tan_v}


def build_worker_request(scene: dict, spec: SceneSpec, *, scene_version: str,
                         lighting: Any = None) -> dict:
    """The Isaac worker request, built from every placed instance."""
    from rfw_w.scenes.workers.isaac.protocol import (
        CameraRequest, LightingRequest, MeshInstance, RoomGeometry, RoomMaterialsRequest,
        RoomOpening, WorkerRequest,
    )

    shell = scene["shell"]
    room_polygon = scene["room_polygon"]
    target_id = scene["target_id"]
    instances: list[MeshInstance] = []
    for entry in scene["instances"]:
        instances.append(MeshInstance(
            instance_id=entry.instance_id, entity_id=entry.entity_id,
            visual_uri=entry.visual_uri, collision_uri=entry.collision_uri,
            translation=(entry.position_m.x, entry.position_m.y, entry.position_m.z),
            # The recorded rotation, not one rebuilt from the yaw. A wall-hung piece is rotated
            # about X as well, and rebuilding from the yaw silently laid it back down -- the
            # worker would have been told to render an upright picture as a horizontal slab.
            rotation_wxyz=(entry.rotation_wxyz if entry.rotation_wxyz is not None
                           else (math.cos(entry.yaw_rad / 2), 0.0, 0.0,
                                 math.sin(entry.yaw_rad / 2))),
            mass_kg=entry.mass_kg, dynamic=entry.dynamic, normalisation=entry.normalisation,
        ))

    centre = interior_point(room_polygon, margin=0.7)

    # The fixed view set from architecture §9: a top-down layout view, two eye-level interior views,
    # and a task closeup per relation. Every eye position is produced by the vantage search rather
    # than by arithmetic on a target, because arithmetic is what put a camera inside a wall and then
    # inside the L-shaped notch.
    layout_target = (centre[0], centre[1], shell.floor_z_m)
    # Framing is chosen from the geometry, not by eye.
    #
    # A 24 mm lens on a 16:9 frame has a *vertical* half-angle of only 23 degrees, so a camera aimed
    # at the middle of a 1.4 m desk puts a cup at one end of it 30 degrees off-axis and outside the
    # picture -- which is what happened, and the render showed an empty tabletop. Each camera below
    # therefore aims within a few degrees of the thing it is required to show, and the target's own
    # extent sets the standoff.
    target_xy = (scene["target_pos"].x, scene["target_pos"].y)
    support_top_z = scene["support_top"].plane_z_m

    # A camera has to stand in the room's **air**, not merely inside its outline. The outline is the
    # wall's outer face and the wall body extends inward from it, so the outline admits points that
    # are inside a wall -- which is where `room_eye_north` ended up (0.076 m from the boundary,
    # against an inner face at ~0.1 m), and why three of the seven views rendered one uniform
    # surface with a single UNLABELLED id. `_inner_floor` is the region a person can occupy, and it
    # is what placement and every geometry gate already measure against; the camera set was the last
    # consumer still calling the outline "the room".
    camera_room = _inner_floor(scene, scene["wall_thickness_m"])
    # How far an eye must stay from a wall or a corner. Roughly a person's half-shoulder: closer
    # than this and the shot is of the wall it is beside.
    CAMERA_CLEARANCE_M = 0.25
    # How many directions around the target the aim solver may consider, beyond the one asked for.
    CAMERA_DIRECTIONS = 16
    camera_notes: list[str] = []

    def _aim(label, target_xy, standoff_xy, height):
        """Eye position: in the room's air, at ``height``, looking at the target.

        The standoff's *direction* is the intent of the shot, and the ring below is what keeps the
        intent from being a cage. When the ladder moves the target against a wall, a single requested
        direction has nowhere to stand and the solve degrades to whatever the fallback path can
        reach -- which is how the previous version produced an eye inside the wall. So the requested
        direction is tried first and every direction gets a fair attempt; the requested one wins
        unless another gives materially more room to stand back, which is the rule
        `point_inside_toward` already applies between its two sides.
        """
        reach = math.hypot(standoff_xy[0], standoff_xy[1])
        requested_angle = math.atan2(standoff_xy[1], standoff_xy[0])
        best: tuple[float, tuple[float, float], int] | None = None
        for index in range(CAMERA_DIRECTIONS + 1):
            angle = (
                requested_angle if index == 0
                else requested_angle + 2.0 * math.pi * index / CAMERA_DIRECTIONS
            )
            desired = (
                target_xy[0] + math.cos(angle) * reach,
                target_xy[1] + math.sin(angle) * reach,
            )
            candidate = _vantage(
                camera_room, target_xy, desired,
                margin=CAMERA_CLEARANCE_M, minimum_clearance_m=CAMERA_CLEARANCE_M,
            )
            if candidate is None:
                continue
            standoff = math.dist(target_xy, candidate)
            score = standoff * (1.25 if index == 0 else 1.0)
            if best is None or score > best[0]:
                best = (score, candidate, index)
        if best is None:
            # The room is too tight for a camera to stand anywhere clear of its walls. Say so, and
            # put the eye at the room's own interior point rather than at a point the solve refused:
            # an honest compromise the run's log carries, not a silent one inside a wall.
            eye = interior_point(camera_room, margin=0.05)
            camera_notes.append(
                f"{label}: no vantage clearing {CAMERA_CLEARANCE_M:.2f} m anywhere in the room; the "
                f"eye is the room's interior point and the shot is not framed as asked"
            )
            return (eye[0], eye[1], height)
        if best[2] != 0:
            degrees = math.degrees(2.0 * math.pi * best[2] / CAMERA_DIRECTIONS)
            camera_notes.append(
                f"{label}: the requested direction has no room to stand (target radius "
                f"{reach:.2f} m); aimed {degrees:.0f} degrees around it instead"
            )
        eye = best[1]
        return (eye[0], eye[1], height)

    # --- whole-room overview ---------------------------------------------------------------
    # Nothing in the five views below shows the room as a whole. Measured from the delivered
    # instance-id channels, four of the twelve instances -- floor_lamp, plant, wall_art and
    # ceiling_light -- have zero pixels across all five, so the visual critique scored
    # `workspace_and_passage: 5` on a third of the scene it had never seen. That is the reason
    # this pair exists: a critic that cannot see the room cannot drive a repair of it.
    #
    # `review_visual` montages every PNG in <out>/isaac, so these join the critique with no
    # further wiring.
    #
    # Their framing is calibrated rather than derived. The nominal USD aperture is 36 mm
    # horizontal by 20.25 mm vertical, but a render measured at 8 m with a 24 mm lens covers
    # 3.9 m, which back-solves to an effective vertical aperture near 12.4 mm -- close to the
    # 20.955 mm 35 mm-still width, and consistent across two renders that agree with each other.
    # So every camera in this set is about 1.7x narrower than its focal length implies, which is
    # the direct cause of the display corner never fitting in frame. The constant below is that
    # measurement; if the worker ever pins the aperture explicitly it should be deleted, not
    # retuned.
    OVERVIEW_APERTURE_V_M = 12.4
    OVERVIEW_TILT_RAD = math.radians(7.0)
    OVERVIEW_EYE_Z_M = 8.0
    OVERVIEW_MARGIN_M = 0.4

    room_xs = [vertex[0] for vertex in room_polygon.vertices]
    room_ys = [vertex[1] for vertex in room_polygon.vertices]
    room_mid_xy = ((min(room_xs) + max(room_xs)) / 2.0, (min(room_ys) + max(room_ys)) / 2.0)
    # Anchor the perspective overview to the solved task. A fixed offset from the room's
    # bounding box can point at an empty alcove when prompt compilation changes the shell.
    corner_target_xy = target_xy

    # Focal chosen so the frame's short axis spans the room plus a margin, at the height above.
    def _overview_span(focal_mm: float) -> float:
        half_v = math.atan((OVERVIEW_APERTURE_V_M / 2.0) / focal_mm)
        low = math.tan(OVERVIEW_TILT_RAD - half_v)
        high = math.tan(OVERVIEW_TILT_RAD + half_v)
        return OVERVIEW_EYE_Z_M * (high - low)

    wanted_span = (max(room_ys) - min(room_ys)) + OVERVIEW_MARGIN_M
    # The smallest focal that still spans the room, so the room fills the frame rather than sitting
    # in the middle of it. Starting from the widest lens tried means a room too big for any of them
    # gets the widest view available instead of silently keeping an unfit default.
    focal_candidates = [value / 2.0 for value in range(20, 200)]   # 10..99.5 mm in 0.5 mm steps
    overview_focal_mm = focal_candidates[0]
    for candidate in focal_candidates:
        if _overview_span(candidate) <= wanted_span:
            overview_focal_mm = candidate
            break

    # The ground point is placed so the covered band is *centred* on the room: a tilted camera's
    # frame is asymmetric about its aim, so aiming at the centre would cut one side.
    half_v_overview = math.atan((OVERVIEW_APERTURE_V_M / 2.0) / overview_focal_mm)
    low_tan = math.tan(OVERVIEW_TILT_RAD - half_v_overview)
    high_tan = math.tan(OVERVIEW_TILT_RAD + half_v_overview)
    overview_ground_y = room_mid_xy[1] - OVERVIEW_EYE_Z_M * (low_tan + high_tan) / 2.0
    # A near-nadir camera would have its up vector parallel to (0,0,1) and SetLookAt's roll would
    # be arbitrary -- the first attempt came back rotated 90 degrees, putting the room's long axis
    # on the frame's short axis. The 7 degree tilt is what makes the roll well defined.
    overview_eye = (room_mid_xy[0], overview_ground_y, OVERVIEW_EYE_Z_M)
    overview_target = (room_mid_xy[0], room_mid_xy[1], scene["shell"].floor_z_m)

    # ADR-011's pre-flight: the whole-room side view is fitted before anything renders, and a
    # footprint this rule cannot frame is a named gap -- the run says which room it could not
    # photograph instead of spending a render on a blank frame (SPEC-015 R4).
    try:
        side_view = long_edge_side_view(
            room_polygon, floor_z_m=scene["shell"].floor_z_m, height_m=scene["shell"].height_m
        )
        camera_notes.append(
            "overview_side: fitted %.2f mm at a %.2f m standoff from the long edge (eye %.2f m in)"
            % (side_view["focal_length_mm"], side_view["distance_m"],
               side_view["eye_offset_from_edge_m"])
        )
    except ValueError as error:
        side_view = None
        camera_notes.append(str(error))

    # The top view's lens, fitted the same way and for the same reason: it has to contain the whole
    # footprint and the task target from its own eye, and a fixed 14 mm did that only by accident.
    layout_eye = ((centre[0] + target_xy[0]) / 2.0, (centre[1] + target_xy[1]) / 2.0,
                  scene["shell"].floor_z_m + scene["shell"].height_m - 0.15)
    layout_target = (layout_eye[0] + 0.02, layout_eye[1], scene["shell"].floor_z_m)
    layout_top_focal_mm = round(fitted_focal(
        layout_eye, layout_target,
        layout_obligation(scene.get("zones"), room_polygon, target_xy,
                          floor_z_m=scene["shell"].floor_z_m, support_top_z=support_top_z),
        aperture_h_mm=20.955, resolution=(1280, 720),
    )[0], 3)

    cameras = [
        CameraRequest(
            camera_id="overview_top", purpose="export",
            # Every delivered view owes at least one instance (SPEC-015 R3): a view that frames
            # none is a named gap for itself rather than a failed delivery (R4).
            minimum_visible_instances=1,
            # Above the roof, and that works because the ceiling mesh is authored
            # `MakeInvisible()` in the worker, so a camera outside the shell sees straight in.
            eye=overview_eye, target=overview_target,
            focal_length_mm=overview_focal_mm, resolution=(1280, 720),
            target_instance_ids=(), require_target_visible=False,
            save_depth=True, save_instance_ids=True,
        ),
        CameraRequest(
            camera_id="overview_corner", purpose="export",
            minimum_visible_instances=1,
            # Inside the room, high in a corner, looking across it. Being inside means the walls
            # are behind the camera rather than between it and the floor, which is the only way a
            # perspective view can show the whole room.
            #
            # Solved rather than placed at the bounding box's corner. It was `min(room_xs) + 0.40,
            # min(room_ys) + 0.35`, which is only inside the room while that corner is intact --
            # and the shell now draws which corner is bitten, so for two of the three experiments
            # the eye stood in the notch, outside the walls. Nothing said so: the render came back
            # non-blank, and the only trace was one camera with no renderer id mapping, which cost
            # a full render to discover. This is the reason for the containment check below.
            eye=_aim("overview_corner", corner_target_xy,
                     (min(room_xs) + 0.40 - corner_target_xy[0],
                      min(room_ys) + 0.35 - corner_target_xy[1]),
                     scene["shell"].floor_z_m + 2.65),
            target=(corner_target_xy[0], corner_target_xy[1], support_top_z),
            focal_length_mm=14.0, resolution=(1280, 720),
            target_instance_ids=(), require_target_visible=False,
            save_depth=True, save_instance_ids=True,
        ),
        CameraRequest(
            camera_id="layout_top", purpose="layout",
            minimum_visible_instances=1,
            # Centred between the room and the task target, because a top view centred only on the
            # room leaves a cup at one end of a desk 45 degrees off-axis. The lens used to be a
            # fixed 14 mm, which was measured to cover 4.06 m whatever the room's size -- in a
            # 67.62 m2 room its whole frame was floor. It is now fitted to this room's footprint
            # and to the task target, which is what this view owes an image of (SPEC-015 R1).
            eye=((centre[0] + target_xy[0]) / 2, (centre[1] + target_xy[1]) / 2,
                 shell.floor_z_m + shell.height_m - 0.15),
            target=((centre[0] + target_xy[0]) / 2 + 0.02, (centre[1] + target_xy[1]) / 2, shell.floor_z_m),
            focal_length_mm=layout_top_focal_mm, resolution=(1280, 720),
            target_instance_ids=(), require_target_visible=False,
            save_depth=True, save_instance_ids=True,
        ),
        CameraRequest(
            camera_id="room_eye_north", purpose="final",
            minimum_visible_instances=1,
            eye=_aim("room_eye_north", target_xy, (0.55, 1.45), support_top_z + 0.80),
            target=(target_xy[0], target_xy[1], support_top_z + 0.10),
            focal_length_mm=22.0, resolution=(1280, 720),
            target_instance_ids=(), require_target_visible=False,
            save_depth=True, save_instance_ids=True,
        ),
        *([] if side_view is None else [CameraRequest(
            camera_id="overview_side", purpose="export",
            minimum_visible_instances=1,
            # ADR-011 replaced the second eye-level view with this one: a second eye-level shot
            # anchored 1.9 m off the room's centre framed nothing once the room grew, while a
            # side view from the long edge's centre is the one framing that scales -- its lens is
            # fitted to the room's own forward half (see `long_edge_side_view`).
            eye=side_view["eye"], target=side_view["target"],
            focal_length_mm=side_view["focal_length_mm"], resolution=(1280, 720),
            target_instance_ids=(), require_target_visible=False,
            save_depth=True, save_instance_ids=True,
        )]),
        CameraRequest(
            camera_id="task_closeup", purpose="relation",
            minimum_visible_instances=1,
            eye=_aim("task_closeup", target_xy, (0.42, 0.62), support_top_z + 0.52),
            target=(target_xy[0], target_xy[1], support_top_z + 0.05),
            focal_length_mm=35.0, resolution=(1280, 720),
            target_instance_ids=(target_id,), require_target_visible=True,
            save_depth=True, save_instance_ids=True,
        ),
        CameraRequest(
            camera_id="desk_workspace", purpose="relation",
            minimum_visible_instances=1,
            # Aimed at the cup itself rather than at the desk's centre, with a wider lens so the
            # desk and the objects sharing it are in frame as well.
            eye=_aim("desk_workspace", target_xy, (0.95, 1.35), support_top_z + 0.92),
            target=(target_xy[0], target_xy[1], support_top_z + 0.03),
            focal_length_mm=20.0, resolution=(1280, 720),
            target_instance_ids=(target_id,), require_target_visible=True,
            save_depth=True, save_instance_ids=True,
        ),
    ]

    if scene.get("shell_uri"):
        # A real roof blocks the historical camera at z=8 m. Keep the reference intact and
        # photograph from inside, with a lens fitted to the footprint using the measured
        # 12.4 mm effective vertical aperture (AGENTS.md camera calibration).
        height = shell.height_m - 0.20
        span = max(max(room_xs) - min(room_xs), max(room_ys) - min(room_ys))
        cameras = [replace(c, eye=(c.eye[0], c.eye[1], shell.floor_z_m + height),
                           focal_length_mm=min(14.0, 12.4 * height / (1.1 * span)))
                   if c.camera_id == "overview_top" else c for c in cameras]

    # Every view that had to be aimed somewhere other than where it was asked to look. A silent
    # change of aim is the sort of thing that gets read as "the shot is fine": the run says which
    # view moved and by how much, so a room too tight to photograph the way the set intends is
    # visible in the log rather than only in a render that came out strange.
    for note in camera_notes:
        step("cameras", note)

    # Nothing checked a camera before a render was spent on it, and the first thing that noticed an
    # unusable one was a blank or half-usable view minutes of Isaac later. Measured, on two of the
    # three experiments: `overview_corner`'s eye sat in the outline's notch, outside the walls,
    # and the only trace was that one camera came back with no renderer id mapping -- after the
    # whole render set had been paid for. The eye must be in the room's air, and the room's air
    # ends at the wall's *inner* face, so that is what this checks against.
    outside = []
    for camera in cameras:
        eye = (camera.eye[0], camera.eye[1])
        if not contains_point(camera_room, eye):
            outside.append(
                f"{camera.camera_id} at ({eye[0]:.3f}, {eye[1]:.3f}), "
                f"{distance_to_boundary(camera_room, eye):.3f} m outside the inner wall face"
            )
    if outside:
        raise SystemExit(
            "CAMERA_OUTSIDE_THE_ROOM: the eye of a camera is not in the room's air, so the render "
            "would show a wall or nothing at all: " + "; ".join(outside)
        )

    return WorkerRequest(
        scene_id=spec.scene_id,
        scene_version=scene_version,
        room=RoomGeometry(
            polygon_xy=shell.polygon_xy, height_m=shell.height_m, floor_z_m=shell.floor_z_m,
            shell_uri=scene.get("shell_uri"),
            openings=tuple(
                RoomOpening(
                    opening_id=opening.opening_id, center_xy=opening.center_xy,
                    width_m=opening.width_m, height_m=opening.height_m,
                    sill_m=opening.sill_z_m, kind=opening.kind.value,
                )
                for opening in shell.openings
            ),
        ),
        instances=tuple(instances),
        cameras=tuple(cameras),
        simulate_seconds=5.0,
        physics_hz=120,
        # The plan's lighting, carried to the renderer. Without this the worker used its own
        # hardcoded numbers and the appearance track could not change the picture it re-rendered --
        # see LightingRequest and Known gaps 13 in AGENTS.md. With a `LightingPlan` (SPEC-008
        # TASK 3) the numbers are the calibrated profile's, sized to this room; a caller that passes
        # none keeps the old appearance path, which is what the recorded pre-TASK-3 deliveries used.
        lighting=(LightingRequest.from_plan(lighting) if lighting is not None
                  else LightingRequest.from_appearance(scene["plan"].appearance)),
        # The plan's room surfaces, on the same reasoning: the shell's materials were module
        # constants in the worker, so `StylePlan.palette` reached nothing and two specs asking for
        # different colour families rendered identical rooms.
        room_materials=RoomMaterialsRequest.from_materials(scene["plan"].room_materials),
    )
