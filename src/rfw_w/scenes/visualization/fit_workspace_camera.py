#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit the SPEC-014 workspace camera to what the figure must actually show.

Why this exists
---------------
The first workspace camera was hand-written
(``experiments/paper_dollhouse/multisim_workspace_camera.json``) and its
``target`` was set to the cup's own position, ``[1.04371, 2.0519, 0.78]``. Its
comment says so outright: "Task desk and manipulable cup". But the cup is the
**edge** of the task cluster, not its centre -- cup, notebook and pen holder
together span x[0.649, 1.093] y[2.001, 2.679], whose centre is
``[0.871, 2.340, 0.809]``. Aiming at one member of a cluster, from 1.34 m,
pushed the desk across two thirds of the frame and cropped the notebook on the
right edge.

The rule here
-------------
Keep the target on the task cluster's centre, keep the bearing the old camera
looked along (so the figure keeps its established three-quarter view), and keep
square publication pixels -- the shared-K argument in ``extract_scene.py``'s
``figure_camera`` applies unchanged: only Isaac can honour two apertures, so the
five-engine claim is only falsifiable with square pixels.

Two free parameters are then solved in order, and the order is forced by the
room. Backing the camera straight off until everything fits wants 2.89 m along
that bearing, which puts the eye at y=4.11 in a room that ends at y=3.686 --
outside, behind the north wall. So distance is taken as far as the room allows
with a declared wall clearance, and the remaining shortfall is taken out of the
focal length, exactly as ``fit_focal_length`` does for the room camera: the
longest focal at which every required box is inside the frame.

Longest, not shortest: the task objects are the subject, and a wider lens than
necessary shrinks them for no gain.

Run
---
    env -u PYTHONPATH -u PYTHONHOME \
      python -m rfw_w.scenes.visualization.fit_workspace_camera \
        --scene outputs/_paper/spec014_common_workspace_20260925_01 \
        --output experiments/paper_multisim/workspace_camera_v2.json
"""

from __future__ import annotations
from rfw_w.scenes.visualization.inputs import resolve_input
from rfw_w.scenes.resources import data_path, module_path

import argparse
import json
import sys
from pathlib import Path

import numpy as np


from rfw_w.scenes.workers.multisim.common import CameraSpec, CommonScene

# The figure's subject. The three manipulable objects must be fully in frame;
# the desk is the surface they sit on and gives the reader the context, so its
# top must be in frame too. Everything else in the room may be cropped -- a
# camera this close to a desk cannot contain the room and should not try.
REQUIRED_TASK_OBJECTS = ("cup", "notebook", "pen_holder")
REQUIRED_SUPPORT = "desk"

# Publication intrinsics. The focal length is the fitted one; the aperture and
# resolution are unchanged from the first workspace camera.
WIDTH, HEIGHT = 1600, 1000
HORIZONTAL_APERTURE_MM = 20.955
MAX_FOCAL_MM = 20.0

# How close the eye may come to a wall or the ceiling. An eye placed exactly on
# a wall plane renders that wall as a near-field smear across the frame.
WALL_CLEARANCE_M = 0.15

# 4% of the short side. The old camera put the notebook's box literally on the
# frame edge; a margin is what makes "not cropped" a measurement instead of an
# opinion.
MARGIN = 0.04


def world_bbox(mesh: Path, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounds of an OBJ under its instance transform."""
    vertices = []
    with Path(mesh).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
    if not vertices:
        raise SystemExit(f"MULTISIM_GEOMETRY_MISMATCH(empty mesh: {mesh})")
    world = (np.asarray(vertices, dtype=np.float64) @ matrix[:3, :3].T) + matrix[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def corners_of(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def distance_inside_room(target: np.ndarray, bearing: np.ndarray, room: dict) -> tuple[float, str]:
    """How far along ``bearing`` the eye may travel and stay in the room.

    Returns the distance and which surface stopped it, so the report says why
    the camera sits where it does rather than asserting a number.
    """
    polygon = room["polygon_xy"]
    lo = np.array([min(p[0] for p in polygon), min(p[1] for p in polygon), 0.0])
    hi = np.array([max(p[0] for p in polygon), max(p[1] for p in polygon),
                   float(room["height_m"])])
    lo = lo + WALL_CLEARANCE_M
    hi = hi - WALL_CLEARANCE_M
    best, why = np.inf, "unbounded"
    for axis, name in enumerate("xyz"):
        if abs(bearing[axis]) < 1e-9:
            continue
        bound = hi[axis] if bearing[axis] > 0 else lo[axis]
        travel = (bound - target[axis]) / bearing[axis]
        if 0 < travel < best:
            best, why = travel, f"{name}{'max' if bearing[axis] > 0 else 'min'}={bound:.3f}"
    return float(best), why


def camera_at(distance: float, focal_mm: float, target: np.ndarray,
              bearing: np.ndarray) -> CameraSpec:
    return CameraSpec(
        name="paper_multisim_workspace_v2",
        eye=target + bearing * distance,
        target=target,
        width=WIDTH,
        height=HEIGHT,
        focal_length_mm=focal_mm,
        horizontal_aperture_mm=HORIZONTAL_APERTURE_MM,
        # Square pixels: the vertical aperture follows the resolution aspect.
        vertical_aperture_mm=HORIZONTAL_APERTURE_MM * HEIGHT / WIDTH,
    )


def outside(camera: CameraSpec, boxes: list[tuple[str, np.ndarray]]) -> list[str]:
    """Which required boxes leave the frame (or the front of the camera)."""
    lo_u, hi_u = MARGIN * camera.width, (1.0 - MARGIN) * camera.width
    lo_v, hi_v = MARGIN * camera.height, (1.0 - MARGIN) * camera.height
    missed = []
    for name, corners in boxes:
        uv, depth = camera.project(corners)
        if (depth <= 0).any() or not np.isfinite(uv).all():
            missed.append(name)
            continue
        if ((uv[:, 0] < lo_u).any() or (uv[:, 0] > hi_u).any()
                or (uv[:, 1] < lo_v).any() or (uv[:, 1] > hi_v).any()):
            missed.append(name)
    return missed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, type=Path,
                        help="common scene bundle to measure the objects from")
    parser.add_argument("--output", required=True, type=Path,
                        help="camera JSON to write")
    parser.add_argument("--reference-camera", type=Path,
                        default=data_path('examples/paper_dollhouse/multisim_workspace_camera.json'),
                        help="camera whose bearing is kept")
    args = parser.parse_args()

    scene = CommonScene.load(args.scene)
    by_id = {o.object_id: o for o in scene.objects()}
    for needed in (*REQUIRED_TASK_OBJECTS, REQUIRED_SUPPORT):
        if needed not in by_id:
            raise SystemExit(f"MULTISIM_SCENE_VERSION_MISMATCH(no object {needed!r} in {args.scene})")

    # Boxes the frame must contain. The desk contributes only its top surface:
    # its legs and underside are behind the tabletop from this bearing and
    # forcing them in would push the camera back for geometry the figure does
    # not need to show.
    boxes: list[tuple[str, np.ndarray]] = []
    task_lo = np.full(3, np.inf)
    task_hi = np.full(3, -np.inf)
    evidence_boxes = {}
    for name in REQUIRED_TASK_OBJECTS:
        obj = by_id[name]
        lo, hi = world_bbox(obj.collision_mesh, obj.matrix_world())
        boxes.append((name, corners_of(lo, hi)))
        evidence_boxes[name] = {"lo_m": lo.tolist(), "hi_m": hi.tolist()}
        task_lo = np.minimum(task_lo, lo)
        task_hi = np.maximum(task_hi, hi)

    desk = by_id[REQUIRED_SUPPORT]
    desk_lo, desk_hi = world_bbox(desk.collision_mesh, desk.matrix_world())
    desk_top = np.array([[x, y, desk_hi[2]] for x in (desk_lo[0], desk_hi[0])
                         for y in (desk_lo[1], desk_hi[1])])
    boxes.append(("desk_top", desk_top))
    evidence_boxes["desk_top"] = {"lo_m": [*desk_lo[:2].tolist(), desk_hi[2]],
                                  "hi_m": [*desk_hi[:2].tolist(), desk_hi[2]]}

    target = (task_lo + task_hi) / 2.0

    reference = json.loads(resolve_input(args.reference_camera).read_text(encoding="utf-8"))
    old_eye = np.asarray(reference["eye"], dtype=np.float64)
    old_target = np.asarray(reference["target"], dtype=np.float64)
    bearing = old_eye - old_target
    old_distance = float(np.linalg.norm(bearing))
    bearing = bearing / old_distance

    # Step 1: back off as far as the room allows. Step 2: take the rest out of
    # the focal length. Doing it the other way round -- widening the lens from
    # where the old camera stood -- would keep the camera 1.34 m from the desk
    # and distort it.
    room = scene.payload["room"]
    distance, limited_by = distance_inside_room(target, bearing, room)
    if not np.isfinite(distance):
        raise SystemExit("MULTISIM_CAMERA_MISMATCH(bearing never leaves the room)")

    chosen_focal = None
    focal = MAX_FOCAL_MM
    while focal >= 3.0:
        if not outside(camera_at(distance, focal, target, bearing), boxes):
            chosen_focal = focal
            break
        focal = round(focal - 0.1, 2)
    if chosen_focal is None:
        raise SystemExit(
            f"MULTISIM_CAMERA_MISMATCH(no focal length <= {MAX_FOCAL_MM} mm fits "
            f"{[n for n, _ in boxes]} from {distance:.2f} m with a {MARGIN:.0%} margin)")

    camera = camera_at(distance, chosen_focal, target, bearing)
    previous = CameraSpec(
        name="previous",
        eye=old_eye,
        target=old_target,
        width=WIDTH,
        height=HEIGHT,
        focal_length_mm=float(reference["focal_length_mm"]),
        horizontal_aperture_mm=float(reference["horizontal_aperture_mm"]),
        vertical_aperture_mm=float(reference["horizontal_aperture_mm"]) * HEIGHT / WIDTH,
    )
    record = {
        "name": camera.name,
        "eye": camera.eye.tolist(),
        "target": camera.target.tolist(),
        "width": WIDTH,
        "height": HEIGHT,
        "focal_length_mm": chosen_focal,
        "horizontal_aperture_mm": HORIZONTAL_APERTURE_MM,
        "comment": (
            "Fitted by python -m rfw_w.scenes.visualization.fit_workspace_camera, not hand-written. Target is the "
            f"centre of the {len(REQUIRED_TASK_OBJECTS)} manipulable task objects' combined box, "
            "not one member of it. The eye is as far back along the previous camera's bearing as "
            f"the room allows ({WALL_CLEARANCE_M} m clearance, stopped by {limited_by}), and the "
            f"focal length is the longest at which cup, notebook, pen holder and the desk top all "
            f"stay inside a {MARGIN:.0%} margin. Square-pixel publication intrinsics."
        ),
        "fit_evidence": {
            "rule": "eye as far back along the reference bearing as the room allows, then the "
                    f"longest focal length at which every required box is inside a {MARGIN:.0%} margin",
            "required_boxes": evidence_boxes,
            "task_cluster_box": {"lo_m": task_lo.tolist(), "hi_m": task_hi.tolist()},
            "target_m": target.tolist(),
            "bearing_unit": bearing.tolist(),
            "distance_m": distance,
            "distance_limited_by": limited_by,
            "wall_clearance_m": WALL_CLEARANCE_M,
            "chosen_focal_length_mm": chosen_focal,
            "focal_ceiling_mm": MAX_FOCAL_MM,
            "margin": MARGIN,
            "previous_camera": {
                "path": str(args.reference_camera),
                "target_m": old_target.tolist(),
                "distance_m": old_distance,
                "focal_length_mm": float(reference["focal_length_mm"]),
                "aimed_at": "the cup's own position, i.e. the edge of the task cluster",
                "boxes_outside_frame": outside(previous, boxes),
            },
            "scene_version": scene.scene_version,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"target     {np.round(target, 4).tolist()}  (was {old_target.tolist()})")
    print(f"distance   {distance:.3f} m  (was {old_distance:.3f} m), stopped by {limited_by}")
    print(f"focal      {chosen_focal} mm  (was {reference['focal_length_mm']} mm)")
    print(f"eye        {np.round(camera.eye, 4).tolist()}")
    print(f"previous camera cropped: {record['fit_evidence']['previous_camera']['boxes_outside_frame']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
