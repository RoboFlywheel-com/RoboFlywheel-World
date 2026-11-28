# SPDX-License-Identifier: Apache-2.0
"""The opening trim: frames, sills and glass, and the fact that they are really authored.

The shell could already cut a hole, and a hole was all it drew. A render of the office then showed a
rectangle of blue where a window should be -- no reveal to say how thick the wall is, no sill, no
glass -- and the visual critic read the room as unfinished, which it was. `RoomMaterials` had
carried `frame` and `glass` (with a `glass_mode` and an opacity) since it was written, and no
geometry had ever been bound to either.

These tests build a real stage with `pxr` and read the prims back, which is the code path the worker
runs under Isaac. The geometry half -- boxes in a wall's own frame -- is checked without `pxr`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from rfw_w.scenes.workers.isaac.geometry import GeometryError, WallOpening, wall_frame

REPO_ROOT = Path(__file__).resolve().parents[3]


class _Room:
    polygon_xy = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
    height_m = 2.8
    floor_z_m = 0.0
    openings: tuple = ()


class TestTheWallFrame:
    """One definition of "along the wall, up it, and through it", shared with `build_wall`."""

    def test_the_inward_normal_points_into_a_counter_clockwise_room(self) -> None:
        """Wall 0 of the outline above runs along +x, so inward is +y."""
        frame = wall_frame((0.0, 0.0), (4.0, 0.0), thickness_m=0.1)
        assert frame.along_x == pytest.approx(1.0) and frame.along_y == pytest.approx(0.0)
        assert frame.across_x == pytest.approx(0.0) and frame.across_y == pytest.approx(1.0)
        assert frame.length_m == pytest.approx(4.0)

    def test_a_point_is_measured_along_then_across(self) -> None:
        frame = wall_frame((0.0, 0.0), (4.0, 0.0), thickness_m=0.1)
        assert frame.point(2.0, 0.0, 1.5) == pytest.approx((2.0, 0.0, 1.5))
        # Across runs from the room-side face towards the far side, so the wall's other face is at
        # the thickness and the box never straddles it by accident.
        assert frame.point(2.0, 0.1, 1.5) == pytest.approx((2.0, 0.1, 1.5))

    def test_a_zero_length_wall_is_refused(self) -> None:
        with pytest.raises(GeometryError, match="zero length"):
            wall_frame((1.0, 1.0), (1.0, 1.0), thickness_m=0.1)

    def test_a_box_needs_volume_in_all_three_axes(self) -> None:
        frame = wall_frame((0.0, 0.0), (4.0, 0.0), thickness_m=0.1)
        with pytest.raises(GeometryError, match="positive extent"):
            frame.box(along_m=(1.0, 1.0), across_m=(0.0, 0.1), z_m=(0.0, 1.0))

    def test_a_box_has_six_faces_and_eight_points(self) -> None:
        frame = wall_frame((0.0, 0.0), (4.0, 0.0), thickness_m=0.1)
        points, faces = frame.box(along_m=(1.0, 1.5), across_m=(0.0, 0.1), z_m=(0.0, 2.0))
        assert len(points) == 8 and len(faces) == 6
        assert all(len(face) == 4 for face in faces)


@pytest.fixture(scope="module")
def worker():
    return importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")


@pytest.fixture(scope="module")
def usd():
    pytest.importorskip("pxr")
    import types

    from pxr import Usd, UsdGeom, UsdShade

    return types.SimpleNamespace(Usd=Usd, UsdGeom=UsdGeom, UsdShade=UsdShade)


def room_request(usd, *, glass_mode: str = "semi_transparent", glass_opacity: float = 0.45):
    from rfw_w.scenes.workers.isaac.protocol import (
        RoomGeometry,
        RoomMaterialsRequest,
        SurfaceRequest,
        WorkerRequest,
    )

    room = RoomGeometry(
        polygon_xy=_Room.polygon_xy, height_m=_Room.height_m, floor_z_m=_Room.floor_z_m,
        openings=(
            # Wall 0 runs along +x from (0,0), so the door sits at x = 1.5 and the window at x = 2.8.
            # A door: sill 0. A window: sill above the floor, which is what leaves wall under it.
            _opening("door_main", (1.5, 0.0), width_m=0.9, height_m=2.05, sill_m=0.0, kind="door"),
            _opening("window_01", (2.8, 0.0), width_m=1.2, height_m=1.2, sill_m=1.0, kind="window"),
        ),
    )
    return WorkerRequest(
        scene_id="s", scene_version="a" * 64, room=room, instances=(_instance(),), cameras=(_camera(),),
        room_materials=RoomMaterialsRequest(
            frame=SurfaceRequest(colour=(0.05, 0.06, 0.07), roughness=0.28, metallic=0.72),
            glass=SurfaceRequest(colour=(0.78, 0.84, 0.88), roughness=0.12),
            glass_mode=glass_mode, glass_opacity=glass_opacity,
        ),
    )


def _camera():
    from rfw_w.scenes.workers.isaac.protocol import CameraRequest

    return CameraRequest(
        camera_id="view", purpose="export", eye=(2.0, 1.5, 1.6), target=(2.0, 1.5, 0.8),
    )


def _instance():
    from rfw_w.scenes.workers.isaac.protocol import MeshInstance

    return MeshInstance(
        instance_id="table", entity_id="table", visual_uri="/nonexistent.obj",
        collision_uri=None, translation=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def _opening(opening_id, center_xy, *, width_m, height_m, sill_m, kind):
    from rfw_w.scenes.workers.isaac.protocol import RoomOpening

    return RoomOpening(
        opening_id=opening_id, center_xy=center_xy, width_m=width_m, height_m=height_m,
        sill_m=sill_m, kind=kind,
    )


def trim_prims(stage) -> list[str]:
    return sorted(
        prim.GetPath().pathString for prim in stage.Traverse()
        if "/World/Room/opening_" in prim.GetPath().pathString
    )


class TestTheOpeningsGetTrim:
    def test_a_door_gets_a_frame_and_no_pane(self, worker, usd) -> None:
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, room_request(usd))
        paths = trim_prims(stage)
        assert [p.rsplit("/", 1)[-1] for p in paths if "door_main" in p] == [
            "opening_door_main_head", "opening_door_main_jamb_far",
            "opening_door_main_jamb_near",
        ]
        assert not [p for p in paths if "door_main" in p and "glass" in p]

    def test_a_window_gets_a_frame_a_sill_and_a_pane(self, worker, usd) -> None:
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, room_request(usd))
        names = [p.rsplit("/", 1)[-1] for p in trim_prims(stage) if "window_01" in p]
        assert "opening_window_01_glass" in names
        assert "opening_window_01_sill" in names
        assert {"opening_window_01_jamb_near", "opening_window_01_jamb_far",
                "opening_window_01_head"} <= set(names)

    def test_the_frame_wears_the_plans_frame_surface(self, worker, usd) -> None:
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, room_request(usd))
        material = usd.UsdShade.Material(stage.GetPrimAtPath("/World/Materials/opening_frame"))
        shader = next(
            usd.UsdShade.Shader(prim) for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith("/World/Materials/opening_frame/")
            and usd.UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface"
        )
        assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.05, 0.06, 0.07))
        assert shader.GetInput("metallic").Get() == pytest.approx(0.72)
        assert material

    def test_the_pane_is_semi_transparent_at_the_requested_opacity(self, worker, usd) -> None:
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, room_request(usd, glass_opacity=0.3))
        shader = next(
            usd.UsdShade.Shader(prim) for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith("/World/Materials/opening_glass/")
            and usd.UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface"
        )
        assert shader.GetInput("opacity").Get() == pytest.approx(0.3)

    def test_the_glass_mode_reaches_the_material(self, worker, usd) -> None:
        """Three modes that were indistinguishable because none of them was ever bound."""
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, room_request(usd, glass_mode="opaque_tint"))
        shader = next(
            usd.UsdShade.Shader(prim) for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith("/World/Materials/opening_glass/")
            and usd.UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface"
        )
        # Opaque: no opacity input at all, rather than one set to 1.0 and left to a renderer's
        # default blending.
        assert not shader.GetInput("opacity") or shader.GetInput("opacity").Get() >= 1.0

    def test_an_emissive_sky_pane_lights_the_room(self, worker, usd) -> None:
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, room_request(usd, glass_mode="emissive_sky"))
        shader = next(
            usd.UsdShade.Shader(prim) for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith("/World/Materials/opening_glass/")
            and usd.UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface"
        )
        assert shader.GetInput("emissiveColor").Get() is not None

    def test_the_pane_is_not_a_collider(self, worker, usd) -> None:
        """A pane that stops the robot is a wall, and the shell's openings are what navigation
        is planned against -- a collider here would block an opening the shell says is open."""
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, room_request(usd))
        pane = stage.GetPrimAtPath("/World/Room/opening_window_01_glass")
        assert pane
        assert not usd.UsdGeom.Imageable(pane).GetPrim().HasAPI(
            __import__("pxr").UsdPhysics.CollisionAPI
        )

    def test_an_opening_too_small_to_case_is_left_as_a_hole(self, worker, usd) -> None:
        """Casing a slot narrower than two frame faces would fill it with frame and no hole.

        The shell refuses to plan one of these; a hand-written request can still state it, and the
        trim has to give way rather than the opening.
        """
        import dataclasses

        stage = usd.Usd.Stage.CreateInMemory()
        request = room_request(usd)
        room = type(request.room)(
            polygon_xy=_Room.polygon_xy, height_m=_Room.height_m, floor_z_m=_Room.floor_z_m,
            openings=(_opening("slot_01", (3.2, 0.0), width_m=0.08, height_m=0.4, sill_m=1.0,
                               kind="window"),),
        )
        worker._author_room(stage, dataclasses.replace(request, room=room))
        assert not [p for p in trim_prims(stage) if "slot_01" in p]
        # And the wall is still cut: dropping the trim must not seal the opening.
        assert stage.GetPrimAtPath("/World/Room/wall_00")
