# SPDX-License-Identifier: Apache-2.0
"""The plan's room surfaces must reach the renderer.

The same regression `test_worker_lighting.py` covers for lighting, one field over: the shell's
materials were module constants in `render_worker`, so `StylePlan.palette` was parsed, validated,
hashed, and read by nothing. `repair.appearance` could change the plan, change the scene version,
invalidate the render and the critique, force both to re-run -- and produce an identical image.

The stage tests build a real USD stage and read the bound `UsdPreviewSurface` inputs back, which is
the code path the worker runs under Isaac. `pxr` arrives with the orchestrator's environment rather
than being a declared dependency, so those tests skip where it is absent.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

from rfw_w.scenes.solvers.materials import materials_for
from rfw_w.scenes.workers.isaac.protocol import (
    ProtocolError,
    RoomMaterialsRequest,
    SurfaceRequest,
    WorkerRequest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def worker():
    return importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")


@pytest.fixture(scope="module")
def usd():
    pytest.importorskip("pxr")
    import types

    from pxr import Gf, Usd, UsdGeom, UsdShade

    return types.SimpleNamespace(Gf=Gf, Usd=Usd, UsdGeom=UsdGeom, UsdShade=UsdShade)


class _Room:
    polygon_xy = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
    height_m = 2.8
    floor_z_m = 0.0
    openings: tuple = ()


class TestTheRequestCarriesThem:
    def test_the_block_round_trips(self) -> None:
        original = RoomMaterialsRequest(
            wall=SurfaceRequest(colour=(0.1, 0.2, 0.3), roughness=0.4),
            floor=SurfaceRequest(colour=(0.2, 0.3, 0.4), roughness=0.5),
            glass_opacity=0.7, glass_mode="opaque_tint",
        )
        assert RoomMaterialsRequest.from_dict(original.to_dict()) == original

    def test_it_is_built_from_the_plans_room_materials(self) -> None:
        from rfw_w.scenes.contracts.scene_spec import StylePlan

        materials = materials_for(
            StylePlan(theme="x", palette=("oak", "off_white", "brass")), seed=3
        )
        request = RoomMaterialsRequest.from_materials(materials)
        assert request.wall.colour == materials.wall.colour
        assert request.floor.colour == materials.floor.colour
        assert request.frame.colour == materials.frame.colour
        assert request.frame.metallic == materials.frame.metallic

    def test_an_omitted_block_defaults_to_what_was_hardcoded(self) -> None:
        """A request written before this field existed must render exactly as it did."""
        restored = RoomMaterialsRequest.from_dict({})
        assert restored.wall.colour == (0.82, 0.79, 0.74)
        assert restored.floor.colour == (0.42, 0.30, 0.21)
        assert restored.ceiling.colour == (0.90, 0.89, 0.87)
        assert restored.facade.colour == (0.55, 0.68, 0.85)

    def test_the_worker_request_defaults_to_the_same_materials(self) -> None:
        import dataclasses

        field = next(f for f in dataclasses.fields(WorkerRequest) if f.name == "room_materials")
        assert field.default_factory is RoomMaterialsRequest

    def test_an_opacity_outside_the_range_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="glass opacity"):
            RoomMaterialsRequest(glass_opacity=1.4)

    def test_an_unknown_glass_mode_is_refused_by_name(self) -> None:
        with pytest.raises(ProtocolError, match="glass mode"):
            RoomMaterialsRequest(glass_mode="transparent-ish")

    def test_a_colour_outside_the_unit_range_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="within"):
            SurfaceRequest(colour=(1.4, 0.5, 0.5))


class TestTheRendererUsesThem:
    """A real USD stage, the real authoring function, no host and no stub."""

    def _materials(self) -> RoomMaterialsRequest:
        return RoomMaterialsRequest(
            wall=SurfaceRequest(colour=(0.11, 0.12, 0.13), roughness=0.21),
            floor=SurfaceRequest(colour=(0.24, 0.25, 0.26), roughness=0.34),
            ceiling=SurfaceRequest(colour=(0.37, 0.38, 0.39), roughness=0.45),
        )

    def _read_surface(self, usd, stage, path: str):
        material = usd.UsdShade.Material(stage.GetPrimAtPath(path))
        shaders = [
            usd.UsdShade.Shader(prim)
            for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith(f"{path}/")
        ]
        shader = next(s for s in shaders if s.GetIdAttr().Get() == "UsdPreviewSurface")
        return {
            "diffuseColor": tuple(shader.GetInput("diffuseColor").Get()),
            "roughness": shader.GetInput("roughness").Get(),
            "material": material,
        }

    def test_the_colours_it_binds_are_the_ones_it_was_given(self, worker, usd, tmp_path) -> None:
        """The regression: this is what a hardcoded renderer cannot pass."""
        request = WorkerRequest(
            scene_id="s", scene_version="a" * 64, room=_Room(),
            instances=(_instance(),), cameras=(_camera(),),
            room_materials=self._materials(),
        )
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, request)

        wall = self._read_surface(usd, stage, "/World/Materials/wall")
        floor = self._read_surface(usd, stage, "/World/Materials/floor")
        ceiling = self._read_surface(usd, stage, "/World/Materials/ceiling")
        assert wall["diffuseColor"] == pytest.approx((0.11, 0.12, 0.13))
        assert wall["roughness"] == pytest.approx(0.21)
        assert floor["diffuseColor"] == pytest.approx((0.24, 0.25, 0.26))
        assert ceiling["diffuseColor"] == pytest.approx((0.37, 0.38, 0.39))

    def test_the_facade_takes_the_surface_it_is_given(self, worker, usd) -> None:
        stage = usd.Usd.Stage.CreateInMemory()
        request = WorkerRequest(
            scene_id="s", scene_version="a" * 64, room=_Room(),
            instances=(_instance(),), cameras=(_camera(),),
            room_materials=RoomMaterialsRequest(
                facade=SurfaceRequest(colour=(0.91, 0.42, 0.07), roughness=0.9)
            ),
        )
        windows = [type("_W", (), {"center_xy": (2.0, 0.0)})(),
                   type("_W", (), {"center_xy": (0.0, 1.5)})()]
        authored = worker._author_exterior(stage, request.room, windows=windows,
                                           surface=request.room_materials.facade)
        sky = self._read_surface(usd, stage, "/World/Materials/exterior_sky")
        assert sky["diffuseColor"] == pytest.approx((0.91, 0.42, 0.07))
        # One backdrop per window: a room with three windows and one backdrop has two of them
        # looking onto nothing.
        assert authored == 2
        for index in range(2):
            assert stage.GetPrimAtPath(f"/World/Exterior/sky_{index:02d}")


class TestTheMaterialHelper:
    """The helper's own contract, because one call site silently stopped honouring it.

    The signature grew a `surface` parameter in front of `colour`, and the asset path went on
    passing its colour positionally. Every test in the suite passed; the failure was
    `'tuple' object has no attribute 'colour'` raised inside Isaac, after a render had been spent
    and with no capture written at all.
    """

    def test_a_positional_colour_is_refused(self, worker) -> None:
        with pytest.raises(TypeError):
            worker._make_material(None, "x", (0.1, 0.2, 0.3))

    def test_neither_source_is_refused(self, worker) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            worker._make_material(None, "x")

    def test_both_sources_are_refused(self, worker) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            worker._make_material(None, "x", surface=SurfaceRequest(), colour=(0.1, 0.2, 0.3))

    def test_every_call_site_names_what_it_passes(self) -> None:
        """Source-level, because the bug was a call site and the helper cannot see its callers.

        Whole call expressions rather than lines: the call that broke spans two, with `colour=` on
        the continuation -- which is also why a line-based check would have missed it.
        """
        import re

        flattened = " ".join(inspect.getsource(importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")).split())
        calls = [
            call for call in re.findall(r"_make_material\((?:[^()]|\([^()]*\))*\)", flattened)
            if not call.startswith("_make_material(stage, name")
        ]
        assert calls, "the scan found no call sites, so it is checking nothing"
        for call in calls:
            assert "surface=" in call or "colour=" in call, call


class TestNoLiteralsLeft:
    def test_the_room_author_does_not_decide_its_own_colours(self) -> None:
        """A literal in `_author_room` is a colour the plan cannot change.

        Source-level rather than stage-level because the failure is a value that never got read,
        not one that came out wrong.
        """
        source = inspect.getsource(importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")._author_room)
        for literal in ("0.82, 0.79", "0.42, 0.30", "0.90, 0.89"):
            assert literal not in source, f"_author_room still carries the colour {literal}"
        assert "room_materials" in source or "surfaces" in source


def _camera():
    from rfw_w.scenes.workers.isaac.protocol import CameraRequest

    return CameraRequest(
        camera_id="view", purpose="export", eye=(2.0, 1.5, 1.6), target=(2.0, 1.5, 0.8),
    )


def _instance():
    from rfw_w.scenes.workers.isaac.protocol import MeshInstance

    return MeshInstance(
        instance_id="cup", entity_id="cup", visual_uri="/tmp/cup.obj",
        collision_uri=None, translation=(0.0, 0.0, 0.0),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

class _ShellRoom(_Room):
    """A room whose surfaces come from a provider's USD rather than from the polygon."""

    shell_uri = "/nonexistent/shell.usdc"


class TestTheShellMaterialOverride:
    """`room_materials` reaches the procedural fallback only, and that was invisible.

    When the room carries a `shell_uri`, `_author_room` hands off to `author_shell` and returns
    before it builds a single material, so the referenced USD's own bindings are what renders.
    Measured 2026-09-25 on the SPEC-014 five-renderer figure: Isaac showed a near-black wall where
    the other four -- which read the bundle's MTL-less OBJ shell -- showed light grey, and no
    lighting change moved it. `shell_material_override` is the opt-in that rebinds the referenced
    shell, and it is off for every delivery.

    `author_shell` is replaced here rather than driven: it resolves, inspects and hashes a real
    provider USD, and none of that is what these tests are about.
    """

    def _request(self, *, override: bool) -> WorkerRequest:
        return WorkerRequest(
            scene_id="s", scene_version="a" * 64, room=_ShellRoom(),
            instances=(_instance(),), cameras=(_camera(),),
            room_materials=RoomMaterialsRequest(
                wall=SurfaceRequest(colour=(0.61, 0.62, 0.63), roughness=0.71)),
            shell_material_override=override,
        )

    def _stub_shell(self, worker, monkeypatch, usd, *, meshes: tuple[str, ...]):
        """Stand in for the provider's USD: `meshes` are the prims it brings under /World/Room."""
        import rfw_w.scenes.workers.isaac.shell_reference as shell_reference

        def author_shell(stage, uri):
            usd.UsdGeom.Xform.Define(stage, "/World/Room")
            for path in meshes:
                usd.UsdGeom.Mesh.Define(stage, path)
                # Infinigen's shells bind per face set, not per mesh: this office puts all four
                # walls in one mesh with a subset each. A subset's binding beats its mesh's, so a
                # rebind that skipped these looked like it worked and changed nothing on screen.
                usd.UsdGeom.Subset.Define(stage, f"{path}/shader_plaster_001")
            return 2

        monkeypatch.setattr(shell_reference, "author_shell", author_shell)

    def _serialisable(self, *, override: bool) -> WorkerRequest:
        """`_ShellRoom` is a stub for the stage tests; the wire needs the real geometry."""
        from rfw_w.scenes.workers.isaac.protocol import RoomGeometry

        return WorkerRequest(
            scene_id="s", scene_version="a" * 64,
            room=RoomGeometry(polygon_xy=_ShellRoom.polygon_xy, height_m=_ShellRoom.height_m,
                              shell_uri=_ShellRoom.shell_uri),
            instances=(_instance(),), cameras=(_camera(),),
            shell_material_override=override,
        )

    def test_it_is_off_by_default(self) -> None:
        """No delivery's appearance moves because this field exists."""
        request = self._serialisable(override=False)
        assert request.shell_material_override is False
        assert request.to_dict()["shell_material_override"] is False
        assert WorkerRequest.from_dict(request.to_dict()).shell_material_override is False

    def test_an_older_request_without_the_field_reads_as_off(self) -> None:
        payload = self._serialisable(override=True).to_dict()
        del payload["shell_material_override"]
        assert WorkerRequest.from_dict(payload).shell_material_override is False

    def test_the_flag_round_trips(self) -> None:
        request = self._serialisable(override=True)
        assert WorkerRequest.from_dict(request.to_dict()).shell_material_override is True

    def test_off_leaves_the_referenced_shell_alone(self, worker, usd, monkeypatch) -> None:
        """The negative that proves the override is doing the work: no binding, no material."""
        self._stub_shell(worker, monkeypatch, usd, meshes=("/World/Room/wall_00",))
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, self._request(override=False))

        assert not stage.GetPrimAtPath("/World/Materials/shell_override").IsValid()
        bound = usd.UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath("/World/Room/wall_00")).GetDirectBinding()
        assert not bound.GetMaterial()

    def test_on_rebinds_every_shell_mesh_to_the_wall_colour(self, worker, usd, monkeypatch) -> None:
        paths = ("/World/Room/wall_00", "/World/Room/floor", "/World/Room/trim/sill")
        self._stub_shell(worker, monkeypatch, usd, meshes=paths)
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, self._request(override=True))

        shader = next(
            usd.UsdShade.Shader(prim) for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith("/World/Materials/shell_override/"))
        assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.61, 0.62, 0.63))
        for path in paths:
            for target in (path, f"{path}/shader_plaster_001"):
                bound = usd.UsdShade.MaterialBindingAPI(
                    stage.GetPrimAtPath(target)).GetDirectBinding().GetMaterial()
                assert bound, f"{target} kept the provider's own material"
                assert bound.GetPath().pathString == "/World/Materials/shell_override"

    def test_it_does_not_reach_outside_the_room(self, worker, usd, monkeypatch) -> None:
        """An instance is not a wall. The override is scoped to the shell, not to the stage."""
        self._stub_shell(worker, monkeypatch, usd, meshes=("/World/Room/wall_00",))
        stage = usd.Usd.Stage.CreateInMemory()
        usd.UsdGeom.Mesh.Define(stage, "/World/Instances/cup")
        worker._author_room(stage, self._request(override=True))

        bound = usd.UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath("/World/Instances/cup")).GetDirectBinding()
        assert not bound.GetMaterial()

    def test_a_shell_it_cannot_bind_fails_by_name(self, worker, usd, monkeypatch) -> None:
        """A gate that cannot fail is the defect: silently binding nothing would look like success.

        That is the exact shape of the bug this field exists to answer -- an override that reached
        no prim and reported a clean render anyway.
        """
        self._stub_shell(worker, monkeypatch, usd, meshes=())
        stage = usd.Usd.Stage.CreateInMemory()
        with pytest.raises(RuntimeError, match="SHELL_MATERIAL_OVERRIDE_BOUND_NOTHING"):
            worker._author_room(stage, self._request(override=True))

    # -- SPEC-016: the same opt-in, one role at a time -----------------------------------

    def _per_role_request(self, roles: tuple[str, ...]) -> WorkerRequest:
        return WorkerRequest(
            scene_id="s", scene_version="a" * 64, room=_ShellRoom(),
            instances=(_instance(),), cameras=(_camera(),),
            room_materials=RoomMaterialsRequest(
                wall=SurfaceRequest(colour=(0.86, 0.80, 0.70), roughness=0.88),
                floor=SurfaceRequest(colour=(0.40, 0.27, 0.17), roughness=0.52)),
            shell_material_override=roles,
        )

    def test_a_role_list_round_trips_as_a_list_not_as_true(self) -> None:
        """`bool(["wall"])` is True, so the obvious reading turns a per-role override into the
        one-albedo-everywhere one -- and the picture changes without the request saying so."""
        request = WorkerRequest(
            scene_id="s", scene_version="a" * 64,
            room=__import__("rfw_w.scenes.workers.isaac.protocol", fromlist=["RoomGeometry"])
            .RoomGeometry(polygon_xy=_ShellRoom.polygon_xy, height_m=_ShellRoom.height_m,
                          shell_uri=_ShellRoom.shell_uri),
            instances=(_instance(),), cameras=(_camera(),),
            shell_material_override=("wall", "floor"))
        assert request.to_dict()["shell_material_override"] == ["wall", "floor"]
        assert WorkerRequest.from_dict(request.to_dict()).shell_material_override == (
            "wall", "floor")

    def test_a_role_with_no_surface_to_bind_from_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="no surface to bind from"):
            self._per_role_request(("skirtingboard",))

    def test_an_empty_role_list_is_refused(self) -> None:
        """Say False to leave the shell alone; an empty declaration is an override that
        binds nothing while provenance claims one."""
        with pytest.raises(ProtocolError, match="empty role list"):
            self._per_role_request(())

    def test_each_role_is_bound_from_its_own_surface(self, worker, usd, monkeypatch) -> None:
        paths = ("/World/Room/wall_00", "/World/Room/floor_01", "/World/Room/ceiling_01")
        self._stub_shell(worker, monkeypatch, usd, meshes=paths)
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, self._per_role_request(("wall", "floor")))

        def colour_of(path: str):
            material = usd.UsdShade.MaterialBindingAPI(
                stage.GetPrimAtPath(path)).GetDirectBinding().GetMaterial()
            assert material, f"{path} kept the provider's own material"
            shader = next(usd.UsdShade.Shader(prim) for prim in stage.Traverse()
                          if prim.GetPath().pathString.startswith(
                              material.GetPath().pathString + "/"))
            return tuple(shader.GetInput("diffuseColor").Get())

        assert colour_of("/World/Room/wall_00") == pytest.approx((0.86, 0.80, 0.70))
        assert colour_of("/World/Room/floor_01") == pytest.approx((0.40, 0.27, 0.17))
        # A role the request did not name keeps whatever the provider authored.
        ceiling = usd.UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath("/World/Room/ceiling_01")).GetDirectBinding()
        assert not ceiling.GetMaterial()

    def test_a_subset_follows_its_mesh_role(self, worker, usd, monkeypatch) -> None:
        """A GeomSubset's own path says nothing about what it is; its mesh's does, and a
        subset's binding beats its mesh's."""
        self._stub_shell(worker, monkeypatch, usd, meshes=("/World/Room/wall_00",))
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_room(stage, self._per_role_request(("wall",)))
        bound = usd.UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath("/World/Room/wall_00/shader_plaster_001")
        ).GetDirectBinding().GetMaterial()
        assert bound and bound.GetPath().pathString.endswith("shell_override_wall")

    def test_a_declared_role_that_binds_nothing_fails_by_name(self, worker, usd, monkeypatch) -> None:
        """Measured 2026-09-26: the dollhouse shell has its ceiling removed, so declaring
        `ceiling` there must fail rather than quietly bind nothing."""
        self._stub_shell(worker, monkeypatch, usd, meshes=("/World/Room/wall_00",))
        stage = usd.Usd.Stage.CreateInMemory()
        with pytest.raises(RuntimeError, match="SHELL_MATERIAL_OVERRIDE_BOUND_NOTHING"):
            worker._author_room(stage, self._per_role_request(("wall", "floor")))
