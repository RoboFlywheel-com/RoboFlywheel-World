# SPDX-License-Identifier: Apache-2.0
"""The plan's lighting must reach the renderer.

This is the test the appearance track was missing. `repair.appearance` moved `AppearanceSettings` in
the plan, which changed the scene version and so invalidated the render and the critique and forced
both to re-run -- while `render_worker._author_lighting` **hardcoded** its three intensities and two
colours. They happened to equal the defaults, so nothing looked wrong. The consequence was exact: the
repair re-rendered an identical image, the critic correctly reported no improvement, and the rung was
recorded as no improvement and rolled back.

Every other test of that track stubs the host, so none of them could see it. These do not: the second
test builds a real USD stage with `pxr` and reads the light prims back, which is the same code path
the worker runs under Isaac. The module imports no `isaacsim` at module level, which is what makes
that possible.

`pxr` is not a declared dependency of the core -- it arrives with the orchestrator's environment --
so the stage test skips where it is absent rather than failing.
"""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path
import sys
from pathlib import Path

import pytest

from rfw_w.scenes.workers.isaac.protocol import LightingRequest, WorkerRequest

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestTheRequestCarriesTheLighting:
    def test_a_lighting_request_round_trips(self) -> None:
        original = LightingRequest(
            dome_intensity=1234.5, sun_intensity=4321.0, ceiling_fill_intensity=111.0,
            dome_color=(0.1, 0.2, 0.3), sun_color=(0.4, 0.5, 0.6),
        )
        restored = LightingRequest.from_dict(original.to_dict())
        assert restored == original

    def test_it_is_built_from_the_plans_appearance_settings(self) -> None:
        from rfw_w.scenes.solvers.furnish import AppearanceSettings

        appearance = AppearanceSettings(
            dome_intensity=1500.0, sun_intensity=3000.0, ceiling_fill_intensity=250.0,
            dome_color=(0.5, 0.5, 0.9), sun_color=(1.0, 0.8, 0.6),
        )
        lighting = LightingRequest.from_appearance(appearance)
        assert lighting.dome_intensity == 1500.0
        assert lighting.sun_intensity == 3000.0
        assert lighting.ceiling_fill_intensity == 250.0
        assert lighting.dome_color == (0.5, 0.5, 0.9)
        assert lighting.sun_color == (1.0, 0.8, 0.6)

    def test_an_omitted_lighting_block_defaults_to_what_was_hardcoded(self) -> None:
        """A request written before this field existed must render exactly as it did."""
        restored = LightingRequest.from_dict({})
        assert (restored.dome_intensity, restored.sun_intensity,
                restored.ceiling_fill_intensity) == (900.0, 2600.0, 700.0)

    def test_a_negative_intensity_is_refused(self) -> None:
        from rfw_w.scenes.workers.isaac.protocol import ProtocolError

        with pytest.raises(ProtocolError, match="non-negative finite intensity"):
            LightingRequest(dome_intensity=-1.0)

    def test_a_colour_needs_three_channels(self) -> None:
        from rfw_w.scenes.workers.isaac.protocol import ProtocolError

        with pytest.raises(ProtocolError, match="three non-negative finite channels"):
            LightingRequest(dome_color=(0.5, 0.5))

    def test_the_worker_request_defaults_to_the_same_lighting(self) -> None:
        """`WorkerRequest.lighting` has a default, so every existing construction still builds."""
        import dataclasses

        assert dataclasses.fields(WorkerRequest)
        field = next(f for f in dataclasses.fields(WorkerRequest) if f.name == "lighting")
        assert field.default_factory is LightingRequest


@pytest.fixture(scope="module")
def worker():
    return importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")


@pytest.fixture(scope="module")
def usd():
    """`pxr` is a namespace package; the submodules have to be imported by name."""
    pytest.importorskip("pxr")
    import types

    from pxr import Gf, Usd, UsdGeom, UsdLux

    return types.SimpleNamespace(Gf=Gf, Usd=Usd, UsdGeom=UsdGeom, UsdLux=UsdLux)


class _Room:
    polygon_xy = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
    height_m = 2.8
    floor_z_m = 0.0


class TestTheRendererUsesIt:
    """A real USD stage, the real authoring function, no host and no stub."""

    def test_the_values_it_writes_are_the_ones_it_was_given(self, worker, usd) -> None:
        """The regression: this is what a hardcoded renderer cannot pass."""
        stage = usd.Usd.Stage.CreateInMemory()
        lighting = LightingRequest(
            dome_intensity=1777.0, sun_intensity=2333.0, ceiling_fill_intensity=311.0,
            dome_color=(0.11, 0.22, 0.33), sun_color=(0.44, 0.55, 0.66),
        )
        applied = worker._author_lighting(
            stage, _Room(), window_center=(2.0, 0.0), lighting=lighting
        )

        dome = usd.UsdLux.DomeLight(stage.GetPrimAtPath("/World/Lights/dome"))
        sun = usd.UsdLux.DistantLight(stage.GetPrimAtPath("/World/Lights/sun"))
        ceiling = usd.UsdLux.RectLight(stage.GetPrimAtPath("/World/Lights/ceiling_fill"))

        assert dome.GetIntensityAttr().Get() == 1777.0
        assert sun.GetIntensityAttr().Get() == 2333.0
        assert ceiling.GetIntensityAttr().Get() == 311.0
        assert tuple(dome.GetColorAttr().Get()) == pytest.approx((0.11, 0.22, 0.33))
        assert tuple(sun.GetColorAttr().Get()) == pytest.approx((0.44, 0.55, 0.66))

        # And what it reports having applied is the same, so a report cannot claim one thing while
        # the stage holds another.
        assert applied["dome_intensity"] == 1777.0
        assert applied["sun_intensity"] == 2333.0
        assert applied["ceiling_fill_intensity"] == 311.0

    def test_two_different_requests_produce_two_different_stages(self, worker, usd) -> None:
        """The property the appearance track needs: a changed plan changes the picture."""
        readings = []
        for intensity in (900.0, 1800.0):
            stage = usd.Usd.Stage.CreateInMemory()
            worker._author_lighting(
                stage, _Room(), window_center=(2.0, 0.0),
                lighting=LightingRequest(dome_intensity=intensity),
            )
            dome = usd.UsdLux.DomeLight(stage.GetPrimAtPath("/World/Lights/dome"))
            readings.append(dome.GetIntensityAttr().Get())
        assert readings == [900.0, 1800.0]

    def test_omitting_the_lighting_still_renders(self, worker, usd) -> None:
        """The default path, for a caller that has no plan -- the smoke fixtures, for instance."""
        stage = usd.Usd.Stage.CreateInMemory()
        applied = worker._author_lighting(stage, _Room(), window_center=(2.0, 0.0))
        assert applied["dome_intensity"] == 900.0
        assert applied["ceiling_fill_intensity"] == 700.0


class TestTheRendererHasNoLiteralsLeft:
    def test_no_intensity_or_colour_is_hardcoded_in_the_lighting_function(self) -> None:
        """A source check, because the failure mode is a literal quietly returning.

        The three values that were hardcoded are exactly the `LightingRequest` defaults, so a test
        that only compares numbers against the defaults cannot tell the two apart. What can tell
        them apart is the literal being absent.
        """
        source = module_path("rfw_w.scenes.workers.isaac.render_worker").read_text()
        start = source.index("def _author_lighting")
        end = source.index("def _author_exterior")
        body = source[start:end]
        for literal in ("CreateIntensityAttr(900)", "CreateIntensityAttr(2600)",
                        "CreateIntensityAttr(700)", "Gf.Vec3f(0.92, 0.94, 1.0)",
                        "Gf.Vec3f(1.0, 0.96, 0.88)"):
            assert literal not in body, (
                f"{literal!r} is back in _author_lighting; the values must come from the request, "
                f"or the appearance track re-renders an identical image"
            )
        assert "lighting.dome_intensity" in body
        assert "lighting.ceiling_fill_intensity" in body

def test_referenced_shell_fill_is_below_ceiling_and_points_down():
    pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom
    from rfw_w.scenes.workers.isaac.protocol import RoomGeometry
    room = RoomGeometry(((0, 0), (4, 0), (4, 3), (0, 3)), 3, shell_uri='shell.usdc')
    stage = Usd.Stage.CreateInMemory()
    importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")._author_lighting(stage, room)
    transform = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Lights/ceiling_fill')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert transform.ExtractTranslation()[2] < 2.95
    assert transform.TransformDir(Gf.Vec3d(0, 0, -1))[2] < 0
