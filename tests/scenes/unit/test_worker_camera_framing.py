# SPDX-License-Identifier: Apache-2.0
"""The renderer's framing must be declared, and must be the framing the probe measures against.

Two halves meet here. `render_worker._author_camera` authors the USD camera; `probe_camera_fov`
recovers the framing from a depth map and assumes a particular camera basis to do it. If those two
disagreed, the probe would report a confident number about a camera nobody rendered with -- so the
basis is checked against a real stage rather than against itself.

The gap this closes: the worker authored only `focalLength`, so its horizontal field of view came
from USD's fallback (117.334 degrees at the recorded focal length) while the Blender side of the
same side-by-side comparison hand-coded 22.044444 mm (119.883 degrees). Nobody had measured either.

`pxr` arrives with the orchestrator's environment rather than the core's, so the stage tests skip
where it is absent.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

from rfw_w.scenes.workers.isaac.protocol import CameraRequest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def worker():
    return importlib.import_module('rfw_w.scenes.workers.isaac.render_worker')


@pytest.fixture(scope="module")
def probe():
    return importlib.import_module('rfw_w.scenes.visualization.probe_camera_fov')


@pytest.fixture(scope="module")
def fixture_module():
    return importlib.import_module('rfw_w.scenes.visualization.appearance_fixture')


@pytest.fixture(scope="module")
def usd():
    pytest.importorskip("pxr")
    import types

    from pxr import Gf, Usd, UsdGeom

    return types.SimpleNamespace(Gf=Gf, Usd=Usd, UsdGeom=UsdGeom)


EYE = (2.3728, 0.7929918609719235, 2.6853)
TARGET = (2.3728, 1.83665, 0.0)
FOCAL_MM = 6.378678048182431


def camera(**overrides) -> CameraRequest:
    payload = {"camera_id": "overview_top", "purpose": "export", "eye": EYE, "target": TARGET,
               "focal_length_mm": FOCAL_MM}
    payload.update(overrides)
    return CameraRequest(**payload)


def _authored(usd, worker, request):
    stage = usd.Usd.Stage.CreateInMemory()
    path = worker._author_camera(stage, request)
    prim = stage.GetPrimAtPath(path)
    assert prim.IsValid(), path
    return stage, prim


class TestThePrimCarriesTheFraming:
    def test_the_apertures_written_are_the_ones_given(self, usd, worker) -> None:
        _, prim = _authored(usd, worker, camera(horizontal_aperture_mm=22.0444444444,
                                                vertical_aperture_mm=12.4))
        camera_schema = usd.UsdGeom.Camera(prim)
        assert camera_schema.GetHorizontalApertureAttr().Get() == pytest.approx(22.0444444444)
        assert camera_schema.GetVerticalApertureAttr().Get() == pytest.approx(12.4)
        assert camera_schema.GetFocalLengthAttr().Get() == pytest.approx(FOCAL_MM)

    def test_a_request_that_omits_them_authors_usds_own(self, usd, worker) -> None:
        """Authored, not inherited -- and equal to the fallback, so nothing re-frames silently."""
        _, prim = _authored(usd, worker, camera())
        camera_schema = usd.UsdGeom.Camera(prim)
        assert camera_schema.GetHorizontalApertureAttr().Get() == pytest.approx(20.955)
        assert camera_schema.GetVerticalApertureAttr().Get() == pytest.approx(15.2908)
        assert camera_schema.GetHorizontalApertureAttr().IsAuthored()

    def test_the_camera_stands_at_the_eye_it_was_given(self, usd, worker) -> None:
        stage, prim = _authored(usd, worker, camera())
        matrix = usd.UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        origin = matrix.Transform(usd.Gf.Vec3d(0.0, 0.0, 0.0))
        assert tuple(origin) == pytest.approx(EYE)
        assert [str(child.GetName()) for child in stage.GetPrimAtPath("/World/Cameras").GetChildren()] \
            == ["overview_top"]


class TestTheProbeAndTheRendererShareACameraBasis:
    def test_the_authored_axis_directions_are_the_probes(self, usd, worker, probe) -> None:
        """The probe reconstructs room geometry from a depth map using this basis, so it has to
        be the renderer's basis, not a plausible-looking one."""
        _, prim = _authored(usd, worker, camera())
        matrix = usd.UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        x_cam, y_cam, z_cam = probe.camera_basis(EYE, TARGET)
        for axis, expected in ((usd.Gf.Vec3d(1, 0, 0), x_cam), (usd.Gf.Vec3d(0, 1, 0), y_cam),
                               (usd.Gf.Vec3d(0, 0, 1), z_cam)):
            assert tuple(matrix.TransformDir(axis)) == pytest.approx(expected, abs=1e-9)

    def test_the_view_direction_is_the_eye_to_target_line(self, usd, worker) -> None:
        _, prim = _authored(usd, worker, camera())
        matrix = usd.UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        forward = matrix.TransformDir(usd.Gf.Vec3d(0.0, 0.0, -1.0))
        span = sum((TARGET[i] - EYE[i]) ** 2 for i in range(3)) ** 0.5
        assert tuple(forward) == pytest.approx(tuple((TARGET[i] - EYE[i]) / span for i in range(3)))


class TestTheFixtureReachesTheWorker:
    def test_the_declared_aperture_is_the_one_authored(self, usd, worker, probe,
                                                      fixture_module) -> None:
        fixture = fixture_module.load(REPO_ROOT / "tests/scenes/fixtures/appearance_fixture_v1.json")
        request = {"cameras": [{"camera_id": "overview_top", "purpose": "export", "eye": EYE,
                                "target": TARGET, "focal_length_mm": FOCAL_MM,
                                "resolution": [1280, 720]}],
                   "room": {"polygon_xy": [[0.0, 0.0], [4.7456, 0.0], [4.7456, 3.7569], [0.0, 3.7569]],
                            "height_m": 2.8852999210357666, "openings": []},
                   "lighting": {"dome_intensity": 650.0, "sun_intensity": 650.0,
                                "ceiling_fill_intensity": 500.0,
                                "dome_color": [0.92, 0.94, 1.0], "sun_color": [1.0, 0.96, 0.88]}}
        rewritten = fixture_module.worker_request(fixture, request, "interior_fill")
        carried = CameraRequest.from_dict(rewritten["cameras"][0])
        _, prim = _authored(usd, worker, carried)
        camera_schema = usd.UsdGeom.Camera(prim)
        assert camera_schema.GetHorizontalApertureAttr().Get() == pytest.approx(
            fixture["camera"]["aperture_mm"])
        # And the framing that aperture implies is the one the probe would measure against.
        measured = probe.hypotheses({"eye": EYE, "target": TARGET, "resolution": [1280, 720],
                                     "focal_length_mm": FOCAL_MM,
                                     "aperture_mm": carried.horizontal_aperture_mm})
        assert measured["asserted_aperture_horizontal"] == pytest.approx(389.6305, abs=1e-3)
        assert measured["retired_blender_sensor_width"] == pytest.approx(370.3749, abs=1e-3)

    def test_the_two_instruments_name_the_same_thing(self, probe, fixture_module) -> None:
        """One calls it a horizontal field of view, the other a focal length in pixels. If they
        described different cameras, a report could quote either and look consistent."""
        fov = fixture_module.horizontal_fov_deg(FOCAL_MM, 20.955)
        focal_px = probe.hypotheses({"eye": EYE, "target": TARGET, "focal_length_mm": FOCAL_MM,
                                     "aperture_mm": 20.955,
                                     "resolution": [1280, 720]})["asserted_aperture_horizontal"]
        # The probe reports its predictions rounded to four decimals, so compare at that scale.
        assert focal_px == pytest.approx((1280 / 2) / math.tan(math.radians(fov) / 2), abs=1e-4)
