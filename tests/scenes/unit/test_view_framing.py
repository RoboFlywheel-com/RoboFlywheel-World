# SPDX-License-Identifier: Apache-2.0
"""SPEC-015 / ADR-011: a delivered view's framing comes from the room, not from a constant.

The measured failure these cover: a 67.62 m2 office failed its delivery because `layout_top` and
`room_eye_south` held fixed focals (14 mm / 18 mm) and framed only floor and wall, so the worker
refused the pass three times over an instance-id channel with an empty id map. Small rooms had
passed all along because the narrow frame happened to still hold the furniture.
"""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import rfw_w.scenes.assembly.cameras as cameras_module
import math
import sys
from pathlib import Path

import pytest

from rfw_w.scenes.geometry.polygon import Polygon

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = module_path('rfw_w.scenes.application')

FRAMING = importlib.import_module('rfw_w.scenes.application')


fitted_focal = cameras_module.fitted_focal
long_edge_side_view = cameras_module.long_edge_side_view
layout_obligation = cameras_module.layout_obligation

RESOLUTION = (1280, 720)
APERTURE_H_MM = 20.955
#: Small / medium / large rectangles from SPEC-006's scale classes, and a U-shaped large room.
SMALL = Polygon(((0.0, 0.0), (4.9, 0.0), (4.9, 3.7), (0.0, 3.7)))
MEDIUM = Polygon(((0.0, 0.0), (6.9, 0.0), (6.9, 5.7), (0.0, 5.7)))
LARGE = Polygon(((0.0, 0.0), (9.8, 0.0), (9.8, 7.4), (0.0, 7.4)))
U_SHAPED = Polygon((
    (0.0, 0.0), (9.8, 0.0), (9.8, 7.4), (6.0, 7.4), (6.0, 3.0), (3.8, 3.0), (3.8, 7.4), (0.0, 7.4),
))


def inside_the_frame(result, polygon, floor_z=0.0, height_m=2.9, margin=1.06) -> None:
    """The view's own obligation: its forward half and the room's height are in the picture."""
    eye = result["eye"]
    target = result["target"]
    forward = [target[i] - eye[i] for i in range(3)]
    length = math.sqrt(sum(component * component for component in forward))
    forward = [component / length for component in forward]
    right = [forward[1], -forward[0], 0.0]
    right_length = math.hypot(*right[:2])
    right = [component / right_length for component in right]
    fx = result["focal_length_mm"] * RESOLUTION[0] / APERTURE_H_MM
    fy = fx
    distance = math.dist(eye[:2], target[:2])
    up = [right[1] * forward[2] - right[2] * forward[1],
          right[2] * forward[0] - right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0]]

    def forward_of(point) -> float:
        return sum((point[i] - eye[i]) * forward[i] for i in range(3))

    # The forward half is the half *in front of the camera*, not the half that is far from it: a
    # wall-hugging eye is closer to the far corners than to the ones beside it (measured: the
    # rectangle's near corners are 2.59 m away against the far wall's 2.85 m).
    far = [v for v in polygon.vertices if forward_of((v[0], v[1], floor_z)) >= 0.5 * distance]
    assert far, "the fixture must have a forward half"
    for vertex in far:
        for z in (floor_z, floor_z + height_m):
            offset = [vertex[0] - eye[0], vertex[1] - eye[1], z - eye[2]]
            forward_m = forward_of((vertex[0], vertex[1], z))
            assert forward_m > 0, "a vertex of the forward half must be in front of the camera"
            assert abs(sum(offset[i] * right[i] for i in range(3)) / forward_m) * fx <= RESOLUTION[0] / 2
            assert abs(sum(offset[i] * up[i] for i in range(3)) / forward_m) * fy <= RESOLUTION[1] / 2


class TestTheSideViewIsFittedToTheRoom:
    @pytest.mark.parametrize("polygon", [SMALL, MEDIUM, LARGE, U_SHAPED])
    def test_every_forward_half_vertex_lands_in_frame(self, polygon):
        result = long_edge_side_view(polygon, floor_z_m=0.0, height_m=2.9)
        inside_the_frame(result, polygon)

    def test_the_standoff_grows_with_the_room(self):
        small = long_edge_side_view(SMALL, floor_z_m=0.0, height_m=2.9)
        large = long_edge_side_view(LARGE, floor_z_m=0.0, height_m=2.9)
        assert large["distance_m"] > small["distance_m"] * 1.5

    def test_the_lens_is_not_a_constant(self):
        # The whole bug was a fixed focal: these three rooms must not share one.
        focals = {long_edge_side_view(p, floor_z_m=0.0, height_m=2.9)["focal_length_mm"]
                  for p in (SMALL, MEDIUM, LARGE)}
        assert len(focals) == 3

    def test_a_long_thin_room_gets_a_wider_lens_than_a_square_one_of_the_same_area(self):
        thin = Polygon(((0.0, 0.0), (11.0, 0.0), (11.0, 2.0), (0.0, 2.0)))     # 22 m2
        square = Polygon(((0.0, 0.0), (4.7, 0.0), (4.7, 4.7), (0.0, 4.7)))    # 22.1 m2
        assert (long_edge_side_view(thin, floor_z_m=0.0, height_m=2.9)["focal_length_mm"]
                < long_edge_side_view(square, floor_z_m=0.0, height_m=2.9)["focal_length_mm"])

    def test_the_eye_stands_inside_a_u_shaped_room_clear_of_its_walls(self):
        result = long_edge_side_view(U_SHAPED, floor_z_m=0.0, height_m=2.9)
        assert long_edge_side_view(U_SHAPED, floor_z_m=0.0, height_m=2.9)["eye_offset_from_edge_m"] >= 0.15
        assert result["eye"][2] == pytest.approx(1.50)


class TestTheApertureModelIsTheMeasuredOne:
    """The renderer reproduces the declared horizontal aperture and squares the pixels.

    Measured 2026-09-24 (`scripts/dev/measure_view_geometry.py`): four independent views of one
    delivery agree within 3% of 20.955 mm, and the declared vertical aperture changes nothing.
    """

    def test_a_tall_room_binds_the_vertical_constraint(self):
        # A short room's far half is bound horizontally; a room taller than it is wide, seen from
        # close by, binds vertically -- and then the lens follows the *aspect*, not the declared
        # vertical aperture (which is why the effective vertical aperture is 11.79 mm, not 15.29).
        eye = (2.0, 0.2, 1.5)
        target = (2.0, 4.0, 1.5)
        near = fitted_focal(eye, target, [(2.0, 4.0, 0.0), (2.0, 4.0, 2.9)],
                            aperture_h_mm=APERTURE_H_MM, resolution=RESOLUTION)
        tall = fitted_focal(eye, target, [(2.0, 4.0, 0.0), (2.0, 4.0, 9.0)],
                            aperture_h_mm=APERTURE_H_MM, resolution=RESOLUTION)
        assert tall[0] < near[0]

    def test_the_focal_is_what_the_horizontal_aperture_says(self):
        # tan(half) * f == width / (2 * aperture_h): the relation the whole design rests on.
        eye = (0.0, 0.0, 1.5)
        target = (0.0, 6.0, 1.5)
        focal, tan_h, _ = fitted_focal(eye, target, [(3.0, 6.0, 1.5)],
                                       aperture_h_mm=APERTURE_H_MM, resolution=RESOLUTION,
                                       margin=1.0)
        # tan(half) = aperture_h / (2 f), and with the point 3 m off-axis at 6 m that is f = a_h / 1
        assert focal * tan_h == pytest.approx(APERTURE_H_MM / 2.0)
        assert focal == pytest.approx(APERTURE_H_MM)


class TestTheSideViewIsACameraTheProtocolAccepts:
    """The bug this exists for: `purpose` is a closed vocabulary.

    It was written as "overview", which `CameraRequest.__post_init__` rejects -- and the resulting
    ProtocolError was swallowed by the orchestrator into a stage outcome, so a full run stopped
    before its first render with no traceback and no "running the worker" line. Runs looked like
    they simply ended. Constructing the request is the cheap check that would have caught it.
    """

    def test_the_built_camera_constructs_with_the_purpose_it_is_given(self):
        from rfw_w.scenes.workers.isaac.protocol import CameraRequest

        view = long_edge_side_view(SMALL, floor_z_m=0.0, height_m=2.9)
        camera = CameraRequest(
            camera_id="overview_side", purpose="export", eye=view["eye"], target=view["target"],
            focal_length_mm=view["focal_length_mm"], resolution=RESOLUTION,
            target_instance_ids=(), require_target_visible=False,
        )
        assert camera.focal_length_mm > 0
        assert camera.purpose in ("layout", "relation", "final", "export")


class TestTheLayoutViewOwesTheTaskAreaNotTheWholeRoom:
    """Fitted to the whole footprint it became a 4.93 mm lens in an 18 m2 room.

    Measured on the same scene, judged through the two camera sets: workspace_and_passage 4 -> 2
    and proportion_density 3 -> 2. The room is `overview_side`'s obligation; the layout view keeps
    the reserved task cell legible, which is what its purpose means.
    """

    def test_a_smaller_obligation_means_a_longer_lens(self):
        eye = (2.45, 1.85, 2.75)
        target = (2.47, 1.85, 0.0)
        whole_room = [(0.0, 0.0, 0.0), (4.9, 0.0, 0.0), (4.9, 3.7, 0.0), (0.0, 3.7, 0.0)]
        task_cell = [(0.3, 0.95, 0.0), (2.0, 0.95, 0.0), (2.0, 2.2, 0.0), (0.3, 2.2, 0.0)]
        wide = fitted_focal(eye, target, whole_room, aperture_h_mm=APERTURE_H_MM,
                            resolution=RESOLUTION)[0]
        tight = fitted_focal(eye, target, task_cell + [(1.26, 2.37, 0.75)],
                             aperture_h_mm=APERTURE_H_MM, resolution=RESOLUTION)[0]
        # Measured on the real rooms: 4.11 -> 8.22 mm (small) and 2.12 -> 4.34 mm (large), about
        # twice as long. This synthetic cell spans nearly the room's whole x range, so it is a
        # weaker case; the semantic claim (the cell and the target are what it owes) is the next
        # test's, and the magnitude is the measurement's.
        assert tight > wide * 1.1

    def test_the_task_target_stays_inside_the_frame(self):
        eye = (2.45, 1.85, 2.75)
        target = (2.47, 1.85, 0.0)
        cell = [(0.3, 0.95, 0.0), (2.0, 0.95, 0.0), (2.0, 2.2, 0.0), (0.3, 2.2, 0.0)]
        cup = (1.26, 2.37, 0.75)
        focal, tan_h, tan_v = fitted_focal(eye, target, cell + [cup],
                                           aperture_h_mm=APERTURE_H_MM, resolution=RESOLUTION)
        fx = focal * RESOLUTION[0] / APERTURE_H_MM
        forward = [target[i] - eye[i] for i in range(3)]
        length = math.sqrt(sum(c * c for c in forward))
        forward = [c / length for c in forward]
        right = [forward[1], -forward[0], 0.0]
        rl = math.hypot(*right[:2]); right = [c / rl for c in right]
        up = [right[1] * forward[2] - right[2] * forward[1],
              right[2] * forward[0] - right[0] * forward[2],
              right[0] * forward[1] - right[1] * forward[0]]
        for point in cell + [cup]:
            offset = [point[i] - eye[i] for i in range(3)]
            depth = sum(offset[i] * forward[i] for i in range(3))
            assert abs(sum(offset[i] * right[i] for i in range(3)) / depth) * fx <= RESOLUTION[0] / 2
            assert abs(sum(offset[i] * up[i] for i in range(3)) / depth) * fx <= RESOLUTION[1] / 2


class TestTheLayoutObligationReadsTheSceneItIsGiven:
    """The shape of `scene["zones"]` is a `ZoneGraph`, not the dict the manifest publishes.

    Reading it as a mapping raised AttributeError inside the request build. The orchestrator turns
    a stage exception into an outcome with no traceback, so a full run stopped before its first
    render with nothing in the log -- the cheap test is the one that would have caught it.
    """

    def test_a_zone_graph_with_a_cell_yields_the_cell_and_the_target(self):
        from types import SimpleNamespace
        from rfw_w.scenes.geometry.polygon import Polygon

        zones = SimpleNamespace(workspace=Polygon(((0.3, 0.95), (2.0, 0.95), (2.0, 2.2), (0.3, 2.2))))
        points = layout_obligation(zones, LARGE, (1.26, 2.37), floor_z_m=0.0, support_top_z=0.75)
        assert len(points) == 5
        assert (1.26, 2.37, 0.75) in points
        assert all(point[2] in (0.0, 0.75) for point in points)

    def test_no_reserved_cell_falls_back_to_the_footprint(self):
        from types import SimpleNamespace

        zones = SimpleNamespace(workspace=None)
        points = layout_obligation(zones, SMALL, (1.26, 2.37), floor_z_m=0.0, support_top_z=0.75)
        assert sum(1 for point in points if point[2] == 0.0) == len(SMALL.vertices)


class TestThePreFlightNamesWhatItCannotFrame:
    def test_a_room_with_no_legal_vantage_is_a_named_gap(self):
        with pytest.raises(ValueError) as error:
            long_edge_side_view(Polygon(((0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.0, 0.3))),
                                floor_z_m=0.0, height_m=2.9)
        assert "VIEW_NO_VANTAGE(long_edge_side_view)" in str(error.value)
