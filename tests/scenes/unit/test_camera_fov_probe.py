# SPDX-License-Identifier: Apache-2.0
"""The framing probe is tested against depth maps whose framing is known by construction.

The room is traced analytically, so the test does not need Isaac to know the right answer: a
rectangular room rendered at a chosen focal length in pixels must be recovered as that focal
length, and a rig declaring a different aperture must be refused. Without that, the probe would
be a number nobody had ever checked -- which is how the two renderers came to be compared across
a 2.55 degree difference in horizontal field of view in the first place.
"""
import importlib
from rfw_w.scenes.resources import module_path, data_path
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_PATH = module_path('rfw_w.scenes.visualization.probe_camera_fov')

PROBE = importlib.import_module('rfw_w.scenes.visualization.probe_camera_fov')


ROOM = {'width': 4.74560022354126, 'depth': 3.7568650932784196, 'height': 2.8852999210357666}
# The recorded office export camera, whose framing is what the fixture has to reproduce.
EYE = [2.3728, 0.7929918609719235, 2.6853]
TARGET = [2.3728, 1.83665, 0.0]
FOCAL_MM = 6.378678048182431


def rig(*, aperture_mm=20.955, resolution=(160, 90), eye=EYE, target=TARGET):
    return {'eye': list(eye), 'target': list(target), 'focal_length_mm': FOCAL_MM,
            'aperture_mm': aperture_mm, 'resolution': [resolution[0], resolution[1]]}


def trace_depth(rig_block, room=ROOM, *, focal_px):
    """The depth map a pinhole renderer at `focal_px` would produce for a rectangular room."""
    width, height = rig_block['resolution']
    eye = np.array(rig_block['eye'], dtype=float)
    x_cam, y_cam, z_cam = (np.array(axis, dtype=float)
                           for axis in PROBE.camera_basis(rig_block['eye'], rig_block['target']))
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    rows, cols = np.mgrid[0:height, 0:width]
    directions = (((cols - cx) / focal_px)[..., None] * x_cam
                  - ((rows - cy) / focal_px)[..., None] * y_cam - z_cam)
    depth = np.full((height, width), np.nan)
    nearest = np.full((height, width), np.inf)
    # (axis, plane value, bounds on the other two axes) for the floor, ceiling and four walls.
    surfaces = [(2, 0.0, (0.0, room['width']), (0.0, room['depth'])),
                (2, room['height'], (0.0, room['width']), (0.0, room['depth'])),
                (0, 0.0, (0.0, room['depth']), (0.0, room['height'])),
                (0, room['width'], (0.0, room['depth']), (0.0, room['height'])),
                (1, 0.0, (0.0, room['width']), (0.0, room['height'])),
                (1, room['depth'], (0.0, room['width']), (0.0, room['height']))]
    for axis, value, bounds_a, bounds_b in surfaces:
        with np.errstate(divide='ignore', invalid='ignore'):
            step = (value - eye[axis]) / directions[..., axis]
        point_a = eye[(axis + 1) % 3] + step * directions[..., (axis + 1) % 3]
        point_b = eye[(axis + 2) % 3] + step * directions[..., (axis + 2) % 3]
        hit = ((step > 1e-6) & np.isfinite(step)
               & (point_a >= bounds_a[0]) & (point_a <= bounds_a[1])
               & (point_b >= bounds_b[0]) & (point_b <= bounds_b[1]))
        depth = np.where(hit & (step < nearest), step, depth)
        nearest = np.minimum(nearest, np.where(hit, step, np.inf))
    return depth


@pytest.mark.parametrize('convention', ['usd_default_horizontal', 'retired_blender_sensor_width',
                                        'usd_default_vertical_fit'])
def test_the_probe_recovers_the_framing_it_was_rendered_at(convention):
    """A renderer that framed at focal length f must be measured as f, whatever the reason."""
    declared = rig()
    focal_px = PROBE.hypotheses(declared)[convention]
    measured = PROBE.fit_camera(trace_depth(declared, focal_px=focal_px), declared, ROOM['height'])
    assert measured['conclusive'] is True
    # The share of plane pixels follows the framing rather than the fit: the vertical-fit case
    # at 160px wide is a 129 degree lens, so walls outnumber floor. What must hold is the peak.
    assert measured['plane_fraction'] > 0.2
    assert measured['explained_fraction'] >= PROBE.PEAK_RATIO * measured['search_background_fraction']
    assert measured['focal_px'] == pytest.approx(focal_px, rel=PROBE.TOLERANCE_FRACTION)


def test_a_consistent_declaration_passes_and_names_its_convention():
    declared = rig(aperture_mm=20.955)
    focal_px = PROBE.hypotheses(declared)['usd_default_horizontal']
    measured = PROBE.fit_camera(trace_depth(declared, focal_px=focal_px), declared, ROOM['height'])
    result = PROBE.check(declared, measured)
    # The asserted aperture *is* USD's default here, so both names describe the same number; what
    # has to hold is that the measured framing is the one the fixture declares.
    assert result['nearest_hypothesis'] in ('asserted_aperture_horizontal', 'usd_default_horizontal')
    assert result['nearest_hypothesis_focal_px'] == pytest.approx(measured['focal_px'],
                                                                 rel=PROBE.TOLERANCE_FRACTION)
    assert result['matches_asserted_aperture'] is True


def test_a_rig_declaring_the_retired_blender_sensor_no_longer_matches():
    """The recorded pair's 22.044444 mm against a renderer that used 20.955: a named failure."""
    declared = rig(aperture_mm=22.0444444444)
    focal_px = PROBE.hypotheses(declared)['usd_default_horizontal']
    measured = PROBE.fit_camera(trace_depth(declared, focal_px=focal_px), declared, ROOM['height'])
    with pytest.raises(ValueError, match=r'CAMERA_FOV_AUTHORED_APERTURE_MISMATCH'):
        PROBE.check(declared, measured)
    assert PROBE.compare(declared, measured)['nearest_hypothesis'] == 'usd_default_horizontal'


def test_a_declaration_that_matches_its_own_render_is_accepted():
    """22.044444 is not wrong in itself; declaring it while rendering at 20.955 is."""
    declared = rig(aperture_mm=22.0444444444)
    focal_px = PROBE.hypotheses(declared)['retired_blender_sensor_width']
    measured = PROBE.fit_camera(trace_depth(declared, focal_px=focal_px), declared, ROOM['height'])
    assert PROBE.check(declared, measured)['matches_asserted_aperture'] is True


def test_a_renderer_that_fits_the_vertical_aperture_is_recognised():
    declared = rig()
    focal_px = PROBE.hypotheses(declared)['usd_default_vertical_fit']
    measured = PROBE.fit_camera(trace_depth(declared, focal_px=focal_px), declared, ROOM['height'])
    result = PROBE.compare(declared, measured)
    assert result['nearest_hypothesis'] == 'usd_default_vertical_fit'
    assert result['matches_asserted_aperture'] is False


def test_the_hypothesis_table_keeps_the_two_old_values_apart():
    """At the recorded 1280px framing: 4.4 percent apart, so the pair was not one camera."""
    predicted = PROBE.hypotheses(rig(resolution=(1280, 720)))
    assert predicted['usd_default_horizontal'] == pytest.approx(389.6305, abs=1e-3)
    assert predicted['retired_blender_sensor_width'] == pytest.approx(370.3749, abs=1e-3)
    assert predicted['usd_default_vertical_fit'] == pytest.approx(300.3537, abs=1e-3)
    assert predicted['asserted_aperture_horizontal'] == predicted['usd_default_horizontal']


def test_an_unexplained_screen_is_inconclusive_rather_than_a_number():
    declared = rig()
    flat = np.full((90, 160), 1.0)
    measured = PROBE.fit_camera(flat, declared, ROOM['height'])
    assert measured['conclusive'] is False
    with pytest.raises(ValueError, match=r'CAMERA_FOV_PROBE_INCONCLUSIVE'):
        PROBE.check(declared, measured)


def test_an_empty_depth_map_is_inconclusive():
    declared = rig()
    measured = PROBE.fit_camera(np.full((90, 160), np.nan), declared, ROOM['height'])
    assert measured['conclusive'] is False


def test_the_reconstruction_puts_floor_pixels_on_the_floor():
    """The fitted point cloud, not just the score: floor pixels land on z=0 inside the room."""
    declared = rig()
    focal_px = PROBE.hypotheses(declared)['usd_default_horizontal']
    depth = trace_depth(declared, focal_px=focal_px)
    points = PROBE.world_points(depth, declared, focal_px)
    on_floor = np.abs(points[:, 2]) <= PROBE.PLANE_BAND_M
    assert on_floor.mean() > 0.25
    inside = points[on_floor]
    assert inside[:, 0].min() > -0.05 and inside[:, 0].max() < ROOM['width'] + 0.05
    assert inside[:, 1].min() > -0.05 and inside[:, 1].max() < ROOM['depth'] + 0.05


def test_one_pixel_against_hand_arithmetic():
    """The one check the tracer cannot satisfy by agreeing with itself.

    A camera at (0,0,1) looking at (0,1,0) with a 3x3 frame at focal length 1: the hand-worked
    axes are x_cam=(1,0,0), y_cam=(0,1,1)/sqrt(2), z_cam=(0,-1,1)/sqrt(2). The centre-bottom
    pixel is the ray (0,0,-sqrt(2)) -- straight down -- so its 0.7071 m hit is the world origin,
    while the centre-top pixel is the horizontal ray (0,sqrt(2),0), whose 0.7071 m point is one
    metre forward at the camera's own height. A flipped row convention swaps those two.
    """
    declared = {'eye': [0.0, 0.0, 1.0], 'target': [0.0, 1.0, 0.0], 'focal_length_mm': 1.0,
                'aperture_mm': 1.0, 'resolution': [3, 3]}
    depth = np.full((3, 3), np.nan)
    depth[2, 1] = math.sqrt(2) / 2
    below = PROBE.world_points(depth, declared, 1.0)
    assert below[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    depth = np.full((3, 3), np.nan)
    depth[0, 1] = math.sqrt(2) / 2
    along = PROBE.world_points(depth, declared, 1.0)
    assert along[0] == pytest.approx([0.0, 1.0, 1.0], abs=1e-12)


def test_the_camera_basis_matches_the_look_at_convention():
    """+X right, +Y up in camera space, `z_cam` behind the camera, up vector +Z."""
    x_cam, y_cam, z_cam = PROBE.camera_basis(EYE, TARGET)
    assert z_cam[2] == pytest.approx(0.9321, abs=1e-3)  # eye above target: it points up and back
    assert all(abs(component) < 1e-9 for component in np.cross(x_cam, y_cam) - np.array(z_cam))
    assert sum(c * c for c in y_cam) == pytest.approx(1.0)
