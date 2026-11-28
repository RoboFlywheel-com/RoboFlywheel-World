# SPDX-License-Identifier: Apache-2.0
"""The three-way fixture must refuse what it cannot honestly support.

Two claims are checked here rather than asserted in prose: that only `interior_fill` can carry a
verdict, and that the aperture is a single authored value instead of one renderer defaulting and
the other hand-coding a different number. The second one was a real gap -- the Blender fixture
carried 22.044444 mm against Isaac's inherited 20.955 mm, a 2.55 degree difference in horizontal
field of view that the recorded pair was compared across.
"""
import copy
import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_PATH = module_path('rfw_w.scenes.visualization.appearance_fixture')

_MODULE = importlib.import_module('rfw_w.scenes.visualization.appearance_fixture')


FIXTURE_PATH = REPO_ROOT / 'tests/scenes' / 'fixtures' / 'appearance_fixture_v1.json'


def _request():
    """The parts of a delivery request the fixture reads, with the recorded office numbers."""
    return {'cameras': [{'camera_id': 'overview_top', 'purpose': 'export',
                         'eye': [2.3728, 0.7929918609719235, 2.6853],
                         'target': [2.3728, 1.83665, 0.0], 'focal_length_mm': 6.378678048182431,
                         'resolution': [1280, 720]},
                        {'camera_id': 'layout_top', 'eye': [2.0, 1.0, 3.0],
                         'target': [2.0, 1.0, 0.0], 'focal_length_mm': 10.0,
                         'resolution': [1024, 1024]}],
            'room': {'polygon_xy': [[0.0, 0.0], [4.74560022354126, 0.0],
                                    [4.74560022354126, 3.7568650932784196], [0.0, 3.7568650932784196]],
                     'height_m': 2.8852999210357666, 'openings': []},
            'lighting': {'dome_intensity': 650.0, 'sun_intensity': 650.0,
                         'ceiling_fill_intensity': 500.0, 'dome_color': [0.92, 0.94, 1.0],
                         'sun_color': [1.0, 0.96, 0.88]}}


@pytest.fixture()
def fixture():
    return _MODULE.load(FIXTURE_PATH)


def _mutated(tmp_path, **changes):
    """A fixture file with one thing changed, to prove the change is what gets refused."""
    raw = json.loads(FIXTURE_PATH.read_text())
    for dotted, value in changes.items():
        head, _, tail = dotted.rpartition('.')
        target = raw
        for step in (head.split('.') if head else []):
            target = target[step]
        if value is _DELETE:
            del target[tail]
        else:
            target[tail] = value
    path = tmp_path / 'fixture.json'
    path.write_text(json.dumps(raw))
    return path


_DELETE = object()


def test_the_recorded_fixture_loads(fixture):
    assert fixture['fixture_id'] == 'pvscene.appearance-three-way/v1'
    assert _MODULE.bake_resolutions(fixture) == (512, 1024)


def test_the_aperture_is_one_authored_value(fixture):
    """Pinned: the old 22.044444 mm never comes back as a second, uncompared camera."""
    assert fixture['camera']['aperture_mm'] == 20.955
    rig = _MODULE.camera_rig(fixture, _request())
    assert rig['aperture_mm'] == 20.955
    assert round(rig['horizontal_fov_deg'], 3) == 117.334


def test_both_horizontal_fields_of_view_are_reported_not_assumed(fixture):
    """The fixture records that equality is unmeasured; it does not assert it."""
    rig = _MODULE.camera_rig(fixture, _request())
    assert rig['fov_equality']['status'] == 'not_run'
    assert rig['fov_equality']['checked_by'] == 'rfw_w.scenes.visualization.probe_camera_fov'


def test_the_old_blender_aperture_would_not_have_matched(fixture):
    """Keeps the evidence for why the aperture had to be unified in the first place."""
    focal = _request()['cameras'][0]['focal_length_mm']
    assert round(_MODULE.horizontal_fov_deg(focal, 22.0444444444), 3) == 119.883
    assert round(_MODULE.horizontal_fov_deg(focal, 20.955), 3) == 117.334


def test_only_the_acceptance_mode_carries_the_verdict(fixture):
    assert _MODULE.mode_block(fixture, 'interior_fill')['role'] == 'acceptance'
    assert _MODULE.mode_block(fixture, 'room_like')['role'] == 'diagnostic'
    with pytest.raises(ValueError, match='APPEARANCE_MODE_NOT_ACCEPTANCE'):
        _MODULE.checklist(fixture, mode='room_like')


@pytest.mark.parametrize('mode', ['interior_fill', 'room_like'])
def test_the_camera_comes_from_the_delivery_request(fixture, mode):
    rig = _MODULE.camera_rig(fixture, _request())
    camera = _request()['cameras'][0]
    assert rig['eye'] == camera['eye'] and rig['target'] == camera['target']
    assert rig['focal_length_mm'] == camera['focal_length_mm']
    assert rig['resolution'] == camera['resolution']
    assert _MODULE.blender_rig(fixture, _request(), mode)['mode'] == mode


def test_a_fixture_camera_the_delivery_lacks_is_named(fixture, tmp_path):
    path = _mutated(tmp_path, **{'camera.camera_id': 'overview_corner'})
    rigged = _MODULE.load(path)
    with pytest.raises(ValueError, match=r'APPEARANCE_CAMERA_MISSING\(overview_corner\)'):
        _MODULE.camera_rig(rigged, _request())


def test_the_acceptance_mode_has_one_fill_and_no_sun(fixture):
    """The sun is off, and both sides carry an interior fill -- without one a sealed room
    renders near black on both sides, which is what the ambient-only attempt measured."""
    block = _MODULE.mode_block(fixture, 'interior_fill')
    assert block['isaac']['sun_intensity'] == 0.0
    assert block['isaac']['ceiling_fill_intensity'] > 0
    assert block['blender']['area_enabled'] is True
    rig = _MODULE.blender_rig(fixture, _request(), 'interior_fill')
    assert rig['area_enabled'] is True and rig['area_energy_W'] > 0


def test_the_area_light_hangs_under_the_room_ceiling(fixture):
    rig = _MODULE.blender_rig(fixture, _request(), 'room_like')
    polygon = _request()['room']['polygon_xy']
    assert rig['area_location'][0] == pytest.approx(sum(p[0] for p in polygon) / len(polygon))
    assert rig['area_location'][2] == pytest.approx(2.8852999210357666 - 0.2)


def test_the_worker_request_carries_the_mode_and_leaves_the_input_alone(fixture):
    request = _request()
    before = copy.deepcopy(request)
    interior = _MODULE.worker_request(fixture, request, 'interior_fill')
    room_like = _MODULE.worker_request(fixture, request, 'room_like')
    assert request == before, 'the delivery request is the input of record'
    assert interior['appearance_fixture']['mode'] == 'interior_fill'
    assert interior['appearance_fixture']['role'] == 'acceptance'
    assert interior['lighting']['sun_intensity'] == 0.0
    assert room_like['lighting']['sun_intensity'] == 650.0
    assert interior['room'] == before['room'], 'the fixture changes lighting and the camera only'


def test_a_resolution_override_reaches_the_worker_request(fixture, tmp_path):
    path = _mutated(tmp_path, **{'camera.render_resolution': [960, 540]})
    overridden = _MODULE.worker_request(_MODULE.load(path), _request(), 'interior_fill')
    camera = next(c for c in overridden['cameras'] if c['camera_id'] == 'overview_top')
    assert camera['resolution'] == [960, 540]
    assert next(c for c in _request()['cameras']
                if c['camera_id'] == 'overview_top')['resolution'] == [1280, 720]


def test_phases_of_the_review_start_at_not_run(fixture):
    review = _MODULE.checklist(fixture)
    assert review['verdict'] == 'not_run'
    assert set(review['items']) == {'texture_present', 'scale_plausible', 'no_seam',
                                    'no_normal_inversion', 'glass_approximation_acceptable'}
    assert set(review['items'].values()) == {'not_run'}


def _renders(**overrides):
    paths = {name: f'/tmp/{name}.png' for name in _MODULE.RENDERERS}
    paths.update(overrides)
    return paths


def test_three_images_are_not_a_passed_review(fixture):
    verdict = _MODULE.three_way_verdict(_renders(), _MODULE.checklist(fixture))
    assert verdict['verdict'] == 'not_run'
    assert sorted(verdict['unmet']) == sorted(_MODULE.checklist(fixture)['items'])


def test_a_complete_sheet_passes_and_one_failure_stops_it(fixture):
    review = _MODULE.checklist(fixture)
    review['items'] = {name: 'pass' for name in review['items']}
    review['reviewer'] = 'human'
    assert _MODULE.three_way_verdict(_renders(), review)['verdict'] == 'pass'
    review['items']['no_seam'] = 'fail'
    verdict = _MODULE.three_way_verdict(_renders(), review)
    assert verdict['verdict'] == 'fail' and verdict['unmet'] == ['no_seam']


def test_a_missing_render_is_named(fixture):
    review = _MODULE.checklist(fixture)
    review['items'] = {name: 'pass' for name in review['items']}
    with pytest.raises(ValueError, match=r'APPEARANCE_RENDER_MISSING\(isaac\)'):
        _MODULE.three_way_verdict(_renders(isaac=None), review)


def test_a_diagnostic_sheet_cannot_be_turned_into_a_verdict(fixture):
    review = _MODULE.checklist(fixture)
    review['mode'] = 'room_like'
    review['items'] = {name: 'pass' for name in review['items']}
    with pytest.raises(ValueError, match=r'APPEARANCE_MODE_NOT_ACCEPTANCE\(room_like\)'):
        _MODULE.three_way_verdict(_renders(), review)


@pytest.mark.parametrize('change,match', [
    ({'version': 'pvscene.appearance-fixture/v0'}, 'APPEARANCE_FIXTURE_VERSION_UNSUPPORTED'),
    ({'photometric_equivalence': 'meters_match'}, 'APPEARANCE_EQUIVALENCE_CLAIMED'),
    ({'camera.sensor_fit': 'AUTO'}, 'APPEARANCE_SENSOR_FIT_UNSUPPORTED'),
    ({'camera.aperture_mm': 0.0}, 'APPEARANCE_APERTURE_INVALID'),
    ({'camera.aperture_mm': _DELETE}, 'APPEARANCE_APERTURE_INVALID'),
    ({'camera.render_resolution': [1280, 720]}, None),
    ({'bake_resolutions_px': [512, 300]}, 'APPEARANCE_RESOLUTION_UNSUPPORTED'),
    ({'bake_resolutions_px': []}, 'APPEARANCE_RESOLUTION_UNSUPPORTED'),
    ({'checklist_items': []}, 'APPEARANCE_CHECKLIST_EMPTY'),
    ({'modes.interior_fill.isaac.sun_intensity': 650.0}, 'APPEARANCE_MODE_CONTRACT'),
    ({'modes.interior_fill.isaac.ceiling_fill_intensity': 0.0}, 'APPEARANCE_MODE_CONTRACT'),
    ({'modes.interior_fill.blender.area_enabled': False}, 'APPEARANCE_MODE_CONTRACT'),
    ({'modes.interior_fill.blender.area_energy_W': 0.0}, 'APPEARANCE_MODE_CONTRACT'),
    ({'modes.interior_fill.isaac.sun_intensity': -1.0}, 'APPEARANCE_INTENSITY_INVALID'),
    ({'modes.room_like.blender.area_energy_W': -5.0}, 'APPEARANCE_INTENSITY_INVALID'),
    ({'modes.room_like.isaac.dome_color': _DELETE}, 'APPEARANCE_MODE_UNSUPPORTED'),
    ({'modes.room_like.role': 'verdict'}, 'APPEARANCE_MODE_UNSUPPORTED'),
    ({'modes.interior_fill.role': 'diagnostic'}, 'APPEARANCE_ACCEPTANCE_MODE_MISSING'),
    ({'modes.room_like': _DELETE}, None),
])
def test_a_broken_fixture_names_the_reason(tmp_path, change, match):
    path = _mutated(tmp_path, **change)
    if match is None:
        assert _MODULE.load(path)['version'] == _MODULE.FIXTURE_VERSION
        return
    with pytest.raises(ValueError, match=match):
        _MODULE.load(path)


def test_an_unsupported_mode_is_named(fixture):
    with pytest.raises(ValueError, match=r'APPEARANCE_MODE_UNSUPPORTED\(evening\)'):
        _MODULE.mode_block(fixture, 'evening')
    with pytest.raises(ValueError, match=r'APPEARANCE_MODE_UNSUPPORTED\(evening\)'):
        _MODULE.lighting(fixture, 'evening')


def test_a_lighting_block_is_exactly_what_the_worker_accepts(fixture):
    """These five names are `LightingRequest`'s fields; a sixth would be dropped silently."""
    for mode in ('interior_fill', 'room_like'):
        assert set(_MODULE.lighting(fixture, mode)) == set(_MODULE.LIGHTING_FIELDS)
