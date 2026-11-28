# SPDX-License-Identifier: Apache-2.0
"""One camera and one lighting mode, shared by the three renderers of a comparison.

Import-safe: it pulls in neither bpy nor the upstream tree, so a fixture can be validated and
its rig derived before Blender or Isaac start, and from a plain interpreter.

Two things this module deliberately does *not* claim:

- **Photometric equivalence.** Cycles and RTX differ in BSDF, sampling and colour management.
  The fixture fixes the declared inputs; it does not make the images equal, and a fixture that
  says otherwise is refused rather than accepted as an ambitious goal.
- **That a verdict may be written against any mode.** Only `ambient_only` carries the
  acceptance role. With sun and fill on, a material loss and a lighting difference cannot be
  told apart, so a checklist for `room_like` is refused. It exists to be looked at.
"""
import copy
import json
import math
from pathlib import Path

FIXTURE_VERSION = 'pvscene.appearance-fixture/v1'
ACCEPTANCE_MODE = 'interior_fill'
EQUIVALENCE = 'not_claimed'
NOT_RUN = 'not_run'
RENDERERS = ('blender_mapped', 'blender_baked', 'isaac')
# Only a horizontal film fit is supported. It is the fit Isaac's USD camera and the measured
# Blender fixture both use, and admitting 'VERTICAL'/'AUTO' would admit a horizontal field of
# view this module cannot compute -- an unsupported value should be a named failure, not a
# silently wrong number.
SENSOR_FITS = ('HORIZONTAL',)
# The same resolutions `MaterialExportProfile` accepts. A fixture may not ask for a bake the
# exporter would refuse, or the refusal would surface late as a bake crash instead of here.
BAKE_RESOLUTIONS = (256, 512, 1024, 2048, 4096)
LIGHTING_FIELDS = ('dome_intensity', 'sun_intensity', 'ceiling_fill_intensity',
                   'dome_color', 'sun_color')


def load(path):
    """Read and validate a fixture. Every rejection names the reason."""
    fixture = json.loads(Path(path).read_text())
    if fixture.get('version') != FIXTURE_VERSION:
        raise ValueError(f'APPEARANCE_FIXTURE_VERSION_UNSUPPORTED({fixture.get("version")})')
    if fixture.get('photometric_equivalence') != EQUIVALENCE:
        raise ValueError(f'APPEARANCE_EQUIVALENCE_CLAIMED({fixture.get("photometric_equivalence")})')
    camera = fixture.get('camera') or {}
    if camera.get('sensor_fit') not in SENSOR_FITS:
        raise ValueError(f'APPEARANCE_SENSOR_FIT_UNSUPPORTED({camera.get("sensor_fit")})')
    for field in ('aperture_mm', 'vertical_aperture_mm'):
        if not float(camera.get(field) or 0) > 0:
            raise ValueError(f'APPEARANCE_APERTURE_INVALID({field}={camera.get(field)})')
    modes = fixture.get('modes') or {}
    acceptance = [name for name, block in modes.items() if block.get('role') == 'acceptance']
    if acceptance != [ACCEPTANCE_MODE]:
        raise ValueError(f'APPEARANCE_ACCEPTANCE_MODE_MISSING({acceptance})')
    for name, block in modes.items():
        _validate_mode(name, block)
    resolutions = fixture.get('bake_resolutions_px') or []
    if not resolutions:
        raise ValueError('APPEARANCE_RESOLUTION_UNSUPPORTED(empty)')
    for px in resolutions:
        if px not in BAKE_RESOLUTIONS:
            raise ValueError(f'APPEARANCE_RESOLUTION_UNSUPPORTED({px})')
    if not fixture.get('checklist_items'):
        raise ValueError('APPEARANCE_CHECKLIST_EMPTY')
    return fixture


def _validate_mode(name, block):
    if block.get('role') not in ('acceptance', 'diagnostic'):
        raise ValueError(f'APPEARANCE_MODE_UNSUPPORTED({name}:role={block.get("role")})')
    for renderer in ('blender', 'isaac'):
        if not isinstance(block.get(renderer), dict):
            raise ValueError(f'APPEARANCE_MODE_UNSUPPORTED({name}:{renderer})')
    for field in LIGHTING_FIELDS:
        if field not in block['isaac']:
            raise ValueError(f'APPEARANCE_MODE_UNSUPPORTED({name}:isaac.{field})')
        value = block['isaac'][field]
        if field.endswith('_intensity') and float(value) < 0:
            raise ValueError(f'APPEARANCE_INTENSITY_INVALID({name}:{field}={value})')
    if float(block['blender'].get('world_strength', 0)) < 0:
        raise ValueError(f'APPEARANCE_INTENSITY_INVALID({name}:world_strength)')
    if float(block['blender'].get('area_energy_W', 0)) < 0:
        raise ValueError(f'APPEARANCE_INTENSITY_INVALID({name}:area_energy_W)')
    # The acceptance mode's whole purpose is that the sun is off and exactly one interior fill
    # lights the room on both sides. A sun in that mode would be judged as a material
    # difference, and no fill at all leaves a sealed room dark on both sides -- which is what
    # an ambient-only mode measured: both renders came out near black through the windows.
    if name == ACCEPTANCE_MODE:
        if float(block['isaac']['sun_intensity']) != 0:
            raise ValueError(f'APPEARANCE_MODE_CONTRACT({name}:sun_on)')
        if float(block['isaac']['ceiling_fill_intensity']) <= 0:
            raise ValueError(f'APPEARANCE_MODE_CONTRACT({name}:no_interior_fill)')
        if not block['blender'].get('area_enabled') or float(block['blender']['area_energy_W']) <= 0:
            raise ValueError(f'APPEARANCE_MODE_CONTRACT({name}:no_blender_area_light)')


def mode_block(fixture, mode):
    try:
        return fixture['modes'][mode]
    except KeyError:
        raise ValueError(f'APPEARANCE_MODE_UNSUPPORTED({mode})') from None


def horizontal_fov_deg(focal_length_mm, aperture_mm):
    """Horizontal field of view of a horizontally fitted camera."""
    if focal_length_mm <= 0:
        raise ValueError(f'APPEARANCE_FOCAL_LENGTH_INVALID({focal_length_mm})')
    return math.degrees(2 * math.atan(aperture_mm / (2 * focal_length_mm)))


def camera_rig(fixture, request):
    """The one camera all three renderers use, taken from the delivery's own CameraRequest.

    Eye, target, focal length and resolution come from the request: those already were shared.
    The aperture comes from the fixture, because it was not -- Isaac wrote only ``focalLength``
    and inherited USD's default while the Blender side hand-coded a different number.
    """
    camera_id = fixture['camera']['camera_id']
    found = next((c for c in request.get('cameras', []) if c.get('camera_id') == camera_id), None)
    if found is None:
        raise ValueError(f'APPEARANCE_CAMERA_MISSING({camera_id})')
    aperture = float(fixture['camera']['aperture_mm'])
    focal = float(found['focal_length_mm'])
    resolution = fixture['camera'].get('render_resolution') or found['resolution']
    return {'camera_id': camera_id, 'mode_camera_source': 'delivery CameraRequest',
            'eye': [float(v) for v in found['eye']], 'target': [float(v) for v in found['target']],
            'focal_length_mm': focal, 'resolution': [int(v) for v in resolution],
            'sensor_fit': fixture['camera']['sensor_fit'], 'aperture_mm': aperture,
            'vertical_aperture_mm': float(fixture['camera']['vertical_aperture_mm']),
            'horizontal_fov_deg': horizontal_fov_deg(focal, aperture),
            'samples': int(fixture['camera'].get('samples', 32)),
            'fov_equality': dict(fixture['camera'].get('fov_check') or {}),
            'photometric_equivalence': EQUIVALENCE}


def blender_rig(fixture, request, mode):
    """Everything the Blender renderer applies: the shared camera plus this mode's lighting.

    The area light hangs under the ceiling at the room's centroid, as the recorded diagnostic
    pair did, so its numbers can be replayed.
    """
    block = mode_block(fixture, mode)
    lighting = block['blender']
    room = request['room']
    polygon = room['polygon_xy']
    centroid = [sum(p[i] for p in polygon) / len(polygon) for i in (0, 1)]
    rig = camera_rig(fixture, request)
    rig['mode'] = mode
    rig['role'] = block['role']
    rig['world_strength'] = float(lighting['world_strength'])
    rig['area_enabled'] = bool(lighting['area_enabled'])
    rig['area_energy_W'] = float(lighting['area_energy_W'])
    rig['area_size_m'] = float(lighting['area_size_m'])
    rig['area_location'] = [centroid[0], centroid[1], room['height_m'] - float(lighting['area_drop_m'])]
    return rig


def lighting(fixture, mode):
    """The Isaac `LightingRequest` fields for this mode."""
    block = mode_block(fixture, mode)['isaac']
    return {field: ([float(c) for c in block[field]] if field.endswith('_color')
                    else float(block[field])) for field in LIGHTING_FIELDS}


def worker_request(fixture, request, mode):
    """A worker request whose lighting and camera resolution come from the fixture.

    A copy: the delivery's own request stays the input of record, and the mode is visible in
    the request the worker actually received.
    """
    rig = camera_rig(fixture, request)
    rewritten = copy.deepcopy(request)
    rewritten['lighting'] = lighting(fixture, mode)
    for camera in rewritten.get('cameras', []):
        if camera.get('camera_id') == rig['camera_id']:
            camera['resolution'] = list(rig['resolution'])
            camera['horizontal_aperture_mm'] = rig['aperture_mm']
            camera['vertical_aperture_mm'] = rig['vertical_aperture_mm']
    rewritten['appearance_fixture'] = {'fixture_id': fixture['fixture_id'], 'mode': mode,
                                       'role': mode_block(fixture, mode)['role'],
                                       'aperture_mm': rig['aperture_mm'],
                                       'vertical_aperture_mm': rig['vertical_aperture_mm'],
                                       'photometric_equivalence': EQUIVALENCE}
    return rewritten


def bake_resolutions(fixture):
    return tuple(int(px) for px in fixture['bake_resolutions_px'])


def checklist(fixture, *, mode=ACCEPTANCE_MODE):
    """The human review sheet, every item at not_run.

    Refused for a diagnostic mode: an empty verdict for `room_like` would look like progress
    toward the one verdict that matters.
    """
    block = mode_block(fixture, mode)
    if block['role'] != 'acceptance':
        raise ValueError(f'APPEARANCE_MODE_NOT_ACCEPTANCE({mode})')
    return {'mode': mode, 'reviewer': '', 'reviewed_at': '',
            'items': {name: NOT_RUN for name in fixture['checklist_items']},
            'verdict': NOT_RUN, 'note': fixture.get('checklist_note', '')}


def three_way_verdict(renders, review):
    """What the three renders and the human sheet add up to.

    Three images on disk is not a result: a verdict needs the acceptance mode's sheet to have
    every item at `pass`. This is the check that keeps `three_way_visual_review` from being
    written in as a constant.
    """
    missing = [name for name in RENDERERS if not renders.get(name)]
    if missing:
        raise ValueError(f'APPEARANCE_RENDER_MISSING({",".join(missing)})')
    if review.get('mode') != ACCEPTANCE_MODE:
        raise ValueError(f'APPEARANCE_MODE_NOT_ACCEPTANCE({review.get("mode")})')
    items = review.get('items') or {}
    unmet = sorted(name for name, value in items.items() if value != 'pass')
    if any(value == 'fail' for value in items.values()):
        verdict = 'fail'
    elif unmet:
        verdict = NOT_RUN
    else:
        verdict = 'pass'
    return {'verdict': verdict, 'unmet': unmet, 'mode': review.get('mode'),
            'reviewer': review.get('reviewer', ''),
            'photometric_equivalence': EQUIVALENCE,
            'renders': {name: str(renders[name]) for name in RENDERERS}}
