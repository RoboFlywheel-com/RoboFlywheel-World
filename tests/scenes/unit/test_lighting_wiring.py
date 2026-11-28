# SPDX-License-Identifier: Apache-2.0
"""SPEC-008 TASK 3: which numbers reach the renderer, and what the delivery says about them.

The defect this wiring fixes is measured, not hypothesised: the task target rendered *essentially
unlit* (median Y 0.0020, 99.89% of its pixels below 0.01) because the worker request was built from
`AppearanceSettings`' defaults. Those defaults were never the calibrated answer -- they were a
starting point for the appearance repair rung -- so the wiring has to keep two things true at once,
and both are pinned here:

* the calibrated profile is what a delivery renders with unless a repair moved the dials;
* a repair that *did* move them still changes the picture (the track AGENTS.md's gap 13 records).
"""
from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import rfw_w.scenes.assembly.lighting as lighting_module
import rfw_w.scenes.solvers.furnish as furnish_module
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rfw_w.scenes.contracts.lighting import (
    CALIBRATION,
    PROFILES,
    PROFILE_LEVELS,
    LightSource,  # noqa: F401  (named in the annotations the plan carries)
    profile as lighting_profile,
)
from rfw_w.scenes.workers.isaac.protocol import LightingRequest

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = module_path('rfw_w.scenes.application')

WIRING = importlib.import_module('rfw_w.scenes.application')


AppearanceSettings = furnish_module.AppearanceSettings

ROOM = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))


def scene(*, appearance=None) -> dict:
    """The driver's scene, in the shape `lighting_plan_for` reads: shell and plan, nothing else."""
    return {"shell": SimpleNamespace(polygon_xy=ROOM, height_m=2.8),
            "plan": SimpleNamespace(appearance=appearance or AppearanceSettings())}


# ---------------------------------------------------------------------------------------------
# The calibrated profile is the source, and it is frozen
# ---------------------------------------------------------------------------------------------

def test_the_profile_is_the_source_and_the_room_is_what_it_was_sized_for():
    plan, source, room = lighting_module.lighting_plan_for(scene(), 'mixed')
    assert source == 'profile'
    assert plan.profile == 'mixed'
    assert room == {'room_width_m': 4.0, 'room_depth_m': 3.0, 'room_height_m': 2.8}
    fields = plan.to_request_fields()
    # The calibrated levels, not the appearance defaults (650/650/500) that rendered the target
    # unlit. These three numbers are the measurement of 2026-09-22 and are frozen with the profile.
    assert fields['dome_intensity'] == PROFILE_LEVELS['mixed']['environment'] == 2700.0
    assert fields['sun_intensity'] == PROFILE_LEVELS['mixed']['sun'] == 4500.0
    assert fields['ceiling_fill_intensity'] == PROFILE_LEVELS['mixed']['fill'] == 3600.0
    assert plan.calibration == CALIBRATION['mixed'] == 'measured(2026-09-22)'


def test_the_profile_is_calibrated_by_name_and_the_other_two_say_they_are_not():
    """A profile nobody measured must not render as if it had been."""
    assert set(PROFILE_LEVELS) == set(PROFILES) == {'daylight', 'warm_artificial', 'mixed'}
    for name in ('daylight', 'warm_artificial'):
        plan, _source, _room = lighting_module.lighting_plan_for(scene(), name)
        assert CALIBRATION[name] == 'not_run'
        assert plan.calibration == 'not_run', f'{name} claims a calibration it has not had'
    with pytest.raises(SystemExit, match='LIGHT_PROFILE_UNSUPPORTED'):
        lighting_module.lighting_plan_for(scene(), 'candlelight')


def test_the_fill_follows_the_room_so_a_large_room_is_not_lit_by_a_small_room_lamp():
    small, _source, _room = lighting_module.lighting_plan_for(scene(), 'mixed')
    large, _source, _room = lighting_module.lighting_plan_for(
        {'shell': SimpleNamespace(polygon_xy=((0.0, 0.0), (9.0, 0.0), (9.0, 7.0), (0.0, 7.0)),
                                  height_m=2.8),
         'plan': SimpleNamespace(appearance=AppearanceSettings())}, 'mixed')
    small_fill = small.fill.size_m
    large_fill = large.fill.size_m
    assert small_fill == pytest.approx((2.8, 2.1))
    assert large_fill == pytest.approx((6.3, 4.9))
    # The same fixture geometry the profile contract promises, and the same intensity: whether that
    # intensity is enough for the larger room is exactly what the matrix measures rather than
    # something this test decides.
    assert small.fill.intensity == large.fill.intensity == 3600.0


# ---------------------------------------------------------------------------------------------
# A repair still wins, and the delivery says which of the two produced the picture
# ---------------------------------------------------------------------------------------------

def test_whatever_moved_the_appearance_multiplies_the_profile_rather_than_replacing_it():
    """Read as a replacement, the appearance threw the calibrated profile away.

    Every randomized run (`--variant-seed` samples dome 720-1080, sun 2080-3120, fill 560-840)
    rendered at a third of the calibrated level and stayed dark: measured on the office, 75% of the
    target's pixels below 0.01 where the threshold is 20%. As a ratio it keeps both claims: the
    picture still varies, and a run nobody varied renders the profile exactly.
    """
    # A repair's own step is multiplicative (`repair._adjust_appearance` scales by 1.35).
    repaired = AppearanceSettings(dome_intensity=650.0 * 1.35, sun_intensity=650.0 * 1.35,
                                 ceiling_fill_intensity=500.0 * 1.35)
    plan, source, _room = lighting_module.lighting_plan_for(scene(appearance=repaired), 'mixed')
    assert source == 'profile x appearance'
    fields = plan.to_request_fields()
    assert fields['dome_intensity'] == pytest.approx(2700.0 * 1.35)
    assert fields['sun_intensity'] == pytest.approx(4500.0 * 1.35)
    assert fields['ceiling_fill_intensity'] == pytest.approx(3600.0 * 1.35)
    # The profile still supplies what a repair cannot: the room-sized fixture and the node losses.
    assert plan.fill.size_m == pytest.approx((2.8, 2.1))
    assert plan.named_gaps()
    # A randomized appearance (the sampled range's midpoint) lands in the same relation.
    sampled = AppearanceSettings(dome_intensity=900.0, sun_intensity=2600.0,
                                 ceiling_fill_intensity=700.0)
    randomized, source, _room = lighting_module.lighting_plan_for(scene(appearance=sampled), 'mixed')
    assert source == 'profile x appearance'
    assert randomized.to_request_fields()['dome_intensity'] == pytest.approx(
        2700.0 * sampled.dome_intensity / 650.0)
    assert randomized.to_request_fields()['dome_intensity'] > 2700.0


def test_the_request_the_worker_gets_is_the_plans_own_five_fields():
    plan = lighting_profile('mixed', room_width_m=4.0, room_depth_m=3.0, room_height_m=2.8)
    request = LightingRequest.from_plan(plan)
    assert json.loads(json.dumps(request.to_dict())) == json.loads(json.dumps(
        plan.to_request_fields()))
    # And it parses back from the wire: the request is written to disk and read by the worker.
    assert LightingRequest.from_dict(json.loads(json.dumps(request.to_dict()))) == request


def test_the_delivery_block_records_the_profile_the_numbers_and_what_could_not_be_carried():
    plan, source, room = lighting_module.lighting_plan_for(scene(), 'mixed')
    block = lighting_module.lighting_block(plan, source, room=room)
    assert block['profile'] == 'mixed' and block['source'] == 'profile'
    assert block['calibration'] == 'measured(2026-09-22)'
    assert block['request']['dome_intensity'] == 2700.0
    assert block['room_m'] == {'room_width_m': 4.0, 'room_depth_m': 3.0, 'room_height_m': 2.8}
    # The honest half: the upstream nodes the mapping does not carry are named, so the picture is
    # not presented as the Nishita sky it was solved from.
    assert any('nishita' in gap for gap in block['named_gaps'])
    kinds = {item['kind']: item for item in block['sources']}
    assert kinds['rect']['size_m'] == [2.8, 2.1]  # the ceiling fill, sized to this room
    assert kinds['rect']['position_m'][2] == round(2.8 - 0.05, 6)  # just under the ceiling
    assert kinds['environment']['intensity'] == 2700.0
    assert json.loads(json.dumps(block)) == block  # it goes into the manifest, so it must serialise


def test_the_manifest_carries_the_block_and_the_contract_accepts_an_empty_one():
    from rfw_w.scenes.contracts.delivery import DeliveryManifest
    from rfw_w.scenes.contracts.readiness import ReadinessLevel, ReadinessResolution
    common = dict(scene_id='s', prompt='p', seed=1, requested_backends=['isaac'],
                  scene_version='a' * 64, gates={},
                  resolution=ReadinessResolution(
                      requested=ReadinessLevel.DRAFT, achieved=ReadinessLevel.DRAFT, outcome=None,
                      scene_version='a' * 64, requested_backends=('isaac',), blockers=()))
    assert DeliveryManifest(**common, lighting={'profile': 'mixed'}).to_dict()['lighting'] == {
        'profile': 'mixed'}
    # A delivery that skipped the backend has no lighting to report, and an empty block says that
    # rather than standing in for a default.
    assert DeliveryManifest(**common).to_dict()['lighting'] == {}

def test_the_block_helper_is_the_one_shape_both_call_sites_use():
    """The bug this pins: the manifest called `lighting_block(*lighting_plan_for(...))`, passing the
    room positionally to a keyword-only parameter. It raised *after* the room had been built and
    rendered -- two SPEC-008 matrix runs died there -- so the two call sites now share one helper
    and the helper's own shape is what is checked."""
    block = lighting_module.lighting_block_for(scene(), 'mixed')
    assert block['profile'] == 'mixed' and block['source'] == 'profile'
    assert block['room_m']['room_height_m'] == 2.8
    with pytest.raises(TypeError):
        lighting_module.lighting_block(lighting_module.lighting_plan_for(scene(), 'mixed')[0], 'profile', {})
