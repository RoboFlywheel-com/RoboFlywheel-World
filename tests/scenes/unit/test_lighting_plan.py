# SPDX-License-Identifier: Apache-2.0
"""A plan must name its units and its losses, and must reach the renderer as what it declared.

The test that matters most is the second one: the fields this plan produces are checked against
`LightingRequest`'s own field names. A plan whose keys drifted from the wire format would render
with defaults -- the room would come back unlit, and every plan-side test would still be green.
"""
from __future__ import annotations

import dataclasses

import pytest

from rfw_w.scenes.contracts.lighting import (
    CALIBRATION,
    CALIBRATION_SCOPE,
    KNOWN_LOSSES,
    PROFILE_LEVELS,
    PROFILES,
    UNIT_MAPPING,
    LightSource,
    LightingError,
    LightingPlan,
    from_appearance,
    profile,
)
from rfw_w.scenes.workers.isaac.protocol import LightingRequest

ROOM = {'room_width_m': 6.9, 'room_depth_m': 5.7, 'room_height_m': 3.0}


class TestTheProfiles:
    @pytest.mark.parametrize('name', PROFILES)
    def test_every_profile_builds_and_declares_its_calibration(self, name) -> None:
        plan = profile(name, **ROOM)
        assert plan.calibration == CALIBRATION[name]
        assert plan.environment is not None
        assert plan.to_request_fields()['dome_intensity'] == PROFILE_LEVELS[name]['environment']

    def test_the_calibrated_profile_says_so_and_the_others_do_not_claim_it(self) -> None:
        """A measured level and a proposed one must not read the same in a report."""
        assert profile('mixed', **ROOM).calibration.startswith('measured(')
        assert profile('daylight', **ROOM).calibration == 'not_run'
        assert profile('warm_artificial', **ROOM).calibration == 'not_run'
        assert CALIBRATION_SCOPE, 'a measurement without its scope is a number without a claim'

    def test_the_calibrated_level_clears_the_exposure_limit_it_was_measured_against(self) -> None:
        """The measured run: 8.67 percent of target pixels below Y=0.01, against a limit of 20."""
        plan = profile('mixed', room_width_m=4.7456, room_depth_m=3.6733, room_height_m=2.8853)
        fields = plan.to_request_fields()
        assert fields['ceiling_fill_intensity'] == 3600.0
        assert fields['dome_intensity'] == 2700.0 and fields['sun_intensity'] == 4500.0

    def test_the_artificial_profile_has_no_sun(self) -> None:
        assert profile('warm_artificial', **ROOM).to_request_fields()['sun_intensity'] == 0.0
        assert profile('daylight', **ROOM).to_request_fields()['sun_intensity'] > 0

    def test_the_fixture_follows_the_room_rather_than_sitting_at_a_fixed_point(self) -> None:
        """SPEC-008 requirement 3: a large room is not lit by a copy of a small room's lamp."""
        small = profile('daylight', room_width_m=3.0, room_depth_m=3.0, room_height_m=2.6)
        large = profile('daylight', room_width_m=9.0, room_depth_m=7.0, room_height_m=3.4)
        assert large.fill.size_m > small.fill.size_m
        assert large.fill.position_m == pytest.approx((4.5, 3.5, 3.35))
        assert small.fill.position_m == pytest.approx((1.5, 1.5, 2.55))
        assert large.fill.size_m == pytest.approx((6.3, 4.9))

    def test_an_unknown_profile_is_named(self) -> None:
        with pytest.raises(LightingError, match=r'LIGHT_PROFILE_UNSUPPORTED\(evening\)'):
            profile('evening', **ROOM)

    @pytest.mark.parametrize('key,label', [('room_width_m', 'width'),
                                           ('room_depth_m', 'depth'),
                                           ('room_height_m', 'height')])
    def test_a_room_that_is_not_a_room_is_named(self, key, label) -> None:
        with pytest.raises(LightingError, match=rf'LIGHT_ROOM_INVALID\({label}'):
            profile('daylight', **{**ROOM, key: 0.0})


class TestTheFieldsReachTheRenderer:
    @pytest.mark.parametrize('name', PROFILES)
    def test_the_field_names_are_the_wire_formats(self, name) -> None:
        produced = profile(name, **ROOM).to_request_fields()
        # A key the worker does not read would be dropped without a word, so the names have to be
        # the wire format's own -- and the values have to be ones it accepts.
        assert set(produced) == {'dome_intensity', 'sun_intensity', 'ceiling_fill_intensity',
                                 'dome_color', 'sun_color'}
        assert set(produced) <= {field.name for field in dataclasses.fields(LightingRequest)}
        assert LightingRequest(**produced).dome_intensity == produced['dome_intensity']

    def test_a_plan_with_nothing_that_lights_the_room_is_refused(self) -> None:
        empty = LightingPlan(profile='daylight')
        with pytest.raises(LightingError, match=r'LIGHT_PLAN_EMPTY\(daylight\)'):
            empty.to_request_fields()


class TestEverySourceNamesItsUnitAndItsLosses:
    @pytest.mark.parametrize('name', PROFILES)
    def test_each_source_has_a_unit_and_a_mapping(self, name) -> None:
        for source in profile(name, **ROOM).sources():
            assert source.unit in {unit for unit, _ in UNIT_MAPPING.values()}
            assert source.mapping, 'a unit without a mapping is a number without a meaning'

    def test_a_recognised_but_unmappable_kind_is_a_named_gap(self) -> None:
        """A spotlight is a real kind and still cannot be mapped. Guessing is not on the table."""
        with pytest.raises(LightingError, match=r'LIGHT_UNIT_MAPPING_MISSING\(spot\)'):
            LightSource(kind='spot', source='plan', intensity=1.0, colour=(1.0, 1.0, 1.0))

    def test_an_unknown_kind_is_named_differently_from_an_unmapped_one(self) -> None:
        with pytest.raises(LightingError, match=r'LIGHT_ENVIRONMENT_UNSUPPORTED\(lantern\)'):
            LightSource(kind='lantern', source='plan', intensity=1.0, colour=(1.0, 1.0, 1.0))

    def test_the_plan_reports_what_it_could_not_carry(self) -> None:
        gaps = profile('daylight', **ROOM).named_gaps()
        assert 'nishita-world-node-not-exported' in gaps
        assert 'environment:nishita-sky-gradient' in gaps
        assert 'sun:angular-size' in gaps
        assert set(gaps) >= {f'environment:{loss}' for loss in KNOWN_LOSSES['environment']}

    @pytest.mark.parametrize('kwargs,match', [
        ({'intensity': -1.0}, r'LIGHT_INTENSITY_INVALID\(rect:-1.0\)'),
        ({'colour': (1.0, 1.0)}, r'LIGHT_COLOUR_INVALID\(rect\)'),
        ({'temperature_k': 0.0}, r'LIGHT_TEMPERATURE_INVALID\(rect:0.0\)'),
    ])
    def test_a_source_that_cannot_be_rendered_is_named(self, kwargs, match) -> None:
        payload = {'kind': 'rect', 'source': 'plan', 'intensity': 1.0, 'colour': (1.0, 1.0, 1.0)}
        payload.update(kwargs)
        with pytest.raises(LightingError, match=match):
            LightSource(**payload)


class TestThePlanRoundTrips:
    def test_a_plan_survives_a_dict_round_trip(self) -> None:
        original = profile('mixed', seed=17, **ROOM)
        restored = LightingPlan.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()
        assert restored.seed == 17

    def test_an_exposure_outside_the_supported_range_is_named(self) -> None:
        with pytest.raises(LightingError, match=r'LIGHT_EXPOSURE_INVALID\(9.0\)'):
            LightingPlan(profile='daylight', exposure=9.0)

    def test_a_changed_plan_is_a_new_value(self) -> None:
        """A repair hands back a plan; the one it was given is not what the last render used."""
        original = profile('daylight', **ROOM)
        stronger = original.with_sources(
            environment=dataclasses.replace(original.environment, intensity=9999.0))
        assert original.environment.intensity != stronger.environment.intensity
        assert stronger.to_request_fields()['dome_intensity'] == 9999.0


class TestTheAppearanceBridge:
    class _Appearance:
        dome_intensity = 650.0
        sun_intensity = 650.0
        ceiling_fill_intensity = 500.0
        dome_color = (0.92, 0.94, 1.0)
        sun_color = (1.0, 0.96, 0.88)

    def test_the_solved_values_reach_the_plan(self) -> None:
        plan = from_appearance(self._Appearance(), profile_name='mixed', **ROOM)
        fields = plan.to_request_fields()
        assert fields['dome_intensity'] == 650.0
        assert fields['sun_intensity'] == 650.0
        assert fields['ceiling_fill_intensity'] == 500.0
        # The plan still says what it cannot carry, whatever the numbers are.
        assert 'environment:nishita-sun-disc' in plan.named_gaps()

    def test_an_artificial_profile_does_not_grow_a_sun(self) -> None:
        plan = from_appearance(self._Appearance(), profile_name='warm_artificial', **ROOM)
        assert plan.sun is None
        assert plan.to_request_fields()['sun_intensity'] == 0.0
