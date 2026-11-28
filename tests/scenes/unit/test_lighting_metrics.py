# SPDX-License-Identifier: Apache-2.0
"""The exposure reading must be linear, and must refuse a target it cannot see.

Checking against hand values matters here for one reason: the measurement decides whether a room
is lit, and it is easy to write it on encoded sRGB bytes -- which weights the shadows heavily
enough that a room nobody can see in reads as acceptable. The hand values below are the check
that the decode happens at all.
"""
import importlib
from rfw_w.scenes.resources import module_path, data_path
from pathlib import Path

import pytest

_PATH = module_path('rfw_w.scenes.visualization.lighting_metrics')

METRICS = importlib.import_module('rfw_w.scenes.visualization.lighting_metrics')


class TestTheDecodeIsLinear:
    def test_black_and_white_are_the_endpoints(self) -> None:
        assert METRICS.luminance_from_srgb(0, 0, 0) == 0.0
        assert METRICS.luminance_from_srgb(255, 255, 255) == pytest.approx(1.0)

    def test_mid_grey_is_regarded_as_mid_light_not_mid_byte(self) -> None:
        """sRGB 128 is 0.216 linear, not 0.502. The difference is the whole point of the decode."""
        assert METRICS.srgb_to_linear(128 / 255.0) == pytest.approx(0.2159, abs=1e-4)
        assert METRICS.luminance_from_srgb(128, 128, 128) == pytest.approx(0.2159, abs=1e-4)

    @pytest.mark.parametrize('encoded,linear', [(0.04045, 0.003131), (0.5, 0.21404),
                                                (0.9, 0.78741)])
    def test_the_piecewise_curve_matches_its_two_branches(self, encoded, linear) -> None:
        assert METRICS.srgb_to_linear(encoded) == pytest.approx(linear, abs=1e-5)

    def test_the_weights_are_rec709_and_apply_to_linear_channels(self) -> None:
        # A saturated green is the brightest primary in Rec.709's weights, blue the darkest.
        assert METRICS.luminance_from_srgb(0, 255, 0) == pytest.approx(0.7152)
        assert METRICS.luminance_from_srgb(0, 0, 255) == pytest.approx(0.0722)
        assert METRICS.luminance(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_a_channel_above_white_is_not_silently_clamped(self) -> None:
        """Values above 1.0 are the caller's business; the metric must not hide them."""
        assert METRICS.luminance(1.2, 1.2, 1.2) > 1.0


class TestTheTargetStatistics:
    def test_a_well_lit_target_passes(self) -> None:
        stats = METRICS.statistics([0.2, 0.3, 0.4, 0.5, 0.6])
        assert stats['median_y'] == pytest.approx(0.4)
        assert stats['underexposed_fraction'] == 0.0
        assert METRICS.check_camera('task_closeup', stats)['target_pixels'] == 5

    def test_a_target_in_the_dark_is_named(self) -> None:
        stats = METRICS.statistics([0.0, 0.001, 0.002, 0.5])
        assert stats['underexposed_fraction'] == 0.75
        with pytest.raises(ValueError, match=r'TARGET_UNDEREXPOSED\(task_closeup'):
            METRICS.check_camera('task_closeup', stats)

    def test_a_blown_target_is_named(self) -> None:
        stats = METRICS.statistics([0.99, 0.995, 0.5, 0.5])
        assert stats['overexposed_fraction'] == 0.5
        with pytest.raises(ValueError, match=r'TARGET_OVEREXPOSED\(overview_alt'):
            METRICS.check_camera('overview_alt', stats)

    def test_the_limits_are_limits_and_not_directions(self) -> None:
        """Exactly at a limit passes; a hair over fails. Otherwise the number is decorative."""
        at_limit = [0.0] * 20 + [0.5] * 80
        assert METRICS.check_camera('c', METRICS.statistics(at_limit))
        over_limit = [0.0] * 21 + [0.5] * 79
        with pytest.raises(ValueError, match='TARGET_UNDEREXPOSED'):
            METRICS.check_camera('c', METRICS.statistics(over_limit))

    def test_a_target_the_camera_cannot_see_is_refused_rather_than_silently_passing(self) -> None:
        with pytest.raises(ValueError, match=r'CAMERA_TARGET_NOT_VISIBLE\(mask empty\)'):
            METRICS.statistics([])

    def test_the_summary_carries_the_bands_with_the_number(self) -> None:
        summary = METRICS.exposure_summary(METRICS.statistics([0.4, 0.45, 0.5]))
        assert 'median Y 0.4500' in summary and 'limit 0.2' in summary and 'limit 0.05' in summary

    def test_an_even_count_medians_between_the_middle_two(self) -> None:
        assert METRICS.statistics([0.1, 0.2, 0.3, 0.4])['median_y'] == pytest.approx(0.25)
