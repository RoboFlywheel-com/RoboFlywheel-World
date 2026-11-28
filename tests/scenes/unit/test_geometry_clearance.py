# SPDX-License-Identifier: Apache-2.0
"""Offsets and clearance.

The margin semantics are the point. The acceptance manual sets an initial 0.05 m margin on top
of the robot's geometry, and a clearance check that reports a bare boolean cannot tell a
candidate that fits with room to spare from one that misses by a millimetre. So the results
carry a signed measured margin, and the tests pin down what the sign means.
"""

from __future__ import annotations

import math

import pytest

from rfw_w.scenes.geometry.clearance import DEFAULT_MARGIN_M, ClearanceError, check_clearance
from rfw_w.scenes.geometry.polygon import Polygon, PolygonError, inset, offset_outward

RECT = Polygon.rectangle(0, 0, 4, 3)
# An L-shaped desk surface: the notch is where an object must not be placed.
L_DESK = Polygon.from_sequence([(0, 0), (1.2, 0), (1.2, 0.3), (0.5, 0.3), (0.5, 0.6), (0, 0.6)])


class TestOffsets:
    def test_growing_increases_area(self) -> None:
        grown = offset_outward(RECT, 0.1)
        assert grown.area > RECT.area

    def test_shrinking_decreases_area(self) -> None:
        shrunk = offset_outward(RECT, -0.1)
        assert shrunk.area < RECT.area

    def test_grow_then_shrink_returns_to_the_original_outline(self) -> None:
        # Areas are not symmetric under ±d (4.2x3.2 is not 3.8x2.8), but the outline is.
        restored = offset_outward(offset_outward(RECT, 0.1), -0.1)
        assert restored.bounds() == pytest.approx(RECT.bounds(), abs=1e-9)

    def test_zero_offset_is_identity(self) -> None:
        assert offset_outward(RECT, 0.0) is RECT

    def test_inward_inset_keeps_the_shape_centred(self) -> None:
        shrunk = inset(RECT, 0.5)
        assert shrunk.area == pytest.approx(3.0 * 2.0)
        assert shrunk.bounds() == pytest.approx((0.5, 0.5, 3.5, 2.5))

    def test_l_shape_inset_preserves_the_notch(self) -> None:
        """Offsetting re-intersects adjacent edges, so a reflex corner stays a reflex corner."""
        shrunk = inset(L_DESK, 0.05)
        assert shrunk.area < L_DESK.area
        assert not shrunk.is_rectangle()

    def test_shrinking_past_collapse_raises(self) -> None:
        """An over-shrunk outline inverts rather than emptying, and grows while doing it.

        Shrinking a 4x3 rectangle by 5 m produces a 6x7 m polygon with the opposite winding,
        which is *larger* than the input. An area-based guard would wave it through, and the
        inverted outline would then satisfy containment tests it should fail.
        """
        with pytest.raises(PolygonError, match="collapses or inverts"):
            inset(RECT, 5.0)

    def test_shrink_result_must_be_contained_in_the_original(self) -> None:
        from rfw_w.scenes.geometry.polygon import contains_polygon

        assert contains_polygon(RECT, inset(RECT, 0.5))

    def test_rejects_non_finite_offset(self) -> None:
        with pytest.raises(PolygonError, match="finite"):
            offset_outward(RECT, math.inf)

    def test_inset_rejects_a_negative_distance(self) -> None:
        with pytest.raises(PolygonError, match="non-negative"):
            inset(RECT, -0.1)


class TestClearance:
    def test_clear_when_well_inside(self) -> None:
        result = check_clearance(Polygon.rectangle(1.0, 1.0, 2.0, 2.0), region=RECT, margin_m=0.5)
        assert result.ok
        assert result.measured_margin_m >= 0.5

    def test_inside_but_too_close_to_the_boundary(self) -> None:
        result = check_clearance(Polygon.rectangle(0.01, 1.0, 1.0, 2.0), region=RECT, margin_m=0.5)
        assert not result.ok
        assert "not contained" in result.reason

    def test_outside_the_region(self) -> None:
        result = check_clearance(Polygon.rectangle(10.0, 10.0, 11.0, 11.0), region=RECT, margin_m=0.1)
        assert not result.ok
        assert result.measured_margin_m < 0

    def test_object_in_the_notch_is_rejected(self) -> None:
        """The L-shape case: a bounding box would accept this placement."""
        footprint = Polygon.rectangle(0.55, 0.35, 0.95, 0.55)
        box_result = check_clearance(footprint, region=L_DESK.bounding_box(), margin_m=0.0)
        real_result = check_clearance(footprint, region=L_DESK, margin_m=0.0)
        assert box_result.ok
        assert not real_result.ok

    def test_overlap_reports_a_negative_margin(self) -> None:
        obstacle = Polygon.rectangle(1.0, 1.0, 2.0, 2.0)
        result = check_clearance(
            Polygon.rectangle(1.5, 1.5, 2.5, 2.5), obstacles=[obstacle], margin_m=0.0
        )
        assert not result.ok
        assert result.measured_margin_m < 0
        assert "overlaps" in result.reason

    def test_touching_is_not_overlapping(self) -> None:
        # Two objects placed flush against each other are legal and common.
        result = check_clearance(
            Polygon.rectangle(0.0, 0.0, 1.0, 1.0),
            obstacles=[Polygon.rectangle(1.0, 0.0, 2.0, 1.0)],
            margin_m=0.0,
        )
        assert result.ok

    def test_near_miss_is_reported_with_the_shortfall(self) -> None:
        obstacle = Polygon.rectangle(1.0, 1.0, 2.0, 2.0)
        # Sitting 0.01 m to the left of the obstacle: it fits geometrically, but not with the
        # 0.05 m engineering margin.
        result = check_clearance(
            Polygon.rectangle(0.97, 1.0, 0.99, 2.0), obstacles=[obstacle], margin_m=DEFAULT_MARGIN_M
        )
        assert not result.ok
        # It fits geometrically but not with the configured margin, and the result says which.
        assert 0 < result.measured_margin_m < DEFAULT_MARGIN_M

    def test_default_margin_is_the_configured_engineering_value(self) -> None:
        assert DEFAULT_MARGIN_M == 0.05

    def test_rejects_a_negative_margin(self) -> None:
        with pytest.raises(ClearanceError, match="non-negative"):
            check_clearance(Polygon.rectangle(0, 0, 1, 1), margin_m=-0.1)

    def test_result_carries_both_numbers(self) -> None:
        payload = check_clearance(
            Polygon.rectangle(1.0, 1.0, 2.0, 2.0), region=RECT, margin_m=0.1
        ).to_dict()
        assert payload["required_margin_m"] == 0.1
        assert "measured_margin_m" in payload
