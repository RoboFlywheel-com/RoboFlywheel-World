# SPDX-License-Identifier: Apache-2.0
"""Support resolution and the ON predicate's basis.

The critical negative is the one the runtime probe demonstrated: an object that falls to the
floor settles stably, so a gate checking only stability reports success. Distinguishing "on the
right surface" from "stable somewhere" is the whole job of this module.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.asset import AffordanceStatus
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.support import (
    DEFAULT_CONTACT_TOLERANCE_M,
    DEFAULT_MIN_COVERAGE,
    PlacedInstance,
    SupportError,
    Surface,
    assert_not_on_floor_when_support_expected,
    polygon_overlap_area,
    resolve_support,
)

FLOOR = Surface(
    surface_id="floor",
    owner_instance=None,
    polygon=Polygon.rectangle(0.0, 0.0, 5.0, 4.0),
    plane_z_m=0.0,
    provenance=AffordanceStatus.MEASURED,
)
# A desk top 1.2 x 0.6 m at 0.75 m, with the near-right corner notched out so a footprint can
# overlap the bounding box while touching nothing.
DESK_TOP = Surface(
    surface_id="desk:top",
    owner_instance="desk",
    polygon=Polygon.from_sequence(
        [(0.0, 0.0), (1.2, 0.0), (1.2, 0.4), (0.9, 0.4), (0.9, 0.6), (0.0, 0.6)]
    ),
    plane_z_m=0.75,
    provenance=AffordanceStatus.MEASURED,
)


def placed(x: float, y: float, *, z: float = 0.75, size: float = 0.1, dynamic: bool = True):
    return PlacedInstance(
        instance_id="cup",
        footprint=Polygon.rectangle(x - size / 2, y - size / 2, x + size / 2, y + size / 2),
        base_z_m=z,
        is_dynamic=dynamic,
    )


class TestOverlapArea:
    def test_identical_polygons(self) -> None:
        square = Polygon.rectangle(0, 0, 1, 1)
        assert polygon_overlap_area(square, square) == pytest.approx(1.0)

    def test_disjoint_polygons(self) -> None:
        assert polygon_overlap_area(
            Polygon.rectangle(0, 0, 1, 1), Polygon.rectangle(2, 2, 3, 3)
        ) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        assert polygon_overlap_area(
            Polygon.rectangle(0, 0, 1, 1), Polygon.rectangle(0.5, 0, 1.5, 1)
        ) == pytest.approx(0.5)

    def test_containment(self) -> None:
        assert polygon_overlap_area(
            Polygon.rectangle(0, 0, 2, 2), Polygon.rectangle(0.5, 0.5, 1.0, 1.0)
        ) == pytest.approx(0.25)

    def test_notch_means_no_overlap_despite_the_bounding_box(self) -> None:
        """A footprint in the desk's notch overlaps its bounding box and none of the desk."""
        notch_footprint = Polygon.rectangle(0.95, 0.45, 1.05, 0.55)
        assert polygon_overlap_area(DESK_TOP.polygon, notch_footprint) == pytest.approx(0.0)
        assert polygon_overlap_area(DESK_TOP.polygon.bounding_box(), notch_footprint) > 0.0

    def test_is_symmetric(self) -> None:
        first = Polygon.rectangle(0, 0, 1, 1)
        second = Polygon.rectangle(0.3, 0.2, 1.4, 1.3)
        assert polygon_overlap_area(first, second) == pytest.approx(polygon_overlap_area(second, first))


class TestResolution:
    def test_finds_the_desk_when_resting_on_it(self) -> None:
        result = resolve_support(placed(0.4, 0.3), [FLOOR, DESK_TOP])
        assert result.resolved
        assert result.witness.support_owner == "desk"
        assert result.witness.covered

    def test_finds_the_floor_when_on_the_floor(self) -> None:
        result = resolve_support(placed(3.0, 3.0, z=0.0), [FLOOR, DESK_TOP])
        assert result.resolved
        assert result.witness.support_owner is None
        assert result.witness.surface_id == "floor"

    def test_floating_object_has_no_support(self) -> None:
        result = resolve_support(placed(0.4, 0.3, z=1.4), [FLOOR, DESK_TOP])
        assert not result.resolved
        assert result.failure_code == "SUPPORT_NOT_FOUND"

    def test_object_in_the_notch_falls_through_to_the_floor(self) -> None:
        """Over the bounding box, not over the desk -- so the floor is what is underneath."""
        result = resolve_support(
            placed(0.98, 0.5, z=0.0), [FLOOR, DESK_TOP]
        )
        assert result.witness.support_owner is None

    def test_partial_overhang_still_counts(self) -> None:
        # Half the footprint off the desk edge is normal and still support. Centred on the edge
        # at x = 1.2, so exactly half of the 0.1 m footprint is over the desk.
        result = resolve_support(placed(1.2, 0.2), [FLOOR, DESK_TOP])
        assert result.witness.support_owner == "desk"
        assert 0.45 < result.witness.coverage_ratio < 0.55

    def test_marginal_coverage_is_rejected(self) -> None:
        # Touching the very corner of the tabletop is not being on it.
        result = resolve_support(placed(1.28, 0.2, z=0.75), [DESK_TOP])
        assert not result.resolved

    def test_highest_covering_surface_wins(self) -> None:
        higher = Surface(
            surface_id="shelf", owner_instance="shelf",
            polygon=Polygon.rectangle(0.0, 0.0, 1.0, 1.0), plane_z_m=1.2,
            provenance=AffordanceStatus.MEASURED,
        )
        result = resolve_support(placed(0.4, 0.3, z=1.2), [FLOOR, DESK_TOP, higher])
        assert result.witness.support_owner == "shelf"

    def test_tolerance_is_bounded_both_ways(self) -> None:
        below = resolve_support(placed(0.4, 0.3, z=0.75 - DEFAULT_CONTACT_TOLERANCE_M / 2), [DESK_TOP])
        assert below.resolved
        too_low = resolve_support(placed(0.4, 0.3, z=0.75 - DEFAULT_CONTACT_TOLERANCE_M * 10), [DESK_TOP])
        assert not too_low.resolved

    def test_rejects_a_footprint_with_no_area(self) -> None:
        # A degenerate rectangle cannot even be constructed, so the zero-area case is built from
        # collinear vertices, which is how a real footprint degenerates.
        collinear = Polygon.from_sequence([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        with pytest.raises(SupportError, match="no area"):
            resolve_support(PlacedInstance("cup", collinear, 0.0, True), [FLOOR])

    def test_rejects_invalid_coverage_threshold(self) -> None:
        with pytest.raises(SupportError, match="min_coverage"):
            resolve_support(placed(0.4, 0.3), [DESK_TOP], min_coverage=0.0)

    def test_rejects_negative_tolerance(self) -> None:
        with pytest.raises(SupportError, match="tolerance"):
            resolve_support(placed(0.4, 0.3), [DESK_TOP], contact_tolerance_m=-0.01)

    def test_default_coverage_leaves_room_for_overhang(self) -> None:
        assert 0.0 < DEFAULT_MIN_COVERAGE < 1.0

    def test_witness_serialises_its_measurements(self) -> None:
        payload = resolve_support(placed(0.4, 0.3), [DESK_TOP]).to_dict()
        witness = payload["witness"]
        assert witness["coverage_ratio"] > 0
        assert "base_gap_m" in witness
        assert witness["tolerance_m"] == DEFAULT_CONTACT_TOLERANCE_M


class TestFloorIsNotTheDesk:
    def test_stable_on_the_floor_fails_a_required_desk_relation(self) -> None:
        """The runtime probe's negative case, as a predicate.

        The cube landed on the floor and stayed still. Stable, and not ON the support.
        """
        resolution = resolve_support(placed(3.0, 3.0, z=0.0), [FLOOR, DESK_TOP])
        assert resolution.resolved, "it did rest on something"
        guarded = assert_not_on_floor_when_support_expected(resolution, expected_owner="desk")
        assert guarded.failure_code == "FUNCTION_PREDICATE_FAILED"
        assert "floor" in guarded.detail

    def test_on_the_desk_passes(self) -> None:
        resolution = resolve_support(placed(0.4, 0.3), [FLOOR, DESK_TOP])
        guarded = assert_not_on_floor_when_support_expected(resolution, expected_owner="desk")
        assert guarded.failure_code is None

    def test_no_support_is_left_as_support_not_found(self) -> None:
        # A different fault from "on the wrong thing": nothing is underneath at all.
        resolution = resolve_support(placed(0.4, 0.3, z=2.0), [FLOOR, DESK_TOP])
        guarded = assert_not_on_floor_when_support_expected(resolution, expected_owner="desk")
        assert guarded.failure_code == "SUPPORT_NOT_FOUND"

    def test_the_two_faults_are_distinguishable(self) -> None:
        on_floor = assert_not_on_floor_when_support_expected(
            resolve_support(placed(3.0, 3.0, z=0.0), [FLOOR, DESK_TOP]), expected_owner="desk"
        )
        nowhere = assert_not_on_floor_when_support_expected(
            resolve_support(placed(3.0, 3.0, z=9.0), [FLOOR, DESK_TOP]), expected_owner="desk"
        )
        assert on_floor.failure_code != nowhere.failure_code


class TestSurfaces:
    def test_floor_is_identified_as_the_floor(self) -> None:
        assert FLOOR.is_floor
        assert not DESK_TOP.is_floor

    def test_surface_reports_its_area_and_centre(self) -> None:
        assert DESK_TOP.area_m2 == pytest.approx(1.2 * 0.6 - 0.3 * 0.2)
        x, y = DESK_TOP.center_xy
        assert 0.0 < x < 1.2 and 0.0 < y < 0.6

    def test_surface_requires_an_id(self) -> None:
        with pytest.raises(SupportError, match="must have an id"):
            Surface("", "desk", Polygon.rectangle(0, 0, 1, 1), 0.5)

    def test_surface_serialises(self) -> None:
        payload = DESK_TOP.to_dict()
        assert payload["owner_instance"] == "desk"
        assert payload["is_floor"] is False
        assert payload["provenance"] == "measured"
