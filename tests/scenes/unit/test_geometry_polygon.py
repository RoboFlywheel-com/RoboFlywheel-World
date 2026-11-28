# SPDX-License-Identifier: Apache-2.0
"""Polygon geometry.

The cases that matter are the ones where a bounding box would give the wrong answer: an
L-shaped room, a notched desk, and a region that contains a polygon's vertices while the
polygon still crosses its boundary.
"""

from __future__ import annotations

import math

import pytest

from rfw_w.scenes.geometry.polygon import (
    Polygon,
    PolygonError,
    area,
    centroid,
    contains_point,
    contains_polygon,
    convex_subtract,
    distance_to_boundary,
    inset,
    intersect_polygons,
    offset_outward,
    overlaps,
    point_inside_toward,
    subtract_polygons,
    validate_simple,
)

# An L-shaped room: a 4x3 rectangle with a 1x1 notch removed from the top-right.
L_ROOM = Polygon.from_sequence(
    [(0, 0), (4, 0), (4, 2), (3, 2), (3, 3), (0, 3)]
)
RECT = Polygon.rectangle(0, 0, 4, 3)


class TestConstruction:
    def test_needs_three_vertices(self) -> None:
        with pytest.raises(PolygonError, match="at least 3"):
            Polygon.from_sequence([(0, 0), (1, 1)])

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(PolygonError, match="finite"):
            Polygon.from_sequence([(0, 0), (1, 1), (math.nan, 2)])

    def test_rejects_degenerate_rectangle(self) -> None:
        with pytest.raises(PolygonError, match="degenerate"):
            Polygon.rectangle(0, 0, 0, 1)


class TestArea:
    def test_rectangle_area(self) -> None:
        assert RECT.area == pytest.approx(12.0)

    def test_winding_does_not_change_area(self) -> None:
        reversed_room = Polygon(tuple(reversed(L_ROOM.vertices)))
        assert reversed_room.area == pytest.approx(L_ROOM.area)

    def test_l_shape_area_excludes_the_notch(self) -> None:
        # 4x3 minus the 1x1 notch. A bounding box would say 12.
        assert L_ROOM.area == pytest.approx(11.0)

    def test_area_ratio_exposes_a_notched_shape(self) -> None:
        """The check that catches a notched room passed off as its bounding box."""
        assert RECT.area_ratio_to_bounding_box() == pytest.approx(1.0)
        ratio = L_ROOM.area_ratio_to_bounding_box()
        assert ratio < 0.95
        assert ratio == pytest.approx(11.0 / 12.0)

    def test_rectangle_detection(self) -> None:
        assert RECT.is_rectangle()
        assert not L_ROOM.is_rectangle()

    def test_free_area_function_matches(self) -> None:
        assert area(L_ROOM) == pytest.approx(L_ROOM.area)


class TestContainment:
    def test_point_inside(self) -> None:
        assert contains_point(RECT, (2.0, 1.5))

    def test_point_outside(self) -> None:
        assert not contains_point(RECT, (5.0, 1.5))

    def test_point_in_the_notch_is_outside_the_real_room(self) -> None:
        """The AABB false pass this whole module exists to prevent.

        (3.5, 2.5) is inside the bounding box and outside the actual room, so a containment
        test built on the box would place an object floating in the notch and report success.
        """
        assert contains_point(L_ROOM.bounding_box(), (3.5, 2.5))
        assert not contains_point(L_ROOM, (3.5, 2.5))

    def test_boundary_counts_as_inside(self) -> None:
        # A support surface's edge is still support; margins are applied explicitly instead.
        assert contains_point(RECT, (0.0, 1.5))
        assert contains_point(RECT, (2.0, 0.0))

    def test_polygon_wholly_inside(self) -> None:
        assert contains_polygon(RECT, Polygon.rectangle(1, 1, 2, 2))

    def test_polygon_crossing_the_notch_is_not_contained(self) -> None:
        assert not contains_polygon(L_ROOM, Polygon.rectangle(2.5, 1.5, 3.5, 2.5))

    def test_vertices_outside_but_edges_crossing_is_not_contained(self) -> None:
        """Testing vertices alone would call this contained."""
        c_shape = Polygon.from_sequence(
            [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (3, 2), (3, 3), (0, 3)]
        )
        bar = Polygon.rectangle(0.5, 1.2, 2.5, 1.8)
        crossing = not contains_polygon(c_shape, bar)
        assert crossing

    def test_distance_to_boundary(self) -> None:
        assert distance_to_boundary(RECT, (2.0, 1.5)) == pytest.approx(1.5)


class TestValidity:
    def test_accepts_a_simple_polygon(self) -> None:
        validate_simple(L_ROOM)

    def test_rejects_a_bow_tie(self) -> None:
        """A self-intersecting outline has a meaningless shoelace area, and every containment
        test built on it inherits the error."""
        bow_tie = Polygon.from_sequence([(0, 0), (2, 2), (2, 0), (0, 2)])
        with pytest.raises(PolygonError, match="self-intersecting"):
            validate_simple(bow_tie)

    def test_rejects_a_zero_length_edge(self) -> None:
        path = Polygon.from_sequence([(0, 0), (1, 0), (1, 0), (1, 1)])
        with pytest.raises(PolygonError, match="zero-length"):
            validate_simple(path)


class TestCentroid:
    def test_rectangle_centroid(self) -> None:
        assert centroid(RECT) == pytest.approx((2.0, 1.5))

    def test_l_shape_centroid_is_pulled_away_from_the_notch(self) -> None:
        # The notch is top-right, so the centroid sits below and left of the rectangle's.
        cx, cy = centroid(L_ROOM)
        assert cx < 2.0
        assert cy < 1.5


def _pieces_area(pieces) -> float:
    return sum(piece.area for piece in pieces)


class TestIntersectPolygons:
    """Exact area of a shared region, including where one side is notched.

    A boolean ``overlaps`` cannot answer "how much", and a bounding-box stand-in answers it
    wrongly, so these tests check the number rather than the fact of intersection.
    """

    def test_two_overlapping_rectangles(self) -> None:
        first = Polygon.rectangle(0, 0, 2, 2)
        second = Polygon.rectangle(1, 1, 3, 3)
        assert _pieces_area(intersect_polygons(first, second)) == pytest.approx(1.0)

    def test_disjoint_rectangles_share_nothing(self) -> None:
        first = Polygon.rectangle(0, 0, 1, 1)
        second = Polygon.rectangle(5, 5, 6, 6)
        assert intersect_polygons(first, second) == ()

    def test_touching_edges_share_no_area(self) -> None:
        first = Polygon.rectangle(0, 0, 1, 1)
        second = Polygon.rectangle(1, 0, 2, 1)
        assert _pieces_area(intersect_polygons(first, second)) == pytest.approx(0.0, abs=1e-9)

    def test_containment_returns_the_inner_polygon(self) -> None:
        outer = Polygon.rectangle(0, 0, 4, 4)
        inner = Polygon.rectangle(1, 1, 2, 3)
        assert _pieces_area(intersect_polygons(outer, inner)) == pytest.approx(inner.area)

    def test_notched_polygon_against_a_rectangle_covering_the_notch(self) -> None:
        """The case Sutherland-Hodgman gets wrong on its own: clipping the L against its own
        notch line would subtract the region beyond it and understate the shared area."""
        box = Polygon.rectangle(3, 2, 4, 3)  # the missing 1x1 notch of the L, exactly
        assert _pieces_area(intersect_polygons(L_ROOM, box)) == pytest.approx(0.0, abs=1e-9)

    def test_notched_polygon_overlap_is_exact(self) -> None:
        # A 2x2 box straddling the notch corner: 3 of its 4 m2 are real floor.
        box = Polygon.rectangle(2, 1, 4, 3)
        shared = intersect_polygons(L_ROOM, box)
        assert _pieces_area(shared) == pytest.approx(3.0)
        # The pieces partition the region: no two of them share area.
        for index, first in enumerate(shared):
            for second in shared[index + 1:]:
                assert _pieces_area(intersect_polygons(first, second)) == pytest.approx(
                    0.0, abs=1e-9
                )

    def test_is_symmetric(self) -> None:
        box = Polygon.rectangle(2, 1, 4, 3)
        assert _pieces_area(intersect_polygons(L_ROOM, box)) == pytest.approx(
            _pieces_area(intersect_polygons(box, L_ROOM))
        )

    def test_degenerate_input_shares_nothing(self) -> None:
        sliver = Polygon.from_sequence([(0, 0), (2, 0), (4, 0)])
        assert intersect_polygons(sliver, RECT) == ()


class TestSubtractPolygons:
    def test_removing_a_subset_leaves_the_remainder(self) -> None:
        outer = Polygon.rectangle(0, 0, 4, 4)
        inner = Polygon.rectangle(1, 1, 2, 3)
        remainder = subtract_polygons(outer, inner)
        assert _pieces_area(remainder) == pytest.approx(outer.area - inner.area)

    def test_disjoint_blocker_changes_nothing(self) -> None:
        first = Polygon.rectangle(0, 0, 1, 1)
        second = Polygon.rectangle(5, 5, 6, 6)
        assert _pieces_area(subtract_polygons(first, second)) == pytest.approx(1.0)

    def test_removing_everything_leaves_nothing(self) -> None:
        assert _pieces_area(subtract_polygons(RECT, RECT)) == pytest.approx(0.0, abs=1e-9)

    def test_notched_blocker_removes_only_the_real_part(self) -> None:
        """The L's notch is empty floor, so blocking the L's bounding box must not remove it."""
        box = Polygon.rectangle(0, 0, 4, 3)
        remaining = subtract_polygons(box, L_ROOM)
        assert _pieces_area(remaining) == pytest.approx(1.0)  # the notch alone
        # And what survives really is the notch.
        assert all(piece.area > 0 for piece in remaining)

    def test_notched_subject_keeps_its_notch(self) -> None:
        # Blocking the L's right half must leave the notch empty and the left half intact.
        right = Polygon.rectangle(2, 0, 4, 3)
        remaining = subtract_polygons(L_ROOM, right)
        assert _pieces_area(remaining) == pytest.approx(6.0)  # 2x3 of real floor

    def test_pieces_are_disjoint(self) -> None:
        box = Polygon.rectangle(0, 0, 4, 3)
        for piece in subtract_polygons(box, L_ROOM) + subtract_polygons(L_ROOM, box):
            assert piece.area > 0


class TestConvexSubtract:
    def test_counts_match_the_difference_without_a_triangulation(self) -> None:
        subject = Polygon.rectangle(0, 0, 10, 1)
        blocker = Polygon.rectangle(4, 0, 6, 1)
        assert _pieces_area(convex_subtract(subject, blocker)) == pytest.approx(8.0)

    def test_blocker_outside_leaves_the_subject_whole(self) -> None:
        subject = Polygon.rectangle(0, 0, 1, 1)
        blocker = Polygon.rectangle(5, 5, 6, 6)
        assert _pieces_area(convex_subtract(subject, blocker)) == pytest.approx(1.0)


class TestVantagePoint:
    """Where a camera may stand.

    The failure these pin is a real one. A camera aimed at a target near a wall has its requested
    direction inside that wall, and the search then relaxes `margin` -- 0.30, 0.15, 0.075 -- until
    *something* qualifies. Measured on a delivered scene: an eye 0.076 m from the boundary, which is
    past the 0.1 m wall thickness, so the camera stood in the wall and the view rendered one uniform
    surface. A vantage point that clear of nothing is not a vantage point, and the answer is to say
    so rather than to relax until a number fits.
    """

    def test_the_requested_direction_wins_when_it_has_room(self) -> None:
        eye = point_inside_toward(RECT, (2.0, 1.5), (3.0, 1.5), margin=0.3)
        assert eye is not None and eye[0] > 2.0

    def test_the_other_side_is_used_when_it_gives_more_room(self) -> None:
        # An object 0.5 m from the east wall can only be photographed from the west.
        eye = point_inside_toward(RECT, (3.5, 1.5), (4.5, 1.5), margin=0.3)
        assert eye is not None and eye[0] < 3.5

    def test_a_returned_point_always_clears_the_floor(self) -> None:
        # The property the caller relies on: whatever comes back is somewhere a camera can stand.
        narrow = Polygon.rectangle(0, 0, 4, 0.4)
        for floor in (0.0, 0.1, 0.2):
            eye = point_inside_toward(
                narrow, (2.0, 0.2), (5.0, 0.2), margin=0.35, minimum_clearance_m=floor
            )
            assert eye is not None, f"a 0.4 m corridor has room for {floor} m of clearance"
            assert distance_to_boundary(narrow, eye) >= floor - 1e-9

    def test_none_rather_than_a_point_in_the_wall(self) -> None:
        """The whole point. A 0.4 m corridor cannot hold a camera 0.25 m clear of both sides, and
        the previous version answered that by relaxing the margin until a point on the wall fit."""
        narrow = Polygon.rectangle(0, 0, 4, 0.4)
        assert point_inside_toward(
            narrow, (2.0, 0.2), (5.0, 0.2), margin=0.35, minimum_clearance_m=0.25
        ) is None

    def test_solving_against_the_inner_face_keeps_the_eye_out_of_the_wall(self) -> None:
        """The camera set's own bug, as a property: the outline is the wall's *outer* face.

        Solved against the outline, a point 0.075 m inside it is a legal answer and is inside the
        wall body. Solved against the inner face, every answer is in the room's air -- which is what
        `_inner_floor` is, and what placement and every geometry gate already measure against.
        """
        wall_thickness = 0.1
        inner = inset(RECT, wall_thickness)
        target = (0.15, 1.5)          # a piece stood against the west wall
        eye = point_inside_toward(inner, target, (-0.35, 1.5), margin=0.25,
                                  minimum_clearance_m=0.25)
        assert eye is not None
        assert contains_point(inner, eye)
        # And therefore clear of the wall solid by the thickness, not merely inside the outline.
        assert distance_to_boundary(RECT, eye) >= wall_thickness
