# SPDX-License-Identifier: Apache-2.0
"""The wall frame, and what it means to hang a flat piece on it.

`Wall.mount_rotation` exists because a yaw cannot describe a picture. The prepared mesh measures
0.70 x 1.00 x 0.005 -- width, height, thickness -- so a rotation about Z alone leaves the panel
lying flat with its five-millimetre axis pointing at the ceiling, and drives its one-metre height
axis through the wall. That is what happened: the delivered `wall_art` reached 0.13 m past a wall
face, and nothing in the repository measured it.

Nothing here needed a room, an asset library or a GPU: the whole question is which way three axes
end up, which is arithmetic on a `Wall`.
"""

from __future__ import annotations

import math

import pytest

from dataclasses import replace as dataclass_replace

from rfw_w.scenes.contracts.units import Quat, Vec3
from rfw_w.scenes.geometry.polygon import Polygon, contains_polygon
from rfw_w.scenes.solvers.furnish import (
    FurnitureRequest,
    LayoutError,
    Orientation,
    PlacementKind,
    ScenePlan,
    SurfaceRequest,
    Wall,
    within_union,
)

# The east wall of the delivered room, exactly as the manifest records it: the outline edge is at
# x = 4.4524 and the inward normal points at -x.
EAST = Wall(index=1, start=(4.4524, 0.0), end=(4.4524, 2.3848), length_m=2.3848, inward=(-1.0, 0.0))
# The same line walked the other way. A wall's stored direction follows the outline's winding, so
# the rotation must not depend on which way round it was written.
EAST_REVERSED = Wall(index=1, start=(4.4524, 2.3848), end=(4.4524, 0.0), length_m=2.3848,
                     inward=(-1.0, 0.0))
NORTH = Wall(index=0, start=(0.0, 3.5598), end=(4.4524, 3.5598), length_m=4.4524, inward=(0.0, -1.0))


def _determinant(rotation: Quat) -> float:
    rows = rotation.to_matrix3()
    return (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )


def _is_same_rotation(first: Quat, second: Quat) -> bool:
    """Quaternions double-cover SO(3), so q and -q are the same rotation."""
    as_list = first.to_list()
    other = second.to_list()
    return all(abs(a - b) < 1e-9 for a, b in zip(as_list, other)) or all(
        abs(a + b) < 1e-9 for a, b in zip(as_list, other)
    )


class TestMountRotation:
    def test_local_x_lies_along_the_wall(self) -> None:
        image = EAST.mount_rotation().rotate(Vec3(1.0, 0.0, 0.0))
        assert image.z == pytest.approx(0.0, abs=1e-12)
        assert abs(image.x * EAST.inward[0] + image.y * EAST.inward[1]) < 1e-12
        assert image.norm() == pytest.approx(1.0)

    def test_local_y_stands_up(self) -> None:
        image = EAST.mount_rotation().rotate(Vec3(0.0, 1.0, 0.0))
        assert image.is_close(Vec3(0.0, 0.0, 1.0))

    def test_local_z_points_into_the_room(self) -> None:
        """Local +z is the thickness axis, and it must run into the room, not into the wall.

        Which way it points decides where the origin has to be put. With +z inward the origin sits
        on the panel's back face and the panel grows into the room; the other way round hangs the
        piece a thickness out through the wall.
        """
        image = EAST.mount_rotation().rotate(Vec3(0.0, 0.0, 1.0))
        assert image.is_close(Vec3(*EAST.inward, 0.0))

    def test_is_a_proper_rotation_for_either_winding(self) -> None:
        """A reflection is not an orientation, and a quaternion recovered from one means nothing.

        Taking the tangent from the wall's own stored direction gives det = -1 for this room's
        winding, so the two directions are checked rather than assumed.
        """
        for wall in (EAST, EAST_REVERSED, NORTH):
            assert _determinant(wall.mount_rotation()) == pytest.approx(1.0, abs=1e-12)

    def test_is_independent_of_the_wall_direction(self) -> None:
        assert _is_same_rotation(EAST.mount_rotation(), EAST_REVERSED.mount_rotation())

    def test_stands_the_piece_up_on_a_wall_facing_another_way(self) -> None:
        """The same three images on a wall whose inward normal is not axis-aligned with -x."""
        image_x = NORTH.mount_rotation().rotate(Vec3(1.0, 0.0, 0.0))
        image_y = NORTH.mount_rotation().rotate(Vec3(0.0, 1.0, 0.0))
        image_z = NORTH.mount_rotation().rotate(Vec3(0.0, 0.0, 1.0))
        assert image_y.is_close(Vec3(0.0, 0.0, 1.0))
        assert image_z.is_close(Vec3(*NORTH.inward, 0.0))
        # Along the wall, so perpendicular to the inward normal and level.
        assert image_x.z == pytest.approx(0.0, abs=1e-12)
        assert abs(image_x.x * NORTH.inward[0] + image_x.y * NORTH.inward[1]) < 1e-12

    def test_is_not_the_yaw_only_rotation_it_replaced(self) -> None:
        """The regression pin: the old answer was a rotation about Z and nothing else."""
        old = Quat.from_yaw(EAST.yaw_facing_into_room())
        assert not _is_same_rotation(EAST.mount_rotation(), old)
        assert old.rotate(Vec3(0.0, 0.0, 1.0)).is_close(Vec3(0.0, 0.0, 1.0))

    def test_the_matrix_is_orthonormal(self) -> None:
        """Catches a composition that is close but not quite a rotation.

        The three axis images are each checked above on their own; this is the check that they are
        mutually perpendicular and unit, which a subtly wrong composition would not be.
        """
        rows = EAST.mount_rotation().to_matrix3()
        for outer in range(3):
            for inner in range(3):
                dot = sum(rows[outer][k] * rows[inner][k] for k in range(3))
                assert dot == pytest.approx(1.0 if outer == inner else 0.0, abs=1e-12)

    def test_the_along_wall_axis_does_not_follow_the_stored_tangent(self) -> None:
        """_is_independent_of_the_wall_direction, stated as the reason rather than the symptom.

        The stored tangent reverses with the outline's winding. If the rotation were built from
        it, the same wall written the other way would hang the piece against the wall's other
        face -- same silhouette, opposite normal.
        """
        assert EAST.tangent == (0.0, 1.0)
        assert EAST_REVERSED.tangent == (0.0, -1.0)
        expected_along = (-EAST.inward[1], EAST.inward[0])
        for wall in (EAST, EAST_REVERSED):
            image = wall.mount_rotation().rotate(Vec3(1.0, 0.0, 0.0))
            assert image.x == pytest.approx(expected_along[0], abs=1e-12)
            assert image.y == pytest.approx(expected_along[1], abs=1e-12)

class TestPlanValidation:
    """The plan's own invariants, which a repair can break by building a bad request.

    These are here because one of them was silently unenforced for a while: `FurnitureRequest` grew
    a second `__post_init__`, and a later definition of the same method replaces the earlier one, so
    the candidate-index checks were dead code. Nothing caught it -- the tests and the run both used
    non-negative indices -- and only a linter reading the module noticed.
    """

    def test_a_negative_candidate_index_is_refused(self) -> None:
        with pytest.raises(LayoutError, match="cannot use candidate"):
            FurnitureRequest(
                entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
                orientation=Orientation.FRONT_TO_ROOM, asset_index=-1,
            )

    def test_a_negative_candidate_skip_is_refused(self) -> None:
        with pytest.raises(LayoutError, match="cannot skip"):
            FurnitureRequest(
                entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
                orientation=Orientation.FRONT_TO_ROOM, skip_candidates=-1,
            )

    def test_a_surface_object_must_name_its_support(self) -> None:
        """A plan that cannot say where a thing goes cannot be repaired to put it elsewhere."""
        with pytest.raises(LayoutError, match="must name the piece it stands on"):
            SurfaceRequest(entity_id="notebook", synset="notebook.n.01", target_width_m=0.2,
                           support_entity="")

    def test_a_facing_piece_without_a_referent_is_refused(self) -> None:
        """Category A without a referent is category C with extra steps."""
        with pytest.raises(LayoutError, match="faces toward something"):
            FurnitureRequest(
                entity_id="chair", synset="chair.n.01", kind=PlacementKind.BESIDE,
                orientation=Orientation.FACES_TOWARD,
            )

    def test_the_plan_refuses_a_duplicate_entity(self) -> None:
        twice = (
            FurnitureRequest(entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
                             orientation=Orientation.FRONT_TO_ROOM),
            FurnitureRequest(entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
                             orientation=Orientation.FRONT_TO_ROOM),
        )
        with pytest.raises(LayoutError, match="more than once"):
            ScenePlan(furniture=twice)

    def test_a_negative_distance_in_the_plan_is_refused(self) -> None:
        with pytest.raises(LayoutError, match="non-negative finite distance"):
            ScenePlan(wall_face_offset_m=-0.1)

    def test_the_plan_serialises_every_field_a_repair_can_change(self) -> None:
        """A field left out of ``to_dict`` is a field a repair can change without changing the
        scene version -- and the loop would then commit it and leave every gate's evidence looking
        current for a scene it never measured."""
        payload = ScenePlan().to_dict()
        for name in ("wall_face_offset_m", "object_gap_m", "mount_height_m", "wall_hang_gap_m",
                     "ceiling_clearance_m"):
            assert name in payload
        request = FurnitureRequest(
            entity_id="chair", synset="chair.n.01", kind=PlacementKind.BESIDE,
            orientation=Orientation.FACES_TOWARD, faces_entity="desk", skip_candidates=2,
        )
        assert request.to_dict()["skip_candidates"] == 2

    def test_replacing_a_request_keeps_everything_else(self) -> None:
        """What an L0 resample does: change one piece's search, leave the plan otherwise alone."""
        plan = ScenePlan(furniture=(
            FurnitureRequest(entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
                             orientation=Orientation.FRONT_TO_ROOM),
        ))
        updated = plan.replaced_furniture(
            dataclass_replace(plan.furniture[0], skip_candidates=3)
        )
        assert updated.furniture_for("desk").skip_candidates == 3
        assert plan.furniture_for("desk").skip_candidates == 0, "the original must be unchanged"

    def test_removing_an_entity_that_is_not_there_is_refused(self) -> None:
        with pytest.raises(LayoutError, match="not in the plan"):
            ScenePlan().without("ghost")


class TestWithinUnion:
    """A published zone is a set of pieces, and a footprint may straddle two of them.

    Every reservation is subtracted with `subtract_polygons`, which returns disjoint convex parts
    that *share edges*. The per-piece test that preceded this asked `contains_polygon(piece,
    candidate)` and therefore refused anything lying across a seam -- which is most of what is left
    once the task's floor is cut out of the room. Measured on the office, a 0.95 x 0.50 cabinet has
    11 legal positions on its storage wall by union and **0** by single piece.
    """

    # Two unit squares meeting along x = 1, the way `subtract_polygons` leaves a split strip.
    LEFT = Polygon.rectangle(0.0, 0.0, 1.0, 1.0)
    RIGHT = Polygon.rectangle(1.0, 0.0, 2.0, 1.0)

    def test_a_footprint_across_the_seam_is_inside_the_union(self) -> None:
        across = Polygon.rectangle(0.75, 0.25, 1.25, 0.75)
        assert not any(contains_polygon(piece, across) for piece in (self.LEFT, self.RIGHT))
        assert within_union(across, (self.LEFT, self.RIGHT))

    def test_the_union_gate_still_refuses_outside_it(self) -> None:
        """The negative test. Widening the permission must not widen it past the region."""
        outside = Polygon.rectangle(1.75, 0.25, 2.25, 0.75)
        assert not within_union(outside, (self.LEFT, self.RIGHT))

    def test_a_footprint_wholly_inside_one_piece_takes_the_cheap_path(self) -> None:
        inside = Polygon.rectangle(0.25, 0.25, 0.75, 0.75)
        assert within_union(inside, (self.LEFT, self.RIGHT))

    def test_no_regions_means_no_restriction(self) -> None:
        """The callers pass `allow or (room,)`, so an empty sequence must not read as "nothing fits"."""
        assert within_union(Polygon.rectangle(0.0, 0.0, 9.0, 9.0), ())

    def test_overlapping_regions_do_not_count_a_square_metre_twice(self) -> None:
        """Why the fallback subtracts instead of summing intersections.

        Two regions that overlap each other would let a sum of intersections exceed the candidate's
        own area, so a footprint half outside the union could pass. The subtraction cannot.
        """
        overlapping = (self.LEFT, Polygon.rectangle(0.5, 0.0, 2.0, 1.0))
        mostly_outside = Polygon.rectangle(1.5, 0.25, 2.5, 0.75)
        assert not within_union(mostly_outside, overlapping)
