# SPDX-License-Identifier: Apache-2.0
"""Unit and frame conventions.

The transform tests matter more than they look. A transposed or inverted transform produces
a scene that renders fine and is subtly wrong, so the conventions are pinned down here with
round trips and with a hand-computed example rather than only self-consistency checks.
"""

from __future__ import annotations

import math

import pytest

from rfw_w.scenes.contracts.units import (
    DegenerateTransformError,
    NonFiniteValueError,
    Quat,
    QuaternionNotNormalisedError,
    Transform,
    UnitsError,
    Vec3,
    quaternion_from_matrix3,
    rotation_distance_radians,
    row_bounds,
)


class TestVec3:
    def test_rejects_non_finite(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(NonFiniteValueError):
                Vec3(bad, 0.0, 0.0)

    @pytest.mark.parametrize("payload", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], "1,2,3", 5.0])
    def test_from_sequence_rejects_wrong_shape(self, payload: object) -> None:
        with pytest.raises(UnitsError):
            Vec3.from_sequence(payload)  # type: ignore[arg-type]

    def test_arithmetic_and_norm(self) -> None:
        assert (Vec3(1, 2, 3) + Vec3(1, 1, 1)).to_list() == [2.0, 3.0, 4.0]
        assert (Vec3(1, 2, 3) - Vec3(1, 1, 1)).to_list() == [0.0, 1.0, 2.0]
        assert Vec3(3, 4, 0).norm() == pytest.approx(5.0)
        assert Vec3(3, 4, 0).normalized().norm() == pytest.approx(1.0)

    def test_cross_product_is_right_handed(self) -> None:
        # +X cross +Y = +Z is what makes the frame right-handed; if this ever flips, every
        # yaw and every up-axis assumption in the project is wrong.
        assert Vec3(1, 0, 0).cross(Vec3(0, 1, 0)).to_list() == [0.0, 0.0, 1.0]

    def test_zero_vector_has_no_direction(self) -> None:
        with pytest.raises(DegenerateTransformError):
            Vec3.origin().normalized()


class TestQuat:
    def test_rejects_unnormalised(self) -> None:
        with pytest.raises(QuaternionNotNormalisedError):
            Quat(2.0, 0.0, 0.0, 0.0)

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(NonFiniteValueError):
            Quat(float("nan"), 0.0, 0.0, 0.0)

    def test_json_order_is_wxyz(self) -> None:
        # Parsing "1,2,3,4" must read w=1, x=2, y=3, z=4. If the order were xyzw, a
        # 90-degree yaw would come back as something else entirely.
        quat = Quat.from_sequence([1.0, 0.0, 0.0, 0.0])
        assert (quat.w, quat.x, quat.y, quat.z) == (1.0, 0.0, 0.0, 0.0)
        assert quat.to_list() == [1.0, 0.0, 0.0, 0.0]

    def test_from_sequence_rejects_wrong_length(self) -> None:
        with pytest.raises(UnitsError):
            Quat.from_sequence([1.0, 0.0, 0.0])

    def test_yaw_rotates_x_toward_y(self) -> None:
        # A +90 degree yaw about +Z takes +X to +Y.
        rotated = Quat.from_yaw(math.pi / 2).rotate(Vec3(1, 0, 0))
        assert rotated.x == pytest.approx(0.0, abs=1e-12)
        assert rotated.y == pytest.approx(1.0, abs=1e-12)

    def test_q_and_minus_q_are_the_same_rotation(self) -> None:
        quat = Quat.from_yaw(0.7)
        negated = Quat(-quat.w, -quat.x, -quat.y, -quat.z)
        assert quat.is_same_rotation(negated)
        assert quat != negated  # storage differs; orientation does not
        assert rotation_distance_radians(quat, negated) == pytest.approx(0.0, abs=1e-9)

    def test_rotation_distance_is_bounded(self) -> None:
        left = Quat.identity()
        right = Quat.from_yaw(math.pi)
        assert rotation_distance_radians(left, right) == pytest.approx(math.pi, abs=1e-9)


class TestTransform:
    def test_applies_rotation_then_translation(self) -> None:
        # Hand-computed: yaw 90 about Z sends (1,0,0) to (0,1,0); translate by (10,0,0)
        # gives (10,1,0). Column-vector order is what makes this the answer.
        transform = Transform(
            translation=Vec3(10, 0, 0),
            rotation=Quat.from_yaw(math.pi / 2),
            parent_frame="world",
            child_frame="child",
        )
        result = transform.apply(Vec3(1, 0, 0))
        assert result.x == pytest.approx(10.0, abs=1e-9)
        assert result.y == pytest.approx(1.0, abs=1e-9)

    def test_frames_are_required_and_non_empty(self) -> None:
        with pytest.raises(UnitsError):
            Transform(Vec3.origin(), Quat.identity(), "", "child")

    def test_inverse_round_trips(self) -> None:
        transform = Transform(Vec3(1, 2, 3), Quat.from_yaw(0.4), "world", "child")
        point = Vec3(0.5, -0.25, 1.5)
        assert transform.inverse().apply(transform.apply(point)).is_close(point, tolerance=1e-12)

    def test_compose_applies_inner_first(self) -> None:
        outer = Transform(Vec3(1, 0, 0), Quat.identity(), "world", "middle")
        inner = Transform(Vec3(0, 2, 0), Quat.identity(), "middle", "leaf")
        composed = outer.compose(inner)
        assert composed.parent_frame == "world"
        assert composed.child_frame == "leaf"
        assert composed.apply(Vec3.origin()).is_close(Vec3(1, 2, 0))

    def test_compose_refuses_mismatched_frames(self) -> None:
        outer = Transform(Vec3.origin(), Quat.identity(), "world", "middle")
        inner = Transform(Vec3.origin(), Quat.identity(), "somewhere_else", "leaf")
        with pytest.raises(UnitsError, match="does not match"):
            outer.compose(inner)

    def test_usd_row_convention_round_trips(self) -> None:
        transform = Transform(Vec3(0.25, -1.5, 2.0), Quat.from_yaw(1.1), "world", "child")
        restored = Transform.from_usd_matrix_rows(
            transform.to_usd_matrix_rows(), "world", "child"
        )
        assert restored.is_close(transform, tolerance=1e-9)

    def test_usd_matrix_puts_translation_in_the_last_row(self) -> None:
        # USD acts on row vectors, so a translation lives in the last row, not the last
        # column. Writing it in the column position is the classic silent transpose.
        rows = Transform(Vec3(1, 2, 3), Quat.identity(), "world", "child").to_usd_matrix_rows()
        assert rows[3] == [1.0, 2.0, 3.0, 1.0]
        assert [row[3] for row in rows[:3]] == [0.0, 0.0, 0.0]


class TestQuaternionRecovery:
    def test_round_trips_near_half_turn(self) -> None:
        # Near 180 degrees is where the naive single-branch formula loses precision, so it
        # is the case worth pinning.
        for angle in (0.0, 0.3, math.pi / 2, math.pi - 1e-6, math.pi - 1e-9):
            quat = Quat.from_yaw(angle)
            recovered = quaternion_from_matrix3(quat.to_matrix3())
            assert rotation_distance_radians(quat, recovered) == pytest.approx(0.0, abs=1e-6)

    def test_rejects_non_3x3(self) -> None:
        with pytest.raises(UnitsError):
            quaternion_from_matrix3([[1.0, 0.0], [0.0, 1.0]])


class TestBounds:
    def test_bounds_of_a_point_set(self) -> None:
        minimum, maximum = row_bounds([Vec3(1, 2, 3), Vec3(-1, 5, 0)])
        assert minimum.to_list() == [-1.0, 2.0, 0.0]
        assert maximum.to_list() == [1.0, 5.0, 3.0]

    def test_empty_set_is_rejected(self) -> None:
        # An empty box would compare equal to any other empty box, and a clearance check
        # reading that would conclude "nothing in the way".
        with pytest.raises(UnitsError, match="empty"):
            row_bounds([])
