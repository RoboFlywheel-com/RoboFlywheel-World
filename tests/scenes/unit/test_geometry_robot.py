# SPDX-License-Identifier: Apache-2.0
"""The robot as a body, and the reach proxy built on it.

The gap these close: `TaskSpec.robot_profile` was required, validated and read by nothing, so the
question "can the robot get to the piece the task is about" had no value in the system that could
express it. Every one of the twenty-six gates passed while a visual critic found a real instance of
the problem by looking at a picture.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.geometry.robot import (
    ROBOT_PROFILES,
    RobotError,
    RobotProfile,
    profile_for,
    standoff_distances,
    standoffs_for,
)

PANDA = profile_for("franka_panda")
ROOM = Polygon.rectangle(0.0, 0.0, 5.0, 4.0)


def count_directions(standoffs, target) -> int:
    import math

    return len({
        round(math.atan2(item.centre[1] - target[1], item.centre[0] - target[0]), 3)
        for item in standoffs
    })


class TestTheProfile:
    def test_the_experiments_robot_has_numbers(self) -> None:
        assert PANDA.name == "franka_panda"
        assert PANDA.reach_m > 0
        assert PANDA.approach_clearance_m > 0

    def test_an_unknown_robot_is_refused_by_name(self) -> None:
        """Not defaulted to another arm's reach: that would be a gate measuring a different
        machine."""
        with pytest.raises(RobotError, match="UNKNOWN_ROBOT_PROFILE"):
            profile_for("ur5")

    def test_a_profile_needs_positive_numbers(self) -> None:
        """A zero footprint is an arm that fits anywhere and a zero reach one that reaches
        nothing; both would make the reach proxy say something false."""
        with pytest.raises(RobotError, match="footprint_m must be positive"):
            RobotProfile(name="x", footprint_m=0.0, reach_m=0.9, approach_clearance_m=0.15)
        with pytest.raises(RobotError, match="reach_m must be positive"):
            RobotProfile(name="x", footprint_m=0.6, reach_m=0.0, approach_clearance_m=0.15)
        with pytest.raises(RobotError, match="approach_clearance_m must be positive"):
            RobotProfile(name="x", footprint_m=0.6, reach_m=0.9, approach_clearance_m=0.0)

    def test_a_profile_must_be_named(self) -> None:
        with pytest.raises(RobotError, match="must be named"):
            RobotProfile(name="", footprint_m=0.6, reach_m=0.9, approach_clearance_m=0.15)

    def test_the_band_is_ordered_and_inside_the_reach(self) -> None:
        band = standoff_distances(PANDA)
        assert band[0] < band[1] < band[2]
        assert band[2] < PANDA.reach_m
        assert band[0] >= PANDA.approach_clearance_m

    def test_an_arm_that_cannot_stand_off_at_all_is_refused(self) -> None:
        """Reach shorter than the approach needs: the arm cannot work on anything, and saying so
        beats returning an empty band that reads as 'nothing was reachable'."""
        stubby = RobotProfile(name="stubby", footprint_m=0.6, reach_m=0.4,
                              approach_clearance_m=0.15)
        with pytest.raises(RobotError, match="cannot work on anything"):
            standoff_distances(stubby)


class TestTheStandoffSolve:
    def test_an_open_room_leaves_the_target_reachable_from_all_round(self) -> None:
        standoffs = standoffs_for((2.5, 2.0), region=ROOM, obstacles=(), profile=PANDA)
        assert count_directions(standoffs, (2.5, 2.0)) > 12

    def test_a_target_at_the_wall_has_fewer_directions_than_one_in_the_open(self) -> None:
        """The measurement a visual critic was making by eye.

        A nightstand against a wall leaves the gripper one side to come from. That is a real
        restriction and it is not a failure -- the measured case is 11 directions against 21 --
        which is why this reports a count rather than refusing.
        """
        open_air = standoffs_for((2.5, 2.0), region=ROOM, obstacles=(), profile=PANDA)
        at_wall = standoffs_for((2.5, 0.12), region=ROOM, obstacles=(), profile=PANDA)
        assert count_directions(at_wall, (2.5, 0.12)) < count_directions(open_air, (2.5, 2.0))
        assert at_wall, "a target against a wall is still reachable from the room side"

    def test_the_support_is_not_an_obstruction(self) -> None:
        """The support does not obstruct the arm lane.

        The base must stay outside the desk, but the arm reaches over the surface it works on.
        Counting the desk as a lane obstruction makes every target resting on anything fail.
        """
        desk = Polygon.rectangle(1.0, 1.0, 3.0, 2.0)
        standoffs = standoffs_for((2.0, 1.5), region=ROOM, obstacles=(desk,), profile=PANDA,
                                  support=desk)
        assert standoffs

    def test_the_robot_base_does_not_stand_inside_the_support(self) -> None:
        from rfw_w.scenes.geometry.polygon import overlaps

        desk = Polygon.rectangle(1.0, 1.0, 3.0, 2.0)
        standoffs = standoffs_for((2.0, 1.5), region=ROOM, obstacles=(desk,), profile=PANDA,
                                  support=desk)
        for item in standoffs:
            half = PANDA.base_radius_m
            base = Polygon.rectangle(
                item.centre[0] - half, item.centre[1] - half,
                item.centre[0] + half, item.centre[1] + half,
            )
            assert not overlaps(base, desk)

    def test_something_else_in_the_way_blocks_it(self) -> None:
        """The other half: a piece that is *not* the support does obstruct, and a target boxed in
        on every side is the failure the gate exists to catch."""
        ring = [
            Polygon.rectangle(-1.0 + 2.5 * index, -1.0, 1.0 + 2.5 * index, 5.0)
            for index in range(4)
        ]
        blocked = standoffs_for((2.5, 2.0), region=ROOM, obstacles=tuple(ring), profile=PANDA)
        assert not blocked, f"boxed in, yet {len(blocked)} standoffs were found"

    def test_the_answer_is_deterministic(self) -> None:
        first = standoffs_for((2.5, 2.0), region=ROOM, obstacles=(), profile=PANDA)
        second = standoffs_for((2.5, 2.0), region=ROOM, obstacles=(), profile=PANDA)
        assert first == second

    def test_a_target_outside_the_room_is_unreachable(self) -> None:
        assert not standoffs_for((9.0, 9.0), region=ROOM, obstacles=(), profile=PANDA)

    def test_every_standoff_is_inside_the_room(self) -> None:
        from rfw_w.scenes.geometry.polygon import contains_polygon

        for item in standoffs_for((2.5, 2.0), region=ROOM, obstacles=(), profile=PANDA):
            half = PANDA.base_radius_m
            base = Polygon.rectangle(
                item.centre[0] - half, item.centre[1] - half,
                item.centre[0] + half, item.centre[1] + half,
            )
            assert contains_polygon(ROOM, base), item.centre


class TestTheRegisteredProfiles:
    def test_every_profile_is_consistent_with_its_own_name(self) -> None:
        for name, profile in ROBOT_PROFILES.items():
            assert profile.name == name
            assert standoff_distances(profile)
