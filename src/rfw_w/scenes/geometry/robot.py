# SPDX-License-Identifier: Apache-2.0
"""The robot as a body, so placement can be judged against it rather than described for it.

Why this exists
---------------

``TaskSpec.robot_profile`` is required by the spec, validated, and read by **nothing**. The only
fact about the robot anywhere in this codebase was its footprint -- 0.6 m, used for the door width
and the corridor -- and that is a fact about what can walk through a doorway, not about what can
pick something up.

So "can the robot actually use this scene" was not merely unchecked, it was *unrepresentable*: no
value in the system could express the distance a target may sit from where the robot stands. That
is the gap a visual critic found by looking at a render -- *"the alarm clock is positioned at the
rear wall corner of the nightstand, restricting gripper approach directions for the pick task"* --
while all twenty-six gates passed, because not one of them had a robot to measure against.

Stdlib only, and walked by ``rfw_w.scenes.validation.check_stdlib_only``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from rfw_w.scenes.geometry.clearance import DEFAULT_MARGIN_M
from rfw_w.scenes.geometry.polygon import Polygon, contains_polygon, overlaps


class RobotError(ValueError):
    """The robot profile is not one this release can reason about."""


@dataclass(frozen=True, slots=True)
class RobotProfile:
    """A robot's dimensions, as the numbers a placement check needs.

    Deliberately four numbers and not a kinematic model. A full model would let a gate lie with
    more confidence: what the pipeline can honestly say is "there is space for the base here, the
    target is inside the arm's reach from it, and nothing stands between the two" -- a *reach
    proxy*, which is what ``robot.ik`` declares as not-run. It cannot say a grasp exists, and it
    must not pretend to.
    """

    name: str
    footprint_m: float
    reach_m: float
    approach_clearance_m: float

    def __post_init__(self) -> None:
        if not self.name:
            raise RobotError("a robot profile must be named")
        for field_name in ("footprint_m", "reach_m", "approach_clearance_m"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0:
                raise RobotError(f"{self.name}: {field_name} must be positive, got {value}")

    @property
    def base_radius_m(self) -> float:
        """The radius the base sweeps, for asking whether it fits somewhere.

        Half the footprint: the footprint is the widest extent, and a base is placed by its centre.
        """
        return self.footprint_m / 2.0


# The profiles this release can reason about, keyed by the name a spec already carries.
#
# `franka_panda` is the arm the three experiments name. Its reach is the manufacturer's 855 mm
# from the flange origin; the footprint is the 0.6 m the corridor and door checks have used all
# along, so those keep their current behaviour. `approach_clearance_m` is the free space a
# parallel-jaw gripper needs in front of what it is about to close on -- the number that decides
# whether a mug in the corner of a desk is graspable, and the one nothing in this repository had.
ROBOT_PROFILES: Mapping[str, RobotProfile] = {
    "franka_panda": RobotProfile(
        name="franka_panda", footprint_m=0.6, reach_m=0.855, approach_clearance_m=0.15,
    ),
    # Measured from the manufacturers' own URDFs by `rfw_w.scenes.visualization.measure_robot_profile`
    # (2026-09-26, seed 11, 1502 samples; `outputs/_paper/robot_profiles_20260926_02`), which was
    # committed before these numbers were used. `reach_m` is the largest horizontal distance from
    # the base axis to the TCP over joint vectors inside the URDF limits -- a placement proxy, not
    # a grasp; `robot.ik` stays not-run for both. `footprint_m` is the support band only (the
    # lowest 0.30 m): the G1's whole-body swept diameter at its ready pose is 0.9297 m, but its
    # outstretched hands need no floor under them.
    "piper_x": RobotProfile(
        name="piper_x", footprint_m=0.4423, reach_m=0.8027, approach_clearance_m=0.15,
    ),
    "unitree_g1": RobotProfile(
        name="unitree_g1", footprint_m=0.4003, reach_m=0.5551, approach_clearance_m=0.18,
    ),
}

def profile_for(name: str) -> RobotProfile:
    """The profile a spec's ``robot_profile`` names.

    Raises for an unknown name rather than guessing dimensions. A robot this release has no numbers
    for is a robot whose reach it cannot check, and quietly substituting another arm's reach would
    be a gate passing on a measurement of a different machine.
    """
    profile = ROBOT_PROFILES.get(name)
    if profile is None:
        raise RobotError(
            f"UNKNOWN_ROBOT_PROFILE({name!r}): this release has numbers for "
            f"{sorted(ROBOT_PROFILES)}. Add the arm to ROBOT_PROFILES with its reach and approach "
            "clearance, or the reach checks would be measuring a different robot."
        )
    return profile


@dataclass(frozen=True, slots=True)
class Standoff:
    """A place the robot's base could stand to work on something, and how far it is from it."""

    centre: tuple[float, float]
    distance_m: float

def _lane(start: tuple[float, float], end: tuple[float, float], width_m: float) -> Polygon:
    """A rectangle of ``width_m`` running from ``start`` to ``end``.

    The shape the approach is checked against: a base may stand beside a target all it likes, but
    the gripper has to get from the base to the target through something, and this is that
    something.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        raise RobotError(f"the approach from {start} to {end} has no length")
    along = (dx / length, dy / length)
    across = (-along[1], along[0])
    half = width_m / 2.0
    corners = [
        (start[0] + across[0] * half, start[1] + across[1] * half),
        (end[0] + across[0] * half, end[1] + across[1] * half),
        (end[0] - across[0] * half, end[1] - across[1] * half),
        (start[0] - across[0] * half, start[1] - across[1] * half),
    ]
    return Polygon(tuple(corners))


def standoffs_for(
    target_xy: tuple[float, float],
    *,
    region: Polygon,
    obstacles: Sequence[Polygon],
    profile: RobotProfile,
    support: Polygon | None = None,
    directions: int = 24,
) -> tuple[Standoff, ...]:
    """Where a base could stand from which ``target_xy`` could be worked on.

    Three things must hold, and each is a named part of the claim rather than a blurred "reachable":

    * the **base fits** -- a footprint-sized box centred on the spot lies inside the room's air;
    * the base is **clear** of everything already placed;
    * the **approach lane** from the base to the target is clear, at the width the gripper needs.

    ``support`` is the surface the target is standing on. It blocks candidate base footprints but
    is removed from the arm-lane obstacles: a mobile manipulator cannot stand inside a desk, while
    its arm must legitimately reach over that desk to the object on top. Treating the support as a
    lane obstruction makes every such target unreachable; removing it from base obstacles lets the
    robot stand inside furniture. The two checks therefore need distinct obstacle sets.

    Empty means no base position works, which is a scene the robot cannot use however good it
    looks. It is not a statement about grasps: see the note on :class:`RobotProfile`.
    """
    # A support blocks the robot base: a mobile manipulator cannot stand inside the desk it works
    # over. It does not block the arm lane, which legitimately crosses the support on the way to
    # an object on its top. Keeping these two obstacle sets distinct is what makes candidate
    # positions on one support have different usable approach directions.
    base_blocked = tuple(obstacles)
    lane_blocked = tuple(item for item in obstacles if item is not support)
    band = standoff_distances(profile)
    half = profile.base_radius_m
    found: list[Standoff] = []
    for index in range(directions):
        angle = 2.0 * math.pi * index / directions
        for distance in band:
            centre = (
                target_xy[0] + math.cos(angle) * distance,
                target_xy[1] + math.sin(angle) * distance,
            )
            base = Polygon.rectangle(
                centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half
            )
            if not contains_polygon(region, base):
                continue
            if base_blocked and any(overlaps(base, item) for item in base_blocked):
                continue
            # From the base's edge to the target, not from its centre: the arm starts at the
            # robot's body, and a lane that begins inside the base would be called blocked by the
            # robot itself.
            start = (
                centre[0] + math.cos(angle + math.pi) * half,
                centre[1] + math.sin(angle + math.pi) * half,
            )
            if lane_blocked and any(
                overlaps(_lane(start, target_xy, profile.approach_clearance_m), item)
                for item in lane_blocked
            ):
                continue
            found.append(Standoff(centre=centre, distance_m=distance))
    return tuple(found)


def standoff_distances(profile: RobotProfile) -> tuple[float, ...]:
    """Where a base could stand to work on something, as distances from it.

    A band rather than a point: a base much closer than the approach clearance crowds the thing it
    is working on, and one further than the reach cannot get there at all. Both ends are the
    profile's own numbers.
    """
    near = profile.approach_clearance_m + profile.base_radius_m
    far = profile.reach_m - DEFAULT_MARGIN_M
    if far <= near:
        raise RobotError(
            f"{profile.name}: reach {profile.reach_m:.3f} m leaves no room to stand outside the "
            f"{near:.3f} m the approach needs; this arm cannot work on anything"
        )
    return (near, (near + far) / 2.0, far)
