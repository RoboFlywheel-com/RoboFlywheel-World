# SPDX-License-Identifier: Apache-2.0
"""What a shell provider reports: the shell it built, the numbers it measured, and what it did not.

A provider that only handed back a `RoomShell` would be claiming the shell *is* the room, and there
would be nothing to compare a plan against. `ShellBuild` is deliberately two things at once — the
shell downstream consumes, and the evidence about how it was arrived at — because the gate that
asks "did the builder build what we asked for" needs both, and a provider that kept its measurements
private could not be checked.

`not_measured` is the half that makes the rest honest. A quantity a provider cannot establish is
*absent with a reason*, never absent silently and never defaulted: a gate reading a missing number
as zero would pass a room nobody measured. `SHELL_MEASUREMENT_ABSENT` is what the gate raises, and
it is a failure of the evidence, not of the room.

Tolerances are **not** constants in this module. They were calibrated from a measurement (SPEC-003
step 5) and are carried in the build so a reader can see which numbers were compared loosely and by
how much -- a tolerance nobody can point at is indistinguishable from a gate nobody is running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from rfw_w.scenes.contracts.room_plan import (
    PlanOpening,
    RoomPlan,
    RoomPlanError,
    opening_host_span,
)
from rfw_w.scenes.contracts.state import Opening, OpeningKind, RoomShell

#: The calibration that `SHELL_DOES_NOT_MATCH_PLAN` compares under.
#:
#: Measured, not chosen. On the 2026-09-21 office build the six wall lines sit at a **constant**
#: 0.05 m from their plan edges and the spread between them is **0.0** to four decimals -- the plan's
#: shape is reproduced exactly, and the 0.05 is the inset Infinigen builds its walls at. The
#: tolerance has to be above float noise and below anything a real shape error would produce; 1 mm
#: is two hundred times the observed spread and fifty times below the smallest opening in the
#: vocabulary.
WALL_LINE_TOLERANCE_M = 1e-3

#: Room height is compared directly: the office build's room object spans [0, 2.9697] against a plan
#: of 2.9697, an exact match. 1 mm leaves room for a float that survived a boolean.
HEIGHT_TOLERANCE_M = 1e-3


class ShellBuildError(ValueError):
    """A shell build could not be read or reported. The message leads with the failure code."""


@dataclass(frozen=True, slots=True)
class ShellBuild:
    """The shell a provider built, plus what it did and did not measure about it."""

    shell: RoomShell
    provider: str
    measurements: Mapping[str, Any] = field(default_factory=dict)
    not_measured: tuple[str, ...] = ()
    geometry_uris: tuple[str, ...] = ()
    material_uris: tuple[str, ...] = ()
    # Quantities that **were** measured and are **not** in the gate's comparison set, each with the
    # reason. Separate from `not_measured` because the two say opposite things and conflating them
    # would misreport the evidence in one direction or the other: "we did not establish this" is
    # not the same claim as "we established it and the plan does not constrain it". A reader
    # asking whether the builder was held to the opening widths needs the second answer, and
    # `not_measured` is the wrong place to find it.
    not_compared: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider:
            raise ShellBuildError("SHELL_BUILD_FAILED(provider): a build must name who made it")
        for entry in self.not_measured:
            if not entry:
                raise ShellBuildError(
                    "SHELL_BUILD_FAILED(not_measured): an absent measurement needs a reason"
                )

    def measured(self, field_name: str) -> Any:
        """A measurement, or a named refusal -- never ``None`` standing in for "did not look"."""
        if field_name not in self.measurements:
            raise ShellBuildError(
                f"SHELL_MEASUREMENT_ABSENT({field_name}): the {self.provider} provider did not "
                f"measure it; it is not zero and it is not a pass"
            )
        return self.measurements[field_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "shell": self.shell.to_dict(),
            "measurements": dict(self.measurements),
            "not_measured": list(self.not_measured),
            "not_compared": list(self.not_compared),
            "geometry_uris": list(self.geometry_uris),
            "material_uris": list(self.material_uris),
        }


def _on_segment(segment, point, *, tolerance_m: float = 1e-3) -> bool:
    """Whether a measured hole's centre lies on the wall stretch the plan gave that opening.

    The stretch is the **host span** (`room_plan.opening_host_span`), not the plan's own
    `wall_segment_xy`, and the distinction is the whole point: a window is handed a span widened to
    what its builder can build in, so it may come out anywhere in that span. Holding it to the
    narrower segment would report a hole as belonging to no planned opening -- `SHELL_OPENING_MISSING`
    and `SHELL_OPENING_UNEXPECTED` for one hole -- over a stretch its builder was never given.
    """
    (x1, y1), (x2, y2) = segment
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared <= 0:
        return False
    t = ((point[0] - x1) * dx + (point[1] - y1) * dy) / length_squared
    if not (-tolerance_m <= t <= 1 + tolerance_m):
        return False
    projection = (x1 + t * dx, y1 + t * dy)
    return math.dist(projection, point) <= tolerance_m + _HALF_WALL_M


def _host_span(plan: RoomPlan, opening: PlanOpening) -> tuple[tuple[float, float], tuple[float, float]]:
    """`opening_host_span` with the malformed-plan case turned into this module's own refusal.

    A plan whose opening is on no wall is refused by `check_plan` before it ever reaches a shell, so
    this is a second line rather than the first. It exists because the alternative is a
    `RoomPlanError` escaping a gate -- and "the plan is malformed" is a fact the gate can report.
    """
    try:
        return opening_host_span(plan, opening)
    except RoomPlanError as error:
        raise ShellBuildError(str(error)) from error


#: How far a hole may sit off its planned wall and still count as that opening's. The measured wall
#: is the interior face, half a wall thickness inside the plan line, so a hole is expected to be
#: about that far off; anything further is a hole in a different wall.
_HALF_WALL_M = 0.2


def openings_from_report(plan: RoomPlan, report: Mapping[str, Any]) -> tuple[Opening, ...]:
    """The openings a measurement report found, as shell openings in the plan's own frame.

    A measured hole is identified by **which plan opening it sits on**, not by a name: the mesh has
    no names, so the only way to say which hole is which is to ask which planned wall segment
    contains it. That makes both failure directions reachable and named -- a plan opening with no
    hole on its segment is ``SHELL_OPENING_MISSING``, and a hole on no planned segment is
    ``SHELL_OPENING_UNEXPECTED``. A whole-count comparison would have collapsed both into one
    number and, worse, gone on to match four planned openings against six measured holes by
    guesswork that would land in the evidence as if it were a measurement.
    """
    planned = list(plan.openings)
    walls = report.get("walls") or []
    measured = [(wall, hole) for wall in walls for hole in wall["openings"]]
    floor_z = report.get("floor_z_m", 0.0)

    openings, used = [], set()
    for plan_opening in planned:
        span = _host_span(plan, plan_opening)
        candidates = []
        for index, (wall, hole) in enumerate(measured):
            if index in used:
                continue
            centre = tuple(hole["centre_xy"])
            if _on_segment(span, centre):
                (x1, y1), (x2, y2) = span
                wanted = ((x1 + x2) / 2, (y1 + y2) / 2)
                candidates.append((math.dist(centre, wanted), index))
        if not candidates:
            raise ShellBuildError(
                f"SHELL_OPENING_MISSING({plan_opening.opening_id}, {plan_opening.kind}, "
                f"{plan_opening.wall_segment_xy}): the plan puts an opening on that wall and no "
                f"measured hole is on the stretch it was given, {span}"
            )
        candidates.sort()
        used.add(candidates[0][1])
        wall, hole = measured[candidates[0][1]]
        openings.append(
            Opening(
                opening_id=plan_opening.opening_id,
                kind=OpeningKind(plan_opening.kind),
                center_xy=tuple(hole["centre_xy"]),
                width_m=round(hole["offset_high"] - hole["offset_low"], 4),
                height_m=round(hole["z_max"] - hole["z_min"], 4),
                sill_z_m=round(hole["z_min"] - floor_z, 4),
            )
        )

    for index, (wall, hole) in enumerate(measured):
        if index not in used:
            raise ShellBuildError(
                f"SHELL_OPENING_UNEXPECTED(centre={tuple(hole['centre_xy'])!r},"
                f" width_m={round(hole['offset_high'] - hole['offset_low'], 4)}): the mesh has a hole "
                "no plan opening sits on"
            )
    return tuple(openings)


def shell_from_report(plan: RoomPlan, report: Mapping[str, Any], *, provider: str) -> ShellBuild:
    """Read a measurement report into a `ShellBuild`.

    The shell's outline comes from the plan rather than from the mesh, and that is deliberate: the
    measurement's contribution is *whether* the built walls lie on the planned ones (the offset
    spread), and the openings that were actually cut. Rebuilding the outline from wall lines would
    mean chaining segments and inventing corners the measurement never established.
    """
    room = plan.rooms[0]
    measurements = dict(report.get("measurements") or {})
    not_measured = list(report.get("not_measured") or ())

    openings = openings_from_report(plan, report)
    shell = RoomShell(
        polygon_xy=room.polygon_xy,
        holes_xy=(),
        height_m=measurements.get("room.height_m", room.ceiling_height_m),
        floor_z_m=room.floor_z_m,
        openings=openings,
        source=f"measured:{provider}",
    )
    for field_name in ("room.wall_line_offset_spread_m", "room.height_m"):
        if field_name not in measurements:
            not_measured.append(f"{field_name}: the measurement report does not carry it")
    return ShellBuild(
        shell=shell,
        provider=provider,
        measurements=measurements,
        not_measured=tuple(not_measured),
    )


def plan_gaps(
    plan: RoomPlan, build: ShellBuild, *, minimum_door_width_m: float | None = None
) -> tuple[str, ...]:
    """What the built shell disagrees with the plan about, as named gaps.

    This is the computation behind the gate ``shell.matches_plan``, and its content is the
    comparison set as SPEC-004 §4 settles it:

    * **the outline**, held to the plan through the *spread* of the wall-line offsets rather than
      through a polygon comparison. Two builders put their walls at different offsets -- this
      repository's own builder sits on the plan lines, Infinigen insets by half a wall -- and both
      reproduce the shape; what distinguishes a reproduced shape from a different one is whether
      those offsets agree with *each other*. A polygon comparison would call the inset a shape
      error, which is why SPEC-003 §4 R2 chose this measure.
    * **the height**.
    * **each opening**: that it exists, that it is the kind the plan named, and that it is on the
      wall the plan put it on -- measured against the **host span** the provider was handed
      (`room_plan.opening_host_span`), which for a window is wider than the plan's own segment.

    Nothing else. Opening widths, heights, sills and positions along their wall are outside it, and
    each build carries them in `not_compared` with the reason -- SPEC-001 R6 measured that
    Infinigen ignores what the plan says about size, so a gate comparing them would fail every
    build for a disagreement the builder was never able to honour.

    A quantity the gate needs and the build does not carry is ``SHELL_MEASUREMENT_ABSENT``: the
    gate cannot answer, and "could not answer" is not "agreed". That is the failure mode this
    whole contract exists to prevent -- a provider that measured nothing must not pass.

    ``minimum_door_width_m`` is the passage floor. It is passed in rather than defaulted so the
    caller names where the number came from, the way `validate_shell` does.
    """
    gaps: list[str] = []
    room = plan.rooms[0]

    spread_field = "room.wall_line_offset_spread_m"
    if spread_field not in build.measurements:
        gaps.append(
            f"SHELL_MEASUREMENT_ABSENT({spread_field}): the {build.provider} build does not report "
            "how far its walls sit from the plan's, so the outline cannot be compared and the gate "
            "does not pass by default"
        )
    elif float(build.measurements[spread_field]) > WALL_LINE_TOLERANCE_M:
        gaps.append(
            f"SHELL_WALLS_NOT_ON_PLAN(spread_m={build.measurements[spread_field]:.6g}, "
            f"tolerance_m={WALL_LINE_TOLERANCE_M:g}): the built walls do not sit at one common "
            "offset from their plan edges, so the outline they describe is not the plan's"
        )

    built_height = build.shell.height_m
    if abs(built_height - room.ceiling_height_m) > HEIGHT_TOLERANCE_M:
        gaps.append(
            f"SHELL_HEIGHT_DISAGREES_WITH_PLAN(built_m={built_height:.4f}, "
            f"planned_m={room.ceiling_height_m:.4f}, tolerance_m={HEIGHT_TOLERANCE_M:g})"
        )

    measured = {opening.opening_id: opening for opening in build.shell.openings}
    for planned in plan.openings:
        try:
            span = opening_host_span(plan, planned)
        except RoomPlanError as error:
            # A plan malformed enough that its opening is on no wall cannot be compared against a
            # build at all, and the gate says so by name rather than by raising: a gate that threw
            # would report "this plan is broken" as a crash rather than as the disagreement it is.
            gaps.append(str(error))
            continue
        found = measured.get(planned.opening_id)
        if found is None:
            gaps.append(
                f"SHELL_OPENING_MISSING({planned.opening_id}, {planned.kind}, "
                f"{planned.wall_segment_xy}): the plan puts an opening on that wall and the built "
                "shell has none"
            )
            continue
        if found.kind.value != planned.kind:
            gaps.append(
                f"SHELL_OPENING_KIND_DISAGREES({planned.opening_id}, built={found.kind.value}, "
                f"planned={planned.kind})"
            )
        if not _on_segment(span, found.center_xy):
            gaps.append(
                f"SHELL_OPENING_ON_WRONG_WALL({planned.opening_id}, centre_xy={found.center_xy}, "
                f"planned_wall_xy={planned.wall_segment_xy}): the opening exists but not on the "
                "wall the plan put it on"
            )

    planned_ids = {opening.opening_id for opening in plan.openings}
    for opening in build.shell.openings:
        if opening.opening_id not in planned_ids:
            gaps.append(
                f"SHELL_OPENING_UNEXPECTED({opening.opening_id}, kind={opening.kind.value}, "
                f"centre_xy={opening.center_xy}): the built shell has an opening the plan does not"
            )

    if minimum_door_width_m is not None:
        doors = [opening for opening in build.shell.openings
                 if opening.kind is OpeningKind.DOOR]
        # Reported against the plan's door by id when there is one, so the gap names the door the
        # task needs rather than whichever door happened to be first in the list.
        wanted = next(
            (opening.opening_id for opening in plan.openings
             if opening.kind == OpeningKind.DOOR.value),
            None,
        )
        candidates = [door for door in doors if wanted is None or door.opening_id == wanted] or doors
        for door in candidates:
            if door.width_m < minimum_door_width_m:
                gaps.append(
                    f"SHELL_DOOR_TOO_NARROW({door.opening_id}, built_m={door.width_m:.4f}, "
                    f"minimum_m={minimum_door_width_m:.4f}): the built door does not admit the "
                    "robot, which no layout behind it can repair"
                )
    return tuple(gaps)
