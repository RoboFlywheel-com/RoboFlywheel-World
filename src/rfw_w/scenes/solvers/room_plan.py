# SPDX-License-Identifier: Apache-2.0
"""Turning the fitted room into a plan.

The room is *drawn* today: `fit_room_around_cell` reads the cell's protected region, samples a
shape from the spec's size bounds, places the openings and returns a `RoomShell` plus the
translation that puts the core inside it. This module expresses the same room as a
`pvscene.room-plan/v1` document, so the room generation stage can consume a plan instead of a
drawing.

**It changes nothing about the room that is drawn.** The geometry is identical; a plan is added
beside it. That is the property the tests pin, and the reason the translation is a pure function of
`FittedRoom` rather than a second call into the fit.

One reconstruction detail is load-bearing. `RoomShell.openings` carries `center_xy` and a width --
**the wall index is already gone by then** -- so the plan's `wall_segment_xy` cannot be recovered
from a shell. It comes from `FittedRoom.options.openings[].wall_index`, which is exact:
`_place_openings` (`room.py:1546`) and `_materialise` (`room.py:1604`) both read
`walls = list(polygon.edges)`, so `wall_index == i` names the polygon's i-th edge.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.room_plan import (
    SCHEMA_VERSION,
    UP_AXIS,
    UNITS,
    PlanCore,
    PlanOpening,
    PlanProvenance,
    PlanRoom,
    RoomPlan,
    RoomPlanError,
)
from rfw_w.scenes.contracts.background_constraints import (
    CIRCULATION_BAND_TOP_M,
    DOOR_SWEEP_BAND_TOP_M,
    REGION_CIRCULATION,
    REGION_DOOR_SWEEP,
    REGION_PROTECTED,
    Reservation,
)
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.solvers.cell import WorkCell
from rfw_w.scenes.solvers.materials import FLOOR, WALL, resolve_roles
from rfw_w.scenes.solvers.room import FittedRoom, fit_room_around_cell

#: Coordinate precision for plan geometry. Six decimals is the same choice `_materialise` makes for
#: an opening's centre, so a plan and the shell it came from round their shared quantities alike.
PLAN_DECIMALS = 6

#: The room type in the plan's room key. Infinigen parses the room type and the storey back out of
#: `"<roomtype>_<storey>/<idx>"`, so this is not free text -- `Semantics.Office` is the vocabulary
#: this has to stay inside.
ROOM_KIND = "office"
ROOM_STOREY = 0
ROOM_INDEX = 0


class RoomPlanProvider(Protocol):
    """Turns a solved cell into a plan. The room line's first provider."""

    def plan(
        self, spec: SceneSpec, core: WorkCell, *, room_seed: int, wall_standoff_m: float = 0.20
    ) -> RoomPlan: ...


def _edges(vertices: tuple[tuple[float, float], ...]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The polygon's edges, cyclically. The same list `_place_openings` indexes into."""
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _segment_for(wall, offset_along_m: float, width_m: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The stretch of wall an opening occupies, as a segment centred on its offset.

    Built as ``offset ± width/2`` rather than by reading a stored segment, so that its length is
    the width by construction -- which is what R7 checks, and why this function cannot produce a
    disagreement for the check to find.
    """
    (ax, ay), (bx, by) = wall
    length = math.dist((ax, ay), (bx, by))
    if length <= 0:
        raise RoomPlanError(f"PLAN_NOT_SERIALISABLE(wall): degenerate wall from {(ax, ay)} to {(bx, by)}")
    ux, uy = (bx - ax) / length, (by - ay) / length
    near = (ax + ux * (offset_along_m - width_m / 2), ay + uy * (offset_along_m - width_m / 2))
    far = (ax + ux * (offset_along_m + width_m / 2), ay + uy * (offset_along_m + width_m / 2))
    round_to = lambda xy: (round(xy[0], PLAN_DECIMALS), round(xy[1], PLAN_DECIMALS))
    return round_to(near), round_to(far)


def plan_from_fitted(
    spec: SceneSpec,
    core: WorkCell,
    fitted: FittedRoom,
    *,
    room_seed: int,
    kind: str | None = None,
) -> RoomPlan:
    """Express an already-fitted room as a plan. Pure: nothing here re-runs the fit.

    Reading `FittedRoom` rather than re-deriving is the whole point. A second call into
    `fit_room_around_cell` could return a different plan from the room that was actually drawn if
    anything in the fit ever became seed-sensitive in a new way, and the difference would show up
    as a provider building a room nobody chose.
    """
    shell = fitted.shell
    if shell.holes_xy:
        raise RoomPlanError(
            f"PLAN_NOT_SERIALISABLE(rooms): the fitted shell carries {len(shell.holes_xy)} hole(s); "
            "the plan vocabulary is simple polygons"
        )
    if not fitted.options.openings and shell.openings:
        # `build_shell` regenerates openings from `f"{spec.seed}:openings"` when the options leave
        # them empty, so the shell would carry openings no plan describes. Refusing beats writing a
        # plan that silently disagrees with the room it claims to be about.
        raise RoomPlanError(
            f"PLAN_NOT_SERIALISABLE(openings): the shell has {len(shell.openings)} opening(s) but "
            "the options that produced it name none; the plan would describe a different room"
        )

    walls = _edges(shell.polygon_xy)
    openings = tuple(
        PlanOpening(
            opening_id=plan.opening_id,
            kind=plan.kind,
            room_id="room_0",
            wall_segment_xy=_segment_for(
                walls[_wall_index(plan.wall_index, len(walls), plan.opening_id)],
                plan.offset_along_m,
                plan.width_m,
            ),
            width_m=plan.width_m,
            height_m=plan.height_m,
            sill_z_m=plan.sill_m,
        )
        for plan in fitted.options.openings
    )

    tokens = resolve_roles(spec.style.palette)
    room = PlanRoom(
        room_id="room_0",
        kind=kind or spec.room.room_kind or ROOM_KIND,
        polygon_xy=shell.polygon_xy,
        floor_z_m=shell.floor_z_m,
        ceiling_height_m=shell.height_m,
        floor_material=tokens.get(FLOOR) or "",
        wall_material=tokens.get(WALL) or "",
    )

    return RoomPlan(
        plan_id=f"{spec.scene_id}:{room_seed}:core",
        source="core",
        rooms=(room,),
        core=PlanCore(
            core_hash=core.core_hash,
            translation_xy=tuple(round(value, PLAN_DECIMALS) for value in fitted.cell_translation_xy),
            rotation_yaw_rad=0.0,
            protected_polygon_xy=tuple(
                (round(x, PLAN_DECIMALS), round(y, PLAN_DECIMALS)) for x, y in core.protected.to_list()
            ),
            support_top_z_m=round(core.support_height_m, PLAN_DECIMALS),
            target_offset_xy=tuple(round(value, PLAN_DECIMALS) for value in core.target_offset),
        ),
        provenance=PlanProvenance(
            spec_hash=content_hash(spec.to_dict()),
            seed=room_seed,
            shell_options_digest=content_hash(fitted.options.to_dict()),
            room_intent=spec.room.to_dict() if spec.room.room_kind is not None else {},
        ),
        units=UNITS,
        up_axis=UP_AXIS,
        openings=openings,
        schema_version=SCHEMA_VERSION,
    )


def _wall_index(index: int, count: int, opening_id: str) -> int:
    if not 0 <= index < count:
        raise RoomPlanError(
            f"PLAN_NOT_SERIALISABLE({opening_id}): wall {index} is outside the outline's "
            f"{count} wall(s)"
        )
    return index


class CoreRoomPlanProvider:
    """The default provider: fit the room around the cell, then describe what was fitted."""

    def plan(
        self, spec: SceneSpec, core: WorkCell, *, room_seed: int, wall_standoff_m: float = 0.20
    ) -> RoomPlan:
        fitted = fit_room_around_cell(
            spec, core, room_seed=room_seed, wall_standoff_m=wall_standoff_m
        )
        return plan_from_fitted(spec, core, fitted, room_seed=room_seed)

def reservations_from_plan(plan: RoomPlan, *, circulation: Sequence[Any],
                           door_sweep: Sequence[Any]) -> tuple[Reservation, ...]:
    """The three reservations a generated periphery must stay out of, from the plan and its zones.

    SPEC-011 requirement 3's regions, and each one's *source* is recorded rather than left implicit:
    the protected region is the plan's own `core.protected_in_room_xy` with the core's
    `support_top_z_m` as its band top, and the corridor and the door sweep are the zone solver's own
    polygons. The band tops for those two come from the contract, not from here, because ADR-009
    decision 3 fixed them (1.8 m) and a second copy of a decided constant is how two copies drift.

    One reservation per polygon: the corridor is several pieces and the door sweep is several more,
    and each is a region a piece of furniture can stand in -- merging them into one polygon would
    describe a shape nobody drew.
    """
    records = [Reservation(region=REGION_PROTECTED, polygon_xy=plan.core.protected_in_room_xy,
                           band_top_m=plan.core.support_top_z_m,
                           source="plan.core.protected_in_room_xy")]
    for region, polygons, band, source in (
        (REGION_CIRCULATION, circulation, CIRCULATION_BAND_TOP_M, "zones.circulation"),
        (REGION_DOOR_SWEEP, door_sweep, DOOR_SWEEP_BAND_TOP_M, "zones.door_sweep"),
    ):
        records.extend(Reservation(region=region, polygon_xy=tuple(polygon.vertices),
                                   band_top_m=band, source=source, index=index)
                       for index, polygon in enumerate(polygons))
    return tuple(records)
