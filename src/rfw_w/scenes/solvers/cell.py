# SPDX-License-Identifier: Apache-2.0
"""Solve the task work cell before a room exists.

This module deliberately accepts no ``RoomShell`` or room polygon. It places the target with
robot-approach scoring and emits the floor region a later room solve must protect.

The support surface comes from one of two places, and the cell never blurs them. By default it is a
conservative envelope derived from the spec vocabulary, and that estimate is **named** as one in
``support_geometry_method``. A caller that has already retrieved the furniture may pass the surface
the asset library states (``MeasuredSupport``); then the cell protects that surface instead, and
records both the asset annotation's method and its status so a reader of the manifest can tell an
estimate from a measurement without asking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rfw_w.scenes.contracts.asset import AffordanceStatus, MeasuredSupport
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_spec import EntitySpec, SceneSpec
from rfw_w.scenes.geometry.polygon import Polygon, convex_hull
from rfw_w.scenes.geometry.robot import Standoff, profile_for, standoffs_for
from rfw_w.scenes.solvers.furnishing_plan import (
    CategoryProfile,
    PlanDerivationError,
    derive_support_and_target,
    profile_for as category_profile_for,
    support_shape_for,
)
from rfw_w.scenes.solvers.layout import (
    ApproachPreference,
    Footprint,
    OccupancyMap,
    SurfacePlacementConfig,
    place_on_surface,
)
from rfw_w.scenes.solvers.support import Surface


class CellError(ValueError):
    """The task cell cannot be solved from the declared semantics."""


@dataclass(frozen=True, slots=True)
class Rigid2D:
    """The one transform a later room solve may choose for the frozen cell."""

    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m, "yaw_rad": self.yaw_rad}


@dataclass(frozen=True, slots=True)
class WorkCell:
    """Task geometry in a local frame, solved without a room."""

    support_id: str
    target_id: str
    support_footprint: Polygon
    support_height_m: float
    target_offset: tuple[float, float]
    approach_arc_deg: tuple[float, float]
    protected: Polygon
    standoffs: tuple[Standoff, ...]
    local_frame: Rigid2D = Rigid2D()
    # The form the support is built to. Always known now that the support is constructed rather than
    # retrieved, which is why it is emitted unconditionally while the two fields below stay
    # conditional -- a cell with no shape would be a cell nobody can build.
    support_shape: str = ""
    support_geometry_method: str = "category_profile_envelope"
    # Present only when a real asset's surface was injected. The estimate path leaves them None and
    # they are then absent from ``to_dict`` entirely, so the default cell's bytes -- and therefore
    # every hash pinned against it -- do not move because a capability was added.
    support_geometry_status: str | None = None
    support_source_polygon_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "support_id": self.support_id,
            "target_id": self.target_id,
            "support_footprint": self.support_footprint.to_list(),
            "support_height_m": round(self.support_height_m, 6),
            "target_offset": [round(value, 6) for value in self.target_offset],
            "approach_arc_deg": [round(value, 6) for value in self.approach_arc_deg],
            "protected": self.protected.to_list(),
            "standoffs": [
                {"centre": [round(value, 6) for value in item.centre],
                 "distance_m": round(item.distance_m, 6)}
                for item in self.standoffs
            ],
            "local_frame": self.local_frame.to_dict(),
            "support_shape": self.support_shape,
            "support_geometry_method": self.support_geometry_method,
        }
        if self.support_geometry_status is not None:
            payload["support_geometry_status"] = self.support_geometry_status
        if self.support_source_polygon_id is not None:
            payload["support_source_polygon_id"] = self.support_source_polygon_id
        return payload

    @property
    def core_hash(self) -> str:
        return content_hash(self.to_dict())

    @property
    def support_centre_offset_m(self) -> tuple[float, float]:
        """Where the support's **centred** frame sits, in this cell's frame.

        Two frames meet here and they are not the same one, which is the whole reason this is a
        named property rather than an expression at the call site.

        The cell puts the support's **front edge on ``y = 0``**, body toward ``+y`` -- that is the
        frame the target offset and the standoffs are solved in, and it is what
        ``assets/support_measurement.py`` converts a library asset into. The mesh
        ``assets/procedural_support.py`` builds is **centred on its own footprint**, spanning
        ``y ∈ [-depth/2, +depth/2]``, because that is the convention the retrieved assets it stands
        in for use. So a support built to this cell's dimensions sits half a depth away from where
        the cell says its origin is.

        The build places the support at ``translation + this``; a caller that placed it at
        ``translation`` would put the task target ``depth/2`` behind the surface it was solved on,
        and nothing downstream measures that.
        """
        min_x, min_y, max_x, max_y = self.support_footprint.bounds()
        return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)


def _midpoint(entity: EntitySpec, axis: int, fallback: float) -> float:
    if entity.size_bounds_m is None:
        return fallback
    bounds = entity.size_bounds_m[axis]
    return (bounds.minimum + bounds.maximum) / 2.0


def _support_envelope(entity: EntitySpec) -> tuple[float, float, float]:
    profile = category_profile_for(entity.category)
    if profile is None or profile.target_extent_m is None:
        raise CellError(
            f"CORE_SUPPORT_SIZE_UNKNOWN({entity.id}): category {entity.category!r} has no "
            "target extent for a room-independent support envelope"
        )
    width = _midpoint(entity, 0, profile.target_extent_m)
    depth = _midpoint(entity, 1, max(0.45, profile.target_extent_m * 0.55))
    height = _midpoint(entity, 2, 0.75)
    if min(width, depth, height) <= 0:
        raise CellError(f"CORE_SUPPORT_SIZE_INVALID({entity.id}): {(width, depth, height)}")
    return width, depth, height


def _profile_for_entity(entity: EntitySpec) -> CategoryProfile:
    """The category profile an entity's support is built from, refused by name when there is none."""
    profile = category_profile_for(entity.category)
    if profile is None:
        raise CellError(
            f"CORE_SUPPORT_SIZE_UNKNOWN({entity.id}): category {entity.category!r} has no profile, "
            "so there is neither a size nor a form to build a support from"
        )
    return profile


def _measured_support(
    entity: EntitySpec, measurement: MeasuredSupport,
) -> tuple[float, float, float]:
    """Check an injected measurement, and hand back the numbers the rest of the solve uses.

    Two things are refused by name rather than absorbed, because both would produce a core that is
    quietly wrong in a way nothing downstream could see:

    * a measurement whose status is not usable for a hard requirement. An estimate handed in *as* a
      measurement would make the cell assert a surface it never measured, which is exactly the
      distinction ``WorkCell.support_geometry_status`` exists to expose.
    * a measurement that is not in the cell's own frame. ``MeasuredSupport`` contracts for "front
      edge at y=0, body toward +y"; anything else would move the protected region the room is later
      fitted around, and the miss would show up as a room that fits, not as an error.
    """
    if not measurement.provenance.status.usable_for_hard_requirement:
        raise CellError(
            f"CORE_SUPPORT_MEASUREMENT_NOT_USABLE({entity.id}): the measurement states status "
            f"{measurement.provenance.status.value!r} via {measurement.provenance.method!r}; a "
            "support whose surface is not measured or annotated cannot be protected as one"
        )
    front_edge = min(y for _, y in measurement.vertices_xy)
    if abs(front_edge) > 1e-6:
        raise CellError(
            f"CORE_SUPPORT_MEASUREMENT_FRAME({entity.id}): the measurement's front edge is at "
            f"y={front_edge:.6f} but the cell frame puts it at y=0; vertices in the asset's frame "
            "would move the protected region rather than fail here"
        )
    width, depth = measurement.width_m, measurement.depth_m
    height = measurement.support_height_m
    if min(width, depth, height) <= 0:
        raise CellError(
            f"CORE_SUPPORT_SIZE_INVALID({entity.id}): measured {(width, depth, height)}"
        )
    return width, depth, height


def solve_cell(
    spec: SceneSpec, *, seed: int, support_measurement: MeasuredSupport | None = None,
) -> WorkCell:
    """Solve the support, target and robot work region without consulting a room.

    ``seed`` is part of the public boundary even though the first deterministic policy needs no
    random draw. Future tie-breaks may use a named substream without admitting a room seed here.

    ``support_measurement`` is where a real library's surface enters, and it is optional on purpose.
    Without it the support is a **named category estimate** (``category_profile_envelope``) derived
    from the spec alone, which is all that can be solved before any asset is retrieved. With it the
    support is the surface the asset actually states, and the cell records which asset and which
    surface it came from. Either way what comes out is a function of ``(spec, seed)`` plus, at most,
    an asset measurement -- never of a room.
    """
    del seed
    gaps: list[str] = []
    try:
        support_id, target_id = derive_support_and_target(spec, gaps)
    except PlanDerivationError as error:
        raise CellError(str(error)) from error
    support_entity = spec.entity(support_id)
    target_entity = spec.entity(target_id)

    shape = support_shape_for(support_entity, _profile_for_entity(support_entity))

    if support_measurement is None:
        width, depth, height = _support_envelope(support_entity)
        # The support faces local -Y. Its front edge is y=0 and the robot aisle lies in front of
        # it; a later room solve may move and rotate this whole frame but may not reshape it.
        support_polygon = Polygon.rectangle(-width / 2.0, 0.0, width / 2.0, depth)
        # Not an estimate any more: these are the dimensions the support is *built* to, so the
        # surface the task is solved against is the surface that will exist. The status says which
        # side of the boundary it came from, and `constructed` is the honest one.
        support_status = AffordanceStatus.CONSTRUCTED
        support_method = f"constructed:{shape.value}"
        support_geometry_status: str | None = AffordanceStatus.CONSTRUCTED.value
        support_source_polygon_id: str | None = None
    else:
        width, depth, height = _measured_support(support_entity, support_measurement)
        support_polygon = Polygon.from_sequence(support_measurement.vertices_xy)
        support_status = support_measurement.provenance.status
        support_method = f"asset_support_polygon:{support_measurement.provenance.method}"
        support_geometry_status = support_measurement.provenance.status.value
        support_source_polygon_id = support_measurement.source_polygon_id

    surface = Surface(
        surface_id=f"{support_id}:planned_top",
        owner_instance=support_id,
        polygon=support_polygon,
        plane_z_m=height,
        provenance=support_status,
    )
    robot = profile_for(spec.task.robot_profile)
    aisle = Polygon.rectangle(
        -width / 2.0 - robot.base_radius_m,
        -robot.reach_m - robot.base_radius_m,
        width / 2.0 + robot.base_radius_m,
        0.0,
    )
    target_width = _midpoint(target_entity, 0, 0.10)
    target_depth = _midpoint(target_entity, 1, target_width)
    target_height = _midpoint(target_entity, 2, 0.10)
    outcome = place_on_surface(
        instance_id=target_id,
        entity_id=target_id,
        footprint=Footprint((target_width / 2.0, target_depth / 2.0)),
        surface=surface,
        object_height_m=target_height,
        occupancy=OccupancyMap(),
        config=SurfacePlacementConfig(step_m=0.04, max_candidates=4096),
        origin_convention="centre",
        approach=ApproachPreference(region=aisle, profile=robot),
    )
    if outcome.placement is None:
        raise CellError(
            f"CORE_TARGET_UNREACHABLE({target_id}): {outcome.detail}; "
            f"{len(outcome.rejections)} candidate(s) rejected"
        )
    target_xy = (outcome.placement.position_m.x, outcome.placement.position_m.y)
    standoffs = standoffs_for(
        target_xy,
        region=aisle,
        obstacles=(support_polygon,),
        profile=robot,
        support=support_polygon,
    )
    if not standoffs:
        raise CellError(f"CORE_TARGET_UNREACHABLE({target_id}): placement produced no standoff")
    angles = sorted(
        math.degrees(math.atan2(item.centre[1] - target_xy[1], item.centre[0] - target_xy[0]))
        for item in standoffs
    )
    protected_points = list(support_polygon.vertices)
    half = robot.base_radius_m
    for item in standoffs:
        protected_points.extend(Polygon.rectangle(
            item.centre[0] - half, item.centre[1] - half,
            item.centre[0] + half, item.centre[1] + half,
        ).vertices)
    return WorkCell(
        support_id=support_id,
        target_id=target_id,
        support_footprint=support_polygon,
        support_height_m=height,
        target_offset=target_xy,
        approach_arc_deg=(angles[0], angles[-1]),
        protected=convex_hull(protected_points),
        standoffs=standoffs,
        support_shape=shape.value,
        support_geometry_method=support_method,
        support_geometry_status=support_geometry_status,
        support_source_polygon_id=support_source_polygon_id,
    )
