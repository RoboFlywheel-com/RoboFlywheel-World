# SPDX-License-Identifier: Apache-2.0
"""The support surface a furniture asset states, said in the task cell's frame.

Why this exists
---------------

The room-free core solves the task before any room exists, and everything it solves is a rigid
transform of one number: the support surface. Until this module existed that number was a category
average -- ``CATEGORY_PROFILES["desk"].target_extent_m`` and a size-band midpoint -- so the protected
region the room is later fitted around described a desk that no asset library contains. The library
does state the real surface; nothing was reading it for the core.

What this module does **not** do is prepare a mesh. ``prepare_furniture`` loads a GLB with trimesh,
decimates a collision hull and estimates a mass, none of which a support rectangle needs. Here the
sidecar alone is read, so measuring a support surface is a gzip decompress and some arithmetic --
and it stays importable by a bare interpreter.

What was measured in the library rather than assumed
----------------------------------------------------

* **coverage.** Of the 168 ``desk.n.01`` assets this library returns, 167 carry a support-surface
  sidecar. The one that does not is refused by name rather than given a default.
* **the base plane.** HSSD authors assets base-at-origin: ``modelbbox.min.y`` lies within
  [-1.9 mm, +1.0 mm] of zero across all 167 sidecars. So a reading's height above its own asset
  frame is its height above the floor, which is why ``base_offset_m`` may default to zero -- and why
  the assumption is named on the record (``height_basis``) instead of being left implicit.
* **the horizontal filter is not cosmetic.** ``_read_support_surfaces_cached`` keeps any reading
  whose flat OBB axis *equals its normal axis*, and a normal may point along x or z -- so a
  **vertical panel** qualifies as a support surface. On ``11c27a0b2950b3a3e9431128cd452ac7cbdf35c1``
  the largest-area entry (0.919 m2, normal along z) is a panel at y = 0.445 m while the desk's real
  top is 0.694 m2 at 0.752 m. Over the library, the overall-largest surface is a panel for **13 of
  167** desks, so the filter changes which surface is chosen for thirteen of them. Its cost is
  measurable too: choosing by area alone leaves **44 of 167** desks more than 10 cm below their own
  top (median error 0.018 m, worst 1.293 m), and the filter brings that to 37 (median 0.005 m). It
  is a real improvement and it is **not a fix** -- largest-area can still select a shelf or a lower
  tier, which is why the choice is recorded by id. The filter lives here rather than in the provider
  because the provider's order is what the shipping build already consumes -- see ``current.md`` for
  the recorded divergence.
* **the front edge is declared, not measured.** A rectangle is symmetric, and HSSD publishes every
  surface twice (``obb`` and its ``modelObb`` mirror, under a 180-degree ``modelToWorld``). Which
  side of it faces a room is therefore not recoverable from this data. The ``min y`` edge is taken as
  the front -- the same assumption the category envelope already made -- and it rides on the record
  as ``front_edge_basis`` so a reader can see that it was assumed.
"""

from __future__ import annotations

from typing import Sequence

from rfw_w.scenes.assets.providers.hssd import (
    HssdConfig,
    PreparedFurniture,
    SupportSurfaceReading,
    read_support_surfaces,
    support_polygons_from_readings,
)
from rfw_w.scenes.contracts.asset import MeasuredSupport, SupportPolygon


class SupportMeasurementError(RuntimeError):
    """The library cannot state a usable support surface for this asset."""


# The axis a horizontal support's normal points along, in the asset's own Y-up frame. Named because
# the whole filter turns on it and a bare "y" in a comparison is the kind of literal that gets
# changed without anyone seeing what it decides.
HORIZONTAL_AXIS = "y"

# A surface narrower than a millimetre is degenerate -- a sliver the OBB calculation can produce
# from a face set that is not a surface at all. Refused rather than placed on.
MIN_EXTENT_M = 0.001

# How close two rectangles must be to be called the same one. Both sides of the comparison come
# from six-decimal prepared records, so this is arithmetic slack, not a fudge factor.
MEASUREMENT_TOLERANCE_M = 1e-6


def select_support_surface(readings: Sequence[SupportSurfaceReading]) -> int | None:
    """The position of the surface the task cell should protect, or ``None`` if there is none.

    Largest area first, which is the order ``read_support_surfaces`` already returns and the same
    rank the shipping build uses -- but over the eligible subset only. A vertical panel is not a
    surface anything can stand on, however much area it has.
    """
    for index, reading in enumerate(readings):
        if reading.normal_axis == HORIZONTAL_AXIS:
            return index
    return None


def measure_support_from_readings(
    readings: Sequence[SupportSurfaceReading],
    *,
    asset_id: str,
    source_uri: str,
    index: int | None = None,
    base_offset_m: float = 0.0,
) -> MeasuredSupport:
    """Turn one reading into a cell-frame support, without touching the filesystem.

    Kept separate from the reading so it can be tested against synthetic surfaces -- the frame
    conversion is the part of this module most worth pinning numerically, and it should not need an
    asset library to exercise.
    """
    if not readings:
        raise SupportMeasurementError(f"HSSD_SUPPORT_SURFACE_ABSENT({asset_id}): no support surface")
    chosen = select_support_surface(readings) if index is None else index
    if chosen is None:
        axes = sorted({reading.normal_axis for reading in readings})
        raise SupportMeasurementError(
            f"HSSD_SUPPORT_SURFACE_HORIZONTAL_ABSENT({asset_id}): the asset states "
            f"{len(readings)} surface(s) and none is horizontal (normal axes: {axes}); a surface "
            "that is not horizontal is a panel, and a target cannot stand on one"
        )
    reading = readings[chosen]
    if reading.extent_x < MIN_EXTENT_M or reading.extent_z < MIN_EXTENT_M:
        raise SupportMeasurementError(
            f"HSSD_SUPPORT_SURFACE_DEGENERATE({asset_id}): surface {chosen} measures "
            f"{reading.extent_x:.6f} x {reading.extent_z:.6f} m, below the {MIN_EXTENT_M} m floor"
        )
    height = reading.plane_height_y + base_offset_m
    if height <= 0:
        raise SupportMeasurementError(
            f"HSSD_SUPPORT_SURFACE_INVALID({asset_id}): surface {chosen} states height "
            f"{height:.6f} m above the asset's base plane, which is not a height"
        )

    # The OBB-to-rectangle construction is the provider's, so there is one definition of what an
    # OBB extent becomes in the prepared Z-up frame. A single reading is passed, so the polygon
    # comes back labelled ":surface0" whatever its position in the file; the record keeps the
    # position instead, which is the fact a reader needs to go and look.
    polygon = support_polygons_from_readings(
        (reading,), asset_id=asset_id, source_uri=source_uri, offset=(0.0, 0.0, base_offset_m),
    )[0]
    vertices = _cell_frame_vertices(polygon.vertices_xy)
    return MeasuredSupport(
        asset_id=asset_id,
        source_polygon_id=f"{asset_id}:surface{chosen}",
        vertices_xy=vertices,
        support_height_m=height,
        provenance=polygon.provenance,
    )


def measure_support_footprint(
    config: HssdConfig, asset_id: str, *, base_offset_m: float = 0.0,
) -> MeasuredSupport:
    """The support surface of one asset, read from the library's own annotation."""
    readings = read_support_surfaces(config, asset_id)
    return measure_support_from_readings(
        readings,
        asset_id=asset_id,
        source_uri=str(config.support_surface_path(asset_id)),
        base_offset_m=base_offset_m,
    )


def assert_measurement_matches_prepared(
    measurement: MeasuredSupport, prepared: PreparedFurniture,
) -> None:
    """Check that the surface the cell protects is the surface the scene will materialise.

    The two come from different calls: the cell is handed a measurement read from the sidecar, while
    the scene places the target on a polygon that ``prepare_furniture`` produced, which carries the
    mesh's own normalisation offset. They agree only if the offset is zero *and* both sides chose
    the same surface -- neither of which is guaranteed by construction.

    Compared by **geometry, not by id**: the prepared polygons are labelled by their position among
    the well-formed readings, and a reading the provider skipped would shift every label after it,
    so a matching id is not evidence of a matching surface. The rectangles are compared instead.
    """
    if not prepared.support_polygons:
        raise SupportMeasurementError(
            f"HSSD_SUPPORT_SURFACE_ABSENT({measurement.asset_id}): the prepared asset carries no "
            "support polygon, so the surface the cell protects is not the one the scene places on"
        )
    wanted = _rounded_vertices(measurement.vertices_xy)
    candidates = {
        _rounded_vertices(_cell_frame_vertices(polygon.vertices_xy))
        for polygon in prepared.support_polygons
    }
    if wanted not in candidates:
        nearest = min(
            candidates,
            key=lambda item: max(
                abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(item, wanted)
            ),
        )
        worst = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(nearest, wanted))
        raise SupportMeasurementError(
            f"PREPARED_SUPPORT_DISAGREES_WITH_MEASUREMENT({measurement.asset_id}): the cell "
            f"protects {wanted} and the closest surface the prepared asset carries is {nearest}, "
            f"off by {worst:.6f} m (tolerance {MEASUREMENT_TOLERANCE_M} m)"
        )


def _cell_frame_vertices(
    vertices_xy: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Move a surface rectangle into the cell frame the core contracts for.

    x is centred on the surface and the ``min y`` edge is placed at ``y = 0``, so the body lies
    toward ``+y`` and the robot aisle at ``y < 0`` -- the convention ``solve_cell`` states and the
    one the category envelope already kept.

    Anchoring on the polygon rather than on the asset's origin is deliberate: the asset origin is
    where the *mesh* was centred, which for a pedestal desk is not where its top is, and it is also
    the coordinate the library's published mirror could move. The surface's own rectangle cannot be
    moved by either.
    """
    xs = [x for x, _ in vertices_xy]
    ys = [y for _, y in vertices_xy]
    dx = (min(xs) + max(xs)) / 2.0
    dy = min(ys)
    return tuple((round(x - dx, 6), round(y - dy, 6)) for x, y in vertices_xy)


def _rounded_vertices(
    vertices_xy: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    return tuple((round(x, 6), round(y, 6)) for x, y in vertices_xy)


def support_polygon_of(measurement: MeasuredSupport) -> SupportPolygon:
    """The same surface as a ``SupportPolygon``, for callers that speak that vocabulary.

    Present because the scene-side code already consumes ``SupportPolygon``; it keeps the two
    records from drifting by making the conversion one function rather than an idiom repeated at
    each call site.
    """
    return SupportPolygon(
        polygon_id=measurement.source_polygon_id,
        vertices_xy=measurement.vertices_xy,
        plane_z_m=measurement.support_height_m,
        provenance=measurement.provenance,
    )
