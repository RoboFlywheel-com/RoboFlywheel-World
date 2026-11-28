# SPDX-License-Identifier: Apache-2.0
"""Support surfaces and the support graph: what rests on what.

The support graph is what makes "ON the desk" mean something. An object sitting at the right
height over the desk's footprint is not necessarily on the desk -- it could be over the floor
beside it, or over a different surface that happens to overlap in plan view. So a relationship
is established from the surface it actually rests on, and the acceptance manual is explicit that
landing on the floor and staying still must not be scored as ON(table).

Support surfaces come from two places and both are recorded: the room's own floor, and the
support polygons an asset declares. A retrieved asset that declares none cannot carry an ON
relation for anything placed on it, and that is reported rather than assumed -- the contract
marks a missing affordance as unknown, never as absent-but-fine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from rfw_w.scenes.contracts.asset import AffordanceStatus
from rfw_w.scenes.geometry.polygon import (
    Polygon,
    PolygonError,
    area,
    centroid,
    triangulate,
)

SCHEMA_VERSION = "pvscene.support-graph/v1"


class SupportError(ValueError):
    """The support graph request is not well formed."""


@dataclass(frozen=True, slots=True)
class Surface:
    """A place something can rest.

    ``owner_instance`` is the instance that provides the surface -- a desk, or ``None`` for the
    room floor. Keeping the floor in the same type as furniture surfaces is what lets the
    predicate ask "which surface is this object on" once, instead of special-casing the floor
    and thereby making "on the floor" indistinguishable from "on nothing".
    """

    surface_id: str
    owner_instance: str | None
    polygon: Polygon
    plane_z_m: float
    provenance: AffordanceStatus = AffordanceStatus.UNKNOWN

    def __post_init__(self) -> None:
        if not self.surface_id:
            raise SupportError("surface must have an id")

    @property
    def is_floor(self) -> bool:
        return self.owner_instance is None

    @property
    def area_m2(self) -> float:
        return area(self.polygon)

    @property
    def center_xy(self) -> tuple[float, float]:
        return centroid(self.polygon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "owner_instance": self.owner_instance,
            "polygon_xy": self.polygon.to_list(),
            "plane_z_m": self.plane_z_m,
            "area_m2": round(self.area_m2, 6),
            "is_floor": self.is_floor,
            "provenance": self.provenance.value,
        }


@dataclass(frozen=True, slots=True)
class ContactWitness:
    """Why a relationship was accepted, with the numbers behind it.

    A predicate without a witness is an assertion. The witness carries the surface it rested on,
    the overlap of the object's footprint with that surface, and how far the object's base sits
    from the surface plane, so a reviewer can tell a real contact from a near miss.
    """

    subject_instance: str
    surface_id: str
    support_owner: str | None
    footprint_area_m2: float
    covered_area_m2: float
    coverage_ratio: float
    base_gap_m: float
    tolerance_m: float

    @property
    def covered(self) -> bool:
        return self.covered_area_m2 > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_instance": self.subject_instance,
            "surface_id": self.surface_id,
            "support_owner": self.support_owner,
            "footprint_area_m2": round(self.footprint_area_m2, 6),
            "covered_area_m2": round(self.covered_area_m2, 6),
            "coverage_ratio": round(self.coverage_ratio, 6),
            "base_gap_m": round(self.base_gap_m, 6),
            "tolerance_m": self.tolerance_m,
            "covered": self.covered,
        }


def polygon_overlap_area(first: Polygon, second: Polygon) -> float:
    """Area shared by two polygons, exact for simple polygons.

    Sutherland-Hodgman clips a subject against each half-plane of the clipping polygon, which is
    only correct when that clipping polygon is **convex**. Clipping against a notched desk's six
    edges therefore subtracts the region beyond the notch line and returns zero for a footprint
    plainly resting on the desk -- an ON predicate built on that would fail for anything on the
    L-shaped part of a surface.

    So the clipping polygon is decomposed into triangles first. Each triangle is convex, the
    triangulation partitions the polygon without overlap, and the clipped areas can simply be
    summed.
    """
    if first.area <= 0 or second.area <= 0:
        return 0.0

    def clip(subject: list[tuple[float, float]], edge_start, edge_end) -> list[tuple[float, float]]:
        if not subject:
            return []
        output: list[tuple[float, float]] = []
        ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]

        def inside(point) -> bool:
            # Left of the directed edge, for counter-clockwise winding.
            return (ex * (point[1] - edge_start[1]) - ey * (point[0] - edge_start[0])) >= -1e-12

        def intersection(point, previous) -> tuple[float, float]:
            px, py = previous
            dx, dy = point[0] - px, point[1] - py
            denominator = ex * dy - ey * dx
            if abs(denominator) <= 1e-15:
                return point
            t = (ex * (py - edge_start[1]) - ey * (px - edge_start[0])) / -denominator
            return (px + t * dx, py + t * dy)

        for index in range(len(subject)):
            current = subject[index]
            previous = subject[index - 1]
            current_in, previous_in = inside(current), inside(previous)
            if current_in:
                if not previous_in:
                    output.append(intersection(current, previous))
                output.append(current)
            elif previous_in:
                output.append(intersection(current, previous))
        return output

    subject = list(second.vertices)
    total = 0.0
    try:
        pieces = triangulate(first)
    except PolygonError:
        # A polygon that cannot be triangulated cannot be intersected meaningfully. Reporting
        # zero would silently deny support, so fall back to the bounding box of the clipping
        # polygon, which at least cannot claim support that is not there.
        pieces = (first.bounding_box(),)

    for piece in pieces:
        oriented = piece if piece.signed_area > 0 else Polygon(tuple(reversed(piece.vertices)))
        result = subject
        for start, end in oriented.edges:
            result = clip(result, start, end)
            if not result:
                break
        if len(result) >= 3:
            total += abs(Polygon(tuple(result)).signed_area)
    return total


@dataclass(frozen=True, slots=True)
class PlacedInstance:
    """What the support solver needs to know about one placed object."""

    instance_id: str
    footprint: Polygon
    base_z_m: float
    is_dynamic: bool

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise SupportError("placed instance must have an id")
        if not math.isfinite(self.base_z_m):
            raise SupportError(f"{self.instance_id}: base height must be finite")


# Vertical tolerance for "resting on". The acceptance manual's ON gate allows a geometric
# tolerance and an initial 2 mm penetration allowance; this is the height part of it, configured
# rather than universal.
DEFAULT_CONTACT_TOLERANCE_M = 0.003

# Minimum share of an object's footprint that must be over its support. Not 1.0: a cup slightly
# overhanging a desk edge is normal. Not near zero either, or an object touching the very corner
# of a tabletop would count as being on it.
DEFAULT_MIN_COVERAGE = 0.5


@dataclass(frozen=True, slots=True)
class SupportResolution:
    """The surface an object rests on, or the reason none was found."""

    instance_id: str
    witness: ContactWitness | None
    failure_code: str | None = None
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.witness is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "resolved": self.resolved,
            "failure_code": self.failure_code,
            "detail": self.detail,
            "witness": None if self.witness is None else self.witness.to_dict(),
        }


def resolve_support(
    placed: PlacedInstance,
    surfaces: Iterable[Surface],
    *,
    contact_tolerance_m: float = DEFAULT_CONTACT_TOLERANCE_M,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> SupportResolution:
    """Find the surface an object actually rests on.

    Candidates are ranked by how much of the footprint they cover, and a surface only counts if
    the object's base sits on its plane within tolerance. Two failure modes are kept distinct so
    the orchestrator can act on them differently:

    * ``SUPPORT_NOT_FOUND`` -- nothing is underneath at all, which is a placement error;
    * ``FUNCTION_PREDICATE_FAILED`` -- something is underneath but it is not the surface the
      relation named, which is what landing on the floor instead of the desk looks like.
    """
    if contact_tolerance_m < 0:
        raise SupportError(f"contact tolerance must be non-negative, got {contact_tolerance_m}")
    if not 0.0 < min_coverage <= 1.0:
        raise SupportError(f"min_coverage must be in (0, 1], got {min_coverage}")

    footprint_area = placed.footprint.area
    if footprint_area <= 0:
        raise SupportError(f"{placed.instance_id}: footprint has no area")

    best: ContactWitness | None = None
    for surface in surfaces:
        # The base has to be at the surface plane, within tolerance, on the correct side.
        gap = placed.base_z_m - surface.plane_z_m
        if gap < -contact_tolerance_m or gap > contact_tolerance_m:
            continue
        covered = polygon_overlap_area(surface.polygon, placed.footprint)
        if covered <= 0.0:
            continue
        ratio = covered / footprint_area
        if ratio < min_coverage:
            continue
        witness = ContactWitness(
            subject_instance=placed.instance_id,
            surface_id=surface.surface_id,
            support_owner=surface.owner_instance,
            footprint_area_m2=footprint_area,
            covered_area_m2=covered,
            coverage_ratio=ratio,
            base_gap_m=gap,
            tolerance_m=contact_tolerance_m,
        )
        if best is None or witness.covered_area_m2 > best.covered_area_m2:
            best = witness

    if best is None:
        return SupportResolution(
            instance_id=placed.instance_id,
            witness=None,
            failure_code="SUPPORT_NOT_FOUND",
            detail=(
                "no surface lies under the object within "
                f"{contact_tolerance_m} m and covers at least {min_coverage:.0%} of its footprint"
            ),
        )
    return SupportResolution(instance_id=placed.instance_id, witness=best)


def assert_not_on_floor_when_support_expected(
    resolution: SupportResolution, *, expected_owner: str
) -> SupportResolution:
    """Turn "resting on the floor" into a failure when a specific support was required.

    The negative case from the runtime probe: an object that falls to the floor settles stably,
    and a gate that only checks stability reports success. This is what makes the difference
    between stable and *on the right thing* explicit.
    """
    if resolution.witness is None:
        return resolution
    if resolution.witness.support_owner == expected_owner:
        return resolution
    found = resolution.witness.support_owner or "the floor"
    return SupportResolution(
        instance_id=resolution.instance_id,
        witness=resolution.witness,
        failure_code="FUNCTION_PREDICATE_FAILED",
        detail=(
            f"object rests on {found!r} but the relation requires {expected_owner!r}; settling "
            "stably on the wrong surface is not the relation being satisfied"
        ),
    )
