# SPDX-License-Identifier: Apache-2.0
"""Clearance checks: whether a footprint can be somewhere without fouling something else.

The acceptance manual sets the initial engineering margin at 0.05 m on top of the robot's own
geometry, and requires that the passage from the requested entrance to every approach pose be
demonstrably legal. Both are computed here from real polygons rather than from bounding boxes,
because a corridor that is wide in its bounding box can still be blocked by an L-shaped desk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from rfw_w.scenes.geometry.polygon import (
    Polygon,
    contains_point,
    contains_polygon,
    distance_to_boundary,
    offset_outward,
    overlaps,
)

# Initial engineering margin from the acceptance manual §6. Configured, not universal: it is
# calibrated on the development set and reported with every measurement.
DEFAULT_MARGIN_M = 0.05


class ClearanceError(ValueError):
    """The clearance request is not well formed."""


@dataclass(frozen=True, slots=True)
class ClearanceResult:
    """Outcome of a single clearance query, with the numbers behind the verdict."""

    ok: bool
    required_margin_m: float
    measured_margin_m: float
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "required_margin_m": self.required_margin_m,
            "measured_margin_m": self.measured_margin_m,
            "reason": self.reason,
        }


def check_clearance(
    footprint: Polygon,
    *,
    region: Polygon | None = None,
    obstacles: Sequence[Polygon] = (),
    margin_m: float = DEFAULT_MARGIN_M,
) -> ClearanceResult:
    """Whether ``footprint`` sits inside ``region`` and clear of every obstacle by ``margin_m``.

    Intersection and insufficient margin are reported as different failures with different
    signs, because they call for different repairs: an intersection means the placement is
    wrong, while a missed margin means it is nearly right and a neighbouring candidate may
    fit. Reporting both as "overlaps" would send the repair loop after the wrong fix, and
    would hide how close a candidate came.
    """
    if margin_m < 0:
        raise ClearanceError(f"margin must be non-negative, got {margin_m}")

    if region is not None:
        grown = footprint if margin_m == 0 else _grown(footprint, margin_m)
        if not contains_polygon(region, grown):
            return ClearanceResult(
                ok=False,
                required_margin_m=margin_m,
                measured_margin_m=_margin_to_region(region, footprint),
                reason=(
                    "footprint with margin is not contained in the allowed region"
                    if contains_polygon(region, footprint)
                    else "footprint is not contained in the allowed region"
                ),
            )

    # Real intersection is tested on the footprint itself, never on the grown one: the grown
    # shape overlapping is exactly what an insufficient margin looks like.
    for index, obstacle in enumerate(obstacles):
        if overlaps(footprint, obstacle):
            return ClearanceResult(
                ok=False,
                required_margin_m=margin_m,
                measured_margin_m=-_overlap_depth(footprint, obstacle),
                reason=f"overlaps obstacle {index}",
            )

    if not obstacles:
        return ClearanceResult(ok=True, required_margin_m=margin_m, measured_margin_m=math.inf)

    best = min(_min_distance(footprint, obstacle) for obstacle in obstacles)
    if best < margin_m:
        return ClearanceResult(
            ok=False,
            required_margin_m=margin_m,
            measured_margin_m=best,
            reason=f"clearance {best:.4f} m is below the required {margin_m:.4f} m",
        )
    return ClearanceResult(ok=True, required_margin_m=margin_m, measured_margin_m=best)


def _grown(footprint: Polygon, margin_m: float) -> Polygon:
    return offset_outward(footprint, margin_m)


def gap_between(first: Polygon, second: Polygon) -> float:
    """Smallest distance between two polygons; ``0.0`` when they touch or overlap.

    The complement of what :func:`check_clearance` answers: "does this fit" is a yes or no, while
    connectivity between regions is a question about *meeting*, and the answer to that is a
    distance of zero rather than a boolean that has to decide what touching means.
    """
    if overlaps(first, second):
        return 0.0
    return _min_distance(first, second)


def _min_distance(first: Polygon, second: Polygon) -> float:
    """Smallest gap between two disjoint polygons, edge to edge."""
    best = math.inf
    for start, end in first.edges:
        for other_start, other_end in second.edges:
            best = min(best, _segment_distance(start, end, other_start, other_end))
    if best is math.inf:
        # One contains the other: the gap is the smaller of the two boundary distances.
        best = min(
            min(distance_to_boundary(second, vertex) for vertex in first.vertices),
            min(distance_to_boundary(first, vertex) for vertex in second.vertices),
        )
    return best


def _segment_distance(a1: tuple[float, float], a2: tuple[float, float],
                      b1: tuple[float, float], b2: tuple[float, float]) -> float:
    """Minimum distance between two segments, including the degenerate cases."""
    if _segments_cross(a1, a2, b1, b2):
        return 0.0
    return min(
        _point_segment_distance(a1, b1, b2),
        _point_segment_distance(a2, b1, b2),
        _point_segment_distance(b1, a1, a2),
        _point_segment_distance(b2, a1, a2),
    )


def _segments_cross(a1, a2, b1, b2) -> bool:
    def orientation(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orientation(b1, b2, a1), orientation(b1, b2, a2)
    d3, d4 = orientation(a1, a2, b1), orientation(a1, a2, b2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _point_segment_distance(point, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
    return math.dist(point, (start[0] + t * dx, start[1] + t * dy))


def _overlap_depth(first: Polygon, second: Polygon) -> float:
    """How far one polygon reaches into the other, as a depth rather than a boolean.

    Uses the inward distance from the first polygon's nearest vertex to the second; enough to
    rank candidates, and deliberately not claimed to be an exact penetration volume.
    """
    candidates: list[float] = []
    for vertex in first.vertices:
        if contains_point(second, vertex):
            candidates.append(distance_to_boundary(second, vertex))
    for vertex in second.vertices:
        if contains_point(first, vertex):
            candidates.append(distance_to_boundary(first, vertex))
    return max(candidates) if candidates else 0.0


def _margin_to_region(region: Polygon, footprint: Polygon) -> float:
    """Signed clearance between the footprint and the region boundary.

    Negative when the footprint reaches outside. Measured to the footprint's vertices, which
    is exact against a convex boundary and a slight overestimate against a reflex one; the
    number is a ranking aid, not a proof, and the containment test above it is the actual gate.
    """
    outside = 0.0
    for vertex in footprint.vertices:
        if not contains_point(region, vertex):
            outside = max(outside, distance_to_boundary(region, vertex))
    if outside > 0.0:
        return -outside
    return min(distance_to_boundary(region, vertex) for vertex in footprint.vertices)
