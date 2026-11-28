# SPDX-License-Identifier: Apache-2.0
"""Planar polygons, in the XY plane at a given height.

Rooms and support surfaces are polygons, not rectangles. The acceptance manual is specific
about why: an L-shaped room or a notched desk cannot be described by a bounding box, and a
box stand-in passes containment checks the real shape would fail -- so an object ends up
placed in the notch, floating, and the geometry gate reports success because the point was
inside the box.

Everything here works on explicit vertex lists and is exact within floating-point tolerance.
Self-intersecting input is rejected rather than silently producing a wrong area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

TOLERANCE = 1e-9
# Below this area a clipped piece is floating-point debris rather than a region. Keeping such
# pieces would multiply the count without adding area, and their vertices are usually collinear.
AREA_EPSILON = 1e-9


class PolygonError(ValueError):
    """The polygon is not well formed for the requested operation."""


Point = tuple[float, float]


def _as_point(value: Sequence[float]) -> Point:
    items = list(value)
    if len(items) != 2:
        raise PolygonError(f"expected an (x, y) pair, got {len(items)} components")
    x, y = float(items[0]), float(items[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise PolygonError(f"coordinates must be finite, got {value!r}")
    return (x, y)


@dataclass(frozen=True, slots=True)
class Polygon:
    """A closed planar polygon. The closing vertex is implicit; do not repeat the first one."""

    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise PolygonError(f"a polygon needs at least 3 vertices, got {len(self.vertices)}")
        for vertex in self.vertices:
            _as_point(vertex)

    @classmethod
    def from_sequence(cls, values: Iterable[Sequence[float]]) -> Polygon:
        return cls(tuple(_as_point(value) for value in values))

    @classmethod
    def rectangle(cls, min_x: float, min_y: float, max_x: float, max_y: float) -> Polygon:
        if min_x >= max_x or min_y >= max_y:
            raise PolygonError(f"degenerate rectangle: ({min_x}, {min_y})-({max_x}, {max_y})")
        return cls(((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)))

    @property
    def signed_area(self) -> float:
        """Shoelace area. Positive for counter-clockwise winding."""
        total = 0.0
        count = len(self.vertices)
        for index in range(count):
            x1, y1 = self.vertices[index]
            x2, y2 = self.vertices[(index + 1) % count]
            total += x1 * y2 - x2 * y1
        return total / 2.0

    @property
    def area(self) -> float:
        return abs(self.signed_area)

    def to_list(self) -> list[list[float]]:
        return [[x, y] for x, y in self.vertices]

    @property
    def edges(self) -> tuple[tuple[Point, Point], ...]:
        count = len(self.vertices)
        return tuple(
            (self.vertices[index], self.vertices[(index + 1) % count]) for index in range(count)
        )

    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)``."""
        xs = [x for x, _ in self.vertices]
        ys = [y for _, y in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    def bounding_box(self) -> Polygon:
        min_x, min_y, max_x, max_y = self.bounds()
        return Polygon.rectangle(min_x, min_y, max_x, max_y)

    def is_rectangle(self, tolerance: float = 1e-9) -> bool:
        """Whether this is an axis-aligned rectangle, not merely four vertices.

        Used to flag where a rectangle is standing in for a real shape, so the loss of
        fidelity is visible in a report rather than assumed away.
        """
        if len(self.vertices) != 4:
            return False
        xs = sorted({round(x, 9) for x, _ in self.vertices})
        ys = sorted({round(y, 9) for _, y in self.vertices})
        return len(xs) == 2 and len(ys) == 2

    def area_ratio_to_bounding_box(self) -> float:
        """How much of the bounding box the polygon actually fills.

        The measure that catches a notched shape passed off as a rectangle: an L-shaped room
        filling 70% of its box cannot be treated as the box.
        """
        box = self.bounding_box()
        if box.area <= 0:
            return 0.0
        return self.area / box.area


def validate_simple(polygon: Polygon, *, tolerance: float = TOLERANCE) -> None:
    """Reject self-intersecting or degenerate polygons.

    A self-intersecting outline has a shoelace area that means nothing, and every containment
    test built on it inherits the error. Failing here is far cheaper than a placed object
    whose supporting surface was computed from a bow-tie.
    """
    edges = polygon.edges
    for index, (start_a, end_a) in enumerate(edges):
        if math.dist(start_a, end_a) <= tolerance:
            raise PolygonError(f"polygon has a zero-length edge at vertex {index}")
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue  # adjacent edges legitimately share a vertex
            if _segments_intersect(edges[i], edges[j], tolerance=tolerance):
                raise PolygonError(
                    f"polygon is self-intersecting: edge {i} crosses edge {j}"
                )
    if polygon.area <= tolerance:
        raise PolygonError(f"polygon has no area ({polygon.area})")


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    first: tuple[Point, Point], second: tuple[Point, Point], *, tolerance: float = TOLERANCE
) -> bool:
    a, b = first
    c, d = second
    d1 = _orientation(c, d, a)
    d2 = _orientation(c, d, b)
    d3 = _orientation(a, b, c)
    d4 = _orientation(a, b, d)
    if ((d1 > tolerance) != (d2 > tolerance)) and ((d3 > tolerance) != (d4 > tolerance)):
        if abs(d1) <= tolerance or abs(d2) <= tolerance or abs(d3) <= tolerance or abs(d4) <= tolerance:
            return False  # touching at an endpoint is not a crossing
        return True
    return False


def contains_point(polygon: Polygon, point: Sequence[float], *, tolerance: float = TOLERANCE) -> bool:
    """Ray casting, with boundary points treated as inside.

    Boundary counts as inside because a support surface's edge is still support; the
    clearance and containment checks that need a margin apply one explicitly rather than
    relying on a half-open test.
    """
    x, y = _as_point(point)
    if distance_to_boundary(polygon, (x, y)) <= tolerance:
        return True

    inside = False
    count = len(polygon.vertices)
    for index in range(count):
        x1, y1 = polygon.vertices[index]
        x2, y2 = polygon.vertices[(index + 1) % count]
        # Half-open rule on y avoids double-counting a vertex that lies exactly on the ray.
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def contains_polygon(polygon: Polygon, other: Polygon, *, tolerance: float = TOLERANCE) -> bool:
    """Whether ``other`` lies wholly inside ``polygon``.

    Checks vertices *and* edge crossings: a large rectangle can have all its vertices outside
    a C-shaped polygon while crossing it, and testing vertices alone would call that contained.
    """
    for vertex in other.vertices:
        if not contains_point(polygon, vertex, tolerance=tolerance):
            return False
    for edge in other.edges:
        for boundary in polygon.edges:
            if _segments_intersect(edge, boundary, tolerance=tolerance):
                return False
    return True


def distance_to_boundary(polygon: Polygon, point: Sequence[float]) -> float:
    """Euclidean distance from a point to the nearest edge."""
    x, y = _as_point(point)
    best = math.inf
    for (x1, y1), (x2, y2) in polygon.edges:
        dx, dy = x2 - x1, y2 - y1
        length_squared = dx * dx + dy * dy
        if length_squared == 0.0:
            best = min(best, math.dist((x, y), (x1, y1)))
            continue
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_squared))
        best = min(best, math.dist((x, y), (x1 + t * dx, y1 + t * dy)))
    return best


def area(polygon: Polygon) -> float:
    return polygon.area


def centroid(polygon: Polygon) -> Point:
    """Area centroid. Falls back to the vertex mean for a degenerate outline."""
    signed = polygon.signed_area
    if abs(signed) <= TOLERANCE:
        xs = [x for x, _ in polygon.vertices]
        ys = [y for _, y in polygon.vertices]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    factor = 1.0 / (6.0 * signed)
    cx = cy = 0.0
    count = len(polygon.vertices)
    for index in range(count):
        x1, y1 = polygon.vertices[index]
        x2, y2 = polygon.vertices[(index + 1) % count]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx * factor, cy * factor)


def convex_hull(points: Iterable[Sequence[float]], *, tolerance: float = TOLERANCE) -> Polygon:
    """The smallest convex polygon containing every point, counter-clockwise.

    Needed to take a *plan* reading off a piece that is not axis-aligned in the world. An object
    with an arbitrary rotation is a box whose eight corners project to at most a hexagon on the
    floor, so the outline that a containment or overlap test must use is the hull of those
    projections -- four corners are only enough when the piece is upright and merely yawed.

    Monotone chain: sort, then walk the lower and upper hulls. Collinear points are dropped, so a
    rectangle comes back as four vertices rather than with its edge midpoints included.
    """
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) < 3:
        raise PolygonError(f"a hull needs at least 3 distinct points, got {len(ordered)}")

    def build(sequence: Sequence[Point]) -> list[Point]:
        chain: list[Point] = []
        for point in sequence:
            while len(chain) >= 2:
                start, end = chain[-2], chain[-1]
                cross = ((end[0] - start[0]) * (point[1] - start[1])
                         - (end[1] - start[1]) * (point[0] - start[0]))
                if cross > tolerance:
                    break
                chain.pop()
            chain.append(point)
        return chain

    lower = build(ordered)
    upper = build(list(reversed(ordered)))
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise PolygonError("every point is collinear, so there is no hull with area")
    return Polygon(tuple(hull))


def _strictly_inside(polygon: Polygon, point: Sequence[float], *, tolerance: float = TOLERANCE) -> bool:
    """Inside the polygon and clear of its boundary.

    ``contains_point`` counts the boundary as inside, which is right for support coverage and
    wrong for overlap: two objects placed flush against each other share an edge and are legal.
    """
    return contains_point(polygon, point, tolerance=tolerance) and (
        distance_to_boundary(polygon, point) > tolerance
    )


def overlaps(first: Polygon, second: Polygon, *, tolerance: float = TOLERANCE) -> bool:
    """Whether two polygons share area.

    Touching edges do not count: two objects placed flush against each other are legal and
    common, so boundary contact is explicitly excluded rather than inherited from the
    containment test.
    """
    for edge in first.edges:
        for other_edge in second.edges:
            if _segments_intersect(edge, other_edge, tolerance=tolerance):
                return True
    # No crossing means containment is all that remains. It has to be *strict* containment,
    # or a pair sharing an edge would be reported as overlapping.
    if any(_strictly_inside(first, vertex, tolerance=tolerance) for vertex in second.vertices):
        return True
    return any(_strictly_inside(second, vertex, tolerance=tolerance) for vertex in first.vertices)


def _ccw(polygon: Polygon) -> Polygon:
    """Return the polygon with counter-clockwise winding, which the offset maths assumes."""
    return polygon if polygon.signed_area > 0 else Polygon(tuple(reversed(polygon.vertices)))


def _line_intersection(
    point_a: Point, direction_a: Point, point_b: Point, direction_b: Point
) -> Point:
    """Intersection of two infinite lines, each given as a point and a direction."""
    denominator = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(denominator) <= TOLERANCE:
        # Parallel edges meet at infinity; returning the shared point keeps a collinear vertex
        # where it is instead of producing a NaN.
        return point_a
    delta = (point_b[0] - point_a[0], point_b[1] - point_a[1])
    t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    return (point_a[0] + t * direction_a[0], point_a[1] + t * direction_a[1])


def offset_outward(polygon: Polygon, distance: float) -> Polygon:
    """Grow (positive) or shrink (negative) a polygon by moving every edge along its normal.

    Adjacent edges are re-intersected, which is exact at convex corners and correct at the
    reflex corners of an L-shape. Shrinking past the point where the outline vanishes raises
    rather than returning an inverted shape: a negative-area inset would silently satisfy
    containment tests it should fail, so a desk "reserved workspace" could end up larger than
    the desk.
    """
    if distance == 0.0:
        return polygon
    if not math.isfinite(distance):
        raise PolygonError(f"offset distance must be finite, got {distance}")

    ccw = _ccw(polygon)
    count = len(ccw.vertices)
    # Inward normal of a counter-clockwise edge (dx, dy) is (-dy, dx); moving outward is the
    # opposite direction, so the sign of `distance` selects which way the edges travel.
    shifted: list[tuple[Point, Point]] = []
    for index in range(count):
        start = ccw.vertices[index]
        end = ccw.vertices[(index + 1) % count]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= TOLERANCE:
            raise PolygonError(f"polygon has a zero-length edge at vertex {index}")
        # Outward normal of a counter-clockwise edge is (dy, -dx) / length.
        nx, ny = dy / length, -dx / length
        shifted.append(((start[0] + nx * distance, start[1] + ny * distance),
                        (dx, dy)))

    result: list[Point] = []
    for index in range(count):
        previous = shifted[index - 1]
        current = shifted[index]
        result.append(
            _line_intersection(previous[0], previous[1], current[0], current[1])
        )

    outset = Polygon(tuple(result))
    if distance < 0 and not contains_polygon(polygon, outset):
        # Containment, not area, is the correct test: shrinking past the collapse point
        # inverts the outline rather than emptying it, and an inverted polygon has a *larger*
        # absolute area than the original (a 4x3 rectangle inset by 5 m comes back as a
        # 6x7 m shape with the opposite winding). An area check therefore passes it through,
        # and the inverted outline then satisfies containment tests it should fail.
        raise PolygonError(
            f"shrinking by {abs(distance)} collapses or inverts the polygon; the requested "
            "inset is larger than the shape can accommodate"
        )
    if distance < 0:
        # Containment is not enough. Where the outline has a neck narrower than twice the inset,
        # the two shifted edges cross and the result is self-intersecting -- while every one of its
        # vertices can still lie inside the original, so the test above passes it through. Measured
        # on a T-shaped room whose middle is 1.079 m wide, inset by 0.75 m: the offset is a bow tie
        # inside the room. A non-simple polygon then fails much later and much less clearly, in
        # `triangulate`, where the message says nothing about the inset that caused it.
        validate_simple(outset)
    return outset


def inset(polygon: Polygon, distance: float) -> Polygon:
    """Shrink by ``distance``. The complement of :func:`offset_outward` with a negative offset."""
    if distance < 0:
        raise PolygonError(f"inset distance must be non-negative, got {distance}")
    return offset_outward(polygon, -distance)


def _ccw_order(polygon: Polygon) -> list[int]:
    """Indices into ``polygon.vertices`` that walk it counter-clockwise.

    Returned as indices rather than a new ``Polygon`` because two callers need different things
    from the same walk: ``triangulate`` wants the vertices, and ``triangulate_indices`` wants the
    triples expressed in the caller's own winding.
    """
    if polygon.signed_area > 0:
        return list(range(len(polygon.vertices)))
    count = len(polygon.vertices)
    return [count - 1 - index for index in range(count)]


def _ear_clip(vertices: Sequence[Point]) -> tuple[tuple[int, int, int], ...]:
    """Ear-clip a counter-clockwise simple polygon into index triples.

    Ear clipping rather than a fan from vertex 0, because a fan is only correct for an outline
    every vertex of which can see vertex 0. A room with a notch in its top-right corner happens to
    satisfy that; the same room with the notch in its top-left does not, and the fan then emits
    triangles covering the notch -- floor where there is no floor, which no polygon-level check
    can see because the polygon is right and only the *mesh* is wrong.
    """
    remaining = list(range(len(vertices)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0

    while len(remaining) > 3:
        guard += 1
        if guard > len(vertices) ** 2:
            raise PolygonError("ear clipping failed; the polygon is probably self-intersecting")
        ear_found = False
        for position in range(len(remaining)):
            previous = remaining[position - 1]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = vertices[previous], vertices[current], vertices[following]
            if _orientation(a, b, c) <= TOLERANCE:
                continue  # reflex or collinear: not an ear
            if any(
                _point_in_triangle(vertices[index], a, b, c)
                for index in remaining
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            remaining.pop(position)
            ear_found = True
            break
        if not ear_found:
            raise PolygonError("no ear found; the polygon is not simple")

    triangles.append((remaining[0], remaining[1], remaining[2]))
    return tuple(triangles)


def triangulate(polygon: Polygon) -> tuple[Polygon, ...]:
    """Split a simple polygon into triangles by ear clipping.

    Needed because the polygon intersection in the support solver clips against half-planes,
    which is only correct when the clipping polygon is convex. A notched desk is not, so it is
    decomposed into convex pieces first and the pieces are summed.

    Winding is normalised to counter-clockwise first; ear clipping on a clockwise polygon finds
    the wrong ears.
    """
    order = _ccw_order(polygon)
    ccw_vertices = [polygon.vertices[index] for index in order]
    return tuple(
        Polygon(tuple(ccw_vertices[index] for index in triangle))
        for triangle in _ear_clip(ccw_vertices)
    )


def triangulate_indices(polygon: Polygon) -> tuple[tuple[int, int, int], ...]:
    """Index triples into ``polygon.vertices``, counter-clockwise, covering it exactly.

    The form a mesh builder needs. Kept beside :func:`triangulate` rather than reimplemented in
    the mesh layer: two copies of one algorithm is the drift this module already refuses for
    constants, and here the drift would be silent -- a second fan would only wrong for the shapes
    the first one is wrong for, so a test of either would pass.
    """
    order = _ccw_order(polygon)
    ccw_vertices = [polygon.vertices[index] for index in order]
    return tuple(
        tuple(order[index] for index in triangle)
        for triangle in _ear_clip(ccw_vertices)
    )


def _point_in_triangle(point: Point, a: Point, b: Point, c: Point) -> bool:
    """Strict interior test, so a vertex shared with the triangle is not counted as inside it."""
    d1 = _orientation(a, b, point)
    d2 = _orientation(b, c, point)
    d3 = _orientation(c, a, point)
    has_negative = d1 < -TOLERANCE or d2 < -TOLERANCE or d3 < -TOLERANCE
    has_positive = d1 > TOLERANCE or d2 > TOLERANCE or d3 > TOLERANCE
    return not (has_negative and has_positive)


def _clean(points: Sequence[Point], *, tolerance: float = TOLERANCE) -> tuple[Point, ...]:
    """Drop repeated vertices so a clipped outline is a polygon rather than a list of duplicates.

    The half-plane clip emits a vertex twice whenever the clip boundary runs exactly through one,
    which is the common case here: zones are built from the same wall lines and so share edges.
    A repeated vertex is not a well-formed outline, and the area and containment tests built on
    it would mean nothing.
    """
    cleaned: list[Point] = []
    for point in points:
        if cleaned and math.dist(cleaned[-1], point) <= tolerance:
            continue
        cleaned.append(point)
    while len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) <= tolerance:
        cleaned.pop()
    return tuple(cleaned)


def _half_plane_clip(
    points: Sequence[Point], start: Point, end: Point, *, keep_left: bool = True,
    tolerance: float = TOLERANCE,
) -> list[Point]:
    """Sutherland-Hodgman step: keep the part of ``points`` on one side of a directed edge.

    ``keep_left`` keeps the counter-clockwise interior side, which is the side an edge bounds when
    the polygon is wound counter-clockwise; ``keep_left=False`` keeps the outside instead, which
    is what subtracting with this edge needs.
    """
    if not points:
        return []
    ex, ey = end[0] - start[0], end[1] - start[1]

    def signed_distance(point: Point) -> float:
        # Cross product of the edge with the point: positive on the left.
        return ex * (point[1] - start[1]) - ey * (point[0] - start[0])

    def inside(point: Point) -> bool:
        distance = signed_distance(point)
        return distance >= -tolerance if keep_left else distance <= tolerance

    def intersection(point: Point, previous: Point) -> Point:
        px, py = previous
        dx, dy = point[0] - px, point[1] - py
        denominator = ex * dy - ey * dx
        if abs(denominator) <= 1e-15:
            return point
        t = (ex * (py - start[1]) - ey * (px - start[0])) / -denominator
        return (px + t * dx, py + t * dy)

    output: list[Point] = []
    for index in range(len(points)):
        current = points[index]
        previous = points[index - 1]
        current_in, previous_in = inside(current), inside(previous)
        if current_in:
            if not previous_in:
                output.append(intersection(current, previous))
            output.append(current)
        elif previous_in:
            output.append(intersection(current, previous))
    return output


def _clip_to_convex(subject: Sequence[Point], clip: Polygon) -> tuple[Point, ...]:
    """The part of ``subject`` inside a **convex** ``clip``.

    Sutherland-Hodgman clips against every half-plane of the clipping polygon, which is only the
    same as clipping against that polygon when the polygon is convex. Against a notched shape it
    cuts the subject against the notch line as though the notch were an outer boundary and returns
    a region that was never shared. Triangles are convex, so the polygon-level operations below
    decompose instead of trusting an arbitrary clip.
    """
    ccw = _ccw(clip)
    points = list(subject)
    for start, end in ccw.edges:
        points = _half_plane_clip(points, start, end, keep_left=True)
        if not points:
            return ()
    return _clean(points)


def _bounds_touch(first: Polygon, second: Polygon) -> bool:
    """Whether two bounding boxes share area. Cheap, and conservative in the safe direction."""
    first_min_x, first_min_y, first_max_x, first_max_y = first.bounds()
    second_min_x, second_min_y, second_max_x, second_max_y = second.bounds()
    return not (
        second_max_x <= first_min_x or second_min_x >= first_max_x
        or second_max_y <= first_min_y or second_min_y >= first_max_y
    )


def is_convex(polygon: Polygon, *, tolerance: float = TOLERANCE) -> bool:
    """Whether every corner turns the same way.

    Reflex corners are exactly what a half-plane clip cannot tolerate, so this is the test that
    decides whether a polygon may be clipped directly or has to be decomposed first.
    """
    vertices = _ccw(polygon).vertices
    count = len(vertices)
    for index in range(count):
        if _orientation(vertices[index - 1], vertices[index], vertices[(index + 1) % count]) < -tolerance:
            return False
    return True


def intersect_polygons(first: Polygon, second: Polygon) -> tuple[Polygon, ...]:
    """The area two simple polygons share, cut into disjoint convex pieces.

    Exact for notched shapes: both polygons are triangulated first, and every triangle pair is
    intersected, since a triangle is always a legal clip polygon. The pieces are disjoint and
    their areas sum to the shared area -- which is what a single ``overlaps`` boolean cannot
    report, and what a bounding-box stand-in would overstate.

    Raises :class:`PolygonError` if either polygon cannot be triangulated, rather than reporting
    no shared area: a shape that cannot be decomposed is a broken input, and silently returning
    nothing would read as "these do not overlap".
    """
    if first.area <= 0 or second.area <= 0:
        return ()
    if contains_polygon(first, second):
        return (second,)
    if contains_polygon(second, first):
        return (first,)
    pieces: list[Polygon] = []
    for left in triangulate(first):
        for right in triangulate(second):
            points = _clip_to_convex(left.vertices, right)
            if len(points) < 3:
                continue
            piece = Polygon(points)
            if piece.area > AREA_EPSILON:
                pieces.append(piece)
    return tuple(pieces)


def convex_subtract(subject: Polygon, blocker: Polygon) -> tuple[Polygon, ...]:
    """``subject`` minus a **convex** ``blocker``, as pieces whose areas sum to the difference.

    Uses the standard exterior decomposition of a convex polygon: with the blocker's edges taken
    counter-clockwise, every point outside it is outside edge *i* while being inside all the edges
    before *i*, for exactly one *i*. Intersecting the subject with each of those convex regions in
    turn removes the blocker exactly, with no general clipping machinery and no approximation at
    the corners. A convex ``subject`` keeps every piece convex.
    """
    if subject.area <= 0:
        return ()
    if blocker.area <= 0:
        return (subject,)

    edges = _ccw(blocker).edges
    pieces: list[Polygon] = []
    for index, (start, end) in enumerate(edges):
        points = _half_plane_clip(list(subject.vertices), start, end, keep_left=False)
        for earlier in range(index):
            if not points:
                break
            before_start, before_end = edges[earlier]
            points = _half_plane_clip(points, before_start, before_end, keep_left=True)
        cleaned = _clean(points)
        if len(cleaned) < 3:
            continue
        piece = Polygon(cleaned)
        if piece.area > AREA_EPSILON:
            pieces.append(piece)
    return tuple(pieces)


def subtract_polygons(first: Polygon, second: Polygon) -> tuple[Polygon, ...]:
    """``first`` minus ``second``, as disjoint convex pieces.

    ``second`` may be any simple polygon. It is triangulated and the pieces are subtracted one
    triangle at a time; because every intermediate piece stays convex, the same exact convex
    routine applies throughout and nothing is approximated away at a notch. The areas of the
    returned pieces sum to ``first.area - second.area`` when the two overlap, and to
    ``first.area`` when they do not.
    """
    if first.area <= 0:
        return ()
    if second.area <= 0 or not _bounds_touch(first, second):
        # Disjoint, or touching only along an edge, where the difference is the subject unchanged.
        # The test is on bounding boxes rather than `overlaps` because that one deliberately
        # excludes flush contact, and two identical polygons are exactly that case.
        return (first,)
    if is_convex(first) and is_convex(second):
        # Both convex: one pass is exact, and the pieces keep their shape. Going through the
        # triangulations would return the same area as a fistful of triangles, which makes every
        # zone the layout has to reason about look far more complicated than it is.
        return convex_subtract(first, second)

    pieces: list[Polygon] = list(triangulate(first))
    for triangle in triangulate(second):
        remaining: list[Polygon] = []
        for piece in pieces:
            remaining.extend(convex_subtract(piece, triangle))
        pieces = remaining
        if not pieces:
            break
    return tuple(pieces)


def interior_point(polygon: Polygon, *, margin: float = 0.0) -> Point:
    """A point guaranteed to be inside the polygon, at least ``margin`` from any edge.

    Camera placement, zone origins and "somewhere in the room" defaults all need this, and doing it
    by arithmetic on the bounding box is how a camera ends up inside a notched corner -- outside the
    room, staring at the back of a wall. The centroid is tried first because it is stable and
    usually inside; an L-shape can pull it out, so a grid scan over the bounding box is the fallback
    and the result is verified rather than assumed.
    """
    def far_enough(point: Point) -> bool:
        return contains_point(polygon, point) and distance_to_boundary(polygon, point) >= margin

    centre = centroid(polygon)
    if far_enough(centre):
        return centre

    min_x, min_y, max_x, max_y = polygon.bounds()
    steps = 64
    best: tuple[float, Point] | None = None
    for i in range(steps + 1):
        for j in range(steps + 1):
            x = min_x + (max_x - min_x) * i / steps
            y = min_y + (max_y - min_y) * j / steps
            candidate = (x, y)
            if not far_enough(candidate):
                continue
            clearance = distance_to_boundary(polygon, candidate)
            if best is None or clearance > best[0]:
                best = (clearance, candidate)
    if best is None:
        raise PolygonError(
            f"no interior point found at least {margin} m from the boundary; the polygon is "
            "degenerate or the margin exceeds what it can hold"
        )
    return best[1]


def point_inside_toward(
    polygon: Polygon,
    target: Sequence[float],
    desired: Sequence[float],
    *,
    margin: float = 0.35,
    min_fraction: float = 0.25,
    max_fraction: float = 1.0,
    minimum_clearance_m: float = 0.0,
) -> Point | None:
    """A vantage point looking at ``target``, as close to ``desired`` as the room allows.

    Camera placement needs this. Working out an eye position by adding offsets to a target is how a
    camera ends up outside the room -- behind a wall, or inside a notched corner -- and a view from
    outside renders the back of a wall: non-blank, valid depth, and showing nothing the scene is
    about.

    Both directions along the target-to-desired line are searched, because the desired point is
    often on the far side of a wall: an object 0.15 m from the south wall can only be photographed
    from the north, and a search that insisted on the requested side would either return the object
    itself or give up. Whichever side yields more standoff wins, and the requested side breaks ties.

    ``minimum_clearance_m`` is the floor the fallback path may not go below, and ``None`` is a
    first-class answer: this either returns a point that clears it or reports that there is none.
    The fallback used to relax ``margin`` all the way to zero and end at
    ``interior_point(polygon, margin=0.05)``. With the **outline** as the polygon that is harmless,
    because 0.05 m inside the outline is still in the room -- but the outline is the wall's outer
    face, so against the inner face every one of those fallbacks returns a point inside the wall
    solid. Measured: a camera 0.076 m from the boundary, rendering one uniform surface. A caller
    that cannot stand a camera anywhere must be told, not handed a point in the wall.
    """
    t = _as_point(target)
    d = _as_point(desired)
    if not contains_point(polygon, t, tolerance=1e-9):
        return interior_point(polygon, margin=margin)

    def clearance(point: Point) -> float:
        return distance_to_boundary(polygon, point)

    def best_along(direction: tuple[float, float]) -> tuple[float, Point] | None:
        """The furthest usable point along a ray from the target, and its standoff."""
        best: tuple[float, Point] | None = None
        steps = 192
        for step in range(steps + 1):
            fraction = min_fraction + (max_fraction - min_fraction) * step / steps
            point = (t[0] + direction[0] * fraction, t[1] + direction[1] * fraction)
            if not contains_point(polygon, point) or clearance(point) < margin:
                continue
            standoff = math.dist(t, point)
            if best is None or standoff > best[0]:
                best = (standoff, point)
        return best

    length = math.dist(t, d)
    if length <= 1e-9:
        return interior_point(polygon, margin=margin)
    unit = ((d[0] - t[0]) / length, (d[1] - t[1]) / length)

    requested = best_along(unit)
    opposite = best_along((-unit[0], -unit[1]))
    if requested is None and opposite is None:
        # Too tight for `margin` anywhere. Relax toward `minimum_clearance_m` and stop there: below
        # it, the point is not somewhere a camera can stand, and returning one is what put three
        # views inside a wall.
        floor = max(0.0, minimum_clearance_m)
        for relaxed in (margin / 2, margin / 4, floor):
            if relaxed < floor:
                continue
            for direction in (unit, (-unit[0], -unit[1])):
                for step in range(192, -1, -1):
                    fraction = min_fraction + (max_fraction - min_fraction) * step / 192
                    point = (t[0] + direction[0] * fraction, t[1] + direction[1] * fraction)
                    if contains_point(polygon, point) and clearance(point) >= relaxed:
                        return point
        return None
    if requested is None:
        return opposite[1]
    if opposite is None:
        return requested[1]
    # Prefer the requested side unless the other gives materially more room to stand back.
    return requested[1] if requested[0] >= opposite[0] * 0.75 else opposite[1]
