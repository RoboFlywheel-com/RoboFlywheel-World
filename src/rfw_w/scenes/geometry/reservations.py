# SPDX-License-Identifier: Apache-2.0
"""Whether a piece of furniture *stands in* one of the task's reservations.

This module exists because two places ask that question and must never answer it differently:

* `rfw_w.scenes.workers.blender.measure_generated_furniture` (the **ruler**) measures a built scene after the
  fact, and its numbers are the ones every recorded intrusion claim in this repository rests on
  (0.1265 m^2 of protected region for the probe's shelf, 0.0400 m^2 for the office's two cabinets).
* `rfw_w.scenes.workers.blender.infinigen_constraint_patch` (the **rejection pool**) decides *during* the build
  which realised objects may stay in the room, and deletes the rest (SPEC-011 requirement 6,
  ADR-009 decision 8). Its judgement is what makes requirement 3's "zero intrusion" true of the
  scene the ruler then reads.

If the two held their own copy of the rule, a build could report "everything refused" while the
ruler still measured area -- and that disagreement would look exactly like an intruder. So the rule
lives here once, and the readers differ only in *which objects they hand in*: the ruler hands in
every object of the scene, the pool hands in the objects one candidate realises.

**The rule, in the order the details matter.**

1. **Footprint = the convex hull of the object's world vertices, projected to the floor.** Not the
   bounding box: a box swells a rotated bookcase into corners it never occupied, and that swelling
   is what a "did this piece intrude" number would report. Not `obj.bound_box` either -- these
   factories' local boxes are not the boxes they look like (a book column's local box measures zero
   in one axis, measured). A concave piece therefore gets its hull and its area is an upper bound;
   the pieces this stage generates (desks, shelves, lamps, books, cabinets, plants) are convex in
   plan, and an upper bound is the honest side to be wrong on when the claim is "it intruded".
2. **A region reserves a band from the floor, not a volume.** An object counts only when it
   *reaches down into* the region: `base_z < band_top_m`, where `base_z` is the lowest world vertex
   of the piece and the band is the region's own height (`protected` reaches the plan's support
   height, `circulation` and `door_sweep` reach 1.8 m -- ADR-009 decision 3). This is what keeps a
   pendant hanging 2.5 m above a desk, and the office's wall art 1.45 m above a 0.75 m desk, out of
   a region they visibly hang over rather than stand in.
3. **Overlap is the shared area of that hull and the region polygon**, summed over the region's
   polygons (a corridor is often several) and rounded to four decimals -- the precision every
   recorded number in this repository already has.

Like `polygon.py` this module is free of `bpy`: the objects arrive duck-typed (`matrix_world`,
`data.vertices`), which is also what lets it be unit-tested with stand-in objects and imported by
the pool from inside Blender without a second implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from rfw_w.scenes.geometry.polygon import Polygon, convex_hull, intersect_polygons

#: Every reported overlap is rounded here rather than by each caller: two readers of one rule that
#: round differently would disagree in the fourth decimal and read as a disagreement about geometry.
AREA_PLACES = 4


@dataclass(frozen=True, slots=True)
class Region:
    """One reserved region as this rule needs it: what it is called, where it is, how high it is.

    ``band_top_m`` is the region's own number: the protected region's top is the plan's
    `core.support_top_z_m` (which is why a *plane* is not enough), the corridor and the door sweep
    reach `CIRCULATION_BAND_TOP_M`/`DOOR_SWEEP_BAND_TOP_M`.
    """

    name: str
    polygon: Polygon
    band_top_m: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a reservation region needs a name")
        if not self.band_top_m > 0:
            raise ValueError(f"reservation region {self.name} needs a positive band")


@dataclass(frozen=True, slots=True)
class Part:
    """What one object measures against the regions: its footprint, its height, its overlaps.

    A *part* rather than a candidate because one piece of furniture can be several Blender objects
    (a factory that did not join its children) and the ruler measures each of them. The pool judges
    the whole tree and refuses it if any part intrudes -- which is the strictly stronger of the two,
    and the direction that keeps the pool from missing what the ruler can see.
    """

    name: str
    base_z_m: float
    top_z_m: float
    footprint_extent_m: tuple[float, float] | None
    overlaps_m2: Mapping[str, float]

    def in_band(self, region: Region) -> bool:
        return self.reaches(region.band_top_m)

    def reaches(self, band_top_m: float) -> bool:
        """Whether the part reaches down into a band of this height -- the band rule itself.

        Written once so a caller that has only the *number* (a report saying which band an object
        stands in, a record naming the band it was judged against) asks the same question as the
        caller that has the region.
        """
        return self.base_z_m < band_top_m

    @property
    def intrudes(self) -> bool:
        """True when this part overlaps any region it also reaches down into."""
        return any(value > 0.0 for value in self.overlaps_m2.values())

    def overlapping(self) -> tuple[tuple[str, float], ...]:
        """The (region, m^2) pairs that are not zero, in the region order the caller handed in."""
        return tuple((name, value) for name, value in self.overlaps_m2.items() if value > 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "base_z_m": round(self.base_z_m, AREA_PLACES),
                "top_z_m": round(self.top_z_m, AREA_PLACES),
                "footprint_extent_m": (None if self.footprint_extent_m is None
                                       else [round(value, AREA_PLACES)
                                             for value in self.footprint_extent_m]),
                "overlaps_m2": {name: value for name, value in self.overlaps_m2.items()}}


def world_vertices(obj: Any) -> tuple[tuple[float, float, float], ...]:
    """The object's vertices in world coordinates, or nothing when it has no mesh to measure."""
    data = getattr(obj, "data", None)
    vertices = getattr(data, "vertices", None)
    if data is None or vertices is None:
        return ()
    matrix = getattr(obj, "matrix_world", None)
    if matrix is None:
        return ()
    return tuple((float(point.x), float(point.y), float(point.z))
                 for point in (matrix @ vertex.co for vertex in vertices))


def footprint(points: Sequence[Sequence[float]]) -> Polygon | None:
    """The floor outline of a point cloud, or None when it has no area to report.

    None is a real answer for a piece with no floor area at all (a degenerate projection); a
    caller must not read it as "zero overlap by measurement".
    """
    if len(points) < 3:
        return None
    try:
        return convex_hull([(point[0], point[1]) for point in points])
    except Exception:  # noqa: BLE001 - every collinear/degenerate projection lands here
        return None


def overlap_m2(outline: Polygon, regions: Iterable[Region]) -> dict[str, float]:
    """How much of each region the outline covers, rounded to four decimals.

    Summed **per region name** and rounded once at the end, because a region is often several
    polygons (the bedroom probe's corridor is four rings): rounding each ring first and adding the
    results would drift in the fourth decimal from the number already recorded for that corridor.
    """
    total: dict[str, float] = {}
    for region in regions:
        total[region.name] = total.get(region.name, 0.0) + sum(
            piece.area for piece in intersect_polygons(outline, region.polygon))
    return {name: round(value, AREA_PLACES) for name, value in total.items()}


def measure_part(obj: Any, regions: Sequence[Region]) -> Part | None:
    """One object against the reservations: the band test first, then the overlap.

    The order is the rule, not an optimisation: an object that does not reach into a region's band
    has *zero* overlap with it whatever its footprint does, so the band decides which regions are
    asked at all. An object with no measurable mesh is `None` -- not a part with zero overlap.
    """
    points = world_vertices(obj)
    if not points:
        return None
    outline = footprint(points)
    base_z = min(point[2] for point in points)
    top_z = max(point[2] for point in points)
    measured = {} if outline is None else overlap_m2(
        outline, [region for region in regions if base_z < region.band_top_m])
    # A region whose band the object does not reach is zero *by the band rule*, which is a different
    # statement from "measured, and nothing overlapped" -- so it is written as a zero here rather
    # than left out, and the two are told apart by `in_band`.
    measured = {region.name: measured.get(region.name, 0.0) for region in regions}
    extent = None if outline is None else (
        max(x for x, _ in outline.vertices) - min(x for x, _ in outline.vertices),
        max(y for _, y in outline.vertices) - min(y for _, y in outline.vertices))
    return Part(name=str(getattr(obj, "name", "")), base_z_m=base_z, top_z_m=top_z,
                footprint_extent_m=extent, overlaps_m2=measured)


def measure_parts(objects: Iterable[Any], regions: Sequence[Region]) -> tuple[Part, ...]:
    """Every measurable object of a sequence, in the order handed in."""
    parts = (measure_part(obj, regions) for obj in objects)
    return tuple(part for part in parts if part is not None)
