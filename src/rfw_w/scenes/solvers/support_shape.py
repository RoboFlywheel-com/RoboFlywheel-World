# SPDX-License-Identifier: Apache-2.0
"""The shape of the surface the task is performed on, as a list of boxes.

Why this exists
---------------

The support -- the desk, the kitchen island, the nightstand -- used to be retrieved from a furniture
library, which meant the task's whole geometry was decided by whichever asset came back and by how
well its annotation described it. Measured on the real library: of 168 desks, 13 have a vertical
panel as their largest "support surface", 37 are placed more than 10 cm below their own top, and one
states no surface at all. The task should not inherit any of that.

So the support is built instead of fetched. What is built is this module's vocabulary: a shape is a
list of axis-aligned boxes, and every shape has exactly one **top slab whose upper face is
``height_m``** -- the surface the task is solved against.

The reference implementation this follows (`~/code/real2sim/codex_assets`,
`jobs/piper_hang_mugs_20260918/build_scene.py:68-77`) emits a whole table as such a list and gets
one thing right that is worth copying exactly: the same tuple produces both engines' geometry, so
they cannot disagree. It has no shape vocabulary at all -- its four "styles" are textures and its
table is a literal -- which is the part extended here.

Conventions, and why they are these
-----------------------------------

* **Metres, Z-up, footprint centred on the origin, base at z = 0.** This is the convention
  ``prepare_furniture`` produces for every HSSD asset and that the placement solver, the origin
  bookkeeping and the Isaac worker all already agree on. A generated support that used any other
  would need a second convention running beside the first.
* **``centre_m`` is a centre and ``size_m`` is a full extent**, matching the reference's ``box()``
  (its USD ``scale`` and its MuJoCo half-extents both come from the same tuple).
* **Parts do not overlap.** Mass is the sum of part volumes, so an overlap would be counted twice,
  and a mesh whose faces interpenetrate is not the shape anyone described. :func:`parts_for` refuses
  an overlapping recipe rather than trusting each one to be written carefully.

The frame here is the *asset's*, not the task cell's. ``solve_cell`` works in a cell frame whose
front edge sits at y = 0 with the body toward +y; this module's top is centred on the origin, because
that is what a placed instance's geometry has to be. The conversion between them is the builder's
job, and it is the same one ``MeasuredSupport`` already carries (``assets/support_measurement.py``).

Stdlib only, and walked by ``rfw_w.scenes.validation.check_stdlib_only``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from rfw_w.scenes.geometry.polygon import Polygon


class SupportShapeError(ValueError):
    """The requested shape cannot be built at the requested size."""


class SupportShape(str, Enum):
    """The support forms this release can build.

    Five, not fifty. Each is a form that actually occurs in the three experiments' prompts -- a desk,
    a desk with side panels, a kitchen island, an island with a lower shelf, a nightstand -- and each
    is a list of boxes, so a new one is a recipe rather than a mesh pipeline.
    """

    SLAB_LEGS = "slab_legs"
    SLAB_PANELS = "slab_panels"
    PLINTH = "plinth"
    TWO_TIER = "two_tier"
    PEDESTAL = "pedestal"


@dataclass(frozen=True, slots=True)
class SupportPart:
    """One box of a support: where its centre is and how big it is, in metres."""

    name: str
    centre_m: tuple[float, float, float]
    size_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise SupportShapeError("a support part must be named")
        for value in (*self.centre_m, *self.size_m):
            if not math.isfinite(value):
                raise SupportShapeError(f"{self.name}: non-finite value {value}")
        for axis, extent in zip("xyz", self.size_m):
            if extent <= 0:
                raise SupportShapeError(f"{self.name}: {axis} extent must be positive, got {extent}")

    @property
    def volume_m3(self) -> float:
        return self.size_m[0] * self.size_m[1] * self.size_m[2]

    @property
    def z_range_m(self) -> tuple[float, float]:
        half = self.size_m[2] / 2.0
        return (self.centre_m[2] - half, self.centre_m[2] + half)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "centre_m": list(self.centre_m),
            "size_m": list(self.size_m),
        }


# The one part whose upper face is the working surface. Named so that "which part is the top" is a
# question with an answer rather than a convention every caller re-derives.
TOP_PART_NAME = "top"

# Default proportions. They are proportions, not measurements: nothing in the task requires a 4 cm
# top or a 5 cm leg, and the reference's own numbers (5 cm top, 4.5 cm legs, 7 cm inset) are engineering
# choices it declares as such. What matters is that they are stated once, here, rather than inline in
# a recipe where a reader cannot see what is a decision and what is a constant.
DEFAULT_TOP_THICKNESS_M = 0.04
DEFAULT_LEG_THICKNESS_M = 0.05
DEFAULT_LEG_INSET_M = 0.07
DEFAULT_SHELF_THICKNESS_M = 0.03
DEFAULT_SHELF_HEIGHT_M = 0.25
DEFAULT_DRAWER_HEIGHT_M = 0.15


def parts_for(
    shape: SupportShape,
    *,
    width_m: float,
    depth_m: float,
    height_m: float,
    top_thickness_m: float = DEFAULT_TOP_THICKNESS_M,
    leg_thickness_m: float = DEFAULT_LEG_THICKNESS_M,
    leg_inset_m: float = DEFAULT_LEG_INSET_M,
    shelf_thickness_m: float = DEFAULT_SHELF_THICKNESS_M,
    shelf_height_m: float = DEFAULT_SHELF_HEIGHT_M,
    drawer_height_m: float = DEFAULT_DRAWER_HEIGHT_M,
) -> tuple[SupportPart, ...]:
    """The boxes one support is made of, top slab first.

    ``leg_inset_m`` is measured from the top's edge to the leg's **outer** face, which is how the
    reference states it (``0.53 = 1.2/2 - 0.07``) -- stated that way so that a wider table gets wider
    leg spacing without anyone recomputing the constant.
    """
    for name, value in (("width_m", width_m), ("depth_m", depth_m), ("height_m", height_m)):
        if not math.isfinite(value) or value <= 0:
            raise SupportShapeError(f"{name} must be positive and finite, got {value}")
    if top_thickness_m <= 0 or top_thickness_m >= height_m:
        raise SupportShapeError(
            f"top_thickness_m={top_thickness_m} must be positive and below height_m={height_m}"
        )
    leg_height = height_m - top_thickness_m

    top = SupportPart(
        name=TOP_PART_NAME,
        centre_m=(0.0, 0.0, height_m - top_thickness_m / 2.0),
        size_m=(width_m, depth_m, top_thickness_m),
    )
    legs = _legs(width_m, depth_m, leg_height, leg_inset_m, leg_thickness_m)
    inner_width = width_m - 2.0 * leg_inset_m - 2.0 * leg_thickness_m
    inner_depth = depth_m - 2.0 * leg_inset_m - 2.0 * leg_thickness_m

    if shape is SupportShape.SLAB_LEGS:
        parts = (top, *legs)
    elif shape is SupportShape.SLAB_PANELS:
        parts = (top, *_panels(width_m, depth_m, leg_height, leg_thickness_m))
    elif shape is SupportShape.PLINTH:
        _require_inner_fits(shape, width_m, depth_m, leg_inset_m, leg_thickness_m)
        parts = (top, SupportPart(
            name="plinth",
            centre_m=(0.0, 0.0, leg_height / 2.0),
            size_m=(inner_width + 2.0 * leg_thickness_m, inner_depth + 2.0 * leg_thickness_m, leg_height),
        ))
    elif shape is SupportShape.TWO_TIER:
        _require_inner_fits(shape, width_m, depth_m, leg_inset_m, leg_thickness_m)
        if shelf_thickness_m <= 0 or shelf_height_m <= 0:
            raise SupportShapeError("a two-tier support needs a positive shelf thickness and height")
        if shelf_height_m + shelf_thickness_m > leg_height:
            raise SupportShapeError(
                f"SUPPORT_SHAPE_TOO_SHORT({shape.value}): a shelf at {shelf_height_m} m plus "
                f"{shelf_thickness_m} m of thickness does not fit under a {leg_height:.3f} m top"
            )
        parts = (top, *legs, SupportPart(
            name="shelf",
            centre_m=(0.0, 0.0, shelf_height_m + shelf_thickness_m / 2.0),
            size_m=(inner_width, inner_depth, shelf_thickness_m),
        ))
    elif shape is SupportShape.PEDESTAL:
        _require_inner_fits(shape, width_m, depth_m, leg_inset_m, leg_thickness_m)
        if drawer_height_m <= 0 or drawer_height_m > leg_height:
            raise SupportShapeError(
                f"SUPPORT_SHAPE_TOO_SHORT({shape.value}): a {drawer_height_m} m drawer does not fit "
                f"under a {leg_height:.3f} m top"
            )
        parts = (top, *legs, SupportPart(
            name="drawer",
            centre_m=(0.0, 0.0, leg_height - drawer_height_m / 2.0),
            size_m=(inner_width, inner_depth, drawer_height_m),
        ))
    else:
        raise SupportShapeError(
            f"UNKNOWN_SUPPORT_SHAPE({shape!r}): this release builds "
            f"{sorted(item.value for item in SupportShape)}"
        )

    _require_top_slab(parts, width_m, depth_m, height_m)
    _require_disjoint(parts)
    return parts


def top_surface(
    shape: SupportShape,
    *,
    width_m: float,
    depth_m: float,
    height_m: float,
    **kwargs: float,
) -> tuple[Polygon, float]:
    """The surface the task is solved against: a rectangle and the height it sits at.

    Footprint-centred, because that is the frame a placed asset's geometry lives in -- see the module
    docstring on why this is not the task cell's frame.
    """
    parts_for(shape, width_m=width_m, depth_m=depth_m, height_m=height_m, **kwargs)
    return (
        Polygon.rectangle(-width_m / 2.0, -depth_m / 2.0, width_m / 2.0, depth_m / 2.0),
        height_m,
    )


def volume_m3(parts: Sequence[SupportPart]) -> float:
    """The support's solid volume, as the sum of its parts.

    Exact for boxes, which is why a constructed support can state an honest mass at all -- the
    library path can only estimate one from a convex hull, and records that it did.
    """
    return sum(part.volume_m3 for part in parts)


def shape_from(name: str | None, *, default: SupportShape = SupportShape.SLAB_LEGS) -> SupportShape:
    """The shape a spec names, or the category's default when it names none.

    An unrecognised name is refused rather than defaulted: a spec asking for a form this release
    cannot build is a fact about the request, and silently building a different table is how a
    manifest ends up describing a scene nobody asked for.
    """
    if name is None:
        return default
    try:
        return SupportShape(name)
    except ValueError as error:
        raise SupportShapeError(
            f"UNKNOWN_SUPPORT_SHAPE({name!r}): this release builds "
            f"{sorted(item.value for item in SupportShape)}"
        ) from error


def _legs(
    width_m: float, depth_m: float, leg_height: float, inset_m: float, thickness_m: float,
) -> tuple[SupportPart, ...]:
    if inset_m < 0:
        raise SupportShapeError(f"leg_inset_m must not be negative, got {inset_m}")
    if thickness_m <= 0:
        raise SupportShapeError(f"leg_thickness_m must be positive, got {thickness_m}")
    if 2.0 * inset_m + thickness_m > min(width_m, depth_m):
        raise SupportShapeError(
            f"SUPPORT_SHAPE_TOO_SMALL(legs): a {width_m:.3f} x {depth_m:.3f} m top leaves no room "
            f"for {thickness_m} m legs inset {inset_m} m from both edges"
        )
    x = width_m / 2.0 - inset_m - thickness_m / 2.0
    y = depth_m / 2.0 - inset_m - thickness_m / 2.0
    return tuple(
        SupportPart(
            name=f"leg_{index}",
            centre_m=(sx * x, sy * y, leg_height / 2.0),
            size_m=(thickness_m, thickness_m, leg_height),
        )
        for index, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
    )


def _panels(
    width_m: float, depth_m: float, panel_height: float, thickness_m: float,
) -> tuple[SupportPart, ...]:
    if 2.0 * thickness_m >= width_m:
        raise SupportShapeError(
            f"SUPPORT_SHAPE_TOO_SMALL(panels): two {thickness_m} m panels do not fit under a "
            f"{width_m:.3f} m top"
        )
    x = width_m / 2.0 - thickness_m / 2.0
    return tuple(
        SupportPart(
            name=f"panel_{index}",
            centre_m=(sx * x, 0.0, panel_height / 2.0),
            size_m=(thickness_m, depth_m, panel_height),
        )
        for index, sx in enumerate((1, -1))
    )


def _require_inner_fits(
    shape: SupportShape, width_m: float, depth_m: float, inset_m: float, thickness_m: float,
) -> None:
    """A shelf or drawer sits *between* the legs, so it needs room the legs leave."""
    inner_width = width_m - 2.0 * inset_m - 2.0 * thickness_m
    inner_depth = depth_m - 2.0 * inset_m - 2.0 * thickness_m
    if min(inner_width, inner_depth) <= 0:
        raise SupportShapeError(
            f"SUPPORT_SHAPE_TOO_SMALL({shape.value}): a {width_m:.3f} x {depth_m:.3f} m top leaves "
            f"{inner_width:.3f} x {inner_depth:.3f} m between its legs, which cannot hold a shelf"
        )


def _require_top_slab(
    parts: Sequence[SupportPart], width_m: float, depth_m: float, height_m: float,
) -> None:
    """The invariant every shape shares: one slab, and its upper face *is* ``height_m``.

    Checked rather than trusted. The core solves the task against ``height_m``; a recipe whose top
    slab stopped 2 mm short would put the target below the surface the arm was judged against, and
    nothing downstream would say so.
    """
    tops = [part for part in parts if part.name == TOP_PART_NAME]
    if len(tops) != 1:
        raise SupportShapeError(f"a support needs exactly one {TOP_PART_NAME!r} part, got {len(tops)}")
    top = tops[0]
    if top.size_m[0] != width_m or top.size_m[1] != depth_m:
        raise SupportShapeError(
            f"the top slab is {top.size_m[0]} x {top.size_m[1]} m but the footprint is "
            f"{width_m} x {depth_m} m"
        )
    lower, upper = top.z_range_m
    if not math.isclose(upper, height_m, abs_tol=1e-9):
        raise SupportShapeError(
            f"the top slab's upper face is z={upper} but the working height is {height_m}"
        )
    if not math.isclose(top.centre_m[0], 0.0, abs_tol=1e-9) or not math.isclose(
        top.centre_m[1], 0.0, abs_tol=1e-9
    ):
        raise SupportShapeError("the top slab must be centred on the footprint origin")
    if lower <= 0:
        raise SupportShapeError("the top slab reaches the floor; that is a plinth, not a slab")


def _require_disjoint(parts: Sequence[SupportPart]) -> None:
    """No two parts may interpenetrate.

    Mass is the sum of part volumes and the mesh is the union of the boxes; an overlap would be
    counted twice in one and produce self-intersecting geometry in the other. Parts that merely touch
    are fine -- a shelf between its legs has to touch them to be attached.
    """
    tolerance = 1e-9
    for index, first in enumerate(parts):
        for second in parts[index + 1:]:
            if all(
                abs(first.centre_m[axis] - second.centre_m[axis])
                < (first.size_m[axis] + second.size_m[axis]) / 2.0 - tolerance
                for axis in range(3)
            ):
                raise SupportShapeError(
                    f"SUPPORT_PARTS_OVERLAP({first.name}, {second.name}): two boxes of one support "
                    "interpenetrate, which would double-count their volume and emit a self-"
                    "intersecting mesh"
                )
