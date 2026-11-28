# SPDX-License-Identifier: Apache-2.0
"""Geometry: polygons, offsets, clearance and support coverage.

Pure stdlib and free of asset or engine dependencies, so layout and predicate logic can be
tested without an asset root, a GPU or a simulator. This is the only part of the system that
decides where things physically are, so it is kept inspectable rather than delegated to a mesh
library.

Polygons here are exact vertex lists, never bounding boxes. The acceptance manual is specific
about why: an L-shaped room or a notched desk cannot be described by a box, and a box stand-in
passes containment checks the real shape would fail -- placing an object in the notch and
reporting the geometry gate as satisfied.
"""

from __future__ import annotations

from rfw_w.scenes.geometry.clearance import (
    DEFAULT_MARGIN_M,
    ClearanceError,
    ClearanceResult,
    check_clearance,
)
from rfw_w.scenes.geometry.polygon import (
    Polygon,
    PolygonError,
    area,
    centroid,
    contains_point,
    contains_polygon,
    distance_to_boundary,
    inset,
    interior_point,
    offset_outward,
    point_inside_toward,
    overlaps,
    validate_simple,
)

__all__ = [
    "DEFAULT_MARGIN_M",
    "ClearanceError",
    "ClearanceResult",
    "Polygon",
    "PolygonError",
    "area",
    "centroid",
    "check_clearance",
    "contains_point",
    "contains_polygon",
    "distance_to_boundary",
    "inset",
    "interior_point",
    "offset_outward",
    "point_inside_toward",
    "overlaps",
    "validate_simple",
]
