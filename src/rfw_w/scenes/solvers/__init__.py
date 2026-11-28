# SPDX-License-Identifier: Apache-2.0
"""Solvers: deterministic construction of the scene's geometry.

Rooms, zones, furniture layout, support relationships and relation predicates are all solved
here rather than asked of a model. The architecture's split is that a model supplies semantics
and constraints while the solvers own the geometry, so everything in this package is
deterministic given a spec and a seed.
"""

from __future__ import annotations

from rfw_w.scenes.solvers.room import (
    ShellOptions,
    ShellValidation,
    build_shell,
    check_door_connectivity,
    validate_shell,
)
from rfw_w.scenes.solvers.zones import (
    MIN_ZONE_AREA_M2,
    WALL_DEPTH_M,
    Zone,
    ZoneError,
    ZoneGraph,
    plan_zones,
)

__all__ = [
    "MIN_ZONE_AREA_M2",
    "WALL_DEPTH_M",
    "ShellOptions",
    "ShellValidation",
    "Zone",
    "ZoneError",
    "ZoneGraph",
    "build_shell",
    "check_door_connectivity",
    "plan_zones",
    "validate_shell",
]
