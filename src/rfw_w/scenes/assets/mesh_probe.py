# SPDX-License-Identifier: Apache-2.0
"""Measure an asset's real geometry from its mesh files.

The catalogue gives a `real_bbox`, but it is a rounded summary and it disagrees with the
meshes. On this library a mug catalogued as 0.10 x 0.08 x 0.09 m measures 0.1005 x 0.0642 x
0.0796 m, and a plate catalogued with two large axes is really a flat disc. Planning a
placement from the catalogue number therefore places objects at the wrong height, and the
error is small enough to look like physics jitter rather than a bad measurement.

So dimensions, bounds and collision part sizes are measured from the actual mesh, and the
provenance says measured rather than declared.

The up-axis is verified rather than trusted, for the reason the architecture records: declared
up-axes have disagreed with the geometry in this asset family before. The check here is
geometric -- a plate is a disc and is therefore thin along the true up axis, a vase is
circular in the plane perpendicular to it -- so the conclusion comes from the shapes instead
of from a metadata field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rfw_w.scenes.contracts.units import Vec3


class MeshProbeError(RuntimeError):
    """The mesh could not be measured."""


@dataclass(frozen=True, slots=True)
class MeshMeasurement:
    """Measured extents, bounds and primitive counts for one mesh."""

    vertex_count: int
    face_count: int
    bounds_min: Vec3
    bounds_max: Vec3

    @property
    def extents(self) -> Vec3:
        return self.bounds_max - self.bounds_min

    @property
    def center(self) -> Vec3:
        midpoint = (self.bounds_min + self.bounds_max).scale(0.5)
        return Vec3(midpoint.x, midpoint.y, midpoint.z)

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bounds_min_m": self.bounds_min.to_list(),
            "bounds_max_m": self.bounds_max.to_list(),
            "extents_m": self.extents.to_list(),
        }


def _obj_tokens(path: Path) -> Iterator[tuple[str, list[str]]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if parts:
                yield parts[0], parts[1:]


def measure_obj(path: Path, *, max_vertices: int = 2_000_000) -> MeshMeasurement:
    """Measure an OBJ's extents and counts.

    Reads only what is needed and refuses implausibly large files rather than exhausting
    memory: a measurement helper that can take the process down is worse than one that reports
    it could not measure.
    """
    if not path.is_file():
        raise MeshProbeError(f"mesh not found: {path}")

    minimum = [math.inf] * 3
    maximum = [math.inf * -1] * 3
    vertices = 0
    faces = 0

    for keyword, values in _obj_tokens(path):
        if keyword == "v":
            if len(values) < 3:
                continue
            try:
                x, y, z = (float(values[0]), float(values[1]), float(values[2]))
            except ValueError as error:
                raise MeshProbeError(f"{path}: malformed vertex {values[:3]!r}") from error
            for index, value in enumerate((x, y, z)):
                if value < minimum[index]:
                    minimum[index] = value
                if value > maximum[index]:
                    maximum[index] = value
            vertices += 1
            if vertices > max_vertices:
                raise MeshProbeError(
                    f"{path}: more than {max_vertices} vertices; refusing to measure a mesh "
                    "this large in-process"
                )
        elif keyword == "f":
            faces += 1

    if vertices == 0:
        raise MeshProbeError(f"{path}: contains no vertices")

    return MeshMeasurement(
        vertex_count=vertices,
        face_count=faces,
        bounds_min=Vec3(*minimum),
        bounds_max=Vec3(*maximum),
    )


# Categories whose proportions are known well enough to check an axis convention against.
# "disc" means the object is round in the horizontal plane and thin along up; "column" means it
# is round in the horizontal plane and tall along up. Both are statements about the shape, so a
# measurement either agrees or it does not.
SHAPE_EXPECTATIONS = {
    "disc": ("Dinner Plate", "Serving Tray", "Soap Dish", "Plant Pot Saucer", "Plate"),
    "column": ("Vase", "Perfume Bottle", "Bulb Planter", "Creamer Pitcher", "Spray Bottle"),
}


UP_AXES = ("x", "y", "z")


@dataclass(frozen=True, slots=True)
class UpAxisVerdict:
    """What the geometry says about which axis is up, and the evidence for it."""

    up_axis: str | None
    confidence: str
    detail: str
    samples: int

    @property
    def decided(self) -> bool:
        return self.up_axis is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "up_axis": self.up_axis,
            "confidence": self.confidence,
            "detail": self.detail,
            "samples": self.samples,
        }


def classify_up_axis(
    measurements: list[tuple[str, Vec3]], *, tolerance_ratio: float = 1.15
) -> UpAxisVerdict:
    """Infer the up axis from the shapes of known flat and tall categories.

    Takes ``(super_category, extents)`` pairs. A disc is flat along up and its two horizontal
    extents are close to equal; a column is long along up with its two horizontal extents close
    to equal. Each measurement votes, and the verdict requires unanimity -- a split vote is
    reported as undecided rather than resolved by majority, because a wrong convention here
    lays every object on its side and is hard to notice afterwards.
    """
    if not measurements:
        return UpAxisVerdict(None, "none", "no measurements were supplied", 0)

    votes: dict[str, int] = {axis: 0 for axis in UP_AXES}
    considered = 0
    detail: list[str] = []

    for category, extents in measurements:
        values = extents.to_list()
        if min(values) <= 0:
            continue
        kind = next((k for k, names in SHAPE_EXPECTATIONS.items() if category in names), None)
        if kind is None:
            continue
        considered += 1
        ordered = sorted(range(3), key=lambda index: values[index])
        if kind == "disc":
            # Flat along up: the smallest extent is the up axis, and the other two agree.
            candidate = ordered[0]
            others = [ordered[1], ordered[2]]
            if values[others[1]] > values[others[0]] * tolerance_ratio:
                detail.append(f"{category}: not disc-like ({values})")
                continue
        else:
            # Long along up: the largest extent is the up axis, and the other two agree.
            candidate = ordered[2]
            others = [ordered[0], ordered[1]]
            if values[others[1]] > values[others[0]] * tolerance_ratio:
                detail.append(f"{category}: not column-like ({values})")
                continue
        votes[UP_AXES[candidate]] += 1
        detail.append(f"{category}: {UP_AXES[candidate]} up ({values})")

    if considered == 0:
        return UpAxisVerdict(None, "none", "no disc- or column-like shape was measured", 0)

    winners = [axis for axis, count in votes.items() if count > 0]
    if len(winners) != 1:
        return UpAxisVerdict(
            None, "conflicting",
            "measurements disagree about the up axis: " + "; ".join(detail),
            considered,
        )
    axis = winners[0]
    return UpAxisVerdict(
        axis,
        "unanimous" if votes[axis] == considered else "partial",
        f"{votes[axis]}/{considered} shapes agree: " + "; ".join(detail),
        considered,
    )


def y_up_to_z_up_rotation():
    """The rotation that maps a Y-up asset frame onto the world's Z-up frame.

    +90 degrees about X sends +Y to +Z, so an object standing up in its own frame stands up in
    the world. The contract forbids applying a normalisation twice, which is why this is
    recorded on the asset rather than applied by every caller.
    """
    from rfw_w.scenes.contracts.units import Quat

    return Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), math.pi / 2.0)


def rotate_extents(extents: Vec3) -> Vec3:
    """Extents after the Y-up to Z-up rotation, as a permutation of the axes."""
    return Vec3(extents.x, extents.z, extents.y)
