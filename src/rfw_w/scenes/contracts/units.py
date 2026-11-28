# SPDX-License-Identifier: Apache-2.0
"""Units, frames and rigid transforms.

Frozen conventions (architecture §5.2). Everything downstream assumes these, so they are
stated once here rather than re-derived per module:

* Units are metres, kilograms, seconds and radians. No other unit is accepted at a
  contract boundary; a value carrying a different unit must have been converted before it
  gets here, and the conversion recorded in the asset's preparation record.
* Right-handed, Z-up, for the world frame.
* A transform is written ``T_parent_child`` and maps a point expressed in the *child*
  frame into the *parent* frame, acting on column vectors: ``p_parent = R * p_child + t``.
* JSON quaternions are ``wxyz``, in that order.

USD's ``Gf`` uses row vectors and row-major storage, so its matrices are the transpose of
ours. That conversion is explicit and named (``to_usd_matrix_rows`` / ``from_usd_matrix_rows``)
rather than hidden in a constructor, because a silently transposed transform is the kind of
error that only shows up as a scene that is subtly wrong.

This module imports nothing outside the standard library on purpose. OpenUSD is needed to
*read* or *write* a USD file, not to represent a transform, so ``pxr`` is never imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# Cosine tolerance for "this quaternion is normalised". Quaternions arriving from JSON or
# from a solver accumulate error; 1e-6 is tight enough that a genuinely unnormalised
# rotation still fails, and loose enough to survive a round trip through decimal text.
NORM_TOLERANCE = 1e-6


class UnitsError(ValueError):
    """Base class for contract violations raised by this module."""


class NonFiniteValueError(UnitsError):
    """A coordinate, quaternion component or scale was NaN or infinite."""


class QuaternionNotNormalisedError(UnitsError):
    """A quaternion's norm was not 1 within :data:`NORM_TOLERANCE`."""


class DegenerateTransformError(UnitsError):
    """An inverse was requested for a transform that cannot have one."""


def _check_finite(value: float, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteValueError(f"{what} must be finite, got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class Vec3:
    """A point or displacement in a named frame, in metres."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        _check_finite(self.x, "Vec3.x")
        _check_finite(self.y, "Vec3.y")
        _check_finite(self.z, "Vec3.z")

    @classmethod
    def origin(cls) -> Vec3:
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_sequence(cls, value: Sequence[float], *, context: str = "Vec3") -> Vec3:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise UnitsError(f"{context} must be a sequence of three numbers, got {type(value).__name__}")
        items = list(value)
        if len(items) != 3:
            raise UnitsError(f"{context} must have exactly 3 components, got {len(items)}")
        return cls(float(items[0]), float(items[1]), float(items[2]))

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> Vec3:
        return Vec3(self.x * factor, self.y * factor, self.z * factor)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> Vec3:
        length = self.norm()
        if length <= 0.0:
            raise DegenerateTransformError("cannot normalise a zero-length vector")
        return self.scale(1.0 / length)

    def is_close(self, other: Vec3, *, tolerance: float = 1e-9) -> bool:
        return all(abs(a - b) <= tolerance for a, b in zip(self.to_list(), other.to_list()))


@dataclass(frozen=True, slots=True)
class Quat:
    """A rotation stored as ``wxyz``.

    ``q`` and ``-q`` denote the same rotation. Equality here is component-wise, so
    comparing two quaternions that represent the same orientation may fail; use
    :meth:`is_same_rotation` when the question is about orientation rather than storage.
    """

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("w", "x", "y", "z"):
            _check_finite(getattr(self, name), f"Quat.{name}")
        norm = self.norm()
        if abs(norm - 1.0) > NORM_TOLERANCE:
            raise QuaternionNotNormalisedError(
                f"quaternion must be normalised, |q| = {norm!r} (tolerance {NORM_TOLERANCE})"
            )

    @classmethod
    def identity(cls) -> Quat:
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_sequence(cls, value: Sequence[float], *, context: str = "Quat") -> Quat:
        """Parse ``wxyz`` from JSON. Rejects ``xyzw`` silently masquerading as ``wxyz``."""
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise UnitsError(f"{context} must be a sequence of four numbers, got {type(value).__name__}")
        items = list(value)
        if len(items) != 4:
            raise UnitsError(f"{context} must have exactly 4 components (wxyz), got {len(items)}")
        return cls(float(items[0]), float(items[1]), float(items[2]), float(items[3]))

    @classmethod
    def from_axis_angle(cls, axis: Vec3, radians: float) -> Quat:
        angle = _check_finite(radians, "axis-angle radians")
        unit = axis.normalized()
        half = angle / 2.0
        sin_half = math.sin(half)
        return cls(math.cos(half), unit.x * sin_half, unit.y * sin_half, unit.z * sin_half)

    @classmethod
    def from_yaw(cls, radians: float) -> Quat:
        """Rotation about +Z, the world up axis. The common case for object placement."""
        return cls.from_axis_angle(Vec3(0.0, 0.0, 1.0), radians)

    def to_list(self) -> list[float]:
        """``wxyz``, the on-disk order."""
        return [self.w, self.x, self.y, self.z]

    def norm(self) -> float:
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def conjugate(self) -> Quat:
        return Quat(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: Quat) -> Quat:
        """Hamilton product: ``self * other`` applies ``other`` first."""
        return Quat(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def to_matrix3(self) -> tuple[tuple[float, float, float], ...]:
        """Rotation matrix acting on column vectors."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )

    def rotate(self, vector: Vec3) -> Vec3:
        rows = self.to_matrix3()
        return Vec3(
            rows[0][0] * vector.x + rows[0][1] * vector.y + rows[0][2] * vector.z,
            rows[1][0] * vector.x + rows[1][1] * vector.y + rows[1][2] * vector.z,
            rows[2][0] * vector.x + rows[2][1] * vector.y + rows[2][2] * vector.z,
        )

    def is_same_rotation(self, other: Quat, *, tolerance: float = 1e-9) -> bool:
        """True when ``self`` and ``other`` denote the same orientation (q == -q allowed)."""
        same = all(abs(a - b) <= tolerance for a, b in zip(self.to_list(), other.to_list()))
        negated = all(abs(a + b) <= tolerance for a, b in zip(self.to_list(), other.to_list()))
        return same or negated


@dataclass(frozen=True, slots=True)
class Transform:
    """``T_parent_child``: maps child-frame coordinates into the parent frame.

    ``parent_frame`` and ``child_frame`` are required rather than optional. A transform
    whose frames are unnamed cannot be checked against the scene graph, and mixing up the
    direction of a transform is otherwise indistinguishable from a correct one.
    """

    translation: Vec3
    rotation: Quat
    parent_frame: str
    child_frame: str

    def __post_init__(self) -> None:
        for name, value in (("parent_frame", self.parent_frame), ("child_frame", self.child_frame)):
            if not isinstance(value, str) or not value:
                raise UnitsError(f"{name} must be a non-empty string")

    @classmethod
    def identity(cls, frame: str) -> Transform:
        return cls(Vec3.origin(), Quat.identity(), frame, frame)

    def apply(self, point: Vec3) -> Vec3:
        """Map a point from the child frame into the parent frame."""
        return self.rotation.rotate(point) + self.translation

    def compose(self, inner: Transform) -> Transform:
        """``self * inner``: apply ``inner`` first, then ``self``.

        ``inner`` must be expressed in ``self.child_frame``, and ``self.parent_frame`` must
        match ``inner.parent_frame``; both are checked because a silently mismatched
        composition produces a plausible but wrong pose.
        """
        if inner.parent_frame != self.child_frame:
            raise UnitsError(
                f"cannot compose: {inner.parent_frame!r} does not match {self.child_frame!r}"
            )
        return Transform(
            translation=self.apply(inner.translation),
            rotation=self.rotation * inner.rotation,
            parent_frame=self.parent_frame,
            child_frame=inner.child_frame,
        )

    def inverse(self) -> Transform:
        """The inverse transform, swapping the frames."""
        inverse_rotation = self.rotation.conjugate()
        return Transform(
            translation=inverse_rotation.rotate(self.translation).scale(-1.0),
            rotation=inverse_rotation,
            parent_frame=self.child_frame,
            child_frame=self.parent_frame,
        )

    def to_usd_matrix_rows(self) -> list[list[float]]:
        """Row-major matrix in USD's row-vector convention.

        USD transforms act as ``p' = p @ M``, so the matrix stored is the transpose of the
        column-vector matrix used here. Returning plain floats keeps ``pxr`` out of the core.
        """
        r = self.rotation.to_matrix3()
        t = self.translation
        # Transposing a 4x4 makes the translation the last *row*, and the rotation's rows
        # become columns.
        return [
            [r[0][0], r[1][0], r[2][0], 0.0],
            [r[0][1], r[1][1], r[2][1], 0.0],
            [r[0][2], r[1][2], r[2][2], 0.0],
            [t.x, t.y, t.z, 1.0],
        ]

    @classmethod
    def from_usd_matrix_rows(
        cls, rows: Sequence[Sequence[float]], parent_frame: str, child_frame: str
    ) -> Transform:
        """Inverse of :meth:`to_usd_matrix_rows`."""
        if len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise UnitsError("USD matrix must be a 4x4 sequence")
        # The rotation is the transpose of the upper-left 3x3, and the translation is the
        # last row of the stored (row-vector) matrix.
        rotation_rows = tuple(
            tuple(float(rows[col][row]) for col in range(3)) for row in range(3)
        )
        translation = Vec3(float(rows[3][0]), float(rows[3][1]), float(rows[3][2]))
        return cls(translation, quaternion_from_matrix3(rotation_rows), parent_frame, child_frame)

    def is_close(self, other: Transform, *, tolerance: float = 1e-9) -> bool:
        return (
            self.translation.is_close(other.translation, tolerance=tolerance)
            and self.rotation.is_same_rotation(other.rotation, tolerance=tolerance)
        )


def quaternion_from_matrix3(rows: Sequence[Sequence[float]]) -> Quat:
    """Recover ``wxyz`` from a rotation matrix by Shepperd's method.

    Branching on the largest diagonal term keeps the division well conditioned; the naive
    single-formula version loses precision near 180-degree rotations, which is exactly
    where a flipped object would go unnoticed.
    """
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise UnitsError("rotation matrix must be 3x3")
    m = [[float(value) for value in row] for row in rows]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = Quat(0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        quat = Quat((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        quat = Quat((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        quat = Quat((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    return Quat(*[value / quat.norm() for value in quat.to_list()])


def rotation_distance_radians(left: Quat, right: Quat) -> float:
    """Angle between two orientations, in radians, in ``[0, pi]``."""
    # |dot| handles the double cover: q and -q are the same rotation.
    dot = abs(sum(a * b for a, b in zip(left.to_list(), right.to_list())))
    return 2.0 * math.acos(min(1.0, dot))


def row_bounds(points: Iterable[Vec3]) -> tuple[Vec3, Vec3]:
    """Axis-aligned bounds of a point set, as ``(minimum, maximum)``.

    Raises on an empty input: an empty bounding box would silently compare equal to any
    other empty box, and downstream clearance checks would read that as "no obstruction".
    """
    items = list(points)
    if not items:
        raise UnitsError("cannot compute bounds of an empty point set")
    return (
        Vec3(min(p.x for p in items), min(p.y for p in items), min(p.z for p in items)),
        Vec3(max(p.x for p in items), max(p.y for p in items), max(p.z for p in items)),
    )
