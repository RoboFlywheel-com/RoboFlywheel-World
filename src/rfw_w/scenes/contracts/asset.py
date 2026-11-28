# SPDX-License-Identifier: Apache-2.0
"""AssetRecordV3 and AssetHandle: what a retrieved asset actually is.

Three rules from architecture §5.2 are enforced here rather than left to convention.

**A missing affordance is neither false nor true.** Support polygons, receptacles, grasp
regions and the rest carry an explicit ``status``. ``unknown`` means nobody measured it.
A hard functional requirement against an ``unknown`` affordance is not "satisfied by
default" and not "failed": it means the candidate must be probed or rejected, which the
retrieval layer does. Representing this as a bool is exactly the mistake the architecture
calls out.

**Scale is not a tuning knob.** Applying a scale factor invalidates the measured mass,
inertia, contact behaviour and every functional witness derived from the geometry, so a
record carries the normalisation that was applied and refuses to apply a second one. Task
objects default to scale 1.

**Paths do not cross the model boundary.** ``AssetHandle`` holds URIs and is internal. What
the compiler and the critic see is the opaque ``semantic_id``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from rfw_w.scenes.contracts.hashing import content_hash, is_digest
from rfw_w.scenes.contracts.scene_spec import PhysicsType
from rfw_w.scenes.contracts.units import Transform, Vec3

SCHEMA_VERSION = "pvscene.asset-record/v3"

# Sources that indicate a stand-in was used instead of a retrieved asset. Architecture §2
# and the acceptance manual both forbid these as a fallback for a retrieval miss: a scene
# built from boxes can pass a contact predicate while containing none of the requested
# objects, which is worse than reporting the miss.
PRIMITIVE_SOURCES = frozenset(
    {
        "primitive_box",
        "primitive_cube",
        "primitive_sphere",
        "primitive_cylinder",
        "primitive_capsule",
        "primitive_plane",
        "synthetic_fixture",
        "placeholder",
    }
)


class AssetError(ValueError):
    """The asset record contradicts the contract."""


class AffordanceStatus(str, Enum):
    """How well an affordance is known. Never collapse these into a boolean."""

    MEASURED = "measured"
    ANNOTATED = "annotated"
    CONSTRUCTED = "constructed"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"

    @property
    def usable_for_hard_requirement(self) -> bool:
        """Whether this status can satisfy a hard functional requirement.

        ``estimated`` is not enough: an estimate has not been verified against the real
        geometry, and a contact predicate decided from one is a guess.

        ``constructed`` is in, and it is a different claim from the two around it. ``measured``
        means somebody else's asset was inspected; ``annotated`` means a dataset stated it. This one
        means the geometry **is ours** -- built to the dimensions the task solver asked for, so
        there is no gap between the number the task was solved against and the number that exists.
        That is the strongest claim in this enum, and recording it as ``measured`` would hide which
        side of the boundary the surface came from.
        """
        return self in (
            AffordanceStatus.MEASURED,
            AffordanceStatus.ANNOTATED,
            AffordanceStatus.CONSTRUCTED,
        )


class Backend(str, Enum):
    ISAAC = "isaac"
    MUJOCO = "mujoco"
    BLENDER = "blender"


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNTESTED = "untested"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a number came from. Every physical and affordance value carries one."""

    method: str
    version: str
    status: AffordanceStatus
    confidence: float | None = None
    source_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.method:
            raise AssetError("provenance.method must be non-empty")
        if not self.version:
            raise AssetError("provenance.version must be non-empty")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise AssetError(f"confidence must be within [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method,
            "version": self.version,
            "status": self.status.value,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.source_uri is not None:
            payload["source_uri"] = self.source_uri
        return payload


@dataclass(frozen=True, slots=True)
class AssetKey:
    """Identity of an asset: provider, library version and asset id together.

    All three are required. Keying on ``asset_id`` alone lets two libraries that both have
    a ``0001`` be treated as the same object, and silently swaps one for the other.
    """

    provider: str
    library_version: str
    asset_id: str

    def __post_init__(self) -> None:
        for name in ("provider", "library_version", "asset_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise AssetError(f"AssetKey.{name} must be a non-empty string")

    @property
    def semantic_id(self) -> str:
        """Opaque handle safe to hand to a model: no path, no library name."""
        return f"asset:{content_hash(self.to_dict())[:16]}"

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "library_version": self.library_version,
            "asset_id": self.asset_id,
        }


@dataclass(frozen=True, slots=True)
class NormalisationRecord:
    """The canonicalisation applied to raw asset geometry, and its source.

    Applies once. A second centre or scale would invalidate mass, inertia, contact and every
    functional witness computed from the first one, so :meth:`assert_not_double_applied`
    exists and the solver path calls it.
    """

    applied_scale: float
    applied_translation: Vec3
    source_frame: str
    target_frame: str
    method: str
    version: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.applied_scale) or self.applied_scale <= 0:
            raise AssetError(f"normalisation scale must be finite and positive, got {self.applied_scale}")
        for name in ("source_frame", "target_frame", "method", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise AssetError(f"NormalisationRecord.{name} must be non-empty")

    @property
    def is_identity(self) -> bool:
        return self.applied_scale == 1.0 and self.applied_translation.is_close(Vec3.origin())

    def assert_not_double_applied(self, requested_scale: float) -> None:
        """Refuse to scale geometry that has already been normalised."""
        if not self.is_identity:
            raise AssetError(
                "asset has already been normalised "
                f"(scale={self.applied_scale}, frame {self.source_frame}->{self.target_frame}); "
                "applying a further scale would invalidate its mass, inertia and contact evidence"
            )
        if requested_scale != 1.0:
            raise AssetError(
                f"task objects default to scale=1; requested scale {requested_scale} would "
                "invalidate measured physical evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied_scale": self.applied_scale,
            "applied_translation": self.applied_translation.to_list(),
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "method": self.method,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class CollisionPart:
    """One convex piece or mesh used for collision.

    ``approximate`` marks hulls and primitives that approximate the visual shape. A part
    that is both approximate and a primitive is the stand-in case: fine for a static support
    proxy, never acceptable for a task target.
    """

    part_id: str
    source: str
    approximate: bool
    vertex_count: int
    face_count: int
    provenance: Provenance

    def __post_init__(self) -> None:
        if not self.part_id:
            raise AssetError("collision part must have an id")
        if self.vertex_count < 0 or self.face_count < 0:
            raise AssetError(f"collision part {self.part_id}: negative counts")

    @property
    def is_primitive_substitute(self) -> bool:
        return self.source in PRIMITIVE_SOURCES

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "source": self.source,
            "approximate": self.approximate,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RigidPhysics:
    """Mass properties and contact parameters for a rigid body.

    ``inertia`` must be symmetric positive definite; a negative or non-SPD inertia is
    accepted by neither PhysX nor MuJoCo and produces a body that explodes or freezes rather
    than one that simulates.
    """

    mass_kg: float
    center_of_mass_m: Vec3
    inertia_kg_m2: tuple[float, float, float, float, float, float, float, float, float]
    static_friction: float
    dynamic_friction: float
    restitution: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise AssetError(f"mass must be finite and positive, got {self.mass_kg}")
        if len(self.inertia_kg_m2) != 9:
            raise AssetError("inertia must be a 3x3 matrix flattened to 9 entries")
        for value in self.inertia_kg_m2:
            if not math.isfinite(value):
                raise AssetError("inertia contains a non-finite entry")
        if self.static_friction < 0 or self.dynamic_friction < 0:
            raise AssetError("friction coefficients must be non-negative")
        if not (0.0 <= self.restitution <= 1.0):
            raise AssetError(f"restitution must be within [0, 1], got {self.restitution}")

    @property
    def inertia_is_spd(self) -> bool:
        """Sylvester's criterion for a symmetric matrix: all leading minors positive."""
        m = self.inertia_kg_m2

        def minor(indices: Sequence[int]) -> float:
            rows = [[m[r * 3 + c] for c in indices] for r in indices]
            if len(rows) == 1:
                return rows[0][0]
            if len(rows) == 2:
                return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
            return (
                rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
                - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
                + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
            )

        return all(minor(list(range(size))) > 0.0 for size in (1, 2, 3))

    def assert_simulable(self) -> None:
        if not self.inertia_is_spd:
            raise AssetError(
                f"inertia {self.inertia_kg_m2} is not symmetric positive definite; "
                "neither PhysX nor MuJoCo will simulate this body correctly"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mass_kg": self.mass_kg,
            "center_of_mass_m": self.center_of_mass_m.to_list(),
            "inertia_kg_m2": list(self.inertia_kg_m2),
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SupportPolygon:
    """A surface that can carry another object, as a horizontal polygon plus its height."""

    polygon_id: str
    vertices_xy: tuple[tuple[float, float], ...]
    plane_z_m: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if len(self.vertices_xy) < 3:
            raise AssetError(
                f"support polygon {self.polygon_id} needs at least 3 vertices, got {len(self.vertices_xy)}"
            )

    @property
    def is_rectangle(self) -> bool:
        """Whether the polygon is a 4-vertex axis-aligned rectangle.

        Recorded rather than assumed: a rectangle stand-in for a real support surface is a
        loss of fidelity worth seeing in the report, and a support surface that is a
        rectangle only because it was derived from an AABB is not the same object.
        """
        if len(self.vertices_xy) != 4:
            return False
        xs = sorted({round(x, 9) for x, _ in self.vertices_xy})
        ys = sorted({round(y, 9) for _, y in self.vertices_xy})
        return len(xs) == 2 and len(ys) == 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon_id": self.polygon_id,
            "vertices_xy": [list(vertex) for vertex in self.vertices_xy],
            "plane_z_m": self.plane_z_m,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MeasuredSupport:
    """A support surface as a *named asset* states it, said in the task cell's own frame.

    Why this is not a ``SupportPolygon``. That record describes a surface in the frame the asset
    library publishes it in, and calls its height ``plane_z_m`` -- a name that reads as an asset
    coordinate. The room-free core needs a different claim: "this much flat area, this high above
    the floor, with the robot working from this side". Handing a raw ``SupportPolygon`` to the core
    would leave the frame conversion to be re-derived at every call site, and would let an asset
    coordinate be read as a floor height without anything saying so. The conversion happens once,
    where the library is read, and the result is this.

    The frame is stated rather than implied: ``vertices_xy`` are metres in the **cell frame** --
    x centred on the surface, the ``min y`` edge at ``y = 0``, the body toward ``+y``, and the robot
    aisle at ``y < 0``. A caller that hands over vertices in any other frame is refused by name
    rather than silently moving the protected region the room is later fitted around.

    ``front_edge_basis`` is load-bearing and is why this record has two string fields that look
    like comments. Which side of a measured rectangle faces a room cannot be recovered from the
    rectangle -- it is symmetric -- so the -Y edge is *declared* the front, exactly as the category
    envelope already assumed. Naming the basis keeps the assumption auditable instead of letting a
    reader believe the library said it.
    """

    asset_id: str
    source_polygon_id: str
    vertices_xy: tuple[tuple[float, float], ...]
    support_height_m: float
    provenance: Provenance
    front_edge_basis: str = "declared_min_y_faces_the_aisle"
    height_basis: str = "asset_local_base_plane"

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise AssetError("MeasuredSupport.asset_id must be non-empty")
        if not self.source_polygon_id:
            raise AssetError("MeasuredSupport.source_polygon_id must be non-empty")
        if not self.front_edge_basis:
            raise AssetError("MeasuredSupport.front_edge_basis must be non-empty")
        if not self.height_basis:
            raise AssetError("MeasuredSupport.height_basis must be non-empty")
        if len(self.vertices_xy) < 3:
            raise AssetError(
                f"measured support {self.source_polygon_id} needs at least 3 vertices, "
                f"got {len(self.vertices_xy)}"
            )
        for x, y in self.vertices_xy:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise AssetError(f"measured support {self.source_polygon_id} has a non-finite vertex")
        if not math.isfinite(self.support_height_m) or self.support_height_m <= 0:
            raise AssetError(
                f"measured support {self.source_polygon_id} needs a positive height, "
                f"got {self.support_height_m}"
            )

    @property
    def width_m(self) -> float:
        """How wide the surface is, derived from the vertices rather than stored beside them.

        Two sources of truth for one number is how a footprint ends up disagreeing with its own
        outline, so this and :attr:`depth_m` are the outline, read.
        """
        return max(x for x, _ in self.vertices_xy) - min(x for x, _ in self.vertices_xy)

    @property
    def depth_m(self) -> float:
        return max(y for _, y in self.vertices_xy) - min(y for _, y in self.vertices_xy)

    @property
    def area_m2(self) -> float:
        return self.width_m * self.depth_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "source_polygon_id": self.source_polygon_id,
            "vertices_xy": [list(vertex) for vertex in self.vertices_xy],
            "support_height_m": self.support_height_m,
            "provenance": self.provenance.to_dict(),
            "front_edge_basis": self.front_edge_basis,
            "height_basis": self.height_basis,
        }


@dataclass(frozen=True, slots=True)
class Receptacle:
    """An inner cavity: where an INSIDE relation is even meaningful.

    The acceptance manual only allows INSIDE against a *certified* rigid receptacle, because
    a point inside a container's bounding box is not inside the cavity. Certification is
    carried here as the provenance status plus an explicit opening path.
    """

    cavity_id: str
    opening_frame: Transform
    capacity_m3: float
    opening_width_m: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.capacity_m3 <= 0:
            raise AssetError(f"receptacle {self.cavity_id}: capacity must be positive")
        if self.opening_width_m <= 0:
            raise AssetError(f"receptacle {self.cavity_id}: opening width must be positive")

    @property
    def is_certified(self) -> bool:
        """Only a measured cavity with a known opening may carry an INSIDE predicate."""
        return self.provenance.status is AffordanceStatus.MEASURED

    def to_dict(self) -> dict[str, Any]:
        return {
            "cavity_id": self.cavity_id,
            "capacity_m3": self.capacity_m3,
            "opening_width_m": self.opening_width_m,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AssetRecordV3:
    """The full record for one retrievable asset.

    ``capabilities`` maps backend to status so that "the catalogue declares this asset" and
    "this asset is known to work in this backend" stay separate claims, which is the
    distinction the acceptance manual §4 insists on.
    """

    key: AssetKey
    source_hash: str
    license: str
    semantic: str
    super_category: str
    physics_type: PhysicsType
    dimensions_m: Vec3
    bounds_min_m: Vec3
    bounds_max_m: Vec3
    canonical_frame: str
    normalisation: NormalisationRecord
    visual_uri: str
    collision_parts: tuple[CollisionPart, ...]
    materials: tuple[str, ...]
    rigid_physics: RigidPhysics | None = None
    support_polygons: tuple[SupportPolygon, ...] = ()
    receptacles: tuple[Receptacle, ...] = ()
    grasp_regions: tuple[str, ...] = ()
    handles: tuple[str, ...] = ()
    capabilities: Mapping[str, CapabilityStatus] = field(default_factory=dict)
    prepared_version: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not is_digest(self.source_hash):
            raise AssetError(f"source_hash must be a sha256 hex digest, got {self.source_hash!r}")
        if not self.license:
            raise AssetError("license must be recorded; an unlicensed asset cannot be shipped")
        if not self.semantic:
            raise AssetError("semantic label must be non-empty")
        if not self.visual_uri:
            raise AssetError("visual_uri must be recorded")
        if not self.canonical_frame:
            raise AssetError("canonical_frame must be recorded")
        if self.dimensions_m.x <= 0 or self.dimensions_m.y <= 0 or self.dimensions_m.z <= 0:
            raise AssetError(f"dimensions must be positive in every axis, got {self.dimensions_m.to_list()}")
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            if self.bounds_min_m.to_list()[index] > self.bounds_max_m.to_list()[index]:
                raise AssetError(f"bounds_min exceeds bounds_max on {axis}")
        if self.physics_type is PhysicsType.RIGID and self.rigid_physics is None:
            raise AssetError(f"rigid asset {self.key.asset_id} has no mass properties")

    @property
    def is_primitive_substitute(self) -> bool:
        return any(part.is_primitive_substitute for part in self.collision_parts)

    def capability_for(self, backend: str) -> CapabilityStatus:
        """Backend support, defaulting to ``untested`` rather than ``supported``.

        Defaulting to supported would let an untested asset and backend combination be
        reported as working.
        """
        return self.capabilities.get(backend, CapabilityStatus.UNTESTED)

    def assert_usable_as_task_target(self) -> None:
        """Reject the substitutions that make a scene look successful but be empty."""
        if self.is_primitive_substitute:
            sources = sorted({part.source for part in self.collision_parts if part.is_primitive_substitute})
            raise AssetError(
                f"asset {self.key.asset_id} uses primitive stand-ins {sources}; a primitive may "
                "never stand in for a retrieved task asset"
            )
        if not self.collision_parts:
            raise AssetError(f"asset {self.key.asset_id} has no collision geometry")
        if self.rigid_physics is not None:
            self.rigid_physics.assert_simulable()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "key": self.key.to_dict(),
            "source_hash": self.source_hash,
            "license": self.license,
            "semantic": self.semantic,
            "super_category": self.super_category,
            "physics_type": self.physics_type.value,
            "dimensions_m": self.dimensions_m.to_list(),
            "bounds_min_m": self.bounds_min_m.to_list(),
            "bounds_max_m": self.bounds_max_m.to_list(),
            "canonical_frame": self.canonical_frame,
            "normalisation": self.normalisation.to_dict(),
            "visual_uri": self.visual_uri,
            "collision_parts": [part.to_dict() for part in self.collision_parts],
            "materials": list(self.materials),
            "support_polygons": [polygon.to_dict() for polygon in self.support_polygons],
            "receptacles": [cavity.to_dict() for cavity in self.receptacles],
            "grasp_regions": list(self.grasp_regions),
            "handles": list(self.handles),
            "capabilities": {backend: status.value for backend, status in sorted(self.capabilities.items())},
            "prepared_version": self.prepared_version,
        }
        if self.rigid_physics is not None:
            payload["rigid_physics"] = self.rigid_physics.to_dict()
        return payload

    def semantic_view(self) -> dict[str, Any]:
        """The path-free projection that may be shown to a model.

        Handles, URIs, hashes and library names are stripped. The architecture forbids
        sending repository files, paths, catalogue records or logs to a model, and this is
        the single place that decision is implemented.
        """
        return {
            "semantic_id": self.key.semantic_id,
            "semantic": self.semantic,
            "category": self.super_category,
            "physics_type": self.physics_type.value,
            "dimensions_m": [round(value, 4) for value in self.dimensions_m.to_list()],
            "materials": list(self.materials),
            "has_support_surface": bool(self.support_polygons),
            "has_receptacle": any(cavity.is_certified for cavity in self.receptacles),
        }


@dataclass(frozen=True, slots=True)
class AssetHandle:
    """Internal binding of a record to concrete files. Never crosses the model boundary.

    Kept as a separate type from :class:`AssetRecordV3` so that "the model saw this asset"
    and "the process can open this asset" are different statements, and so that a leak of a
    URI into a prompt becomes a type error rather than a review catch.
    """

    key: AssetKey
    visual_uri: str
    collision_uris: tuple[str, ...]
    metadata_uri: str | None
    package_hash: str

    def __post_init__(self) -> None:
        if not self.visual_uri:
            raise AssetError("handle must carry a visual uri")
        if not self.collision_uris:
            raise AssetError(f"handle for {self.key.asset_id} has no collision geometry")
        if not is_digest(self.package_hash):
            raise AssetError(f"package_hash must be a sha256 digest, got {self.package_hash!r}")

    @property
    def semantic_id(self) -> str:
        return self.key.semantic_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "visual_uri": self.visual_uri,
            "collision_uris": list(self.collision_uris),
            "metadata_uri": self.metadata_uri,
            "package_hash": self.package_hash,
        }
