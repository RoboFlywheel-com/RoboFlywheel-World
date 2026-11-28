# SPDX-License-Identifier: Apache-2.0
"""SceneIR: the single cross-engine source of truth (architecture §5.3).

Two properties here are easy to get wrong and expensive to get wrong.

**The relation graph may contain cycles, and a cycle is not an error.** The support graph is
a DAG, but relations such as mutual contact or "this box sits between these two" can form
cycles. Treating the relation graph as a tree and dropping the back edges would silently
discard constraints, so traversal is cycle-safe and cycles are reported rather than pruned.

**Moving a parent invalidates its descendants' evidence.** Transform, support, navigation,
relation, render and export evidence all derive from a pose. If moving the desk does not
invalidate the cup's ON(desk) witness and the render approved for the old layout, the system
will report a stale pass. :meth:`SceneIR.dependency_closure` computes that set.

Instances sharing a mesh still have independent identity and dynamic state: mesh reuse is a
memory optimisation, never an identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from rfw_w.scenes.contracts.hashing import SceneHashes, content_hash
from rfw_w.scenes.contracts.randomization import DomainRandomizationRecord
from rfw_w.scenes.contracts.scene_spec import Mobility, PhysicsType, Predicate
from rfw_w.scenes.contracts.units import Transform, Vec3

SCHEMA_VERSION = "pvscene.scene-ir/v3"


class StateError(ValueError):
    """The scene state contradicts the contract."""


class Component(str, Enum):
    """Which hash component a change touches. Drives what evidence must be recomputed."""

    SPEC = "spec"
    ASSETS = "assets"
    GEOMETRY = "geometry"
    PHYSICS = "physics"
    APPEARANCE = "appearance"


class EvidenceKind(str, Enum):
    """Kinds of derived evidence a state change can invalidate."""

    TRANSFORM = "transform"
    SUPPORT = "support"
    RELATION = "relation"
    NAVIGATION = "navigation"
    RENDER = "render"
    VISUAL = "visual"
    PHYSICS = "physics"
    EXPORT = "export"
    # The room shell as built. A kind of its own rather than folded into `TRANSFORM`: nothing in
    # this repository can move a wall, so the set of changes that invalidate `shell.matches_plan`
    # is empty -- which is a statement worth being able to read off the table rather than infer
    # from a gate that never re-ran. `EXPORT` is the precedent for a kind whose gate list is empty.
    SHELL = "shell"


class OpeningKind(str, Enum):
    DOOR = "door"
    WINDOW = "window"


@dataclass(frozen=True, slots=True)
class Opening:
    """A door or window in the shell.

    Doors matter beyond appearance: the clearance and navigability checks depend on the door
    actually admitting the robot, so the width is a functional quantity, not decor.
    """

    opening_id: str
    kind: OpeningKind
    center_xy: tuple[float, float]
    width_m: float
    height_m: float
    sill_z_m: float = 0.0

    def __post_init__(self) -> None:
        if not self.opening_id:
            raise StateError("opening must have an id")
        if self.width_m <= 0 or self.height_m <= 0:
            raise StateError(f"opening {self.opening_id}: width and height must be positive")
        if self.sill_z_m < 0:
            raise StateError(f"opening {self.opening_id}: sill height must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "opening_id": self.opening_id,
            "kind": self.kind.value,
            "center_xy": list(self.center_xy),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "sill_z_m": self.sill_z_m,
        }


@dataclass(frozen=True, slots=True)
class RoomShell:
    """The room outline: an outer polygon with optional holes, plus its openings.

    Stored as an explicit polygon rather than a bounding box because an L-shaped room or a
    notch is not describable by a rectangle, and a rectangle stand-in passes containment
    checks that the real room would fail.

    ``holes_xy`` is **reserved and refused**. No shape in the shell vocabulary produces one, and
    every reader of ``polygon_xy`` -- ``door_sweep_for``, ``plan_zones``, ``validate_shell`` and
    the wall and floor authoring in the Isaac worker -- takes the outline as a simple polygon, so a
    courtyard would be floor to all of them. An inert field that consumers silently treat as
    absent is exactly the class of false pass this contract exists to prevent, so
    ``validate_shell`` fails a shell that carries one by name. Making the field real is a whole
    stage: ``Polygon`` has no holes notion and every consumer above would have to grow one.
    """

    polygon_xy: tuple[tuple[float, float], ...]
    holes_xy: tuple[tuple[tuple[float, float], ...], ...] = ()
    height_m: float = 2.8
    floor_z_m: float = 0.0
    openings: tuple[Opening, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.polygon_xy) < 3:
            raise StateError(f"room polygon needs at least 3 vertices, got {len(self.polygon_xy)}")
        for index, hole in enumerate(self.holes_xy):
            if len(hole) < 3:
                raise StateError(f"room hole {index} needs at least 3 vertices, got {len(hole)}")
        if self.height_m <= 0:
            raise StateError(f"room height must be positive, got {self.height_m}")

    @property
    def is_rectangle(self) -> bool:
        if len(self.polygon_xy) != 4 or self.holes_xy:
            return False
        xs = sorted({round(x, 9) for x, _ in self.polygon_xy})
        ys = sorted({round(y, 9) for _, y in self.polygon_xy})
        return len(xs) == 2 and len(ys) == 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon_xy": [list(vertex) for vertex in self.polygon_xy],
            "holes_xy": [[list(vertex) for vertex in hole] for hole in self.holes_xy],
            "height_m": self.height_m,
            "floor_z_m": self.floor_z_m,
            "openings": [opening.to_dict() for opening in self.openings],
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class JointState:
    """Articulated state for one instance.

    ``q`` and ``qd`` are kept per joint so that a settled door's angle survives into the
    evidence, and so that a task's ``initial_q`` can be compared against where the joint
    actually ended up.
    """

    joint_id: str
    position: float
    velocity: float = 0.0

    def __post_init__(self) -> None:
        if not self.joint_id:
            raise StateError("joint state must name a joint")

    def to_dict(self) -> dict[str, Any]:
        return {"joint_id": self.joint_id, "position": self.position, "velocity": self.velocity}


@dataclass(frozen=True, slots=True)
class Instance:
    """One placed object.

    ``local_transform`` is expressed in the frame of ``parent_id`` (the supporting or owning
    instance, or the world when there is none), and ``world`` is the resolved pose. Both are
    stored because the support graph needs the parent-relative form to re-solve a subtree
    after a parent moves, while the validators and the exporter need the world form.
    """

    instance_id: str
    entity_id: str
    asset_ref: str
    physics_type: PhysicsType
    mobility: Mobility
    world_transform: Transform
    local_transform: Transform
    parent_id: str | None = None
    scale: float = 1.0
    joints: tuple[JointState, ...] = ()
    mass_kg: float | None = None
    bounds_min_m: Vec3 | None = None
    bounds_max_m: Vec3 | None = None
    prepared_version: str = ""

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise StateError("instance must have an id")
        if not self.entity_id:
            raise StateError(f"instance {self.instance_id} must reference an entity")
        if not self.asset_ref:
            raise StateError(f"instance {self.instance_id} must reference an asset")
        if self.scale != 1.0:
            # Task objects default to scale=1 (§5.2). A scaled instance has mass, inertia and
            # contact evidence that no longer describe it, so the value cannot simply be
            # carried; the record has to have been re-derived.
            raise StateError(
                f"instance {self.instance_id} has scale {self.scale}; scaling invalidates "
                "measured mass, inertia and contact evidence"
            )
        if self.mass_kg is not None and self.mass_kg <= 0:
            raise StateError(f"instance {self.instance_id}: mass must be positive")

    @property
    def is_dynamic(self) -> bool:
        return self.mobility is Mobility.DYNAMIC

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instance_id": self.instance_id,
            "entity_id": self.entity_id,
            "asset_ref": self.asset_ref,
            "physics_type": self.physics_type.value,
            "mobility": self.mobility.value,
            "world_transform": {
                "translation": self.world_transform.translation.to_list(),
                "rotation_wxyz": self.world_transform.rotation.to_list(),
            },
            "local_transform": {
                "translation": self.local_transform.translation.to_list(),
                "rotation_wxyz": self.local_transform.rotation.to_list(),
            },
            "scale": self.scale,
            "joints": [joint.to_dict() for joint in self.joints],
            "prepared_version": self.prepared_version,
        }
        if self.parent_id is not None:
            payload["parent_id"] = self.parent_id
        if self.mass_kg is not None:
            payload["mass_kg"] = self.mass_kg
        if self.bounds_min_m is not None and self.bounds_max_m is not None:
            payload["bounds_min_m"] = self.bounds_min_m.to_list()
            payload["bounds_max_m"] = self.bounds_max_m.to_list()
        return payload


@dataclass(frozen=True, slots=True)
class RelationEdge:
    """A relation holding in the current state, with the witness that supports it."""

    relation_id: str
    subject: str
    predicate: Predicate
    object: str
    hard: bool
    satisfied: bool
    witness_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.relation_id:
            raise StateError("relation edge must have an id")
        if self.subject == self.object:
            raise StateError(f"relation {self.relation_id}: a relation needs two distinct entities")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate.value, self.object)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "subject": self.subject,
            "predicate": self.predicate.value,
            "object": self.object,
            "hard": self.hard,
            "satisfied": self.satisfied,
            "witness_refs": list(self.witness_refs),
        }


@dataclass(frozen=True, slots=True)
class SupportEdge:
    """``child`` rests on ``parent``.

    The support graph is required to be acyclic: a cycle would mean an object supports itself
    transitively, and settling it is not defined. Relation cycles are permitted; support
    cycles are not, which is the distinction between the two graphs.
    """

    child_id: str
    parent_id: str
    contact_polygon_area_m2: float = 0.0

    def __post_init__(self) -> None:
        if not self.child_id or not self.parent_id:
            raise StateError("support edge needs both a child and a parent")
        if self.child_id == self.parent_id:
            raise StateError(f"support edge {self.child_id} -> {self.parent_id} is self-supporting")

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "contact_polygon_area_m2": self.contact_polygon_area_m2,
        }


@dataclass(frozen=True, slots=True)
class CameraRig:
    """A named camera and the entities it is aimed at.

    ``target_ids`` is what makes a camera invalidatable: a render approved for a layout where
    the camera saw the desk must not be reused after the desk moved.
    """

    camera_id: str
    purpose: str
    target_ids: tuple[str, ...] = ()
    require_target_visible: bool = True
    save_depth: bool = False

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise StateError("camera rig must have an id")
        if self.purpose not in ("layout", "relation", "final", "export"):
            raise StateError(f"camera {self.camera_id}: unknown purpose {self.purpose!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "purpose": self.purpose,
            "target_ids": list(self.target_ids),
            "require_target_visible": self.require_target_visible,
            "save_depth": self.save_depth,
        }


# Which kinds of derived evidence a pose change destroys. Kept as a table so the mapping is
# reviewable in one place rather than scattered through the invalidation code.
_POSE_EVIDENCE: tuple[EvidenceKind, ...] = (
    EvidenceKind.TRANSFORM,
    EvidenceKind.SUPPORT,
    EvidenceKind.RELATION,
    EvidenceKind.NAVIGATION,
    EvidenceKind.RENDER,
    EvidenceKind.VISUAL,
    EvidenceKind.PHYSICS,
    EvidenceKind.EXPORT,
)

# Evidence kinds to the readiness gates they satisfy. Backend-specific gates are added by the
# caller, which is the only place that knows which backends were requested.
_EVIDENCE_GATES: Mapping[EvidenceKind, tuple[str, ...]] = {
    EvidenceKind.SUPPORT: ("geometry.supports",),
    EvidenceKind.TRANSFORM: ("geometry.in_bounds", "geometry.no_penetration"),
    EvidenceKind.RELATION: ("function.initial_relations",),
    EvidenceKind.PHYSICS: ("physics.settle_stability", "physics.clock_advanced"),
    EvidenceKind.RENDER: ("visual.required_views",),
    EvidenceKind.VISUAL: ("visual.critique",),
    EvidenceKind.NAVIGATION: (
        "function.space_conditions",
        "robot.nav",
        "robot.ik",
        "robot.motion",
        "task.rollout",
    ),
    EvidenceKind.EXPORT: (),
    # Empty for the same reason `EXPORT`'s is: no atom in this repository changes the shell, so
    # there is no change for which this evidence must be recomputed. Registering the kind at all
    # is what keeps the gate from being stale-and-unfixable -- an unregistered gate is one the
    # repair layer will never re-run, and a gate that is never re-run while the scene moves under
    # it blocks the level it belongs to with no way back.
    EvidenceKind.SHELL: (),
}


def gates_for_evidence(kinds: Iterable[EvidenceKind]) -> frozenset[str]:
    """The gates a set of evidence kinds answers for.

    Public because the repair layer has to say what an atom invalidates, and that answer must be
    this table rather than a second copy of it: two lists of which gate a moved box invalidates
    would drift, and the drift would show up as a gate re-run that should not have been, or --
    worse -- one that should have been and was not.
    """
    gates: set[str] = set()
    for kind in kinds:
        gates.update(_EVIDENCE_GATES[kind])
    return frozenset(gates)


# What a change to a *pose* invalidates. Named because the repair atoms all need it, and because
# reading "a moved box invalidates navigation too" is the sort of thing that should be checkable
# rather than inferred from a call site.
POSE_INVALIDATION: tuple[EvidenceKind, ...] = _POSE_EVIDENCE

# What a lighting or material change invalidates. Deliberately narrower than a pose change: the
# executable design §9 requires a material/lighting edit to redo the render and the critique and
# explicitly does not require it to re-pass the physics gates, because nothing moved.
APPEARANCE_INVALIDATION: tuple[EvidenceKind, ...] = (
    EvidenceKind.RENDER,
    EvidenceKind.VISUAL,
)


@dataclass(frozen=True, slots=True)
class DependencyClosure:
    """What a set of changes invalidates.

    ``moved`` is the geometric closure: the changed instances plus everything they support
    transitively. ``evidence_affected`` is wider, because a relation witness for a neighbour
    also depends on the moved object's pose, and the relation graph may contain cycles.
    """

    moved: frozenset[str]
    evidence_affected: frozenset[str]
    affected_cameras: frozenset[str]
    affected_gates: frozenset[str]
    touched_components: frozenset[Component]

    def to_dict(self) -> dict[str, Any]:
        return {
            "moved": sorted(self.moved),
            "evidence_affected": sorted(self.evidence_affected),
            "affected_cameras": sorted(self.affected_cameras),
            "affected_gates": sorted(self.affected_gates),
            "touched_components": sorted(component.value for component in self.touched_components),
        }


@dataclass(frozen=True, slots=True)
class SceneIR:
    """The whole scene state: the single fact source every backend is derived from."""

    scene_id: str
    room: RoomShell
    instances: tuple[Instance, ...]
    relations: tuple[RelationEdge, ...] = ()
    supports: tuple[SupportEdge, ...] = ()
    cameras: tuple[CameraRig, ...] = ()
    randomization: DomainRandomizationRecord | None = None
    spec_hash: str = ""
    hashes: SceneHashes | None = None
    revision: int = 0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise StateError("scene_id must be non-empty")

        ids = [instance.instance_id for instance in self.instances]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise StateError(f"duplicate instance ids: {duplicates}")
        known = set(ids)

        for instance in self.instances:
            if instance.parent_id is not None and instance.parent_id not in known:
                raise StateError(
                    f"instance {instance.instance_id} has unknown parent {instance.parent_id!r}"
                )

        for edge in self.supports:
            for end in (edge.child_id, edge.parent_id):
                if end not in known:
                    raise StateError(f"support edge references unknown instance {end!r}")
        self._assert_support_acyclic()

        for relation in self.relations:
            for end in (relation.subject, relation.object):
                if end not in known:
                    raise StateError(f"relation {relation.relation_id} references unknown {end!r}")

        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(set(camera_ids)) != len(camera_ids):
            raise StateError(f"duplicate camera ids: {camera_ids}")

    # -- topology ---------------------------------------------------------------

    def _assert_support_acyclic(self) -> None:
        """Support cycles are not definable, unlike relation cycles which are legitimate."""
        parents: dict[str, list[str]] = {}
        for edge in self.supports:
            parents.setdefault(edge.child_id, []).append(edge.parent_id)

        state: dict[str, int] = {}

        def visit(node: str, trail: tuple[str, ...]) -> None:
            if state.get(node) == 1:
                cycle = " -> ".join((*trail, node))
                raise StateError(f"support graph contains a cycle: {cycle}")
            if state.get(node) == 2:
                return
            state[node] = 1
            for parent in parents.get(node, ()):
                visit(parent, (*trail, node))
            state[node] = 2

        for instance in self.instances:
            visit(instance.instance_id, ())

    def supported_children(self, instance_id: str) -> tuple[str, ...]:
        return tuple(edge.child_id for edge in self.supports if edge.parent_id == instance_id)

    def relation_neighbours(self, instance_id: str) -> tuple[str, ...]:
        found: list[str] = []
        for relation in self.relations:
            if relation.subject == instance_id:
                found.append(relation.object)
            elif relation.object == instance_id:
                found.append(relation.subject)
        return tuple(found)

    def instance(self, instance_id: str) -> Instance:
        for candidate in self.instances:
            if candidate.instance_id == instance_id:
                return candidate
        raise StateError(f"unknown instance {instance_id!r}")

    # -- invalidation -----------------------------------------------------------

    def dependency_closure(
        self,
        changed_ids: Iterable[str],
        *,
        requested_backends: Sequence[str] = (),
    ) -> DependencyClosure:
        """Everything a change to ``changed_ids`` invalidates.

        Traversal is cycle-safe on both graphs. The geometric closure walks the support DAG
        downward (a moved desk moves nothing, but its dependents' support evidence changes);
        the evidence closure additionally expands through the relation graph, which may be
        cyclic, so a visited set is required rather than optional.
        """
        seeds = tuple(dict.fromkeys(changed_ids))
        known = {instance.instance_id for instance in self.instances}
        unknown = sorted(set(seeds) - known)
        if unknown:
            raise StateError(f"dependency closure over unknown instances: {unknown}")

        moved: set[str] = set()
        frontier = list(seeds)
        while frontier:
            node = frontier.pop()
            if node in moved:
                continue
            moved.add(node)
            frontier.extend(self.supported_children(node))

        evidence: set[str] = set()
        frontier = list(moved)
        while frontier:
            node = frontier.pop()
            if node in evidence:
                continue
            evidence.add(node)
            frontier.extend(self.relation_neighbours(node))

        gates: set[str] = set()
        components: set[Component] = set()
        for kind in _POSE_EVIDENCE:
            gates.update(_EVIDENCE_GATES[kind])
        components.update({Component.GEOMETRY, Component.PHYSICS})

        for backend in requested_backends:
            gates.update({f"backend.{backend}.run", f"backend.{backend}.render"})

        cameras = frozenset(
            camera.camera_id
            for camera in self.cameras
            if set(camera.target_ids) & evidence
        )

        return DependencyClosure(
            moved=frozenset(moved),
            evidence_affected=frozenset(evidence),
            affected_cameras=cameras,
            affected_gates=frozenset(gates),
            touched_components=frozenset(components),
        )

    # -- hashes -----------------------------------------------------------------

    def derive_hashes(self, *, spec: Any = None, appearance: Any = None) -> SceneHashes:
        """Recompute the six hashes from current content.

        ``spec`` and ``appearance`` come from outside because the spec is the request rather
        than the scene, and appearance intent lives with the style and material layers.
        """
        assets = sorted(
            {
                f"{instance.entity_id}:{instance.asset_ref}:{instance.prepared_version}"
                for instance in self.instances
            }
        )
        geometry = {
            "room": self.room.to_dict(),
            "instances": {
                instance.instance_id: {
                    "world": instance.world_transform.translation.to_list(),
                    "rotation": instance.world_transform.rotation.to_list(),
                    "bounds_min": None if instance.bounds_min_m is None else instance.bounds_min_m.to_list(),
                    "bounds_max": None if instance.bounds_max_m is None else instance.bounds_max_m.to_list(),
                }
                for instance in sorted(self.instances, key=lambda item: item.instance_id)
            },
            "supports": [edge.to_dict() for edge in self.supports],
        }
        physics = {
            "instances": {
                instance.instance_id: {
                    "mass_kg": instance.mass_kg,
                    "mobility": instance.mobility.value,
                    "physics_type": instance.physics_type.value,
                    "joints": [joint.to_dict() for joint in instance.joints],
                }
                for instance in sorted(self.instances, key=lambda item: item.instance_id)
            }
        }
        return SceneHashes.from_components(
            spec=spec if spec is not None else {"scene_id": self.scene_id, "spec_hash": self.spec_hash},
            assets=assets,
            geometry=geometry,
            physics=physics,
            appearance=appearance if appearance is not None else {"instance_count": len(self.instances)},
        )

    def with_hashes(self, hashes: SceneHashes) -> SceneIR:
        return SceneIR(
            scene_id=self.scene_id,
            room=self.room,
            instances=self.instances,
            relations=self.relations,
            supports=self.supports,
            cameras=self.cameras,
            randomization=self.randomization,
            spec_hash=self.spec_hash,
            hashes=hashes,
            revision=self.revision + 1,
        )

    def content_payload(self) -> dict[str, Any]:
        """The scene content, excluding bookkeeping.

        ``revision`` and the recorded ``hashes`` are deliberately absent. They are counters
        and derived values, not content: including them would make a verifier recomputing
        from the delivered artefacts disagree with the generator purely because the generator
        had attached its own hashes.
        """
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "room": self.room.to_dict(),
            "instances": [instance.to_dict() for instance in self.instances],
            "relations": [relation.to_dict() for relation in self.relations],
            "supports": [edge.to_dict() for edge in self.supports],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "randomization": (
                None if self.randomization is None else self.randomization.to_dict()
            ),
            "spec_hash": self.spec_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.content_payload()
        payload["revision"] = self.revision
        if self.hashes is not None:
            payload["hashes"] = self.hashes.to_dict()
        return payload

    def content_hash(self) -> str:
        """Digest of the current content, independent of the stored hashes and revision.

        The independent verifier recomputes this, which is how a delivered manifest claiming a
        scene version that does not match its own geometry is caught.
        """
        return content_hash(self.content_payload())
