# SPDX-License-Identifier: Apache-2.0
"""SceneSpec: what the user asked for, frozen before any solving happens.

The one distinction this module exists to protect is the one architecture §5.1 calls out:

* ``initial_relations`` must hold in the **generated** scene -- "the cup is on the desk".
* ``goal_predicates`` must hold **after the robot acts** -- "the cup is inside the cabinet".

Collapsing the two lets a system place the cup in the cabinet at generation time and then
claim the pick-and-place task succeeded, because the goal predicate is already satisfied and
nothing ever had to move. So they are separate fields with separate types, the goal side is
parsed only from ``task.goal_predicates``, and a goal predicate is never satisfiable at
generation time by construction -- the solver never reads it.

Articulated tasks store ``initial_q``, ``test_q_schedule`` and ``goal_q`` as three distinct
fields rather than overwriting one another, so that the state a door started in stays
recoverable after the door has been opened and closed again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pvscene.scene-spec/v3"


class SpecError(ValueError):
    """The requested scene contradicts the contract."""


class PhysicsType(str, Enum):
    """Physics class of an entity. Determines which providers and predicates apply."""

    RIGID = "rigid"
    ARTICULATED = "articulated"
    DEFORMABLE = "deformable"


class Mobility(str, Enum):
    """Whether the solver may move an instance after placement."""

    FIXED = "fixed"
    DYNAMIC = "dynamic"


class EntityRole(str, Enum):
    """Why an entity is in the scene. Drives the deletion and substitution rules."""

    TASK_TARGET = "task_target"
    SUPPORT = "support"
    CONTEXT = "context"
    DECORATION = "decoration"


class SceneScope(str, Enum):
    """Whether an entity belongs to the task cell or to its surroundings."""

    CORE = "core"
    PERIPHERY = "periphery"


class Density(str, Enum):
    """How full a zone should be. Drives counts and how much floor stays open."""

    SPARSE = "sparse"
    MEDIUM = "medium"
    DENSE = "dense"

    @property
    def floor_coverage_target(self) -> float:
        """The fraction of floor area furniture and coverings may occupy.

        This is the "whitespace" the deterministic rules own (architecture §9 lists 留白 among
        them). A room furnished to 100% is not a dense room, it is an unusable one: circulation is
        a floor area, and something has to reserve it.
        """
        return {"sparse": 0.22, "medium": 0.35, "dense": 0.50}[self.value]


class ZoneKind(str, Enum):
    """Functional regions a room is divided into.

    Zone assignment is what stops every piece of furniture being placed against the longest wall.
    Each kind carries its own placement rule and its own density, so a storage zone can be dense
    while the area in front of the door stays empty.
    """

    WORK = "work"                # the strip on the wall nearest the task's work area
    SEATING = "seating"          # chairs, which face the work zone
    STORAGE = "storage"          # tall pieces that present a front to the room
    DISPLAY = "display"          # wall-hung pieces and plants
    AMBIENT = "ambient"          # lamps, floor coverings
    CIRCULATION = "circulation"  # must stay clear; nothing is placed here


class ElementEmphasis(str, Enum):
    """Primary pieces get the best position and the largest share of the style budget."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    ACCENT = "accent"


class Predicate(str, Enum):
    """Relation predicates. First release covers the first block of this list (§8)."""

    ON = "ON"
    INSIDE = "INSIDE"
    AGAINST_WALL = "AGAINST_WALL"
    FACING = "FACING"
    NEAR = "NEAR"
    OPENABLE = "OPENABLE"
    CLOSABLE = "CLOSABLE"
    CAPPING = "CAPPING"
    SLOTTING = "SLOTTING"
    HOOKING = "HOOKING"
    PEGGING = "PEGGING"
    SOFT_INSIDE = "SOFT_INSIDE"
    DRAPED_ON = "DRAPED_ON"


@dataclass(frozen=True, slots=True)
class Range1D:
    """A closed interval in metres, used for per-axis room size bounds."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not (self.minimum <= self.maximum):
            raise SpecError(f"range minimum {self.minimum} exceeds maximum {self.maximum}")

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum

    def to_list(self) -> list[float]:
        return [self.minimum, self.maximum]

    @classmethod
    def from_sequence(cls, value: Sequence[float], *, context: str = "range") -> Range1D:
        items = list(value)
        if len(items) != 2:
            raise SpecError(f"{context} must be [min, max], got {len(items)} entries")
        return cls(float(items[0]), float(items[1]))


# Versioned finite defaults for unspecified dimensions. Explicit bounds always win.
# These profiles are the SPEC-006 small/medium/large acceptance bands, not global limits.
ROOM_SIZE_POLICY_VERSION = "room-size/v1"
ROOM_KINDS = ("office", "kitchen", "bedroom")
ROOM_SCALE_BOUNDS = {
    "small": ((4., 5.), (3., 4.), (2.7, 3.)),
    "medium": ((6., 7.), (5., 6.), (2.7, 3.)),
    "large": ((8., 10.), (6., 8.), (2.7, 3.)),
}


@dataclass(frozen=True, slots=True)
class RoomSpec:
    size_bounds_m: tuple[Range1D, Range1D, Range1D]
    room_kind: str | None = None
    size_policy: str = "explicit"
    scale_class: str | None = None
    size_policy_version: str | None = None

    def __post_init__(self) -> None:
        if len(self.size_bounds_m) != 3:
            raise SpecError("room size_bounds_m must have three axis ranges (x, y, z)")
        if any(not math.isfinite(v) or v <= 0 for axis in self.size_bounds_m
               for v in (axis.minimum, axis.maximum)):
            raise SpecError("ROOM_SIZE_CONFLICT(bounds must be finite and positive)")
        if self.room_kind is not None and self.room_kind not in ROOM_KINDS:
            raise SpecError(f"ROOM_KIND_UNSUPPORTED({self.room_kind})")
        if self.size_policy not in ("explicit", "inferred"):
            raise SpecError(f"ROOM_SCALE_POLICY_MISSING({self.size_policy})")
        if self.scale_class is not None and self.scale_class not in ROOM_SCALE_BOUNDS:
            raise SpecError(f"ROOM_SCALE_POLICY_MISSING({self.scale_class})")
        if self.size_policy == "inferred":
            if (self.room_kind is None or self.scale_class is None
                    or self.size_policy_version != ROOM_SIZE_POLICY_VERSION):
                raise SpecError("ROOM_SCALE_POLICY_MISSING(inferred needs kind, scale and version)")
            expected = ROOM_SCALE_BOUNDS[self.scale_class]
            if tuple((v.minimum, v.maximum) for v in self.size_bounds_m) != expected:
                raise SpecError("ROOM_SIZE_CONFLICT(inferred bounds disagree with versioned policy)")
        elif self.size_policy_version is not None:
            raise SpecError("ROOM_SIZE_CONFLICT(explicit bounds have no inferred policy version)")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoomSpec:
        unknown = set(payload) - {"size_bounds_m", "room_kind", "size_policy", "scale_class", "size_policy_version"}
        if unknown:
            raise SpecError(f"room has unknown keys: {sorted(unknown)}")
        bounds = payload.get("size_bounds_m")
        policy = payload.get("size_policy", "explicit" if bounds is not None else "inferred")
        kind = payload.get("room_kind")
        if kind is not None and kind not in ROOM_KINDS:
            raise SpecError(f"ROOM_KIND_UNSUPPORTED({kind})")
        scale = payload.get("scale_class")
        version = payload.get("size_policy_version")
        if policy == "inferred":
            scale = scale or "medium"
            if kind not in ROOM_KINDS or scale not in ROOM_SCALE_BOUNDS:
                raise SpecError(f"ROOM_SCALE_POLICY_MISSING({kind}, {scale})")
            if bounds is None:
                bounds = ROOM_SCALE_BOUNDS[scale]
            version = version or ROOM_SIZE_POLICY_VERSION
        if not isinstance(bounds, Sequence) or len(bounds) != 3:
            raise SpecError("ROOM_SIZE_CONFLICT(room.size_bounds_m must be three [min, max] pairs)")
        return cls(
            size_bounds_m=tuple(Range1D.from_sequence(axis, context=f"size_bounds_m[{index}]")
                                for index, axis in enumerate(bounds)),
            room_kind=kind, size_policy=policy, scale_class=scale, size_policy_version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {"size_bounds_m": [axis.to_list() for axis in self.size_bounds_m]}
        # Preserve legacy documents/hashes. New intent is explicit whenever supplied.
        if self.room_kind is not None:
            payload["room_kind"] = self.room_kind
        if self.scale_class is not None:
            payload["scale_class"] = self.scale_class
        if self.room_kind is not None or self.size_policy != "explicit":
            payload["size_policy"] = self.size_policy
        if self.size_policy_version is not None:
            payload["size_policy_version"] = self.size_policy_version
        return payload


@dataclass(frozen=True, slots=True)
class StylePlan:
    """Aesthetic and density intent, produced once at compile time (architecture §9).

    §9 asks this to carry the furniture families, the palette, materials, the primary/secondary
    hierarchy and a density range per zone. It is the one place where the model's judgement about
    *how a room should be furnished* enters the system: the solvers own orientation, scale matching,
    repetition, whitespace, occlusion and passage, and they read all of those from here.

    A StylePlan that carries nothing is not harmless -- the solvers then have no intent to apply and
    fall back to filling whichever wall is longest, which is exactly what a crowded corner looks
    like. So the fields below are the ones the furnishing solver actually consumes, not decoration.
    """

    theme: str
    palette: tuple[str, ...]
    density: Density = Density.MEDIUM
    families: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    hierarchy: Mapping[str, ElementEmphasis] = field(default_factory=dict)
    zone_density: Mapping[ZoneKind, Density] = field(default_factory=dict)
    coverage_target: float | None = None

    def __post_init__(self) -> None:
        if not self.theme:
            raise SpecError("style.theme must be non-empty")
        if not self.palette:
            raise SpecError("style.palette must name at least one colour family")
        target = self.effective_coverage_target
        if not 0.0 < target <= 0.9:
            raise SpecError(
                f"style coverage target must be within (0, 0.9], got {target}; a room furnished to "
                "the last square metre has no circulation left"
            )

    @property
    def effective_coverage_target(self) -> float:
        """The coverage to aim for: explicit when given, otherwise derived from density."""
        return self.coverage_target if self.coverage_target is not None else self.density.floor_coverage_target

    def density_for(self, zone: ZoneKind) -> Density:
        """A zone's density, falling back to the room's."""
        return self.zone_density.get(zone, self.density)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "theme": self.theme,
            "palette": list(self.palette),
            "density": self.density.value,
        }
        if self.families:
            payload["families"] = list(self.families)
        if self.materials:
            payload["materials"] = list(self.materials)
        if self.hierarchy:
            payload["hierarchy"] = {k: v.value for k, v in sorted(self.hierarchy.items())}
        if self.zone_density:
            payload["zone_density"] = {k.value: v.value for k, v in sorted(self.zone_density.items(), key=lambda i: i[0].value)}
        if self.coverage_target is not None:
            payload["coverage_target"] = self.coverage_target
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StylePlan:
        known = {"theme", "palette", "density", "families", "materials", "hierarchy",
                 "zone_density", "coverage_target"}
        unknown = set(payload) - known
        if unknown:
            raise SpecError(f"style has unknown keys: {sorted(unknown)}")
        try:
            density = Density(payload.get("density", Density.MEDIUM.value))
        except ValueError as error:
            raise SpecError(f"unknown density {payload.get('density')!r}") from error
        palette = payload.get("palette", [])
        if isinstance(palette, (str, bytes)) or not isinstance(palette, Sequence):
            raise SpecError("style.palette must be a list of colour families")

        hierarchy: dict[str, ElementEmphasis] = {}
        for key, value in (payload.get("hierarchy") or {}).items():
            try:
                hierarchy[str(key)] = ElementEmphasis(value)
            except ValueError as error:
                raise SpecError(f"unknown emphasis {value!r} for {key!r}") from error

        zone_density: dict[ZoneKind, Density] = {}
        for key, value in (payload.get("zone_density") or {}).items():
            try:
                zone = ZoneKind(key)
            except ValueError as error:
                raise SpecError(f"unknown zone {key!r}") from error
            try:
                zone_density[zone] = Density(value)
            except ValueError as error:
                raise SpecError(f"unknown density {value!r} for zone {key!r}") from error

        coverage = payload.get("coverage_target")
        return cls(
            theme=str(payload.get("theme", "")),
            palette=tuple(str(item) for item in palette),
            density=density,
            families=tuple(str(item) for item in payload.get("families", ()) or ()),
            materials=tuple(str(item) for item in payload.get("materials", ()) or ()),
            hierarchy=hierarchy,
            zone_density=zone_density,
            coverage_target=None if coverage is None else float(coverage),
        )


@dataclass(frozen=True, slots=True)
class EntitySpec:
    """One requested object, with its semantic identity but no asset and no pose.

    The compiler chooses semantics and constraints. It does not choose asset IDs or write
    poses: those come from retrieval and the solvers, and letting the model pick them is
    what the architecture §4 table forbids.
    """

    id: str
    category: str
    physics_type: PhysicsType
    mobility: Mobility
    hard_required: bool
    role: EntityRole
    quantity: int = 1
    size_bounds_m: tuple[Range1D, Range1D, Range1D] | None = None
    notes: str = ""
    material_tokens: tuple[str, ...] = ()
    requires_support_surface: bool = False
    # The form a constructed support is built to, for an entity that is one. Empty means "whatever
    # the category's own profile builds", which is what every spec in the repository leaves it as.
    # The *name* is validated where the shapes are known (``solvers/support_shape.py``) rather than
    # here: the contract layer holds no geometry vocabulary, and duplicating the list would let the
    # two drift.
    support_shape: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.isidentifier():
            raise SpecError(f"entity id {self.id!r} must be a non-empty identifier")
        if not self.category:
            raise SpecError(f"entity {self.id}: category must be non-empty")
        if self.quantity < 1:
            raise SpecError(f"entity {self.id}: quantity must be at least 1, got {self.quantity}")
        if self.size_bounds_m is not None and len(self.size_bounds_m) != 3:
            raise SpecError(f"entity {self.id}: size_bounds_m must have three axis ranges")
        if any(not token.strip() for token in self.material_tokens):
            raise SpecError(f"entity {self.id}: material_tokens may not contain empty values")
        if self.support_shape and not self.support_shape.strip():
            raise SpecError(f"entity {self.id}: support_shape may not be blank")

    @property
    def is_task_target(self) -> bool:
        return self.role is EntityRole.TASK_TARGET

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EntitySpec:
        known = {
            "id", "category", "physics_type", "mobility", "hard_required", "role",
            "quantity", "size_bounds_m", "notes", "material_tokens", "requires_support_surface",
            "support_shape",
        }
        unknown = set(payload) - known
        if unknown:
            raise SpecError(f"entity has unknown keys: {sorted(unknown)}")
        for required in ("id", "category", "physics_type", "mobility"):
            if required not in payload:
                raise SpecError(f"entity is missing required key {required!r}: {dict(payload)}")
        try:
            physics_type = PhysicsType(payload["physics_type"])
        except ValueError as error:
            raise SpecError(f"unknown physics_type {payload['physics_type']!r}") from error
        try:
            mobility = Mobility(payload["mobility"])
        except ValueError as error:
            raise SpecError(f"unknown mobility {payload['mobility']!r}") from error
        role_raw = payload.get("role")
        try:
            role = EntityRole(role_raw) if role_raw is not None else EntityRole.CONTEXT
        except ValueError as error:
            raise SpecError(f"unknown role {role_raw!r}") from error
        bounds = payload.get("size_bounds_m")
        parsed_bounds: tuple[Range1D, Range1D, Range1D] | None = None
        if bounds is not None:
            if not isinstance(bounds, Sequence) or len(bounds) != 3:
                raise SpecError(f"entity {payload['id']}: size_bounds_m must be three [min, max] pairs")
            parsed_bounds = tuple(
                Range1D.from_sequence(axis, context=f"{payload['id']}.size_bounds_m[{index}]")
                for index, axis in enumerate(bounds)
            )  # type: ignore[arg-type]
        material_tokens = payload.get("material_tokens", ())
        if isinstance(material_tokens, (str, bytes)) or not isinstance(material_tokens, Sequence):
            raise SpecError(f"entity {payload['id']}: material_tokens must be a list")
        requires_support_surface = payload.get("requires_support_surface", False)
        if not isinstance(requires_support_surface, bool):
            raise SpecError(f"entity {payload['id']}: requires_support_surface must be a boolean")
        support_shape = payload.get("support_shape", "")
        if not isinstance(support_shape, str):
            raise SpecError(
                f"entity {payload['id']}: support_shape must be a string naming a built form"
            )
        return cls(
            id=str(payload["id"]),
            category=str(payload["category"]),
            physics_type=physics_type,
            mobility=mobility,
            # Default to hard-required: a silently optional object would be the easy way to
            # make a failing scene pass by deleting the thing that failed.
            hard_required=bool(payload.get("hard_required", True)),
            role=role,
            quantity=int(payload.get("quantity", 1)),
            size_bounds_m=parsed_bounds,
            notes=str(payload.get("notes", "")),
            material_tokens=tuple(str(token) for token in material_tokens),
            requires_support_surface=requires_support_surface,
            support_shape=support_shape,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "physics_type": self.physics_type.value,
            "mobility": self.mobility.value,
            "hard_required": self.hard_required,
            "role": self.role.value,
            "quantity": self.quantity,
        }
        if self.size_bounds_m is not None:
            payload["size_bounds_m"] = [axis.to_list() for axis in self.size_bounds_m]
        if self.notes:
            payload["notes"] = self.notes
        if self.material_tokens:
            payload["material_tokens"] = list(self.material_tokens)
        if self.requires_support_surface:
            payload["requires_support_surface"] = True
        if self.support_shape:
            payload["support_shape"] = self.support_shape
        return payload


def scope_of(entity: EntitySpec) -> SceneScope:
    """Classify an entity from its role, the sole authority for task scope.

    This is deliberately narrower than :func:`relation_closure_ids`: a soft relation may name a
    contextual entity without making that entity part of the task cell.
    """
    if entity.role in (EntityRole.TASK_TARGET, EntityRole.SUPPORT):
        return SceneScope.CORE
    return SceneScope.PERIPHERY


@dataclass(frozen=True, slots=True)
class RelationSpec:
    """A relation that must hold in the generated scene.

    ``hard`` relations may not be dropped, relaxed or replaced by a different predicate.
    A patch that tries is rejected by the goal-preservation check, not by this type.
    """

    id: str
    subject: str
    predicate: Predicate
    object: str
    hard: bool = True
    tolerance_m: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise SpecError("relation id must be non-empty")
        if not self.subject or not self.object:
            raise SpecError(f"relation {self.id}: subject and object must both be named")
        if self.subject == self.object:
            raise SpecError(f"relation {self.id}: subject and object must differ")
        if self.tolerance_m is not None and self.tolerance_m < 0:
            raise SpecError(f"relation {self.id}: tolerance_m must be non-negative")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RelationSpec:
        known = {"id", "subject", "predicate", "object", "hard", "tolerance_m"}
        unknown = set(payload) - known
        if unknown:
            raise SpecError(f"relation has unknown keys: {sorted(unknown)}")
        for required in ("id", "subject", "predicate", "object"):
            if required not in payload:
                raise SpecError(f"relation is missing required key {required!r}: {dict(payload)}")
        try:
            predicate = Predicate(payload["predicate"])
        except ValueError as error:
            raise SpecError(f"unknown predicate {payload['predicate']!r}") from error
        tolerance = payload.get("tolerance_m")
        return cls(
            id=str(payload["id"]),
            subject=str(payload["subject"]),
            predicate=predicate,
            object=str(payload["object"]),
            hard=bool(payload.get("hard", True)),
            tolerance_m=None if tolerance is None else float(tolerance),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate.value,
            "object": self.object,
            "hard": self.hard,
        }
        if self.tolerance_m is not None:
            payload["tolerance_m"] = self.tolerance_m
        return payload


@dataclass(frozen=True, slots=True)
class GoalPredicate:
    """A predicate that must hold **after** the robot acts.

    Deliberately a separate type from :class:`RelationSpec`. The generation solvers never
    read these, so a goal cannot be satisfied by placing the object there at build time.
    """

    subject: str
    predicate: Predicate
    object: str
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoalPredicate:
        known = {"subject", "predicate", "object", "description"}
        unknown = set(payload) - known
        if unknown:
            raise SpecError(f"goal predicate has unknown keys: {sorted(unknown)}")
        for required in ("subject", "predicate", "object"):
            if required not in payload:
                raise SpecError(f"goal predicate is missing {required!r}: {dict(payload)}")
        try:
            predicate = Predicate(payload["predicate"])
        except ValueError as error:
            raise SpecError(f"unknown goal predicate {payload['predicate']!r}") from error
        return cls(
            subject=str(payload["subject"]),
            predicate=predicate,
            object=str(payload["object"]),
            description=str(payload.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "subject": self.subject,
            "predicate": self.predicate.value,
            "object": self.object,
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class JointTask:
    """Articulation task state, kept as three distinct quantities (§5.1).

    ``initial_q`` is what the joint is at when delivered, ``goal_q`` is where the task wants
    it, and ``test_q_schedule`` is the ordered sweep the validator drives. Overwriting one
    with another is how "the door started open" gets lost.
    """

    joint_id: str
    initial_q: float
    goal_q: float
    test_q_schedule: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.joint_id:
            raise SpecError("joint task must name a joint_id")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JointTask:
        known = {"joint_id", "initial_q", "goal_q", "test_q_schedule"}
        unknown = set(payload) - known
        if unknown:
            raise SpecError(f"joint task has unknown keys: {sorted(unknown)}")
        for required in ("joint_id", "initial_q", "goal_q"):
            if required not in payload:
                raise SpecError(f"joint task is missing {required!r}")
        schedule = payload.get("test_q_schedule", [])
        if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
            raise SpecError("joint task test_q_schedule must be a list of joint values")
        return cls(
            joint_id=str(payload["joint_id"]),
            initial_q=float(payload["initial_q"]),
            goal_q=float(payload["goal_q"]),
            test_q_schedule=tuple(float(value) for value in schedule),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "initial_q": self.initial_q,
            "goal_q": self.goal_q,
            "test_q_schedule": list(self.test_q_schedule),
        }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """The task, its robot profile, and the post-execution goals."""

    action: str
    robot_profile: str
    targets: tuple[str, ...]
    goal_predicates: tuple[GoalPredicate, ...] = ()
    joint_task: JointTask | None = None

    def __post_init__(self) -> None:
        if not self.action:
            raise SpecError("task.action must be non-empty")
        if not self.robot_profile:
            raise SpecError("task.robot_profile must be non-empty")
        if not self.targets:
            raise SpecError("task.targets must name at least one entity")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskSpec:
        known = {"action", "robot_profile", "targets", "goal_predicates", "joint_task"}
        unknown = set(payload) - known
        if unknown:
            raise SpecError(f"task has unknown keys: {sorted(unknown)}")
        for required in ("action", "robot_profile", "targets"):
            if required not in payload:
                raise SpecError(f"task is missing {required!r}")
        targets = payload["targets"]
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence) or not targets:
            raise SpecError("task.targets must be a non-empty list of entity ids")
        raw_goals = payload.get("goal_predicates", [])
        if isinstance(raw_goals, (str, bytes)) or not isinstance(raw_goals, Sequence):
            raise SpecError("task.goal_predicates must be a list")
        joint_raw = payload.get("joint_task")
        return cls(
            action=str(payload["action"]),
            robot_profile=str(payload["robot_profile"]),
            targets=tuple(str(item) for item in targets),
            goal_predicates=tuple(GoalPredicate.from_dict(item) for item in raw_goals),
            joint_task=None if joint_raw is None else JointTask.from_dict(joint_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action,
            "robot_profile": self.robot_profile,
            "targets": list(self.targets),
            "goal_predicates": [goal.to_dict() for goal in self.goal_predicates],
        }
        if self.joint_task is not None:
            payload["joint_task"] = self.joint_task.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """The frozen request. Everything downstream is derived from this plus the seed."""

    scene_id: str
    seed: int
    prompt: str
    room: RoomSpec
    style: StylePlan
    requested_backends: tuple[str, ...]
    requested_readiness: str
    entities: tuple[EntitySpec, ...]
    initial_relations: tuple[RelationSpec, ...]
    task: TaskSpec
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise SpecError("scene_id must be non-empty")
        if not self.requested_backends:
            raise SpecError("at least one requested backend is required")
        if len(set(self.requested_backends)) != len(self.requested_backends):
            raise SpecError(f"requested_backends has duplicates: {self.requested_backends}")
        if not self.entities:
            raise SpecError("a scene must request at least one entity")

        ids = [entity.id for entity in self.entities]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise SpecError(f"duplicate entity ids: {duplicates}")
        known = set(ids)

        for relation in self.initial_relations:
            for end in (relation.subject, relation.object):
                if end not in known:
                    raise SpecError(
                        f"relation {relation.id} references unknown entity {end!r}; known: {sorted(known)}"
                    )
        relation_ids = [relation.id for relation in self.initial_relations]
        if len(set(relation_ids)) != len(relation_ids):
            raise SpecError(f"duplicate relation ids: {relation_ids}")

        for target in self.task.targets:
            if target not in known:
                raise SpecError(f"task.targets references unknown entity {target!r}")

        # A goal predicate that is also an initial relation can be satisfied without the
        # robot doing anything, which is the failure mode §5.1 warns about.
        initial_keys = {
            (relation.subject, relation.predicate, relation.object) for relation in self.initial_relations
        }
        for goal in self.task.goal_predicates:
            if (goal.subject, goal.predicate, goal.object) in initial_keys:
                raise SpecError(
                    f"goal predicate {goal.subject} {goal.predicate.value} {goal.object} is also an "
                    "initial relation; a goal that already holds at generation time cannot "
                    "demonstrate that the task was performed"
                )
            for end in (goal.subject, goal.object):
                if end not in known:
                    raise SpecError(f"goal predicate references unknown entity {end!r}")

        # Deformable entities in a first-release request are a capability question, not a
        # parse question. The preflight reports the gap; the spec still has to be parseable.
        if self.task.joint_task is not None:
            articulated = {e.id for e in self.entities if e.physics_type is PhysicsType.ARTICULATED}
            if not articulated:
                raise SpecError(
                    f"task declares a joint_task for {self.task.joint_task.joint_id!r} but no "
                    "articulated entity is requested"
                )

    @property
    def hard_entities(self) -> tuple[EntitySpec, ...]:
        """Entities that may never be deleted, substituted by a primitive, or downgraded."""
        return tuple(entity for entity in self.entities if entity.hard_required)

    @property
    def hard_relations(self) -> tuple[RelationSpec, ...]:
        return tuple(relation for relation in self.initial_relations if relation.hard)

    @property
    def task_targets(self) -> tuple[EntitySpec, ...]:
        return tuple(entity for entity in self.entities if entity.is_task_target)

    def entity(self, entity_id: str) -> EntitySpec:
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        raise SpecError(f"unknown entity {entity_id!r}")

    def protected_summary(self) -> dict[str, Any]:
        """The goal set a semantic patch must preserve.

        Handed to the failure replanner and checked against every patch: the patch may change
        how a goal is met, never whether it is met.
        """
        return {
            "hard_entities": sorted(entity.id for entity in self.hard_entities),
            "task_targets": sorted(entity.id for entity in self.task_targets),
            "hard_relations": sorted(
                f"{relation.subject}:{relation.predicate.value}:{relation.object}"
                for relation in self.hard_relations
            ),
            "task_action": self.task.action,
            "goal_predicates": sorted(
                f"{goal.subject}:{goal.predicate.value}:{goal.object}"
                for goal in self.task.goal_predicates
            ),
            "requested_backends": sorted(self.requested_backends),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "seed": self.seed,
            "prompt": self.prompt,
            "room": self.room.to_dict(),
            "style": self.style.to_dict(),
            "requested_backends": list(self.requested_backends),
            "requested_readiness": self.requested_readiness,
            "entities": [entity.to_dict() for entity in self.entities],
            "initial_relations": [relation.to_dict() for relation in self.initial_relations],
            "task": self.task.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SceneSpec:
        known = {
            "schema_version", "scene_id", "seed", "prompt", "room", "style",
            "requested_backends", "requested_readiness", "entities", "initial_relations", "task",
        }
        unknown = set(payload) - known
        if unknown:
            raise SpecError(f"scene spec has unknown keys: {sorted(unknown)}")
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise SpecError(f"unsupported scene spec schema_version {version!r}, expected {SCHEMA_VERSION!r}")
        for required in ("scene_id", "room", "entities", "task"):
            if required not in payload:
                raise SpecError(f"scene spec is missing {required!r}")
        backends = payload.get("requested_backends")
        if isinstance(backends, (str, bytes)) or not isinstance(backends, Sequence) or not backends:
            raise SpecError("requested_backends must be a non-empty list")
        readiness = payload.get("requested_readiness")
        if not isinstance(readiness, str) or not readiness:
            raise SpecError("requested_readiness must be a non-empty string")
        return cls(
            scene_id=str(payload["scene_id"]),
            seed=int(payload.get("seed", 0)),
            prompt=str(payload.get("prompt", "")),
            room=RoomSpec.from_dict(payload["room"]),
            style=StylePlan.from_dict(payload.get("style", {"theme": "unspecified", "palette": ["neutral"]})),
            requested_backends=tuple(str(item) for item in backends),
            requested_readiness=readiness,
            entities=tuple(EntitySpec.from_dict(item) for item in payload["entities"]),
            initial_relations=tuple(
                RelationSpec.from_dict(item) for item in payload.get("initial_relations", [])
            ),
            task=TaskSpec.from_dict(payload["task"]),
            schema_version=version,
        )


def relation_closure_ids(spec: SceneSpec) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Entities and relations named by the initial task structure.

    The entity set includes every task target and both endpoints of every initial relation,
    including soft relations. Randomization and requested-entity coverage intentionally use this
    wider set; it must not be replaced by the role-based core.
    """
    entities = set(spec.task.targets)
    for relation in spec.initial_relations:
        entities.update((relation.subject, relation.object))
    return tuple(sorted(entities)), tuple(sorted(relation.id for relation in spec.initial_relations))
