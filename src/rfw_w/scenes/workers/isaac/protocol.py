# SPDX-License-Identifier: Apache-2.0
"""The request and report format shared between the orchestrator and the Isaac worker.

Stdlib-only on purpose: the orchestrator writes a request and reads a report without loading
Isaac, and the worker parses the same format under Isaac's interpreter. Neither imports the
other's code.

Everything the worker needs to be reproducible is in the request, and everything it must be
judged by is in the report: the scene hash it rendered, the Isaac version, the camera intrinsics,
and the depth convention. A render without those cannot be tied to a scene version, and an image
that cannot be tied to a version cannot be allowed to approve one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "pvscene.isaac-worker-request/v1"
REPORT_SCHEMA_VERSION = "pvscene.isaac-worker-report/v1"

class ProtocolError(ValueError):
    """The worker request or report contradicts the contract."""


@dataclass(frozen=True, slots=True)
class MeshInstance:
    """One object to put in the stage.

    ``normalisation`` names the frame conversion the asset carries, so the worker applies it
    rather than guessing: the simready library is Y-up and the record says so. Applying it twice,
    or once when the asset is already Z-up, lays the object on its side.
    """

    instance_id: str
    entity_id: str
    visual_uri: str
    collision_uri: str | None
    translation: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    mass_kg: float | None = None
    dynamic: bool = False
    normalisation: str = "none"

    def __post_init__(self) -> None:
        if not self.instance_id or not self.entity_id:
            raise ProtocolError("instance must name both an instance_id and an entity_id")
        if not self.visual_uri:
            raise ProtocolError(f"{self.instance_id}: a visual mesh is required")
        if len(self.translation) != 3 or len(self.rotation_wxyz) != 4:
            raise ProtocolError(f"{self.instance_id}: pose must be translation(3) + quaternion(4)")
        if self.dynamic and (self.mass_kg is None or self.mass_kg <= 0):
            raise ProtocolError(
                f"{self.instance_id}: a dynamic body needs a positive mass, otherwise it cannot "
                "be settled and the settle result would describe nothing"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instance_id": self.instance_id,
            "entity_id": self.entity_id,
            "visual_uri": self.visual_uri,
            "translation": list(self.translation),
            "rotation_wxyz": list(self.rotation_wxyz),
            "dynamic": self.dynamic,
            "normalisation": self.normalisation,
        }
        if self.collision_uri:
            payload["collision_uri"] = self.collision_uri
        if self.mass_kg is not None:
            payload["mass_kg"] = self.mass_kg
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MeshInstance:
        return cls(
            instance_id=str(payload["instance_id"]),
            entity_id=str(payload["entity_id"]),
            visual_uri=str(payload["visual_uri"]),
            collision_uri=payload.get("collision_uri"),
            translation=tuple(float(v) for v in payload["translation"]),  # type: ignore[arg-type]
            rotation_wxyz=tuple(float(v) for v in payload["rotation_wxyz"]),  # type: ignore[arg-type]
            mass_kg=None if payload.get("mass_kg") is None else float(payload["mass_kg"]),
            dynamic=bool(payload.get("dynamic", False)),
            normalisation=str(payload.get("normalisation", "none")),
        )


@dataclass(frozen=True, slots=True)
class RoomOpening:
    """A doorway or window, positioned on the room outline.

    Carried in the request because the alternative is what happened before this existed: the shell
    computed openings, nothing passed them to the worker, and every wall was authored solid. The
    render then showed a sealed box while the manifest listed a door, and a spatial gate that only
    read the shell data went on passing.
    """

    opening_id: str
    center_xy: tuple[float, float]
    width_m: float
    height_m: float
    sill_m: float = 0.0
    kind: str = "door"

    def __post_init__(self) -> None:
        if not self.opening_id:
            raise ProtocolError("opening must have an id")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ProtocolError(f"opening {self.opening_id}: width and height must be positive")
        if self.sill_m < 0:
            raise ProtocolError(f"opening {self.opening_id}: sill must not be negative")

    def as_tuple(self) -> tuple[str, float, float, float, float, float]:
        return (
            self.opening_id, self.center_xy[0], self.center_xy[1],
            self.width_m, self.height_m, self.sill_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "opening_id": self.opening_id,
            "kind": self.kind,
            "center_xy": list(self.center_xy),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "sill_m": self.sill_m,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoomOpening:
        return cls(
            opening_id=str(payload["opening_id"]),
            center_xy=(float(payload["center_xy"][0]), float(payload["center_xy"][1])),
            width_m=float(payload["width_m"]),
            height_m=float(payload["height_m"]),
            sill_m=float(payload.get("sill_m", 0.0)),
            kind=str(payload.get("kind", "door")),
        )


@dataclass(frozen=True, slots=True)
class RoomGeometry:
    """The floor polygon, the room height, and the openings cut into its walls.

    The worker builds the floor and walls from these vertices, so a notched room keeps its notch
    in the render. Generating a box from the bounding extent would render a room the scene does
    not have, and the image would then disagree with the geometry the gates validated.
    """

    polygon_xy: tuple[tuple[float, float], ...]
    height_m: float
    floor_z_m: float = 0.0
    openings: tuple[RoomOpening, ...] = ()
    shell_uri: str | None = None

    def __post_init__(self) -> None:
        if len(self.polygon_xy) < 3:
            raise ProtocolError("room polygon needs at least 3 vertices")
        if self.height_m <= 0:
            raise ProtocolError(f"room height must be positive, got {self.height_m}")
        ids = [opening.opening_id for opening in self.openings]
        if len(set(ids)) != len(ids):
            raise ProtocolError(f"duplicate opening ids: {sorted(ids)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon_xy": [list(vertex) for vertex in self.polygon_xy],
            "height_m": self.height_m,
            "floor_z_m": self.floor_z_m,
            "openings": [opening.to_dict() for opening in self.openings],
            **({"shell_uri": self.shell_uri} if self.shell_uri is not None else {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoomGeometry:
        return cls(
            polygon_xy=tuple((float(v[0]), float(v[1])) for v in payload["polygon_xy"]),
            height_m=float(payload["height_m"]),
            floor_z_m=float(payload.get("floor_z_m", 0.0)),
            openings=tuple(RoomOpening.from_dict(item) for item in payload.get("openings", ())),
            shell_uri=payload.get("shell_uri"),
        )


@dataclass(frozen=True, slots=True)
class CameraRequest:
    """One camera: where it stands, what it looks at, and what it must capture."""

    camera_id: str
    purpose: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_length_mm: float = 24.0
    # Carried rather than left to the renderer. Until these existed the worker authored only
    # `focalLength`, so the horizontal aperture was whatever USD defaults to (20.955 mm) while the
    # Blender side of the same comparison used a hand-coded 22.044444 mm -- a 2.55 degree
    # difference in horizontal field of view across two images that were judged side by side.
    # The defaults are USD's own, so a request that omits them still frames exactly as before.
    horizontal_aperture_mm: float = 20.955
    vertical_aperture_mm: float = 15.2908
    resolution: tuple[int, int] = (1280, 720)
    target_instance_ids: tuple[str, ...] = ()
    require_target_visible: bool = True
    #: What this view owes an image of: at least this many *entity* instances have to be visible
    #: in it. 0 means the view promises nothing, which is the right declaration for a shot whose
    #: job is the shell or a wall. SPEC-015 R3/R4: a view that misses its own declaration is a
    #: named gap for *that view*, not a failed delivery -- the run needed 20 minutes and every
    #: other view was fine (measured: two views framing nothing cost the whole 2026-09-24 large
    #: office run its render, its critique and its readiness level).
    minimum_visible_instances: int = 0
    save_depth: bool = True
    save_instance_ids: bool = True

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ProtocolError("camera must have an id")
        if self.purpose not in ("layout", "relation", "final", "export"):
            raise ProtocolError(f"camera {self.camera_id}: unknown purpose {self.purpose!r}")
        if self.focal_length_mm <= 0:
            raise ProtocolError(f"camera {self.camera_id}: focal length must be positive")
        if self.minimum_visible_instances < 0:
            raise ProtocolError(
                f"camera {self.camera_id}: minimum_visible_instances cannot be negative"
            )
        if self.horizontal_aperture_mm <= 0 or self.vertical_aperture_mm <= 0:
            raise ProtocolError(
                f"camera {self.camera_id}: apertures must be positive, got "
                f"{self.horizontal_aperture_mm}x{self.vertical_aperture_mm}"
            )
        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise ProtocolError(f"camera {self.camera_id}: resolution must be positive")
        if tuple(self.eye) == tuple(self.target):
            raise ProtocolError(
                f"camera {self.camera_id}: eye and target coincide, so the view direction is undefined"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "purpose": self.purpose,
            "eye": list(self.eye),
            "target": list(self.target),
            "focal_length_mm": self.focal_length_mm,
            "horizontal_aperture_mm": self.horizontal_aperture_mm,
            "vertical_aperture_mm": self.vertical_aperture_mm,
            "resolution": list(self.resolution),
            "target_instance_ids": list(self.target_instance_ids),
            "require_target_visible": self.require_target_visible,
            "minimum_visible_instances": self.minimum_visible_instances,
            "save_depth": self.save_depth,
            "save_instance_ids": self.save_instance_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraRequest:
        return cls(
            camera_id=str(payload["camera_id"]),
            purpose=str(payload["purpose"]),
            eye=tuple(float(v) for v in payload["eye"]),  # type: ignore[arg-type]
            target=tuple(float(v) for v in payload["target"]),  # type: ignore[arg-type]
            focal_length_mm=float(payload.get("focal_length_mm", 24.0)),
            horizontal_aperture_mm=float(payload.get("horizontal_aperture_mm", 20.955)),
            vertical_aperture_mm=float(payload.get("vertical_aperture_mm", 15.2908)),
            resolution=tuple(int(v) for v in payload.get("resolution", (1280, 720))),  # type: ignore[arg-type]
            target_instance_ids=tuple(str(v) for v in payload.get("target_instance_ids", ())),
            require_target_visible=bool(payload.get("require_target_visible", True)),
            minimum_visible_instances=int(payload.get("minimum_visible_instances", 0)),
            save_depth=bool(payload.get("save_depth", True)),
            save_instance_ids=bool(payload.get("save_instance_ids", True)),
        )


@dataclass(frozen=True, slots=True)
class LightingRequest:
    """The lighting the renderer is asked for, carried rather than assumed.

    It exists because the plan's ``AppearanceSettings`` had **no path to the renderer**:
    ``render_worker._author_lighting`` hardcoded its three intensities and two colours, and this
    request carried no lighting field at all. They happened to equal the defaults, so nothing looked
    wrong -- but it meant `repair.appearance` changed the plan, changed the scene version, invalidated
    the render and the critique, forced both to re-run, and produced an identical image. The critic
    then correctly reported no improvement and the rung was rolled back: an entire track that could
    not change anything it claimed to change.

    Defaults are the values the renderer used to hardcode, so a request built without lighting
    renders exactly as before.
    """

    dome_intensity: float = 900.0
    sun_intensity: float = 2600.0
    ceiling_fill_intensity: float = 700.0
    dome_color: tuple[float, float, float] = (0.92, 0.94, 1.0)
    sun_color: tuple[float, float, float] = (1.0, 0.96, 0.88)

    def __post_init__(self) -> None:
        for name in ("dome_intensity", "sun_intensity", "ceiling_fill_intensity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ProtocolError(f"{name} must be a non-negative finite intensity, got {value}")
        for name in ("dome_color", "sun_color"):
            colour = getattr(self, name)
            if len(colour) != 3 or any(not math.isfinite(channel) or channel < 0 for channel in colour):
                raise ProtocolError(f"{name} must be three non-negative finite channels, got {colour}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dome_intensity": self.dome_intensity,
            "sun_intensity": self.sun_intensity,
            "ceiling_fill_intensity": self.ceiling_fill_intensity,
            "dome_color": list(self.dome_color),
            "sun_color": list(self.sun_color),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LightingRequest:
        return cls(
            dome_intensity=float(payload.get("dome_intensity", 900.0)),
            sun_intensity=float(payload.get("sun_intensity", 2600.0)),
            ceiling_fill_intensity=float(payload.get("ceiling_fill_intensity", 700.0)),
            dome_color=tuple(float(c) for c in payload.get("dome_color", (0.92, 0.94, 1.0))),
            sun_color=tuple(float(c) for c in payload.get("sun_color", (1.0, 0.96, 0.88))),
        )

    @classmethod
    def from_appearance(cls, appearance: Any) -> LightingRequest:
        """Build from a `solvers.furnish.AppearanceSettings`.

        Takes the object structurally rather than importing it: this module is the wire format and
        is parsed from both sides, so it must not depend on the solvers.
        """
        return cls(
            dome_intensity=float(appearance.dome_intensity),
            sun_intensity=float(appearance.sun_intensity),
            ceiling_fill_intensity=float(appearance.ceiling_fill_intensity),
            dome_color=tuple(float(c) for c in appearance.dome_color),
            sun_color=tuple(float(c) for c in appearance.sun_color),
        )

    @classmethod
    def from_plan(cls, plan: Any) -> LightingRequest:
        """Build from a `contracts.lighting.LightingPlan` -- the calibrated profile's own numbers.

        The appearance defaults were the reason a delivered room rendered its task target
        essentially unlit (SPEC-008 TASK 1 measured median Y = 0.0020 with 99.89% of the target's
        pixels below 0.01). A plan carries what the profile was calibrated to, the room it was sized
        for and the node losses it knows about, so the renderer is told the profile rather than a
        number that happened to be the default.

        Structural, like `from_appearance` and for the same reason: the wire format is parsed from
        both sides and must not import the core. The colours are put in the same type every other
        constructor uses -- a plan serialises them as lists, and a request whose colours were lists
        would compare unequal to the same request read back from disk.
        """
        fields = plan.to_request_fields()
        return cls(
            dome_intensity=float(fields["dome_intensity"]),
            sun_intensity=float(fields["sun_intensity"]),
            ceiling_fill_intensity=float(fields["ceiling_fill_intensity"]),
            dome_color=tuple(float(channel) for channel in fields["dome_color"]),
            sun_color=tuple(float(channel) for channel in fields["sun_color"]),
        )


# How a window pane is rendered. Carried on the request rather than chosen inside the worker,
# because which of these is right is a property of the renderer's behaviour, not of the scene, and
# a stage that picked for itself could not be compared against a stage that picked differently.
#
#   semi_transparent  a tinted pane at `glass_opacity`. The default.
#   opaque_tint       fully opaque: the fallback if the semi-transparent one renders as a mirror.
#   emissive_sky      a sky colour in both diffuse and emissive, which is what v2 shipped in place
#                     of glass after measuring that headless RTX turns a near-transparent
#                     UsdPreviewSurface into a mirror that hides the backdrop behind it.
GLASS_MODES = ("semi_transparent", "opaque_tint", "emissive_sky")

#: The shell roles a request may rebind one at a time, each from the `RoomMaterialsRequest` field
#: of the same name. Only these five: the rest of the shell (skirting, doors, window glass) keeps
#: the provider's own material, because the request carries no surface that means "skirting".
OVERRIDABLE_SHELL_ROLES = frozenset({"wall", "floor", "ceiling", "facade", "frame"})


def _shell_override_from_wire(value: Any) -> bool | tuple[str, ...]:
    """Read the field either way round. A list on the wire is a role list; a bool is a bool.

    Written out because ``bool(["wall"])`` is ``True``, i.e. the obvious reading turns a per-role
    override into the one-albedo-everywhere one and the picture changes without the request saying
    it did.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(str(role) for role in value)
    raise ProtocolError(
        f"shell_material_override must be a bool or a list of role names, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class SurfaceRequest:
    """One flat surface on the wire: a colour, a roughness and a metallic."""

    colour: tuple[float, float, float] = (0.82, 0.79, 0.74)
    roughness: float = 0.85
    metallic: float = 0.0

    def __post_init__(self) -> None:
        if len(self.colour) != 3:
            raise ProtocolError(f"surface colour must be three channels, got {self.colour}")
        for channel in self.colour:
            if not math.isfinite(channel) or not 0.0 <= channel <= 1.0:
                raise ProtocolError(f"surface colour channels must be within [0, 1], got {self.colour}")
        for name in ("roughness", "metallic"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ProtocolError(f"surface {name} must be within [0, 1], got {value}")

    def to_dict(self) -> dict[str, Any]:
        return {"colour": list(self.colour), "roughness": self.roughness, "metallic": self.metallic}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SurfaceRequest:
        return cls(
            colour=tuple(float(c) for c in payload.get("colour", (0.82, 0.79, 0.74))),
            roughness=float(payload.get("roughness", 0.85)),
            metallic=float(payload.get("metallic", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class RoomMaterialsRequest:
    """The shell's surfaces, carried rather than hardcoded in the renderer.

    Same defect as ``LightingRequest`` before it, and the same shape of fix: ``_author_room`` built
    its three materials from module constants, so a spec that asked for a different palette changed
    the plan, changed the scene version, invalidated the render, and produced an identical image.

    Defaults are exactly the constants the renderer used, so a request built without materials
    renders byte-identically to one built before this existed.
    """

    wall: SurfaceRequest = SurfaceRequest()
    floor: SurfaceRequest = SurfaceRequest(colour=(0.42, 0.30, 0.21), roughness=0.6)
    ceiling: SurfaceRequest = SurfaceRequest(colour=(0.90, 0.89, 0.87), roughness=0.9)
    # What shows through a window and behind its pane.
    facade: SurfaceRequest = SurfaceRequest(colour=(0.55, 0.68, 0.85), roughness=0.9)
    frame: SurfaceRequest = SurfaceRequest(colour=(0.055, 0.065, 0.07), roughness=0.28, metallic=0.72)
    glass: SurfaceRequest = SurfaceRequest(colour=(0.78, 0.84, 0.88), roughness=0.12)
    glass_opacity: float = 0.45
    glass_mode: str = "semi_transparent"

    def __post_init__(self) -> None:
        if not math.isfinite(self.glass_opacity) or not 0.0 <= self.glass_opacity <= 1.0:
            raise ProtocolError(
                f"glass opacity must be within [0, 1], got {self.glass_opacity}"
            )
        if self.glass_mode not in GLASS_MODES:
            raise ProtocolError(
                f"glass mode {self.glass_mode!r} is not one of {list(GLASS_MODES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall": self.wall.to_dict(),
            "floor": self.floor.to_dict(),
            "ceiling": self.ceiling.to_dict(),
            "facade": self.facade.to_dict(),
            "frame": self.frame.to_dict(),
            "glass": self.glass.to_dict(),
            "glass_opacity": self.glass_opacity,
            "glass_mode": self.glass_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoomMaterialsRequest:
        base = cls()
        return cls(
            wall=SurfaceRequest.from_dict(payload["wall"]) if "wall" in payload else base.wall,
            floor=SurfaceRequest.from_dict(payload["floor"]) if "floor" in payload else base.floor,
            ceiling=(SurfaceRequest.from_dict(payload["ceiling"])
                     if "ceiling" in payload else base.ceiling),
            facade=SurfaceRequest.from_dict(payload["facade"]) if "facade" in payload else base.facade,
            frame=SurfaceRequest.from_dict(payload["frame"]) if "frame" in payload else base.frame,
            glass=SurfaceRequest.from_dict(payload["glass"]) if "glass" in payload else base.glass,
            glass_opacity=float(payload.get("glass_opacity", base.glass_opacity)),
            glass_mode=str(payload.get("glass_mode", base.glass_mode)),
        )

    @classmethod
    def from_materials(cls, materials: Any) -> RoomMaterialsRequest:
        """Build from a `solvers.materials.RoomMaterials`.

        Takes the object structurally rather than importing it: this module is the wire format,
        parsed from both sides, so it must not depend on the solvers.
        """
        def surface(item: Any) -> SurfaceRequest:
            return SurfaceRequest(
                colour=tuple(float(c) for c in item.colour),
                roughness=float(item.roughness),
                metallic=float(item.metallic),
            )

        return cls(
            wall=surface(materials.wall),
            floor=surface(materials.floor),
            ceiling=surface(materials.ceiling),
            facade=surface(materials.facade),
            frame=surface(materials.frame),
            glass=surface(materials.glass),
            glass_opacity=float(materials.glass_opacity),
            glass_mode=str(materials.glass_mode),
        )


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """Everything the worker needs, and nothing that identifies the machine it runs on."""

    scene_id: str
    scene_version: str
    room: RoomGeometry
    instances: tuple[MeshInstance, ...]
    cameras: tuple[CameraRequest, ...]
    simulate_seconds: float = 5.0
    physics_hz: int = 120
    active_gpu: int = 0
    # Carried, not assumed. See LightingRequest: the renderer hardcoded these and the appearance
    # track could therefore not change the picture.
    lighting: LightingRequest = field(default_factory=LightingRequest)
    room_materials: RoomMaterialsRequest = field(default_factory=RoomMaterialsRequest)
    # Opt-in, and false by default so no delivery's appearance moves. `room_materials` reaches the
    # procedural fallback only: when the room carries a `shell_uri`, `_author_room` hands off to
    # `author_shell` and returns before a single material is built, so the shell renders in whatever
    # the provider authored into its USD. That is right for a delivery -- Infinigen's paint is part
    # of the scene -- and wrong for a figure that puts five renderers side by side, because the
    # other four read the bundle's OBJ shell, which carries no MTL at all. Measured 2026-09-25 on
    # SPEC-014: the wall the camera faces came out near-black in Isaac and light grey in the other
    # four, and no lighting change touched it -- raising the ceiling fill 18x only revealed it as
    # Infinigen's dark purple accent wall. With this set, every mesh under /World/Room is rebound to
    # `room_materials.wall`, which is the same one-albedo-everywhere treatment the OBJ shell gets.
    #
    # SPEC-016 widened it: `True` keeps that behaviour byte for byte, and a **tuple of role names**
    # rebinds only those roles, each from its own field of `room_materials`. One albedo everywhere
    # is right for the five-renderer figure (the other four read an MTL-less OBJ) and wrong for a
    # room portrait, where a floor that is the same flat colour as the walls is the reason the
    # picture reads as a white box. The role of a prim is decided by its path
    # (`shell_reference.shell_role`), which is the provider's own factory naming.
    shell_material_override: bool | tuple[str, ...] = False
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ProtocolError("scene_id must be non-empty")
        if len(self.scene_version) != 64:
            raise ProtocolError(
                f"scene_version must be the scene hash it was built from, got {self.scene_version!r}"
            )
        if not self.instances:
            raise ProtocolError("a scene must contain at least one instance")
        if not self.cameras:
            raise ProtocolError("at least one camera is required; a render with no view is not a render")
        if self.simulate_seconds < 0:
            raise ProtocolError(f"simulate_seconds must not be negative, got {self.simulate_seconds}")
        if self.physics_hz <= 0:
            raise ProtocolError(f"physics_hz must be positive, got {self.physics_hz}")

        ids = [instance.instance_id for instance in self.instances]
        if len(set(ids)) != len(ids):
            raise ProtocolError(f"duplicate instance ids: {sorted(ids)}")
        known = set(ids)
        for camera in self.cameras:
            missing = sorted(set(camera.target_instance_ids) - known)
            if missing:
                raise ProtocolError(
                    f"camera {camera.camera_id} targets unknown instances {missing}"
                )
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(set(camera_ids)) != len(camera_ids):
            raise ProtocolError(f"duplicate camera ids: {sorted(camera_ids)}")
        if not isinstance(self.shell_material_override, bool):
            unknown = sorted(set(self.shell_material_override) - OVERRIDABLE_SHELL_ROLES)
            if unknown:
                raise ProtocolError(
                    f"shell_material_override names roles with no surface to bind from: {unknown}; "
                    f"known roles are {sorted(OVERRIDABLE_SHELL_ROLES)}"
                )
            if not self.shell_material_override:
                raise ProtocolError(
                    "shell_material_override is an empty role list: say False to leave the shell's "
                    "own materials alone, rather than declaring an override that binds nothing"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "scene_version": self.scene_version,
            "room": self.room.to_dict(),
            "instances": [instance.to_dict() for instance in self.instances],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "simulate_seconds": self.simulate_seconds,
            "physics_hz": self.physics_hz,
            "active_gpu": self.active_gpu,
            "lighting": self.lighting.to_dict(),
            "room_materials": self.room_materials.to_dict(),
            "shell_material_override": (self.shell_material_override
                                        if isinstance(self.shell_material_override, bool)
                                        else list(self.shell_material_override)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkerRequest:
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported request schema_version {version!r}")
        return cls(
            scene_id=str(payload["scene_id"]),
            scene_version=str(payload["scene_version"]),
            room=RoomGeometry.from_dict(payload["room"]),
            instances=tuple(MeshInstance.from_dict(item) for item in payload["instances"]),
            cameras=tuple(CameraRequest.from_dict(item) for item in payload["cameras"]),
            simulate_seconds=float(payload.get("simulate_seconds", 5.0)),
            physics_hz=int(payload.get("physics_hz", 120)),
            active_gpu=int(payload.get("active_gpu", 0)),
            lighting=LightingRequest.from_dict(payload.get("lighting") or {}),
            room_materials=RoomMaterialsRequest.from_dict(payload.get("room_materials") or {}),
            shell_material_override=_shell_override_from_wire(
                payload.get("shell_material_override", False)),
            schema_version=version,
        )


@dataclass(frozen=True, slots=True)
class CaptureReport:
    """What one camera produced, with the numbers that let the image be audited."""

    camera_id: str
    rgb_path: str
    depth_path: str | None
    instance_id_path: str | None
    resolution: tuple[int, int]
    depth_convention: str
    depth_valid_fraction: float
    depth_min_m: float | None
    depth_max_m: float | None
    color_std: float
    visible_instance_ids: tuple[int, ...]
    # Renderer id -> scene instance id for this camera. ID spaces may differ by render product, so
    # a first-camera global map cannot label every saved channel correctly.
    instance_id_map: Mapping[str, str]
    target_instance_ids: tuple[str, ...]
    target_visible: bool | None
    nonblank: bool
    #: The camera saved an instance-id channel and no entity was in it. Not a defect on its own --
    #: a wall-filling view legitimately contains nothing -- but a reader has to be able to tell it
    #: apart from a view that contained things, and the view's own declaration decides whether it
    #: is a gap. (The renderer's channel is all-zero in this case; `visible_instance_ids` still
    #: lists the shell, which is what separates "nothing to see" from "the annotator produced
    #: nothing at all".)
    contains_nothing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "instance_id_path": self.instance_id_path,
            "resolution": list(self.resolution),
            "depth_convention": self.depth_convention,
            "depth_valid_fraction": round(self.depth_valid_fraction, 6),
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "color_std": round(self.color_std, 6),
            "visible_instance_ids": sorted(self.visible_instance_ids),
            "instance_id_map": dict(sorted(self.instance_id_map.items())),
            "target_instance_ids": list(self.target_instance_ids),
            "target_visible": self.target_visible,
            "nonblank": self.nonblank,
            "contains_nothing": self.contains_nothing,
        }


@dataclass(frozen=True, slots=True)
class WorkerReport:
    """The worker's whole output.

    ``depth_convention`` is not decoration. The v2 renderer produced `distance_to_camera` (radial
    Euclidean) and this worker produces camera-z, and a downstream check that compares a depth
    reading against a height without knowing which one it holds is comparing two different
    quantities. So the convention is recorded per capture and asserted by the tests.

    ``observed_label_payload_kind`` records the shape the instance-segmentation annotator actually
    returned. That payload's structure is the least certain part of the Isaac-side integration, so
    the first real run reports what it saw rather than the code assuming it.
    """

    scene_id: str
    scene_version: str
    status: str
    isaac_version: str = ""
    captures: tuple[CaptureReport, ...] = ()
    instance_id_map: Mapping[str, str] = field(default_factory=dict)
    # Where the mapping came from, so a report cannot present an assumed map as an observed one.
    instance_id_map_source: str = ""
    # Renderer ids two cameras mapped to *different* instances. Each camera's own id space is valid
    # and stays on its capture; what a disagreement rules out is a single global meaning for that
    # id, so it is dropped from `instance_id_map` and named here rather than resolved to whichever
    # camera happened to be read last.
    instance_id_conflicts: tuple[str, ...] = ()
    # The request's own index -> instance_id order, kept separate because it is *not* the
    # renderer's id space.
    instance_index_map: Mapping[str, str] = field(default_factory=dict)
    observed_label_payload_kind: str = ""
    # Which annotator produced the id channel, and what filter was requested for it. The families
    # are not interchangeable: measured on this build, `instance_segmentation` emits ids
    # 0=BACKGROUND and 1=UNLABELLED while `instance_id_segmentation` emits neither and starts at 1.
    # A silent swap would change the id space without changing the artefact layout, so it is named.
    annotator_family: str = ""
    semantic_filter: str = ""
    # How many of the requested openings were actually cut into wall geometry. A door in the
    # manifest and a sealed wall in the render is the exact discrepancy this records: the scene
    # claims a passage the image does not contain.
    #
    # **None means "this run did not measure it"**, which is what the failure path produces: a
    # report that never reached the render used to publish `0` and `{}` here, and the manifest then
    # derived `door_cut=false` from a default -- a delivery asserting something about a room it had
    # never looked at (SPEC-015 R5). Measured values are ints/mappings; unmeasured ones are null.
    openings_requested: int | None = None
    openings_cut: int | None = None
    shell_reference: Mapping[str, Any] | None = None
    lighting: Mapping[str, Any] | None = None
    # How many sky backdrops were authored -- one per window, not one per room. A bool said
    # only that there was at least one, which a room with three windows and one backdrop would
    # have passed while two of its windows looked onto nothing.
    exterior_backdrops: int = 0
    # Which asset textures the render actually used, so "it is textured" is checkable.
    asset_textures: Mapping[str, str] = field(default_factory=dict)
    settle: Mapping[str, Any] = field(default_factory=dict)
    # Where the dynamic bodies were at each stage of the run, for diagnosing motion that happens
    # somewhere other than the settle.
    position_probe: Mapping[str, Any] = field(default_factory=dict)
    capture_state: str = "authored"
    #: Per view: what it contained against what it declared, plus the named gaps
    #: (`VIEW_EMPTY(camera_id)`, `VIEW_TARGET_MISSING(camera_id,target)`). Kept on a *passing*
    #: report: the render did what it could and the run decides what a miss means for the delivery.
    view_coverage: tuple[dict[str, Any], ...] = ()
    view_gaps: tuple[str, ...] = ()
    error_type: str = ""
    error_detail: str = ""
    wall_seconds: float = 0.0
    schema_version: str = REPORT_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "scene_version": self.scene_version,
            "status": self.status,
            "isaac_version": self.isaac_version,
            "depth_convention": "camera_z_meters",
            "captures": [capture.to_dict() for capture in self.captures],
            "instance_id_map": dict(sorted(self.instance_id_map.items())),
            "instance_id_map_source": self.instance_id_map_source,
            "instance_id_conflicts": list(self.instance_id_conflicts),
            "instance_index_map": dict(sorted(self.instance_index_map.items())),
            "observed_label_payload_kind": self.observed_label_payload_kind,
            "openings_requested": self.openings_requested,
            "openings_cut": self.openings_cut,
            "shell_reference": None if self.shell_reference is None else dict(self.shell_reference),
            "lighting": None if self.lighting is None else dict(self.lighting),
            "exterior_backdrops": self.exterior_backdrops,
            "asset_textures": dict(sorted(self.asset_textures.items())),
            "annotator_family": self.annotator_family,
            "semantic_filter": self.semantic_filter,
            "settle": dict(self.settle),
            "position_probe": {k: v for k, v in self.position_probe.items()},
            "capture_state": self.capture_state,
            "view_coverage": [dict(entry) for entry in self.view_coverage],
            "view_gaps": list(self.view_gaps),
            "error_type": self.error_type,
            "error_detail": self.error_detail,
            "wall_seconds": round(self.wall_seconds, 4),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkerReport:
        return cls(
            scene_id=str(payload.get("scene_id", "")),
            scene_version=str(payload.get("scene_version", "")),
            status=str(payload.get("status", "failed")),
            isaac_version=str(payload.get("isaac_version", "")),
            captures=tuple(
                CaptureReport(
                    camera_id=str(item["camera_id"]),
                    rgb_path=str(item["rgb_path"]),
                    depth_path=item.get("depth_path"),
                    instance_id_path=item.get("instance_id_path"),
                    resolution=tuple(int(v) for v in item["resolution"]),  # type: ignore[arg-type]
                    depth_convention=str(item.get("depth_convention", "")),
                    depth_valid_fraction=float(item.get("depth_valid_fraction", 0.0)),
                    depth_min_m=item.get("depth_min_m"),
                    depth_max_m=item.get("depth_max_m"),
                    color_std=float(item.get("color_std", 0.0)),
                    visible_instance_ids=tuple(int(v) for v in item.get("visible_instance_ids", ())),
                    instance_id_map={
                        str(k): str(v) for k, v in item.get("instance_id_map", {}).items()
                    },
                    target_instance_ids=tuple(str(v) for v in item.get("target_instance_ids", ())),
                    target_visible=item.get("target_visible"),
                    nonblank=bool(item.get("nonblank", False)),
                )
                for item in payload.get("captures", ())
            ),
            instance_id_map={str(k): str(v) for k, v in payload.get("instance_id_map", {}).items()},
            instance_id_map_source=str(payload.get("instance_id_map_source", "")),
            instance_id_conflicts=tuple(
                str(value) for value in payload.get("instance_id_conflicts", ())
            ),
            instance_index_map={str(k): str(v) for k, v in payload.get("instance_index_map", {}).items()},
            observed_label_payload_kind=str(payload.get("observed_label_payload_kind", "")),
            openings_requested=(None if payload.get("openings_requested") is None
                                else int(payload["openings_requested"])),
            openings_cut=(None if payload.get("openings_cut") is None else int(payload["openings_cut"])),
            shell_reference=(None if payload.get("shell_reference") is None
                             else dict(payload["shell_reference"])),
            lighting=None if payload.get("lighting") is None else dict(payload["lighting"]),
            exterior_backdrops=int(payload.get("exterior_backdrops", 0)),
            asset_textures={str(k): str(v) for k, v in payload.get("asset_textures", {}).items()},
            annotator_family=str(payload.get("annotator_family", "")),
            semantic_filter=str(payload.get("semantic_filter", "")),
            settle=dict(payload.get("settle", {})),
            position_probe=dict(payload.get("position_probe", {})),
            capture_state=str(payload.get("capture_state", "authored")),
            view_coverage=tuple(dict(entry) for entry in payload.get("view_coverage", ())),
            view_gaps=tuple(str(gap) for gap in payload.get("view_gaps", ())),
            error_type=str(payload.get("error_type", "")),
            error_detail=str(payload.get("error_detail", "")),
            wall_seconds=float(payload.get("wall_seconds", 0.0)),
        )

    def assert_self_consistent(self) -> None:
        """Fail if the report claims a pass that its own numbers contradict.

        Run by the orchestrator on receipt. A worker that returns ``passed`` with a blank image or
        no visible target is the most damaging kind of bug here, because everything downstream
        trusts the report instead of looking at the pixels.
        """
        if not self.passed:
            return
        problems: list[str] = []
        if not self.captures:
            problems.append("status is passed but no capture was recorded")
        if not self.isaac_version:
            problems.append("status is passed but the Isaac version was not recorded")
        if (self.openings_requested is not None and self.openings_cut is not None
                and self.openings_requested != self.openings_cut):
            problems.append(
                f"{self.openings_requested - self.openings_cut} requested opening(s) were not authored or measured "
                "into the wall geometry, so the render shows a sealed room where the scene claims a "
                "passage"
            )
        if not self.annotator_family:
            problems.append(
                "status is passed but the annotator that produced the instance-id channel was not "
                "recorded; the families have different id spaces"
            )
        if not self.instance_id_map:
            problems.append(
                "status is passed but no instance-id mapping was obtained; without it a rendered "
                "id cannot be resolved to an entity, and a critic would have to guess which "
                "object is which"
            )
        elif self.instance_id_map_source != "annotator_idToLabels":
            problems.append(
                f"the instance-id mapping came from {self.instance_id_map_source!r} rather than "
                "the renderer's own labels, so it is an assumption rather than an observation"
            )
        for capture in self.captures:
            if not capture.nonblank:
                problems.append(f"{capture.camera_id}: blank image")
            if capture.depth_path and capture.depth_valid_fraction <= 0.0:
                problems.append(f"{capture.camera_id}: depth buffer is entirely invalid")
            if capture.depth_convention != "camera_z_meters":
                problems.append(
                    f"{capture.camera_id}: depth convention {capture.depth_convention!r} is not "
                    "the declared camera-z"
                )
            if capture.instance_id_path and not capture.instance_id_map:
                # An empty map is legitimate when there was nothing to label -- a view of a wall or
                # of the shell alone renders an all-zero channel, and `visible_instance_ids` still
                # records what the renderer saw. It is a defect only when *nothing at all* was
                # labelled, because then the channel cannot be interpreted at all: that is the
                # annotator having produced nothing, not a view that contained nothing.
                if not capture.visible_instance_ids:
                    problems.append(
                        f"{capture.camera_id}: an instance-id channel was saved with neither an id "
                        "mapping nor a single visible instance, so the channel cannot be read"
                    )
                elif not capture.contains_nothing:
                    problems.append(
                        f"{capture.camera_id}: the capture holds no entity and does not say so"
                    )
            if capture.target_instance_ids and capture.target_visible is False and not any(
                gap.startswith(f"VIEW_TARGET_MISSING({capture.camera_id}") for gap in self.view_gaps
            ):
                problems.append(
                    f"{capture.camera_id}: required targets "
                    f"{list(capture.target_instance_ids)} are not visible and the report does not "
                    "name it (SPEC-015 R4: an unmet declaration is a named gap for this view)"
                )
        # A view that missed its own declaration has to say so by name. The report may still pass:
        # the render did what it could, and what a miss means for the delivery is the run's call.
        for entry in self.view_coverage:
            if entry.get("visible_instances", 0) < entry.get("minimum_visible_instances", 0):
                if f"VIEW_EMPTY({entry.get('camera_id')})" not in self.view_gaps:
                    problems.append(
                        f"{entry.get('camera_id')}: the view contains "
                        f"{entry.get('visible_instances')} instance(s) against a declared minimum of "
                        f"{entry.get('minimum_visible_instances')} and the report does not name it"
                    )
        if self.settle.get("performed") and self.capture_state != "settled":
            problems.append(
                "physics settling was performed but captures are not declared to describe the "
                "settled state"
            )
        if problems:
            raise ProtocolError(
                "the worker reported a pass that its own captures contradict: "
                + "; ".join(problems)
            )
