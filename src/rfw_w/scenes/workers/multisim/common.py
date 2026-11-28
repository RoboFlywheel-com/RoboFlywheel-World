# SPDX-License-Identifier: Apache-2.0
"""Shared camera, scene and depth conventions for the SPEC-014 five-backend figure.

Every backend driver in this directory consumes the same ``common_scene.json`` and
``camera.json`` and emits the same products:

* ``rgb.png`` — the visible geometry, as that engine renders it.
* ``depth_visible.npy`` — float32 metres along the **camera forward axis**
  (``camera_z_meters``) of the *visible* geometry. This is the number the main
  figure's depth row shows.
* ``depth_visible_mask.npy`` — bool, the pixels that depth is defined on. A
  background / invalid pixel is never encoded as ``0`` and never enters a
  statistic.
* ``depth_collision.npy`` — the same quantity for the *collision* geometry, kept
  in a separate file under a separate name so the two can never be confused
  (SPEC-014 requirement 2).
* ``report.json`` — engine name and version, the scene version and camera digest
  it consumed, per-object loading evidence, and the conversion applied.

Conventions, fixed here so no driver may vary them:

* World is **Z-up, right-handed, metres**, matching the delivery's ``scene_ir``.
* Cameras follow the OpenGL / USD convention: looking down local ``-Z``, ``+Y``
  up. ``depth`` is ``-z_cam``, i.e. positive in front of the camera.
* Intrinsics come from the delivery's own camera record: ``focal_length_mm``
  against the declared ``horizontal_aperture_mm`` / ``vertical_aperture_mm``.
  Both apertures are used as declared — the two are not in the same ratio as the
  resolution on this pipeline, so deriving one from the aspect would silently
  change the field of view (ADR-011).

This module is deliberately import-light: it is imported by five separate
processes under different interpreters, so nothing here may import an engine.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The depth colour scale is fixed in metres and shared by every depth panel, so a
# colour means the same distance in all five columns. Chosen once for the office
# room: its diagonal is ~6.1 m.
DEPTH_CMAP = "turbo_r"
DEPTH_VMIN_M = 0.0
DEPTH_VMAX_M = 6.0

# Room shell meshes that are not part of the room's interior. Rendering the outer
# facade from an interior camera only occludes the room; it is hidden by the same
# rule in all five backends, and the rule is recorded in every report.
HIDDEN_SHELL_ROLES = ("exterior",)

# Where "white" starts, for the clipping measurement. 250 rather than 255 so a
# surface that is one quantisation step from saturation still counts as lost.
CLIPPING_LUMA = 250.0


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CameraSpec:
    """One camera, pinned by the delivery record. ``eye``/``target`` are world metres."""

    name: str
    eye: np.ndarray  # (3,)
    target: np.ndarray  # (3,)
    width: int
    height: int
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float

    @property
    def fx(self) -> float:
        return self.width * self.focal_length_mm / self.horizontal_aperture_mm

    @property
    def fy(self) -> float:
        return self.height * self.focal_length_mm / self.vertical_aperture_mm

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def hfov_rad(self) -> float:
        return 2.0 * math.atan(self.horizontal_aperture_mm / (2.0 * self.focal_length_mm))

    @property
    def vfov_rad(self) -> float:
        return 2.0 * math.atan(self.vertical_aperture_mm / (2.0 * self.focal_length_mm))

    @property
    def forward(self) -> np.ndarray:
        fwd = self.target - self.eye
        return fwd / np.linalg.norm(fwd)

    @property
    def rotation_world_to_cam(self) -> np.ndarray:
        """Rows ``[right, up, -forward]``: ``p_cam = R @ (p_world - eye)``."""
        fwd = self.forward
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        return np.stack([right, up, -fwd], axis=0)

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """World points (n,3) -> pixel (u,v) and ``camera_z`` (n,)."""
        cam = (np.asarray(points_world, dtype=np.float64) - self.eye) @ \
            self.rotation_world_to_cam.T
        z = -cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.fx * cam[:, 0] / z
            v = self.cy - self.fy * cam[:, 1] / z
        return np.stack([u, v], axis=-1), z

    def as_record(self) -> dict:
        return {
            "name": self.name,
            "eye": self.eye.tolist(),
            "target": self.target.tolist(),
            "width": self.width,
            "height": self.height,
            "focal_length_mm": self.focal_length_mm,
            "horizontal_aperture_mm": self.horizontal_aperture_mm,
            "vertical_aperture_mm": self.vertical_aperture_mm,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "fx_over_fy": self.fx / self.fy,
            "depth_definition": "camera_z_meters",
            "up_axis": "Z",
        }

    @property
    def fovy_rad(self) -> float:
        """MuJoCo / PyBullet / Genesis / SAPIEN all take a single vertical field
        of view with square pixels; SPEC-014 pins the figure camera to exactly
        that so this number is the same quantity in all four."""
        return 2.0 * math.atan(self.height / (2.0 * self.fy))


def load_camera(path: str | Path) -> CameraSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CameraSpec(
        name=payload["name"],
        eye=np.asarray(payload["eye"], dtype=np.float64),
        target=np.asarray(payload["target"], dtype=np.float64),
        width=int(payload["width"]),
        height=int(payload["height"]),
        focal_length_mm=float(payload["focal_length_mm"]),
        horizontal_aperture_mm=float(payload["horizontal_aperture_mm"]),
        vertical_aperture_mm=float(payload["vertical_aperture_mm"]),
    )


@dataclass(frozen=True)
class SceneObject:
    """One loadable piece of the scene: an instance or a room shell mesh."""

    object_id: str
    kind: str  # "furniture" | "shell"
    role: str  # entity_id for furniture, shell role for room meshes
    visual_mesh: Path
    collision_mesh: Path
    translation: np.ndarray  # (3,) world metres
    rotation_wxyz: np.ndarray  # (4,) world quaternion, w first
    scale: float

    def matrix_world(self) -> np.ndarray:
        """4x4 world matrix. Rotation is applied before translation."""
        w, x, y, z = self.rotation_wxyz
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        out = np.eye(4)
        out[:3, :3] = rot * self.scale
        out[:3, 3] = self.translation
        return out


class CommonScene:
    """The frozen scene: the same meshes and poses for all five backends."""

    def __init__(self, root: Path, payload: dict):
        self.root = Path(root)
        self.payload = payload

    @classmethod
    def load(cls, root: str | Path) -> "CommonScene":
        root = Path(root)
        payload = json.loads((root / "common_scene.json").read_text(encoding="utf-8"))
        return cls(root, payload)

    @property
    def scene_version(self) -> str:
        return self.payload["scene_version"]

    @property
    def camera(self) -> CameraSpec:
        return CameraSpec(
            name=self.payload["camera"]["name"],
            eye=np.asarray(self.payload["camera"]["eye"], dtype=np.float64),
            target=np.asarray(self.payload["camera"]["target"], dtype=np.float64),
            width=int(self.payload["camera"]["width"]),
            height=int(self.payload["camera"]["height"]),
            focal_length_mm=float(self.payload["camera"]["focal_length_mm"]),
            horizontal_aperture_mm=float(self.payload["camera"]["horizontal_aperture_mm"]),
            vertical_aperture_mm=float(self.payload["camera"]["vertical_aperture_mm"]),
        )

    def objects(self, include_hidden: bool = False) -> list[SceneObject]:
        out: list[SceneObject] = []
        for rec in self.payload["objects"]:
            if not include_hidden and rec.get("hidden"):
                continue
            out.append(
                SceneObject(
                    object_id=rec["object_id"],
                    kind=rec["kind"],
                    role=rec["role"],
                    visual_mesh=self.root / rec["visual_mesh"],
                    collision_mesh=self.root / rec["collision_mesh"],
                    translation=np.asarray(rec["translation"], dtype=np.float64),
                    rotation_wxyz=np.asarray(rec["rotation_wxyz"], dtype=np.float64),
                    scale=float(rec.get("scale", 1.0)),
                )
            )
        return out

    @property
    def instance_ids(self) -> list[str]:
        return [o.object_id for o in self.objects() if o.kind == "furniture"]

    def hidden_rule(self) -> dict:
        return dict(self.payload.get("hidden_rule", {}))


def camera_from_isaac_record(record: dict, name: str | None = None) -> CameraSpec:
    """Build a CameraSpec from the delivery's own camera record (already the
    authority for eye/target/focal/aperture)."""
    return CameraSpec(
        name=name or record["camera_id"],
        eye=np.asarray(record["eye"], dtype=np.float64),
        target=np.asarray(record["target"], dtype=np.float64),
        width=int(record["resolution"][0]),
        height=int(record["resolution"][1]),
        focal_length_mm=float(record["focal_length_mm"]),
        horizontal_aperture_mm=float(record["horizontal_aperture_mm"]),
        vertical_aperture_mm=float(record["vertical_aperture_mm"]),
    )


def depth_to_world_normal_rgb(
    depth: np.ndarray, camera: CameraSpec, far_threshold_m: float = DEPTH_VMAX_M
) -> np.ndarray:
    """Linear camera-z depth (metres) -> world-space normal map, ``(n+1)/2``.

    Invalid and background pixels come out neutral gray so the companion figure
    shows coverage, not a fabricated surface.
    """
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0) & (depth < far_threshold_m)

    us = (np.arange(w) + 0.5 - camera.cx) / camera.fx
    vs = (np.arange(h) + 0.5 - camera.cy) / camera.fy
    uu, vv = np.meshgrid(us, vs)
    pts = np.stack([uu * depth, -vv * depth, -depth], axis=-1)

    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1, :] = pts[2:, :] - pts[:-2, :]
    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.where(norm > 0, norm, 1.0)
    normal[np.einsum("hwc,hwc->hw", normal, pts) > 0] *= -1.0

    interior = np.zeros_like(valid)
    interior[1:-1, 1:-1] = (
        valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    )
    valid = valid & interior

    normal_world = normal @ camera.rotation_world_to_cam
    rgb = ((normal_world + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    rgb[~valid] = (128, 128, 128)
    return rgb


def colourise_depth(depth: np.ndarray) -> np.ndarray:
    """Depth -> RGB uint8 under the one fixed metre scale every panel shares."""
    import matplotlib

    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    normed = np.clip((depth - DEPTH_VMIN_M) / (DEPTH_VMAX_M - DEPTH_VMIN_M), 0.0, 1.0)
    rgba = matplotlib.colormaps[DEPTH_CMAP](normed)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = (32, 32, 32)
    return rgb


def save_depth_products(
    out_dir: Path,
    depth: np.ndarray,
    prefix: str = "depth_visible",
) -> dict:
    """Write one depth array plus its mask. The raw float32 array is never
    filtered, colourised in place, or overwritten by a later step."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.isfinite(depth) & (depth > 0)
    np.save(out_dir / f"{prefix}.npy", depth)
    np.save(out_dir / f"{prefix}_mask.npy", mask)
    valid = depth[mask]
    return {
        "array": f"{prefix}.npy",
        "mask": f"{prefix}_mask.npy",
        "shape": list(depth.shape),
        "dtype": "float32",
        "units": "m",
        "definition": "camera_z_meters",
        "valid_pixel_fraction": float(mask.mean()) if mask.size else 0.0,
        "valid_min_m": float(valid.min()) if valid.size else None,
        "valid_max_m": float(valid.max()) if valid.size else None,
    }


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def look_at_quaternion_wxyz(camera: CameraSpec) -> np.ndarray:
    """Camera world orientation as ``wxyz``, in the GL convention (-Z forward).

    Every backend needs this in some form; keeping one implementation here means
    an error in it shows up in all five columns at once rather than in one.
    """
    r = camera.rotation_world_to_cam  # world -> cam
    cam_to_world = r.T
    m = np.eye(3)
    m[:, 0] = cam_to_world[:, 0]
    m[:, 1] = cam_to_world[:, 1]
    m[:, 2] = cam_to_world[:, 2]
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def mtl_definitions(source: Path) -> dict[str, dict]:
    """Read the diffuse MTL fields a renderer can reproduce for the RGB pass.

    Lived in ``render_mujoco.py`` until PyBullet needed the same split (measured:
    PyBullet rendered ``usemtl``-multi meshes as flat white because it only ever
    built a collision shape, and one shape carries one material).
    """
    libraries: list[Path] = []
    with Path(source).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("mtllib "):
                libraries.append(Path(source).parent / line.split(maxsplit=1)[1].strip())
    definitions: dict[str, dict] = {}
    for library in libraries:
        if not library.is_file():
            raise ValueError(f"MULTISIM_PROVENANCE_MISSING({library})")
        current = None
        for line in library.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            if parts[0] == "newmtl":
                current = parts[1]
                definitions[current] = {"diffuse_rgb": (1.0, 1.0, 1.0), "texture": None}
            elif current is not None and parts[0] == "Kd":
                definitions[current]["diffuse_rgb"] = tuple(float(v) for v in parts[1].split()[:3])
            elif current is not None and parts[0] == "map_Kd":
                texture = library.parent / parts[1].strip()
                if not texture.is_file():
                    raise ValueError(f"MULTISIM_PROVENANCE_MISSING({texture})")
                definitions[current]["texture"] = texture
    return definitions


def split_visual_materials(source: Path, prefix: Path,
                           default_diffuse: tuple[float, float, float] = (1.0, 1.0, 1.0)
                           ) -> list[dict]:
    """Write one OBJ per material without changing any face or UV index.

    ``default_diffuse`` is what a face with no material at all is given. It
    matters more than it looks: the room shell meshes carry no ``mtllib`` and
    no ``usemtl`` whatsoever, so every engine was falling through to its own
    white default, and a white wall saturates before a brown desk reaches
    mid-grey. That is what made a shared exposure unreachable.
    """
    source = Path(source)
    definitions = mtl_definitions(source)
    vertices: list[str] = []
    faces: dict[str | None, list[str]] = {}
    active: str | None = None
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        tag = line.split(maxsplit=1)[0] if line.split() else ""
        if tag in ("v", "vt", "vn"):
            vertices.append(line)
        elif tag == "usemtl":
            active = line.split(maxsplit=1)[1].strip()
        elif tag == "f":
            faces.setdefault(active, []).append(line)
    if not faces:
        raise ValueError(f"MULTISIM_GEOMETRY_MISMATCH({source}: no faces)")
    records = []
    for index, (name, material_faces) in enumerate(faces.items()):
        if name is not None and name not in definitions:
            raise ValueError(f"MULTISIM_PROVENANCE_MISSING({source}: material {name})")
        target = Path(prefix).with_name(f"{Path(prefix).name}_{index}.obj")
        target.write_text("".join(vertices + material_faces), encoding="utf-8")
        records.append({"name": name, "mesh": target, "faces": len(material_faces),
                        **definitions.get(name, {"diffuse_rgb": tuple(default_diffuse),
                                                 "texture": None})})
    return records


def resolves_to_no_material(mesh: str | Path) -> bool:
    """True when an OBJ declares no material library at all.

    This is the exact population the rig's ``shell_albedo`` applies to, and it
    is decided by reading the file rather than by guessing from an object's
    role: the room shell meshes (walls, floor, ceiling, window and door
    reveals) carry no ``mtllib`` and no ``usemtl``, so every backend was
    falling through to its own near-white default. Every backend asks this same
    question of the same bytes, which is what keeps the answer uniform.
    """
    with Path(mesh).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("mtllib ") or line.startswith("usemtl "):
                return False
    return True


@dataclass(frozen=True)
class LightRig:
    """The one light rig all five backends are driven from (SPEC-014 R5).

    Before this existed only MuJoCo and SAPIEN declared any lighting; Genesis,
    PyBullet and Isaac each took an engine or stage default, which is why the
    five columns did not read as one set. Engines differ in what they can
    honour -- PyBullet's TinyRenderer takes a single directional light and no
    point light at all -- so each backend records what it *applied* and what it
    could not, as a named gap rather than as silence.
    """

    name: str
    ambient: tuple[float, float, float]
    direction: tuple[float, float, float]
    directional_colour: tuple[float, float, float]
    point_lights: tuple[dict, ...]
    gain: dict  # backend name -> scalar multiplier on that engine's light intensity
    target_mean_luminance: float
    luminance_tolerance: float
    calibration_patch: dict  # world-space rectangle the exposure is measured on
    noise_max: float
    clipping_max: float
    shell_albedo: tuple[float, float, float]

    def gain_for(self, backend: str) -> float:
        """Per-engine intensity multiplier.

        Exposure is unified **inside the engine** -- the published PNG stays a
        byte-for-byte copy of what the renderer produced (SPEC-014 R3). A gain
        here is a light-intensity change followed by a re-render, never a
        post-multiply on the image.
        """
        return float(self.gain.get(backend, 1.0))


def load_light_rig(path: str | Path) -> LightRig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    exposure = payload["exposure"]
    return LightRig(
        name=payload["name"],
        ambient=tuple(float(v) for v in payload["ambient"]),
        direction=tuple(float(v) for v in payload["directional"]["direction"]),
        directional_colour=tuple(float(v) for v in payload["directional"]["colour"]),
        point_lights=tuple(dict(p) for p in payload.get("point_lights", ())),
        gain=dict(payload.get("gain", {})),
        target_mean_luminance=float(exposure["target_mean_luminance"]),
        luminance_tolerance=float(exposure["tolerance"]),
        calibration_patch=dict(exposure["calibration_patch"]),
        noise_max=float(payload["noise"]["max_high_frequency_residual"]),
        clipping_max=float(payload["clipping"]["max_clipped_fraction"]),
        shell_albedo=tuple(float(v) for v in payload["shell_albedo"]["rgb"]),
    )


def calibration_pixels(rig: LightRig, camera: CameraSpec) -> tuple[slice, slice]:
    """Project the rig's world-space calibration rectangle into pixel bounds.

    Declared in **world** metres (a patch of bare desk top), not in pixels, so
    that moving the camera does not silently move the surface the exposure is
    measured on.
    """
    patch = rig.calibration_patch
    x0, x1 = patch["x_m"]
    y0, y1 = patch["y_m"]
    z = float(patch["z_m"])
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1)], dtype=np.float64)
    uv, depth = camera.project(corners)
    if not np.isfinite(uv).all() or (depth <= 0).any():
        raise ValueError(f"MULTISIM_EXPOSURE_PATCH_OFF_CAMERA({rig.name})")
    u0, v0 = np.ceil(uv[:, 0].min()).astype(int), np.ceil(uv[:, 1].min()).astype(int)
    u1, v1 = np.floor(uv[:, 0].max()).astype(int), np.floor(uv[:, 1].max()).astype(int)
    u0, u1 = max(0, u0), min(camera.width, u1)
    v0, v1 = max(0, v0), min(camera.height, v1)
    if u1 - u0 < 8 or v1 - v0 < 8:
        raise ValueError(
            f"MULTISIM_EXPOSURE_PATCH_OFF_CAMERA({rig.name}: {u1 - u0}x{v1 - v0} px in frame)")
    return slice(v0, v1), slice(u0, u1)


def measure_exposure(rgb: np.ndarray, rig: LightRig, camera: CameraSpec) -> dict:
    """Mean luminance on the calibration patch, and whether it is in band.

    Rec. 709 luma on the 0-255 render. The gate can fail: a backend whose lights
    are left at an engine default lands far outside the band, which is exactly
    the defect this measurement exists to catch.
    """
    rows, cols = calibration_pixels(rig, camera)
    patch = np.asarray(rgb, dtype=np.float64)[rows, cols, :3]
    luma = 0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]
    measured = float(luma.mean())
    delta = measured - rig.target_mean_luminance
    return {
        "measured_mean_luminance": measured,
        "target_mean_luminance": rig.target_mean_luminance,
        "tolerance": rig.luminance_tolerance,
        "delta": delta,
        "in_band": bool(abs(delta) <= rig.luminance_tolerance),
        "patch_px": [int(cols.start), int(rows.start), int(cols.stop), int(rows.stop)],
        "patch_world": dict(rig.calibration_patch),
        "definition": "Rec.709 luma mean over the projected world calibration patch",
    }


def measure_noise(rgb: np.ndarray, rig: LightRig, camera: CameraSpec) -> dict:
    """High-frequency residual on the same flat patch.

    A path tracer at too few samples per pixel leaves salt-and-pepper that no
    geometry or camera gate notices. The calibration patch is bare desk top, so
    on a converged render its pixel-to-pixel difference is near zero and any
    residual is sampling noise. Measured on the 8 spp / no-denoiser render that
    prompted this gate, and on a converged one, so the threshold has both sides.
    """
    rows, cols = calibration_pixels(rig, camera)
    patch = np.asarray(rgb, dtype=np.float64)[rows, cols, :3]
    luma = 0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]
    # Laplacian: flat shading and a smooth gradient both pass, per-pixel scatter does not.
    residual = (4.0 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1]
                - luma[1:-1, :-2] - luma[1:-1, 2:])
    measured = float(np.abs(residual).mean())
    return {
        "high_frequency_residual": measured,
        "max_allowed": rig.noise_max,
        "within_limit": bool(measured <= rig.noise_max),
        "patch_px": [int(cols.start), int(rows.start), int(cols.stop), int(rows.stop)],
        "definition": "mean |4-neighbour Laplacian| of Rec.709 luma on the calibration patch",
    }


def measure_clipping(rgb: np.ndarray, rig: LightRig) -> dict:
    """Fraction of the WHOLE frame driven to white.

    The exposure gate above looks at one mid-tone patch, and on its own that is
    not enough -- measured 2026-09-25, driving MuJoCo's desk patch to the 140
    target put 62.7% of its frame at >=250 luma, and PyBullet's 66.7%, while the
    patch itself read perfectly in band. The room's walls are untextured white
    and saturate long before a brown desk reaches mid-grey, and how early they
    saturate is a property of each engine's shading, not of the rig. A one-point
    exposure gate cannot see that, so it has to be measured separately.
    """
    image = np.asarray(rgb, dtype=np.float64)
    luma = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    clipped = float((luma >= CLIPPING_LUMA).mean())
    return {
        "clipped_fraction": clipped,
        "max_allowed": rig.clipping_max,
        "within_limit": bool(clipped <= rig.clipping_max),
        "threshold_luma": CLIPPING_LUMA,
        "frame_mean": float(image.mean()),
        "definition": "fraction of frame pixels with Rec.709 luma >= 250",
    }


def appearance_gates(rgb: np.ndarray, rig: LightRig, camera: CameraSpec,
                     backend: str) -> tuple[dict, list[str]]:
    """Exposure, noise and clipping, and the named gaps for whichever failed."""
    exposure = measure_exposure(rgb, rig, camera)
    noise = measure_noise(rgb, rig, camera)
    clipping = measure_clipping(rgb, rig)
    gaps: list[str] = []
    if not exposure["in_band"]:
        gaps.append(
            f"MULTISIM_EXPOSURE_OUTSIDE_BAND({backend},"
            f"{exposure['measured_mean_luminance']:.2f},"
            f"{exposure['target_mean_luminance']:.2f})")
    if not noise["within_limit"]:
        gaps.append(f"MULTISIM_RGB_NOISY({backend},{noise['high_frequency_residual']:.3f})")
    if not clipping["within_limit"]:
        gaps.append(f"MULTISIM_RGB_CLIPPED({backend},{clipping['clipped_fraction']:.4f})")
    return {"lighting_rig": rig.name, "exposure": exposure, "noise": noise,
            "clipping": clipping}, gaps


def backend_report(
    backend: str,
    engine_version: str,
    scene: CommonScene,
    camera: CameraSpec,
    extra: dict | None = None,
) -> dict:
    """The report every backend must emit. A missing field is a visible hole in
    the ledger rather than a silent zero."""
    report = {
        "backend": backend,
        "engine_version": engine_version,
        "scene_version": scene.scene_version,
        "object_ids": scene.instance_ids,
        "object_count": len(scene.instance_ids),
        "hidden_rule": scene.hidden_rule(),
        "camera": camera.as_record(),
        "depth_products": {
            "visible": "depth_visible.npy",
            "visible_mask": "depth_visible_mask.npy",
            "collision": "depth_collision.npy",
            "collision_mask": "depth_collision_mask.npy",
        },
        "geometry_source": {
            "rgb_and_visible_depth": "visual_mesh",
            "collision_depth": "collision_mesh",
        },
        "status": "passed",
    }
    report.update(extra or {})
    return report
