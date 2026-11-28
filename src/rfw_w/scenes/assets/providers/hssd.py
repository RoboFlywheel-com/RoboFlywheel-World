# SPDX-License-Identifier: Apache-2.0
"""The HSSD furniture provider: raw GLB turned into something placeable.

HSSD ships meshes with a synset index and a set of precomputed support surfaces, and no mass, no
collision geometry and no USD. The architecture is explicit that this preparation is an offline step
which must be *visible* rather than hidden inside a generation run, so it is a module of its own
with its own record and its own evidence.

Two things were measured rather than assumed:

* **the up axis.** A desk's bounding box is ambiguous -- 1.295 x 0.762 x 0.621 could be a low desk or
  a deep one. The support surface settles it: its normal lies along -Y and its OBB is
  1.213 x 2e-06 x 0.606, a flat rectangle perpendicular to Y, sitting at Y = 0.593 inside a Y range
  of [0, 0.762]. So Y is vertical, the floor is at Y = 0 and the desktop at Y = 0.593.
* **the support polygon.** Taken from the data's own support surface, not from the mesh bounding
  box. A bounding box for an L-shaped or pedestal desk would claim support where there is none, and
  an object placed there would float.

Mass is estimated, and recorded as estimated: HSSD carries none.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from rfw_w.scenes.contracts.asset import (
    AffordanceStatus,
    AssetKey,
    Provenance,
    SupportPolygon,
)
from rfw_w.scenes.contracts.sizing import (
    DEFAULT_RATIO_LIMIT,
    MIN_SCREENED_AXIS_M,
    midpoint_of,
    ratio_violation,
)
from rfw_w.scenes.contracts.units import Vec3
from rfw_w.scenes.assets.prepared_obj import (
    UP_FIX_ROWS,
    sanitise as _sanitise,
    write_scene_obj as _write_scene_obj,
)

SCHEMA_VERSION = "pvscene.hssd-prepared/v1"

# The precision a prepared asset's measurements are stored and compared at. Six decimals is a
# micrometre on a metre-scale piece, so it is far below anything placement or a gate can observe --
# what matters is that a value read back from the sidecar equals the value that was written, since
# the two feed the same placement and the same scene version.
STORED_DECIMALS = 6
PROVIDER = "hssd"
LIBRARY_VERSION = "hssd-models"

# Estimated, because HSSD has no mass. A wood-composite bulk density; the number is labelled
# estimated everywhere it appears, and M1 uses HSSD only for fixed furniture where mass does not
# enter a stability gate.
DEFAULT_DENSITY_KG_M3 = 400.0


def _written_texture_sizes(obj_path: Path) -> dict[str, tuple[int, int]]:
    """The size of every texture the OBJ export wrote beside it, keyed by file name.

    Read from disk rather than from the material object: the question is what the prepared asset
    carries, and a synthesised placeholder is the same size as a real map to everything except a
    pixel count.

    Every map, not the first: an asset's materials each carry their own, and an asset whose first
    material has no image is not an asset without textures -- the office notebook keeps its spiral
    and its cover in three separate materials, two of which state a colour and none an image.
    """
    sizes: dict[str, tuple[int, int]] = {}
    for candidate in sorted(obj_path.parent.glob("*.png")) + sorted(obj_path.parent.glob("*.jpg")):
        try:
            from PIL import Image

            with Image.open(candidate) as image:
                sizes[candidate.name] = image.size
        except Exception:  # noqa: BLE001 - an unreadable image counts as no image
            continue
    return sizes


class HssdError(RuntimeError):
    """The furniture could not be prepared."""


@dataclass(frozen=True, slots=True)
class HssdConfig:
    """Where the library is, and where prepared output goes."""

    models_root: Path
    synset_index_path: Path
    prepared_root: Path

    def __post_init__(self) -> None:
        if not self.models_root.is_dir():
            raise HssdError(f"HSSD models root not found: {self.models_root}")
        if not self.synset_index_path.is_file():
            raise HssdError(f"HSSD synset index not found: {self.synset_index_path}")

    def glb_path(self, asset_id: str) -> Path:
        # Sharded by the first hex character, as the dataset is laid out.
        return self.models_root / "objects" / asset_id[0] / f"{asset_id}.glb"

    def support_surface_path(self, asset_id: str) -> Path:
        return self.models_root / "support-surfaces" / asset_id / f"{asset_id}.supportSurface.json.gz"


def _file_identity(path: Path) -> tuple[str, int, int] | None:
    """A cache key which also invalidates when an asset-side file changes."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=8)
def _load_synset_index_cached(
    path_text: str, _mtime_ns: int, _size: int,
) -> dict[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(Path(path_text).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise HssdError(f"unreadable HSSD synset index: {error}") from error
    if not isinstance(payload, dict):
        raise HssdError("HSSD synset index must be a mapping of synset to assets")
    return payload


@lru_cache(maxsize=4_096)
def _candidates_for_synset_cached(
    index_path: str,
    index_mtime_ns: int,
    index_size: int,
    models_root_text: str,
    synset: str,
) -> tuple[str, ...]:
    index = _load_synset_index_cached(index_path, index_mtime_ns, index_size)
    entries = index.get(synset)
    if not entries:
        return ()
    models_root = Path(models_root_text)
    found = [
        str(entry["id"])
        for entry in entries
        if (
            isinstance(entry, dict)
            and entry.get("id")
            and (
                models_root / "objects" / str(entry["id"])[0] / f"{entry['id']}.glb"
            ).is_file()
        )
    ]
    # Sorted so that selecting the k-th candidate is reproducible across runs.
    return tuple(sorted(found))


def load_synset_index(config: HssdConfig) -> dict[str, list[dict[str, Any]]]:
    """Cached catalogue metadata; callers must not mutate the returned mapping."""
    identity = _file_identity(config.synset_index_path)
    if identity is None:
        raise HssdError("HSSD synset index is unavailable")
    return _load_synset_index_cached(*identity)


def candidates_for_synset(config: HssdConfig, synset: str) -> tuple[str, ...]:
    """Existing asset ids, scanned once per synset and unchanged catalogue in this process."""
    identity = _file_identity(config.synset_index_path)
    if identity is None:
        raise HssdError(f"unreadable HSSD synset index: {config.synset_index_path}")
    return _candidates_for_synset_cached(
        *identity, str(config.models_root.resolve()), synset,
    )


@dataclass(frozen=True, slots=True)
class SupportSurfaceReading:
    """A support surface as the data describes it, in the model's Y-up frame."""

    centroid: tuple[float, float, float]
    extent_x: float
    extent_z: float
    normal_axis: str
    area_m2: float

    @property
    def plane_height_y(self) -> float:
        return self.centroid[1]


@lru_cache(maxsize=65_536)
def _read_support_surfaces_cached(
    path_text: str, _mtime_ns: int, _size: int,
) -> tuple[SupportSurfaceReading, ...]:
    """Read the precomputed support surfaces.

    The largest surface by area is the usable top; the rest are shelves or lower tiers. All are
    kept, because a desk with a hutch genuinely has two, and discarding the smaller one would make
    a shelf unusable without saying so.
    """
    path = Path(path_text)
    try:
        payload = json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, json.JSONDecodeError) as error:
        raise HssdError(f"unreadable support surface {path.name}: {error}") from error

    readings: list[SupportSurfaceReading] = []
    for surface in payload.get("supportSurfaces", []):
        if not isinstance(surface, Mapping):
            continue
        obb = surface.get("obb") or {}
        centroid = obb.get("centroid")
        axes = obb.get("axesLengths")
        normal = surface.get("normal") or {}
        if not (isinstance(centroid, Sequence) and len(centroid) == 3 and isinstance(axes, Sequence) and len(axes) == 3):
            continue
        # The vertical axis is the one the OBB is flat along, which is also the one the normal
        # points down. Both are checked so a malformed entry is skipped rather than assumed flat.
        flat_index = min(range(3), key=lambda index: abs(float(axes[index])))
        normal_values = [abs(float(normal.get(axis, 0.0))) for axis in ("x", "y", "z")]
        normal_index = max(range(3), key=lambda index: normal_values[index])
        if flat_index != normal_index or normal_values[flat_index] < 0.5:
            continue
        horizontal = [index for index in range(3) if index != flat_index]
        readings.append(
            SupportSurfaceReading(
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                extent_x=float(axes[horizontal[0]]),
                extent_z=float(axes[horizontal[1]]),
                normal_axis="xyz"[flat_index],
                area_m2=float(surface.get("area", 0.0)),
            )
        )
    readings.sort(key=lambda reading: -reading.area_m2)
    return tuple(readings)


def read_support_surfaces(config: HssdConfig, asset_id: str) -> tuple[SupportSurfaceReading, ...]:
    """Read precomputed surfaces once per unchanged file in this process."""
    identity = _file_identity(config.support_surface_path(asset_id))
    return () if identity is None else _read_support_surfaces_cached(*identity)


def support_polygons_from_readings(
    readings: Sequence[SupportSurfaceReading],
    *,
    asset_id: str,
    source_uri: str,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[SupportPolygon, ...]:
    """Convert readings into Z-up support polygons, in the prepared mesh's frame.

    ``offset`` is the translation the mesh received during preparation. It has to be applied here
    too: the support surfaces are described in the *original* asset frame, and the prepared mesh has
    been moved to put its footprint centre on the origin with its base at zero. Leaving the offset
    out places the surface where the asset used to be rather than where it now is -- which is how a
    cup ends up beside the desk top instead of on it, falls to the floor, and is then reported as
    failing the settle gate for reasons that have nothing to do with physics.

    The rectangle comes from the OBB extents, which is the shape the data states. It is recorded as
    a rectangle rather than presented as a general polygon, so a consumer can see that the surface
    description is a box and not an arbitrary outline.
    """
    ox, oy, oz = offset
    polygons: list[SupportPolygon] = []
    for index, reading in enumerate(readings):
        half_x, half_y = reading.extent_x / 2.0, reading.extent_z / 2.0
        if half_x <= 0 or half_y <= 0:
            continue
        # Model is Y-up and the prepared frame is Z-up, reached by the rotation in UP_FIX_ROWS:
        # the surface's height y becomes the prepared z, and the reading's z becomes the prepared
        # *-y*. The sign is load-bearing -- it is what puts this polygon in the same frame as the
        # mesh, and a surface object placed on this support is positioned from both.
        plane_z = reading.plane_height_y + oz
        centre_x, centre_y = reading.centroid[0] + ox, -reading.centroid[2] + oy
        polygons.append(
            SupportPolygon(
                polygon_id=f"{asset_id}:surface{index}",
                vertices_xy=(
                    (centre_x - half_x, centre_y - half_y),
                    (centre_x + half_x, centre_y - half_y),
                    (centre_x + half_x, centre_y + half_y),
                    (centre_x - half_x, centre_y + half_y),
                ),
                plane_z_m=plane_z,
                provenance=Provenance(
                    method="hssd_precomputed_support_surface_obb",
                    version=SCHEMA_VERSION,
                    # Annotated by the dataset, not measured by us: honest about which it is.
                    status=AffordanceStatus.ANNOTATED,
                    source_uri=source_uri,
                ),
            )
        )
    return tuple(polygons)


@dataclass(frozen=True, slots=True)
class PreparedFurniture:
    """A prepared asset: loadable meshes plus everything measured about it."""

    asset_key: AssetKey
    visual_obj: Path
    collision_obj: Path
    dimensions_m: Vec3
    bounds_min_m: Vec3
    bounds_max_m: Vec3
    mass_kg: float
    support_polygons: tuple[SupportPolygon, ...]
    vertex_count: int
    face_count: int
    collision_face_count: int
    provenance: Provenance
    # Whether the asset carried a real diffuse map, or only a base-colour factor. Recorded because
    # the two are not interchangeable and a synthesised texture is not a substitute for either.
    has_diffuse_texture: bool = False
    # Which point of the mesh its own origin is. Recorded because a placement that assumes the
    # wrong one lifts the object by half its height or drops it by half -- a silent error that
    # leaves every image valid and every depth plausible while the object is somewhere else.
    origin_convention: str = "footprint_centre_base_at_zero"

    def __post_init__(self) -> None:
        """Round the measurements to the precision the sidecar stores them at.

        Not cosmetic. `to_dict` rounded on the way out, so a *cache hit* read back values that
        differed from a fresh preparation by ~1e-7 m -- and those values feed the placement
        footprint, the recorded bounds and therefore the scene version. Rebuilding a plan whose
        assets were prepared earlier then produced a different scene, which is how the loop's
        restore path came to land on a version no gate had measured.

        Rounding where the value is made, rather than where it is written, gives the object and its
        sidecar one representation: whatever a fresh preparation returns is what a cache hit
        returns, so the scene stays a function of the plan.
        """
        for name in ("dimensions_m", "bounds_min_m", "bounds_max_m"):
            vector = getattr(self, name)
            object.__setattr__(
                self, name, Vec3(*[round(value, STORED_DECIMALS) for value in vector.to_list()])
            )
        object.__setattr__(self, "mass_kg", round(self.mass_kg, STORED_DECIMALS))

    def to_dict(self) -> dict[str, Any]:
        # The numbers are written as they are: `__post_init__` has already put them at the stored
        # precision, and rounding again here would be the second copy of that rule that drifts.
        return {
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset_key.to_dict(),
            "visual_obj": self.visual_obj.name,
            "collision_obj": self.collision_obj.name,
            "dimensions_m": self.dimensions_m.to_list(),
            "bounds_min_m": self.bounds_min_m.to_list(),
            "bounds_max_m": self.bounds_max_m.to_list(),
            "mass_kg": self.mass_kg,
            "mass_provenance": self.provenance.to_dict(),
            "support_polygons": [polygon.to_dict() for polygon in self.support_polygons],
            "origin_convention": self.origin_convention,
            "has_diffuse_texture": self.has_diffuse_texture,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "collision_face_count": self.collision_face_count,
        }


def _provenance_from_dict(payload: Mapping[str, Any]) -> Provenance:
    return Provenance(
        method=str(payload["method"]), version=str(payload["version"]),
        status=AffordanceStatus(payload["status"]),
        confidence=None if payload.get("confidence") is None else float(payload["confidence"]),
        source_uri=payload.get("source_uri"),
    )


def _support_polygon_from_dict(payload: Mapping[str, Any]) -> SupportPolygon:
    return SupportPolygon(
        polygon_id=str(payload["polygon_id"]),
        vertices_xy=tuple((float(vertex[0]), float(vertex[1])) for vertex in payload["vertices_xy"]),
        plane_z_m=float(payload["plane_z_m"]),
        provenance=_provenance_from_dict(payload["provenance"]),
    )


def _prepared_from_dict(payload: Mapping[str, Any], *, directory: Path) -> PreparedFurniture:
    """Rebuild a prepared asset from its recorded metadata.

    The paths are stored as names and re-joined to ``directory`` here, so the sidecar stays portable
    and a moved preparation directory does not silently point at the old one.
    """
    return PreparedFurniture(
        asset_key=AssetKey(**payload["asset"]),
        visual_obj=directory / str(payload["visual_obj"]),
        collision_obj=directory / str(payload["collision_obj"]),
        dimensions_m=Vec3(*[float(value) for value in payload["dimensions_m"]]),
        bounds_min_m=Vec3(*[float(value) for value in payload["bounds_min_m"]]),
        bounds_max_m=Vec3(*[float(value) for value in payload["bounds_max_m"]]),
        mass_kg=float(payload["mass_kg"]),
        support_polygons=tuple(
            _support_polygon_from_dict(item) for item in payload["support_polygons"]
        ),
        vertex_count=int(payload["vertex_count"]),
        face_count=int(payload["face_count"]),
        collision_face_count=int(payload["collision_face_count"]),
        provenance=_provenance_from_dict(payload["mass_provenance"]),
        has_diffuse_texture=bool(payload["has_diffuse_texture"]),
        origin_convention=str(payload["origin_convention"]),
    )


METADATA_FILENAME = "prepared.json"


def prepared_metadata_path(config: HssdConfig, asset_id: str) -> Path:
    return config.prepared_root / asset_id / METADATA_FILENAME


def _cache_hit(config: HssdConfig, asset_id: str, glb: Path) -> PreparedFurniture | None:
    """The prepared asset, if it is already on disk and not older than the GLB it came from.

    ``prepare_furniture`` re-loaded the GLB and re-exported both meshes on every call, and a repair
    loop calls it once per entity per iteration: measured on the M1 scene, one build takes over two
    minutes, so a run that takes four rungs would spend eight minutes re-deriving meshes that had
    not changed. The mtime comparison is the load-bearing part -- a hit on a *stale* preparation
    would ship artefacts that do not match the asset, which is worse than the cost it saves.
    """
    metadata = prepared_metadata_path(config, asset_id)
    if not metadata.is_file():
        return None
    try:
        payload = json.loads(metadata.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    directory = config.prepared_root / asset_id
    prepared = _prepared_from_dict(payload, directory=directory)
    if not prepared.visual_obj.is_file() or not prepared.collision_obj.is_file():
        return None
    prepared_at = metadata.stat().st_mtime
    if glb.stat().st_mtime > prepared_at:
        return None
    return prepared


def prepare_furniture(
    config: HssdConfig,
    asset_id: str,
    *,
    density_kg_m3: float = DEFAULT_DENSITY_KG_M3,
    hull_target_faces: int = 256,
) -> PreparedFurniture:
    """Turn one GLB into Z-up OBJ meshes with a support surface and an estimated mass.

    The mesh origin is moved to the footprint centre with the base at z = 0, which is the convention
    the placement solver expects. Centring is recorded as part of the preparation, and the contract
    refuses a second normalisation, so the placement cannot silently re-centre it.
    """
    try:
        import trimesh  # imported here so the core stays stdlib-only
        import numpy as np
    except ImportError as error:
        raise HssdError(
            "preparing HSSD furniture needs trimesh and numpy; install the geometry extra"
        ) from error

    glb = config.glb_path(asset_id)
    if not glb.is_file():
        raise HssdError(f"no GLB for {asset_id} at {glb}")

    cached = _cache_hit(config, asset_id, glb)
    if cached is not None:
        return cached

    try:
        loaded = trimesh.load(str(glb))
    except Exception as error:  # noqa: BLE001 - trimesh raises several types on bad GLB
        raise HssdError(f"could not load {glb}: {type(error).__name__}: {error}") from error
    # A GLB with one geometry loads as a bare mesh, and a bare mesh has nowhere to keep a material.
    # Wrapping it makes the single- and multi-material cases one code path -- and multi-material is
    # the case that matters: `force="mesh"` (what this used to ask for) concatenates the GLB's
    # geometries and leaves the result with `visual.material is None`, so the export fell back to
    # trimesh's default `Kd 0.4 0.4 0.4` and a three-material notebook rendered flat grey.
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise HssdError(
            f"{glb} loaded as {type(loaded).__name__}, which is neither a mesh nor a scene"
        )
    if not scene.geometry or not any(
        len(geometry.vertices) for geometry in scene.geometry.values()
    ):
        raise HssdError(f"{glb} contains no geometry")

    scene.apply_transform(np.asarray(UP_FIX_ROWS, dtype=float))

    bounds_min = np.asarray(scene.bounds[0], dtype=float)
    bounds_max = np.asarray(scene.bounds[1], dtype=float)
    extents = bounds_max - bounds_min

    # Footprint centring with the base at zero. This is the one normalisation the asset receives.
    offset = np.array([
        -(bounds_min[0] + bounds_max[0]) / 2.0,
        -(bounds_min[1] + bounds_max[1]) / 2.0,
        -bounds_min[2],
    ])
    centring = np.eye(4)
    centring[:3, 3] = offset
    scene.apply_transform(centring)

    target = config.prepared_root / asset_id
    target.mkdir(parents=True, exist_ok=True)
    visual_obj = target / f"{asset_id}.obj"
    collision_obj = target / f"{asset_id}_collision.obj"
    # One `usemtl` group per material, each with its own map, written here rather than by trimesh's
    # exporter -- see `_write_scene_obj` for the indexing defect that made an asset's map
    # unreachable. A material that is only a base-colour factor (`image is None`, which is most HSSD
    # furniture) is written as that factor, in the linear space the GLB and the renderer both use.
    _write_scene_obj(scene, visual_obj)
    # Whether real maps came out is decided by reading what was written, not by asking the in-memory
    # material: trimesh's visual round-trips through the OBJ exporter in ways that made an earlier
    # check report "no image" for assets that do have one, and it wrote no material at all.
    written_textures = _written_texture_sizes(visual_obj)

    # One merged mesh for the measurements and the collision hull. A hull is a physics proxy and is
    # never textured, so the merged form is the right input for it, and taking it from the same
    # transformed scene keeps the two descriptions of one asset the same geometry.
    remeshed = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)

    try:
        hull = remeshed.convex_hull
        # Decimate so the collision mesh stays useful for physics; the hull of a 60k-vertex chair is
        # exact and slow, and a convex hull of a piece of furniture is already an approximation.
        if len(hull.faces) > hull_target_faces:
            hull = hull.simplify_quadric_decimation(face_count=hull_target_faces)
        hull.export(str(collision_obj))
        collision_face_count = int(len(hull.faces))
    except Exception as error:  # noqa: BLE001 - hulling can fail on degenerate geometry
        raise HssdError(f"could not build a collision hull for {asset_id}: {error}") from error

    volume_m3 = float(abs(remeshed.convex_hull.volume))
    mass_kg = max(volume_m3 * density_kg_m3, 0.1)

    readings = read_support_surfaces(config, asset_id)
    polygons = support_polygons_from_readings(
        readings, asset_id=asset_id, source_uri=str(config.support_surface_path(asset_id)),
        # The mesh was rotated Y-up -> Z-up and then translated by `offset`, so the reading's plane
        # height (its original Y) lands on the prepared Z and takes offset[2], while the reading's X
        # lands on the prepared X and takes offset[0] -- and its Z lands on the prepared *-Y*, which
        # is what `support_polygons_from_readings` applies. See UP_FIX_ROWS.
        offset=(float(offset[0]), float(offset[1]), float(offset[2])),
    )

    prepared = PreparedFurniture(
        asset_key=AssetKey(provider=PROVIDER, library_version=LIBRARY_VERSION, asset_id=asset_id),
        visual_obj=visual_obj,
        collision_obj=collision_obj,
        dimensions_m=Vec3(float(extents[0]), float(extents[1]), float(extents[2])),
        bounds_min_m=Vec3(float(bounds_min[0] + offset[0]), float(bounds_min[1] + offset[1]), 0.0),
        bounds_max_m=Vec3(
            float(bounds_max[0] + offset[0]),
            float(bounds_max[1] + offset[1]),
            float(extents[2]),
        ),
        mass_kg=mass_kg,
        support_polygons=polygons,
        vertex_count=int(len(remeshed.vertices)),
        face_count=int(len(remeshed.faces)),
        has_diffuse_texture=any(min(size) >= 64 for size in written_textures.values()),
        collision_face_count=collision_face_count,
        provenance=Provenance(
            method="convex_hull_volume_times_assumed_density",
            version=SCHEMA_VERSION,
            # Estimated, and labelled so: HSSD carries no mass, and a functional decision must not
            # be made on this silently.
            status=AffordanceStatus.ESTIMATED,
            source_uri=str(glb),
        ),
    )

    # Written last, so a partial preparation leaves no sidecar and the next call recomputes rather
    # than rebuilding a `PreparedFurniture` around meshes that were never written.
    prepared_metadata_path(config, asset_id).write_text(
        json.dumps(prepared.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return prepared


_GLB_MAGIC = b"glTF"
_GLB_JSON_CHUNK = 0x4E4F534A
# A ceiling on the JSON chunk a reader will accept. Real assets carry 50-500 KB of JSON; the cap is
# there so a corrupt or hostile length field cannot turn a size screen into a large allocation.
_MAX_JSON_CHUNK_BYTES = 8 * 1024 * 1024


@lru_cache(maxsize=65_536)
def _measure_glb_extents_cached(
    path_text: str, _mtime_ns: int, _size: int,
) -> Vec3 | None:
    """A GLB's bounding box, read from the POSITION accessors, in Z-up metres.

    Cheap on purpose: it reads the JSON chunk's accessor ``min``/``max`` and never touches the
    binary buffer, so screening a candidate costs a file read rather than a mesh import. That is
    what makes it affordable to screen *before* preparing, which is where it has to happen -- a
    rejection after `prepare_furniture` has already paid for the GLB load and two OBJ exports is a
    rejection that costs as much as an acceptance.

    Returns None rather than raising when the file is not a GLB, carries no accessor bounds, or has
    node transforms this reader cannot compose. The caller treats that as *unmeasurable*, not as a
    violation: rejecting an asset on a bound nobody measured is exactly the assumption this file's
    own docstring warns about. All 8 sampled HSSD assets have identity transforms, so the
    transform case is rare rather than hypothetical.

    Y-up to Z-up, matching `prepare_furniture`: a GLB says (x, y, z) and the room says (x, z, y).

    **Only the head of the file is read.** This used to `read_bytes()` the whole thing, and the
    library's meshes have a 2.3 MB median and an 18.7 MB maximum -- so screening the 24 candidates of
    nine entities pulled well over a gigabyte off a network filesystem to answer a question that
    lives in the first few hundred kilobytes. glTF requires the JSON chunk to be the *first* chunk,
    which is what makes a bounded read correct rather than a shortcut. Measured on a real run, the
    build spent 6.5 minutes of wall clock on 18 seconds of CPU before this.
    """
    path = Path(path_text)
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] != _GLB_MAGIC:
                return None
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                return None
            length, kind = struct.unpack("<II", chunk_header)
            if kind != _GLB_JSON_CHUNK or length > _MAX_JSON_CHUNK_BYTES:
                # Not a JSON-first GLB, or a length no real file carries -- a corrupt header must
                # not turn into a multi-gigabyte allocation.
                return None
            raw = handle.read(length)
    except OSError:
        return None
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    accessors = document.get("accessors") or []
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    for mesh in document.get("meshes") or []:
        for primitive in mesh.get("primitives") or []:
            index = (primitive.get("attributes") or {}).get("POSITION")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                continue
            accessor = accessors[index]
            minimum, maximum = accessor.get("min"), accessor.get("max")
            if not (isinstance(minimum, Sequence) and isinstance(maximum, Sequence)):
                continue
            if len(minimum) != 3 or len(maximum) != 3:
                continue
            for axis in range(3):
                low[axis] = min(low[axis], float(minimum[axis]))
                high[axis] = max(high[axis], float(maximum[axis]))
    if low[0] > high[0]:
        return None
    return Vec3(round(high[0] - low[0], 4), round(high[2] - low[2], 4), round(high[1] - low[1], 4))


def measure_glb_extents(path: Path) -> Vec3 | None:
    """Measure an unchanged GLB header once per process; missing files remain unmeasurable."""
    identity = _file_identity(path)
    return None if identity is None else _measure_glb_extents_cached(*identity)


def clear_hssd_metadata_caches() -> None:
    """Clear process-local read caches, primarily for tests and long-lived workers."""
    _load_synset_index_cached.cache_clear()
    _candidates_for_synset_cached.cache_clear()
    _read_support_surfaces_cached.cache_clear()
    _measure_glb_extents_cached.cache_clear()


# The placement kinds whose frame differs from the asset's own. A picture's prepared mesh states its
# thickness on z and its height on y, because it is authored lying down; the placement stands it up
# with a mount rotation, so the installed height is the asset's y. Comparing in the wrong frame is
# not a small error: `wall_art` scores 0 in-bound out of 24 candidates in its own frame and 8 of 24
# once permuted, so getting this wrong deletes the picture from every spec.
_INSTALLED_FRAME_PERMUTATION = {"wall_hung": (0, 2, 1)}


def size_bound_violation(
    extents: Vec3,
    bounds: Sequence[Any] | None,
    *,
    kind: str = "",
    ratio_limit: float = DEFAULT_RATIO_LIMIT,
) -> str | None:
    """Whether a measured asset is outside the size its entity declared, and by which axis.

    ``bounds`` is the entity's own ``size_bounds_m``; None means the spec declared no bound, and an
    undeclared bound is not a violation -- there is nothing to be outside of.

    The comparison is against the midpoint of each declared range, through the shared rule in
    `contracts.sizing`. This is the check the furniture path never had while the SimReady path
    always did, and its absence is why a spec asking for a 0.12-0.35 m ceiling light was handed a
    1.67 m pendant.
    """
    if bounds is None:
        return None
    measured = extents.to_list()
    permutation = _INSTALLED_FRAME_PERMUTATION.get(kind)
    if permutation is not None:
        measured = [measured[index] for index in permutation]
    return ratio_violation(measured, midpoint_of(bounds), ratio_limit,
                           axis_names="xyz" if permutation is None else "xzy",
                           screen_below_m=MIN_SCREENED_AXIS_M)


@dataclass(frozen=True, slots=True)
class SizeScreen:
    """The result of screening a candidate list against one entity's size bounds.

    Three lists rather than one, because the three cases call for different behaviour and
    collapsing them loses the reason. ``admitted`` may be used. ``relaxed`` are out of bounds but
    measurable, worst-first, and are the fallback when nothing is admitted -- shipping the least-bad
    and *saying so* beats failing the run, and beats silently taking the first (which is what the
    pipeline did, and is the bug). ``unmeasurable`` are kept as candidates too: a bound nobody
    measured is not evidence against an asset.
    """

    admitted: tuple[str, ...]
    relaxed: tuple[tuple[str, str], ...] = ()
    unmeasurable: tuple[str, ...] = ()
    # Candidates the screen never reached, in their original order. They are admissible by default:
    # not measuring an asset is not evidence against it.
    unexamined: tuple[str, ...] = ()

    @property
    def usable(self) -> tuple[str, ...]:
        """What the caller may actually place, admitted first."""
        return (
            self.admitted + self.unexamined + self.unmeasurable
            + tuple(asset for asset, _ in self.relaxed)
        )


def screen_candidates_by_size(
    config: HssdConfig,
    candidates: Sequence[str],
    *,
    bounds: Sequence[Any] | None,
    kind: str = "",
    ratio_limit: float = DEFAULT_RATIO_LIMIT,
    max_examined: int = 24,
    enough: int = 6,
) -> SizeScreen:
    """Split candidates into those that fit the declared size, those that do not, and those that
    could not be measured.

    Only the first ``max_examined`` are read, and reading stops once ``enough`` fit -- the same
    bargain `rank_candidates_by_size` makes. A synset can hold hundreds of assets, each read is a
    file, and the build materialises three: measuring all 24 of nine entities' candidates read 216
    GLBs per rebuild to decide an ordering whose only job is to make the first three plausible.
    Everything not examined keeps its order behind the screened ones rather than being dropped, so
    the pool is not narrowed -- only the effort spent on it.
    """
    admitted: list[str] = []
    unmeasurable: list[str] = []
    relaxed: list[tuple[float, str, str]] = []
    examined = list(candidates)[:max_examined]
    stopped_at = len(examined)
    for index, asset_id in enumerate(examined):
        if len(admitted) >= enough:
            # Enough fit. Everything from here on keeps its order, unmeasured and unjudged: the
            # build materialises three, so reading the rest buys an ordering nobody uses.
            stopped_at = index
            break
        path = config.glb_path(asset_id)
        extents = measure_glb_extents(path) if path.is_file() else None
        if extents is None:
            unmeasurable.append(asset_id)
            continue
        violation = size_bound_violation(extents, bounds, kind=kind, ratio_limit=ratio_limit)
        if violation is None:
            admitted.append(asset_id)
            continue
        # How far out, as the worst ratio across the axes, so "least bad" is a real ordering rather
        # than the order the candidates happened to arrive in.
        wanted = midpoint_of(bounds) if bounds else (0.0, 0.0, 0.0)
        measured = extents.to_list()
        severity = max(
            (abs(math.log(max(value, 1e-6) / target)) for value, target in zip(measured, wanted)
             if target > 0),
            default=0.0,
        )
        relaxed.append((severity, asset_id, violation))

    relaxed.sort(key=lambda item: (item[0], item[1]))
    # The unexamined keep their order behind the admitted ones, and the pool is never narrowed.
    remainder = [*examined[stopped_at:], *candidates[max_examined:]]
    return SizeScreen(
        admitted=tuple(admitted),
        relaxed=tuple((asset_id, reason) for _severity, asset_id, reason in relaxed),
        unmeasurable=tuple(unmeasurable),
        unexamined=tuple(remainder),
    )


def rank_candidates_by_size(
    config: HssdConfig,
    candidates: Sequence[str],
    *,
    target_width_m: float,
    max_examined: int = 24,
) -> list[str]:
    """Order candidates by how close their width is to the target.

    A cheap pre-ranking so the first attempt is plausible rather than arbitrary. It reads the GLB
    bounding box through the support surface when one exists, and falls back to file order when it
    does not, which keeps the ordering deterministic either way.

    Only the first ``max_examined`` candidates are measured. Reading a support surface means
    gunzipping and parsing a file, and a synset can hold hundreds of assets -- measuring all 170
    desks to rank them cost minutes per run for an ordering that only needs to be good enough to
    pick a plausible first candidate. The remainder keep their stable sorted order behind the
    measured ones, so the pool is not narrowed, only the ranking effort.
    """
    examined = list(candidates)[:max_examined]
    remainder = list(candidates)[max_examined:]
    scored: list[tuple[float, str]] = []
    for asset_id in examined:
        readings = read_support_surfaces(config, asset_id)
        if readings:
            width = max(readings[0].extent_x, readings[0].extent_z)
            scored.append((abs(width - target_width_m), asset_id))
        else:
            scored.append((float("inf"), asset_id))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [asset_id for _, asset_id in scored] + remainder
