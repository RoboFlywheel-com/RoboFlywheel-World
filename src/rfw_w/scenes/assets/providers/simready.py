# SPDX-License-Identifier: Apache-2.0
"""Preparing a SimReady target: the library's own mesh, in one frame we can trust.

v1.3 ships each asset twice -- ``mesh/<leaf>.obj`` in the source's own frame and ``mesh/m_<leaf>.obj``
in a normalised one -- and a collision mesh only in the normalised family. Whether the normalised
family is upright is a per-asset decision the release made with a heuristic, and its result is not
consistent: measured on the release's own files, the condiment jar ``3_22_0027`` comes out with its
axis moved from Y to Z (upright as intended), while the mug ``4_3_0325`` keeps its axis on Y and has
only its X and Z extents swapped. Loading the normalised family as Z-up therefore stands the mug on
its side, which is what the first acceptance run against v1.3 reported verbatim: "the cup is lying
on its side, creating an unstable and implausible initial pose for the pick task".

Its ``up_axis_fix`` block is a record of what was done, not a guarantee of what came out, and a
loader cannot repair a sideways asset from it.

So the target is prepared here instead, from the ``<leaf>.obj`` the source shipped: rotated once by
the same Y-up correction every asset this project has loaded has needed, with a collision hull built
from that same geometry. One frame for the visual and the proxy, taken from the mesh that is
actually placed, and nothing inferred from a statement about a different file.

The cost is physics fidelity: the preparation's proxy is a single decimated convex hull rather than
the release's per-part decomposition. That is the same trade the HSSD path has always made, and the
release's decomposition is in a frame that would have to be corrected before it could be used at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rfw_w.scenes.assets.prepared_obj import UP_FIX_ROWS, sanitise, write_scene_obj
from rfw_w.scenes.contracts.units import Vec3

SCHEMA_VERSION = "pvscene.simready-prepared/v1"
METADATA_FILENAME = "prepared.json"

# The proxy's face budget, matching the furniture path's: a hull of an 60k-vertex asset is exact and
# slow, and a convex hull of a rigid object is already an approximation.
HULL_TARGET_FACES = 256


class SimreadyPrepareError(RuntimeError):
    """The target could not be prepared."""


@dataclass(frozen=True, slots=True)
class SimreadyPrepareConfig:
    """Where the library is, and where prepared output goes."""

    assets_root: Path
    prepared_root: Path

    def __post_init__(self) -> None:
        if not self.assets_root.is_dir():
            raise SimreadyPrepareError(f"SimReady assets root not found: {self.assets_root}")

    def source_obj(self, prefix: str, leaf: str) -> Path:
        """The mesh the source shipped, in its own frame.

        ``prefix`` is the asset's own directory under the library root (``<domain>/<class>/<leaf>``),
        which is what the catalogue index records; ``leaf`` is the asset's name in the release.
        """
        return self.assets_root / prefix / "mesh" / f"{leaf}.obj"

    def metadata_path(self, asset_id: str) -> Path:
        return self.prepared_root / asset_id / METADATA_FILENAME


@dataclass(frozen=True, slots=True)
class PreparedRigid:
    """A prepared target: a mesh, a proxy, and what was measured about both."""

    asset_id: str
    visual_obj: Path
    collision_obj: Path
    dimensions_m: Vec3
    bounds_min_m: Vec3
    bounds_max_m: Vec3
    vertex_count: int
    face_count: int
    collision_face_count: int
    has_diffuse_texture: bool
    source_uri: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "asset_id": self.asset_id,
            "visual_obj": self.visual_obj.name,
            "collision_obj": self.collision_obj.name,
            "dimensions_m": self.dimensions_m.to_list(),
            "bounds_min_m": self.bounds_min_m.to_list(),
            "bounds_max_m": self.bounds_max_m.to_list(),
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "collision_face_count": self.collision_face_count,
            "has_diffuse_texture": self.has_diffuse_texture,
            "source_uri": self.source_uri,
        }


def _prepared_from_dict(payload: Mapping[str, Any], *, directory: Path) -> PreparedRigid:
    return PreparedRigid(
        asset_id=str(payload["asset_id"]),
        visual_obj=directory / str(payload["visual_obj"]),
        collision_obj=directory / str(payload["collision_obj"]),
        dimensions_m=Vec3(*[float(value) for value in payload["dimensions_m"]]),
        bounds_min_m=Vec3(*[float(value) for value in payload["bounds_min_m"]]),
        bounds_max_m=Vec3(*[float(value) for value in payload["bounds_max_m"]]),
        vertex_count=int(payload["vertex_count"]),
        face_count=int(payload["face_count"]),
        collision_face_count=int(payload["collision_face_count"]),
        has_diffuse_texture=bool(payload["has_diffuse_texture"]),
        source_uri=str(payload.get("source_uri", "")),
    )


def _cache_hit(config: SimreadyPrepareConfig, asset_id: str, source: Path) -> PreparedRigid | None:
    """A preparation already on disk, when it was made from this same source file.

    The source's size and modification time are the identity: a release that re-exports an asset in
    place would otherwise keep being served the mesh prepared from the file it replaced, and the
    scene would then be built from geometry that is not on disk.
    """
    metadata = config.metadata_path(asset_id)
    if not metadata.is_file():
        return None
    try:
        payload = json.loads(metadata.read_text())
        recorded = json.loads(metadata.read_text()).get("source_identity")
        stat = source.stat()
    except (OSError, ValueError):
        return None
    if recorded != {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}:
        return None
    try:
        prepared = _prepared_from_dict(payload, directory=metadata.parent)
    except (KeyError, TypeError, ValueError):
        return None
    if not prepared.visual_obj.is_file() or not prepared.collision_obj.is_file():
        return None
    return prepared


def prepare_rigid(
    config: SimreadyPrepareConfig,
    *,
    prefix: str,
    leaf: str,
    asset_id: str | None = None,
) -> PreparedRigid:
    """Turn one library mesh into a Z-up OBJ pair with a collision hull, and record it.

    ``asset_id`` defaults to the release's own leaf name, so the prepared directory is named after
    the asset a reader can go and look up.
    """
    try:
        import trimesh  # imported here so the core stays stdlib-only
    except ImportError as error:
        raise SimreadyPrepareError(
            "preparing a SimReady target needs trimesh; install the geometry extra"
        ) from error

    name = asset_id or leaf
    source = config.source_obj(prefix, leaf)
    if not source.is_file():
        raise SimreadyPrepareError(
            f"no source mesh for {leaf} at {source}; the release ships the asset's own frame at "
            "mesh/<leaf>.obj, and the normalised family is not a substitute"
        )

    cached = _cache_hit(config, name, source)
    if cached is not None:
        return cached

    try:
        loaded = trimesh.load(str(source))
    except Exception as error:  # noqa: BLE001 - trimesh raises several types on a bad mesh
        raise SimreadyPrepareError(f"could not load {source}: {type(error).__name__}: {error}") from error
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise SimreadyPrepareError(
            f"{source} loaded as {type(loaded).__name__}, which is neither a mesh nor a scene"
        )
    if not scene.geometry or not any(len(g.vertices) for g in scene.geometry.values()):
        raise SimreadyPrepareError(f"{source} contains no geometry")

    import numpy as np

    scene.apply_transform(np.asarray(UP_FIX_ROWS, dtype=float))
    target = config.prepared_root / name
    target.mkdir(parents=True, exist_ok=True)
    visual_obj = target / f"{name}.obj"
    collision_obj = target / f"{name}_collision.obj"
    write_scene_obj(scene, visual_obj)

    merged = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    try:
        hull = merged.convex_hull
        if len(hull.faces) > HULL_TARGET_FACES:
            hull = hull.simplify_quadric_decimation(face_count=HULL_TARGET_FACES)
        hull.export(str(collision_obj))
    except Exception as error:  # noqa: BLE001 - hulling can fail on degenerate geometry
        raise SimreadyPrepareError(f"could not build a collision hull for {leaf}: {error}") from error

    bounds_min = merged.bounds[0]
    bounds_max = merged.bounds[1]
    extents = bounds_max - bounds_min
    written = sorted(p.name for p in target.glob("*.png")) + sorted(p.name for p in target.glob("*.jpg"))
    textured = False
    for image in written:
        try:
            from PIL import Image

            with Image.open(target / image) as opened:
                if min(opened.size) >= 64:
                    textured = True
                    break
        except Exception:  # noqa: BLE001 - an unreadable image counts as no image
            continue

    prepared = PreparedRigid(
        asset_id=name,
        visual_obj=visual_obj,
        collision_obj=collision_obj,
        dimensions_m=Vec3(*(round(float(value), 6) for value in extents)),
        bounds_min_m=Vec3(*(round(float(value), 6) for value in bounds_min)),
        bounds_max_m=Vec3(*(round(float(value), 6) for value in bounds_max)),
        vertex_count=int(len(merged.vertices)),
        face_count=int(len(merged.faces)),
        collision_face_count=int(len(hull.faces)),
        has_diffuse_texture=textured,
        source_uri=str(source),
    )
    stat = source.stat()
    payload = prepared.to_dict()
    payload["source_identity"] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    config.metadata_path(name).write_text(json.dumps(payload, indent=1, sort_keys=True))
    return prepared


def prepared_name(prefix: str, leaf: str) -> str:
    """The directory name a preparation gets: the release's leaf, made file-name safe."""
    return sanitise(f"{prefix.split('/')[0]}_{leaf}")
