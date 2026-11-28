# SPDX-License-Identifier: Apache-2.0
"""The generated furniture a background build kept, as prepared assets and delivery instances.

SPEC-011's rejection pool decides which objects `populate_assets` realised may stay, and the
ledger names every one of them -- but until now they lived **only inside the build's own scene**.
The delivery carried the room, the retrieved furniture and the ledger; it did not carry the kept
objects, so Isaac never settled them and no gate ever judged them against the retrieval periphery.
The delivery said so itself (`background.delivery_note`). This module is the missing half.

Two decisions are made here rather than in the Blender-side export, because they are the ones a
reader of the delivery has to be able to check without Blender:

* **The frame.** A kept object arrives as world-space vertices. Re-expressed in the object's own
  yaw-aligned frame with the footprint centre at the origin and the base at z = 0 -- the convention
  every prepared asset in this repository uses -- an instance's box is the object's own box. Taking
  the world axis-aligned bounds instead would overstate a yawed object by the corners it never
  occupied, which is exactly the mistake the audit tool refuses to make.
* **The identity.** A generated instance is named `generated_<category>_<ordinal>`, so the category
  a delivery reports is readable from the entity id alone, and the asset key is a digest over the
  factory, the object's name, its measured size and the upstream commit -- the same object in the
  same build always lands in the same prepared directory.

The mesh work itself is Blender-side (`rfw_w.scenes.workers.blender.export_generated_furniture` writes the raw
OBJ in this frame and the pose that puts it back). Here the raw mesh becomes a prepared asset:
visual OBJ, convex-hull collision mesh, and a `prepared.json` in the same shape the retrieved
libraries write, so one reader serves all three.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rfw_w.scenes.assets.mesh_probe import measure_obj
from rfw_w.scenes.assets.prepared_obj import write_scene_obj
from rfw_w.scenes.assets.providers.hssd import METADATA_FILENAME, PreparedFurniture
from rfw_w.scenes.contracts.asset import AffordanceStatus, AssetKey, Provenance
from rfw_w.scenes.contracts.background_constraints import GENERATED_PROVIDER
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.units import Vec3

#: The assembly the Blender-side export writes, and the only file the Python side reads.
ASSEMBLY_FILENAME = "objects.json"
ASSEMBLY_SCHEMA_VERSION = "pvscene.generated-assembly/v1"

#: A generated object's sidecar is written in the HSSD schema's shape (`PreparedFurniture`), because
#: one shape means one reader; this string is what tells a reader that the *provenance* is
#: different -- a mesh that came out of a generated scene, not out of a library.
PREPARED_SCHEMA_VERSION = "pvscene.generated-prepared/v1"

#: What a generated object is assumed to weigh. Upstream's furniture factories do not carry mass
#: and the mesh does not say what it is made of, so the mass is a volume-times-density estimate and
#: is recorded as one (`AffordanceStatus.ESTIMATED`): enough for a static piece to be settled, never
#: presented as a measurement.
DENSITY_KG_M3 = 500.0

#: A collision hull is a physics proxy, never the visual mesh -- the same rule the retrieved
#: libraries follow (`hssd.prepare_furniture`).
HULL_TARGET_FACES = 256


def library_version(upstream_pin: str) -> str:
    """Which upstream tree the furniture was realised by, in the `asset_ref` string.

    A generated asset's library version is not a published release; it is the commit of the
    pinned Infinigen tree that solved it, so the ref carries the commit rather than a claim about
    a version nobody tagged.
    """
    pin = str(upstream_pin).strip()
    if not pin:
        raise ValueError("GENERATED_PIN_MISSING: a generated asset must name the upstream commit")
    return f"infinigen@{pin[:12]}"


def asset_key(*, factory: str, name: str, upstream_pin: str,
              dimensions_m: Sequence[float]) -> str:
    """A stable directory name for one generated object.

    A digest rather than the object's name: Blender names are long (`MonitorFactory(8973379)
    .spawn_asset(9247255)`), carry parentheses a path should not, and are not unique across two
    builds. The digest is over what identifies the object -- its factory, its realised name, its
    measured size and the upstream commit -- so the same object in a rebuilt scene lands in the
    same directory and a different one cannot reuse its artefacts.
    """
    return content_hash({
        "factory": factory, "name": name, "upstream_pin": str(upstream_pin),
        "dimensions_m": [round(float(value), 6) for value in dimensions_m],
    })[:40]


@dataclass(frozen=True, slots=True)
class LocalFrame:
    """One kept object's world geometry, re-expressed in its own frame.

    ``points`` are in the object's yaw-aligned frame with the footprint centre at the origin and
    the base at z = 0; ``position_m`` is where that origin sits in the room, and ``yaw_rad`` is the
    rotation that puts the local frame back. The pair is exact: rotating the local box by ``yaw``
    and placing it at ``position_m`` reproduces the box the object really occupies, which a
    world-axis-aligned bound would not.
    """

    points: tuple[tuple[float, float, float], ...]
    bounds_min: Vec3
    bounds_max: Vec3
    position_m: Vec3
    yaw_rad: float

    @property
    def dimensions_m(self) -> Vec3:
        return self.bounds_max - self.bounds_min


def recentre(points: Iterable[Sequence[float]], *, yaw_rad: float) -> LocalFrame:
    """Re-express world-space points in the object's yaw-aligned frame.

    The rotation is the only place a sign can hide, so it is written once: the local frame is the
    world frame turned by ``-yaw``, the geometry is centred in *that* frame, and the origin is
    turned back by ``+yaw`` to get the pose that returns the object to where it really is.
    """
    cos, sin = math.cos(-yaw_rad), math.sin(-yaw_rad)
    turned = [(cos * float(x) - sin * float(y), sin * float(x) + cos * float(y), float(z))
              for x, y, z in points]
    if not turned:
        raise ValueError("GENERATED_ASSET_EMPTY(no vertices): an object with no geometry is not one")
    low = [min(point[axis] for point in turned) for axis in range(3)]
    high = [max(point[axis] for point in turned) for axis in range(3)]
    # Centre in x and y, **base** in z: the prepared convention is `footprint_centre_base_at_zero`,
    # and the instances say `origin_at_base=True` -- a mesh whose z was centred would float by half
    # its height, which is a silent error the read side has no way to notice from the numbers alone.
    centre = [(low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0, low[2]]
    local = tuple((point[0] - centre[0], point[1] - centre[1], point[2] - centre[2])
                  for point in turned)
    # Back to world: the origin the local frame was centred on, turned by +yaw.
    back_cos, back_sin = math.cos(yaw_rad), math.sin(yaw_rad)
    origin_x = back_cos * centre[0] - back_sin * centre[1]
    origin_y = back_sin * centre[0] + back_cos * centre[1]
    return LocalFrame(
        points=local,
        bounds_min=Vec3(low[0] - centre[0], low[1] - centre[1], low[2] - centre[2]),
        bounds_max=Vec3(high[0] - centre[0], high[1] - centre[1], high[2] - centre[2]),
        position_m=Vec3(origin_x, origin_y, centre[2]),
        yaw_rad=float(yaw_rad),
    )


def read_assembly(path: Path) -> dict[str, Any]:
    """The Blender-side export, validated by name rather than trusted."""
    payload = json.loads(Path(path).read_text())
    version = payload.get("schema_version")
    if version != ASSEMBLY_SCHEMA_VERSION:
        raise ValueError(f"GENERATED_ASSEMBLY_SCHEMA_UNSUPPORTED({version})")
    if "objects" not in payload:
        raise ValueError("GENERATED_ASSEMBLY_MALFORMED(no objects list)")
    return payload


def kept_objects(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The kept objects the export wrote geometry for, in a deterministic order."""
    return sorted((item for item in payload.get("objects", ()) if item.get("outcome") == "kept"),
                  key=lambda item: str(item.get("name", "")))


def prepare_asset(record: Mapping[str, Any], raw_obj: Path, out_dir: Path, *,
                  upstream_pin: str, density_kg_m3: float = DENSITY_KG_M3) -> PreparedFurniture:
    """Write one generated mesh as a prepared asset, and return what was written about it.

    The mesh is measured **after** it is written (`mesh_probe.measure_obj`), not from the in-memory
    geometry: the numbers that reach the delivery are the numbers on disk, which is the same rule
    the constructed-support path follows and the reason a mismatch is caught here rather than in a
    render.
    """
    try:
        import trimesh  # imported here so the core stays stdlib-only
    except ImportError as error:
        raise RuntimeError(
            "preparing generated furniture needs trimesh; install the geometry extra"
        ) from error

    dimensions = [float(value) for value in record["dimensions_m"]]
    key = asset_key(factory=str(record["factory"]), name=str(record["name"]),
                    upstream_pin=upstream_pin, dimensions_m=dimensions)
    directory = Path(out_dir) / key
    directory.mkdir(parents=True, exist_ok=True)
    visual_obj = directory / f"{key}.obj"
    collision_obj = directory / f"{key}_collision.obj"

    loaded = trimesh.load(str(raw_obj), force='scene')
    meshes = (list(loaded.geometry.values()) if hasattr(loaded, "geometry")
              else [loaded])
    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    if merged.faces is None or len(merged.faces) == 0:
        raise ValueError(f"GENERATED_ASSET_EMPTY({record.get('name')}): the mesh has no faces")
    # The visual OBJ is rewritten in this repository's own writer, so the Isaac reader gets the
    # per-material groups and the base-colour factors it was built for rather than a second
    # dialect. The mesh is already in the prepared frame -- the Blender export centred it.
    write_scene_obj(loaded, visual_obj)

    hull = merged.convex_hull
    if len(hull.faces) > HULL_TARGET_FACES:
        hull = hull.simplify_quadric_decimation(face_count=HULL_TARGET_FACES)
    hull.export(str(collision_obj))

    measured = measure_obj(visual_obj)
    collision = measure_obj(collision_obj)
    for value, expected in zip(measured.extents.to_list(), dimensions):
        if abs(value - expected) > 1e-4:
            raise ValueError(
                f"GENERATED_ASSET_MEASUREMENT_MISMATCH({record.get('name')}): the written mesh is "
                f"{measured.extents.to_list()} but the export recorded {dimensions}"
            )
    volume_m3 = float(abs(merged.convex_hull.volume))
    provenance = Provenance(
        method="generated_hull_volume_times_assumed_density",
        version=library_version(upstream_pin),
        status=AffordanceStatus.ESTIMATED,
        source_uri=str(record.get("name", "")),
    )
    prepared = PreparedFurniture(
        asset_key=AssetKey(provider=GENERATED_PROVIDER,
                           library_version=library_version(upstream_pin), asset_id=key),
        visual_obj=visual_obj,
        collision_obj=collision_obj,
        dimensions_m=measured.extents,
        bounds_min_m=Vec3(measured.bounds_min.x, measured.bounds_min.y, measured.bounds_min.z),
        bounds_max_m=Vec3(measured.bounds_max.x, measured.bounds_max.y, measured.bounds_max.z),
        mass_kg=max(volume_m3 * density_kg_m3, 0.1),
        # No support surface is claimed: a generated shelf is not a declared support of anything in
        # the task, and a polygon invented here would be a placeable surface nobody asked for.
        support_polygons=(),
        vertex_count=measured.vertex_count,
        face_count=measured.face_count,
        collision_face_count=collision.face_count,
        provenance=provenance,
        has_diffuse_texture=bool(record.get("has_texture", False)),
        origin_convention="footprint_centre_base_at_zero",
    )
    (directory / METADATA_FILENAME).write_text(
        json.dumps(prepared.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return prepared
