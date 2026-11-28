# SPDX-License-Identifier: Apache-2.0
"""Turn a support shape into the artefacts the rest of the pipeline already expects.

``solvers/support_shape.py`` says what a support is made of; this module makes it exist. It writes
the mesh, measures the mesh it wrote, and returns a record shaped like the one the HSSD provider
returns for a retrieved asset -- so the placement solver, the gates, ``scene_ir`` and the Isaac
worker need no second code path for a support that was built rather than fetched.

Why the measurement round trip matters
--------------------------------------

The dimensions recorded here are **measured from the OBJ after it is written**, not copied from the
numbers the caller asked for. That is the whole reason a constructed support can honestly claim
:attr:`AffordanceStatus.CONSTRUCTED` and be usable for a hard requirement: the number the task was
solved against and the number that exists in the scene are the same number *because it was checked*,
not because both came from one variable. A writer bug that dropped a leg or mis-scaled a slab shows
up here as a disagreement rather than three months later as a target floating above a table.

The mass is the exception and says so: it is the sum of the parts' volumes times a density, because
the parts are boxes and their volume is exact -- no hull, no estimate. The method string records that
this is where the number came from.

Stdlib only (measurement included), and walked by ``rfw_w.scenes.validation.check_stdlib_only``.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Sequence

from rfw_w.scenes.assets.mesh_probe import measure_obj
from rfw_w.scenes.assets.providers.hssd import METADATA_FILENAME, PreparedFurniture
from rfw_w.scenes.contracts.asset import (
    AffordanceStatus,
    AssetKey,
    Provenance,
    SupportPolygon,
)
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.units import Vec3
from rfw_w.scenes.solvers.support_shape import SupportPart, SupportShape, parts_for, top_surface, volume_m3


class ProceduralSupportError(RuntimeError):
    """The support could not be built, or what was built is not what was asked for."""


PROVIDER = "constructed"
LIBRARY_VERSION = "pvscene.procedural/v1"

# The density a built support is given. A real desk is mostly air and veneer, but these boxes are
# solids standing in for furniture, so this is a *solid* density: 500 kg/m3 sits between the light
# hardwoods. Recorded through the provenance method rather than presented as a measurement, because
# nothing weighed anything.
DEFAULT_DENSITY_KG_M3 = 500.0

# A generated support is a handful of boxes; anything larger means the recipe ran away.
MAX_PARTS = 32


def write_obj(
    parts: Sequence[SupportPart],
    path: Path,
    *,
    material: str | None = None,
    material_library: str | None = None,
    textured: bool = False,
) -> None:
    """Write the parts as one Wavefront OBJ, boxes only.

    Normals are derived by the renderer. Textured visual meshes carry face-corner UVs;
    collision meshes omit them. Faces are quads, as for the worker's other box primitives.

    ``material``/``material_library`` name the sidecar ``.mtl`` the mesh wears, which is how a
    built support gets a colour at all: without them the renderer falls back to a colour it invents
    -- measured, a lavender desk in a room whose prompt asked for warm wood.
    """
    if not parts:
        raise ProceduralSupportError("a support needs at least one part")
    if len(parts) > MAX_PARTS:
        raise ProceduralSupportError(
            f"REFUSED_TOO_MANY_PARTS({len(parts)}): a support of boxes should be a handful, and "
            f"the limit is {MAX_PARTS}"
        )
    lines: list[str] = [
        "# A support built by solvers/support_shape.py. Metres, Z-up, footprint centred, base at z=0.",
    ]
    if material_library:
        lines.append(f"mtllib {material_library}")
    if material:
        lines.append(f"usemtl {material}")
    faces: list[str] = []
    if textured:
        lines.extend(("vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"))
    # Counted, not derived from `len(lines)`: the header carries a comment and possibly an `mtllib`
    # and a `usemtl`, and treating those as vertices shifts every face index -- which the mesh
    # *measurement* does not notice, because it counts `v` and `f` lines without checking that the
    # indices resolve. The worker's `load_obj` does notice, and refuses the file.
    emitted_vertices = 0
    for part in parts:
        half_x, half_y, half_z = (extent / 2.0 for extent in part.size_m)
        cx, cy, cz = part.centre_m
        base = emitted_vertices
        corners = [
            (cx - half_x, cy - half_y, cz - half_z),
            (cx + half_x, cy - half_y, cz - half_z),
            (cx + half_x, cy + half_y, cz - half_z),
            (cx - half_x, cy + half_y, cz - half_z),
            (cx - half_x, cy - half_y, cz + half_z),
            (cx + half_x, cy - half_y, cz + half_z),
            (cx + half_x, cy + half_y, cz + half_z),
            (cx - half_x, cy + half_y, cz + half_z),
        ]
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in corners)
        emitted_vertices += len(corners)
        # Outward winding, so the mesh is not inside-out for anything that cares.
        for a, b, c, d in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                           (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            indices = (base + a + 1, base + b + 1, base + c + 1, base + d + 1)
            faces.append("f " + " ".join(
                f"{vertex}/{uv}" if textured else str(vertex)
                for uv, vertex in enumerate(indices, 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*lines, *faces, ""]), encoding="utf-8")


def materialise(
    shape: SupportShape,
    *,
    width_m: float,
    depth_m: float,
    height_m: float,
    out_dir: Path,
    asset_id: str | None = None,
    density_kg_m3: float = DEFAULT_DENSITY_KG_M3,
    colour: tuple[float, float, float] | None = None,
    roughness: float = 0.6,
    wood_grain: bool = False,
    **shape_kwargs: float,
) -> PreparedFurniture:
    """Build the support and return the record everything downstream already knows how to read.

    ``out_dir`` is the run's prepared-asset directory -- the same one the HSSD provider writes into,
    so the worker's paths, the manifest's inventory and the run's gitignore all keep working.

    ``asset_id`` defaults to a digest of the shape and its dimensions, so the same request produces
    the same directory name and a different table cannot silently reuse the last one's artefacts.
    """
    parts = parts_for(shape, width_m=width_m, depth_m=depth_m, height_m=height_m, **shape_kwargs)
    polygon, top_z = top_surface(shape, width_m=width_m, depth_m=depth_m, height_m=height_m,
                                **shape_kwargs)
    if colour is not None:
        _require_colour(colour)
    key = asset_id or _digest(shape, parts, colour)
    if wood_grain and colour is not None:
        key = content_hash({"base": key, "finish": "wood_grain_v1", "roughness": roughness})[:40]
    directory = out_dir / key
    visual_obj = directory / f"{key}.obj"
    collision_obj = directory / f"{key}_collision.obj"

    material_name = f"support_{key[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    if colour is not None:
        library = f"{key}.mtl"
        (directory / library).write_text(
            "\n".join([
                f"# The support's finish, from the spec's palette (role: furniture).",
                f"newmtl {material_name}",
                f"Kd {colour[0]:.6f} {colour[1]:.6f} {colour[2]:.6f}",
                f"Ks 0.000000 0.000000 0.000000",
                f"Ns {max(1.0, min(1000.0, (1.0 - roughness) * 1000.0)):.6f}",
                *(["map_Kd wood_grain.png"] if wood_grain else []),
                "",
            ]),
            encoding="utf-8",
        )
        if wood_grain:
            _write_wood_grain(directory / "wood_grain.png", colour)
    write_obj(
        parts, visual_obj,
        material=material_name if colour is not None else None,
        material_library=f"{key}.mtl" if colour is not None else None,
        textured=wood_grain and colour is not None,
    )
    # The collision mesh is the visual mesh, not a convex hull. A table is not convex, and a hull
    # around one would collide as a solid block -- an arm could not reach under it and a physics
    # settle would report a contact where the scene has empty space. No material: it is never
    # rendered.
    write_obj(parts, collision_obj)

    measured = measure_obj(visual_obj)
    collision = measure_obj(collision_obj)
    _require_measured_matches(shape, measured.bounds_min, measured.bounds_max, width_m, depth_m, height_m)
    _require_measured_matches(
        shape, collision.bounds_min, collision.bounds_max, width_m, depth_m, height_m
    )

    provenance = Provenance(
        method="constructed_part_volume_times_density",
        version=LIBRARY_VERSION,
        status=AffordanceStatus.CONSTRUCTED,
        source_uri=str(visual_obj),
    )
    prepared = PreparedFurniture(
        asset_key=AssetKey(provider=PROVIDER, library_version=LIBRARY_VERSION, asset_id=key),
        visual_obj=visual_obj,
        collision_obj=collision_obj,
        dimensions_m=measured.extents,
        bounds_min_m=Vec3(measured.bounds_min.x, measured.bounds_min.y, 0.0),
        bounds_max_m=Vec3(measured.bounds_max.x, measured.bounds_max.y, measured.extents.z),
        mass_kg=max(volume_m3(parts) * density_kg_m3, 0.1),
        support_polygons=(
            SupportPolygon(
                polygon_id=f"{key}:top",
                vertices_xy=polygon.vertices,
                plane_z_m=top_z,
                provenance=provenance,
            ),
        ),
        vertex_count=measured.vertex_count,
        face_count=measured.face_count,
        collision_face_count=collision.face_count,
        provenance=provenance,
        has_diffuse_texture=wood_grain and colour is not None,
        origin_convention="footprint_centre_base_at_zero",
    )
    (directory / METADATA_FILENAME).write_text(
        json.dumps(prepared.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return prepared


def _write_wood_grain(path: Path, colour: tuple[float, float, float]) -> None:
    """A deterministic procedural veneer, explicitly synthetic, with no external image dependency."""
    size = 256
    pixels = bytearray()
    for y in range(size):
        pixels.append(0)  # PNG filter: none
        for x in range(size):
            wave = y + 2.5 * math.sin(x / 41.0) + 1.2 * math.sin(x / 19.0 + y / 37.0)
            grain = 0.92 + 0.09 * math.sin(wave * 1.8) + 0.035 * math.sin(wave * 6.2)
            # Palette colours are linear; the PNG texture is sampled as sRGB by the renderer.
            pixels.extend(max(0, min(255, round((channel * grain) ** (1 / 2.2) * 255)))
                          for channel in colour)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(bytes(pixels))) + chunk(b"IEND", b""))


def _require_colour(colour: tuple[float, float, float]) -> None:
    if len(colour) != 3 or any(not math.isfinite(c) or not 0.0 <= c <= 1.0 for c in colour):
        raise ProceduralSupportError(
            f"a support's colour must be three channels within [0, 1], got {colour}"
        )


def _digest(
    shape: SupportShape, parts: Sequence[SupportPart], colour: tuple[float, float, float] | None,
) -> str:
    """A stable name for one built support, from the shape it is, not from when it was built.

    The colour is in the digest, not just the geometry: two supports of the same form in different
    finishes are different artefacts, and a name that ignored the finish would let the second
    silently reuse the first one's mesh and sidecar.
    """
    return content_hash({
        "shape": shape.value,
        "parts": [part.to_dict() for part in parts],
        "colour": None if colour is None else [round(channel, 6) for channel in colour],
    })[:40]


def _require_measured_matches(
    shape: SupportShape,
    bounds_min: Vec3,
    bounds_max: Vec3,
    width_m: float,
    depth_m: float,
    height_m: float,
) -> None:
    """The mesh that was written must be the support that was asked for.

    Checked against the file, after the file exists. This is the step that makes the difference
    between "we recorded the numbers we chose" and "we measured what we built", and it is the only
    reason a constructed support may be treated as a hard-requirement affordance.
    """
    tolerance = 1e-6
    if (
        abs(bounds_min.x + width_m / 2.0) > tolerance
        or abs(bounds_max.x - width_m / 2.0) > tolerance
        or abs(bounds_min.y + depth_m / 2.0) > tolerance
        or abs(bounds_max.y - depth_m / 2.0) > tolerance
        or abs(bounds_min.z) > tolerance
        or abs(bounds_max.z - height_m) > tolerance
    ):
        raise ProceduralSupportError(
            f"CONSTRUCTED_SUPPORT_MEASURES_WRONG({shape.value}): the written mesh spans "
            f"({bounds_min.x:.6f}, {bounds_min.y:.6f}, {bounds_min.z:.6f})-"
            f"({bounds_max.x:.6f}, {bounds_max.y:.6f}, {bounds_max.z:.6f}) but the support asked for "
            f"is {width_m} x {depth_m} m with its top at {height_m} m, centred with its base at zero"
        )
