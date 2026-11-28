# SPDX-License-Identifier: Apache-2.0
"""Stage geometry: room solids and OBJ meshes, authored into USD.

Runs under Isaac's interpreter, so it may use ``pxr``; it must not use anything from the
orchestrator's runtime.

Two deliberate choices:

* the floor and walls are built from the room's **actual polygon**, extruded. Building a box from
  the bounding extent would render a room the scene does not have, and the delivered image would
  then disagree with the geometry the gates validated -- the image would look right while showing
  a different room.
* OBJ meshes are loaded directly rather than converted through an asset pipeline, so there is no
  hidden conversion step whose output could differ from the file the measurement was taken from.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# The installed worker and orchestrator share the same pure polygon implementation.
from rfw_w.scenes.geometry.polygon import (  # noqa: E402
    Polygon,
    PolygonError,
    triangulate_indices,
    validate_simple,
)

# Parser tolerances and limits, so a malformed asset cannot take the worker down mid-render.
MAX_VERTICES = 4_000_000


class GeometryError(RuntimeError):
    """The geometry could not be authored."""


def load_obj(path: Path) -> "ObjMesh":
    """Read an OBJ into points, polygons and texture coordinates.

    Handles ``v``, ``vt`` and ``f``, which is what these assets contain. Face references may be
    ``v``, ``v/vt`` or ``v/vt/vn``; the texture index is the second field and the normal the third,
    and all three forms are common in the same library.

    Texture coordinates are read because leaving them behind is what produced untextured renders
    from fully textured assets: the files carry a ``vt`` per vertex and an ``mtllib`` pointing at a
    2048x2048 map, and a loader that reads only ``v`` and ``f`` silently drops all of it.

    Face indices are converted from OBJ's 1-based (and possibly negative, meaning relative) form to
    0-based, because getting that wrong produces a mesh that renders without complaint and is wrong.
    """
    points: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    faces: list[list[int]] = []
    face_uvs: list[list[int]] = []
    materials: list[str] = []
    library: str | None = None
    current_material = ""

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                if len(parts) < 4:
                    continue
                points.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(points) > MAX_VERTICES:
                    raise GeometryError(
                        f"{path}: more than {MAX_VERTICES} vertices; refusing to author a mesh "
                        "this large"
                    )
            elif tag == "vt":
                if len(parts) < 3:
                    continue
                texcoords.append((float(parts[1]), float(parts[2])))
            elif tag == "mtllib":
                if len(parts) > 1:
                    library = " ".join(parts[1:])
            elif tag == "usemtl":
                current_material = " ".join(parts[1:]) if len(parts) > 1 else ""
            elif tag == "f":
                indices: list[int] = []
                uvs: list[int] = []
                for token in parts[1:]:
                    fields = token.split("/")
                    raw = fields[0]
                    if not raw:
                        continue
                    value = int(raw)
                    # OBJ is 1-based; negative values are relative to the end.
                    index = value - 1 if value > 0 else len(points) + value
                    if not 0 <= index < len(points):
                        raise GeometryError(
                            f"{path}: face references vertex {value}, which is out of range"
                        )
                    indices.append(index)
                    if len(fields) > 1 and fields[1]:
                        uv_value = int(fields[1])
                        uv_index = uv_value - 1 if uv_value > 0 else len(texcoords) + uv_value
                        if 0 <= uv_index < len(texcoords):
                            uvs.append(uv_index)
                if len(indices) >= 3:
                    faces.append(indices)
                    face_uvs.append(uvs if len(uvs) == len(indices) else [])
                    materials.append(current_material)

    if not points:
        raise GeometryError(f"{path}: contains no vertices")
    return ObjMesh(
        points=points, faces=faces, texcoords=texcoords, face_texcoords=face_uvs,
        materials=materials, material_library=library,
    )


@dataclass(frozen=True, slots=True)
class ObjMesh:
    """An OBJ's geometry plus whatever material information came with it.

    ``face_texcoords`` is empty for a face when the file gave fewer texture indices than vertices,
    which is how a partially textured mesh appears; the caller decides whether that is fatal rather
    than this reader guessing.
    """

    points: list[tuple[float, float, float]]
    faces: list[list[int]]
    texcoords: list[tuple[float, float]] = field(default_factory=list)
    face_texcoords: list[list[int]] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    material_library: str | None = None

    @property
    def is_textured(self) -> bool:
        return bool(self.texcoords) and any(self.face_texcoords)

    def material_names(self) -> tuple[str, ...]:
        """The materials this mesh's faces name, in the order the faces first name them."""
        names: list[str] = []
        for name in self.materials:
            if name not in names:
                names.append(name)
        return tuple(names)

    def group_has_uvs(self, material: str) -> bool:
        """Whether every face of one material carries a texture coordinate.

        Asked per group rather than per mesh because a real asset is mixed: the office ceiling
        light's metal ships a 588 KB map and its glass ships none, and asking the mesh as a whole
        (`is_textured`) answers for whichever group happened to come first -- which is how the
        metal's map was dropped. A map needs coordinates on all of its own faces, so this is the
        question that has to be true before one is bound.
        """
        for name, faces in self.material_groups():
            if name == material:
                return bool(faces) and all(self.face_texcoords[index] for index in faces)
        return False

    def material_groups(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Faces grouped by the material they name, in first-appearance order.

        A mesh whose faces name one material -- or none, which is how a single-material export
        writes them -- comes back as a single group, so a caller that binds a material per group
        binds exactly what it bound when there was only one.
        """
        order: list[str] = []
        groups: dict[str, list[int]] = {}
        for index, name in enumerate(self.materials):
            if name not in groups:
                groups[name] = []
                order.append(name)
            groups[name].append(index)
        if not order:
            return (("", ()),)
        return tuple((name, tuple(groups[name])) for name in order)

    def _material_fields(self, obj_path: Path) -> dict[str, dict[str, str]]:
        """Every material the OBJ's libraries state, keyed by name, in declaration order.

        One read answers both "what colour is this material" and "what map does it carry", and both
        answers come from the same library and the same entry -- two walks of one file is how the
        two would come to disagree about which material they had read.
        """
        fields: dict[str, dict[str, str]] = {}
        if not self.material_library:
            return fields
        current: dict[str, str] | None = None
        for name in str(self.material_library).split():
            mtl = obj_path.parent / name
            if not mtl.is_file():
                continue
            for line in mtl.read_text(errors="replace").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "newmtl":
                    key = " ".join(parts[1:])
                    current = fields.setdefault(key, {})
                elif current is not None and len(parts) > 1:
                    # First declaration wins, as it did when this returned on the first match: a
                    # material named by two libraries is the first one's, not the last one's.
                    current.setdefault(parts[0], " ".join(parts[1:]))
        return fields

    def _fields_for(
        self, fields: dict[str, dict[str, str]], material: str | None
    ) -> dict[str, str] | None:
        """One material's entry, defaulting to the first the faces name.

        Faces that name nothing (`usemtl` absent, which is how a single-material export writes
        them) take the first material the file declares, and a name the libraries do not state
        resolves to nothing rather than to somebody else's material.
        """
        if not fields:
            return None
        if material is not None:
            return fields.get(material)
        for name in self.materials:
            if name in fields:
                return fields[name]
        return next(iter(fields.values()))

    def material_colour(
        self, obj_path: Path, material: str | None = None
    ) -> tuple[float, float, float] | None:
        """One material's ``Kd``, when it carries a colour rather than a map.

        A material with no image is not a material with no appearance: these assets are coloured by
        a factor. Returning the factor lets the renderer show the colour the asset specifies instead
        of the black a placeholder would give -- and in a multi-material asset the factor is the
        only thing separating the parts, so it is read per material rather than once for the mesh.
        """
        entry = self._fields_for(self._material_fields(obj_path), material)
        if entry is None:
            return None
        stated = entry.get("Kd")
        parts = stated.split() if stated else []
        if len(parts) < 3:
            return None
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            return None

    def diffuse_texture(self, obj_path: Path, material: str | None = None) -> Path | None:
        """Resolve one material's ``map_Kd``, relative to the OBJ.

        Relative paths are honoured and absolute ones are not: a package must not depend on where
        it happened to be built, and an absolute texture reference is the first thing a delivered
        package breaks on.
        """
        entry = self._fields_for(self._material_fields(obj_path), material)
        if entry is None:
            return None
        reference = entry.get("map_Kd")
        if reference is None:
            return None
        texture = (obj_path.parent / reference).resolve()
        if not texture.is_file():
            return None
        # A handful of pixels is a synthesised placeholder, not a map. Binding it renders the
        # object the placeholder's colour, which is how a desk came out black; a colour factor is
        # the honest source in that case.
        try:
            from PIL import Image as _Image

            with _Image.open(texture) as image:
                if min(image.size) < 64:
                    return None
        except Exception:  # noqa: BLE001 - unreadable is treated as absent
            return None
        return texture


def extrude_polygon(
    polygon: Iterable[Sequence[float]], *, base_z: float, height: float
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """A prism: the polygon at ``base_z`` and at ``base_z + height``, with side quads.

    Used for walls, and for the floor when ``height`` is the slab thickness. Assumes a simple
    polygon; a self-intersecting outline would produce a solid whose faces cross.
    """
    outline = [(float(point[0]), float(point[1])) for point in polygon]
    if len(outline) < 3:
        raise GeometryError("a prism needs at least 3 outline vertices")
    if height <= 0:
        raise GeometryError(f"prism height must be positive, got {height}")

    top_z = base_z + height
    points: list[tuple[float, float, float]] = []
    for x, y in outline:
        points.append((x, y, base_z))
    for x, y in outline:
        points.append((x, y, top_z))

    count = len(outline)
    faces: list[list[int]] = []
    # Bottom face, wound downwards so it faces away from the interior.
    faces.append(list(range(count))[::-1])
    # Top face.
    faces.append([count + index for index in range(count)])
    # Sides.
    for index in range(count):
        nxt = (index + 1) % count
        faces.append([index, nxt, count + nxt, count + index])
    return points, faces


def box_mesh(
    minimum: Sequence[float], maximum: Sequence[float]
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """An axis-aligned box, for the floor slab under a polygon's bounds."""
    min_x, min_y, min_z = (float(value) for value in minimum)
    max_x, max_y, max_z = (float(value) for value in maximum)
    if max_x <= min_x or max_y <= min_y or max_z <= min_z:
        raise GeometryError("box must have positive extent in every axis")
    points = [
        (min_x, min_y, min_z), (max_x, min_y, min_z), (max_x, max_y, min_z), (min_x, max_y, min_z),
        (min_x, min_y, max_z), (max_x, min_y, max_z), (max_x, max_y, max_z), (min_x, max_y, max_z),
    ]
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7],
        [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    return points, faces


def triangulated_floor(
    polygon: Iterable[Sequence[float]], *, z: float
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """A flat polygon triangulated at ``z``.

    A polygon plane rather than a box, so a notched room has a floor with a notch. Ray casting and
    renders both then agree with the validated geometry.

    Ear clipping, not a fan from vertex 0. A fan is correct only when every vertex can see vertex
    0, which a notch in the *top-right* corner happens to satisfy and no other corner does.
    Measured on a 5.0 x 4.0 room with a 1.5 x 1.4 notch, using this repository's own triangulator:
    the outline covers 17.9 m2 and the fan emits 17.9 m2 with the notch top-right but 20.0 m2 with
    it top-left -- 2.1 m2 of floor over the notch, exactly the notch's own area. The polygon is
    right in both cases; only the *mesh* is wrong, which is why no polygon-level check can see it
    and why this has to be ear clipping rather than a checked assumption.
    """
    outline = [(float(point[0]), float(point[1])) for point in polygon]
    if len(outline) < 3:
        raise GeometryError("a floor needs at least 3 outline vertices")
    points = [(x, y, z) for x, y in outline]
    try:
        shape = Polygon(tuple(outline))
        # Checked here rather than trusted from the caller. A self-intersecting outline has a
        # meaningless shoelace area and ear clipping still emits triangles for it: measured on a
        # bow tie whose signed area is 0.0, the mesh covered 4.0 m2. A floor that covers area the
        # room does not have is the failure this module exists to prevent, and the polygon-level
        # validator lives in the solver -- which a hand-written request never went through.
        validate_simple(shape)
        faces = [list(triangle) for triangle in triangulate_indices(shape)]
    except PolygonError as error:
        raise GeometryError(f"the floor outline is not a simple polygon: {error}") from error
    return points, faces


def y_up_to_z_up_matrix() -> tuple[tuple[float, ...], ...]:
    """The 4x4 row-major USD matrix that maps a Y-up asset frame onto Z-up.

    USD uses row vectors, so this is the transpose of the column-vector rotation about +X by
    +90 degrees, which sends +Y to +Z.
    """
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True, slots=True)
class WallOpening:
    """A hole in a wall, positioned along the wall rather than in world XY.

    ``offset_along_m`` is the distance from the wall's start vertex to the opening's centre, so a
    wall can be segmented without re-deriving which edge the opening belongs to. ``sill_m`` is the
    height of the opening's bottom: zero for a door, non-zero for a window, and it is what makes a
    window leave a wall below it.
    """

    opening_id: str
    offset_along_m: float
    width_m: float
    height_m: float
    sill_m: float = 0.0

    def __post_init__(self) -> None:
        if not self.opening_id:
            raise GeometryError("opening must have an id")
        if self.width_m <= 0 or self.height_m <= 0:
            raise GeometryError(f"opening {self.opening_id}: width and height must be positive")
        if self.sill_m < 0:
            raise GeometryError(f"opening {self.opening_id}: sill must not be negative")

    @property
    def start_m(self) -> float:
        return self.offset_along_m - self.width_m / 2.0

    @property
    def end_m(self) -> float:
        return self.offset_along_m + self.width_m / 2.0

    @property
    def top_m(self) -> float:
        return self.sill_m + self.height_m


@dataclass(frozen=True, slots=True)
class WallPanel:
    """One solid rectangle of wall: a horizontal span at a vertical span.

    A wall with a door becomes three panels (left, right, lintel above); a wall with a window
    becomes four (left, right, below, above). Building the wall this way rather than subtracting a
    solid keeps it exact and needs no boolean kernel.
    """

    start_m: float
    end_m: float
    bottom_m: float
    top_m: float

    @property
    def is_degenerate(self) -> bool:
        return self.end_m - self.start_m <= 1e-9 or self.top_m - self.bottom_m <= 1e-9


def wall_panels(
    wall_length_m: float,
    openings: Iterable[WallOpening],
    *,
    wall_height_m: float,
) -> tuple[WallPanel, ...]:
    """Split a wall into the solid panels left by its openings.

    Openings are laid out along the wall; the wall is then cut at every vertical band an opening
    introduces. Each opening contributes up to three horizontal cuts (its two sides) and up to two
    vertical bands (below its sill, above its head), which is what makes a door and a window differ
    without special-casing either.

    An opening that does not fit is an error rather than something to clamp: silently narrowing a
    door to fit would produce a wall that looks right and a doorway that is the wrong size, which is
    exactly the kind of quiet geometry lie the rest of this project is built to avoid.
    """
    if wall_length_m <= 0:
        raise GeometryError(f"wall length must be positive, got {wall_length_m}")
    if wall_height_m <= 0:
        raise GeometryError(f"wall height must be positive, got {wall_height_m}")

    ordered = sorted(openings, key=lambda item: item.start_m)
    for opening in ordered:
        if opening.start_m < -1e-9 or opening.end_m > wall_length_m + 1e-9:
            raise GeometryError(
                f"opening {opening.opening_id} spans [{opening.start_m:.3f}, {opening.end_m:.3f}] m "
                f"but the wall is only {wall_length_m:.3f} m long"
            )
        if opening.top_m > wall_height_m + 1e-9:
            raise GeometryError(
                f"opening {opening.opening_id} reaches {opening.top_m:.3f} m but the wall is "
                f"{wall_height_m:.3f} m tall"
            )

    # Horizontal spans: start with the whole wall, then remove each opening's footprint.
    spans: list[tuple[float, float]] = [(0.0, wall_length_m)]
    for opening in ordered:
        remaining: list[tuple[float, float]] = []
        for start, end in spans:
            if opening.end_m <= start + 1e-9 or opening.start_m >= end - 1e-9:
                remaining.append((start, end))  # no overlap
                continue
            if opening.start_m > start + 1e-9:
                remaining.append((start, opening.start_m))
            if opening.end_m < end - 1e-9:
                remaining.append((opening.end_m, end))
        spans = remaining

    panels: list[WallPanel] = [
        WallPanel(start, end, 0.0, wall_height_m) for start, end in spans if end - start > 1e-9
    ]

    # Vertical bands under and over each opening, within the opening's own horizontal span. Where
    # openings overlap horizontally the bands are unioned by walking the sorted boundaries, so two
    # overlapping windows cannot produce a panel that is emitted twice.
    boundaries = sorted({0.0, wall_height_m} | {o.sill_m for o in ordered} | {o.top_m for o in ordered})
    for start, end in ((o.start_m, o.end_m) for o in ordered):
        for low, high in zip(boundaries, boundaries[1:]):
            if high - low <= 1e-9:
                continue
            # A band is solid here when it lies inside the opening's span but outside its void.
            covered_by_void = any(
                o.sill_m <= low + 1e-9 and o.top_m >= high - 1e-9
                and o.start_m <= start + 1e-9 and o.end_m >= end - 1e-9
                for o in ordered
            )
            if not covered_by_void:
                panels.append(WallPanel(start, end, low, high))

    solid = [panel for panel in panels if not panel.is_degenerate]
    if not solid:
        raise GeometryError(
            "no wall remains after its openings; an opening cannot consume the whole wall"
        )
    return tuple(solid)


@dataclass(frozen=True, slots=True)
class WallFrame:
    """A wall edge as a frame: along it, up from the floor, and through it.

    ``across_m`` runs from the room-side face towards the far side, so 0 is the surface a person
    sees and ``thickness_m`` is the wall's other face -- and the inward normal is the same one
    ``build_wall`` offsets its panels along, so trim placed through this frame cannot end up on the
    wrong side of the wall.
    """

    origin_x: float
    origin_y: float
    along_x: float
    along_y: float
    across_x: float
    across_y: float
    length_m: float
    thickness_m: float

    def point(self, along_m: float, across_m: float, z_m: float) -> tuple[float, float, float]:
        return (
            self.origin_x + self.along_x * along_m + self.across_x * across_m,
            self.origin_y + self.along_y * along_m + self.across_y * across_m,
            z_m,
        )

    def box(
        self,
        *,
        along_m: tuple[float, float],
        across_m: tuple[float, float],
        z_m: tuple[float, float],
    ) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
        """An axis-aligned-in-this-frame box, with the winding ``box_mesh`` uses."""
        a0, a1 = float(along_m[0]), float(along_m[1])
        c0, c1 = float(across_m[0]), float(across_m[1])
        z0, z1 = float(z_m[0]), float(z_m[1])
        if a1 <= a0 or c1 <= c0 or z1 <= z0:
            raise GeometryError(
                f"a wall box needs positive extent in all three axes, got along={along_m} "
                f"across={across_m} z={z_m}"
            )
        points = [
            self.point(a0, c0, z0), self.point(a1, c0, z0),
            self.point(a1, c1, z0), self.point(a0, c1, z0),
            self.point(a0, c0, z1), self.point(a1, c0, z1),
            self.point(a1, c1, z1), self.point(a0, c1, z1),
        ]
        faces = [
            [0, 3, 2, 1], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ]
        return points, faces


def wall_frame(
    start: Sequence[float], end: Sequence[float], *, thickness_m: float
) -> WallFrame:
    """The frame of one wall edge.

    The inward normal is the direction rotated +90 degrees from start->end, which is what makes an
    outline walked counter-clockwise yield walls inside the room rather than outside it.
    """
    import math as _math

    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length = _math.hypot(dx, dy)
    if length <= 1e-9:
        raise GeometryError("wall edge has zero length")
    ux, uy = dx / length, dy / length
    return WallFrame(
        origin_x=ax, origin_y=ay, along_x=ux, along_y=uy,
        across_x=-uy, across_y=ux, length_m=length, thickness_m=float(thickness_m),
    )


def build_wall(
    start: Sequence[float],
    end: Sequence[float],
    *,
    thickness_m: float,
    base_z: float,
    height_m: float,
    openings: Iterable[WallOpening] = (),
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """A wall as one mesh, segmented around its openings.

    The wall is offset *inward* from the edge (to the left of the start->end direction), so an
    outline walked counter-clockwise yields walls inside the room rather than outside it.
    """
    import math as _math

    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length = _math.hypot(dx, dy)
    if length <= 1e-9:
        raise GeometryError("wall edge has zero length")
    ux, uy = dx / length, dy / length
    # Inward normal for counter-clockwise winding: rotate the direction by +90 degrees.
    nx, ny = -uy * thickness_m, ux * thickness_m

    panels = wall_panels(length, openings, wall_height_m=height_m)

    points: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for panel in panels:
        a0 = (ax + ux * panel.start_m, ay + uy * panel.start_m)
        a1 = (ax + ux * panel.end_m, ay + uy * panel.end_m)
        b0 = (a0[0] + nx, a0[1] + ny)
        b1 = (a1[0] + nx, a1[1] + ny)
        z0, z1 = base_z + panel.bottom_m, base_z + panel.top_m
        base = len(points)
        points.extend([
            (a0[0], a0[1], z0), (a1[0], a1[1], z0), (b1[0], b1[1], z0), (b0[0], b0[1], z0),
            (a0[0], a0[1], z1), (a1[0], a1[1], z1), (b1[0], b1[1], z1), (b0[0], b0[1], z1),
        ])
        faces.extend([
            [base + 0, base + 3, base + 2, base + 1],
            [base + 4, base + 5, base + 6, base + 7],
            [base + 0, base + 1, base + 5, base + 4],
            [base + 1, base + 2, base + 6, base + 5],
            [base + 2, base + 3, base + 7, base + 6],
            [base + 3, base + 0, base + 4, base + 7],
        ])
    return points, faces


def match_openings_to_edges(
    polygon_xy: Sequence[Sequence[float]],
    openings: Iterable[Sequence[float]],
    *,
    tolerance_m: float = 1e-6,
) -> dict[int, tuple[WallOpening, ...]]:
    """Assign each opening to the wall edge it lies on, as ``{edge_index: (opening, ...)}``.

    ``openings`` are ``(opening_id, centre_x, centre_y, width, height, sill)``. An opening whose
    centre is not on any edge raises rather than being dropped: dropping it yields a sealed wall and
    a scene that looks fine while the doorway the request asked for is simply absent, which is the
    failure this whole function exists to prevent.
    """
    outline = [(float(p[0]), float(p[1])) for p in polygon_xy]
    if len(outline) < 3:
        raise GeometryError("room outline needs at least 3 vertices")

    by_edge: dict[int, list[WallOpening]] = {}
    count = len(outline)
    for raw in openings:
        opening_id, cx, cy, width, height, sill = (
            str(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]), float(raw[4]), float(raw[5]),
        )
        for index in range(count):
            ax, ay = outline[index]
            bx, by = outline[(index + 1) % count]
            dx, dy = bx - ax, by - ay
            length_squared = dx * dx + dy * dy
            if length_squared <= 1e-18:
                continue
            t = ((cx - ax) * dx + (cy - ay) * dy) / length_squared
            if not (-1e-9 <= t <= 1.0 + 1e-9):
                continue
            px, py = ax + t * dx, ay + t * dy
            if math.hypot(px - cx, py - cy) > tolerance_m:
                continue
            by_edge.setdefault(index, []).append(
                WallOpening(
                    opening_id=opening_id,
                    offset_along_m=t * math.sqrt(length_squared),
                    width_m=width,
                    height_m=height,
                    sill_m=sill,
                )
            )
            break
        else:
            raise GeometryError(
                f"opening {opening_id} at ({cx:.3f}, {cy:.3f}) does not lie on any wall edge; "
                "an opening in the middle of the floor is not a doorway, and dropping it would "
                "leave a sealed wall where the request asked for one"
            )
    return {index: tuple(items) for index, items in by_edge.items()}
