# SPDX-License-Identifier: Apache-2.0
"""Writing a prepared asset's OBJ, MTL and maps, for both asset libraries to share.

Each library's preparation once wrote its own files, and both had the same two defects: a
multi-material asset came out as one material (the office notebook's pink cover, red body and pale
pages rendered as a single grey), and the base-colour factors came out in whatever encoding the
upstream tool happened to use. One writer means one answer to "what does a prepared asset carry",
and one place to test it.

The reader of this output is `workers/isaac/geometry.ObjMesh`, which groups faces by `usemtl`,
resolves `map_Kd` per material and reads each material's `Kd`. Everything written here is written
for that reader.
"""

from __future__ import annotations

from pathlib import Path

# Y-up (the GLB frame) to Z-up (the prepared frame), as a rotation about +X by +90 degrees:
# ``(x, y, z) -> (x, -z, y)``. Written as rows for the 4x4 trimesh wants.
#
# It is *not* the axis swap ``(x, z, y)`` this used to be. A swap maps the two axes onto each other
# but its determinant is -1, so it is a reflection: it mirrors the asset and reverses every face's
# winding. Measured on the office notebook's GLB, the swap took the mesh's signed volume from
# +3.7e-4 to -3.7e-4 m^3 -- the same shape, inside out -- and these are the meshes the renderer
# authors, so every HSSD asset in every delivered scene had its normals pointed inwards and its
# handedness flipped. The SimReady path has always sent +Y to +Z with a rotation
# (`workers/isaac/geometry.y_up_to_z_up_matrix`); this makes the two libraries arrive in one frame.
UP_FIX_ROWS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def sanitise(name: str) -> str:
    """A file-name-safe token for a material name, which the libraries spell inconsistently."""
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in name)


def material_name(geometry, index: int) -> str:
    """The name one geometry's material is written under.

    The asset's own material name when it has one, because that is what a reader of the OBJ sees and
    what the renderer records; a numbered fallback when it does not, because a material with no name
    is still a material and an unnamed `usemtl` would bind the wrong one.
    """
    material = getattr(getattr(geometry, "visual", None), "material", None)
    name = getattr(material, "name", None)
    return str(name) if name else f"material_{index}"


def write_texture(directory: Path, name: str, material) -> str | None:
    """Write a material's base-colour image into the prepared directory, or None if it has none.

    Only an image the material actually carries: trimesh synthesises a tiny placeholder from a
    colour factor when asked for a texture it does not have, and a 16x8 near-black PNG bound as if
    it were the asset's map is how a desk came out black.
    """
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if image is None:
        return None
    file_name = f"{sanitise(name)}.png"
    try:
        from PIL import Image as _Image

        if hasattr(image, "save"):
            image.save(directory / file_name)
        else:  # a numpy array, as trimesh hands some materials back
            import numpy as np

            _Image.fromarray(np.asarray(image)).save(directory / file_name)
    except Exception:  # noqa: BLE001 - an unwritable map must not lose the geometry
        return None
    return file_name


def material_lines(name: str, material, image_name: str | None) -> list[str]:
    """One MTL entry for a material.

    ``Kd`` is the material's own base-colour factor in the *linear* space the GLB states it in and
    ``UsdPreviewSurface.diffuseColor`` expects. trimesh's exporter wrote the sRGB-encoded bytes
    instead, which this renderer then took as linear -- so every constant-coloured asset rendered
    lighter than the asset says it is. trimesh hands the factor back in the byte encoding, so the
    range is normalised here; a material that states no factor at all gets no `Kd` line, and the
    renderer then reports it as having no colour rather than reading a fabricated one.
    """
    lines = [f"newmtl {name}", "Ka 0.40000000 0.40000000 0.40000000"]
    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        # A material loaded from an MTL -- trimesh's `SimpleMaterial`, which is what an OBJ with a
        # sibling .mtl loads as -- states its colour as `diffuse`, in the same byte encoding.
        factor = getattr(material, "diffuse", None)
    if factor is not None:
        channels = [float(value) for value in list(factor)[:3]]
        if len(channels) == 3 and any(channel > 1.0 for channel in channels):
            channels = [channel / 255.0 for channel in channels]
        if len(channels) == 3 and all(0.0 <= channel <= 1.0 for channel in channels):
            lines.append("Kd " + " ".join(f"{channel:.8f}" for channel in channels))
    lines.extend(["Ks 0.40000000 0.40000000 0.40000000", "Ns 1.00000000"])
    if image_name is not None:
        lines.append(f"map_Kd {image_name}")
    return lines


def write_scene_obj(scene, obj_path: Path) -> None:
    """Write a scene as one OBJ with per-material groups, its MTL, and every material's map.

    Written here rather than by trimesh's exporter because that exporter is not self-consistent
    when one geometry carries UVs and another does not. Measured on the office ceiling light: its
    OBJ declares 6028 `vt` lines while the second geometry's faces reference texture indices up to
    14427 -- the vertex indices, against a texture array that is not the same length. A reader can
    only drop those UVs (this project's did) or read out of range, and the asset lost the 588 KB map
    its GLB ships. Writing the file here also keeps the indexing local per geometry, so a group
    without UVs says so by having none rather than by pointing somewhere arbitrary.

    Nothing is inferred: a geometry with UVs writes `f v/vt`, one without writes `f v`, and a
    material's colour is written only when the asset states one.

    Each geometry is written through its place in the scene graph, because that is where the
    rotations and translations live: `Scene.apply_transform` moves the *nodes*, and the geometry
    objects keep their own coordinates. A writer that read `scene.geometry` directly therefore wrote
    the asset in its original orientation -- measured on the office plant, a 0.90 m tall asset whose
    prepared mesh then had its 0.90 on Y and its base 0.16 m below the floor.
    """
    import trimesh

    header = ["# pvscene prepared asset", f"mtllib {obj_path.stem}.mtl"]
    vertices: list[str] = []
    faces: list[str] = []
    materials: list[str] = []
    written: set[str] = set()
    vertex_offset = 0
    uv_offset = 0
    nodes = list(getattr(getattr(scene, "graph", None), "nodes_geometry", ()))
    entries = (
        [(node, *scene.graph[node]) for node in nodes] if nodes else
        [(None, None, name) for name in scene.geometry]
    )
    for index, (node, transform, geometry_name) in enumerate(entries):
        geometry = scene.geometry[geometry_name]
        name = material_name(geometry, index)
        material = getattr(getattr(geometry, "visual", None), "material", None)
        image_name = None
        if material is not None:
            image_name = write_texture(obj_path.parent, name, material)
        if name not in written:
            written.add(name)
            materials.extend(material_lines(name, material, image_name))
            materials.append("")
        points = getattr(geometry, "vertices", ())
        if transform is not None:
            points = trimesh.transform_points(points, transform)
        uvs = getattr(getattr(geometry, "visual", None), "uv", None)
        # Coordinates are written only for a geometry that has one per vertex. They are numbered in
        # the texture array, not in the vertex array: numbering them by vertex is the same defect
        # trimesh's exporter has -- a group after an untextured one then references coordinates the
        # file never declares.
        textured = uvs is not None and len(uvs) == len(points) and len(points) > 0
        for position in points:
            x, y, z = (float(value) for value in position[:3])
            vertices.append(f"v {x:.8f} {y:.8f} {z:.8f}")
        if textured:
            for uv in uvs:
                vertices.append(f"vt {float(uv[0]):.8f} {float(uv[1]):.8f}")
        obj_faces = getattr(geometry, "faces", ())
        if obj_faces is not None and len(obj_faces):
            faces.append(f"o {sanitise(name)}")
            faces.append(f"usemtl {name}")
            for face in obj_faces:
                if textured:
                    refs = [
                        f"{vertex_offset + int(vertex) + 1}/{uv_offset + int(vertex) + 1}"
                        for vertex in face
                    ]
                else:
                    refs = [f"{vertex_offset + int(vertex) + 1}" for vertex in face]
                if len(refs) >= 3:
                    faces.append("f " + " ".join(refs))
        vertex_offset += len(points)
        if textured:
            uv_offset += len(uvs)
    obj_path.write_text("\n".join(header + vertices + faces) + "\n")
    obj_path.with_suffix(".mtl").write_text("\n".join(materials) + "\n")
