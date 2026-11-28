#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a scene in an installed Isaac environment.

    python -m rfw_w.scenes.workers.isaac.render_worker --request request.json --output-dir out

Reports RGB, metric camera-z depth and instance IDs. Failed runs write a report
and exit nonzero; absence of the report is also failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

from rfw_w.scenes.workers.isaac.geometry import (  # noqa: E402
    GeometryError,
    build_wall,
    extrude_polygon,
    match_openings_to_edges,
    load_obj,
    triangulated_floor,
    wall_frame,
    y_up_to_z_up_matrix,
)
from rfw_w.scenes.workers.isaac.protocol import WorkerReport, WorkerRequest  # noqa: E402

DEPTH_CONVENTION = "camera_z_meters"
WALL_THICKNESS_M = 0.1

# Opening trim. These are a domestic proportion, not a measurement: a 6 cm casing on a 0.9 m door
# is about what the real thing has, a frame stands 1.5 cm proud of each wall face, and a window
# sill reaches 4.5 cm into the room. Named so that changing the look is one edit, and no gate reads
# them -- the openings the shell computed are unchanged by any of this.
FRAME_FACE_M = 0.06
FRAME_PROUD_M = 0.015
SILL_DEPTH_M = 0.045
GLASS_THICKNESS_M = 0.012


def _author_mesh(stage, path: str, points, faces, *, transform_matrix=None, mesh_data=None):
    """Create a UsdGeom.Mesh, with texture coordinates when the source carried them.

    The UVs are authored as ``primvars:st`` because without them a bound texture has nothing to
    sample: the material arrives, the map is readable, and the surface still renders flat. Both
    indexings occur in these libraries -- a UV per vertex, and a UV per face-corner -- so whichever
    the file used decides the interpolation, rather than assuming one and silently smearing the map.
    """
    from pxr import Gf, Sdf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if transform_matrix is not None:
        matrix = Gf.Matrix4d(*[value for row in transform_matrix for value in row])
        UsdGeom.Xformable(mesh).AddTransformOp().Set(matrix)

    if mesh_data is not None and mesh_data.is_textured:
        per_vertex = all(
            len(face_uvs) == len(face) and all(a == b for a, b in zip(face_uvs, face))
            for face, face_uvs in zip(mesh_data.faces, mesh_data.face_texcoords)
        )
        if per_vertex:
            values = [Gf.Vec2f(u, v) for u, v in mesh_data.texcoords]
            interpolation = UsdGeom.Tokens.vertex
        else:
            # One value per face corner, in face order. A face whose material carries no map has no
            # coordinates, and it still needs an entry: a faceVarying primvar shorter than the
            # corner list is a primvar USD cannot interpret, and the groups without maps are bound
            # to colours that never sample it. Padding rather than dropping the UVs is what keeps
            # the textured groups' maps -- see `ObjMesh.group_has_uvs`.
            values = [
                Gf.Vec2f(*mesh_data.texcoords[uv]) if uv < len(mesh_data.texcoords) else Gf.Vec2f(0.0, 0.0)
                for face, face_uvs in zip(mesh_data.faces, mesh_data.face_texcoords)
                for uv in (face_uvs if len(face_uvs) == len(face) else [0] * len(face))
            ]
            interpolation = UsdGeom.Tokens.faceVarying
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation
        )
        primvar.Set(values)
    return mesh


def _make_textured_material(stage, name: str, texture: Path, *, roughness: float = 0.55):
    """A UsdPreviewSurface whose base colour is a texture on disk.

    The texture is referenced by a path relative to the stage's root layer, so a package stays
    self-contained: an absolute reference into the asset library is what makes a delivered scene
    open correctly only on the machine that built it.
    """
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Materials/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    reader = UsdShade.Shader.Define(stage, f"/World/Materials/{name}/Texture")
    reader.CreateIdAttr("UsdUVTexture")
    reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    reader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _compose_transform(translation, rotation_wxyz, *, y_up: bool):
    """Build the USD row-major matrix for an instance.

    USD is row-vector, so the stored matrix is the transpose of the column-vector transform. The
    Y-up correction is composed *outside* the instance's own rotation, so an asset that was
    measured Y-up is stood upright before its placement is applied.
    """
    from pxr import Gf

    w, x, y, z = rotation_wxyz
    translation_matrix = Gf.Matrix4d(1.0)
    translation_matrix.SetTranslate(Gf.Vec3d(*translation))
    rotation_matrix = Gf.Matrix4d(
        1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 0,
        2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x), 0,
        2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y), 0,
        0, 0, 0, 1,
    )
    # USD multiplies row vectors on the left (p' = p @ M), so in a product the *leftmost* matrix
    # is applied first. The chain is therefore correction, then rotation, then translation --
    # the reverse of the column-vector order, and writing `T * R` (as the column-vector habit
    # suggests) would translate before rotating and misplace every instance.
    composed = rotation_matrix * translation_matrix
    if y_up:
        correction = Gf.Matrix4d(*[value for row in y_up_to_z_up_matrix() for value in row])
        composed = correction * composed
    return composed


def _author_room(stage, request: WorkerRequest) -> int:
    """Floor, walls and ceiling from the room's real polygon. Returns openings cut."""
    from pxr import Sdf, UsdGeom, UsdPhysics

    room = request.room
    if getattr(room, "shell_uri", None):
        from rfw_w.scenes.workers.isaac.shell_reference import author_shell
        openings = author_shell(stage, room.shell_uri)
        override = getattr(request, "shell_material_override", False)
        if override:
            # See WorkerRequest.shell_material_override for why this is opt-in. `True` is one
            # material for every shell surface, matching what the bundle's MTL-less OBJ gives the
            # other four backends. A role list instead binds each named role from its own
            # `room_materials` field, using the provider's factory naming in the prim path -- the
            # only thing in the USD that says which prim is a floor.
            #
            # GeomSubsets are rebound as well as Meshes, and they are the ones that matter. A
            # subset's binding beats its mesh's, so rebinding only the Mesh prims changed nothing
            # visible: measured 2026-09-25, this Infinigen office shell puts all four walls in ONE
            # mesh carrying two subsets, and the dark half of the frame was the subset bound to a
            # Blender `Image_Texture`/`UV_Map` node graph that RTX cannot evaluate and renders
            # black. That is also why no light reached it -- there was nothing wrong with the light.
            from rfw_w.scenes.workers.isaac.shell_reference import shell_role
            per_role = None if override is True else tuple(override)
            materials = {}
            if per_role is None:
                materials["*"] = _make_material(stage, "shell_override",
                                                surface=request.room_materials.wall)
            else:
                for role in per_role:
                    materials[role] = _make_material(
                        stage, f"shell_override_{role}",
                        surface=getattr(request.room_materials, role))
            rebound = {role: 0 for role in materials}
            shell_root = Sdf.Path("/World/Room")
            for prim in stage.Traverse():
                if not prim.GetPath().HasPrefix(shell_root):
                    continue
                if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):
                    continue
                if per_role is None:
                    _bind(prim, materials["*"])
                    rebound["*"] += 1
                    continue
                # A GeomSubset's own path says nothing; its mesh's does.
                path = str(prim.GetPath())
                role = shell_role(path)
                if role == "shell_other" and prim.IsA(UsdGeom.Subset):
                    role = shell_role(str(prim.GetPath().GetParentPath()))
                # Infinigen names the outer shell `exterior`; the request calls that surface the
                # facade, which is what shows through a window.
                role = "facade" if role == "exterior" else role
                if role in materials:
                    _bind(prim, materials[role])
                    rebound[role] += 1
            empty = sorted(role for role, count in rebound.items() if count == 0)
            if empty:
                raise RuntimeError(
                    f"SHELL_MATERIAL_OVERRIDE_BOUND_NOTHING({room.shell_uri}: {empty})")
        return openings
    # These surface parameters belong to the procedural fallback only.
    surfaces = request.room_materials
    wall_material = _make_material(stage, "wall", surface=surfaces.wall)
    floor_material = _make_material(stage, "floor", surface=surfaces.floor)
    ceiling_material = _make_material(stage, "ceiling", surface=surfaces.ceiling)
    frame_material = _make_material(stage, "opening_frame", surface=surfaces.frame)
    glass_material = _glass_material(stage, surfaces)
    min_x = min(x for x, _ in room.polygon_xy)
    max_x = max(x for x, _ in room.polygon_xy)
    min_y = min(y for _, y in room.polygon_xy)
    max_y = max(y for _, y in room.polygon_xy)

    # Floor as a triangle fan of the actual outline, so a notched room has a notched floor,
    # sitting on a thin slab that gives it collision volume.
    points, faces = triangulated_floor(room.polygon_xy, z=room.floor_z_m)
    floor = _author_mesh(stage, "/World/Room/floor_visual", points, faces)
    _bind(floor.GetPrim(), floor_material)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(floor.GetPrim())
    # Exact triangles rather than a hull: a hull would bridge the notch and let objects rest on
    # floor that is not there.
    mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)

    # Walls, segmented around the openings so a doorway is a hole in the geometry rather than a
    # line in a manifest. Before this, every wall was authored solid: the shell computed a door and
    # a window, nothing passed them here, and the render showed a sealed box while a spatial gate
    # that only read the shell data kept passing.
    by_edge = match_openings_to_edges(
        room.polygon_xy, [opening.as_tuple() for opening in room.openings]
    )
    for index, (start, end) in enumerate(_edges(room.polygon_xy)):
        openings = by_edge.get(index, ())
        points, faces = build_wall(
            start, end,
            thickness_m=WALL_THICKNESS_M,
            base_z=room.floor_z_m,
            height_m=room.height_m,
            openings=openings,
        )
        prim = _author_mesh(stage, f"/World/Room/wall_{index:02d}", points, faces)
        _bind(prim.GetPrim(), wall_material)
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        _author_opening_trim(
            stage, wall_frame(start, end, thickness_m=WALL_THICKNESS_M),
            openings, floor_z_m=room.floor_z_m,
            frame_material=frame_material, glass_material=glass_material,
        )

    # A ceiling and a floor slab close the shell; without them a render from outside looks into the
    # room and the "walls" read as disconnected slabs.
    ceiling_points, ceiling_faces = triangulated_floor(
        room.polygon_xy, z=room.floor_z_m + room.height_m
    )
    ceiling = _author_mesh(stage, "/World/Room/ceiling", ceiling_points, ceiling_faces)
    _bind(ceiling.GetPrim(), ceiling_material)
    UsdGeom.Imageable(ceiling).MakeInvisible()  # visible only from above, and it darkens the room
    return sum(len(items) for items in by_edge.values())


def _glass_material(stage, surfaces):
    """The pane material for the requested glass mode.

    ``glass_mode`` is part of the room's material block and reached no geometry at all before this:
    the request carried it, the shell ignored it, and three modes were indistinguishable because
    none of them was ever bound. Binding it here is what makes the field mean something.
    """
    if surfaces.glass_mode == "opaque_tint":
        return _make_material(stage, "opening_glass", surface=surfaces.glass)
    if surfaces.glass_mode == "emissive_sky":
        # A pane that lights the room rather than showing what is behind it: the sky's own colour,
        # emitted. For a window on a facade the renderer cannot see, this is the honest way to say
        # "daylight".
        return _make_material(
            stage, "opening_glass", surface=surfaces.glass, emissive=surfaces.facade.colour,
        )
    return _make_material(
        stage, "opening_glass", surface=surfaces.glass, opacity=surfaces.glass_opacity,
    )


def _author_opening_trim(
    stage, frame, openings, *, floor_z_m: float, frame_material, glass_material
) -> None:
    """Frames, sills and glass for the openings in one wall.

    An earlier shell cut a genuine hole and stopped there, so a window was a void with a coloured
    plane somewhere behind it: no reveal, no sill, nothing at the edge to say how thick the wall is
    or where the glass sits. That reads as a hole in a set, not as a window, and it was the whole of
    the difference a reader could see between this shell and a room generator that models joinery.

    Sizes are a domestic proportion rather than a measurement -- a 6 cm casing on a 0.9 m door is
    roughly what the real thing has, and the constants are named so that changing them is one edit.
    Nothing here is a gate: the frames are geometry, and the openings themselves are still the ones
    the shell computed.

    Prims are named after the *opening*, not after its position in a list: two openings on one wall
    would otherwise author the same four paths twice, and USD would redefine the prim rather than
    complain -- leaving one frame in the stage and two holes expecting one.
    """
    from pxr import UsdPhysics

    for opening in openings:
        stem = f"/World/Room/opening_{_material_slug(opening.opening_id)}"
        z0, z1 = floor_z_m + opening.sill_m, floor_z_m + opening.top_m
        if z1 - z0 <= 2 * FRAME_FACE_M or opening.width_m <= 2 * FRAME_FACE_M:
            # An opening too small to case is left as a plain hole rather than filled with a frame
            # that has no middle. The shell refuses to plan one, but a hand-written request can.
            continue
        across = (-FRAME_PROUD_M, frame.thickness_m + FRAME_PROUD_M)
        pieces: list[tuple[str, tuple[list, list]]] = [
            ("jamb_near", frame.box(
                along_m=(opening.start_m - FRAME_FACE_M, opening.start_m), across_m=across,
                z_m=(z0, z1),
            )),
            ("jamb_far", frame.box(
                along_m=(opening.end_m, opening.end_m + FRAME_FACE_M), across_m=across,
                z_m=(z0, z1),
            )),
            ("head", frame.box(
                along_m=(opening.start_m - FRAME_FACE_M, opening.end_m + FRAME_FACE_M),
                across_m=across, z_m=(z1, z1 + FRAME_FACE_M),
            )),
        ]
        if opening.sill_m > 0.0:
            # A window: a sill across the bottom, reaching into the room, and a pane. A door keeps
            # its threshold clear -- a frame across a doorway is a step a robot would have to climb
            # and a gate that measured the opening would no longer be measuring what it says.
            pieces.append((
                "sill", frame.box(
                    along_m=(opening.start_m - FRAME_FACE_M, opening.end_m + FRAME_FACE_M),
                    across_m=(-SILL_DEPTH_M, frame.thickness_m + FRAME_PROUD_M),
                    z_m=(z0 - FRAME_FACE_M, z0),
                )
            ))
        for name, (points, faces) in pieces:
            prim = _author_mesh(stage, f"{stem}_{name}", points, faces)
            _bind(prim.GetPrim(), frame_material)
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        if opening.sill_m <= 0.0:
            continue
        middle = frame.thickness_m / 2.0
        points, faces = frame.box(
            along_m=(opening.start_m, opening.end_m),
            across_m=(middle - GLASS_THICKNESS_M / 2, middle + GLASS_THICKNESS_M / 2),
            z_m=(z0, z1),
        )
        pane = _author_mesh(stage, f"{stem}_glass", points, faces)
        _bind(pane.GetPrim(), glass_material)
        # Not a collider: a pane that stops the robot is a wall, and the openings the shell
        # computed are what navigation is planned against.


def _edges(polygon) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    count = len(polygon)
    return [(polygon[index], polygon[(index + 1) % count]) for index in range(count)]


def _apply_semantic_label(prim, entity_id: str) -> None:
    """Tag a prim so the instance-segmentation annotator can name it.

    Without a semantic label the annotator still returns ids, but nothing maps them to entity ids,
    and an unmapped id is exactly the "let the critic guess which object is which" the architecture
    forbids.
    """
    from pxr import Semantics

    api = Semantics.SemanticsAPI.Apply(prim, "pvscene")
    api.CreateSemanticTypeAttr().Set("class")
    api.CreateSemanticDataAttr().Set(entity_id)


def _material_slug(name: str) -> str:
    """A USD-safe token for one of the library's material names.

    The names carry spaces, dots and slashes ("Metal___Chrome_2017", "5/8 in"), and a prim name
    that is not an identifier makes the stage unreadable rather than rendering wrongly -- so this
    is a translation with a stated alphabet, not a guess at what the name meant.
    """
    slug = "".join(
        character if (character.isascii() and (character.isalnum() or character == "_")) else "_"
        for character in name
    ).strip("_") or "unnamed"
    return slug if not slug[0].isdigit() else f"material_{slug}"


def _asset_material(
    stage, obj, visual_path: Path, *, out_dir: Path, prim_name: str, texture_prefix: str,
    material_name: str, has_uvs: bool,
) -> tuple[object | None, str]:
    """The material one group of faces wears, and a record of what it turned out to be.

    ``material_name`` is the ``usemtl`` the faces named, empty when they named none; an empty name
    resolves to the first material the file declares, which is what a single-material export means.

    ``has_uvs`` is whether *this group's* faces carry coordinates, which is what a map needs; a
    group whose faces have none is coloured from its factor instead, however many maps its sibling
    groups ship.

    ``None`` comes back only when the asset states neither a map nor a colour factor, which is the
    caller's cue to use a generated display colour and say so.
    """
    requested = material_name or None
    # The asset's own texture when it has one. A delivery that shows flat colour where the library
    # ships a 2048x2048 map is not a faithful render, and a critic scoring material coherence is
    # scoring a colour that was chosen by this script rather than by the asset.
    texture = obj.diffuse_texture(visual_path, requested) if has_uvs else None
    if texture is not None:
        # Copied beside the deliverable and referenced relatively: an absolute path into the asset
        # library is what makes a package open correctly only on the machine that built it.
        target = out_dir / "textures" / f"{texture_prefix}{texture.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            import shutil as _shutil

            _shutil.copy2(texture, target)
        return (
            _make_textured_material(stage, prim_name, target),
            f"texture:{target.name}",
        )
    # No map: use the material's own colour factor if it has one, and only fall back to a generated
    # display colour when the asset states nothing. These assets are coloured by a factor rather
    # than a map, so showing a colour this script chose instead hides what the asset specifies --
    # and in a multi-material asset that factor is the only thing telling the parts apart.
    colour = obj.material_colour(visual_path, requested) if obj.material_library else None
    if colour is None:
        return None, "fallback_display_colour"
    return (
        _make_material(stage, prim_name, colour=colour, roughness=0.6),
        f"colour:{tuple(round(v, 3) for v in colour)}",
    )


def _author_instance(stage, instance, *, index: int, out_dir: Path, asset_materials: dict):
    """Author one retrieved mesh, with its materials, collision geometry and rigid-body state."""
    from pxr import UsdShade, UsdGeom, UsdPhysics

    visual_path = Path(instance.visual_uri)
    if not visual_path.is_file():
        raise GeometryError(f"{instance.instance_id}: visual mesh not found at {instance.visual_uri}")

    obj = load_obj(visual_path)
    prim_path = f"/World/Instances/{instance.instance_id}"
    y_up = instance.normalisation == "y_up_to_z_up"
    matrix = _compose_transform(instance.translation, instance.rotation_wxyz, y_up=y_up)
    mesh = _author_mesh(
        stage, prim_path, obj.points, obj.faces, transform_matrix=matrix, mesh_data=obj
    )
    prim = mesh.GetPrim()

    # One material per group of faces. Binding `materials[0]` to the whole mesh is what made a
    # three-material asset wear one of its parts everywhere -- the office notebook's pink cover,
    # red body and pale pages arrived as a single grey before this, and the chair's fabric was
    # stretched over its chrome and its black frame. A single-material asset is one group and takes
    # exactly the path it took before, down to the display-colour fallback.
    groups = obj.material_groups()
    if len(groups) == 1:
        material, record = _asset_material(
            stage, obj, visual_path, out_dir=out_dir, prim_name=f"asset_{instance.instance_id}",
            texture_prefix=f"{instance.instance_id}_", material_name=groups[0][0],
            has_uvs=obj.group_has_uvs(groups[0][0]),
        )
        if material is not None:
            _bind(prim, material)
        else:
            mesh.CreateDisplayColorAttr([_display_color(index)])
        asset_materials[instance.instance_id] = record
    else:
        records: list[str] = []
        for name, face_indices in groups:
            slug = _material_slug(name)
            material, record = _asset_material(
                stage, obj, visual_path, out_dir=out_dir,
                prim_name=f"asset_{instance.instance_id}_{slug}",
                texture_prefix=f"{instance.instance_id}_{slug}_", material_name=name,
                has_uvs=obj.group_has_uvs(name),
            )
            if material is None:
                material = _make_material(
                    stage, f"asset_{instance.instance_id}_{slug}",
                    colour=_display_color(index), roughness=0.7,
                )
            # A GeomSubset per material, in the `materialBind` family that USD reserves for exactly
            # this. The groups partition the faces, which is the completeness the family requires:
            # a mesh with a materialBind subset and an uncovered face renders that face's material
            # as the prim's, and the prim has none.
            subset = UsdGeom.Subset.CreateGeomSubset(
                mesh, f"materialBind_{slug}", UsdGeom.Tokens.face, face_indices,
                familyName=UsdShade.Tokens.materialBind,
            )
            _bind(subset.GetPrim(), material)
            records.append(f"{name or 'unnamed'}={record}")
        asset_materials[instance.instance_id] = "; ".join(records)

    if instance.collision_uri:
        collision_path = Path(instance.collision_uri)
        if collision_path.is_file():
            try:
                collision_obj = load_obj(collision_path)
                # Collision geometry is never textured: it is invisible and exists to be simulated.
                child = _author_mesh(
                    stage, f"{prim_path}/collision", collision_obj.points, collision_obj.faces
                )
                UsdPhysics.CollisionAPI.Apply(child.GetPrim())
                # A collision proxy is not appearance. A convex hull of a desk with legs is a solid
                # block, so leaving it visible renders that block instead of the desk -- the images
                # looked plausible, the depth was valid, and every check passed while the render
                # showed a collider rather than the asset. Hulls are physics-only.
                UsdGeom.Imageable(child).MakeInvisible()
                hull = UsdPhysics.MeshCollisionAPI.Apply(child.GetPrim())
                # The library's collision meshes are convex exports, so a hull approximation
                # matches what they already are; an exact triangle mesh would be slower and no
                # more correct for them.
                hull.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            except GeometryError:
                # A missing collision mesh is reported by the caller as a degraded instance rather
                # than aborting the render, so a visual-only check is still possible.
                pass

    if instance.dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(float(instance.mass_kg))
    return prim


def _display_color(index: int):
    from pxr import Gf

    # Deterministic, well separated hues; enough contrast for the non-blank check.
    return Gf.Vec3f(*((0.35 + (index * 0.37) % 0.5), (0.30 + (index * 0.61) % 0.5), (0.40 + (index * 0.13) % 0.5)))


def _capture(
    camera_request,
    annotators: dict,
    out_dir: Path,
    entity_by_instance: dict[str, str],
) -> "CaptureReport":
    """Render one frame and write every channel, with the statistics that audit it."""
    import numpy as np
    from PIL import Image

    from rfw_w.scenes.workers.isaac.protocol import CaptureReport
    import omni.replicator.core as rep

    rep.orchestrator.step(rt_subframes=16, pause_timeline=True)

    width, height = camera_request.resolution
    rgb = np.asarray(annotators["rgb"].get_data())
    if rgb.shape[:2] != (height, width):
        raise GeometryError(
            f"{camera_request.camera_id}: renderer returned {rgb.shape[:2]}, expected {(height, width)}"
        )
    Image.fromarray(rgb[..., :3].astype("uint8")).save(out_dir / f"{camera_request.camera_id}.png")

    color_std = float(rgb[..., :3].astype("float").std(axis=(0, 1)).mean())

    depth_path = None
    valid_fraction = 0.0
    depth_min = depth_max = None
    if camera_request.save_depth:
        depth = np.asarray(annotators["depth"].get_data(), dtype=np.float32)
        # Float metres, camera-z -- not the eight-bit colourised image v2 wrote. Downstream checks
        # compare this against heights and thresholds, so losing the values would lose the check.
        np.save(out_dir / f"{camera_request.camera_id}_depth_m.npy", depth)
        depth_path = f"{camera_request.camera_id}_depth_m.npy"
        valid = np.isfinite(depth) & (depth > 0)
        valid_fraction = float(valid.mean())
        if valid.any():
            depth_min = float(depth[valid].min())
            depth_max = float(depth[valid].max())

    instance_path = None
    visible: tuple[int, ...] = ()
    payload_kind = ""
    if camera_request.save_instance_ids:
        payload = annotators["instance"].get_data()
        instance_ids, payload_kind = _extract_instance_ids(payload, np)
        # The first priming frame may not carry idToLabels yet. Refresh after the actual 16-subframe
        # capture and before constructing CaptureReport; doing it in the caller afterwards updates
        # only the global map and leaves this camera's immutable map empty.
        labels, _ = _id_map(annotators["instance"])
        annotators["_labels"] = labels
        np.save(out_dir / f"{camera_request.camera_id}_instance_ids.npy", instance_ids)
        instance_path = f"{camera_request.camera_id}_instance_ids.npy"
        visible = tuple(int(value) for value in np.unique(instance_ids))

    target_visible = None
    if camera_request.target_instance_ids:
        # Visibility is decided from the renderer's own idToLabels, and is `None` (unknown) when
        # no mapping is available. Reporting `False` from an assumption would be a confident
        # wrong answer, and reporting `True` from one would be worse.
        source = annotators.get("_labels") or {}
        if source:
            # The annotator's keys are strings and the id array is integers, so the two are
            # compared in one space. Comparing across them silently matched nothing and reported
            # every target as invisible while its pixels were plainly in the frame.
            visible_keys = {str(value) for value in visible}
            visible_names = {name for index, name in source.items() if index in visible_keys}
            target_visible = bool(set(camera_request.target_instance_ids) & visible_names)

    return CaptureReport(
        camera_id=camera_request.camera_id,
        rgb_path=f"{camera_request.camera_id}.png",
        depth_path=depth_path,
        instance_id_path=instance_path,
        resolution=(width, height),
        depth_convention=DEPTH_CONVENTION,
        depth_valid_fraction=valid_fraction,
        depth_min_m=depth_min,
        depth_max_m=depth_max,
        color_std=color_std,
        visible_instance_ids=visible,
        instance_id_map=dict(annotators.get("_labels") or {}),
        target_instance_ids=tuple(camera_request.target_instance_ids),
        target_visible=target_visible,
        # Same threshold shape as the existing runtime probe, so "non-blank" means the same thing
        # across the project.
        nonblank=color_std > 5.0 and (not camera_request.save_depth or valid_fraction > 0.1),
        # A saved channel with no entity in it: legitimately empty, and said so (SPEC-015 R3/R4).
        contains_nothing=bool(instance_path and not (annotators.get("_labels") or {}) and visible),
    )


def _extract_instance_ids(payload, np):
    """Pull the id array out of whatever the annotator returned, and say what that was.

    The payload's shape is the least certain part of this integration, so the observed kind is
    returned and reported rather than the code asserting a structure nobody has seen. Both known
    shapes are handled; an unrecognised one raises with what it saw.
    """
    if isinstance(payload, dict):
        data = payload.get("data")
        if data is None:
            raise GeometryError(
                f"instance annotator returned a mapping without 'data': keys {sorted(payload)}"
            )
        kind = "dict_with_data"
        array = np.asarray(data)
        if array.ndim == 3:
            # Colourised: three channels of the same id. Take one.
            array = array[..., 0]
        return array.astype("uint32"), kind
    array = np.asarray(payload)
    if array.ndim == 3:
        return array[..., 0].astype("uint32"), "ndarray_hwc"
    return array.astype("uint32"), "ndarray"


# Labels the annotator emits for things that are not instances.
NON_INSTANCE_LABELS = frozenset({"BACKGROUND", "UNLABELLED", "UNLABELLED_OBJECT"})


def _instance_id_from_label(value: object) -> str | None:
    """The instance id a label denotes, or None if it denotes no instance.

    Measured against a live run: `idToLabels` maps ids to prim paths, e.g.
    ``{"0": "BACKGROUND", "1": "UNLABELLED", "2": "/World/Instances/cup_0"}``. It does *not*
    return the semantic `class` string, so a reader looking for one finds nothing and concludes no
    mapping exists. Background and unlabelled entries are not instances and are dropped rather
    than mapped to something.
    """
    if isinstance(value, dict):
        for key in ("class", "instance"):
            if key in value:
                return _instance_id_from_label(value[key])
        return None
    text = str(value)
    if not text or text in NON_INSTANCE_LABELS:
        return None
    # A prim path: the instance id is its last component under the instances scope.
    if "/" in text:
        parts = [part for part in text.split("/") if part]
        if not parts:
            return None
        return parts[-1]
    return text


def _id_map(annotator) -> tuple[dict[str, str], str]:
    """The renderer's id-to-instance map, read from the annotator after a frame has been drawn.

    Must be called *after* at least one ``orchestrator.step``: before the first render the
    annotator has no data, and reading it early yields an empty map that looks like "the renderer
    provides no labels" rather than "the labels were read too soon".
    """
    try:
        payload = annotator.get_data()
    except Exception:  # noqa: BLE001 - metadata is optional; absence is reported, not fatal
        return {}, "unavailable"
    if isinstance(payload, dict):
        info_payload = payload.get("info")
    else:
        info_payload = getattr(annotator, "get_metadata", lambda: None)()
    if not isinstance(info_payload, dict):
        return {}, "no_info_key"
    labels = info_payload.get("idToLabels")
    if not isinstance(labels, dict):
        return {}, f"info_keys={sorted(info_payload)}"

    mapping: dict[str, str] = {}
    for key, value in labels.items():
        instance_id = _instance_id_from_label(value)
        if instance_id:
            mapping[str(key)] = instance_id
    return mapping, "idToLabels"


def _merge_camera_labels(
    per_camera: list[tuple[str, dict[str, str]]],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """One global map from the cameras' own maps, keeping only the ids they agree about.

    The renderer's id space belongs to the render product, so two cameras may give the same number
    to different instances -- `CaptureReport.instance_id_map` says as much. Merging them as the
    cameras were read therefore has two failure modes, and the previous version had both: the last
    camera read silently decided what an id meant, and a camera whose annotator produced nothing
    could not be seen at all, because the other cameras had already filled the map that was being
    checked.

    An id the cameras disagree about is dropped and named rather than resolved to whichever was
    read last. The per-camera maps are unaffected -- they are the ones a saved channel is labelled
    by -- so a disagreement costs the global map a key, not a delivery its evidence.
    """
    merged: dict[str, str] = {}
    conflicted: set[str] = set()
    for _camera_id, labels in per_camera:
        for key, value in labels.items():
            if key in conflicted:
                continue
            seen = merged.get(key)
            if seen is None:
                merged[key] = value
            elif seen != value:
                conflicted.add(key)
                del merged[key]
    return merged, tuple(sorted(conflicted))


def _author_camera(stage, camera_request):
    """Author one USD camera, its framing included.

    Split out of `_setup_camera` so the framing can be read back off a real stage without Kit:
    the render product and the annotators need a running host, the camera prim does not.
    """
    from pxr import Gf, UsdGeom

    path = f"/World/Cameras/{camera_request.camera_id}"
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(camera_request.focal_length_mm)
    # Both apertures are authored even when the request kept USD's defaults. Until they were,
    # this function wrote only `focalLength`, so the horizontal field of view came from a
    # renderer fallback nobody had compared against the Blender side's hand-coded value.
    camera.CreateHorizontalApertureAttr(camera_request.horizontal_aperture_mm)
    camera.CreateVerticalApertureAttr(camera_request.vertical_aperture_mm)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    eye = Gf.Vec3d(*camera_request.eye)
    target = Gf.Vec3d(*camera_request.target)
    matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
    return path


def _setup_camera(stage, camera_request, *, entity_by_instance):
    """Author one camera and its render product with the three annotators."""
    import omni.replicator.core as rep

    path = _author_camera(stage, camera_request)
    width, height = camera_request.resolution
    product = rep.create.render_product(path, (width, height))

    # `instance_segmentation` is used because it is what this worker was verified against. The
    # name is recorded in the report so a switch to `instance_id_segmentation` -- which uses a
    # different id space -- cannot pass unnoticed.
    annotator_family = "instance_segmentation"
    annotators: dict = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb")}
    # distance_to_image_plane is camera-z; v2 used distance_to_camera, which is radial. The two
    # differ by the off-axis angle, so the convention is recorded rather than assumed.
    annotators["depth"] = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    try:
        annotators["instance"] = rep.AnnotatorRegistry.get_annotator(
            annotator_family, init_params={"colorize": False}
        )
    except Exception:  # noqa: BLE001 - fall back to the default initialisation
        annotators["instance"] = rep.AnnotatorRegistry.get_annotator(annotator_family)
    annotators["_family"] = annotator_family
    for annotator in annotators.values():
        if hasattr(annotator, "attach"):
            annotator.attach(product)

    # Prime the pipeline once so the annotators have data. The renderer assigns instance ids on
    # its first frame, so reading labels before this yields an empty map, and target visibility
    # computed from an empty map is reported as unknown -- correct, but useless when a frame has
    # in fact been drawn.
    rep.orchestrator.step(rt_subframes=1, pause_timeline=True)
    labels, _ = _id_map(annotators["instance"])
    annotators["_labels"] = labels
    return product, annotators


def _settle(app, request: WorkerRequest, *, entity_by_instance) -> dict:
    """Run the physics clock and report what each dynamic object did.

    Deliberately records the *simulated* clock as well as wall time. The acceptance manual requires
    both, because a step loop that stops early still finishes quickly and would otherwise look like
    a successful five-second settle.
    """
    import numpy as np
    import omni.timeline
    from pxr import Usd, UsdGeom

    dynamic = [instance for instance in request.instances if instance.dynamic]
    if not dynamic or request.simulate_seconds <= 0:
        return {"performed": False, "reason": "no dynamic instances or zero duration"}

    from pxr import Sdf

    stage = omni.usd.get_context().get_stage()
    physics_prim = stage.GetPrimAtPath("/World/physicsScene")
    if physics_prim:
        # Enabled only here, so the authored state is what the images show and what the settle
        # starts from.
        physics_prim.CreateAttribute(
            "physics:simulationEnabled", Sdf.ValueTypeNames.Bool
        ).Set(True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.play()

    steps = int(round(request.simulate_seconds * request.physics_hz))
    sim_seconds = 0.0
    dt = 1.0 / request.physics_hz
    trace_path = None
    samples: dict[str, list[list[float]]] = {instance.instance_id: [] for instance in dynamic}

    started = time.monotonic()
    for _ in range(steps):
        app.update()
        sim_seconds += dt
        for instance in dynamic:
            prim = stage.GetPrimAtPath(f"/World/Instances/{instance.instance_id}")
            if not prim:
                continue
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            position = matrix.ExtractTranslation()
            samples[instance.instance_id].append([position[0], position[1], position[2]])
    timeline.stop()
    wall_seconds = time.monotonic() - started

    outcomes = []
    for instance in dynamic:
        positions = samples[instance.instance_id]
        if not positions:
            outcomes.append({"instance_id": instance.instance_id, "readable": False})
            continue
        array = np.asarray(positions, dtype=float)
        final = array[-1]
        tail = array[-max(1, request.physics_hz):]
        drift = float(np.linalg.norm(tail - final, axis=1).max()) if len(tail) > 1 else 0.0
        outcomes.append({
            "instance_id": instance.instance_id,
            "entity_id": instance.entity_id,
            "readable": True,
            "initial_position_m": array[0].tolist(),
            "final_position_m": final.tolist(),
            "drop_m": float(array[0][2] - final[2]),
            "last_second_drift_m": drift,
            "sample_count": len(positions),
        })

    # The per-step trace is written out because v2's settle report has none: its stability is a
    # final-window difference, which cannot distinguish a settled object from one that was
    # prevented from moving. A trace makes that checkable after the fact.
    return {
        "performed": True,
        "requested_sim_seconds": request.simulate_seconds,
        "observed_sim_seconds": sim_seconds,
        "wall_seconds": wall_seconds,
        "physics_hz": request.physics_hz,
        "step_count": steps,
        "objects": outcomes,
        "trace": {name: positions[::max(1, len(positions) // 200)] for name, positions in samples.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-gpu", type=int, default=None)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    payload = json.loads(args.request.read_text())
    if args.active_gpu is not None:
        payload["active_gpu"] = args.active_gpu
    if payload.get("room", {}).get("shell_uri"):
        payload["room"]["shell_uri"] = str((args.request.resolve().parent / payload["room"]["shell_uri"]).resolve())
    request = WorkerRequest.from_dict(payload)
    report = WorkerReport(scene_id=request.scene_id, scene_version=request.scene_version, status="failed")
    started = time.monotonic()
    app = None
    code = 1
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({
            "headless": True,
            "renderer": "RaytracedLighting",
            "active_gpu": request.active_gpu,
            "physics_gpu": request.active_gpu,
            "multi_gpu": False,
            "width": max(camera.resolution[0] for camera in request.cameras),
            "height": max(camera.resolution[1] for camera in request.cameras),
        })

        import omni.timeline
        import omni.usd
        from isaacsim.core.version import get_version
        from pxr import UsdGeom, UsdLux, UsdPhysics

        report = WorkerReport(
            scene_id=request.scene_id,
            scene_version=request.scene_version,
            status="failed",
            isaac_version=".".join(list(get_version())[:2]),
        )

        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # The delivery package must not embed a physics scene; the running wrapper creates the one
        # it uses. That is what this is.
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        # Simulation starts off, and only the settle turns it on.
        #
        # `pause_timeline=True` on a Replicator step does not stop PhysX: measured on a real scene,
        # a cup sat exactly where it was authored, was unchanged by timeline.stop(), and then moved
        # 0.44 m sideways with a 0.14 m drop *during the render pass* -- so the delivered images
        # showed a scene that had already fallen over, and the settle that followed measured a fall
        # that had partly happened before it began. Disabling the scene is explicit and does not
        # depend on how the renderer treats the timeline.
        # Set as a raw attribute: this USD version's UsdPhysics.Scene has no helper for it.
        from pxr import Sdf as _Sdf

        physics_scene.GetPrim().CreateAttribute(
            "physics:simulationEnabled", _Sdf.ValueTypeNames.Bool
        ).Set(False)

        openings_cut = _author_room(stage, request)
        shell_reference = {"mode": "procedural"}
        if request.room.shell_uri:
            from rfw_w.scenes.workers.isaac.shell_reference import inspect_shell
            shell_reference = inspect_shell(request.room.shell_uri)
        # The windows, all of them, rather than "the" window. The sun is aimed through the largest,
        # because that is the one whose light reaches furthest into the room, and the backdrop is
        # authored per window -- a room with three windows and one backdrop has two of them looking
        # onto nothing.
        windows = [o for o in request.room.openings if o.kind == "window"]
        widest = max(windows, key=lambda o: o.width_m, default=None)
        lighting = _author_lighting(stage, request.room, lighting=request.lighting,
                                    window_center=widest.center_xy if widest else None)
        exterior = _author_exterior(stage, request.room, windows=windows,
                                    surface=request.room_materials.facade)

        entity_by_instance: dict[int, str] = {}
        asset_materials: dict = {}
        for index, instance in enumerate(request.instances):
            prim = _author_instance(stage, instance, index=index, out_dir=out,
                                    asset_materials=asset_materials)
            _apply_semantic_label(prim, instance.entity_id)
            entity_by_instance[index] = instance.entity_id

        report = WorkerReport(
            scene_id=request.scene_id,
            scene_version=request.scene_version,
            status="failed",
            isaac_version=report.isaac_version,
        )

        # Stop the timeline before rendering.
        #
        # The stage is authored with rigid bodies already in place, and if the clock is running they
        # begin to fall the moment the app updates -- including during the render priming steps. The
        # images then show the *settled* state, not the generated one, and the settle that follows
        # measures a fall that had already happened. Measured on a real scene: a cup's first settle
        # sample sat 0.18 m below where it was placed, with a lateral offset, while an object that
        # rested stably was unaffected -- the signature of motion during capture rather than during
        # the settle.
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        app.update()

        # Diagnostic: where are the dynamic bodies at three points in the run? A body that arrives
        # at its first settle sample hundreds of millimetres from where it was authored moved
        # somewhere other than the settle, and the three readings say which stage.
        probe_positions: dict[str, Any] = {}

        def _snapshot(label: str) -> None:
            from pxr import Usd, UsdGeom as _G

            readings = {}
            for instance in request.instances:
                if not instance.dynamic:
                    continue
                prim = stage.GetPrimAtPath(f"/World/Instances/{instance.instance_id}")
                if not prim:
                    continue
                t_world = _G.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                p3 = t_world.ExtractTranslation()
                readings[instance.instance_id] = [round(float(v), 4) for v in p3]
            probe_positions[label] = readings

        _snapshot("after_authored")
        _snapshot("after_stop")

        # Dataset channels must describe the state whose stability was measured. Settling after
        # capture produced valid images and valid physics evidence for two different poses.
        settle = _settle(app, request, entity_by_instance=entity_by_instance)
        _snapshot("after_settle")

        captures = []
        payload_kind = ""
        # One entry per camera, not one map merged as we go. `_capture` refreshes this camera's map
        # after its own 16-subframe capture and puts it on the report; the caller's job is only to
        # keep them apart until every camera has been read.
        per_camera_labels: list[tuple[str, dict[str, str]]] = []
        for camera_request in request.cameras:
            product, annotators = _setup_camera(
                stage, camera_request, entity_by_instance=entity_by_instance
            )
            capture = _capture(camera_request, annotators, out, entity_by_instance)
            captures.append(capture)
            per_camera_labels.append((camera_request.camera_id, dict(capture.instance_id_map)))
            if not payload_kind:
                try:
                    _, payload_kind = _extract_instance_ids(annotators["instance"].get_data(), __import__("numpy"))
                except Exception:  # noqa: BLE001
                    payload_kind = "unavailable"

        _snapshot("after_capture")

        annotator_labels, id_conflicts = _merge_camera_labels(per_camera_labels)

        # The instance-id map comes from the annotator's own idToLabels and nothing else.
        #
        # An earlier version seeded this map from the request's instance order, which produced a
        # map that read as correct and was not: the renderer assigns its own ids, so the image came
        # back with one id covering 97% of the frame -- the room -- labelled as a mug. A fabricated
        # mapping is worse than a missing one, because every downstream check that resolves an id
        # to an entity would resolve it to the wrong object and sound confident doing it.
        # The annotator labels prim paths; the request knows which instance each path names. The
        # map is therefore annotator-derived in its keys and request-derived only in translating a
        # prim name to an entity, which is a lookup rather than an assumption about id numbering.
        instance_by_id = {instance.instance_id: instance.entity_id for instance in request.instances}
        instance_by_index = {
            index: instance.instance_id for index, instance in enumerate(request.instances)
        }
        merged_map: dict[str, str] = {}
        for key, label in annotator_labels.items():
            entity = instance_by_id.get(label)
            if entity:
                merged_map[key] = entity

        # What each view contained against what it declared. A miss is named per view and does not
        # fail the report: the render did what it could, and the run decides what it means.
        declarations = {camera.camera_id: camera for camera in request.cameras}
        coverage: list[dict[str, Any]] = []
        view_gaps: list[str] = []
        for capture in captures:
            declared = declarations.get(capture.camera_id)
            minimum = getattr(declared, "minimum_visible_instances", 0)
            visible_entities = len(capture.instance_id_map)
            coverage.append({
                "camera_id": capture.camera_id,
                "visible_instances": visible_entities,
                "minimum_visible_instances": minimum,
                "obligation_met": visible_entities >= minimum,
                "contains_nothing": capture.contains_nothing,
                "target_required": bool(capture.target_instance_ids),
                "target_visible": capture.target_visible,
            })
            if visible_entities < minimum:
                view_gaps.append(f"VIEW_EMPTY({capture.camera_id})")
            if capture.target_instance_ids and capture.target_visible is False:
                view_gaps.append(
                    f"VIEW_TARGET_MISSING({capture.camera_id},"
                    f"{','.join(capture.target_instance_ids)})"
                )

        report = WorkerReport(
            scene_id=request.scene_id,
            scene_version=request.scene_version,
            status="passed",
            isaac_version=report.isaac_version,
            captures=tuple(captures),
            instance_id_map=merged_map,
            instance_id_map_source=(
                "annotator_idToLabels" if annotator_labels else "unavailable"
            ),
            instance_id_conflicts=id_conflicts,
            instance_index_map={str(k): v for k, v in instance_by_index.items()},
            observed_label_payload_kind=payload_kind,
            openings_requested=len(request.room.openings),
            openings_cut=openings_cut,
            shell_reference=shell_reference,
            lighting=lighting,
            exterior_backdrops=exterior,
            asset_textures=asset_materials,
            annotator_family=annotators.get("_family", ""),
            semantic_filter=annotators.get("_filter", ""),
            settle=settle,
            position_probe=probe_positions,
            capture_state="settled" if settle.get("performed") else "authored",
            view_coverage=tuple(coverage),
            view_gaps=tuple(view_gaps),
            wall_seconds=time.monotonic() - started,
        )
        report.assert_self_consistent()
        code = 0
    except Exception as error:  # noqa: BLE001 - the report is the point of the failure path
        report = WorkerReport(
            scene_id=report.scene_id,
            scene_version=report.scene_version,
            status="failed",
            isaac_version=report.isaac_version,
            captures=report.captures,
            instance_id_map=report.instance_id_map,
            observed_label_payload_kind=report.observed_label_payload_kind,
            # Carried through so a failed run still records which id space it was working in, which
            # is what a reader needs to interpret the captures it did manage to write.
            annotator_family=report.annotator_family,
            semantic_filter=report.semantic_filter,
            instance_id_map_source=report.instance_id_map_source,
            settle=report.settle,
            # Carried through the failure path too: a failed run is exactly when the diagnostic is
            # wanted, and dropping it there made the probe useless the first time it was needed.
            position_probe=getattr(report, "position_probe", {}),
            error_type=type(error).__name__,
            error_detail=str(error)[:2000],
            wall_seconds=time.monotonic() - started,
        )
        traceback.print_exc()
    finally:
        (out / "report.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report.status, "report": str(out / "report.json")}), flush=True)
        # os._exit rather than SimulationApp.close(): on the installed Isaac 4.5.0-rc.36 the Kit
        # shutdown path segfaults after the report is written, turning a passing run into a
        # non-zero exit. The report is on disk and flushed, so the exit code is all that is left
        # to get right, and this is the only way to get it right on this build.
        import os

        os._exit(code)
    return code




# The room's surfaces come from the request, not from here. They used to be four constants in this
# module, which meant `StylePlan.palette` was read by nothing and two specs asking for different
# colour families rendered identical rooms. The defaults now live on `protocol.RoomMaterialsRequest`
# -- where a request built without materials still gets these exact values -- and
# `tests/unit/test_worker_room_materials.py` fails if a literal reappears here.


def _make_material(stage, name: str, *, surface=None, colour=None, roughness: float = 0.7,
                   metallic: float = 0.0, opacity: float = 1.0, emissive=None):
    """A UsdPreviewSurface material. RTX renders these without a texture file.

    ``opacity`` below 1 makes the surface see-through. It is deliberately not taken to the
    near-transparent end: v2 measured that headless RTX renders a near-transparent
    ``UsdPreviewSurface`` as a **mirror** which hides whatever is behind it, and answered that by
    shipping no glass geometry at all. A window pane here is semi-opaque for that reason.
    """
    from pxr import Gf, Sdf, UsdShade

    if (surface is None) == (colour is None):
        raise ValueError(
            f"{name}: exactly one of `surface` and `colour` is required, got "
            f"surface={type(surface).__name__} colour={type(colour).__name__}"
        )
    if surface is not None:
        colour = surface.colour
        roughness = surface.roughness
        metallic = surface.metallic
    material = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Materials/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _author_lighting(stage, room, *, window_center: tuple[float, float] | None = None,
                     lighting=None) -> dict:
    """Dome fill, a sun through the window, and a ceiling light.

    A single dim dome was what made earlier renders read as a grey box: no directional light means
    no shading, no shadows and nothing to suggest depth. The sun is aimed along the window's inward
    direction so the light in the image comes from somewhere in the room rather than from nowhere.

    **The values come from `lighting`, not from literals.** They were hardcoded here while the plan's
    `AppearanceSettings` had no path to the renderer at all -- and the literals happened to equal the
    defaults, so nothing looked wrong. What it meant in practice was that `repair.appearance` could
    change the plan, change the scene version, invalidate the render and force a re-render, and
    produce a byte-identical image: the whole appearance track could not change anything it claimed
    to. `lighting` defaults to the same numbers, so a caller that omits it renders as before.
    """
    from pxr import Gf, UsdGeom, UsdLux

    if lighting is None:
        from rfw_w.scenes.workers.isaac.protocol import LightingRequest

        lighting = LightingRequest()

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    dome.CreateIntensityAttr(lighting.dome_intensity)
    dome.CreateColorAttr(Gf.Vec3f(*lighting.dome_color))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/sun")
    sun.CreateIntensityAttr(lighting.sun_intensity)
    sun.CreateColorAttr(Gf.Vec3f(*lighting.sun_color))
    sun.CreateAngleAttr(1.5)
    # Aim from the window toward the room centre, so the doorway of light lands inside.
    target = Gf.Vec3d(
        sum(x for x, _ in room.polygon_xy) / len(room.polygon_xy),
        sum(y for _, y in room.polygon_xy) / len(room.polygon_xy),
        room.floor_z_m,
    )
    if window_center is not None:
        eye = Gf.Vec3d(window_center[0], window_center[1], room.height_m * 0.8)
    else:
        eye = Gf.Vec3d(target[0] + 4.0, target[1] - 4.0, room.height_m * 1.4)
    # A distant light has no direction attribute: it points along the prim's own orientation, so
    # the aim is the transform. (Setting an attribute here raised AttributeError on this build.)
    UsdGeom.Xformable(sun).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    )

    ceiling = UsdLux.RectLight.Define(stage, "/World/Lights/ceiling_fill")
    ceiling.CreateIntensityAttr(lighting.ceiling_fill_intensity)
    ceiling.CreateWidthAttr(max(1.0, max(x for x, _ in room.polygon_xy) - min(x for x, _ in room.polygon_xy)) * 0.7)
    ceiling.CreateHeightAttr(max(1.0, max(y for _, y in room.polygon_xy) - min(y for _, y in room.polygon_xy)) * 0.7)
    referenced = bool(getattr(room, "shell_uri", None))
    # Infinigen's inner ceiling is 5 cm below the plan datum: the old emitter coincided
    # with that surface. USD rect lights emit along local -Z; 180 degrees points upward.
    UsdGeom.Xformable(ceiling).AddTranslateOp().Set(
        Gf.Vec3d(target[0], target[1], room.floor_z_m + room.height_m - (0.20 if referenced else 0.05))
    )
    UsdGeom.Xformable(ceiling).AddRotateXYZOp().Set(
        Gf.Vec3f(0.0 if referenced else 180.0, 0.0, 0.0)
    )

    # What was applied, read back off the request so the report can be compared with the plan that
    # asked for it rather than with a literal that happens to match.
    return {
        "dome_intensity": lighting.dome_intensity,
        "sun_intensity": lighting.sun_intensity,
        "ceiling_fill_intensity": lighting.ceiling_fill_intensity,
        "dome_color": list(lighting.dome_color),
        "sun_color": list(lighting.sun_color),
    }


def _author_exterior(stage, room, *, windows, surface) -> int:
    """A sky plane just outside each window, so glazing shows sky rather than an empty void.

    Without it a window renders as a blown-out white rectangle: the camera sees past the wall into
    nothing, and the brightest thing in the image is a hole. A backdrop gives the window something
    to look onto, which is what makes it read as a window.

    One plane *per window*, not one for the room. While the shell had exactly one window those were
    the same thing; a room with three windows and one backdrop has two of them looking onto nothing.

    ``surface`` is required rather than defaulted: a facade colour decided here is a facade colour
    the plan cannot change, which is the defect this whole change is about.
    """
    from pxr import Gf, UsdGeom, UsdShade

    if not windows:
        return 0
    sky = _make_material(stage, "exterior_sky", surface=surface)
    # Outside the wall, on the far side of the opening from the room centre.
    cx = sum(x for x, _ in room.polygon_xy) / len(room.polygon_xy)
    cy = sum(y for _, y in room.polygon_xy) / len(room.polygon_xy)
    distance = 2.5
    for index, window in enumerate(windows):
        wx, wy = window.center_xy
        dx, dy = wx - cx, wy - cy
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        plane = UsdGeom.Plane.Define(stage, f"/World/Exterior/sky_{index:02d}")
        UsdGeom.Xformable(plane).AddScaleOp().Set(Gf.Vec3f(3.0, 3.0, 3.0))
        UsdGeom.Xformable(plane).AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
        UsdGeom.Xformable(plane).AddTranslateOp().Set(
            Gf.Vec3d(wx + dx / norm * distance, wy + dy / norm * distance, room.height_m * 0.5)
        )
        _bind(plane.GetPrim(), sky)
    return len(windows)


if __name__ == "__main__":
    raise SystemExit(main())
