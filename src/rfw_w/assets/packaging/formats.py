# SPDX-License-Identifier: Apache-2.0
"""Format assembly from determined mesh, collision and physical parameters."""

from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET


def _numbers(values):
    return " ".join(format(float(v), ".12g") for v in values)


def _appearance(mesh):
    """Resolve the two material representations produced by GLB/OBJ loaders."""
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    color = getattr(material, "baseColorFactor", None)
    if color is None:
        color = getattr(material, "diffuse", None)
    if color is None:
        color = [255, 255, 255, 255]
    color = [float(v) for v in color]
    if max(color) > 1:
        color = [v / 255 for v in color]
    if len(color) == 3:
        color.append(1.0)
    roughness = getattr(material, "roughnessFactor", None)
    metallic = getattr(material, "metallicFactor", None)
    return (
        image,
        color,
        float(roughness) if roughness is not None else 0.6,
        float(metallic) if metallic is not None else 0.0,
    )


def urdf(key, visual, hulls, physics, out):
    root = ET.Element("robot", name=key)
    body = ET.SubElement(root, "link", name=key)
    vis = ET.SubElement(body, "visual")
    ET.SubElement(
        ET.SubElement(vis, "geometry"), "mesh", filename=visual, scale="1 1 1"
    )
    for hull in hulls:
        col = ET.SubElement(body, "collision")
        ET.SubElement(
            ET.SubElement(col, "geometry"), "mesh", filename=hull, scale="1 1 1"
        )
    inertial = ET.SubElement(body, "inertial")
    ET.SubElement(inertial, "origin", xyz=_numbers(physics["com"]), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(physics["total_mass"]))
    a = physics["inertia"]
    ET.SubElement(
        inertial,
        "inertia",
        **{
            k: str(v)
            for k, v in zip(
                ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"),
                (a[0][0], a[0][1], a[0][2], a[1][1], a[1][2], a[2][2]),
            )
        },
    )
    extra = ET.SubElement(body, "extra_info")
    for name, value in (
        ("uuid", key),
        ("mass", physics["total_mass"]),
        ("mu1", physics["whole_mu1"]),
        ("mu2", physics["whole_mu2"]),
        ("real_bbox", _numbers(physics["dimensions_m"])),
    ):
        ET.SubElement(extra, name).text = str(value)
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)


def mjcf(key, visual, hulls, physics, out, visual_mesh=None):
    root = ET.Element("mujoco", model=key)
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "mesh", name="visual", file="../" + visual)
    image, color, roughness, metallic = (
        _appearance(visual_mesh)
        if visual_mesh is not None
        else (None, [1, 1, 1, 1], 0.6, 0.0)
    )
    material_attributes = {
        "name": "visual_material",
        "rgba": _numbers(color),
        "roughness": str(roughness),
        "metallic": str(metallic),
    }
    if image is not None and getattr(visual_mesh.visual, "uv", None) is not None:
        texture = Path(out).parent.parent / "mesh" / "mjcf-texture.png"
        image.save(texture)
        ET.SubElement(
            assets,
            "texture",
            name="visual_texture",
            type="2d",
            file="../mesh/mjcf-texture.png",
        )
        material_attributes["texture"] = "visual_texture"
    ET.SubElement(assets, "material", **material_attributes)
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name=key)
    ET.SubElement(body, "freejoint", name="root")
    a = physics["inertia"]
    ET.SubElement(
        body,
        "inertial",
        pos=_numbers(physics["com"]),
        mass=str(physics["total_mass"]),
        fullinertia=_numbers((a[0][0], a[1][1], a[2][2], a[0][1], a[0][2], a[1][2])),
    )
    ET.SubElement(
        body,
        "geom",
        type="mesh",
        mesh="visual",
        material="visual_material",
        contype="0",
        conaffinity="0",
        group="2",
        mass="0",
    )
    for i, hull in enumerate(hulls):
        name = "collision_" + str(i)
        ET.SubElement(assets, "mesh", name=name, file="../" + hull)
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=name,
            contype="1",
            conaffinity="1",
            group="3",
            rgba="0 0 0 0",
            mass="0",
            friction=_numbers((physics["whole_mu1"], physics["whole_mu2"], 0.0001)),
        )
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)


def usd(key, mesh, hull_paths, physics, out):
    import numpy as np, trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    body = UsdGeom.Xform.Define(
        stage, "/obj_" + key.replace("-", "_").replace(".", "_")
    )
    prim = body.GetPrim()
    stage.SetDefaultPrim(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(physics["total_mass"])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*physics["com"]))
    values, vectors = np.linalg.eigh(physics["inertia"])
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] *= -1
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(vectors).as_quat()
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))

    def author(path, source, collision=False):
        node = UsdGeom.Mesh.Define(stage, path)
        node.CreatePointsAttr([Gf.Vec3f(*v) for v in source.vertices])
        node.CreateFaceVertexCountsAttr([3] * len(source.faces))
        node.CreateFaceVertexIndicesAttr(source.faces.flatten().tolist())
        node.CreateSubdivisionSchemeAttr("none")
        if collision:
            UsdPhysics.CollisionAPI.Apply(node.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(node.GetPrim()).CreateApproximationAttr(
                "convexHull"
            )
            node.CreatePurposeAttr(UsdGeom.Tokens.guide)
        return node

    visual_node = author(str(prim.GetPath()) + "/visual", mesh)
    uv = getattr(mesh.visual, "uv", None)
    image, color, roughness, metallic = _appearance(mesh)
    visual_material = UsdShade.Material.Define(
        stage, str(prim.GetPath()) + "/visual_material"
    )
    surface = UsdShade.Shader.Define(
        stage, str(prim.GetPath()) + "/visual_material/surface"
    )
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color[:3])
    )
    surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
    visual_material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(visual_node.GetPrim()).Bind(visual_material)
    if uv is not None and image is not None:
        texture = Path(out).parent / "mesh" / "usd-texture.png"
        image.save(texture)
        texcoords = UsdGeom.PrimvarsAPI(visual_node).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        texcoords.Set([Gf.Vec2f(float(v[0]), float(v[1])) for v in uv])
        texture_shader = UsdShade.Shader.Define(
            stage, str(prim.GetPath()) + "/visual_material/texture"
        )
        texture_shader.CreateIdAttr("UsdUVTexture")
        texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("mesh/usd-texture.png")
        )
        reader = UsdShade.Shader.Define(
            stage, str(prim.GetPath()) + "/visual_material/uv"
        )
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture_shader.ConnectableAPI(), "rgb"
        )
        visual_material.CreateSurfaceOutput().ConnectToSource(
            surface.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(visual_node.GetPrim()).Bind(visual_material)
    material = UsdShade.Material.Define(stage, str(prim.GetPath()) + "/material")
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(physics["whole_mu1"])
    mat.CreateDynamicFrictionAttr(physics["whole_mu2"])
    for i, path in enumerate(hull_paths):
        node = author(
            str(prim.GetPath()) + "/hull_" + str(i),
            trimesh.load(path, force="mesh"),
            True,
        )
        UsdShade.MaterialBindingAPI.Apply(node.GetPrim()).Bind(
            material, materialPurpose="physics"
        )
    stage.GetRootLayer().Save()
