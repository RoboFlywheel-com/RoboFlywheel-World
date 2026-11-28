# SPDX-License-Identifier: Apache-2.0
"""Real textured GLB input survives both exported formats and relocation."""

from __future__ import annotations
import json, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest
import trimesh
from PIL import Image
from rfw_w.assets.packaging import formats


def roundtrip_mesh(tmp_path, representation):
    mesh = trimesh.creation.box()
    image = Image.new("RGB", (8, 8), (220, 40, 30))
    if representation == "pbr":
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=image,
            baseColorFactor=[180, 220, 255, 255],
            roughnessFactor=0.3,
            metallicFactor=0.2,
        )
    else:
        material = trimesh.visual.material.SimpleMaterial(
            image=image, diffuse=[180, 220, 255, 255]
        )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[i % 2, (i // 2) % 2] for i in range(len(mesh.vertices))]),
        material=material,
    )
    path = tmp_path / (representation + ".glb")
    mesh.export(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    assert loaded.visual.uv is not None
    return loaded


@pytest.mark.parametrize("representation", ["pbr", "simple"])
def test_textured_glb_exported_to_mjcf_and_usd_and_relocated(tmp_path, representation):
    Usd = pytest.importorskip("pxr.Usd")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    mujoco = pytest.importorskip("mujoco")
    mesh = roundtrip_mesh(tmp_path, representation)
    package = tmp_path / "package"
    (package / "mesh").mkdir(parents=True)
    (package / "mjcf").mkdir()
    mesh.export(package / "mesh/visual.obj")
    hull = mesh.convex_hull
    hull.export(package / "mesh/collision.obj")
    physical = {
        "total_mass": 1.0,
        "com": [0, 0, 0],
        "inertia": np.eye(3).tolist(),
        "whole_mu1": 0.7,
        "whole_mu2": 0.5,
    }
    formats.mjcf(
        "cube",
        "mesh/visual.obj",
        ["mesh/collision.obj"],
        physical,
        package / "mjcf/cube.xml",
        mesh,
    )
    formats.usd(
        "cube", mesh, [package / "mesh/collision.obj"], physical, package / "cube.usd"
    )
    copied = tmp_path / "copied"
    shutil.copytree(package, copied)
    model = mujoco.MjModel.from_xml_path(str(copied / "mjcf/cube.xml"))
    assert model.ntex == 1 and model.nmat == 1
    assert np.any(model.mat_texid >= 0)
    assert model.mesh_texcoord.shape[0] > 0
    stage = Usd.Stage.Open(str(copied / "cube.usd"))
    textures = [
        UsdShade.Shader(p)
        for p in stage.Traverse()
        if p.IsA(UsdShade.Shader)
        and UsdShade.Shader(p).GetIdAttr().Get() == "UsdUVTexture"
    ]
    assert len(textures) == 1
    ref = textures[0].GetInput("file").Get().path
    assert not Path(ref).is_absolute() and (copied / ref).is_file()
    assert Image.open(copied / ref).getpixel((0, 0))[:3] == (220, 40, 30)
    surface = UsdShade.Shader(stage.GetPrimAtPath("/obj_cube/visual_material/surface"))
    assert surface.GetInput("diffuseColor").HasConnectedSource()


def test_simple_material_adapter_preserves_original_image():
    mesh = trimesh.creation.box()
    image = Image.new("RGB", (4, 4), "blue")
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.SimpleMaterial(image=image),
    )
    assert formats._appearance(mesh)[0] is image
