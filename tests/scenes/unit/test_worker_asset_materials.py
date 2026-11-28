# SPDX-License-Identifier: Apache-2.0
"""One material per face group, so a multi-material asset keeps its parts.

The defect these pin, measured on the real library: the office notebook's GLB states three
materials -- a pale metallic cover, a red body and a pink cover -- and the prepared OBJ the renderer
read stated one, because the provider asked trimesh for a single concatenated mesh
(``force="mesh"``). Three materials in, one grey out, and the visual critic then scored the grey as
"insufficient tonal and material separation". The chair had the same shape of loss: chrome, black
and fabric became fabric everywhere.

The USD half of these tests builds a real stage with ``pxr`` and reads the subsets back, which is
the code path the worker runs under Isaac. ``pxr`` arrives with the orchestrator's environment and is
not a declared dependency of the core, so that class skips where it is absent rather than failing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from rfw_w.scenes.workers.isaac.geometry import load_obj

REPO_ROOT = Path(__file__).resolve().parents[3]

PLAIN_OBJ = """mtllib material.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
v 1 1 0
v 1 0 1
usemtl Shader 3
f 1 2 3
f 1 3 4
usemtl Shader2
f 2 5 3
usemtl 5/8 in
f 3 5 6
"""

PLAIN_MTL = """newmtl Shader 3
Kd 0.78 0.78 0.78
newmtl Shader2
Kd 0.92 0.36 0.36
newmtl 5/8 in
Kd 0.96 0.75 0.88
"""

# Two materials, one of which carries a map: the case where the two kinds of appearance have to
# coexist on one mesh.
TEXTURED_OBJ = """mtllib material.mtl
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
usemtl Fabric
f 1/1 2/2 3/3
usemtl Black
f 1/1 3/3 2/2
"""

TEXTURED_MTL = """newmtl Fabric
Kd 1.00 1.00 1.00
map_Kd Fabric.png
newmtl Black
Kd 0.00 0.00 0.00
"""


def write_asset(directory: Path, obj: str, mtl: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "material.mtl").write_text(mtl)
    visual = directory / "asset.obj"
    visual.write_text(obj)
    return visual


def write_map(directory: Path, name: str = "Fabric.png", size: int = 64) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", (size, size), (200, 180, 160)).save(directory / name)


class TestTheGroupsPartitionTheFaces:
    """The grouping is what the binder iterates, so its shape is the contract."""

    def test_faces_come_back_grouped_in_first_appearance_order(self, tmp_path: Path) -> None:
        visual = write_asset(tmp_path / "a", PLAIN_OBJ, PLAIN_MTL)
        groups = load_obj(visual).material_groups()
        assert [name for name, _ in groups] == ["Shader 3", "Shader2", "5/8 in"]
        assert [indices for _, indices in groups] == [(0, 1), (2,), (3,)]

    def test_a_single_material_is_one_group(self, tmp_path: Path) -> None:
        """The path every SimReady asset and every built support takes, unchanged."""
        visual = write_asset(
            tmp_path / "b",
            "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl material_0\nf 1 2 3\n",
            "newmtl material_0\nKd 1 1 1\n",
        )
        groups = load_obj(visual).material_groups()
        assert groups == (("material_0", (0,)),)

    def test_faces_that_name_nothing_are_one_unnamed_group(self, tmp_path: Path) -> None:
        visual = write_asset(
            tmp_path / "c", "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            "newmtl material_0\nKd 1 1 1\n",
        )
        assert load_obj(visual).material_groups() == (("", (0,)),)

    def test_a_mesh_with_no_faces_still_has_a_group(self, tmp_path: Path) -> None:
        """So a caller never has to special-case an empty mesh before binding something."""
        visual = write_asset(tmp_path / "d", "v 0 0 0\n", "")
        assert load_obj(visual).material_groups() == (("", ()),)


class TestTheLookupsAnswerForOneMaterial:
    """One read of the MTL answers both questions, and both must be about the same material."""

    def test_each_materials_colour_is_read_from_its_own_entry(self, tmp_path: Path) -> None:
        visual = write_asset(tmp_path / "a", PLAIN_OBJ, PLAIN_MTL)
        obj = load_obj(visual)
        assert obj.material_colour(visual, "Shader2") == pytest.approx((0.92, 0.36, 0.36))
        assert obj.material_colour(visual, "5/8 in") == pytest.approx((0.96, 0.75, 0.88))

    def test_the_default_is_the_first_material_the_faces_name(self, tmp_path: Path) -> None:
        """A single-material caller that passes no name reads what it always read."""
        visual = write_asset(tmp_path / "a", PLAIN_OBJ, PLAIN_MTL)
        assert load_obj(visual).material_colour(visual) == pytest.approx((0.78, 0.78, 0.78))

    def test_a_name_the_file_does_not_state_resolves_to_nothing(self, tmp_path: Path) -> None:
        """Not to somebody else's colour: the parts of a mesh are not interchangeable."""
        visual = write_asset(tmp_path / "a", PLAIN_OBJ, PLAIN_MTL)
        obj = load_obj(visual)
        assert obj.material_colour(visual, "Absent") is None
        assert obj.diffuse_texture(visual, "Absent") is None

    def test_a_map_is_resolved_per_material(self, tmp_path: Path) -> None:
        directory = tmp_path / "b"
        visual = write_asset(directory, TEXTURED_OBJ, TEXTURED_MTL)
        write_map(directory)
        obj = load_obj(visual)
        assert obj.diffuse_texture(visual, "Fabric") == (directory / "Fabric.png").resolve()
        # The material beside it has no map, and must not be handed the other one's.
        assert obj.diffuse_texture(visual, "Black") is None

    def test_a_placeholder_map_is_not_a_map_for_any_material(self, tmp_path: Path) -> None:
        """The 16x8 synthesised image renders the object its colour; a factor is honest."""
        directory = tmp_path / "c"
        visual = write_asset(directory, TEXTURED_OBJ, TEXTURED_MTL)
        write_map(directory, size=16)
        assert load_obj(visual).diffuse_texture(visual, "Fabric") is None


class TestTheMaterialNamesBecomeUsdNames:
    def test_a_name_that_is_not_an_identifier_is_translated(self) -> None:
        worker = importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")
        assert worker._material_slug("Shader 3") == "Shader_3"
        assert worker._material_slug("Metal___Chrome_2017") == "Metal___Chrome_2017"

    def test_a_name_starting_with_a_digit_is_prefixed(self) -> None:
        worker = importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")
        assert worker._material_slug("5/8 in") == "material_5_8_in"

    def test_a_name_with_nothing_usable_in_it_is_named(self) -> None:
        worker = importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")
        assert worker._material_slug("///") == "unnamed"


@pytest.fixture(scope="module")
def worker():
    return importlib.import_module("rfw_w.scenes.workers.isaac.render_worker")


@pytest.fixture(scope="module")
def usd():
    pytest.importorskip("pxr")
    import types

    from pxr import Usd, UsdGeom, UsdShade

    return types.SimpleNamespace(Usd=Usd, UsdGeom=UsdGeom, UsdShade=UsdShade)


def instance_for(visual: Path):
    from rfw_w.scenes.workers.isaac.protocol import MeshInstance

    return MeshInstance(
        instance_id="table", entity_id="table", visual_uri=str(visual),
        collision_uri=None, translation=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def subsets_of(usd, stage, path: str = "/World/Instances/table") -> dict:
    prim = stage.GetPrimAtPath(path)
    return {
        child.GetPath().name: child
        for child in prim.GetChildren()
        if child.IsA(usd.UsdGeom.Subset)
    }


def surface_shader_of(usd, subset_prim):
    material, _ = usd.UsdShade.MaterialBindingAPI(subset_prim).ComputeBoundMaterial()
    assert material, f"{subset_prim.GetPath()} is bound to no material"
    shader, _, _ = material.ComputeSurfaceSource()
    assert shader, f"{subset_prim.GetPath()}'s material has no surface shader"
    return shader


class TestTheAuthoredStageBindsEachPart:
    def test_a_multi_material_instance_authors_one_subset_per_material(
        self, worker, usd, tmp_path: Path
    ) -> None:
        visual = write_asset(tmp_path / "a", PLAIN_OBJ, PLAIN_MTL)
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_instance(
            stage, instance_for(visual), index=0, out_dir=tmp_path, asset_materials={}
        )
        subsets = subsets_of(usd, stage)
        assert sorted(subsets) == ["materialBind_Shader2", "materialBind_Shader_3",
                                   "materialBind_material_5_8_in"]
        for prim in subsets.values():
            assert usd.UsdGeom.Subset(prim).GetFamilyNameAttr().Get() == "materialBind"
            assert usd.UsdGeom.Subset(prim).GetElementTypeAttr().Get() == "face"

    def test_each_subset_wears_its_own_colour(self, worker, usd, tmp_path: Path) -> None:
        """The measured loss: three materials in, one colour out."""
        visual = write_asset(tmp_path / "a", PLAIN_OBJ, PLAIN_MTL)
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_instance(
            stage, instance_for(visual), index=0, out_dir=tmp_path, asset_materials={}
        )
        by_name = subsets_of(usd, stage)
        expected = {
            "materialBind_Shader_3": (0.78, 0.78, 0.78),
            "materialBind_Shader2": (0.92, 0.36, 0.36),
            "materialBind_material_5_8_in": (0.96, 0.75, 0.88),
        }
        for name, colour in expected.items():
            shader = surface_shader_of(usd, by_name[name])
            got = shader.GetInput("diffuseColor").Get()
            assert tuple(round(float(channel), 2) for channel in got) == pytest.approx(colour), name

    def test_the_subsets_cover_every_face_exactly_once(self, worker, usd, tmp_path: Path) -> None:
        """USD requires the materialBind family to be a partition; an uncovered face renders as
        the prim's material, and the prim has none."""
        visual = write_asset(tmp_path / "a", PLAIN_OBJ, PLAIN_MTL)
        stage = usd.Usd.Stage.CreateInMemory()
        worker._author_instance(
            stage, instance_for(visual), index=0, out_dir=tmp_path, asset_materials={}
        )
        assigned: list[int] = []
        for prim in subsets_of(usd, stage).values():
            assigned.extend(usd.UsdGeom.Subset(prim).GetIndicesAttr().Get())
        assert sorted(assigned) == list(range(len(load_obj(visual).faces)))
        assert len(assigned) == len(set(assigned))

    def test_a_textured_part_and_a_coloured_part_coexist(self, worker, usd, tmp_path: Path) -> None:
        directory = tmp_path / "b"
        visual = write_asset(directory, TEXTURED_OBJ, TEXTURED_MTL)
        write_map(directory)
        stage = usd.Usd.Stage.CreateInMemory()
        records: dict = {}
        worker._author_instance(
            stage, instance_for(visual), index=0, out_dir=directory, asset_materials=records
        )
        by_name = subsets_of(usd, stage)
        fabric = surface_shader_of(usd, by_name["materialBind_Fabric"])
        connected, _, _ = fabric.GetInput("diffuseColor").GetConnectedSource()
        assert connected is not None, "the fabric part has no map bound"
        reader = usd.UsdShade.Shader(connected.GetPrim())
        assert reader.GetIdAttr().Get() == "UsdUVTexture"
        black = surface_shader_of(usd, by_name["materialBind_Black"])
        assert tuple(round(float(c), 2) for c in black.GetInput("diffuseColor").Get()) == (0.0, 0.0, 0.0)

    def test_the_record_says_which_material_wore_what(self, worker, usd, tmp_path: Path) -> None:
        visual = write_asset(tmp_path / "a", PLAIN_OBJ, PLAIN_MTL)
        stage = usd.Usd.Stage.CreateInMemory()
        records: dict = {}
        worker._author_instance(
            stage, instance_for(visual), index=0, out_dir=tmp_path, asset_materials=records
        )
        assert records["table"] == (
            "Shader 3=colour:(0.78, 0.78, 0.78); "
            "Shader2=colour:(0.92, 0.36, 0.36); "
            "5/8 in=colour:(0.96, 0.75, 0.88)"
        )

    def test_a_single_material_instance_authors_no_subsets(self, worker, usd, tmp_path: Path) -> None:
        """The path every SimReady target takes: a prim-level binding, and no subsets at all."""
        visual = write_asset(
            tmp_path / "c",
            "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl material_0\nf 1 2 3\n",
            "newmtl material_0\nKd 1 1 1\n",
        )
        stage = usd.Usd.Stage.CreateInMemory()
        records: dict = {}
        worker._author_instance(
            stage, instance_for(visual), index=0, out_dir=tmp_path, asset_materials=records
        )
        assert subsets_of(usd, stage) == {}
        assert records["table"] == "colour:(1.0, 1.0, 1.0)"

    def test_an_asset_that_states_nothing_keeps_the_display_colour(
        self, worker, usd, tmp_path: Path
    ) -> None:
        """The fallback that existed before must survive: it is recorded, not silently bound."""
        visual = write_asset(
            tmp_path / "d",
            "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl material_0\nf 1 2 3\n",
            "newmtl material_0\nKa 0.4 0.4 0.4\n",
        )
        stage = usd.Usd.Stage.CreateInMemory()
        records: dict = {}
        worker._author_instance(
            stage, instance_for(visual), index=0, out_dir=tmp_path, asset_materials=records
        )
        assert records["table"] == "fallback_display_colour"
        assert usd.UsdGeom.Mesh(stage.GetPrimAtPath("/World/Instances/table")).GetDisplayColorAttr()
