# SPDX-License-Identifier: Apache-2.0
"""The prepared OBJ: self-consistent indices, one material group per part, real colours.

Two defects live here, both measured on the office scene's own furniture.

The first is indexing. Until this was written, the OBJ came from trimesh's exporter, which numbers
the faces of a geometry with *global* vertex indices while emitting a `vt` block that is shorter
than that -- measured on the ceiling light: 6028 `vt` lines, faces referencing up to 14427. A reader
can only drop those UVs (this one did) or read out of range, and the asset lost the 588 KB map its
GLB ships.

The second is colour. The same exporter wrote each material's base-colour factor as sRGB-encoded
bytes while `UsdPreviewSurface.diffuseColor` is linear, so every constant-coloured asset rendered
lighter than the asset states -- and it wrote trimesh's default grey for a material that states no
factor at all, which is a colour no asset ever asked for.

These tests need no asset library: a GLB with the same shape as the ones that broke is built here,
with trimesh, and put through the real `prepare_furniture`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfw_w.scenes.assets.prepared_obj import sanitise as _sanitise
from rfw_w.scenes.assets.providers.hssd import HssdConfig, prepare_furniture
from rfw_w.scenes.workers.isaac.geometry import load_obj

trimesh = pytest.importorskip("trimesh")
np = pytest.importorskip("numpy")

ASSET_ID = "abcdef0123456789"


def _material(name: str, *, colour=None, image: bool = False):
    from trimesh.visual.material import PBRMaterial

    kwargs = {"name": name}
    if colour is not None:
        kwargs["baseColorFactor"] = [colour[0], colour[1], colour[2], 1.0]
    if image:
        from PIL import Image

        kwargs["baseColorTexture"] = Image.new("RGB", (128, 128), (200, 180, 160))
    return PBRMaterial(**kwargs)


def write_mixed_glb(directory: Path, *, first_has_uvs: bool) -> Path:
    """A two-material GLB where exactly one geometry carries texture coordinates.

    That asymmetry is the shape of the real assets: the ceiling light's glass has no UVs and its
    metal does, and it is what trimesh's exporter cannot index correctly.
    """
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    sphere = trimesh.creation.icosphere(radius=0.1)

    def uvs_for(points):
        """One coordinate per vertex, which is what a glTF primitive carries."""
        uv = np.zeros((len(points), 2), dtype=float)
        uv[:, 0] = np.linspace(0.0, 1.0, len(points))
        return uv

    for geometry, material, has_uvs in (
        (box, _material("Glass_Clear"), first_has_uvs),
        (sphere, _material("Metal", colour=(0.25, 0.5, 0.75), image=True), True),
    ):
        geometry.visual = trimesh.visual.TextureVisuals(
            uv=uvs_for(geometry.vertices) if has_uvs else None, material=material,
        )
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="glass", geom_name="Glass_Clear")
    scene.add_geometry(sphere, node_name="metal", geom_name="Metal")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{ASSET_ID}.glb"
    scene.export(str(target))
    return target


def config_for(tmp_path: Path, glb: Path) -> HssdConfig:
    models = tmp_path / "models"
    (models / "objects" / ASSET_ID[0]).mkdir(parents=True, exist_ok=True)
    (models / "objects" / ASSET_ID[0] / f"{ASSET_ID}.glb").write_bytes(glb.read_bytes())
    index = tmp_path / "synset.json"
    index.write_text("{}")
    return HssdConfig(
        models_root=models, synset_index_path=index, prepared_root=tmp_path / "prepared",
    )


class TestTheExportedObjIsSelfConsistent:
    def test_every_texture_reference_is_inside_the_declared_vt_block(self, tmp_path: Path) -> None:
        """The defect, as an invariant: a reader must not have to guess or drop."""
        glb = write_mixed_glb(tmp_path / "glb", first_has_uvs=False)
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        declared = 0
        referenced: list[int] = []
        for line in prepared.visual_obj.read_text().splitlines():
            if line.startswith("vt "):
                declared += 1
            elif line.startswith("f ") and "/" in line:
                referenced.extend(int(token.split("/")[1]) for token in line.split()[1:])
        assert referenced, "no face references texture coordinates; the fixture is not mixed"
        assert max(referenced) <= declared, f"faces reference uv {max(referenced)} of {declared}"

    def test_a_group_without_coordinates_writes_faces_without_them(self, tmp_path: Path) -> None:
        glb = write_mixed_glb(tmp_path / "glb", first_has_uvs=False)
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        body = prepared.visual_obj.read_text()
        glass = body.split("usemtl Glass_Clear", 1)[1].split("usemtl Metal", 1)[0]
        assert "/" not in glass, "the untextured group's faces carry texture references"

    def test_the_two_groups_keep_their_own_coordinates(self, tmp_path: Path) -> None:
        """Local indexing per group: the second group's uv 1 is its own first coordinate."""
        glb = write_mixed_glb(tmp_path / "glb", first_has_uvs=True)
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        obj = load_obj(prepared.visual_obj)
        assert obj.group_has_uvs("Metal")
        assert obj.group_has_uvs("Glass_Clear")
        assert obj.is_textured


class TestTheMaterialsComeOutOfTheGlb:
    def test_a_factor_is_written_in_the_space_the_renderer_reads(self, tmp_path: Path) -> None:
        glb = write_mixed_glb(tmp_path / "glb", first_has_uvs=False)
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        mtl = prepared.visual_obj.with_suffix(".mtl").read_text()
        metal = mtl.split("newmtl Metal", 1)[1]
        kd = next(line for line in metal.splitlines() if line.startswith("Kd "))
        # Within a byte: trimesh writes the factor into the GLB's 0-255 encoding and reads it back
        # the same way, so the fixture cannot round-trip more precisely than 1/255.
        assert [float(value) for value in kd.split()[1:]] == pytest.approx([0.25, 0.5, 0.75], abs=0.005)

    def test_a_material_that_states_no_colour_gets_no_kd(self, tmp_path: Path) -> None:
        """Rather than trimesh's default grey, which is a colour no asset ever asked for."""
        glb = write_mixed_glb(tmp_path / "glb", first_has_uvs=False)
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        mtl = prepared.visual_obj.with_suffix(".mtl").read_text()
        glass = mtl.split("newmtl Glass_Clear", 1)[1].split("newmtl ", 1)[0]
        assert not [line for line in glass.splitlines() if line.startswith("Kd ")]

    def test_the_map_is_written_and_referenced_by_its_own_material(self, tmp_path: Path) -> None:
        glb = write_mixed_glb(tmp_path / "glb", first_has_uvs=False)
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        directory = prepared.visual_obj.parent
        image = directory / f"{_sanitise('Metal')}.png"
        assert image.is_file(), sorted(p.name for p in directory.iterdir())
        mtl = prepared.visual_obj.with_suffix(".mtl").read_text()
        assert f"map_Kd {image.name}" in mtl.split("newmtl Metal", 1)[1]
        assert prepared.has_diffuse_texture

    def test_the_texture_resolves_through_the_worker_reader(self, tmp_path: Path) -> None:
        glb = write_mixed_glb(tmp_path / "glb", first_has_uvs=False)
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        obj = load_obj(prepared.visual_obj)
        resolved = obj.diffuse_texture(prepared.visual_obj, "Metal")
        assert resolved is not None and resolved.name == f"{_sanitise('Metal')}.png"
        # And the group with no coordinates is not handed its sibling's map.
        assert not obj.group_has_uvs("Glass_Clear")


class TestThePreparedAssetIsStillTheSameAsset:
    def test_the_dimensions_are_the_glb_bounds_stood_upright(self, tmp_path: Path) -> None:
        """The export change must not move the asset: it is the same geometry, written by us.

        Checked on the written file as well as on the record, because the two can disagree: the
        record is measured from the merged mesh (rotation applied) while the file is written from
        the scene graph, and a writer that ignores the graph writes the pre-rotation coordinates.
        """
        glb = write_mixed_glb(tmp_path / "glb", first_has_uvs=False)
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        assert prepared.dimensions_m.to_list() == pytest.approx([0.2, 0.2, 0.2], abs=1e-6)
        assert prepared.bounds_min_m.z == pytest.approx(0.0)
        written = load_obj(prepared.visual_obj)
        for axis in range(3):
            values = [point[axis] for point in written.points]
            assert max(values) - min(values) == pytest.approx(0.2, abs=1e-6)

    def test_a_single_material_asset_is_still_one_group(self, tmp_path: Path) -> None:
        directory = tmp_path / "glb"
        box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        box.visual = trimesh.visual.TextureVisuals(material=_material("wood", colour=(0.4, 0.3, 0.2)))
        scene = trimesh.Scene()
        scene.add_geometry(box, node_name="box", geom_name="wood")
        glb = directory / f"{ASSET_ID}.glb"
        directory.mkdir(parents=True, exist_ok=True)
        scene.export(str(glb))
        prepared = prepare_furniture(config_for(tmp_path, glb), ASSET_ID)
        obj = load_obj(prepared.visual_obj)
        assert len(obj.material_groups()) == 1
        assert obj.material_colour(prepared.visual_obj, "wood") == pytest.approx(
            (0.4, 0.3, 0.2), abs=0.005
        )
