# SPDX-License-Identifier: Apache-2.0
"""Preparing a v1.3 target: the source's mesh, upright, with a proxy in the same frame.

The defect this exists for was measured in a real run. v1.3 ships each asset twice and recommends
loading the normalised family as Z-up; measured against the release's own files, the jar's axis moves
to Z and the mug's stays on Y with only its X and Z extents swapped -- so loading the family as Z-up
stands the mug on its side, which is what the critique of the first v1.3 run reported ("the cup is
lying on its side, creating an unstable and implausible initial pose for the pick task").

`up_axis_fix` records what the release did, not what it produced, and it is per asset. So the loader
takes the ``<leaf>.obj`` the source shipped -- the frame every SimReady asset in this project has
been loaded from -- and builds its own pair. The collision mesh exists only in the normalised family,
which is the reason a pair has to be built rather than picked.

No asset library is needed: a two-material OBJ, in the shape v1.3 writes, is built here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfw_w.scenes.assets.providers.simready import (
    SimreadyPrepareConfig,
    SimreadyPrepareError,
    prepare_rigid,
)
from rfw_w.scenes.workers.isaac.geometry import load_obj

trimesh = pytest.importorskip("trimesh")
np = pytest.importorskip("numpy")

LEAF = "9_9_0001"
PREFIX = f"9/9/{LEAF}"

# A mug-shaped mesh, Y-up: a body taller than it is wide, with a handle. Written as an OBJ in the
# shape the release writes, with the material and its map beside it.
OBJ = """mtllib material.mtl
o Body
usemtl ceramic
v -0.04 0.000 0.04
v  0.04 0.000 0.04
v  0.04 0.000 -0.04
v -0.04 0.000 -0.04
v -0.04 0.100 0.04
v  0.04 0.100 0.04
v  0.04 0.100 -0.04
v -0.04 0.100 -0.04
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
o Handle
usemtl brass
v -0.07 0.030 0.00
v -0.07 0.070 0.00
v -0.04 0.070 0.00
v -0.04 0.030 0.00
f 9 10 11 12
"""

MTL = """newmtl ceramic
Kd 0.90000000 0.88000000 0.85000000
map_Kd material_0.png
newmtl brass
Kd 0.72000000 0.58000000 0.32000000
"""


def library(tmp_path: Path, *, with_map: bool = True) -> SimreadyPrepareConfig:
    mesh_dir = tmp_path / "library" / PREFIX / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / f"{LEAF}.obj").write_text(OBJ)
    (mesh_dir / "material.mtl").write_text(MTL)
    if with_map:
        from PIL import Image

        Image.new("RGB", (128, 128), (230, 220, 200)).save(mesh_dir / "material_0.png")
    return SimreadyPrepareConfig(
        assets_root=tmp_path / "library", prepared_root=tmp_path / "prepared",
    )


class TestTheSourceMeshIsTheOneLoaded:
    def test_a_missing_source_mesh_fails_by_name(self, tmp_path: Path) -> None:
        """The normalised family is not a substitute, and saying so is the whole point."""
        config = library(tmp_path)
        (tmp_path / "library" / PREFIX / "mesh" / f"{LEAF}.obj").unlink()
        with pytest.raises(SimreadyPrepareError, match="no source mesh"):
            prepare_rigid(config, prefix=PREFIX, leaf=LEAF)

    def test_the_prepared_mesh_is_the_source_stood_upright(self, tmp_path: Path) -> None:
        """Y-up becomes Z-up by rotation, so the mug's height moves from Y to Z.

        Asserted on the *file*, not on the record: the record is measured from the merged mesh,
        where the rotation is already applied, so it said 0.10 m while the written OBJ still had
        the source's orientation -- the scene graph holds the rotation, and a writer that reads
        `scene.geometry` writes the pre-rotation coordinates.
        """
        prepared = prepare_rigid(library(tmp_path), prefix=PREFIX, leaf=LEAF)
        assert prepared.dimensions_m.z == pytest.approx(0.100, abs=1e-6)
        assert prepared.dimensions_m.x == pytest.approx(0.110, abs=1e-6)  # body plus handle
        assert prepared.bounds_min_m.z == pytest.approx(0.0, abs=1e-6)
        written = load_obj(prepared.visual_obj)
        zs = [point[2] for point in written.points]
        ys = [point[1] for point in written.points]
        assert max(zs) - min(zs) == pytest.approx(0.100, abs=1e-6), "the written mesh is not upright"
        assert min(zs) == pytest.approx(0.0, abs=1e-6)
        assert max(ys) - min(ys) == pytest.approx(0.080, abs=1e-6)

    def test_winding_survives_the_rotation(self, tmp_path: Path) -> None:
        """A swap would mirror it: this correction is the one the worker's own matrix uses."""
        prepared = prepare_rigid(library(tmp_path), prefix=PREFIX, leaf=LEAF)
        mesh = trimesh.load(str(prepared.visual_obj), force="mesh", process=False)
        triangles = mesh.vertices[mesh.faces]
        volume = float(
            np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
            / 6.0
        )
        assert volume > 0, "the prepared mesh is inside out"

    def test_a_multi_material_asset_keeps_its_parts(self, tmp_path: Path) -> None:
        prepared = prepare_rigid(library(tmp_path), prefix=PREFIX, leaf=LEAF)
        obj = load_obj(prepared.visual_obj)
        assert sorted(name for name, _ in obj.material_groups()) == ["brass", "ceramic"]
        assert obj.material_colour(prepared.visual_obj, "brass") == pytest.approx(
            (0.72, 0.58, 0.32), abs=0.005
        )

    def test_the_map_is_carried_when_the_asset_has_one(self, tmp_path: Path) -> None:
        prepared = prepare_rigid(library(tmp_path), prefix=PREFIX, leaf=LEAF)
        assert any(prepared.visual_obj.parent.glob("*.png"))
        assert prepared.has_diffuse_texture
        with_map = prepare_rigid(library(tmp_path / "second", with_map=False),
                                 prefix=PREFIX, leaf=LEAF)
        assert not with_map.has_diffuse_texture


class TestTheProxyIsInTheSameFrame:
    def test_the_collision_hull_covers_the_prepared_mesh(self, tmp_path: Path) -> None:
        """The reason a pair is prepared rather than picked: the release's collision is in a frame
        the visual is not, and a proxy measured in the wrong frame misplaces the object."""
        prepared = prepare_rigid(library(tmp_path), prefix=PREFIX, leaf=LEAF)
        hull = trimesh.load(str(prepared.collision_obj), force="mesh", process=False)
        extents = hull.bounds[1] - hull.bounds[0]
        assert extents[2] == pytest.approx(prepared.dimensions_m.z, abs=0.02)
        assert extents[0] == pytest.approx(prepared.dimensions_m.x, abs=0.02)

    def test_the_proxy_is_decimated_to_the_face_budget(self, tmp_path: Path) -> None:
        prepared = prepare_rigid(library(tmp_path), prefix=PREFIX, leaf=LEAF)
        assert prepared.collision_face_count <= 256


class TestThePreparationIsRecorded:
    def test_the_sidecar_round_trips(self, tmp_path: Path) -> None:
        config = library(tmp_path)
        fresh = prepare_rigid(config, prefix=PREFIX, leaf=LEAF)
        again = prepare_rigid(config, prefix=PREFIX, leaf=LEAF)
        assert again.to_dict() == fresh.to_dict()
        assert again.visual_obj == fresh.visual_obj

    def test_a_changed_source_invalidates_the_cache(self, tmp_path: Path) -> None:
        """An asset re-exported in place must not keep being served from the old mesh."""
        config = library(tmp_path)
        first = prepare_rigid(config, prefix=PREFIX, leaf=LEAF)
        source = tmp_path / "library" / PREFIX / "mesh" / f"{LEAF}.obj"
        body = source.read_text().replace("0.100", "0.200")
        source.write_text(body)
        second = prepare_rigid(config, prefix=PREFIX, leaf=LEAF)
        assert second.dimensions_m.z == pytest.approx(0.200, abs=1e-6)
        assert first.dimensions_m.z == pytest.approx(0.100, abs=1e-6)

    def test_the_record_names_the_file_it_came_from(self, tmp_path: Path) -> None:
        prepared = prepare_rigid(library(tmp_path), prefix=PREFIX, leaf=LEAF)
        payload = json.loads(
            (tmp_path / "prepared" / LEAF / "prepared.json").read_text()
        )
        assert payload["source_uri"].endswith(f"{LEAF}.obj")
        assert payload["source_identity"]["size"] > 0
        assert prepared.source_uri.endswith(f"{LEAF}.obj")
