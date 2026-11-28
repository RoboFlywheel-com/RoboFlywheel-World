# SPDX-License-Identifier: Apache-2.0
"""The figure shell copy: roof mode drops only ceiling/exterior; cutaway clips a wall."""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("pxr")

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = module_path('rfw_w.scenes.visualization.cutaway_shell')


cutaway = importlib.import_module('rfw_w.scenes.visualization.cutaway_shell')


from pxr import Usd, UsdGeom  # noqa: E402


def quad_mesh(stage, path: str, points: list, counts: list, indices: list):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    return mesh


def shell_fixture(tmp_path: Path) -> Path:
    """A minimal shell with one prim per role the cut logic distinguishes.

    WallSouth and WallWest are two stacked faces each (split at z=1.35), so a
    height clip above 1.35 drops exactly their upper faces, and one below 1.35
    removes the prims entirely (a straddling face is dropped with the cut). WallEast is two faces split at
    y=0.1, so a south cut plane drops exactly one of them -- that prim also
    carries a two-face GeomSubset, which is what exercises the subset-index
    remap. WallNorth is two stacked faces that no south/west clip should touch.
    """
    stage = Usd.Stage.CreateNew(str(tmp_path / "shell.usda"))
    root = UsdGeom.Xform.Define(stage, "/Shell")
    stage.SetDefaultPrim(root.GetPrim())
    quad_mesh(stage, "/Shell/Floor",
              [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], [4], [0, 1, 2, 3])
    quad_mesh(stage, "/Shell/Ceiling",
              [(0, 0, 2.7), (4, 0, 2.7), (4, 3, 2.7), (0, 3, 2.7)], [4], [0, 1, 2, 3])
    quad_mesh(stage, "/Shell/SkirtingBoard_Ceiling",
              [(0, 0, 2.6), (4, 0, 2.6), (4, 0.05, 2.7), (0, 0.05, 2.7)], [4], [0, 1, 2, 3])
    quad_mesh(stage, "/Shell/ExteriorFacade",
              [(-0.2, 0, 0), (-0.2, 3, 0), (-0.2, 3, 2.7), (-0.2, 0, 2.7)], [4], [0, 1, 2, 3])
    quad_mesh(stage, "/Shell/WallSouth",
              [(0, 0, 0), (4, 0, 0), (4, 0, 1.35), (0, 0, 1.35),
               (0, 0, 1.35), (4, 0, 1.35), (4, 0, 2.7), (0, 0, 2.7)],
              [4, 4], list(range(8)))
    quad_mesh(stage, "/Shell/WallWest",
              [(0, 0, 0), (0, 3, 0), (0, 3, 1.35), (0, 0, 1.35),
               (0, 0, 1.35), (0, 3, 1.35), (0, 3, 2.7), (0, 0, 2.7)],
              [4, 4], list(range(8)))
    quad_mesh(stage, "/Shell/WallNorth",
              [(0, 3, 0), (4, 3, 0), (4, 3, 1.35), (0, 3, 1.35),
               (0, 3, 1.35), (4, 3, 1.35), (4, 3, 2.7), (0, 3, 2.7)],
              [4, 4], list(range(8)))
    east = quad_mesh(stage, "/Shell/WallEast",
                     [(4, 0, 0), (4, 0.1, 0), (4, 0.1, 2.7), (4, 0, 2.7),
                      (4, 0.1, 0), (4, 3, 0), (4, 3, 2.7), (4, 0.1, 2.7)],
                     [4, 4], list(range(8)))
    subset = UsdGeom.Subset.Define(stage, "/Shell/WallEast/material")
    subset.CreateIndicesAttr([0, 1])
    subset.CreateElementTypeAttr("face")
    stage.GetRootLayer().Save()
    return tmp_path / "shell.usda"


def face_counts(path: Path, prim: str) -> list[int] | None:
    stage = Usd.Stage.Open(str(path))
    found = stage.GetPrimAtPath(prim)
    if not found.IsValid():
        return None
    return list(UsdGeom.Mesh(found).GetFaceVertexCountsAttr().Get())


def test_roof_mode_keeps_walls_full_height(tmp_path: Path) -> None:
    shell = shell_fixture(tmp_path)
    record = cutaway.build_cutaway_shell(shell, tmp_path / "out" / "shell_roof.usdc",
                                         None, mode="roof")
    assert record["mode"] == "roof"
    assert record["open_side"] is None
    out = Path(record["cutaway_shell"])
    # The dropped roles are gone entirely.
    assert face_counts(out, "/Shell/Ceiling") is None
    assert face_counts(out, "/Shell/SkirtingBoard_Ceiling") is None
    assert face_counts(out, "/Shell/ExteriorFacade") is None
    # Walls keep every face -- no cut plane ran, so nothing is clipped.
    assert face_counts(out, "/Shell/WallSouth") == [4, 4]
    assert face_counts(out, "/Shell/WallEast") == [4, 4]
    assert face_counts(out, "/Shell/Floor") == [4]
    by_path = {p["prim_path"]: p for p in record["prims"]}
    assert by_path["/Shell/WallSouth"]["action"] == "kept"
    assert by_path["/Shell/WallSouth"]["faces_kept"] == 2
    assert by_path["/Shell/Floor"]["action"] == "kept"
    # The subset survived untouched: no face moved, so no remap was needed.
    subset = by_path["/Shell/WallEast"]["subsets"][0]
    assert subset["faces_in"] == 2
    assert subset["faces_kept"] == 2
    sidecar = json.loads(out.with_suffix(".export.json").read_text())
    assert sidecar["mode"] == "roof"
    assert sidecar["figure_only"] is True


def test_cutaway_mode_still_clips_the_near_wall(tmp_path: Path) -> None:
    """Regression for the default mode: roof mode must not have weakened it."""
    shell = shell_fixture(tmp_path)
    cut = {"axis": "y", "keep_above": 0.0, "open_side": "south"}
    record = cutaway.build_cutaway_shell(shell, tmp_path / "out" / "shell_cutaway.usdc", cut)
    assert record["mode"] == "cutaway"
    out = Path(record["cutaway_shell"])
    # Both WallSouth faces sit below the cut plane: the prim is removed whole.
    assert face_counts(out, "/Shell/WallSouth") is None
    # WallEast loses exactly its face below the inset, and the subset is
    # remapped onto the surviving face rather than left pointing at a gap.
    assert face_counts(out, "/Shell/WallEast") == [4]
    stage = Usd.Stage.Open(str(out))
    subset = UsdGeom.Subset(stage.GetPrimAtPath("/Shell/WallEast/material"))
    assert list(subset.GetIndicesAttr().Get()) == [0]
    assert face_counts(out, "/Shell/Ceiling") is None
    assert face_counts(out, "/Shell/Floor") == [4]
    sidecar = json.loads(out.with_suffix(".export.json").read_text())
    assert sidecar["mode"] == "cutaway"


def test_cutaway_mode_requires_a_cut(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CUTAWAY_CUT_MISSING"):
        cutaway.build_cutaway_shell(shell_fixture(tmp_path), tmp_path / "o" / "s.usdc",
                                    None, mode="cutaway")


def test_unknown_mode_is_named(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CUTAWAY_MODE_UNKNOWN"):
        cutaway.build_cutaway_shell(shell_fixture(tmp_path), tmp_path / "o" / "s.usdc",
                                    None, mode="slice")


# ---------------------------------------------------------------------------
# Roof-mode wall clip: the two walls nearest the camera corner become a stub.
# ---------------------------------------------------------------------------

POLYGON = [[0, 0], [4, 0], [4, 3], [0, 3]]


def test_wall_clip_edges_uses_the_nearest_vertex() -> None:
    edges = cutaway.wall_clip_edges(POLYGON, "sw")
    assert [0, 0, 4, 0] in edges  # south edge
    assert [0, 3, 0, 0] in edges  # west edge (polygon wound counter-clockwise)
    edges = cutaway.wall_clip_edges(POLYGON, "se")
    assert [0, 0, 4, 0] in edges  # south edge
    assert [4, 0, 4, 3] in edges  # east edge
    with pytest.raises(ValueError, match="WALL_CLIP_CORNER_INVALID"):
        cutaway.wall_clip_edges(POLYGON, "up")


def test_wall_clip_edges_all_returns_the_full_loop() -> None:
    edges = cutaway.wall_clip_edges(POLYGON, "all")
    assert len(edges) == len(POLYGON)
    assert [0, 0, 4, 0] in edges
    assert [4, 0, 4, 3] in edges
    assert [4, 3, 0, 3] in edges
    assert [0, 3, 0, 0] in edges


def test_wall_clip_drops_only_high_near_wall_faces(tmp_path: Path) -> None:
    shell = shell_fixture(tmp_path)
    # 1.5 sits above the lower face's top (1.35) and below the upper face's
    # (2.7), so the height split lands exactly on the fixture's face boundary.
    clip = cutaway.wall_clip_for({"polygon_xy": POLYGON}, "sw", 1.5)
    record = cutaway.build_cutaway_shell(shell, tmp_path / "out" / "shell_roof.usdc",
                                         None, mode="roof", wall_clip=clip)
    out = Path(record["cutaway_shell"])
    # Near walls: the face above the clip height is gone, the stub below stays.
    assert face_counts(out, "/Shell/WallSouth") == [4]
    assert face_counts(out, "/Shell/WallWest") == [4]
    # WallEast's first face sits within the near radius of the south edge and is
    # full height: it is clipped too, and the subset remaps onto the survivor.
    assert face_counts(out, "/Shell/WallEast") == [4]
    stage = Usd.Stage.Open(str(out))
    subset = UsdGeom.Subset(stage.GetPrimAtPath("/Shell/WallEast/material"))
    assert list(subset.GetIndicesAttr().Get()) == [0]
    # The far wall keeps every face, including the high ones.
    assert face_counts(out, "/Shell/WallNorth") == [4, 4]
    assert face_counts(out, "/Shell/Floor") == [4]
    assert face_counts(out, "/Shell/Ceiling") is None
    by_path = {p["prim_path"]: p for p in record["prims"]}
    assert by_path["/Shell/WallSouth"]["action"] == "height_clipped"
    assert by_path["/Shell/WallWest"]["action"] == "height_clipped"
    assert by_path["/Shell/WallNorth"]["action"] == "kept"
    # The clip lands on a face boundary and the height test is on the highest
    # vertex: the kept stub face tops out at 1.35 (not the requested 1.5), and
    # the lowest dropped top is the full-height faces' 2.7.
    assert record["wall_clip"]["height_m"] == 1.5
    assert record["wall_clip"]["achieved_max_kept_z_m"] == pytest.approx(1.35)
    assert record["wall_clip"]["dropped_min_top_z_m"] == pytest.approx(2.7)
    assert record["wall_clip"]["corner"] == "sw"
    sidecar = json.loads(out.with_suffix(".export.json").read_text())
    assert sidecar["wall_clip"]["corner"] == "sw"


def test_wall_clip_drops_a_straddling_face(tmp_path: Path) -> None:
    """Conservative cut: a face whose top is above the clip is dropped even when
    most of it is below -- the stub is ragged at face granularity, never bladed."""
    shell = shell_fixture(tmp_path)
    clip = cutaway.wall_clip_for({"polygon_xy": POLYGON}, "sw", 1.25)
    record = cutaway.build_cutaway_shell(shell, tmp_path / "out" / "shell_roof.usdc",
                                         None, mode="roof", wall_clip=clip)
    out = Path(record["cutaway_shell"])
    # The fixture's lower near-wall faces top out at 1.35 > 1.25: they straddle
    # the cut and go with it, so the near-wall prims are removed entirely.
    assert face_counts(out, "/Shell/WallSouth") is None
    assert face_counts(out, "/Shell/WallWest") is None
    removed = {p["prim_path"] for p in record["prims_removed_entirely"]}
    assert "/Shell/WallSouth" in removed
    # Nothing near survived, so the achieved height records are empty.
    assert record["wall_clip"]["achieved_max_kept_z_m"] is None
    assert record["wall_clip"]["dropped_min_top_z_m"] == pytest.approx(1.35)
    # The far wall is untouched.
    assert face_counts(out, "/Shell/WallNorth") == [4, 4]


def test_wall_clip_drops_a_diagonal_full_height_triangle(tmp_path: Path) -> None:
    """The Infinigen wall triangulation regression: a diagonal triangle spans the
    full wall height with a centroid below the cut, and must still be dropped."""
    stage = Usd.Stage.CreateNew(str(tmp_path / "shell.usda"))
    root = UsdGeom.Xform.Define(stage, "/Shell")
    stage.SetDefaultPrim(root.GetPrim())
    quad_mesh(stage, "/Shell/Floor",
              [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], [4], [0, 1, 2, 3])
    # One south-wall prim, two triangles: a long diagonal spanning z 0 -> 2.7
    # (centroid z 0.9, below the 1.25 cut) and a low one entirely under it.
    quad_mesh(stage, "/Shell/WallSouth",
              [(0, 0.02, 0), (4, 0.02, 2.7), (0, 0.02, 2.7),
               (4, 0.02, 0), (4, 0.02, 1.0), (0, 0.02, 1.0)],
              [3, 3], list(range(6)))
    stage.GetRootLayer().Save()
    clip = cutaway.wall_clip_for({"polygon_xy": POLYGON}, "sw", 1.25)
    record = cutaway.build_cutaway_shell(tmp_path / "shell.usda",
                                         tmp_path / "out" / "shell_roof.usdc",
                                         None, mode="roof", wall_clip=clip)
    out = Path(record["cutaway_shell"])
    # The diagonal blade is gone; the fully-low triangle stays.
    assert face_counts(out, "/Shell/WallSouth") == [3]
    by_path = {p["prim_path"]: p for p in record["prims"]}
    assert by_path["/Shell/WallSouth"]["action"] == "height_clipped"
    assert by_path["/Shell/WallSouth"]["faces_kept"] == 1
    assert record["wall_clip"]["achieved_max_kept_z_m"] == pytest.approx(1.0)
    assert record["wall_clip"]["dropped_min_top_z_m"] == pytest.approx(2.7)


def test_wall_clip_needs_roof_mode(tmp_path: Path) -> None:
    clip = cutaway.wall_clip_for({"polygon_xy": POLYGON}, "sw", 1.25)
    with pytest.raises(ValueError, match="CUTAWAY_WALL_CLIP_MODE"):
        cutaway.build_cutaway_shell(shell_fixture(tmp_path), tmp_path / "o" / "s.usdc",
                                    {"axis": "y", "keep_above": 0.0, "open_side": "south"},
                                    mode="cutaway", wall_clip=clip)
