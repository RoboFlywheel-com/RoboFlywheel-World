# SPDX-License-Identifier: Apache-2.0
"""The shapes a support can be built to, and the artefacts built from them.

The support used to be retrieved, so its geometry was whatever the library had. It is constructed
now, which means the shape vocabulary is the thing that decides what the task stands on -- and a
vocabulary is only as good as the invariants it refuses to break.

Every test here is either an invariant that must hold for *every* shape or a refusal that must name
what went wrong. "It built something" is not evidence: a recipe that dropped a leg or stopped the
top 2 mm short would satisfy it.
"""

from __future__ import annotations

import json

import pytest

from rfw_w.scenes.assets.procedural_support import (
    ProceduralSupportError,
    materialise,
    write_obj,
)
from rfw_w.scenes.contracts.asset import AffordanceStatus
from rfw_w.scenes.solvers.support_shape import (
    TOP_PART_NAME,
    SupportShape,
    SupportShapeError,
    parts_for,
    shape_from,
    top_surface,
    volume_m3,
)

# A size per shape that is plausible for the furniture it stands for, so a recipe is exercised at the
# proportions it will actually meet rather than at one convenient size.
SIZES = {
    SupportShape.SLAB_LEGS: (1.35, 0.75, 0.75),
    SupportShape.SLAB_PANELS: (1.35, 0.75, 0.75),
    SupportShape.PLINTH: (1.55, 0.90, 0.90),
    SupportShape.TWO_TIER: (1.55, 0.90, 0.90),
    SupportShape.PEDESTAL: (0.60, 0.45, 0.575),
}


class TestEveryShapeKeepsTheInvariants:
    @pytest.mark.parametrize("shape", list(SupportShape))
    def test_the_top_slabs_upper_face_is_the_working_height(self, shape: SupportShape) -> None:
        """The one invariant every shape shares, and the one the task is solved against.

        The core decides a target is reachable from a standoff at `height_m`; a recipe whose top
        stopped short would put the object below the surface the arm was judged against, and nothing
        downstream measures that.
        """
        width, depth, height = SIZES[shape]
        parts = parts_for(shape, width_m=width, depth_m=depth, height_m=height)
        tops = [part for part in parts if part.name == TOP_PART_NAME]
        assert len(tops) == 1
        assert tops[0].z_range_m[1] == pytest.approx(height, abs=1e-9)
        assert tops[0].size_m == (width, depth, tops[0].size_m[2])

    @pytest.mark.parametrize("shape", list(SupportShape))
    def test_nothing_reaches_below_the_floor_or_above_the_top(self, shape: SupportShape) -> None:
        width, depth, height = SIZES[shape]
        parts = parts_for(shape, width_m=width, depth_m=depth, height_m=height)
        for part in parts:
            lower, upper = part.z_range_m
            assert lower >= -1e-9, part.name
            assert upper <= height + 1e-9, part.name

    @pytest.mark.parametrize("shape", list(SupportShape))
    def test_nothing_stands_outside_the_declared_footprint(self, shape: SupportShape) -> None:
        """The footprint is what the placement solver reserves floor for."""
        width, depth, height = SIZES[shape]
        parts = parts_for(shape, width_m=width, depth_m=depth, height_m=height)
        for part in parts:
            assert abs(part.centre_m[0]) + part.size_m[0] / 2 <= width / 2 + 1e-9, part.name
            assert abs(part.centre_m[1]) + part.size_m[1] / 2 <= depth / 2 + 1e-9, part.name

    @pytest.mark.parametrize("shape", list(SupportShape))
    def test_the_top_surface_is_the_footprint_at_the_height(self, shape: SupportShape) -> None:
        width, depth, height = SIZES[shape]
        polygon, z = top_surface(shape, width_m=width, depth_m=depth, height_m=height)
        assert z == pytest.approx(height)
        assert polygon.bounds() == pytest.approx((-width / 2, -depth / 2, width / 2, depth / 2))
        assert polygon.is_rectangle()

    def test_a_two_tier_support_has_a_shelf_below_its_top(self) -> None:
        """Named for the shape rather than asserted generically: a shelf *at* the top is not a shelf."""
        parts = parts_for(SupportShape.TWO_TIER, width_m=1.55, depth_m=0.90, height_m=0.90)
        shelf = next(part for part in parts if part.name == "shelf")
        assert shelf.z_range_m[1] < 0.90 - 0.04  # strictly below the underside of the top slab
        # Between the legs, not through them: a shelf passing through a leg would double-count its
        # volume and self-intersect in the mesh.
        legs = [part for part in parts if part.name.startswith("leg_")]
        assert legs
        for leg in legs:
            assert not _overlap(shelf, leg)

    def test_the_volume_is_the_sum_of_the_parts(self) -> None:
        """Exact for boxes, which is the whole reason a constructed mass can be honest."""
        parts = parts_for(SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75)
        assert volume_m3(parts) == pytest.approx(sum(part.volume_m3 for part in parts))
        assert volume_m3(parts) > 0


class TestRefusalsByName:
    def test_an_unknown_shape_is_refused(self) -> None:
        with pytest.raises(SupportShapeError, match="UNKNOWN_SUPPORT_SHAPE"):
            shape_from("chaise_longue")

    def test_a_shape_that_cannot_fit_its_parts_is_refused(self) -> None:
        # A 0.10 m nightstand has no room for 0.07 m insets and 0.05 m legs.
        with pytest.raises(SupportShapeError, match="SUPPORT_SHAPE_TOO_SMALL"):
            parts_for(SupportShape.SLAB_LEGS, width_m=0.10, depth_m=0.10, height_m=0.575)

    def test_a_top_thicker_than_the_whole_support_is_refused(self) -> None:
        with pytest.raises(SupportShapeError, match="top_thickness_m"):
            parts_for(SupportShape.SLAB_LEGS, width_m=1.0, depth_m=0.6, height_m=0.03)

    def test_a_shelf_that_does_not_fit_under_the_top_is_refused(self) -> None:
        with pytest.raises(SupportShapeError, match="SUPPORT_SHAPE_TOO_SHORT"):
            parts_for(SupportShape.TWO_TIER, width_m=1.0, depth_m=0.6, height_m=0.10)

    def test_overlapping_parts_are_refused(self, monkeypatch) -> None:
        """The check that stops a recipe double-counting its own volume.

        Exercised by making a leg overlap the top slab, because no shipped recipe does -- which is
        exactly why the guard has to be tested rather than assumed to be unreachable.
        """
        from rfw_w.scenes.solvers import support_shape as module

        original = module._legs

        def legs_through_the_top(*args, **kwargs):
            parts = original(*args, **kwargs)
            first = parts[0]
            return (
                module.SupportPart(
                    name=first.name,
                    centre_m=(first.centre_m[0], first.centre_m[1], 0.74),
                    size_m=(first.size_m[0], first.size_m[1], first.size_m[2]),
                ),
                *parts[1:],
            )

        monkeypatch.setattr(module, "_legs", legs_through_the_top)
        with pytest.raises(SupportShapeError, match="SUPPORT_PARTS_OVERLAP"):
            parts_for(SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75)

    def test_a_non_positive_extent_is_refused(self) -> None:
        with pytest.raises(SupportShapeError):
            parts_for(SupportShape.PLINTH, width_m=-1.0, depth_m=0.6, height_m=0.75)


class TestTheBuiltArtefacts:
    def test_the_written_mesh_measures_the_support_that_was_asked_for(self, tmp_path) -> None:
        """The round trip that makes the provenance honest.

        The dimensions on the record come from measuring the file, not from the numbers the caller
        passed -- so this test is what stands between "constructed" and "we wrote down what we
        chose".
        """
        prepared = materialise(
            SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75,
            out_dir=tmp_path,
        )
        assert prepared.dimensions_m.to_list() == pytest.approx([1.35, 0.75, 0.75])
        assert prepared.bounds_min_m.to_list() == pytest.approx([-0.675, -0.375, 0.0])
        assert prepared.bounds_max_m.to_list() == pytest.approx([0.675, 0.375, 0.75])
        assert prepared.vertex_count == 8 * 5   # five boxes
        assert prepared.face_count == 6 * 5

    def test_the_support_surface_is_the_top_slab_and_says_it_was_constructed(self, tmp_path) -> None:
        prepared = materialise(
            SupportShape.PEDESTAL, width_m=0.60, depth_m=0.45, height_m=0.575, out_dir=tmp_path,
        )
        top = prepared.support_polygons[0]
        assert top.plane_z_m == pytest.approx(0.575)
        assert top.is_rectangle
        assert top.provenance.status is AffordanceStatus.CONSTRUCTED
        assert top.provenance.method == "constructed_part_volume_times_density"
        assert prepared.provenance.status is AffordanceStatus.CONSTRUCTED

    def test_the_mass_is_the_parts_and_the_method_says_so(self, tmp_path) -> None:
        prepared = materialise(
            SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75, out_dir=tmp_path,
        )
        parts = parts_for(SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75)
        assert prepared.mass_kg == pytest.approx(volume_m3(parts) * 500.0)
        assert prepared.provenance.method == "constructed_part_volume_times_density"

    def test_the_collision_mesh_is_the_parts_not_a_hull(self, tmp_path) -> None:
        """A table is not convex.

        A hull around one collides as a solid block: an arm could not reach under it and a settle
        would report contact where the scene has empty space. The prepared record must therefore
        carry the same geometry twice, not a simplified silhouette.
        """
        prepared = materialise(
            SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75, out_dir=tmp_path,
        )
        assert prepared.collision_face_count == prepared.face_count
        assert prepared.collision_obj.read_text() == prepared.visual_obj.read_text()

    def test_the_same_request_builds_the_same_artefact(self, tmp_path) -> None:
        """Determinism, because a scene version is a hash of what was built."""
        first = materialise(
            SupportShape.TWO_TIER, width_m=1.55, depth_m=0.90, height_m=0.90,
            out_dir=tmp_path / "a",
        )
        second = materialise(
            SupportShape.TWO_TIER, width_m=1.55, depth_m=0.90, height_m=0.90,
            out_dir=tmp_path / "b",
        )
        assert first.asset_key.asset_id == second.asset_key.asset_id
        assert first.visual_obj.read_bytes() == second.visual_obj.read_bytes()

    def test_a_different_support_builds_a_different_artefact(self, tmp_path) -> None:
        first = materialise(
            SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75,
            out_dir=tmp_path / "a",
        )
        second = materialise(
            SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.80,
            out_dir=tmp_path / "b",
        )
        assert first.asset_key.asset_id != second.asset_key.asset_id

    def test_the_sidecar_is_written_beside_the_mesh(self, tmp_path) -> None:
        """The run directory's prepared-asset layout is what the worker's paths are built from."""
        prepared = materialise(
            SupportShape.PLINTH, width_m=1.55, depth_m=0.90, height_m=0.90, out_dir=tmp_path,
        )
        sidecar = prepared.visual_obj.parent / "prepared.json"
        assert sidecar.is_file()
        payload = json.loads(sidecar.read_text())
        assert payload["support_polygons"][0]["plane_z_m"] == pytest.approx(0.90)

    def test_a_mesh_that_does_not_match_the_request_is_refused(self, tmp_path) -> None:
        """The guard itself, exercised: a writer bug must not become a floating target."""
        from rfw_w.scenes.assets.procedural_support import _require_measured_matches
        from rfw_w.scenes.contracts.units import Vec3

        with pytest.raises(ProceduralSupportError, match="CONSTRUCTED_SUPPORT_MEASURES_WRONG"):
            _require_measured_matches(
                SupportShape.SLAB_LEGS,
                Vec3(-0.675, -0.375, 0.0), Vec3(0.675, 0.375, 0.70),   # 5 cm short
                1.35, 0.75, 0.75,
            )

    def test_the_mesh_survives_the_readers_the_worker_uses(self, tmp_path) -> None:
        """Parsed by the worker's own OBJ reader, not only measured.

        This is the check that catches a bad face index. `measure_obj` counts `v` and `f` lines and
        never resolves an index, so a writer whose offset was one too large -- which the first
        version of this one was, because it counted *lines* rather than *vertices* past a header
        holding a comment and an ``mtllib`` -- measures perfectly and then fails inside Isaac.
        """
        from rfw_w.scenes.workers.isaac.geometry import load_obj

        prepared = materialise(
            SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75,
            out_dir=tmp_path, colour=(0.42, 0.30, 0.21),
        )
        mesh = load_obj(prepared.visual_obj)
        assert mesh.faces
        assert load_obj(prepared.collision_obj).faces

    def test_the_support_wears_the_colour_it_was_given(self, tmp_path) -> None:
        """The finish travels in the mesh, which is how it reaches the renderer at all.

        Without a material the worker falls back to a colour it invents -- measured, a lavender desk
        in a room whose prompt asked for warm wood.
        """
        from rfw_w.scenes.workers.isaac.geometry import load_obj

        prepared = materialise(
            SupportShape.PEDESTAL, width_m=0.60, depth_m=0.45, height_m=0.575,
            out_dir=tmp_path, colour=(0.42, 0.30, 0.21), roughness=0.6,
        )
        mesh = load_obj(prepared.visual_obj)
        assert not mesh.is_textured          # a colour factor, not a map
        assert mesh.material_colour(prepared.visual_obj) == pytest.approx((0.42, 0.30, 0.21))

    def test_the_colour_is_part_of_the_asset_name(self, tmp_path) -> None:
        """Two finishes of one form are two artefacts, not one reused."""
        first = materialise(
            SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75,
            out_dir=tmp_path / "a", colour=(0.42, 0.30, 0.21),
        )
        second = materialise(
            SupportShape.SLAB_LEGS, width_m=1.35, depth_m=0.75, height_m=0.75,
            out_dir=tmp_path / "b", colour=(0.82, 0.79, 0.74),
        )
        assert first.asset_key.asset_id != second.asset_key.asset_id

    def test_a_colour_outside_the_unit_cube_is_refused(self, tmp_path) -> None:
        with pytest.raises(ProceduralSupportError, match="colour"):
            materialise(
                SupportShape.PLINTH, width_m=1.55, depth_m=0.90, height_m=0.90,
                out_dir=tmp_path, colour=(1.4, 0.3, 0.2),
            )

    def test_an_empty_part_list_is_refused(self, tmp_path) -> None:
        with pytest.raises(ProceduralSupportError, match="at least one part"):
            write_obj((), tmp_path / "nothing.obj")


def _overlap(first, second) -> bool:
    return all(
        abs(first.centre_m[axis] - second.centre_m[axis])
        < (first.size_m[axis] + second.size_m[axis]) / 2.0 - 1e-9
        for axis in range(3)
    )
