# SPDX-License-Identifier: Apache-2.0
"""Retrieval adapter logic that does not need the real library.

These are the parts that decide what a candidate looks like, so they are worth pinning down
without an asset root: the size re-filter that catches the catalogue disagreeing with the
geometry, and the projections that decide what a model is allowed to see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfw_w.scenes.adapters.simready_retrieval import (
    LIBRARY_FRAMES,
    LICENCE_UNRECORDED,
    CandidateView,
    RetrievalAdapterError,
    frame_for,
    measured_size_violation,
    worker_normalisation,
)
from rfw_w.scenes.contracts.asset import AssetKey
from rfw_w.scenes.contracts.scene_spec import EntityRole, EntitySpec, PhysicsType
from rfw_w.scenes.contracts.units import Vec3


class TestMeasuredSizeFilter:
    """v2's size gate runs on the catalogue's rounded bbox; this runs on the measurement."""

    def test_within_range_passes(self) -> None:
        assert measured_size_violation(Vec3(0.09, 0.09, 0.10), Vec3(0.09, 0.09, 0.105), 2.0) is None

    def test_too_tall_is_rejected_with_the_axis_and_the_factor(self) -> None:
        reason = measured_size_violation(Vec3(0.10, 0.10, 0.31), Vec3(0.09, 0.09, 0.105), 2.0)
        assert reason is not None
        assert "axisz" in reason
        assert "2.0x" in reason

    def test_too_small_is_rejected(self) -> None:
        reason = measured_size_violation(Vec3(0.01, 0.09, 0.10), Vec3(0.09, 0.09, 0.105), 2.0)
        assert reason is not None
        assert "below" in reason

    def test_ratio_is_relative_to_the_target_not_absolute(self) -> None:
        # A 0.10 m axis is fine against a 0.09 m target and out of range against a 0.02 m one.
        assert measured_size_violation(Vec3(0.10, 0.10, 0.10), Vec3(0.09, 0.09, 0.09), 2.0) is None
        assert measured_size_violation(Vec3(0.10, 0.10, 0.10), Vec3(0.02, 0.02, 0.02), 2.0)

    def test_non_positive_target_axis_is_ignored(self) -> None:
        assert measured_size_violation(Vec3(5.0, 0.09, 0.10), Vec3(0.0, 0.09, 0.105), 2.0) is None


class TestMeshMeasurementCache:
    def test_an_unchanged_obj_is_parsed_once(self, tmp_path: Path, monkeypatch) -> None:
        from rfw_w.scenes.adapters import simready_retrieval as retrieval

        path = tmp_path / "mesh.obj"
        path.write_text("v 0 0 0\nv 1 2 3\nf 1 2 2\n")
        retrieval._measure_obj_cached.cache_clear()
        calls = 0
        original = retrieval.measure_obj

        def counting_measure(candidate):
            nonlocal calls
            calls += 1
            return original(candidate)

        monkeypatch.setattr(retrieval, "measure_obj", counting_measure)
        assert retrieval._measure_path(path).extents == Vec3(1.0, 2.0, 3.0)
        assert retrieval._measure_path(path).extents == Vec3(1.0, 2.0, 3.0)
        assert calls == 1


def view(**overrides: object) -> CandidateView:
    payload: dict[str, object] = {
        "asset_key": AssetKey(provider="simready", library_version="v0.2", asset_id="12_17_0197"),
        "semantic": "Coffee Cup",
        "super_category": "Coffee Cup",
        "dimensions_m": Vec3(0.0948, 0.0952, 0.1007),
        "mass_kg": 0.4055,
        "material": "ceramic/porcelain",
        "total_score": 0.7221,
        "semantic_score": 0.9,
        "size_score": 0.8,
        "readiness_score": 0.7,
        "affordability_score": 0.5,
        "up_axis": "y",
        "library_prefix": "12/17/12_17_0197",
    }
    payload.update(overrides)
    return CandidateView(**payload)  # type: ignore[arg-type]


class TestProjections:
    def test_semantic_view_leaks_no_paths_or_hashes(self) -> None:
        """What a model may see: an opaque id and semantics, nothing resolvable."""
        projected = view().semantic_view()
        assert set(projected) == {
            "semantic_id", "semantic", "category", "dimensions_m", "material",
        }
        blob = str(projected)
        assert "12/17" not in blob
        assert ".obj" not in blob
        assert "/data" not in blob

    def test_audit_projection_carries_the_score_breakdown(self) -> None:
        payload = view().to_dict()
        assert payload["score_breakdown"]["semantic"] == 0.9
        assert payload["asset"]["asset_id"] == "12_17_0197"

    def test_audit_projection_omits_the_library_prefix(self) -> None:
        # The prefix is internal plumbing for resolving files, not evidence.
        assert "library_prefix" not in view().to_dict()
        assert "library_prefix" not in view().semantic_view()

    def test_friction_is_not_part_of_either_projection(self) -> None:
        for projected in (view().to_dict(), view().semantic_view()):
            assert "friction_static" not in projected


class TestTheLibraryFrameIsDeclaredNotAssumed:
    """Which way up a library's meshes are is a property of the library, and it is stated per version.

    v0.2 ships the GLB's own Y-up meshes. v1.3 ships those *and* a normalised family whose up axis
    is decided per asset by a heuristic -- measured on the release's own files, the jar's axis moves
    to Z and the mug's stays on Y -- so v1.3's loader-entry mesh is the source's frame too, and what
    it needs instead is a pair prepared from it (`LibraryFrame.prepare`), because the release's
    collision mesh exists only in the family we do not load.
    """

    def test_both_versions_are_loaded_from_the_source_frame(self) -> None:
        for version in ("v0.2", "v1.3"):
            frame = frame_for(version)
            assert frame.source_frame == "simready_y_up", version
            assert frame.method, version

    def test_only_the_version_without_a_usable_pair_asks_for_preparation(self) -> None:
        assert not frame_for("v0.2").prepare
        assert frame_for("v1.3").prepare

    def test_a_prepared_version_is_not_rotated_again(self) -> None:
        """The second half of the mug defect: the preparation rotates, so nothing else may.

        Applying the Y-up correction to a pair that was already prepared turned the mug's 0.10 m
        handle span into its height, and the next critique reported it lying on its side again.
        """
        assert frame_for("v0.2").rotate_on_read
        assert not frame_for("v1.3").rotate_on_read

    def test_an_undeclared_version_is_refused_rather_than_assumed(self) -> None:
        """The failure mode this exists for: a new export silently loaded as if it were v0.2."""
        with pytest.raises(RetrievalAdapterError, match="has no declared frame"):
            frame_for("v9.9")
        with pytest.raises(RetrievalAdapterError, match="has no declared frame"):
            worker_normalisation("v9.9")

    def test_the_worker_rotates_exactly_the_meshes_that_still_need_it(self) -> None:
        assert worker_normalisation("v0.2") == "y_up_to_z_up"
        assert worker_normalisation("v1.3") == "none"

    def test_every_declared_frame_is_complete(self) -> None:
        """A half-filled entry would reach the pipeline as an AttributeError, not as a claim."""
        for version, frame in LIBRARY_FRAMES.items():
            assert frame.source_frame and frame.method, version
            assert frame.alignment in ("rotate_on_read", "prepared_in_world_frame"), version
            assert frame.prepare == (frame.alignment == "prepared_in_world_frame"), version


class TestLicenceRecording:
    def test_licence_absence_is_recorded_rather_than_left_empty(self) -> None:
        # A contract violation would be to ship an asset with no licence; writing "unknown"
        # would hide that the catalogue has no such field, so the absence is spelled out.
        assert LICENCE_UNRECORDED.startswith("unrecorded")
        assert "catalogue" in LICENCE_UNRECORDED


class TestEntityConversion:
    def test_target_size_is_the_midpoint_of_the_requested_range(self) -> None:
        """Aiming at the maximum of a range systematically picks the largest asset in it,
        which for a cup means retrieving a bowl."""
        from rfw_w.scenes.adapters.simready_retrieval import target_size_for

        entity = EntitySpec.from_dict({
            "id": "cup", "category": "cup", "physics_type": "rigid", "mobility": "dynamic",
            "role": "task_target", "size_bounds_m": [[0.06, 0.12], [0.08, 0.10], [0.07, 0.14]],
        })
        size = target_size_for(entity)
        assert size.to_list() == pytest.approx([0.09, 0.09, 0.105])

    def test_missing_bounds_fall_back_to_a_documented_default(self) -> None:
        from rfw_w.scenes.adapters.simready_retrieval import CATEGORY_TARGET_SIZE_M, target_size_for

        entity = EntitySpec.from_dict({
            "id": "cup", "category": "cup", "physics_type": "rigid", "mobility": "dynamic",
        })
        assert target_size_for(entity) == CATEGORY_TARGET_SIZE_M["cup"]


class TestExplicitPlateCoarseFrame:
    def test_swapped_axis_only_passes_explicit_coarse_opt_in(self):
        from rfw_w.scenes.adapters.simready_retrieval import coarse_size_violation
        actual,target=Vec3(.22,.025,.22),Vec3(.22,.22,.03)
        assert coarse_size_violation(actual,target,2) is not None
        assert coarse_size_violation(actual,target,2,allow_dimension_permutation=True) is None

    def test_axis_opt_in_does_not_admit_large_or_cube_candidate(self):
        from rfw_w.scenes.adapters.simready_retrieval import coarse_size_violation
        target=Vec3(.22,.22,.03)
        for actual in (Vec3(.6,.6,.03),Vec3(.15,.15,.15)):
            assert coarse_size_violation(actual,target,2,allow_dimension_permutation=True) is not None
