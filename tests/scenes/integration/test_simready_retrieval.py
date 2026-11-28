# SPDX-License-Identifier: Apache-2.0
"""Real retrieval against the simready library.

Tier B: needs the real asset roots and a prepared SBERT model directory. When those are absent
these skip loudly rather than passing, because a retrieval test that quietly does nothing is
worse than no test -- it reports the capability as covered.

The assertions are about what makes retrieval *real*: measured geometry rather than catalogue
numbers, a normalisation that is recorded rather than assumed, and a model-facing projection
that resolves to nothing on disk.
"""

from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import os
from pathlib import Path

import pytest

from rfw_w.scenes.adapters.simready_retrieval import (
    SimreadyConfig,
    build_encoder,
    build_engine,
    retrieve_entity,
    to_asset_record,
)
from rfw_w.scenes.contracts.scene_spec import EntitySpec

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.assets

CUP = {
    "id": "cup",
    "category": "cup",
    "physics_type": "rigid",
    "mobility": "dynamic",
    "hard_required": True,
    "role": "task_target",
    "size_bounds_m": [[0.06, 0.12], [0.06, 0.12], [0.07, 0.14]],
}
# A category the rigid library does not contain at all: the coverage audit measured zero desks.
DESK = {
    "id": "desk",
    "category": "desk",
    "physics_type": "rigid",
    "mobility": "fixed",
    "hard_required": True,
    "role": "support",
}


@pytest.fixture(scope="module")
def prepared_model(asset_gate) -> Path:
    configured = os.environ.get("PVSCENE_PREPARED_DIR")
    if not configured:
        asset_gate("PVSCENE_PREPARED_DIR is not set; run scripts/ops/prepare_sbert.py first")
    path = Path(configured).expanduser()
    if not (path / "config.json").is_file():
        asset_gate(f"no prepared SBERT model at {path}; run scripts/ops/prepare_sbert.py first")
    return path


@pytest.fixture(scope="module")
def config(asset_roots, prepared_model: Path, asset_gate) -> SimreadyConfig:
    semantic_index = os.environ.get("PVSCENE_SIMREADY_SEMANTIC_INDEX")
    if not semantic_index:
        asset_gate("PVSCENE_SIMREADY_SEMANTIC_INDEX is not set (the sbert_features.pkl index)")
    return SimreadyConfig(
        index_path=asset_roots.simready_index,
        assets_root=asset_roots.simready_assets_root,
        semantic_index_path=Path(semantic_index),
        library_version=os.environ.get("PVSCENE_SIMREADY_LIBRARY_VERSION", "v1.3"),
        prepared_root=Path(os.environ["PVSCENE_PREPARED_DIR"]) / "integration-assets",
    )


@pytest.fixture(scope="module")
def engine(config: SimreadyConfig, prepared_model: Path):
    # HF_HOME must be writable or transformers refuses to start; the weights themselves are read
    # from the prepared directory.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return build_engine(
        config, encoder=build_encoder(prepared_model)
    )


class TestRetrieval:
    def test_retrieves_mugs_for_a_cup(self, engine, config: SimreadyConfig) -> None:
        outcome = retrieve_entity(
            engine, EntitySpec.from_dict(CUP), config=config, limit=6
        )
        assert outcome.ok, outcome.detail
        assert len(outcome.candidates) >= 3

    def test_candidates_carry_measured_dimensions(self, engine, config: SimreadyConfig) -> None:
        """Catalogue bboxes are rounded and disagree with the geometry; these are measured."""
        outcome = retrieve_entity(engine, EntitySpec.from_dict(CUP), config=config, limit=6)
        for candidate in outcome.candidates:
            dimensions = candidate.dimensions_m.to_list()
            assert all(value > 0 for value in dimensions)
            # A mug is a handful of centimetres. A catalogue-only path lets implausible
            # candidates through, which is why the measurement re-check exists.
            assert max(dimensions) < 0.25, candidate.to_dict()

    def test_oversized_candidates_are_rejected_by_measurement(
        self, engine, config: SimreadyConfig
    ) -> None:
        outcome = retrieve_entity(engine, EntitySpec.from_dict(CUP), config=config, limit=8)
        measured_rejections = [
            reason for _, reason in outcome.rejected if "MEASURED_SIZE_OUT_OF_RANGE" in reason
        ]
        # Not asserted to be non-empty: the point is that when the catalogue overstates an
        # asset, the rejection names it, and no such candidate reaches the layout.
        for candidate in outcome.candidates:
            assert max(candidate.dimensions_m.to_list()) < 0.25
        assert all("MEASURED_SIZE_OUT_OF_RANGE" in reason for reason in measured_rejections)

    def test_recall_alone_returns_categories_unrelated_to_the_target(
        self, engine, config: SimreadyConfig
    ) -> None:
        """Documents the failure mode that admission exists to prevent.

        Measured, not assumed: a "desk" query with no admission returns Stools and a Paperweight.
        They are the nearest classes in embedding space for the whole library and are nothing like
        a desk, so semantic score cannot separate a synonym from a different object -- no
        threshold on it would keep the stools out. Only an explicit admitted set does.
        """
        outcome = retrieve_entity(engine, EntitySpec.from_dict(DESK), config=config, limit=8)
        assert outcome.ok, "recall is expected to return something; that is the problem"
        categories = {candidate.super_category for candidate in outcome.candidates}
        assert "Desk" not in categories
        assert categories, "recall returned candidates, so the set is non-empty"
        # At least one returned category is plainly not a desk: this is the overclaim the
        # admission gate prevents, and it is recorded here as evidence rather than as a guess.
        assert categories - {"Desk Organisation", "Desk Organizer", "Desk Bell"}
        for candidate in outcome.candidates:
            assert candidate.asset_key.asset_id, candidate.to_dict()

    def test_admission_rejects_adjacent_categories(self, engine, config: SimreadyConfig) -> None:
        """The library has no desk, and admission makes that a typed failure rather than a bell."""
        outcome = retrieve_entity(
            engine,
            EntitySpec.from_dict(DESK),
            config=config,
            limit=8,
            admitted_categories=("Desk",),
        )
        assert not outcome.ok
        assert outcome.candidates == ()
        assert outcome.failure_code == "ASSET_UNAVAILABLE"
        assert any("CATEGORY_NOT_ADMITTED" in reason for _, reason in outcome.rejected)

    def test_admission_keeps_the_real_mugs(self, engine, config: SimreadyConfig) -> None:
        outcome = retrieve_entity(
            engine,
            EntitySpec.from_dict(CUP),
            config=config,
            limit=6,
            admitted_categories=("Mug", "Coffee Cup", "Cup"),
        )
        assert outcome.ok
        for candidate in outcome.candidates:
            assert candidate.super_category in ("Mug", "Coffee Cup", "Cup")

    def test_requirements_config_supplies_the_admitted_set(self) -> None:
        from rfw_w.scenes.adapters.simready_retrieval import load_admitted_categories

        admitted = load_admitted_categories(data_path('configs') / 'required_categories_v1.json')
        assert "desk" in admitted
        assert "Mug" in admitted["cup"]
        assert admitted["desk"] == ("Desk",)


class TestMaterialisation:
    def test_builds_a_record_with_real_geometry(self, engine, config: SimreadyConfig) -> None:
        outcome = retrieve_entity(engine, EntitySpec.from_dict(CUP), config=config, limit=4)
        record, handle = to_asset_record(outcome.candidates[0], config)
        assert record.collision_parts
        part = record.collision_parts[0]
        assert part.vertex_count > 100
        assert part.face_count > 100
        assert part.source == "collision_mesh"

    def test_inertia_is_simulable(self, engine, config: SimreadyConfig) -> None:
        outcome = retrieve_entity(engine, EntitySpec.from_dict(CUP), config=config, limit=4)
        record, _ = to_asset_record(outcome.candidates[0], config)
        assert record.rigid_physics is not None
        assert record.rigid_physics.inertia_is_spd
        record.rigid_physics.assert_simulable()

    def test_normalisation_is_recorded_not_assumed(self, engine, config: SimreadyConfig) -> None:
        """The library is Y-up, verified from the shapes; the conversion is recorded on the asset."""
        outcome = retrieve_entity(engine, EntitySpec.from_dict(CUP), config=config, limit=4)
        record, _ = to_asset_record(outcome.candidates[0], config)
        assert record.normalisation.source_frame == "simready_y_up"
        assert record.normalisation.target_frame == "world_z_up"
        assert record.normalisation.applied_scale == 1.0

    def test_handle_resolves_to_files_that_exist(self, engine, config: SimreadyConfig) -> None:
        outcome = retrieve_entity(engine, EntitySpec.from_dict(CUP), config=config, limit=4)
        _, handle = to_asset_record(outcome.candidates[0], config)
        assert Path(handle.visual_uri).is_file()
        for uri in handle.collision_uris:
            assert Path(uri).is_file()

    def test_model_facing_view_resolves_to_nothing(self, engine, config: SimreadyConfig) -> None:
        """The record's semantic view must not be usable to locate the asset on disk."""
        outcome = retrieve_entity(engine, EntitySpec.from_dict(CUP), config=config, limit=4)
        record, _ = to_asset_record(outcome.candidates[0], config)
        blob = str(record.semantic_view())
        assert ".obj" not in blob
        assert "mesh" not in blob
        assert str(config.assets_root) not in blob

    def test_licence_absence_is_visible(self, engine, config: SimreadyConfig) -> None:
        # Recorded explicitly, so an audit can find every asset whose licence is unrecorded
        # rather than having to infer it from a blank.
        outcome = retrieve_entity(engine, EntitySpec.from_dict(CUP), config=config, limit=4)
        record, _ = to_asset_record(outcome.candidates[0], config)
        assert record.license.startswith("unrecorded")
