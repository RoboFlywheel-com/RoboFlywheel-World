# SPDX-License-Identifier: Apache-2.0
"""Synthetic catalogues test retrieval without another source checkout or model download."""

import json
from pathlib import Path
import sys

import pytest

from rfw_w.scenes.adapters.simready_retrieval import (
    SimreadyConfig, RetrievalAdapterError, build_engine, retrieve_entity,
)
from rfw_w.scenes.contracts.scene_spec import EntitySpec


def test_retrieval_measures_geometry_and_never_imports_external_project(tmp_path: Path):
    assets = tmp_path / "assets"
    mesh = assets / "0/0/0_0_0000/mesh"
    mesh.mkdir(parents=True)
    # Source is Y-up. Reported world extents must exchange Y/Z once.
    (mesh / "0_0_0000.obj").write_text("v 0 0 0\nv .09 .11 .08\nf 1 2 1\n")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"_meta": {"library_version": "v1.3"},
        "0_0_0000": {"leaf": "0_0_0000", "assets_prefix": "0/0/0_0_0000",
                     "super_category": "Mug", "physics_type": "rigid",
                     "dimensions_m": [.09, .11, .08], "real_bbox": [.09, .08, .11],
                     "mass": .2, "mu1": .5, "mu2": .4}}))
    config = SimreadyConfig(index, assets, library_version="v1.3")
    engine = build_engine(config, scoring="lexical")
    entity = EntitySpec.from_dict({"id": "cup", "category": "Mug", "physics_type": "rigid",
                                  "mobility": "dynamic", "role": "task_target",
                                  "size_bounds_m": [[.08,.1],[.07,.09],[.1,.12]]})
    outcome = retrieve_entity(engine, entity, config=config, admitted_categories=("Mug",))
    assert outcome.ok, outcome.detail
    assert outcome.candidates[0].dimensions_m.to_list() == [.09, .08, .11]
    assert not any(name == "pvscene" or name.startswith("pvscene.") for name in sys.modules)


def test_retrieval_accepts_release_sequence_numbers_above_9999(tmp_path: Path):
    """Release sequence numbers are padded to four digits, not truncated to four."""
    assets = tmp_path / "assets"
    mesh = assets / "24/40/24_40_10000/mesh"
    mesh.mkdir(parents=True)
    (mesh / "24_40_10000.obj").write_text("v 0 0 0\nv .09 .11 .08\nf 1 2 1\n")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"_meta": {"library_version": "v1.3"},
        "24_40_10000": {"leaf": "24_40_10000", "assets_prefix": "24/40/24_40_10000",
                        "super_category": "Mug", "dimensions_m": [.09, .11, .08],
                        "mass": .2, "mu1": .5}}))
    config = SimreadyConfig(index, assets, library_version="v1.3")
    entity = EntitySpec.from_dict({"id": "cup", "category": "Mug", "physics_type": "rigid",
                                  "mobility": "dynamic", "role": "task_target",
                                  "size_bounds_m": [[.08, .1], [.07, .09], [.1, .12]]})
    outcome = retrieve_entity(build_engine(config, scoring="lexical"), entity,
                              config=config, admitted_categories=("Mug",))
    assert outcome.ok, outcome.detail
    assert outcome.candidates[0].asset_key.asset_id == "24_40_10000"


def test_semantic_recall_uses_class_vectors_without_lexical_fallback(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    rows = {"_meta": {"library_version": "v1.3"}}
    for key, category in (("0_0_0000", "Mug"), ("0_1_0000", "Bowl")):
        mesh = assets / "/".join((*key.split("_")[:2], key)) / "mesh"
        mesh.mkdir(parents=True)
        (mesh / f"{key}.obj").write_text("v 0 0 0\nv .1 .1 .1\nf 1 2 1\n")
        rows[key] = {"assets_prefix": str(mesh.parent.relative_to(assets)),
                     "super_category": category, "dimensions_m": [.1,.1,.1],
                     "real_bbox": [.1,.1,.1], "mass": .2, "mu1": .5}
    index = tmp_path / "index.json"
    index.write_text(json.dumps(rows))
    semantic = tmp_path / "classes.json"
    semantic.write_text(json.dumps({"schema_version": "pvscene.semantic-class-index/v1",
        "model_id": "synthetic-two-dimensional", "classes": [["", "Mug"], ["", "Bowl"]],
        "text_features": [[1.,0.], [0.,1.]]}))

    class Encoder:
        model_id = "synthetic-two-dimensional"

        def encode(self, texts):
            return [[1.,0.] for _ in texts]

    config = SimreadyConfig(index, assets, semantic_index_path=semantic, library_version="v1.3")
    engine = build_engine(config, encoder=Encoder())
    entity = EntitySpec.from_dict({"id": "vessel", "category": "drinking vessel",
                                  "physics_type": "rigid", "mobility": "dynamic"})
    outcome = retrieve_entity(engine, entity, config=config)
    assert outcome.ok
    assert [candidate.super_category for candidate in outcome.candidates] == ["Mug"]
    assert any(reason == "SEMANTIC_BELOW_THRESHOLD" for _, reason in outcome.rejected)
    with pytest.raises(RetrievalAdapterError, match="requires an encoder"):
        build_engine(config)


def test_thin_source_mesh_is_ranked_in_the_same_frame_as_measured_admission(tmp_path):
    assets = tmp_path / "assets"
    mesh = assets / "0/0/0_0_0000/mesh"
    mesh.mkdir(parents=True)
    (mesh / "0_0_0000.obj").write_text("v 0 0 0\nv .3 .02 .3\nf 1 2 1\n")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"_meta": {"library_version": "v1.3"}, "0_0_0000": {
        "assets_prefix": "0/0/0_0_0000", "super_category": "Plate",
        "dimensions_m": [.3,.02,.3], "real_bbox": [.3,.02,.3], "mass": .2}}))
    config = SimreadyConfig(index, assets, library_version="v1.3")
    entity = EntitySpec.from_dict({"id":"plate", "category":"Plate", "physics_type":"rigid",
        "mobility":"dynamic", "size_bounds_m":[[.28,.32],[.28,.32],[.018,.022]]})
    outcome = retrieve_entity(build_engine(config, scoring="lexical"), entity, config=config)
    assert outcome.ok, outcome.detail
    assert outcome.candidates[0].dimensions_m.to_list() == [.3,.3,.02]


@pytest.mark.parametrize("vector", [[float("nan"), 1.], [float("inf"), 0.], [0.,0.]])
def test_invalid_semantic_vectors_are_rejected(tmp_path, vector):
    from rfw_w.scenes.assets.semantic_index import load_semantic_class_index
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"schema_version":"pvscene.semantic-class-index/v1",
        "model_id":"synthetic", "classes":[["", "Mug"]], "text_features":[vector]}))
    with pytest.raises(ValueError, match="finite|zero"):
        load_semantic_class_index(path)


def test_exact_category_name_does_not_override_semantic_rejection(tmp_path):
    from rfw_w.scenes.assets.retrieval_engine import ReleaseCatalog, RetrievalEngine, RetrievalGoal
    from rfw_w.scenes.contracts.scene_spec import PhysicsType
    from rfw_w.scenes.contracts.units import Vec3
    assets = tmp_path / "assets"
    assets.mkdir()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"_meta":{"library_version":"v1.3"}, "0_0_0000":{
        "assets_prefix":"0/0/0_0_0000", "super_category":"Mug",
        "dimensions_m":[.1,.1,.1], "real_bbox":[.1,.1,.1]}}))

    class Scorer:
        def score_many(self, terms, labels):
            return {label:0. for label in labels}

    config = SimreadyConfig(index, assets, library_version="v1.3")
    result = RetrievalEngine(ReleaseCatalog(config, scorer=Scorer())).retrieve(
        RetrievalGoal("cup", "Mug", "Mug", PhysicsType.RIGID, Vec3(.1,.1,.1)))
    assert not result.candidates
    assert result.rejected == (("0_0_0000", "SEMANTIC_BELOW_THRESHOLD"),)


def test_same_dimension_encoder_from_another_model_is_refused(tmp_path):
    from rfw_w.scenes.assets.semantic_index import SemanticClassIndex, SbertSemanticScorer

    class Encoder:
        model_id = "another-model"

    index = SemanticClassIndex((("", "Mug"),), [[1.,0.]], "expected-model", "hash", tmp_path)
    with pytest.raises(ValueError, match="identity"):
        SbertSemanticScorer(index, encoder=Encoder())


def test_index_built_with_local_encoder_can_be_read_by_the_same_encoder(tmp_path):
    from rfw_w.scenes.assets.semantic_index import (
        build_semantic_class_index, load_semantic_class_index, SbertSemanticScorer,
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"rfw_model_id":"synthetic-model"}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"asset":{"super_category":"Mug"}}))

    class Encoder:
        model_id = str(model)

        def encode(self, texts):
            return [[1.,0.] for _ in texts]

    path = build_semantic_class_index(catalog, tmp_path / "classes.json", encoder=Encoder())
    index = load_semantic_class_index(path)
    assert index.model_id == "synthetic-model"
    assert SbertSemanticScorer(index, encoder=Encoder()).score(("Mug",), "Mug") == 1.
