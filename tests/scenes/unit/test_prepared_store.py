# SPDX-License-Identifier: Apache-2.0
"""Blob identification for the local model preparation.

The failure this guards against actually happened: a 438 MB safetensors blob was named
``pytorch_model.bin`` on the assumption that large means torch, and the resulting load error
pointed at the file rather than at the misnaming. So the rules read the container format, and
here they are tested against synthetic blobs of each shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfw_w.scenes.assets.prepared_store import (
    PreparationError,
    identify_blob,
    identify_blobs,
    materialize_local_model,
    snapshot_tree_present,
)


def safetensors(path: Path, header: dict | None = None) -> Path:
    payload = json.dumps(header or {"__metadata__": {"format": "pt"}}).encode()
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + b"\x00" * 32)
    return path


def torch_zip(path: Path) -> Path:
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    return path


def legacy_pickle(path: Path) -> Path:
    path.write_bytes(b"\x80\x04" + b"\x00" * 64)
    return path


def json_blob(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


class TestWeightFormats:
    def test_safetensors_is_not_called_pytorch_model_bin(self, tmp_path: Path) -> None:
        """The bug that motivated content-based identification."""
        assert identify_blob(safetensors(tmp_path / "b")) == "model.safetensors"

    def test_torch_zip_is_recognised(self, tmp_path: Path) -> None:
        assert identify_blob(torch_zip(tmp_path / "b")) == "pytorch_model.bin"

    def test_legacy_pickle_is_recognised(self, tmp_path: Path) -> None:
        assert identify_blob(legacy_pickle(tmp_path / "b")) == "pytorch_model.bin"

    def test_arbitrary_binary_is_not_a_weight_file(self, tmp_path: Path) -> None:
        blob = tmp_path / "b"
        blob.write_bytes(b"\xde\xad\xbe\xef" * 32)
        assert identify_blob(blob) is None


class TestConfigFormats:
    def test_readme(self, tmp_path: Path) -> None:
        blob = tmp_path / "b"
        blob.write_text("---\nlanguage: en\nlicense: apache-2.0\n---\n# Model\n")
        assert identify_blob(blob) == "README.md"

    def test_vocab(self, tmp_path: Path) -> None:
        blob = tmp_path / "b"
        blob.write_text("<s> <pad> </s> <unk> [PAD] [unused0]\n[UNK]\n")
        assert identify_blob(blob) == "vocab.txt"

    def test_tokenizer_config(self, tmp_path: Path) -> None:
        blob = json_blob(tmp_path / "b", {"do_lower_case": True, "bos_token": "<s>"})
        assert identify_blob(blob) == "tokenizer_config.json"

    def test_special_tokens_map(self, tmp_path: Path) -> None:
        blob = json_blob(tmp_path / "b", {"bos_token": "<s>", "unk_token": "[UNK]"})
        assert identify_blob(blob) == "special_tokens_map.json"

    def test_model_config(self, tmp_path: Path) -> None:
        blob = json_blob(tmp_path / "b", {"architectures": ["MPNetForMaskedLM"], "model_type": "mpnet"})
        assert identify_blob(blob) == "config.json"

    def test_pooling_config(self, tmp_path: Path) -> None:
        blob = json_blob(tmp_path / "b", {"word_embedding_dimension": 768, "pooling_mode_mean_tokens": True})
        assert identify_blob(blob) == "1_Pooling/config.json"

    def test_modules_list(self, tmp_path: Path) -> None:
        blob = json_blob(tmp_path / "b", [{"idx": 0, "name": "0", "type": "sentence_transformers.models.Transformer"}])
        assert identify_blob(blob) == "modules.json"

    def test_tokenizer_json(self, tmp_path: Path) -> None:
        blob = json_blob(tmp_path / "b", {"truncation": {}, "padding": {}, "model": {"type": "WordPiece"}})
        assert identify_blob(blob) == "tokenizer.json"

    def test_unknown_json_is_not_named(self, tmp_path: Path) -> None:
        blob = json_blob(tmp_path / "b", {"something": "unexpected"})
        assert identify_blob(blob) is None


def build_cache(root: Path) -> Path:
    """A minimal cache entry shaped like the real one, minus the snapshot tree."""
    model = root / "hub" / "models--sentence-transformers--all-mpnet-base-v2"
    (model / "blobs").mkdir(parents=True)
    (model / "refs").mkdir()
    (model / "refs" / "main").write_text("e8c3b32edf5434bc2275fc9bab85f82640a19130\n")
    json_blob(model / "blobs" / "aaa", {"architectures": ["MPNetForMaskedLM"]})
    json_blob(model / "blobs" / "bbb", {"do_lower_case": True, "bos_token": "<s>"})
    (model / "blobs" / "ccc").write_text("<s> <pad> </s> [PAD]\n")
    safetensors(model / "blobs" / "ddd")
    return model


class TestSnapshotDetection:
    def test_no_snapshots_directory_means_not_present(self, tmp_path: Path) -> None:
        assert not snapshot_tree_present(build_cache(tmp_path))

    def test_empty_snapshots_directory_is_not_present(self, tmp_path: Path) -> None:
        # An empty snapshots directory is exactly the state that makes local_files_only fail,
        # so it must not read as usable.
        model = build_cache(tmp_path)
        (model / "snapshots").mkdir()
        assert not snapshot_tree_present(model)

    def test_populated_snapshot_is_present(self, tmp_path: Path) -> None:
        model = build_cache(tmp_path)
        snapshot = model / "snapshots" / "e8c3b32e"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        assert snapshot_tree_present(model)


class TestIdentificationFailures:
    def test_unidentified_blob_is_an_error_not_a_guess(self, tmp_path: Path) -> None:
        """A mis-named file loads as a wrong model rather than failing, so guessing is worse
        than stopping."""
        model = build_cache(tmp_path)
        (model / "blobs" / "eee").write_bytes(b"\xde\xad\xbe\xef" * 16)
        with pytest.raises(PreparationError, match="could not identify"):
            identify_blobs(model)

    def test_ambiguous_blobs_are_rejected(self, tmp_path: Path) -> None:
        model = build_cache(tmp_path)
        json_blob(model / "blobs" / "fff", {"architectures": ["BertForMaskedLM"]})
        with pytest.raises(PreparationError, match="both look like"):
            identify_blobs(model)

    def test_missing_blobs_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(PreparationError, match="no blobs directory"):
            identify_blobs(tmp_path)

    def test_all_blobs_in_a_healthy_cache_are_identified(self, tmp_path: Path) -> None:
        identified = identify_blobs(build_cache(tmp_path))
        assert "model.safetensors" in identified
        assert "config.json" in identified


class TestMaterialisation:
    def test_builds_a_loadable_directory(self, tmp_path: Path) -> None:
        model = build_cache(tmp_path)
        target = tmp_path / "prepared"
        record = materialize_local_model(model, target)
        assert (target / "config.json").is_file()
        assert (target / "model.safetensors").is_file()
        assert "model.safetensors" in record.files

    def test_record_names_the_source_blob_for_every_file(self, tmp_path: Path) -> None:
        # Provenance is the point: a reviewer must be able to trace each file back.
        record = materialize_local_model(build_cache(tmp_path), tmp_path / "prepared")
        for name, origin in record.files.items():
            assert origin.startswith("symlink <- blobs/") or origin.startswith("copy <- blobs/"), (
                name,
                origin,
            )

    def test_refuses_when_the_cache_is_already_loadable(self, tmp_path: Path) -> None:
        model = build_cache(tmp_path)
        snapshot = model / "snapshots" / "e8c3b32e"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        with pytest.raises(PreparationError, match="already has a snapshots tree"):
            materialize_local_model(model, tmp_path / "prepared")

    def test_refuses_when_a_required_file_is_absent(self, tmp_path: Path) -> None:
        """No substitution and no download: a missing file stops the preparation."""
        model = build_cache(tmp_path)
        (model / "blobs" / "ccc").unlink()  # vocab.txt
        with pytest.raises(PreparationError, match="vocab.txt"):
            materialize_local_model(model, tmp_path / "prepared")

    def test_refuses_when_there_is_no_weight_file(self, tmp_path: Path) -> None:
        model = build_cache(tmp_path)
        (model / "blobs" / "ddd").unlink()  # the weights
        with pytest.raises(PreparationError, match="model.safetensors"):
            materialize_local_model(model, tmp_path / "prepared")

    def test_record_round_trips(self, tmp_path: Path) -> None:
        record = materialize_local_model(build_cache(tmp_path), tmp_path / "prepared")
        assert record.to_dict()["method"] == "reconstruct_local_model_from_hf_blobs"
        assert record.to_dict()["version"] == "v1"
