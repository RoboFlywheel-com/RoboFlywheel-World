# SPDX-License-Identifier: Apache-2.0
"""The emitted JSON Schemas must stay faithful to the contracts.

Deriving schemas introspectively is only worth it if drift is impossible, so this checks the
two ways it could drift: a field the emitter cannot express (which must be a loud failure, not
a quiet omission), and non-deterministic output (which would make a schema change
unreviewable).
"""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EMITTER = module_path('rfw_w.scenes.preparation.emit_schemas')


@pytest.fixture(scope="module")
def emitter():
    return importlib.import_module('rfw_w.scenes.preparation.emit_schemas')


def test_registry_covers_every_cross_process_contract(emitter) -> None:
    names = {name for name, _ in emitter.REGISTRY}
    assert names == {
        "scene-spec", "asset-record", "scene-ir", "scene-hashes",
        "tool-request", "tool-result", "evidence", "readiness-resolution",
        "delivery-manifest",
    }


def test_every_field_is_expressed(emitter) -> None:
    """An incomplete schema that looks complete is worse than one that admits its gaps.

    A field landing in x-pvscene-unexpressed means the emitter could not describe it, which
    would let a type change slip past schema review.
    """
    gaps: dict[str, list[str]] = {}
    for name, cls in emitter.REGISTRY:
        unexpressed = emitter.build_document(name, cls).get("x-pvscene-unexpressed", [])
        if unexpressed:
            gaps[name] = unexpressed
    assert gaps == {}, f"emitter could not express: {gaps}"


def test_documents_are_deterministic(emitter) -> None:
    for name, cls in emitter.REGISTRY:
        first = emitter.build_document(name, cls)
        second = emitter.build_document(name, cls)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True), name


def test_optional_fields_are_not_required(emitter) -> None:
    # `tolerance_m` has a default, so a spec omitting it is valid.
    document = emitter.build_document("scene-spec", emitter.contracts.SceneSpec)
    assert "tolerance_m" not in json.dumps(document.get("required", []))


def test_required_fields_match_the_constructor(emitter) -> None:
    document = emitter.build_document("evidence", emitter.contracts.Evidence)
    assert set(document["required"]) == {
        "gate_id", "status", "state_hash", "backend", "tool_version", "method",
    }


def test_enums_become_string_enums(emitter) -> None:
    document = emitter.build_document("evidence", emitter.contracts.Evidence)
    assert document["properties"]["status"]["enum"] == ["pass", "fail", "not_run", "unsupported"]


def test_unknown_keys_are_disallowed(emitter) -> None:
    # Constructors reject unexpected keys, so the schema must say so too.
    for name, cls in emitter.REGISTRY:
        assert emitter.build_document(name, cls)["additionalProperties"] is False, name


def test_schema_version_is_pinned_to_the_contract_constant(emitter) -> None:
    document = emitter.build_document("scene-spec", emitter.contracts.SceneSpec)
    assert document["properties"]["schema_version"]["const"] == "pvscene.scene-spec/v3"


def test_the_committed_schemas_are_what_the_emitter_produces(emitter, tmp_path) -> None:
    """Nothing read `schemas/` before this, so a committed schema could drift for as long as nobody
    looked. Two already had: `scene-spec` was missing `StylePlan.coverage_target`, which is in the
    code and was not in the document generated from it.

    Regenerated into a temporary directory and compared byte for byte, because "close enough" is not
    a property a schema has -- a missing property is a field nothing validates.
    """
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", 'rfw_w.scenes.preparation.emit_schemas', "--output-dir", str(tmp_path)],
        check=True, capture_output=True,
    )
    committed = data_path('schemas')
    drifted = []
    for name, _cls in emitter.REGISTRY:
        produced = (tmp_path / f"{name}.schema.json").read_text()
        on_disk = (committed / f"{name}.schema.json")
        if not on_disk.is_file():
            drifted.append(f"{name}: not committed")
        elif on_disk.read_text() != produced:
            drifted.append(f"{name}: differs; run scripts/ops/emit_schemas.py")
    assert not drifted, drifted


def test_the_untyped_holes_are_the_ones_that_were_decided(emitter) -> None:
    """A `Mapping[str, Any]` field emits an object that constrains nothing.

    That is a real hole, and it is the shape a field added carelessly takes -- so the list is pinned
    here. Adding an open field without meaning to fails this test rather than passing unnoticed,
    which is exactly how it stays a decision instead of becoming an oversight.
    """
    document = emitter.build_document("delivery-manifest", emitter.contracts.DeliveryManifest)
    assert document.get("x-pvscene-untyped") == [
        # Compilation contains a local field diff with heterogeneous old/new values, analogous
        # to the existing critic record. It is validated by the compiler before being recorded.
        "assets_used[]", "background", "compilation", "external_calls", "furnishing", "lighting",
        "loop", "placement", "plan", "visual_critique", "zones",
    ]


def test_an_open_object_is_detected_in_both_spellings(emitter) -> None:
    """`Mapping[str, Any]` emits `additionalProperties: {}`; a bare mapping emits `{"type":
    "object"}`. Both accept anything; a detector that misses one reports a clean schema."""
    assert list(emitter._open_object_paths(
        {"properties": {"a": {"type": "object", "additionalProperties": {}}}}
    )) == ["a"]
    assert list(emitter._open_object_paths({"properties": {"a": {"type": "object"}}})) == ["a"]
    assert list(emitter._open_object_paths(
        {"properties": {"a": {"type": "object", "additionalProperties": False, "properties": {}}}}
    )) == []
