# SPDX-License-Identifier: Apache-2.0
"""Immutable background evidence is rechecked after copying delivery to a new directory."""
import json
import shutil

import pytest

from rfw_w.scenes.validation.verify_delivery import VerificationFailure, verify_generated_assembly


def record(root):
    generation = "a" * 64
    name = f"generated_furniture/{generation}/attempt_001/assembly.json"
    assembly = {"generation_hash": generation, "attempt": 1, "artifact_path": name, "status": "no-kept-objects"}
    artifact = root / name
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(assembly))
    return {"background": {"assembly": assembly}}, artifact


def test_copied_evidence_can_be_verified_and_tampering_is_rejected(tmp_path):
    source = tmp_path / "original"
    manifest, artifact = record(source)
    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    verify_generated_assembly(copied, manifest)
    (copied / artifact.relative_to(source)).write_text("{}")
    with pytest.raises(VerificationFailure, match="differs"):
        verify_generated_assembly(copied, manifest)


def test_missing_and_external_artifacts_fail(tmp_path):
    manifest, artifact = record(tmp_path)
    artifact.unlink()
    with pytest.raises(VerificationFailure, match="missing"):
        verify_generated_assembly(tmp_path, manifest)
    manifest["background"]["assembly"]["artifact_path"] = "../outside.json"
    with pytest.raises(VerificationFailure):
        verify_generated_assembly(tmp_path, manifest)
