# SPDX-License-Identifier: Apache-2.0
"""The independent verifier must catch a delivery that overclaims.

A verifier that only passes is worthless, so every test here is a tampered manifest that must fail.
The scenarios are the ways a delivery could claim more than its evidence supports.
"""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFIER = module_path('rfw_w.scenes.validation.verify_delivery')

VERSION = "a" * 64


@pytest.fixture(scope="module")
def verifier():
    return importlib.import_module('rfw_w.scenes.validation.verify_delivery')


def gate(gate_id: str, status: str, *, version: str = VERSION, **extra) -> dict:
    payload = {
        "gate_id": gate_id, "status": status, "state_hash": version,
        "backend": "isaac", "tool_version": "t/1", "method": "fixture",
        "witness_refs": [],
    }
    if status == "pass":
        payload["threshold_profile"] = "engineering_v1"
    payload.update(extra)
    return payload


REQUIRED = [
    "semantics.hard_entities_preserved", "semantics.task_structure_covered",
    "semantics.randomization_invariants",
    "geometry.task_cell_clear", "geometry.no_penetration", "geometry.supports",
    "geometry.in_bounds", "physics.settle_stability", "physics.clock_advanced",
    "visual.required_views", "visual.critique", "function.initial_relations",
    "function.space_conditions", "function.task_reachable", "shell.matches_plan",
]


def manifest(**overrides) -> dict:
    """A manifest v2, which embeds the resolution whole.

    v1 restated it flat under `requested_readiness`/`achieved_readiness`/`meets_request`, which is
    the hand-copy the nesting removes. The fixture follows the producer, so a change to the shape
    has to be made in both places rather than only in the code the tests do not exercise.
    """
    payload = {
        "scene_version": VERSION,
        "resolution": {
            "requested": "functional_scene_ready",
            "achieved": "functional_scene_ready",
            "meets_request": True,
            "blockers": [],
        },
        "requested_backends": ["isaac"],
        "gates": [gate(name, "pass") for name in REQUIRED] + [
            gate("backend.isaac.load", "pass"), gate("backend.isaac.run", "pass"),
            gate("backend.isaac.render", "pass"),
        ],
        "not_run": [], "unsupported": [],
        # A visual pass is verified against the critique that produced it, so the fixture carries a
        # matching one; otherwise every test trips that check before reaching its own.
        "visual_critique": {"critique": {"reviewed_scene_hash": VERSION}},

    }
    payload.update(overrides)
    return payload


def write(tmp_path: Path, payload: dict, *, with_isaac: bool = True) -> Path:
    """Write a manifest, and by default the worker artefacts a render pass depends on.

    Without them the artefact check fires before whichever check a test is aiming at, and the test
    would pass for the wrong reason.
    """
    (tmp_path / "delivery_manifest.json").write_text(json.dumps(payload))
    if with_isaac:
        isaac = tmp_path / "isaac"
        isaac.mkdir(exist_ok=True)
        for name in ("overview.png", "overview_depth_m.npy", "overview_instance_ids.npy"):
            (isaac / name).write_bytes(b"")
        (isaac / "report.json").write_text(json.dumps({
            "scene_version": payload["scene_version"],
            "captures": [{
                "rgb_path": "overview.png", "depth_path": "overview_depth_m.npy",
                "instance_id_path": "overview_instance_ids.npy",
            }],
            "instance_id_map_source": "annotator_idToLabels",
        }))
    return tmp_path


class TestAccepts:
    def test_render_of_another_scene_is_rejected(self, verifier, tmp_path: Path) -> None:
        delivery = write(tmp_path, manifest())
        report_path = delivery / "isaac" / "report.json"
        report = json.loads(report_path.read_text())
        report["scene_version"] = "b" * 64
        report_path.write_text(json.dumps(report))
        with pytest.raises(verifier.VerificationFailure, match="different scene version"):
            verifier.verify(delivery)

    def test_a_consistent_delivery_verifies(self, verifier, tmp_path: Path) -> None:
        summary = verifier.verify(write(tmp_path, manifest()))
        assert summary["verified"] is True
        assert summary["recomputed_readiness"] == "functional_scene_ready"

    def test_a_delivery_with_declared_gaps_still_verifies(self, verifier, tmp_path: Path) -> None:
        """Declared gaps are the point: they do not stop a delivery, they bound its claim."""
        payload = manifest()
        payload["gates"].append(gate("robot.ik", "not_run"))
        payload["not_run"] = ["robot.ik"]
        assert verifier.verify(write(tmp_path, payload))["verified"] is True


class TestRejects:
    def test_a_missing_manifest_is_a_failure(self, verifier, tmp_path: Path) -> None:
        with pytest.raises(verifier.VerificationFailure, match="no manifest"):
            verifier.verify(tmp_path)

    def test_a_tampered_scene_graph_is_rejected(self, verifier, tmp_path: Path) -> None:
        payload = manifest()
        payload["scene_graph"] = {
            "path": "scene_ir.json", "content_hash": "b" * 64, "state": "authored",
        }
        (tmp_path / "scene_ir.json").write_text(json.dumps({"scene_id": "changed"}))
        with pytest.raises(verifier.VerificationFailure, match="content hash"):
            verifier.verify(write(tmp_path, payload))

    def test_a_claim_higher_than_the_gates_support_is_a_failure(
        self, verifier, tmp_path: Path
    ) -> None:
        """The central overclaim: the manifest says more than its evidence does."""
        payload = manifest()
        payload["gates"] = [g for g in payload["gates"] if g["gate_id"] != "function.initial_relations"]
        with pytest.raises(verifier.VerificationFailure, match="its own gates support"):
            verifier.verify(write(tmp_path, payload))

    def test_a_stale_pass_is_a_failure(self, verifier, tmp_path: Path) -> None:
        """A pass measured on another scene version cannot approve this one."""
        payload = manifest()
        for entry in payload["gates"]:
            if entry["gate_id"] == "visual.critique":
                entry["state_hash"] = "b" * 64
        with pytest.raises(verifier.VerificationFailure, match="is recorded as a pass for scene"):
            verifier.verify(write(tmp_path, payload))

    def test_the_specific_diagnosis_precedes_the_generic_one(self, verifier, tmp_path: Path) -> None:
        """A stale pass is a more useful message than "the level came out lower than claimed"."""
        payload = manifest()
        for entry in payload["gates"]:
            if entry["gate_id"] == "geometry.supports":
                entry["state_hash"] = "b" * 64
        with pytest.raises(verifier.VerificationFailure) as raised:
            verifier.verify(write(tmp_path, payload))
        assert "is recorded as a pass for scene" in str(raised.value)

    def test_an_undeclared_failure_is_a_failure(self, verifier, tmp_path: Path) -> None:
        """A gate that failed, is not required at the claimed level, and is not declared as a gap.

        This is the hidden failure: it does not lower the achieved level, so only an explicit scan
        for undeclared non-passes finds it.
        """
        payload = manifest()
        payload["gates"].append(gate("robot.ik", "fail", detail="IK did not converge"))
        with pytest.raises(verifier.VerificationFailure, match="is not declared"):
            verifier.verify(write(tmp_path, payload))

    def test_meets_request_with_blockers_is_a_failure(self, verifier, tmp_path: Path) -> None:
        payload = manifest()
        # Two levels up: a robot gate that was never run blocks robot_feasible.
        payload["resolution"]["requested"] = "robot_feasible_scene_ready"
        payload["resolution"]["achieved"] = "robot_feasible_scene_ready"
        payload["resolution"]["meets_request"] = True
        payload["gates"].append(gate("robot.nav", "not_run"))
        payload["not_run"] = ["robot.nav"]
        with pytest.raises(verifier.VerificationFailure):
            verifier.verify(write(tmp_path, payload))

    def test_a_missing_scene_version_is_a_failure(self, verifier, tmp_path: Path) -> None:
        payload = manifest()
        payload["scene_version"] = ""
        with pytest.raises(verifier.VerificationFailure, match="no scene version"):
            verifier.verify(write(tmp_path, payload))

    def test_missing_requested_backends_is_a_failure(self, verifier, tmp_path: Path) -> None:
        """Inferring them from the gates present would read a declared gap as a requested backend."""
        payload = manifest()
        del payload["requested_backends"]
        payload["gates"].append(gate("backend.mujoco.load", "not_run"))
        payload["not_run"] = ["backend.mujoco.load"]
        with pytest.raises(verifier.VerificationFailure, match="which backends were requested"):
            verifier.verify(write(tmp_path, payload))

    def test_a_pass_with_an_invalid_record_is_a_failure(self, verifier, tmp_path: Path) -> None:
        payload = manifest()
        for entry in payload["gates"]:
            if entry["gate_id"] == "geometry.supports":
                entry["state_hash"] = "not-a-hash"
        with pytest.raises(verifier.VerificationFailure, match="not valid"):
            verifier.verify(write(tmp_path, payload))

    def test_a_fabricated_instance_map_is_a_failure(self, verifier, tmp_path: Path) -> None:
        """A map not sourced from the renderer cannot resolve an id to an entity."""
        payload = manifest()
        directory = write(tmp_path, payload)
        isaac = directory / "isaac"
        (isaac / "report.json").write_text(json.dumps({
            "scene_version": VERSION,
            "captures": [], "instance_id_map_source": "request_instance_order",
        }))
        with pytest.raises(verifier.VerificationFailure, match="own labels"):
            verifier.verify(directory)

    def test_a_capture_naming_a_missing_file_is_a_failure(self, verifier, tmp_path: Path) -> None:
        payload = manifest()
        directory = write(tmp_path, payload)
        isaac = directory / "isaac"
        (isaac / "report.json").write_text(json.dumps({
            "scene_version": VERSION,
            "captures": [{"rgb_path": "missing.png", "depth_path": None, "instance_id_path": None}],
            "instance_id_map_source": "annotator_idToLabels",
        }))
        with pytest.raises(verifier.VerificationFailure, match="not on disk"):
            verifier.verify(directory)

    def test_a_visual_pass_without_a_matching_critique_is_a_failure(
        self, verifier, tmp_path: Path
    ) -> None:
        payload = manifest()
        payload["visual_critique"] = {"critique": {"reviewed_scene_hash": "b" * 64}}
        with pytest.raises(verifier.VerificationFailure, match="not recorded against this scene"):
            verifier.verify(write(tmp_path, payload))

    def test_a_render_pass_without_artefacts_is_a_failure_without_isaac(
        self, verifier, tmp_path: Path
    ) -> None:
        with pytest.raises(verifier.VerificationFailure, match="no worker report"):
            verifier.verify(write(tmp_path, manifest(), with_isaac=False))


@pytest.mark.parametrize("tamper", ["missing_compiled", "changed_compiled", "changed_request", "changed_effective"])
def test_compilation_claim_requires_its_actual_artifacts(verifier, tmp_path, tamper):
    spec = {"scene_id": "test", "seed": 11, "prompt": "pick the cup"}
    record = {"status": "compiled", "compiled_spec_hash": verifier.content_hash(spec),
              "effective_spec_hash": verifier.content_hash(spec),
              "compile_prompt_hash": verifier.content_hash({"prompt": spec["prompt"], "seed": 11})}
    for filename in ("compiled_spec.json", "effective_spec.json", "compile_request.json"):
        (tmp_path / filename).write_text(json.dumps(spec))
    payload = {**spec, "compilation": record}
    verifier.verify_compilation(tmp_path, payload)
    if tamper == "missing_compiled":
        (tmp_path / "compiled_spec.json").unlink()
    else:
        filename = {"changed_compiled": "compiled_spec.json", "changed_request": "compile_request.json",
                    "changed_effective": "effective_spec.json"}[tamper]
        (tmp_path / filename).write_text(json.dumps({**spec, "prompt": "different task"}))
    with pytest.raises(verifier.VerificationFailure):
        verifier.verify_compilation(tmp_path, payload)


class TestShellReference:
    def test_missing_uri_fails_without_isaac(self, verifier, tmp_path):
        delivery = write(tmp_path, manifest())
        (delivery / 'shell_reference.json').write_text(json.dumps({
            'uri': 'missing.usdc', 'scene_version': VERSION,
        }))
        with pytest.raises(verifier.VerificationFailure, match='SHELL_URI_UNRESOLVABLE'):
            verifier.verify(delivery)

    def test_referenced_render_requires_export_evidence(self, verifier, tmp_path):
        delivery = write(tmp_path, manifest())
        path = delivery / 'isaac/report.json'
        report = json.loads(path.read_text())
        report['shell_reference'] = {'mode': 'referenced'}
        path.write_text(json.dumps(report))
        with pytest.raises(verifier.VerificationFailure, match='reference evidence absent'):
            verifier.verify(delivery)

    def test_relocated_self_contained_shell_verifies(self, verifier, tmp_path):
        import hashlib
        import shutil
        pytest.importorskip('pxr')
        from pxr import Usd, UsdGeom
        from rfw_w.scenes.workers.isaac.shell_reference import inspect_shell
        original = tmp_path / 'original'
        original.mkdir()
        payload = manifest(isaac_request_path='request.json')
        write(original, payload)
        path = original / 'shell.usda'
        stage = Usd.Stage.CreateNew(str(path))
        root = UsdGeom.Xform.Define(stage, '/Shell')
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, '/Shell/floor')
        mesh.CreatePointsAttr([(0, 0, 0), (2, 0, 0), (0, 2, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        stage.GetRootLayer().Save()
        path.with_suffix('.export.json').write_text(json.dumps({
            'usd_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_sha256': 'fixture', 'materials': 'fixture', 'openings_measured': 0,
        }))
        record = inspect_shell(str(path), delivery=original)
        (original / 'shell_reference.json').write_text(json.dumps({
            **record, 'uri': 'shell.usda', 'scene_version': VERSION,
        }))
        (original / 'request.json').write_text(json.dumps({'room': {'shell_uri': 'shell.usda'}}))
        report_path = original / 'isaac/report.json'
        report = json.loads(report_path.read_text())
        report['shell_reference'] = record
        report_path.write_text(json.dumps(report))
        relocated = tmp_path / 'relocated'
        shutil.copytree(original, relocated)
        shutil.rmtree(original)
        assert verifier.verify(relocated)['verified']
        # It must actually inspect the geometry after relocation, not trust the old report.
        (relocated / 'shell.usda').unlink()
        with pytest.raises(verifier.VerificationFailure, match='SHELL_URI_UNRESOLVABLE'):
            verifier.verify(relocated)
