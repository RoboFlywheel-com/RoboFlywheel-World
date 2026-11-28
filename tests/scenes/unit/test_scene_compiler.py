# SPDX-License-Identifier: Apache-2.0
"""Compiler boundary tests: model responses are untrusted and calls are budgeted."""
from rfw_w.scenes.resources import data_path, module_path
import importlib
import rfw_w.scenes.application as application_module
import rfw_w.scenes.assembly.placement as placement_module
import rfw_w.scenes.commands.generate as generate_module
import json
import sys
from pathlib import Path
from dataclasses import replace

import pytest

from rfw_w.scenes.agents import scene_compiler as compiler
from rfw_w.scenes.agents.llm_client import LLMConfig, LLMResponse, LLMError
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.orchestrator.budgets import Budget, BudgetExhausted
from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.furnishing_plan import plan_from_spec

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ("office_pick_cup", "kitchen_fetch_jar", "bedroom_pick_alarm_clock")


def spec_for(name="office_pick_cup"):
    return SceneSpec.from_dict(json.loads((data_path('examples') / name / 'task.json').read_text()))


def response_for(spec):
    return {"prompt_hash": compiler.prompt_hash(spec.prompt, spec.seed),
            "spec": spec.to_dict(), "assumptions": [], "gaps": []}


def parse(doc, spec):
    return compiler.parse_compilation(json.dumps(doc), prompt=spec.prompt, seed=spec.seed, scene_id=spec.scene_id)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_golden_response_drives_the_existing_core(name):
    spec = spec_for(name)
    compiled, _ = parse(response_for(spec), spec)
    assert compiled == spec
    assert plan_from_spec(compiled).target_id == spec.task.targets[0]
    assert solve_cell(compiled, seed=spec.seed).core_hash == solve_cell(spec, seed=spec.seed).core_hash


@pytest.mark.parametrize("mutation,reason", [
    (lambda d: d.update(prompt_hash="wrong"), "PROMPT_HASH_MISMATCH"),
    (lambda d: d["spec"].update(seed=True), "INPUT_IDENTITY_CHANGED"),
    (lambda d: d["spec"]["entities"][0].update(category="unknown_furniture"), "CATEGORY_UNKNOWN"),
    (lambda d: d["spec"]["entities"][0].update(hard_required="false"), "BOOLEAN_REQUIRED"),
    (lambda d: d["spec"]["entities"][0].update(hard_required=False), "CORE_NOT_REQUIRED"),
    (lambda d: d["spec"]["entities"][0].update(material_tokens=["unknown_material"]), "MATERIAL_UNKNOWN"),
    (lambda d: d["spec"]["entities"][0].update(quantity=2), "QUANTITY_UNSUPPORTED"),
    (lambda d: d["spec"]["task"].update(robot_profile="fixed_piper"), "ROBOT_UNSUPPORTED"),
    (lambda d: d["spec"].update(requested_readiness="draft"), "READINESS_OR_BACKEND_UNSUPPORTED"),
    (lambda d: d["spec"]["room"].update(size_bounds_m=[[1, 4], [3, 4], [2.6, 3]]), "SIZE_OUT_OF_RANGE"),
    (lambda d: d["spec"]["initial_relations"][0].update(object="missing"), "SPEC_INVALID"),
    (lambda d: d["spec"]["initial_relations"][0].update(hard=False), "HARD_ON_REQUIRED"),
    (lambda d: d["spec"]["task"].update(targets=["cup", "desk"]), "TARGET_NOT_UNIQUE"),
])
def test_invalid_response_is_named(mutation, reason):
    spec = spec_for()
    doc = response_for(spec)
    mutation(doc)
    with pytest.raises(compiler.CompileError, match=reason):
        parse(doc, spec)


def test_duplicate_keys_and_nan_are_not_json_evidence():
    spec = spec_for()
    for text, reason in [('{"spec":{},"spec":{}}', "DUPLICATE_JSON_KEY"), ('{"spec":NaN}', "NONFINITE_NUMBER")]:
        with pytest.raises(compiler.CompileError, match=reason):
            compiler.parse_compilation(text, prompt=spec.prompt, seed=spec.seed, scene_id=spec.scene_id)


def test_no_permission_means_no_call_and_no_charge(monkeypatch):
    monkeypatch.delenv("PVSCENE_ALLOW_EXTERNAL_VLM", raising=False)
    monkeypatch.setattr(compiler, "call_vision", lambda *a, **k: pytest.fail("call must not occur"))
    spec, budget = spec_for(), Budget()
    with pytest.raises(SystemExit, match="PVSCENE_ALLOW_EXTERNAL_VLM"):
        compiler.compile_scene(LLMConfig("https://example.invalid", "test", "UNUSED"),
                               prompt=spec.prompt, seed=spec.seed, scene_id=spec.scene_id,
                               on_request=budget.charge_model_request)
    assert budget.model_requests_used == 0


def test_transport_failure_is_recorded_and_spends_one_request(monkeypatch):
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    def fail(*a, **k):
        raise LLMError("provider detail must not be stored")
    monkeypatch.setattr(compiler, "call_vision", fail)
    spec, budget = spec_for(), Budget()
    result = compiler.compile_scene(LLMConfig("https://example.invalid", "test", "UNUSED"),
                                    prompt=spec.prompt, seed=spec.seed, scene_id=spec.scene_id,
                                    on_request=budget.charge_model_request)
    assert result.spec is None and result.record["reason"] == "LLMError"
    assert budget.model_requests_used == 1
    assert "provider detail" not in json.dumps(result.record)


def test_exhausted_budget_cannot_send(monkeypatch):
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.setattr(compiler, "call_vision", lambda *a, **k: pytest.fail("no budget"))
    spec, budget = spec_for(), Budget(max_model_requests=1, model_requests_used=1)
    with pytest.raises(BudgetExhausted):
        compiler.compile_scene(LLMConfig("https://example.invalid", "test", "UNUSED"),
                               prompt=spec.prompt, seed=spec.seed, scene_id=spec.scene_id,
                               on_request=budget.charge_model_request)


def test_diff_exposes_changed_semantics():
    spec = spec_for()
    altered = replace(spec, seed=42)
    assert compiler.field_diff(spec.to_dict(), altered.to_dict()) == [
        {"path": "/seed", "requested": 11, "compiled": 42}]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_prompt_cli_writes_compiled_spec_and_real_geometry_preview(name, monkeypatch, tmp_path):
    tmp_path = tmp_path / "run"
    spec = spec_for(name)
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.setenv("PVSCENE_VLM_BASE_URL", "https://example.invalid")
    monkeypatch.setattr(compiler, "call_vision", lambda *a, **k: LLMResponse(json.dumps(response_for(spec)), "mock"))
    monkeypatch.setattr(sys, "argv", ["run_m1_acceptance", "--prompt", spec.prompt,
                        "--scene-id", spec.scene_id, "--seed", str(spec.seed),
                        "--compiler-model", "mock", "--compile-only", "--output-dir", str(tmp_path)])
    assert generate_module.main() == 0
    assert SceneSpec.from_dict(json.loads((tmp_path / "compiled_spec.json").read_text())) == spec
    record = json.loads((tmp_path / "compilation.json").read_text())
    assert record["status"] == "compiled" and record["preview"]["status"] == "ran"
    assert record["model_requests"] == 1
    assert (tmp_path / "core_room.svg").exists()


def test_cli_failed_compile_without_reference_stops_before_build(monkeypatch, tmp_path):
    tmp_path = tmp_path / "run"
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.setenv("PVSCENE_VLM_BASE_URL", "https://example.invalid")
    monkeypatch.setattr(compiler, "call_vision", lambda *a, **k: LLMResponse("{}", "mock"))
    monkeypatch.setattr(sys, "argv", ["run_m1_acceptance", "--prompt", "a task", "--compiler-model", "mock",
                                    "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="no reference spec"):
        generate_module.main()
    assert json.loads((tmp_path / "compilation.json").read_text())["status"] == "failed"
    assert not (tmp_path / "effective_spec.json").exists()


def test_empty_prompt_is_a_recorded_failure_without_a_request(monkeypatch):
    monkeypatch.setattr(compiler, "call_vision", lambda *a, **k: pytest.fail("empty prompt"))
    result = compiler.compile_scene(LLMConfig("https://example.invalid", "test", "UNUSED"),
                                    prompt="", seed=11, scene_id="test", on_request=lambda: pytest.fail("charge"))
    assert result.spec is None and result.record["reason"] == "PROMPT_LENGTH"
    assert result.record["model_requests"] == 0


def test_failed_compile_falls_back_only_to_explicit_reference(monkeypatch, tmp_path):
    tmp_path = tmp_path / "run"
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.setenv("PVSCENE_VLM_BASE_URL", "https://example.invalid")
    monkeypatch.setattr(compiler, "call_vision", lambda *a, **k: LLMResponse("{}", "mock"))
    monkeypatch.setattr(sys, "argv", ["run_m1_acceptance", "--experiment", "office_pick_cup", "--compile",
                                    "--compiler-model", "mock", "--compile-only", "--output-dir", str(tmp_path)])
    assert generate_module.main() == 1  # fallback is reviewable but is not a successful compilation
    record = json.loads((tmp_path / "compilation.json").read_text())
    assert record["status"] == "fallback" and record["reason"] == "RESPONSE_SHAPE"
    assert (tmp_path / "effective_spec.json").exists()
    assert not (tmp_path / "compiled_spec.json").exists()


def test_fresh_run_refuses_to_overwrite_evidence(monkeypatch, tmp_path, capsys):
    (tmp_path / "compilation.json").write_text("previous evidence")
    monkeypatch.setattr(sys, "argv", ["run_m1_acceptance", "--experiment", "office_pick_cup",
                                    "--no-compile", "--compile-only", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        generate_module.main()
    assert error.value.code == 2
    assert "fresh directory" in capsys.readouterr().err
    assert (tmp_path / "compilation.json").read_text() == "previous evidence"


def test_model_cannot_hide_unsupported_task_in_a_valid_spec():
    spec = spec_for()
    doc = response_for(spec)
    doc["gaps"] = ["TASK_UNSUPPORTED"]
    with pytest.raises(compiler.CompileError, match="MODEL_REPORTED_GAPS:TASK_UNSUPPORTED"):
        parse(doc, spec)


def test_compiled_spec_replaces_reference_before_the_real_pipeline(monkeypatch, tmp_path):
    tmp_path = tmp_path / "run"
    spec = spec_for("kitchen_fetch_jar")
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    monkeypatch.setenv("PVSCENE_VLM_BASE_URL", "https://example.invalid")
    monkeypatch.setattr(compiler, "call_vision", lambda *a, **k: LLMResponse(json.dumps(response_for(spec)), "mock"))
    monkeypatch.setattr(sys, "argv", ["run_m1_acceptance", "--experiment", "office_pick_cup",
        "--prompt", spec.prompt, "--scene-id", spec.scene_id, "--seed", str(spec.seed),
        "--compiler-model", "mock", "--skip-isaac", "--no-replan", "--output-dir", str(tmp_path)])
    entry = importlib.import_module('rfw_w.scenes.application')
    class ReachedPipeline(Exception):
        pass
    def host(**kwargs):
        assert kwargs["spec"] == spec
        assert placement_module.build_plan(kwargs["spec"]).target_id == "jar"
        raise ReachedPipeline
    monkeypatch.setattr(application_module, "SceneHost", host)
    with pytest.raises(ReachedPipeline):
        generate_module.main()
    record = json.loads((tmp_path / "compilation.json").read_text())
    assert any(row["path"] == "/scene_id" for row in record["field_diff"])


def test_critic_checks_permission_and_shared_budget_at_call_site(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from rfw_w.scenes.agents import llm_client, montage
    from rfw_w.scenes.contracts.tool import EvidenceLedger
    spec = spec_for()
    version = "a" * 64
    (tmp_path / "view.png").write_bytes(b"fixture")
    (tmp_path / "report.json").write_text(json.dumps({"scene_version": version}))
    monkeypatch.setattr(montage, "montage_data_url", lambda *a, **k: ("data:image/png;base64,fixture", ["view"]))
    monkeypatch.setenv("PVSCENE_VLM_BASE_URL", "https://example.invalid")
    monkeypatch.setattr(llm_client, "call_vision", lambda *a, **k: pytest.fail("must not send"))
    entry = importlib.import_module('rfw_w.scenes.application')
    host = object.__new__(application_module.SceneHost)
    host.args = SimpleNamespace(vlm_model="mock")
    host.worker_dir = tmp_path
    host.spec = spec
    host.scene = {"instances": [SimpleNamespace(instance_id="cup")]}
    host.ledger = EvidenceLedger()
    host.model_budget = Budget(max_model_requests=1, model_requests_used=1)
    monkeypatch.delenv("PVSCENE_ALLOW_EXTERNAL_VLM", raising=False)
    with pytest.raises(SystemExit, match="PVSCENE_ALLOW_EXTERNAL_VLM"):
        host.run_critique(version)
    monkeypatch.setenv("PVSCENE_ALLOW_EXTERNAL_VLM", "1")
    assert host.run_critique(version) is False
    assert host.ledger.get("visual.critique", version).status.value == "not_run"
    assert host.model_budget.model_requests_used == 1


def test_binary_relation_failure_keeps_a_bounded_semantic_diagnosis():
    spec = spec_for()
    doc = response_for(spec)
    doc['spec']['initial_relations'][0].update(subject='desk', object='desk')
    with pytest.raises(compiler.CompileError) as caught:
        parse(doc, spec)
    assert caught.value.reason == 'SPEC_INVALID'
    assert 'subject and object must differ' in caught.value.detail
    assert len(caught.value.detail) <= 400
