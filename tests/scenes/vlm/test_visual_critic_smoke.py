# SPDX-License-Identifier: Apache-2.0
"""One real Visual Critic call, on a real rendered scene. MANUAL OPT-IN ONLY.

This sends a rendered image to an external service, so it never runs as part of a default gate. It
requires PVSCENE_ALLOW_EXTERNAL_VLM=1, which `scripts/dev/run_vlm_tests.sh` sets only when asked.

What it verifies is the channel, not the picture: that an image produced by the Isaac worker can
reach a multimodal model, come back as a strict-schema critique, and be gated. Whether the critique
is *good* is a calibration question for human-labelled data, and this test does not pretend
otherwise -- it asserts the plumbing and records what the model said.

The image is the real output of `scripts/run_isaac_worker.py`, not a synthetic picture, because the
question is whether the rendered channel works end to end.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rfw_w.scenes.agents.llm_client import (
    CRITIC_SYSTEM_PROMPT,
    CredentialsMissing,
    LLMConfig,
    LLMError,
    build_critic_prompt,
    call_vision,
    image_data_url,
)
from rfw_w.scenes.agents.visual_critic import (
    CritiqueStatus,
    critique_evidence,
    evaluate_visual_gate,
    parse_critique,
    sanitised_payload,
)

pytestmark = pytest.mark.vlm

REPO_ROOT = Path(__file__).resolve().parents[3]
# One output root: every run lands under outputs/. The scene runs are grouped by experiment and the
# smoke fixtures are not, which is why the two globs below differ in depth rather than in prefix.
OUTPUTS = REPO_ROOT / "outputs"

# Configured, not hardcoded into behaviour: the endpoint and model are deployment choices.
#
# The default is what .claude/settings.local.json configures for this project. Measured on the
# configured
# gateway, qwen3-vl-plus, qwen-vl-plus, qwen3-vl-flash, qwen-omni-turbo and qwen3-omni-flash all
# return HTTP 403 "Model access denied" while appearing in the /models listing, so a listing is not
# evidence of access and the working model has to be named. Override with PVSCENE_VLM_MODEL.
MODEL = os.environ.get("PVSCENE_VLM_MODEL", "qwen3.8-max")
BASE_URL_ENV = "SIMFOUNDRY_OPENAI_BASE_URL"
KEY_ENV = "DASHSCOPE_API_KEY"


def _latest_render() -> tuple[Path, Path]:
    """The most recent real scene render, preferring an accepted delivery over a smoke run.

    The smoke fixture is a runtime check: two objects on a floor with no room around them. Judging
    a model against it measures the model's opinion of a test fixture, and comparing two models on
    different fixtures measures nothing at all. A delivery render is a scene the system actually
    claims something about, so it is preferred when one exists.
    """
    candidates: list[Path] = []
    # A scene run: outputs/<experiment>/<run>/isaac/
    candidates += sorted(
        (p for p in OUTPUTS.glob("*/*/isaac") if (p / "report.json").is_file()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    # A runtime smoke fixture: outputs/_smoke/<name>/
    candidates += sorted(
        (p for p in OUTPUTS.glob("_smoke/*") if (p / "report.json").is_file()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for directory in candidates:
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text())
        for capture in report.get("captures", ()):
            image = directory / capture["rgb_path"]
            if image.is_file():
                return report_path, image
    pytest.skip(
        "no Isaac render found under outputs/; run scripts/run_m1_acceptance.py first. A synthetic "
        "image would not answer the question this test asks."
    )


@pytest.fixture(scope="module")
def render() -> tuple[Path, Path]:
    if os.environ.get("PVSCENE_ALLOW_EXTERNAL_VLM") != "1":
        pytest.skip("PVSCENE_ALLOW_EXTERNAL_VLM is not set; this test sends an image externally")
    return _latest_render()


def test_rendered_image_reaches_the_model_and_returns_a_gated_critique(render) -> None:
    report_path, image_path = render
    report = json.loads(report_path.read_text())
    scene_version = report["scene_version"]

    payload = sanitised_payload(
        scene_version=scene_version,
        views=[{"label": "overview", "image_path": str(image_path)}],
        targets=[{"semantic_id": key, "category": value} for key, value in report["instance_id_map"].items()],
    )
    # The sanitised payload must not carry a filesystem path, or this test would be shipping one.
    assert str(image_path.parent) not in json.dumps(payload)

    config = LLMConfig(
        base_url=os.environ.get(BASE_URL_ENV, ""),
        model=MODEL,
        api_key_env=KEY_ENV,
        timeout_s=float(os.environ.get("PVSCENE_VLM_TIMEOUT", "180")),
    )
    if not config.credential_present():
        pytest.skip(f"{KEY_ENV} is not set")

    try:
        response = call_vision(
            config,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=build_critic_prompt(payload),
            images=[image_data_url(str(image_path))],
        )
    except CredentialsMissing as error:
        pytest.skip(str(error))
    except LLMError as error:
        pytest.fail(f"the model call failed: {error}")

    critique = parse_critique(
        response.text,
        reviewed_scene_hash=scene_version,
        current_scene_hash=scene_version,
        model_id=response.model_id,
        tokens_used=response.tokens_used,
    )
    gate = evaluate_visual_gate(critique)
    evidence = critique_evidence(critique, gate, scene_version=scene_version)

    record = {
        "model_id": response.model_id,
        "tokens_used": response.tokens_used,
        "finish_reason": response.finish_reason,
        "critique": critique.to_dict(),
        "gate": gate.to_dict(),
        "evidence_status": evidence.status.value,
        "image": image_path.name,
    }
    out = OUTPUTS / "_probe" / "vlm_smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "critique.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    # Either outcome is acceptable: a valid critique with scores, or an invalid one with a reason.
    # What is not acceptable is a critique that is invalid yet carries scores, or one that passes
    # the gate without scores.
    if critique.status is CritiqueStatus.VALID:
        assert critique.scores is not None
        assert set(critique.scores) == {
            "prompt_adherence", "functionality", "style_material_coherence", "proportion_density",
            "composition_lighting", "lifestyle_plausibility", "workspace_and_passage",
        }
        assert all(1 <= value <= 5 for value in critique.scores.values())
        assert critique.reviewed_scene_hash == scene_version
    else:
        assert critique.scores is None, "an invalid critique must not carry partial scores"
        assert critique.reason_code is not None
    assert not (gate.passed and gate.scores is None)
    assert response.text, "the model returned an empty response"
