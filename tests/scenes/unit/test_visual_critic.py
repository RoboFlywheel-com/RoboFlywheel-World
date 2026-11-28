# SPDX-License-Identifier: Apache-2.0
"""The Visual Critic's strictness.

Every test here is a way a visual gate stops meaning anything. The most important is that a missing
score is `invalid` rather than a default: filling in a middling value for an absent judgement is the
single easiest way to make a failed scene pass.
"""

from __future__ import annotations

import json

import pytest

from rfw_w.scenes.agents.visual_critic import (
    CRITERIA,
    Critique,
    CritiqueError,
    CritiqueStatus,
    InvalidReason,
    VisualThresholds,
    critique_evidence,
    evaluate_visual_gate,
    parse_critique,
    sanitised_payload,
)
from rfw_w.scenes.contracts.tool import EvidenceStatus

HASH_A = "a" * 64
HASH_B = "b" * 64


def good_scores(**overrides: int) -> dict[str, int]:
    base = {criterion: 4 for criterion in CRITERIA}
    base.update(overrides)
    return base


def response(**overrides: object) -> str:
    payload: dict[str, object] = {"scene_version": HASH_A, "scores": good_scores()}
    payload.update(overrides)
    return json.dumps(payload)


def parse(raw: str, **kwargs: object) -> Critique:
    kwargs.setdefault("current_scene_hash", HASH_A)
    return parse_critique(raw, reviewed_scene_hash=HASH_A, **kwargs)  # type: ignore[arg-type]


class TestInvalidCritiques:
    def test_empty_response_is_invalid(self) -> None:
        critique = parse("")
        assert critique.status is CritiqueStatus.INVALID
        assert critique.scores is None
        assert critique.reason_code is InvalidReason.UNPARSEABLE

    def test_truncated_json_is_invalid(self) -> None:
        """A response cut off mid-object must not be partially read."""
        critique = parse('{"scene_version": "' + HASH_A + '", "scores": {"prompt_adherence": 4')
        assert critique.status is CritiqueStatus.INVALID
        assert critique.scores is None

    def test_prose_without_json_is_invalid(self) -> None:
        assert parse("The scene looks quite nice to me.").status is CritiqueStatus.INVALID

    def test_a_missing_criterion_is_invalid_not_a_default(self) -> None:
        scores = good_scores()
        del scores["functionality"]
        critique = parse(response(scores=scores))
        assert critique.status is CritiqueStatus.INVALID
        assert critique.reason_code is InvalidReason.MISSING_CRITERION
        assert "functionality" in critique.reason_detail
        assert critique.scores is None

    def test_a_score_out_of_range_is_invalid(self) -> None:
        critique = parse(response(scores=good_scores(prompt_adherence=6)))
        assert critique.status is CritiqueStatus.INVALID
        assert critique.reason_code is InvalidReason.SCORE_OUT_OF_RANGE

    def test_a_zero_score_is_invalid(self) -> None:
        assert parse(response(scores=good_scores(functionality=0))).status is CritiqueStatus.INVALID

    def test_a_non_integer_score_is_invalid(self) -> None:
        assert parse(response(scores=good_scores(prompt_adherence=3.5))).status is CritiqueStatus.INVALID

    def test_a_boolean_score_is_invalid(self) -> None:
        # bool is an int subclass; True would otherwise be read as a score of 1.
        assert parse(response(scores=good_scores(prompt_adherence=True))).status is CritiqueStatus.INVALID

    def test_scores_must_be_an_object(self) -> None:
        assert parse(response(scores=[4, 4, 4])).status is CritiqueStatus.INVALID

    def test_an_invalid_critique_never_passes_the_gate(self) -> None:
        gate = evaluate_visual_gate(parse("nonsense"))
        assert not gate.passed
        assert gate.scores is None
        assert "invalid" in gate.reason


class TestStaleness:
    def test_a_response_without_a_scene_version_is_invalid(self) -> None:
        """Without the hash the critique cannot be tied to a render, and could be approving a
        different one."""
        critique = parse(json.dumps({"scores": good_scores()}))
        assert critique.status is CritiqueStatus.INVALID
        assert critique.reason_code is InvalidReason.STALE_IMAGE

    def test_a_critique_of_another_scene_is_invalid(self) -> None:
        critique = parse(response(scene_version=HASH_B))
        assert critique.status is CritiqueStatus.INVALID
        assert critique.reason_code is InvalidReason.STALE_IMAGE
        assert HASH_B[:12] in critique.reason_detail

    def test_stale_evidence_does_not_approve_the_current_scene(self) -> None:
        critique = parse(response(scene_version=HASH_B))
        gate = evaluate_visual_gate(critique)
        evidence = critique_evidence(critique, gate, scene_version=HASH_A)
        assert evidence.status is EvidenceStatus.NOT_RUN
        assert not evidence.satisfies(HASH_A)


class TestGating:
    def test_a_good_critique_passes(self) -> None:
        critique = parse(response())
        gate = evaluate_visual_gate(critique)
        assert gate.passed
        assert gate.failures == ()

    def test_prompt_adherence_below_four_fails(self) -> None:
        """The manual's initial gate: prompt and functionality at least 4."""
        gate = evaluate_visual_gate(parse(response(scores=good_scores(prompt_adherence=3))))
        assert not gate.passed
        assert ("prompt_adherence", 3, 4) in gate.failures

    def test_functionality_below_four_fails(self) -> None:
        gate = evaluate_visual_gate(parse(response(scores=good_scores(functionality=3))))
        assert not gate.passed

    def test_a_three_on_a_secondary_criterion_passes(self) -> None:
        gate = evaluate_visual_gate(parse(response(scores=good_scores(composition_lighting=3))))
        assert gate.passed

    def test_a_critical_defect_fails_despite_perfect_scores(self) -> None:
        """Averaging is how "the cup is inside the desk" survives a review that liked the lighting."""
        critique = parse(response(critical_defect=True))
        gate = evaluate_visual_gate(critique)
        assert not gate.passed
        assert gate.critical_defect
        assert "no score can offset it" in gate.reason

    def test_thresholds_are_configurable_and_validated(self) -> None:
        strict = VisualThresholds(minimums={criterion: 5 for criterion in CRITERIA})
        assert not evaluate_visual_gate(parse(response()), thresholds=strict).passed
        with pytest.raises(CritiqueError, match="do not cover"):
            VisualThresholds(minimums={"prompt_adherence": 4})

    def test_out_of_range_threshold_is_rejected(self) -> None:
        with pytest.raises(CritiqueError, match="outside"):
            VisualThresholds(minimums={**{c: 3 for c in CRITERIA}, "prompt_adherence": 9})


class TestEvidence:
    def test_a_passing_critique_produces_current_evidence(self) -> None:
        critique = parse(response())
        evidence = critique_evidence(critique, evaluate_visual_gate(critique), scene_version=HASH_A)
        assert evidence.status is EvidenceStatus.PASS
        assert evidence.satisfies(HASH_A)
        assert evidence.threshold_profile
        assert evidence.unit == "score_1_to_5"

    def test_a_failing_gate_produces_failing_evidence(self) -> None:
        critique = parse(response(scores=good_scores(functionality=2)))
        evidence = critique_evidence(critique, evaluate_visual_gate(critique), scene_version=HASH_A)
        assert evidence.status is EvidenceStatus.FAIL

    def test_evidence_records_the_reviewed_hash(self) -> None:
        # Reporting the current hash for a critique of another version would launder a stale
        # approval into a current one.
        critique = parse(response(scene_version=HASH_B))
        assert critique.reviewed_scene_hash == HASH_B

    def test_evidence_carries_the_object_observations_as_witnesses(self) -> None:
        critique = parse(response(object_evidence=[{"object_id": "cup", "observation": "on the desk"}]))
        evidence = critique_evidence(critique, evaluate_visual_gate(critique), scene_version=HASH_A)
        assert any("cup" in ref for ref in evidence.witness_refs)

    def test_the_suggested_objects_are_readable_as_repair_targets(self) -> None:
        """The failure this pins: the critic named the objects it wanted changed, the evidence
        carried them, and none of them reached the event -- so every geometric rung was skipped for
        a complaint whose only admissible answer was `swap_asset`."""
        from rfw_w.scenes.orchestrator.events import targets_from_witnesses

        critique = parse(
            response(repairs=[
                {"criterion": "prompt_adherence", "target_id": "cup", "action": "swap_asset",
                 "constraint": "a handled mug"},
            ]),
            known_targets=("cup", "desk"),
        )
        assert critique.suggestions, "the suggestion was refused before it could be witnessed"
        evidence = critique_evidence(critique, evaluate_visual_gate(critique), scene_version=HASH_A)
        assert targets_from_witnesses(evidence.witness_refs) == ("cup",)

    def test_the_observations_alone_do_not_name_repair_targets(self) -> None:
        """They describe every object the critic looked at, which is not a list of what it asked to
        change: parsing them would aim a repair at the first plan row in the room."""
        from rfw_w.scenes.orchestrator.events import targets_from_witnesses

        critique = parse(response(object_evidence=[
            {"object_id": "desk", "observation": "a desk"}, {"object_id": "cup", "observation": "a cup"},
        ]))
        evidence = critique_evidence(critique, evaluate_visual_gate(critique), scene_version=HASH_A)
        assert targets_from_witnesses(evidence.witness_refs) == ()


class TestSanitisation:
    def test_payload_carries_only_the_image_name_not_its_path(self) -> None:
        payload = sanitised_payload(
            scene_version=HASH_A,
            views=[{"label": "overview", "image_path": "/data/secret/scene/overview.png"}],
            targets=[{"semantic_id": "asset:abc", "category": "Mug"}],
        )
        blob = json.dumps(payload)
        assert "/data/secret" not in blob
        assert "overview.png" in blob

    def test_payload_admits_no_paths_logs_or_catalogue_fields(self) -> None:
        payload = sanitised_payload(
            scene_version=HASH_A,
            views=[{"label": "overview", "image_path": "/a/b/c.png", "image_base64": "AAAA"}],
            targets=[{"semantic_id": "asset:abc", "category": "Mug"}],
        )
        assert set(payload) == {"scene_version", "views", "task_targets", "task_prompt", "task"}
        assert set(payload["task_targets"][0]) == {"semantic_id", "category"}

    def test_payload_carries_task_semantics_for_prompt_adherence(self) -> None:
        payload = sanitised_payload(
            scene_version=HASH_A,
            views=[{"label": "overview"}],
            targets=[{"semantic_id": "cup", "category": "cup"}],
            task_prompt="a warm office with a desk",
            task={"action": "pick", "targets": ["cup"]},
        )
        assert payload["task_prompt"].startswith("a warm office")
        assert payload["task"]["targets"] == ["cup"]

    def test_a_view_must_be_labelled(self) -> None:
        with pytest.raises(CritiqueError, match="label"):
            sanitised_payload(scene_version=HASH_A, views=[{}], targets=[])

    def test_an_unhashed_scene_version_is_refused(self) -> None:
        with pytest.raises(CritiqueError, match="scene hash"):
            sanitised_payload(scene_version="office_scene", views=[{"label": "v"}], targets=[])


class TestReviewedVersusSent:
    def test_the_reviewed_hash_comes_from_the_response(self) -> None:
        """A field echoing the caller's argument would make staleness compare the caller to itself."""
        critique = parse(response(scene_version=HASH_B))
        assert critique.reviewed_scene_hash == HASH_B
        assert critique.sent_scene_hash == HASH_A

    def test_a_valid_critique_records_both_hashes_as_equal(self) -> None:
        critique = parse(response())
        assert critique.reviewed_scene_hash == critique.sent_scene_hash == HASH_A

    def test_an_invalid_critique_still_carries_a_usable_digest(self) -> None:
        # Evidence demands a valid digest, so an unparseable response falls back to what was sent.
        critique = parse("not json at all")
        assert len(critique.reviewed_scene_hash) == 64
        assert critique.sent_scene_hash == HASH_A
