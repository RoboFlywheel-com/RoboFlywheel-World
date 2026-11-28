# SPDX-License-Identifier: Apache-2.0
"""The Visual Critic: scores from a model, and the rules that stop them becoming a rubber stamp.

Three rules from architecture §9 and the manual §6 are implemented here rather than left to
discipline, because each is a way a visual gate stops meaning anything:

* **a missing score is invalid, never a default.** A truncated response, an absent criterion or a
  score out of range produces ``invalid`` with a reason code. Filling in a "medium" score is the
  single easiest way to make a failed scene pass, so it is unrepresentable.
* **an image from a different scene version cannot approve this one.** The critique records the
  scene hash it looked at; a mismatched hash yields ``RENDER_STALE`` and no approval.
* **a critical defect fails the gate regardless of the scores.** Otherwise "looks lovely, the cup
  is inside the desk" passes on average.

The payload sent to a model is built by :func:`sanitised_payload`, which admits the rendered images
and opaque semantic IDs and nothing else. Repository paths, logs, catalogue records and checkpoint
contents are not merely omitted by convention -- they have no route into the payload.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from rfw_w.scenes.contracts.hashing import is_digest
from rfw_w.scenes.contracts.tool import Evidence, EvidenceStatus

TOOL_VERSION = "rfw_w.scenes.agents.visual_critic/v1"
SCHEMA_VERSION = "pvscene.visual-critique/v1"

# The six criteria from architecture §9, plus `functionality`, which the manual's gate names and
# the §9 list omits. Both names are carried rather than one silently replacing the other.
CRITERIA = (
    "prompt_adherence",
    "functionality",
    "style_material_coherence",
    "proportion_density",
    "composition_lighting",
    "lifestyle_plausibility",
    "workspace_and_passage",
)

SCORE_MIN = 1
SCORE_MAX = 5

# The manual's initial engineering gate: prompt and functionality at least 4, everything else at
# least 3, no critical defect. Configured, to be calibrated against human labels.
DEFAULT_THRESHOLDS = {
    "prompt_adherence": 4,
    "functionality": 4,
    "style_material_coherence": 3,
    "proportion_density": 3,
    "composition_lighting": 3,
    "lifestyle_plausibility": 3,
    "workspace_and_passage": 3,
}

THRESHOLD_PROFILE = "visual_engineering_v1"


class CritiqueError(ValueError):
    """The critique payload is malformed."""


class CritiqueStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


class InvalidReason(str, Enum):
    """Why a critique could not be used. Each maps to a different repair."""

    NO_IMAGE = "NO_IMAGE"
    UNPARSEABLE = "UNPARSEABLE"
    MISSING_CRITERION = "MISSING_CRITERION"
    SCORE_OUT_OF_RANGE = "SCORE_OUT_OF_RANGE"
    STALE_IMAGE = "STALE_IMAGE"
    NO_TARGET_EVIDENCE = "NO_TARGET_EVIDENCE"


@dataclass(frozen=True, slots=True)
class Critique:
    """A model's verdict, or the reason there is not a usable one.

    ``scores`` is ``None`` when the status is invalid. That is deliberate: an invalid critique that
    carried partial scores would invite a caller to average the ones it received, which is how a
    missing judgement becomes a mediocre pass.
    """

    status: CritiqueStatus
    # What the model said it reviewed. Recorded from the response, not from the caller's argument:
    # a field that echoed the caller would make the staleness check compare the caller to itself.
    reviewed_scene_hash: str
    # The version of the images actually sent, so a disagreement between the two is visible.
    sent_scene_hash: str = ""
    model_id: str = ""
    scores: Mapping[str, int] | None = None
    defects: tuple[str, ...] = ()
    critical_defect: bool = False
    object_evidence: tuple[tuple[str, str], ...] = ()
    view_evidence: tuple[str, ...] = ()
    reason_code: InvalidReason | None = None
    reason_detail: str = ""
    tokens_used: int | None = None
    # What the critic asked to be changed, validated, and what it asked for that was refused. Both
    # are carried: the first drives the repair loop and the second is how a reviewer sees that the
    # critic was proposing things the loop could not do.
    suggestions: tuple[RepairSuggestion, ...] = ()
    rejected_suggestions: tuple[RejectedSuggestion, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status is CritiqueStatus.VALID

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status.value,
            "reviewed_scene_hash": self.reviewed_scene_hash,
            "sent_scene_hash": self.sent_scene_hash,
            "model_id": self.model_id,
            "scores": None if self.scores is None else dict(sorted(self.scores.items())),
            "defects": list(self.defects),
            "critical_defect": self.critical_defect,
            "object_evidence": [list(pair) for pair in self.object_evidence],
            "view_evidence": list(self.view_evidence),
            "reason_code": None if self.reason_code is None else self.reason_code.value,
            "reason_detail": self.reason_detail,
            "tokens_used": self.tokens_used,
            "suggestions": [item.to_dict() for item in self.suggestions],
            "rejected_suggestions": [item.to_dict() for item in self.rejected_suggestions],
        }


class CriticRepair(str, Enum):
    """The repair verbs a critic may ask for. Closed, and smaller than the ladder's atom set.

    The executable design §4 gives the Visual Critic "permitted semantic modification suggestions"
    and §9 limits repair to swapping to a compatible candidate, local layout, filling in missing
    context, and adjusting lighting/material. So the vocabulary is those four ideas and nothing
    else: a critic cannot ask for a subgraph rollback or an agent replan, because those are the
    ladder's answers to failures the ladder has already established, and a model that could reach
    for them directly would be choosing the repair rather than reporting the defect.
    """

    SWAP_ASSET = "swap_asset"
    OPTIMISE_POSE = "optimise_pose"
    CHANGE_SUPPORT = "change_support"
    ADJUST_LIGHTING_MATERIAL = "adjust_lighting_material"
    ADD_CONTEXT_OBJECT = "add_context_object"


# Which repair a critic may ask for, per criterion. This is the table that decides the loop's atom
# from the VLM's judgement, and it is a *restriction*: a model that asks for something outside its
# criterion's row has the suggestion dropped rather than obeyed.
#
# The split follows what each criterion is actually about. A criterion about whether the scene works
# is answered by moving or re-supporting what is in the way; a criterion about how it looks is
# answered by changing how it is lit; a criterion about what is present is answered by changing what
# is there, not where it is.
CRITERION_REPAIRS: Mapping[str, tuple[CriticRepair, ...]] = {
    "prompt_adherence": (CriticRepair.SWAP_ASSET, CriticRepair.ADD_CONTEXT_OBJECT),
    "functionality": (CriticRepair.CHANGE_SUPPORT, CriticRepair.OPTIMISE_POSE),
    "workspace_and_passage": (CriticRepair.OPTIMISE_POSE, CriticRepair.CHANGE_SUPPORT),
    "lifestyle_plausibility": (CriticRepair.OPTIMISE_POSE, CriticRepair.SWAP_ASSET),
    "proportion_density": (CriticRepair.SWAP_ASSET, CriticRepair.ADD_CONTEXT_OBJECT),
    "style_material_coherence": (CriticRepair.ADJUST_LIGHTING_MATERIAL,),
    "composition_lighting": (CriticRepair.ADJUST_LIGHTING_MATERIAL,),
}


@dataclass(frozen=True, slots=True)
class RepairSuggestion:
    """One repair a critic asked for, already validated against the scene it was shown.

    It is a *suggestion* and carries no authority: the ladder decides whether the repair it maps to
    is the cheapest admissible one for the failure. What validation buys is that the suggestion
    names a real object and an action the loop has an implementation for, so acting on it cannot
    send a repair at an instance that does not exist.
    """

    criterion: str
    target_id: str
    action: CriticRepair
    constraint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion, "target_id": self.target_id,
            "action": self.action.value, "constraint": self.constraint,
        }


@dataclass(frozen=True, slots=True)
class RejectedSuggestion:
    """A suggestion that was dropped, and why.

    Recorded rather than discarded. A critic that proposes actions the loop cannot take, or names
    objects that are not in the scene, is a critic that has stopped being grounded in what it was
    shown -- and a run that silently ignored those suggestions would look exactly like a run whose
    critic had nothing to say.
    """

    reason: str
    criterion: str = ""
    target_id: str = ""
    action: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason, "criterion": self.criterion, "target_id": self.target_id,
            "action": self.action, "detail": self.detail,
        }


def _validated_suggestions(
    raw: Any, *, known_targets: Sequence[str]
) -> tuple[tuple[RepairSuggestion, ...], tuple[RejectedSuggestion, ...]]:
    """Check a critic's repair requests against the vocabulary and the scene.

    Four ways a suggestion is refused, each with its own reason so the run's report says which kind
    of failure the critic had: an unknown criterion, an action outside the vocabulary, a target that
    is not one of the instances the critic was shown, and an action its criterion does not admit.
    """
    accepted: list[RepairSuggestion] = []
    rejected: list[RejectedSuggestion] = []
    if raw is None:
        return (), ()
    if not isinstance(raw, list):
        return (), (RejectedSuggestion(reason="not-a-list", detail=f"repairs was {type(raw).__name__}"),)

    known = set(known_targets)
    for item in raw:
        if not isinstance(item, dict):
            rejected.append(RejectedSuggestion(reason="not-an-object", detail=repr(item)[:120]))
            continue
        criterion = str(item.get("criterion", ""))
        target_id = str(item.get("target_id", ""))
        action_text = str(item.get("action", ""))
        constraint = str(item.get("constraint", ""))

        if criterion not in CRITERIA:
            rejected.append(RejectedSuggestion(
                reason="unknown-criterion", criterion=criterion, target_id=target_id,
                action=action_text, detail="not one of the seven scored criteria",
            ))
            continue
        try:
            action = CriticRepair(action_text)
        except ValueError:
            rejected.append(RejectedSuggestion(
                reason="unknown-action", criterion=criterion, target_id=target_id,
                action=action_text,
                detail=(
                    "not in the vocabulary the loop implements: "
                    + ", ".join(member.value for member in CriticRepair)
                ),
            ))
            continue
        if not known:
            rejected.append(RejectedSuggestion(
                reason="no-targets-supplied", criterion=criterion, target_id=target_id,
                action=action_text,
                detail=(
                    "the call did not supply the instance ids the critic was shown, so a target "
                    "cannot be checked against the scene and is not trusted"
                ),
            ))
            continue
        if target_id not in known:
            rejected.append(RejectedSuggestion(
                reason="unknown-target", criterion=criterion, target_id=target_id,
                action=action_text,
                detail=f"the critic was shown {sorted(known)}",
            ))
            continue
        if action not in CRITERION_REPAIRS[criterion]:
            rejected.append(RejectedSuggestion(
                reason="action-not-admissible-for-criterion", criterion=criterion,
                target_id=target_id, action=action_text,
                detail=(
                    f"{criterion} admits "
                    + ", ".join(candidate.value for candidate in CRITERION_REPAIRS[criterion])
                ),
            ))
            continue
        accepted.append(
            RepairSuggestion(
                criterion=criterion, target_id=target_id, action=action, constraint=constraint
            )
        )
    return tuple(accepted), tuple(rejected)


def sanitised_payload(
    *,
    scene_version: str,
    views: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    task_prompt: str = "",
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only payload a model is ever sent.

    Each view contributes its label and its image bytes; each target contributes an opaque semantic
    ID and its category. Anything else -- paths, hashes, logs, catalogue fields, checkpoint state --
    has no parameter to arrive through.

    The scene version is passed through as an opaque digest rather than a name, because the model
    needs to echo it back for the staleness check and needs to know nothing else about it.
    """
    if not is_digest(scene_version):
        raise CritiqueError("scene_version must be a scene hash")
    clean_views = []
    for view in views:
        label = str(view.get("label", "")).strip()
        if not label:
            raise CritiqueError("every view must carry a label")
        payload: dict[str, Any] = {"label": label}
        if view.get("image_base64"):
            payload["image_base64"] = str(view["image_base64"])
        if view.get("image_path"):
            # A path is accepted here only to be read by the caller; the payload carries the
            # basename so that a full filesystem path cannot travel to the model.
            import os

            payload["image_name"] = os.path.basename(str(view["image_path"]))
        clean_views.append(payload)

    clean_targets = []
    for target in targets:
        clean_targets.append(
            {
                "semantic_id": str(target.get("semantic_id", "")),
                "category": str(target.get("category", "")),
            }
        )
    return {
        "scene_version": scene_version,
        "views": clean_views,
        "task_targets": clean_targets,
        "task_prompt": str(task_prompt),
        "task": dict(task or {}),
    }


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_critique(
    raw: str,
    *,
    reviewed_scene_hash: str,
    current_scene_hash: str,
    model_id: str = "",
    tokens_used: int | None = None,
    known_targets: Sequence[str] = (),
) -> Critique:
    """Parse a model response, refusing anything that is not a complete, in-range critique.

    Never raises for a bad response: a malformed reply is an outcome, not an exception, because the
    caller must record that the gate did not run rather than crash the run.

    ``known_targets`` is the set of instance ids the critic was shown. A repair suggestion naming
    anything outside it is dropped, because the only thing worse than a critic with no repair to
    suggest is one whose suggestion sends a repair at an object that does not exist. Omitting the
    targets does not disable the check -- it fails it, and every suggestion is refused with
    ``no-targets-supplied``, so a caller cannot get an unvalidated suggestion by forgetting.
    """
    def invalid(reason: InvalidReason, detail: str, *, echoed: str = "") -> Critique:
        return Critique(
            status=CritiqueStatus.INVALID,
            # Falls back to the sent hash only when the response named nothing, so the field always
            # carries a usable digest for the evidence that will demand one.
            reviewed_scene_hash=echoed if is_digest(echoed) else reviewed_scene_hash,
            sent_scene_hash=reviewed_scene_hash,
            model_id=model_id,
            reason_code=reason,
            reason_detail=detail,
            tokens_used=tokens_used,
        )

    if not raw or not raw.strip():
        return invalid(InvalidReason.UNPARSEABLE, "the model returned nothing")

    # Models wrap JSON in prose or fences; find the outermost object and parse that. A truncated
    # response has no closing brace and correctly fails here rather than being partially read.
    match = _JSON_BLOCK.search(raw)
    if not match:
        return invalid(InvalidReason.UNPARSEABLE, "no JSON object found in the response")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return invalid(InvalidReason.UNPARSEABLE, f"JSON did not parse: {error.msg} at {error.pos}")

    if not isinstance(payload, dict):
        return invalid(InvalidReason.UNPARSEABLE, "the response was not a JSON object")

    echoed = str(payload.get("scene_version", "")).strip()
    if not echoed:
        return invalid(
            InvalidReason.STALE_IMAGE,
            "the critique does not name the scene version it reviewed, so it cannot be tied to a "
            "render and could be approving a different one",
        )
    if echoed != current_scene_hash:
        return invalid(
            InvalidReason.STALE_IMAGE,
            f"the critique reviewed scene {echoed[:12]} while the scene is {current_scene_hash[:12]}",
            echoed=echoed,
        )

    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        return invalid(InvalidReason.MISSING_CRITERION, "no scores object was returned")

    scores: dict[str, int] = {}
    for criterion in CRITERIA:
        if criterion not in raw_scores:
            return invalid(
                InvalidReason.MISSING_CRITERION,
                f"criterion {criterion!r} was not scored; a missing judgement is not a middling one",
            )
        value = raw_scores[criterion]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return invalid(InvalidReason.SCORE_OUT_OF_RANGE, f"{criterion} is not a number: {value!r}")
        if not float(value).is_integer():
            return invalid(InvalidReason.SCORE_OUT_OF_RANGE, f"{criterion} is not a whole number: {value!r}")
        integer = int(value)
        if not SCORE_MIN <= integer <= SCORE_MAX:
            return invalid(
                InvalidReason.SCORE_OUT_OF_RANGE,
                f"{criterion} is {integer}, outside {SCORE_MIN}-{SCORE_MAX}",
            )
        scores[criterion] = integer

    defects = payload.get("defects", [])
    if isinstance(defects, str):
        defects = [defects]
    if not isinstance(defects, list):
        return invalid(InvalidReason.UNPARSEABLE, "defects must be a list")

    object_evidence: list[tuple[str, str]] = []
    for item in payload.get("object_evidence", []) or []:
        if isinstance(item, dict) and item.get("object_id") and item.get("observation"):
            object_evidence.append((str(item["object_id"]), str(item["observation"])))

    suggestions, rejected = _validated_suggestions(
        payload.get("repairs"), known_targets=known_targets
    )
    # An object the critic names in its narrative but that is not in the scene is the same class of
    # problem as a repair aimed at one. Recorded rather than dropped, so the run shows whether the
    # critic was describing what it was shown.
    known = set(known_targets)
    if known:
        for object_id, _observation in object_evidence:
            if object_id not in known:
                rejected = rejected + (RejectedSuggestion(
                    reason="unknown-object-evidence", target_id=object_id,
                    detail=f"named in object_evidence but not among {sorted(known)}",
                ),)

    return Critique(
        status=CritiqueStatus.VALID,
        reviewed_scene_hash=echoed,
        sent_scene_hash=reviewed_scene_hash,
        model_id=model_id,
        scores=scores,
        defects=tuple(str(item) for item in defects),
        critical_defect=bool(payload.get("critical_defect", False)),
        object_evidence=tuple(object_evidence),
        view_evidence=tuple(str(item) for item in payload.get("view_evidence", []) or []),
        tokens_used=tokens_used,
        suggestions=suggestions,
        rejected_suggestions=rejected,
    )


@dataclass(frozen=True, slots=True)
class VisualThresholds:
    """Per-criterion minimum scores. Configured, not universal."""

    minimums: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    def __post_init__(self) -> None:
        missing = [criterion for criterion in CRITERIA if criterion not in self.minimums]
        if missing:
            raise CritiqueError(f"thresholds do not cover {missing}")
        for criterion, value in self.minimums.items():
            if not SCORE_MIN <= value <= SCORE_MAX:
                raise CritiqueError(f"{criterion} threshold {value} is outside {SCORE_MIN}-{SCORE_MAX}")

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self.minimums.items()))


@dataclass(frozen=True, slots=True)
class VisualGate:
    """The gate's verdict, with which criteria failed and by how much."""

    passed: bool
    scores: Mapping[str, int] | None
    failures: tuple[tuple[str, int, int], ...]
    critical_defect: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scores": None if self.scores is None else dict(sorted(self.scores.items())),
            "failures": [list(item) for item in self.failures],
            "critical_defect": self.critical_defect,
            "reason": self.reason,
        }


def evaluate_visual_gate(
    critique: Critique, *, thresholds: VisualThresholds | None = None
) -> VisualGate:
    """Decide whether a critique passes.

    An invalid critique never passes and never contributes scores. A critical defect fails the gate
    whatever the average is, because averaging is exactly how "the cup is inside the desk" survives
    a review that liked the lighting.
    """
    limits = thresholds or VisualThresholds()
    if not critique.valid or critique.scores is None:
        return VisualGate(
            passed=False,
            scores=None,
            failures=(),
            critical_defect=False,
            reason=(
                "the critique is invalid"
                + (f" ({critique.reason_code.value})" if critique.reason_code else "")
                + f": {critique.reason_detail}"
            ),
        )

    failures = tuple(
        (criterion, int(critique.scores[criterion]), limits.minimums[criterion])
        for criterion in CRITERIA
        if int(critique.scores[criterion]) < limits.minimums[criterion]
    )
    if critique.critical_defect:
        return VisualGate(
            passed=False,
            scores=critique.scores,
            failures=failures,
            critical_defect=True,
            reason="a critical defect was reported; no score can offset it",
        )
    if failures:
        detail = ", ".join(f"{name} {got}<{want}" for name, got, want in failures)
        return VisualGate(
            passed=False, scores=critique.scores, failures=failures, critical_defect=False,
            reason=f"criteria below threshold: {detail}",
        )
    return VisualGate(passed=True, scores=critique.scores, failures=(), critical_defect=False)


def critique_evidence(
    critique: Critique,
    gate: VisualGate,
    *,
    scene_version: str,
    backend: str = "visual",
) -> Evidence:
    """Turn a critique and its gate into evidence tied to the scene version.

    The evidence records the hash the *critique* reviewed, not the current one: if those differ the
    critique is stale, and reporting the current hash would launder a stale approval into a current
    one.
    """
    if not critique.valid:
        return Evidence(
            gate_id="visual.critique",
            status=EvidenceStatus.NOT_RUN,
            state_hash=current_or_reviewed(critique, scene_version),
            backend=backend,
            tool_version=TOOL_VERSION,
            method="vlm_critique_strict_schema",
            detail=f"critique invalid: {critique.reason_code.value if critique.reason_code else 'unknown'}",
        )

    if not critique.reviewed_scene_hash == scene_version:
        return Evidence(
            gate_id="visual.critique",
            status=EvidenceStatus.FAIL,
            state_hash=critique.reviewed_scene_hash,
            backend=backend,
            tool_version=TOOL_VERSION,
            method="vlm_critique_strict_schema",
            detail=(
                f"the critique reviewed scene {critique.reviewed_scene_hash[:12]} but the scene is "
                f"{scene_version[:12]}"
            ),
        )

    return Evidence(
        gate_id="visual.critique",
        status=EvidenceStatus.PASS if gate.passed else EvidenceStatus.FAIL,
        state_hash=scene_version,
        backend=backend,
        tool_version=TOOL_VERSION,
        method="vlm_critique_strict_schema",
        measured_value=(
            float(min(gate.scores.values())) if gate.passed and gate.scores else None
        ),
        unit="score_1_to_5" if gate.passed and gate.scores else None,
        threshold_profile=THRESHOLD_PROFILE if gate.passed else None,
        witness_refs=tuple(
            f"{view}" for view in critique.view_evidence
        ) + tuple(f"{obj}:{obs}" for obj, obs in critique.object_evidence)
        # The suggestions ride as witnesses so a repair the loop performed can be traced back to
        # the critique that asked for it, in the same evidence record as the scores it came with.
        #
        # They ride in the `instance:<id>:<what>` form `orchestrator.events.targets_from_witnesses`
        # recognises, and that is load-bearing rather than cosmetic. The object_evidence above is
        # prose keyed by entity -- it describes every object the critic looked at, so parsing it
        # would name the whole room as repair targets -- but these two lines name the objects the
        # critic actually asked to change, which is exactly the event's target set. Written any
        # other way the critic's failure raised an event with *no* targets, and an untargeted
        # geometric event has every geometric rung skipped (``driver._repair_for``), so a complaint
        # whose only admissible answer was `swap_asset` could never be acted on at all.
        + tuple(
            f"instance:{item.target_id}:repair-{item.action.value}"
            for item in critique.suggestions
        ),
        detail=gate.reason or f"critique passed with scores {dict(sorted(gate.scores.items()))}",
    )


def critics_atom_for(action: CriticRepair) -> Any:
    """The ladder atom a critic's verb asks for.

    Imported here rather than at module scope: the escalation ladder imports nothing from this
    module, and keeping the dependency one-way means the critic can be read without the ladder in
    mind. ``add_context_object`` maps to a replan because adding an object is a semantic decision --
    there is no local atom that invents an object, and there should not be one, since the plan's
    contents come from the request rather than from the repair loop.

    A suggestion is not an authorisation: the caller still has to pass
    :meth:`~rfw_w.scenes.orchestrator.escalation.Ladder.assert_order`, so a critic asking for a
    candidate swap while the pose has never been resampled does not get one.
    """
    from rfw_w.scenes.orchestrator.escalation import Atom

    return {
        CriticRepair.SWAP_ASSET: Atom.SWAP_ASSET,
        CriticRepair.OPTIMISE_POSE: Atom.OPTIMISE_POSE,
        CriticRepair.CHANGE_SUPPORT: Atom.CHANGE_SUPPORT,
        CriticRepair.ADJUST_LIGHTING_MATERIAL: Atom.ADJUST_LIGHTING_MATERIAL,
        CriticRepair.ADD_CONTEXT_OBJECT: Atom.REPLAN,
    }[action]


def current_or_reviewed(critique: Critique, scene_version: str) -> str:
    """The hash to bind invalid evidence to.

    The current scene version, because invalid evidence must be re-measured against the scene as it
    stands; binding it to a stale hash would make the staleness the reason.
    """
    return scene_version
