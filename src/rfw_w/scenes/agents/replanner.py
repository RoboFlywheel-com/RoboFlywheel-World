# SPDX-License-Identifier: Apache-2.0
"""The Failure Replanner: the one model call the ladder may make (L5).

The executable design gives this role a precise contract:

* §4 -- triggered *"only when local repair is exhausted or there is a structural conflict"*; it
  returns a schema'd ``SemanticPatch``; it *"may not delete hard entities, change the task, or relax
  hard relations"*.
* §19.2 -- it sees a **compressed** summary (goal, stage, task-critical statuses, the active
  failure, global flags, suggested repairs), not the scene.
* §10 -- it is given ``allowed_actions``, and progress is judged by the failure set shrinking, not
  by the patch claiming success.

Three things are implemented here rather than left to discipline, because each is a way an L5 call
becomes either useless or dangerous:

* **the patch is validated, not trusted.** Every operation is checked against the plan and against
  the request's hard constraints. A rejected operation is recorded with its reason; it is never
  silently dropped, because a replan whose operations were all refused must read as "the model had
  nothing usable to say" rather than as "the model agreed".
* **the vocabulary is closed and the unimplemented members are named.** §9.1 lists ten orchestrator
  actions; four of them (``retrieve_again``, ``change_relation``, ``replan_layout`` and a general
  ``add`` of an arbitrary asset) have no implementation in this version. They are refused with a
  reason that says so, rather than accepted and ignored.
* **an added object must come from a declared vocabulary.** A model choosing a synset freely is
  choosing what to retrieve from the asset library. The caller supplies the vocabulary and this
  module refuses anything outside it, so an L5 patch cannot ask the retriever for something the
  scene was never specified to contain.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace as dataclass_replace
from enum import Enum
from typing import Any, Mapping, Sequence

from rfw_w.scenes.agents.llm_client import (
    LLMConfig,
    LLMError,
    REPLANNER_SYSTEM_PROMPT,
    build_replanner_prompt,
    call_vision,
)
from rfw_w.scenes.contracts.tool import FailureCode
from rfw_w.scenes.solvers.furnish import FurnitureRequest, Orientation, PlacementKind, ScenePlan

# An added object goes against a wall and faces the room, which is the ordinary case for the context
# objects a critic is most likely to ask for (a bin, a shelf, a second cabinet). Placement is not the
# model's decision here for the same reason it is not its decision anywhere else: the solver places,
# and the model says what should be added rather than where it ends up.
_DEFAULT_ADDED_KIND = PlacementKind.AGAINST_WALL
_DEFAULT_ADDED_ORIENTATION = Orientation.FRONT_TO_ROOM

SCHEMA_VERSION = "pvscene.replanner/v1"

# The free-text cap. A replanner that returns an essay has not returned a patch, and the diagnosis
# is recorded in the run's own artefacts, so it must not be able to carry arbitrary length there.
MAX_DIAGNOSIS_CHARS = 400


class PatchAction(str, Enum):
    """What a patch may ask for. A subset of §9.1's orchestrator actions.

    The four §9.1 actions with no implementation here are deliberately absent rather than present
    and ignored: a closed vocabulary that lies about what it can do is worse than a small one.
    """

    MOVE = "move"
    SWAP = "swap"
    CHANGE_SUPPORT = "change_support"
    ADD = "add"
    REMOVE = "remove"
    ROLLBACK = "rollback"
    TERMINATE = "terminate"


# What this version can actually carry out. ``terminate`` is a legitimate answer -- an agent that
# has established the task cannot be met should say so rather than thrash -- and is handled by the
# driver rather than by a repair atom.
IMPLEMENTED_ACTIONS: tuple[str, ...] = (
    # Keep this list identical to application_requests below. Advertising MOVE, CHANGE_SUPPORT,
    # ADD or ROLLBACK while silently producing no edit made a valid-looking L5 patch restart the
    # loop on an unchanged scene.
    PatchAction.SWAP.value,
    PatchAction.REMOVE.value,
    PatchAction.TERMINATE.value,
)

# Named so a refusal can say which design action it corresponds to. Without this the reason reads
# "unimplemented", which does not tell a reader whether the vocabulary or the code is short.
UNIMPLEMENTED_DESIGN_ACTIONS = ("retrieve_again", "change_relation", "replan_layout")


@dataclass(frozen=True, slots=True)
class PatchOperation:
    """One operation in a patch, validated."""

    action: PatchAction
    # Empty for ``terminate``, which is addressed at the task rather than at an object.
    target_id: str = ""
    synset: str = ""
    constraint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value, "target_id": self.target_id,
            "synset": self.synset, "constraint": self.constraint,
        }


@dataclass(frozen=True, slots=True)
class RejectedOperation:
    """An operation the patch asked for and the loop refused, with the reason."""

    reason: str
    action: str = ""
    target_id: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason, "action": self.action,
            "target_id": self.target_id, "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SemanticPatch:
    """What the replanner returned: its diagnosis, the operations that survived, and the rest."""

    diagnosis: str
    operations: tuple[PatchOperation, ...] = ()
    rejected: tuple[RejectedOperation, ...] = ()
    model_id: str = ""
    tokens_used: int | None = None

    @property
    def usable(self) -> bool:
        """Whether the patch has anything the loop can act on."""
        return any(operation.action is not PatchAction.TERMINATE for operation in self.operations)

    @property
    def terminates(self) -> bool:
        return any(operation.action is PatchAction.TERMINATE for operation in self.operations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "diagnosis": self.diagnosis,
            "operations": [operation.to_dict() for operation in self.operations],
            "rejected": [operation.to_dict() for operation in self.rejected],
            "usable": self.usable,
            "terminates": self.terminates,
            "model_id": self.model_id,
            "tokens_used": self.tokens_used,
        }


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _validate_operations(
    raw: Any, *, plan: ScenePlan, synset_vocabulary: Sequence[str]
) -> tuple[tuple[PatchOperation, ...], tuple[RejectedOperation, ...]]:
    """Check every operation against the plan and the request's hard constraints.

    The refusals, each with its own reason code so the run's report says what the model tried to do:
    an action outside the vocabulary, an action this version does not implement, a target that is
    not in the plan, deleting a hard-required entity, deleting a non-optional entity, adding a
    synset outside the declared vocabulary, and adding an object with no synset at all.
    """
    accepted: list[PatchOperation] = []
    rejected: list[RejectedOperation] = []
    if raw is None:
        return (), ()
    if not isinstance(raw, list):
        return (), (RejectedOperation(reason="not-a-list", detail=f"operations was {type(raw).__name__}"),)

    vocabulary = {str(name) for name in synset_vocabulary}
    for item in raw:
        if not isinstance(item, dict):
            rejected.append(RejectedOperation(reason="not-an-object", detail=repr(item)[:120]))
            continue
        action_text = str(item.get("action", ""))
        target_id = str(item.get("target_id", ""))
        synset = str(item.get("synset", ""))
        constraint = str(item.get("constraint", ""))

        if action_text in UNIMPLEMENTED_DESIGN_ACTIONS:
            rejected.append(RejectedOperation(
                reason="action-not-implemented", action=action_text, target_id=target_id,
                detail=(
                    f"{action_text} is one of the design's orchestrator actions but has no "
                    f"implementation in this version; implemented: {list(IMPLEMENTED_ACTIONS)}"
                ),
            ))
            continue
        try:
            action = PatchAction(action_text)
        except ValueError:
            rejected.append(RejectedOperation(
                reason="unknown-action", action=action_text, target_id=target_id,
                detail=f"not in the vocabulary: {list(IMPLEMENTED_ACTIONS)}",
            ))
            continue

        if action.value not in IMPLEMENTED_ACTIONS:
            rejected.append(RejectedOperation(
                reason="action-not-implemented", action=action_text, target_id=target_id,
                detail=f"the application layer implements only {list(IMPLEMENTED_ACTIONS)}",
            ))
            continue

        if action is PatchAction.TERMINATE:
            # A termination is addressed at the task, not at an object, so it carries no target and
            # must not be refused for lacking one.
            accepted.append(PatchOperation(action=action, constraint=constraint))
            continue

        if action is PatchAction.ADD:
            if not synset:
                rejected.append(RejectedOperation(
                    reason="add-without-synset", action=action_text,
                    detail="adding an object requires the synset to retrieve",
                ))
                continue
            if synset not in vocabulary:
                rejected.append(RejectedOperation(
                    reason="synset-outside-vocabulary", action=action_text, target_id=target_id,
                    detail=(
                        f"{synset} is not in the declared vocabulary {sorted(vocabulary)}; an L5 "
                        "patch names what the request already permits, it does not choose what the "
                        "scene contains"
                    ),
                ))
                continue
            accepted.append(
                PatchOperation(action=action, target_id=target_id, synset=synset, constraint=constraint)
            )
            continue

        request = plan.furniture_for(target_id) or plan.surface_for(target_id)
        if request is None:
            rejected.append(RejectedOperation(
                reason="unknown-target", action=action_text, target_id=target_id,
                detail=f"the plan contains {sorted(plan.entity_ids)}",
            ))
            continue
        if action is PatchAction.REMOVE:
            if getattr(request, "hard_required", False):
                rejected.append(RejectedOperation(
                    reason="hard-entity-removal", action=action_text, target_id=target_id,
                    detail=(
                        "the request marks this entity as required; the design forbids a replan "
                        "from dropping a hard goal"
                    ),
                ))
                continue
            if not getattr(request, "optional", False):
                rejected.append(RejectedOperation(
                    reason="non-optional-removal", action=action_text, target_id=target_id,
                    detail="only entities the request marked optional may be removed",
                ))
                continue
        accepted.append(
            PatchOperation(action=action, target_id=target_id, synset=synset, constraint=constraint)
        )
    return tuple(accepted), tuple(rejected)


def parse_patch(
    raw: str,
    *,
    plan: ScenePlan,
    synset_vocabulary: Sequence[str] = (),
    model_id: str = "",
    tokens_used: int | None = None,
) -> SemanticPatch:
    """Parse and validate a replanner response. Never raises for a bad one.

    A malformed patch is an outcome the driver records, not an exception: the run has already spent
    the request, and crashing would lose the record that it spent it.
    """
    if not raw or not raw.strip():
        return SemanticPatch(
            diagnosis="", rejected=(RejectedOperation(reason="empty-response"),),
            model_id=model_id, tokens_used=tokens_used,
        )
    match = _JSON_BLOCK.search(raw)
    if not match:
        return SemanticPatch(
            diagnosis="", rejected=(RejectedOperation(reason="unparseable", detail="no JSON object"),),
            model_id=model_id, tokens_used=tokens_used,
        )
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return SemanticPatch(
            diagnosis="",
            rejected=(RejectedOperation(reason="unparseable", detail=error.msg),),
            model_id=model_id, tokens_used=tokens_used,
        )
    if not isinstance(payload, dict):
        return SemanticPatch(
            diagnosis="",
            rejected=(RejectedOperation(reason="not-an-object", detail="response was not a mapping"),),
            model_id=model_id, tokens_used=tokens_used,
        )

    operations, rejected = _validate_operations(
        payload.get("operations"), plan=plan, synset_vocabulary=synset_vocabulary
    )
    return SemanticPatch(
        diagnosis=str(payload.get("diagnosis", ""))[:MAX_DIAGNOSIS_CHARS],
        operations=operations, rejected=rejected, model_id=model_id, tokens_used=tokens_used,
    )


def scene_summary(
    plan: ScenePlan,
    *,
    goal: str,
    stage: str,
    active_failure: Mapping[str, Any],
    global_flags: Mapping[str, bool] | None = None,
    suggested_repairs: Sequence[str] = (),
    known_targets: Sequence[str] = (),
) -> dict[str, Any]:
    """The compressed summary §19.2 gives the replanner.

    Deliberately not the scene: poses, mesh paths, catalogue records and evidence are all absent.
    What is here is what a decision needs -- the goal, how far the run has got, which pieces are
    required, what survived the ladder, and which repairs were already tried and failed. The last of
    those is the one that stops the model proposing the repair the local search has just exhausted.
    """
    task_critical: dict[str, Any] = {}
    for name in known_targets:
        request = plan.furniture_for(name) or plan.surface_for(name)
        if request is None:
            continue
        task_critical[name] = {
            "kind": getattr(request, "kind", None) and request.kind.value,
            "required": bool(getattr(request, "hard_required", False)),
            "optional": bool(getattr(request, "optional", True)),
        }
    return {
        "goal": goal,
        "stage": stage,
        "task_critical": task_critical,
        "active_failure": dict(active_failure),
        "global": dict(global_flags or {}),
        "suggested_repair": list(suggested_repairs),
        "plan": {
            "entities": list(plan.entity_ids),
            "wall_face_offset_m": plan.wall_face_offset_m,
            "object_gap_m": plan.object_gap_m,
        },
    }


def replan(
    config: LLMConfig,
    *,
    plan: ScenePlan,
    summary: Mapping[str, Any],
    allowed_actions: Sequence[str] = IMPLEMENTED_ACTIONS,
    synset_vocabulary: Sequence[str] = (),
    on_request: Any = None,
) -> SemanticPatch:
    """Spend one model request on a replan.

    ``on_request`` is called before the request goes out, so the caller can charge it against the
    budget. It is a callback rather than an import because the budget belongs to the orchestrator and
    a client that charged it itself would be spending a request the loop never counted -- the failure
    §4 warns about when it says a Router's internal calls must not be hidden inside a tool.
    """
    if on_request is not None:
        on_request()

    try:
        response = call_vision(
            config,
            system_prompt=REPLANNER_SYSTEM_PROMPT,
            user_prompt=build_replanner_prompt(summary, allowed_actions=allowed_actions),
        )
    except LLMError as error:
        # A transport failure is not a patch. Reported as an empty patch with the reason, so the
        # driver records that the request was spent and produced nothing rather than pretending the
        # model declined.
        return SemanticPatch(
            diagnosis="", rejected=(RejectedOperation(reason="call-failed", detail=str(error)),)
        )

    return parse_patch(
        response.text, plan=plan, synset_vocabulary=synset_vocabulary,
        model_id=response.model_id, tokens_used=response.tokens_used,
    )


def application_requests(
    patch: SemanticPatch, plan: ScenePlan
) -> tuple[tuple[str, FurnitureRequest | None], ...]:
    """The plan edits a patch asks for, as ``(entity_id, request_or_None)`` pairs.

    Kept separate from applying them so a caller can decide *whether* to apply -- and so the
    operations stay a description of a change rather than a change, which is what makes them
    reviewable before they land.
    """
    edits: list[tuple[str, FurnitureRequest | None]] = []
    for operation in patch.operations:
        if operation.action is PatchAction.ADD:
            edits.append((
                operation.target_id,
                FurnitureRequest(
                    entity_id=operation.target_id, synset=operation.synset,
                    kind=_DEFAULT_ADDED_KIND, orientation=_DEFAULT_ADDED_ORIENTATION,
                ),
            ))
        elif operation.action is PatchAction.REMOVE:
            edits.append((operation.target_id, None))
        elif operation.action is PatchAction.SWAP:
            request = plan.furniture_for(operation.target_id)
            if request is not None:
                edits.append((
                    operation.target_id,
                    dataclass_replace(request, asset_index=request.asset_index + 1),
                ))
    return tuple(edits)


def failure_code_for_patch(patch: SemanticPatch) -> FailureCode:
    """The typed code a patch's outcome is recorded under.

    A patch with no usable operation is a failure to replan, and that is distinct from a patch that
    chose to terminate: the first says the model had nothing to offer and the second says the model
    judged the task unmeetable. Recording both as "failed" would lose which happened.
    """
    if patch.terminates:
        return FailureCode.HARD_GOAL_DROPPED
    if not patch.usable:
        return FailureCode.PATCH_INVALID
    return FailureCode.NO_PROGRESS
