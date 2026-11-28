# SPDX-License-Identifier: Apache-2.0
"""Compile task text to the same SceneSpec the deterministic pipeline already consumes.

Only task text, public vocabulary and output contract cross the model boundary. No file paths,
repository content, asset catalogue, or execution logs are accepted by the prompt builder.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Callable

from rfw_w.scenes.agents.llm_client import LLMConfig, LLMError, call_vision, require_external_permission
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_spec import SceneSpec, EntityRole, Predicate, Mobility
from rfw_w.scenes.contracts.readiness import ReadinessLevel
from rfw_w.scenes.geometry.robot import ROBOT_PROFILES
from rfw_w.scenes.solvers.furnishing_plan import CATEGORY_PROFILES, plan_from_spec
from rfw_w.scenes.solvers.materials import PALETTE_LIBRARY
from rfw_w.scenes.solvers.support_shape import SupportShape

MAX_PROMPT_CHARS = 16000
MAX_RESPONSE_CHARS = 100000
MODEL_GAPS = frozenset({"TASK_UNSUPPORTED", "PHYSICS_UNSUPPORTED", "CATEGORY_UNKNOWN",
                        "AMBIGUOUS_TARGET", "ROOM_CONSTRAINT_UNSUPPORTED", "ROBOT_UNSUPPORTED"})


class CompileError(ValueError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"compile_failed({reason})")
        self.reason = reason
        self.detail = detail[:400]


def prompt_hash(prompt: str, seed: int) -> str:
    return content_hash({"prompt": prompt, "seed": seed})


def build_compile_prompt(prompt: str, seed: int, scene_id: str) -> tuple[str, str]:
    if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise CompileError("PROMPT_LENGTH")
    vocabulary = {
        "category": sorted(CATEGORY_PROFILES), "palette": sorted(PALETTE_LIBRARY),
        "predicate": [p.value for p in Predicate], "role": [r.value for r in EntityRole],
        "mobility": [m.value for m in Mobility], "robot_profile": sorted(ROBOT_PROFILES),
        "support_shape": [s.value for s in SupportShape],
    }
    system = '''Compile a robotics scene request into task semantics, never assets or poses.
Return exactly one JSON object with keys prompt_hash, spec, assumptions, gaps. assumptions is a list
of short strings explaining every inferred constraint not explicit in the request. Do not change
the requested objects or task to make generation easier. Unsupported requests must be reported
in gaps and must not be disguised as supported tasks. gaps is [] for a supported unambiguous
request, otherwise one or more of TASK_UNSUPPORTED, PHYSICS_UNSUPPORTED, CATEGORY_UNKNOWN,
AMBIGUOUS_TARGET, ROOM_CONSTRAINT_UNSUPPORTED, ROBOT_UNSUPPORTED. A nonempty gaps list stops
compilation; do not turn an unsupported request into a different supported task.
spec has keys schema_version ("pvscene.scene-spec/v3"), scene_id, seed, prompt (verbatim input),
room, style, requested_backends (["isaac"]), requested_readiness ("functional_scene_ready"),
entities, initial_relations, task.
room must include room_kind (office/kitchen/bedroom), size_policy (explicit/inferred), and
scale_class (small/medium/large). Use explicit and size_bounds_m
[[width_min,width_max],[depth_min,depth_max],[height_min,height_max]] only when dimensions are
specified by the user; preserve those bounds. XY must be within 2..20 metres and height 2..6 metres.
When dimensions are absent, use inferred and OMIT size_bounds_m: the versioned policy supplies
small=4..5 by 3..4, medium=6..7 by 5..6, large=8..10 by 6..8 metres, with height 2.7..3.
Choose the scale requested by the user; absent a scale, use medium and explain the assumption.
Do not assign office to a kitchen or bedroom. Core geometry is solved first and the room is fitted
around its protected floor.
Supported solver behavior: the room generator supplies a door and windows; the layout reserves the
robot work region and a passage from the door. Requests to preserve workspace or door-to-seat
circulation are supported, even though they have no separate JSON fields. Furniture category
profiles supply wall-oriented storage, a chair near its support, wall-hung art and floor rugs;
use notes and relations between requested objects to express ordinary wall/desk orientation.
Do not invent wall/door entities. Do not report ROOM_CONSTRAINT_UNSUPPORTED merely because the
request mentions walls, a doorway, workspace or a passage. That gap is for room constraints beyond
this contract, such as an exact fixed floor plan or exact opening positions. Compilation records
intent; later geometry and visual gates measure whether the generated scene actually satisfies it.
style: {"theme": string, "palette": [tokens], "density": "sparse"|"medium"|"dense"}.
Each entity: {"id": identifier, "category": token, "physics_type": "rigid",
"mobility": token, "hard_required": boolean, "role": token, "quantity": 1,
"size_bounds_m": [[xmin,xmax],[ymin,ymax],[zmin,zmax]], "notes": string,
"material_tokens": [tokens], "requires_support_surface": boolean}.
Entity bounds are optional, must be positive and within .001..10 metres if present.
Exactly one task_target and one support; both hard_required true. Target mobility is dynamic.
The target must have a hard ON initial relation to the support. Other objects resting on the
support have ON relations; surrounding contextual objects may have soft NEAR or FACING relations.
Use separate entity IDs for multiple objects. Do not put a surrounding bed or chair in core by
marking it as support. Preserve explicitly required context with hard_required or relations.
Each initial relation: {"id": identifier, "subject": entity_id, "predicate": token,
"object": entity_id, "hard": boolean}. No unrecognised fields. Every relation is binary:
subject and object must be DIFFERENT IDs of requested entities. For wall orientation, use the
entity's notes and category profile; do not emit a self-relation such as desk AGAINST_WALL desk,
and do not add a wall entity to make a relation possible.
task: {"action": "pick_place", "robot_profile": token, "targets": [the target id],
"goal_predicates": []}. This release supports tabletop pick_place with a mobile manipulator;
it does not support fixed-base mounts, articulated tasks or deformable objects.
No executable instructions, paths, synsets, asset IDs, poses, or room sketches in the output.
notes and each assumption are at most 400 characters. Echo prompt_hash exactly.
The following public semantic vocabulary is closed:
'''+json.dumps(vocabulary, sort_keys=True)
    user = json.dumps({"prompt": prompt, "seed": seed, "scene_id": scene_id,
                       "prompt_hash": prompt_hash(prompt, seed)}, ensure_ascii=False)
    return system, user


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CompileError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def field_diff(before, after, path="") -> list[dict]:
    """A local review artifact; it is never part of the model input."""
    if isinstance(before, dict) and isinstance(after, dict):
        return [row for key in sorted(set(before) | set(after))
                for row in field_diff(before.get(key), after.get(key), f"{path}/{key}")]
    if isinstance(before, list) and isinstance(after, list):
        return [row for i in range(max(len(before), len(after)))
                for row in field_diff(before[i] if i < len(before) else None,
                                      after[i] if i < len(after) else None, f"{path}/{i}")]
    return [] if type(before) is type(after) and before == after else [
        {"path": path or "/", "requested": before, "compiled": after}]


def _check_bounds(value, limits):
    if not isinstance(value, list) or len(value) != 3:
        raise CompileError("SIZE_SHAPE")
    for axis, (low, high) in zip(value, limits):
        if not isinstance(axis, list) or len(axis) != 2 or any(
            type(n) not in (int, float) or not math.isfinite(n) for n in axis
        ):
            raise CompileError("SIZE_SHAPE")
        if not low <= axis[0] <= axis[1] <= high:
            # Refuse instead of silently weakening an explicitly requested dimension.
            raise CompileError("SIZE_OUT_OF_RANGE")


def parse_compilation(text: str, *, prompt: str, seed: int, scene_id: str) -> tuple[SceneSpec, list[str]]:
    if len(text) > MAX_RESPONSE_CHARS:
        raise CompileError("RESPONSE_LENGTH")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        doc = json.loads(text, object_pairs_hook=_unique_object,
                         parse_constant=lambda _: (_ for _ in ()).throw(CompileError("NONFINITE_NUMBER")))
    except (json.JSONDecodeError, TypeError):
        raise CompileError("UNPARSEABLE") from None
    if not isinstance(doc, dict) or set(doc) != {"prompt_hash", "spec", "assumptions", "gaps"}:
        raise CompileError("RESPONSE_SHAPE")
    if doc["prompt_hash"] != prompt_hash(prompt, seed):
        raise CompileError("PROMPT_HASH_MISMATCH")
    gaps = doc["gaps"]
    if not isinstance(gaps, list) or any(not isinstance(g, str) or g not in MODEL_GAPS for g in gaps):
        raise CompileError("GAPS_SHAPE")
    if gaps:
        raise CompileError("MODEL_REPORTED_GAPS:" + ",".join(sorted(set(gaps))))
    assumptions = doc["assumptions"]
    if not isinstance(assumptions, list) or len(assumptions) > 40 or any(
        not isinstance(s, str) or len(s) > 400 for s in assumptions
    ):
        raise CompileError("ASSUMPTIONS_SHAPE")
    raw = doc["spec"]
    if not isinstance(raw, dict):
        raise CompileError("SPEC_SHAPE")
    if (raw.get("prompt") != prompt or raw.get("seed") != seed or type(raw.get("seed")) is not int
            or raw.get("scene_id") != scene_id):
        raise CompileError("INPUT_IDENTITY_CHANGED")
    try:
        from rfw_w.scenes.contracts.scene_spec import RoomSpec
        resolved_room = RoomSpec.from_dict(raw["room"])
        _check_bounds([axis.to_list() for axis in resolved_room.size_bounds_m], ((2, 20), (2, 20), (2, 6)))
        if not isinstance(raw["entities"], list) or not 2 <= len(raw["entities"]) <= 64:
            raise CompileError("ENTITY_COUNT")
        for entity in raw["entities"]:
            if type(entity.get("hard_required")) is not bool:
                raise CompileError("BOOLEAN_REQUIRED")
            if type(entity.get("quantity", 1)) is not int or entity.get("quantity", 1) != 1:
                raise CompileError("QUANTITY_UNSUPPORTED")
            if len(entity.get("notes", "")) > 400:
                raise CompileError("NOTES_LENGTH")
            if "size_bounds_m" in entity:
                _check_bounds(entity["size_bounds_m"], ((.001, 10),) * 3)
        for relation in raw.get("initial_relations", []):
            if type(relation.get("hard")) is not bool:
                raise CompileError("BOOLEAN_REQUIRED")
        spec = SceneSpec.from_dict(raw)
        if spec.requested_backends != ("isaac",) or spec.requested_readiness != ReadinessLevel.FUNCTIONAL_SCENE_READY.value:
            raise CompileError("READINESS_OR_BACKEND_UNSUPPORTED")
        if spec.task.action != "pick_place" or spec.task.joint_task is not None:
            raise CompileError("TASK_UNSUPPORTED")
        if spec.task.robot_profile not in ROBOT_PROFILES:
            raise CompileError("ROBOT_UNSUPPORTED")
        if len(spec.task.targets) != 1 or len(spec.task_targets) != 1 or spec.task_targets[0].id != spec.task.targets[0]:
            raise CompileError("TARGET_NOT_UNIQUE")
        supports = [e for e in spec.entities if e.role is EntityRole.SUPPORT]
        if len(supports) != 1 or not supports[0].hard_required or not spec.task_targets[0].hard_required:
            raise CompileError("CORE_NOT_REQUIRED")
        if spec.task_targets[0].mobility is not Mobility.DYNAMIC:
            raise CompileError("TARGET_NOT_DYNAMIC")
        tokens = set(spec.style.palette) | set(spec.style.materials)
        for entity in spec.entities:
            if entity.category not in CATEGORY_PROFILES:
                raise CompileError("CATEGORY_UNKNOWN")
            if entity.physics_type.value != "rigid":
                raise CompileError("PHYSICS_UNSUPPORTED")
            tokens.update(entity.material_tokens)
        if tokens - set(PALETTE_LIBRARY):
            raise CompileError("MATERIAL_UNKNOWN")
        if not any(r.subject == spec.task.targets[0] and r.object == supports[0].id
                   and r.predicate is Predicate.ON and r.hard for r in spec.initial_relations):
            raise CompileError("HARD_ON_REQUIRED")
        plan = plan_from_spec(spec)
        if plan.support_id != supports[0].id or plan.derivation_gaps:
            raise CompileError("PLAN_DERIVATION_GAP")
    except CompileError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        # This exception is from semantic validation, not transport: a bounded field-level
        # diagnosis explains a refusal without recording a provider response or credentials.
        raise CompileError("SPEC_INVALID", str(error)) from None
    return spec, assumptions


@dataclass(frozen=True)
class Compilation:
    spec: SceneSpec | None
    record: dict


def compile_scene(config: LLMConfig, *, prompt: str, seed: int, scene_id: str,
                  on_request: Callable[[], None], reference: SceneSpec | None = None) -> Compilation:
    record = {"status": "failed", "compile_prompt_hash": prompt_hash(prompt, seed),
              "model": config.model, "model_requests": 0, "assumptions": [], "field_diff": []}
    try:
        system, user = build_compile_prompt(prompt, seed, scene_id)
    except CompileError as error:
        record["reason"] = error.reason
        return Compilation(None, record)
    require_external_permission()
    on_request()
    record["model_requests"] = 1
    try:
        response = call_vision(config, system_prompt=system, user_prompt=user)
        record["response_hash"] = content_hash({"text": response.text})
        record["finish_reason"] = response.finish_reason
        spec, assumptions = parse_compilation(response.text, prompt=prompt, seed=seed, scene_id=scene_id)
        record.update(status="compiled", compiled_spec_hash=content_hash(spec.to_dict()),
                      model_id=response.model_id, tokens_used=response.tokens_used,
                      assumptions=assumptions,
                      field_diff=[] if reference is None else field_diff(reference.to_dict(), spec.to_dict()))
        return Compilation(spec, record)
    except (LLMError, CompileError) as error:
        # Transport diagnostics can contain provider response text. Keep only the named class.
        record["reason"] = error.reason if isinstance(error, CompileError) else type(error).__name__
        if isinstance(error, CompileError):
            record["failure_detail"] = error.detail
        else:
            cause = error.__cause__
            record["transport_error_type"] = type(cause).__name__ if cause else None
            record["http_status"] = getattr(cause, "code", None)
        return Compilation(None, record)
