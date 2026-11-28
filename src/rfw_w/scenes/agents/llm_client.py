# SPDX-License-Identifier: Apache-2.0
"""A minimal multimodal client, for the Scene Compiler and the Visual Critic.

Stdlib only, and deliberately narrow: one call with text and optionally an image, in whichever of
the two wire formats :data:`WIRES` names. There is no retry loop, no streaming and no tool calling,
because the thing being protected here is the model-request budget from architecture §4 -- a client
that retried internally would spend requests the orchestrator never counted.

Credential handling: the key is read from an environment variable at call time and never stored in
a result, a report or an exception message. A leaked key in an evidence artefact is a real
disclosure, and artefacts are the things this project ships.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "pvscene.llm/v2"

# The two request shapes this client speaks. They are not dialects of one another: a
# chat-completions call carries `messages` and answers in `choices[0].message.content`, a
# responses call carries `instructions` plus `input` and answers in `output[].content[]`. Naming
# the wire explicitly rather than sniffing it means a misconfiguration fails at the endpoint with
# a 400 instead of being silently misread into an empty critique that looks like a model that
# found nothing wrong.
WIRES = ("chat_completions", "responses")


class LLMError(RuntimeError):
    """The call failed, with the reason but never the credential."""

    def __init__(self, message: str, *, code: str = "LLM_ERROR") -> None:
        super().__init__(message)
        self.code = code


class CredentialsMissing(LLMError):
    """No credential is configured for the requested endpoint."""

    def __init__(self) -> None:
        super().__init__("credential environment variable is not set", code="LLM_CREDENTIAL_MISSING")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Where to call and which variable holds the key.

    The key itself is not a field. A config object holding a secret ends up in logs and reprs;
    naming the environment variable and reading it at call time does not.
    """

    base_url: str
    model: str
    api_key_env: str
    timeout_s: float = 240.0
    # Enough for the critique on either wire. The old 1024 was a chat-completions-era number and it
    # is not enough for a reasoning model: asked for the same seven-criterion JSON, gpt-5.6-sol
    # spent the whole ceiling on reasoning and came back with 329 characters of a truncated object
    # -- `finish_reason: max_output_tokens`. A truncated critique is UNPARSEABLE rather than wrong,
    # so the run records a critique that did not happen and the cause is not visible in the ledger.
    max_tokens: int = 4096
    wire: str = "chat_completions"

    def __post_init__(self) -> None:
        if not self.base_url:
            raise LLMError("base_url must be non-empty")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise LLMError("base_url must not contain credentials or query parameters",
                           code="LLM_CONFIG_INVALID")
        if not self.model:
            raise LLMError("model must be non-empty")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", self.api_key_env):
            raise LLMError("api_key_env must name the environment variable holding the key")
        if self.timeout_s <= 0:
            raise LLMError("timeout_s must be positive")
        if self.wire not in WIRES:
            raise LLMError(
                f"wire must be one of {WIRES}; a base_url ending in "
                f"/v1/responses needs 'responses', and one serving /chat/completions needs "
                f"'chat_completions'"
            )

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        suffix = "/responses" if self.wire == "responses" else "/chat/completions"
        # A base_url that already spells the full path is used as given. Providers hand out both
        # forms -- ".../v1" and ".../v1/responses" -- and appending blindly turns the second into
        # ".../v1/responses/responses", which 404s in a way that reads like a wrong model name.
        if base.endswith(suffix):
            return base
        return base + suffix

    def credential_present(self) -> bool:
        """Whether a key is configured. The value is not read."""
        return bool(os.environ.get(self.api_key_env))

    def to_dict(self) -> dict[str, Any]:
        """Safe to serialise: names the variable, never the value."""
        return {
            "schema_version": SCHEMA_VERSION,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "wire": self.wire,
            "credential_present": self.credential_present(),
        }


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model_id: str
    tokens_used: int | None = None
    finish_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "text_length": len(self.text),
        }


def image_data_url(path: str, *, mime: str = "image/png") -> str:
    """Read an image and encode it as a data URL for the request body."""
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _request_body(
    config: LLMConfig, *, system_prompt: str, user_prompt: str, images: Sequence[str]
) -> dict[str, Any]:
    """The wire's own request shape.

    Three differences matter, and each is a silent-corruption risk rather than a loud one. The
    Responses API takes the system text as top-level ``instructions`` instead of a ``system``
    message, names its images ``input_image`` inside typed content parts instead of ``image_url``,
    and caps length with ``max_output_tokens``. Sending a chat body to /responses does not fail
    loudly at the provider -- it can be accepted and answered from the text alone, which produces a
    critique that scored seven criteria without ever seeing a pixel.
    """
    if config.wire == "responses":
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        content.extend({"type": "input_image", "image_url": image} for image in images)
        return {
            "model": config.model,
            "instructions": system_prompt,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": config.max_tokens,
        }
    chat_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    chat_content.extend({"type": "image_url", "image_url": {"url": image}} for image in images)
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_content},
        ],
    }


def _read_response(config: LLMConfig, payload: Mapping[str, Any]) -> LLMResponse:
    """The wire's own response shape, read into the one shape the callers know."""
    usage = payload.get("usage") or {}
    total = usage.get("total_tokens")
    if config.wire == "responses":
        parts: list[str] = []
        for item in payload.get("output") or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    parts.append(str(part.get("text", "")))
        text = "".join(parts)
        raw_status = str(payload.get("status", ""))
        status = raw_status if raw_status in {"completed", "incomplete", "failed"} else "unknown"
        incomplete = payload.get("incomplete_details") or {}
        raw_reason = str(incomplete.get("reason", ""))
        reason = raw_reason if raw_reason in {"max_output_tokens", "content_filter"} else ""
        if not text:
            # The two ways to get nothing back are different failures with different fixes, and
            # reporting one message for both sends the reader to the wrong place. A refusal or a
            # token ceiling is named; anything else is left as the raw status.
            raise LLMError(
                f"the response carried no output_text (status={status!r}, "
                f"incomplete={(reason or 'other')!r}); if reason is 'max_output_tokens', raise max_tokens"
            )
        return LLMResponse(
            text=text,
            model_id=str(payload.get("model", config.model)),
            tokens_used=total if isinstance(total, int) else None,
            finish_reason=reason or status,
        )

    choices = payload.get("choices") or []
    if not choices:
        raise LLMError("no choices in the response")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if isinstance(text, list):
        # Some providers return content parts; concatenate the text ones.
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    finish_reason = str(choices[0].get("finish_reason", ""))
    if finish_reason not in {"stop", "length", "tool_calls", "content_filter"}:
        finish_reason = "unknown"
    return LLMResponse(
        text=str(text or ""),
        model_id=str(payload.get("model", config.model)),
        tokens_used=total if isinstance(total, int) else None,
        finish_reason=finish_reason,
    )


def call_vision(
    config: LLMConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    images: Sequence[str] = (),
) -> LLMResponse:
    """One multimodal completion, with the request budget spent exactly once."""
    key = os.environ.get(config.api_key_env)
    if not key:
        raise CredentialsMissing()

    body = _request_body(
        config, system_prompt=system_prompt, user_prompt=user_prompt, images=images
    )
    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            # The key travels in the header and is never placed in the body or a log line.
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # Providers can echo request headers and credentials in error bodies. Keep only the status.
        raise LLMError(f"HTTP {error.code}", code=f"LLM_HTTP_{error.code}") from None
    except urllib.error.URLError:
        raise LLMError("model endpoint unavailable", code="LLM_NETWORK_ERROR") from None
    except json.JSONDecodeError:
        raise LLMError("model endpoint returned invalid JSON", code="LLM_RESPONSE_INVALID") from None
    except (TimeoutError, OSError) as error:
        # A read timeout arrives as a bare TimeoutError, not a URLError, so an uncaught one escaped
        # as an exception and took a whole acceptance run down after every earlier stage had
        # succeeded. A slow or unreachable endpoint is an outcome, not a crash.
        code = "LLM_TIMEOUT" if isinstance(error, TimeoutError) else "LLM_NETWORK_ERROR"
        raise LLMError("model request failed", code=code) from None

    return _read_response(config, payload)


CRITIC_SYSTEM_PROMPT = (
    "You are reviewing a rendered 3D indoor scene for a robotics simulation. "
    "Score each criterion from 1 to 5, where 5 is fully satisfactory. "
    "Report a critical defect when the scene is unusable for its stated task. "
    "Respond with a single JSON object and nothing else, using exactly these keys: "
    "scene_version (echo the value given to you), "
    "scores (an object with prompt_adherence, functionality, style_material_coherence, "
    "proportion_density, composition_lighting, lifestyle_plausibility, workspace_and_passage), "
    "critical_defect (boolean), defects (list of strings), "
    "object_evidence (list of objects with object_id and observation), "
    "view_evidence (list of view labels you actually used), "
    "repairs (list of objects with criterion, target_id, action and constraint). "
    "For every defect you report, add one entry to repairs saying what should change. "
    "target_id must be one of the object ids you were given; you cannot name an object that is not "
    "in the scene, and a repair naming one will be discarded. "
    "action must be exactly one of: swap_asset (use a different asset of the same kind), "
    "optimise_pose (move or turn this object), change_support (stand it on something else), "
    "adjust_lighting_material (change lighting or materials, which moves nothing), "
    "add_context_object (add an object the scene is missing). "
    "Which actions are allowed depends on the criterion, so match the repair to the defect: a "
    "complaint about lighting or materials takes adjust_lighting_material, a complaint about how "
    "the scene works takes optimise_pose or change_support, and a complaint about what is present "
    "takes swap_asset or add_context_object. "
    "Do not omit any score: if you cannot judge one, say so in defects, but you must still return a "
    "number for every criterion."
)


def build_critic_prompt(payload: Mapping[str, Any]) -> str:
    """The user message. Built from the sanitised payload and nothing else."""
    targets = payload.get("task_targets", [])
    views = [view.get("label") for view in payload.get("views", [])]
    # The instance ids the critic may name are listed explicitly rather than left implicit: a model
    # asked to name an object it was not given has no way to know it should not, and the validation
    # on the way back would then be discarding suggestions the prompt invited.
    instance_ids = sorted(
        {str(target.get("semantic_id", "")) for target in targets} - {""}
    )
    return (
        f"Scene version (echo this back exactly): {payload.get('scene_version')}\n"
        f"Views provided, in order: {', '.join(str(v) for v in views)}\n"
        f"Task targets: {json.dumps(targets)}\n"
        f"Original scene request: {payload.get('task_prompt', '')}\n"
        f"Structured task: {json.dumps(payload.get('task', {}), sort_keys=True)}\n"
        f"Object ids you may name in target_id or object_id: {json.dumps(instance_ids)}\n"
        "Judge prompt adherence against the original request and functionality against the "
        "structured task. Score the criteria, propose your repairs, and return the JSON object."
    )


REPLANNER_SYSTEM_PROMPT = (
    "You are repairing a 3D indoor scene for a robotics simulation after the local repair ladder "
    "has been exhausted. You are given a compressed scene summary, the failure that survived, and "
    "the actions that are available to you. "
    "Respond with a single JSON object and nothing else, using exactly these keys: "
    "diagnosis (one sentence: why the local repairs could not fix this), "
    "operations (a list of objects, each with action, target_id and optionally synset or "
    "constraint). "
    "action must be exactly one of: move (place this object somewhere else), "
    "swap (use a different asset of the same kind), change_support (stand this object on "
    "something else), add (add a new object of a given synset), remove (delete an optional "
    "object), rollback (undo the most recent changes to this object's neighbourhood), "
    "terminate (report that the task cannot be met). "
    "You may not delete an object the summary marks as required, and you may not weaken the task. "
    "Prefer the fewest operations that address the stated failure; a scene rewritten wholesale is "
    "not a repair."
)


def build_replanner_prompt(summary: Mapping[str, Any], *, allowed_actions: Sequence[str]) -> str:
    """The user message for a replan. Built from the summary and the allowed actions only.

    The summary is already compressed and sanitised by its producer; nothing else is added here,
    because a replanner that can see the repository can be asked questions it should not answer.
    """
    return (
        "Scene summary:\n"
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n"
        f"Available actions: {json.dumps(sorted(allowed_actions))}\n"
        "Return the JSON object."
    )


def require_external_permission() -> None:
    """Acceptance/compiler opt-in, checked before any task text or image is sent."""
    if os.environ.get("PVSCENE_ALLOW_EXTERNAL_VLM") != "1":
        raise SystemExit(
            "external model calls require PVSCENE_ALLOW_EXTERNAL_VLM=1; "
            "this sends task text and, for the visual critic, render images off this machine"
        )
