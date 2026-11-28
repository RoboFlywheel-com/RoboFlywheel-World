#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ask two base models to critique the same frozen scene, and put the answers side by side.

The question this answers is "is it the scene or is it the judge?". Generation in this project is
deterministic -- layout, retrieval and placement are solver code with no model in the path -- so
re-running a generation with a different model would produce an identical scene and answer nothing.
The only model in the loop is the critic, and the way to measure it is to hold the scene fixed and
vary the critic.

So this reads a finished delivery, rebuilds the exact prompt and the exact montage that the run
sent, and sends them to each configured model. Everything is shared between the arms: the payload,
the images, the system prompt, the validation. Only the endpoint and the model name differ.

**This sends a rendered scene to an external service.** It is a manual tool, never a test, and it
refuses to run without PVSCENE_ALLOW_EXTERNAL_VLM=1 for the same reason the VLM test tier does.

    PVSCENE_ALLOW_EXTERNAL_VLM=1 python -m rfw_w.scenes.validation.compare_critics \
        --delivery outputs/office_pick_cup/m1_009 --env-file .env.local \
        --out-dir outputs/_probe/critic_comparison

The output directory is required rather than defaulted into the delivery. It used to default to
`<delivery>/critic_comparison`, which writes four new files -- including a montage -- into a tree
that `AGENTS.md` calls evidence and says is never edited in place. Two critics on one scene is
evidence; it belongs beside the comparison that produced it, not inside the delivery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# The criteria, in the manual's order, so a side-by-side table reads the same way every time.
CRITERIA = (
    "prompt_adherence",
    "functionality",
    "style_material_coherence",
    "proportion_density",
    "composition_lighting",
    "lifestyle_plausibility",
    "workspace_and_passage",
)


def load_env_file(path: Path) -> None:
    """Same loader the acceptance script uses, so both read the environment identically."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def arms_from_env(spec: str) -> list[dict[str, str]]:
    """Parse `label=model` pairs, resolving endpoint and credential from the environment.

    The endpoint and the key variable are looked up per arm rather than passed in, so the same
    invocation compares a DashScope model against a gateway model with no flags beyond the two
    names.
    """
    arms: list[dict[str, str]] = []
    for item in spec.split(","):
        label, _, model = item.strip().partition("=")
        if not model:
            raise SystemExit(f"--models entries must be label=model, not {item!r}")
        prefix = label.upper().replace("-", "_")
        arms.append(
            {
                "label": label,
                "model": model,
                "base_url": (
                    os.environ.get(f"{prefix}_BASE_URL")
                    or os.environ.get("SIMFOUNDRY_OPENAI_BASE_URL", "")
                ),
                "api_key_env": os.environ.get(f"{prefix}_API_KEY_ENV", "DASHSCOPE_API_KEY"),
                "wire": os.environ.get(f"{prefix}_WIRE", "chat_completions"),
            }
        )
    return arms


def build_case(delivery: Path) -> tuple[dict, str, list[str], bytes]:
    """Rebuild exactly what the run sent: the payload, the prompt, the labels, the montage bytes."""
    from rfw_w.scenes.agents.llm_client import build_critic_prompt
    from rfw_w.scenes.agents.montage import build_montage
    from rfw_w.scenes.agents.visual_critic import sanitised_payload

    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    scene_version = manifest["scene_version"]
    images = sorted((delivery / "isaac").glob("*.png"))
    if not images:
        raise SystemExit(f"no renders under {delivery / 'isaac'}; nothing to critique")
    png, labels = build_montage([(image.stem, image) for image in images], columns=3)

    instance_ids = sorted(manifest["placement"]["instances"])
    payload = sanitised_payload(
        scene_version=scene_version,
        views=[{"label": label} for label in labels],
        targets=[{"semantic_id": i, "category": i} for i in instance_ids],
    )
    return payload, build_critic_prompt(payload), labels, png


def ask(arm: dict[str, str], *, prompt: str, montage_url: str, scene_version: str,
        known_targets: tuple[str, ...], max_tokens: int = 4096) -> dict:
    """One arm: one request, one strict-schema parse, every refusal recorded."""
    from rfw_w.scenes.agents.llm_client import (
        CRITIC_SYSTEM_PROMPT,
        LLMConfig,
        LLMError,
        call_vision,
    )
    from rfw_w.scenes.agents.visual_critic import parse_critique

    config = LLMConfig(
        base_url=arm["base_url"], model=arm["model"],
        api_key_env=arm["api_key_env"], wire=arm["wire"], max_tokens=max_tokens,
    )
    record: dict = {"label": arm["label"], "model": arm["model"], "config": config.to_dict()}
    try:
        response = call_vision(
            config, system_prompt=CRITIC_SYSTEM_PROMPT, user_prompt=prompt, images=[montage_url]
        )
    except LLMError as error:
        record["error"] = str(error)
        return record

    critique = parse_critique(
        response.text, reviewed_scene_hash=scene_version, current_scene_hash=scene_version,
        model_id=response.model_id, tokens_used=response.tokens_used, known_targets=known_targets,
    )
    record["response"] = response.to_dict()
    record["critique"] = critique.to_dict()
    record["raw_text"] = response.text
    return record


def mean_score(critique: dict) -> str:
    """The average of the seven scores, which is what a one-line comparison actually needs.

    A model that scores every criterion the same has a flatter, less discriminating judgement than
    one that separates them, and the mean alone hides that -- which is why the mean is printed next
    to the per-criterion row rather than instead of it.
    """
    scores = critique.get("scores") or {}
    values = [scores[name] for name in CRITERIA if name in scores]
    return f"{sum(values) / len(values):.2f}" if values else "-"


def render(records: list[dict]) -> str:
    """The side-by-side, in the form a human reads rather than the form a machine stores."""
    out: list[str] = []
    ok = [r for r in records if "critique" in r]
    for record in records:
        out.append("=" * 78)
        out.append(
            f"ARM {record['label']}  model={record['model']}  "
            f"wire={record['config']['wire']}  attempt={record.get('attempt', 0)}"
        )
        out.append(f"  endpoint: {record['config']['base_url']}")
        if "error" in record:
            out.append(f"  ERROR: {record['error']}")
            continue
        critique = record["critique"]
        out.append(
            f"  status={critique['status']}  critical_defect={critique['critical_defect']}  "
            f"tokens={critique['tokens_used']}"
        )
        if critique.get("reason_code"):
            out.append(f"  INVALID ({critique['reason_code']}): {critique['reason_detail']}")
        scores = critique.get("scores") or {}
        out.append("  scores: " + "  ".join(
            f"{name}={scores.get(name, '-')}" for name in CRITERIA
        ))
        # An invalid critique carries no scores, and the raw text is the only thing that says why
        # the model's reply could not be read -- which is the whole finding when one arm is invalid
        # and the other is not.
        if critique["status"] != "valid":
            out.append("  raw reply (first 1500 chars):")
            out.append("    " + (record.get("raw_text") or "<empty>")[:1500].replace("\n", "\n    "))
        for defect in critique["defects"]:
            out.append(f"  defect: {defect}")
        for evidence in critique["object_evidence"]:
            out.append(f"  saw {evidence[0]}: {evidence[1]}")
        for suggestion in critique["suggestions"]:
            out.append(
                f"  repair: [{suggestion['criterion']}] {suggestion['action']} "
                f"-> {suggestion['target_id']}: {suggestion['constraint']}"
            )
        for refused in critique.get("rejected_suggestions", []):
            out.append(f"  REFUSED: {json.dumps(refused, sort_keys=True)}")
        out.append(f"  views claimed: {', '.join(critique['view_evidence'])}")
    if len(ok) == 2:
        out.append("=" * 78)
        out.append("DELTA (second minus first, per criterion)")
        first, second = ok
        if first["critique"].get("scores") and second["critique"].get("scores"):
            for name in CRITERIA:
                a = first["critique"]["scores"].get(name)
                b = second["critique"]["scores"].get(name)
                flag = "" if a == b else "   <-- differs"
                out.append(f"  {name:28s} {a} -> {b}{flag}")
        else:
            out.append("  not comparable: one arm did not return a valid critique")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--models", default="dashscope=,",
        help="comma-separated label=model. A label also names the environment prefix for its "
             "endpoint and key: DASHSCOPE_BASE_URL / DASHSCOPE_API_KEY_ENV / DASHSCOPE_WIRE.",
    )
    parser.add_argument("--montage-out", type=Path, default=None,
                        help="write the montage that was sent, for a human to look at too")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="output ceiling per arm; a reasoning model spends much of it on "
                             "reasoning and returns a truncated JSON object if this is too low")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="where to write the comparison. Required, because the delivery is the "
                             "input and writing into it puts a montage and a comparison beside the "
                             "evidence they are about")
    parser.add_argument("--repeat", type=int, default=1,
                        help="requests per arm. At temperature 1.0 a single sample per model is "
                             "weak evidence: two runs of the same model can differ by a point, so "
                             "a one-shot gap between models needs the within-model spread to read "
                             "it against")
    args = parser.parse_args()

    if os.environ.get("PVSCENE_ALLOW_EXTERNAL_VLM") != "1":
        raise SystemExit(
            "refusing to send a rendered scene to an external service without "
            "PVSCENE_ALLOW_EXTERNAL_VLM=1"
        )
    if args.env_file:
        load_env_file(args.env_file)

    payload, prompt, labels, png = build_case(args.delivery)
    known_targets = tuple(t["semantic_id"] for t in payload["task_targets"])
    montage_url = "data:image/png;base64," + __import__("base64").b64encode(png).decode("ascii")
    # Flushed, because a run of six requests against two providers takes minutes and Python block-
    # buffers stdout to a file: the first version of this printed nothing at all until it finished,
    # which is indistinguishable from a hang and gives no way to see which arm is slow.
    def say(line: str) -> None:
        print(line, flush=True)

    say(f"[case] scene {payload['scene_version'][:12]}  {len(labels)} views  "
        f"{len(png)/1e6:.2f} MB montage  {len(known_targets)} nameable objects")

    records = []
    for arm in arms_from_env(args.models):
        if not arm["model"]:
            arm["model"] = os.environ.get("PVSCENE_VLM_MODEL", "")
        say(f"[arm] {arm['label']}: {arm['model']} via {arm['wire']} at {arm['base_url']}")
        for attempt in range(args.repeat):
            record = ask(arm, prompt=prompt, montage_url=montage_url,
                         scene_version=payload["scene_version"], known_targets=known_targets,
                         max_tokens=args.max_tokens)
            record["attempt"] = attempt
            records.append(record)
            if "error" in record:
                say(f"  [{attempt}] error: {record['error']}")
            else:
                scores = record["critique"].get("scores") or {}
                say(f"  [{attempt}] status={record['critique']['status']} "
                      f"suggestions={len(record['critique']['suggestions'])} "
                      f"refused={len(record['critique'].get('rejected_suggestions', []))} "
                      f"mean={mean_score(record['critique'])}  "
                      + " ".join(str(scores.get(name, '-')) for name in CRITERIA))

    if args.repeat > 1:
        say("\nSPREAD ACROSS REPEATS (how much of a one-shot difference is just sampling)")
        for arm in arms_from_env(args.models):
            mine = [r for r in records if r["label"] == arm["label"] and "critique" in r]
            if len(mine) < 2:
                continue
            per_criterion = {}
            for name in CRITERIA:
                values = [
                    (r["critique"].get("scores") or {}).get(name)
                    for r in mine
                    if (r["critique"].get("scores") or {}).get(name) is not None
                ]
                if values:
                    per_criterion[name] = (min(values), max(values))
            means = [mean_score(r["critique"]) for r in mine]
            usable = [m for m in means if m is not None]
            say(f"  {arm['label']:10s} mean across repeats: "
                  f"{min(usable) if usable else '-'} .. {max(usable) if usable else '-'}"
                  + "".join(
                      f"\n      {name:28s} {lo}..{hi}" for name, (lo, hi) in per_criterion.items()
                      if lo != hi
                  )
                  + ("" if per_criterion else ""))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt)
    (out_dir / "montage.png").write_bytes(png)
    (args.montage_out or (out_dir / "montage.png")).write_bytes(png)
    (out_dir / "comparison.json").write_text(
        json.dumps(
            {
                "schema_version": "pvscene.critic-comparison/v1",
                "delivery": str(args.delivery),
                "scene_version": payload["scene_version"],
                "views": labels,
                "prompt": prompt,
                "prompt_note": (
                    "the original user request is NOT part of this prompt; the critic is given a "
                    "fixed task sentence and no statement of what was asked for"
                ),
                "arms": records,
            },
            indent=2, sort_keys=True,
        )
        + "\n"
    )
    table = render(records)
    (out_dir / "comparison.txt").write_text(table + "\n")
    print()
    say(table)
    say(f"\n[written] {out_dir}/comparison.txt  comparison.json  prompt.txt  montage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
