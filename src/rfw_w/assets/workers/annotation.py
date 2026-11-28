# SPDX-License-Identifier: Apache-2.0
"""Part VLM worker; geometry and physical calculations remain independent."""

from __future__ import annotations
import argparse, base64, json, os
from pathlib import Path

MATERIAL_ENUM = (
    "metal, plastic, wood, glass, ceramic, rubber, fabric, leather, foam, "
    "food, stone, paper, unknown"
)

SYSTEM_PROMPT = (
    "You annotate rigid 3D object parts for robotic manipulation. Image 1 is a "
    "front/left/right RGB grid. Image 2 uses the same cameras and gives each part "
    "a solid color. Infer each colored part's semantics, real material and friction "
    "against rubber. Also estimate whole-object dimensions and mass. Return JSON only."
)

USER_PROMPT = """Object category hint: {category}
The mask has exactly {count} parts named by these colors: {colors}.
Return this JSON schema:
{{"object": {{"name": "...", "category": "...", "dimension_cm": [L, W, H], "mass_g": 0}},
 "parts": [{{"mask_color": "one listed color", "name": "functional part name",
 "material": {{"category": "one of: {materials}", "subtype": "specific material or empty"}},
 "static_friction": 0.0, "dynamic_friction": 0.0,
 "priority_rank": 1, "graspable": true,
 "Basic_description": "appearance and location",
 "Functional_description": "function",
 "Grasped_description": "how a gripper should approach"}}]}}
dimension_cm contains three positive centimeter dimensions, longest first. mass_g is positive.
Return one parts entry per listed color, including small or occluded parts. This is one rigid body;
do not report joints, movement ranges, damping, parents or neighbors."""


def _image_data(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _parse_json(text: str) -> dict:
    import json_repair

    if "```" in text:
        text = text.split("```", 2)[1].removeprefix("json").strip()
    parsed = json_repair.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("VLM response is not a JSON object")
    return parsed


def _client(args):
    from openai import AzureOpenAI, OpenAI

    if not args.endpoint or not args.model:
        raise ValueError("--endpoint and --model are required")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"API key environment variable is not set: {args.api_key_env}")
    if args.api_version:
        return AzureOpenAI(
            azure_endpoint=args.endpoint,
            api_key=api_key,
            api_version=args.api_version,
            timeout=300,
            max_retries=3,
        )
    return OpenAI(base_url=args.endpoint, api_key=api_key, timeout=300, max_retries=3)


def _query(client, model: str, prompt: str, rgb: Path, mask: Path) -> dict:
    content = [{"type": "text", "text": prompt}]
    for path in (rgb, mask):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_image_data(path)}"},
            }
        )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }
    if "gpt-5" in model.lower() or "gpt5" in model.lower():
        kwargs["max_completion_tokens"] = 8192
    else:
        kwargs.update({"max_tokens": 8192, "temperature": 0.1})
    response = client.chat.completions.create(**kwargs)
    return _parse_json(response.choices[0].message.content or "{}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--parts-dir", required=True)
    p.add_argument("--renders-dir", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", required=True)
    p.add_argument("--api-version")
    p.add_argument("--render-size", type=int, default=512)
    args = p.parse_args()
    import numpy as np
    from rfw_w.assets.contracts.io import load_manifest, part_dir, write_json
    from rfw_w.assets.annotation.rendering import build_grids
    from rfw_w.assets.annotation.palette import NAMES
    from rfw_w.assets.annotation.parts import validate

    client = _client(args)
    for row in load_manifest(args.manifest):
        root = part_dir(args.parts_dir, row)
        ids = sorted(
            int(i) for i in np.unique(np.load(root / "face_ids.npy")) if i >= 0
        )
        rgb, mask = build_grids(
            row["asset_id"],
            root / "mesh_part_seg.glb",
            part_dir(args.renders_dir, row),
            root / "vlm_grids",
            ["front", "left", "right"],
            args.render_size,
        )
        prompt = USER_PROMPT.format(
            category=row.get("category", row["class_id"]),
            count=len(ids),
            colors=", ".join(NAMES[i] for i in ids),
            materials=MATERIAL_ENUM,
        )
        raw = _query(client, args.model, prompt, rgb, mask)
        validate(raw, ids)
        write_json(root / "annotations.json", raw)


if __name__ == "__main__":
    main()
