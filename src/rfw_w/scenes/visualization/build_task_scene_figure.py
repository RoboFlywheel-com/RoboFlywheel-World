#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One figure of task + scene: each panel is a scene, captioned with its own task.

SPEC-016. The earlier paper figures compared *renderers* (five columns, one
scene). This compares *scenes*: what the task is, and what the room looks like
when a robot is in it to do it.

The caption is **derived from the spec**, not typed here: the action, the target
and the surface the target starts on come from ``task`` and
``initial_relations``, so a caption cannot drift from the scene it sits under.
A row may override the wording, and then the override is recorded as such.

Writes, under ``--output-dir``:

    figure_task_scenes.pdf / .png   183 mm wide, 300 dpi
    panels/<scene>_rgb.png          byte-for-byte copies of what the figure shows
    provenance.json                 per scene: spec digest, task fields, caption,
                                    render report digest, panel digest

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.visualization.build_task_scene_figure \\
      --selection experiments/paper_embodiment_matrix/figure_selection.json \\
      --output-dir outputs/_paper/task_scene_figure_NEW
"""

from __future__ import annotations
from rfw_w.scenes.visualization.inputs import resolve_input

import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MM = 1 / 25.4
#: How the spec's action reads in a caption. Unknown actions are written through
#: as-is rather than guessed at -- a caption that invents a verb is a caption
#: that no longer describes the scene.
ACTION_VERBS = {"pick_place": "Pick"}
CAMERA = "paper_workspace_detail"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def caption_for(spec: dict) -> tuple[str, dict]:
    """The task sentence, derived from the spec's own task and relations."""
    task = spec["task"]
    target = task["targets"][0]
    support = None
    for relation in spec.get("initial_relations", []):
        if relation.get("subject") == target and relation.get("predicate") == "ON":
            support = relation.get("object")
            break
    entities = {e["id"]: e for e in spec["entities"]}
    target_name = entities.get(target, {}).get("category", target).replace("_", " ")
    support_name = entities.get(support, {}).get("category", support or "surface").replace("_", " ")
    verb = ACTION_VERBS.get(task["action"], task["action"])
    text = f"{verb} the {target_name} from the {support_name}"
    return text, {"action": task["action"], "target": target, "target_category": target_name,
                  "support": support, "support_category": support_name,
                  "robot_profile": task.get("robot_profile")}


def build(selection_path: Path, output: Path, columns: int) -> dict:
    selection = json.loads(selection_path.read_text())
    rows = selection["rows"]
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"OUTPUT_DIR_NOT_FRESH({output})")
    (output / "panels").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for row in rows:
        scene_id = row["scene_id"]
        spec_path = resolve_input(row["spec"], base=selection_path.parent)
        spec = json.loads(spec_path.read_text())
        render_dir = Path(row["render_dir"]).resolve()
        report_path = render_dir / "isaac" / "report.json"
        report = json.loads(report_path.read_text())
        if report.get("status") != "passed":
            raise ValueError(f"TASK_SCENE_RENDER_NOT_PASSED({scene_id})")
        capture = next((c for c in report["captures"] if c["camera_id"] == row.get("camera", CAMERA)),
                       None)
        if capture is None:
            raise ValueError(f"TASK_SCENE_CAPTURE_MISSING({scene_id},{row.get('camera', CAMERA)})")
        source = render_dir / "isaac" / capture["rgb_path"]
        panel = output / "panels" / f"{scene_id}_rgb.png"
        shutil.copy2(source, panel)

        derived, task_fields = caption_for(spec)
        caption = row.get("caption", derived)
        records.append({
            "scene_id": scene_id,
            "room_kind": spec["room"].get("room_kind"),
            "seed": spec.get("seed"),
            "prompt": spec.get("prompt"),
            "caption": caption,
            "caption_source": "spec" if caption == derived else "manual override",
            "caption_derived_from_spec": derived,
            "task": task_fields,
            "spec": str(spec_path),
            "spec_sha256": sha(spec_path),
            "render_dir": str(render_dir),
            "report_sha256": sha(report_path),
            "panel": str(panel),
            "panel_sha256": sha(panel),
            "source_rgb": str(source),
            "resolution": capture["resolution"],
        })

    count = len(records)
    cols = min(columns, count)
    rows_n = math.ceil(count / cols)
    with Image.open(records[0]["panel"]) as first:
        aspect = first.size[1] / first.size[0]
    width_mm = 183.0
    cell_w = width_mm / cols
    subtitle_lines = (len(textwrap.wrap(selection.get("subtitle", ""), 150))
                      if selection.get("subtitle") else 0)
    # The header is measured, not guessed: a four-line subtitle over a fixed 12 mm
    # band printed straight through the first row's captions.
    header_mm = 7.0 + 2.6 * subtitle_lines
    # 7 mm per row for the caption above, 5 mm at the foot for the last row's
    # scene line -- cut to 1%, the bottom row's label printed off the page.
    fig_h = rows_n * (cell_w * aspect + 7.0) + header_mm + 5.0
    fig, axes = plt.subplots(rows_n, cols, figsize=(width_mm * MM, fig_h * MM),
                             facecolor="white", squeeze=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - header_mm / fig_h, bottom=5.0 / fig_h,
                        wspace=0.03, hspace=0.30)
    for index, ax in enumerate(axes.ravel()):
        if index >= count:
            ax.axis("off")
            continue
        record = records[index]
        with Image.open(record["panel"]) as image:
            ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{index + 1}. {record['caption']}", loc="left", fontsize=6.5, pad=2)
        ax.text(0.0, -0.045, f"{record['room_kind']} · seed {record['seed']} · {record['scene_id']}",
                transform=ax.transAxes, fontsize=5.2, color="#555555", va="top")
    fig.text(0.008, 0.995, selection.get("title", "Tasks and the scenes generated for them"),
             fontsize=9, weight="bold", va="top")
    subtitle = selection.get("subtitle")
    if subtitle:
        # Wrapped rather than clipped: a one-line subtitle runs off a 183 mm page
        # and the reader loses the half that says what is not being claimed.
        fig.text(0.008, 0.995 - 4.2 / fig_h, "\n".join(textwrap.wrap(subtitle, 150)),
                 fontsize=5.4, color="#444444", va="top", linespacing=1.45)
    fig.savefig(output / "figure_task_scenes.pdf", dpi=300)
    fig.savefig(output / "figure_task_scenes.png", dpi=300)
    plt.close(fig)

    payload = {
        "claim": "one panel per scene, captioned from that scene's own spec; the panels are "
                 "byte-for-byte copies of the renders",
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha(selection_path),
        "columns": cols,
        "scenes": records,
        "figure_width_mm": width_mm,
        "figure_sha256": {name: sha(output / name)
                          for name in ("figure_task_scenes.pdf", "figure_task_scenes.png")},
        "user_visual_review": "not_run",
        "robot_physics": "not_run",
        "robot_ik": "not_run",
    }
    (output / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                       ensure_ascii=False) + "\n")
    return {"output": str(output), "scenes": [r["scene_id"] for r in records],
            "captions": [r["caption"] for r in records], "columns": cols}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--columns", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.selection, args.output_dir.resolve(), args.columns),
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
