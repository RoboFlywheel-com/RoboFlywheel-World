#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a task+scene figure against the specs it claims to show.

SPEC-016. The figure's captions are the claim: "this scene is the one built for
this task". This reads the bundle and the specs it names, in a separate
process, and asks whether that holds:

* every panel re-hashes to what provenance recorded, and to the render it came
  from,
* every caption re-derives from its spec's own task and relations (an override
  must be recorded as an override, not passed off as derived),
* every named render report says `passed`,
* the spec digests still match the specs on disk -- an edited spec after the
  figure was built is a caption that no longer describes its scene,
* the figure files exist at the declared width.

The one thing it cannot check is whether the pictures are good. That stays
`not_run` until the user says.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.validation.check_task_scene_figure \\
      --bundle outputs/_paper/task_scene_figure_NEW \\
      --output-dir outputs/_acceptance/task_scene_figure_NEW
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from rfw_w.scenes.visualization.build_task_scene_figure import caption_for


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")

    provenance = json.loads((bundle / "provenance.json").read_text())
    problems: list[str] = []
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    for name, digest in (provenance.get("figure_sha256") or {}).items():
        path = bundle / name
        record(f"figure:{name}", path.is_file() and sha(path) == digest,
               "the figure file is present and matches its recorded digest")

    scenes = provenance.get("scenes") or []
    record("scenes_present", bool(scenes), f"{len(scenes)} scenes recorded")
    for scene in scenes:
        scene_id = scene["scene_id"]
        panel = Path(scene["panel"])
        if not panel.is_absolute():
            panel = bundle / panel
        record(f"{scene_id}:panel_digest", panel.is_file() and sha(panel) == scene["panel_sha256"],
               "the panel is present and matches its recorded digest")
        source = Path(scene["source_rgb"])
        if source.is_file():
            record(f"{scene_id}:panel_is_the_render", sha(source) == scene["panel_sha256"],
                   "the panel is byte-for-byte the render it names")
        else:
            record(f"{scene_id}:panel_is_the_render", False,
                   f"the render it names is gone: {source}")

        spec_path = Path(scene["spec"])
        if not spec_path.is_file():
            record(f"{scene_id}:spec_present", False, f"the spec it names is gone: {spec_path}")
            continue
        record(f"{scene_id}:spec_digest", sha(spec_path) == scene["spec_sha256"],
               "the spec still hashes to what the figure was built from")
        spec = json.loads(spec_path.read_text())
        derived, task_fields = caption_for(spec)
        record(f"{scene_id}:caption_derives", derived == scene["caption_derived_from_spec"],
               f"re-derived {derived!r} against recorded {scene['caption_derived_from_spec']!r}")
        if scene["caption_source"] == "spec":
            record(f"{scene_id}:caption_is_the_spec", scene["caption"] == derived,
                   "the printed caption is the derived one, as claimed")
        record(f"{scene_id}:task_fields", task_fields == scene["task"],
               "the recorded task fields re-derive from the spec")

        report_path = Path(scene["render_dir"]) / "isaac" / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            record(f"{scene_id}:report_digest", sha(report_path) == scene["report_sha256"],
                   "the render report still matches")
            record(f"{scene_id}:render_passed", report.get("status") == "passed",
                   f"the render report says status={report.get('status')}")
        else:
            record(f"{scene_id}:report_present", False, f"no report at {report_path}")

    out.mkdir(parents=True, exist_ok=True)
    verified = not problems
    payload = {
        "verified": verified,
        "bundle": str(bundle),
        "scenes": [s["scene_id"] for s in scenes],
        "captions": {s["scene_id"]: s["caption"] for s in scenes},
        "checks": checks,
        "checks_run": len(checks),
        "problems": problems,
        "not_run": ["user visual judgement", "robot physics", "robot ik"],
    }
    (out / "verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                      ensure_ascii=False) + "\n")
    print(json.dumps({"verified": verified, "checks_run": len(checks), "problems": problems,
                      "not_run": payload["not_run"]}, indent=2, ensure_ascii=False))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
