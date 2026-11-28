#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gather the paper's finished asset sets into one reviewable directory.

This is a **convenience bundle for a human to look through**, not a new piece of evidence.
Every file is copied byte for byte from a set that was already packaged and, where a verifier
exists, already independently checked; nothing is re-rendered, re-scaled or recoloured here.
The manifest records where each file came from and re-hashes it after the copy, so the bundle
can be held to its sources rather than trusted.

What it deliberately does NOT do: re-run any gate, or upgrade any `not_run` to a pass. Every
set in here still carries `user_visual_review: not_run` -- the machine checks being green is
what lets these be looked at, not a substitute for looking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()

#: Each entry is one already-packaged set. `source` is relative to the repo root so the bundle
#: names no absolute path on this machine (AGENTS.md rule 3).
SETS = [
    {
        "dir": "01_three_rooms_workspace",
        "source": "outputs/_paper/three_scene_assets_v2_20260926_01",
        "what": "office / counter / bedroom, each: room overview RGB, workspace close-up RGB, "
                "the same view's collision RGB, a viridis_r depth preview, and the float32 "
                "camera-z depth in metres behind it",
        "engine": "Isaac Sim 4.5.0-rc.36",
        "verified": "independent panel check: verified=true, problems=[]; all three renders passed with render_metrics_gaps empty; close-ups 1600x1200",
        "spec": "SPEC-013",
    },
    {
        "dir": "02_three_rooms_topview",
        "source": "outputs/_paper/dollhouse_topview_panels_02",
        "what": "the same three rooms from above (dollhouse), RGB + collision + depth preview + "
                "raw metric depth, plus a 3x3 montage. Depth colour scale is shared across the "
                "three rooms at 4.42-6.61 m",
        "engine": "Isaac Sim 4.5.0-rc.36",
        "verified": "six worker passes all passed, render_metrics with no gaps; specialised "
                    "tests 43 passed",
        "spec": "SPEC-013",
    },
    {
        "dir": "03_five_engines_one_viewpoint",
        "source": "outputs/_paper/spec014_five_sim_rgb_assets_v4_20260926_01",
        "what": "ONE office workspace scene and ONE frozen camera, rendered by five simulators: "
                "Genesis, Isaac Sim, MuJoCo, PyBullet, SAPIEN. Shared light rig, one exposure "
                "target, one camera. Five 1600x1000 raw PNGs plus each backend's own report",
        "engine": "Genesis / Isaac Sim / MuJoCo 3.12.0 / PyBullet / SAPIEN",
        "verified": "independent verifier (separate process, reads only the package): "
                    "verified=true, problems=[], 32/32 machine checks, not_run empty. "
                    "Calibration-patch luma 69.68-70.44 across all five",
        "spec": "SPEC-014",
    },
    {
        "dir": "04_room_alignment_figure",
        "source": "outputs/_paper/spec013_alignment_20260924_06",
        "what": "the composed room-alignment figure (PDF + PNG, 183 mm at 300 dpi) with its "
                "depth companion page, the per-panel sources, and the measurements behind it",
        "engine": "Isaac Sim 4.5.0-rc.36",
        "verified": "independent check: verified=true, problems=[], 39 machine checks, all 30 "
                    "panels traceable; specialised tests 13 passed",
        "spec": "SPEC-013",
    },
]

#: Stated once here rather than per set, because it is true of every set and the whole point of
#: the bundle. A green machine gate is not a visual acceptance and never becomes one by copying.
VISUAL_REVIEW = "not_run"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def copy_set(entry: dict, output_dir: Path) -> dict:
    source = REPO_ROOT / entry["source"]
    if not source.is_dir():
        raise SystemExit(f"PAPER_BUNDLE_SOURCE_MISSING({entry['source']})")
    destination = output_dir / entry["dir"]
    files = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        before, after = digest(path), digest(target)
        if before != after:
            raise SystemExit(f"PAPER_BUNDLE_COPY_MISMATCH({entry['dir']}/{relative})")
        files.append({"path": str(relative), "sha256": after,
                      "source": f"{entry['source']}/{relative}"})
    return {**{k: v for k, v in entry.items() if k != "source"},
            "source": entry["source"], "file_count": len(files),
            "bytes": sum((destination / f["path"]).stat().st_size for f in files),
            "user_visual_review": VISUAL_REVIEW, "files": files}


README = """# Paper assets, {date}

Four finished sets, copied byte for byte from where each was packaged. Nothing here was
re-rendered or edited; `MANIFEST.json` carries every file's SHA-256 and the path it came from.

{table}

## The one thing to read before judging these

Every set says `user_visual_review: not_run`. The machine checks in each set are green, and
that is exactly what they are for -- they say the image is the resolution it claims, from the
scene version it claims, copied without edits, exposed inside a declared band. **None of them
can say the picture is good.** That judgement is not automated anywhere in this repository and
is not recorded as passed until a person says so.

## Known gaps, already measured, not fixed

- **`03`, Isaac column**: the framed art on the wall renders as a flat colour, where MuJoCo and
  PyBullet show the actual orange/black painting. Isaac's RTX renderer cannot evaluate the
  Blender `Image_Texture`/`UV_Map` node graph these assets carry. It sits on an instance rather
  than on the room shell, so the shell-material override that fixed the same problem for the
  walls does not reach it.
- **`03`, SAPIEN and Genesis**: the 0.80 white walls bounce enough light in a path tracer to
  flatten the contrast; the earlier 0.62 grey shell had more depth to it. Changing back is one
  number in `experiments/paper_multisim/lighting.json` and one re-render per backend.
- **`01`, the `counter` room**: this row was called `kitchen` until 2026-09-26. The room has an
  island, a stool, a cabinet and a plant and no stove, sink or fridge -- its scene prompt never
  asked for any -- so the close-up could not be made to read as a kitchen by any camera, and the
  label was the thing that was wrong. It is now named for what is in frame. Making it an actual
  kitchen means changing the scene spec and re-running that room's pipeline; that has not been
  done.
- **`01`, office overview**: most of the frame is empty floor with the desk pushed into a corner.
- **`02`** is still framed by the older fixed-offset rule and its `kitchen` row still carries
  that name. Only `01` was re-rendered and re-labelled.

## Provenance

Each subdirectory keeps the `provenance.json` it was packaged with. The top-level
`MANIFEST.json` is this bundle's own record and does not replace them.

Rebuild with:

    env -u PYTHONPATH -u PYTHONHOME /opt/conda/envs/scenesmith/bin/python \\
      python -m rfw_w.scenes.visualization.bundle_paper_assets --output-dir <fresh directory>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="fresh directory; refuses one that already has files in it")
    parser.add_argument("--date", default="2026-09-26")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        # AGENTS.md rule 2: a re-run writes a new directory, it never repairs an old one.
        raise SystemExit(f"PAPER_BUNDLE_OUTPUT_NOT_EMPTY({output_dir})")
    output_dir.mkdir(parents=True, exist_ok=True)

    sets = [copy_set(entry, output_dir) for entry in SETS]
    manifest = {
        "claim": "four already-packaged paper asset sets, copied byte for byte into one place",
        "bundled_on": args.date,
        "image_processing": "none; every file is a byte-for-byte copy",
        "user_visual_review": VISUAL_REVIEW,
        "builder": "python -m rfw_w.scenes.visualization.bundle_paper_assets",
        "sets": sets,
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    rows = ["| dir | what | engine | spec |", "|---|---|---|---|"]
    rows += [f"| `{s['dir']}` | {s['what']} | {s['engine']} | {s['spec']} |" for s in sets]
    (output_dir / "README.md").write_text(
        README.format(date=args.date, table="\n".join(rows)))

    for record in sets:
        print(f"{record['dir']:32s} {record['file_count']:3d} files  "
              f"{record['bytes'] / 1e6:7.1f} MB  <- {record['source']}")
    print(f"total {sum(s['file_count'] for s in sets)} files -> {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
