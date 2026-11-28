#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit the room shells for the three reviewed production experiments.

Why this exists
---------------

The room shell is about to gain shape families, per-kind size profiles and multi-opening
placement. Every one of those changes the polygon a spec produces, and a polygon is not
something a reviewer can eyeball in a diff. So the current geometry is frozen here first, and
the change that follows is judged by the **diff** against this file rather than by description.

The property this file protects is not "the rooms are right" -- it is "here is exactly what
changed, and by how much". A shape change that is intended shows up as a handful of moved
numbers; one that is not shows up the same way, which is the point.

It also pins reproducibility across seeds: the same ``(experiment, seed)`` must keep producing
the same shell, so a later phase that quietly re-orders a random draw fails here rather than at
render time.

Output is deterministic -- sorted keys, no timestamp, no paths -- so emitting twice produces
byte-identical files and this can be reviewed as a diff.

    New experiment coverage belongs to the diversity/window checks. This historical
    baseline keeps its explicit population until a reviewed baseline expansion.

    python -m rfw_w.scenes.preparation.emit_room_shell_baseline --check     # fail if the file is stale
    python -m rfw_w.scenes.preparation.emit_room_shell_baseline             # rewrite it
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.solvers.room import build_shell, generate_options

SCHEMA = "pvscene.room-shell-baseline/v1"
DEFAULT_OUT = data_path('fixtures/room_shell_baseline.json')

# Enough seeds to cover each `spec.seed % 3` residue at least six times, so a change to the
# notch dice cannot hide behind a run of unlucky draws.
SEEDS_PER_EXPERIMENT = 20


# This historical ruler was reviewed for these three production experiments. A paper
# case added to the working tree must not silently redefine the frozen population;
# current experiment coverage is checked separately by the diversity/window tests.
BASELINE_EXPERIMENTS = (
    "bedroom_pick_alarm_clock", "kitchen_fetch_jar", "office_pick_cup",
)


def experiments(*, root: Path | None = None) -> list[Path]:
    paths = [(root if root is not None else data_path('examples')) / name / 'task.json' for name in BASELINE_EXPERIMENTS]
    missing = [path.parent.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"BASELINE_EXPERIMENT_MISSING({','.join(missing)})")
    return paths


def shell_record(spec: SceneSpec, seed: int) -> dict:
    """One shell, recorded as the numbers that would have to move for anything to have changed.

    The resolved geometry is the record; ``options`` rides along so a change to the *draw* is
    separable from a change to the *outline*. Both moving together is a deliberate shape change;
    only one moving is a bug in the other.
    """
    sampled = replace(spec, seed=seed)
    options = generate_options(sampled)
    shell = build_shell(sampled, options=options)

    polygon = [[round(x, 6), round(y, 6)] for x, y in shell.polygon_xy]
    openings = [
        {
            "opening_id": opening.opening_id,
            "kind": opening.kind.value,
            "center_xy": [round(opening.center_xy[0], 6), round(opening.center_xy[1], 6)],
            "width_m": opening.width_m,
            "height_m": opening.height_m,
            "sill_z_m": opening.sill_z_m,
        }
        for opening in shell.openings
    ]
    return {
        "seed": seed,
        "family": options.family.value,
        "polygon_xy": polygon,
        "height_m": shell.height_m,
        "vertex_count": len(polygon),
        "openings": openings,
        # The whole resolved program, verbatim. Recording a hand-picked subset is how the fixture
        # would stop noticing a new field, which is the opposite of what it is for.
        "options": options.to_dict(),
        # One line to read in a diff: equal means nothing about this shell moved.
        "shell_hash": content_hash({"polygon_xy": polygon, "height_m": shell.height_m,
                                    "openings": openings}),
    }


def build_document(*, root: Path | None = None) -> dict:
    document: dict = {
        "schema": SCHEMA,
        "note": (
            "The room shell each spec produces, per seed, frozen so a shape change is reviewed as "
            "a diff against these numbers rather than described. Regenerate with "
            "python -m rfw_w.scenes.preparation.emit_room_shell_baseline only when the change is intended, and read "
            "the diff. T is now published: seed populations are reweighted across four families.\n"
            "Generated by python -m rfw_w.scenes.preparation.emit_room_shell_baseline. The unified shell policy "
            "chooses a shape family (rect/l/u/t) and the notches within it, and random draws come from "
            "named substreams seeded by 'SEED:shape' rather than by the integer seed directly -- "
            "so both the dimensions and the outline moved when families landed, which is expected "
            "and is why this file was regenerated in that change.\n"
            "Regenerated again when the bite became a drawn band (NOTCH_FRACTION_BAND) instead of "
            "the fixed 0.25 fraction. That time every seed's width, depth and height were "
            "deliberately unchanged -- the size draws sit on the integer seed and were not touched "
            "-- so the diff was 0 moved footprints, 0 moved heights, 0 moved families and 40 moved "
            "notches. A reader regenerating this file should run the same check: if a size moves "
            "when only the shape was meant to, that is the regression, not the review."
        ),
        "seeds_per_experiment": SEEDS_PER_EXPERIMENT,
        "experiments": {},
    }
    for path in experiments(root=root):
        spec = SceneSpec.from_dict(json.loads(path.read_text()))
        document["experiments"][path.parent.name] = {
            "base_seed": spec.seed,
            "shells": [
                shell_record(spec, spec.seed + offset)
                for offset in range(SEEDS_PER_EXPERIMENT)
            ],
        }
    return document


def serialise(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit non-zero if the committed file differs")
    args = parser.parse_args()

    if args.out is None:
        if not args.check:
            parser.error("--out is required when emitting a new baseline")
        args.out = DEFAULT_OUT
    document = build_document()
    wanted = serialise(document)
    if args.check:
        if not args.out.is_file():
            print(f"no baseline at {args.out}", file=sys.stderr)
            return 1
        if args.out.read_text() != wanted:
            print(
                f"{args.out} is stale: the shell a spec produces no longer matches the "
                "baseline. If the shape change was intended, regenerate it and review the "
                "diff; if it was not, this is the regression.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} is current")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(wanted)
    total = sum(len(item["shells"]) for item in document["experiments"].values())
    print(f"[written] {args.out}  ({total} shells across "
          f"{len(document['experiments'])} experiments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
