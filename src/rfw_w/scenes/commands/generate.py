# SPDX-License-Identifier: Apache-2.0
"""Generate a scene from a task specification or an explicitly enabled compiler."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
from rfw_w.scenes.assembly.lighting import LIGHTING_PROFILES
from rfw_w.scenes.runtime import DEFAULT_EXPERIMENT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Where the run goes. `--experiment` + `--run` is the normal form and resolves to a fixed
    # path under `outputs/`; `--spec` + `--output-dir` remains for a probe or a one-off that has
    # no experiment of its own.
    parser.add_argument("--experiment", default=None,
                        help=f"a bundled example (default {DEFAULT_EXPERIMENT!r} "
                             f"when --spec is not given). Resolves the spec and the output path")
    parser.add_argument("--run", default=None,
                        help="the run name, one level under the experiment: "
                             "outputs/<experiment>/<run>")
    parser.add_argument("--spec", type=Path, default=None,
                        help="a spec file, instead of --experiment")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="an explicit directory, instead of --run. Required with --spec")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--prompt", help="task text to compile into a SceneSpec")
    inputs.add_argument("--prompt-file", type=Path, help="UTF-8 task text to compile")
    compiler = parser.add_mutually_exclusive_group()
    compiler.add_argument("--compile", action="store_true", help="compile the experiment prompt before generation")
    compiler.add_argument("--no-compile", action="store_true", help="use the supplied JSON spec directly (default)")
    parser.add_argument("--compile-only", action="store_true", help="write the spec and core/room review sketch, without assets or Isaac")
    parser.add_argument("--compiler-model", help="compiler model (or PVSCENE_COMPILER_MODEL / PVSCENE_VLM_MODEL)")
    parser.add_argument("--seed", type=int, default=None, help="seed for prompt input; defaults to reference seed or 11")
    parser.add_argument("--scene-id", default=None, help="identity for prompt input; defaults to reference ID or prompt_scene")
    parser.add_argument("--asset-bindings", type=Path, help="JSON mapping task target entity IDs to completed asset-package directories; relative paths belong to the JSON file")
    parser.add_argument("--asset-license", help="explicit license declaration for the bound assets, independent of the code license")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument(
        "--domain-randomization", action="store_true",
        help="sample the recorded non-task domain while freezing the task-relation closure",
    )
    parser.add_argument(
        "--variant-seed", type=int, default=None,
        help="replayable domain seed; supplying it also enables domain randomization",
    )
    parser.add_argument(
        "--max-variant-attempts", type=int, default=8,
        help="bounded room-seed resamples used to preserve the recorded task closure",
    )
    parser.add_argument("--shell-provider", choices=("builtin", "infinigen"), default="builtin")
    parser.add_argument(
        "--lighting-profile", choices=LIGHTING_PROFILES, default="mixed",
        help="SPEC-008's calibrated lighting profile, sized to the room it lights. `mixed` is the "
             "only one measured; the other two render and say `not_run` in the delivery's "
             "`lighting` block rather than pretending to be calibrated",
    )
    parser.add_argument(
        "--room-bounds", default=None,
        help="JSON room size_bounds_m overriding the spec's own, for the lighting matrix's "
             "small/large rooms (e.g. '[[4,5],[3,4],[2.7,3]]'). Needs --spec, and the run says so "
             "in the delivery: the effective spec carries the overridden bounds",
    )
    parser.add_argument(
        "--background-generation", action="store_true",
        help="SPEC-011's opt-in: build the shell with the task's reservations injected, let "
             "Infinigen realise background content, and refuse (inside the build) anything that "
             "stands in a reserved region or is not one of --background-gaps. Needs "
             "--shell-provider infinigen",
    )
    parser.add_argument(
        "--background-gaps", default="",
        help="comma-separated categories generation may add (ADR-009 decision 4). Required with "
             "--background-generation: without it the pool would refuse everything the build makes",
    )
    parser.add_argument("--skip-isaac", action="store_true")
    parser.add_argument(
        "--request-only", action="store_true",
        help="write the Isaac request and stop, for iterating on cameras without re-running the "
             "slow furnishing stage",
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help="run the pipeline straight through without the escalation ladder. Kept so a delivery "
             "can be produced for comparison; the loop is the default because a run without it "
             "cannot report where its failures were repaired",
    )
    parser.add_argument(
        "--deadline-s", type=float, default=1800.0,
        help="the whole run's wall-clock budget. Generous by default because the ladder may "
             "re-render after an appearance repair, and each render is minutes",
    )
    parser.add_argument(
        "--max-model-requests", type=int, default=3,
        help="the design's budget: compile, an optional stage critic, and a final critic, with 5 "
             "as the hard ceiling. A replan and a re-critique both count",
    )
    parser.add_argument(
        "--max-local-repairs", type=int, default=6,
        help="how many distinct failures may be given a local repair job",
    )
    parser.add_argument(
        "--no-replan", action="store_true",
        help="never call the model from the loop. The ladder still runs to L4 and then reports the "
             "exhaustion; used to demonstrate a repair that provably spends no model request",
    )
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument(
        "--agent-model", default=None,
        help="text-only L5 replanner model; independent of --vlm-model so the agent brain can be "
             "pinned without sending render images to it (or set PVSCENE_AGENT_MODEL)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from rfw_w.scenes.application import GenerationOptions, generate
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return generate(GenerationOptions(**vars(args)))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
