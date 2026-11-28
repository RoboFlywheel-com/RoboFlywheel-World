#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is the scene a function of the plan? Build one, build another, build the first again.

The loop's restore path depends on the answer. When every rung of the local ladder fails, the driver
rebuilds the plan the job started from (`driver.py`, the exhaustion branch) and expects the scene
version it started with. If it gets a different one, the restored scene is one no gate has measured,
and the run reports `draft` with every gate stale -- however well the restored scene had measured.

The note the driver emits says that happened and not where. This script says where: it builds the
same plan twice with a different plan built in between, which is the restore's own sequence, and
prints what moved. No Isaac and no model, so it costs a few minutes rather than half an hour.

    python -m rfw_w.scenes.validation.check_build_purity --experiment office_pick_cup --env-file .env.local \\
        --output-dir outputs/_probe/build_purity_20260917_001

A fresh output directory is required, as for every other producer here: the run writes prepared
assets into it, and a reused directory would let the previous run's cache answer for this one --
which is one of the things this script exists to detect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()


def _plan_a_change(host: object) -> object:
    """A repair that genuinely edits the plan, taken from the ladder rather than invented.

    `resample_pose` advances the placement sweep of a floor piece, which the plan records as
    `skip_candidates` -- a field its `to_dict` carries, so the edit is visible to the version.
    """
    from rfw_w.scenes.contracts.tool import FailureCode
    from rfw_w.scenes.orchestrator.escalation import Atom
    from rfw_w.scenes.orchestrator.repair import RepairAction, apply as apply_repair

    plan = host.current_plan()
    floor = tuple(
        request.entity_id for request in plan.furniture
        if request.kind.value not in ("wall_hung", "ceiling")
    )
    if not floor:
        raise SystemExit("this plan has no floor piece to resample, so the script cannot make it move")
    outcome = apply_repair(
        plan, RepairAction(atom=Atom.RESAMPLE_POSE, targets=(floor[0],), source="gate"),
        wall_thickness_m=host.wall_thickness_m(),
        failure_code=FailureCode.GEOMETRY_OUT_OF_BOUNDS.value,
    )
    if not outcome.changed:
        raise SystemExit(f"the repair did not change the plan: {'; '.join(outcome.notes)}")
    return outcome.plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="office_pick_cup")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env.local")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import run_m1_acceptance as acceptance

    if args.env_file.is_file():
        acceptance.load_env_file(args.env_file)
    else:
        raise SystemExit(f"env file not found: {args.env_file}")

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already has files in it; every run needs a fresh directory")

    spec = acceptance.load_spec(args.experiment)
    host = acceptance.M1Host(
        spec=spec,
        out=out,
        args=argparse.Namespace(skip_isaac=True, vlm_model=None),
        requested=acceptance.ReadinessLevel(spec.requested_readiness),
        requested_backends=spec.requested_backends,
    )
    plan, _randomization = acceptance.randomize_plan(
        spec, acceptance.build_m1_plan(spec), variant_seed=spec.seed, enabled=False
    )
    host.randomization = _randomization
    host.generation_seed = spec.seed

    print(f"building the unmodified plan ...", flush=True)
    first = host.build(plan)
    print(f"building a repaired plan ...", flush=True)
    moved = _plan_a_change(host)
    second = host.build(moved)
    print(f"rebuilding the unmodified plan, as the restore path does ...", flush=True)
    third = host.build(plan)

    identical = first == third
    report = {
        "first": first,
        "repaired": second,
        "restored": third,
        "the_repaired_plan_differed": first != second,
        "the_restore_reproduced_its_own_version": identical,
        "build_diagnostics": host.build_diagnostics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not identical:
        print(
            "\nFAIL: rebuilding the plan the run started from produced a different version, so the "
            "scene is not a function of the plan. Whatever `build_diagnostics` names is an input "
            "outside the plan, and the loop's restore path cannot be relied on until it is inside.",
            file=sys.stderr,
        )
        return 1
    print("\nPASS: the scene version is a function of the plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
