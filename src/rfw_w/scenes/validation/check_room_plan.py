#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a room plan file: read it, hash it, and report every named gap in it.

    python -m rfw_w.scenes.validation.check_room_plan --plan tests/fixtures/office_room_plan.json

Prints the plan's identity and its content hash, then the named gaps. Exits non-zero when there
are any, so the command doubles as a check that a plan is self-consistent.

This is the SPEC-001 acceptance surface: it is the command a reader can run to see what the
contract actually enforces, and the negative case -- edit one ``width_m`` by 5 cm and leave the
segment alone -- produces ``PLAN_OPENING_WIDTH_DISAGREES_WITH_SEGMENT`` rather than a silent pass.

It deliberately does **not** compare the plan against built geometry. That is
``shell.matches_plan``'s job (SPEC-004), and conflating the two would make a plan fail for a
builder's choice of door width -- a quantity the plan was never able to control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from rfw_w.scenes.contracts.room_plan import (
    RoomPlan,
    RoomPlanError,
    check_plan,
    plan_to_shell,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True, help="a pvscene.room-plan/v1 JSON file")
    parser.add_argument(
        "--show-shell",
        action="store_true",
        help="also print the shell this plan asks for (the request path, not anything built)",
    )
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.plan.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"PLAN_NOT_SERIALISABLE(read): {error}")
        return 2

    try:
        plan = RoomPlan.from_dict(payload)
    except RoomPlanError as error:
        # The code is already the first token of the message; printing it as-is keeps one
        # spelling of every failure.
        print(error)
        return 1

    print(f"plan_id       {plan.plan_id}")
    print(f"source        {plan.source}")
    print(f"units/up_axis {plan.units} / {plan.up_axis}")
    print(f"rooms         {len(plan.rooms)}: " + ", ".join(r.room_id for r in plan.rooms))
    print(f"openings      {len(plan.openings)}: " + ", ".join(o.opening_id for o in plan.openings))
    print(f"core_hash     {plan.core.core_hash}")
    print(f"hash          {plan.content_hash}")

    gaps = check_plan(plan)
    if gaps:
        print()
        print(f"invariants: {len(gaps)} violated")
        for gap in gaps:
            print(f"  {gap}")
        return 1

    print("invariants: ok")

    if args.show_shell:
        shell = plan_to_shell(plan)
        print()
        print(f"shell.source  {shell.source}")
        print(f"shell.height  {shell.height_m}")
        for opening in shell.openings:
            x, y = opening.center_xy
            print(
                f"  {opening.opening_id:<12} {opening.kind.value:<7} "
                f"centre=({x:.4f}, {y:.4f}) w={opening.width_m:.4f} "
                f"h={opening.height_m:.4f} sill={opening.sill_z_m:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
