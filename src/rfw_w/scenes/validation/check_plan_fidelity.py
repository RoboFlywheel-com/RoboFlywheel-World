#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the plan describe the room that was drawn?

    python -m rfw_w.scenes.validation.check_plan_fidelity --output-dir outputs/_probe/room_probe

Reads `shell.json` and `plan.json` from one run of `python -m rfw_w.scenes.commands.room --fit-cell`,
converts the plan back to a shell through the request path, and compares the two field by field.

This is SPEC-002's acceptance surface. The claim it checks is narrow and worth stating exactly:
the plan is a **faithful restatement** of the fitted room, so that a provider consuming the plan
builds the room the fit chose. It does not check that anything was built, and it does not check
that the room is any good -- those are SPEC-003 and SPEC-004.

`source` is deliberately excluded from the comparison. The fitted shell says `procedural_shell`
because that is what drew it; a shell derived from a plan says `plan:core`. Two names for two
different provenances is the field working, not a disagreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from rfw_w.scenes.contracts.room_plan import RoomPlan, plan_to_shell
from rfw_w.scenes.contracts.state import Opening, RoomShell

#: How close two coordinates have to be. The plan writes 6 decimals and `_materialise` rounds an
#: opening centre to the same 6, so the two sides are equal up to that rounding -- not exactly
#: equal, because the plan's segment is rebuilt from an offset rather than copied.
TOLERANCE_M = 1e-6


def _opening_key(opening: Opening) -> tuple:
    return (
        opening.opening_id,
        opening.kind.value,
        opening.center_xy,
        opening.width_m,
        opening.height_m,
        opening.sill_z_m,
    )


def compare(drawn: RoomShell, requested: RoomShell) -> list[str]:
    """Named differences between the shell that was drawn and the one the plan asks for."""
    gaps: list[str] = []

    if drawn.polygon_xy != requested.polygon_xy:
        gaps.append(
            f"PLAN_POLYGON_DIFFERS(drawn={drawn.polygon_xy}, requested={requested.polygon_xy})"
        )
    for name, left, right in (
        ("height_m", drawn.height_m, requested.height_m),
        ("floor_z_m", drawn.floor_z_m, requested.floor_z_m),
    ):
        if abs(left - right) > TOLERANCE_M:
            gaps.append(f"PLAN_{name.upper()}_DIFFERS(drawn={left}, requested={right})")

    by_id = {opening.opening_id: opening for opening in requested.openings}
    for opening in drawn.openings:
        counterpart = by_id.pop(opening.opening_id, None)
        if counterpart is None:
            gaps.append(f"PLAN_OPENING_MISSING({opening.opening_id})")
            continue
        if counterpart.kind != opening.kind:
            gaps.append(
                f"PLAN_OPENING_KIND_DIFFERS({opening.opening_id}, {opening.kind.value}, "
                f"{counterpart.kind.value})"
            )
        for field in ("center_xy",):
            drawn_value = getattr(opening, field)
            other = getattr(counterpart, field)
            if any(abs(a - b) > TOLERANCE_M for a, b in zip(drawn_value, other)):
                gaps.append(
                    f"PLAN_OPENING_{field.upper()}_DIFFERS({opening.opening_id}, {drawn_value}, {other})"
                )
        for field in ("width_m", "height_m", "sill_z_m"):
            drawn_value = getattr(opening, field)
            other = getattr(counterpart, field)
            if abs(drawn_value - other) > TOLERANCE_M:
                gaps.append(
                    f"PLAN_OPENING_{field.upper()}_DIFFERS({opening.opening_id}, {drawn_value}, {other})"
                )
    for extra in by_id:
        gaps.append(f"PLAN_OPENING_UNEXPECTED({extra})")
    return gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="a directory written by run_room_only.py --fit-cell (needs shell.json and plan.json)",
    )
    args = parser.parse_args(argv)

    try:
        shell_payload = json.loads((args.output_dir / "shell.json").read_text())
        plan_payload = json.loads((args.output_dir / "plan.json").read_text())
    except OSError as error:
        print(f"PLAN_FIDELITY_CANNOT_READ: {error}")
        return 2

    drawn = RoomShell(**{
        "polygon_xy": tuple(tuple(vertex) for vertex in shell_payload["shell"]["polygon_xy"]),
        "holes_xy": (),
        "height_m": shell_payload["shell"]["height_m"],
        "floor_z_m": shell_payload["shell"]["floor_z_m"],
        "openings": tuple(
            Opening(
                opening_id=item["opening_id"],
                kind=item["kind"],
                center_xy=tuple(item["center_xy"]),
                width_m=item["width_m"],
                height_m=item["height_m"],
                sill_z_m=item["sill_z_m"],
            )
            for item in shell_payload["shell"]["openings"]
        ),
        "source": shell_payload["shell"]["source"],
    })
    plan = RoomPlan.from_dict(plan_payload)
    requested = plan_to_shell(plan)

    print(f"drawn shell   source={drawn.source!r} openings={len(drawn.openings)}")
    print(f"plan          plan_id={plan.plan_id} hash={plan.content_hash}")
    print(f"plan->shell   source={requested.source!r} openings={len(requested.openings)}")

    gaps = compare(drawn, requested)
    if gaps:
        print()
        print(f"fidelity: {len(gaps)} difference(s)")
        for gap in gaps:
            print(f"  {gap}")
        return 1
    print("fidelity: ok -- the plan restates the drawn room")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
