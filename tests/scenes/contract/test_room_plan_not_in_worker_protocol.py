# SPDX-License-Identifier: Apache-2.0
"""`RoomPlan` must not reach the Isaac worker.

SPEC-001 R1: the worker protocol carries a `RoomShell` -- polygon, height, openings -- and nothing
about where that shell came from. The plan is the contract between core and room *generation*, a
stage that has already finished by the time a worker request exists.

The rule matters because the protocol is the wire format the worker and the orchestrator have to
agree on, and the worker is a separate process: a plan field leaking in would be a schema change
that only shows up as a KeyError on the far side.

This runs in a *subprocess* for the same reason `test_no_v2_import` does -- `sys.modules` is
process-global, so an in-process assertion would answer "has some earlier test imported it" rather
than "does the protocol need it".

The import check alone is not enough, though: a module can reach the plan through a
re-export without importing it by name. So the payload check builds a real request and asserts
none of the plan's own vocabulary appears in it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"

#: Words that only a plan uses. The shell contract shares none of them, which is the point --
#: `polygon_xy`, `height_m` and `openings` cross over; `plan_id`, `core_hash` and `wall_segment_xy`
#: do not.
PLAN_ONLY_KEYS = {
    "plan_id",
    "core",
    "core_hash",
    "provenance",
    "shell_options_digest",
    "wall_segment_xy",
    "ceiling_height_m",
    "up_axis",
    "source",
}


def _run(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}},
    )


def test_the_worker_protocol_does_not_import_the_room_plan() -> None:
    program = (
        "import sys\n"
        "import rfw_w.scenes.workers.isaac.protocol\n"
        "print(sorted(n for n in sys.modules if 'room_plan' in n))\n"
    )
    result = _run(program)
    assert result.returncode == 0, f"importing the protocol failed:\n{result.stderr}"
    assert result.stdout.strip() == "[]", (
        "the Isaac worker protocol imported rfw_w.scenes.contracts.room_plan; the worker takes a "
        "RoomShell, not a plan (SPEC-001 R1)"
    )


@pytest.mark.parametrize(
    "module",
    [
        "rfw_w.scenes.workers.isaac.protocol",
        "rfw_w.scenes.workers.isaac.geometry",
    ],
)
def test_a_worker_module_does_not_reach_the_plan_by_any_name(module: str) -> None:
    """A re-export would escape the import check above, so this asks what got pulled in."""
    program = (
        "import sys\n"
        f"import {module}\n"
        "leaked = [n for n in sys.modules if 'room_plan' in n]\n"
        "print(','.join(leaked))\n"
    )
    result = _run(program)
    assert result.returncode == 0, f"importing {module} failed:\n{result.stderr}"
    assert result.stdout.strip() == "", (
        f"importing {module} pulled {result.stdout.strip()!r} into sys.modules"
    )


def test_a_worker_request_payload_carries_no_plan_vocabulary() -> None:
    """The wire format itself: build a request and look at the keys that come out.

    This is the half the import check cannot see. `RoomGeometry` and `RoomPlan` both describe a
    room, so a field could be copied across without anything importing anything -- and the copy
    would be exactly the silent second representation this contract exists to prevent.
    """
    from rfw_w.scenes.workers.isaac.protocol import RoomGeometry, RoomOpening

    geometry = RoomGeometry(
        polygon_xy=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
        height_m=2.8,
        openings=(
            RoomOpening(opening_id="door_main", center_xy=(2.0, 3.0), width_m=0.9, height_m=2.05),
        ),
    )
    payload = json.dumps(geometry.to_dict())

    found = sorted(key for key in PLAN_ONLY_KEYS if f'"{key}"' in payload)
    assert found == [], (
        f"the worker's room geometry carries plan-only keys {found}; the worker reads a shell, and "
        "a plan field on the wire has no reader on the far side (SPEC-001 R1)"
    )


def test_the_room_plan_module_is_where_this_repository_says_it_is() -> None:
    """Guards the tests above: a rename would make them pass by checking nothing."""
    assert (SRC / "rfw_w" / "scenes" / "contracts" / "room_plan.py").is_file()

    program = (
        "import rfw_w.scenes.contracts.room_plan as m\n"
        "print(m.RoomPlan.__name__, m.SCHEMA_VERSION)\n"
    )
    result = _run(program)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "RoomPlan pvscene.room-plan/v1"
