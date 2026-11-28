# SPDX-License-Identifier: Apache-2.0
"""Reading a shell measurement into evidence.

`ShellBuild` is the seam where "what got built" becomes something a gate can compare. The negative
tests carry the weight for the usual reason, and here they have a second one: the report is
produced by a Blender script in another process, so this parser is the only place that can notice a
report which is well-formed JSON and still not say what it claims to.

The fixture is the real 2026-09-21 office measurement, trimmed to the fields this parser reads --
the measured widths are the ones the ruler produced (0.9296 for the door where the plan asked 0.9),
so a change to these tests' numbers is a change in what the tool measures and not a restatement of
what it was told to expect.
"""

from __future__ import annotations

import copy
import json

import pytest

from rfw_w.scenes.contracts.room_plan import RoomPlan
from rfw_w.scenes.contracts.shell_build import (
    HEIGHT_TOLERANCE_M,
    WALL_LINE_TOLERANCE_M,
    ShellBuildError,
    openings_from_report,
    shell_from_report,
)

PLAN = {
    "schema_version": "pvscene.room-plan/v1",
    "plan_id": "office_pick_cup:11:probe",
    "source": "core",
    "units": "m",
    "up_axis": "Z",
    "rooms": [{
        "room_id": "room_0", "kind": "office",
        "polygon_xy": [[0.0, 0.746], [0.864, 0.746], [0.864, 0.0],
                       [4.4524, 0.0], [4.4524, 3.5598], [0.0, 3.5598]],
        "floor_z_m": 0.0, "ceiling_height_m": 2.9697,
        "floor_material": "oak", "wall_material": "off_white",
    }],
    "openings": [
        {"opening_id": "door_main", "kind": "door", "room_id": "room_0",
         "wall_segment_xy": [[1.804213, 3.5598], [2.704213, 3.5598]],
         "width_m": 0.9, "height_m": 2.05, "sill_z_m": 0.0},
        {"opening_id": "window_01", "kind": "window", "room_id": "room_0",
         "wall_segment_xy": [[1.016366, 0.0], [2.407366, 0.0]],
         "width_m": 1.391, "height_m": 1.2, "sill_z_m": 0.9},
        {"opening_id": "window_02", "kind": "window", "room_id": "room_0",
         "wall_segment_xy": [[4.4524, 0.59873], [4.4524, 1.80473]],
         "width_m": 1.206, "height_m": 1.2, "sill_z_m": 0.9},
        {"opening_id": "window_03", "kind": "window", "room_id": "room_0",
         "wall_segment_xy": [[0.0, 1.889299], [0.0, 0.909299]],
         "width_m": 0.98, "height_m": 1.2, "sill_z_m": 0.9},
    ],
    "core": {
        "core_hash": "979875297fcd621734151a7b8daf9507c735d5c8bb7b148e2a85940fa6b9c4a1",
        "translation_xy": [3.37525, 2.5173], "rotation_yaw_rad": 0.0,
        "protected_polygon_xy": [[-0.96371, -0.68871], [-0.9225, -0.94215], [0.08, -1.05]],
        "support_top_z_m": 0.75, "target_offset_xy": [-0.22, 0.055],
    },
    "provenance": {"spec_hash": "4b219d11f8653c384669341d3608dc1454ee5525572424f9a869c4beebba0c50",
                   "seed": 11},
}

#: The measured holes, as `scripts/ops/measure_built_shell.py` reported them. `offset_low`/`high`
#: are in the wall's own frame and `centre_xy` is in the room's -- the tool emits both so a reader
#: never has to redo the conversion.
REPORT = {
    "floor_z_m": 0.05,
    "measurements": {
        "room.height_m": 2.9697,
        "room.wall_line_offset_m": 0.05,
        "room.wall_line_offset_spread_m": 0.0,
        "wall.depth_m": 0.05,
        "openings.count": 4,
        "openings.plan_count": 4,
    },
    "not_measured": [],
    "walls": [
        {"plan_edge": 0, "point": [0.914, 0.796], "end": [0.05, 0.796],
         "length_m": 0.864, "z_min": 0.05, "z_max": 2.9197, "openings": []},
        {"plan_edge": 1, "point": [0.914, 0.05], "end": [0.914, 0.796],
         "length_m": 0.746, "z_min": 0.05, "z_max": 2.9197, "openings": []},
        {"plan_edge": 2, "point": [4.4024, 0.05], "end": [0.914, 0.05],
         "length_m": 3.4884, "z_min": 0.05, "z_max": 2.9197,
         "openings": [{"offset_low": -2.1678, "offset_high": -1.2497, "centre_xy": [1.7087, 0.05],
                       "z_min": 0.9669, "z_max": 2.0663}]},
        {"plan_edge": 3, "point": [4.4024, 0.05], "end": [4.4024, 3.5098],
         "length_m": 3.4598, "z_min": 0.05, "z_max": 2.9197,
         "openings": [{"offset_low": 0.7983, "offset_high": 1.7140, "centre_xy": [4.4024, 1.257],
                       "z_min": 0.9669, "z_max": 2.0663}]},
        {"plan_edge": 4, "point": [4.4024, 3.5098], "end": [0.05, 3.5098],
         "length_m": 4.3524, "z_min": 0.05, "z_max": 2.9197,
         "openings": [{"offset_low": -2.8217, "offset_high": -1.8921, "centre_xy": [2.3569, 3.5098],
                       "z_min": 0.05, "z_max": 2.0721}]},
        {"plan_edge": 5, "point": [0.05, 0.796], "end": [0.05, 3.5098],
         "length_m": 2.7138, "z_min": 0.05, "z_max": 2.9197,
         "openings": [{"offset_low": 1.0970, "offset_high": 1.7425, "centre_xy": [0.05, 1.4198],
                       "z_min": 0.9669, "z_max": 2.0663}]},
    ],
}


def plan() -> RoomPlan:
    return RoomPlan.from_dict(PLAN)


def report() -> dict:
    return copy.deepcopy(REPORT)


def build(edited: dict | None = None):
    return shell_from_report(plan(), edited or report(), provider="infinigen")


class TestTheBuildIsReadBack:
    def test_the_provider_is_named_on_the_shell(self) -> None:
        """ADR-001 asks a reader to be able to tell where a shell came from."""
        assert build().shell.source == "measured:infinigen"
        assert build().provider == "infinigen"

    def test_the_outline_comes_from_the_plan(self) -> None:
        assert build().shell.polygon_xy == plan().rooms[0].polygon_xy
        assert build().shell.height_m == plan().rooms[0].ceiling_height_m

    def test_the_height_is_the_measured_one_when_the_report_carries_it(self) -> None:
        edited = report()
        edited["measurements"]["room.height_m"] = 2.4
        assert build(edited).shell.height_m == 2.4

    def test_every_planned_opening_is_identified(self) -> None:
        assert [opening.opening_id for opening in build().shell.openings] == [
            opening.opening_id for opening in plan().openings
        ]

    def test_the_openings_carry_what_was_measured_not_what_was_asked_for(self) -> None:
        """The whole point of measuring: 0.9296 is the door, 0.9 is the request."""
        measured = {opening.opening_id: opening for opening in build().shell.openings}
        assert measured["door_main"].width_m == pytest.approx(0.9296, abs=1e-3)
        assert measured["door_main"].height_m == pytest.approx(2.0221, abs=1e-3)
        assert measured["door_main"].sill_z_m == pytest.approx(0.0, abs=1e-3)
        assert measured["window_02"].width_m == pytest.approx(0.9156, abs=1e-3)

    def test_a_sill_is_measured_from_the_interior_floor(self) -> None:
        """Not from z=0: Infinigen puts the floor surface half a slab above the plan's datum."""
        measured = {opening.opening_id: opening for opening in build().shell.openings}
        assert measured["window_01"].sill_z_m == pytest.approx(0.9169, abs=1e-3)
        assert measured["door_main"].sill_z_m == pytest.approx(0.0, abs=1e-3)

    def test_a_door_and_a_window_keep_their_kinds(self) -> None:
        kinds = {opening.opening_id: opening.kind.value for opening in build().shell.openings}
        assert kinds["door_main"] == "door"
        assert kinds["window_01"] == "window"


class TestWhatWasNotMeasured:
    def test_an_absent_measurement_is_refused_by_name(self) -> None:
        """Not ``None`` and not zero: a gate reading a missing number as zero would pass anything."""
        with pytest.raises(ShellBuildError, match=r"SHELL_MEASUREMENT_ABSENT\(room.floor_area_m2\)"):
            build().measured("room.floor_area_m2")

    def test_a_report_missing_a_compared_quantity_says_so(self) -> None:
        edited = report()
        del edited["measurements"]["room.wall_line_offset_spread_m"]
        assert any("room.wall_line_offset_spread_m" in entry for entry in build(edited).not_measured)

    def test_an_absent_measurement_must_carry_a_reason(self) -> None:
        from rfw_w.scenes.contracts.shell_build import ShellBuild

        from rfw_w.scenes.contracts.state import RoomShell

        with pytest.raises(ShellBuildError, match=r"SHELL_BUILD_FAILED\(not_measured\)"):
            ShellBuild(shell=RoomShell(polygon_xy=((0, 0), (1, 0), (1, 1))),
                       provider="builtin", not_measured=("",))

    def test_a_build_must_name_its_provider(self) -> None:
        from rfw_w.scenes.contracts.shell_build import ShellBuild

        from rfw_w.scenes.contracts.state import RoomShell

        with pytest.raises(ShellBuildError, match=r"SHELL_BUILD_FAILED\(provider\)"):
            ShellBuild(shell=RoomShell(polygon_xy=((0, 0), (1, 0), (1, 1))), provider="")


class TestOpeningsThatDoNotMatch:
    def test_a_plan_opening_with_no_hole_on_its_wall_is_named(self) -> None:
        edited = report()
        edited["walls"][4]["openings"] = []          # the north wall's door is gone
        with pytest.raises(ShellBuildError, match=r"SHELL_OPENING_MISSING\(door_main"):
            build(edited)

    def test_a_hole_no_plan_opening_sits_on_is_named(self) -> None:
        edited = report()
        edited["walls"][2]["openings"].append(
            {"offset_low": -3.0, "offset_high": -2.5, "centre_xy": [2.75, 0.05],
             "z_min": 1.0, "z_max": 2.0}
        )
        with pytest.raises(ShellBuildError, match="SHELL_OPENING_UNEXPECTED"):
            build(edited)

    def test_a_hole_moved_to_another_wall_is_a_missing_opening(self) -> None:
        """The failure a whole-count comparison would have missed entirely."""
        edited = report()
        door = edited["walls"][4]["openings"].pop()
        edited["walls"][3]["openings"].append(
            {"offset_low": 2.0, "offset_high": 2.9, "centre_xy": [4.4024, 2.45],
             "z_min": door["z_min"], "z_max": door["z_max"]}
        )
        with pytest.raises(ShellBuildError, match="SHELL_OPENING_MISSING"):
            build(edited)

    def test_openings_without_a_floor_level_still_parse(self) -> None:
        edited = report()
        del edited["floor_z_m"]
        measured = {opening.opening_id: opening for opening in
                    openings_from_report(plan(), edited)}
        assert measured["window_01"].sill_z_m == pytest.approx(0.9669, abs=1e-3)


class TestTheCalibration:
    def test_the_tolerances_are_the_calibrated_ones(self) -> None:
        """Pinned so a loosening has to be deliberate and reviewable."""
        assert WALL_LINE_TOLERANCE_M == 1e-3
        assert HEIGHT_TOLERANCE_M == 1e-3

    def test_the_office_build_is_inside_the_calibration(self) -> None:
        """The numbers the tolerances were derived from, read back from the report."""
        build_ = build()
        assert build_.measured("room.wall_line_offset_spread_m") <= WALL_LINE_TOLERANCE_M
        assert abs(build_.measured("room.height_m") - PLAN["rooms"][0]["ceiling_height_m"]) <= (
            HEIGHT_TOLERANCE_M
        )

    def test_the_build_survives_a_json_round_trip(self) -> None:
        original = build()
        again = json.loads(json.dumps(original.to_dict()))
        assert again["measurements"] == original.to_dict()["measurements"]
        assert again["shell"]["polygon_xy"] == [list(v) for v in original.shell.polygon_xy]
