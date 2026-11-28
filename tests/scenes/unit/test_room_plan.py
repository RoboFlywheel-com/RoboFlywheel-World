# SPDX-License-Identifier: Apache-2.0
"""The room plan contract.

The negative tests carry the weight. A contract that only ever says "yes" is a set of field names,
so the question each test here asks is *what does this refuse, and does it say why*: an undeclared
unit, a plan edited after it was frozen, an opening whose two width representations disagree, a
core that moved when only its translation was allowed to.

The fixture is a real plan -- generated from `office_pick_cup`'s solved cell and fitted room, not
hand-written -- because a contract exercised only against made-up numbers does not tell you
whether it can carry the numbers that actually occur.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.room_plan import (
    WINDOW_HOST_SPAN_M,
    RoomPlan,
    RoomPlanError,
    buildable_window_span_m,
    check_plan,
    core_gaps,
    core_identity,
    floor_plan_payload,
    opening_host_span,
    plan_to_shell,
    segment_literal,
    shape_literal,
)
from rfw_w.scenes.contracts.state import OpeningKind

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "office_room_plan.json"


def _within(span, point, *, tolerance_m: float = 1e-6) -> bool:
    """Whether a point is inside a segment's stretch, endpoints included."""
    (x1, y1), (x2, y2) = span
    dx, dy = x2 - x1, y2 - y1
    t = ((point[0] - x1) * dx + (point[1] - y1) * dy) / (dx * dx + dy * dy)
    projection = (x1 + t * dx, y1 + t * dy)
    return -tolerance_m <= t <= 1 + tolerance_m and math.dist(projection, point) <= tolerance_m


def payload() -> dict:
    """A fresh copy of the fixture, so a test that edits one cannot affect another."""
    return json.loads(FIXTURE.read_text())


def plan() -> RoomPlan:
    return RoomPlan.from_dict(payload())


def refreeze(edited: dict) -> RoomPlan:
    """Read an edited plan as a *new* plan, rather than carrying the old plan's hash.

    Editing a frozen plan is what these tests are about, so the staleness check -- which exists
    precisely to catch that -- has to be stepped over deliberately. Dropping the ``hash`` key says
    "this is a different plan", which is the honest framing: a plan whose content changed *is* a
    new version, and ``test_an_edited_plan_is_refused`` covers the case where it is not relabelled.
    """
    edited.pop("hash", None)
    return RoomPlan.from_dict(edited)


class TestRoundTrip:
    def test_a_plan_survives_a_json_round_trip(self) -> None:
        original = plan()
        again = RoomPlan.from_dict(json.loads(json.dumps(original.to_dict())))
        assert again == original

    def test_the_hash_is_stable_across_reads(self) -> None:
        assert plan().content_hash == plan().content_hash

    def test_serialising_twice_is_byte_identical(self) -> None:
        assert json.dumps(plan().to_dict()) == json.dumps(plan().to_dict())

    def test_the_hash_covers_every_field_but_itself(self) -> None:
        """The digest is over ``payload()``; the ``hash`` key is added beside it, never inside."""
        base = plan()
        assert "hash" not in base.payload()
        assert set(base.to_dict()) == set(base.payload()) | {"hash"}

        # Both replacements have to be *valid* values, or the plan is refused for a different
        # reason and the hash claim goes untested.
        for key, value in (("plan_id", "other_id"), ("source", "external")):
            assert refreeze({**payload(), key: value}).content_hash != base.content_hash

        moved = payload()
        moved["core"]["translation_xy"] = [0.0, 0.0]
        assert refreeze(moved).content_hash != base.content_hash

    def test_an_edited_plan_is_refused(self) -> None:
        """A plan carries its own identity, so a silent edit is caught on read."""
        edited = payload()
        edited["rooms"][0]["ceiling_height_m"] = 3.5
        with pytest.raises(RoomPlanError, match=r"PLAN_NOT_SERIALISABLE\(hash\)"):
            RoomPlan.from_dict(edited)


class TestDeclarations:
    """R1: units and up-axis are declared, never inferred from this repository's conventions."""

    def test_a_plan_without_units_is_refused(self) -> None:
        missing = payload()
        del missing["units"]
        with pytest.raises(RoomPlanError, match=r"PLAN_UNITS_UNDECLARED\(units\)"):
            RoomPlan.from_dict(missing)

    def test_a_plan_without_an_up_axis_is_refused(self) -> None:
        missing = payload()
        del missing["up_axis"]
        with pytest.raises(RoomPlanError, match=r"PLAN_UNITS_UNDECLARED\(up_axis\)"):
            RoomPlan.from_dict(missing)

    def test_a_plan_in_feet_is_refused(self) -> None:
        imperial = payload()
        imperial["units"] = "ft"
        with pytest.raises(RoomPlanError, match=r"PLAN_UNITS_UNDECLARED\(units\)"):
            RoomPlan.from_dict(imperial)

    def test_a_y_up_plan_is_refused(self) -> None:
        y_up = payload()
        y_up["up_axis"] = "Y"
        with pytest.raises(RoomPlanError, match=r"PLAN_UNITS_UNDECLARED\(up_axis\)"):
            RoomPlan.from_dict(y_up)

    def test_an_unknown_source_is_refused(self) -> None:
        invented = payload()
        invented["source"] = "vibes"
        with pytest.raises(RoomPlanError, match=r"PLAN_NOT_SERIALISABLE\(source\)"):
            RoomPlan.from_dict(invented)

    def test_a_window_kind_of_something_else_is_refused(self) -> None:
        """`kind` is Infinigen's door/window vocabulary, not free text."""
        strange = payload()
        strange["openings"][0]["kind"] = "skylight"
        with pytest.raises(RoomPlanError, match=r"PLAN_NOT_SERIALISABLE\(door_main\)"):
            RoomPlan.from_dict(strange)

    def test_an_opening_in_a_room_that_does_not_exist_is_refused(self) -> None:
        elsewhere = payload()
        elsewhere["openings"][0]["room_id"] = "room_7"
        with pytest.raises(RoomPlanError, match="is not in this plan"):
            RoomPlan.from_dict(elsewhere)

    def test_duplicate_opening_ids_are_refused(self) -> None:
        twice = payload()
        twice["openings"].append(copy.deepcopy(twice["openings"][0]))
        with pytest.raises(RoomPlanError, match="duplicate opening ids"):
            RoomPlan.from_dict(twice)


class TestOpeningWidthInvariant:
    """R7: the two representations of one opening's width are not allowed to disagree."""

    def test_the_fixture_is_self_consistent(self) -> None:
        assert check_plan(plan()) == ()

    def test_every_opening_width_equals_its_segment_length(self) -> None:
        for opening in plan().openings:
            assert opening.width_m == pytest.approx(opening.segment_length_m, abs=1e-9)

    def test_a_five_centimetre_change_is_reported_with_the_difference(self) -> None:
        widened = payload()
        widened["openings"][0]["width_m"] += 0.05
        gaps = check_plan(refreeze(widened))
        assert len(gaps) == 1
        assert gaps[0].startswith("PLAN_OPENING_WIDTH_DISAGREES_WITH_SEGMENT(door_main, 0.950000,")
        # The gap carries the segment length it disagreed with, so a reader can see by how much.
        assert "0.900000" in gaps[0]

    def test_moving_the_segment_instead_fails_the_same_way(self) -> None:
        """Either representation moving alone is the same defect."""
        stretched = payload()
        (x1, y1), (x2, y2) = stretched["openings"][0]["wall_segment_xy"]
        stretched["openings"][0]["wall_segment_xy"] = [[x1 + 0.05, y1], [x2, y2]]
        gaps = check_plan(refreeze(stretched))
        assert len(gaps) == 1
        assert "PLAN_OPENING_WIDTH_DISAGREES_WITH_SEGMENT" in gaps[0]

    def test_the_tolerance_covers_the_worst_case_a_six_decimal_write_can_produce(self) -> None:
        """The tolerance is derived from the 6-decimal write, not picked.

        Two endpoints rounded to 6 decimals move a *diagonal* segment by up to 1.42e-6, which a
        1e-6 tolerance would report as the plan disagreeing with itself. Asserted as arithmetic
        rather than on a plan, because the worst case needs a diagonal wall and the fixture's room
        has none -- which is the next test's subject.
        """
        from rfw_w.scenes.contracts.room_plan import SEGMENT_LENGTH_TOLERANCE_M

        assert SEGMENT_LENGTH_TOLERANCE_M >= 2 * (2**0.5) * 5e-7

    def test_a_sub_tolerance_move_is_not_a_disagreement(self) -> None:
        """JSON round trips move floats in the last places; that is not a disagreement."""
        nudged = payload()
        (x1, y1), (x2, y2) = nudged["openings"][0]["wall_segment_xy"]
        nudged["openings"][0]["wall_segment_xy"] = [[x1 - 5e-7, y1], [x2 - 5e-7, y2]]
        nudged["openings"][0]["width_m"] -= 1e-6
        assert check_plan(refreeze(nudged)) == ()

    def test_a_segment_on_no_wall_is_reported(self) -> None:
        """R7's second claim, and the one the serialiser depends on.

        `floor_plan_payload` clamps a window's host span to the wall behind it, which it can only
        do if the segment is on one. Before this check the only plans in existence came from the
        drawer, which derives the segment from a wall it already knows, so the claim was implicit;
        `RoomPlan.source` admits ``external`` and ``bank``, where it is not.
        """
        adrift = payload()
        adrift["openings"][0]["wall_segment_xy"] = [[0.0000005, 0.0000005], [0.9, 0.9]]
        adrift["openings"][0]["width_m"] = 1.2727922061357855
        gaps = check_plan(refreeze(adrift))
        assert len(gaps) == 1
        assert gaps[0].startswith("PLAN_OPENING_NOT_ON_A_WALL(door_main,")

    def test_a_sub_micron_disagreement_is_tolerated(self) -> None:
        """JSON round trips move floats in the last places; that is not a disagreement."""
        nudged = payload()
        nudged["openings"][0]["width_m"] += 1e-9
        assert check_plan(refreeze(nudged)) == ()


class TestCoreFrozen:
    """R5: the room solve may move the core, never reshape it."""

    def test_only_the_translation_may_differ(self) -> None:
        before = plan()
        moved = payload()
        moved["core"]["translation_xy"] = [
            before.core.translation_xy[0] + 1.0,
            before.core.translation_xy[1] + 1.0,
        ]
        assert core_gaps(before, refreeze(moved)) == ()

    def test_a_yaw_change_is_allowed(self) -> None:
        before = plan()
        turned = payload()
        turned["core"]["rotation_yaw_rad"] = 0.5
        assert core_gaps(before, refreeze(turned)) == ()

    def test_a_changed_core_field_is_named(self) -> None:
        before = plan()
        altered = payload()
        altered["core"]["support_top_z_m"] = 0.9
        gaps = core_gaps(before, refreeze(altered))
        assert len(gaps) == 1
        assert gaps[0].startswith("PLAN_CORE_MUTATED(core.support_top_z_m, 0.75, 0.9)")

    def test_core_identity_ignores_the_translation(self) -> None:
        """The pinned value tests freeze: a moved core is the same core."""
        before = plan()
        moved = payload()
        moved["core"]["translation_xy"] = [0.0, 0.0]
        assert core_identity(before) == core_identity(refreeze(moved))

    def test_core_identity_changes_with_a_core_field(self) -> None:
        altered = payload()
        altered["core"]["target_offset_xy"] = [0.0, 0.0]
        assert core_identity(plan()) != core_identity(refreeze(altered))

    def test_the_plan_carries_the_cells_own_hash(self) -> None:
        """The core block is evidence of the frozen cell, so it names it."""
        from rfw_w.scenes.contracts.hashing import is_digest

        assert is_digest(plan().core.core_hash)

    def test_the_protected_region_moves_by_exactly_the_translation(self) -> None:
        core = plan().core
        dx, dy = core.translation_xy
        for (x, y), (rx, ry) in zip(core.protected_polygon_xy, core.protected_in_room_xy):
            assert (rx, ry) == pytest.approx((x + dx, y + dy))


class TestRequestPath:
    """The only conversion plan -> shell, and it produces a request, not a report."""

    def test_the_segment_midpoint_becomes_the_opening_centre(self) -> None:
        shell = plan_to_shell(plan())
        by_id = {opening.opening_id: opening for opening in shell.openings}
        for opening in plan().openings:
            (x1, y1), (x2, y2) = opening.wall_segment_xy
            assert by_id[opening.opening_id].center_xy == pytest.approx(
                ((x1 + x2) / 2, (y1 + y2) / 2)
            )

    def test_the_room_outline_and_height_come_across_unchanged(self) -> None:
        source = plan()
        shell = plan_to_shell(source)
        assert shell.polygon_xy == source.rooms[0].polygon_xy
        assert shell.height_m == source.rooms[0].ceiling_height_m
        assert shell.holes_xy == ()

    def test_the_source_says_the_shell_came_from_a_plan(self) -> None:
        """ADR-001 asks a reader to be able to tell where a shell came from."""
        assert plan_to_shell(plan()).source == "plan:core"

    def test_opening_kinds_survive_the_conversion(self) -> None:
        kinds = {opening.kind for opening in plan_to_shell(plan()).openings}
        assert kinds == {OpeningKind.DOOR, OpeningKind.WINDOW}

    def test_the_conversion_reproduces_the_openings_it_was_asked_for(self) -> None:
        """The width is passed through as-is. Whether it is *built* is a different document."""
        source = plan()
        by_id = {opening.opening_id: opening for opening in plan_to_shell(source).openings}
        for opening in source.openings:
            converted = by_id[opening.opening_id]
            assert converted.width_m == opening.width_m
            assert converted.height_m == opening.height_m
            assert converted.sill_z_m == opening.sill_z_m

    def test_a_second_room_is_refused_rather_than_ignored(self) -> None:
        """Single-room is a constraint, so a second room must not be silently dropped."""
        two = payload()
        second = copy.deepcopy(two["rooms"][0])
        second["room_id"] = "room_1"
        two["rooms"].append(second)
        two["hash"] = RoomPlan.from_dict({k: v for k, v in two.items() if k != "hash"}).content_hash
        with pytest.raises(RoomPlanError, match="single-room"):
            plan_to_shell(RoomPlan.from_dict(two))


class TestSerialisation:
    """Shape strings are ours to build and nobody else's to interpolate into."""

    def test_a_polygon_literal_is_a_shapely_expression(self) -> None:
        literal = shape_literal([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
        assert literal == "shapely.Polygon([(0.0,0.0),(1.0,0.0),(1.0,1.0)])"

    def test_a_segment_literal_is_a_linestring_expression(self) -> None:
        literal = segment_literal([(0.0, 0.5), (1.5, 0.5)])
        assert literal == "shapely.LineString([(0.0,0.5),(1.5,0.5)])"

    def test_a_two_vertex_polygon_is_refused(self) -> None:
        with pytest.raises(RoomPlanError, match=r"PLAN_NOT_SERIALISABLE\(polygon\)"):
            shape_literal([(0.0, 0.0), (1.0, 1.0)])

    def test_a_non_finite_coordinate_is_refused(self) -> None:
        with pytest.raises(RoomPlanError, match="non-finite"):
            shape_literal([(0.0, 0.0), (1.0, 0.0), (float("nan"), 1.0)])

    def test_a_segment_needs_exactly_two_points(self) -> None:
        with pytest.raises(RoomPlanError, match=r"PLAN_NOT_SERIALISABLE\(segment\)"):
            segment_literal([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    def test_the_floor_plan_uses_infinigens_own_keys(self) -> None:
        """`predefined.py` parses the room type and storey out of the key, so it must be shaped."""
        payload_ = floor_plan_payload(plan())
        assert set(payload_) == {"rooms", "doors", "interiors", "opens", "windows"}
        assert list(payload_["rooms"]) == ["office_0/0"]
        assert payload_["rooms"]["office_0/0"]["shape"].startswith("shapely.Polygon([")
        assert set(payload_["doors"]) == {"door_main"}
        assert set(payload_["windows"]) == {"window_01", "window_02", "window_03"}
        # Single-room means there is no interior wall for either of these to sit in.
        assert payload_["opens"] == {} and payload_["interiors"] == {}


class TestTheHostSpanAWindowIsGiven:
    """`floor_plan_payload` hands a window a wider stretch than the plan asked for, and why.

    Infinigen decides whether a window exists from the length of the line it is given: a piece at or
    below its `door_width` yields no cutter and no hole, silently (SPEC-004 §8 has the measurement --
    office's 0.926 m piece against a 0.9296 m `door_width`, 4 mm short and never built). The plan's
    own width is the *request*, and SPEC-001 already says the widths are not a promise about built
    geometry; this is where that freedom is spent.
    """

    def _window(self, plan_: RoomPlan):
        return next(o for o in plan_.openings if o.opening_id == "window_02")

    def test_the_short_window_is_the_one_that_was_not_built(self) -> None:
        """The fixture is a real office plan, so the premise is a measurement and not a setup."""
        window = self._window(plan())
        assert window.segment_length_m == pytest.approx(0.926)
        assert window.segment_length_m < buildable_window_span_m(0.1)

    def test_the_span_is_long_enough_that_no_draw_can_drop_it(self) -> None:
        """The whole point: whatever `door_width` the seed lands on, the piece is longer than it."""
        span = opening_host_span(plan(), self._window(plan()))
        assert math.dist(*span) == pytest.approx(WINDOW_HOST_SPAN_M)
        assert math.dist(*span) > buildable_window_span_m(0.1)

    def test_the_constant_clears_the_ceiling_for_any_wall_thickness_we_might_ask_for(self) -> None:
        """The margin is for the thickness, which is the one number the plan cannot supply.

        `buildable_window_span_m` falls as the wall gets thicker, so the claim `WINDOW_HOST_SPAN_M`
        has to survive is made about thicknesses, not about the 0.1 m this repository happens to
        use. The crossover is exact -- `1.10 > (1.4 - t) * 0.8` means `t > 0.025` -- so the range
        covered is every wall from 2.5 cm up, which includes Infinigen's own default draw
        (`wall_thickness=("uniform", 0.2, 0.3)`, which this repository overrides) and every
        thickness anybody would call a wall.
        """
        assert WINDOW_HOST_SPAN_M > buildable_window_span_m(0.1)
        # And it stays a *margin* rather than a coincidence: 6 cm at the thickness in use.
        assert WINDOW_HOST_SPAN_M - buildable_window_span_m(0.1) == pytest.approx(0.06)
        for thickness in (0.05, 0.1, 0.15, 0.2, 0.3):
            assert WINDOW_HOST_SPAN_M > buildable_window_span_m(thickness), thickness
        # Below the crossover it does not, and that is the honest boundary rather than a claim that
        # the constant is safe everywhere: a 2.5 cm wall is thinner than a window frame.
        assert WINDOW_HOST_SPAN_M == pytest.approx(buildable_window_span_m(0.025))

    def test_the_plan_is_not_touched_by_it(self) -> None:
        """The segment, the width and the hash are the drawer's, and stay that way.

        Widening the plan's own segment was the alternative, and it was refused: it would move every
        plan hash and every `scene_version`, and make the document claim a width nobody drew.
        """
        before = payload()
        _ = floor_plan_payload(plan())
        assert plan().to_dict() == before

    def test_the_span_contains_the_segment_it_came_from(self) -> None:
        test_plan = plan()
        for opening in test_plan.openings:
            span = opening_host_span(test_plan, opening)
            start, end = span
            (sx, sy), (ex, ey) = opening.wall_segment_xy
            for point in (opening.centre_m, (sx, sy), (ex, ey)):
                assert _within(span, point), (opening.opening_id, point, span)
            assert math.dist(start, end) >= opening.segment_length_m - 1e-9

    def test_a_door_gets_its_own_segment(self) -> None:
        """The door cutter has no length test, so there is nothing about a door to widen."""
        test_plan = plan()
        door = next(o for o in test_plan.openings if o.kind == "door")
        assert opening_host_span(test_plan, door) == door.wall_segment_xy

    def test_the_payload_carries_the_span_for_windows_and_the_segment_for_doors(self) -> None:
        """What Infinigen is actually handed, read out of the shape strings rather than assumed."""
        test_plan = plan()
        payload_ = floor_plan_payload(test_plan)
        for opening in test_plan.openings:
            table = payload_["doors" if opening.kind == "door" else "windows"]
            literal = table[opening.opening_id]["shape"]
            start, end = opening_host_span(test_plan, opening)
            assert literal == segment_literal([start, end])
        # The room is untouched: only the openings' stretches are ever widened.
        assert payload_["rooms"]["office_0/0"]["shape"] == shape_literal(
            test_plan.rooms[0].polygon_xy
        )

    def test_a_wall_too_short_to_hold_the_span_gets_the_whole_wall(self) -> None:
        """The clamp. A stretch that does not fit slides back inside the wall instead of running
        through the corner -- and the provider that cannot build in what is left says so by name
        (`SHELL_WINDOW_SPAN_UNBUILDABLE`, covered in `test_shell_gates.py`)."""
        short = payload()
        short["rooms"][0]["polygon_xy"] = [[0.0, 0.0], [4.0, 0.0], [4.0, 0.8], [0.0, 0.8]]
        short["openings"] = [
            {**short["openings"][0], "wall_segment_xy": [[1.55, 0.0], [2.45, 0.0]]},
            {**short["openings"][1], "wall_segment_xy": [[0.0, 0.15], [0.0, 0.65]],
             "width_m": 0.5},
        ]
        test_plan = refreeze(short)
        window = next(o for o in test_plan.openings if o.opening_id == "window_01")
        start, end = opening_host_span(test_plan, window)
        assert start == (0.0, 0.0) and end == (0.0, 0.8), (start, end)
        assert check_plan(test_plan) == ()

    def test_the_span_is_a_function_of_the_plan_alone(self) -> None:
        """Two readers of one plan get one span, so a report and a gate cannot disagree."""
        first = opening_host_span(plan(), self._window(plan()))
        second = opening_host_span(RoomPlan.from_dict(json.loads(json.dumps(plan().to_dict()))),
                                   self._window(plan()))
        assert first == second

    def test_an_opening_in_a_room_the_plan_does_not_have_is_refused(self) -> None:
        """`from_dict` refuses it first; this is the second line, because the serialiser is where a
        missing wall becomes a wrong span rather than an error."""
        elsewhere = payload()
        elsewhere["openings"][1]["room_id"] = "room_7"
        with pytest.raises(RoomPlanError, match="is not in this plan"):
            RoomPlan.from_dict(elsewhere)

    def test_a_missing_coordinate_pair_is_refused_rather_than_defaulted(self) -> None:
        broken = payload()
        broken["core"]["translation_xy"] = [1.0]
        with pytest.raises(RoomPlanError, match="core.translation_xy"):
            RoomPlan.from_dict(broken)
