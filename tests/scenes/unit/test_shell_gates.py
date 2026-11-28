# SPDX-License-Identifier: Apache-2.0
"""`shell.matches_plan`: the gate that asks whether the builder built the plan.

The five negatives below are not decoration. A gate is only worth its wiring if somebody has shown
it *failing*, and each of these is one of the ways a real builder can disagree with a plan that no
other gate in this repository would notice -- the walls moving, the height changing, an opening
disappearing, a provider that measured nothing, and a door too narrow for the robot. SPEC-004 §8.3
requires exactly this list, and this file is where it lives.

The fixtures are synthetic so that each negative can move exactly one thing. The measured-office
numbers are in `test_shell_build.py`; these are about the gate, not the ruler.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest

from rfw_w.scenes.contracts.room_plan import RoomPlan, plan_to_shell
from rfw_w.scenes.contracts.shell_build import (
    HEIGHT_TOLERANCE_M,
    WALL_LINE_TOLERANCE_M,
    ShellBuild,
    plan_gaps,
)
from rfw_w.scenes.solvers.shell_providers import (
    INFINIGEN_PIN,
    NOT_COMPARED_FIELDS,
    PROVIDERS,
    BuiltinShellProvider,
    ShellProviderError,
    minimum_door_width_m,
    provider_named,
)


def _fields(entries: tuple[str, ...]) -> set[str]:
    """The quantity each `not_compared` entry is about, before its colon and reason."""
    return {entry.split(":", 1)[0].strip() for entry in entries}

PLAN = {
    "schema_version": "pvscene.room-plan/v1",
    "plan_id": "gate_fixture",
    "source": "core",
    "units": "m",
    "up_axis": "Z",
    "rooms": [{
        "room_id": "room_0", "kind": "office",
        "polygon_xy": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
        "floor_z_m": 0.0, "ceiling_height_m": 2.8,
        "floor_material": "oak", "wall_material": "off_white",
    }],
    "openings": [
        {"opening_id": "door_main", "kind": "door", "room_id": "room_0",
         "wall_segment_xy": [[1.55, 0.0], [2.45, 0.0]],
         "width_m": 0.9, "height_m": 2.05, "sill_z_m": 0.0},
        {"opening_id": "window_01", "kind": "window", "room_id": "room_0",
         "wall_segment_xy": [[1.4, 3.0], [2.6, 3.0]],
         "width_m": 1.2, "height_m": 1.2, "sill_z_m": 0.9},
    ],
    "core": {
        "core_hash": "979875297fcd621734151a7b8daf9507c735d5c8bb7b148e2a85940fa6b9c4a1",
        "translation_xy": [2.0, 1.5], "rotation_yaw_rad": 0.0,
        "protected_polygon_xy": [[1.0, 1.0], [3.0, 1.0], [3.0, 2.0]],
        "support_top_z_m": 0.75, "target_offset_xy": [0.0, 0.0],
    },
    "provenance": {"spec_hash": "4b219d11f8653c384669341d3608dc1454ee5525572424f9a869c4beebba0c50",
                   "seed": 11},
}


def plan() -> RoomPlan:
    return RoomPlan.from_dict(PLAN)


def build(**overrides) -> ShellBuild:
    """A build that matches its plan, as the control, with one thing movable at a time.

    Modelled on what the `builtin` provider reports because that is the shape both providers
    report: a shell, a spread for the outline, a height, and per-opening sizes.
    """
    shell = replace(plan_to_shell(plan()), source="builtin:test")
    measurements = {
        "room.height_m": PLAN["rooms"][0]["ceiling_height_m"],
        "room.wall_line_offset_m": 0.0,
        "room.wall_line_offset_spread_m": 0.0,
        "wall.thickness_m": 0.1,
        "openings.count": len(shell.openings),
        "openings.plan_count": len(shell.openings),
    }
    measurements.update(overrides.pop("measurements", {}))
    return ShellBuild(
        shell=overrides.pop("shell", shell),
        provider=overrides.pop("provider", "builtin"),
        measurements=measurements,
        not_measured=overrides.pop("not_measured", ()),
        not_compared=overrides.pop("not_compared", ()),
        **overrides,
    )


def gaps(**overrides) -> tuple[str, ...]:
    minimum = overrides.pop("minimum_door_width_m", None)
    return plan_gaps(plan(), build(**overrides), minimum_door_width_m=minimum)


class TestTheControlPasses:
    def test_a_build_that_matches_its_plan_has_no_gaps(self) -> None:
        """The negative tests below only mean something if this one is empty."""
        assert gaps() == ()

    def test_the_control_passes_the_passage_check_too(self) -> None:
        assert gaps(minimum_door_width_m=minimum_door_width_m()) == ()

    def test_the_passage_floor_is_the_shell_policys_own_number(self) -> None:
        """Read from `ShellOptions`, not restated, so the two cannot drift apart."""
        assert minimum_door_width_m() == pytest.approx(0.65, abs=1e-9)


class TestTheWallsMoved:
    def test_a_height_ten_centimetres_off_is_named(self) -> None:
        """Negative 1 (SPEC-004 §8.3): the built room is a different height."""
        shell = replace(build().shell, height_m=2.9)
        found = gaps(shell=shell, measurements={"room.height_m": 2.9})
        assert len(found) == 1
        assert found[0].startswith("SHELL_HEIGHT_DISAGREES_WITH_PLAN")
        assert "built_m=2.9000" in found[0] and "planned_m=2.8000" in found[0]

    def test_a_height_inside_the_tolerance_is_not_named(self) -> None:
        """The other half of a tolerance: half of it must pass, or it is not a tolerance."""
        shell = replace(build().shell, height_m=2.8 + HEIGHT_TOLERANCE_M / 2)
        assert gaps(shell=shell, measurements={"room.height_m": 2.8}) == ()

    def test_a_moved_vertex_scatters_the_wall_line_offsets(self) -> None:
        """Negative 2: the outline changed, and this is how a measurement sees that.

        A measurement does not hand back a polygon -- it hands back where each built wall sits
        relative to its plan edge, and the spread of those offsets. Moving one vertex makes the two
        walls that meet there sit at *different* offsets from their plan edges, which is a
        non-zero spread even though no single wall is far away. Comparing polygons instead would
        also flag Infinigen's uniform half-wall inset, which is not a shape error at all.
        """
        found = gaps(measurements={"room.wall_line_offset_spread_m": 0.05})
        assert len(found) == 1
        assert found[0].startswith("SHELL_WALLS_NOT_ON_PLAN")
        assert "spread_m=0.05" in found[0]

    def test_a_uniform_inset_is_not_a_moved_vertex(self) -> None:
        """The measurement that decided this design: every wall 0.05 m in, spread 0.0."""
        assert gaps(measurements={"room.wall_line_offset_m": 0.05}) == ()

    def test_the_spread_is_compared_under_the_calibrated_tolerance(self) -> None:
        assert gaps(measurements={"room.wall_line_offset_spread_m": WALL_LINE_TOLERANCE_M / 2}) == ()


class TestTheOpeningsMoved:
    def test_a_deleted_opening_is_named_with_the_wall_it_should_be_on(self) -> None:
        """Negative 3: the plan asks for a window and the build has none."""
        shell = replace(
            build().shell,
            openings=tuple(o for o in build().shell.openings if o.opening_id != "window_01"),
        )
        found = gaps(shell=shell)
        assert len(found) == 1
        assert found[0].startswith("SHELL_OPENING_MISSING(window_01, window")
        assert "(1.4, 3.0)" in found[0]

    def test_a_door_that_came_back_as_a_window_is_named(self) -> None:
        from rfw_w.scenes.contracts.state import OpeningKind

        shell = replace(
            build().shell,
            openings=tuple(
                replace(o, kind=OpeningKind.WINDOW) if o.opening_id == "door_main" else o
                for o in build().shell.openings
            ),
        )
        assert any(gap.startswith("SHELL_OPENING_KIND_DISAGREES(door_main") for gap in gaps(shell=shell))

    def test_an_opening_on_the_wrong_wall_is_named(self) -> None:
        """The failure a whole-count comparison would have missed entirely."""
        shell = build().shell
        moved = tuple(
            replace(o, center_xy=(1.0, 3.0)) if o.opening_id == "door_main" else o
            for o in shell.openings
        )
        found = gaps(shell=replace(shell, openings=moved))
        assert any(gap.startswith("SHELL_OPENING_ON_WRONG_WALL(door_main") for gap in found)

    def test_an_opening_the_plan_never_asked_for_is_named(self) -> None:
        from rfw_w.scenes.contracts.state import Opening, OpeningKind

        shell = build().shell
        extra = Opening(
            opening_id="window_99", kind=OpeningKind.WINDOW,
            center_xy=(3.0, 3.0), width_m=1.0, height_m=1.2, sill_z_m=0.9,
        )
        found = gaps(shell=replace(shell, openings=shell.openings + (extra,)))
        assert any(gap.startswith("SHELL_OPENING_UNEXPECTED(window_99") for gap in found)


class TestWhatTheGateCannotAnswer:
    def test_a_build_that_measured_nothing_does_not_pass(self) -> None:
        """Negative 4: the provider reports no wall offsets, so the outline is unknown.

        This is the failure the whole `ShellBuild` contract exists to make possible. A gate that
        read a missing number as zero would report a room nobody measured as matching its plan --
        and it would look exactly like a pass.
        """
        stripped = build()
        without = {k: v for k, v in stripped.measurements.items()
                   if k != "room.wall_line_offset_spread_m"}
        found = plan_gaps(plan(), replace(stripped, measurements=without))
        assert len(found) == 1
        assert found[0].startswith("SHELL_MEASUREMENT_ABSENT(room.wall_line_offset_spread_m)")

    def test_a_missing_measurement_is_not_read_as_zero(self) -> None:
        """The gate says it could not answer, rather than answering with an absent number."""
        stripped = build()
        without = {k: v for k, v in stripped.measurements.items()
                   if k != "room.wall_line_offset_spread_m"}
        found = plan_gaps(plan(), replace(stripped, measurements=without))
        assert found[0].startswith("SHELL_MEASUREMENT_ABSENT(")
        assert "does not pass by default" in found[0]


class TestTheDoorHasToAdmitTheRobot:
    def test_a_door_below_the_floor_is_refused(self) -> None:
        """Negative 5: narrowed to below what the robot needs.

        Not a cosmetic check. A door narrower than the footprint plus margin cannot be repaired by
        any layout behind it, which is why this is the one opening dimension the gate *does*
        compare even though every other one is out of the comparison set.
        """
        shell = build().shell
        narrowed = tuple(
            replace(o, width_m=0.55) if o.opening_id == "door_main" else o
            for o in shell.openings
        )
        found = gaps(shell=replace(shell, openings=narrowed), minimum_door_width_m=0.65)
        assert any(gap.startswith("SHELL_DOOR_TOO_NARROW(door_main") for gap in found)
        assert "minimum_m=0.6500" in next(g for g in found if g.startswith("SHELL_DOOR_TOO_NARROW"))

    def test_a_door_exactly_at_the_floor_is_admitted(self) -> None:
        shell = build().shell
        narrowed = tuple(
            replace(o, width_m=0.65) if o.opening_id == "door_main" else o
            for o in shell.openings
        )
        assert gaps(shell=replace(shell, openings=narrowed), minimum_door_width_m=0.65) == ()


class TestTheBuildNamesWhatItDidNotCompare:
    def test_a_measured_but_unconstrained_quantity_is_not_reported_as_unmeasured(self) -> None:
        """`not_compared` and `not_measured` say opposite things and must not be merged.

        The opening widths *are* measured -- the report carries 0.9296 where the plan asked 0.9.
        Filing them under `not_measured` would understate the evidence; filing them nowhere would
        leave a reader unable to tell whether they were compared.
        """
        fake = build(not_compared=("openings.*.width_m: the plan does not constrain it",))
        assert "width_m" in fake.not_compared[0]
        assert fake.not_measured == ()
        assert "not_compared" in fake.to_dict()


class TestTheProviders:
    def test_both_providers_set_the_same_quantities_aside(self, tmp_path) -> None:
        """The comparison set belongs to the gate, so both reports must agree on it.

        Measured: `builtin` first reported an empty `not_compared` while `infinigen` reported five,
        and side by side they read as "one builder's opening widths were held to the plan and the
        other's were not". Neither is true. The *reasons* differ per provider; the fields do not.
        """
        built = BuiltinShellProvider().build(plan(), output_dir=tmp_path)
        assert _fields(built.not_compared) == set(NOT_COMPARED_FIELDS)

    def test_a_measured_provider_sets_the_same_fields_aside(self) -> None:
        """Pinned from the other side, so a change to one list cannot quietly miss the other."""
        from rfw_w.scenes.solvers.shell_providers import _MEASURED_NOT_COMPARED

        assert _fields(_MEASURED_NOT_COMPARED) == set(NOT_COMPARED_FIELDS)
    def test_both_providers_are_nameable(self) -> None:
        assert sorted(PROVIDERS) == ["builtin", "infinigen"]

    def test_the_infinigen_binding_is_a_commit_and_not_an_installation(self) -> None:
        """ADR-002: two Infinigen trees on this machine, only one of which takes a floor plan."""
        assert INFINIGEN_PIN == "892947be71c4ff102c6a558d088ffe149561c11f"

    def test_an_unknown_provider_fails_by_name(self) -> None:
        with pytest.raises(ShellProviderError, match=r"SHELL_PROVIDER_UNKNOWN\(meshroom\)"):
            provider_named("meshroom")

    def test_the_builtin_provider_builds_the_room_the_plan_describes(self, tmp_path) -> None:
        built = BuiltinShellProvider().build(plan(), output_dir=tmp_path)
        assert built.shell.polygon_xy == tuple(
            tuple(vertex) for vertex in PLAN["rooms"][0]["polygon_xy"]
        )
        assert built.shell.height_m == PLAN["rooms"][0]["ceiling_height_m"]
        assert built.shell.source == "builtin:core"
        assert plan_gaps(plan(), built) == ()

    def test_the_builtin_provider_says_what_it_did_not_establish(self, tmp_path) -> None:
        """A provider with an empty `not_measured` would be claiming it established everything."""
        built = BuiltinShellProvider().build(plan(), output_dir=tmp_path)
        assert built.not_measured and all(": " in entry for entry in built.not_measured)

    def test_the_same_plan_twice_gives_the_same_shell(self, tmp_path) -> None:
        """SPEC-004 §4 R6. Deterministic, so a changed shell is a changed plan and not luck."""
        first = BuiltinShellProvider().build(plan(), output_dir=tmp_path / "a")
        second = BuiltinShellProvider().build(plan(), output_dir=tmp_path / "b")
        assert first.shell == second.shell

    def test_a_build_that_cannot_run_says_so_by_name(self, tmp_path, monkeypatch) -> None:
        """No `.env.local`, no build -- and the message names the variable that was missing."""
        from rfw_w.scenes.solvers.shell_providers import INFINIGEN_ROOT_ENV, InfinigenShellProvider

        monkeypatch.delenv(INFINIGEN_ROOT_ENV, raising=False)
        with pytest.raises(ShellProviderError, match=r"SHELL_BUILD_FAILED\(RFW_INFINIGEN_ROOT"):
            InfinigenShellProvider().build(plan(), output_dir=tmp_path)

    def test_a_build_will_not_write_into_a_directory_that_already_has_one(self, tmp_path) -> None:
        """Evidence is never merged into: a second build goes in a second directory."""
        from rfw_w.scenes.solvers.shell_providers import InfinigenShellProvider

        run_dir = tmp_path / "infinigen_build"
        run_dir.mkdir()
        (run_dir / "floor_plan.json").write_text("{}")
        with pytest.raises(ShellProviderError, match=r"already has files"):
            InfinigenShellProvider(infinigen_root=tmp_path).build(plan(), output_dir=tmp_path)


class TestAWindowIsHeldToTheSpanItWasGiven:
    """The comparison uses the **host span**, not the plan's own segment, and this is the proof.

    A plan's window is a request for a hole on a wall. Infinigen honours the wall and places the
    hole where it likes within the stretch it was handed -- and that stretch is wider than the
    plan's segment, because a stretch as narrow as the plan's own width is one Infinigen silently
    builds nothing in (`room_plan.WINDOW_HOST_SPAN_M`). So the two ends of the handoff have to agree
    about the stretch, or a hole that landed at the far end of what it was given is reported as
    belonging to no planned opening at all.
    """

    #: 0.9 m, the width the office plan asks for and the width that produced the missing window:
    #: Infinigen's `door_width` on that build measured 0.9296 m, and a stretch at or below it yields
    #: no cutter (SPEC-004 §8).
    NARROW = 0.9

    def _plan(self) -> RoomPlan:
        narrowed = copy.deepcopy(PLAN)
        narrowed["openings"][1]["wall_segment_xy"] = [[1.55, 3.0], [2.45, 3.0]]
        narrowed["openings"][1]["width_m"] = self.NARROW
        return RoomPlan.from_dict(narrowed)

    def _gaps_at(self, centre_x: float) -> tuple[str, ...]:
        test_plan = self._plan()
        shell = plan_to_shell(test_plan)
        moved = tuple(
            replace(opening, center_xy=(centre_x, 3.0)) if opening.opening_id == "window_01"
            else opening
            for opening in shell.openings
        )
        build_ = build(shell=replace(shell, openings=moved))
        return plan_gaps(test_plan, build_, minimum_door_width_m=minimum_door_width_m())

    def test_the_premise_the_span_is_wider_than_the_segment(self) -> None:
        from rfw_w.scenes.contracts.room_plan import WINDOW_HOST_SPAN_M, opening_host_span

        test_plan = self._plan()
        window = next(o for o in test_plan.openings if o.opening_id == "window_01")
        start, end = opening_host_span(test_plan, window)
        assert window.segment_length_m == pytest.approx(self.NARROW)
        assert math.dist(start, end) == pytest.approx(WINDOW_HOST_SPAN_M)
        # The plan itself is untouched by it: the width it asked for is still the width it asks for.
        assert window.wall_segment_xy == ((1.55, 3.0), (2.45, 3.0))

    def test_a_hole_beyond_the_segment_but_inside_the_span_is_still_that_window(self) -> None:
        """The case the widening exists for. x=2.53 is 0.08 m past the segment and inside the span."""
        assert self._gaps_at(2.53) == ()

    def test_a_hole_past_the_span_is_not(self) -> None:
        """And the span is a bound, not a licence: 0.1 m further and the hole is off its wall.

        `SHELL_OPENING_ON_WRONG_WALL` rather than `SHELL_OPENING_MISSING`, because the shell
        carries the opening's id: the gate is told *which* planned opening this is and has only to
        decide whether it is on the stretch that opening was given. The id-less direction -- a hole
        in a mesh, which has no name -- is `openings_from_report`'s, and `test_shell_build.py`
        covers it there.
        """
        found = self._gaps_at(2.60)
        assert len(found) == 1
        assert found[0].startswith("SHELL_OPENING_ON_WRONG_WALL(window_01")

    def test_the_clamp_cannot_fire_on_a_plan_this_drawer_draws(self) -> None:
        """The arithmetic that makes the widen-and-clamp a no-op for our own plans.

        Two constants from two modules, and neither knows about the other. The drawer
        (`solvers/room.py`) will draw a window only where there is room for the *segment* plus
        `WINDOW_EDGE_INSET_M` at each end, and `_segment_centre` holds the segment to that margin --
        so the plan's own stretch is inside the wall by construction, and the only new thing the
        serialiser asks the wall to hold is `WINDOW_HOST_SPAN_M`. That is the inequality:

            window_width_m + 2 * WINDOW_EDGE_INSET_M  >=  WINDOW_HOST_SPAN_M

        1.4 m of wall for a 1.10 m span. If it fails, a drawer-made plan starts landing in the
        clamp -- still correct behaviour, and still a refusal by name where it cannot fit, but it
        would mean the floor `_windows_on_wall` enforces and the span the builder needs have
        drifted apart with nothing saying so.
        """
        from rfw_w.scenes.contracts.room_plan import WINDOW_HOST_SPAN_M
        from rfw_w.scenes.solvers.room import WINDOW_EDGE_INSET_M, ShellOptions

        shortest_wall_that_can_carry_a_window = (
            ShellOptions().window_width_m + 2 * WINDOW_EDGE_INSET_M
        )
        assert shortest_wall_that_can_carry_a_window >= WINDOW_HOST_SPAN_M, (
            shortest_wall_that_can_carry_a_window, WINDOW_HOST_SPAN_M
        )

    def test_a_window_already_wider_than_the_span_keeps_its_own(self) -> None:
        """Widening is a floor, never a cut: the gate fixture's 1.2 m window is its own span."""
        from rfw_w.scenes.contracts.room_plan import opening_host_span

        window = next(o for o in plan().openings if o.opening_id == "window_01")
        assert opening_host_span(plan(), window) == window.wall_segment_xy


class TestTheProviderRefusesAWindowItCannotBuild:
    """`SHELL_WINDOW_SPAN_UNBUILDABLE`: the one place that can say *why* a window is missing.

    Infinigen does not reject a window span that is too short, it drops it -- no cutter, no hole,
    and nothing in the scene to say one was ever asked for -- so without this the run would reach a
    room missing a window it was told to build, and the gate would report the absence without the
    reason. The plan here is one this repository's drawer never makes; the contract permits it
    (`RoomPlan.source` may be ``external`` or ``bank``), which is why the check exists.
    """

    NARROW_WALL = {
        **PLAN,
        "plan_id": "short_wall",
        "rooms": [{**PLAN["rooms"][0],
                   "polygon_xy": [[0.0, 0.0], [4.0, 0.0], [4.0, 0.8], [0.0, 0.8]]}],
        "openings": [{**PLAN["openings"][0], "wall_segment_xy": [[1.55, 0.0], [2.45, 0.0]]},
                     {**PLAN["openings"][1], "wall_segment_xy": [[0.0, 0.15], [0.0, 0.65]],
                      "width_m": 0.5}],
    }

    def test_the_premise_the_wall_cannot_hold_the_span(self) -> None:
        from rfw_w.scenes.contracts.room_plan import (
            buildable_window_span_m,
            opening_host_span,
        )

        test_plan = RoomPlan.from_dict(self.NARROW_WALL)
        window = next(o for o in test_plan.openings if o.opening_id == "window_01")
        start, end = opening_host_span(test_plan, window)
        assert math.dist(start, end) == pytest.approx(0.8)  # the whole wall, which is all it has
        assert math.dist(start, end) <= buildable_window_span_m(0.1)

    def test_the_build_is_refused_by_name_before_anything_is_written(self, tmp_path) -> None:
        from rfw_w.scenes.solvers.shell_providers import InfinigenShellProvider

        provider = InfinigenShellProvider(infinigen_root=tmp_path)
        with pytest.raises(ShellProviderError, match=r"SHELL_WINDOW_SPAN_UNBUILDABLE\(window_01"):
            provider.build(RoomPlan.from_dict(self.NARROW_WALL), output_dir=tmp_path / "out")
        # Refused before the directory exists, so a refusal leaves no half-made build behind.
        assert not (tmp_path / "out" / "infinigen_build").exists()


class TestTheGateIsWiredIntoTheLadder:
    def test_the_gate_id_is_a_named_gate(self) -> None:
        from rfw_w.scenes.contracts.readiness import GateId

        assert GateId.SHELL_MATCHES_PLAN.value == "shell.matches_plan"

    def test_it_is_required_for_functional_scene_ready_and_above(self) -> None:
        from rfw_w.scenes.contracts.readiness import GateId, ReadinessLevel, required_gates

        for level in (ReadinessLevel.FUNCTIONAL_SCENE_READY,
                      ReadinessLevel.ROBOT_FEASIBLE_SCENE_READY,
                      ReadinessLevel.TASK_VERIFIED):
            assert GateId.SHELL_MATCHES_PLAN.value in required_gates(level, ["isaac"])

    def test_it_is_not_required_for_a_draft_or_a_simready_scene(self) -> None:
        """It is a claim about a *built* room; a shell that was never built has nothing to check."""
        from rfw_w.scenes.contracts.readiness import GateId, ReadinessLevel, required_gates

        for level in (ReadinessLevel.DRAFT, ReadinessLevel.SIMREADY_SCENE_READY):
            assert GateId.SHELL_MATCHES_PLAN.value not in required_gates(level, ["isaac"])

    def test_a_failure_raises_an_event_rather_than_being_unclassified(self) -> None:
        """Without the `shell.` prefix `kind_for_gate` returns `None` and the gate is silent."""
        from rfw_w.scenes.orchestrator.events import EventKind, code_for_gate, kind_for_gate

        assert kind_for_gate("shell.matches_plan") is EventKind.SEMANTIC_CONFLICT
        assert code_for_gate("shell.matches_plan").value == "PRECONDITION_UNMET"

    def test_the_repair_layer_knows_the_kind_so_the_gate_never_goes_stale(self) -> None:
        """An unregistered gate is one the repair layer never re-runs."""
        from rfw_w.scenes.contracts.state import EvidenceKind, gates_for_evidence

        assert EvidenceKind.SHELL.value == "shell"
        assert gates_for_evidence([EvidenceKind.SHELL]) == frozenset()

    def test_the_evidence_recorder_and_the_independent_verifier_agree(self) -> None:
        """`test_verify_delivery.py` carries its own copy of the required list; keep them equal."""
        from tests.scenes.unit import test_verify_delivery

        from rfw_w.scenes.contracts.readiness import ReadinessLevel, required_gates

        required = set(required_gates(ReadinessLevel.FUNCTIONAL_SCENE_READY, ["isaac"]))
        required -= {"backend.isaac.load", "backend.isaac.run", "backend.isaac.render"}
        assert required == set(test_verify_delivery.REQUIRED)
