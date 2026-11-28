# SPDX-License-Identifier: Apache-2.0
"""Zone planning.

The plan is evidence: `placeable_area_m2` is the denominator a coverage target is measured
against, and the keepouts are what the layout is told to avoid. So the tests here are not that
`plan_zones` returns something, but that what it returns is *true about the room* -- every
published polygon is real floor, no two placeable zones claim the same square metre, and nothing
placeable sits in a keepout.

A bounding-box stand-in would pass all of these on a rectangular room and fail on the notched
one, which is why the seeded sweep below runs real shells rather than fixtures.
"""

from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.geometry.polygon import Polygon, intersect_polygons
from rfw_w.scenes.solvers.room import Corner, Family, Notch, ShellOptions, build_shell
from rfw_w.scenes.solvers.zones import ZoneError, ZoneGraph, ZoneKind, plan_zones

REPO_ROOT = Path(__file__).resolve().parents[3]

SPEC = SceneSpec.from_dict({
    "schema_version": "pvscene.scene-spec/v3",
    "scene_id": "office_pick_cup_m1_001",
    "seed": 11,
    "prompt": "warm wood office",
    "room": {"size_bounds_m": [[4, 5], [3, 4], [2.6, 3.0]]},
    "style": {"theme": "warm_minimal", "palette": ["wood"]},
    "requested_backends": ["isaac"],
    "requested_readiness": "functional_scene_ready",
    "entities": [
        {"id": "desk", "category": "desk", "physics_type": "rigid", "mobility": "fixed"},
        {"id": "cup", "category": "cup", "physics_type": "rigid", "mobility": "dynamic",
         "role": "task_target"},
    ],
    "initial_relations": [{"id": "r1", "subject": "cup", "predicate": "ON", "object": "desk"}],
    "task": {"action": "pick_place", "robot_profile": "franka_panda", "targets": ["cup"]},
})

# Wide enough to sweep every room shape the generator produces, including the notched outline
# and a door on either wall.
SWEEP_SEEDS = range(40)

TOLERANCE_M2 = 1e-6


def shared_area(left, right) -> float:
    """Area two groups of polygons have in common."""
    return sum(
        piece.area
        for first in left
        for second in right
        for piece in intersect_polygons(first, second)
    )


def outside_area(pieces, reference: Polygon) -> float:
    """How much of ``pieces`` is not inside ``reference``."""
    return sum(
        max(0.0, piece.area - sum(p.area for p in intersect_polygons(piece, reference)))
        for piece in pieces
    )


# How many of the swept seeds the planner may refuse before the sweep stops meaning anything.
# The shell gained shape families, and some of those rooms -- a T whose stem the corridor cannot
# reach the work strip through -- are genuinely unplannable by the fixed strip-and-centre scheme.
# Measured 9 in 900 across the three experiments; 3 in 40 for this spec. A ratio rather than zero
# because refusing such a room is correct, and a bound rather than nothing because a sweep that
# silently shrank would keep passing while covering less and less.
MAX_REFUSALS = 6


@pytest.fixture(scope="module")
def refusals() -> list[str]:
    return _sweep()[1]


def _sweep() -> tuple[list[tuple[int, ZoneGraph]], list[str]]:
    planned: list[tuple[int, ZoneGraph]] = []
    refused: list[str] = []
    for seed in SWEEP_SEEDS:
        spec = replace(SPEC, seed=seed)
        shell = build_shell(spec)
        try:
            planned.append((seed, plan_zones(shell, spec.style)))
        except ZoneError as error:
            refused.append(f"seed {seed}: {error}")
    return planned, refused


@pytest.fixture(scope="module")
def graphs() -> list[tuple[int, ZoneGraph]]:
    """One plan per seed, over real generated shells, minus the seeds the planner refuses."""
    return _sweep()[0]


class TestTheSweepStillSweeps:
    def test_the_refusal_rate_is_bounded_and_named(self, refusals: list[str]) -> None:
        assert len(refusals) <= MAX_REFUSALS, (
            f"{len(refusals)} of {len(list(SWEEP_SEEDS))} seeds refused, over the bound of "
            f"{MAX_REFUSALS}: " + "; ".join(refusals)
        )
        # Every refusal must name what is wrong, not just fail.
        for refusal in refusals:
            assert any(
                named in refusal
                for named in ("work strip", "CIRCULATION", "cannot hold the task")
            ), refusal


class TestInvariantSweep:
    """The whole point of the module: a plan that is checked rather than asserted."""

    def test_no_seed_violates_its_own_invariants(self, graphs) -> None:
        violations = {seed: gaps for seed, graph in graphs if (gaps := graph.check_invariants())}
        assert violations == {}, (
            "zone plans that misdescribe their room: "
            + "; ".join(f"seed {seed}: {gaps}" for seed, gaps in sorted(violations.items()))
        )

    def test_every_zone_is_inside_the_room(self, graphs) -> None:
        for seed, graph in graphs:
            for zone in graph.zones:
                escaped = outside_area(zone.polygons, graph.room)
                assert escaped <= TOLERANCE_M2, (
                    f"seed {seed}: {zone.kind.value} zone puts {escaped:.4f} m2 outside the room"
                )

    def test_circulation_is_inside_the_room(self, graphs) -> None:
        """The corridor is drawn from the door centre, which is *on* the wall line, so half its
        width starts outside unless it is clipped. An unclipped corridor reserves floor that is
        not in the building."""
        for seed, graph in graphs:
            escaped = outside_area(graph.circulation, graph.room)
            assert escaped <= TOLERANCE_M2, (
                f"seed {seed}: circulation puts {escaped:.4f} m2 outside the room"
            )

    def test_placeable_zones_do_not_overlap_each_other(self, graphs) -> None:
        for seed, graph in graphs:
            placeable = [zone for zone in graph.zones if not zone.keepout]
            for index, first in enumerate(placeable):
                for second in placeable[index + 1:]:
                    overlap = shared_area(first.polygons, second.polygons)
                    assert overlap <= TOLERANCE_M2, (
                        f"seed {seed}: {first.kind.value} and {second.kind.value} both claim "
                        f"{overlap:.4f} m2 of floor"
                    )

    def test_placeable_zones_clear_the_keepouts(self, graphs) -> None:
        for seed, graph in graphs:
            keepouts = graph.keepouts
            for zone in (z for z in graph.zones if not z.keepout):
                overlap = shared_area(zone.polygons, keepouts)
                assert overlap <= TOLERANCE_M2, (
                    f"seed {seed}: {zone.kind.value} zone sits {overlap:.4f} m2 inside a keepout"
                )

    def test_placeable_area_counts_no_keepout_floor(self, graphs) -> None:
        """`placeable_area_m2` is the denominator of the coverage target. Summing the zones
        without removing the keepouts makes the room look bigger than it is."""
        for seed, graph in graphs:
            keepouts = graph.keepouts
            truth = sum(
                max(0.0, zone.area_m2 - shared_area(zone.polygons, keepouts))
                for zone in graph.zones
                if not zone.keepout
            )
            assert graph.placeable_area_m2 == pytest.approx(truth, abs=TOLERANCE_M2), (
                f"seed {seed}: reports {graph.placeable_area_m2:.4f} m2 placeable, "
                f"{truth:.4f} m2 after the keepouts"
            )


class TestNotchedRoom:
    """A bounding box would report the notch as usable floor."""

    def test_zones_never_reach_into_the_notch(self) -> None:
        spec = replace(SPEC, seed=3)
        notched = ShellOptions(
            family=Family.L,
            notches=(Notch(corner=Corner.TOP_RIGHT, width_m=1.0, depth_m=1.0),),
            door_center=(2.0, 0.0),
        )
        shell = build_shell(spec, options=notched)
        graph = plan_zones(shell, spec.style)
        room = Polygon(shell.polygon_xy)
        assert room.area_ratio_to_bounding_box() < 0.95, "fixture is not actually notched"
        for zone in graph.zones:
            for piece in zone.polygons:
                assert outside_area((piece,), room) <= TOLERANCE_M2, (
                    f"{zone.kind.value} zone leaves the room outline"
                )


class TestRefusesToLie:
    def test_rejects_a_non_positive_footprint(self) -> None:
        shell = build_shell(replace(SPEC, seed=1))
        with pytest.raises(ZoneError, match="footprint must be positive"):
            plan_zones(shell, SPEC.style, robot_footprint_m=0.0)

    def test_the_gate_actually_gates(self, graphs) -> None:
        """`check_invariants` must be capable of failing, or the sweep above proves nothing.

        Feeding it a graph with a zone deliberately pushed outside the room is how we know the
        check is measuring rather than returning a constant.
        """
        _, graph = graphs[0]
        outside = Polygon.rectangle(-10.0, -10.0, -9.0, -9.0)
        zones = tuple(
            replace(zone, polygons=(outside,)) if index == 0 else zone
            for index, zone in enumerate(graph.zones)
        )
        broken = replace(graph, zones=zones)
        gaps = broken.check_invariants()
        assert any(gap.startswith("ZONE_OUTSIDE_ROOM") for gap in gaps), gaps


class TestTheExperimentsDefaultSeeds:
    """A shape the shell can draw is not the same as a room the pipeline can furnish.

    The regression this exists for: widening the shape vocabulary to T and U left
    `office_pick_cup`'s own seed as a T whose work strip was too short for the desk, and the build
    died with ``'desk' could not be placed, so nothing can support the task target``. Nothing
    between the shell and the placement reported the shape as the cause, and no test in the suite
    covered the three shipped experiments at their own seeds.
    """

    # The work strip is 1.25 m deep (`zones.WALL_DEPTH_M`) and its length comes from the wall it
    # runs along, so the length is what decides whether the support fits. The office desk is
    # 1.370 m long; 1.4 m leaves it 3 cm of slack rather than a knife edge. The kitchen island and
    # the bedroom nightstand are shorter, so this bound is the strictest of the three.
    MIN_WORK_STRIP_LENGTH_M = 1.4

    @pytest.mark.parametrize(
        "experiment",
        ["office_pick_cup", "kitchen_fetch_jar", "bedroom_pick_alarm_clock"],
    )
    def test_the_default_seed_plans_a_work_zone_the_task_fits_in(self, experiment: str) -> None:
        task = data_path('examples') / experiment / 'task.json'
        spec = SceneSpec.from_dict(json.loads(task.read_text()))
        graph = plan_zones(build_shell(spec), spec.style)
        work = graph.zone_for(ZoneKind.WORK)
        assert work is not None, f"{experiment}: the plan has no work zone at all"
        longest = max(
            max(
                max(point[0] for point in piece.vertices)
                - min(point[0] for point in piece.vertices),
                max(point[1] for point in piece.vertices)
                - min(point[1] for point in piece.vertices),
            )
            for piece in work.polygons
        )
        assert longest >= self.MIN_WORK_STRIP_LENGTH_M, (
            f"{experiment} seed {spec.seed}: the work zone is {longest:.3f} m along its longest "
            f"axis, short of the {self.MIN_WORK_STRIP_LENGTH_M} m the task's support needs"
        )


class TestThePlanAroundATaskCell:
    """What changes when the room was fitted around a task cell, and what must not.

    `cell_protected` is the task's own floor: the support, the places the robot may stand, and the
    approach between them -- solved before the room existed. Every zone here is carved around it and
    the door corridor ends on its boundary, so the task stops competing with the periphery for the
    same square metres. The disclosure matters as much as the reservation: the region is *published*
    on the graph, because a plan that quietly shrank to make room for something would be
    indistinguishable from a plan for a smaller room.
    """

    @pytest.mark.parametrize("seed", [11, 23, 37, 51])
    def test_no_zone_and_no_corridor_claims_floor_the_task_reserved(self, seed: int) -> None:
        """The property the whole reservation exists for.

        A zone overlapping the workspace is a piece of furniture planned onto the desk or into the
        aisle the arm stands in; a corridor overlapping it is the way in crossing the work. Both
        would be invisible in the manifest -- the zones would look plausible and the counts would
        agree with themselves.
        """
        spec = replace(SPEC, seed=seed)
        shell = build_shell(spec)
        protected = Polygon.rectangle(1.4, 1.0, 2.9, 2.4)
        graph = plan_zones(shell, spec.style, cell_protected=protected)

        assert graph.workspace is protected
        placeable = [piece for zone in graph.zones if not zone.keepout for piece in zone.polygons]
        assert shared_area(placeable, [protected]) <= TOLERANCE_M2
        assert shared_area(list(graph.circulation), [protected]) <= TOLERANCE_M2

    def test_the_corridor_reaches_the_task_rather_than_the_work_strip(self) -> None:
        """The invariant that used to be about the work zone, aimed at the cell instead.

        The work strip is a strip on whichever wall looked longest -- a guess at where the task
        would be. The corridor was drawn to that guess. With a cell in hand, "the passage arrives
        somewhere" has to mean "it arrives at the task".
        """
        from rfw_w.scenes.geometry.clearance import gap_between

        spec = replace(SPEC, seed=11)
        shell = build_shell(spec)
        protected = Polygon.rectangle(1.4, 1.0, 2.9, 2.4)
        graph = plan_zones(shell, spec.style, cell_protected=protected)
        assert graph.circulation
        assert min(gap_between(piece, protected) for piece in graph.circulation) <= TOLERANCE_M2

    def test_a_workspace_outside_the_room_is_refused_by_name(self) -> None:
        """Zoning around a region the room does not contain would plan over floor that is not there."""
        spec = replace(SPEC, seed=11)
        shell = build_shell(spec)
        elsewhere = Polygon.rectangle(50.0, 50.0, 51.0, 51.0)
        with pytest.raises(ZoneError, match="CELL_PROTECTED_OUTSIDE_ROOM"):
            plan_zones(shell, spec.style, cell_protected=elsewhere)

    def test_a_plan_without_a_cell_publishes_no_workspace(self) -> None:
        """The room-only path is untouched, and says so rather than carrying an empty region.

        Every other test in this file runs this path -- the sweep over forty seeds, the refusal
        rate, the placeable-area identity -- and passes unchanged, which is the evidence that
        nothing about the default plan moved.
        """
        spec = replace(SPEC, seed=11)
        graph = plan_zones(build_shell(spec), spec.style)
        assert graph.workspace is None
        assert graph.to_dict()["workspace_xy"] is None

    def test_the_seating_band_is_not_published_when_the_task_anchors_it(self) -> None:
        """A wall strip is the wrong shape for "beside the desk" once the desk's place is known.

        The band is drawn beyond the work strip on the longest wall, which is a proxy for "beyond
        the work area" that only holds while the work area is on that wall. Measured on
        `office_pick_cup` seed 11 it landed on the wall the fitted workspace itself occupies, and the
        seating zone came out at 0.11-0.72 m2 -- smaller than a chair.
        """
        spec = replace(SPEC, seed=11)
        shell = build_shell(spec)
        with_cell = plan_zones(
            shell, spec.style, cell_protected=Polygon.rectangle(1.4, 1.0, 2.9, 2.4)
        )
        assert with_cell.zone_for(ZoneKind.SEATING) is None
        assert plan_zones(shell, spec.style).zone_for(ZoneKind.SEATING) is not None


    def test_a_zone_over_the_task_floor_is_named(self) -> None:
            """The invariant can fail, and this proves it.

            The protected region is a *reservation*, not a keepout, so it is in neither `keepouts` nor
            any zone -- which is why only a test watched this until the invariant was added. The
            doctored graph is what a future change to `plan_zones` would produce: floor published on top
            of the task's own, with every other invariant still satisfied.
            """
            spec = replace(SPEC, seed=11)
            shell = build_shell(spec)
            protected = Polygon.rectangle(1.4, 1.0, 2.9, 2.4)
            graph = plan_zones(shell, spec.style, cell_protected=protected)
            assert not any("TASK_FLOOR" in gap for gap in graph.check_invariants()), (
                "the invariant must be silent on a plan that respects the reservation"
            )

            placeable_index = next(i for i, zone in enumerate(graph.zones) if not zone.keepout)
            doctored = replace(
                graph,
                zones=tuple(
                    replace(zone, polygons=(protected,)) if i == placeable_index else zone
                    for i, zone in enumerate(graph.zones)
                ),
            )
            gaps = doctored.check_invariants()
            assert any(gap.startswith("ZONE_OVERLAPS_THE_TASK_FLOOR(") for gap in gaps), gaps

    def test_the_periphery_area_is_the_room_less_the_task_floor(self) -> None:
        """The number the "the periphery samples in `room - protected`" claim rests on.

        Without it a reader can see `placeable_area_m2` shrink after a room is fitted around a cell
        and cannot see why; with it the ceiling on what the periphery could ever be offered is a
        figure that travels with the plan.
        """
        spec = replace(SPEC, seed=11)
        shell = build_shell(spec)
        room_area = Polygon(shell.polygon_xy).area

        without = plan_zones(shell, spec.style)
        assert without.periphery_area_m2 == round(room_area, 6), (
            "with no task in the frame the periphery may sample the whole room"
        )

        protected = Polygon.rectangle(1.4, 1.0, 2.9, 2.4)
        with_cell = plan_zones(shell, spec.style, cell_protected=protected)
        assert with_cell.periphery_area_m2 == round(room_area - protected.area, 6)
        assert with_cell.periphery_area_m2 < without.periphery_area_m2

    def test_every_zone_of_a_kind_is_reachable_through_region_for(self) -> None:
        """`zone_for` hands back the largest, and the siblings carry floor nothing can reach.

        Measured on the office: three `DISPLAY` strips at 1.02, 0.20 and 0.06 m2. A plan row
        confined to `DISPLAY` could only ever stand on the first, so the other two were floor the
        plan published and no piece could use -- which is a loss that looks like nothing at all in
        the manifest.
        """
        spec = replace(SPEC, seed=11)
        shell = build_shell(spec)
        graph = plan_zones(
            shell, spec.style, cell_protected=Polygon.rectangle(1.4, 1.0, 2.9, 2.4)
        )

        for kind in ZoneKind:
            found = graph.of_kind(kind)
            region = graph.region_for(kind)
            if not found:
                assert region is None, f"{kind.value}: no zone, so no region"
                continue
            assert region is not None and region.kind is kind
            assert region.area_m2 == pytest.approx(sum(zone.area_m2 for zone in found))
            assert len(region.polygons) == sum(len(zone.polygons) for zone in found)
            assert region.wall_index == max(found, key=lambda zone: zone.area_m2).wall_index

        assert any(len(graph.of_kind(kind)) > 1 for kind in ZoneKind), (
            "no kind has siblings on this room, so this test would pass without proving anything"
        )


class TestZoneForRefusesAValueThatIsNotAZoneKind:
    """The silent `None` that hid a caller asking for a zone that could never exist.

    `run_m1_acceptance.py` passed a `PlacementKind` -- `"against_wall"` -- into `zone_for`, and
    `of_kind` compares with `is` against the `ZoneKind` members, so it returned `None` every time.
    The caller read that as "the room has no such zone" and placed the support unzoned. The check
    would have named it; this is the negative test that proves the check can fail.
    """

    def test_a_placement_kind_is_refused_by_name(self) -> None:
        from rfw_w.scenes.solvers.furnish import PlacementKind

        graph = plan_zones(build_shell(replace(SPEC, seed=11)), SPEC.style)
        with pytest.raises(ZoneError, match="ZONE_KIND_EXPECTED"):
            graph.zone_for(PlacementKind.AGAINST_WALL)

    def test_a_plain_string_is_refused_too(self) -> None:
        graph = plan_zones(build_shell(replace(SPEC, seed=11)), SPEC.style)
        with pytest.raises(ZoneError, match="ZONE_KIND_EXPECTED"):
            graph.zone_for("work")

    def test_a_real_zone_kind_still_answers_with_none_when_absent(self) -> None:
        """The refusal must not swallow the legitimate answer."""
        graph = plan_zones(build_shell(replace(SPEC, seed=11)), SPEC.style)
        assert graph.zone_for(ZoneKind.CIRCULATION) is None or graph.zone_for(ZoneKind.CIRCULATION)
        assert graph.zone_for(ZoneKind.WORK) is not None

    def test_region_for_refuses_it_the_same_way(self) -> None:
        """Both readers share one check, so neither can go back to answering a silent `None`."""
        from rfw_w.scenes.solvers.furnish import PlacementKind

        graph = plan_zones(build_shell(replace(SPEC, seed=11)), SPEC.style)
        for bad in (PlacementKind.AGAINST_WALL, "work"):
            with pytest.raises(ZoneError, match="ZONE_KIND_EXPECTED"):
                graph.region_for(bad)
