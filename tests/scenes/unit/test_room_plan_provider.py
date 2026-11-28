# SPDX-License-Identifier: Apache-2.0
"""Turning the fitted room into a plan.

The claim under test is narrow: the plan is a faithful restatement of the room the fit drew, so a
provider consuming the plan builds the room that was chosen. The refusals matter as much as the
restatement -- a plan that silently described a different room from the one that was fitted would
be worse than no plan, because it would look like evidence.

The inline spec keeps this independent of `experiments/`, so the file runs without any asset root.
"""

from __future__ import annotations

import math

import pytest

from rfw_w.scenes.contracts.room_plan import (
    RoomPlanError,
    check_plan,
    floor_plan_payload,
    plan_to_shell,
)
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.materials import FLOOR, WALL, resolve_roles
from rfw_w.scenes.solvers.room import fit_room_around_cell
from rfw_w.scenes.solvers.room_plan import (
    CoreRoomPlanProvider,
    _edges,
    plan_from_fitted,
)

SPEC = SceneSpec.from_dict({
    "schema_version": "pvscene.scene-spec/v3",
    "scene_id": "office_pick_cup_m1_001",
    "seed": 11,
    "prompt": "warm wood office",
    "room": {"size_bounds_m": [[4, 5], [3, 4], [2.6, 3.0]]},
    "style": {"theme": "warm_minimal", "palette": ["oak", "off_white", "muted_green", "brass"]},
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


def fitted_and_planned(spec: SceneSpec = SPEC):
    """(cell, fitted room, plan) for one spec -- the three things every test here compares."""
    cell = solve_cell(spec, seed=spec.seed)
    fitted = fit_room_around_cell(spec, cell, room_seed=spec.seed)
    plan = plan_from_fitted(spec, cell, fitted, room_seed=spec.seed)
    return cell, fitted, plan


class TestThePlanRestatesTheRoom:
    def test_the_outline_and_height_come_across_unchanged(self) -> None:
        _, fitted, plan = fitted_and_planned()
        assert plan.rooms[0].polygon_xy == fitted.shell.polygon_xy
        assert plan.rooms[0].ceiling_height_m == fitted.shell.height_m
        assert plan.rooms[0].floor_z_m == fitted.shell.floor_z_m

    def test_every_drawn_opening_is_in_the_plan(self) -> None:
        _, fitted, plan = fitted_and_planned()
        assert [opening.opening_id for opening in plan.openings] == [
            opening.opening_id for opening in fitted.shell.openings
        ]

    def test_the_request_path_reproduces_the_drawn_shell(self) -> None:
        """The acceptance claim, in one assertion: plan -> shell == the shell that was fitted."""
        _, fitted, plan = fitted_and_planned()
        requested = plan_to_shell(plan)
        assert requested.polygon_xy == fitted.shell.polygon_xy
        assert requested.height_m == fitted.shell.height_m
        assert len(requested.openings) == len(fitted.shell.openings)

        drawn = {opening.opening_id: opening for opening in fitted.shell.openings}
        for opening in requested.openings:
            other = drawn[opening.opening_id]
            assert opening.kind == other.kind
            assert opening.center_xy == pytest.approx(other.center_xy, abs=1e-6)
            assert opening.width_m == pytest.approx(other.width_m, abs=1e-6)
            assert opening.height_m == other.height_m
            assert opening.sill_z_m == other.sill_z_m

    def test_the_plan_is_self_consistent(self) -> None:
        _, _, plan = fitted_and_planned()
        assert check_plan(plan) == ()

    def test_it_needs_no_spec_to_read_back(self) -> None:
        """A plan is a document, not a view: everything a builder needs is in it."""
        payload = floor_plan_payload(fitted_and_planned()[2])
        assert list(payload["rooms"]) and payload["doors"]


class TestSegmentReconstruction:
    """`wall_index` is the only link back to the outline, and it must be exact."""

    def test_every_segment_lies_on_the_wall_it_names(self) -> None:
        _, fitted, plan = fitted_and_planned()
        walls = _edges(fitted.shell.polygon_xy)
        plans = fitted.options.openings

        for opening, source in zip(plan.openings, plans):
            (ax, ay), (bx, by) = walls[source.wall_index]
            for x, y in opening.wall_segment_xy:
                # Collinear with the wall and between its endpoints.
                cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
                wall_length = math.dist((ax, ay), (bx, by))
                assert abs(cross) / wall_length < 1e-6
                t = ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / wall_length**2
                assert -1e-6 <= t <= 1 + 1e-6

    def test_the_segment_is_centred_on_the_wall_index_the_shell_lost(self) -> None:
        """A shell keeps only `center_xy`; the plan keeps the wall identity too."""
        _, fitted, plan = fitted_and_planned()
        walls = _edges(fitted.shell.polygon_xy)
        for opening, source in zip(plan.openings, fitted.options.openings):
            (ax, ay), (bx, by) = walls[source.wall_index]
            centre = opening.centre_m
            expected = (
                ax + (bx - ax) * source.offset_along_m / math.dist((ax, ay), (bx, by)),
                ay + (by - ay) * source.offset_along_m / math.dist((ax, ay), (bx, by)),
            )
            assert centre == pytest.approx(expected, abs=1e-6)

    def test_the_width_is_the_segment_by_construction(self) -> None:
        _, _, plan = fitted_and_planned()
        for opening in plan.openings:
            assert opening.segment_length_m == pytest.approx(opening.width_m, abs=1e-5)

    def test_a_wall_index_off_the_end_is_refused(self) -> None:
        """Rather than wrapping to the wrong wall, which is what `% len` would do silently."""
        from rfw_w.scenes.solvers.room_plan import _wall_index

        with pytest.raises(RoomPlanError, match=r"PLAN_NOT_SERIALISABLE\(door_main\)"):
            _wall_index(7, 6, "door_main")


class TestDeterminism:
    def test_the_same_spec_gives_the_same_plan_hash(self) -> None:
        assert fitted_and_planned()[2].content_hash == fitted_and_planned()[2].content_hash

    def test_a_different_seed_gives_a_different_plan(self) -> None:
        other = SceneSpec.from_dict({**SPEC.to_dict(), "seed": 99})
        assert fitted_and_planned(other)[2].content_hash != fitted_and_planned()[2].content_hash


class TestTheCoreIsEvidence:
    def test_the_cell_hash_is_carried_verbatim(self) -> None:
        cell, _, plan = fitted_and_planned()
        assert plan.core.core_hash == cell.core_hash

    def test_the_translation_is_the_one_the_fit_applied(self) -> None:
        _, fitted, plan = fitted_and_planned()
        assert plan.core.translation_xy == pytest.approx(fitted.cell_translation_xy, abs=1e-6)

    def test_the_protected_region_is_in_the_cell_frame(self) -> None:
        """Applying the translation has to land on the region the fit reserved."""
        cell, fitted, plan = fitted_and_planned()

        def flat(points) -> list[float]:
            # Flattened because `approx` compares nested sequences badly, and a polygon is a
            # sequence of pairs where a list-versus-tuple difference reads as a mismatch.
            return [coordinate for point in points for coordinate in point]

        assert flat(plan.core.protected_polygon_xy) == pytest.approx(
            flat(cell.protected.to_list()), abs=1e-6
        )
        assert flat(plan.core.protected_in_room_xy) == pytest.approx(
            flat(fitted.protected_in_room.to_list()), abs=1e-6
        )

    def test_the_support_and_target_come_from_the_cell(self) -> None:
        cell, _, plan = fitted_and_planned()
        assert plan.core.support_top_z_m == pytest.approx(cell.support_height_m, abs=1e-6)
        assert plan.core.target_offset_xy == pytest.approx(cell.target_offset, abs=1e-6)

    def test_the_provenance_names_the_spec_and_the_options(self) -> None:
        """SPEC-001 notes `source` says who signed it; provenance says from what."""
        from rfw_w.scenes.contracts.hashing import content_hash, is_digest

        _, fitted, plan = fitted_and_planned()
        assert plan.source == "core"
        assert plan.provenance.spec_hash == content_hash(SPEC.to_dict())
        assert plan.provenance.seed == SPEC.seed
        assert plan.provenance.shell_options_digest == content_hash(fitted.options.to_dict())
        assert is_digest(plan.core.core_hash)


class TestMaterials:
    def test_the_refs_are_palette_tokens_not_colours(self) -> None:
        _, _, plan = fitted_and_planned()
        tokens = resolve_roles(SPEC.style.palette)
        assert plan.rooms[0].floor_material == tokens[FLOOR]
        assert plan.rooms[0].wall_material == tokens[WALL]
        # A token, not an RGB triple and not the role's own name.
        assert plan.rooms[0].floor_material in SPEC.style.palette

    def test_a_role_the_palette_did_not_name_is_empty_rather_than_invented(self) -> None:
        from rfw_w.scenes.solvers.materials import resolve_roles as roles

        # A palette with nothing that can serve a floor leaves the role on its default colour.
        assert roles(("brass",))[FLOOR] is None


class TestRefusals:
    def test_openings_the_options_do_not_name_are_refused(self) -> None:
        """`build_shell` regenerates openings when the options leave them empty...

        ...so the shell would carry openings no plan describes. A plan that disagreed with the room
        it claims to be about is worse than no plan.
        """
        from dataclasses import replace

        cell = solve_cell(SPEC, seed=SPEC.seed)
        fitted = fit_room_around_cell(SPEC, cell, room_seed=SPEC.seed)
        hollowed = replace(fitted, options=replace(fitted.options, openings=()))
        assert fitted.shell.openings, "the fit should have produced openings for this to be a test"
        with pytest.raises(RoomPlanError, match=r"PLAN_NOT_SERIALISABLE\(openings\)"):
            plan_from_fitted(SPEC, cell, hollowed, room_seed=SPEC.seed)

    def test_a_shell_with_a_hole_is_refused(self) -> None:
        from dataclasses import replace

        cell = solve_cell(SPEC, seed=SPEC.seed)
        fitted = fit_room_around_cell(SPEC, cell, room_seed=SPEC.seed)
        holed = replace(fitted, shell=replace(fitted.shell, holes_xy=(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),)))
        with pytest.raises(RoomPlanError, match=r"PLAN_NOT_SERIALISABLE\(rooms\)"):
            plan_from_fitted(SPEC, cell, holed, room_seed=SPEC.seed)


class TestTheProvider:
    def test_the_provider_agrees_with_the_pure_translation(self) -> None:
        """`CoreRoomPlanProvider` fits and translates; the tests above translate a given fit.

        Equal hashes mean the provider's own fit produced the same room, so the two entry points
        cannot drift apart.
        """
        cell = solve_cell(SPEC, seed=SPEC.seed)
        fitted = fit_room_around_cell(SPEC, cell, room_seed=SPEC.seed)
        expected = plan_from_fitted(SPEC, cell, fitted, room_seed=SPEC.seed)
        provided = CoreRoomPlanProvider().plan(SPEC, cell, room_seed=SPEC.seed)
        assert provided.content_hash == expected.content_hash

    def test_the_standoff_reaches_the_fit(self) -> None:
        """A caller that changes the standoff must not get a plan for the old one."""
        cell = solve_cell(SPEC, seed=SPEC.seed)
        tight = CoreRoomPlanProvider().plan(SPEC, cell, room_seed=SPEC.seed, wall_standoff_m=0.20)
        roomy = CoreRoomPlanProvider().plan(SPEC, cell, room_seed=SPEC.seed, wall_standoff_m=0.80)
        assert tight.content_hash != roomy.content_hash
