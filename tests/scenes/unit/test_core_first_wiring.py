# SPDX-License-Identifier: Apache-2.0
"""The build's order, and the two placement rules the reordering had to correct.

S3a left `fit_room_around_cell` written, tested and **unwired**: `build_scene` drew the room first
and solved the task cell afterwards, so the room still decided where the desk stood, the desk
decided where the cup stood, and the cup decided how many ways the arm could reach it. Nothing in
the test suite noticed, because nothing pins the order in which `build_scene` calls anything.

So the first test here reads the script's syntax tree rather than running it -- the same approach
`tests/unit/test_acceptance_script_scope.py` takes, and for the same reason: importing the M1 script
needs a spec, a plan and an asset environment, and the question "is this wired" is answerable
without any of them.

The rest are the defects the reordering exposed, each of which was invisible until the task's floor
started being reserved.
"""

from __future__ import annotations

import ast
import importlib
from rfw_w.scenes.resources import module_path, data_path
import rfw_w.scenes.application as application_module
import rfw_w.scenes.assembly.placement as placement_module
import rfw_w.scenes.delivery.compose as compose_module
import rfw_w.scenes.delivery.evidence as evidence_module
import rfw_w.scenes.runtime as runtime_module
import rfw_w.scenes.solvers.layout as layout_module
import rfw_w.scenes.solvers.shell_providers as shell_providers_module
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.furnish import (
    FurnitureRequest,
    Orientation,
    PlacementKind,
    room_walls,
)
from rfw_w.scenes.solvers.layout import FLOOR_KEEPOUT_PREFIX, Footprint

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE = module_path('rfw_w.scenes.assembly.placement')


@pytest.fixture(scope="module")
def acceptance():
    return importlib.import_module('rfw_w.scenes.application')


class TestTheBuildSolvesTheTaskBeforeTheRoom:
    def test_build_scene_fits_the_room_around_the_cell(self) -> None:
        """The reversal, pinned where it can actually regress.

        `build_scene` has to call `fit_room_around_cell`, and it has to call it with the cell it
        solved -- a room fitted around something else is not the claim. Read as source because the
        alternative is an end-to-end run, and an ordering that only fails under a GPU is an ordering
        that fails late.
        """
        tree = ast.parse(ACCEPTANCE.read_text())
        build_scene = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_scene"
        )
        calls = {
            node.func.id
            for node in ast.walk(build_scene)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "fit_room_around_cell" in calls
        assert "solve_cell" in calls
        # And not the room-only derivation: a build that still draws its shell from
        # `generate_options` has the fit in it but not in the path.
        assert "generate_options" not in calls

    def test_the_fit_is_given_the_cell_it_just_solved(self) -> None:
        """`fit_room_around_cell(spec, cell, ...)` in that argument order, and not a rebuilt one."""
        tree = ast.parse(ACCEPTANCE.read_text())
        build_scene = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_scene"
        )
        fits = [
            node for node in ast.walk(build_scene)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "fit_room_around_cell"
        ]
        assert len(fits) == 1
        args = fits[0].args
        assert isinstance(args[1], ast.Name) and args[1].id == "cell"


class TestTheSceneVersionDoesNotMoveWhenTheSceneDictGrows:
    def test_the_version_payload_names_its_keys_rather_than_hashing_the_build(self, acceptance) -> None:
        """Adding a key to `build_scene`'s return must not renumber every scene version.

        `build_scene` now hands back `room_plan` beside the room (SPEC-004, so that
        `shell.matches_plan` has something to compare against). The version is a hash of an
        **explicit** payload -- `scene_ir` and `plan` -- and not of the scene mapping, which is the
        only reason that addition left every existing version alone. A reader who widened this to
        `content_hash(scene)` would change every version in the ledger, and the evidence recorded
        against them would go stale without a single gate failing.

        Pinned by reading the key set, because the alternative is two eight-minute builds differing
        by one line.
        """
        tree = ast.parse(module_path("rfw_w.scenes.application").read_text())
        # Walked rather than scanned at module level: this one is a method of the host class, and
        # the sibling test above reads `build_scene` off `tree.body` only because that one is a
        # module-level function.
        payload = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_version_payload"
        )
        returned = next(node for node in ast.walk(payload) if isinstance(node, ast.Return))
        keys = {key.value for key in returned.value.keys}
        assert keys == {"scene_ir", "plan"}
        assert "room_plan" not in keys


class TestARugRespectsTheFloorReservations:
    """`FLOOR_COVERING` took the room's interior point and consulted no occupancy at all.

    Invisible while everything stood against the walls. Not invisible once the task's workspace was
    reserved as a floor keepout: a 2.44 x 1.52 m rug was laid across the desk *and* the floor the
    arm stands on, and `function.task_reachable` then found no base position -- on a scene where
    every other gate passed.
    """

    ROOM = Polygon(((0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0)))
    # A 2.44 x 1.52 m rug: large enough that the room has to be searched rather than assumed.
    KEEPOUT = Polygon.rectangle(3.0, 2.2, 5.0, 3.8)

    def _place(self, acceptance, *, keepout: Polygon | None):
        polygon = self.ROOM
        occupancy = layout_module.OccupancyMap()
        if keepout is not None:
            occupancy.add(f"{FLOOR_KEEPOUT_PREFIX}task_workspace", keepout)
        request = FurnitureRequest(
            entity_id="rug", synset="rug", kind=PlacementKind.FLOOR_COVERING,
            orientation=Orientation.SYMMETRIC,
        )
        return placement_module._furniture_placement(
            request, Footprint((1.22, 0.76)), 0.01,
            polygon=polygon, walls=room_walls(polygon), occupancy=occupancy,
            placed_by_entity={}, opening_spans={}, room_height_m=2.8,
            wall_thickness_m=0.1, plan=placement_module.build_plan(runtime_module.load_spec()), zone=None,
        )

    def test_a_free_room_still_gets_its_rug_at_the_interior_point(self, acceptance) -> None:
        """The old behaviour, kept: a room whose centre is free places its rug where it always did."""
        from rfw_w.scenes.geometry.polygon import interior_point

        placed = self._place(acceptance, keepout=None)
        assert placed is not None
        centre, _, yaw, _ = placed
        expected = interior_point(self.ROOM, margin=1.22 + 0.2)
        assert centre == pytest.approx(expected)
        assert yaw == 0.0

    def test_a_rug_is_not_laid_over_the_reserved_floor(self, acceptance) -> None:
        """The defect. The keepout is the task's own workspace, and the rug must go elsewhere."""
        from rfw_w.scenes.geometry.polygon import intersect_polygons

        from rfw_w.scenes.geometry.polygon import interior_point

        keepout = self.KEEPOUT
        # The interior point is *inside* the reservation, so the old code put the rug there and
        # this branch had nowhere else to look.
        assert keepout.area > 0
        blocked = Footprint((1.22, 0.76)).at(*interior_point(self.ROOM, margin=1.22 + 0.2))
        assert sum(p.area for p in intersect_polygons(blocked, keepout)) > 0.1

        placed = self._place(acceptance, keepout=keepout)
        assert placed is not None, "the room has free floor; the rug should have found it"
        centre, _, _, _ = placed
        footprint = Footprint((1.22, 0.76)).at(*centre)
        assert sum(p.area for p in intersect_polygons(footprint, keepout)) <= 1e-6

class TestThePeripheryPlacementOrder:
    """The order the furniture loop consumes the plan in, and what the loop does with a zone.

    The plan's order is the spec's entity order -- a statement about the *request*, not about the
    room -- and the loop was consuming it as a placement priority. Measured on the office, that put
    the rug (2.2 m, the largest and most constrained piece in the scene) fifth of eight, so every
    smaller and freer piece had already taken the floor it needed by the time it was tried. Measured
    on `--skip-isaac` deliveries, ordering largest-first took the three experiments from 16 unplaced
    items to 8: the kitchen's three plants and the bedroom's wardrobe and rug all came back.
    """

    @staticmethod
    def _request(entity_id: str, extent: float, kind: PlacementKind = PlacementKind.AGAINST_WALL):
        return FurnitureRequest(
            entity_id=entity_id, synset=entity_id, kind=kind,
            orientation=Orientation.FRONT_TO_ROOM, target_extent_m=extent,
        )

    def test_the_largest_piece_goes_first(self, acceptance) -> None:
        requests = [
            self._request("chair", 0.55), self._request("storage", 0.95),
            self._request("rug", 2.2, PlacementKind.FLOOR_COVERING),
        ]
        ordered = [r.entity_id for r in placement_module._most_constrained_first(requests)]
        assert ordered == ["rug", "storage", "chair"]

    def test_the_order_is_a_function_of_the_plan_not_of_the_file_order(self, acceptance) -> None:
        """Two specs naming the same entities in a different order must place them the same way.

        Without the tie-break this is the property that fails: equal extents would keep whatever
        order the JSON happened to have.
        """
        forward = [self._request("a", 0.8), self._request("b", 0.8), self._request("c", 0.8)]
        backward = list(reversed(forward))
        assert (
            [r.entity_id for r in placement_module._most_constrained_first(forward)]
            == [r.entity_id for r in placement_module._most_constrained_first(backward)]
        )

    def test_a_zone_that_cannot_hold_the_piece_is_detected(self, acceptance) -> None:
        """The capacity test, and the anti-vacuity half of it.

        Both directions matter: a zone too small for the footprint must be reported as unable to
        hold it (so the piece is retried on the room's floor), and a zone that *could* hold it must
        not be -- that case stays a refusal, which is what `_furniture_placement` documents.
        """
        from rfw_w.scenes.solvers.zones import Zone, ZoneKind
        from rfw_w.scenes.contracts.scene_spec import Density

        footprint = Footprint((0.25, 0.25))  # 0.5 x 0.5
        small = Zone(
            kind=ZoneKind.STORAGE, polygons=(Polygon.rectangle(0.0, 0.0, 0.4, 0.4),),
            density=Density.MEDIUM,
        )
        roomy = Zone(
            kind=ZoneKind.STORAGE, polygons=(Polygon.rectangle(0.0, 0.0, 3.0, 1.0),),
            density=Density.MEDIUM,
        )
        assert not placement_module._zone_could_hold(small, footprint)
        assert placement_module._zone_could_hold(roomy, footprint)

    def test_the_capacity_test_reads_the_union_not_one_piece(self, acceptance) -> None:
        """A footprint straddling two touching pieces is inside the zone, and this is the same
        union the permission test uses. Piece-by-piece it would read as unplaceable, and the piece
        would be retried on the room's floor for no reason."""
        from rfw_w.scenes.solvers.zones import Zone, ZoneKind
        from rfw_w.scenes.contracts.scene_spec import Density

        split = Zone(
            kind=ZoneKind.STORAGE,
            polygons=(
                Polygon.rectangle(0.0, 0.0, 1.0, 1.0),
                Polygon.rectangle(1.0, 0.0, 2.0, 1.0),
            ),
            density=Density.MEDIUM,
        )
        assert placement_module._zone_could_hold(split, Footprint((0.25, 0.25)))

    def test_the_drop_message_names_the_arena_that_was_searched(self, acceptance) -> None:
        """A rug is not confined to a zone, so its failure must not claim one.

        Measured on the before-delivery: the office's three rugs were reported as having "no position
        in the ambient zone" while `FLOOR_COVERING` never consulted that zone -- a constraint named
        in the evidence that was never in effect.
        """
        from rfw_w.scenes.contracts.scene_spec import ZoneKind

        assert placement_module._arena_name(PlacementKind.FLOOR_COVERING, ZoneKind.AMBIENT) == "the room"
        assert placement_module._arena_name(PlacementKind.CEILING, ZoneKind.AMBIENT) == "the room"
        # A piece the zone really does confine still names it.
        assert placement_module._arena_name(PlacementKind.AGAINST_WALL, ZoneKind.STORAGE) == "storage zone"
        # ... and an unzoned piece falls back to the room, which is what it searched.
        assert placement_module._arena_name(PlacementKind.AGAINST_WALL, None) == "the room"


def test_rollback_restores_only_the_current_plan_shell_export(acceptance, tmp_path):
    from types import SimpleNamespace
    host = application_module.SceneHost.__new__(application_module.SceneHost)
    host._backend_by_version = {}
    host._critique_by_version = {}
    exported = tmp_path / 'shell.usdc'
    host._shell_exports = {'infinigen:plan_a': exported}
    host.scene = {'room_plan': SimpleNamespace(content_hash='plan_a')}
    host._restore_observations('version_a')
    assert host.scene['shell_uri'] == str(exported)
    # A rebuilt room must never inherit another plan's shell, even if a stale URI survived.
    host.scene['room_plan'] = SimpleNamespace(content_hash='plan_b')
    host._restore_observations('version_b')
    assert 'shell_uri' not in host.scene
    host._shell_exports['infinigen:plan_b'] = ValueError('failed export')
    host._restore_observations('version_b')
    assert 'shell_uri' not in host.scene


class TestAViewThatMissedItsDeclarationIsNamed:
    """SPEC-015 R3/R4, the acceptance half: the gap is the *view's*, and the delivery continues."""

    def _report(self, **overrides):
        capture = {"camera_id": "overview_top", "nonblank": True, "depth_path": "d.npy",
                   "depth_valid_fraction": 0.9, "target_visible": None}
        payload = {"captures": (capture,),
                   "view_coverage": ({"camera_id": "overview_top", "visible_instances": 9,
                                      "minimum_visible_instances": 1, "obligation_met": True},),
                   "view_gaps": ()}
        payload.update(overrides)
        return payload

    def test_a_view_that_met_its_declaration_satisfies_the_gate(self, acceptance):
        ok, gaps = evidence_module.views_satisfied(self._report())
        assert ok is True and gaps == ()

    def test_a_view_that_missed_it_fails_the_gate_and_names_itself(self, acceptance):
        report = self._report(
            view_coverage=({"camera_id": "overview_top", "visible_instances": 0,
                            "minimum_visible_instances": 1, "obligation_met": False},),
            view_gaps=("VIEW_EMPTY(overview_top)",),
        )
        ok, gaps = evidence_module.views_satisfied(report)
        assert ok is False
        assert gaps == ("VIEW_EMPTY(overview_top)",)

    def test_an_unusable_view_still_fails_it(self, acceptance):
        report = self._report(captures=({"camera_id": "overview_top", "nonblank": False},))
        ok, _ = evidence_module.views_satisfied(report)
        assert ok is False


class TestTheManifestDoesNotInventADoor:
    """SPEC-015 R5, the derived half: `door_cut` is null when nothing measured it."""

    def test_a_failed_report_yields_no_claim(self, acceptance):
        assert evidence_module.door_cut_from_report({"status": "failed"}) is None
        assert evidence_module.door_cut_from_report({"openings_cut": None}) is None

    def test_a_measured_report_yields_its_measurement(self, acceptance):
        assert evidence_module.door_cut_from_report({"openings_cut": 4}) is True
        assert evidence_module.door_cut_from_report({"openings_cut": 0}) is False


class TestTheManifestResolvesTheShellReferenceAgain:
    """SPEC-005's shell reference, at the point where it used to go missing.

    A rollback rebuilds the scene and a rebuilt scene carries no `shell_uri`, so the manifest's
    write read an empty dict entry and published nothing -- silently. The backend path had been
    fixed; the manifest's copy had not, and enabling the Visual Critic made every failed verdict a
    rollback. Measured on the 2026-09-24 backfill: three deliveries rendered against the referenced
    shell, gate evidence saying so, and no `shell_reference.json`. Only the independent verifier
    noticed.
    """

    class _Host:
        def __init__(self, *, provider="infinigen", restored=None, raises=None):
            self.args = SimpleNamespace(shell_provider=provider)
            self.scene = {}
            self._restored = restored
            self._raises = raises
            self.prepared = 0

        def prepare_shell_reference(self):
            self.prepared += 1
            if self._raises is not None:
                raise self._raises
            if self._restored:
                self.scene["shell_uri"] = self._restored

    def test_a_rebuilt_scene_gets_its_uri_back_from_the_cached_export(self, acceptance) -> None:
        host = self._Host(restored="/tmp/shell.usdc")
        assert compose_module.resolve_shell_uri(host, host.scene) == "/tmp/shell.usdc"
        assert host.prepared == 1

    def test_an_unresolvable_reference_returns_nothing_rather_than_raising(self, acceptance) -> None:
        """The backend has already recorded `backend.isaac.load` FAIL for this run."""
        host = self._Host(raises=shell_providers_module.ShellProviderError("EXPORT_BACKEND_UNAVAILABLE(no shell)"))
        assert compose_module.resolve_shell_uri(host, host.scene) is None

    def test_a_builtin_run_never_exports_a_shell_it_does_not_reference(self, acceptance) -> None:
        host = self._Host(provider="builtin")
        assert compose_module.resolve_shell_uri(host, host.scene) is None
        assert host.prepared == 0

    def test_the_scene_dict_is_not_edited_by_resolving(self, acceptance) -> None:
        """The caller writes the file from the returned value; mutating the scene here would make
        the resolution invisible to a reader of the call site."""
        host = self._Host(restored="/tmp/shell.usdc")
        scene: dict = {}
        compose_module.resolve_shell_uri(host, scene)
        assert scene == {}
