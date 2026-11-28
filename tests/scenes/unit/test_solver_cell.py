# SPDX-License-Identifier: Apache-2.0
"""The task cell is solved before, and independently of, the room shell."""

from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import inspect
import json
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.geometry.polygon import contains_point
from rfw_w.scenes.solvers.cell import CellError, solve_cell
from rfw_w.scenes.solvers.room import RoomError, fit_room_around_cell

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ("office_pick_cup", "kitchen_fetch_jar", "bedroom_pick_alarm_clock")


def load_spec(name: str) -> SceneSpec:
    return SceneSpec.from_dict(json.loads((data_path('examples') / name / 'task.json').read_text()))


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_all_experiments_produce_a_reachable_local_cell(name: str) -> None:
    spec = load_spec(name)
    cell = solve_cell(spec, seed=spec.seed)
    assert cell.support_id == next(entity.id for entity in spec.entities if entity.role.value == "support")
    assert cell.target_id == spec.task.targets[0]
    assert contains_point(cell.support_footprint, cell.target_offset)
    assert cell.standoffs
    assert cell.protected.area >= cell.support_footprint.area


def test_cell_solve_has_no_room_input() -> None:
    """The core is a function of the spec and, at most, a measured asset surface.

    The parameter list is pinned so that adding one is a decision rather than a habit -- but what
    the pin protects is *no room, no shell, nothing spatial*, not "this signature never grows". A
    measured support surface is an asset fact, not a room fact, and it enters as an optional
    keyword so that not passing it stays the default.
    """
    parameters = inspect.signature(solve_cell).parameters
    assert tuple(parameters) == ("spec", "seed", "support_measurement")
    measured = parameters["support_measurement"]
    assert measured.kind is inspect.Parameter.KEYWORD_ONLY
    assert measured.default is None
    for name, parameter in parameters.items():
        annotation = str(parameter.annotation)
        assert "Room" not in annotation and "Shell" not in annotation, name


def support_measurement(**overrides):
    """A measured support, in the cell frame, with only the piece under test varying."""
    from rfw_w.scenes.contracts.asset import AffordanceStatus, MeasuredSupport, Provenance

    fields = {
        "asset_id": "00f5afbca9f9bb5eadf718e81a66f0973ed67898",
        "source_polygon_id": "00f5afbca9f9bb5eadf718e81a66f0973ed67898:surface0",
        "vertices_xy": ((-0.6455, 0.0), (0.6455, 0.0), (0.6455, 0.606), (-0.6455, 0.606)),
        "support_height_m": 0.762,
        "provenance": Provenance(
            method="hssd_precomputed_support_surface_obb",
            version="pvscene.hssd-prepared/v1",
            status=AffordanceStatus.ANNOTATED,
        ),
    }
    fields.update(overrides)
    return MeasuredSupport(**fields)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_same_spec_and_seed_have_a_byte_stable_core_hash(name: str) -> None:
    spec = load_spec(name)
    assert solve_cell(spec, seed=spec.seed).core_hash == solve_cell(spec, seed=spec.seed).core_hash


@pytest.mark.parametrize("measured", [False, True])
def test_room_dimensions_are_not_part_of_the_core_hash(measured: bool) -> None:
    """Room intent must not reach the core with an estimate *or* with a real surface.

    Checked both ways because the measurement is the one new input to the solve: if a real support
    surface let room dimensions leak back in, the whole ordering would be undone quietly, and the
    estimate-only version of this test would still pass.
    """
    kwargs = {"support_measurement": support_measurement()} if measured else {}
    spec = load_spec("office_pick_cup")
    payload = spec.to_dict()
    payload["room"]["size_bounds_m"] = [[8.0, 9.0], [7.0, 8.0], [3.0, 3.2]]
    larger_room_intent = SceneSpec.from_dict(payload)
    assert (
        solve_cell(spec, seed=11, **kwargs).core_hash
        == solve_cell(larger_room_intent, seed=11, **kwargs).core_hash
    )


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_room_fit_contains_the_protected_cell(name: str) -> None:
    spec = load_spec(name)
    cell = solve_cell(spec, seed=spec.seed)
    fitted = fit_room_around_cell(spec, cell, room_seed=101)
    from rfw_w.scenes.geometry.polygon import Polygon, contains_polygon, inset

    air = inset(Polygon(fitted.shell.polygon_xy), fitted.options.wall_thickness_m)
    assert contains_polygon(air, fitted.protected_in_room)
    from rfw_w.scenes.geometry.polygon import overlaps
    from rfw_w.scenes.solvers.zones import door_sweep_for

    assert not any(overlaps(piece, fitted.protected_in_room)
                   for piece in door_sweep_for(fitted.shell))


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_built_support_lands_where_the_cell_says_it_does(name: str) -> None:
    """The two frames are not the same one, and this is the conversion between them.

    The cell puts the support's **front edge on y = 0** with the body toward +y, which is the frame
    the target offset and the standoffs are solved in. The mesh `assets/procedural_support.py` writes
    is **centred on its own footprint**, because that is the convention the retrieved assets it
    stands in for use. So a support built to this cell's dimensions sits half a depth away from
    where the cell says its origin is, and a build that placed it at the translation alone would put
    the task target `depth/2` behind the surface it was solved on -- with nothing downstream
    measuring that.

    Checked through the same property the build calls, so the test fails when the build's arithmetic
    would.
    """
    from rfw_w.scenes.geometry.polygon import Polygon
    from rfw_w.scenes.solvers.support_shape import SupportShape, top_surface

    spec = load_spec(name)
    cell = solve_cell(spec, seed=spec.seed)
    fitted = fit_room_around_cell(spec, cell, room_seed=spec.seed)

    min_x, min_y, max_x, max_y = cell.support_footprint.bounds()
    offset = cell.support_centre_offset_m
    assert offset == pytest.approx(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0))

    # The top slab the generator writes, in its own centred frame, translated to where the build
    # puts it: the cell frame's rectangle moved by the translation *plus* the frame offset.
    polygon, _ = top_surface(
        SupportShape(cell.support_shape),
        width_m=max_x - min_x, depth_m=max_y - min_y, height_m=cell.support_height_m,
    )
    world = Polygon(tuple(
        (x + fitted.cell_translation_xy[0] + offset[0], y + fitted.cell_translation_xy[1] + offset[1])
        for x, y in polygon.vertices
    ))
    expected = Polygon(tuple(
        (x + fitted.cell_translation_xy[0], y + fitted.cell_translation_xy[1])
        for x, y in cell.support_footprint.vertices
    ))
    assert world.bounds() == pytest.approx(expected.bounds())
    assert world.area == pytest.approx(expected.area, abs=1e-9)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_fit_keeps_the_placement_that_leaves_the_most_usable_floor(name: str) -> None:
    """Holding the cell is necessary and not sufficient.

    Where the cell is decides whether the rest of the room is one region or a scatter of slivers.
    Measured on `office_pick_cup` seed 11, the admissible placements span 5.93 to 7.47 m2 of
    placeable floor and the seating zone ranges from 0.11 m2 -- no chair fits -- to 0.72 m2. The fit
    used to return the *first* admissible candidate, which was that 0.11 m2 end, and the run then
    failed its own randomization closure because a chair named by a soft relation could not be
    placed.

    So this recomputes every admissible placement and requires the chosen one to be the best of
    them. It is the property, not a stored number: it holds for any seed and any arm.
    """
    from rfw_w.scenes.geometry.polygon import Polygon, inset
    from rfw_w.scenes.solvers.room import _fit_candidates
    from rfw_w.scenes.solvers.zones import door_sweep_for, plan_zones

    spec = load_spec(name)
    cell = solve_cell(spec, seed=spec.seed)
    fitted = fit_room_around_cell(spec, cell, room_seed=spec.seed)
    available = inset(Polygon(fitted.shell.polygon_xy), fitted.options.wall_thickness_m)
    candidates = _fit_candidates(
        cell.protected, available,
        margin_m=fitted.wall_standoff_m, blockers=door_sweep_for(fitted.shell),
    )
    assert candidates
    assert fitted.cell_translation_xy in {placement for placement, _ in candidates}

    chosen = plan_zones(
        fitted.shell, spec.style, cell_protected=fitted.protected_in_room
    ).placeable_area_m2
    for placement, moved in candidates:
        try:
            other = plan_zones(fitted.shell, spec.style, cell_protected=moved).placeable_area_m2
        except Exception:  # a placement the zone plan refuses is not a placement
            continue
        assert other <= chosen + 1e-9, (placement, other, chosen)


@pytest.mark.parametrize("measured", [False, True])
def test_twenty_room_seeds_do_not_change_the_core_hash(measured: bool) -> None:
    spec = load_spec("office_pick_cup")
    kwargs = {"support_measurement": support_measurement()} if measured else {}
    cell = solve_cell(spec, seed=spec.seed, **kwargs)
    core_hashes = set()
    shell_hashes = set()
    for room_seed in range(20):
        fitted = fit_room_around_cell(spec, cell, room_seed=room_seed)
        core_hashes.add(cell.core_hash)
        shell_hashes.add(tuple(fitted.shell.polygon_xy))
    assert len(core_hashes) == 1
    assert len(shell_hashes) > 1


def test_a_cell_that_leaves_no_room_for_the_door_sweep_is_rejected() -> None:
    from types import SimpleNamespace
    from rfw_w.scenes.geometry.polygon import Polygon

    spec = load_spec("office_pick_cup")
    payload = spec.to_dict()
    payload["room"]["size_bounds_m"] = [[4.0, 4.0], [3.0, 3.0], [2.8, 2.8]]
    exact_room = SceneSpec.from_dict(payload)
    # It fits the inner face with the requested 0.2 m standoff, but occupies every remaining
    # strip of floor, so any inward-opening door must sweep through it.
    fills_room = SimpleNamespace(protected=Polygon.rectangle(0.0, 0.0, 3.2, 2.2))
    with pytest.raises(RoomError, match="DOOR_SWEEP_CROSSES_TASK_CELL"):
        fit_room_around_cell(exact_room, fills_room, room_seed=0)


def test_room_too_small_for_the_cell_fails_before_placement() -> None:
    spec = load_spec("office_pick_cup")
    cell = solve_cell(spec, seed=spec.seed)
    payload = spec.to_dict()
    payload["room"]["size_bounds_m"] = [[1.0, 1.1], [1.0, 1.1], [2.6, 3.0]]
    too_small = SceneSpec.from_dict(payload)
    with pytest.raises(RoomError, match="ROOM_TOO_SMALL_FOR_TASK_CELL"):
        fit_room_around_cell(too_small, cell, room_seed=0)


# The core hashes of the constructed path. They are literals rather than recomputed on purpose: a
# test that recomputes the value it guards cannot notice that the value moved.
#
# These moved once, deliberately, when the support stopped being an estimate and became a
# construction spec -- the method string, the status and the shape all entered the payload. The
# previous three values are no longer recorded anywhere reachable (the run directories that held
# them were cleared 2026-09-22); what remains of that diff is the commit that made it. They are
# literals anyway, so the point holds: a move is a decision someone has to write down here.
CONSTRUCTED_CORE_HASHES = {
    "office_pick_cup": "979875297fcd621734151a7b8daf9507c735d5c8bb7b148e2a85940fa6b9c4a1",
    # Solid island base replaces the visually table-like two-tier frame; workspace is unchanged.
    "kitchen_fetch_jar": "199072cbe681999d4725f2c3ec07d02fe8f6141055f695f19756e4ead8c20fa5",
    "bedroom_pick_alarm_clock": "dbde1d697ed905f6d695463eb194fcf2ce24acf7d2f77babd8395cfc0fc6a7a4",
}

CONSTRUCTED_KEYS = (
    "approach_arc_deg", "local_frame", "protected", "standoffs", "support_footprint",
    "support_geometry_method", "support_geometry_status", "support_height_m", "support_id",
    "support_shape", "target_id", "target_offset",
)

# The form each experiment's support is built to, from its category profile.
EXPECTED_SHAPES = {
    "office_pick_cup": "slab_legs",
    "kitchen_fetch_jar": "plinth",
    "bedroom_pick_alarm_clock": "pedestal",
}


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_constructed_path_is_pinned(name: str) -> None:
    """The core names the support it is built from, and the payload is frozen.

    Both halves matter: the hash covers the payload, and the key set catches a field that was added
    without anyone deciding to add it, which a hash comparison alone would not name.
    """
    spec = load_spec(name)
    cell = solve_cell(spec, seed=spec.seed)
    assert cell.core_hash == CONSTRUCTED_CORE_HASHES[name]
    assert tuple(sorted(cell.to_dict())) == CONSTRUCTED_KEYS
    assert cell.support_shape == EXPECTED_SHAPES[name]
    assert cell.support_geometry_method == f"constructed:{cell.support_shape}"
    assert cell.support_geometry_status == "constructed"
    assert cell.support_source_polygon_id is None
    assert cell.to_dict() == solve_cell(spec, seed=spec.seed, support_measurement=None).to_dict()


def test_a_measured_cell_names_the_asset_annotation() -> None:
    """A reader of the core must be able to tell a measurement from an estimate without asking.

    The method string, the status and the surface's own id all travel with the cell, because a
    delivery that protects a real desk's surface and one that protects a category average are not
    the same claim, and the manifest is where that difference has to be legible.
    """
    spec = load_spec("office_pick_cup")
    measurement = support_measurement()
    measured = solve_cell(spec, seed=spec.seed, support_measurement=measurement)
    estimated = solve_cell(spec, seed=spec.seed)

    assert measured.support_geometry_method == (
        "asset_support_polygon:hssd_precomputed_support_surface_obb"
    )
    assert measured.support_geometry_status == "annotated"
    assert measured.support_source_polygon_id == measurement.source_polygon_id
    assert measured.support_height_m == 0.762
    assert measured.support_footprint.area == pytest.approx(1.291 * 0.606, abs=1e-6)
    assert measured.core_hash != estimated.core_hash
    assert "support_geometry_status" in measured.to_dict()


def test_a_measured_cell_keeps_the_core_invariants() -> None:
    """What the measured surface changes is the numbers, not the shape of the solve."""
    spec = load_spec("office_pick_cup")
    cell = solve_cell(spec, seed=spec.seed, support_measurement=support_measurement())
    assert contains_point(cell.support_footprint, cell.target_offset)
    assert cell.standoffs
    assert cell.protected.area >= cell.support_footprint.area
    assert cell.approach_arc_deg[0] <= cell.approach_arc_deg[1]
    # The cell frame the measurement contracts for: front edge at 0, body toward +y.
    min_x, min_y, max_x, max_y = cell.support_footprint.bounds()
    assert (min_x, min_y, max_y) == pytest.approx((-0.6455, 0.0, 0.606), abs=1e-9)
    assert max_x == pytest.approx(0.6455, abs=1e-9)


def test_an_estimated_measurement_is_refused() -> None:
    """Injecting an estimate as a measurement must not be silently equivalent to measuring."""
    from rfw_w.scenes.contracts.asset import AffordanceStatus, Provenance

    spec = load_spec("office_pick_cup")
    estimated = support_measurement(provenance=Provenance(
        method="hssd_precomputed_support_surface_obb",
        version="pvscene.hssd-prepared/v1",
        status=AffordanceStatus.ESTIMATED,
    ))
    with pytest.raises(CellError, match="CORE_SUPPORT_MEASUREMENT_NOT_USABLE"):
        solve_cell(spec, seed=spec.seed, support_measurement=estimated)


def test_a_measurement_not_in_the_cell_frame_is_refused() -> None:
    """Asset-frame vertices would move the protected region rather than fail; so they fail."""
    spec = load_spec("office_pick_cup")
    shifted = support_measurement(
        vertices_xy=((0.0, 0.05), (1.291, 0.05), (1.291, 0.656), (0.0, 0.656)),
    )
    with pytest.raises(CellError, match="CORE_SUPPORT_MEASUREMENT_FRAME"):
        solve_cell(spec, seed=spec.seed, support_measurement=shifted)


def test_a_measured_surface_too_small_for_the_target_fails_by_name() -> None:
    """The existing placement failure, reached with a measured surface -- no new code invented."""
    spec = load_spec("office_pick_cup")
    tiny = support_measurement(
        vertices_xy=((-0.025, 0.0), (0.025, 0.0), (0.025, 0.05), (-0.025, 0.05)),
    )
    with pytest.raises(CellError, match="CORE_TARGET_UNREACHABLE"):
        solve_cell(spec, seed=spec.seed, support_measurement=tiny)


@pytest.mark.parametrize("name", ["office_pick_cup", "kitchen_fetch_jar", "bedroom_pick_alarm_clock"])
def test_compiler_target_core_is_byte_stable_across_room_draws(name):
    import json
    spec = load_spec(name)
    cell = solve_cell(spec, seed=spec.seed)
    frozen = json.dumps(cell.to_dict(), sort_keys=True)
    shells = set()
    for room_seed in range(8):
        fitted = fit_room_around_cell(spec, cell, room_seed=room_seed)
        assert json.dumps(cell.to_dict(), sort_keys=True) == frozen
        shells.add(tuple(fitted.shell.polygon_xy))
    assert len(shells) > 1
