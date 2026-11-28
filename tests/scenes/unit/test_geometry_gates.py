# SPDX-License-Identifier: Apache-2.0
"""The two geometry gates must be computed, and must be able to fail.

Both used to be unconditional `PASS` records with `measured=0.0` and a fixed detail string, on
gates that every level above `draft` requires. Their details said so themselves -- "the desk and
cup footprints share no area", "both footprints lie inside the 6-vertex room outline" -- while
twelve instances were placed, and one of those twelve reached 0.13 m through a wall.

So the tests here are mostly *negative*. The manual's rule is that a gate which cannot fail is the
defect, and only a failing case proves it is gone. The positive cases that follow are not filler
either: the first attempt at a penetration test flagged seven pairs on a correct room, five of them
legitimate, so "does not fire on a normal room" is a real property to pin down.

No assets, no GPU, no Isaac: the gates read only the room outline and the placed poses.
"""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import rfw_w.scenes.assembly.geometry as geometry_module
import rfw_w.scenes.assembly.placement as placement_module
import rfw_w.scenes.runtime as runtime_module
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.units import Vec3
from rfw_w.scenes.geometry.polygon import Polygon

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE = module_path('rfw_w.scenes.application')

# The delivered room: 6 vertices with the notch at the north-east, 2.9697 m to the ceiling.
ROOM = Polygon(((0.0, 0.0), (4.4524, 0.0), (4.4524, 2.3848), (3.3394, 2.3848),
                (3.3394, 3.5598), (0.0, 3.5598)))
ROOM_HEIGHT_M = 2.9697


@pytest.fixture(scope="module")
def acceptance():
    return importlib.import_module('rfw_w.scenes.application')


@dataclass
class Placed:
    """The fields the gates read off a placed instance, and nothing else."""

    instance_id: str
    dimensions_m: Vec3
    position_m: Vec3
    yaw_rad: float = 0.0
    rotation_wxyz: tuple[float, float, float, float] | None = None
    origin_at_base: bool = True


class _Shell:
    floor_z_m = 0.0
    height_m = ROOM_HEIGHT_M


def _yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


# The wall thickness the delivered room uses. The gates read it because the room polygon is the
# wall's outer face, so "inside the room" means inside the outline inset by this much.
WALL_THICKNESS_M = 0.1


def _scene(placed) -> dict:
    return {
        "room_polygon": ROOM,
        "shell": _Shell(),
        "instances": list(placed),
        # The gates read the floor-covering exemption off the plan, so a scene without one is not a
        # scene these gates can judge.
        "plan": placement_module.build_plan(runtime_module.load_spec()),
    }


def _bounds(acceptance, placed):
    return geometry_module._geometry_in_bounds(_scene(placed), wall_thickness_m=WALL_THICKNESS_M)


# A desk against the west wall's *inner face*: 1.37 x 0.76 x 0.75, yawed a quarter turn so its
# length runs along the wall. Its back sits at x = 0.1, which is where the wall's visible surface
# is -- the outline is the wall's outer face, so a desk flush to the outline is 0.1 m inside the
# wall solid. That difference is the whole reason this gate measures the inner face.
DESK = Placed("desk", Vec3(1.37, 0.76, 0.75), Vec3(0.4802, 2.6747, 0.0),
              yaw_rad=-math.pi / 2, rotation_wxyz=_yaw(-math.pi / 2))

# The same desk as the pipeline placed it before the loop existed: flush to the outline.
DESK_FLUSH_TO_OUTLINE = Placed("desk", Vec3(1.37, 0.76, 0.75), Vec3(0.3802, 2.6747, 0.0),
                               yaw_rad=-math.pi / 2, rotation_wxyz=_yaw(-math.pi / 2))


class TestInBoundsCatchesRealEscapes:
    """The two escapes that were actually delivered, at the poses that were actually recorded."""

    def test_a_panel_through_the_wall_fails(self, acceptance) -> None:
        """`wall_art` on m1_004: 0.70 x 1.00 x 0.005 lying flat, centre 4.0824, wall at 4.4524."""
        flat = Placed("wall_art", Vec3(0.70, 1.00, 0.005), Vec3(4.0824, 0.75, 1.4525),
                      yaw_rad=math.pi / 2, rotation_wxyz=_yaw(math.pi / 2))
        ok, measured, detail, witnesses = _bounds(acceptance, [flat])
        assert not ok
        # 0.1300 m past the outline, which is 0.2300 m past the wall's inner face at 4.3524.
        assert measured == pytest.approx(0.2300, abs=1e-3)
        assert "wall_art" in detail
        assert witnesses == ("instance:wall_art:outside-room",)

    def test_a_pendant_through_the_ceiling_fails(self, acceptance) -> None:
        """`ceiling_light` on m1_004: origin read as the centre, so 1.67 m grew upwards."""
        pendant = Placed("ceiling_light", Vec3(0.2578, 0.2578, 1.6692), Vec3(2.076, 1.673, 2.0851))
        ok, measured, detail, _ = _bounds(acceptance, [pendant])
        assert not ok
        assert measured == pytest.approx(0.7846, abs=1e-3)
        assert "ceiling_light" in detail

    def test_a_piece_under_the_floor_fails(self, acceptance) -> None:
        """A base-origin piece placed 0.30 m below the floor it should stand on."""
        sunken = Placed("desk", Vec3(1.37, 0.76, 0.75), Vec3(1.0, 1.0, -0.30), origin_at_base=True)
        ok, measured, _detail, _ = _bounds(acceptance, [sunken])
        assert not ok
        # It is also 0.1 m inside the west wall, and the gate reports the worse of the two.
        assert measured == pytest.approx(0.30, abs=1e-6)

    def test_the_height_range_follows_the_origin_convention(self, acceptance) -> None:
        """The same numbers under the two conventions differ by half the height, and the gate
        must read the recorded one rather than assume. A centre-origin desk at z=-0.30 reaches
        0.675 m below the floor, not 0.30 m, because its own box is centred on its origin."""
        centred = Placed("desk", Vec3(1.37, 0.76, 0.75), Vec3(1.0, 1.0, -0.30), origin_at_base=False)
        ok, measured, _detail, _ = _bounds(acceptance, [centred])
        assert not ok
        assert measured == pytest.approx(0.675, abs=1e-6)

    def test_a_piece_flush_to_the_outline_is_inside_the_wall(self, acceptance) -> None:
        """The defect the gate could not see while it measured the outline.

        This is the same desk the pipeline actually placed, at the same pose: back flush to the
        outline at x = 0, which is one wall thickness inside the wall's visible face. The gate used
        to pass it, and no render could tell the difference -- a desk flush to the wall and a desk
        sunk into it look identical from every camera inside the room.
        """
        ok, measured, detail, witnesses = _bounds(acceptance, [DESK_FLUSH_TO_OUTLINE])
        assert not ok
        assert measured == pytest.approx(WALL_THICKNESS_M, abs=1e-3)
        assert "desk" in detail
        assert witnesses == ("instance:desk:inside-wall",)

    def test_the_same_desk_against_the_wall_face_passes(self, acceptance) -> None:
        """The negative case that proves the gate is measuring the wall and not the desk."""
        ok, measured, detail, _ = _bounds(acceptance, [DESK])
        assert ok, detail
        assert measured == 0.0


class TestInBoundsPassesACorrectRoom:
    def test_the_fixed_poses_pass(self, acceptance) -> None:
        """The same three pieces at the poses this change produces."""
        fixed = [
            DESK,
            # Hung on the east wall: back 0.02 m off the wall's inner face, 1.00 m tall, spanning
            # 0.95..1.95, and its 5 mm thickness running into the room.
            Placed("wall_art", Vec3(0.70, 1.00, 0.005), Vec3(4.3274, 1.50, 1.45),
                   rotation_wxyz=(0.5, 0.5, -0.5, -0.5)),
            Placed("ceiling_light", Vec3(0.2578, 0.2578, 1.6692), Vec3(2.076, 1.673, 1.2505)),
        ]
        ok, measured, detail, witnesses = _bounds(acceptance, fixed)
        assert ok, detail
        assert measured == 0.0
        assert len(witnesses) == len(fixed), "a passing gate must name everything it looked at"

    def test_the_gate_looks_at_every_instance_not_just_the_named_ones(self, acceptance) -> None:
        """The escape is on a piece the old detail never mentioned."""
        ok, _, detail, _ = _bounds(
            acceptance, [DESK, Placed("plant", Vec3(0.24, 0.20, 0.61), Vec3(4.40, 3.50, 0.0))]
        )
        assert not ok
        assert "plant" in detail
        assert "desk" not in detail


class TestNoPenetrationFiresOnRealOverlap:
    def test_two_pieces_in_the_same_space_fail(self, acceptance) -> None:
        """A cabinet standing where the desk already is: overlapping in plan *and* in height."""
        blocker = Placed("storage", Vec3(0.933, 0.471, 1.079), Vec3(0.3802, 2.4, 0.0))
        ok, measured, detail, witnesses = geometry_module._geometry_no_penetration(
            _scene([DESK, blocker])
        )
        assert not ok
        assert measured > 0.0
        assert "desk" in detail and "storage" in detail
        assert witnesses == ("pair:desk|storage:overlap",)


class TestNoPenetrationPassesAnOrdinaryRoom:
    """Every one of these was a false positive in the first, plan-only version of the test."""

    def test_an_object_resting_on_its_support_passes(self, acceptance) -> None:
        """The cup's plan outline is inside the desk's; its base is 2 mm above the desk top."""
        cup = Placed("cup", Vec3(0.095, 0.095, 0.101), Vec3(0.1085, 2.0982, 0.8023),
                     origin_at_base=False)
        ok, measured, detail, _ = geometry_module._geometry_no_penetration(_scene([DESK, cup]))
        assert ok, detail
        assert measured == 0.0

    def test_furniture_standing_on_a_rug_passes(self, acceptance) -> None:
        """A rug is a floor covering; `FURNISHING_PLAN` says it is meant to lie under furniture."""
        rug = Placed("rug", Vec3(2.438, 1.524, 0.015), Vec3(2.076, 1.673, 0.006))
        chair = Placed("chair", Vec3(0.572, 0.623, 0.878), Vec3(1.65, 2.58, 0.0),
                       yaw_rad=1.4963, rotation_wxyz=_yaw(1.4963))
        ok, _measured, detail, _ = geometry_module._geometry_no_penetration(_scene([rug, chair]))
        assert ok, detail

    def test_a_pendant_above_a_rug_passes(self, acceptance) -> None:
        """Same plan footprint, disjoint heights."""
        rug = Placed("rug", Vec3(2.438, 1.524, 0.015), Vec3(2.076, 1.673, 0.006))
        pendant = Placed("ceiling_light", Vec3(0.2578, 0.2578, 1.6692), Vec3(2.076, 1.673, 1.2505))
        ok, _measured, detail, _ = geometry_module._geometry_no_penetration(_scene([rug, pendant]))
        assert ok, detail

    def test_a_panel_that_clears_the_furniture_passes(self, acceptance) -> None:
        """The fixed picture, with the lamp and plant it had to be moved away from."""
        lamp = Placed("floor_lamp", Vec3(0.236, 0.274, 1.121), Vec3(4.2251, 0.35, 0.0),
                      yaw_rad=1.0629, rotation_wxyz=_yaw(1.0629))
        plant = Placed("plant", Vec3(0.24, 0.20, 0.61), Vec3(4.2656, 0.75, 0.0),
                       yaw_rad=1.2004, rotation_wxyz=_yaw(1.2004))
        panel = Placed("wall_art", Vec3(0.70, 1.00, 0.005), Vec3(4.3274, 1.50, 1.45),
                       rotation_wxyz=(0.5, 0.5, -0.5, -0.5))
        ok, _measured, detail, _ = geometry_module._geometry_no_penetration(_scene([lamp, plant, panel]))
        assert ok, detail

    def test_the_same_panel_over_the_lamp_fails(self, acceptance) -> None:
        """The negative of the case above: hung where the lamp stands, it really does intersect.

        This is not hypothetical -- it is what happened when the mount was fixed and the panel was
        left on the first span the sweep tried, before the sweep learned to check what was already
        standing there. The lamp reaches 1.12 m and the panel's lower edge is at 0.95 m.
        """
        lamp = Placed("floor_lamp", Vec3(0.236, 0.274, 1.121), Vec3(4.2251, 0.35, 0.0),
                      yaw_rad=1.0629, rotation_wxyz=_yaw(1.0629))
        panel = Placed("wall_art", Vec3(0.70, 1.00, 0.005), Vec3(4.3274, 0.75, 1.45),
                       rotation_wxyz=(0.5, 0.5, -0.5, -0.5))
        ok, measured, detail, _ = geometry_module._geometry_no_penetration(_scene([lamp, panel]))
        assert not ok
        assert measured > 0.0
        assert "wall_art" in detail and "floor_lamp" in detail

class TestTaskCellClear:
    def scene(self, acceptance, name="chair", kind=None, x=2.0, z=0.0):
        from dataclasses import replace
        from rfw_w.scenes.solvers.furnish import FurnitureRequest, PlacementKind, Orientation
        placed = Placed(name, Vec3(0.5, 0.5, 0.1), Vec3(x, 2.0, z))
        scene = _scene([placed])
        scene.update(workspace=Polygon(((1, 1), (3, 1), (3, 3), (1, 3))),
                     support_id="desk", floor_occupants={name})
        scene["plan"] = replace(placement_module.build_plan(runtime_module.load_spec()), furniture=(
            FurnitureRequest(entity_id=name, synset="chair.n.01", kind=kind or PlacementKind.ANYWHERE, orientation=Orientation.FRONT_TO_ROOM),
        ))
        return scene

    def test_floor_intrusion_fails(self, acceptance):
        ok, area, detail, witnesses = geometry_module._geometry_task_cell_clear(self.scene(acceptance))
        assert not ok and area == pytest.approx(0.25) and witnesses

    def test_clear_periphery_passes(self, acceptance):
        assert geometry_module._geometry_task_cell_clear(self.scene(acceptance, x=4.0))[0]

    def test_floor_covering_cannot_bypass_floor_registry(self, acceptance):
        from rfw_w.scenes.solvers.furnish import PlacementKind
        scene = self.scene(acceptance, name="rug", kind=PlacementKind.FLOOR_COVERING)
        scene["floor_occupants"] = set()
        assert not geometry_module._geometry_task_cell_clear(scene)[0]

    def test_ceiling_projection_is_diagnostic(self, acceptance):
        from rfw_w.scenes.solvers.furnish import PlacementKind
        result = geometry_module._geometry_task_cell_clear(
            self.scene(acceptance, name="light", kind=PlacementKind.CEILING, z=2.8))
        assert result[0] and "air projection=0.250000" in result[2]

    def test_missing_workspace_cannot_pass(self, acceptance):
        scene = self.scene(acceptance)
        del scene["workspace"]
        assert not geometry_module._geometry_task_cell_clear(scene)[0]


class TestTheAssemblyAsksTheGateQuestionBeforeAdmittingAPiece:
    """SPEC-009's assembly step decides one candidate at a time, by the gate's own rule.

    A kept generated object is refused at assembly when it would share space with a piece the
    delivery already places. Measured on the first assembled delivery, one monitor and the
    retrieved floor lamp shared 0.0301 m2 and the run ended with `geometry.no_penetration` red --
    so the assembly asks *that* question, with the same tolerance and the same exemptions, rather
    than a second rule of its own. These tests pin the two ways it could disagree with the gate.
    """

    def _refusal(self, acceptance, placed, candidate):
        return geometry_module._assembly_refusal(
            _scene(placed), dimensions=candidate.dimensions_m, position=candidate.position_m,
            rotation_wxyz=candidate.rotation_wxyz or _yaw(candidate.yaw_rad),
            wall_thickness_m=WALL_THICKNESS_M)

    def test_a_generated_piece_where_the_desk_stands_is_refused_with_the_gates_own_area(self):
        acceptance = importlib.import_module('rfw_w.scenes.application')
        blocker = Placed("generated_monitor_1", Vec3(0.933, 0.471, 1.079), Vec3(0.6, 2.4, 0.0))
        refusal = self._refusal(acceptance, [DESK], blocker)
        # The gate's number for the same pair, to the last digit: one rule, two readers.
        ok, measured, _detail, _ = geometry_module._geometry_no_penetration(_scene([DESK, blocker]))
        assert not ok
        assert refusal == f"retrieved-overlap({measured:.4f}m2,desk)"

    def test_a_generated_piece_inside_the_wall_is_refused_by_the_bounds_rule(self):
        """The other question: kitchen seed 11 kept a mirror frame 0.0400 m inside the wall."""
        acceptance = importlib.import_module('rfw_w.scenes.application')
        # Inside the outline, but 0.05 m past the wall's inner face at x = 0.1.
        into_the_wall = Placed("generated_mirror_frame_1", Vec3(0.3, 0.3, 0.5), Vec3(0.2, 1.2, 0.0))
        refusal = self._refusal(acceptance, [], into_the_wall)
        assert refusal.startswith("inside-wall(")
        ok, measured, _detail, _ = geometry_module._geometry_in_bounds(
            _scene([into_the_wall]), wall_thickness_m=WALL_THICKNESS_M)
        assert not ok
        assert float(refusal.split("(")[1].rstrip("m)")) == pytest.approx(measured, abs=1e-4)

    def test_a_piece_past_the_outline_is_refused_as_having_left_the_building(self):
        acceptance = importlib.import_module('rfw_w.scenes.application')
        through_it = Placed("generated_shelf_1", Vec3(0.3, 0.3, 0.5), Vec3(-0.1, 1.2, 0.0))
        assert self._refusal(acceptance, [], through_it).startswith("outside-room(")

    def test_a_piece_resting_on_its_support_is_not_refused(self):
        """Same exemption as the gate: the thing holding you up does not obstruct you."""
        acceptance = importlib.import_module('rfw_w.scenes.application')
        cup_on_desk = Placed("generated_cup_1", Vec3(0.095, 0.095, 0.101),
                             Vec3(0.4802, 2.4, 0.8023), origin_at_base=False)
        assert self._refusal(acceptance, [DESK], cup_on_desk) == ""

    def test_a_piece_over_a_floor_covering_is_not_refused(self):
        acceptance = importlib.import_module('rfw_w.scenes.application')
        rug = Placed("rug", Vec3(2.438, 1.524, 0.015), Vec3(2.076, 1.673, 0.006))
        on_the_rug = Placed("generated_book_stack_1", Vec3(0.5, 0.5, 0.4), Vec3(2.0, 1.6, 0.0))
        assert self._refusal(acceptance, [rug], on_the_rug) == ""

    def test_a_piece_clear_of_everything_is_not_refused(self):
        acceptance = importlib.import_module('rfw_w.scenes.application')
        elsewhere = Placed("generated_shelf_trinkets_1", Vec3(0.4, 0.4, 0.3), Vec3(2.5, 1.0, 0.0))
        assert self._refusal(acceptance, [DESK], elsewhere) == ""
