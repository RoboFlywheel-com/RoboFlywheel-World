# SPDX-License-Identifier: Apache-2.0
"""The `function.space_conditions` gate must be computed, not asserted.

It used to be recorded as an unconditional pass whose detail claimed "the desk zone leaves the
approach from the door clear" -- a sentence no code in the repository evaluated, on a gate that
`FUNCTIONAL_SCENE_READY` and above require. So the tests here are mostly *negative*: they place
furniture that blocks the way and require the gate to fail. A gate that cannot fail is the defect
being fixed, and only a failing case proves it is gone.

No assets, no GPU, no Isaac: a synthetic scene is enough, because the gate reads only the shell, the
zone plan and the placed footprints.
"""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import rfw_w.scenes.assembly.geometry as geometry_module
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from rfw_w.scenes.contracts.scene_spec import SceneSpec, ZoneKind
from rfw_w.scenes.contracts.units import Vec3
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.room import build_shell, generate_options
from rfw_w.scenes.solvers.zones import plan_zones

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE = module_path('rfw_w.scenes.application')

SPEC = SceneSpec.from_dict({
    "schema_version": "pvscene.scene-spec/v3",
    "scene_id": "unit_space_conditions",
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

# The robot the shell options declare, so the gate's numbers are the ones it was built against.
ROBOT_FOOTPRINT_M = 0.6
MARGIN_M = 0.05


@pytest.fixture(scope="module")
def acceptance():
    return importlib.import_module('rfw_w.scenes.application')


@dataclass
class Placed:
    """The fields the gate reads off a placed instance, and nothing else."""

    instance_id: str
    dimensions_m: Vec3
    yaw_rad: float
    position_m: Vec3


def _scene(shell, zones, placed, workspace=None):
    return {
        "shell": shell,
        "zones": zones,
        "scene_id": SPEC.scene_id,
        "room_polygon": Polygon(shell.polygon_xy),
        "floor_occupants": frozenset(entry.instance_id for entry in placed),
        "desk_entry": next((e for e in placed if e.instance_id == "desk"), None),
        "instances": list(placed),
        # The task's workspace, which the real build takes from the task cell's protected region.
        # A synthetic scene has no cell, so this stands one in with the region the gate used to be
        # pointed at: what these tests are about is the route from the door, not what it is aimed at.
        "workspace": zones.zone_for(ZoneKind.WORK).polygons[0] if workspace is None else workspace,
    }


def _fixture(seed: int = 11, *, door_width_m: float | None = None):
    spec = replace(SPEC, seed=seed)
    options = generate_options(spec)
    if door_width_m is not None:
        # Through the method, not a `replace` on the field: the field is what the drawer used and
        # the resolved plan is what the shell is built from, so replacing the field alone left the
        # plan saying 0.9 and the shell honouring the plan.
        options = options.with_door_width(door_width_m)
    shell = build_shell(spec, options=options)
    zones = plan_zones(shell, spec.style)
    return spec, shell, zones


class TestComputesAPass:
    def test_empty_room_passes(self, acceptance) -> None:
        spec, shell, zones = _fixture()
        ok, measured, detail, witnesses = geometry_module._space_conditions(
            _scene(shell, zones, [Placed("desk", Vec3(1.37, 0.76, 0.75), 0.0, Vec3(0.4, 2.5, 0.0))]),
            spec,
        )
        assert ok, detail
        assert measured == pytest.approx(ROBOT_FOOTPRINT_M + 2 * MARGIN_M)
        assert witnesses, "a passing gate must name the geometry it measured"

    def test_a_door_narrower_than_the_robot_fails(self, acceptance) -> None:
        spec, shell, zones = _fixture(door_width_m=0.5)
        ok, measured, detail, _ = geometry_module._space_conditions(_scene(shell, zones, []), spec)
        assert not ok
        assert "narrower than" in detail

    def test_a_scene_with_no_workspace_fails_by_name(self, acceptance) -> None:
        """It must not pass a scene it cannot aim at.

        The goal used to be read off the zone plan, and a plan with no work zone made the gate fall
        through to a failure -- but only because a *zone* was missing, which is a statement about the
        plan and not about the task. The workspace is now recorded by the build, and a scene without
        one is a scene this gate cannot answer. That has to read as a named failure, not as silence.
        """
        spec, shell, zones = _fixture()
        scene = _scene(shell, zones, [])
        del scene["workspace"]
        ok, _, detail, _ = geometry_module._space_conditions(scene, spec)
        assert not ok
        assert "no task workspace" in detail

    def test_a_workspace_the_route_cannot_reach_fails(self, acceptance) -> None:
        """The property the gate is for, stated against the workspace rather than against a zone.

        A workspace on floor that is not in this room at all: no piece of the robot's centre-space
        can meet it, so the answer must be no. Nothing about the furniture changes -- the goal does.
        """
        spec, shell, zones = _fixture()
        elsewhere = Polygon.rectangle(50.0, 50.0, 51.0, 51.0)
        ok, _, detail, _ = geometry_module._space_conditions(
            _scene(shell, zones, [], workspace=elsewhere), spec
        )
        assert not ok
        assert "task workspace" in detail


class TestDetectsABlockedRoute:
    """The manual's own example: a desk seated across its own doorway must not pass."""

    def test_blocking_the_corridor_fails(self, acceptance) -> None:
        spec, shell, zones = _fixture()
        corridor = zones.circulation
        assert corridor, "fixture has no corridor to block"
        xs = [vertex[0] for piece in corridor for vertex in piece.vertices]
        ys = [vertex[1] for piece in corridor for vertex in piece.vertices]
        middle = (sum(xs) / len(xs), sum(ys) / len(ys))

        blocker = Placed("storage", Vec3(2.5, 2.5, 1.0), 0.0, Vec3(middle[0], middle[1], 0.0))
        ok, _, detail, _ = geometry_module._space_conditions(_scene(shell, zones, [blocker]), spec)
        assert not ok, "a piece standing across the route was reported as a legal passage"
        assert "obstruct" in detail or "reach" in detail

    def test_blocking_the_doorway_fails(self, acceptance) -> None:
        spec, shell, zones = _fixture()
        doors = [o for o in shell.openings if o.kind.value == "door"]
        door = doors[0]
        # A wide piece pressed along the inside of the door wall, admitting nothing.
        blocker = Placed(
            "storage", Vec3(3.0, 0.9, 1.0), 0.0,
            Vec3(door.center_xy[0], door.center_xy[1] + 0.45, 0.0),
        )
        ok, _, detail, _ = geometry_module._space_conditions(_scene(shell, zones, [blocker]), spec)
        assert not ok, "furniture across the doorway was reported as a legal passage"
        assert detail

    def test_a_route_that_is_merely_narrow_fails(self, acceptance) -> None:
        """Connectivity alone would pass this: the gap is real floor, just not robot-width."""
        spec, shell, zones = _fixture()
        room = Polygon(shell.polygon_xy)
        min_x, min_y, max_x, max_y = room.bounds()
        # Two tall walls of furniture with a 0.3 m slot between them -- narrower than the robot.
        walls = [
            Placed("storage", Vec3(max_x - min_x, 2.0, 1.0), 0.0,
                   Vec3((min_x + max_x) / 2, min_y + 1.0, 0.0)),
            Placed("side_table", Vec3(max_x - min_x, 2.0, 1.0), 0.0,
                   Vec3((min_x + max_x) / 2, min_y + 3.3, 0.0)),
        ]
        ok, _, detail, _ = geometry_module._space_conditions(_scene(shell, zones, walls), spec)
        assert not ok, "a passage 0.3 m wide was accepted for a 0.6 m robot"
        assert detail
