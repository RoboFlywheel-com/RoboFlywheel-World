# SPDX-License-Identifier: Apache-2.0
"""The solved scene must become the typed graph shipped to dataset consumers."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rfw_w.scenes.contracts.scene_spec import SceneSpec, SpecError
from rfw_w.scenes.contracts.state import RoomShell
from rfw_w.scenes.contracts.units import Vec3
from rfw_w.scenes.solvers.furnish import (
    FurnitureRequest,
    Orientation,
    PlacementKind,
    ScenePlan,
    SurfaceRequest,
)
from rfw_w.scenes.solvers.scene_ir import (
    missing_core_structure,
    requested_entity_coverage,
    scene_ir_from_build,
)


def spec() -> SceneSpec:
    return SceneSpec.from_dict({
        "scene_id": "graph_test",
        "seed": 7,
        "prompt": "pick up the cup from the desk",
        "room": {"size_bounds_m": [[4, 4], [3, 3], [2.8, 2.8]]},
        "style": {"theme": "plain", "palette": ["neutral"]},
        "requested_backends": ["isaac"],
        "requested_readiness": "functional_scene_ready",
        "entities": [
            {"id": "desk", "category": "desk", "physics_type": "rigid", "mobility": "fixed",
             "hard_required": True, "role": "support"},
            {"id": "cup", "category": "cup", "physics_type": "rigid", "mobility": "dynamic",
             "hard_required": True, "role": "task_target"},
            {"id": "notebook", "category": "notebook", "physics_type": "rigid",
             "mobility": "dynamic", "hard_required": False, "role": "context"},
        ],
        "initial_relations": [
            {"id": "r1", "subject": "cup", "predicate": "ON", "object": "desk", "hard": True},
            {"id": "r2", "subject": "notebook", "predicate": "ON", "object": "desk", "hard": False},
        ],
        "task": {"action": "pick", "robot_profile": "franka_panda", "targets": ["cup"]},
    })


def entry(name: str, position: tuple[float, float, float], *, dynamic: bool = False):
    return SimpleNamespace(
        instance_id=name,
        entity_id=name,
        provider="test",
        library_version="v1",
        asset_id=f"asset-{name}",
        source_hash=f"hash-{name}",
        position_m=Vec3(*position),
        dimensions_m=Vec3(1.0, 0.6, 0.8) if name == "desk" else Vec3(0.1, 0.1, 0.1),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        origin_at_base=name != "cup",
        dynamic=dynamic,
        mass_kg=0.2 if dynamic else None,
    )


def solved(*, include_notebook: bool = True, include_cup: bool = True):
    plan = ScenePlan(
        furniture=(FurnitureRequest(
            entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
            orientation=Orientation.FRONT_TO_ROOM, hard_required=True, optional=False,
        ),),
        surfaces=(SurfaceRequest(
            entity_id="notebook", synset="notebook.n.01", target_width_m=0.2,
            support_entity="desk",
        ),),
        support_id="desk",
        target_id="cup",
    )
    instances = [entry("desk", (1.0, 1.0, 0.0))]
    if include_cup:
        instances.append(entry("cup", (1.0, 1.0, 0.85), dynamic=True))
    if include_notebook:
        instances.append(entry("notebook", (1.2, 1.0, 0.8), dynamic=True))
    return {
        "plan": plan,
        "instances": instances,
        "support_id": "desk",
        "target_id": "cup",
        "shell": RoomShell(
            polygon_xy=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
            height_m=2.8,
        ),
    }


def test_support_topology_and_assets_reach_the_scene_ir() -> None:
    graph = scene_ir_from_build(spec(), solved())
    assert graph.instance("cup").parent_id == "desk"
    assert graph.instance("notebook").parent_id == "desk"
    assert {edge.child_id for edge in graph.supports} == {"cup", "notebook"}
    assert graph.instance("cup").asset_ref == "test:v1:asset-cup"


def test_missing_optional_requested_entity_is_still_a_coverage_gap() -> None:
    task_coverage, missing_task, total_coverage, missing_all = requested_entity_coverage(
        spec(), solved(include_notebook=False)
    )
    assert task_coverage == 2 / 3
    assert missing_task == ("notebook",)
    assert total_coverage == 2 / 3
    assert missing_all == ("notebook",)


def test_missing_core_node_is_a_preservation_failure() -> None:
    graph = scene_ir_from_build(spec(), solved(include_cup=False))
    assert missing_core_structure(spec(), graph) == (("cup",), ("r1",))


def test_missing_hard_relation_is_a_preservation_failure_even_when_nodes_exist() -> None:
    graph = scene_ir_from_build(spec(), solved())
    graph_without_relations = replace(graph, relations=())
    assert missing_core_structure(spec(), graph_without_relations) == ((), ("r1",))


def test_optional_soft_relation_is_not_part_of_core_preservation() -> None:
    graph = scene_ir_from_build(spec(), solved(include_notebook=False))
    assert missing_core_structure(spec(), graph) == ((), ())


def test_a_generated_object_reaches_the_graph_without_being_a_spec_entity() -> None:
    """SPEC-009 TASK 2's remaining half: a kept generated piece becomes a real instance.

    The graph used to refuse any entity id the spec did not declare, which is right for a typo and
    wrong for the background branch: the task declares what must be there, and a generated object
    is what the room gained beyond it. It arrives rigid and fixed -- SPEC-009 forbids claiming a
    generated piece articulates or can be operated -- and it takes part in no relation or support.
    """
    scene = solved()
    scene["instances"].append(entry("generated_monitor_1", (3.0, 2.0, 0.0)))
    graph = scene_ir_from_build(spec(), scene)
    monitor = graph.instance("generated_monitor_1")
    assert monitor.physics_type.value == "rigid"
    assert monitor.mobility.value == "fixed"
    assert monitor.parent_id is None
    assert not [edge for edge in graph.supports if edge.child_id == "generated_monitor_1"]
    # It is not a requested entity, so it cannot make coverage look better or a preservation check
    # look worse: those are about what the task asked for.
    assert requested_entity_coverage(spec(), scene)[3] == ()
    assert missing_core_structure(spec(), graph) == ((), ())


def test_an_entity_the_spec_never_declared_is_still_a_fault() -> None:
    scene = solved()
    scene["instances"].append(entry("monitor_typo", (3.0, 2.0, 0.0)))
    with pytest.raises(SpecError, match="unknown entity 'monitor_typo'"):
        scene_ir_from_build(spec(), scene)
