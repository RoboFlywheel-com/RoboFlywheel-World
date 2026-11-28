# SPDX-License-Identifier: Apache-2.0
"""SceneIR topology and invalidation.

The invalidation tests are the important ones. If moving the desk does not invalidate the
cup's ON(desk) witness and the render that was approved for the old layout, the system
reports a stale pass -- which is the failure this whole contract exists to prevent.

The graph tests pin down a distinction that is easy to lose: relation cycles are legitimate
(mutual contact, "between these two"), support cycles are not definable.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_spec import Mobility, PhysicsType, Predicate
from rfw_w.scenes.contracts.state import (
    CameraRig,
    EvidenceKind,
    Instance,
    Opening,
    OpeningKind,
    RelationEdge,
    RoomShell,
    SceneIR,
    StateError,
    SupportEdge,
)
from rfw_w.scenes.contracts.units import Quat, Transform, Vec3

# An L-shaped room: the notch is the point. A rectangle stand-in would contain points the
# real room does not, and those points are where an object would end up floating.
L_ROOM = (
    (0.0, 0.0), (4.0, 0.0), (4.0, 2.2), (3.0, 2.2), (3.0, 3.2), (0.0, 3.2),
)


def shell(**overrides: object) -> RoomShell:
    payload: dict[str, object] = {
        "polygon_xy": L_ROOM,
        "height_m": 2.8,
        "openings": (Opening("door_main", OpeningKind.DOOR, (3.5, 0.0), 0.9, 2.05),),
        "source": "procedural_shell",
    }
    payload.update(overrides)
    return RoomShell(**payload)  # type: ignore[arg-type]


def instance(name: str, *, parent: str | None = None, mobility: Mobility = Mobility.FIXED) -> Instance:
    pose = Transform(Vec3(1.0, 1.0, 0.75), Quat.identity(), "world", name)
    return Instance(
        instance_id=name,
        entity_id=name,
        asset_ref=f"asset:{name}",
        physics_type=PhysicsType.RIGID,
        mobility=mobility,
        world_transform=pose,
        local_transform=Transform.identity(name),
        parent_id=parent,
        mass_kg=0.2 if mobility is Mobility.DYNAMIC else None,
    )


def scene(**overrides: object) -> SceneIR:
    payload: dict[str, object] = {
        "scene_id": "office_pick_cup_m1_001",
        "room": shell(),
        "instances": (
            instance("desk"),
            instance("chair"),
            instance("cup", parent="desk", mobility=Mobility.DYNAMIC),
            instance("notebook", parent="desk", mobility=Mobility.DYNAMIC),
        ),
        "relations": (
            RelationEdge("r1", "cup", Predicate.ON, "desk", True, True),
            RelationEdge("r2", "notebook", Predicate.ON, "desk", False, True),
        ),
        "supports": (
            SupportEdge("cup", "desk", 0.0021),
            SupportEdge("notebook", "desk", 0.019),
        ),
        "cameras": (
            CameraRig("overview_top", "layout", ()),
            CameraRig("desk_closeup", "relation", ("desk", "cup"), save_depth=True),
        ),
        "spec_hash": content_hash({"scene": "spec"}),
    }
    payload.update(overrides)
    return SceneIR(**payload)  # type: ignore[arg-type]


class TestGraphTopology:
    def test_support_cycle_is_rejected(self) -> None:
        # Self-supporting transitively is not definable, so settling has no meaning.
        with pytest.raises(StateError, match="cycle"):
            scene(supports=(
                SupportEdge("cup", "desk"),
                SupportEdge("desk", "cup"),
            ))

    def test_relation_cycle_is_allowed(self) -> None:
        """Mutual relations are legitimate; only the support graph must be acyclic.

        Pruning relation back edges as if the graph were a tree would silently drop a
        constraint, which is why the two graphs are validated differently.
        """
        cyclic = scene(relations=(
            RelationEdge("r1", "cup", Predicate.NEAR, "notebook", False, True),
            RelationEdge("r2", "notebook", Predicate.NEAR, "cup", False, True),
        ))
        assert len(cyclic.relations) == 2

    def test_rejects_unknown_parent(self) -> None:
        with pytest.raises(StateError, match="unknown parent"):
            scene(instances=(instance("cup", parent="ghost"),))

    def test_rejects_duplicate_instance_ids(self) -> None:
        with pytest.raises(StateError, match="duplicate instance ids"):
            scene(instances=(instance("cup"), instance("cup")))

    def test_rejects_support_edge_to_unknown_instance(self) -> None:
        with pytest.raises(StateError, match="unknown instance"):
            scene(supports=(SupportEdge("cup", "ghost"),))

    def test_rejects_relation_to_unknown_instance(self) -> None:
        with pytest.raises(StateError, match="references unknown"):
            scene(relations=(RelationEdge("r1", "cup", Predicate.ON, "ghost", True, True),))

    def test_room_polygon_needs_three_vertices(self) -> None:
        with pytest.raises(StateError, match="at least 3 vertices"):
            shell(polygon_xy=((0.0, 0.0), (1.0, 0.0)))

    def test_l_shaped_room_is_not_a_rectangle(self) -> None:
        assert not shell().is_rectangle

    def test_rectangular_room_with_a_hole_is_not_a_rectangle(self) -> None:
        # A hole changes what is inside the room, so the outline alone is not the room.
        holed = shell(
            polygon_xy=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
            holes_xy=(((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)),),
        )
        assert not holed.is_rectangle


class TestInstanceRules:
    def test_scale_other_than_one_is_refused(self) -> None:
        # Carrying a scale would silently invalidate measured mass, inertia and contact
        # evidence, so it cannot be stored as an ordinary field.
        with pytest.raises(StateError, match="scale"):
            Instance(
                instance_id="cup", entity_id="cup", asset_ref="asset:cup",
                physics_type=PhysicsType.RIGID, mobility=Mobility.DYNAMIC,
                world_transform=Transform.identity("world"),
                local_transform=Transform.identity("world"),
                scale=1.5,
            )

    def test_dynamic_flag_follows_mobility(self) -> None:
        assert scene().instance("cup").is_dynamic
        assert not scene().instance("desk").is_dynamic


class TestDependencyClosure:
    def test_moving_the_desk_invalidates_what_rests_on_it(self) -> None:
        closure = scene().dependency_closure(["desk"])
        assert closure.moved == {"desk", "cup", "notebook"}

    def test_moving_the_desk_invalidates_relation_evidence_to_its_children(self) -> None:
        closure = scene().dependency_closure(["desk"])
        assert {"desk", "cup", "notebook"} <= set(closure.evidence_affected)

    def test_closure_expands_through_the_relation_graph(self) -> None:
        """Moving a relation neighbour affects the witness even if geometry does not move.

        The cup does not move when the chair moves, but a NEAR(chair) witness for the cup
        does, so the expansion has to follow relations as well as support.
        """
        near = scene(relations=(
            RelationEdge("r1", "cup", Predicate.NEAR, "chair", False, True),
        ))
        closure = near.dependency_closure(["chair"])
        assert closure.moved == {"chair"}
        assert "cup" in closure.evidence_affected

    def test_closure_is_cycle_safe(self) -> None:
        """A relation cycle must terminate rather than recurse forever."""
        cyclic = scene(relations=(
            RelationEdge("r1", "cup", Predicate.NEAR, "chair", False, True),
            RelationEdge("r2", "chair", Predicate.NEAR, "cup", False, True),
        ))
        closure = cyclic.dependency_closure(["cup"])
        assert {"cup", "chair"} <= set(closure.evidence_affected)

    def test_closure_does_not_walk_support_upward(self) -> None:
        # Moving the cup does not move the desk it sits on.
        closure = scene().dependency_closure(["cup"])
        assert "desk" not in closure.moved

    def test_cameras_aimed_at_affected_instances_are_invalidated(self) -> None:
        closure = scene().dependency_closure(["desk"])
        assert closure.affected_cameras == {"desk_closeup"}

    def test_cameras_with_no_target_are_not_invalidated(self) -> None:
        closure = scene().dependency_closure(["desk"])
        assert "overview_top" not in closure.affected_cameras

    def test_closure_names_the_gates_that_must_be_recomputed(self) -> None:
        closure = scene().dependency_closure(["desk"], requested_backends=("isaac",))
        assert "function.initial_relations" in closure.affected_gates
        assert "geometry.supports" in closure.affected_gates
        assert "physics.settle_stability" in closure.affected_gates
        assert "visual.required_views" in closure.affected_gates
        assert "visual.critique" in closure.affected_gates
        assert "backend.isaac.render" in closure.affected_gates

    def test_backend_gates_are_namespaced_by_backend(self) -> None:
        closure = scene().dependency_closure(["desk"], requested_backends=("isaac", "mujoco"))
        assert "backend.mujoco.render" in closure.affected_gates
        assert "backend.isaac.render" in closure.affected_gates

    def test_closure_touches_geometry_and_physics_components(self) -> None:
        values = {component.value for component in scene().dependency_closure(["desk"]).touched_components}
        assert "geometry" in values
        assert "physics" in values

    def test_unknown_instance_is_rejected(self) -> None:
        with pytest.raises(StateError, match="unknown instances"):
            scene().dependency_closure(["ghost"])

    def test_closure_serialises(self) -> None:
        payload = scene().dependency_closure(["desk"]).to_dict()
        assert payload["moved"] == ["cup", "desk", "notebook"]
        assert payload["affected_cameras"] == ["desk_closeup"]


class TestHashes:
    def test_derived_hashes_are_stable(self) -> None:
        left = scene().derive_hashes()
        right = scene().derive_hashes()
        assert left.version == right.version

    def test_moving_an_instance_changes_the_geometry_hash(self) -> None:
        baseline = scene().derive_hashes()
        moved = Instance(
            instance_id="desk", entity_id="desk", asset_ref="asset:desk",
            physics_type=PhysicsType.RIGID, mobility=Mobility.FIXED,
            world_transform=Transform(Vec3(2.0, 1.0, 0.75), Quat.identity(), "world", "desk"),
            local_transform=Transform.identity("desk"),
        )
        changed = scene(instances=(
            moved,
            instance("chair"),
            instance("cup", parent="desk", mobility=Mobility.DYNAMIC),
            instance("notebook", parent="desk", mobility=Mobility.DYNAMIC),
        )).derive_hashes()
        assert changed.geometry_hash != baseline.geometry_hash
        assert changed.version != baseline.version

    def test_content_hash_is_independent_of_stored_hashes(self) -> None:
        # The verifier recomputes from content, so two IRs differing only in recorded hashes
        # must hash the same.
        plain = scene()
        recorded = scene(hashes=plain.derive_hashes())
        assert plain.content_hash() == recorded.content_hash()

    def test_revision_increments_when_hashes_are_attached(self) -> None:
        base = scene()
        assert base.with_hashes(base.derive_hashes()).revision == base.revision + 1
