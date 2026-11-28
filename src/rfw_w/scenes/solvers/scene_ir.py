# SPDX-License-Identifier: Apache-2.0
"""Build the canonical SceneIR from a solved scene.

The acceptance script historically carried the solved state as a dictionary and independently
rebuilt its version payload, worker request and manifest placement section.  This adapter is the
single boundary that turns solver output into the typed graph all downstream consumers can share.
It performs no I/O and imports no backend.
"""

from __future__ import annotations

from typing import Any, Mapping

from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.background_constraints import generated_category
from rfw_w.scenes.contracts.scene_spec import (
    Mobility,
    PhysicsType,
    Predicate,
    SceneScope,
    SceneSpec,
    SpecError,
    relation_closure_ids,
    scope_of,
)
from rfw_w.scenes.contracts.state import Instance, RelationEdge, SceneIR, SupportEdge
from rfw_w.scenes.contracts.units import Quat, Transform, Vec3


def _world_transform(entry: Any) -> Transform:
    rotation = Quat.from_sequence(
        entry.rotation_wxyz or (1.0, 0.0, 0.0, 0.0),
        context=f"{entry.instance_id}.rotation_wxyz",
    )
    return Transform(
        translation=Vec3(entry.position_m.x, entry.position_m.y, entry.position_m.z),
        rotation=rotation,
        parent_frame="world",
        child_frame=entry.instance_id,
    )


def scene_ir_from_build(spec: SceneSpec, scene: Mapping[str, Any]) -> SceneIR:
    """Convert one completed build into the graph delivered to downstream consumers."""
    entries = {entry.instance_id: entry for entry in scene["instances"]}
    worlds = {name: _world_transform(entry) for name, entry in entries.items()}

    parents: dict[str, str] = {}
    relation_satisfied: dict[str, bool] = {}
    for relation in spec.initial_relations:
        present = relation.subject in entries and relation.object in entries
        satisfied = False
        if present and relation.predicate is Predicate.ON:
            # Surface objects are placed through the named support surface. Other predicates are
            # carried but remain false until a predicate solver produces a witness.
            requested_parent = next(
                (
                    item.support_entity
                    for item in scene["plan"].surfaces
                    if item.entity_id == relation.subject
                ),
                scene["support_id"] if relation.subject == scene["target_id"] else None,
            )
            satisfied = requested_parent == relation.object
            if satisfied:
                parents[relation.subject] = relation.object
        relation_satisfied[relation.id] = satisfied

    instances: list[Instance] = []
    declared = {entity.id for entity in spec.entities}
    for instance_id, entry in entries.items():
        # A generated instance is not a spec entity: the task declares what the scene must contain,
        # and a background object is what the room gained beyond that (`generated_<category>_<n>`,
        # contracts/background_constraints.generated_entity_id). Anything else the spec does not
        # declare is still a fault -- a typo in an entity id must not read as a background piece.
        entity = spec.entity(entry.entity_id) if entry.entity_id in declared else None
        if entity is None and generated_category(entry.entity_id) is None:
            raise SpecError(f"unknown entity {entry.entity_id!r}")
        parent_id = parents.get(instance_id)
        world = worlds[instance_id]
        local = world if parent_id is None else worlds[parent_id].inverse().compose(world)
        half = entry.dimensions_m.scale(0.5)
        bounds_min = getattr(entry, "local_bounds_min_m", None) or Vec3(-half.x, -half.y, 0.0 if entry.origin_at_base else -half.z)
        bounds_max = getattr(entry, "local_bounds_max_m", None) or Vec3(half.x, half.y, entry.dimensions_m.z if entry.origin_at_base else half.z)
        instances.append(
            Instance(
                instance_id=instance_id,
                entity_id=entry.entity_id,
                asset_ref=f"{entry.provider}:{entry.library_version}:{entry.asset_id}",
                # A generated object is a rigid static piece: upstream realises furniture, and
                # SPEC-009 forbids claiming that a generated piece opens, articulates or can be
                # operated. `RIGID` here is that refusal, not an assumption about the mesh.
                physics_type=entity.physics_type if entity is not None else PhysicsType.RIGID,
                mobility=Mobility.DYNAMIC if entry.dynamic else Mobility.FIXED,
                world_transform=world,
                local_transform=local,
                parent_id=parent_id,
                mass_kg=entry.mass_kg,
                bounds_min_m=bounds_min,
                bounds_max_m=bounds_max,
                prepared_version=entry.source_hash or entry.library_version,
            )
        )

    relations = tuple(
        RelationEdge(
            relation_id=relation.id,
            subject=relation.subject,
            predicate=relation.predicate,
            object=relation.object,
            hard=relation.hard,
            satisfied=relation_satisfied[relation.id],
            witness_refs=(f"surface:{relation.object}:top",)
            if relation_satisfied[relation.id] and relation.predicate is Predicate.ON
            else (),
        )
        for relation in spec.initial_relations
        if relation.subject in entries and relation.object in entries
    )
    supports = tuple(
        SupportEdge(
            child_id=child,
            parent_id=parent,
            contact_polygon_area_m2=(
                entries[child].dimensions_m.x * entries[child].dimensions_m.y
            ),
        )
        for child, parent in sorted(parents.items())
    )
    return SceneIR(
        scene_id=spec.scene_id,
        room=scene["shell"],
        instances=tuple(instances),
        relations=relations,
        supports=supports,
        randomization=scene.get("randomization"),
        spec_hash=content_hash(spec.to_dict()),
    )


def requested_entity_coverage(
    spec: SceneSpec, scene: Mapping[str, Any]
) -> tuple[float, tuple[str, ...], float, tuple[str, ...]]:
    """Return task-structure and total requested-entity coverage."""
    placed = {entry.entity_id for entry in scene["instances"]}
    requested = {entity.id for entity in spec.entities}
    task_entities = set(relation_closure_ids(spec)[0])
    missing_task = tuple(sorted(task_entities - placed))
    missing_all = tuple(sorted(requested - placed))
    return (
        (len(task_entities) - len(missing_task)) / len(task_entities),
        missing_task,
        (len(requested) - len(missing_all)) / len(requested),
        missing_all,
    )


def missing_core_structure(
    spec: SceneSpec, graph: SceneIR
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return core nodes and hard relation edges absent from the delivered graph.

    Scope comes from entity roles. Relation preservation remains explicit because an instance-only
    check would pass a graph that contains the cup and desk but has dropped the hard ON edge.
    """
    observed_entities = {instance.entity_id for instance in graph.instances}
    observed_relations = {relation.relation_id for relation in graph.relations}
    required_entities = {
        entity.id for entity in spec.entities if scope_of(entity) is SceneScope.CORE
    }
    required_relations = {relation.id for relation in spec.hard_relations}
    return (
        tuple(sorted(required_entities - observed_entities)),
        tuple(sorted(required_relations - observed_relations)),
    )
