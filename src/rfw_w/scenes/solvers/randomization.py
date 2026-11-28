# SPDX-License-Identifier: Apache-2.0
"""Sample only the domain outside the task-relation closure, with a replayable seed."""

from __future__ import annotations

import random
from dataclasses import replace

from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.randomization import (
    AppearanceVariation,
    DomainRandomizationRecord,
    RandomizedEntityChoice,
)
from rfw_w.scenes.contracts.scene_spec import SceneSpec, relation_closure_ids
from rfw_w.scenes.solvers.furnish import AppearanceSettings, ScenePlan


def task_invariant_ids(spec: SceneSpec) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Compatibility name for the relation closure frozen by randomization records."""
    return relation_closure_ids(spec)


def invariant_spec_hash(spec: SceneSpec) -> str:
    entity_ids, relation_ids = task_invariant_ids(spec)
    return content_hash({
        "entities": [spec.entity(entity_id).to_dict() for entity_id in entity_ids],
        "relations": [
            relation.to_dict() for relation in spec.initial_relations if relation.id in relation_ids
        ],
        "task": spec.task.to_dict(),
    })


def randomize_plan(
    spec: SceneSpec,
    plan: ScenePlan,
    *,
    variant_seed: int,
    enabled: bool,
) -> tuple[ScenePlan, DomainRandomizationRecord]:
    """Return a deterministic variant and the complete choice record used to reproduce it."""
    invariant_entities, invariant_relations = task_invariant_ids(spec)
    if not enabled:
        return plan, DomainRandomizationRecord(
            enabled=False,
            variant_seed=variant_seed,
            room_seed=None,
            invariant_entity_ids=invariant_entities,
            invariant_relation_ids=invariant_relations,
            invariant_spec_hash=invariant_spec_hash(spec),
        )

    rng = random.Random(variant_seed)
    choices: list[RandomizedEntityChoice] = []
    furniture = []
    protected = set(invariant_entities)
    for request in plan.furniture:
        if request.entity_id in protected:
            furniture.append(request)
            continue
        # Candidate zero remains reachable for half the seeds; candidate one introduces asset
        # diversity without jumping so far down the pool that valid variants become exceptional.
        index = rng.randrange(2)
        furniture.append(replace(request, asset_index=index))
        choices.append(RandomizedEntityChoice(request.entity_id, index))

    appearance = AppearanceSettings(
        dome_intensity=round(rng.uniform(720.0, 1080.0), 3),
        sun_intensity=round(rng.uniform(2080.0, 3120.0), 3),
        ceiling_fill_intensity=round(rng.uniform(560.0, 840.0), 3),
        dome_color=plan.appearance.dome_color,
        sun_color=plan.appearance.sun_color,
    )
    varied = replace(plan, furniture=tuple(furniture), appearance=appearance)
    return varied, DomainRandomizationRecord(
        enabled=True,
        variant_seed=variant_seed,
        room_seed=variant_seed,
        room_seed_attempts=(variant_seed,),
        invariant_entity_ids=invariant_entities,
        invariant_relation_ids=invariant_relations,
        invariant_spec_hash=invariant_spec_hash(spec),
        entity_choices=tuple(choices),
        appearance=AppearanceVariation(
            dome_intensity=appearance.dome_intensity,
            sun_intensity=appearance.sun_intensity,
            ceiling_fill_intensity=appearance.ceiling_fill_intensity,
        ),
    )
