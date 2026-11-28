# SPDX-License-Identifier: Apache-2.0
"""Deterministic geometry, relation and shell evidence for a scene version."""

from __future__ import annotations
from rfw_w.scenes.contracts.shell_build import HEIGHT_TOLERANCE_M, WALL_LINE_TOLERANCE_M, plan_gaps
from rfw_w.scenes.contracts.tool import Evidence, EvidenceStatus
from rfw_w.scenes.solvers.relation import RelationRequest, evaluate_relation
from rfw_w.scenes.solvers.shell_providers import ShellProviderError, minimum_door_width_m
from rfw_w.scenes.solvers.scene_ir import missing_core_structure, requested_entity_coverage
from rfw_w.scenes.solvers.randomization import invariant_spec_hash
from rfw_w.scenes.assembly.geometry import _geometry_in_bounds, _geometry_no_penetration, _geometry_task_cell_clear, _space_conditions, _task_reachable
from rfw_w.scenes.delivery.evidence import evidence
from rfw_w.scenes.runtime import step

def measure_shell_matches_plan(host, scene_version: str) -> None:
    """Did the builder build the room the plan describes?

    The comparison set is in `plan_gaps`, and it is deliberately small: the outline (through
    the spread of wall-line offsets), the height, and per opening its existence, its kind and
    which wall it is on. Opening *sizes* are not in it, because the plan cannot constrain them
    -- SPEC-001 R6 measured Infinigen producing the same 0.929 m door from a 0.9 m and a 2.0 m
    segment -- and a gate that fails every build for something the builder was never able to
    honour is a gate that gets switched off.

    The provider is `--shell-provider`, `builtin` by default. That is today's honest state:
    the room the worker renders is still the one this repository draws (SPEC-005 is the step
    that changes it), so gating a room nobody renders would be worse than gating this one. It
    also means the gate is not yet load-bearing for Infinigen in a full run -- which is why
    the acceptance evidence for the measured provider is the three `--provider infinigen`
    builds recorded in the manual, not this line.
    """
    room_plan = host.scene.get("room_plan")
    if room_plan is None:
        raise SystemExit(
            "the build produced no room plan, so shell.matches_plan has nothing to compare "
            "against; the plan is written beside the room in build_scene"
        )
    provider_name = getattr(host.args, "shell_provider", "builtin")
    build = host._shell_build_lazy(room_plan)
    if isinstance(build, ShellProviderError):
        host.ledger.record(evidence(
            "shell.matches_plan", EvidenceStatus.FAIL, scene_version,
            method=f"{provider_name}_shell_build",
            detail=str(build), witnesses=("region:shell",),
        ))
        step("shell", f"shell.matches_plan = failed (the {provider_name} build failed)")
        return

    gaps = plan_gaps(room_plan, build, minimum_door_width_m=minimum_door_width_m())
    # The report has to name all three of what was measured, what tolerance it was held to and
    # what was set aside -- otherwise "passed" is a word with no content. `not_compared` is
    # carried in the detail rather than in `not_measured`, because the two say opposite things:
    # a quantity that was measured but is not constrained by the plan is not a quantity nobody
    # looked at.
    detail = (
        f"{provider_name} built the room; compared the outline (spread tolerance "
        f"{WALL_LINE_TOLERANCE_M:g} m), the height (tolerance {HEIGHT_TOLERANCE_M:g} m) and "
        f"{len(room_plan.openings)} opening(s) for existence, kind and wall; "
        f"set aside {len(build.not_compared)} measured quantity(ies) the plan does not "
        f"constrain; did not measure {len(build.not_measured)}"
    )
    host.ledger.record(evidence(
        "shell.matches_plan",
        EvidenceStatus.PASS if not gaps else EvidenceStatus.FAIL,
        scene_version,
        method=f"{provider_name}_shell_compared_against_the_plan",
        measured=round(
            float(build.measurements.get("room.wall_line_offset_spread_m", 0.0)), 6
        ),
        unit="m",
        profile="shell_plan_v1",
        detail=detail if not gaps else f"{detail}; gaps: " + "; ".join(gaps),
        witnesses=tuple(f"region:shell:{gap.split('(')[0]}" for gap in gaps) or ("region:shell",),
    ))
    step("shell", f"shell.matches_plan = {'passed' if not gaps else 'failed'} "
                  f"({provider_name}, {len(build.not_compared)} not compared)")


def measure_deterministic_gates(host, scene_version: str) -> None:
    """Every gate that needs no simulator and no model.

    Cheap enough to re-run after each repair, which is what keeps the geometry phase dense: the
    render is not touched until these all pass, so a run can take several rungs without paying
    for a picture of a scene that is about to change again.
    """
    scene = host.scene
    randomization = scene["randomization"]
    graph = scene["scene_ir"]
    measure_shell_matches_plan(host, scene_version)
    observed_entities = {instance.entity_id for instance in graph.instances}
    observed_relations = {relation.relation_id for relation in graph.relations}
    invariant_missing = sorted(
        set(randomization.invariant_entity_ids) - observed_entities
    )
    relation_missing = sorted(
        set(randomization.invariant_relation_ids) - observed_relations
    )
    invariant_ok = (
        not invariant_missing
        and not relation_missing
        and randomization.invariant_spec_hash == invariant_spec_hash(host.spec)
    )
    host.ledger.record(evidence(
        "semantics.randomization_invariants",
        EvidenceStatus.PASS if invariant_ok else EvidenceStatus.FAIL,
        scene_version,
        method="recorded_task_closure_after_randomization",
        measured=float(len(invariant_missing) + len(relation_missing)),
        unit="missing_invariant_nodes_or_edges",
        profile=randomization.policy_version,
        detail=(
            f"variant seed {randomization.variant_seed} preserved "
            f"{len(randomization.invariant_entity_ids)} task entities and "
            f"{len(randomization.invariant_relation_ids)} relations"
            if invariant_ok else
            f"randomization invariant mismatch: entities={invariant_missing}, "
            f"relations={relation_missing}"
        ),
        witnesses=tuple(
            [f"entity:{name}:missing-after-randomization" for name in invariant_missing]
            + [f"relation:{name}:missing-after-randomization" for name in relation_missing]
        ),
    ))
    coverage, missing, total_coverage, missing_all = requested_entity_coverage(host.spec, scene)
    host.ledger.record(evidence(
        "semantics.task_structure_covered",
        EvidenceStatus.PASS if not missing else EvidenceStatus.FAIL,
        scene_version,
        method="scene_ir_entity_coverage",
        measured=coverage,
        unit="fraction_of_task_structure_entities",
        profile="task_relation_closure_v1",
        detail=(
            f"all task targets and relation endpoints are present; total requested entity "
            f"coverage is {total_coverage:.3f}"
            if not missing else
            f"task structure is missing {', '.join(missing)}; total requested entity "
            f"coverage is {total_coverage:.3f}, with {', '.join(missing_all)} absent"
        ),
        witnesses=tuple(f"entity:{name}:missing" for name in missing),
    ))
    missing_core, missing_hard_relations = missing_core_structure(host.spec, graph)
    core_ok = not missing_core and not missing_hard_relations
    host.ledger.record(evidence(
        "semantics.hard_entities_preserved",
        EvidenceStatus.PASS if core_ok else EvidenceStatus.FAIL,
        scene_version,
        method="scene_ir_core_and_hard_relation_presence",
        measured=float(len(missing_core) + len(missing_hard_relations)),
        unit="missing_core_nodes_or_hard_edges",
        profile="goal_closure_v2",
        detail=(
            f"all role-defined core entities and {len(host.spec.hard_relations)} hard "
            "relations are present"
            if core_ok else
            f"core preservation mismatch: entities={list(missing_core)}, "
            f"hard_relations={list(missing_hard_relations)}"
        ),
        witnesses=tuple(
            [f"entity:{name}:missing-core" for name in missing_core]
            + [f"relation:{name}:missing-hard" for name in missing_hard_relations]
        ),
    ))

    penetration_ok, penetration_measured, penetration_detail, penetration_witnesses = (
        _geometry_no_penetration(scene)
    )
    step("validate_geometry",
         f"geometry.no_penetration = {'passed' if penetration_ok else 'failed'}")
    host.ledger.record(evidence(
        "geometry.no_penetration",
        EvidenceStatus.PASS if penetration_ok else EvidenceStatus.FAIL,
        scene_version,
        method="plan_overlap_with_height_separation_over_all_placed_instances",
        measured=penetration_measured, unit="m2", profile="engineering_v1",
        detail=penetration_detail, witnesses=penetration_witnesses,
    ))

    clear, shared, detail, witnesses = _geometry_task_cell_clear(scene)
    host.ledger.record(evidence(
        "geometry.task_cell_clear", EvidenceStatus.PASS if clear else EvidenceStatus.FAIL,
        scene_version, method="placed_floor_footprint_intersection_with_task_workspace",
        measured=shared, unit="m2", profile="engineering_v1", detail=detail, witnesses=witnesses,
    ))

    bounds_ok, bounds_measured, bounds_detail, bounds_witnesses = _geometry_in_bounds(
        scene, wall_thickness_m=host.wall_thickness_m_value,
    )
    step("validate_geometry", f"geometry.in_bounds = {'passed' if bounds_ok else 'failed'}")
    host.ledger.record(evidence(
        "geometry.in_bounds",
        EvidenceStatus.PASS if bounds_ok else EvidenceStatus.FAIL,
        scene_version,
        method="placed_box_containment_inside_wall_inner_face_and_height",
        measured=bounds_measured, unit="m", profile="engineering_v1",
        detail=bounds_detail, witnesses=bounds_witnesses,
    ))

    # The support surface the task target stands on, as a gate that can fail. It used to be an
    # unconditional PASS with a detail string naming a hardcoded desk -- a gate that cannot
    # fail, on the one surface the whole task depends on. `AffordanceStatus` already carries the
    # judgement ("estimated is not enough: an estimate has not been verified against the real
    # asset"), so the gate reads it rather than inventing a second rule.
    top_provenance = scene["support_top"].provenance
    support_declared = bool(scene["support"].support_polygons) and (
        top_provenance.usable_for_hard_requirement
    )
    host.ledger.record(evidence(
        "geometry.supports",
        EvidenceStatus.PASS if support_declared else EvidenceStatus.FAIL,
        scene_version,
        method="support_surface_declared_and_measured", profile="support_declared_v1",
        detail=(
            f"{host.support_id!r} stands the task on a {top_provenance.value} surface"
            if support_declared else
            f"{host.support_id!r} declares "
            f"{len(scene['support'].support_polygons)} support surface(s) with provenance "
            f"{top_provenance.value}; the task target {host.target_id!r} has nothing "
            f"verified to stand on"
        ),
        witnesses=(
            () if support_declared
            else (f"instance:{host.support_id}:support-unverified",)
        ),
    ))
    step("validate_geometry",
         f"geometry.supports = {'passed' if support_declared else 'failed'}")

    hard_relations = tuple(relation for relation in host.spec.initial_relations if relation.hard)
    relation_results: list[Evidence] = []
    for relation in hard_relations:
        subject = scene["subjects"].get(relation.subject)
        if subject is None:
            relation_results.append(evidence(
                "function.initial_relations", EvidenceStatus.FAIL, scene_version,
                method="hard_relation_subject_presence",
                detail=(
                    f"{relation.id}: {relation.subject} {relation.predicate.value} "
                    f"{relation.object} cannot be evaluated because the subject is absent"
                ),
                witnesses=(f"relation:{relation.id}:subject-missing",),
            ))
            continue
        relation_results.append(evaluate_relation(
            RelationRequest(
                relation_id=relation.id,
                subject_instance=relation.subject,
                predicate=relation.predicate,
                object_instance=relation.object,
                scene_version=scene_version,
                hard=True,
            ),
            subject=subject,
            support_instance=relation.object,
            surfaces=scene["surfaces"],
        ))
    if not relation_results:
        host.relation_evidence = evidence(
            "function.initial_relations", EvidenceStatus.FAIL, scene_version,
            method="all_hard_initial_relations",
            detail="the task declares no hard initial relation to validate",
        )
    else:
        statuses = {item.status for item in relation_results}
        aggregate_status = (
            EvidenceStatus.FAIL if EvidenceStatus.FAIL in statuses else
            EvidenceStatus.UNSUPPORTED if EvidenceStatus.UNSUPPORTED in statuses else
            EvidenceStatus.NOT_RUN if EvidenceStatus.NOT_RUN in statuses else
            EvidenceStatus.PASS
        )
        host.relation_evidence = evidence(
            "function.initial_relations", aggregate_status, scene_version,
            method="all_hard_initial_relations",
            measured=float(sum(item.status is EvidenceStatus.PASS for item in relation_results)),
            unit="relations_passed",
            profile="all_hard_relations_v1",
            detail="; ".join(item.detail for item in relation_results),
            witnesses=tuple(
                witness
                for item in relation_results
                for witness in item.witness_refs
            ),
        )
    step("validate_function", f"{len(hard_relations)} hard initial relation(s) = "
         f"{host.relation_evidence.status.value}")
    host.ledger.record(host.relation_evidence)

    space_ok, space_measured, space_detail, space_witnesses = _space_conditions(scene, host.spec)
    step("validate_function",
         f"function.space_conditions = {'passed' if space_ok else 'failed'}")
    host.ledger.record(evidence(
        "function.space_conditions",
        EvidenceStatus.PASS if space_ok else EvidenceStatus.FAIL,
        scene_version,
        method="robot_centre_space_reaches_work_zone",
        measured=space_measured, unit="m", profile="engineering_v1",
        detail=space_detail, witnesses=space_witnesses,
    ))

    # The passage gate asks whether the robot can get *into* the room. This asks whether it can
    # get to the thing the task is about once it is there -- the question a visual critic was
    # answering alone, because no gate had a robot to measure against.
    reach_ok, reach_measured, reach_detail, reach_witnesses = _task_reachable(scene, host.spec)
    step("validate_function",
         f"function.task_reachable = {'passed' if reach_ok else 'failed'}")
    host.ledger.record(evidence(
        "function.task_reachable",
        EvidenceStatus.PASS if reach_ok else EvidenceStatus.FAIL,
        scene_version,
        method="reach_proxy_standoff_solve",
        measured=reach_measured, unit="directions", profile="engineering_v1",
        detail=reach_detail, witnesses=reach_witnesses,
    ))
