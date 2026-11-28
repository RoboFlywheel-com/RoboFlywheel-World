# SPDX-License-Identifier: Apache-2.0
"""Version-bound delivery composition from one scene run."""

from __future__ import annotations
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence
from rfw_w.scenes.contracts.delivery import DeliveryManifest, RoomFitRecord, RoomSummary, RunBudget, SceneGraphRecord, VlmRecord
from rfw_w.scenes.contracts.scene_spec import SceneScope, scope_of, relation_closure_ids
from rfw_w.scenes.contracts.tool import EvidenceStatus
from rfw_w.scenes.contracts.background_constraints import INJECTION_FILENAME
from rfw_w.scenes.solvers.shell_providers import BACKGROUND_GENERATION_POLICY, ShellProviderError, background_block
from rfw_w.scenes.assembly.lighting import lighting_block_for
from rfw_w.scenes.delivery.evidence import door_cut_from_report
from rfw_w.scenes.runtime import step, vlm_config


def _loop_record(loop_run: Any, diagnostics: Sequence[str]) -> dict | None:
    """The loop's own report, with the build's purity diagnostics folded into its notes.

    They ride here rather than in a field of their own because they are notes about the run -- why
    a restore landed where it did -- and `loop` is already the block a reader goes to for that. A
    run with no diagnostics is unchanged.
    """
    if loop_run is None:
        return None
    record = loop_run.to_dict()
    if diagnostics:
        record["notes"] = list(record.get("notes", ())) + list(diagnostics)
    return record


def _placement_records(scene: dict) -> dict:
    """Every placed instance's pose, as the manifest and the independent verifier need it.

    The whole rotation and the origin convention are recorded, not just a yaw: a yaw cannot describe
    a piece that had to be stood up, and the geometry gates replay these poses rather than
    re-deriving them.
    """
    return {
        "instances": {
            entry.instance_id: {
                "position_m": [round(value, 4) for value in entry.position_m.to_list()],
                "yaw_rad": round(entry.yaw_rad, 4),
                "rotation_wxyz": [round(value, 6) for value in (
                    entry.rotation_wxyz
                    or (math.cos(entry.yaw_rad / 2), 0.0, 0.0, math.sin(entry.yaw_rad / 2))
                )],
                "origin_at_base": entry.origin_at_base,
                "dynamic": entry.dynamic,
            }
            for entry in scene["instances"]
        },
        "support_surface": {
            "plane_z_m": scene["support_top"].plane_z_m,
            "provenance": scene["support_top"].provenance.value,
            "polygon_xy": scene["support_top"].polygon.to_list(),
        },
    }


def _room_fit_record(fitted: Any) -> RoomFitRecord | None:
    """The room-versus-cell arithmetic, as the manifest records it.

    ``None`` when the scene carries no fit. There is no such path in this build, but a manifest
    assembled by hand or by an older entry point must say "no fit" rather than carry a fit of
    zeroes, which would read as a room exactly the size of the task it holds.
    """
    if fitted is None:
        return None
    p_min_x, p_min_y, p_max_x, p_max_y = fitted.protected_in_room.bounds()
    standoff = fitted.wall_standoff_m
    protected_w, protected_d = p_max_x - p_min_x, p_max_y - p_min_y
    return RoomFitRecord(
        protected_width_m=round(protected_w, 6),
        protected_depth_m=round(protected_d, 6),
        needed_width_m=round(protected_w + 2.0 * standoff, 6),
        needed_depth_m=round(protected_d + 2.0 * standoff, 6),
        sampled_width_m=round(fitted.options.width_m or 0.0, 6),
        sampled_depth_m=round(fitted.options.depth_m or 0.0, 6),
        cell_x_m=round(fitted.cell_translation_xy[0], 6),
        cell_y_m=round(fitted.cell_translation_xy[1], 6),
        wall_standoff_m=round(standoff, 6),
        opening_attempts=fitted.opening_attempts,
        size_margin_x_m=round((fitted.size_margin_m or (0.0, 0.0))[0], 6),
        size_margin_y_m=round((fitted.size_margin_m or (0.0, 0.0))[1], 6),
        shape_family=fitted.options.family.value,
        shape_fallback=fitted.shape_fallback or "",
        door_bearing_deg=fitted.door_bearing_deg,
        door_arc_margin_deg=fitted.door_arc_margin_deg,
        door_in_approach_arc=fitted.door_in_approach_arc,
    )


def resolve_shell_uri(host: Any, scene: dict) -> str | None:
    """The referenced shell this delivery must publish, resolved again at manifest time.

    Read off the scene dict, it was *silently* absent on any run the loop rolled back: a rollback
    rebuilds the scene, and a rebuilt scene carries no `shell_uri` (SPEC-005's own record). The fix
    when that was first measured covered the backend path; the manifest's copy of the evidence was
    left reading the dict. Enabling the Visual Critic exposed it -- a failed verdict is a rollback --
    and the 2026-09-24 backfill's three deliveries were all rendered against the referenced shell,
    said so in their gate evidence, and published no `shell_reference.json`. Only the independent
    verifier noticed, which is the wrong place to find it: a delivery that omits its own evidence
    should say so by name.

    `prepare_shell_reference` is idempotent and cached per room plan, so resolving here is the same
    exported file the worker was handed, not a second export. If it cannot be resolved the backend
    has already recorded `backend.isaac.load` FAIL, and this returns None so nothing is written.
    """
    if getattr(getattr(host, "args", None), "shell_provider", "builtin") == "infinigen":
        try:
            host.prepare_shell_reference()
        except (ShellProviderError, ValueError) as error:
            step("manifest", f"shell reference could not be resolved: {error}")
    return scene.get("shell_uri")



def write_delivery(host, *, args, compilation: dict, loop_run, budget, started: float) -> int:
    """Write a delivery solely from the final host state and recorded measurements."""
    out, spec, resolution = host.out, host.spec, host._resolution()
    ledger = host.ledger
    scene = host.scene
    scene_version = host.version
    scene_ir_path = out / "scene_ir.json"
    scene_ir_path.write_text(
        json.dumps(scene["scene_ir"].to_dict(), indent=2, sort_keys=True) + "\n"
    )
    worker_report = host.worker_report
    critique_record = {} if host.critique is None else {"critique": host.critique.to_dict()}
    vlm_model = args.vlm_model or os.environ.get("PVSCENE_VLM_MODEL", "")
    if vlm_model and not args.skip_isaac:
        configured = vlm_config(vlm_model)
        requested_vlm = VlmRecord(
            base_url=configured.base_url, model=configured.model,
            api_key_env=configured.api_key_env, wire=configured.wire,
            credential_present=configured.credential_present(),
        )
    else:
        # One shape, not two. This used to be `{"model": "", "configured": False}` when no model was
        # named and a full config dict when one was -- two different documents under one key.
        requested_vlm = VlmRecord(
            base_url="", model="", api_key_env="", wire="", credential_present=False,
        )
    agent_model = args.agent_model or os.environ.get("PVSCENE_AGENT_MODEL", "")
    if agent_model and not args.no_replan and not args.no_loop:
        configured_agent = vlm_config(agent_model)
        requested_agent = VlmRecord(
            base_url=configured_agent.base_url,
            model=configured_agent.model,
            api_key_env=configured_agent.api_key_env,
            wire=configured_agent.wire,
            credential_present=configured_agent.credential_present(),
        )
    else:
        requested_agent = VlmRecord(
            base_url="", model="", api_key_env="", wire="", credential_present=False,
        )

    not_run = sorted(
        item.gate_id for item in ledger.entries.values() if item.status is EvidenceStatus.NOT_RUN
    )
    unsupported = sorted(
        item.gate_id for item in ledger.entries.values() if item.status is EvidenceStatus.UNSUPPORTED
    )

    shell_uri = resolve_shell_uri(host, scene)
    if shell_uri:
        from rfw_w.scenes.workers.isaac.shell_reference import inspect_shell
        shell_record = inspect_shell(shell_uri, delivery=out)
        shell_record["uri"] = str(Path(shell_uri).relative_to(out))
        shell_record["scene_version"] = host.version
        (out / "shell_reference.json").write_text(json.dumps(shell_record, indent=2) + "\n")

    manifest = DeliveryManifest(
        scene_id=spec.scene_id,
        prompt=spec.prompt,
        seed=spec.seed,
        # Recorded rather than left to be inferred: a verifier that deduced the backends from which
        # backend gates exist would treat a declared out-of-scope gap (mujoco) as a requested
        # backend and conclude the delivery failed.
        requested_backends=tuple(spec.requested_backends),
        periphery_named_by_relations=tuple(
            e.id for e in spec.entities
            if scope_of(e) is SceneScope.PERIPHERY and e.id in relation_closure_ids(spec)[0]
        ),
        scene_version=scene_version,
        # The resolution whole, rather than restated by hand under names of this file's own. The
        # hand-written copy used `requested_readiness`/`achieved_readiness` while the type it was
        # copying used `requested`/`achieved`, which is how two names for one fact start to drift.
        resolution=resolution,
        gates=tuple(ledger.entries.values()),
        not_run=tuple(not_run),
        unsupported=tuple(unsupported),
        isaac_version=worker_report.get("isaac_version", ""),
        isaac_artifact_dir=(
            str(host.worker_dir.relative_to(out)) if host.worker_dir is not None else ""
        ),
        isaac_request_path=(
            str(host.worker_request_path.relative_to(out))
            if host.worker_request_path is not None else ""
        ),
        room=RoomSummary(
            vertices=len(scene["room_polygon"].vertices),
            area_m2=round(scene["room_polygon"].area, 4),
            area_ratio_to_bounding_box=round(scene["room_polygon"].area_ratio_to_bounding_box(), 4),
            height_m=scene["shell"].height_m,
            openings=len(scene["shell"].openings),
            door_cut=door_cut_from_report(worker_report),
            fit=_room_fit_record(scene["room_fit"]),
        ),
        scene_graph=SceneGraphRecord(
            path="scene_ir.json",
            content_hash=scene["scene_ir"].content_hash(),
            state="authored",
        ),
        randomization=scene["randomization"],
        assets_used=tuple(entry.to_manifest() for entry in scene["instances"]),
        compilation=compilation,
        external_calls={
            "permission_granted": os.environ.get("PVSCENE_ALLOW_EXTERNAL_VLM") == "1",
            "model_requests_attempted": budget.model_requests_used,
            "transfer_confirmation": "not_measured",
        },
        # SPEC-011's opt-in, when this run took it: mode, gaps, policy and the build's own records.
        # SPEC-008 TASK 3: the lighting the delivery was rendered with. Read from the plan and the
        # run's own flag rather than from `self.lighting_block`, for `background`'s reason -- this
        # manifest may be composed after a repair rebuilt the scene, and a block a dict remembers
        # would report the lighting of a version nobody rendered.
        lighting=(lighting_block_for(scene, getattr(args, "lighting_profile", "mixed"))
                  if scene.get("shell") else {}),
        background=background_block(
            generation=getattr(args, "background_generation", False),
            gaps=[item.strip() for item in
                  str(getattr(args, "background_gaps", "")).split(",") if item.strip()],
            out=out, policy=BACKGROUND_GENERATION_POLICY,
            assembly=scene.get("generated_assembly") or None,
            injection_record=(out / 'shell_build' / scene['room_plan'].content_hash[:12]
                              / 'infinigen_build' / INJECTION_FILENAME
                              if scene.get('room_plan') else None)),
        furnishing={
            "instances_placed": len(scene["instances"]),
            "target_range": list(scene["furniture_target"]),
            **{key: scene[key] for key in (
                "unplaced_core", "unplaced_task_closure", "unplaced_periphery", "periphery_dropped",
            )},
            "candidate_attempts": [item.to_dict() for item in scene["failures"]],
            "target_placement": scene["target_source"],
            "room_area_m2": round(scene["room_polygon"].area, 3),
            # Added by the loop: the candidates the placement sweeps refused while producing this
            # scene. They were collected and discarded before, so a piece that could not be placed
            # said nothing about where it had been tried or how close each candidate came.
            "placement_rejections": list(scene["assembly_audit"].placement_rejections),
            # What the entity's own declared size removed from the candidate pool. Empty is the
            # healthy case; a long list means the spec asked for something the library does not
            # have, which is worth seeing before blaming the layout.
            "size_bound_rejections": list(scene["assembly_audit"].size_rejections),
            # Pieces a zone could not hold anywhere, and the floor they were given instead.
            # Distinct from a refusal: the zone was never able to hold these.
            "zone_fallbacks": list(scene["assembly_audit"].zone_fallbacks),
        },
        # The zone plan as evidence rather than an internal step: it is what the placement was
        # confined to, so a reader can check the layout against the regions it was allowed.
        zones=scene["zones"].to_dict(),
        # The plan is recorded because it is what a repair changed. Without it a reader can see that
        # the scene moved and not what decision moved it.
        plan=host.plan.to_dict(),
        placement=_placement_records(scene),
        visual_critique=critique_record or None,
        # What was *asked for*, next to what the provider echoed. `critique.model_id` comes from
        # the response body, so a gateway that rewrites the name leaves the delivery unable to say
        # which model it was -- and a run comparing two base models on one scene cannot be audited
        # at all. The key is named here and never carried.
        vlm=requested_vlm,
        agent=requested_agent,
        loop=_loop_record(loop_run, host.build_diagnostics),
        budget=RunBudget(
            elapsed_s=round(time.monotonic() - started, 3),
            deadline_s=args.deadline_s,
        ),
    )
    assembly = scene.get('generated_assembly')
    if assembly and assembly.get('artifact_path'):
        assembly_path = out / assembly['artifact_path']
        persisted = json.loads(assembly_path.read_text())
        if persisted != assembly:
            raise ValueError('final generated assembly differs from its immutable artifact')
    (out / "delivery_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "delivery_manifest": str(out / "delivery_manifest.json"),
        "requested": resolution.requested.value,
        "achieved": resolution.achieved.value,
        "outcome": None if resolution.outcome is None else resolution.outcome.value,
        "meets_request": resolution.meets_request,
        "blockers": len(resolution.blockers),
        "not_run": not_run,
        "unsupported": unsupported,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    if loop_run is not None:
        summary["loop"] = {
            "terminal": loop_run.terminal,
            "segments": loop_run.segments,
            "events": len(loop_run.events),
            "repairs": len(loop_run.ladder["attempts"]),
            "resolved_without_model": loop_run.ladder["resolved_without_model"],
            "local_recovery_rate": loop_run.ladder["local_recovery_rate"],
            "agent_intervention_rate": loop_run.agent_intervention_rate,
            "removed": list(loop_run.removed),
        }
    print(json.dumps(summary, indent=2))

    return 0 if resolution.meets_request else 1
