# SPDX-License-Identifier: Apache-2.0
"""Generated background assembly into the same measured scene instances."""

from __future__ import annotations
import json
from pathlib import Path
from rfw_w.scenes.contracts.hashing import content_hash
from typing import Any
from rfw_w.scenes.contracts.units import Quat
from rfw_w.scenes.contracts.units import Vec3
from rfw_w.scenes.solvers.shell_providers import ShellProviderError
from rfw_w.scenes.assembly.geometry import _assembly_refusal
from rfw_w.scenes.assembly.placement import Furnished
from rfw_w.scenes.runtime import step

def assemble_generated(host, scene: dict) -> dict:
    """Give the kept generated objects a place in this delivery, or say why they have none.

    The gap this closes: SPEC-011's pool decides what may stay, and everything downstream of it
    measured that room -- but the delivery never *carried* the objects, so the scene a worker
    settled and a critique looked at was the room without them. The delivery said so in its own
    `background.delivery_note`.

    One append per object, into the same list every retrieved piece goes through, so
    `assets_used`, `placement.instances`, the worker request, the scene graph and the geometry
    gates all see generated furniture at once. A second code path for generated objects is what
    this deliberately does not have: `geometry.no_penetration` judging a generated shelf against
    the retrieved periphery is only meaningful if both are the same kind of record.

    Only the opt-in path runs this. The default path must be byte-identical to what it was, and
    a delivery without the flag gets `{}` rather than an empty assembly.
    """
    if not getattr(host.args, "background_generation", False):
        return {}
    from rfw_w.scenes.assets.generated_furniture import (
        kept_objects, prepare_asset, read_assembly,
    )
    from rfw_w.scenes.contracts.background_constraints import generated_entity_id
    from rfw_w.scenes.solvers.shell_providers import export_generated_furniture

    build = host._shell_build_lazy(scene.get("room_plan"))
    if isinstance(build, ShellProviderError):
        # Written rather than returned in silence: `background_block` reads the record from disk,
        # so an assembly that could not run has to leave the *reason* where the delivery can
        # carry it -- otherwise the delivery says "asked for generation, carries nothing", and
        # the audit cannot tell that from an assembly nobody ran.
        return write_assembly(host, {"status": "not_run",
                                     "reason": f"the shell build failed: {build}"})
    try:
        assembly_path = export_generated_furniture(build)
    except ShellProviderError as error:
        # Named, not raised: a delivery whose generated objects could not be exported is still a
        # delivery, and `BACKGROUND_ASSEMBLY_MISSING`/`BACKGROUND_ASSEMBLY_FAILED` is the gap
        # the audit reports for it.
        step("background", f"generated furniture not assembled: {error}")
        return write_assembly(host, {"status": "failed", "reason": str(error)})
    payload = read_assembly(assembly_path)
    kept = kept_objects(payload)
    summary: dict[str, Any] = {
        "status": "assembled" if kept else "no-kept-objects",
        "assembly": Path(assembly_path).resolve().relative_to(host.out.resolve()).as_posix(),
        "assembly_digest": str(payload.get("digest", "")),
        "blend_sha256": str(payload.get("blend_sha256", "")),
        "upstream_pin": str(payload.get("upstream_pin", "")),
        "kept_in_ledger": int(payload.get("kept_in_ledger", 0)),
        "exported": int(payload.get("exported", 0)),
        "skipped": list(payload.get("skipped", ())),
        "instances": [],
    }
    raw_dir = Path(assembly_path).parent / "raw"
    ordinals: dict[str, int] = {}
    # The room the delivery already places, and the same rule the gate will apply to the whole
    # set. A generated object that would share space with a *retrieved* piece is refused here
    # and the refusal is recorded with the number and the piece it collided with: measured on
    # the first assembled delivery, one monitor shared 0.0301 m2 of plan area with the floor
    # lamp, and the run then ended with `geometry.no_penetration` red and no settle.
    #
    # Why here and not in the build's pool: the pool judges against the task's reservations and
    # the declared gaps, and it runs *inside* the build -- while the retrieval placement is a
    # delivery-level fact. Refusing after the build means the build keeps the room it built
    # (SPEC-011's ledger is unchanged) and the delivery says what it carries and what it left
    # behind; the alternative, handing the pool the retrieval footprints, is a bundle change
    # recorded in the PLAN as the next step rather than a thing this delivery silently assumed.
    refused: list[dict[str, Any]] = []
    for record in kept:
        category = str(record.get("category", ""))
        prepared = prepare_asset(record, raw_dir / str(record["raw_obj"]),
                                 host.out / "prepared_assets",
                                 upstream_pin=str(payload.get("upstream_pin", "")))
        rotation_wxyz = tuple(Quat.from_yaw(float(record["yaw_rad"])).to_list())
        position = Vec3(*[float(value) for value in record["position_m"]])
        refusal = _assembly_refusal(
            scene, dimensions=prepared.dimensions_m, position=position,
            rotation_wxyz=rotation_wxyz,
            wall_thickness_m=float(scene.get("wall_thickness_m", 0.0)))
        if refusal:
            refused.append({
                "generated_name": str(record.get("name", "")), "category": category,
                "asset_id": prepared.asset_key.asset_id, "reason": refusal,
            })
            step("background", f"refused {record.get('name')} at assembly: {refusal}")
            continue
        ordinals[category] = ordinals.get(category, 0) + 1
        entity_id = generated_entity_id(category, ordinals[category])
        entry = Furnished(
            instance_id=entity_id, entity_id=entity_id,
            visual_uri=str(prepared.visual_obj), collision_uri=str(prepared.collision_obj),
            position_m=Vec3(*[float(value) for value in record["position_m"]]),
            yaw_rad=float(record["yaw_rad"]),
            mass_kg=prepared.mass_kg, dynamic=False, normalisation="none",
            dimensions_m=prepared.dimensions_m,
            provider=prepared.asset_key.provider,
            library_version=prepared.asset_key.library_version,
            asset_id=prepared.asset_key.asset_id,
            mass_provenance=prepared.provenance.status.value,
            rotation_wxyz=tuple(Quat.from_yaw(float(record["yaw_rad"])).to_list()),
            origin_at_base=True,
        )
        scene["instances"].append(entry)
        summary["instances"].append({
            "instance_id": entity_id, "category": category,
            "generated_name": str(record.get("name", "")),
            "asset_id": prepared.asset_key.asset_id,
            "dimensions_m": [round(value, 4) for value in prepared.dimensions_m.to_list()],
            "mass_kg": round(prepared.mass_kg, 3),
            "mass_provenance": prepared.provenance.status.value,
            "tilt_deg": float(record.get("tilt_deg", 0.0)),
        })
    tolerance = float(payload.get("tilt_tolerance_deg", 1.0))
    if kept and not summary["instances"]:
        summary["status"] = "all-refused-at-assembly"
    summary["tilted_beyond_tolerance"] = sorted(
        item["instance_id"] for item in summary["instances"]
        if abs(float(item["tilt_deg"])) > tolerance
    )
    summary["categories"] = sorted({item["category"] for item in summary["instances"]})
    # The refusals are part of the delivery's account, not a log line: a reader asking "why are
    # there five monitors in the ledger and four in the scene" has to find the answer beside the
    # objects themselves.
    summary["refused_at_assembly"] = refused
    summary["refused_categories"] = sorted({item["category"] for item in refused})
    step("background", f"assembled {len(summary['instances'])} generated object(s) as "
                       f"instances: {summary['categories']} "
                       f"({len(refused)} refused at assembly)")
    return write_assembly(host, summary)


def write_assembly(host, summary: dict) -> dict:
    """Persist one immutable assembly record for this plan, geometry and run attempt."""
    attempt = getattr(host, '_assembly_attempt', 0) + 1
    host._assembly_attempt = attempt
    inputs = {
        'spec_hash': content_hash(host.spec.to_dict()),
        'plan_hash': content_hash(host.plan.to_dict()),
        'room_plan_hash': host.scene['room_plan'].content_hash,
        'generation_seed': host.generation_seed,
        'instances': [{
            'instance_id': item.instance_id,
            'asset_id': item.asset_id,
            'position_m': item.position_m.to_list(),
            'rotation_wxyz': item.rotation_wxyz,
            'dimensions_m': item.dimensions_m.to_list(),
        } for item in host.scene['instances']],
    }
    generation_hash = content_hash(inputs)
    directory = host.out / 'generated_furniture' / generation_hash / f'attempt_{attempt:03d}'
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(summary)
    record['generation_hash'] = generation_hash
    record['attempt'] = attempt
    record['artifact_path'] = (directory / 'assembly.json').relative_to(host.out).as_posix()
    (directory / 'assembly.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return record
