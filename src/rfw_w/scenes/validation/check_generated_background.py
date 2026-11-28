#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit ownership, the frozen core and the reservations a background run may not enter.

TASK 1 of the SPEC-009 PLAN, which is "the measurement tool and the failing negatives, before
anything changes behaviour". It audits **deliveries that already exist on disk**, so the baseline
is read off evidence rather than asserted:

* every requested entity and its single owner (`retrieved`, or `generated` for a delivery that
  took the opt-in);
* the task dependency closure -- the target, its supports and both endpoints of every relation --
  and whether each is actually placed;
* the core hash, recomputed from the delivery's own effective spec, and the frozen core asset
  digest, which no generated object may change;
* the XY overlap of every placed footprint with the task's protected region and with the floor
  reservations (circulation, door sweep) -- generated instances included, since they enter the
  delivery as ordinary instances;
* the six upstream background stages, in the five states SPEC-009 requires, from three kinds of
  evidence: the command we ran (gin overrides), the build's own log, and solve_state.json;
* for a generated delivery: gap alignment (ADR-009 decision 11), the stage's allowance, and that
  every object the delivery *claims* is a prepared asset with collision mesh on disk.

The stage half is why this tool exists rather than a table. `compose_indoors` runs
`solve_large`/`solve_medium`/`solve_small` and *solves furniture placeholders for them* even with
`populate_assets` switched off, and the export discards the result -- so the baseline question is
not "does Infinigen support background generation" but "what is already being solved, at what
cost, and where does it go".

`--matrix` is the SPEC's acceptance entry point (3 types x 2 seeds, each with Isaac settle and an
independent verification, plus its negatives). A case whose delivery does not exist is reported
`not_run` with its reason, never passed; and a case may declare the named failures its delivery
*must* show, so an honest, recorded boundary is not mistaken for a pass.

A pass on the retrieved path is not a SPEC-009 acceptance on its own: the report says
`spec_accepted: false` and leaves the human checklist `not_run` until the matrix has run and a
person has looked at the pictures.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path.cwd()

from rfw_w.scenes.contracts.background import (
    BACKGROUND_VERSION,
    EVIDENCE_LOG,
    EVIDENCE_NONE,
    EVIDENCE_OVERRIDE,
    RESERVATION_KEEPOUT,
    RESERVATION_PROTECTED,
    STAGES,
    BackgroundManifest,
    BackgroundRequest,
    OwnedEntity,
    StageRecord,
    core_entity_ids,
    enabled_param,
    periphery_entity_ids,
    retrieved_ownership,
)
from rfw_w.scenes.contracts.background_constraints import (
    BUNDLE_FILENAME,
    GAP_UNFILLED_KINDS,
    GENERATED_OBJECTS_MAX_PER_STAGE,
    GENERATED_PROVIDER,
    INJECTION_FILENAME,
    REALISING_STAGE,
    generated_category,
)
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.geometry.polygon import Polygon, intersect_polygons
from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.room import fit_room_around_cell
from rfw_w.scenes.solvers.room_plan import plan_from_fitted

#: Both reservations are *floor* regions (`solvers/zones.py`): the protected region is the task
#: cell's own footprint plus the way in, and circulation is a floor area sized to the robot. What
#: has to stay clear is the band the cell occupies -- from the floor up to the task support's own
#: top -- so an object is measured only when its lowest point falls inside that band. The bound
#: comes from the plan's own `core.support_top_z_m`, not from a constant, and the two cases that
#: decided it are measured: the office's wall art has its base at 1.450 m over a 0.750 m desk, and
#: the bedroom's pendant at 2.528 m over a 0.575 m nightstand -- both hang over the reservation.
#: The rug that made this check necessary in the first place (`run_m1_acceptance.py:1266`) had its
#: base at 0.006 m, which is *in* the band, and that is the intrusion this reports.

#: `[name] finished in H:MM:SS.ffffff` -- the stage runner's own line, and the only place a
#: duration exists.
_STAGE_FINISHED = re.compile(r"\| \[(?P<name>\w+)\] finished in (?P<time>[\d:.]+)")
_SKIPPED = re.compile(r"Skipping run_stage\((?P<name>\w+)[^)]*\)")
_OVERRIDE = re.compile(r"compose_indoors\.(?P<stage>\w+)_enabled=(?P<value>True|False)")


class AuditError(RuntimeError):
    """The audit cannot be run on the input it was given, named rather than reported as a pass."""


@dataclass
class RunFiles:
    """One delivery's on-disk evidence, loaded but not yet judged."""

    run_dir: Path
    spec: SceneSpec
    delivery: Mapping[str, Any]
    scene: Mapping[str, Any]
    generation: Mapping[str, Any]
    build_dir: Path | None
    plan_core_hash: str = ""
    support_top_z_m: float = 0.0
    protected_xy: tuple[tuple[float, float], ...] = ()
    room_polygon: Polygon | None = None
    keepouts: tuple[Polygon, ...] = ()
    replay_note: str = "not_run"
    stage_log: str = ""
    command: str = ""
    placeholders: int = 0
    real_placeholders: int = 0
    placeholder_generators: tuple[str, ...] = ()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise AuditError(f"BACKGROUND_AUDIT_INPUT_MISSING({path.name}): {path}")
    return json.loads(path.read_text())


def _build_dir(run_dir: Path) -> Path | None:
    roots = sorted((run_dir / 'shell_build').glob('*/infinigen_build'))
    return roots[0] if roots else None


def load_run(run_dir: Path) -> RunFiles:
    """Load a delivery and replay its room, so the audit reads the same room the run did.

    The room is re-derived from the effective spec and the recorded room seed, the way
    `check_room_intent.py` does for SPEC-006. The protection region and the core hash exist nowhere
    in a delivery manifest, and inventing a second copy of them would be a second source of truth
    for a number whose whole job is to prove the first one did not move.
    """
    spec = SceneSpec.from_dict(_read_json(run_dir / 'effective_spec.json'))
    delivery = _read_json(run_dir / 'delivery_manifest.json')
    scene = _read_json(run_dir / 'scene_ir.json')
    generation = _read_json(run_dir / 'generation_attempt_001.json')
    run = RunFiles(run_dir=run_dir, spec=spec, delivery=delivery, scene=scene,
                   generation=generation, build_dir=_build_dir(run_dir))
    core = solve_cell(spec, seed=spec.seed)
    fitted = fit_room_around_cell(spec, core, room_seed=int(generation['room_seed']))
    plan = plan_from_fitted(spec, core, fitted, room_seed=int(generation['room_seed']))
    run.plan_core_hash = plan.core.core_hash
    run.support_top_z_m = plan.core.support_top_z_m
    run.protected_xy = plan.core.protected_in_room_xy
    # The replay is only evidence about *this* delivery if it is the same room. The manifest
    # publishes the room polygon the zones were planned against, so the two are compared rather
    # than assumed equal.
    zones = delivery.get('zones', {})
    published = zones.get('room_polygon_xy')
    if published:
        replayed = Polygon(plan.rooms[0].polygon_xy)
        if _ring(replayed) != _ring(Polygon(published)):
            raise AuditError('BACKGROUND_ROOM_REPLAY_MISMATCH: the replayed room is not the '
                             'delivered room, so its protected region is not evidence either')
        run.replay_note = 'matched'
        run.room_polygon = replayed
    else:
        run.replay_note = 'room_polygon_xy absent from the manifest; replay not compared'
    rings: list[Polygon] = []
    for key in ('circulation', 'door_sweep'):
        for ring in zones.get(key, ()):
            rings.append(Polygon(ring))
    run.keepouts = tuple(rings)
    if run.build_dir is not None:
        run.stage_log = (run.build_dir / 'stderr.txt').read_text(errors='replace') \
            if (run.build_dir / 'stderr.txt').is_file() else ''
        run.command = (run.build_dir / 'command.txt').read_text(errors='replace') \
            if (run.build_dir / 'command.txt').is_file() else ''
        state_path = run.build_dir / 'scene' / 'solve_state.json'
        if state_path.is_file():
            objs = list(json.loads(state_path.read_text())['objs'].values())
            generated = [obj for obj in objs if obj.get('generator')]
            run.placeholders = len(generated)
            run.real_placeholders = sum(1 for obj in objs
                                        if 'Semantics(real-placeholder)' in obj.get('tags', ()))
            run.placeholder_generators = tuple(sorted({
                str(obj['generator']).split('(')[0] for obj in generated}))
    return run


def _relative(path: Path) -> str:
    """A path as written in the report: repo-relative when it is inside the repo, else absolute."""
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if ROOT in resolved.parents else str(resolved)


def _ring(polygon: Polygon, *, places: int = 6) -> tuple[tuple[float, float], ...]:
    """A polygon rounded for comparison, so two spellings of one room are not two rooms."""
    return tuple(sorted((round(x, places), round(y, places)) for x, y in polygon.vertices))


# ---------------------------------------------------------------------------------------------
# Footprints and reservations
# ---------------------------------------------------------------------------------------------

def footprint_polygon(position_m: Sequence[float], dimensions_m: Sequence[float],
                      yaw_rad: float) -> Polygon:
    """The XY footprint of a placed box, turned by its yaw.

    A yaw-rotated rectangle rather than an axis-aligned box: the intrusion measurement is the thing
    the whole negative fixture is built on, and a bounding box of a rotated wardrobe overstates it
    by the corners it never occupied.
    """
    half_x, half_y = float(dimensions_m[0]) / 2.0, float(dimensions_m[1]) / 2.0
    cos, sin = math.cos(yaw_rad), math.sin(yaw_rad)
    corners = ((+half_x, +half_y), (-half_x, +half_y), (-half_x, -half_y), (+half_x, -half_y))
    return Polygon([(float(position_m[0]) + cos * x - sin * y,
                     float(position_m[1]) + sin * x + cos * y) for x, y in corners])


def overlap_area_m2(footprint: Polygon, regions: Sequence[Polygon]) -> float:
    """Shared area with a set of regions, summed over the disjoint pieces they are cut into."""
    return sum(piece.area for region in regions for piece in intersect_polygons(footprint, region))


def measure_overlaps(run: RunFiles, core: set[str]) -> dict[str, dict[str, float]]:
    """Per-entity overlap with each reservation, for entities that can stand in one.

    Core entities are excluded from the *measurement*, not excused after it: the protected region
    is the core's own footprint, so a support is inside it by construction and reporting the
    nightstand as an intruder would be a rule that fails every valid delivery.
    """
    protected = [Polygon(run.protected_xy)] if run.protected_xy else []
    layout = run.delivery.get('placement', {}).get('instances', {})
    dimensions = {asset['entity_id']: asset['dimensions_m']
                  for asset in run.delivery.get('assets_used', ())}
    overlaps: dict[str, dict[str, float]] = {}
    for entity_id, placed in sorted(layout.items()):
        if entity_id in core or entity_id not in dimensions:
            continue
        position = placed['position_m']
        if float(position[2]) >= run.support_top_z_m:
            continue  # above the work surface: hanging over the reservation, not in it
        footprint = footprint_polygon(position, dimensions[entity_id],
                                      float(placed.get('yaw_rad', 0.0)))
        record = {RESERVATION_PROTECTED: overlap_area_m2(footprint, protected),
                  RESERVATION_KEEPOUT: overlap_area_m2(footprint, run.keepouts)}
        if any(value > 0.0 for value in record.values()):
            overlaps[entity_id] = record
    return overlaps


def core_asset_digest(run: RunFiles, core: set[str]) -> str:
    """A digest over the core's own identity: which asset each is, and where it stands.

    This is the "core assets stay frozen" half of SPEC-009 requirement 2. The core *hash* covers
    the cell's derivation; this covers what actually reached the scene, so a generated object that
    displaced a support would change it even if the room solve never noticed.

    A generated instance can never enter this digest: `core` comes from the spec's own closure, and
    a generated entity id is not one of its entities. The exclusion is by construction rather than
    by a filter here, which is the only version of it a reader can check.
    """
    payload = []
    for instance in sorted(run.scene.get('instances', ()), key=lambda item: item['entity_id']):
        if instance['entity_id'] not in core:
            continue
        payload.append({
            'entity_id': instance['entity_id'],
            'asset_ref': instance['asset_ref'],
            'world_transform': instance['world_transform'],
            'parent_id': instance.get('parent_id'),
        })
    return content_hash(payload)


# ---------------------------------------------------------------------------------------------
# The generated half: what the delivery carries, and whether it matches what was declared
# ---------------------------------------------------------------------------------------------

def generated_rows(run: RunFiles) -> list[dict[str, Any]]:
    """The generated instances this delivery carries, read from its own manifest.

    A generated object reaches a delivery the same way a retrieved one does (`assets_used` plus a
    `placement.instances` row), which is the point of the assembly step: there is no second kind of
    record for the audit to learn. What makes it discoverable is its entity id --
    `generated_<category>_<ordinal>` (`background_constraints.generated_entity_id`) -- so the
    category a reader needs for gap alignment is *inside* the id rather than in a parallel field.
    """
    rows = []
    for asset in run.delivery.get('assets_used', ()):
        entity_id = str(asset.get('entity_id', ''))
        category = generated_category(entity_id)
        if category is None:
            continue
        rows.append({
            'entity_id': entity_id, 'category': category,
            'provider': str(asset.get('provider', '')), 'asset_id': str(asset.get('asset_id', '')),
            'dimensions_m': asset.get('dimensions_m'),
        })
    return sorted(rows, key=lambda row: row['entity_id'])


def assembly_gaps(run: RunFiles) -> list[str]:
    """Every way a delivery can *say* it carries generated objects and not carry them.

    Three claims have to agree, and each has a failure mode a reader would otherwise have to take
    on trust: an `assets_used` row nothing places (`BACKGROUND_ASSEMBLY_MISMATCH` -- the inventory
    claims an object the scene does not have), an instance the inventory never mentions
    (`BACKGROUND_INSTANCE_UNRECORDED` -- the delivery carries what the manifest cannot show), and a
    generated asset missing the OBJ or the collision mesh on disk
    (`BACKGROUND_GENERATED_ASSET_MISSING` -- SPEC-009 requirement 5: a delivery is self-contained
    and a static piece still needs stable collision). Each is named with the entity, so the repair
    is obvious.
    """
    rows = generated_rows(run)
    if not rows:
        # A delivery that asked for generation and carries none. Two innocent explanations -- the
        # build kept nothing, or every kept object was refused at assembly -- and one that is not:
        # the assembly never ran, so objects the pool kept are in no delivery at all. Told apart by
        # the records rather than by inference, and named only in the case that is a defect.
        block = run.delivery.get('background', {}) or {}
        if not block.get('generation'):
            return []
        assembly = block.get('assembly') or {}
        if assembly.get('status') in ('assembled', 'all-refused-at-assembly', 'no-kept-objects'):
            return []
        kept = _kept_in_ledger(run)
        if kept:
            return [f'BACKGROUND_ASSEMBLY_MISSING({kept} kept in the pool ledger, none delivered, '
                    f'none refused)']
        if not assembly:
            return ['BACKGROUND_ASSEMBLY_MISSING(the run asked for generation and the delivery '
                    'carries no assembly record)']
        return []
    instances = {str(item['entity_id']) for item in run.scene.get('instances', ())
                 if generated_category(str(item['entity_id'])) is not None}
    placed = {str(row['entity_id']) for row in rows}
    gaps = [f'BACKGROUND_ASSEMBLY_MISMATCH({entity_id})' for entity_id in sorted(placed - instances)]
    gaps += [f'BACKGROUND_INSTANCE_UNRECORDED({entity_id})' for entity_id in sorted(instances - placed)]
    for row in rows:
        directory = run.run_dir / 'prepared_assets' / row['asset_id']
        for name in (f"{row['asset_id']}.obj", f"{row['asset_id']}_collision.obj"):
            if not (directory / name).is_file():
                gaps.append(f'BACKGROUND_GENERATED_ASSET_MISSING({row["entity_id"]},{name})')
    return gaps


def declared_gaps(run: RunFiles) -> tuple[tuple[str, ...], str]:
    """The categories this room declared as gaps, and where that declaration was read.

    The build's own request (`reservations.json`) is the primary evidence: it is what the pool
    judged against, written before anything was generated. The delivery's `background` block is a
    copy of it, and the two are compared rather than either being trusted -- a delivery whose block
    declared work the build was never given would be a fixture passing itself.
    """
    if run.build_dir is None:
        return (), 'no build directory'
    bundle_path = run.build_dir / BUNDLE_FILENAME
    if not bundle_path.is_file():
        return (), f'{BUNDLE_FILENAME} absent from the build'
    bundle = json.loads(bundle_path.read_text())
    declared = tuple(sorted(str(item) for item in bundle.get('gaps', ())))
    block = run.delivery.get('background', {}) or {}
    if block and tuple(sorted(str(item) for item in block.get('gaps', ()))) != declared:
        raise AuditError(
            'BACKGROUND_GAP_DECLARATION_MISMATCH: the delivery block declares '
            f'{sorted(block.get("gaps", ()))} and the build was given {list(declared)}'
        )
    return declared, f'{BUNDLE_FILENAME} (the request the build was handed)'


def gap_reasons(run: RunFiles, gaps: Sequence[str]) -> dict[str, str]:
    """Why each declared-but-unfilled gap is empty, from the two records that know.

    The kinds are the ones `background_constraints.GAP_UNFILLED_KINDS` names, and they are derived
    rather than recorded because the records already carry the facts they rest on. Two records,
    because the two halves of the branch are recorded in different places: the pool's ledger says
    what the *build* did with each category (a rejected candidate was proposed and refused, silence
    is `not-proposed-by-this-run`, no ledger at all is `not-in-the-room`), and the delivery's own
    assembly record says what happened to the objects the pool *kept* -- measured on office seed 11,
    where every kept `book_stack` was refused at assembly and the category would otherwise read as
    unfilled with no reason.
    """
    if run.build_dir is None:
        return {category: 'not-in-the-room' for category in gaps}
    record_path = run.build_dir / INJECTION_FILENAME
    if not record_path.is_file():
        return {category: 'not-in-the-room' for category in gaps}
    pool = json.loads(record_path.read_text()).get('rejection_pool') or {}
    candidates = list(pool.get('candidates', ()))
    delivered = {row['category'] for row in generated_rows(run)}
    refused = {str(item.get('category', '')) for item in
               (run.delivery.get('background', {}).get('assembly', {}) or {}).get(
                   'refused_at_assembly', ())}
    reasons: dict[str, str] = {}
    for category in gaps:
        if category in delivered:
            continue  # filled, not a gap any more
        own = [item for item in candidates if str(item.get('category', '')) == category]
        kept = [item for item in own if item.get('outcome') == 'kept']
        if kept and category in refused:
            reasons[category] = 'refused-at-assembly'
        elif kept:
            # Kept by the pool, carried by nobody, refused by nobody: the assembly's absence. The
            # delivery-side name for it is BACKGROUND_ASSEMBLY_MISSING; this is the gap's own name.
            reasons[category] = 'kept-but-not-delivered'
        elif own:
            reasons[category] = 'proposed-but-rejected'
        else:
            reasons[category] = 'not-proposed-by-this-run'
    return reasons


def alignment_gaps(manifest: BackgroundManifest) -> list[str]:
    """ADR-009 decision 11, judged on the delivery: every generated category is a gap, every gap is
    accounted for.

    The build's own pool enforces the first half while it runs; this is the delivery-side statement
    of the same rule, and it is not redundant -- the pool judges *candidates*, and this judges what
    actually reached the delivery. A gap that is neither filled nor explained is the second half:
    `BACKGROUND_GAP_UNFILLED(category,reason)` where the reason is one of the named kinds, so a
    reader can tell "the sampler never proposed it" from "we have no idea".
    """
    if not manifest.gaps:
        # A room that declared no gaps has made no claim about what generation may produce, and the
        # alignment rule is about the declaration. Saying so is the difference between a rule that
        # is absent and a rule that was satisfied.
        return []
    generated = manifest.entities_owned_by('generated')
    filled: set[str] = set()
    gaps: list[str] = []
    for entity in generated:
        category = generated_category(entity.entity_id) or ''
        if category in manifest.gaps:
            filled.add(category)
        else:
            gaps.append(f'BACKGROUND_GENERATED_CATEGORY_NOT_A_GAP({entity.entity_id},'
                        f'{category or "unnamed"})')
    for category in manifest.gaps:
        if category in filled:
            continue
        reason = str(manifest.gap_reasons.get(category, ''))
        if reason not in GAP_UNFILLED_KINDS:
            gaps.append(f'BACKGROUND_GAP_UNFILLED({category},'
                        f'{reason or "no-fill-and-no-reason-recorded"})')
    return gaps


def audited_gaps(manifest: BackgroundManifest) -> list[str]:
    """Every named failure of one manifest: the contract's own rules plus the delivery-side ones."""
    return list(manifest.audit()) + alignment_gaps(manifest)


# ---------------------------------------------------------------------------------------------
# The stage report
# ---------------------------------------------------------------------------------------------

def stage_records(run: RunFiles) -> tuple[StageRecord, ...]:
    """The six background stages, each with the evidence its state rests on.

    Three sources, kept apart because they say different things: the gin overrides in the run's own
    command (`command.txt`) are evidence about what *we* asked for; a `finished in` line in
    `stderr.txt` is evidence about what the build did; and silence is evidence of neither.
    `populate_assets` is the case that matters -- upstream's stage runner says nothing at all about
    a stage it is told not to run, so its state is only knowable from the override we passed.
    """
    overrides = {match.group('stage'): match.group('value') == 'True'
                 for match in _OVERRIDE.finditer(run.command)}
    # A stage that *failed* is not a stage nobody wrote down. The build's own record says so in its
    # `injection_reason` ("<stage> failed"), and its pool record names why -- measured on bedroom
    # seed 11, where `populate_assets` died on a missing Blender addon and the delivery would
    # otherwise have reported the stage as unreported, which reads as "we did not look".
    failed_stage, failure_reason = "", ""
    if run.build_dir is not None:
        record_path = run.build_dir / INJECTION_FILENAME
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            stated = str(record.get('injection_reason', ''))
            pool = record.get('rejection_pool') or {}
            for stage in STAGES:
                if f'{stage} failed' in stated:
                    failed_stage = stage
                    failure_reason = str(pool.get('reason') or stated)
                    break
    durations: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for line in run.stage_log.splitlines():
        finished = _STAGE_FINISHED.search(line)
        if finished and finished.group('name') in STAGES:
            hours, minutes, seconds = finished.group('time').split(':')
            durations[finished.group('name')] = (int(hours) * 3600 + int(minutes) * 60
                                                 + float(seconds))
        for match in _SKIPPED.finditer(line):
            skipped[match.group('name')] = line.split('|')[-1].strip()
    records = []
    for stage in STAGES:
        if stage == failed_stage:
            records.append(StageRecord(stage=stage, state='failed', evidence=EVIDENCE_LOG,
                                       reason=failure_reason))
        elif stage in durations:
            records.append(StageRecord(stage=stage, state='executed', evidence=EVIDENCE_LOG,
                                       duration_s=round(durations[stage], 3)))
        elif stage in overrides and not overrides[stage]:
            records.append(StageRecord(
                stage=stage, state='not_run', evidence=EVIDENCE_OVERRIDE,
                reason=f'our own command passed {enabled_param(stage)}=False'))
        elif stage in skipped:
            records.append(StageRecord(stage=stage, state='not_run', evidence=EVIDENCE_LOG,
                                       reason=skipped[stage]))
        else:
            records.append(StageRecord(
                stage=stage, state='not_run', evidence=EVIDENCE_NONE,
                reason='absent from the build log and named by no override: the build says '
                       'nothing about this stage, which is not the same as it not running'))
    return tuple(records)


def request_for(run: RunFiles, core: set[str], *, mode: str,
                gaps: Sequence[str]) -> BackgroundRequest:
    """What this delivery's run asked of the background branch.

    Read from the delivery rather than assumed: a `retrieved` delivery asked for no generation and
    has no allowance anywhere, while a delivery carrying generated instances asked for the
    `populate_assets` allowance its own build recorded. Recorded as a *request* rather than
    described in prose so the report and SPEC-009's matrix read the same fields -- and so that the
    day a run asks for something else, the change shows up as a value rather than as a sentence
    somebody has to notice.
    """
    budget = dict(GENERATED_OBJECTS_MAX_PER_STAGE) if mode == 'generated' else {}
    return BackgroundRequest(
        scene_id=str(run.delivery.get('scene_id', run.run_dir.name)),
        seed=int(run.delivery.get('seed', 0)), mode=mode, stages=STAGES, budget=budget,
        core_hash=run.plan_core_hash, required_entity_ids=tuple(sorted(core)),
        protected_polygon_xy=run.protected_xy,
        keepout_polygons_xy=tuple(tuple(region.vertices) for region in run.keepouts))


def manifest_for(run: RunFiles, *, run_started: float) -> BackgroundManifest:
    """The manifest for one delivery: who owns what, what the stages did, what was measured."""
    core = set(core_entity_ids(run.spec))
    rows = generated_rows(run)
    # What the run *asked for*, not what it got: a delivery that requested generation and carries
    # nothing must still be judged against the gaps it declared -- and against the assembly rule,
    # which is where "asked for generation, delivered none" is named. Deriving the mode from the
    # outcome would audit exactly the deliveries the rule exists for as if they never asked.
    asked_for_generation = bool((run.delivery.get('background') or {}).get('generation'))
    mode = 'generated' if (rows or asked_for_generation) else 'retrieved'
    gaps, gap_source = declared_gaps(run) if mode == 'generated' else ((), 'not asked for')
    gap_reasons_by_category = gap_reasons(run, gaps) if mode == 'generated' else {}
    request = request_for(run, core, mode=mode, gaps=gaps)
    ownership = {entity.entity_id: entity for entity in retrieved_ownership(run.spec)}
    assets = {asset['entity_id']: asset for asset in run.delivery.get('assets_used', ())}
    entities = []
    for entity_id, owned in sorted(ownership.items()):
        asset = assets.get(entity_id, {})
        entities.append(OwnedEntity(entity_id=entity_id, owner=owned.owner, scope=owned.scope,
                                    provider=str(asset.get('provider', '')),
                                    asset_ref=str(asset.get('asset_id', ''))))
    # The generated instances, owned by the stage that realised them. The scope is `periphery` by
    # the ownership contract's own rule (`retrieved_ownership` keeps every spec entity's scope) and
    # the entity ids come from the delivery, so a generated object can only ever appear here as
    # itself -- there is no path by which one is recorded against a spec entity.
    for row in rows:
        entities.append(OwnedEntity(entity_id=row['entity_id'], owner='generated',
                                    scope='periphery', stage=REALISING_STAGE,
                                    provider=row['provider'] or GENERATED_PROVIDER,
                                    asset_ref=row['asset_id']))
    unrequested = sorted(set(assets) - set(ownership) - {row['entity_id'] for row in rows})
    return BackgroundManifest(
        scene_id=str(run.delivery.get('scene_id', run.run_dir.name)),
        seed=int(run.delivery.get('seed', 0)), mode=mode,
        entities=tuple(entities),
        stages=stage_records(run),
        required_entity_ids=request.required_entity_ids,
        core_hash=run.plan_core_hash, core_hash_requested=request.core_hash,
        # Empty in `retrieved` mode because that mode allows no generated entity anywhere; the
        # generated mode's allowance is the frozen constant the build itself was held to.
        budget=dict(request.budget),
        gaps=gaps, gap_reasons=gap_reasons_by_category,
        overlap_by_entity=measure_overlaps(run, core),
        measurements={
            'core_asset_digest': core_asset_digest(run, core),
            'periphery_entity_ids': list(periphery_entity_ids(run.spec)),
            'placed_instances': len(assets),
            'unrequested_entities': unrequested,
            'generated_instances': rows,
            'generated_categories': sorted({row['category'] for row in rows}),
            'gap_declaration': {'categories': list(gaps), 'read_from': gap_source},
            'gap_reasons': dict(gap_reasons_by_category),
            'assembly': run.delivery.get('background', {}).get('assembly', {}) or {},
            'floor_reservations': {'protected_vertices': len(run.protected_xy),
                                   'keepout_polygons': len(run.keepouts),
                                   'protection_band_top_m': run.support_top_z_m},
            'room_replay': run.replay_note,
            'placeholder_census': {
                'objects_with_generator': run.placeholders,
                'tagged_real_placeholder': run.real_placeholders,
                'factories': list(run.placeholder_generators),
                'note': 'objects in solve_state.json carrying a generator -- the population targets '
                        'of populate_intermediate_pholders/populate_assets (`populate.py:62` skips '
                        'only the generator-less and the already-spawned) -- and how many of them '
                        'are tagged Semantics(real-placeholder)',
            },
        },
        provenance={'audited_by': 'python -m rfw_w.scenes.validation.check_generated_background',
                    'run_dir': _relative(run.run_dir),
                    'build_dir': None if run.build_dir is None else _relative(run.build_dir),
                    'request': request.to_dict(),
                    'core_replay': 'solve_cell + fit_room_around_cell + plan_from_fitted',
                    'elapsed_s': round(time.monotonic() - run_started, 3)})


# ---------------------------------------------------------------------------------------------
# The negatives: each is a mutation of a real manifest, and each must be caught by name
# ---------------------------------------------------------------------------------------------

def _intruding_entity(run: RunFiles, core: set[str]) -> tuple[str, float] | None:
    """A placed, non-core, floor-standing entity, so the intrusion negative has something real.

    Taken from the delivery rather than invented: the fixture has to be a piece that *could* stand
    there, or the negative proves nothing about the geometry that would catch it.
    """
    dimensions = {asset['entity_id']: asset['dimensions_m']
                  for asset in run.delivery.get('assets_used', ())}
    for entity_id, placed in sorted(run.delivery.get('placement', {}).get('instances', {}).items()):
        if entity_id in core or entity_id not in dimensions:
            continue
        if float(placed['position_m'][2]) >= run.support_top_z_m:
            continue
        return entity_id, 0.0
    return None


def negatives(run: RunFiles, manifest: BackgroundManifest) -> list[dict[str, Any]]:
    """The four mutations SPEC-009 requires to fail, plus the two the stage rules add.

    Each is applied to a *copy* of the real manifest through `with_changes`, so a negative can
    never edit the evidence it was derived from.
    """
    core = set(core_entity_ids(run.spec))
    cases: list[dict[str, Any]] = []
    intruder = _intruding_entity(run, core)
    if intruder is not None:
        entity_id, _ = intruder
        overlaps = dict(manifest.overlap_by_entity)
        overlaps[entity_id] = {RESERVATION_PROTECTED: 0.125, RESERVATION_KEEPOUT: 0.0}
        cases.append({'name': 'protected_intrusion', 'expect': 'BACKGROUND_PROTECTED_OVERLAP',
                      'manifest': manifest.with_changes(overlap_by_entity=overlaps)})
        keepout = dict(manifest.overlap_by_entity)
        keepout[entity_id] = {RESERVATION_PROTECTED: 0.0, RESERVATION_KEEPOUT: 0.0625}
        cases.append({'name': 'keepout_intrusion', 'expect': 'BACKGROUND_KEEPOUT_OVERLAP',
                      'manifest': manifest.with_changes(overlap_by_entity=keepout)})
    core_entity = sorted(core)[0] if core else ''
    if core_entity:
        entities = tuple(
            entity if entity.entity_id != core_entity
            else OwnedEntity(entity_id=entity.entity_id, owner='generated', scope=entity.scope,
                             stage='solve_medium')
            for entity in manifest.entities)
        cases.append({'name': 'core_forced_generated', 'expect': 'BACKGROUND_CORE_OWNER_GENERATED',
                      'manifest': manifest.with_changes(entities=entities)})
    periphery = [entity for entity in manifest.entities if not entity.is_core]
    if periphery:
        # A periphery entity, so the mutation adds one fact and not three: a *core* piece handed
        # to the generator also trips CORE_OWNER_GENERATED, and the row would no longer say which
        # rule caught it. The stage's allowance is opened only enough to keep the budget quiet.
        target = periphery[0]
        entities = manifest.entities + (OwnedEntity(entity_id=target.entity_id, owner='generated',
                                                    scope=target.scope, stage='solve_medium'),)
        cases.append({'name': 'duplicate_owner', 'expect': 'BACKGROUND_DUPLICATE_OWNER',
                      'manifest': manifest.with_changes(
                          entities=entities, mode='generated',
                          budget={stage: len(manifest.entities) for stage in STAGES})})
    missing = manifest.required_entity_ids[:1]
    if missing:
        entities = tuple(entity for entity in manifest.entities
                         if entity.entity_id not in set(missing))
        cases.append({'name': 'required_entity_dropped',
                      'expect': 'BACKGROUND_REQUIRED_ENTITY_MISSING',
                      'manifest': manifest.with_changes(entities=entities)})
    cases.append({'name': 'tiny_budget', 'expect': 'BACKGROUND_BUDGET_EXCEEDED',
                  'manifest': manifest.with_changes(mode='generated',
                                                    budget={stage: 0 for stage in STAGES},
                                                    entities=tuple(
                                                        OwnedEntity(entity_id=entity.entity_id,
                                                                    owner='generated',
                                                                    scope=entity.scope,
                                                                    stage='solve_medium')
                                                        for entity in manifest.entities))})
    if manifest.stages:
        stages = tuple(record for record in manifest.stages if record.stage != 'solve_medium')
        cases.append({'name': 'stage_left_unreported',
                      'expect': 'BACKGROUND_STAGE_NOT_REPORTED',
                      'manifest': manifest.with_changes(stages=stages + (
                          StageRecord(stage='solve_medium', state='not_run', evidence=EVIDENCE_NONE,
                                      reason='no line in the build log'),))})
    # The two delivery-side rules (ADR-009 decision 11). Each mutation is chosen to trip exactly
    # one of them: a category is swapped for one the room did not declare, on a delivery where the
    # category it left behind is still filled by another object -- else the swap would also make
    # that gap unfilled, and the row would no longer say which rule caught it.
    if manifest.gaps:
        counts: dict[str, int] = {}
        for entity in manifest.entities_owned_by('generated'):
            category = generated_category(entity.entity_id) or ''
            counts[category] = counts.get(category, 0) + 1
        for entity in manifest.entities_owned_by('generated'):
            category = generated_category(entity.entity_id) or ''
            if counts.get(category, 0) < 2:
                continue
            entities = tuple(
                item if item.entity_id != entity.entity_id
                else OwnedEntity(entity_id='generated_wardrobe_1', owner='generated',
                                 scope=item.scope, stage=item.stage)
                for item in manifest.entities)
            cases.append({'name': 'generated_category_not_a_gap',
                          'expect': 'BACKGROUND_GENERATED_CATEGORY_NOT_A_GAP',
                          'manifest': manifest.with_changes(entities=entities)})
            break
        # A category the retrieval vocabulary covers and this room did not declare: adding it to the
# declaration is the mutation. Chosen from the vocabulary rather than hardcoded, because a fixed
# name (`mirror_frame`) is a *declared* gap in the bedroom and the kitchen -- where the mutation
# would have been a no-op and the negative would have proved nothing.
        undeclared = next((category for category in ('wardrobe', 'chair', 'stool', 'rug')
                           if category not in set(manifest.gaps)), None)
        if undeclared is not None:
            cases.append({'name': 'gap_declared_but_unfilled',
                          'expect': 'BACKGROUND_GAP_UNFILLED',
                          'manifest': manifest.with_changes(
                              gaps=tuple(sorted(set(manifest.gaps) | {undeclared})))})
    return cases


def _kept_in_ledger(run: RunFiles) -> int:
    """How many objects the build's own pool ledger says it kept, or 0 without a ledger."""
    if run.build_dir is None:
        return 0
    record_path = run.build_dir / INJECTION_FILENAME
    if not record_path.is_file():
        return 0
    pool = json.loads(record_path.read_text()).get('rejection_pool') or {}
    return sum(1 for item in pool.get('candidates', ()) if item.get('outcome') == 'kept')


def _with(run: RunFiles, **changes: Any) -> RunFiles:
    """A shallow copy of the run with fields changed: a negative never edits the evidence."""
    clone = copy.copy(run)
    for name, value in changes.items():
        setattr(clone, name, value)
    return clone


def assembly_negatives(run: RunFiles) -> list[dict[str, Any]]:
    """The delivery-level assembly rules, each tripped by a mutated *copy* of the real delivery.

    These read three places at once -- the manifest's `assets_used`, the graph's instances and the
    prepared assets on disk -- so they cannot be expressed as manifest mutations. Each case keeps
    the delivery it was derived from untouched and changes one thing in the copy, which is the same
    rule the manifest negatives follow.
    """
    rows = generated_rows(run)
    cases: list[dict[str, Any]] = []
    if not rows:
        return cases
    first = rows[0]['entity_id']
    # The manifest row is kept and the instance removed: the delivery claims an object that nothing
    # places. This is the direction that matters -- an inventory entry with nothing behind it reads
    # as a scene that has the object.
    without_instance = tuple(item for item in run.scene.get('instances', ())
                             if str(item.get('entity_id')) != first)
    cases.append({'name': 'assembly_row_without_instance',
                  'expect': 'BACKGROUND_ASSEMBLY_MISMATCH',
                  'run': _with(run, scene={**run.scene, 'instances': without_instance})})
    # And the reverse: the scene has it, the inventory does not, so a reader of the manifest cannot
    # see what the delivery carries.
    without_row = tuple(asset for asset in run.delivery.get('assets_used', ())
                        if str(asset.get('entity_id')) != first)
    cases.append({'name': 'instance_without_assembly_row',
                  'expect': 'BACKGROUND_INSTANCE_UNRECORDED',
                  'run': _with(run, delivery={**run.delivery, 'assets_used': without_row})})
    cases.append({'name': 'generated_asset_not_on_disk',
                  'expect': 'BACKGROUND_GENERATED_ASSET_MISSING',
                  'run': _with(run, run_dir=run.run_dir / 'not-a-delivery')})
    return cases


def run_negatives(run: RunFiles, manifest: BackgroundManifest) -> list[dict[str, Any]]:
    """Run each negative and require it to be caught by the name it announces."""
    rows = []
    for case in negatives(run, manifest):
        gaps = audited_gaps(case['manifest'])
        found = any(gap.startswith(case['expect']) for gap in gaps)
        rows.append({'name': case['name'], 'expected': case['expect'], 'observed': list(gaps),
                     'caught': found})
    for case in assembly_negatives(run):
        gaps = assembly_gaps(case['run'])
        found = any(gap.startswith(case['expect']) for gap in gaps)
        rows.append({'name': case['name'], 'expected': case['expect'], 'observed': list(gaps),
                     'caught': found})
    return rows


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def audit_delivery(run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {'run_dir': str(run_dir)}
    try:
        run = load_run(run_dir)
        manifest = manifest_for(run, run_started=started)
        row['manifest'] = manifest.to_dict()
        row['gaps'] = list(audited_gaps(manifest)) + assembly_gaps(run)
        row['negatives'] = run_negatives(run, manifest)
    except Exception as error:  # a delivery the audit cannot read is reported, never skipped
        row['gaps'] = [f'{type(error).__name__}({error})']
        row['manifest'] = None
        row['negatives'] = []
    row['passed'] = not row['gaps']
    row['negatives_passed'] = all(item['caught'] for item in row['negatives'])
    row['elapsed_s'] = round(time.monotonic() - started, 3)
    return row


MATRIX_SCHEMA_VERSION = 'pvscene.background-matrix/v1'


def _verified_independently(run_dir: Path) -> dict[str, Any]:
    """Run the independent verifier on this delivery, as SPEC-009's matrix requires per scene.

    A subprocess rather than an import: the verifier is meant to be a *second* reader of the
    delivery, and a matrix that called it in-process could pass it state the delivery never had.
    """
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-m", 'rfw_w.scenes.validation.verify_delivery',
         '--delivery', str(run_dir)], capture_output=True, text=True)
    tail = (completed.stdout or '').strip().splitlines()[-1:] or ['']
    answer: dict[str, Any] = {'verified': completed.returncode == 0, 'exit_code': completed.returncode,
                              'summary': tail[0], 'elapsed_s': round(time.monotonic() - started, 3)}
    # `passed` is what the row's criteria loop reads; the verifier's own word (`verified`) is kept
    # beside it so a reader of the report sees the verifier's verdict, not just the summary line.
    answer['passed'] = bool(answer['verified'])
    try:
        payload = json.loads(completed.stdout)
        answer['recomputed_readiness'] = payload.get('recomputed_readiness')
    except (ValueError, TypeError):
        pass
    return answer


def _final_failures(run_dir: Path) -> list[str]:
    """The gates whose *last* recorded verdict is a failure, for one delivery.

    The last verdict, not any verdict: a run whose first version was out of bounds and whose final
    version is inside is a run that repaired itself, and counting the repaired row as a failure
    would report the loop's work as a defect.
    """
    manifest = json.loads((run_dir / 'delivery_manifest.json').read_text())
    last: dict[str, str] = {}
    for row in manifest.get('gates', ()):
        last[str(row.get('gate_id'))] = str(row.get('status'))
    return sorted(gate for gate, status in last.items() if status == 'fail')


def _isaac_evidence(run: RunFiles) -> dict[str, Any]:
    """Whether this delivery's own manifest says the worker settled and rendered it.

    The *last* verdict per gate, for the reason `_final_failures` gives: a run that retried the
    worker and passed on the second attempt has passed, and the first attempt's row is history.
    """
    last: dict[str, str] = {}
    for row in run.delivery.get('gates', ()):
        last[str(row.get('gate_id'))] = str(row.get('status'))
    wanted = ('backend.isaac.load', 'backend.isaac.run', 'backend.isaac.render',
              'physics.settle_stability', 'physics.clock_advanced', 'visual.required_views')
    return {name: last.get(name, 'absent') for name in wanted}


def matrix_case(run_dir: Path, case: Mapping[str, Any], *,
                verify: bool = True) -> dict[str, Any]:
    """One matrix row: what the delivery measured, judged against what the case declares.

    `expect_failures` is the same contract SPEC-011's matrix uses, and it is what keeps an honest
    boundary from being mistaken for a pass: a case declares the named failures its delivery *must*
    show -- the abort switch that cannot be on in this environment, for one -- and any red the case
    did not declare is an `unexpected_red`, which fails the row.
    """
    row: dict[str, Any] = {'id': str(case['id']), 'room_type': str(case.get('room_type', '')),
                           'seed': int(case.get('seed', 0)), 'role': str(case.get('role', 'scene')),
                           'delivery': _relative(run_dir), 'status': 'ran'}
    if not (run_dir / 'delivery_manifest.json').is_file():
        row['status'] = 'not_run'
        row['not_run_reason'] = str(case.get(
            'not_run_reason', 'the delivery does not exist yet; the case is listed, not passed'))
        return row
    audit = audit_delivery(run_dir)
    manifest = audit['manifest'] or {}
    measurements = manifest.get('measurements', {})
    categories = sorted(str(item) for item in measurements.get('generated_categories', ()))
    minimum = int(case.get('min_generated_categories', 0))
    criteria: dict[str, Any] = {
        'owned_and_frozen': {'passed': bool(audit['passed']), 'gaps': list(audit['gaps'])},
        'generated_categories': {'measured': categories, 'minimum': minimum,
                                 'passed': len(categories) >= minimum},
        'gap_alignment': {'declared': list(case.get('gaps', ())),
                          'measured': categories,
                          'passed': set(categories) <= set(case.get('gaps', ()))},
        'isaac': _isaac_evidence(load_run(run_dir)),
    }
    baseline = case.get('baseline')
    if baseline:
        baseline_dir = (ROOT / str(baseline)) if not Path(str(baseline)).is_absolute() \
            else Path(str(baseline))
        if (baseline_dir / 'delivery_manifest.json').is_file():
            baseline_audit = audit_delivery(baseline_dir)
            baseline_manifest = baseline_audit['manifest'] or {}
            same_digest = (measurements.get('core_asset_digest')
                           == baseline_manifest.get('measurements', {}).get('core_asset_digest'))
            same_hash = manifest.get('core_hash') == baseline_manifest.get('core_hash')
            criteria['core_unchanged'] = {
                'baseline': _relative(baseline_dir), 'digest_equal': bool(same_digest),
                'core_hash_equal': bool(same_hash), 'passed': bool(same_digest and same_hash)}
            # And no *new* red: the generated delivery's final gate verdicts must not fail anything
            # the same room's default-path delivery did not also fail. The core hash says the plan
            # did not move; this says the delivery did not get worse -- which is the half a room
            # with a pre-existing defect (a retrieval piece the loop could not repair) would
            # otherwise turn into a red against the background branch.
            baseline_fails = _final_failures(baseline_dir)
            measured_fails = _final_failures(run_dir)
            criteria['no_new_red'] = {
                'baseline_failures': baseline_fails, 'measured_failures': measured_fails,
                'new': sorted(set(measured_fails) - set(baseline_fails)),
                'passed': not (set(measured_fails) - set(baseline_fails))}
        else:
            criteria['core_unchanged'] = {
                'baseline': str(baseline), 'passed': False,
                'reason': 'the declared baseline delivery does not exist'}
    else:
        criteria['core_unchanged'] = {'passed': False, 'reason': 'no baseline declared for this case'}
    if verify:
        criteria['independent_verification'] = _verified_independently(run_dir)
    expected = [str(item) for item in case.get('expect_failures', ())]
    reds = list(audit['gaps'])
    row['expected_red_confirmed'] = all(
        any(red.startswith(prefix) for red in reds) for prefix in expected)
    row['unexpected_reds'] = [red for red in reds
                              if not any(red.startswith(prefix) for prefix in expected)]
    row['criteria'] = criteria
    blocked_by = str(case.get('blocked_by', ''))
    if blocked_by:
        # A blocked case is not a pass and not a failure of the branch: something *outside* it
        # stopped the scene. The fixture has to name the cause and the evidence for it, and the
        # evidence is looked for in this delivery -- a block is a claim like any other, and one
        # nobody can check would be a way to file any red under "blocked".
        evidence = str(case.get('blocked_evidence', ''))
        observed = list(reds) + [f'{gate}:fail' for gate, state in criteria['isaac'].items()
                                 if state == 'fail']
        observed += [f'{gate}:fail' for gate in _final_failures(run_dir)]
        row['status'] = 'blocked'
        row['blocked_by'] = blocked_by
        row['blocked_evidence'] = evidence
        row['blocked_confirmed'] = bool(evidence) and any(evidence in item for item in observed)
        row['passed'] = False
        row['audit_s'] = audit['elapsed_s']
        return row
    row['negatives'] = {str(item['name']): bool(item['caught']) for item in audit['negatives']}
    row['passed'] = (not row['unexpected_reds'] and row['expected_red_confirmed']
                     and bool(row['negatives']) and all(row['negatives'].values())
                     and all(bool(item.get('passed')) for name, item in criteria.items()
                             if name != 'isaac')
                     and not any(state == 'fail' for state in criteria['isaac'].values()))
    row['audit_s'] = audit['elapsed_s']
    return row


def run_matrix(fixture_path: Path, output_dir: Path) -> int:
    """SPEC-009's acceptance matrix: one row per scene, plus the negative cases' verdicts."""
    fixture = json.loads(fixture_path.read_text())
    version = fixture.get('schema_version')
    if version != MATRIX_SCHEMA_VERSION:
        raise AuditError(f'BACKGROUND_MATRIX_SCHEMA_UNSUPPORTED({version}): '
                         f'expected {MATRIX_SCHEMA_VERSION}')
    cases = list(fixture.get('cases', ()))
    if not cases:
        raise AuditError('BACKGROUND_MATRIX_EMPTY(no cases)')
    rows = []
    for case in cases:
        delivery = str(case.get('delivery', ''))
        if not delivery:
            raise AuditError(f'BACKGROUND_MATRIX_CASE_INVALID({case.get("id")}:no delivery)')
        run_dir = Path(delivery) if Path(delivery).is_absolute() else ROOT / delivery
        print(f'{case["id"]}: auditing {delivery}')
        rows.append(matrix_case(run_dir, case))
    declared_negatives = sorted(
        # A name, or an object carrying one: the fixture is allowed to be a list of names, and a
        # reader should not have to care which spelling was chosen.
        str(item['name'] if isinstance(item, Mapping) else item)
        for item in fixture.get('negatives', ()))
    observed = {name for row in rows for name in (row.get('negatives') or {})}
    failed = sorted({name for row in rows for name, caught in (row.get('negatives') or {}).items()
                     if not caught})
    without_a_verdict = sorted(set(declared_negatives) - observed)
    report = {
        'tool': 'python -m rfw_w.scenes.validation.check_generated_background',
        'spec': 'SPEC-009-infinigen-background-generation',
        'stage': 'TASK 2 (the remaining half: generated objects as delivery instances) and '
                 'TASK 3 (the six-scene matrix)',
        'background_version': BACKGROUND_VERSION,
        'matrix_schema_version': MATRIX_SCHEMA_VERSION,
        'fixture': _relative(fixture_path),
        'cases': rows,
        'cases_ran': sum(1 for row in rows if row['status'] == 'ran'),
        'cases_not_run': [row['id'] for row in rows if row['status'] != 'ran'],
        'cases_blocked': [{'id': row['id'], 'by': row.get('blocked_by', ''),
                           'confirmed': bool(row.get('blocked_confirmed'))}
                          for row in rows if row['status'] == 'blocked'],
        'cases_blocked_unconfirmed': [row['id'] for row in rows if row['status'] == 'blocked'
                                      and not row.get('blocked_confirmed')],
        'negatives': {'declared': declared_negatives, 'observed': sorted(observed),
                      'failed': failed, 'without_a_verdict': without_a_verdict},
        'human_review': {'status': 'not_run',
                         'items': ['lighting/shadow/colour cast on a generated periphery',
                                   'whether generated furniture reads as furniture',
                                   'whether generated materials read as the room\'s materials']},
        'matrix_run': True,
        'measured_passed': all(row['passed'] for row in rows if row['status'] == 'ran'),
        'spec_accepted': False,
    }
    report['spec_accepted'] = bool(report['matrix_run']
                                   and report['cases_ran'] == len(rows)
                                   and report['measured_passed']
                                   and not failed and not without_a_verdict
                                   and not report['cases_blocked_unconfirmed'])
    output_dir.mkdir(parents=True)
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for row in rows:
        if row['status'] == 'blocked':
            print(f"  {row['id']}: blocked ({row['blocked_by']}); "
                  f"cause {'confirmed' if row['blocked_confirmed'] else 'NOT confirmed'}")
            continue
        if row['status'] != 'ran':
            print(f"  {row['id']}: not_run ({row['not_run_reason']})")
            continue
        print(f"  {row['id']}: {'PASS' if row['passed'] else 'FAIL'}; "
              f"categories {row['criteria']['generated_categories']['measured']}; "
              f"negatives {sum(1 for ok in row['negatives'].values() if ok)}/{len(row['negatives'])}")
        for red in row['unexpected_reds']:
            print(f'      unexpected: {red}')
    return 0 if report['spec_accepted'] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--delivery', type=Path, action='append', default=[],
                        help='a delivery directory to audit; repeat for several')
    parser.add_argument('--deliveries-from', type=Path, action='append', default=[],
                        help='a directory tree to walk for */delivery_manifest.json; repeatable')
    parser.add_argument('--matrix', type=Path, default=None,
                        help='the SPEC-009 acceptance matrix: one row per scene, and its verdicts')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('use a fresh output directory: an audit never overwrites evidence')
    if args.matrix is not None:
        if not args.matrix.is_file():
            parser.error(f'BACKGROUND_MATRIX_MISSING(no fixture at {args.matrix})')
        try:
            return run_matrix(args.matrix, args.output_dir)
        except AuditError as error:
            parser.error(str(error))
    targets = list(args.delivery)
    for tree in args.deliveries_from:
        targets.extend(sorted(path.parent for path
                              in tree.glob('*/delivery_manifest.json')))
        targets.extend(sorted(path.parent.parent for path
                              in tree.glob('*/*/delivery_manifest.json')))
    targets = sorted(dict.fromkeys(targets))
    if not targets:
        parser.error('name at least one --delivery, or a tree with --deliveries-from')
    for target in targets:
        if not (target / 'delivery_manifest.json').is_file():
            parser.error(f'{target} has no delivery_manifest.json')

    args.output_dir.mkdir(parents=True)
    rows = [audit_delivery(target) for target in targets]
    report = {
        'tool': 'python -m rfw_w.scenes.validation.check_generated_background',
        'spec': 'SPEC-009-infinigen-background-generation',
        'stage': 'TASK 1 (measurement tool and baseline); the acceptance matrix is not implemented',
        'background_version': BACKGROUND_VERSION,
        'mode': 'retrieved (the only mode that exists on this machine)',
        'stages': list(STAGES),
        'deliveries': rows,
        'matrix_run': False,
        'human_review': {'status': 'not_run',
                         'items': ['lighting/shadow/colour cast on a generated periphery',
                                   'whether generated furniture reads as furniture']},
        'measured_passed': all(row['passed'] for row in rows),
        'negatives_passed': all(row['negatives_passed'] for row in rows),
        # A pass here says the retrieved baseline is owned, frozen and clear of the reservations.
        # It says nothing about generation, which has not run.
        'spec_accepted': False,
    }
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))
    for row in rows:
        print(f"{row['run_dir']}: {'PASS' if row['passed'] else 'FAIL'} "
              f"in {row['elapsed_s']} s, negatives "
              f"{'ok' if row['negatives_passed'] else 'FAILED'}")
        for gap in row['gaps']:
            print(f'    {gap}')
    return 0 if report['measured_passed'] and report['negatives_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
