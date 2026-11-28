#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure what Infinigen's `populate_assets` actually left in the room.

    blender -b <build>/scene/scene.blend -P python -m rfw_w.scenes.workers.blender.measure_generated_furniture -- \\
        --delivery outputs/<scene>/<run> --out /tmp/generated_furniture.json

Run under **Infinigen's bundled Blender Python**. It answers one question SPEC-009 has to answer
before anything is wired into the default path: the generated furniture is placed by *Infinigen's
own* constraint solver, which has never heard of the task cell's protected region or of our
circulation corridor. Does it stand in them?

Two facts make this measurable at all:

* Every asset `populate_assets` realised is named ``<Factory>(n).spawn_asset(m)``, and every
  placeholder it left is ``...spawn_placeholder(m)`` -- so "what did generation produce" is read
  from the scene rather than from a report about the scene.
* The reservations are not in the built scene; they are re-derived here from the delivery's own
  effective spec and room seed, the same replay `check_generated_background.py` uses
  (`solve_cell` -> `fit_room_around_cell` -> `plan_from_fitted`). Both tools therefore measure
  against the same polygon rather than each holding its own copy.

It measures, it does not judge: the overlap numbers and the band rule are here, the verdict
(`BACKGROUND_PROTECTED_OVERLAP(entity, m2)`) stays in the audit tool and the contract.

The geometry itself moved to `rfw_w.scenes.geometry.reservations` in SPEC-011 TASK 4, when the
build's own rejection pool started judging realised objects during the build: two copies of one
rule could only ever disagree, and a disagreement between "the pool refused everything" and "the
ruler still measures area" is indistinguishable from an intruder.

The built shell is inset half a wall thickness from the plan outline (measured 0.05 m, SPEC-004),
so a footprint that grazes a reservation edge by centimetres is a real geometric fact but a weak
verdict; the report says which objects overlap by how much so a reader can see the difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path.cwd()

from rfw_w.scenes.contracts.background_constraints import (
    CIRCULATION_BAND_TOP_M,
    DOOR_SWEEP_BAND_TOP_M,
)
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.geometry.reservations import Region, measure_part
from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.room import fit_room_around_cell
from rfw_w.scenes.solvers.room_plan import plan_from_fitted

#: How an object is named tells whether the populate stage realised it.
SPAWNED = 'spawn_asset'
PLACEHOLDER = 'spawn_placeholder'


def _read_json(path: Path) -> dict:
    import bpy
    return json.loads(path.read_text())


def reservations(delivery: Path) -> dict:
    """The task cell's protected region, the corridor, the door sweep, and the bands they reserve.

    Re-derived, not read from the scene: the built geometry has no notion of a task cell. Same
    replay as the audit tool's, so the two measurements rest on one polygon.

    The three regions are reported **separately** because SPEC-011 requirement 3 asks for the
    protected region and the corridor as two reports, and because a piece can stand in either --
    the older aggregate `keepout_polygons_xy` is kept alongside so the numbers already recorded in
    SPEC-009's evidence can still be reproduced from this tool.
    """
    import bpy
    spec = SceneSpec.from_dict(_read_json(delivery / 'effective_spec.json'))
    seed = int(_read_json(delivery / 'generation_attempt_001.json')['room_seed'])
    core = solve_cell(spec, seed=spec.seed)
    fitted = fit_room_around_cell(spec, core, room_seed=seed)
    plan = plan_from_fitted(spec, core, fitted, room_seed=seed)
    zones = _read_json(delivery / 'delivery_manifest.json').get('zones', {})
    circulation = [[list(v) for v in ring] for ring in zones.get('circulation', ())]
    door_sweep = [[list(v) for v in ring] for ring in zones.get('door_sweep', ())]
    return {
        'protected_polygon_xy': [list(v) for v in plan.core.protected_in_room_xy],
        'circulation_polygons_xy': circulation,
        'door_sweep_polygons_xy': door_sweep,
        'keepout_polygons_xy': circulation + door_sweep,
        'protection_band_top_m': plan.core.support_top_z_m,
        'circulation_band_top_m': CIRCULATION_BAND_TOP_M,
        'door_sweep_band_top_m': DOOR_SWEEP_BAND_TOP_M,
        'room_polygon_xy': [list(v) for v in plan.rooms[0].polygon_xy],
    }


def footprint_rule() -> str:
    """Where the geometry rule lives, so a record can name the rule it was measured by.

    Both this tool and the build's own rejection pool judge by
    `rfw_w.scenes.geometry.reservations`: one rule, two readers (SPEC-011 TASK 4). The string travels
    in the report because a number is only evidence next to the rule that produced it.
    """
    import bpy
    return "rfw_w.scenes.geometry.reservations.measure_part (hull footprint, region band, m2)"


def background_factories(build: Path) -> set[str]:
    """The generators the stage ladder solved *background* content with.

    Read from the build's own `solve_state.json`: an entry carrying a generator is a piece
    `populate_assets` would realise. The set matters because the same ``spawn_asset`` naming is
    used for the building's own spawned parts -- the probe's scene has three windows and a door
    under it -- and counting those as generated furniture would report the architecture as
    background content.
    """
    import bpy
    state = _read_json(build / 'scene' / 'solve_state.json')
    return {str(entry['generator']).split('(')[0]
            for entry in state['objs'].values() if entry.get('generator')}


def main() -> int:
    # Blender takes everything after `--` as the script's own arguments and leaves its own flags in
    # `sys.argv` ahead of them, so the split is the `--` and not an offset.
    import bpy
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True,
                        help='the delivery whose spec/seed define the reservations')
    parser.add_argument('--build', type=Path, required=True,
                        help='the infinigen_build directory this blend was written by')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    factories = background_factories(args.build)

    where = reservations(args.delivery)
    band_top = float(where['protection_band_top_m'])
    circulation_top = float(where['circulation_band_top_m'])
    sweep_top = float(where['door_sweep_band_top_m'])
    # One region per polygon, named by the region it belongs to; the shared rule sums a region's
    # polygons and rounds once, which is what keeps these numbers the ones already on record.
    regions = []
    if where['protected_polygon_xy']:
        regions.append(Region('protected', Polygon(where['protected_polygon_xy']), band_top))
    regions.extend(Region('circulation', Polygon(ring), circulation_top)
                   for ring in where['circulation_polygons_xy'])
    regions.extend(Region('door_sweep', Polygon(ring), sweep_top)
                   for ring in where['door_sweep_polygons_xy'])
    # The SPEC-009 aggregate, kept on the band that produced the recorded numbers (0.0315 m2 for the
    # probe's shelf), so this tool can still reproduce that evidence. It is the protected band on
    # purpose: the older name predates the corridor's own 1.8 m band.
    regions.extend(Region('keepout', Polygon(ring), band_top)
                   for ring in (where['circulation_polygons_xy']
                                + where['door_sweep_polygons_xy']))

    objects = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj.data is None:
            continue
        factory = obj.name.split('(')[0]
        if factory not in factories:
            continue
        realised = SPAWNED in obj.name
        if not realised and PLACEHOLDER not in obj.name:
            continue
        part = measure_part(obj, regions)
        if part is None:
            continue
        measured = part.overlaps_m2
        entry = {
            'name': obj.name,
            'factory': factory,
            'generated': realised,
            'base_z_m': round(part.base_z_m, 4),
            'top_z_m': round(part.top_z_m, 4),
            # The band rule, applied to the object's real world height: the reservations are *floor*
            # regions, so only what reaches down into them counts. A pendant above the desk hangs
            # over the reservation, it does not stand in it -- and the corridor reaches higher than
            # the work surface (ADR-009 decision 3), so the bands are asked separately.
            'in_protection_band': part.reaches(band_top),
            'in_circulation_band': part.reaches(circulation_top),
            'in_door_sweep_band': part.reaches(sweep_top),
        }
        entry['footprint_extent_m'] = [] if part.footprint_extent_m is None else [
            round(part.footprint_extent_m[0], 4), round(part.footprint_extent_m[1], 4)]
        entry['protected_overlap_m2'] = measured.get('protected', 0.0)
        entry['circulation_overlap_m2'] = measured.get('circulation', 0.0)
        entry['door_sweep_overlap_m2'] = measured.get('door_sweep', 0.0)
        entry['keepout_overlap_m2'] = measured.get('keepout', 0.0)
        objects.append(entry)

    generated = [entry for entry in objects if entry['generated']]
    report = {
        'tool': 'python -m rfw_w.scenes.workers.blender.measure_generated_furniture',
        'tool_version': 'regions/v3',
        'rule': footprint_rule(),
        'blend': bpy.data.filepath,
        'delivery': str(args.delivery),
        'build': str(args.build),
        'protection_band_top_m': band_top,
        'bands': {'protected_top_m': band_top, 'circulation_top_m': circulation_top,
                  'door_sweep_top_m': sweep_top},
        'reservations': where,
        'background_factories': sorted(factories),
        'counts': {
            'spawned': len(generated),
            'left_as_placeholder': len([entry for entry in objects
                                        if not entry['generated']]),
            'spawned_standing_in_the_band': len([entry for entry in generated
                                                 if entry['in_protection_band']]),
            'spawned_standing_in_the_corridor_band': len([entry for entry in generated
                                                          if entry['in_circulation_band']]),
        },
        'objects': sorted(objects, key=lambda entry: entry['name']),
    }
    report['overlaps'] = sorted(
        ({'name': entry['name'], 'factory': entry['factory'],
          'footprint_extent_m': entry['footprint_extent_m'],
          'base_z_m': entry['base_z_m'],
          'protected_overlap_m2': entry['protected_overlap_m2'],
          'circulation_overlap_m2': entry['circulation_overlap_m2'],
          'door_sweep_overlap_m2': entry['door_sweep_overlap_m2'],
          'keepout_overlap_m2': entry['keepout_overlap_m2']}
         for entry in generated
         if max(entry['protected_overlap_m2'], entry['circulation_overlap_m2'],
                entry['door_sweep_overlap_m2']) > 0.0),
        key=lambda entry: -entry['protected_overlap_m2'])
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{report['counts']} -> {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
