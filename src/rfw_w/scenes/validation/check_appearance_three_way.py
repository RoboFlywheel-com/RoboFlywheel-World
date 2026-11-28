#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble the three-way appearance sheet and report the numbers behind it.

Three renders of one delivery -- Blender with the source procedural materials, Blender with the
exported PBR bake, and Isaac on the same USD -- under one camera and one declared mode from
`appearance_fixture.py`. This step adds no rendering of its own: the legs are produced by
`python -m rfw_w.scenes.workers.blender.render_material_pair` and `python -m rfw_w.scenes.workers.launch`, and if any of them is
missing this refuses rather than reporting a partial comparison as one.

It does not pass or fail an appearance. Mean luminance, clipping and the framing both legs
declare are recorded so a human verdict has something to sit against, and the checklist stays at
not_run until a person fills it in -- `appearance_fixture.three_way_verdict` is what turns that
sheet into a verdict, and it refuses a sheet for a diagnostic mode.
"""
from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import argparse
import importlib
import json
from pathlib import Path
import sys



def _fixture_module():
    return importlib.import_module('rfw_w.scenes.visualization.appearance_fixture')


def _luminance(path: Path) -> dict:
    import numpy as np
    from PIL import Image

    pixels = np.asarray(Image.open(path).convert('L'), dtype=float)
    return {'mean': round(float(pixels.mean()), 2),
            'p05': int(np.percentile(pixels, 5)), 'p95': int(np.percentile(pixels, 95)),
            'clipped_fraction': round(float((pixels >= 254).mean()), 4),
            'resolution': list(Image.open(path).size)}


def main() -> int:
    from PIL import Image, ImageDraw

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-dir', type=Path, required=True,
                        help='output of render_material_pair.py (mapped.png, baked.png, report.json)')
    parser.add_argument('--isaac-dir', type=Path, required=True, help='output of run_isaac_worker.py')
    parser.add_argument('--camera', default=None, help='defaults to the fixture camera id')
    parser.add_argument('--review', type=Path,
                        help='a filled-in checklist; the verdict is computed from it, never hand-written')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output exists; use fresh evidence')
    args.output_dir.mkdir(parents=True)

    fixture_module = _fixture_module()
    fixture = fixture_module.load(data_path('fixtures/appearance_fixture_v1.json'))
    camera_id = args.camera or fixture['camera']['camera_id']

    pair = json.loads((args.pair_dir / 'report.json').read_text())
    isaac = json.loads((args.isaac_dir / 'report.json').read_text())
    renders = {'blender_mapped': args.pair_dir / 'mapped.png',
               'blender_baked': args.pair_dir / 'baked.png',
               'isaac': args.isaac_dir / f'{camera_id}.png'}
    for name, path in renders.items():
        if not path.is_file():
            raise SystemExit(f'APPEARANCE_RENDER_MISSING({name}:{path})')

    # Both legs must be the same declared camera, or the three images are three views rather than
    # one. The pair reports its rig; the worker reports the lighting it was handed.
    pair_fov = pair['rig']['horizontal_fov_deg']
    isaac_lighting = isaac['lighting']
    if isaac['status'] != 'passed':
        raise SystemExit(f'ISAAC_CAPTURE_NOT_PASSED({isaac["status"]})')

    sheet = Image.new('RGB', (640 * 3, 388), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    panels = [('Blender source (procedural)', 'blender_mapped'),
              ('Blender exported (PBR bake)', 'blender_baked'),
              (f'Isaac (same USD, RTX)', 'isaac')]
    luminance = {}
    for index, (label, name) in enumerate(panels):
        image = Image.open(renders[name]).convert('RGB')
        sheet.paste(image.resize((640, 360)), (640 * index, 28))
        draw.text((640 * index + 8, 9), label, fill=(235, 235, 235))
        luminance[name] = _luminance(renders[name])
    sheet.save(args.output_dir / 'three_way.jpg', quality=93)
    review = json.loads(args.review.read_text()) if args.review else fixture_module.checklist(fixture)

    report = {
        'fixture': {'fixture_id': fixture['fixture_id'],
                    'mode': pair['fixture']['mode'], 'role': pair['fixture']['role']},
        'camera': {'camera_id': camera_id,
                   'horizontal_fov_deg': pair_fov,
                   'aperture_mm': pair['rig']['aperture_mm'],
                   'resolution': pair['rig']['resolution'],
                   'source': 'delivery CameraRequest, aperture from the fixture'},
        'lighting': {'blender': {'world_strength': pair['rig']['world_strength'],
                                 'area_enabled': pair['rig']['area_enabled'],
                                 'area_energy_W': pair['rig']['area_energy_W']},
                     'isaac': isaac_lighting,
                     'photometric_equivalence': pair['rig']['photometric_equivalence']},
        'luminance': luminance,
        'renders': {name: str(path) for name, path in renders.items()},
        'known_differences': [
            'The Blender legs render architectural surfaces only; the Isaac leg carries the room'
            ' instances too, so furniture covers part of the floor there.',
            'Cycles and RTX differ in BSDF, sampling and colour management. Each side was'
            ' calibrated to a comparable mean luminance; that is an exposure choice, not a match.',
        ],
        'review': review,
        'review_note': 'An assistant inspection is not a verdict. Fill the items in by hand;'
                       ' three_way_verdict then accepts or refuses the sheet.',
        'sheet': str(args.output_dir / 'three_way.jpg'),
    }
    if args.review:
        # The verdict is computed, so a report cannot claim a pass the sheet does not support.
        report['verdict'] = fixture_module.three_way_verdict(renders, review)
    (args.output_dir / 'three_way.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'fixture': report['fixture'], 'luminance': luminance,
                      'verdict': report.get('verdict', 'not_run'), 'sheet': report['sheet']},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
