#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run real PBR exports and relocation/texture negatives; never certify unreviewed visuals."""
from rfw_w.scenes.resources import module_path, data_path
import argparse
import hashlib
import json
import os
from pathlib import Path
from rfw_w.scenes.runtime import load_env_file
import shutil
import subprocess
import sys
import time

ROOT = Path.cwd()
from rfw_w.scenes.solvers.shell_providers import InfinigenShellProvider
from rfw_w.scenes.workers.isaac.shell_reference import inspect_shell, ShellReferenceError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=ROOT/'.env.local')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--machine-only', action='store_true')
    parser.add_argument('--three-way-report', type=Path,
                        help='three_way.json from check_appearance_three_way.py; without it the'
                             ' visual item stays not_run rather than being assumed either way')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error('output exists; use fresh evidence')
    out.mkdir(parents=True)
    load_env_file(args.env_file)
    root = InfinigenShellProvider()._root()
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    environment['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    matrix = json.loads(args.matrix.read_text())
    (out/'matrix.json').write_text(json.dumps(matrix, indent=2))
    code_files = [module_path('rfw_w.scenes.workers.blender.export_built_shell'), module_path('rfw_w.scenes.workers.blender.infinigen_materials'),
                  module_path('rfw_w.scenes.workers.blender.probe_material_bake_regression'), Path(__file__).resolve(),
                  module_path('rfw_w.scenes.visualization.appearance_fixture'),
                  module_path('rfw_w.scenes.validation.check_appearance_three_way'),
                  module_path('rfw_w.scenes.workers.isaac.shell_reference')]
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
    with (out/'negative.log').open('w') as log:
        negative_run = subprocess.run([str(root/'blender/blender'), '-b',
            '--python-exit-code','1','--python-expr', "import runpy; runpy.run_module('rfw_w.scenes.workers.blender.probe_material_failures', run_name='__main__')",
            '--','--out',str(out/'negative.json')],env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    negative = json.loads((out/'negative.json').read_text()) if (out/'negative.json').exists() else {'passed':False}
    with (out/'channel_regression.log').open('w') as log:
        channel_run = subprocess.run([str(root/'blender/blender'), '-b',
            '--python-exit-code','1','--python-expr', "import runpy; runpy.run_module('rfw_w.scenes.workers.blender.probe_material_bake_regression', run_name='__main__')",
            '--','--out',str(out/'channel_regression')],env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    channel_report = out/'channel_regression/report.json'
    channels = json.loads(channel_report.read_text()) if channel_report.exists() else {'passed':False}
    rows = []
    for case in matrix['cases']:
        started = time.monotonic()
        row = {'case': case['id'], 'gaps': []}
        directory = out/case['id']
        directory.mkdir()
        try:
            build = json.loads((ROOT/case['build_report']).read_text())
            source = Path(build['geometry_uris'][0])
            evidence = directory/'source'
            evidence.mkdir()
            shutil.copy2(source, evidence/'scene.blend')
            shutil.copy2(source.parent.parent/'measurement.json', evidence/'measurement.json')
            plan = json.loads((source.parent.parent/'room_plan.json').read_text())
            plan['rooms'][0]['floor_material'] = case['floor_material']
            plan['rooms'][0]['wall_material'] = case['wall_material']
            (evidence/'room_plan.json').write_text(json.dumps(plan, indent=2))
            export = directory/'export'
            with (directory/'export.log').open('w') as log:
                result = subprocess.run([str(root/'blender/blender'),
                    '-b',str(evidence/'scene.blend'),'--python-exit-code','1',
                    '--python-expr', "import runpy; runpy.run_module('rfw_w.scenes.workers.blender.export_built_shell', run_name='__main__')",'--','--out',str(export/'shell.usdc'),
                    '--measurement',str(evidence/'measurement.json'),'--room-plan',str(evidence/'room_plan.json'),
                    '--material-profile','pbr-v1','--resolution',str(args.resolution),'--bake-device','CUDA'],
                    env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            if result.returncode:
                raise ValueError('MATERIAL_BAKE_UNAVAILABLE(see export.log)')
            row['verification'] = inspect_shell(str(export/'shell.usdc'),delivery=directory)
            moved = out/(case['id']+'_relocated')
            shutil.copytree(directory, moved)
            row['relocation'] = inspect_shell(str(moved/'export/shell.usdc'),delivery=moved)
            # Damage only the independent copy after relocation has been verified.
            manifest = json.loads((moved/'export/material_manifest.json').read_text())
            target = moved/'export'/manifest['surfaces'][0]['channels']['NORMAL']['uri']
            target.unlink()
            try:
                inspect_shell(str(moved/'export/shell.usdc'),delivery=moved)
                row['gaps'].append('MATERIAL_NEGATIVE_DID_NOT_FAIL(missing texture)')
            except ShellReferenceError as error:
                row['missing_texture_negative'] = str(error)
                if not str(error).startswith('MATERIAL_DEPENDENCY_MISSING'):
                    row['gaps'].append('MATERIAL_NEGATIVE_WRONG_FAILURE')
        except Exception as error:
            row['gaps'].append(str(error))
        row.update(passed=not row['gaps'], elapsed_s=round(time.monotonic()-started,3))
        (directory/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row)
    unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in code_hashes.items())
    # The visual item is *read*, never assumed. A verdict only exists when a human filled the
    # sheet in and `three_way_verdict` computed it; two images on disk are not a result.
    visual, visual_report = 'not_run', None
    if args.three_way_report:
        visual_report = json.loads(args.three_way_report.read_text())
        visual = (visual_report.get('verdict') or {}).get('verdict', 'not_run')
    machine_passed = (bool(rows) and all(r['passed'] for r in rows) and unchanged and
                      negative_run.returncode==0 and negative.get('passed',False) and
                      channel_run.returncode==0 and channels.get('passed',False))
    report = {'scope':'real export, binding, dependency and relocation checks', 'cases':rows,
              'machine_passed':machine_passed,
              'implementation_sha256':code_hashes, 'implementation_unchanged':unchanged,
              'three_way_visual_review':visual,
              'three_way_report':None if visual_report is None else {
                  'path':str(args.three_way_report),
                  'mode':(visual_report.get('verdict') or {}).get('mode'),
                  'reviewer':(visual_report.get('verdict') or {}).get('reviewer'),
                  'unmet':(visual_report.get('verdict') or {}).get('unmet'),
                  'scope_note':visual_report.get('review',{}).get('scope')},
              'unsupported_shader_negative':negative,
              'channel_regression':channels,
              'spec_accepted':machine_passed and visual=='pass'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # `--machine-only` keeps its old meaning: machine green is green even with no verdict yet.
    # Without it, the run passes only when the visual item has actually passed too.
    return 0 if report['spec_accepted'] or (args.machine_only and report['machine_passed']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
