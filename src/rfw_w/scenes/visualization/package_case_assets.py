#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble an audited local handoff without editing any render or acceptance evidence."""
import argparse
from pathlib import Path
import shutil
import zipfile
from rfw_w.scenes.visualization.paper_case_common import content_hash,digest,file_inventory,fresh,read,write


def package(render,audit,review,output):
    render=Path(render).resolve();audit=read(audit);review=read(review)
    if not audit['machine_passed'] or audit['gaps']:raise ValueError('PAPER_UNSUPPORTED_CLAIM(machine audit)')
    if review['status']!='passed' or review['scene_sha256']!=digest(render/'scene/case.usda'):raise ValueError('PAPER_EVIDENCE_SCENE_MISMATCH(visual review)')
    for rel,sha in review['artifact_hashes'].items():
        if digest(render/'assets'/rel)!=sha:raise ValueError('PAPER_EVIDENCE_SCENE_MISMATCH(reviewed artifact)')
    output=fresh(output);case_id=read(render/'case_config.json')['case_id'];bundle=output/case_id
    shutil.copytree(render,bundle,ignore=shutil.ignore_patterns('*.log','*.txt'))
    write(bundle/'verification.json',{**audit,'visual_review':review,'spec_accepted':True,'scope':'SPEC-012 static asset package; not production readiness'})
    (bundle/'README.md').write_text('''# Paper dual-Piper workbench — curated static case

Start with `preview.jpg`. Independent PNGs are in `assets/`; all but `isaac_view.png`
have transparent backgrounds. `assets/room_plan.svg` contains editable geometry and text.

- `core_iso.png`: mug, bowl, plate, soap bottle, hinged microwave, sponge, plush bear, two Piper X arms and edge mounts.
- `core_operating_view.png`, `core_topdown.png`: door-facing view and overhead arrangement.
- `shell_clay_iso.png`: the same generated architectural shell, cut away only for the view.
- `material_01..03.png`: actual exported PBR sampled inside source UV islands.
- `periphery_01..03.png`: the storage, chair and plant used in the room.
- `target_*.png`, `robot_piper_left.png`, `robot_piper_right.png`: isolated details of each selected target and robot.
- `delivery_iso.png`, `isaac_view.png`: the same case in Isaac, with no dynamics executed.

`scene/case.usda` references the local `scene/sources/` and `scene/shell/` packages.
The USD preserves source materials; studio lighting, palette adjustments and cutaways are
view-session overrides, recorded in `isaac/render_report.json` and reproduced by the renderer.
`frozen_core.json`, `room_fit.json` and `provenance.json` identify the frozen core and transforms.
The RoomPlan core field carries the *figure case* digest, not a production solver certificate.

Two Piper X arms stand side by side on the front long edge of a 1.8 x 0.9 m table.
The microwave faces the left arm; tableware, cleaning items and the bear are grouped
compactly in front of the arms. This is a visual arrangement, not a reachability certificate. The door approach gap is checked in XY; it is not a motion sweep certificate.
`case_config.json` records these groups, operating-face direction and all local poses.
`selection/` preserves real-source Blender candidate sheets; final assets are Isaac renders.
New sponge/bear visuals come from the softsim catalog and original physics.json provenance.
They were not locally tetrahedralised and are not simulation-ready USD certificates.

The source libraries were selected manually. No retrieval, LLM compilation or VLM was run.
No settle, soft-body simulation, articulated task, IK, navigation or rollout is certified.
The workbench and mounting plate are explicitly constructed geometry. Both Piper arms and the microwave have recorded
static joint poses. Source USD dependencies were checked again after relocating the package.
OmniPBR.mdl is an Isaac-provided runtime material dependency.

This is a local research handoff. Asset redistribution licenses remain unverified.
Final PPTX/SVG pipeline layout and publication typography are outside this asset package.
''')
    write(bundle/'figure_asset_map.json',{
        'A01':{'files':['case_config.json'],'status':'curated input; compilation not_run'},
        'A02':{'files':['assets/core_iso.png','assets/core_operating_view.png','assets/core_topdown.png','frozen_core.json'],'status':'rendered'},
        'A03':{'files':['assets/room_plan.svg','room_plan.json'],'status':'generated from coordinates'},
        'A04':{'files':['assets/shell_clay_iso.png','shell_gate.json'],'status':'rendered; host gate passed'},
        'A05':{'files':[f'assets/material_{i:02d}.png' for i in range(1,4)],'status':'rendered'},
        'A06':{'files':['render_config.json','isaac/render_report.json'],'status':'figure lighting recorded; icon deferred to layout'},
        'A07':{'files':[f'assets/periphery_{i:02d}.png' for i in range(1,4)],'status':'curated; rendered'},
        'A08':{'files':['assets/delivery_iso.png','scene/case.usda'],'status':'figure package; not production SceneIR'},
        'A09':{'files':['assets/isaac_view.png','isaac/render_report.json'],'status':'rendered; settle not_run'},
        'A10':{'files':['verification.json'],'status':'static geometry/source audit only'},
        'A11':{'files':['provenance.json'],'status':'VLM not_run'},
        'A12':{'files':['provenance.json'],'status':'refinement/task rollout not_run; icons deferred to layout'}
    })
    write(bundle/'package_files.json',file_inventory(bundle))
    archive=output/(case_id+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(output))
    print('package',output.name,'zip_bytes',archive.stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render-run',required=True);p.add_argument('--audit',required=True);p.add_argument('--visual-review',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();package(a.render_run,a.audit,a.visual_review,a.output_dir)
