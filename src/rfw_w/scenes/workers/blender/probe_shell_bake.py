#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe the pinned upstream bake on one source floor; never overwrite the source blend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    import bpy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resolution',type=int,default=256)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output_dir.exists():raise RuntimeError('output exists; use fresh evidence')
    args.output_dir.mkdir(parents=True)
    root=Path(os.environ['RFW_INFINIGEN_ROOT'])
    from infinigen.tools.export import bake_object
    source=Path(bpy.data.filepath);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    floors=[o for o in bpy.context.scene.objects if o.type=='MESH' and any(c.name=='unique_assets:room_floor' for c in o.users_collection)]
    if len(floors)!=1:raise RuntimeError(f'MATERIAL_BINDING_MISMATCH(floors={len(floors)})')
    obj=floors[0]
    preferences=bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type='CUDA';preferences.get_devices()
    devices=[d for d in preferences.devices if d.type=='CUDA']
    if not devices:raise RuntimeError('MATERIAL_BAKE_UNAVAILABLE(no CUDA device)')
    for d in preferences.devices:d.use=d in devices
    bpy.context.scene.render.engine='CYCLES'
    bpy.context.scene.cycles.device='GPU'
    bpy.context.scene.cycles.samples=8
    for other in bpy.context.scene.objects:other.hide_render=other!=obj
    obj.hide_viewport=False;obj.hide_set(False)
    started=time.monotonic()
    bake_object(obj,args.output_dir,args.resolution,True,export_name='floor')
    files=sorted(args.output_dir.glob('*.png'))
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_dir/'baked_probe.blend'))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    report={'source_sha256':source_hash,'object':obj.name,'resolution':args.resolution,
            'elapsed_s':round(time.monotonic()-started,3),'textures':[p.name for p in files],
            'required_channels_present':all((args.output_dir/f'floor_{c}.png').is_file() for c in ['DIFFUSE','ROUGHNESS','NORMAL']),
            'scope':'upstream single floor probe; material fidelity not accepted'}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    if not report['required_channels_present']:raise RuntimeError('MATERIAL_CHANNEL_MISSING')


if __name__=='__main__':main()
