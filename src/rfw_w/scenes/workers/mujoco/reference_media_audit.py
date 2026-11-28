# SPDX-License-Identifier: Apache-2.0
"""Verify media time/source binding and report task-object visibility honestly."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from rfw_w.scenes.assets.articulated_urdf import digest


def audit(run,media):
    run,media=Path(run),Path(media)
    validation=json.loads((run/'task_validation.json').read_text())
    raw=json.loads((run/'trajectory.json').read_text());rows=raw['samples']
    manifest=json.loads((media/'media.json').read_text());visibility=json.loads((media/'visibility.json').read_text())
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height,r_frame_rate','-of','json',str(media/'task.mp4')],text=True))['streams'][0]
    checks=dict(task_passed=validation['status']=='passed',source_hash=manifest['source_trajectory_sha256']==digest(run/'trajectory.json'),
        frame_count=int(probe['nb_frames'])==len(rows)==manifest['frames']==len(visibility),
        resolution=probe['width']==1280 and probe['height']==720,frame_rate=probe['r_frame_rate']=='30/1')
    if manifest.get('source_world_sha256'):
        recording=json.loads((run/'recording.json').read_text())
        checks['source_world_binding']=manifest['source_world_sha256']==recording['world_sha256']
    details={}
    tracked = [(obj['role'], obj['leaf']) for obj in manifest['tracked_objects']] if manifest.get('tracked_objects') else [('caliper','11_2_0007'),('screwdriver','9_6_0000')]
    for label,leaf in tracked:
        frames=[r for r in visibility if r['phase'] in (label+'_close',label+'_lift',label+'_transfer',label+'_lower',label+'_release')]
        counts=[r['pixels']['V13_'+leaf] for r in frames]
        ratio=sum(p>8 for p in counts)/len(counts) if counts else 0.
        longest=streak=0
        for p in counts:
            streak=0 if p>8 else streak+1;longest=max(longest,streak)
        details[label]=dict(action_frames=len(counts),visible_fraction=ratio,longest_occlusion_s=longest/30,
            minimum_pixels=min(counts,default=0),note='finger occlusion at contact permitted; uncut timeline retained')
        checks[label+'_mostly_visible']=ratio>.8
    return dict(status='passed' if all(checks.values()) else 'failed',checks=checks,objects=details,
        camera_review='automated per-frame visibility; human review recorded separately',rgb_depth_instance_scope='RGB full timeline; depth/instance phase keyframes')
