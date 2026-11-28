# SPDX-License-Identifier: Apache-2.0
"""Gate completed Isaac media using actual video decoding and source bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visual_check(video, seconds):
    import numpy as np
    samples=[]
    for t in [0, seconds*.25, seconds*.5, seconds*.75, max(0,seconds-.1)]:
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(video),'-frames:v','1','-vf','scale=320:180','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
        pixels=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3).astype(float)
        red=((pixels[:,0]>100)&(pixels[:,0]>1.45*pixels[:,1])&(pixels[:,0]>1.45*pixels[:,2])).mean()
        samples.append({'time_s':t,'red_dominant_fraction':float(red),'std':float(pixels.std())})
    return {'status':'passed' if all(s['red_dominant_fraction']<.65 and s['std']>8 for s in samples) else 'failed','samples':samples,'scope':'Sampled pathological red cast and blank-frame screen; human composition review still required'}


def validate(root):
    provenance=json.loads((root/'render_provenance.json').read_text())
    stage=json.loads((root/'stage_validation.json').read_text())
    video=root/'task.mp4'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(video)]))['streams'][0]
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    visual=visual_check(video,float(probe['duration']))
    checks={'visual_screen_passed':visual['status']=='passed','complete_render':provenance['status']=='rendered_pending_media_check',
        'isaac_renderer':provenance['renderer'].startswith('Isaac Sim 4.5 RTX '),
        'stage_passed':stage['status']=='passed' and bool(stage['checks']) and all(stage['checks'].values()),
        'stage_hash_matches':sha(root/'scene.usdc')==stage['scene_sha256']==provenance['scene_sha256'],
        'trajectory_hash_matches':sha(root/'trajectory.json')==stage['trajectory_sha256']==provenance['trajectory_sha256'],
        'video_hash_matches':sha(video)==provenance['video_sha256'],
        'all_frames_present':int(probe['nb_read_frames'])==provenance['frames'],
        'resolution_1080p':(probe['width'],probe['height'])==(1920,1080),
        'fps_30':probe['r_frame_rate']=='30/1',
        'duration_matches':abs(float(probe['duration'])-provenance['frames']/30)<.035,
        'full_decode_passed':decode.returncode==0 and not decode.stderr.strip(),
        'poster_and_result_present':all((root/n).is_file() for n in ('poster.png','result.png'))}
    report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,renderer=provenance['renderer'],
        visual_screen=visual,frames=int(probe['nb_read_frames']),seconds=float(probe['duration']),video_sha256=sha(video),
        scene_sha256=stage['scene_sha256'],trajectory_sha256=stage['trajectory_sha256'])
    (root/'media_validation.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);a=p.parse_args()
    result=validate(a.stage);print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(1)
