# SPDX-License-Identifier: Apache-2.0
"""Fresh physical episodes and Isaac RTX media; local JSON contains resource paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EPISODES = "rfw_w.scenes.workers.isaac.episodes"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--physics-only', action='store_true')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    env = os.environ.copy()
    for k in ['PYTHONPATH','PYTHONHOME','CONDA_PREFIX','CONDA_DEFAULT_ENV']: env.pop(k,None)
    report = {'target':32,'clips':[]}
    def run(args, log):
        with log.open('w') as f:
            subprocess.run([str(v) for v in args], stdout=f, stderr=subprocess.STDOUT, env=env, check=True)
    for item in cfg['episodes']:
        if not item['id'].replace('_','').replace('-','').isalnum():
            raise ValueError('Invalid episode ID')
        folder = a.output/item['id'];folder.mkdir(exist_ok=a.resume)
        row = {'id':item['id'],'physics':'not_run','media':'not_run'}
        try:
            base = json.loads(Path(item['base_config']).read_text())
            base.update(item['overrides']);base['output_dir']=str(folder/'run');base['capture']=False
            cp=folder/'execution.local';cp.write_text(json.dumps(base))
            runroot=folder/'run'
            if not (runroot/'trajectory.json').exists():
                run([cfg['isaac_python'],'-m', EPISODES + '.expert_pick_place','--config',cp],folder/'physics.log.local')
            run([cfg['runtime_python'],'-m', EPISODES + ('.validate_interaction' if base.get('interaction_task') else '.validate_scene'),'--config',cp],folder/'physics_validation.log.local')
            row['physics']='passed';row['trajectory_sha256']=sha(runroot/'trajectory.json')
            if not a.physics_only:
                stage=folder/'stage'
                if not (stage/'scene.usdc').exists():
                    media_cfg={'id':item['id'],'title':item['title'],'output':str(stage),'trajectory':str(runroot/'trajectory.json'),'validation':str(runroot/'validation.json'),'blend':item['source_blend'],'layout':item['layout'],'g1_urdf':base['g1_urdf']}
                    mp=folder/'stage.local';mp.write_text(json.dumps(media_cfg))
                    run([cfg['runtime_python'],'-m', EPISODES + '.prepare_unified_stage','--','--config',mp],folder/'prepare.log.local')
                run([cfg['runtime_python'],'-m', EPISODES + '.validate_unified_stage',stage],folder/'stage_validation.log.local')
                if not (stage/'task.mp4').exists():
                    run([cfg['isaac_python'],'-m', EPISODES + '.render_unified_rtx','--stage',stage,'--renderer','realtime','--gpu',cfg.get('gpu',0)],folder/'render.log.local')
                run([cfg['runtime_python'],'-m', EPISODES + '.validate_unified_media',stage],folder/'media_validation.log.local')
                row['media']='passed'
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            if row['physics']=='not_run': row['physics']='failed'
            else: row['media']='failed'
            row['failure']=str(e)
        report['clips'].append(row)
        report['accepted_physics'] = sum(x['physics']=='passed' for x in report['clips'])
        report['accepted_media'] = sum(x['media']=='passed' for x in report['clips'])
        (a.output/'progress.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
    return 0 if all(x['physics']=='passed' and (a.physics_only or x['media']=='passed') for x in report['clips']) else 1


if __name__=='__main__':raise SystemExit(main())
