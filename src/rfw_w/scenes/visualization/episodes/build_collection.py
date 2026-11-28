# SPDX-License-Identifier: Apache-2.0
"""Build a local playback collection from current independently accepted materials."""
import argparse
import hashlib
import html
from importlib.resources import files
import json
from pathlib import Path


def accepted(folder):
    reports=[json.loads((folder/n).read_text()) for n in ['physics_validation.json','stage_validation.json','media_validation.json']]
    if any(r.get('status')!='passed' or not r.get('checks') or not all(r['checks'].values()) for r in reports):return False
    media=reports[-1]
    visual=folder/'visual_review.json'
    if visual.exists() and json.loads(visual.read_text()).get('status')!='passed':return False
    stage=reports[1]
    if not media.get('renderer','').startswith('Isaac Sim 4.5 RTX '):return False
    if any(stage.get(k)!=media.get(k) for k in ['scene_sha256','trajectory_sha256']):return False
    return all(hashlib.sha256((folder/n).read_bytes()).hexdigest()==media[k] for n,k in [('task.mp4','video_sha256'),('scene.usdc','scene_sha256'),('trajectory.json','trajectory_sha256')])


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--batches',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    candidates=list(a.baseline.iterdir())
    for root in a.batches:candidates.extend(root.glob('*/stage'))
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'media').mkdir()
    items=[];seen=set()
    for folder in candidates:
        try:
            if not accepted(folder):continue
            r=json.loads((folder/'render_provenance.json').read_text());digest=json.loads((folder/'media_validation.json').read_text())['trajectory_sha256']
            if digest in seen:continue
            seen.add(digest);key=r['id']
            for name in ['task.mp4','poster.png','result.png','physics_validation.json','stage_validation.json','media_validation.json']:
                (a.output/'media'/f'{key}_{name}').symlink_to((folder/name).resolve())
            items.append({'id':key,'title':r['title'],'layout':r['layout'],'trajectory_sha256':digest,'video':f'media/{key}_task.mp4','poster':f'media/{key}_poster.png','scope':'局部操作' if key.startswith('T12_') else '已验执行'})
        except (OSError,KeyError,ValueError):continue
    template=files('rfw_w.scenes.visualization.episodes').joinpath('collection.html').read_text()
    data=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
    (a.output/'index.html').write_text(template.replace('__DATA__',data).replace('__COUNT__',str(len(items))))
    (a.output/'catalog.json').write_text(json.dumps({'target':32,'accepted':len(items),'clips':items},ensure_ascii=False,indent=2));print('ACCEPTED',len(items))


if __name__=='__main__':main()
