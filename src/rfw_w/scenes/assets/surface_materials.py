# SPDX-License-Identifier: Apache-2.0
"""Local PBR retrieval with explicit vocabulary; no network or silent style fallback."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
import shutil
from rfw_w.scenes.contracts.surface_appearance import CHANNEL_SPACES, require, relative_ref, SurfaceAppearanceError

SOLIDS = {'solid:sage': [0.30,0.46,0.34,1], 'solid:terracotta':[0.58,0.26,0.16,1],
          'solid:blue':[0.20,0.33,0.49,1]}
WORDS = {'wood':{'wood','oak','wooden','hardwood','floor','timber','木','木纹','木地板'},
         'plaster':{'plaster','stucco','paint','wall','gray','grey','灰泥','墙','墙壁'}}
SUFFIXES = {'base_color':'_color','normal_gl':'_normalgl','roughness':'_roughness'}


def materials_root():
    value=os.environ.get('PVSCENE_MATERIALS_ROOT')
    require(value and Path(value).is_dir(),'MATERIAL_ENV_MISSING','PVSCENE_MATERIALS_ROOT')
    return Path(value)


def search_surface_materials(description, surface_kind, *, limit=8):
    require(isinstance(description,str) and 0<len(description)<=400,'MATERIAL_NOT_FOUND','description')
    require(surface_kind in ('floor','wall_set') and type(limit) is int and 1<=limit<=8,
            'SURFACE_BINDING_INVALID','query')
    tokens=set(re.findall(r'[a-z]+',description.lower()))
    found=[]
    for directory in sorted(materials_root().iterdir()):
        if not directory.is_dir(): continue
        family=next((f for f in WORDS if f in directory.name.lower()),None)
        matches=(tokens & WORDS.get(family,set())) | {w for w in WORDS.get(family,set()) if w in description and not w.isascii()}
        if directory.name==description:matches={'exact_id'}
        if not matches: continue
        channels={k:any(p.stem.lower().endswith(suffix) for p in directory.iterdir()
                      if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png')) for k,suffix in SUFFIXES.items()}
        found.append({'material_id':directory.name,'provider':'local_pbr','channels':channels,
                      'matched_terms':sorted(matches),'score':len(matches),'surface_kind':surface_kind})
    for mid,rgba in SOLIDS.items():
        name=mid.split(':')[1]
        if name in tokens or mid==description:
            found.append({'material_id':mid,'provider':'solid','channels':{},'matched_terms':[name],
                          'score':1,'surface_kind':surface_kind})
    found.sort(key=lambda c:(-c['score'],c['material_id']))
    return {'description':description,'surface_kind':surface_kind,'ranking_method':'explicit_local_vocabulary',
            'coverage':'local directories and declared solid palette; no CLIP or artwork generation',
            'candidates':found[:limit],'rejected':[]}


def prepare_material(mid, root):
    if mid in SOLIDS:
        return {'provider':'solid','base_color_rgba':SOLIDS[mid],'roughness':.82,'channels':{}}
    relative_ref(mid)
    directory=materials_root()/mid
    require(directory.is_dir() and directory.resolve().is_relative_to(materials_root().resolve()),'MATERIAL_NOT_FOUND',mid)
    result={'provider':'local_pbr','base_color_rgba':[1,1,1,1],'roughness':.82,'channels':{}}
    from PIL import Image
    for channel,suffix in SUFFIXES.items():
        files=sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png')
                     and p.stem.lower().endswith(suffix))
        require(len(files)==1,'MATERIAL_CHANNEL_MISSING',mid+':'+channel)
        path=files[0]
        require(path.resolve().is_relative_to(directory.resolve()),'SURFACE_BINDING_INVALID','texture symlink')
        try:
            with Image.open(path) as im:
                im.load();require(min(im.size)>0,'MATERIAL_IMAGE_INVALID',mid)
        except (OSError,ValueError):raise SurfaceAppearanceError('MATERIAL_IMAGE_INVALID',mid) from None
        dest=root/'materials'/mid/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists(): require(dest.read_bytes()==path.read_bytes(),'MATERIAL_DIGEST_MISMATCH',mid)
        else:shutil.copyfile(path,dest)
        result['channels'][channel]={'ref':dest.relative_to(root).as_posix(),
                                    'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                                    'color_space':CHANNEL_SPACES[channel]}
    return result
