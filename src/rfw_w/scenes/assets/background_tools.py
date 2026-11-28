# SPDX-License-Identifier: Apache-2.0
"""Measured optional furnishing primitives shared by floor and wall tools.

Only geometry and deterministic screening run here. Absolute library locations stay
inside HssdConfig; output ledgers carry prepared paths relative to their bundle.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from rfw_w.scenes.assets.providers.hssd import load_synset_index, measure_glb_extents, prepare_furniture, HssdError
from rfw_w.scenes.assets.room_tools import RoomAssetError, _instance
from rfw_w.scenes.solvers.house_tools import _read_obj_groups, measure_house_access

SYNSETS={'cabinet':('cabinet.n.01',),'bookshelf':('bookcase.n.01',),'chair':('chair.n.01',),
         'side_table':('end_table.n.01',),'picture':('picture_frame.n.01',),'mirror':('wall_mirror.n.01',)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounds(item):
    d=item['dimensions_m']; p=item['pose_xyz_m']; yaw=item.get('yaw_rad',0)
    w=abs(math.cos(yaw))*d[0]+abs(math.sin(yaw))*d[1]
    depth=abs(math.sin(yaw))*d[0]+abs(math.cos(yaw))*d[1]
    return [p[0]-w/2,p[1]-depth/2,p[2],p[0]+w/2,p[1]+depth/2,p[2]+d[2]]


def intersects(a,b,margin=0):
    return all(a[k]<b[k+3]+margin-1e-6 and b[k]<a[k+3]+margin-1e-6 for k in range(3))


def keep_out(plan, core):
    c=plan['corridor']; zones=[[*[c['x_span_m'][0],c['y_span_m'][0]],0,c['x_span_m'][1],c['y_span_m'][1],3]]
    for item in core['instances']:
        if item['priority']=='core' and item.get('support_surface'):
            box=bounds(item)
            # Keep an approach band around both task support fixtures. This is a static
            # reservation, not a claim of measured reachability for a particular robot.
            zones.append([box[0]-.3,box[1]-.3,0,box[3]+.3,box[4]+.65,3])
    return zones


def screen_floor(item,plan,core,accepted):
    b=bounds(item); r=next(r for r in plan['rooms'] if r['id']==item['room_id'])['bounds_xy_m']
    if not all(math.isfinite(v) for v in b) or abs(b[2])>1e-5 or b[0]<r[0]+.06 or b[1]<r[1]+.06 or b[3]>r[2]-.06 or b[4]>r[3]-.06:
        raise RoomAssetError('FURNISHING_OUT_OF_BOUNDS',item['role_id'])
    if any(intersects(b,z) for z in keep_out(plan,core)):
        raise RoomAssetError('CORE_KEEP_OUT_VIOLATION',item['role_id'])
    if any(intersects(b,bounds(other),.05) for other in core['instances']+accepted):
        raise RoomAssetError('FURNISHING_COLLISION',item['role_id'])


def retrieve_background_candidates(role,config, *, limit=8, scan_limit=64):
    index=load_synset_index(config); rejected=[]; admitted=[]
    rows=[r for syn in SYNSETS.get(role['category'],()) for r in index.get(syn,[])][:scan_limit]
    for row in rows:
        aid=str(row['id']); extent=measure_glb_extents(config.glb_path(aid))
        if extent is None:
            rejected.append({'asset_id':aid,'code':'ASSET_UNAVAILABLE'}); continue
        dims=extent.to_list()
        if role['placement_kind']=='floor':
            usable=all(.08<=d<=cap for d,cap in zip(dims,(1.25,1.05,2.2)))
        else:
            # A certified static panel requires a horizontal thin axis and substantial
            # upright height. Reject horizontal/stand-mounted mirrors rather than rotate
            # arbitrary assets into a wall fixture with no mounting evidence.
            usable=.2<=dims[0]<=1.3 and .001<=dims[1]<=.16 and .2<=dims[2]<=1.4
        if not usable:
            rejected.append({'asset_id':aid,'code':'SIZE_UNSUPPORTED' if role['placement_kind']=='floor' else 'WALL_MOUNT_UNCERTIFIED'}); continue
        admitted.append((abs(dims[0]-.65),aid,dims))
    admitted.sort()
    return {'role_id':role['id'],'synsets':list(SYNSETS.get(role['category'],())),
            'candidates':[{'asset_id':a,'dimensions_m':d} for _,a,d in admitted[:limit]],
            'rejected':rejected,'examined':len(rows),'scan_limit':scan_limit,'max_candidates':limit}


def place_floor(role,query,config,plan,core,accepted,root):
    attempts=[]; room=next(r for r in plan['rooms'] if r['id']==role['room_id'])['bounds_xy_m']
    for candidate in query['candidates']:
        try:
            prepared=prepare_furniture(config,candidate['asset_id'])
        except (HssdError,OSError,ValueError):
            attempts.append({'asset_id':candidate['asset_id'],'code':'ASSET_PREPARE_FAILED'}); continue
        d=prepared.dimensions_m.to_list()
        for y in (3.5,3.0,2.8,.5):
            for x in (room[0]+.12+d[0]/2,room[2]-.12-d[0]/2,(room[0]+room[2])/2):
                item=_instance(role,prepared,[x,y,0],output_root=root)
                item.update(purpose=role['purpose'],placement_kind='floor')
                try:
                    screen_floor(item,plan,core,accepted)
                except RoomAssetError as error:
                    attempts.append({'asset_id':candidate['asset_id'],'pose_xyz_m':item['pose_xyz_m'],'code':error.code}); continue
                return item,attempts
    return None,attempts


def measure_wall_mount_surfaces(plan,geometry):
    measured=measure_house_access(plan,geometry); groups=_read_obj_groups(geometry/'house.obj'); anchors={}
    for name,box in groups.items():
        if not name.startswith('wall_') or 'lintel' in name: continue
        x0,y0,z0,x1,y1,z1=box
        axis='x' if x1-x0<y1-y0 else 'y'
        for room in plan['rooms']:
            r=room['bounds_xy_m']; cx=(r[0]+r[2])/2; cy=(r[1]+r[3])/2
            if axis=='x':
                if y1<=r[1] or y0>=r[3] or x1<r[0]-.06 or x0>r[2]+.06: continue
                normal=[1,0,0] if cx>(x0+x1)/2 else [-1,0,0]
                plane=x1 if normal[0]>0 else x0; span=[max(y0,r[1]),min(y1,r[3])]
            else:
                if x1<=r[0] or x0>=r[2] or y1<r[1]-.06 or y0>r[3]+.06: continue
                normal=[0,1,0] if cy>(y0+y1)/2 else [0,-1,0]
                plane=y1 if normal[1]>0 else y0; span=[max(x0,r[0]),min(x1,r[2])]
            key=name+':'+room['id']
            anchors[key]={'wall_segment_id':name,'room_id':room['id'],'normal':normal,'axis':axis,
                          'plane_m':plane,'span_m':span,'height_m':[z0,z1], 'mesh_sha256':measured['mesh_sha256']}
    return anchors


def screen_wall(item,anchor,plan,core,accepted):
    if not all(math.isfinite(v) for v in item['pose_xyz_m']+item['dimensions_m']) or not math.isfinite(item.get('yaw_rad',0)):
        raise RoomAssetError('WALL_SIDE_MISMATCH',item['role_id'])
    if anchor['room_id']!=item['room_id'] or item['mount_normal']!=anchor['normal']:
        raise RoomAssetError('WALL_SIDE_MISMATCH',item['role_id'])
    b=bounds(item); axis=0 if anchor['axis']=='x' else 1; tangent=1-axis; n=anchor['normal'][axis]
    plane=anchor['plane_m']; nearest=b[axis] if n>0 else b[axis+3]
    if abs(nearest-plane-n*.002)>1e-5:
        raise RoomAssetError('WALL_SIDE_MISMATCH',item['role_id'])
    low,high=anchor['span_m']; z0,z1=anchor['height_m']
    if b[tangent]<low+.08 or b[tangent+3]>high-.08 or b[2]<z0+.15 or b[5]>z1-.15:
        raise RoomAssetError('WALL_OPENING_OVERLAP',item['role_id'])
    if any(intersects(b,z) for z in keep_out(plan,core)):
        raise RoomAssetError('CORE_KEEP_OUT_VIOLATION',item['role_id'])
    if any(intersects(b,bounds(o),.01) for o in core['instances']+accepted):
        raise RoomAssetError('WALL_FIXTURE_COLLISION',item['role_id'])


def place_wall(role,query,config,plan,core,accepted,root,anchors):
    attempts=[]
    for candidate in query['candidates']:
        try: prepared=prepare_furniture(config,candidate['asset_id'])
        except (HssdError,OSError,ValueError):
            attempts.append({'asset_id':candidate['asset_id'],'code':'ASSET_PREPARE_FAILED'}); continue
        d=prepared.dimensions_m.to_list()
        if not (.001<=d[1]<=.16 and .2<=d[2]<=1.4):
            attempts.append({'asset_id':candidate['asset_id'],'code':'WALL_MOUNT_UNCERTIFIED'});continue
        # The far exterior wall avoids the task approach and stays visible in a
        # front-wall cutaway preview. Other measured walls remain valid fallbacks.
        for key,a in sorted(anchors.items(),key=lambda pair: (0 if pair[0].startswith('wall_top') else 1,pair[0])):
            if a['room_id']!=role['room_id']: continue
            axis=0 if a['axis']=='x' else 1; n=a['normal'][axis]
            pose=[0.,0.,1.8-d[2]/2];pose[axis]=a['plane_m']+n*(d[1]/2+.002)
            pose[1-axis]=sum(a['span_m'])/2
            item=_instance(role,prepared,pose,output_root=root)
            item.update(purpose=role['purpose'],placement_kind='wall',wall_anchor=key,mount_normal=a['normal'],
                        yaw_rad=(-math.pi/2 if n>0 else math.pi/2) if axis==0 else (0 if n>0 else math.pi))
            try: screen_wall(item,a,plan,core,accepted)
            except RoomAssetError as error:
                attempts.append({'asset_id':candidate['asset_id'],'wall_anchor':key,'code':error.code});continue
            item['mount_certificate']={'method':'upright_thin_panel_aabb','thickness_axis':'Y','measured_dimensions_m':d,
                                       'physical_attachment':'not_run','rear_face_orientation':'not_run'}
            return item,attempts
    return None,attempts


def verify_prepared_instance(item,root,query):
    if item['provider']!='hssd' or item['asset_id'] not in {c['asset_id'] for c in query['candidates']}:
        raise RoomAssetError('ASSET_UNCERTIFIED',item['role_id'])
    metadata=json.loads((root/'prepared'/item['asset_id']/'prepared.json').read_text())
    if metadata['asset']['asset_id']!=item['asset_id'] or metadata['asset']['provider']!='hssd':
        raise RoomAssetError('ASSET_UNCERTIFIED',item['role_id'])
    if metadata['dimensions_m']!=item['dimensions_m']:
        raise RoomAssetError('ASSET_UNCERTIFIED',item['role_id'])
    # Measure the visual mesh: the simplified collision hull can shrink slightly
    # (chair probe lost 3.4 mm of height), so it cannot certify visible extents.
    vertices=[]
    for kind in ('visual','collision'):
        ref=Path(item[kind+'_ref'])
        if ref.is_absolute() or '..' in ref.parts: raise RoomAssetError('ASSET_UNCERTIFIED',item['role_id'])
        if ref.as_posix()!=f"prepared/{item['asset_id']}/{metadata[kind+'_obj']}":
            raise RoomAssetError('ASSET_UNCERTIFIED',item['role_id'])
        path=root/ref
        if digest(path)!=item[kind+'_sha256']: raise RoomAssetError('ASSET_UNCERTIFIED',item['role_id'])
        if kind=='visual':
            vertices=[list(map(float,l.split()[1:4])) for l in path.read_text().splitlines() if l.startswith('v ')]
    if not vertices: raise RoomAssetError('ASSET_UNCERTIFIED',item['role_id'])
    dims=[max(v[k] for v in vertices)-min(v[k] for v in vertices) for k in range(3)]
    if any(abs(a-b)>1e-4 for a,b in zip(dims,item['dimensions_m'])):
        raise RoomAssetError('ASSET_UNCERTIFIED',item['role_id'])
