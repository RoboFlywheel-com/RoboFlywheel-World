# SPDX-License-Identifier: Apache-2.0
"""Measured thin-covering boundary ruler. No physics or visual judgement is inferred."""
import math
from rfw_w.scenes.contracts.surface_appearance import require
from rfw_w.scenes.assets.background_tools import keep_out,bounds,intersects


def covering_vertices(surface,center,size):
    require(isinstance(center,list) and len(center)==2 and isinstance(size,list) and len(size)==2
            and all(type(x) in (int,float) and math.isfinite(x) for x in center+size)
            and all(.05<=x<=3 for x in size),'COVERING_OUTSIDE_SURFACE','coordinates')
    u,v=center;w,h=size
    local=[[u-w/2,v-h/2],[u+w/2,v-h/2],[u+w/2,v+h/2],[u-w/2,v+h/2]]
    vertices=[]
    for a,b in local:
        # One millimetre offsets the decorative face without altering the support mesh.
        plane=surface['plane_m']+.001*surface['normal'][{'x':0,'y':1,'z':2}[surface['axis']]]
        vertices.append([plane,a,b] if surface['axis']=='x' else [a,plane,b] if surface['axis']=='y' else [a,b,plane])
    return vertices


def verify_covering(item,surfaces,plan,core,background,accepted):
    surface=surfaces.get(item['surface_id'])
    require(surface is not None,'COVERING_SURFACE_MISSING',item['surface_id'])
    require(item['room_id']==surface['room_id'] and item['normal']==surface['normal'],
            'COVERING_NORMAL_INVALID',item['instance_id'])
    vertices=covering_vertices(surface,item['local_position_m'],item['size_m'])
    require(item['vertices_m']==vertices,'COVERING_NORMAL_INVALID','world transform')
    lo_u,lo_v,hi_u,hi_v=surface['bounds_uv_m'];u,v=item['local_position_m'];w,h=item['size_m']
    require(u-w/2>=lo_u+.02 and u+w/2<=hi_u-.02 and v-h/2>=lo_v+.02 and v+h/2<=hi_v-.02,
            'COVERING_OUTSIDE_SURFACE',item['instance_id'])
    box=[min(p[k] for p in vertices) for k in range(3)]+[max(p[k] for p in vertices) for k in range(3)]
    if surface['kind']=='floor':
        require(item['collision_policy']=='visual_only','COVERING_NORMAL_INVALID','floor collision policy')
        # Footprint screening uses a finite height only as a geometric reservation.
        box[2]=0;box[5]=.01
        require(not any(intersects(box,z) for z in keep_out(plan,core)),
                'COVERING_KEEP_OUT_VIOLATION',item['instance_id'])
        require(not any(intersects(box,bounds(i),.02) for i in core['instances']+background['instances']),
                'COVERING_OVERLAP',item['instance_id'])
    else:
        require(item['collision_policy']=='static_clearance_proxy','COVERING_NORMAL_INVALID','wall policy')
        axis={'x':0,'y':1}[surface['axis']];box[axis]-=.002;box[axis+3]+=.002
        require(not any(intersects(box,bounds(i),.02) for i in core['instances']+background['instances']),
                'COVERING_OVERLAP',item['instance_id'])
    for other in accepted:
        if other['surface_id']!=item['surface_id']:continue
        x,y=other['local_position_m'];ow,oh=other['size_m']
        require(abs(u-x)>=(w+ow)/2+.02 or abs(v-y)>=(h+oh)/2+.02,'COVERING_OVERLAP',item['instance_id'])
    return {'verified':True,'collision_policy':item['collision_policy'],'physical':'not_run'}
