# SPDX-License-Identifier: Apache-2.0
"""Deterministic bounded pose candidates and independent rotated footprint checks."""
from copy import deepcopy
import math
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.layout_revision import require, validate_permission, verify_pose_difference, dependents
from rfw_w.scenes.geometry.polygon import Polygon, contains_polygon, overlaps


def footprint(item):
    x, y, _ = item['pose_xyz_m']; w, d, _ = item['dimensions_m']
    c, s = math.cos(item.get('yaw_rad', 0)), math.sin(item.get('yaw_rad', 0))
    return Polygon.from_sequence([(x+c*a-s*b, y+s*a+c*b)
                                  for a,b in ((-w/2,-d/2),(w/2,-d/2),(w/2,d/2),(-w/2,d/2))])


def measure_layout(core, background, plan):
    """Actual oriented rectangles; support contact allowed, volume penetration rejected."""
    items = core['instances'] + background.get('instances', [])
    by_id = {i['role_id']: i for i in items}
    rooms = {r['id']: Polygon.rectangle(*r['bounds_xy_m']) for r in plan['rooms']}
    corridor = plan['corridor']
    keepout = Polygon.rectangle(corridor['x_span_m'][0], corridor['y_span_m'][0],
                                corridor['x_span_m'][1], corridor['y_span_m'][1])
    for item in items:
        poly = footprint(item)
        require(contains_polygon(rooms[item['room_id']], poly), 'LAYOUT_COLLISION', 'room ' + item['role_id'])
        support = item.get('support_role_id')
        if support:
            surface = by_id.get(support, {}).get('support_surface')
            require(surface is not None and contains_polygon(Polygon.from_sequence(surface['polygon_xy']), poly),
                    'LAYOUT_COLLISION', 'support ' + item['role_id'])
            require(abs(item['pose_xyz_m'][2]-surface['plane_z_m']-.002)<1e-5,
                    'LAYOUT_COLLISION', 'support height')
        elif item.get('placement_kind') != 'wall':
            require(abs(item['pose_xyz_m'][2])<1e-5 and not overlaps(poly, keepout),
                    'LAYOUT_COLLISION', 'floor/corridor ' + item['role_id'])
    for index, item in enumerate(items):
        for other in items[:index]:
            if item.get('support_role_id') == other['role_id'] or other.get('support_role_id') == item['role_id']:
                continue
            az = item['pose_xyz_m'][2]; bz = other['pose_xyz_m'][2]
            if min(az+item['dimensions_m'][2],bz+other['dimensions_m'][2])-max(az,bz)>1e-6:
                require(not overlaps(footprint(item),footprint(other)), 'LAYOUT_COLLISION',
                        item['role_id']+':'+other['role_id'])
    goal = core['goal_placement']; target = by_id[goal['target_role_id']]
    desired = {**target, 'pose_xyz_m': goal['candidate_pose_xyz_m']}
    support = by_id[goal['destination_support_role_id']]['support_surface']
    require(contains_polygon(Polygon.from_sequence(support['polygon_xy']),footprint(desired)) and
            abs(desired['pose_xyz_m'][2]-support['plane_z_m']-.002)<1e-5,
            'LAYOUT_COLLISION','goal support')
    return {'status':'passed','instances':len(items),'method':'oriented rectangular proxies',
            'continuous_motion':'not_run'}


def transformed(core, background, target, operation, dx=0, dy=0, yaw_delta=0, support_id=None):
    result = deepcopy(core); bg = deepcopy(background)
    items = result['instances'] + bg.get('instances', [])
    old = {i['role_id']:deepcopy(i) for i in items}
    chosen = old[target]; child_ids = dependents(items,target) if operation=='move_component' else []
    moving = {target,*child_ids}
    ox, oy, _ = chosen['pose_xyz_m']; c,s=math.cos(yaw_delta),math.sin(yaw_delta)
    for item in items:
        if item['role_id'] not in moving:continue
        x,y,z=item['pose_xyz_m'];item['pose_xyz_m']=[ox+dx+c*(x-ox)-s*(y-oy),oy+dy+s*(x-ox)+c*(y-oy),z]
        item['yaw_rad']=item.get('yaw_rad',0)+yaw_delta
        if operation=='change_support':
            destination=old[support_id];item['support_role_id']=support_id
            item['pose_xyz_m']=[destination['pose_xyz_m'][0]+dx,destination['pose_xyz_m'][1]+dy,
                                destination['support_surface']['plane_z_m']+.002]
        if operation=='move_component' and item.get('support_surface'):
            surface=item['support_surface']
            surface['polygon_xy']=[[ox+dx+c*(x-ox)-s*(y-oy),oy+dy+s*(x-ox)+c*(y-oy)]
                                   for x,y in surface['polygon_xy']]
    if operation=='move_component' and result['goal_placement']['destination_support_role_id'] in moving:
        x,y,z=result['goal_placement']['candidate_pose_xyz_m']
        result['goal_placement']['candidate_pose_xyz_m']=[ox+dx+c*(x-ox)-s*(y-oy),oy+dy+s*(x-ox)+c*(y-oy),z]
    return result,bg


def propose(graph, core, background, plan, grant, operation, *, source_digest, limit=8):
    target = validate_permission(graph,grant,operation)
    require(type(limit) is int and 1<=limit<=32,'LAYOUT_INPUT_INVALID','candidate limit')
    all_items=core['instances']+background.get('instances',[])
    require(any(i['role_id']==target for i in all_items),'LAYOUT_INPUT_INVALID','target missing')
    children=dependents(all_items,target)
    require(not children or operation=='move_component','LAYOUT_DEPENDENTS_UNSUPPORTED',','.join(children))
    candidates=[];rejected=[]
    if operation=='rotate':options=[(0,0,yaw,None) for yaw in (math.pi/2,-math.pi/2,math.pi)]
    elif operation=='change_support':
        options=[(x,y,0,i['role_id']) for i in all_items if i.get('support_surface')
                 and i['room_id']==next(j['room_id'] for j in all_items if j['role_id']==target)
                 for x,y in ((0,0),(.15,0),(-.15,0),(0,.15))]
    else:options=[(x,y,0,None) for distance in (.1,.2,.3,.4,.6)
                  for x,y in ((-distance,0),(distance,0),(0,-distance),(0,distance),
                              (-distance,-distance),(-distance,distance),(distance,-distance),(distance,distance))]
    for dx,dy,yaw,support in options:
        revised,bg=transformed(core,background,target,operation,dx,dy,yaw,support)
        combined_before={**core,'instances':all_items}
        combined_after={**revised,'instances':revised['instances']+bg.get('instances',[])}
        try:
            verify_pose_difference(graph,combined_before,combined_after,grant,operation,component=operation=='move_component')
            measured=measure_layout(revised,bg,plan)
        except ValueError as error:
            rejected.append({'offset_m':[dx,dy],'yaw_delta_rad':yaw,'support_id':support,
                             'failure_code':getattr(error,'code','LAYOUT_INPUT_INVALID')})
            continue
        value={'graph_hash':graph.content_hash,'source_digest':source_digest,'permission_hash':grant['content_hash'],
               'target_id':target,'operation':operation,'core':revised,'background':bg,'measurement':measured}
        value['candidate_id']=content_hash(value);candidates.append(value)
        if len(candidates)>=limit:break
    report={'schema_version':'pvscene.layout-candidates/v1','graph_hash':graph.content_hash,
            'source_digest':source_digest,'permission':grant,'operation':operation,
            'candidates':candidates,'rejected':rejected}
    report['content_hash']=content_hash(report)
    return report
