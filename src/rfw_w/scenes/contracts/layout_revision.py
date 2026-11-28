# SPDX-License-Identifier: Apache-2.0
"""Explicit, version-bound pose permissions; task and asset identity remain frozen."""
from copy import deepcopy
import math
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError

SCHEMA = 'pvscene.layout-permission/v1'
OPERATIONS = ('translate', 'rotate', 'change_support', 'move_component')
INVALIDATES = ('static_occupation', 'robot_installation', 'target_ik', 'approach_path', 'render', 'critic')


def require(condition, code, detail=''):
    if not condition:
        raise GraphError(code, detail)


def permission(graph, target_id, *, operations, scope_ids, locked_fields=()):
    """Caller explicitly grants inferred-pose edits; user semantic roles grant no rights."""
    raw = graph.to_dict()
    nodes = {n['id']: n for n in raw['nodes']}
    require(target_id in nodes and nodes[target_id]['kind'] == 'asset_role', 'LAYOUT_INPUT_INVALID', 'target role')
    require(isinstance(operations, (list, tuple)) and operations and
            all(isinstance(o, str) and o in OPERATIONS for o in operations) and len(set(operations)) == len(operations),
            'LAYOUT_INPUT_INVALID', 'operations')
    require(isinstance(scope_ids, (list, tuple)) and scope_ids and
            all(isinstance(s, str) and s in nodes for s in scope_ids) and len(set(scope_ids)) == len(scope_ids),
            'LAYOUT_SCOPE_VIOLATION', 'scope')
    require(nodes[target_id]['room_id'] in scope_ids, 'LAYOUT_SCOPE_VIOLATION', 'target room')
    require(isinstance(locked_fields, (list, tuple)) and
            all(v in ('pose_xyz_m', 'yaw_rad', 'support_role_id') for v in locked_fields),
            'LAYOUT_INPUT_INVALID', 'locked fields')
    value = {'schema_version': SCHEMA, 'graph_hash': graph.content_hash, 'target_id': target_id,
             'operations': list(operations), 'scope_ids': list(scope_ids), 'locked_fields': list(locked_fields),
             'pose_source': 'inferred', 'semantic_identity_locked': True}
    value['content_hash'] = content_hash(value)
    return value


def validate_permission(graph, value, operation):
    require(isinstance(value, dict), 'LAYOUT_LOCKED', 'explicit permission missing')
    expected = permission(graph, value.get('target_id'), operations=value.get('operations'),
                          scope_ids=value.get('scope_ids'), locked_fields=value.get('locked_fields'))
    require(value == expected, 'LAYOUT_CANDIDATE_STALE', 'permission identity/version')
    require(operation in value['operations'], 'LAYOUT_LOCKED', operation)
    return value['target_id']


def dependents(instances, target_id):
    """Complete support dependency closure, including optional children without graph edges."""
    result = {target_id}
    while True:
        expanded = result | {i['role_id'] for i in instances if i.get('support_role_id') in result}
        if expanded == result:
            return sorted(result - {target_id})
        result = expanded


def verify_pose_difference(graph, before, after, grant, operation, *, component=False):
    target = validate_permission(graph, grant, operation)
    old = {i['role_id']: i for i in before['instances']}
    new = {i['role_id']: i for i in after['instances']}
    require(len(old) == len(before['instances']) and len(new) == len(after['instances']) and set(old) == set(new),
            'LAYOUT_SCOPE_VIOLATION', 'instance identity')
    require(target in old, 'LAYOUT_INPUT_INVALID', 'unbuilt target')
    children = dependents(before['instances'], target)
    require(not children or component, 'LAYOUT_DEPENDENTS_UNSUPPORTED', ','.join(children))
    require(not component or operation == 'move_component', 'LAYOUT_INPUT_INVALID', 'component operation')
    changed_ids = {target, *children} if component else {target}
    for rid in changed_ids:
        require(old[rid]['room_id'] in grant['scope_ids'], 'LAYOUT_SCOPE_VIOLATION', 'component room')
        for key in grant['locked_fields']:
            require(old[rid].get(key) == new[rid].get(key), 'LAYOUT_LOCKED', key)
        pose = new[rid].get('pose_xyz_m')
        yaw = new[rid].get('yaw_rad')
        require(isinstance(pose, list) and len(pose) == 3 and
                all(type(v) in (int, float) and math.isfinite(v) for v in pose) and
                type(yaw) in (int, float) and math.isfinite(yaw), 'LAYOUT_INPUT_INVALID', 'finite pose')
    allowed = {'pose_xyz_m', 'yaw_rad'}
    if operation == 'translate':
        allowed = {'pose_xyz_m'}
    if operation == 'rotate':
        allowed = {'yaw_rad'}
    if operation == 'change_support':
        allowed.add('support_role_id')
    if component:
        allowed.add('support_surface')
    for rid in old:
        fields = allowed if rid in changed_ids else set()
        require({k:v for k,v in old[rid].items() if k not in fields} ==
                {k:v for k,v in new[rid].items() if k not in fields},
                'LAYOUT_SCOPE_VIOLATION', 'asset/other instance ' + rid)
    if component:
        dx=new[target]['pose_xyz_m'][0]-old[target]['pose_xyz_m'][0]
        dy=new[target]['pose_xyz_m'][1]-old[target]['pose_xyz_m'][1]
        dz=new[target]['pose_xyz_m'][2]-old[target]['pose_xyz_m'][2]
        theta=new[target]['yaw_rad']-old[target]['yaw_rad'];c,s=math.cos(theta),math.sin(theta)
        ox,oy,_=old[target]['pose_xyz_m']
        for rid in changed_ids:
            x,y,z=old[rid]['pose_xyz_m']
            wanted=[ox+dx+c*(x-ox)-s*(y-oy),oy+dy+s*(x-ox)+c*(y-oy),z+dz]
            require(all(abs(a-b)<1e-8 for a,b in zip(wanted,new[rid]['pose_xyz_m'])) and
                    abs(new[rid]['yaw_rad']-old[rid]['yaw_rad']-theta)<1e-8,
                    'LAYOUT_RELATION_VIOLATION','rigid component '+rid)
    a, b = deepcopy(before), deepcopy(after)
    for value in (a, b):
        value.pop('content_hash', None)
        value.pop('instances', None)
    if component and before['goal_placement']['destination_support_role_id'] in changed_ids:
        a['goal_placement'].pop('candidate_pose_xyz_m', None)
        b['goal_placement'].pop('candidate_pose_xyz_m', None)
    require(a == b, 'LAYOUT_SCOPE_VIOLATION', 'ledger semantics/selection')
    for predicate in graph.to_dict()['initial_state']:
        if predicate['subject'] in changed_ids and predicate['predicate'] == 'ON':
            require(new[predicate['subject']]['support_role_id'] == predicate['object'],
                    'LAYOUT_RELATION_VIOLATION', 'initial ON')
    for edge in graph.to_dict().get('edges', []):
        if edge['kind'] == 'supports' and edge['target_id'] in changed_ids:
            require(new[edge['target_id']]['support_role_id'] == edge['source_id'],
                    'LAYOUT_RELATION_VIOLATION', 'supports edge')
    require(any(old[rid] != new[rid] for rid in changed_ids), 'LAYOUT_NO_PROGRESS', 'unchanged pose')
    return {'changed_role_ids': sorted(changed_ids), 'invalidates': list(INVALIDATES)}
