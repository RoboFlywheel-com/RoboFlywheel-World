# SPDX-License-Identifier: Apache-2.0
"""Read-only critique results distinguish evaluation completion from measured quality."""
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError

SCHEMA='pvscene.scene-critique/v1'
PROFILES=('static','visual','robot_layout')
OPERATIONS=('translate','rotate','move_component','revise_robot_installation','inspect')


def require(condition,code,detail=''):
    if not condition:raise GraphError(code,detail)


def validate_request(graph,scope_ids,profiles):
    ids={n['id'] for n in graph.to_dict()['nodes']}
    require(isinstance(scope_ids,(list,tuple)) and bool(scope_ids) and
            all(isinstance(s,str) and s in ids for s in scope_ids) and len(set(scope_ids))==len(scope_ids),
            'CRITIC_SCOPE_VIOLATION','scope ids')
    require(isinstance(profiles,(list,tuple)) and bool(profiles) and
            all(isinstance(p,str) and p in PROFILES for p in profiles) and len(set(profiles))==len(profiles),
            'CRITIC_PROFILE_UNSUPPORTED','profiles')


def validate_visual_response(graph,value,scope_ids):
    require(isinstance(value,dict) and set(value)=={'graph_hash','quality_pass','issues'} and
            value['graph_hash']==graph.content_hash and type(value['quality_pass']) is bool and
            isinstance(value['issues'],list),'CRITIC_RESPONSE_INVALID','response/version')
    nodes={n['id']:n for n in graph.to_dict()['nodes']}
    result=[]
    for item in value['issues']:
        require(isinstance(item,dict) and set(item)=={'scope_ids','severity','code','detail','suggested_operation'}
                and isinstance(item['scope_ids'],list) and item['scope_ids'] and
                all(isinstance(s,str) and s in nodes and
                    (s in scope_ids or nodes[s].get('room_id') in scope_ids) for s in item['scope_ids']) and
                item['severity'] in ('warning','error') and item['suggested_operation'] in OPERATIONS and
                item['code']=='VISUAL_QUALITY' and isinstance(item['detail'],str) and 0<len(item['detail'])<=1000,
                'CRITIC_RESPONSE_INVALID','issue; no model physics claims')
        result.append(issue('visual',item['scope_ids'],item['code'],item['detail'],
                            severity=item['severity'],operation=item['suggested_operation']))
    require(not value['quality_pass'] or not any(i['severity']=='error' for i in result),
            'CRITIC_RESPONSE_INVALID','quality contradicts error')
    return result


def issue(source,scope_ids,code,detail,*,severity='error',operation='inspect',evidence_refs=()):
    value={'source':source,'scope_ids':list(scope_ids),'severity':severity,'code':code,
           'detail':detail,'evidence_refs':list(evidence_refs),'suggested_operation':operation,
           'constraints':['preserve_task_identity','explicit_pose_permission','current_version_reverification']}
    value['id']=content_hash(value)
    return value
