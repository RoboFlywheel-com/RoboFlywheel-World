# SPDX-License-Identifier: Apache-2.0
"""Independent scope ruler for adding one optional surface manipuland."""
from rfw_w.scenes.contracts.surface_appearance import require


def verify_optional_extension(old, new, role_id, surface_id):
    for key in ('graph_id','composition_schema','initial_state','steps','goals','requirements'):
        require(old.get(key)==new.get(key),'ROLLBACK_CORE_VIOLATION',key)
    old_nodes={n['id']:n for n in old['nodes']};new_nodes={n['id']:n for n in new['nodes']}
    require(role_id not in old_nodes and set(new_nodes)-set(old_nodes)=={role_id},
            'COMPOSITION_ID_CONFLICT',role_id)
    require(surface_id in old_nodes and old_nodes[surface_id]['kind']=='surface',
            'COMPOSITION_ANCHOR_INVALID',surface_id)
    parent={e['target_id']:e['source_id'] for e in old['edges'] if e['kind']=='contains'}
    policies=[n for n in old_nodes.values() if n['kind']=='distractor_policy' and parent.get(n['id'])==surface_id]
    require(len(policies)==1,'COMPOSITION_ANCHOR_INVALID','surface policy')
    policy=policies[0];rid=new_nodes[role_id]
    require(rid.get('category') in {'bowl','jar'} and rid.get('physical_class')=='manipuland'
            and rid.get('placement_kind')=='surface' and rid.get('priority')=='optional'
            and rid.get('purpose')=='distraction' and rid.get('task_role')=='distractor'
            and rid.get('room_id')==old_nodes[surface_id]['room_id'],
            'RIGID_CATEGORY_UNSUPPORTED',role_id)
    for nid,node in old_nodes.items():
        if nid==policy['id']:
            changed=new_nodes[nid]
            require({k:v for k,v in node.items() if k not in {'candidate_categories','max_count'}}
                    =={k:v for k,v in changed.items() if k not in {'candidate_categories','max_count'}},
                    'ROLLBACK_CORE_VIOLATION','policy identity')
            require(changed['candidate_categories']==list(dict.fromkeys(node['candidate_categories']+[rid['category']]))
                    and changed['max_count']==node['max_count']+1 and changed['max_count']<=4,
                    'FURNISHING_BUDGET_EXCEEDED','policy')
        else:require(node==new_nodes.get(nid),'ROLLBACK_CORE_VIOLATION',nid)
    added=[e for e in new['edges'] if e not in old['edges']]
    require(len(added)==1 and added[0]['kind']=='contains' and added[0]['source_id']==surface_id
            and added[0]['target_id']==role_id and all(e in new['edges'] for e in old['edges']),
            'ROLLBACK_CORE_VIOLATION','edges')


def verify_optional_ledger(old, new, role_id):
    for key in ('house_plan_hash','goal_placement','unplaced_optional'):
        require(old[key]==new[key],'ROLLBACK_CORE_VIOLATION',key)
    require(old['instances']==[i for i in new['instances'] if i['role_id']!=role_id],
            'ROLLBACK_CORE_VIOLATION','preserved instances')
    for key in ('queries','selection'):
        require(old[key]=={k:v for k,v in new[key].items() if k!=role_id},
                'ROLLBACK_CORE_VIOLATION',key)
    require(len([i for i in new['instances'] if i['role_id']==role_id])==1,
            'ASSET_LEDGER_MISMATCH','new instance')
