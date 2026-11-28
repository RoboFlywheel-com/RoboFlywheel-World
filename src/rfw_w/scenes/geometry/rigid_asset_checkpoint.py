# SPDX-License-Identifier: Apache-2.0
"""Independent immutable artifact and single binding scope checks for restoration."""
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.surface_appearance import require


def verify_checkpoint_payload(checkpoint, saved_graph, saved_artifacts):
    require(checkpoint.get('schema_version') == 'pvscene.rigid-checkpoint/v1',
            'CHECKPOINT_TAMPERED', 'schema')
    require(content_hash({k: v for k, v in checkpoint.items() if k != 'content_hash'})
            == checkpoint.get('content_hash'), 'CHECKPOINT_TAMPERED', 'checkpoint digest')
    require(checkpoint.get('graph_hash') == saved_graph.content_hash,
            'CHECKPOINT_TAMPERED', 'saved graph')
    require(checkpoint.get('artifacts') == saved_artifacts,
            'CHECKPOINT_TAMPERED', 'saved artifact digest')


def verify_binding_scope(current_graph, saved_graph, current_core, saved_core, role_id):
    """No other task semantics or core binding may be restored by a target rollback."""
    for key in ('graph_id', 'composition_schema', 'initial_state', 'steps', 'goals', 'requirements', 'edges'):
        require(current_graph.get(key) == saved_graph.get(key), 'ROLLBACK_CORE_VIOLATION', key)
    def nodes(graph):
        return [{k: v for k, v in n.items() if not (n['id'] == 'inst_' + role_id
                 and n['kind'] == 'asset_instance' and k in ('asset_ref', 'source_reason'))}
                for n in graph['nodes']]
    require(nodes(current_graph) == nodes(saved_graph), 'ROLLBACK_CORE_VIOLATION', 'nodes')
    require(current_graph['built']['house_shell'] == saved_graph['built']['house_shell'],
            'ROLLBACK_CORE_VIOLATION', 'house')
    for key in ('source_graph_hash', 'house_plan_hash', 'unplaced_optional'):
        require(current_core[key] == saved_core[key], 'ROLLBACK_CORE_VIOLATION', key)
    for key in ('instances',):
        require([i for i in current_core[key] if i['role_id'] != role_id]
                == [i for i in saved_core[key] if i['role_id'] != role_id],
                'ROLLBACK_CORE_VIOLATION', 'other instances')
    for key in ('queries', 'selection'):
        require({k: v for k, v in current_core[key].items() if k != role_id}
                == {k: v for k, v in saved_core[key].items() if k != role_id},
                'ROLLBACK_CORE_VIOLATION', 'other ' + key)
    require(current_core['goal_placement']['target_role_id'] == role_id
            == saved_core['goal_placement']['target_role_id'],
            'ROLLBACK_SCOPE_INVALID', 'target role only')

    require({k:v for k,v in current_core['goal_placement'].items() if k!='candidate_pose_xyz_m'}
            == {k:v for k,v in saved_core['goal_placement'].items() if k!='candidate_pose_xyz_m'},
            'ROLLBACK_CORE_VIOLATION', 'goal semantics')
