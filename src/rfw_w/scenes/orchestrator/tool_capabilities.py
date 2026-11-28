# SPDX-License-Identifier: Apache-2.0
"""Stage-qualified capability metadata; state machines remain the authority for legality.

Invalidation metadata documents obligations; executors still perform and record the
actual invalidation and revalidation. Failure-code lists are representative, not an
exhaustive replacement for the original tool contracts.
"""
from copy import deepcopy
import re
from rfw_w.scenes.contracts.task_world_graph import GraphError

DOMAINS = ('scene', 'assets', 'layout', 'appearance', 'robot', 'review', 'recovery', 'delivery')
STAGES = ('scene', 'robot_layout', 'critic_repair')
FIELDS = {'stage', 'name', 'version', 'writes', 'scope', 'requires', 'effect',
          'domain', 'operation', 'scope_kind', 'reads', 'failure_codes',
          'invalidates', 'reverify', 'cost_class', 'parameter_schema'}

# Existing executable capabilities only. Future layout/critic operations register
# when their executors and contracts are implemented, not when their plans exist.
_RECORDS = [
    {'stage': 'scene',
     'name': 'add_surface_rigid_asset',
     'version': 'v1',
     'writes': True,
     'scope': 'optional surface role',
     'requires': 'built backgrounds',
     'effect': 'first admitted SimReady candidate placed on measured support',
     'domain': 'assets',
     'operation': 'add_surface',
     'scope_kind': 'optional_surface_role',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['ASSET_UNAVAILABLE', 'COMPOSITION_ID_CONFLICT'],
     'invalidates': ['static', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'retrieval',
     'parameter_schema': {'surface_id': 'optional_surface_ids entry',
                          'role_id': 'unique optional_ identifier',
                          'category': 'bowl or jar, once each',
                          'description': 'task semantic description'}},
    {'stage': 'scene',
     'name': 'save_rigid_checkpoint',
     'version': 'v1',
     'writes': False,
     'scope': 'target',
     'requires': 'verified core',
     'effect': 'immutable target checkpoint',
     'domain': 'recovery',
     'operation': 'checkpoint_target',
     'scope_kind': 'target_role',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['CHECKPOINT_TAMPERED'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'local',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'restore_rigid_asset',
     'version': 'v1',
     'writes': True,
     'scope': 'target',
     'requires': 'target checkpoint and changed binding',
     'effect': 'remeasured target restoration',
     'domain': 'recovery',
     'operation': 'restore_target',
     'scope_kind': 'target_role',
     'reads': ['scene_graph', 'assets', 'checkpoint'],
     'failure_codes': ['ROLLBACK_CORE_VIOLATION', 'CHECKPOINT_TAMPERED'],
     'invalidates': ['static', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'geometry',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'search_rigid_assets',
     'version': 'v1',
     'writes': False,
     'scope': 'target role',
     'requires': 'verified core',
     'effect': 'bounded SimReady candidates',
     'domain': 'assets',
     'operation': 'search',
     'scope_kind': 'target_role',
     'reads': ['scene_graph', 'asset_index'],
     'failure_codes': ['ASSET_UNAVAILABLE', 'RIGID_CATEGORY_UNSUPPORTED'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'retrieval',
     'parameter_schema': {'role_id': 'rigid_role_id', 'description': 'task object description'}},
    {'stage': 'scene',
     'name': 'select_rigid_asset',
     'version': 'v1',
     'writes': True,
     'scope': 'target role',
     'requires': 'rigid query',
     'effect': 'mixed provider target assembly',
     'domain': 'assets',
     'operation': 'select',
     'scope_kind': 'target_role',
     'reads': ['scene_graph', 'assets', 'rigid_query'],
     'failure_codes': ['RIGID_QUERY_STALE', 'RIGID_CANDIDATE_NOT_ADMITTED'],
     'invalidates': ['static', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'geometry',
     'parameter_schema': {'candidate_key': 'rigid_candidates candidate_key'}},
    {'stage': 'scene',
     'name': 'create_surface_covering',
     'version': 'v1',
     'writes': True,
     'scope': 'measured surface',
     'requires': 'room appearance',
     'effect': 'verified thin covering',
     'domain': 'appearance',
     'operation': 'create_covering',
     'scope_kind': 'surface',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['COVERING_CORE_FORBIDDEN', 'COVERING_SURFACE_MISSING', 'COVERING_MATERIAL_INVALID'],
     'invalidates': ['static', 'coverings', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'local',
     'parameter_schema': {'instance_id': 'unique cover_ token',
                          'surface_id': 'covering_surfaces surface_id',
                          'material_id': 'covering_material_ids entry',
                          'local_position_m': 'two numbers, face-local meters',
                          'size_m': 'two positive numbers'}},
    {'stage': 'scene',
     'name': 'move_surface_covering',
     'version': 'v1',
     'writes': True,
     'scope': 'covering instance',
     'requires': 'covering',
     'effect': 'new checked position',
     'domain': 'appearance',
     'operation': 'move_covering',
     'scope_kind': 'covering_instance',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['COVERING_CORE_FORBIDDEN', 'COVERING_SURFACE_MISSING', 'COVERING_MATERIAL_INVALID'],
     'invalidates': ['static', 'coverings', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'local',
     'parameter_schema': {'instance_id': 'covering_instances entry', 'local_position_m': 'two numbers'}},
    {'stage': 'scene',
     'name': 'remove_surface_covering',
     'version': 'v1',
     'writes': True,
     'scope': 'covering instance',
     'requires': 'covering',
     'effect': 'remove optional covering',
     'domain': 'appearance',
     'operation': 'remove_covering',
     'scope_kind': 'covering_instance',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['COVERING_CORE_FORBIDDEN', 'COVERING_SURFACE_MISSING', 'COVERING_MATERIAL_INVALID'],
     'invalidates': ['static', 'coverings', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'local',
     'parameter_schema': {'instance_id': 'covering_instances entry'}},
    {'stage': 'scene',
     'name': 'search_surface_materials',
     'version': 'v1',
     'writes': False,
     'scope': 'room surface',
     'requires': 'built backgrounds',
     'effect': 'bounded material candidates',
     'domain': 'appearance',
     'operation': 'search',
     'scope_kind': 'room_surface',
     'reads': ['material_index'],
     'failure_codes': ['MATERIAL_ENV_MISSING', 'MATERIAL_NOT_FOUND'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'local',
     'parameter_schema': {'description': 'string task style, e.g. wood oak or plaster wall',
                          'surface_kind': 'floor or wall_set, exactly material_query_needed'}},
    {'stage': 'scene',
     'name': 'set_room_surface_materials',
     'version': 'v1',
     'writes': True,
     'scope': 'room surfaces',
     'requires': 'material queries',
     'effect': 'prepared appearance binding',
     'domain': 'appearance',
     'operation': 'set',
     'scope_kind': 'room_surface',
     'reads': ['scene_graph', 'assets', 'material_query'],
     'failure_codes': ['MATERIAL_CHANNEL_MISSING', 'SURFACE_BINDING_INVALID'],
     'invalidates': ['appearance', 'render', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'local',
     'parameter_schema': {'floor_material_id': 'material_queries.floor.candidates material_id',
                          'wall_material_id': 'material_queries.wall_set.candidates material_id',
                          'meters_per_tile': 'number 0.01 to 100'}},
    {'stage': 'scene',
     'name': 'revise_composition',
     'version': 'v1',
     'writes': True,
     'scope': 'rooms',
     'requires': 'v2 intent, no built artifacts',
     'effect': 'new semantic graph',
     'domain': 'scene',
     'operation': 'configure',
     'scope_kind': 'room',
     'reads': ['scene_graph'],
     'failure_codes': ['COMPOSITION_PATCH_INVALID', 'PATCH_SCOPE_VIOLATION', 'GRAPH_VERSION_CONFLICT'],
     'invalidates': ['built_geometry', 'static', 'render', 'critic'],
     'reverify': ['build_house'],
     'cost_class': 'local',
     'parameter_schema': {'patch_index': 'integer 0 or 1'}},
    {'stage': 'scene',
     'name': 'build_house',
     'version': 'v1',
     'writes': True,
     'scope': 'house',
     'requires': 'configured intent',
     'effect': 'measured house shell',
     'domain': 'scene',
     'operation': 'build',
     'scope_kind': 'house',
     'reads': ['scene_graph'],
     'failure_codes': ['HOUSE_TOPOLOGY_UNSUPPORTED', 'PORTAL_GEOMETRY_INVALID', 'HOUSE_SHELL_MISMATCH'],
     'invalidates': ['asset_assembly', 'static', 'render', 'robot_layout', 'critic'],
     'reverify': ['assemble_core'],
     'cost_class': 'geometry',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'assemble_core',
     'version': 'v1',
     'writes': True,
     'scope': 'task_core',
     'requires': 'measured house shell',
     'effect': 'verified core assembly',
     'domain': 'scene',
     'operation': 'assemble',
     'scope_kind': 'task_core',
     'reads': ['scene_graph', 'house_shell'],
     'failure_codes': ['ASSET_UNAVAILABLE', 'CORE_KEEP_OUT_VIOLATION'],
     'invalidates': ['static', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'retrieval',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'save_checkpoint',
     'version': 'v1',
     'writes': False,
     'scope': 'graph',
     'requires': 'verified core',
     'effect': 'immutable checkpoint',
     'domain': 'recovery',
     'operation': 'checkpoint',
     'scope_kind': 'scene',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['CHECKPOINT_TAMPERED'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'local',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'furnish_floor',
     'version': 'v1',
     'writes': True,
     'scope': 'floor roles',
     'requires': 'checkpoint and pending floor',
     'effect': 'measured floor background',
     'domain': 'layout',
     'operation': 'furnish_floor',
     'scope_kind': 'floor_roles',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['CORE_KEEP_OUT_VIOLATION', 'FURNISHING_BUDGET_EXCEEDED', 'ASSET_UNAVAILABLE'],
     'invalidates': ['static', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'retrieval',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'mount_wall',
     'version': 'v1',
     'writes': True,
     'scope': 'wall roles',
     'requires': 'checkpoint and pending wall',
     'effect': 'measured wall background',
     'domain': 'layout',
     'operation': 'mount_wall',
     'scope_kind': 'wall_roles',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['WALL_MOUNT_UNCERTIFIED', 'ASSET_UNAVAILABLE'],
     'invalidates': ['static', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'retrieval',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'verify_scene',
     'version': 'v1',
     'writes': False,
     'scope': 'scene',
     'requires': 'built backgrounds',
     'effect': 'current static verification',
     'domain': 'review',
     'operation': 'verify_static',
     'scope_kind': 'scene',
     'reads': ['scene_graph', 'assets'],
     'failure_codes': ['CORE_KEEP_OUT_VIOLATION', 'ASSET_LEDGER_MISMATCH', 'ASSET_BINDING_INVALID'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'geometry',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'rollback_scope',
     'version': 'v1',
     'writes': True,
     'scope': 'failed optional roles',
     'requires': 'failure and checkpoint',
     'effect': 'new revision, invalidated background evidence',
     'domain': 'recovery',
     'operation': 'restore_scope',
     'scope_kind': 'optional_roles',
     'reads': ['scene_graph', 'assets', 'checkpoint'],
     'failure_codes': ['ROLLBACK_SCOPE_INVALID', 'CHECKPOINT_TAMPERED'],
     'invalidates': ['static', 'render', 'robot_layout', 'critic'],
     'reverify': ['verify_scene'],
     'cost_class': 'geometry',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'finish',
     'version': 'v1',
     'writes': False,
     'scope': 'scene',
     'requires': 'current verification and all declared roles accounted',
     'effect': 'terminal success',
     'domain': 'delivery',
     'operation': 'finish',
     'scope_kind': 'scene',
     'reads': ['scene_graph', 'evidence'],
     'failure_codes': ['PRECONDITION_UNMET'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'local',
     'parameter_schema': {}},
    {'stage': 'scene',
     'name': 'terminate',
     'version': 'v1',
     'writes': False,
     'scope': 'scene',
     'requires': 'named failure',
     'effect': 'terminal named failure',
     'domain': 'delivery',
     'operation': 'terminate',
     'scope_kind': 'scene',
     'reads': ['scene_graph', 'failure'],
     'failure_codes': [],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'local',
     'parameter_schema': {}},
    {'stage': 'robot_layout',
     'name': 'install_robot',
     'version': 'v1',
     'writes': True,
     'scope': 'fixed robot mount',
     'effect': 'measured mount',
     'requires': 'no current installation',
     'domain': 'robot',
     'operation': 'install',
     'scope_kind': 'robot_mount',
     'reads': ['scene_graph', 'assets', 'robot_model'],
     'failure_codes': ['INSTALLATION_UNAVAILABLE', 'ROBOT_LAYOUT_INPUT_INVALID', 'ROBOT_TOOL_TIMEOUT'],
     'invalidates': ['target_ik', 'approach_samples', 'robot_render', 'critic'],
     'reverify': ['check_target_reachability', 'check_approach_path'],
     'cost_class': 'simulation',
     'parameter_schema': {}},
    {'stage': 'robot_layout',
     'name': 'check_target_reachability',
     'version': 'v1',
     'writes': False,
     'scope': 'target',
     'effect': 'real target IK',
     'requires': 'current installation, target evidence absent',
     'domain': 'robot',
     'operation': 'check_target',
     'scope_kind': 'target_role',
     'reads': ['robot_installation', 'assets'],
     'failure_codes': ['TARGET_IK_UNREACHABLE', 'TARGET_IK_COLLISION', 'ROBOT_LAYOUT_INPUT_INVALID'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'simulation',
     'parameter_schema': {}},
    {'stage': 'robot_layout',
     'name': 'check_approach_path',
     'version': 'v1',
     'writes': False,
     'scope': 'approach',
     'effect': 'sampled collision evidence',
     'requires': 'current target IK, approach evidence absent',
     'domain': 'robot',
     'operation': 'check_approach',
     'scope_kind': 'approach',
     'reads': ['robot_installation', 'target_ik'],
     'failure_codes': ['APPROACH_PATH_COLLISION', 'ROBOT_LAYOUT_INPUT_INVALID'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'simulation',
     'parameter_schema': {}},
    {'stage': 'robot_layout',
     'name': 'revise_robot_installation',
     'version': 'v1',
     'writes': True,
     'scope': 'fixed robot mount',
     'effect': 'new mount, invalidate IK/path',
     'requires': 'repairable failure and revision budget remaining',
     'domain': 'robot',
     'operation': 'revise_installation',
     'scope_kind': 'robot_mount',
     'reads': ['scene_graph', 'assets', 'robot_model'],
     'failure_codes': ['INSTALLATION_UNAVAILABLE', 'ROBOT_LAYOUT_INPUT_INVALID', 'ROBOT_TOOL_TIMEOUT'],
     'invalidates': ['target_ik', 'approach_samples', 'robot_render', 'critic'],
     'reverify': ['check_target_reachability', 'check_approach_path'],
     'cost_class': 'simulation',
     'parameter_schema': {}},
    {'stage': 'robot_layout',
     'name': 'finish',
     'version': 'v1',
     'writes': False,
     'scope': 'robot layout',
     'effect': 'layout checks complete',
     'requires': 'current installation, target IK and approach evidence',
     'domain': 'delivery',
     'operation': 'finish',
     'scope_kind': 'robot_layout',
     'reads': ['scene_graph', 'evidence'],
     'failure_codes': ['PRECONDITION_UNMET'],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'local',
     'parameter_schema': {}},
    {'stage': 'robot_layout',
     'name': 'terminate',
     'version': 'v1',
     'writes': False,
     'scope': 'robot layout',
     'effect': 'named failure',
     'requires': 'named failure',
     'domain': 'delivery',
     'operation': 'terminate',
     'scope_kind': 'robot_layout',
     'reads': ['scene_graph', 'failure'],
     'failure_codes': [],
     'invalidates': [],
     'reverify': [],
     'cost_class': 'local',
     'parameter_schema': {}},
]


def validate_catalog(records):
    if not isinstance(records, (list, tuple)) or not records:
        raise GraphError('TOOL_CAPABILITY_INVALID', 'empty catalog')
    seen = set()
    for item in records:
        if not isinstance(item, dict) or set(item) != FIELDS:
            raise GraphError('TOOL_CAPABILITY_INVALID', 'fields')
        if not isinstance(item['stage'], str) or not isinstance(item['name'], str) or not isinstance(item['domain'], str):
            raise GraphError('TOOL_CAPABILITY_INVALID', 'identity type')
        identity = (item['stage'], item['name'])
        if item['stage'] not in STAGES or item['domain'] not in DOMAINS:
            raise GraphError('TOOL_CAPABILITY_INVALID', 'stage or domain')
        if identity in seen:
            raise GraphError('TOOL_CAPABILITY_INVALID', 'duplicate stage/name')
        seen.add(identity)
        if type(item['writes']) is not bool:
            raise GraphError('TOOL_CAPABILITY_INVALID', 'writes')
        for key in ('name', 'version', 'scope', 'requires', 'effect', 'operation', 'scope_kind'):
            if not isinstance(item[key], str) or not item[key]:
                raise GraphError('TOOL_CAPABILITY_INVALID', key)
        if item['cost_class'] not in ('local', 'geometry', 'retrieval', 'simulation'):
            raise GraphError('TOOL_CAPABILITY_INVALID', 'cost class')
        for key in ('reads', 'failure_codes', 'invalidates', 'reverify'):
            values = item[key]
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values) or len(values) != len(set(values)):
                raise GraphError('TOOL_CAPABILITY_INVALID', key)
        if not item['reads'] or (not item['writes'] and item['invalidates']):
            raise GraphError('TOOL_CAPABILITY_INVALID', 'read-only invalidation')
        if any(not re.fullmatch(r'[A-Z][A-Z0-9_]*', v) for v in item['failure_codes']):
            raise GraphError('TOOL_CAPABILITY_INVALID', 'failure code')
        schema = item['parameter_schema']
        if not isinstance(schema, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in schema.items()):
            raise GraphError('TOOL_CAPABILITY_INVALID', 'parameter schema')
    for item in records:
        if any((item['stage'], name) not in seen for name in item['reverify']):
            raise GraphError('TOOL_CAPABILITY_INVALID', 'unknown reverify tool')
    return True


# Registered only after their local executors and independent checks exist.
for name, domain, operation, writes, schema in (
    ('critique_scene','review','critique',False,{'scope_ids':'known IDs within granted scope','profiles':'static, visual or robot_layout with explicit executor'}),
    ('propose_layout_candidates','layout','propose',False,{'target_id':'granted built role','operation':'granted translate, rotate, change_support or move_component'}),
    ('apply_layout_candidate','layout','apply',True,{'candidate_id':'current deterministic candidate ID'}),
    ('finish','delivery','finish',False,{}),
    ('terminate','delivery','terminate',False,{}),
):
    _RECORDS.append({'stage':'critic_repair','name':name,'version':'v1','writes':writes,
        'scope':'explicit repair scope','requires':'current version, granted scope and phase preconditions',
        'effect':'scoped '+operation,'domain':domain,'operation':operation,'scope_kind':'repair_scope',
        'reads':['scene_graph','assets','current_evidence'],'failure_codes':
        ['CRITIC_INPUT_INVALID','CRITIC_SOURCE_CHANGED'] if name=='critique_scene' else
        ['LAYOUT_CANDIDATE_STALE','LAYOUT_SCOPE_VIOLATION'] if domain=='layout' else [],
        'invalidates':['static','target_ik','approach_path','render','critic'] if writes else [],
        'reverify':['critique_scene'] if writes else [],'cost_class':'geometry' if domain in ('review','layout') else 'local',
        'parameter_schema':schema})


validate_catalog(_RECORDS)


def catalog():
    """Return a detached snapshot so callers cannot change registry metadata."""
    return deepcopy(_RECORDS)


def describe_tools(stage, names):
    """Describe an already-computed legal set without reordering or broadening it."""
    if stage not in STAGES:
        raise GraphError('TOOL_CAPABILITY_INVALID', 'unknown stage')
    if not isinstance(names, (list, tuple)) or any(not isinstance(n, str) for n in names) or len(names) != len(set(names)):
        raise GraphError('TOOL_CAPABILITY_INVALID', 'legal names')
    by_name = {r['name']: r for r in _RECORDS if r['stage'] == stage}
    result = []
    for name in names:
        if name not in by_name:
            raise GraphError('TOOL_NOT_AVAILABLE', stage + ':' + name)
        # Stage is selected by the caller; it is unnecessary model payload.
        result.append(deepcopy({k: v for k, v in by_name[name].items() if k != 'stage'}))
    return result


def group_tools(descriptions):
    """Group just the current legal tools, preserving their order within each domain."""
    groups = {}
    for tool in descriptions:
        domain = tool['domain']
        if domain not in DOMAINS:
            raise GraphError('TOOL_CAPABILITY_INVALID', 'domain')
        groups.setdefault(domain, []).append(tool['name'])
    return groups


def legacy_tools(stage):
    """Compatibility view for existing transaction metadata consumers."""
    keys = ('version', 'writes', 'scope', 'effect')
    if stage == 'scene':
        keys = ('version', 'writes', 'scope', 'requires', 'effect')
    if stage not in STAGES:
        raise GraphError('TOOL_CAPABILITY_INVALID', 'unknown stage')
    return {r['name']: {k: deepcopy(r[k]) for k in keys}
            for r in _RECORDS if r['stage'] == stage}
