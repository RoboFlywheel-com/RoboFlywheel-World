# SPDX-License-Identifier: Apache-2.0
"""Pure surface appearance contract. File and image checks belong to the verifier."""
from __future__ import annotations
import math
from pathlib import PurePosixPath
from rfw_w.scenes.contracts.hashing import is_digest

SCHEMA = 'pvscene.surface-appearance/v1'
CHANNEL_SPACES = {'base_color': 'sRGB', 'normal_gl': 'Non-Color', 'roughness': 'Non-Color'}


class SurfaceAppearanceError(ValueError):
    def __init__(self, code, detail=''):
        self.code, self.detail = code, detail
        super().__init__(f'{code}({detail})')


def require(condition, code, detail=''):
    if not condition:
        raise SurfaceAppearanceError(code, detail)


def relative_ref(ref):
    require(isinstance(ref, str) and bool(ref) and '\\' not in ref,
            'SURFACE_BINDING_INVALID', 'relative reference')
    p = PurePosixPath(ref)
    require(not p.is_absolute() and '..' not in p.parts and ':' not in ref,
            'SURFACE_BINDING_INVALID', 'relative reference')
    return p


def validate_bindings(bindings, nodes):
    require(isinstance(bindings, list) and bool(bindings), 'SURFACE_BINDING_INVALID', 'bindings')
    seen = set()
    for b in bindings:
        require(isinstance(b, dict) and set(b) == {'surface_id', 'room_id', 'scope_id',
                'material_id', 'meters_per_tile', 'intent'}, 'SURFACE_BINDING_INVALID', 'fields')
        sid = b['surface_id']
        require(isinstance(sid, str) and sid and sid not in seen, 'SURFACE_BINDING_INVALID', 'surface ID')
        seen.add(sid)
        scope = nodes.get(b['scope_id'], {})
        require(scope.get('kind') in ('floor', 'wall_set') and scope.get('room_id') == b['room_id'],
                'SURFACE_BINDING_INVALID', sid)
        require(isinstance(b['material_id'], str) and bool(b['material_id']), 'MATERIAL_NOT_FOUND', sid)
        tile = b['meters_per_tile']
        require(type(tile) in (float, int) and math.isfinite(tile) and 0.01 <= tile <= 100,
                'SURFACE_UV_INVALID', sid)
        intent = b['intent']
        require(isinstance(intent, dict) and set(intent) == {'source', 'description'}
                and intent['source'] in ('explicit', 'inferred')
                and isinstance(intent['description'], str) and 0 < len(intent['description']) <= 400,
                'SURFACE_BINDING_INVALID', 'intent')


def validate_graph_appearance(value, nodes):
    require(isinstance(value, dict) and set(value) == {'schema_version', 'ledger_ref', 'ledger_hash', 'bindings'},
            'SURFACE_BINDING_INVALID', 'appearance fields')
    require(value['schema_version'] == SCHEMA and is_digest(value['ledger_hash']),
            'SURFACE_BINDING_INVALID', 'schema/hash')
    relative_ref(value['ledger_ref'])
    validate_bindings(value['bindings'], nodes)


def validate_material(mid, material):
    require(isinstance(material, dict) and set(material) == {'provider', 'base_color_rgba', 'roughness', 'channels'},
            'SURFACE_BINDING_INVALID', mid)
    require(material['provider'] in ('local_pbr', 'solid'), 'SURFACE_BINDING_INVALID', mid)
    rgba = material['base_color_rgba']
    require(isinstance(rgba, list) and len(rgba) == 4 and all(type(x) in (int, float)
            and math.isfinite(x) and 0 <= x <= 1 for x in rgba), 'SURFACE_BINDING_INVALID', mid)
    require(type(material['roughness']) in (int, float) and math.isfinite(material['roughness'])
            and 0 <= material['roughness'] <= 1, 'SURFACE_BINDING_INVALID', mid)
    channels = material['channels']
    require(isinstance(channels, dict), 'MATERIAL_CHANNEL_MISSING', mid)
    if material['provider'] == 'solid':
        require(channels == {}, 'SURFACE_BINDING_INVALID', mid)
        return
    require(set(channels) == set(CHANNEL_SPACES), 'MATERIAL_CHANNEL_MISSING', mid)
    for channel, space in CHANNEL_SPACES.items():
        item = channels[channel]
        require(isinstance(item, dict) and set(item) == {'ref', 'sha256', 'color_space'}
                and is_digest(item['sha256']) and item['color_space'] == space,
                'MATERIAL_CHANNEL_MISSING', mid + ':' + channel)
        relative_ref(item['ref'])
