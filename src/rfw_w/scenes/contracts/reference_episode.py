# SPDX-License-Identifier: Apache-2.0
"""Versioned measured-stage intent. Standard library only, no symbolic success."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping
from rfw_w.scenes.contracts.hashing import content_hash, is_digest


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: str
    graph_hash: str
    bindings: Mapping[str, str]
    stages: tuple[dict, ...]

    def __post_init__(self):
        if not self.task_id or not is_digest(self.graph_hash) or not self.bindings or not self.stages:
            raise ValueError('EPISODE_SPEC_INVALID')
        object.__setattr__(self, 'bindings', deepcopy(dict(self.bindings)))
        object.__setattr__(self, 'stages', deepcopy(self.stages))
        names = set()
        for stage in self.stages:
            if set(stage) != {'name', 'command', 'targets', 'requires', 'ensures'}:
                raise ValueError('EPISODE_STAGE_INVALID')
            if stage['name'] in names or not stage['requires'] or not stage['ensures']:
                raise ValueError('EPISODE_STAGE_INVALID')
            if any(name not in self.bindings for name in stage['targets']):
                raise ValueError('EPISODE_BINDING_MISSING')
            names.add(stage['name'])

    def to_dict(self):
        return {'schema_version': 'pvscene.reference-episode/v1', 'task_id': self.task_id,
                'graph_hash': self.graph_hash, 'bindings': dict(self.bindings),
                'stages': deepcopy(list(self.stages)), 'state_authority': 'runtime_measurement'}

    @property
    def content_hash(self):
        return content_hash(self.to_dict())
