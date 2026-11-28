# SPDX-License-Identifier: Apache-2.0
import pytest
from rfw_w.scenes.contracts.reference_episode import EpisodeSpec


def stage(**changes):
    value = dict(name='open', command='grasp_pull', targets=['drawer'],
                 requires=['closed'], ensures=['opened'])
    value.update(changes)
    return value


def test_unknown_binding_and_duplicate_stage_rejected():
    with pytest.raises(ValueError, match='EPISODE_BINDING_MISSING'):
        EpisodeSpec('T03', 'a'*64, {'drawer': 'body'}, (stage(targets=['missing']),))
    with pytest.raises(ValueError, match='EPISODE_STAGE_INVALID'):
        EpisodeSpec('T03', 'a'*64, {'drawer': 'body'}, (stage(), stage()))


def test_revision_hash_changes_with_motion_obligation():
    first = EpisodeSpec('T03', 'a'*64, {'drawer': 'body'}, (stage(),))
    second = EpisodeSpec('T03', 'a'*64, {'drawer': 'body'}, (stage(ensures=['opened', 'passively_held']),))
    assert first.content_hash != second.content_hash
    assert first.to_dict()['state_authority'] == 'runtime_measurement'
