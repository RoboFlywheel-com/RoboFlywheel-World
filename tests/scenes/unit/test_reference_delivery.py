# SPDX-License-Identifier: Apache-2.0
"""Evidence rejection precedes optional simulator imports and output mutations."""
import json
import pytest
from rfw_w.scenes.workers.mujoco.reference_export import export_episode


def test_failed_reference_cannot_be_exported(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (run/'task_validation.json').write_text(json.dumps({'status':'failed','checks':{'grasp':False}}))
    out=tmp_path/'export'
    with pytest.raises(ValueError,match='REQUIRES_PASSED'):
        export_episode(tmp_path/'world',run,out)
    assert not out.exists()


def test_stale_reference_cannot_be_exported(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (run/'task_validation.json').write_text(json.dumps({'status':'passed','checks':{'grasp':True},'trajectory_sha256':'0'*64}))
    (run/'trajectory.json').write_text(json.dumps({'samples':[]}))
    out=tmp_path/'export'
    with pytest.raises(ValueError,match='STALE_VALIDATION'):
        export_episode(tmp_path/'world',run,out)
    assert not out.exists()
