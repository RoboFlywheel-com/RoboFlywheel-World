# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import pytest
from rfw_w.scenes.contracts.scene_composition import background_role,migrate_composition
from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.orchestrator.scene_checkpoints import save_scene_checkpoint,rollback_scene_scope,read_checkpoint

def graph():
    root=Path(__file__).resolve().parents[1]/'fixtures/task_world'
    old=compile_pick_place_slots(json.loads((root/'cross_room_slots_response.json').read_text())['slots'],json.loads((root/'cross_room_request.json').read_text()))
    return migrate_composition(old,expected_hash=old.content_hash)

def test_rollback_creates_new_revision_and_preserves_checkpoint_bytes(tmp_path):
    old=graph();save_scene_checkpoint(old,tmp_path/'saved');before=(tmp_path/'saved/checkpoint.json').read_bytes()
    raw=old.to_dict();role,edge=background_role(raw,'study','cabinet','cabinet','floor');raw['nodes'].append(role);raw['edges'].append(edge)
    current=old.revise(raw,expected_hash=old.content_hash)
    new,report=rollback_scene_scope(current,tmp_path/'saved',tmp_path/'rollback',expected_hash=current.content_hash,scope_ids=('cabinet',),reason_code='FURNISHING_COLLISION')
    assert new.to_dict()['parent_hash']==current.content_hash
    assert len({new.content_hash,old.content_hash,current.content_hash})==3
    assert (tmp_path/'saved/checkpoint.json').read_bytes()==before
    assert all(n['id']!='cabinet' for n in new.to_dict()['nodes'])
    assert report['invalidated_evidence']
    with pytest.raises(GraphError,match='GRAPH_STALE_EVIDENCE'):new.check_evidence({'graph_hash':old.content_hash})

@pytest.mark.parametrize('scope,code',[(('cup',),'ROLLBACK_CORE_VIOLATION'),(('door',),'ROLLBACK_SCOPE_INVALID'),(('missing',),'ROLLBACK_SCOPE_INVALID')])
def test_rollback_rejects_unowned_scope(tmp_path,scope,code):
    old=graph();save_scene_checkpoint(old,tmp_path/'saved')
    with pytest.raises(GraphError,match=code):rollback_scene_scope(old,tmp_path/'saved',tmp_path/'bad',expected_hash=old.content_hash,scope_ids=scope,reason_code='TEST')
    assert json.loads((tmp_path/'bad/transaction.json').read_text())['result']['status']=='failed'

def test_checkpoint_tampering_fails(tmp_path):
    save_scene_checkpoint(graph(),tmp_path/'saved');p=tmp_path/'saved/checkpoint.json';raw=json.loads(p.read_text());raw['graph_hash']='0'*64;p.write_text(json.dumps(raw))
    with pytest.raises(GraphError,match='CHECKPOINT_TAMPERED'):read_checkpoint(tmp_path/'saved')

def test_core_inference_rollback_invalidates_built_artifacts(tmp_path):
    old=graph();raw=old.to_dict()
    node=next(n for n in raw['nodes'] if n['id']=='desk');node['source']='inferred';node['source_reason']='unspecified destination support'
    old=old.revise(raw,expected_hash=old.content_hash)
    save_scene_checkpoint(old,tmp_path/'saved')
    raw=old.to_dict();next(n for n in raw['nodes'] if n['id']=='desk')['category']='table'
    raw['built']['house_shell']={'status':'built'}
    current=old.revise(raw,expected_hash=old.content_hash)
    new,report=rollback_scene_scope(current,tmp_path/'saved',tmp_path/'rollback',expected_hash=current.content_hash,scope_ids=('desk',),reason_code='TEST')
    assert new.to_dict()['built']=={}
    assert next(n for n in new.to_dict()['nodes'] if n['id']=='desk')['category']=='desk'
    assert 'house_shell' in report['invalidated_evidence']
    assert 'build_house' in report['reverify_tools']

def test_stale_rollback_records_failure(tmp_path):
    old=graph();save_scene_checkpoint(old,tmp_path/'saved')
    with pytest.raises(GraphError,match='GRAPH_VERSION_CONFLICT'):
        rollback_scene_scope(old,tmp_path/'saved',tmp_path/'bad',expected_hash='0'*64,scope_ids=('study',),reason_code='TEST')
    assert not (tmp_path/'bad/graph.json').exists()

def test_repeated_scope_restore_is_rejected_without_new_graph(tmp_path):
 old=graph();raw=old.to_dict();r,e=background_role(raw,'study','cabinet','cabinet','floor');raw['nodes'].append(r);raw['edges'].append(e)
 current=old.revise(raw,expected_hash=old.content_hash);save_scene_checkpoint(current,tmp_path/'saved')
 with pytest.raises(GraphError,match='ROLLBACK_SCOPE_INVALID'):
  rollback_scene_scope(current,tmp_path/'saved',tmp_path/'noop',expected_hash=current.content_hash,scope_ids=('cabinet',),reason_code='REPEAT')
 assert not (tmp_path/'noop/graph.json').exists()
