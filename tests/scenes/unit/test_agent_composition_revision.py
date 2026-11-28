# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import pytest
from rfw_w.scenes.agents.composition_revision import validate_composition_patch, propose_composition_patch
from rfw_w.scenes.agents.llm_client import LLMConfig
from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.contracts.scene_composition import migrate_composition
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.orchestrator.composition_tools import commit_composition_patch


def graph():
    root=Path(__file__).resolve().parents[1]/'fixtures/task_world'
    old=compile_pick_place_slots(json.loads((root/'cross_room_slots_response.json').read_text())['slots'],json.loads((root/'cross_room_request.json').read_text()))
    return migrate_composition(old,expected_hash=old.content_hash)

def add():
    return {'operations':[{'action':'add_optional_role','room_id':'study','role_id':'cabinet','category':'cabinet','placement_kind':'floor','purpose':'background'}]}

def test_semantic_edit_can_add_and_remove_optional_without_changing_task(tmp_path):
    old=graph(); patch=add()
    new=commit_composition_patch(old,patch,tmp_path/'add',expected_hash=old.content_hash,scope_ids=('study',))
    assert new.task_view()['goals']==old.task_view()['goals']
    assert new.content_hash!=old.content_hash
    removed=validate_composition_patch(new,{'operations':[{'action':'remove_optional_role','target_id':'cabinet'}]},expected_hash=new.content_hash,scope_ids=('cabinet',))
    assert all(n['id']!='cabinet' for n in removed.to_dict()['nodes'])

@pytest.mark.parametrize('patch,scope,code',[
    ({'operations':[{'action':'remove_optional_role','target_id':'cup'}]},('cup',),'CORE_CONSTRAINT_CHANGED'),
    ({'operations':[{'action':'add_optional_role'}]},('study',),'COMPOSITION_PATCH_INVALID'),
    ({'operations':[{'action':'move','target_id':'cup'}]},('cup',),'COMPOSITION_PATCH_INVALID'),
    ({'operations':[], 'built':{}},(),'COMPOSITION_PATCH_INVALID'),
    (add(),(),'PATCH_SCOPE_VIOLATION'),
])
def test_rejected_patch_is_recorded_without_mutation(tmp_path,patch,scope,code):
    old=graph(); before=old.to_dict()
    with pytest.raises(GraphError,match=code):
        commit_composition_patch(old,patch,tmp_path/'rejected',expected_hash=old.content_hash,scope_ids=scope)
    assert old.to_dict()==before
    event=json.loads((tmp_path/'rejected/transaction.json').read_text())
    assert event['result']['status']=='failed'
    assert not (tmp_path/'rejected/graph.json').exists()

def test_online_permission_precedes_model(monkeypatch):
    monkeypatch.delenv('PVSCENE_ALLOW_EXTERNAL_VLM',raising=False)
    monkeypatch.setattr('rfw_w.scenes.agents.composition_revision.call_vision',lambda *a,**k:pytest.fail('external call'))
    with pytest.raises(SystemExit):
        propose_composition_patch({},[add()],LLMConfig('https://example.invalid','test','UNUSED'))

@pytest.mark.parametrize('patch',[{'operations':[{'action':'replace_inferred_category','target_id':[],'category':{},'reason_code':'TEST'}]},
                                {'operations':[{'action':'add_optional_role','room_id':'study','role_id':'bad','category':{},'placement_kind':'floor','purpose':'background'}]}])
def test_malformed_patch_types_are_named_failures(patch):
 old=graph()
 with pytest.raises(GraphError,match='COMPOSITION_PATCH_INVALID'):
  validate_composition_patch(old,patch,expected_hash=old.content_hash,scope_ids=('study',))
