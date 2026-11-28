# SPDX-License-Identifier: Apache-2.0
"""Final loss partition cannot be gamed by plan removal or candidate retries."""
from rfw_w.scenes.resources import data_path, module_path
import json
from pathlib import Path
import pytest
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.contracts.unplaced import UnplacedItem, split_unplaced


def office():
    return SceneSpec.from_dict(json.loads((data_path('examples/office_pick_cup/task.json')).read_text()))


def test_retry_success_is_not_a_missing_entity():
    spec = office()
    ids = [e.id for e in spec.entities]
    result = split_unplaced(spec, ids, ids, [UnplacedItem("chair", "chair", "NO_POSITION", "seating")])
    assert all(not items for items in result.values())


def test_core_closure_periphery_and_plan_drop_are_distinct():
    spec = office()
    missing = {"cup", "chair", "rug", "plant"}
    placed = {e.id for e in spec.entities} - missing
    planned = {e.id for e in spec.entities} - {"chair", "plant"}
    result = split_unplaced(spec, placed, planned, [])
    assert [i["entity_id"] for i in result["unplaced_core"]] == ["cup"]
    assert [i["entity_id"] for i in result["unplaced_task_closure"]] == ["chair"]
    assert [i["entity_id"] for i in result["unplaced_periphery"]] == ["rug"]
    assert [i["entity_id"] for i in result["periphery_dropped"]] == ["plant"]


def test_loss_count_is_per_entity_not_per_attempt():
    spec = office()
    ids = {e.id for e in spec.entities}
    result = split_unplaced(spec, ids - {"rug"}, ids,
                            [UnplacedItem("rug", "rug", "NO_POSITION", "room")] * 3)
    assert len(result["unplaced_periphery"]) == 1


def test_unknown_reason_is_refused():
    with pytest.raises(ValueError):
        UnplacedItem("rug", "rug", "whatever", "room")
