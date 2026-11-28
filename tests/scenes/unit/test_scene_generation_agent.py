# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.assets.room_tools import RoomAssetError
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.orchestrator.scene_generation_agent import SceneGenerationAgent

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/task_world"


def graph():
    request = json.loads((FIXTURES / "cross_room_request.json").read_text())
    slots = json.loads((FIXTURES / "cross_room_slots_response.json").read_text())["slots"]
    return compile_pick_place_slots(slots, request)


def test_agent_refuses_existing_evidence_directory(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    with pytest.raises(GraphError, match="AGENT_OUTPUT_EXISTS"):
        SceneGenerationAgent().run(graph(), directory)


def test_agent_records_named_asset_environment_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("PVSCENE_HSSD_ROOT", raising=False)
    monkeypatch.delenv("PVSCENE_HSSD_SYNSET_INDEX", raising=False)
    directory = tmp_path / "run"
    with pytest.raises(RoomAssetError, match="ASSET_ENV_MISSING"):
        SceneGenerationAgent().run(graph(), directory)
    result = json.loads((directory / "workflow.json").read_text())
    assert result["phase"] == "G3" and result["failure_code"] == "ASSET_ENV_MISSING"
    assert result["model_requests_used"] == 0
    assert (directory / "house/geometry/house.obj").is_file()
    failure = json.loads((directory / "assets/failure.json").read_text())
    event = failure["events"][-1]
    assert event["request"]["tool_version"] == "retrieve_role_candidates/v1"
    assert event["result"]["status"] == "failed"
    assert event["result"]["failure_code"] == "ASSET_IO_FAILURE"
