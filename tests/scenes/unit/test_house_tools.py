# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.orchestrator.house_tools import run_house_tools
from rfw_w.scenes.solvers.house_tools import HouseToolError, measure_house_access, solve_house_layout


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/task_world"


def graph():
    request = json.loads((FIXTURES / "cross_room_request.json").read_text())
    slots = json.loads((FIXTURES / "cross_room_slots_response.json").read_text())["slots"]
    return compile_pick_place_slots(slots, request)


def test_two_room_shell_has_one_measured_opening(tmp_path):
    source = graph()
    saved, report = run_house_tools(source, tmp_path / "house")
    plan = json.loads((tmp_path / "house/geometry/house_plan.json").read_text())
    independent = measure_house_access(plan, tmp_path / "house/geometry")
    assert report["rooms"] == 2 and report["portals"] == 1
    assert report["measured_clearance_m"] == independent["measured_clearance_m"] == 1.0
    assert independent["wall_groups"] == 9 and independent["floor_groups"] == 2
    assert saved.content_hash != source.content_hash
    assert TaskWorldGraph.from_dict(json.loads((tmp_path / "house/graph.json").read_text())).content_hash == saved.content_hash
    assert saved.to_dict()["built"]["house_shell"]["status"] == "built"
    assert report["robot_navigation"] == "not_run"


def test_stale_version_and_existing_output_are_rejected(tmp_path):
    source = graph()
    with pytest.raises(GraphError, match="GRAPH_VERSION_CONFLICT"):
        run_house_tools(source, tmp_path / "stale", expected_hash="0"*64)
    run_house_tools(source, tmp_path / "good")
    with pytest.raises(GraphError, match="HOUSE_OUTPUT_EXISTS"):
        run_house_tools(source, tmp_path / "good")


def test_unbuildable_portal_rejected(tmp_path):
    raw = graph().to_dict()
    next(n for n in raw["nodes"] if n["kind"] == "portal")["requested_clearance_m"] = 10.0
    raw.pop("content_hash", None)
    bad = TaskWorldGraph.from_dict(raw)
    with pytest.raises(HouseToolError, match="PORTAL_GEOMETRY_INVALID"):
        solve_house_layout(bad)
    with pytest.raises(HouseToolError, match="PORTAL_GEOMETRY_INVALID"):
        run_house_tools(bad, tmp_path / "bad")
    failure = json.loads((tmp_path / "bad/failure.json").read_text())
    assert failure["reason_code"] == "PORTAL_GEOMETRY_INVALID"
    assert failure["events"][-1]["result"]["failure_code"] == "CLEARANCE_VIOLATION"


def test_tampered_shared_wall_rejected(tmp_path):
    source = graph()
    run_house_tools(source, tmp_path / "house")
    directory = tmp_path / "house/geometry"
    plan = json.loads((directory / "house_plan.json").read_text())
    mesh = (directory / "house.obj").read_text()
    (directory / "house.obj").write_text(mesh.replace("1.500000", "1.400000"))
    with pytest.raises(HouseToolError, match="HOUSE_SHELL_MISMATCH"):
        measure_house_access(plan, directory)
