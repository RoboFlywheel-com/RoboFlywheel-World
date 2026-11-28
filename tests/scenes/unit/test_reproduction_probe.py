# SPDX-License-Identifier: Apache-2.0
"""Native generated-request physics acceptance, including invalid inputs."""

import json

from rfw_w.scenes.validation.probe_request import main


def test_generated_two_body_request_steps_with_authored_pose(tmp_path):
    import trimesh

    mesh = tmp_path / "cube.obj"
    trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(mesh)
    request = {
        "physics_hz": 120,
        "instances": [{"instance_id": "block", "collision_uri": str(mesh),
                       "visual_uri": str(mesh), "translation": [0, 0, 0.5],
                       "rotation_wxyz": [1, 0, 0, 0], "dynamic": True, "mass_kg": 0.1}],
    }
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request))
    out = tmp_path / "check"
    assert main(["--request", str(source), "--out", str(out), "--steps", "12"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["initial_positions"]["block"] == [0, 0, 0.5]
    assert report["final_positions"]["block"][2] < 0.5
    assert report["task_run"] == report["settle"] == "not_run"
    assert report["physics"] == "request_mass_engine_derived_inertia"


def test_missing_request_mesh_cannot_pass(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"physics_hz": 120, "instances": [
        {"collision_uri": str(tmp_path / "missing.obj"), "visual_uri": "missing.obj"}]}))
    out = tmp_path / "check"
    assert main(["--request", str(request), "--out", str(out)]) == 1
    assert json.loads((out / "report.json").read_text())["error_type"] == "FileNotFoundError"
