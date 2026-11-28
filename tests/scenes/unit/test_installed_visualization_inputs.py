# SPDX-License-Identifier: Apache-2.0
"""Installed example references work after leaving the repository checkout."""
import json
from pathlib import Path

from rfw_w.scenes.resources import data_path
from rfw_w.scenes.visualization.inputs import resolve_input


def test_all_packaged_example_input_references_resolve_from_external_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    count = 0
    def check(value):
        nonlocal count
        if isinstance(value, str) and value.startswith("examples/"):
            assert resolve_input(value).is_file(), value
            count += 1
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    for source in data_path("examples").rglob("*.json"):
        check(json.loads(source.read_text()))
    assert count >= 20


def test_explicit_local_input_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "examples" / "office_pick_cup" / "task.json"
    local.parent.mkdir(parents=True)
    local.write_text('{"local":true}')
    assert resolve_input("examples/office_pick_cup/task.json") == local
    assert resolve_input(local) == local
