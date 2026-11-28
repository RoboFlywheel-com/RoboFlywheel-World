# SPDX-License-Identifier: Apache-2.0
"""Task identity and source preparation failures precede simulator execution."""
import json
from pathlib import Path
import pytest
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph


def test_garden_graph_does_not_start_at_goal():
    path = Path(__file__).resolve().parents[1] / 'fixtures/reference_rigid/graph.json'
    graph = TaskWorldGraph.from_dict(json.loads(path.read_text()))
    raw = graph.to_dict()
    assert len(raw['steps']) == 6
    assert any(g['subject'] == 'pot' and g['object'] == 'saucer' for g in raw['goals'])
    assert all(s['object'] == 'source_area' for s in raw['initial_state'] if s['predicate'] == 'ON')


def test_garden_requires_task_source_files(tmp_path):
    pytest.importorskip('trimesh')
    from rfw_w.scenes.adapters.reference_rigid_world import build
    demo = Path(__file__).resolve().parents[1] / 'fixtures/reference_rigid'
    graph = json.loads((demo / 'graph.json').read_text())
    profile = json.loads((demo / 'profile.json').read_text())
    with pytest.raises(FileNotFoundError):
        build(dict(robot_sources=str(tmp_path / 'missing'), scratch=str(tmp_path)), tmp_path / 'world',
              graph=graph, profile=profile, template=demo / 'world.xml.template')
    assert not (tmp_path / 'world/manifest.json').exists()


def test_bench_rejects_unsupported_support_relationship(tmp_path):
    pytest.importorskip('trimesh')
    from rfw_w.scenes.adapters.reference_rigid_world import build
    demo = Path(__file__).resolve().parents[1] / 'fixtures/reference_rigid'
    graph = json.loads((demo / 'graph.json').read_text())
    profile = json.loads((demo / 'profile.json').read_text())
    profile.update(task_id='tools', preset='bench_pick_place_v1')
    with pytest.raises(ValueError, match='BENCH_PROFILE_UNSUPPORTED'):
        build({}, tmp_path/'world', graph=graph, profile=profile, template=demo/'world.xml.template')
    assert not (tmp_path/'world').exists()
