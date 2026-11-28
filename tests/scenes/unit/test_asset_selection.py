# SPDX-License-Identifier: Apache-2.0
import importlib
from rfw_w.scenes.resources import module_path, data_path
import rfw_w.scenes.application as application_module
from dataclasses import replace
from types import SimpleNamespace

from rfw_w.scenes.assets import selection
from rfw_w.scenes.assets.procedural_support import materialise
from rfw_w.scenes.contracts.scene_spec import EntitySpec, Range1D
from rfw_w.scenes.contracts.units import Vec3
from rfw_w.scenes.solvers.support_shape import SupportShape
from rfw_w.scenes.workers.isaac.geometry import load_obj


def test_drawer_box_does_not_outrank_pen_organiser():
    assert selection.name_score("pen_holder", "Spinning Desk Organiser") > selection.name_score(
        "pen_holder", "Acrylic 3 Drawer Box")


def test_recall_is_limited_after_category_admission(monkeypatch):
    from rfw_w.scenes.adapters import simready_retrieval as retrieval
    entity = EntitySpec.from_dict({"id": "cup", "category": "cup", "role": "task_target",
                                  "physics_type": "rigid", "mobility": "dynamic"})
    goal = SimpleNamespace(target_size_m=Vec3(.1, .1, .1))
    monkeypatch.setattr(retrieval, "build_goal", lambda *a, **kw: goal)
    monkeypatch.setattr(retrieval, "_measure_dimensions", lambda *a: (Vec3(.1, .1, .1), None))
    def hit(index, category):
        asset = SimpleNamespace(asset_id=str(index), semantic_class=category, mass_kg=.1,
                                material_label="ceramic", metadata={"super_category": category},
                                friction=None, friction_secondary=None)
        evidence = SimpleNamespace(total_score=1., semantic_score=1., size_score=1.,
                                   readiness_score=1., affordance_score=1.)
        return SimpleNamespace(asset=asset, evidence=evidence)
    hits = [hit(i, "Bowl") for i in range(8)] + [hit(8, "Coffee Cup"), hit(9, "Mug")]
    def retrieve(goal, *, limit):
        return SimpleNamespace(candidates=hits[:limit], rejected=(), failure_code=None)
    engine = SimpleNamespace(retrieve=retrieve, policy=SimpleNamespace(max_dimension_ratio=2.))
    config = SimpleNamespace(provider="simready", library_version="test")
    result = retrieval.retrieve_entity(engine, entity, config=config, limit=2,
                                       admitted_categories=("Mug", "Coffee Cup"))
    assert [view.super_category for view in result.candidates] == ["Mug", "Coffee Cup"]
    assert len(result.rejected) == 8


def test_size_violation_cannot_be_hidden_by_semantic_match(monkeypatch):
    entity = EntitySpec.from_dict({"id": "pen", "category": "pen_holder", "role": "context",
                                  "physics_type": "rigid", "mobility": "dynamic"})
    entity = replace(entity, size_bounds_m=(Range1D(.06, .12),) * 3)
    config = SimpleNamespace(glb_path=lambda asset: asset)
    monkeypatch.setattr(selection, "load_synset_index", lambda _: {"a": [
        {"id": "large", "name": "Spinning Pen Holder"}, {"id": "small", "name": "Desk Organiser"}]})
    monkeypatch.setattr(selection, "candidates_for_synset", lambda *_: ("large", "small"))
    monkeypatch.setattr(selection, "measure_glb_extents", lambda asset:
                        Vec3(1, 1, 1) if asset == "large" else Vec3(.1, .1, .1))
    assert selection.rank_for_entity(config, entity, synsets=("a",), target_width_m=.1)[0] == "small"

    # The looser ratio gate also permits 0.17 m; the actual requested interval still wins.
    monkeypatch.setattr(selection, "measure_glb_extents", lambda asset:
                        Vec3(.17, .17, .17) if asset == "large" else Vec3(.1, .1, .1))
    assert selection.rank_for_entity(config, entity, synsets=("a",), target_width_m=.1)[0] == "small"


def test_wood_finish_reaches_worker_without_changing_collision(tmp_path):
    kwargs = dict(width_m=1.2, depth_m=.6, height_m=.75, colour=(.5, .3, .12), out_dir=tmp_path)
    plain = materialise(SupportShape.SLAB_LEGS, **kwargs)
    wood = materialise(SupportShape.SLAB_LEGS, wood_grain=True, **kwargs)
    assert plain.asset_key != wood.asset_key
    mesh = load_obj(wood.visual_obj)
    assert mesh.is_textured
    assert mesh.diffuse_texture(wood.visual_obj).is_file()
    assert wood.has_diffuse_texture
    assert plain.collision_obj.read_bytes() == wood.collision_obj.read_bytes()
    assert plain.dimensions_m == wood.dimensions_m
    assert materialise(SupportShape.SLAB_LEGS, wood_grain=True, **kwargs).asset_key == wood.asset_key


def test_near_wall_search_prefers_anchor_but_respects_reserved_floor():
    from rfw_w.scenes.solvers.furnish import Wall, place_against_wall
    from rfw_w.scenes.solvers.layout import Footprint, OccupancyMap
    wall = Wall(index=0, start=(0, 0), end=(6, 0), length_m=6, inward=(0, 1))
    occupancy = OccupancyMap()
    occupancy.add("reserved", Footprint((.6, .6)).at(4.5, .5))
    kwargs = dict(instance_id="bed", entity_id="bed", footprint=Footprint((.5, .5)),
                  wall=wall, occupancy=occupancy, object_height_m=0, wall_face_offset_m=.1)
    plain = place_against_wall(**kwargs).placement
    near = place_against_wall(**kwargs, preferred_xy=(4.5, .5)).placement
    assert plain is not None and near is not None
    assert abs(near.position_m.x - 4.5) < abs(plain.position_m.x - 4.5)
    assert abs(near.position_m.x - 4.5) >= 1.1
    short = replace(wall, end=(1.19, 0), length_m=1.19)
    assert place_against_wall(**{**kwargs, "wall": short}).placement is None


def test_rollback_restores_matching_render_and_critique():
    from pathlib import Path
    host = application_module.SceneHost.__new__(application_module.SceneHost)
    host._backend_by_version = {'a': ({'scene_version': 'a'}, Path('a'), Path('request_a'))}
    host._critique_by_version = {'a': ('critique_a', 'gate_a')}
    host._restore_observations('a')
    assert host.worker_report['scene_version'] == 'a' and host.critique == 'critique_a'
    host._restore_observations('unmeasured')
    assert host.worker_dir is None and host.critique is None
    host._restore_observations('a')
    assert host.worker_dir == Path('a') and host.visual_gate == 'gate_a'
