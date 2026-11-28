# SPDX-License-Identifier: Apache-2.0
"""The half of SPEC-009's background branch that turns a kept object into a delivery instance.

The pool (SPEC-011) decides what may stay; these tests are about everything after that decision:
the frame an object's world geometry is re-expressed in, the identity a delivery can read a
generated object by, and the prepared asset a generated mesh is written as. Each of them has a way
to be *silently* wrong -- an axis-aligned box that overstates a yawed object, an entity id whose
category a reader cannot recover, a sidecar that describes a mesh nobody wrote -- so each is pinned
against the wrong answer rather than only against the right one.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rfw_w.scenes.assets.generated_furniture import (
    ASSEMBLY_SCHEMA_VERSION,
    LocalFrame,
    asset_key,
    kept_objects,
    library_version,
    prepare_asset,
    read_assembly,
    recentre,
)
from rfw_w.scenes.contracts.background_constraints import (
    GENERATED_PROVIDER,
    generated_category,
    generated_entity_id,
)
from rfw_w.scenes.contracts.hashing import content_hash

PIN = 'a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4'


def _box(centre, size, *, yaw_rad=0.0, base_z=0.0):
    """A box's eight world corners, turned about Z by ``yaw_rad``."""
    cos, sin = math.cos(yaw_rad), math.sin(yaw_rad)
    points = []
    for dx in (-size[0] / 2.0, size[0] / 2.0):
        for dy in (-size[1] / 2.0, size[1] / 2.0):
            for dz in (0.0, size[2]):
                x, y = centre[0] + dx, centre[1] + dy
                points.append((cos * x - sin * y, sin * x + cos * y, base_z + dz))
    return points


# ---------------------------------------------------------------------------------------------
# The frame: an instance's box has to be the object's box, at any yaw
# ---------------------------------------------------------------------------------------------

def test_a_yawed_object_keeps_its_own_box_and_its_pose_puts_it_back():
    points = _box((1.25, -0.5), (1.2, 0.4, 0.75), yaw_rad=math.radians(37.0))
    frame = recentre(points, yaw_rad=math.radians(37.0))
    # The local box is the *object's* box: a world axis-aligned bound of the same object would be
    # 0.09 m^2 larger in footprint (a 1.44 x 0.94 rectangle around a 1.2 x 0.4 one).
    assert frame.dimensions_m.x == pytest.approx(1.2, abs=1e-9)
    assert frame.dimensions_m.y == pytest.approx(0.4, abs=1e-9)
    assert frame.dimensions_m.z == pytest.approx(0.75, abs=1e-9)
    # Put the local points back: rotate by the yaw and translate to the recorded origin, and every
    # vertex lands where it started. This is the claim `geometry.no_penetration` rests on.
    cos, sin = math.cos(frame.yaw_rad), math.sin(frame.yaw_rad)
    for (x, y, z), original in zip(frame.points, points):
        restored = (cos * x - sin * y + frame.position_m.x,
                    sin * x + cos * y + frame.position_m.y,
                    z + frame.position_m.z)
        assert restored == pytest.approx(original, abs=1e-9)


def test_the_origin_is_the_footprint_centre_with_the_base_at_zero():
    # The convention every prepared asset in this repository uses, checked where it is made: the
    # instance's `origin_at_base=True` is only true if this function centred the base to z=0.
    frame = recentre(_box((0.4, 0.9), (0.5, 0.5, 1.1)), yaw_rad=0.0)
    assert frame.bounds_min.z == pytest.approx(0.0, abs=1e-12)
    assert frame.bounds_min.x == pytest.approx(-0.25, abs=1e-12)
    assert frame.bounds_max.x == pytest.approx(0.25, abs=1e-12)
    assert (frame.position_m.x, frame.position_m.y) == pytest.approx((0.4, 0.9))


def test_an_object_with_no_geometry_is_named_not_measured():
    with pytest.raises(ValueError, match='GENERATED_ASSET_EMPTY'):
        recentre([], yaw_rad=0.0)


# ---------------------------------------------------------------------------------------------
# The identity: the category is inside the id, and the asset key is a function of the object
# ---------------------------------------------------------------------------------------------

def test_a_generated_entity_id_carries_its_category_and_refuses_a_stranger():
    assert generated_entity_id('shelf_trinkets', 2) == 'generated_shelf_trinkets_2'
    assert generated_category('generated_shelf_trinkets_2') == 'shelf_trinkets'
    # The ids this contract did not mint are not categories: a spec entity id, a hand-written one
    # with a category that is not an ordinal, and one with no category at all.
    for stranger in ('desk', 'generated_monitor', 'generated__1', 'generated_monitor_x',
                     'monitor_1', 'generated_monitor_1_x'):
        assert generated_category(stranger) is None
    with pytest.raises(ValueError, match='GENERATED_CATEGORY_MISSING'):
        generated_entity_id('', 1)


def test_the_asset_key_is_stable_for_one_object_and_moves_with_it():
    base = dict(factory='MonitorFactory', name='MonitorFactory(1).spawn_asset(2)',
                upstream_pin=PIN, dimensions_m=[0.6, 0.2, 0.4])
    assert asset_key(**base) == asset_key(**base)
    assert asset_key(**base) != asset_key(**{**base, 'name': 'MonitorFactory(1).spawn_asset(3)'})
    assert asset_key(**base) != asset_key(**{**base, 'dimensions_m': [0.61, 0.2, 0.4]})
    # A different upstream commit is a different asset: the tree that solved it is part of what the
    # object is, and a delivery that mixed two pins would otherwise look like one build.
    assert asset_key(**base) != asset_key(**{**base, 'upstream_pin': 'f' * 40})


def test_the_library_version_names_the_commit_and_refuses_to_invent_one():
    assert library_version(PIN) == f'infinigen@{PIN[:12]}'
    assert GENERATED_PROVIDER == 'generated'
    with pytest.raises(ValueError, match='GENERATED_PIN_MISSING'):
        library_version('')


# ---------------------------------------------------------------------------------------------
# The assembly record: read by name, and only the kept objects are instances
# ---------------------------------------------------------------------------------------------

def _assembly(tmp_path: Path, objects) -> Path:
    payload = {'schema_version': ASSEMBLY_SCHEMA_VERSION, 'objects': objects,
               'ledger_sha256': 'a' * 64, 'blend_sha256': 'b' * 64, 'upstream_pin': PIN,
               'kept_in_ledger': len(objects), 'exported': len(objects), 'skipped': [],
               'tilt_tolerance_deg': 1.0}
    payload['digest'] = content_hash({key: value for key, value in payload.items()
                                      if key != 'digest'})
    path = tmp_path / 'objects.json'
    path.write_text(json.dumps(payload))
    return path


def _record(name='MonitorFactory(1).spawn_asset(2)', category='monitor', tilt=0.0):
    return {'name': name, 'factory': name.split('(')[0], 'category': category, 'outcome': 'kept',
            'slug': '00_MonitorFactory', 'raw_obj': '00_MonitorFactory.obj', 'parts': 1,
            'vertex_count': 8, 'face_count': 12, 'dimensions_m': [0.6, 0.2, 0.4],
            'bounds_min_m': [-0.3, -0.1, 0.0], 'bounds_max_m': [0.3, 0.1, 0.4],
            'position_m': [1.5, 0.5, 0.0], 'yaw_rad': 0.0, 'tilt_deg': tilt,
            'materials': ['Metal'], 'has_texture': False,
            'origin_convention': 'footprint_centre_base_at_zero'}


def test_the_assembly_is_read_by_name_and_a_kept_object_keeps_its_category(tmp_path):
    path = _assembly(tmp_path, [_record(category='monitor'), _record(name='CellShelfFactory(1)'
                                                                    '.spawn_asset(2)',
                                                                    category='shelf')])
    payload = read_assembly(path)
    # Sorted by the object's own name, so two reads of one assembly cannot disagree about order.
    assert [item['category'] for item in kept_objects(payload)] == ['shelf', 'monitor']
    # A refused candidate cannot reach a delivery even if it is in the file: the ledger is the
    # evidence of the decision, and an export that carried a rejected object would be a second,
    # silent decision.
    payload['objects'].append({**_record(), 'outcome': 'rejected'})
    assert len(kept_objects(payload)) == 2


def test_an_assembly_of_an_unknown_schema_is_refused_by_name(tmp_path):
    path = tmp_path / 'objects.json'
    path.write_text(json.dumps({'schema_version': 'pvscene.generated-assembly/v0', 'objects': []}))
    with pytest.raises(ValueError, match='GENERATED_ASSEMBLY_SCHEMA_UNSUPPORTED'):
        read_assembly(path)
    path.write_text(json.dumps({'schema_version': ASSEMBLY_SCHEMA_VERSION}))
    with pytest.raises(ValueError, match='GENERATED_ASSEMBLY_MALFORMED'):
        read_assembly(path)


# ---------------------------------------------------------------------------------------------
# The prepared asset: written, then measured from disk
# ---------------------------------------------------------------------------------------------

def test_a_generated_mesh_becomes_a_prepared_asset_measured_from_disk(tmp_path):
    trimesh = pytest.importorskip('trimesh')
    raw = tmp_path / 'raw' / '00_MonitorFactory.obj'
    raw.parent.mkdir(parents=True)
    # A 0.6 x 0.2 x 0.4 box in the prepared frame (footprint centre at the origin, base at z=0).
    box = trimesh.creation.box(extents=(0.6, 0.2, 0.4))
    box.apply_translation((0.0, 0.0, 0.2))
    box.export(str(raw))
    record = _record()
    prepared = prepare_asset(record, raw, tmp_path / 'prepared_assets', upstream_pin=PIN)
    assert prepared.asset_key.provider == GENERATED_PROVIDER
    assert prepared.asset_key.library_version == f'infinigen@{PIN[:12]}'
    assert prepared.dimensions_m.to_list() == pytest.approx([0.6, 0.2, 0.4], abs=1e-6)
    assert prepared.origin_convention == 'footprint_centre_base_at_zero'
    # The sidecar is on disk, in the shape the retrieved libraries write, and describes the mesh
    # that is actually there -- a preparation that wrote one thing and reported another is the
    # failure this shape exists to make impossible.
    sidecar = json.loads((prepared.visual_obj.parent / 'prepared.json').read_text())
    assert sidecar['asset']['asset_id'] == prepared.asset_key.asset_id
    assert sidecar['visual_obj'] == prepared.visual_obj.name
    assert (prepared.visual_obj.parent / sidecar['collision_obj']).is_file()
    assert sidecar['mass_provenance']['status'] == 'estimated'
    # No support surface is claimed: a generated shelf is not a declared support of anything.
    assert sidecar['support_polygons'] == []


def test_a_mesh_that_does_not_match_the_recorded_size_is_refused(tmp_path):
    trimesh = pytest.importorskip('trimesh')
    raw = tmp_path / 'raw' / '00_MonitorFactory.obj'
    raw.parent.mkdir(parents=True)
    box = trimesh.creation.box(extents=(0.6, 0.2, 0.4))
    box.export(str(raw))
    record = {**_record(), 'dimensions_m': [0.9, 0.2, 0.4]}  # the export says something else
    with pytest.raises(ValueError, match='GENERATED_ASSET_MEASUREMENT_MISMATCH'):
        prepare_asset(record, raw, tmp_path / 'prepared_assets', upstream_pin=PIN)
