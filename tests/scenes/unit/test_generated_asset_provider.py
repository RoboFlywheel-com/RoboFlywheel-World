# SPDX-License-Identifier: Apache-2.0
"""Synthetic metric packages exercise the independent scene-side data boundary."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from rfw_w.scenes.assets.providers.generated_package import GeneratedPackageError, prepare_package
from rfw_w.scenes.assembly.geometry import _placed_extents
from rfw_w.scenes.assembly.placement import build_scene
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.resources import data_path
from rfw_w.scenes.solvers.scene_ir import scene_ir_from_build
from rfw_w.scenes.solvers.layout import SurfacePlacementConfig


def package_fixture(root, *, hull_count=2):
    root.mkdir()
    mesh = trimesh.creation.box(extents=(0.08, 0.09, 0.10))
    # Intentionally offset origin: recentering would invalidate the original COM and inertia.
    mesh.apply_translation((0.025, -0.012, 0.035))
    (root / 'mesh').mkdir()
    mesh.export(root / 'mesh/visual.obj')
    hull_paths = []
    for index in range(hull_count):
        part = trimesh.creation.box(extents=(0.08 / hull_count, 0.09, 0.10))
        part.apply_translation((0.025 - 0.04 + (index + 0.5) * 0.08 / hull_count, -0.012, 0.035))
        member = f'mesh/collision_{index}.obj'
        part.export(root / member)
        hull_paths.append(member)
    frame = {'source_up_axis': 'Y', 'target_up_axis': 'Z', 'scale': 0.1, 'rotation': np.eye(4).tolist(), 'dimensions_m': [0.08, 0.09, 0.1]}
    physics = {'total_mass': 0.137, 'com': [0.025, -0.012, 0.035], 'inertia': [[0.0002, 0.00001, 0], [0.00001, 0.00023, 0], [0, 0, 0.00027]], 'whole_mu1': 0.65, 'whole_mu2': 0.45, 'n_collision_hulls': hull_count, 'dimensions_m': frame['dimensions_m'], 'frame': frame}
    (root / 'part_physics.json').write_text(json.dumps(physics))
    robot = ET.Element('robot', name='fixture')
    link = ET.SubElement(robot, 'link', name='fixture')
    for kind, members in [('visual', ['mesh/visual.obj']), ('collision', hull_paths)]:
        for member in members:
            ET.SubElement(ET.SubElement(ET.SubElement(link, kind), 'geometry'), 'mesh', filename=member, scale='1 1 1')
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz='0.025 -0.012 0.035', rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value='0.137')
    ET.SubElement(inertial, 'inertia', ixx='0.0002', ixy='0.00001', ixz='0', iyy='0.00023', iyz='0', izz='0.00027')
    ET.ElementTree(robot).write(root / 'fixture.urdf')
    manifest = {'schema': 'rfw.asset-package/v1', 'asset_id': 'fixture', 'frame': frame, 'physics': 'part_physics.json', 'production_status': 'package_complete', 'formats': {'urdf': 'fixture.urdf'}, 'files': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file()}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / '_SUCCESS').write_text('package_complete\n')
    return root, physics


def rehash(root):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files'] = {member: hashlib.sha256((root / member).read_bytes()).hexdigest() for member in manifest['files']}
    path.write_text(json.dumps(manifest))


def minimal_spec():
    payload = json.loads(data_path('examples/office_pick_cup/task.json').read_text())
    payload['entities'] = [entity for entity in payload['entities'] if entity['id'] in ('desk', 'cup')]
    payload['initial_relations'] = [relation for relation in payload['initial_relations'] if relation['subject'] in ('desk', 'cup') and relation['object'] in ('desk', 'cup')]
    payload['requested_readiness'] = 'draft'
    return SceneSpec.from_dict(payload)


def test_provider_preserves_all_hulls_frame_physics_and_independent_copy(tmp_path):
    package, physics = package_fixture(tmp_path / 'source')
    original = trimesh.load(package / 'mesh/visual.obj', process=False).vertices.copy()
    prepared = prepare_package(package, prepared_root=tmp_path / 'out', license='CC0-1.0', semantic='cup')
    assert prepared.record.normalisation.is_identity
    assert prepared.record.rigid_physics.mass_kg == physics['total_mass']
    assert prepared.record.rigid_physics.center_of_mass_m.to_list() == physics['com']
    assert prepared.record.rigid_physics.inertia_kg_m2 == tuple(np.asarray(physics['inertia']).flat)
    assert len(prepared.handle.collision_uris) == 2
    assert prepared.collision_realization == 'merged_source_hulls_convex_proxy'
    assert np.array_equal(trimesh.load(prepared.handle.visual_uri, process=False).vertices, original)
    shutil.rmtree(package)
    assert all(Path(path).is_file() for path in prepared.handle.collision_uris)
    assert Path(prepared.handle.metadata_uri).is_file()


@pytest.mark.parametrize('failure', ['hash', 'success', 'escape', 'scale', 'physics', 'dimensions', 'hull_count'])
def test_provider_rejects_invalid_package(tmp_path, failure):
    package, physics = package_fixture(tmp_path / 'source')
    if failure == 'hash':
        (package / 'mesh/visual.obj').write_text('tampered')
    elif failure == 'success':
        (package / '_SUCCESS').write_text('partial')
    elif failure in ('escape', 'scale'):
        urdf = ET.parse(package / 'fixture.urdf')
        mesh = urdf.getroot().find('link/visual/geometry/mesh')
        mesh.set('filename' if failure == 'escape' else 'scale', '../escape.obj' if failure == 'escape' else '2 2 2')
        urdf.write(package / 'fixture.urdf'); rehash(package)
    else:
        if failure == 'physics': physics['inertia'][0][1] = 1
        if failure == 'dimensions': physics['dimensions_m'] = [1, 1, 1]
        if failure == 'hull_count': physics['n_collision_hulls'] = 1
        (package / 'part_physics.json').write_text(json.dumps(physics)); rehash(package)
    with pytest.raises(GeneratedPackageError):
        prepare_package(package, prepared_root=tmp_path / 'out', license='CC0-1.0', semantic='cup')
    assert not (tmp_path / 'out').exists()


def test_direct_binding_runs_placement_and_cli_without_external_assets(tmp_path, monkeypatch):
    package, physics = package_fixture(tmp_path / 'source')
    for name in list(__import__('os').environ):
        if name.startswith('PVSCENE_'):
            monkeypatch.delenv(name)
    spec = minimal_spec()
    scene = build_scene(spec, output_dir=tmp_path / 'scene', seed=spec.seed, asset_bindings={'cup': package}, asset_license='CC0-1.0')
    assert len(scene['instances']) == 2
    target = next(item for item in scene['instances'] if item.entity_id == 'cup')
    assert target.normalisation == 'none'
    assert scene['target_pos'] == target.geometry_centre_m()
    assert target.position_m.z + target.local_bounds_min_m.z == pytest.approx(scene['support_top'].plane_z_m + SurfacePlacementConfig().spawn_clearance_m)
    extent = dict(_placed_extents(scene))['cup']
    assert extent.z_low_m == pytest.approx(target.position_m.z + target.local_bounds_min_m.z)
    ir = scene_ir_from_build(spec, scene)
    item = next(item for item in ir.instances if item.entity_id == 'cup')
    assert item.bounds_min_m == target.local_bounds_min_m
    assert target.to_manifest()['rigid_physics']['inertia_kg_m2'] == list(np.asarray(physics['inertia']).flat)
    spec_path = tmp_path / 'spec.json'; spec_path.write_text(json.dumps(spec.to_dict()))
    bindings = tmp_path / 'bindings.json'; bindings.write_text(json.dumps({'cup': 'source'}))
    monkeypatch.setenv('PVSCENE_VLM_MODEL', 'dummy-model-not-called')
    monkeypatch.setenv('PVSCENE_AGENT_MODEL', 'dummy-agent-not-called')
    result = subprocess.run([sys.executable, '-m', 'rfw_w.scenes.commands.generate', '--spec', str(spec_path), '--output-dir', str(tmp_path / 'cli-scene'), '--asset-bindings', str(bindings), '--asset-license', 'CC0-1.0', '--request-only', '--no-replan'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list((tmp_path / 'cli-scene').rglob('isaac_request.json'))
    # Delivery serialisation must carry the true physics and honest worker fidelity too.
    delivery_result = subprocess.run([sys.executable, '-m', 'rfw_w.scenes.commands.generate', '--spec', str(spec_path), '--output-dir', str(tmp_path / 'delivery-scene'), '--asset-bindings', str(bindings), '--asset-license', 'CC0-1.0', '--skip-isaac', '--no-loop', '--no-replan'], capture_output=True, text=True)
    assert delivery_result.returncode == 1, delivery_result.stdout + delivery_result.stderr
    delivery = json.loads((tmp_path / 'delivery-scene/delivery_manifest.json').read_text())
    bound = next(item for item in delivery['assets_used'] if item['entity_id'] == 'cup')
    assert bound['rigid_physics']['center_of_mass_m'] == physics['com']
    assert bound['source_collision_hull_count'] == 2
    assert bound['physics_realization'] == 'mass_only_worker_inertia_estimated'
    assert not delivery['resolution']['meets_request']
    assert {item['gate_id'] for item in delivery['resolution']['blockers']} == {'backend.isaac.load', 'backend.isaac.run', 'backend.isaac.render'}
    from rfw_w.scenes.validation.verify_delivery import verify
    assert verify(tmp_path / 'delivery-scene')['verified'] is True
    verification = subprocess.run([sys.executable, '-m', 'rfw_w.scenes.validation.verify_delivery', '--delivery', str(tmp_path / 'delivery-scene')], capture_output=True, text=True)
    assert verification.returncode == 0, verification.stdout + verification.stderr


def test_explicit_license_and_unsupported_binding_fail(tmp_path):
    package, _ = package_fixture(tmp_path / 'source')
    with pytest.raises(GeneratedPackageError, match='license'):
        prepare_package(package, prepared_root=tmp_path / 'out', license='', semantic='cup')
    spec = minimal_spec()
    with pytest.raises(ValueError, match='GENERATED_BINDING_UNSUPPORTED'):
        build_scene(spec, output_dir=tmp_path / 'out', seed=spec.seed, asset_bindings={'desk': package}, asset_license='CC0-1.0')


def test_reused_prepared_proxy_is_verified_against_all_source_hulls(tmp_path):
    package, _ = package_fixture(tmp_path / 'source')
    prepared = prepare_package(package, prepared_root=tmp_path / 'out', license='CC0-1.0', semantic='cup')
    Path(prepared.worker_collision_uri).write_text('tampered')
    with pytest.raises(GeneratedPackageError, match='collision proxy'):
        prepare_package(package, prepared_root=tmp_path / 'out', license='CC0-1.0', semantic='cup')
