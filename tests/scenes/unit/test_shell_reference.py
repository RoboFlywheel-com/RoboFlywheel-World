# SPDX-License-Identifier: Apache-2.0
"""The render must reference the delivered mesh, and missing/tampered evidence must fail."""
import importlib
import hashlib
import json
from pathlib import Path

import pytest

from rfw_w.scenes.workers.isaac.protocol import RoomGeometry, WorkerReport
from rfw_w.scenes.workers.isaac.shell_reference import (
    ShellReferenceError, author_shell, inspect_shell,
)


def shell_file(tmp_path, *, empty=False):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    path = tmp_path / 'shell.usda'
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, '/Shell')
    stage.SetDefaultPrim(root.GetPrim())
    if not empty:
        mesh = UsdGeom.Mesh.Define(stage, '/Shell/wall')
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 0, 2)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()
    path.with_suffix('.export.json').write_text(json.dumps(dict(
        usd_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256='source', openings_measured=2, materials='native USD',
    )))
    return path


def test_protocol_keeps_missing_uri_payload_unchanged():
    room = RoomGeometry(((0, 0), (2, 0), (0, 2)), 3)
    assert 'shell_uri' not in room.to_dict()
    assert RoomGeometry.from_dict(room.to_dict()) == room


def test_protocol_round_trip_reference():
    room = RoomGeometry(((0, 0), (2, 0), (0, 2)), 3, shell_uri='shell.usdc')
    assert RoomGeometry.from_dict(room.to_dict()) == room
    report = WorkerReport('scene', 'version', 'failed', shell_reference={'mode': 'referenced'})
    assert WorkerReport.from_dict(report.to_dict()).shell_reference == report.shell_reference


def test_missing_shell_is_named(tmp_path):
    pytest.importorskip('pxr')
    with pytest.raises(ShellReferenceError, match='SHELL_URI_UNRESOLVABLE'):
        inspect_shell(str(tmp_path / 'missing.usda'))


def test_empty_shell_is_named(tmp_path):
    with pytest.raises(ShellReferenceError, match='SHELL_URI_EMPTY_GEOMETRY'):
        inspect_shell(str(shell_file(tmp_path, empty=True)))


def test_changed_geometry_cannot_reuse_export_evidence(tmp_path):
    path = shell_file(tmp_path)
    path.write_text(path.read_text() + '\n# tampered\n')
    with pytest.raises(ShellReferenceError, match='digest mismatch'):
        inspect_shell(str(path))


def test_external_reference_is_not_self_contained(tmp_path):
    path = shell_file(tmp_path)
    with pytest.raises(ShellReferenceError, match='outside delivery'):
        inspect_shell(str(path), delivery=tmp_path / 'other')


def test_missing_dependency_is_rejected(tmp_path):
    from pxr import Usd
    path = shell_file(tmp_path)
    stage = Usd.Stage.Open(str(path))
    stage.GetDefaultPrim().GetReferences().AddReference('missing.usda')
    stage.GetRootLayer().Save()
    with pytest.raises(ShellReferenceError, match='missing dependencies'):
        inspect_shell(str(path))


def test_reference_authors_source_without_procedural_walls(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    path = shell_file(tmp_path)
    stage = Usd.Stage.CreateInMemory()
    assert author_shell(stage, str(path)) == 2
    mesh = stage.GetPrimAtPath('/World/Room/wall')
    assert mesh.IsA(UsdGeom.Mesh)
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath('/World/Room/floor_visual')
    assert inspect_shell(str(path), delivery=tmp_path)['mesh_count'] == 1


def test_export_bounds_must_match_evaluated_source(tmp_path):
    path = shell_file(tmp_path)
    sidecar = path.with_suffix('.export.json')
    record = json.loads(sidecar.read_text())
    record['source_bounds_min'] = [10, 10, 10]
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ShellReferenceError, match='changed source bounds'):
        inspect_shell(str(path))


def test_worker_room_dispatches_to_reference(tmp_path):
    path = shell_file(tmp_path)
    module = importlib.import_module('rfw_w.scenes.workers.isaac.render_worker')
    import sys
    from types import SimpleNamespace
    from pxr import Usd
    request = SimpleNamespace(room=RoomGeometry(((0, 0), (2, 0), (0, 2)), 3, shell_uri=str(path)))
    stage = Usd.Stage.CreateInMemory()
    assert module._author_room(stage, request) == 2
    assert stage.GetPrimAtPath('/World/Room/wall')
    assert not stage.GetPrimAtPath('/World/Room/wall_00')


def test_changed_source_evidence_is_rejected(tmp_path):
    path = shell_file(tmp_path)
    source = tmp_path / 'source.blend'
    source.write_bytes(b'original')
    sidecar = path.with_suffix('.export.json')
    record = json.loads(sidecar.read_text())
    record.update(source_file=source.name,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    sidecar.write_text(json.dumps(record))
    source.write_bytes(b'changed')
    with pytest.raises(ShellReferenceError, match='source evidence changed'):
        inspect_shell(str(path))


def test_export_selects_architecture_not_placeholders_or_furniture():
    module = importlib.import_module('rfw_w.scenes.workers.blender.export_built_shell')
    from types import SimpleNamespace

    def obj(collection):
        return SimpleNamespace(type='MESH', hide_render=False, users_collection=[SimpleNamespace(name=collection)])
    assert module.architectural_mesh(obj('unique_assets:room_wall'))
    assert module.architectural_mesh(obj('unique_assets:doors'))
    assert not module.architectural_mesh(obj('placeholders:portal_cutters'))
    assert not module.architectural_mesh(obj('unique_assets'))
    assert not module.architectural_mesh(obj('Scene Collection'))


def test_reference_identity_allows_usd_roundoff_but_not_changed_geometry():
    from rfw_w.scenes.workers.isaac.shell_reference import reference_matches
    actual = {'usd_sha256': 'same', 'bounds_max': [5., 4.65292769, 3.]}
    assert reference_matches({'usd_sha256': 'same', 'bounds_max': [5., 4.65292788, 3.]}, actual)
    assert not reference_matches({'usd_sha256': 'other', 'bounds_max': actual['bounds_max']}, actual)
    assert not reference_matches({'usd_sha256': 'same', 'bounds_max': [5., 4.654, 3.]}, actual)


def test_worker_only_shell_rejects_hidden_environment_light(tmp_path):
    from pxr import Usd, UsdLux
    path = shell_file(tmp_path)
    sidecar = path.with_suffix('.export.json')
    record = json.loads(sidecar.read_text())
    record['lighting_policy'] = 'worker_only'
    sidecar.write_text(json.dumps(record))
    assert inspect_shell(str(path))['mesh_count'] == 1
    stage = Usd.Stage.Open(str(path))
    UsdLux.DomeLight.Define(stage, '/Shell/env_light')
    stage.GetRootLayer().Save()
    record['usd_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ShellReferenceError, match='SHELL_UNEXPECTED_LIGHT'):
        inspect_shell(str(path))


def test_material_manifest_cannot_claim_absent_channels(tmp_path):
    path = shell_file(tmp_path)
    sidecar = path.with_suffix('.export.json')
    record = json.loads(sidecar.read_text())
    manifest = tmp_path / 'material_manifest.json'
    manifest.write_text(json.dumps({'schema_version': 'pvscene.material-manifest/v1',
                                    'surfaces': [{'object': 'wall', 'channels': {}}]}))
    record.update(material_manifest=manifest.name,
                  material_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ShellReferenceError, match='MATERIAL_DEPENDENCY_MISSING'):
        inspect_shell(str(path))


def test_material_manifest_cannot_escape_export(tmp_path):
    path = shell_file(tmp_path)
    sidecar = path.with_suffix('.export.json')
    record = json.loads(sidecar.read_text())
    record['material_manifest'] = '../outside.json'
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ShellReferenceError, match='MATERIAL_DEPENDENCY_MISSING'):
        inspect_shell(str(path))


def pbr_shell_file(tmp_path):
    from pxr import Sdf, Usd, UsdShade
    path = shell_file(tmp_path)
    stage = Usd.Stage.Open(str(path))
    material = UsdShade.Material.Define(stage, '/Shell/Material')
    shader = UsdShade.Shader.Define(stage, '/Shell/Material/Preview')
    shader.CreateIdAttr('UsdPreviewSurface')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    channels = {}
    for channel, input_name, value_type in [('DIFFUSE','diffuseColor',Sdf.ValueTypeNames.Color3f),
                                            ('ROUGHNESS','roughness',Sdf.ValueTypeNames.Float),
                                            ('NORMAL','normal',Sdf.ValueTypeNames.Normal3f)]:
        texture = tmp_path/(channel+'.png')
        texture.write_bytes(b'fixture dependency; this gate does not decode image pixels')
        node = UsdShade.Shader.Define(stage, '/Shell/Material/'+channel)
        node.CreateIdAttr('UsdUVTexture')
        node.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture.name))
        shader.CreateInput(input_name,value_type).ConnectToSource(node.ConnectableAPI(),'rgb')
        channels[channel] = {'uri':texture.name,'sha256':hashlib.sha256(texture.read_bytes()).hexdigest()}
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/Shell/wall')).Bind(material)
    stage.GetRootLayer().Save()
    manifest = tmp_path/'material_manifest.json'
    manifest.write_text(json.dumps({'schema_version':'pvscene.material-manifest/v1',
                                    'surfaces':[{'object':'wall','channels':channels}]}))
    sidecar = path.with_suffix('.export.json')
    record = json.loads(sidecar.read_text())
    record.update(usd_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),material_manifest=manifest.name,
                  material_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    sidecar.write_text(json.dumps(record))
    return path


def test_pbr_material_validation_survives_relocation(tmp_path):
    import shutil
    original = tmp_path/'original'
    original.mkdir()
    path = pbr_shell_file(original)
    expected = inspect_shell(str(path),delivery=original)['material_validation']
    moved = tmp_path/'moved'
    shutil.copytree(original,moved)
    shutil.rmtree(original)
    assert inspect_shell(str(moved/path.name),delivery=moved)['material_validation'] == expected


def test_pbr_existing_texture_is_not_enough_without_normal_connection(tmp_path):
    from pxr import Usd, UsdShade
    path = pbr_shell_file(tmp_path)
    stage = Usd.Stage.Open(str(path))
    UsdShade.Shader(stage.GetPrimAtPath('/Shell/Material/Preview')).GetInput('normal').DisconnectSource()
    stage.GetRootLayer().Save()
    sidecar = path.with_suffix('.export.json')
    record = json.loads(sidecar.read_text())
    record['usd_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ShellReferenceError,match='MATERIAL_BINDING_MISMATCH.*normal'):
        inspect_shell(str(path))


def test_pbr_texture_tampering_fails(tmp_path):
    path = pbr_shell_file(tmp_path)
    (tmp_path/'NORMAL.png').write_bytes(b'tampered')
    with pytest.raises(ShellReferenceError,match='MATERIAL_BINDING_MISMATCH'):
        inspect_shell(str(path))
