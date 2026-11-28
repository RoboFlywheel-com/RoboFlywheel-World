# SPDX-License-Identifier: Apache-2.0
"""Inspect a delivered shell without Isaac. USD imports stay outside the stdlib core."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class ShellReferenceError(ValueError):
    pass


# A built shell's prims carry the provider's factory names, which is the only thing in the USD
# that says what a surface is: nothing marks a prim as "the floor". Longest hints first, because
# `skirtingboard_ceiling` contains both `skirtingboard` and `ceiling` and the coarser hint wins
# otherwise. Shared with `rfw_w.scenes.workers.multisim.extract_scene`, which names the same roles.
SHELL_ROLE_HINTS = (
    ('exterior', 'exterior'),
    ('skirtingboard_ceiling', 'ceiling_skirting'),
    ('skirtingboard', 'skirtingboard'),
    ('wall', 'wall'),
    ('floor', 'floor'),
    ('ceiling', 'ceiling'),
    ('door', 'door'),
    ('window', 'window'),
)


def shell_role(prim_path: str) -> str:
    """The role a shell prim's path names, or `shell_other` when it names none."""
    lowered = str(prim_path).lower()
    for hint, role in SHELL_ROLE_HINTS:
        if hint in lowered:
            return role
    return 'shell_other'


def inspect_materials(path: Path, stage, record: dict, assets) -> dict | None:
    """Recompute texture integrity and actual USD binding coverage, not manifest booleans."""
    from pxr import UsdGeom, UsdShade

    if 'material_manifest' not in record:
        return None
    manifest = (path.parent / record['material_manifest']).resolve()
    if not manifest.is_relative_to(path.parent) or not manifest.is_file():
        raise ShellReferenceError('MATERIAL_DEPENDENCY_MISSING(manifest)')
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != record.get('material_manifest_sha256'):
        raise ShellReferenceError('MATERIAL_BINDING_MISMATCH(manifest digest)')
    data = json.loads(manifest.read_text())
    if data.get('schema_version') != 'pvscene.material-manifest/v1':
        raise ShellReferenceError('MATERIAL_BINDING_MISMATCH(manifest version)')
    asset_hashes = {hashlib.sha256(Path(a).read_bytes()).hexdigest() for a in assets if Path(a).is_file()}
    textures = 0
    for surface in data.get('surfaces', []):
        channels = surface.get('channels', {})
        for channel in ('DIFFUSE', 'ROUGHNESS', 'NORMAL'):
            if channel not in channels:
                raise ShellReferenceError(f'MATERIAL_DEPENDENCY_MISSING({surface.get("object")}:{channel})')
        for channel, item in channels.items():
            dependency = (path.parent / item['uri']).resolve()
            if not dependency.is_relative_to(path.parent) or not dependency.is_file():
                raise ShellReferenceError(f'MATERIAL_DEPENDENCY_MISSING({item["uri"]})')
            digest = hashlib.sha256(dependency.read_bytes()).hexdigest()
            converted = item.get('converted_to')
            if converted and (channel != 'TRANSMISSION' or converted != 'OPACITY' or converted not in channels):
                raise ShellReferenceError('MATERIAL_BINDING_MISMATCH(channel conversion)')
            if digest != item.get('sha256') or (not converted and digest not in asset_hashes):
                raise ShellReferenceError(f'MATERIAL_BINDING_MISMATCH({item["uri"]}:digest or USD dependency)')
            textures += 1
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != len(data.get('surfaces', [])):
        raise ShellReferenceError('MATERIAL_BINDING_MISMATCH(surface count)')
    for prim in meshes:
        binding = UsdShade.MaterialBindingAPI(prim)
        material, _ = binding.ComputeBoundMaterial()
        if material:
            shader, _, _ = material.ComputeSurfaceSource()
            for channel in ('diffuseColor', 'roughness', 'normal'):
                if not shader or not shader.GetInput(channel).HasConnectedSource():
                    raise ShellReferenceError(f'MATERIAL_BINDING_MISMATCH({prim.GetPath()}:{channel})')
            continue
        coverage = set()
        for subset in binding.GetMaterialBindSubsets():
            material, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
            if material:
                shader, _, _ = material.ComputeSurfaceSource()
                for channel in ('diffuseColor', 'roughness', 'normal'):
                    if not shader or not shader.GetInput(channel).HasConnectedSource():
                        raise ShellReferenceError(f'MATERIAL_BINDING_MISMATCH({subset.GetPath()}:{channel})')
                coverage.update(subset.GetIndicesAttr().Get() or [])
        count = len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or [])
        if coverage != set(range(count)):
            raise ShellReferenceError(f'MATERIAL_BINDING_MISMATCH({prim.GetPath()}:unbound faces)')
    return {'surface_count': len(meshes), 'texture_count': textures,
            'manifest_sha256': record['material_manifest_sha256']}


def inspect_shell(uri: str, *, delivery: Path | None = None) -> dict:
    from pxr import Usd, UsdGeom, UsdLux, UsdUtils

    path = Path(uri).resolve()
    if not path.is_file():
        raise ShellReferenceError(f'SHELL_URI_UNRESOLVABLE({uri})')
    if delivery is not None and not path.is_relative_to(delivery.resolve()):
        raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(outside delivery)')
    try:
        stage = Usd.Stage.Open(str(path))
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as error:
        raise ShellReferenceError(f'SHELL_URI_UNRESOLVABLE({uri})') from error
    if unresolved:
        raise ShellReferenceError(f'SHELL_URI_UNRESOLVABLE(missing dependencies: {unresolved})')
    if delivery is not None:
        for dep in [layer.realPath for layer in layers] + list(assets):
            if dep and not Path(dep).resolve().is_relative_to(delivery.resolve()):
                raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(dependency outside delivery)')
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)
              and UsdGeom.Mesh(p).GetPointsAttr().Get()
              and UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get()]
    if not meshes or not stage.GetDefaultPrim():
        raise ShellReferenceError(f'SHELL_URI_EMPTY_GEOMETRY({uri})')
    sidecar = path.with_suffix('.export.json')
    if not sidecar.is_file():
        raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(export evidence absent)')
    record = json.loads(sidecar.read_text())
    material_result = inspect_materials(path, stage, record, assets)
    if record.get('lighting_policy') == 'worker_only':
        lights = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
        if lights:
            raise ShellReferenceError(f'SHELL_UNEXPECTED_LIGHT({lights})')
    for label in ('source', 'measurement'):
        if f'{label}_file' not in record:
            continue
        dependency = (path.parent / record[f'{label}_file']).resolve()
        if delivery is not None and not dependency.is_relative_to(delivery.resolve()):
            raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(source evidence outside delivery)')
        if not dependency.is_file() or hashlib.sha256(dependency.read_bytes()).hexdigest() != record[f'{label}_sha256']:
            raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(source evidence changed)')
        if label == 'measurement':
            count = json.loads(dependency.read_text()).get('measurements', {}).get('openings.count')
            if count != record.get('openings_measured'):
                raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(opening measurement mismatch)')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if record.get('usd_sha256') != digest:
        raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(export digest mismatch)')
    if not isinstance(record.get('openings_measured'), int) or record['openings_measured'] < 0:
        raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(opening measurement absent)')
    # Transform vertices, not local axis-aligned boxes: rotating the latter inflates the world
    # bound (kitchen door: 4.7558 versus actual 4.6529 m on 2026-09-22).
    transforms = UsdGeom.XformCache()
    low, high = [float('inf')] * 3, [float('-inf')] * 3
    for prim in meshes:
        transform = transforms.GetLocalToWorldTransform(prim)
        for point in UsdGeom.Mesh(prim).GetPointsAttr().Get():
            world = transform.Transform(point)
            for axis in range(3):
                low[axis] = min(low[axis], world[axis])
                high[axis] = max(high[axis], world[axis])
    # Export may change topology, but it must preserve evaluated world-space geometry bounds.
    for key, actual in [('source_bounds_min', low), ('source_bounds_max', high)]:
        if key in record and any(abs(float(a) - float(b)) > 1e-3 for a, b in zip(record[key], actual)):
            raise ShellReferenceError('SHELL_URI_UNRESOLVABLE(export changed source bounds)')
    result = dict(mode='referenced', usd_sha256=digest, mesh_count=len(meshes),
                openings_measured=record['openings_measured'],
                bounds_min=low, bounds_max=high,
                source_sha256=record['source_sha256'], materials=record['materials'])
    if material_result is not None:
        result['material_validation'] = material_result
    return result


def author_shell(stage, uri: str) -> int:
    from pxr import UsdGeom, UsdPhysics

    record = inspect_shell(uri)
    root = UsdGeom.Xform.Define(stage, '/World/Room').GetPrim()
    if not root.GetReferences().AddReference(str(Path(uri).resolve())):
        raise ShellReferenceError(f'SHELL_URI_UNRESOLVABLE({uri})')
    # Architectural surfaces are static triangle meshes; convex hulls would seal doors.
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(root.GetPath()) and prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set('none')
    return record['openings_measured']


def reference_matches(record: dict, actual: dict) -> bool:
    """Identity is exact; world bounds allow float32/float64 USD transform differences.

    Isaac versus orchestrator on three 2026-09-22 renders differed by at most 1.82e-7 m.
    One micrometre covers this arithmetic without accepting a millimetre geometry change.
    """
    for key, value in actual.items():
        other = record.get(key)
        if key in {"bounds_min", "bounds_max"}:
            if not isinstance(other, (list, tuple)) or len(other) != 3:
                return False
            if not all(math.isclose(float(a), float(b), abs_tol=1e-6, rel_tol=0.0)
                       for a, b in zip(other, value)):
                return False
        elif other != value:
            return False
    return True
