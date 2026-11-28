# SPDX-License-Identifier: Apache-2.0
"""Read a completed asset package without importing the asset production package.

This boundary checks the bytes and preserves the package's metric Z-up geometry,
collision decomposition and mass properties. It never normalises an asset again.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import xml.etree.ElementTree as ET

from rfw_w.scenes.contracts.asset import (
    AffordanceStatus, AssetHandle, AssetKey, AssetRecordV3, CollisionPart,
    NormalisationRecord, Provenance, RigidPhysics,
)
from rfw_w.scenes.contracts.scene_spec import PhysicsType
from rfw_w.scenes.contracts.units import Vec3

SCHEMA = "rfw.asset-package/v1"


class GeneratedPackageError(ValueError):
    """A package cannot preserve the physical asset contract."""


def _member(root: Path, relative: str, declared: dict | None = None) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise GeneratedPackageError("package member must be a relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise GeneratedPackageError("package member escapes its package")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise GeneratedPackageError("package member resolves outside its package")
    if not result.is_file():
        raise GeneratedPackageError(f"missing package member: {relative}")
    if declared is not None and relative not in declared:
        raise GeneratedPackageError(f"unhashed package member: {relative}")
    return result


def _manifest(root: Path) -> tuple[dict, str]:
    marker = _member(root, "_SUCCESS").read_text(encoding="utf-8").strip()
    if marker != "package_complete":
        raise GeneratedPackageError("_SUCCESS does not state package_complete")
    raw = _member(root, "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or manifest.get("production_status") != "package_complete":
        raise GeneratedPackageError("a completed rfw.asset-package/v1 package is required")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise GeneratedPackageError("package has no file hashes")
    for relative, expected in declared.items():
        actual = hashlib.sha256(_member(root, relative).read_bytes()).hexdigest()
        if not isinstance(expected, str) or actual != expected:
            raise GeneratedPackageError(f"package hash mismatch: {relative}")
    return manifest, hashlib.sha256(raw).hexdigest()


def load_bindings(path: Path, *, license: str) -> dict[str, Path]:
    """Resolve explicit entity identities; relative directories belong to this JSON file."""
    if not license or not license.strip():
        raise GeneratedPackageError("--asset-bindings requires an explicit --asset-license")
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise GeneratedPackageError("asset bindings must be a non-empty entity-to-package object")
    result = {}
    for entity, directory in payload.items():
        if not isinstance(entity, str) or not entity or not isinstance(directory, str) or not directory:
            raise GeneratedPackageError("each asset binding needs an entity identity and package directory")
        candidate = Path(directory)
        result[entity] = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
    return result


def _identity_geometry(element: ET.Element) -> None:
    origin = element.find("origin")
    if origin is not None:
        for attribute in ("xyz", "rpy"):
            values = [float(value) for value in origin.get(attribute, "0 0 0").split()]
            if len(values) != 3 or not all(value == 0.0 for value in values):
                raise GeneratedPackageError("package geometry must already be in its physical frame")
    meshes = element.findall("geometry/mesh")
    if len(meshes) != 1:
        raise GeneratedPackageError("every package geometry must reference one mesh")
    scale = [float(value) for value in meshes[0].get("scale", "1 1 1").split()]
    if scale != [1.0, 1.0, 1.0]:
        raise GeneratedPackageError("package geometry must not require another scale")


@dataclass(frozen=True)
class PreparedGeneratedPackage:
    record: AssetRecordV3
    handle: AssetHandle
    worker_collision_uri: str
    collision_realization: str
    package_dir: Path


def prepare_package(package: Path, *, prepared_root: Path, license: str, semantic: str) -> PreparedGeneratedPackage:
    """Verify, copy and describe one independently movable production artifact."""
    if not isinstance(license, str) or not license.strip():
        raise GeneratedPackageError("the caller must explicitly declare the asset license")
    import numpy as np
    import trimesh

    package = package.resolve()
    manifest, digest = _manifest(package)
    declared = manifest["files"]
    frame = manifest.get("frame", {})
    if frame.get("target_up_axis") != "Z":
        raise GeneratedPackageError("asset package must already use the metric Z-up physical frame")
    formats = manifest.get("formats", {})
    if "urdf" not in formats:
        raise GeneratedPackageError("the generated package needs URDF geometry references")
    urdf = _member(package, formats["urdf"], declared)
    robot = ET.parse(urdf).getroot()
    if len(robot.findall("link")) != 1 or robot.findall("joint"):
        raise GeneratedPackageError("only a single rigid-body package is supported")
    visuals, collisions = robot.findall("link/visual"), robot.findall("link/collision")
    if len(visuals) != 1 or not collisions:
        raise GeneratedPackageError("package requires one visual mesh and all collision meshes")
    geometry_paths = []
    for geometry in [*visuals, *collisions]:
        _identity_geometry(geometry)
        relative = geometry.find("geometry/mesh").get("filename", "")
        # URDF references are relative to its own directory, and must remain hashed members.
        resolved = _member(urdf.parent, relative)
        member = str(resolved.relative_to(package))
        _member(package, member, declared)
        geometry_paths.append(member)
    physics_member = manifest.get("physics", "")
    physics = json.loads(_member(package, physics_member, declared).read_text())
    inertia = np.asarray(physics.get("inertia"), dtype=float)
    if inertia.shape != (3, 3) or not np.isfinite(inertia).all() or not np.allclose(inertia, inertia.T, rtol=0, atol=1e-10):
        raise GeneratedPackageError("physical inertia must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[0] <= 0 or eigenvalues[2] > eigenvalues[:2].sum() + 1e-10:
        raise GeneratedPackageError("physical inertia is not physically valid")
    provenance = Provenance("asset_package_physics", SCHEMA, AffordanceStatus.ESTIMATED)
    rigid = RigidPhysics(
        mass_kg=float(physics["total_mass"]), center_of_mass_m=Vec3.from_sequence(physics["com"]),
        inertia_kg_m2=tuple(float(value) for value in inertia.flat),
        static_friction=float(physics["whole_mu1"]), dynamic_friction=float(physics["whole_mu2"]),
        restitution=float(physics.get("restitution", 0.0)), provenance=provenance,
    )
    if not all(math.isfinite(value) for value in (rigid.static_friction, rigid.dynamic_friction)):
        raise GeneratedPackageError("physical friction must be finite")
    rigid.assert_simulable()
    if physics.get("frame") != frame:
        raise GeneratedPackageError("manifest frame disagrees with the physical sidecar")
    inertial = robot.find("link/inertial")
    if inertial is None:
        raise GeneratedPackageError("URDF has no physical inertia")
    actual_mass = float(inertial.find("mass").get("value"))
    origin = inertial.find("origin")
    actual_com = [float(value) for value in origin.get("xyz", "0 0 0").split()]
    if any(float(value) != 0 for value in origin.get("rpy", "0 0 0").split()):
        raise GeneratedPackageError("URDF inertia must be expressed in the package physical frame")
    attributes = inertial.find("inertia").attrib
    expected_inertia = (inertia[0, 0], inertia[0, 1], inertia[0, 2], inertia[1, 1], inertia[1, 2], inertia[2, 2])
    if not np.isclose(actual_mass, rigid.mass_kg, rtol=1e-9) or not np.allclose(actual_com, rigid.center_of_mass_m.to_list(), rtol=1e-9, atol=1e-10) or not np.allclose([float(attributes[name]) for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")], expected_inertia, rtol=1e-9, atol=1e-12):
        raise GeneratedPackageError("URDF mass properties disagree with the physical sidecar")
    meshes = []
    for member in geometry_paths:
        mesh = trimesh.load(_member(package, member, declared), force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise GeneratedPackageError("package contains unusable mesh geometry")
        meshes.append(mesh)
    visual = meshes[0]
    dimensions = visual.bounds[1] - visual.bounds[0]
    if not np.allclose(dimensions, physics["dimensions_m"], rtol=1e-5, atol=1e-7) or not np.allclose(dimensions, frame["dimensions_m"], rtol=1e-5, atol=1e-7):
        raise GeneratedPackageError("measured visual dimensions disagree with physical dimensions")
    if physics.get("n_collision_hulls") != len(collisions):
        raise GeneratedPackageError("URDF collision count disagrees with the physical decomposition")
    destination = prepared_root.resolve() / "generated" / digest
    if destination.exists():
        _, copied_digest = _manifest(destination)
        if copied_digest != digest:
            raise GeneratedPackageError("prepared package does not match this immutable package")
    else:
        destination.mkdir(parents=True)
        # Copy only checked members, never unpublished extras or symlinked directories.
        for member in [*declared, "manifest.json", "_SUCCESS"]:
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_member(package, member), target)
        _, copied_digest = _manifest(destination)
        if copied_digest != digest:
            raise GeneratedPackageError("package changed while being copied")
    collision_members = geometry_paths[1:]
    if len(collisions) == 1:
        worker_collision = destination / collision_members[0]
        realization = "single_source_hull_convex_proxy"
    else:
        worker_collision = destination / "scene_collision_proxy.obj"
        # Concatenate in the original metric frame; the worker currently convexifies this mesh.
        proxy_bytes = trimesh.util.concatenate(meshes[1:]).export(file_type="obj").encode("utf-8")
        if worker_collision.exists():
            if worker_collision.read_bytes() != proxy_bytes:
                raise GeneratedPackageError("prepared collision proxy does not match the original hulls")
        else:
            worker_collision.write_bytes(proxy_bytes)
        realization = "merged_source_hulls_convex_proxy"
    key = AssetKey("generated_package", SCHEMA, str(manifest["asset_id"]))
    geometry_provenance = Provenance("package_collision_mesh", SCHEMA, AffordanceStatus.MEASURED)
    record = AssetRecordV3(
        key=key, source_hash=digest, license=license.strip(), semantic=semantic, super_category=semantic,
        physics_type=PhysicsType.RIGID, dimensions_m=Vec3.from_sequence(dimensions.tolist()),
        bounds_min_m=Vec3.from_sequence(visual.bounds[0].tolist()), bounds_max_m=Vec3.from_sequence(visual.bounds[1].tolist()),
        canonical_frame="world_z_up", normalisation=NormalisationRecord(1.0, Vec3.origin(), "world_z_up", "world_z_up", "preserved_asset_package_frame", SCHEMA),
        visual_uri=str(destination / geometry_paths[0]),
        collision_parts=tuple(CollisionPart(f"hull_{index}", "collision_mesh", True, len(mesh.vertices), len(mesh.faces), geometry_provenance) for index, mesh in enumerate(meshes[1:])),
        materials=(), rigid_physics=rigid, prepared_version=digest,
    )
    record.assert_usable_as_task_target()
    handle = AssetHandle(key, record.visual_uri, tuple(str(destination / member) for member in collision_members), str(destination / physics_member), digest)
    return PreparedGeneratedPackage(record, handle, str(worker_collision), realization, destination)
