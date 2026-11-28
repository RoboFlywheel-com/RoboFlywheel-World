# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import pytest

from rfw_w.assets.contracts.request import AssetRequest
from rfw_w.assets.runtime import Runtime
from rfw_w.assets.application import generate
from rfw_w.assets.simulation.validation import validate, verify_files


def test_missing_sonata_checkpoint_fails_before_model_import(tmp_path):
    from rfw_w.assets.workers.p3sam_patch import patch_p3sam

    with pytest.raises(FileNotFoundError, match="Sonata checkpoint"):
        patch_p3sam(str(tmp_path), sonata_ckpt=str(tmp_path / "missing.pth"))


def supplied_asset(tmp_path):
    import numpy as np, trimesh

    mesh = tmp_path / "cube.glb"
    trimesh.creation.box().export(mesh)
    ids = tmp_path / "face_ids.npy"
    np.save(ids, np.zeros(12, dtype=int))
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "object": {"dimension_cm": [10, 10, 10], "mass_g": 100},
                "parts": [
                    {
                        "mask_color": "Red",
                        "name": "body",
                        "material": {"category": "plastic", "subtype": "ABS"},
                        "static_friction": 0.7,
                        "dynamic_friction": 0.5,
                    }
                ],
            }
        )
    )
    return {
        "asset_id": "cube_0001",
        "class_id": "cube",
        "input": {"mesh": str(mesh)},
        "annotations": str(annotations),
        "face_ids": str(ids),
        "formats": ["urdf", "mjcf", "usd"],
        "source_up_axis": "Z",
    }


def test_real_public_mesh_to_package_and_relocated_mujoco(tmp_path):
    request = supplied_asset(tmp_path)
    runtime = Runtime(
        {"physical": {"python": sys.executable, "options": {"voxel-resolution": 8}}}
    )
    report = generate(request, runtime, tmp_path / "run")
    assert report["status"] == "package_complete"
    assert report["simulation_validation"] == "not_run"
    package = tmp_path / "run/package"
    manifest = verify_files(package)
    assert set(manifest["formats"]) == {"urdf", "mjcf", "usd"}
    physics = json.loads((package / "part_physics.json").read_text())
    assert physics["total_mass"] == pytest.approx(0.1)
    assert physics["n_collision_hulls"] >= 1
    result = validate(package, tmp_path / "verify", steps=12)
    assert result["checks"]["step"] == "passed"
    assert result["simulated_seconds"] > 0
    assert result["settle"] == "not_run"
    with pytest.raises(FileExistsError):
        generate(request, runtime, tmp_path / "run")
    (package / "part_physics.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify_files(package)


@pytest.mark.parametrize("invalid", ["../escape", "x/y", "", "..", "x\\y"])
def test_reject_unsafe_identity(tmp_path, invalid):
    request = supplied_asset(tmp_path)
    request["asset_id"] = invalid
    with pytest.raises(ValueError):
        AssetRequest.from_dict(request)


def test_preflight_missing_environment_does_not_create_output(tmp_path):
    request = supplied_asset(tmp_path)
    with pytest.raises(ValueError):
        generate(request, Runtime(), tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_annotations_fail_before_collision_when_parts_missing(tmp_path):
    from rfw_w.assets.annotation.parts import validate as validate_annotation

    with pytest.raises(ValueError, match="every"):
        validate_annotation({"object": {"dimension_cm": [1, 2, 3]}, "parts": []}, [0])


def test_density_table_preserves_production_materials():
    from rfw_w.assets.physics.materials import density_of

    assert density_of("metal", "lead")[0] == 11340
    assert density_of("metal", "magnesium alloy")[0] == 1800


def test_mass_anchor_scales_inertia_preserves_com():
    from rfw_w.assets.physics.estimate import anchor

    data = {
        "total_mass": 1.0,
        "composed_mass": 1.0,
        "com": [0.1, 0.2, 0.3],
        "inertia": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "parts": {"0": {"mu1": 0.7, "mu2": 0.5}},
    }
    out = anchor(data, {0: 1.0}, 0.1, "anchor_vlm")
    assert out["com"] == [0.1, 0.2, 0.3]
    assert out["inertia"][0][0] == pytest.approx(0.1)


def test_core_import_has_no_model_dependencies(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(tmp_path / "invalid-source-path")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import rfw_w.assets.application, rfw_w.assets.cli, rfw_w.assets.runtime; import sys; assert "torch" not in sys.modules; assert "numpy" not in sys.modules',
        ],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_validation_requests_create_no_output(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        validate(tmp_path / "missing", tmp_path / "out", steps=0)
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="supports"):
        validate(tmp_path / "missing", tmp_path / "out", backend="unknown")
    assert not (tmp_path / "out").exists()


def test_package_rejects_unhashed_external_format(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "physics.json").write_text("{}")
    import hashlib

    digest = hashlib.sha256(b"{}").hexdigest()
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "rfw.asset-package/v1",
                "physics": "physics.json",
                "files": {"physics.json": digest},
                "formats": {"urdf": "../outside.urdf"},
            }
        )
    )
    with pytest.raises(ValueError, match="hashed"):
        verify_files(package)


@pytest.mark.parametrize(
    "config",
    [
        {"allow_remote_models": "false"},
        {"timeout_s": 0},
        {"timeout_s": float("nan")},
        {"stages": {"physical": {"env": {"PYTHONPATH": "bad"}}}},
        {"stages": {"physical": {"env": {"API_KEY": "bad"}}}},
    ],
)
def test_invalid_runtime_configuration(config, tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        Runtime.from_file(path)


@pytest.mark.parametrize("alteration", ["translate", "scale"])
def test_segmentation_frame_mismatch_is_rejected(alteration):
    import numpy as np, trimesh
    from rfw_w.assets.geometry.prepare import prepare_pair

    original = trimesh.creation.box()
    segmented = original.copy()
    if alteration == "translate":
        segmented.apply_translation([0.2, 0, 0])
    else:
        segmented.apply_scale(0.7)
    with pytest.raises(ValueError, match="source frame"):
        prepare_pair(original, segmented, np.zeros(12, dtype=int), [0.1, 0.1, 0.1], "Y")


def test_single_frame_transform_matches_visual_and_segmentation():
    import numpy as np, trimesh
    from rfw_w.assets.geometry.prepare import prepare_pair

    mesh = trimesh.creation.box(extents=[2, 3, 4])
    mesh.apply_translation([1, 0.5, 0])
    visual, segmented, frame = prepare_pair(
        mesh, mesh.copy(), np.zeros(12, dtype=int), [0.4, 0.3, 0.2], "Y"
    )
    assert np.allclose(visual.vertices, segmented.vertices)
    assert frame["scale"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "change",
    [
        {"asset_id": "../escape"},
        {"class_id": "../escape"},
        {"input_kind": "unknown"},
        {"input_value": ""},
        {"seed": True},
        {"seed": -1},
        {"source_up_axis": "X"},
        {"formats": "urdf"},
        {"formats": ("urdf", "urdf")},
        {"formats": ("unknown",)},
        {"mass_policy": "unknown"},
        {"category": None},
        {"annotations": "/missing/annotations.json"},
        {"face_ids": "/missing/face_ids.npy"},
        {"input_kind": "mesh", "input_value": "/missing/mesh.glb"},
    ],
)
def test_direct_request_constructor_validates_all_fields(change, tmp_path):
    kwargs = {
        "asset_id": "cube_0001",
        "class_id": "cube",
        "input_kind": "prompt",
        "input_value": "A cube",
    }
    kwargs.update(change)
    with pytest.raises((ValueError, FileNotFoundError)):
        request = AssetRequest(**kwargs)
        generate(request, Runtime(), tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_unknown_request_fields_fail_before_output(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        generate(
            {
                "asset_id": "cube",
                "class_id": "cube",
                "input": {"prompt": "cube"},
                "unsupported": True,
            },
            Runtime(),
            tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


def test_preflight_dependencies_and_requested_usd_before_output(tmp_path):
    import venv

    environment = tmp_path / "core-env"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    # Minimal installed worker package simulates a core-only installation.
    probe = subprocess.run(
        [str(python), "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        capture_output=True,
        text=True,
        check=True,
    )
    package = Path(probe.stdout.strip()) / "rfw_w" / "assets"
    (package / "workers").mkdir(parents=True)
    for path in (
        package.parent / "__init__.py",
        package / "__init__.py",
        package / "workers/__init__.py",
        package / "workers/physical.py",
    ):
        path.write_text("")
    runtime = Runtime({"physical": {"python": str(python)}})
    request = supplied_asset(tmp_path)
    with pytest.raises(RuntimeError, match="coacd") as error:
        generate(request, runtime, tmp_path / "run")
    assert "pxr" in str(error.value)
    assert not (tmp_path / "run").exists()


def test_preflight_usd_dependency_is_format_specific(tmp_path):
    import venv

    environment = tmp_path / "core-env"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    result = subprocess.run(
        [str(python), "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        capture_output=True,
        text=True,
        check=True,
    )
    site = Path(result.stdout.strip())
    package = site / "rfw_w" / "assets"
    (package / "workers").mkdir(parents=True)
    for path in (
        package.parent / "__init__.py",
        package / "__init__.py",
        package / "workers/__init__.py",
        package / "workers/physical.py",
    ):
        path.write_text("")
    for name in ("numpy", "scipy", "trimesh", "coacd", "psutil", "PIL"):
        (site / (name + ".py")).write_text(
            'raise AssertionError("preflight must not import dependency")'
        )
    runtime = Runtime({"physical": {"python": str(python)}})
    runtime.preflight(["physical"], ["urdf", "mjcf"])
    with pytest.raises(RuntimeError, match="pxr"):
        runtime.preflight(["physical"], ["usd"])
