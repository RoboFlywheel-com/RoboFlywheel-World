# SPDX-License-Identifier: Apache-2.0
"""Independent portable-package validation in fresh output directories."""

from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import xml.etree.ElementTree as ET


def verify_files(package):
    package = Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest.get("schema") != "rfw.asset-package/v1":
        raise ValueError("unsupported package schema")
    files = manifest.get("files")
    formats = manifest.get("formats")
    if (
        not isinstance(files, dict)
        or not isinstance(formats, dict)
        or not formats
        or set(formats) - {"urdf", "mjcf", "usd"}
    ):
        raise ValueError("invalid package file or format contract")

    def safe(name):
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("unsafe package reference")
        path = (package / name).resolve()
        if not path.is_relative_to(package) or not path.is_file():
            raise ValueError("unsafe or missing package file")
        return path

    for name, digest in files.items():
        path = safe(name)
        if (
            not isinstance(digest, str)
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("package checksum mismatch")
    physics = manifest.get("physics")
    if physics not in files:
        raise ValueError("physics must be a hashed package file")
    safe(physics)
    for kind, name in formats.items():
        if name not in files:
            raise ValueError("format must be a hashed package file")
        path = safe(name)
        if kind in ("urdf", "mjcf"):
            for element in ET.parse(path).iter():
                for attribute in ("file", "filename"):
                    ref = element.get(attribute)
                    if ref:
                        target = (path.parent / ref).resolve()
                        if (
                            Path(ref).is_absolute()
                            or not target.is_relative_to(package)
                            or not target.exists()
                            or str(target.relative_to(package)) not in files
                        ):
                            raise ValueError("invalid format resource reference")
    return manifest


def validate(package, out, backend="mujoco", steps=240):
    if backend != "mujoco":
        raise ValueError("independent dynamic validation currently supports mujoco")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    package = Path(package).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("validation output must be new")
    if out.is_relative_to(package):
        raise ValueError("validation output cannot be inside package")
    manifest = verify_files(package)
    out.mkdir(parents=True)
    relocated = out / "relocated"
    shutil.copytree(package, relocated)
    verify_files(relocated)
    report = {
        "schema": "rfw.asset-validation/v1",
        "backend": backend,
        "checks": {"files": "passed", "relocate": "passed"},
        "task_run": "not_run",
    }
    try:
        import numpy as np, mujoco

        if "mjcf" not in manifest["formats"]:
            raise ValueError("MuJoCo requires requested MJCF export")
        physical = json.loads((relocated / manifest["physics"]).read_text())
        from rfw_w.assets.physics.estimate import validate_physics

        validate_physics(physical)
        model = mujoco.MjModel.from_xml_path(
            str(relocated / manifest["formats"]["mjcf"])
        )
        data = mujoco.MjData(model)
        bodies = [i for i in range(1, model.nbody) if model.body_mass[i] > 0]
        if len(bodies) != 1 or model.nq < 7:
            raise ValueError("expected one dynamic free rigid body")
        body = bodies[0]
        if not np.isclose(model.body_mass[body], physical["total_mass"], rtol=1e-6):
            raise ValueError("simulator mass differs from sidecar")
        if not np.allclose(model.body_ipos[body], physical["com"], atol=1e-6):
            raise ValueError("simulator COM differs from sidecar")
        if not np.allclose(
            np.sort(model.body_inertia[body]),
            np.linalg.eigvalsh(physical["inertia"]),
            rtol=1e-5,
            atol=1e-9,
        ):
            raise ValueError("simulator inertia differs from sidecar")
        report["checks"]["load"] = "passed"
        report["checks"]["mass"] = "passed"
        report["checks"]["com"] = "passed"
        report["checks"]["inertia"] = "passed"
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or data.time <= 0
        ):
            raise ValueError("invalid simulation state")
        report.update(
            status="passed",
            steps=steps,
            simulated_seconds=data.time,
            settle="not_run",
            render="not_run",
        )
        report["checks"]["step"] = "passed"
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
