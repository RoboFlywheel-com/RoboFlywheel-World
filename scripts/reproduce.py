# SPDX-License-Identifier: Apache-2.0
"""Run installed asset and scene pipelines, recording each independent verdict.

Local JSON supplies interpreters and optional real inputs. Output is always new;
missing engines remain blocked. Logs/configuration are local, never publication data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(config, out):
    out = Path(out).resolve()
    release = Path(config["release_root"]).resolve() if config.get("release_root") else None
    if release and (out.is_relative_to(release) or release.is_relative_to(out)):
        raise ValueError("output must be separate from the read-only release")
    if out.exists():
        raise FileExistsError("reproduction output must be new")
    python = config["geometry_python"]
    if not Path(python).is_file():
        raise FileNotFoundError("geometry interpreter unavailable")
    out.mkdir(parents=True)
    summary = {"schema": "rfw.reproduction/v1", "checks": {},
               "model_chain": "not_run", "full_reproduction": False}
    checks = summary["checks"]

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def command(name, interpreter, args, environment=None, timeout=3600):
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["RFW_ASSET_PYTHON"] = interpreter
        env["RFW_SCENE_PYTHON"] = interpreter
        for key in environment or {}:
            if key in ("PYTHONPATH", "PYTHONHOME") or any(
                word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIAL")
            ):
                raise ValueError("backend env cannot inject package paths or credential values")
        env.update(environment or {})
        if not Path(interpreter).is_file():
            checks[name] = {"status": "blocked", "reason": "interpreter_unavailable"}
            save()
            return None
        with (out / (name + ".log")).open("w") as log:
            try:
                result = subprocess.run([interpreter, *map(str, args)], cwd=out,
                                        env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=timeout)
                checks[name] = {"status": "passed" if result.returncode == 0 else "failed",
                                "exit_code": result.returncode}
            except subprocess.TimeoutExpired:
                checks[name] = {"status": "failed", "reason": "timeout"}
        save()
        return checks[name].get("exit_code")

    def probes(name, root, urdf, mjcf, physics):
        for backend in ("mujoco", "pybullet", "sapien", "genesis", "isaac"):
            entry = name + "_" + backend
            interpreter = config.get("backends", {}).get(backend)
            if not interpreter:
                checks[entry] = {"status": "blocked", "reason": "backend_not_configured"}
                continue
            if backend == "isaac":
                checks[entry] = {"status": "not_run", "reason": "use_scene_worker_for_isaac"}
                continue
            destination = out / entry
            command(entry, interpreter, ["-m", "rfw_w.assets.simulation.probe",
                    "--urdf", root / urdf, "--mjcf", root / mjcf,
                    "--physics", root / physics, "--backend", backend,
                    "--steps", config.get("steps", 240), "--out", destination],
                    config.get("backend_env", {}).get(backend), timeout=1200)
            if (destination / "report.json").exists():
                checks[entry]["evidence"] = json.loads((destination / "report.json").read_text())
        save()

    if command("synthetic_input", python, ["-m", "rfw_w.assets.examples.synthetic_mesh",
                                           "--out", out / "input"]) != 0:
        return summary
    if command("asset_generate", python, ["-m", "rfw_w", "asset", "generate",
            "--request", out / "input/request.json", "--config", out / "input/runtime.json",
            "--out", out / "asset"]) != 0:
        return summary
    package = out / "asset/package"
    manifest = json.loads((package / "manifest.json").read_text())
    command("asset_relocate", python, ["-m", "rfw_w", "asset", "validate",
            "--bundle", package, "--out", out / "asset_verify",
            "--steps", config.get("steps", 240)])
    copied = out / "asset_verify/relocated"
    if copied.is_dir():
        probes("asset", copied, manifest["formats"]["urdf"],
               manifest["formats"]["mjcf"], manifest["physics"])
    (out / "bindings.json").write_text(json.dumps({"block": "asset/package"}) + "\n")
    base = ["-m", "rfw_w", "scene", "generate", "--experiment", "synthetic_asset",
            "--asset-bindings", out / "bindings.json", "--asset-license", "CC0-1.0",
            "--no-loop", "--no-replan"]
    code = command("scene_generate", python, [*base, "--output-dir", out / "scene", "--skip-isaac"])
    if (out / "scene/delivery_manifest.json").is_file():
        checks["scene_generate"]["status"] = "partial" if code == 1 else checks["scene_generate"]["status"]
        checks["scene_generate"]["reason"] = "readiness_requires_backend_evidence"
        command("scene_validate", python, ["-m", "rfw_w", "scene", "validate",
                                            "--delivery", out / "scene"])
    if command("scene_request", python, [*base, "--output-dir", out / "scene_request",
                                         "--request-only"]) == 0:
        command("scene_mujoco", python, ["-m", "rfw_w.scenes.validation.probe_request",
                "--request", out / "scene_request/isaac_request.json", "--out", out / "scene_mujoco",
                "--steps", config.get("steps", 240)])
        report = out / "scene_mujoco/report.json"
        if report.exists():
            checks["scene_mujoco"]["evidence"] = json.loads(report.read_text())

    if config.get("asset_request") and config.get("asset_runtime"):
        real_code = command("real_asset_generate", python, ["-m", "rfw_w", "asset", "generate",
                "--request", config["asset_request"], "--config", config["asset_runtime"],
                "--out", out / "real_asset"])
        if real_code == 0:
            real_package = out / "real_asset/package"
            real_manifest = json.loads((real_package / "manifest.json").read_text())
            command("real_asset_relocate", python, ["-m", "rfw_w", "asset", "validate",
                    "--bundle", real_package, "--out", out / "real_asset_verify",
                    "--steps", config.get("steps", 240)])
            real_copied = out / "real_asset_verify/relocated"
            if real_copied.is_dir():
                probes("real_asset", real_copied, real_manifest["formats"]["urdf"],
                       real_manifest["formats"]["mjcf"], real_manifest["physics"])
        if real_code == 0 and config.get("scene_spec"):
            (out / "real_bindings.json").write_text(json.dumps(
                {config.get("scene_entity", "block"): "real_asset/package"}) + "\n")
            if command("real_scene_request", python, ["-m", "rfw_w", "scene", "generate",
                    "--spec", config["scene_spec"], "--asset-bindings", out / "real_bindings.json",
                    "--asset-license", config.get("asset_license", "unrecorded"),
                    "--output-dir", out / "real_scene_request", "--request-only", "--no-replan"]) == 0:
                command("real_scene_mujoco", python, ["-m", "rfw_w.scenes.validation.probe_request",
                        "--request", out / "real_scene_request/isaac_request.json",
                        "--out", out / "real_scene_mujoco", "--steps", config.get("steps", 240)])
    if release:
        audit = Path(__file__).with_name("audit_release_asset.py")
        for key in config.get("release_keys", []):
            destination = out / ("release_" + key)
            if command("release_" + key, python, [audit, "--release-root", release,
                    "--key", key, "--out", destination, "--mujoco-python", python]) == 0:
                root = destination / "relocated/assets" / Path(*key.split("_")[:2]) / key
                probes("release_" + key, root, f"m_{key}.urdf", f"mjcf/m_{key}.xml",
                       "part_physics.json")
    save()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = run(json.loads(args.config.read_text()), args.out)
    print(json.dumps({name: item["status"] for name, item in summary["checks"].items()}, indent=2))
    # Partial or blocked coverage must remain visible to batch callers.
    return 0 if all(item["status"] == "passed" for item in summary["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
