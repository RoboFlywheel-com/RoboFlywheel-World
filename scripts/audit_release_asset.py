# SPDX-License-Identifier: Apache-2.0
"""Audit a v1.3 release leaf without modifying the release library.

Use --release-root or RFW_RELEASE_ROOT, --key and a fresh --out directory.
Optional --mujoco-python compiles and steps a copied MJCF in a clean process.
This checks loading and stepping only; it does not certify dynamic stability.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET


def inspect_asset(root, key):
    if not re.fullmatch(r"\d+_\d+_\d{4}", key):
        raise ValueError("invalid release leaf; expected <domain>_<class>_<NNNN>")
    domain, category, _ = key.split("_")
    leaf = root / "assets" / domain / category / key
    if not leaf.is_dir():
        raise FileNotFoundError("release leaf is missing")
    required = ["_SUCCESS", "part_physics.json", f"m_{key}.urdf",
                "coacd_multi.urdf", f"{key}.usd", "mesh", "mjcf", "affordance"]
    missing = [name for name in required if not (leaf / name).exists()]
    if missing:
        raise ValueError(f"missing release files: {missing}")
    with (leaf / "part_physics.json").open() as stream:
        json.load(stream)
    documents = [leaf / f"m_{key}.urdf", leaf / "coacd_multi.urdf"]
    mjcf = leaf / "mjcf" / f"m_{key}.xml"
    if not mjcf.is_file():
        raise FileNotFoundError("canonical MJCF is missing")
    documents.append(mjcf)
    results = []
    for document in documents:
        tree = ET.parse(document).getroot()
        compiler = tree.find("compiler")
        refs = []
        for element in tree.iter():
            ref = element.get("filename") if element.tag == "mesh" else None
            if document == mjcf and element.tag in ("mesh", "texture", "include", "hfield"):
                ref = element.get("file")
            if not ref:
                continue
            directory = ""
            if compiler is not None and element.tag in ("mesh", "texture"):
                directory = compiler.get("assetdir", "")
                directory = compiler.get("meshdir" if element.tag == "mesh" else "texturedir", directory)
            candidate = (document.parent / directory / ref).resolve()
            if Path(ref).is_absolute() or not candidate.is_relative_to(leaf.resolve()):
                raise ValueError(f"nonportable reference in {document.name}")
            if not candidate.is_file():
                raise FileNotFoundError(f"missing dependency in {document.name}: {ref}")
            refs.append(ref)
        results.append({"file": str(document.relative_to(leaf)), "references": len(refs),
                        "free_joints": len(tree.findall(".//freejoint")) +
                        len(tree.findall(".//joint[@type='free']"))})
    return leaf, {"key": key, "layout": "v1.3", "static_references": "passed",
                  "documents": results, "physics_schema": "not_run",
                  "usd_dependencies": "not_run", "stability": "not_run"}


MUJOCO_CHECK = r'''
import json, sys
try:
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_path(sys.argv[1])
    data = mujoco.MjData(model)
    for _ in range(120):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("nonfinite state")
    if data.time <= 0:
        raise ValueError("physics clock did not advance")
    result = {"status": "passed", "version": mujoco.__version__,
              "steps": 120, "time_s": data.time, "bodies": model.nbody,
              "joints": model.njnt, "stability": "not_run"}
except Exception as exc:
    result = {"status": "failed", "error_type": type(exc).__name__}
with open(sys.argv[2], "w") as stream:
    json.dump(result, stream, indent=2)
sys.exit(0 if result["status"] == "passed" else 1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", default=os.environ.get("RFW_RELEASE_ROOT"))
    parser.add_argument("--key", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mujoco-python")
    args = parser.parse_args()
    if not args.release_root:
        parser.error("--release-root or RFW_RELEASE_ROOT is required")
    release = Path(args.release_root).resolve()
    output = args.out.resolve()
    if output.is_relative_to(release) or release.is_relative_to(output):
        parser.error("output must be separate from the release root")
    output.mkdir(parents=True, exist_ok=False)
    report = {"key": args.key, "status": "failed"}
    exit_code = 1
    try:
        leaf, report = inspect_asset(release, args.key)
        copied_root = output / "relocated"
        copied = copied_root / "assets" / Path(*args.key.split("_")[:2]) / args.key
        shutil.copytree(leaf, copied, symlinks=False)
        # Resolve dependencies again from the new root.
        inspect_asset(copied_root, args.key)
        report["relocated_static_references"] = "passed"
        report["mujoco"] = {"status": "not_run"}
        if args.mujoco_python:
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            result_file = output / "mujoco.json"
            result = subprocess.run(
                [args.mujoco_python, "-c", MUJOCO_CHECK,
                 str(copied / "mjcf" / f"m_{args.key}.xml"), str(result_file)],
                cwd=output, env=env, capture_output=True, timeout=120)
            report["mujoco"] = (json.loads(result_file.read_text()) if result_file.exists()
                                 else {"status": "failed", "error_type": "runtime_startup"})
            if result.returncode:
                raise RuntimeError("MuJoCo load/step failed")
        report["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
