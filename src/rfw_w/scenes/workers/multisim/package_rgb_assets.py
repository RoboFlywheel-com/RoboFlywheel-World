#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Copy five native simulator RGB captures into one traceable asset directory.

The images are copied byte for byte. This entry never assembles a figure or
changes exposure, gamma, colour, crop or resolution.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from rfw_w.scenes.workers.multisim.common import CommonScene, digest, write_json


BACKENDS = ("genesis", "isaac", "mujoco", "pybullet", "sapien")


def package(scene_dir: Path, renders_dir: Path, mujoco_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"OUTPUT_DIR_NOT_FRESH({output_dir})")
    scene = CommonScene.load(scene_dir)
    camera = scene.camera
    sources = {name: renders_dir / name for name in BACKENDS}
    sources["mujoco"] = mujoco_dir

    verified: dict[str, dict] = {}
    for name, directory in sources.items():
        rgb_path = directory / "rgb.png"
        report_path = directory / "report.json"
        if not rgb_path.is_file() or not report_path.is_file():
            raise ValueError(f"MULTISIM_PROVENANCE_MISSING({name}: rgb/report)")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            raise ValueError(f"MULTISIM_BACKEND_UNAVAILABLE({name}: {report.get('status')})")
        if report.get("scene_version") != scene.scene_version:
            raise ValueError(f"MULTISIM_SCENE_VERSION_MISMATCH({name})")
        if report.get("backend") != name or report.get("object_ids") != scene.instance_ids:
            raise ValueError(f"MULTISIM_GEOMETRY_MISMATCH({name}: object map)")
        if report.get("camera") != camera.as_record():
            raise ValueError(f"MULTISIM_CAMERA_MISMATCH({name})")
        with Image.open(rgb_path) as image:
            if image.mode != "RGB" or image.size != (camera.width, camera.height):
                raise ValueError(f"MULTISIM_RGB_INVALID({name}: {image.mode} {image.size})")
            pixels = np.asarray(image)
            chroma = np.abs(pixels.astype(np.int16) - pixels[..., :1].astype(np.int16))
        verified[name] = {
            "source_rgb": str(rgb_path.resolve()),
            "source_report": str(report_path.resolve()),
            "rgb_sha256": digest(rgb_path),
            "report_sha256": digest(report_path),
            "engine_version": report.get("engine_version"),
            "mean_channel_difference": float(chroma.mean()),
        }
    if verified["mujoco"]["mean_channel_difference"] <= 2.0:
        raise ValueError("MULTISIM_RGB_MONOCHROME(mujoco: material mapping did not render colour)")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir()
    for name, record in verified.items():
        target = output_dir / f"{name}_rgb.png"
        shutil.copyfile(record["source_rgb"], target)
        if digest(target) != record["rgb_sha256"]:
            raise ValueError(f"MULTISIM_RGB_COPY_MISMATCH({name})")
        record["asset"] = target.name
        report_target = reports_dir / f"{name}.json"
        shutil.copyfile(record["source_report"], report_target)
        if digest(report_target) != record["report_sha256"]:
            raise ValueError(f"MULTISIM_REPORT_COPY_MISMATCH({name})")
        record["report_asset"] = str(report_target.relative_to(output_dir))
    shutil.copyfile(scene_dir / "camera.json", output_dir / "camera.json")
    provenance = {
        "claim": "five simulator native RGB renders of one frozen static scene and camera",
        "scene_version": scene.scene_version,
        "camera_sha256": digest(scene_dir / "camera.json"),
        "resolution_px": [camera.width, camera.height],
        "image_processing": "none; raw PNG copied byte for byte",
        "backends": verified,
    }
    write_json(output_dir / "provenance.json", provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--mujoco", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    record = package(args.scene, args.renders, args.mujoco, args.output_dir)
    for name in BACKENDS:
        print(name, record["backends"][name]["asset"],
              record["backends"][name]["rgb_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
