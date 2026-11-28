# SPDX-License-Identifier: Apache-2.0
"""Create explicitly synthetic mesh inputs without model calls or release data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def create(out: str | Path, python: str = sys.executable) -> dict[str, str]:
    """Write a fresh, self-contained fixture for the actual physical worker."""
    out = Path(out).expanduser().resolve()
    if out.exists():
        raise FileExistsError("synthetic input directory must be fresh")
    interpreter = shutil.which(python)
    if not interpreter or not Path(interpreter).is_file():
        raise FileNotFoundError("physical interpreter is unavailable")
    # Keep a virtual environment's interpreter symlink; resolving it loses the venv.
    interpreter = os.path.abspath(interpreter)

    import numpy as np
    import trimesh

    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    mesh.visual.face_colors = [220, 40, 30, 255]
    out.mkdir(parents=True)
    mesh.export(out / "synthetic_cube.glb")
    np.save(out / "face_ids.npy", np.zeros(len(mesh.faces), dtype=np.int64))

    annotations = {
        "object": {"dimension_cm": [10, 10, 10], "mass_g": 100},
        "parts": [
            {
                "mask_color": "Red",
                "name": "synthetic_body",
                "material": {"category": "plastic", "subtype": "ABS"},
                "static_friction": 0.7,
                "dynamic_friction": 0.5,
            }
        ],
    }
    request = {
        "asset_id": "synthetic_cube_0001",
        "class_id": "synthetic_cube",
        "category": "synthetic cube",
        "input": {"mesh": str(out / "synthetic_cube.glb")},
        "annotations": str(out / "annotations.json"),
        "face_ids": str(out / "face_ids.npy"),
        "formats": ["urdf", "mjcf", "usd"],
        "source_up_axis": "Z",
        "seed": 2025,
    }
    runtime = {
        "stages": {
            "physical": {
                "python": interpreter,
                "options": {"voxel-resolution": 16},
            }
        },
        "allow_remote_models": False,
        "timeout_s": 3600,
    }
    provenance = {
        "synthetic": True,
        "description": "A 10 cm cube with one synthetic part and assumed material, mass and friction.",
        "source_up_axis": "Z",
        "real_semantic_validation": "not_run",
        "visual_quality": "not_run",
        "task_validation": "not_run",
        "model_calls": "not_run",
    }
    for filename, data in (
        ("annotations.json", annotations),
        ("request.json", request),
        ("runtime.json", runtime),
        ("fixture.json", provenance),
    ):
        (out / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {
        "input_kind": "synthetic_mesh",
        "request": str(out / "request.json"),
        "runtime": str(out / "runtime.json"),
        "provenance": str(out / "fixture.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="fresh synthetic input directory")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="installed physical-worker interpreter; defaults to the current Python",
    )
    args = parser.parse_args(argv)
    print(json.dumps(create(args.out, args.python), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
