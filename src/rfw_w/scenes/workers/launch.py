#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive the v3 Isaac worker from the orchestrator side.

Two modes:

    # a runtime smoke test: a real room, two real retrieved objects, one camera
    python -m rfw_w.scenes.workers.launch --smoke --output-dir outputs/_smoke/isaac_worker_smoke --env-file .env.local

    # a request built elsewhere
    python -m rfw_w.scenes.workers.launch --request <request.json> --output-dir <new dir>

The orchestrator never imports Isaac: it writes a request, runs the worker under Isaac's
interpreter as a subprocess, and validates the report it gets back. Isaac loads a large runtime
with global state, and keeping it out of the orchestrator's process is what lets the deterministic
core stay testable.

A run that produces no report is a failure even if the process exited zero, because that is exactly
what Kit's shutdown path can produce.
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path.cwd()

from rfw_w.scenes.workers.isaac.protocol import (
    CameraRequest,
    MeshInstance,
    ProtocolError,
    WorkerReport,
    WorkerRequest,
)


def _env(name: str, default: str | None = None) -> str | None:
    import os

    return os.environ.get(name) or default


def load_env_file(path: Path) -> None:
    import os

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_smoke_request() -> WorkerRequest:
    """A runtime smoke test: the real room, two real retrieved mugs, floor-resting.

    The objects are genuinely retrieved, so this exercises the mesh authoring, the Y-up
    correction and the physics on real geometry rather than on primitives. It is still only a
    runtime check: it certifies nothing about scene quality, and the report says so.
    """
    from rfw_w.scenes.adapters.simready_retrieval import (
        SimreadyConfig,
        build_encoder,
        build_engine,
        retrieve_entity,
        to_asset_record,
    )
    from rfw_w.scenes.contracts.scene_spec import SceneSpec
    from rfw_w.scenes.solvers.room import build_shell, generate_options

    spec = SceneSpec.from_dict(
        json.loads((data_path('examples') / 'office_pick_cup' / 'task.json').read_text())
    )
    options = generate_options(spec)
    shell = build_shell(spec, options=options)

    lib = _env("PVSCENE_SIMREADY_LIB")
    index_path = _env("PVSCENE_SIMREADY_INDEX") or (str(Path(lib) / "index.json") if lib else None)
    assets_root = _env("PVSCENE_SIMREADY_ASSETS_ROOT")
    semantic_index = _env("PVSCENE_SIMREADY_SEMANTIC_INDEX")
    prepared = _env("PVSCENE_PREPARED_DIR")
    missing = [
        name
        for name, value in (
            ("PVSCENE_SIMREADY_INDEX", index_path),
            ("PVSCENE_SIMREADY_ASSETS_ROOT", assets_root),
            ("PVSCENE_SIMREADY_SEMANTIC_INDEX", semantic_index),
            ("PVSCENE_PREPARED_DIR", prepared),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "the smoke test needs the real libraries; missing "
            + ", ".join(missing)
            + " (run with --env-file .env.local)"
        )

    config = SimreadyConfig(
        index_path=Path(index_path),
        assets_root=Path(assets_root),
        semantic_index_path=Path(semantic_index),
        library_version=os.environ.get("PVSCENE_SIMREADY_LIBRARY_VERSION", "v1.3"),
        prepared_root=Path(_env("PVSCENE_PREPARED_ASSETS_DIR") or "prepared/smoke-assets"),
    )
    engine = build_engine(
        config, encoder=build_encoder(Path(prepared))
    )
    cup = spec.entity("cup")
    outcome = retrieve_entity(
        engine, cup, config=config, limit=4, admitted_categories=("Mug", "Coffee Cup", "Cup")
    )
    if not outcome.ok:
        raise SystemExit(f"smoke test could not retrieve a cup: {outcome.detail}")

    # Two distinct candidates so the instance-id map has more than one entry to check.
    instances: list[MeshInstance] = []
    for index, candidate in enumerate(outcome.candidates[:2]):
        record, handle = to_asset_record(candidate, config)
        x = 1.2 + index * 0.5
        instances.append(
            MeshInstance(
                instance_id=f"cup_{index}",
                entity_id=f"cup_{index}",
                visual_uri=handle.visual_uri,
                collision_uri=handle.collision_uris[0],
                # Resting on the floor: the mesh origin is the bounding-box centre, so the centre
                # goes half its height up.
                translation=(x, 1.5, shell.floor_z_m + record.dimensions_m.z / 2.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                mass_kg=record.rigid_physics.mass_kg,
                dynamic=True,
                normalisation="y_up_to_z_up",
            )
        )

    # Inside the room, looking at the objects. A camera outside would render the outside of a
    # wall: the image would be non-blank and the depth valid, so every "did it render" check would
    # pass while showing nothing the scene is about.
    center_x = sum(x for x, _ in shell.polygon_xy) / len(shell.polygon_xy)
    center_y = sum(y for _, y in shell.polygon_xy) / len(shell.polygon_xy)
    cup_x = sum(instance.translation[0] for instance in instances) / len(instances)
    cup_y = sum(instance.translation[1] for instance in instances) / len(instances)
    camera = CameraRequest(
        camera_id="overview",
        purpose="layout",
        eye=(cup_x + 0.9, cup_y - 1.0, 1.5),
        target=(cup_x, cup_y, 0.12),
        resolution=(960, 640),
        target_instance_ids=("cup_0", "cup_1"),
        require_target_visible=True,
        save_depth=True,
        save_instance_ids=True,
    )

    from rfw_w.scenes.workers.isaac.protocol import RoomGeometry

    return WorkerRequest(
        scene_id="isaac_worker_smoke",
        scene_version="0" * 64,
        room=RoomGeometry(polygon_xy=shell.polygon_xy, height_m=shell.height_m, floor_z_m=shell.floor_z_m),
        instances=tuple(instances),
        cameras=(camera,),
        simulate_seconds=2.0,
        physics_hz=120,
    )


def invoke(request_path: Path, output_dir: Path, *, isaac_python: str, active_gpu: int, timeout_s: int) -> tuple[int, WorkerReport | None]:
    """Run the worker and read its report back."""
    out = output_dir.resolve()
    if out.exists():
        raise SystemExit(f"output directory already exists; a stale report must not look current: {out}")

    command = [
        isaac_python,
        "-m", "rfw_w.scenes.workers.isaac.render_worker",
        "--request",
        str(request_path),
        "--output-dir",
        str(out),
        "--active-gpu", str(active_gpu),
    ]
    print(f"running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    if completed.stdout:
        print(completed.stdout[-4000:])
    if completed.stderr:
        print(completed.stderr[-4000:], file=sys.stderr)

    report_path = out / "report.json"
    if not report_path.is_file():
        print(
            f"FAIL: the worker exited {completed.returncode} without writing a report; a killed "
            "process or a crashed interpreter is a failure, not a silent success",
            file=sys.stderr,
        )
        return completed.returncode or 1, None

    report = WorkerReport.from_dict(json.loads(report_path.read_text()))
    try:
        report.assert_self_consistent()
    except ProtocolError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1, report
    return completed.returncode, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--build-only", action="store_true",
                        help="write the request without running Isaac, for inspection")
    args = parser.parse_args()

    if args.env_file:
        if not args.env_file.is_file():
            raise SystemExit(f"env file not found: {args.env_file}")
        load_env_file(args.env_file)

    # Only the request directory is created here. The worker's output directory is deliberately
    # left absent so that invoke() can refuse to run into an existing one: a stale report must not
    # be able to look like this run's result.
    if args.build_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.request:
        request_path = args.request
    elif args.smoke:
        request = build_smoke_request()
        request_path = args.output_dir / "request.json"
        request_path.write_text(json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"wrote request to {request_path}")
    else:
        raise SystemExit("either --request or --smoke is required")

    if args.build_only:
        return 0

    isaac_python = _env("PVSCENE_ISAAC_PYTHON", "/isaac-sim/python.sh")
    started = time.monotonic()
    code, report = invoke(
        request_path, args.output_dir, isaac_python=isaac_python,
        active_gpu=args.active_gpu, timeout_s=args.timeout,
    )
    elapsed = round(time.monotonic() - started, 2)

    if report is None:
        return 1
    summary = {
        "status": report.status,
        "elapsed_seconds": elapsed,
        "isaac_version": report.isaac_version,
        "captures": [
            {
                "camera_id": capture.camera_id,
                "nonblank": capture.nonblank,
                "color_std": round(capture.color_std, 3),
                "depth_valid_fraction": round(capture.depth_valid_fraction, 3),
                "visible_instance_ids": list(capture.visible_instance_ids),
                "target_visible": capture.target_visible,
            }
            for capture in report.captures
        ],
        "observed_label_payload_kind": report.observed_label_payload_kind,
        "settle_performed": report.settle.get("performed"),
        "error": report.error_detail or None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if report.passed and code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
