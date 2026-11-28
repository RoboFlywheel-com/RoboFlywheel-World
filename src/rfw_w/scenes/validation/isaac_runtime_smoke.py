#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isaac runtime probe: a falling cube, support contact, RGB and metric depth.

This synthetic fixture certifies the runtime only, never scene quality, retrieval,
robot feasibility, or multi-simulator readiness. Run with /isaac-sim/python.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fixture", choices=("supported", "missing_support"), default="supported")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 2:
        parser.error("--seconds must be finite and >= 2")
    # A fresh directory prevents stale images/reports from appearing successful.
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "pvscene.runtime-probe/v1",
        "status": "failed",
        "scope": "synthetic_runtime_fixture_only",
        "fixture": args.fixture,
        "scene_quality": "not_run",
        "retrieval": "not_run",
        "robot_rollout": "not_run",
        "multisim": "not_run",
    }
    started = time.monotonic()
    app = None
    code = 1
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({
            "headless": True,
            "renderer": "RaytracedLighting",
            "active_gpu": args.active_gpu,
            "physics_gpu": args.active_gpu,
            "multi_gpu": False,
            "width": 640,
            "height": 480,
        })
        report["startup_seconds"] = time.monotonic() - started
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.version import get_version
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux

        report["isaac_version"] = list(get_version())
        report["depth_convention"] = "camera_z_meters"
        world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60, device="cpu")
        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        world.scene.add(FixedCuboid(
            prim_path="/World/floor", name="floor", size=1,
            position=np.array([0., 0., -0.05]), scale=np.array([4., 4., 0.1]),
            color=np.array([0.45, 0.47, 0.5]),
        ))
        if args.fixture == "supported":
            world.scene.add(FixedCuboid(
                prim_path="/World/support", name="support", size=1,
                position=np.array([0., 0., 0.70]), scale=np.array([1.2, 0.8, 0.1]),
                color=np.array([0.42, 0.24, 0.10]),
            ))
        cube = world.scene.add(DynamicCuboid(
            prim_path="/World/probe", name="probe", size=0.08, mass=0.2,
            position=np.array([0., 0., 0.99]), color=np.array([0.08, 0.36, 0.78]),
        ))
        light = UsdLux.DomeLight.Define(stage, "/World/light")
        light.CreateIntensityAttr(1000)
        camera = UsdGeom.Camera.Define(stage, "/World/camera")
        camera.CreateFocalLengthAttr(24)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
        eye, target = Gf.Vec3d(1.8, -2.2, 1.8), Gf.Vec3d(0, 0, 0.65)
        matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
        # Probe stage contains its runtime PhysicsScene; this is not a delivery package.
        stage.Export(str(out / "fixture.usda"))
        rep.orchestrator.set_capture_on_play(False)
        product = rep.create.render_product(str(camera.GetPath()), (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb.attach(product)
        depth.attach(product)

        def capture(name: str) -> dict:
            rep.orchestrator.step(rt_subframes=8, pause_timeline=True)
            pixels = np.asarray(rgb.get_data())
            meters = np.asarray(depth.get_data(), dtype=np.float32)
            if pixels.shape[:2] != (480, 640) or meters.shape != (480, 640):
                raise RuntimeError("RENDER_SHAPE_MISMATCH")
            Image.fromarray(pixels[..., :3].astype(np.uint8)).save(out / f"{name}.png")
            np.save(out / f"{name}_depth_m.npy", meters)
            valid = np.isfinite(meters) & (meters > 0) & (meters < 100)
            stats = {
                "rgb": f"{name}.png", "depth": f"{name}_depth_m.npy",
                "color_std": float(pixels[..., :3].astype(float).std(axis=(0, 1)).mean()),
                "valid_depth_fraction": float(valid.mean()),
            }
            stats["passed"] = stats["color_std"] > 5 and stats["valid_depth_fraction"] > 0.1
            return stats

        world.reset()
        world.pause()
        report["renders"] = [capture("before")]
        world.play()
        physics_elapsed = [0.0]
        def count_physics(dt: float) -> None:
            physics_elapsed[0] += float(dt)
        world.add_physics_callback("probe_clock", count_physics)
        samples = []
        sim_started = time.monotonic()
        for _ in range(math.ceil(args.seconds * 120)):
            world.step(render=False)
            position, orientation = cube.get_world_pose()
            linear, angular = cube.get_linear_velocity(), cube.get_angular_velocity()
            if not all(np.isfinite(x).all() for x in (position, orientation, linear, angular)):
                raise RuntimeError("NONFINITE_PHYSICS_STATE")
            samples.append(np.array(position, copy=True))
        final = samples[-1]
        tail = np.asarray(samples[-120:])
        drift = float(np.linalg.norm(tail - final, axis=1).max())
        report["physics"] = {
            "requested_sim_seconds": args.seconds,
            "observed_sim_seconds": physics_elapsed[0],
            "wall_seconds": time.monotonic() - sim_started,
            "initial_xyz_m": [0., 0., 0.99],
            "final_xyz_m": final.tolist(),
            "expected_support_center_z_m": 0.79,
            "last_second_drift_m": drift,
            "linear_speed_m_s": float(np.linalg.norm(linear)),
            "angular_speed_rad_s": float(np.linalg.norm(angular)),
            "checks": {
                "clock_advanced": physics_elapsed[0] >= args.seconds - 1 / 120 - 1e-6,
                "real_drop": float(0.99 - final[2]) > 0.05,
                "on_intended_support": abs(float(final[2]) - 0.79) < 0.01,
                "stable_tail": drift < 0.001,
                "low_velocity": float(np.linalg.norm(linear)) < 0.01 and float(np.linalg.norm(angular)) < 0.05,
            },
        }
        world.pause()
        report["renders"].append(capture("after"))
        passed = all(report["physics"]["checks"].values()) and all(x["passed"] for x in report["renders"])
        report["status"] = "passed" if passed else "failed"
        code = 0 if passed else 1
    except Exception as error:
        report["error_type"] = type(error).__name__
        traceback.print_exc()
    finally:
        report["wall_seconds_before_shutdown"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "report": "report.json"}), flush=True)
    # Leave through os._exit rather than SimulationApp.close().
    #
    # The installed build is Isaac 4.5.0-rc.36 (see /isaac-sim/VERSION), and on it the Kit
    # shutdown path segfaults *after* the report has been written. That kills the process with
    # SIGSEGV, so python.sh reports a non-zero code for the positive case that the acceptance
    # manual requires to exit 0 -- the verdict in the report is right while the exit code says
    # otherwise. faulthandler even dumps 71 loaded extension modules on the way out.
    #
    # The report is already written and its handles flushed above, so exiting directly is the only
    # way the exit code can be trusted to mean what the manual says it means.
    import os

    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
