# SPDX-License-Identifier: Apache-2.0
"""Load exported rigid bodies in a fresh engine process without altering source data.

This checks native mass properties and finite stepping. It does not certify
settlement, manipulation or visual fidelity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def check_properties(physical, mass, com, inertia):
    import numpy as np

    expected = np.asarray(physical["inertia"], dtype=float)
    measured = np.asarray(inertia, dtype=float)
    if measured.shape == (3,):
        measured = np.diag(measured)
    return {
        "mass": bool(np.isclose(mass, physical["total_mass"], rtol=1e-5)),
        "com": bool(np.allclose(com, physical["com"], atol=1e-5)),
        "inertia": bool(np.allclose(
            np.linalg.eigvalsh(measured), np.linalg.eigvalsh(expected),
            rtol=1e-4, atol=1e-8,
        )),
    }


def mujoco_probe(urdf, mjcf, physical, steps):
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    bodies = [i for i in range(1, model.nbody) if model.body_mass[i] > 0]
    if len(bodies) != 1 or model.nq < 7:
        raise ValueError("expected one dynamic rigid body")
    body = bodies[0]
    checks = check_properties(physical, model.body_mass[body],
                              model.body_ipos[body], model.body_inertia[body])
    for _ in range(steps):
        mujoco.mj_step(model, data)
    checks["step"] = bool(np.isfinite(data.qpos).all()
                          and np.isfinite(data.qvel).all() and data.time > 0)
    return checks, {"simulated_seconds": float(data.time), "dynamic_bodies": 1}


def pybullet_probe(urdf, mjcf, physical, steps):
    import numpy as np
    import pybullet as p

    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        body = p.loadURDF(str(urdf), useFixedBase=False,
                          flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client)
        if p.getNumJoints(body, physicsClientId=client):
            raise ValueError("expected one rigid body")
        info = p.getDynamicsInfo(body, -1, physicsClientId=client)
        checks = check_properties(physical, info[0], info[3], info[2])
        for _ in range(steps):
            p.stepSimulation(physicsClientId=client)
        pose = p.getBasePositionAndOrientation(body, physicsClientId=client)
        velocity = p.getBaseVelocity(body, physicsClientId=client)
        checks["step"] = bool(np.isfinite(np.concatenate((*pose, *velocity))).all())
        return checks, {"dynamic_bodies": 1, "simulated_seconds": steps / 240.0}
    finally:
        p.disconnect(client)


def sapien_probe(urdf, mjcf, physical, steps):
    import numpy as np
    import sapien

    # Physics-only loading keeps this usable on servers without a Vulkan renderer.
    scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    loader.load_multiple_collisions_from_file = True
    articulations, builders, cameras = loader.parse(str(urdf))
    if articulations or len(builders) != 1:
        raise ValueError("expected one single-body URDF")
    builder = builders[0]
    builder.visual_records.clear()
    actor = builder.build()
    body = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if body is None:
        raise ValueError("export did not load as a dynamic body")
    checks = check_properties(physical, body.mass, body.cmass_local_pose.p, body.inertia)
    for _ in range(steps):
        scene.step()
    checks["step"] = bool(np.isfinite(np.concatenate((actor.pose.p, actor.pose.q,
                         body.linear_velocity, body.angular_velocity))).all())
    evidence = {"dynamic_bodies": 1,
                "simulated_seconds": steps * float(scene.get_timestep()),
                "visuals": "not_loaded_physics_only"}
    scene.remove_entity(actor)
    return checks, evidence


def genesis_probe(urdf, mjcf, physical, steps):
    import genesis as gs
    import numpy as np

    gs.init(backend=gs.gpu, logging_level="warning", seed=2025)
    scene = gs.Scene(show_viewer=False,
                     sim_options=gs.options.SimOptions(dt=1 / 240),
                     rigid_options=gs.options.RigidOptions(enable_collision=True))
    body = scene.add_entity(gs.morphs.MJCF(file=str(mjcf), align=False,
                                         recompute_inertia=False, visualization=False))
    scene.build()
    links = [link for link in body.links if link.inertial_mass and link.inertial_mass > 0]
    if len(links) != 1:
        raise ValueError("expected one dynamic rigid body")
    link = links[0]
    checks = check_properties(physical, link.inertial_mass, link.inertial_pos, link.inertial_i)
    for _ in range(steps):
        scene.step()
    checks["step"] = bool(np.isfinite(body.get_pos().cpu().numpy()).all()
                          and np.isfinite(body.get_quat().cpu().numpy()).all())
    return checks, {"dynamic_bodies": 1, "simulated_seconds": steps / 240.0}


PROBES = {"mujoco": mujoco_probe, "pybullet": pybullet_probe,
          "sapien": sapien_probe, "genesis": genesis_probe}
VERSIONS = {"mujoco": "mujoco", "pybullet": "pybullet",
            "sapien": "sapien", "genesis": "genesis-world"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--physics", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=PROBES)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("steps must be positive")
    if args.out.exists():
        parser.error("output must be new")
    args.out.mkdir(parents=True)
    report = {"schema": "rfw.rigid-probe/v1", "backend": args.backend,
              "status": "failed", "steps": args.steps,
              "render": "not_run", "settle": "not_run", "task_run": "not_run"}
    try:
        physical = json.loads(args.physics.read_text())
        from rfw_w.assets.physics.estimate import validate_physics

        validate_physics(physical)
        report["version"] = importlib.metadata.version(VERSIONS[args.backend])
        checks, evidence = PROBES[args.backend](args.urdf.resolve(), args.mjcf.resolve(),
                                              physical, args.steps)
        report.update(checks=checks, evidence=evidence,
                      status="passed" if all(checks.values()) else "failed")
    except Exception as error:
        report.update(error_type=type(error).__name__)
        # Log details locally; the portable report contains no deployment paths.
        import traceback

        traceback.print_exc()
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
