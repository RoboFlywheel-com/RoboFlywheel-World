# SPDX-License-Identifier: Apache-2.0
"""Native physics smoke check of a generated mesh request.

Single collision meshes are convex proxies. Request mass is preserved; inertia
is engine-derived. This does not replace delivery or robot-task acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def build_mjcf(request, out):
    root = ET.Element("mujoco", model="rfw_scene_probe")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=str(1 / request["physics_hz"]), gravity="0 0 -9.81")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="plane", size="20 20 .1", pos="0 0 0")
    bodies = []
    for index, record in enumerate(request["instances"]):
        mesh = Path(record.get("collision_uri") or record["visual_uri"])
        if not mesh.is_file():
            raise FileNotFoundError("request geometry missing")
        ET.SubElement(asset, "mesh", name=f"mesh_{index}", file=str(mesh.resolve()))
        body = ET.SubElement(world, "body", name=record["instance_id"],
                             pos=" ".join(map(str, record["translation"])),
                             quat=" ".join(map(str, record["rotation_wxyz"])))
        if record["dynamic"]:
            ET.SubElement(body, "freejoint")
            bodies.append(record["instance_id"])
        ET.SubElement(body, "geom", type="mesh", mesh=f"mesh_{index}",
                      mass=str(record["mass_kg"]))
    path = out / "scene.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path, bodies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("steps must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "rfw.scene-physics-probe/v1", "status": "failed",
              "backend": "mujoco", "steps": args.steps,
              "collision": "single_mesh_convex_proxy",
              "physics": "request_mass_engine_derived_inertia",
              "room": "ground_only_no_wall_collision",
              "settle": "not_run", "render": "not_run", "task_run": "not_run"}
    try:
        import mujoco
        import numpy as np

        request = json.loads(args.request.read_text())
        xml, names = build_mjcf(request, args.out)
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        if not names:
            raise ValueError("request contains no dynamic instances")
        mujoco.mj_forward(model, data)
        before = {name: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].tolist()
                  for name in names}
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("nonfinite scene state")
        report.update(status="passed", version=mujoco.__version__,
                      simulated_seconds=float(data.time), instance_count=len(request["instances"]),
                      dynamic_instances=names, initial_positions=before,
                      final_positions={name: data.xpos[mujoco.mj_name2id(
                          model, mujoco.mjtObj.mjOBJ_BODY, name)].tolist() for name in names})
    except Exception as error:
        report["error_type"] = type(error).__name__
        import traceback

        traceback.print_exc()
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
