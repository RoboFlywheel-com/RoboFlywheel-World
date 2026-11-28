#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bake a robot's URDF at a declared joint vector into one figure-only OBJ.

SPEC-016 TASK 1. The pipeline has never had a robot in it: ``task.robot_profile``
is validated, hashed and rendered by nothing, so every paper figure shows a task
with no machine to do it. This bakes the declared arm or humanoid into a single
world-space mesh that the figure request can carry as one more instance.

What it claims: these links, at these joint angles, in this pose. What it does
not claim, and records as ``not_run``: physics, IK, reachability of any grasp.
The joint vector is *authored*, not solved.

The baked frame puts the robot's support surface at ``z = 0`` (the lowest vertex
after FK), so the placement rule only has to put the base on a table top or the
floor; the shift applied is recorded.

Writes, under ``--output-dir``:

    robot.obj / robot.mtl   one mesh, one material group per (link, visual)
    robot_asset.json        source digests, joint vector, frames, bounds

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.visualization.robot_asset \\
      --robot piper --pose experiments/paper_dollhouse/pose_piper_ready.json \\
      --output-dir outputs/_paper/robot_piper_01
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from rfw_w.scenes.visualization.paper_case_common import digest, write
from rfw_w.scenes.visualization.urdf_kinematics import fk, parse_urdf, transform

# Robot sources are named by environment variable, never by path: an internal
# asset path in the repository is exactly what `.env.local` exists to prevent.
ROBOTS = {
    "piper": {
        "root_env": "PVSCENE_PAPER_PIPER_ROOT",
        "urdf": "piper.urdf",
        "palette": "palette.json",
        "mount": "surface",
        "tcp": {"link": "gripper_base", "offset_m": [0.0, 0.0, 0.138]},
        # Which way the baked mesh's +x points relative to the robot's own front, so a declaration
        # can say "face this bearing" and mean it. Both robots front on +x, and that is read off
        # the kinematics rather than off a render: the Piper's tool point comes out at
        # [0.492, -0.089, 0.257] and the G1's right hand at [0.295, -0.227, 0.796], both ahead of
        # the base on +x. An earlier 180 deg offset here was inferred from where the "UNITREE"
        # lettering appeared in a render, and it turned the humanoid away from the nightstand it
        # was reaching for -- the lettering is on both sides of that torso.
        "facing_offset_deg": 0.0,
        "description": "AgileX Piper 6-DoF arm, table-mounted",
    },
    "g1": {
        "root_env": "PVSCENE_PAPER_G1_ROOT",
        "urdf": "g1_29dof_lock_waist_rev_1_0.urdf",
        "palette": None,
        "mount": "floor",
        "tcp": {"link": "right_wrist_yaw_link", "offset_m": [0.05, 0.0, 0.0]},
        "facing_offset_deg": 0.0,
        "description": "Unitree G1 29-DoF humanoid, standing on the floor",
    },
}

DEFAULT_COLOUR = (0.62, 0.62, 0.64)


def resolve_root(robot: str) -> Path:
    variable = ROBOTS[robot]["root_env"]
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(
            f"ROBOT_SOURCE_UNSET({variable}): declare the checkout in .env.local; "
            "robot source paths are deliberately not in the repository")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise SystemExit(f"ROBOT_SOURCE_MISSING({variable})")
    return root


def find_mesh(root: Path, urdf_dir: Path, filename: str) -> Path:
    """Resolve a URDF mesh filename against the layouts the two robots use.

    Piper names bare files that live in ``meshes/``; the G1 names
    ``meshes/<file>`` relative to its URDF. ``package://`` is stripped and then
    tried the same way.
    """
    name = filename.split("package://", 1)[-1]
    if name.startswith("//"):
        name = name[2:]
    candidates = [urdf_dir / name, urdf_dir / "meshes" / name, root / name, root / "meshes" / name]
    if "/" in name:
        tail = name.split("/", 1)[1]
        candidates += [urdf_dir / tail, urdf_dir / "meshes" / tail, root / "meshes" / tail]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"ROBOT_MESH_MISSING({filename})")


def visual_colour(link: str, visual: dict, palette: dict | None, materials: dict) -> tuple:
    """Colour for one visual, and where it came from.

    Order matters: a palette captured from the manufacturer's own asset beats
    the URDF's named material, which beats grey. The G1 has no palette and
    names ``dark``/``white``; Piper has a palette and names nothing.
    """
    index = visual["index"]
    if palette is not None:
        entry = palette.get(link)
        if entry and index < len(entry):
            return tuple(float(v) for v in entry[index][:3]), "palette"
    if visual.get("material_rgba"):
        return tuple(float(v) for v in visual["material_rgba"][:3]), "visual_material"
    name = visual.get("material_name")
    if name and name in materials:
        return tuple(float(v) for v in materials[name][:3]), f"urdf_material:{name}"
    return DEFAULT_COLOUR, "default"


def bake(robot: str, pose_degrees: dict, output_dir: Path) -> dict:
    import trimesh

    spec = ROBOTS[robot]
    root = resolve_root(robot)
    urdf_path = root / spec["urdf"]
    if not urdf_path.is_file():
        raise SystemExit(f"ROBOT_URDF_MISSING({spec['urdf']})")
    model = parse_urdf(urdf_path)
    palette = None
    if spec["palette"] and (root / spec["palette"]).is_file():
        palette = json.loads((root / spec["palette"]).read_text())

    known = {j["name"] for j in model["joints"]}
    unknown = sorted(set(pose_degrees) - known)
    if unknown:
        raise SystemExit(f"ROBOT_POSE_UNKNOWN_JOINT({unknown})")
    q = {}
    for name, value in pose_degrees.items():
        joint = next(j for j in model["joints"] if j["name"] == name)
        # A prismatic joint is declared in metres; a revolute one in degrees.
        q[name] = float(value) if joint["type"] == "prismatic" else math.radians(float(value))
    frames = fk(model, q)

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    groups: list[tuple[str, int, int]] = []   # (material name, first face, face count)
    colours: dict[str, tuple] = {}
    records: list[dict] = []
    skipped: list[dict] = []
    offset = 0
    for link, visuals in sorted(model["links"].items()):
        if link not in frames:
            skipped.append({"link": link, "why": "not reached by the kinematic walk"})
            continue
        for visual in visuals:
            if not visual.get("mesh"):
                skipped.append({"link": link, "index": visual["index"],
                                "why": visual.get("skipped", "no mesh")})
                continue
            source = find_mesh(root, urdf_path.parent, visual["mesh"])
            mesh = trimesh.load(str(source), force="mesh", process=False)
            local = transform(visual["origin_xyz"], visual["origin_rpy"])
            scale = np.asarray(visual.get("scale") or [1.0, 1.0, 1.0], dtype=np.float64)
            points = np.asarray(mesh.vertices, dtype=np.float64) * scale
            world = frames[link] @ local
            points = points @ world[:3, :3].T + world[:3, 3]
            triangles = np.asarray(mesh.faces, dtype=np.int64) + offset
            material = f"{link}_{visual['index']}"
            colour, colour_source = visual_colour(link, visual, palette, model["materials"])
            colours[material] = colour
            groups.append((material, sum(len(f) for f in faces), len(triangles)))
            vertices.append(points)
            faces.append(triangles)
            offset += len(points)
            records.append({
                "link": link, "visual_index": visual["index"],
                "mesh": visual["mesh"], "sha256": digest(source),
                "colour_rgb": list(colour), "colour_source": colour_source,
                "vertex_count": int(len(points)), "face_count": int(len(triangles)),
            })
    if not vertices:
        raise SystemExit("ROBOT_BAKE_EMPTY(no mesh visual survived the walk)")

    points = np.concatenate(vertices)
    triangles = np.concatenate(faces)
    # Put the support surface at z = 0 so placement only has to name a surface.
    shift = float(points[:, 2].min())
    points[:, 2] -= shift

    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / "robot.obj"
    mtl_path = output_dir / "robot.mtl"
    mtl_path.write_text("".join(
        f"newmtl {name}\nKd {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\nKa 0 0 0\nKs 0.04 0.04 0.04\n"
        f"Ns 24\nd 1\nillum 2\n\n" for name, c in colours.items()), encoding="utf-8")

    lines = [f"# pvscene v3 SPEC-016 figure robot: {robot}", "mtllib robot.mtl"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]
    cursor = 0
    for material, first, count in groups:
        lines.append(f"g {material}")
        lines.append(f"usemtl {material}")
        for face in triangles[first:first + count]:
            lines.append("f " + " ".join(str(int(i) + 1) for i in face))
        cursor += count
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tcp_link = spec["tcp"]["link"]
    tcp_world = None
    if tcp_link in frames:
        local = np.asarray(spec["tcp"]["offset_m"], dtype=np.float64)
        tcp = frames[tcp_link][:3, :3] @ local + frames[tcp_link][:3, 3]
        tcp_world = [float(tcp[0]), float(tcp[1]), float(tcp[2] - shift)]

    payload = {
        "schema_version": "pvscene.figure-robot/v1",
        "robot": robot,
        "description": spec["description"],
        "mount": spec["mount"],
        "source_urdf": spec["urdf"],
        "source_urdf_sha256": digest(urdf_path),
        "source_root_env": spec["root_env"],
        "pose_degrees": dict(sorted(pose_degrees.items())),
        "joint_count": len(model["joints"]),
        "posed_joint_count": len(pose_degrees),
        "base_shift_m": shift,
        "base_frame": "the baked mesh's z = 0 is the support surface; x/y are the robot's own axes",
        "tcp_link": tcp_link,
        "tcp_offset_m": spec["tcp"]["offset_m"],
        "tcp_in_baked_frame_m": tcp_world,
        "tcp_definition": "a pre-operation point derived from the source URDF, not a calibrated "
                          "grasp frame",
        "bounds_min_m": points.min(axis=0).tolist(),
        "bounds_max_m": points.max(axis=0).tolist(),
        "vertex_count": int(len(points)),
        "face_count": int(len(triangles)),
        "visuals": records,
        "skipped": skipped,
        "visual_obj_sha256": digest(obj_path),
        "material_lib_sha256": digest(mtl_path),
        "physics_simulation": "not_run",
        "ik": "not_run",
        "collision_model": "not_run",
        "claim": "figure_only_static_pose",
    }
    write(output_dir / "robot_asset.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    parser.add_argument("--pose", type=Path, required=True,
                        help='JSON {"joint_degrees": {"joint1": -35, ...}}; prismatic joints in metres')
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")
    pose = json.loads(args.pose.read_text())
    payload = bake(args.robot, pose.get("joint_degrees", {}), out)
    print(json.dumps({
        "output_dir": str(out), "robot": payload["robot"],
        "links_baked": len(payload["visuals"]), "skipped": len(payload["skipped"]),
        "bounds_min_m": payload["bounds_min_m"], "bounds_max_m": payload["bounds_max_m"],
        "tcp_in_baked_frame_m": payload["tcp_in_baked_frame_m"],
        "ik": "not_run",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
