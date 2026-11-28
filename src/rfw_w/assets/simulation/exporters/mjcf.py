# SPDX-License-Identifier: Apache-2.0
"""Mjcf for one explicitly supplied asset directory."""

import argparse

import json

import os

import xml.etree.ElementTree as ET

ROLL_FRICTION = 0.0001  # MuJoCo default rolling friction; not tracked in v1.3


def fix_mjcf(asset_dir, out_path, collision=None):
    """Write a physics-complete MJCF to out_path. Returns (key, status, detail).

    collision="teleop" swaps the per-hull collision geoms for the
    collision_budget hull set; any other value keeps the stock behavior.
    """
    key = os.path.basename(asset_dir.rstrip("/"))
    xml_path = os.path.join(asset_dir, "mjcf", f"{key}.xml")
    phys_path = os.path.join(asset_dir, "part_physics.json")
    sidecar_path = os.path.join(asset_dir, "mesh", f"{key}_teleop_collision.json")
    fresh_sources = [xml_path, phys_path]
    if collision == "teleop":
        if not os.path.isfile(sidecar_path):
            return key, "fail", "no teleop sidecar; run collision_budget first"
        fresh_sources.append(sidecar_path)
    if os.path.isfile(out_path) and os.path.getmtime(out_path) >= max(
        os.path.getmtime(p) for p in fresh_sources
    ):
        # a stale output from the OTHER collision mode must not be treated
        # as up-to-date just because it is newer
        with open(out_path) as f:
            is_teleop_out = "teleop_collision_mesh" in f.read()
        if is_teleop_out == (collision == "teleop"):
            return key, "skip", "up-to-date"
    phys = json.load(open(phys_path))
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bodies = [b for b in root.iter("body") if b.find("geom") is not None]
    if len(bodies) != 1:
        return (
            key,
            "fail",
            f"{len(bodies)} geom-bearing bodies; "
            "whole-object part_physics cannot be distributed",
        )
    body = bodies[0]
    if body.find("inertial") is not None:
        return key, "fail", "already has <inertial> (stale fix output?)"

    inertia = phys["inertia"]
    ET.SubElement(
        body,
        "inertial",
        attrib={
            "mass": f"{float(phys['total_mass'])}",
            "pos": " ".join(f"{float(v)}" for v in (phys.get("com") or [0, 0, 0])),
            "fullinertia": " ".join(
                str(inertia[r][c])
                for r, c in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
            ),
        },
    )

    mu1 = float(phys.get("whole_mu1", 0.5))
    mu2 = float(phys.get("whole_mu2", 0.005))
    friction = f"{mu1} {mu2} {ROLL_FRICTION}"

    if collision == "teleop":
        # swap the stock per-hull collision set for the budgeted whole-mesh
        # re-decomposition from collision_budget
        n_hulls = int(json.load(open(sidecar_path))["n_hulls"])
        hull_files = [
            os.path.join(asset_dir, "mjcf", "mesh", f"{key}_teleop_collision_{i}.obj")
            for i in range(n_hulls)
        ]
        missing = [f for f in hull_files if not os.path.isfile(f)]
        if missing:
            return (
                key,
                "fail",
                f"{len(missing)}/{n_hulls} teleop hull objs "
                "missing; re-run collision_budget",
            )
        asset = root.find("asset")
        old_mesh_names = set()
        for geom in list(body.iter("geom")):
            if geom.get("contype") == "1":
                old_mesh_names.add(geom.get("mesh"))
                body.remove(geom)
        if asset is not None:
            for m in list(asset.iter("mesh")):
                if m.get("name") in old_mesh_names:
                    asset.remove(m)
        if asset is None:
            # MJCF top-level element order matters: <asset> precedes <worldbody>
            asset = ET.Element("asset")
            worldbody = root.find("worldbody")
            root.insert(
                (
                    list(root).index(worldbody)
                    if worldbody is not None
                    else len(list(root))
                ),
                asset,
            )
        for i in range(n_hulls):
            name = f"teleop_collision_mesh_{i}"
            ET.SubElement(
                asset,
                "mesh",
                attrib={
                    "name": name,
                    "file": f"mesh/{key}_teleop_collision_{i}.obj",
                    "scale": "1 1 1",
                    "inertia": "shell",
                },
            )
            ET.SubElement(
                body,
                "geom",
                attrib={
                    "type": "mesh",
                    "mesh": name,
                    "contype": "1",
                    "conaffinity": "1",
                    "rgba": "1 1 1 0",
                    "friction": friction,
                },
            )
        n_coll = n_hulls
    else:
        n_coll = 0
        for geom in body.iter("geom"):
            if geom.get("contype") == "1":
                geom.set("friction", friction)
                n_coll += 1
        if n_coll != int(phys.get("n_collision_hulls", n_coll)):
            return (
                key,
                "fail",
                f"collision geom count {n_coll} != " "part_physics n_collision_hulls",
            )

    # make the repaired file location-independent: absolute meshdir/texturedir
    # (files are referenced as "mesh/..."; MjSpec does not default
    # texturedir to meshdir, so set both)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", os.path.join(asset_dir, "mjcf"))
    compiler.set("texturedir", os.path.join(asset_dir, "mjcf"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.indent(tree, space="  ", level=0)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return (
        key,
        "ok",
        f"{n_coll} collision geoms, mass={phys['total_mass']} -> {out_path}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collision", choices=["teleop"])
    args = parser.parse_args()
    key, status, detail = fix_mjcf(args.asset_dir, args.out, args.collision)
    print(json.dumps({"key": key, "status": status, "detail": detail}))
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
