# SPDX-License-Identifier: Apache-2.0
"""URDF parsing and forward kinematics, shared by the figure-only robot authoring.

Why this module exists
----------------------

SPEC-012 authored the dual-Piper case by reading ``piper.urdf`` directly
(``python -m rfw_w.scenes.visualization.piper_case_robot``). SPEC-016 needs the same three
operations for a second robot (a Unitree G1 with 39 joints and STL meshes), so
the kinematics live here once instead of twice.

What it deliberately is not: a robot model. These functions place links at a
**declared** joint vector. No solver, no collision model, no dynamics -- the
figure claims a static pose and nothing more, and every caller records
``ik: not_run`` beside the pose it chose.

Two URDF details that bite, both measured on the two robots on this machine:

* the root link is the one that is never a joint's child. Piper's URDF names a
  ``world`` parent, the G1's does not -- seeding the walk with a hard-coded
  ``world`` frame silently drops the whole G1 tree.
* a ``<visual>`` may carry its own ``<origin>`` *and* its own ``<material>``,
  and a link may carry several visuals. Colour therefore belongs to the visual,
  not to the link, which is why the palette is indexed by ``(link, index)``.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


class UrdfError(ValueError):
    """The URDF is not one this module can place."""


def vector(node, key: str, default: str = "0 0 0") -> list[float]:
    """A URDF ``xyz``/``rpy``/``axis`` attribute as three floats."""
    raw = node.get(key, default) if node is not None else default
    return [float(x) for x in raw.split()]


def transform(xyz, rpy) -> np.ndarray:
    """A 4x4 homogeneous transform from a URDF origin (xyz + fixed-axis rpy)."""
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def parse_urdf(path: Path) -> dict:
    """Links (with their visuals), joints and named materials, as plain data."""
    root = ET.parse(path).getroot()
    materials: dict[str, list[float]] = {}
    for node in root.findall("material"):
        colour = node.find("color")
        if colour is not None and node.get("name"):
            materials[node.get("name")] = [float(v) for v in colour.get("rgba").split()]

    links: dict[str, list[dict]] = {}
    for link in root.findall("link"):
        visuals = []
        for index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                # A primitive-geometry visual: recorded as skipped rather than
                # approximated, so the baked mesh never silently loses a part.
                visuals.append({"index": index, "mesh": None, "skipped": "non_mesh_geometry"})
                continue
            material = visual.find("material")
            name = material.get("name") if material is not None else None
            inline = material.find("color") if material is not None else None
            rgba = [float(v) for v in inline.get("rgba").split()] if inline is not None else None
            origin = visual.find("origin")
            visuals.append({
                "index": index,
                "mesh": mesh.get("filename"),
                "scale": vector(mesh, "scale", "1 1 1"),
                "origin_xyz": vector(origin, "xyz"),
                "origin_rpy": vector(origin, "rpy"),
                "material_name": name,
                "material_rgba": rgba,
            })
        links[link.get("name")] = visuals

    joints = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        joints.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "xyz": vector(joint.find("origin"), "xyz"),
            "rpy": vector(joint.find("origin"), "rpy"),
            "axis": vector(joint.find("axis"), "xyz", "0 0 1"),
            "limits": ([float(limit.get("lower")), float(limit.get("upper"))]
                       if limit is not None and limit.get("lower") is not None else None),
        })
    return {"links": links, "joints": joints, "materials": materials}


def root_links(model: dict) -> list[str]:
    """Declared links that are never a joint's child -- the trees' roots.

    Only **declared** links: a joint whose parent is a name with no ``<link>``
    of its own (a typo, or a fragment pasted out of another URDF) would
    otherwise become a silent second root at the origin, and its whole subtree
    would render in the wrong place with nothing saying so. ``fk`` reports that
    as ``URDF_DISCONNECTED``.
    """
    children = {j["child"] for j in model["joints"]}
    return sorted(name for name in model["links"] if name not in children)


def fk(model: dict, q: dict[str, float]) -> dict[str, np.ndarray]:
    """World transform per link at the declared joint vector (radians).

    Joints absent from ``q`` are at zero. The walk repeats until no new frame
    appears, so a URDF whose joints are not in topological order still resolves;
    a joint whose parent never resolves is a named failure rather than a link
    quietly left at the origin.
    """
    from scipy.spatial.transform import Rotation

    frames: dict[str, np.ndarray] = {name: np.eye(4) for name in root_links(model)}
    pending = list(model["joints"])
    while pending:
        progressed = False
        remaining = []
        for joint in pending:
            parent = frames.get(joint["parent"])
            if parent is None:
                remaining.append(joint)
                continue
            motion = np.eye(4)
            value = float(q.get(joint["name"], 0.0))
            if joint["type"] == "revolute" or joint["type"] == "continuous":
                motion[:3, :3] = Rotation.from_rotvec(np.asarray(joint["axis"]) * value).as_matrix()
            elif joint["type"] == "prismatic":
                motion[:3, 3] = np.asarray(joint["axis"]) * value
            frames[joint["child"]] = parent @ transform(joint["xyz"], joint["rpy"]) @ motion
            progressed = True
        if not progressed:
            raise UrdfError(
                f"URDF_DISCONNECTED({sorted({j['child'] for j in remaining})}): "
                "these links hang off a parent that is not in the tree")
        pending = remaining
    return frames
