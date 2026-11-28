# SPDX-License-Identifier: Apache-2.0
"""Calculate physical parameters once from metric geometry and part estimates."""

from __future__ import annotations
import math


def validate_physics(data):
    import numpy as np

    mass = data["total_mass"]
    com = np.asarray(data["com"])
    inertia = np.asarray(data["inertia"])
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be finite and positive")
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise ValueError("COM must be a finite vector")
    if (
        inertia.shape != (3, 3)
        or not np.all(np.isfinite(inertia))
        or not np.allclose(inertia, inertia.T, atol=1e-10)
    ):
        raise ValueError("inertia must be finite and symmetric")
    values = np.linalg.eigvalsh(inertia)
    if values[0] <= 0 or values[2] > values[:2].sum() + 1e-10:
        raise ValueError("inertia is not physically valid")
    for name in ("whole_mu1", "whole_mu2"):
        if not math.isfinite(data[name]) or data[name] < 0:
            raise ValueError("invalid friction")


def estimate(
    mesh, face_ids, annotations, collision, policy="anchor_on_flag", voxel_resolution=64
):
    from .mass_properties import voxelize_parts, mass_com_inertia
    from .materials import density_of

    parts, centers, vox_ids = voxelize_parts(
        mesh, face_ids, pitch_rel=1 / voxel_resolution
    )
    props = {}
    masses = {}
    for pid, annotation in annotations.items():
        material = annotation["material"]
        density, source = density_of(material["category"], material.get("subtype", ""))
        geom = parts.get(pid)
        if geom is None:
            raise ValueError("part has no occupied voxel volume")
        convex = collision[pid]
        volume = float(convex["convex_volume_m3"])
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("invalid collision volume")
        masses[pid] = volume * density
        props[str(pid)] = {
            "material_category": material["category"],
            "material_subtype": material.get("subtype", ""),
            "density": density,
            "density_source": source,
            "convex_volume": volume,
            "voxel_volume": geom["volume_m3"],
            "volume_m3": volume,
            "centroid": geom["centroid_m"],
            "n_voxels": geom["n_voxels"],
            "hulls": convex["hulls"],
            "mu1": annotation["static_friction"],
            "mu2": annotation["dynamic_friction"],
        }
    composed, com, inertia = mass_com_inertia(centers, vox_ids, masses)
    return {
        "total_mass": composed,
        "composed_mass": composed,
        "com": com.tolist(),
        "inertia": inertia.tolist(),
        "parts": props,
    }, masses


def anchor(data, masses, target_mass, policy):
    composed = data["composed_mass"]
    ratio = composed / target_mass if target_mass else None
    flag = ratio is not None and not 0.5 <= ratio <= 2
    if policy == "anchor_vlm" and target_mass is None:
        raise ValueError("anchor_vlm requires measured/estimated mass")
    factor = (
        target_mass / composed
        if target_mass
        and (policy == "anchor_vlm" or policy == "anchor_on_flag" and flag)
        else 1.0
    )
    data["total_mass"] = composed * factor
    data["inertia"] = [[v * factor for v in row] for row in data["inertia"]]
    final = {pid: m * factor for pid, m in masses.items()}
    for pid, mass in final.items():
        data["parts"][str(pid)]["mass_kg"] = mass
    data.update(
        vlm_mass=target_mass,
        mass_ratio=ratio,
        mass_check="flag" if flag else "ok" if ratio is not None else "no_vlm",
        mass_policy=policy,
        mass_source="vlm_anchored" if factor != 1 else "composed",
    )
    total = data["total_mass"]
    data["whole_mu1"] = (
        sum(data["parts"][str(pid)]["mu1"] * mass for pid, mass in final.items())
        / total
    )
    data["whole_mu2"] = (
        sum(data["parts"][str(pid)]["mu2"] * mass for pid, mass in final.items())
        / total
    )
    validate_physics(data)
    return data
