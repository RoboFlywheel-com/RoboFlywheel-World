# SPDX-License-Identifier: Apache-2.0
"""Per-part voxel volume and aggregate rigid-body mass-property calculations."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def voxelize_parts(mesh, face_ids, pitch_rel: float = 1 / 64):
    extent = float(np.ptp(mesh.vertices, axis=0).max())
    pitch = max(extent * pitch_rel, 1e-5)
    voxels = mesh.voxelized(pitch=pitch)
    try:
        voxels = voxels.fill()
    except Exception:
        pass
    centers = np.asarray(voxels.points)
    voxel_volume = float(pitch**3)
    if not len(centers):
        centers = np.asarray(mesh.triangles_center)
        voxel_volume = float(mesh.area / max(len(centers), 1) * pitch)
    _, nearest_faces = cKDTree(mesh.triangles_center).query(centers, k=1)
    voxel_parts = np.asarray(face_ids)[nearest_faces]
    parts = {}
    for part_id in np.unique(voxel_parts):
        if part_id < 0:
            continue
        mask = voxel_parts == part_id
        parts[int(part_id)] = {
            "volume_m3": float(mask.sum() * voxel_volume),
            "n_voxels": int(mask.sum()),
            "centroid_m": centers[mask].mean(axis=0).tolist(),
        }
    return parts, centers, voxel_parts


def mass_com_inertia(centers, voxel_parts, part_masses):
    counts = {
        int(part): int((voxel_parts == part).sum())
        for part in np.unique(voxel_parts)
        if part >= 0
    }
    voxel_masses = np.asarray(
        [
            (
                part_masses.get(int(part), 0.0) / counts[int(part)]
                if int(part) >= 0 and counts.get(int(part))
                else 0.0
            )
            for part in voxel_parts
        ]
    )
    total_mass = float(voxel_masses.sum())
    com = (voxel_masses[:, None] * centers).sum(axis=0) / max(total_mass, 1e-12)
    offsets = centers - com
    squared_radii = (offsets**2).sum(axis=1)
    inertia = np.zeros((3, 3), dtype=float)
    for row in range(3):
        for column in range(3):
            diagonal = 1.0 if row == column else 0.0
            inertia[row, column] = (
                voxel_masses
                * (squared_radii * diagonal - offsets[:, row] * offsets[:, column])
            ).sum()
    return total_mass, com, inertia
