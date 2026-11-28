# SPDX-License-Identifier: Apache-2.0
"""Prepare one metric Z-up asset frame and preserve face assignments."""

from __future__ import annotations


def prepare(mesh, face_ids, dimensions_m, source_up_axis="Y"):
    import numpy as np

    if len(face_ids) != len(mesh.faces) or np.asarray(face_ids).ndim != 1:
        raise ValueError("segmentation face assignments do not match mesh")
    if not np.all(np.isfinite(mesh.vertices)) or len(mesh.faces) == 0:
        raise ValueError("mesh must be finite and nonempty")
    dims = np.asarray(dimensions_m, dtype=float)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        raise ValueError("dimensions must be three positive meter values")
    extent = float(np.ptp(mesh.vertices, axis=0).max())
    if extent <= 0:
        raise ValueError("mesh has zero extent")
    result = mesh.copy()
    scale = float(max(dims) / extent)
    result.apply_scale(scale)
    transform = np.eye(4)
    if source_up_axis == "Y":
        transform[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        result.apply_transform(transform)
    measured = np.ptp(result.vertices, axis=0).tolist()
    return result, {
        "source_up_axis": source_up_axis,
        "target_up_axis": "Z",
        "scale": scale,
        "rotation": transform.tolist(),
        "dimensions_m": measured,
    }


def prepare_pair(visual, segmented, face_ids, dimensions_m, source_up_axis="Y"):
    """One visual-derived transform applies identically to segmented geometry."""
    import numpy as np

    if not np.allclose(visual.bounds, segmented.bounds, rtol=1e-5, atol=1e-7):
        raise ValueError("segmented geometry does not share the visual source frame")
    prepared, frame = prepare(
        visual, np.zeros(len(visual.faces), dtype=int), dimensions_m, source_up_axis
    )
    if len(face_ids) != len(segmented.faces):
        raise ValueError("face assignments do not match segmented geometry")
    metric = segmented.copy()
    metric.apply_scale(frame["scale"])
    metric.apply_transform(np.asarray(frame["rotation"]))
    if not np.allclose(prepared.bounds, metric.bounds, rtol=1e-5, atol=1e-7):
        raise ValueError("prepared visual and segmented bounds differ")
    return prepared, metric, frame
