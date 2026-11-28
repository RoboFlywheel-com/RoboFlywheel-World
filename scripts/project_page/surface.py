"""Repair triangle orientation without merging authored UV vertices."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def orient_faces(vertices, faces):
    points = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64).copy()
    if not len(faces) or not np.isfinite(points).all():
        raise ValueError("missing or nonfinite geometry")
    # Discover adjacency by coincident positions only. Keep the original indices
    # for rendering: the two sides of a UV seam can legitimately use different UVs.
    _, welded = np.unique(points, axis=0, return_inverse=True)
    tri = points[faces].astype(float)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    _, unique = np.unique(np.sort(welded[faces], axis=1), axis=0, return_index=True)
    keep = np.sort(unique[area[unique] > 0])
    if len(keep) < .98 * len(faces) or not len(keep):
        raise ValueError("too many duplicate or degenerate triangles")
    faces, tri, cross, area = faces[keep], tri[keep], cross[keep], area[keep]
    n = len(faces)
    f = welded[faces]
    a, b = f.ravel(), np.roll(f, -1, axis=1).ravel()
    keys = np.minimum(a,b).astype(np.int64) * (int(welded.max())+1) + np.maximum(a,b)
    order = np.argsort(keys)
    starts = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    if (counts > 2).mean() > .005:
        raise ValueError("excess nonmanifold edges")
    pairs = starts[counts == 2]
    u, v = order[pairs], order[pairs+1]
    parity = ((a[u] < b[u]) == (a[v] < b[v])).astype(int)
    # In the doubled adjacency graph each face has a forward and reversed copy.
    # Opposite edge directions connect equal copies; equal directions cross over.
    x = np.r_[u//3, u//3+n]
    y = np.r_[v//3+parity*n, v//3+(1-parity)*n]
    graph = coo_matrix((np.ones(len(x), dtype=bool), (x,y)), shape=(2*n,2*n))
    _, labels = connected_components(graph, directed=False)
    if (labels[:n] == labels[n:]).any():
        raise ValueError("nonorientable surface")
    _, component = np.unique(np.minimum(labels[:n],labels[n:]), return_inverse=True)
    flip = labels[:n] > labels[n:]
    sign = np.where(flip, -1, 1)
    majority = np.bincount(component, weights=area*sign)
    # Closed shells use signed volume; open patches retain the predominant
    # authored side. There is no intrinsic notion of 'outside' for a sheet.
    closed = np.ones(len(majority), dtype=bool)
    closed[component[order[np.repeat(counts != 2, counts)]//3]] = False
    volume = np.bincount(component, weights=np.einsum("ij,ij->i", tri[:,0],cross)*sign)
    flip ^= np.where(closed, volume < 0, majority < 0)[component]
    faces[flip] = faces[flip, ::-1]
    return faces, int(flip.sum())
