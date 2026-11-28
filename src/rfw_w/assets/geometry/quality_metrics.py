# SPDX-License-Identifier: Apache-2.0
"""Single-mesh topology, fragments, degeneracy and shape measurements."""

import argparse

import gc

import json

import time

from pathlib import Path

import numpy as np

import psutil

import trimesh

from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import connected_components

from scipy.spatial import ConvexHull

from rfw_w.assets.geometry.quality_policy import RED_FLAGS

MEM_CAP_MB = 1500

HULL_SAMPLE_MAX = 5000


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def edge_topology(faces):
    """Return sorted edges, signs, face indices, groups and per-edge counts."""
    F = len(faces)
    e0 = np.stack([faces[:, 0], faces[:, 1]], axis=1)
    e1 = np.stack([faces[:, 1], faces[:, 2]], axis=1)
    e2 = np.stack([faces[:, 2], faces[:, 0]], axis=1)
    directed = np.concatenate([e0, e1, e2], axis=0)
    fidx = np.tile(np.arange(F), 3)
    sorted_edges = np.sort(directed, axis=1)
    sign_pos = directed[:, 0] < directed[:, 1]
    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    se = sorted_edges[order]
    sp = sign_pos[order]
    fi = fidx[order]
    diff = np.any(se[1:] != se[:-1], axis=1)
    group_starts = np.concatenate([[0], np.where(diff)[0] + 1])
    group_ends = np.concatenate([group_starts[1:], [len(se)]])
    n_unique = len(group_starts)
    counts = group_ends - group_starts
    return se, sp, fi, group_starts, group_ends, counts, n_unique


def scan_one(glb_path: Path, domain, class_name, obj_id: str, verbose: bool = False):
    t0 = time.time()
    try:
        scene = trimesh.load(str(glb_path), force="scene", process=False)
        if isinstance(scene, trimesh.Scene):
            geoms = [g for g in scene.geometry.values() if hasattr(g, "vertices")]
            if not geoms:
                return {
                    "domain": domain,
                    "class": class_name,
                    "obj_id": obj_id,
                    "error": "no_geometry",
                }
            mesh = trimesh.util.concatenate(geoms)
            n_parts_glb = len(geoms)
            del scene, geoms
        else:
            mesh = scene
            n_parts_glb = 1
            del scene

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        n_verts = int(len(verts))
        n_faces = int(len(faces))
        if n_faces == 0:
            return {
                "domain": domain,
                "class": class_name,
                "obj_id": obj_id,
                "error": "no_faces",
            }

        if rss_mb() > MEM_CAP_MB:
            return {
                "domain": domain,
                "class": class_name,
                "obj_id": obj_id,
                "error": "oom_skip_after_load",
                "n_verts": n_verts,
                "n_faces": n_faces,
            }

        extents = np.asarray(mesh.extents, dtype=float)
        ext_sorted = np.sort(extents)
        aspect_ratio_max = float(ext_sorted[-1] / max(ext_sorted[0], 1e-9))
        thinness = float(ext_sorted[0] / max(ext_sorted[1], 1e-9))
        bbox_volume = float(np.prod(extents)) if (extents > 0).all() else 0.0
        bbox_diag = float(np.linalg.norm(extents))

        bbox_center = (verts.min(0) + verts.max(0)) / 2
        verts_centroid = verts.mean(0)
        centroid_offset_norm = float(
            np.linalg.norm(verts_centroid - bbox_center) / max(bbox_diag, 1e-9)
        )

        se, sp, fi, gs, ge, ecounts, n_unique = edge_topology(faces)
        boundary = int((ecounts == 1).sum())
        manifold = int((ecounts == 2).sum())
        non_manifold = int((ecounts >= 3).sum())
        is_watertight = bool(boundary == 0 and non_manifold == 0)
        boundary_edges_ratio = float(boundary / max(n_unique, 1))
        non_manifold_edges_ratio = float(non_manifold / max(n_unique, 1))

        n_wind_ok = 0
        n_wind_bad = 0
        if manifold > 0:
            mask2 = ecounts == 2
            starts2 = gs[mask2]
            sa = sp[starts2]
            sb = sp[starts2 + 1]
            n_wind_ok = int((sa != sb).sum())
            n_wind_bad = int((sa == sb).sum())
        winding_consistent_ratio = (
            float(n_wind_ok / max(manifold, 1)) if manifold > 0 else 0.0
        )

        mask2 = ecounts == 2
        starts2 = gs[mask2]
        if len(starts2) > 0:
            adj_a = fi[starts2]
            adj_b = fi[starts2 + 1]
            rows = np.concatenate([adj_a, adj_b])
            cols = np.concatenate([adj_b, adj_a])
            data = np.ones(len(rows), dtype=np.int8)
            g = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
            n_components, labels = connected_components(g, directed=False)
            del g, rows, cols, data
            face_counts = np.bincount(labels, minlength=n_components)
        else:
            n_components = n_faces
            face_counts = np.ones(n_faces, dtype=np.int64)
            labels = np.arange(n_faces)

        sorted_counts = np.sort(face_counts)[::-1]
        largest_frac = float(sorted_counts[0] / n_faces)
        top2_frac = float(sorted_counts[:2].sum() / n_faces)
        top3_frac = float(sorted_counts[:3].sum() / n_faces)
        top5_frac = float(sorted_counts[:5].sum() / n_faces)
        top1_top2_gap = float(
            (
                sorted_counts[0] - sorted_counts[1]
                if len(sorted_counts) >= 2
                else sorted_counts[0]
            )
            / n_faces
        )
        tiny_thresh = max(5, int(0.01 * n_faces))
        n_tiny = int(np.sum(face_counts < tiny_thresh))
        n_singleton = int(np.sum(face_counts == 1))

        del se, sp, fi, gs, ge, ecounts, labels, face_counts, sorted_counts
        gc.collect()

        if rss_mb() > MEM_CAP_MB:
            return {
                "domain": domain,
                "class": class_name,
                "obj_id": obj_id,
                "error": "oom_skip_after_components",
                "n_verts": n_verts,
                "n_faces": n_faces,
                "n_components": int(n_components),
            }

        chunk = 50000
        n_degen = 0
        total_area = 0.0
        for s in range(0, n_faces, chunk):
            f = faces[s : s + chunk]
            tri = verts[f]
            e1 = tri[:, 1] - tri[:, 0]
            e2 = tri[:, 2] - tri[:, 0]
            cross = np.cross(e1, e2)
            a = np.linalg.norm(cross, axis=1)
            total_area += float(a.sum() / 2.0)
            n_degen += int(np.sum(a < 1e-12))
            del f, tri, e1, e2, cross, a
        degen_ratio = float(n_degen / n_faces)

        hull_volume = 0.0
        hull_solidity = 0.0
        try:
            if n_verts > HULL_SAMPLE_MAX:
                idx = np.random.RandomState(0).choice(
                    n_verts, HULL_SAMPLE_MAX, replace=False
                )
                pts = verts[idx]
            else:
                pts = verts
            hull = ConvexHull(pts)
            hull_volume = float(hull.volume)
            hull_solidity = float(hull_volume / bbox_volume) if bbox_volume > 0 else 0.0
            del hull, pts
        except Exception:
            hull_volume = -1.0
            hull_solidity = -1.0

        scan_time = round(time.time() - t0, 2)
        rss_peak = rss_mb()

        del mesh, verts, faces
        gc.collect()

        rec = {
            "domain": domain,
            "class": class_name,
            "obj_id": obj_id,
            "n_verts": n_verts,
            "n_faces": n_faces,
            "n_parts_glb": int(n_parts_glb),
            "total_area": total_area,
            "n_components": int(n_components),
            "largest_frac": largest_frac,
            "top2_frac": top2_frac,
            "top3_frac": top3_frac,
            "top5_frac": top5_frac,
            "top1_top2_gap": top1_top2_gap,
            "n_tiny_components": n_tiny,
            "n_singleton_components": n_singleton,
            "is_watertight": is_watertight,
            "n_unique_edges": int(n_unique),
            "boundary_edges_count": boundary,
            "boundary_edges_ratio": boundary_edges_ratio,
            "non_manifold_edges_count": non_manifold,
            "non_manifold_edges_ratio": non_manifold_edges_ratio,
            "manifold_edges_count": manifold,
            "winding_consistent_ratio": winding_consistent_ratio,
            "extents": [float(x) for x in extents],
            "aspect_ratio_max": aspect_ratio_max,
            "thinness": thinness,
            "bbox_volume": bbox_volume,
            "bbox_diag": bbox_diag,
            "centroid_offset_norm": centroid_offset_norm,
            "hull_volume": hull_volume,
            "hull_solidity": hull_solidity,
            "degen_face_ratio": degen_ratio,
            "scan_time_s": scan_time,
            "rss_peak_mb": round(rss_peak, 0),
            "error": None,
        }

        # Add hard-hit booleans for convenience.
        for metric, op, thr, short, _ in RED_FLAGS:
            val = rec[metric]
            hit = (op == ">=" and val >= thr) or (op == "<=" and val <= thr)
            rec[f"{short}_hit"] = bool(hit)

        if verbose:
            print(
                f"  {obj_id} V={n_verts} F={n_faces} comps={n_components} "
                f"watertight={is_watertight} bedge={boundary} nm={non_manifold} "
                f"solid={hull_solidity:.2f} thin={thinness:.2f} "
                f"rss={rss_peak:.0f}MB t={scan_time}s",
                flush=True,
            )
        return rec

    except MemoryError as e:
        gc.collect()
        return {
            "domain": domain,
            "class": class_name,
            "obj_id": obj_id,
            "error": "MemoryError: " + str(e),
        }
    except Exception as e:
        gc.collect()
        return {
            "domain": domain,
            "class": class_name,
            "obj_id": obj_id,
            "error": f"{type(e).__name__}: {e}",
            "scan_time_s": round(time.time() - t0, 2),
        }
