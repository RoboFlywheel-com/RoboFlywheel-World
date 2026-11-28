# SPDX-License-Identifier: Apache-2.0
"""Collision budget for one explicitly supplied asset directory."""

import argparse

import glob

import json

import os

import sys

THRESHOLDS = (0.05, 0.02, 0.01)

NONCONVEX_RATIO = 1.05


def _load_hulls(path):
    """Load a merged hull OBJ and split into convex pieces by connectivity."""
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    pieces = mesh.split(only_watertight=False)
    if isinstance(pieces, trimesh.Trimesh):
        pieces = [pieces]
    return [p for p in pieces if len(p.vertices) > 0 and len(p.faces) > 0]


def _is_nonconvex(visual_obj):
    """True when the visual mesh is clearly not its own convex hull.

    Returns True when the volume ratio cannot be computed (non-watertight
    input etc.) -- retrying is the safe choice there.
    """
    import numpy as np
    import trimesh

    try:
        mesh = trimesh.load(visual_obj, force="mesh")
        ratio = float(mesh.convex_hull.volume) / float(abs(mesh.volume))
        return bool(not np.isfinite(ratio) or ratio > NONCONVEX_RATIO)
    except Exception:  # noqa: BLE001
        return True


def redecompose(asset_dir, min_hulls=4, max_hulls=64, threshold=None, verbose=False):
    """Build the teleop collision set for one asset. Returns (key, status, detail)."""
    key = os.path.basename(asset_dir.rstrip("/"))
    from rfw_w.assets.catalog.release import visual_mesh

    visual_obj = str(visual_mesh(asset_dir, key))
    out_obj = os.path.join(asset_dir, "mesh", f"{key}_teleop_collision.obj")
    sidecar_path = os.path.join(asset_dir, "mesh", f"{key}_teleop_collision.json")
    hull_dir = os.path.join(asset_dir, "mjcf", "mesh")
    if not os.path.isfile(visual_obj):
        return key, "fail", f"no visual mesh {visual_obj}"
    if (
        os.path.isfile(out_obj)
        and os.path.isfile(sidecar_path)
        and os.path.getmtime(out_obj) >= os.path.getmtime(visual_obj)
    ):
        return key, "skip", "up-to-date"

    from rfw_w.assets.geometry.coacd import decompose_convex_mp

    nonconvex = _is_nonconvex(visual_obj)
    best = None  # (threshold, pieces) with the most hulls, as fallback
    used = None
    thresholds = (threshold,) if threshold is not None else THRESHOLDS
    for thr in thresholds:
        tmp_obj = f"{out_obj}.thr{thr}.tmp.obj"
        try:
            decompose_convex_mp(
                visual_obj,
                tmp_obj,
                threshold=thr,
                max_convex_hull=max_hulls,
                verbose=verbose,
            )
            pieces = _load_hulls(tmp_obj)
        except Exception as e:  # noqa: BLE001
            pieces = None
            if verbose:
                print(f"coacd failed at threshold={thr}: {e}")
        finally:
            for f in (tmp_obj, tmp_obj + ".coacd.log"):
                if os.path.isfile(f):
                    os.remove(f)
        if not pieces:
            continue
        if best is None or len(pieces) > len(best[1]):
            best = (thr, pieces)
        if len(pieces) >= min_hulls or not nonconvex:
            used = (thr, pieces)
            break
    if used is None:
        used = best  # lower bound unreachable: keep the richest decomposition
    if used is None:
        return key, "fail", "coacd failed at all thresholds"
    thr, pieces = used

    os.makedirs(hull_dir, exist_ok=True)
    # drop stale per-hull files first: a re-run may produce FEWER hulls
    for f in glob.glob(os.path.join(hull_dir, f"{key}_teleop_collision_*.obj")):
        os.remove(f)
    for i, piece in enumerate(pieces):
        piece.export(os.path.join(hull_dir, f"{key}_teleop_collision_{i}.obj"))

    import trimesh

    scene = trimesh.Scene()
    for i, piece in enumerate(pieces):
        scene.add_geometry(piece, geom_name=f"hull_{i}")
    scene.export(out_obj)

    per_hull_verts = sorted((len(p.vertices) for p in pieces), reverse=True)
    sidecar = {
        "n_hulls": len(pieces),
        "threshold_used": thr,
        "min_hulls": min_hulls,
        "max_hulls": max_hulls,
        "min_hulls_met": len(pieces) >= min_hulls,
        "nonconvex": nonconvex,
        "per_hull_verts": per_hull_verts,
        "over_64vert_hulls": sum(1 for v in per_hull_verts if v > 64),
        "source_mesh": os.path.relpath(visual_obj, asset_dir),
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    return (
        key,
        "ok",
        (
            f"{len(pieces)} hulls (threshold={thr}, "
            f"max_vert={per_hull_verts[0]}) -> {out_obj}"
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--max-hulls", type=int, default=32)
    args = parser.parse_args()
    print(redecompose(args.asset_dir, max_hulls=args.max_hulls))


if __name__ == "__main__":
    main()
