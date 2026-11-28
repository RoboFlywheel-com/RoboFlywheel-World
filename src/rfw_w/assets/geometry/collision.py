# SPDX-License-Identifier: Apache-2.0
"""Crash-isolated CoACD with individually preserved collision hulls."""

from __future__ import annotations
from pathlib import Path


def decompose_parts(mesh, face_ids, part_ids, output_dir, timeout=600, seed=0):
    import trimesh
    from .coacd import decompose_convex_mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    hulls = []
    for pid in part_ids:
        sub = mesh.submesh([face_ids == pid], append=True)
        if sub is None or not len(sub.faces):
            raise ValueError("part has no faces")
        source = output_dir / f"part-{pid}-source.obj"
        merged = output_dir / f"part-{pid}-merged.obj"
        sub.export(source)
        decompose_convex_mp(str(source), str(merged), timeout=timeout, seed=seed)
        pieces = trimesh.load(merged, force="mesh").split(only_watertight=False)
        if not len(pieces):
            raise ValueError("CoACD produced no collision hulls")
        paths = []
        for i, piece in enumerate(pieces):
            path = output_dir / f"part-{pid}-hull-{i}.obj"
            piece.export(path)
            paths.append(path)
            hulls.append(path)
        records[pid] = {
            "hulls": len(paths),
            "convex_volume_m3": sum(abs(part.volume) for part in pieces),
            "paths": paths,
        }
        source.unlink()
        merged.unlink()
        Path(str(merged) + ".coacd.log").unlink(missing_ok=True)
    if not hulls:
        raise ValueError("no collision hulls")
    return records, hulls
