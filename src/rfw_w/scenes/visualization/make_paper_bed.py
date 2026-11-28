#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Make a small, figure-only upholstered bed inside the delivered bed footprint.

The authored asset is intentionally separate from the production delivery. The
render provenance records both OBJ hashes and the original URI hashes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


MATERIALS = {
    "walnut": (0.39, 0.27, 0.20),
    "linen": (0.78, 0.70, 0.60),
    "sheet": (0.92, 0.89, 0.82),
    "sage": (0.35, 0.49, 0.46),
    "sage_light": (0.46, 0.58, 0.53),
    "pillow": (0.96, 0.95, 0.90),
    "stitch": (0.82, 0.76, 0.66),
}


def box(extents, centre):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(centre)
    return mesh


def rounded_cushion(extents, centre):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale(np.asarray(extents) / 2.0)
    mesh.apply_translation(centre)
    return mesh


def make_bed(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[tuple[str, trimesh.Trimesh]] = []
    add = pieces.append
    # Local x +/-0.75, y +/-1.00; after the delivery's 180-degree rotation,
    # the headboard faces the north wall and the nightstand sits beside it.
    add(("walnut", box((1.48, 1.93, 0.16), (0, 0, 0.17))))
    add(("linen", box((1.43, 1.82, 0.20), (0, 0.01, 0.35))))
    add(("sheet", box((1.37, 1.73, 0.10), (0, 0.02, 0.49))))
    add(("walnut", box((1.48, 0.09, 0.65), (0, -0.94, 0.53))))
    add(("linen", box((1.39, 0.065, 0.48), (0, -0.875, 0.59))))
    # The duvet and its folded edge are separate tones so the top view reads
    # as bedding, rather than the dark untextured rectangle in the source bed.
    add(("sage", box((1.35, 1.10, 0.07), (0, 0.32, 0.575))))
    add(("sage_light", box((1.35, 0.16, 0.045), (0, -0.30, 0.61))))
    for x in (-0.37, 0.37):
        add(("pillow", rounded_cushion((0.56, 0.37, 0.13), (x, -0.62, 0.59))))
    for x in (-0.68, 0.68):
        add(("walnut", box((0.045, 1.89, 0.12), (x, 0, 0.16))))
    visual = output_dir / "paper_bed.obj"
    mtl = output_dir / "paper_bed.mtl"
    mtl.write_text("".join(
        f"newmtl {name}\nKd {rgb[0]} {rgb[1]} {rgb[2]}\nKa 0 0 0\nKs 0.04 0.04 0.04\nNs 8\n\n"
        for name, rgb in MATERIALS.items()))
    lines = ["# figure-only upholstered bed\n", f"mtllib {mtl.name}\n"]
    offset = 0
    for name, mesh in pieces:
        lines.append(f"o {name}_{offset}\nusemtl {name}\n")
        lines.extend(f"v {x:.7f} {y:.7f} {z:.7f}\n" for x, y, z in mesh.vertices)
        lines.extend(f"f {a+offset+1} {b+offset+1} {c+offset+1}\n"
                     for a, b, c in mesh.faces)
        offset += len(mesh.vertices)
    visual.write_text("".join(lines))
    collision = output_dir / "paper_bed_collision.obj"
    collision.write_text(trimesh.creation.box(
        extents=(1.48, 1.93, 0.82), transform=trimesh.transformations.translation_matrix(
            (0, 0, 0.41))).export(file_type="obj"))
    return visual, collision


def make_jar(output_dir: Path) -> tuple[Path, Path]:
    """A 15 cm ceramic jar with a readable lid, within the source target bounds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {"clay": (0.65, 0.43, 0.31), "label": (0.89, 0.83, 0.69),
               "lid": (0.34, 0.26, 0.23), "rim": (0.77, 0.63, 0.47)}
    pieces = []
    for name, radius, height, z in (
            ("clay", 0.058, 0.108, -0.009),
            ("label", 0.059, 0.035, -0.014),
            ("rim", 0.061, 0.009, 0.051),
            ("lid", 0.068, 0.020, 0.060)):
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
        mesh.apply_translation((0, 0, z))
        pieces.append((name, mesh))
    visual = output_dir / "paper_jar.obj"
    mtl = output_dir / "paper_jar.mtl"
    mtl.write_text("".join(
        f"newmtl {name}\nKd {c[0]} {c[1]} {c[2]}\nKa 0 0 0\nKs 0.08 0.08 0.08\nNs 20\n\n"
        for name, c in palette.items()))
    lines = ["# figure-only ceramic jar\n", f"mtllib {mtl.name}\n"]
    offset = 0
    for name, mesh in pieces:
        lines.append(f"o {name}_{offset}\nusemtl {name}\n")
        lines.extend(f"v {x:.7f} {y:.7f} {z:.7f}\n" for x, y, z in mesh.vertices)
        lines.extend(f"f {a+offset+1} {b+offset+1} {c+offset+1}\n"
                     for a, b, c in mesh.faces)
        offset += len(mesh.vertices)
    visual.write_text("".join(lines))
    collision = output_dir / "paper_jar_collision.obj"
    collision.write_text(trimesh.creation.cylinder(radius=0.068, height=0.138,
                                                    sections=24).export(file_type="obj"))
    return visual, collision
