#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A figure-only work table of one declared size, for every room to share.

SPEC-016. The three rooms' task supports are three retrieved assets with three
different tops -- the office desk at 0.750 m, the kitchen island at 0.900 m and
the bedroom nightstand at 0.575 m, with footprints from 1.35 x 0.74 down to
0.60 x 0.45. Read as a set, the rows do not compare: the same arm and the same
props sit at three heights on three shapes.

So the figure substitutes one constructed table into each room, the way
``make_paper_bed`` substitutes a bed: same size, same height, same colour,
placed where the delivered support stood. The delivery itself is untouched, and
the substitution is recorded with both meshes' digests.

Its size is a declaration, not a measurement -- ``TABLE_SIZE_M`` below carries
why those numbers and not others.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

#: Width, depth, top height. The height is the office desk's own 0.750 m, which is
#: an ordinary work height and the one the arm's poses were swept against. The
#: footprint is the largest that fits **all three** rooms once the bedroom's table
#: moves off the nightstand's corner into the room's open middle: measured on the
#: three deliveries, 1.20 x 0.70 m clears the bed (x >= 3.095), the wardrobe
#: (y <= 0.602) and the doorway strip (y >= 3.373) there, and still leaves the
#: office and kitchen tables inside their rooms.
TABLE_SIZE_M = (1.20, 0.70, 0.75)

TOP_THICKNESS_M = 0.04
LEG_M = 0.06
MATERIALS = {"top": (0.62, 0.46, 0.31), "leg": (0.38, 0.28, 0.20), "rail": (0.45, 0.33, 0.23)}


def make_table(output_dir: Path, size_m: tuple[float, float, float] = TABLE_SIZE_M
               ) -> tuple[Path, Path]:
    """Write the table's visual and collision OBJ; return both paths.

    The mesh is authored about its own centre in x/y with its feet at z = 0, so
    it drops straight into the pose the delivered support already had.
    """
    width, depth, height = (float(v) for v in size_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[tuple[str, trimesh.Trimesh]] = []

    def box(extents, centre, material):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(centre)
        pieces.append((material, mesh))

    box((width, depth, TOP_THICKNESS_M), (0, 0, height - TOP_THICKNESS_M / 2), "top")
    inset = LEG_M / 2 + 0.03
    leg_height = height - TOP_THICKNESS_M
    for sx in (-1, 1):
        for sy in (-1, 1):
            box((LEG_M, LEG_M, leg_height),
                (sx * (width / 2 - inset), sy * (depth / 2 - inset), leg_height / 2), "leg")
    # A rail under the top, so the table reads as furniture from the dollhouse
    # camera instead of as a slab on four sticks.
    box((width - 0.10, 0.04, 0.06), (0, depth / 2 - inset, leg_height - 0.06), "rail")
    box((width - 0.10, 0.04, 0.06), (0, -(depth / 2 - inset), leg_height - 0.06), "rail")

    visual = output_dir / "paper_table.obj"
    mtl = output_dir / "paper_table.mtl"
    mtl.write_text("".join(
        f"newmtl {name}\nKd {rgb[0]} {rgb[1]} {rgb[2]}\nKa 0 0 0\nKs 0.04 0.04 0.04\nNs 12\n\n"
        for name, rgb in MATERIALS.items()))
    lines = [f"# figure-only work table {width} x {depth} x {height} m\n", f"mtllib {mtl.name}\n"]
    offset = 0
    for name, mesh in pieces:
        lines.append(f"o {name}_{offset}\nusemtl {name}\n")
        lines.extend(f"v {x:.7f} {y:.7f} {z:.7f}\n" for x, y, z in mesh.vertices)
        lines.extend(f"f {a+offset+1} {b+offset+1} {c+offset+1}\n" for a, b, c in mesh.faces)
        offset += len(mesh.vertices)
    visual.write_text("".join(lines))

    collision = output_dir / "paper_table_collision.obj"
    # The collision proxy is the table's own shape, not a solid block. A block
    # from floor to top is a fair *physics* proxy and a bad *picture*: measured
    # 2026-09-26 on the office, the collision pass then reported
    # TARGET_NOT_VISIBLE for the cup and lost both flat books, because a small
    # object resting a couple of millimetres above a large coplanar face is not
    # what a renderer resolves in its favour. Nothing here runs physics, so the
    # honest proxy is the shape the figure is about.
    collision.write_text(visual.read_text(encoding="utf-8"), encoding="utf-8")
    return visual, collision


def table_bounds(size_m: tuple[float, float, float] = TABLE_SIZE_M
                 ) -> tuple[list[float], list[float]]:
    """The table's local AABB, in the form the placement rules read."""
    width, depth, height = (float(v) for v in size_m)
    return [-width / 2, -depth / 2, 0.0], [width / 2, depth / 2, height]
