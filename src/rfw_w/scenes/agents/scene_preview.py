# SPDX-License-Identifier: Apache-2.0
"""Local, asset-free review sketch of a compiled task cell and fitted shell."""
from html import escape

from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.room import fit_room_around_cell


def preview_scene(spec) -> tuple[str, dict]:
    cell = solve_cell(spec, seed=spec.seed)
    fit = fit_room_around_cell(spec, cell, room_seed=spec.seed)
    shell = fit.shell
    points = shell.polygon_xy
    xmin, ymin = min(x for x, y in points), min(y for x, y in points)
    xmax, ymax = max(x for x, y in points), max(y for x, y in points)
    scale = min(720 / (xmax - xmin), 520 / (ymax - ymin))

    def polygon(vertices, fill, stroke):
        coords = " ".join(f"{40+(x-xmin)*scale:.2f},{70+(ymax-y)*scale:.2f}" for x, y in vertices)
        return f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'

    tx, ty = fit.cell_translation_xy
    support = [(x + tx, y + ty) for x, y in cell.support_footprint.vertices]
    target_x, target_y = cell.target_offset[0] + tx, cell.target_offset[1] + ty
    def dot(x, y, radius, colour):
        return f'<circle cx="{40+(x-xmin)*scale:.2f}" cy="{70+(ymax-y)*scale:.2f}" r="{radius}" fill="{colour}"/>'
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="650" viewBox="0 0 800 650">',
           '<rect width="800" height="650" fill="white"/>',
           f'<text x="30" y="30" font-family="sans-serif" font-size="20">{escape(spec.scene_id)}</text>',
           '<text x="30" y="53" font-family="sans-serif" font-size="13">Room outline and protected task floor; no assets, physics or grasp verified.</text>',
           polygon(points, "#f4f4f4", "#333"),
           polygon(fit.protected_in_room.vertices, "#c9e5f5", "#246a90"),
           polygon(support, "#bd9368", "#614a30"),
           dot(target_x, target_y, 5, "#c43c39"),
           *[dot(item.centre[0] + tx, item.centre[1] + ty, 2.5, "#246a90") for item in cell.standoffs],
           '<text x="30" y="625" font-family="sans-serif" font-size="14">Blue: protected floor / approach samples. Brown: support. Red: target. Grey: room.</text>',
           '</svg>']
    return "\n".join(svg), {"status": "ran", "method": "asset_free_cell_and_room_fit",
                             "core_hash": cell.core_hash, "room_family": fit.options.family.value}
