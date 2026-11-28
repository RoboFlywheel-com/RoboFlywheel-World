# SPDX-License-Identifier: Apache-2.0
"""One rule for "does this piece stand in the task's reservation", read by two tools.

`rfw_w.scenes.geometry.reservations` exists because the ruler
(`scripts/ops/measure_generated_furniture.py`, which measures a built scene) and the build's own
rejection pool (`scripts/ops/infinigen_constraint_patch.py`, which deletes what may not stay) ask
the same question -- and a disagreement between them would look exactly like an intruder. The tests
below pin the three parts of the rule that can be got subtly wrong, with stand-in objects rather
than Blender: the hull footprint, the band test, and *when* the rounding happens (once per region,
not once per polygon, or a four-ring corridor would drift in the fourth decimal from the number
already on record).

The module is deliberately free of `bpy`, which is also what makes these tests possible; the last
test pins that, so a future edit that imports Blender cannot quietly make the rule untestable here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rfw_w.scenes.geometry.reservations import (
    AREA_PLACES,
    Part,
    Region,
    footprint,
    measure_part,
    measure_parts,
    overlap_m2,
    world_vertices,
)
from rfw_w.scenes.geometry.polygon import Polygon

REPO_ROOT = Path(__file__).resolve().parents[3]


class Point:
    """What `matrix @ vertex.co` returns: three attributes, the way Blender's `Vector` reads."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)


class Translation:
    """The only matrix operation the rule needs: a world offset."""

    def __init__(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        self.offset = (dx, dy, dz)

    def __matmul__(self, co) -> Point:
        return Point(co.x + self.offset[0], co.y + self.offset[1], co.z + self.offset[2])


class Mesh:
    def __init__(self, vertices) -> None:
        self.vertices = [type('Vertex', (), {'co': Point(*vertex)})() for vertex in vertices]


class StandIn:
    """A Blender object as this rule sees it: a name, a mesh, and where it is in the world."""

    def __init__(self, name='Obj', vertices=(), offset=(0.0, 0.0, 0.0)) -> None:
        self.name = name
        self.data = Mesh(vertices)
        self.matrix_world = Translation(*offset)


def box(x0, y0, x1, y1, *, z0=0.0, z1=0.5, name='Box', offset=(0.0, 0.0, 0.0)) -> StandIn:
    """An axis-aligned box's eight corners, which is what a factory's mesh reduces to here."""
    corners = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    return StandIn(name=name, vertices=corners, offset=offset)


def square(size: float, *, x=0.0, y=0.0) -> Polygon:
    return Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])


# ---------------------------------------------------------------------------------------------
# The footprint
# ---------------------------------------------------------------------------------------------

def test_the_footprint_is_the_hull_of_the_world_vertices_not_the_local_box():
    piece = box(0.0, 0.0, 0.4, 0.2, offset=(1.0, 2.0, 0.0))
    hull = footprint(world_vertices(piece))
    assert hull.vertices == ((1.0, 2.0), (1.4, 2.0), (1.4, 2.2), (1.0, 2.2))
    # A piece with no floor area at all (a column whose vertices are collinear in plan) has no
    # footprint: `None` is an answer, and it is not "zero overlap by measurement".
    assert footprint([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]) is None
    assert footprint([(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]) is None


def test_an_object_with_no_mesh_is_not_a_zero_overlap_part():
    assert measure_part(StandIn('empty'), [Region('protected', square(1.0), 0.6)]) is None
    assert world_vertices(StandIn('empty')) == ()
    assert measure_parts([StandIn('a'), box(0, 0, 0.1, 0.1)], [])[0].name == 'Box'


# ---------------------------------------------------------------------------------------------
# The band rule
# ---------------------------------------------------------------------------------------------

def test_a_piece_that_hangs_above_the_band_is_not_in_the_region():
    regions = [Region('protected', square(1.0), 0.575)]
    standing = box(0.2, 0.2, 0.8, 0.8, z0=0.0, z1=0.9)
    hanging = box(0.2, 0.2, 0.8, 0.8, z0=2.0, z1=2.4)
    assert measure_part(standing, regions).overlaps_m2 == {'protected': 0.36}
    # The office's wall art at 1.45 m over a 0.75 m desk and the bedroom's pendant at 2.528 m over
    # a 0.575 m nightstand hang *over* the reservation; only what reaches down into it stands in it.
    part = measure_part(hanging, regions)
    assert part.overlaps_m2 == {'protected': 0.0}
    assert part.reaches(2.5) and not part.in_band(regions[0])
    # The same piece under the corridor's own 1.8 m band is a different question, asked separately.
    corridor = [Region('circulation', square(1.0), 1.8)]
    assert measure_part(hanging, corridor).overlaps_m2 == {'circulation': 0.0}
    assert measure_part(box(0.2, 0.2, 0.8, 0.8, z0=1.0, z1=2.4), corridor).overlaps_m2 == {
        'circulation': 0.36}


def test_a_region_is_summed_over_its_polygons_and_rounded_once():
    """A corridor is often four rings. Rounding each ring first and adding would drift."""
    strip = box(0.0, 0.0, 3.0, 0.12346)
    # Two 1.0 x 0.12346 m rings of one region: each is 0.12346 m2, and 0.12346 rounds to 0.1235.
    # Summed first and rounded once the answer is 0.2469; rounded per ring it would be 0.2470.
    rings = [Region('circulation', Polygon([(x, 0.0), (x + 1.0, 0.0), (x + 1.0, 0.12346),
                                            (x, 0.12346)]), 1.8) for x in (0.0, 2.0)]
    measured = measure_part(strip, rings)
    assert measured.overlaps_m2 == {'circulation': 0.2469}
    assert round(0.12346, 4) * 2 == 0.2470  # what the per-ring rounding would have produced
    single = Region('circulation', square(1.0), 1.8)
    assert overlap_m2(square(0.25), [single]) == {'circulation': 0.0625}
    # A splash of float debris is not an intrusion: areas below the fourth decimal read as zero.
    assert measure_part(box(0.0, 0.0, 0.0001, 0.0001), [single]).overlaps_m2 == {
        'circulation': 0.0}


def test_a_region_has_to_be_named_and_to_reach_somewhere():
    with pytest.raises(ValueError, match='needs a name'):
        Region('', square(1.0), 0.6)
    with pytest.raises(ValueError, match='positive band'):
        Region('protected', square(1.0), 0.0)


def test_a_part_reports_which_regions_it_stands_in(clean_part=Part(
        name='Shelf', base_z_m=0.05, top_z_m=2.01, footprint_extent_m=(0.83, 0.36),
        overlaps_m2={'protected': 0.1265, 'circulation': 0.0315, 'door_sweep': 0.0})):
    assert clean_part.intrudes
    assert clean_part.overlapping() == (('protected', 0.1265), ('circulation', 0.0315))
    assert 'overlaps_m2' in clean_part.to_dict() and clean_part.to_dict()['base_z_m'] == 0.05


def test_the_rule_carries_no_blender_import():
    source = (REPO_ROOT / 'src/rfw_w/scenes/geometry/reservations.py').read_text()
    assert 'import bpy' not in source and 'from bpy' not in source
    assert AREA_PLACES == 4
