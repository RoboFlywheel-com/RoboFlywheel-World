# SPDX-License-Identifier: Apache-2.0
"""Stage geometry authoring.

The polygon-based floor and the Y-up correction are the two things that would silently render the
wrong room or lay every object on its side, so both are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfw_w.scenes.workers.isaac.geometry import (
    GeometryError,
    WallOpening,
    box_mesh,
    extrude_polygon,
    load_obj,
    match_openings_to_edges,
    triangulated_floor,
    wall_panels,
    y_up_to_z_up_matrix,
)


def write_obj(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


class TestObjLoading:
    def test_reads_points_and_faces(self, tmp_path: Path) -> None:
        path = write_obj(tmp_path / "a.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        obj = load_obj(path)
        assert len(obj.points) == 3
        assert obj.faces == [[0, 1, 2]]

    def test_indices_are_converted_from_one_based(self, tmp_path: Path) -> None:
        """Getting this wrong produces a mesh that renders without complaint and is wrong."""
        path = write_obj(tmp_path / "a.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 3 2 1\n")
        assert load_obj(path).faces == [[2, 1, 0]]

    def test_negative_indices_are_relative(self, tmp_path: Path) -> None:
        path = write_obj(tmp_path / "a.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
        assert load_obj(path).faces == [[0, 1, 2]]

    def test_ignores_normals_but_keeps_texture_coordinates(self, tmp_path: Path) -> None:
        """Normals are unnecessary; texture coordinates are not.

        Dropping the UVs is what left fully textured assets rendering as flat colour: the files
        carry a ``vt`` per vertex and a material library beside them, and a loader that read only
        ``v`` and ``f`` discarded all of it without saying so.
        """
        path = write_obj(
            tmp_path / "a.obj",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvt 0.5 0.25\nf 1/1/1 2/1/1 3/1/1\n",
        )
        obj = load_obj(path)
        assert len(obj.points) == 3
        assert obj.faces == [[0, 1, 2]]
        assert obj.texcoords == [(0.5, 0.25)]
        assert obj.face_texcoords == [[0, 0, 0]]

    def test_rejects_an_out_of_range_index(self, tmp_path: Path) -> None:
        path = write_obj(tmp_path / "a.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n")
        with pytest.raises(GeometryError, match="out of range"):
            load_obj(path)

    def test_rejects_a_mesh_with_no_vertices(self, tmp_path: Path) -> None:
        path = write_obj(tmp_path / "a.obj", "# nothing here\n")
        with pytest.raises(GeometryError, match="no vertices"):
            load_obj(path)

    def test_resolves_the_diffuse_texture_from_the_material_library(self, tmp_path: Path) -> None:
        (tmp_path / "material.mtl").write_text(
            "newmtl material_0\nmap_Kd material_0.png\n"
        )
        # A real-sized image: a map smaller than 64 px is treated as a synthesised placeholder and
        # refused, so a one-byte fixture would test the guard rather than the resolution.
        from PIL import Image

        Image.new("RGB", (128, 128), (200, 120, 60)).save(tmp_path / "material_0.png")
        path = write_obj(
            tmp_path / "a.obj",
            "mtllib material.mtl\nusemtl material_0\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1\n",
        )
        assert load_obj(path).diffuse_texture(path) == tmp_path / "material_0.png"

    def test_a_placeholder_sized_map_is_refused(self, tmp_path: Path) -> None:
        """trimesh synthesises a 16x8 image from a base-colour factor when a material has no map.

        Binding that as a texture renders the object the placeholder's colour -- which is how a desk
        came out black -- so a map this small is treated as absent and the colour factor is used.
        """
        from PIL import Image

        (tmp_path / "material.mtl").write_text("newmtl material_0\nmap_Kd material_0.png\nKd 0.6 0.4 0.2\n")
        Image.new("RGB", (16, 8), (8, 8, 8)).save(tmp_path / "material_0.png")
        path = write_obj(
            tmp_path / "a.obj",
            "mtllib material.mtl\nusemtl material_0\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1\n",
        )
        assert load_obj(path).diffuse_texture(path) is None
        assert load_obj(path).material_colour(path) == (0.6, 0.4, 0.2)

    def test_an_absolute_texture_reference_resolves_to_nothing(self, tmp_path: Path) -> None:
        """A package must not depend on where it was built."""
        (tmp_path / "material.mtl").write_text(
            "newmtl material_0\nmap_Kd /somewhere/else/material_0.png\n"
        )
        path = write_obj(
            tmp_path / "a.obj",
            "mtllib material.mtl\nusemtl material_0\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1\n",
        )
        assert load_obj(path).diffuse_texture(path) is None

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_obj(tmp_path / "absent.obj")


class TestRoomSolids:
    def test_floor_keeps_the_notch(self) -> None:
        """The floor is the real outline, so a notched room has a notched floor.

        Building the floor from the bounding extent would render a room the scene does not have,
        and the delivered image would then disagree with the geometry the gates validated.
        """
        l_shape = ((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (3.0, 2.0), (3.0, 3.0), (0.0, 3.0))
        points, faces = triangulated_floor(l_shape, z=0.0)
        assert len(points) == 6
        assert self._mesh_area(points, faces) == pytest.approx(11.0)

    def _mesh_area(self, points, faces) -> float:
        """The area the emitted triangles actually cover, in XY.

        The vertices are ``(x, y, z)``; the third component is dropped because the floor is flat.
        """
        total = 0.0
        for a, b, c in faces:
            ax, ay = points[a][0], points[a][1]
            bx, by = points[b][0], points[b][1]
            cx, cy = points[c][0], points[c][1]
            total += abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2
        return total

    def test_every_notch_corner_triangulates_to_the_polygon_area(self) -> None:
        """The fan bug, as a property. Counting vertices and faces does not catch it.

        A fan from vertex 0 is correct only when every vertex can see vertex 0. A notch in the
        *top-right* corner satisfies that; the same notch in any other corner does not, and the
        fan then lays floor over the notch. Measured on this outline (5.0 x 4.0, notch 1.5 x 1.4,
        so 17.9 m2 of real floor): the fan emits 17.9 m2 top-right and 20.0 m2 top-left or
        bottom-right. The polygon is right either way, so `validate_shell` passes and the
        delivered render shows a floor where the room has none -- which is why this is asserted
        as area rather than as a face count.
        """
        width, depth, notch_w, notch_d = 5.0, 4.0, 1.5, 1.4
        real_area = width * depth - notch_w * notch_d
        corners = {
            "top_right": ((0.0, 0.0), (width, 0.0), (width, depth - notch_d),
                          (width - notch_w, depth - notch_d), (width - notch_w, depth),
                          (0.0, depth)),
            "top_left": ((0.0, 0.0), (width, 0.0), (width, depth), (notch_w, depth),
                         (notch_w, depth - notch_d), (0.0, depth - notch_d)),
            "bottom_right": ((0.0, 0.0), (width - notch_w, 0.0), (width - notch_w, notch_d),
                             (width, notch_d), (width, depth), (0.0, depth)),
            "bottom_left": ((notch_w, 0.0), (width, 0.0), (width, depth), (0.0, depth),
                            (0.0, notch_d), (notch_w, notch_d)),
        }
        for name, outline in corners.items():
            points, faces = triangulated_floor(outline, z=0.0)
            assert self._mesh_area(points, faces) == pytest.approx(real_area, abs=1e-9), (
                f"the {name} notch triangulated to the wrong area: the mesh covers floor the "
                f"room does not have"
            )

    def test_floor_rejects_a_self_intersecting_outline(self) -> None:
        """A bow tie has a meaningless area, and a mesh built from it would inherit the error."""
        with pytest.raises(GeometryError, match="simple polygon"):
            triangulated_floor(((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)), z=0.0)

    def test_floor_rejects_too_few_vertices(self) -> None:
        with pytest.raises(GeometryError, match="at least 3"):
            triangulated_floor(((0.0, 0.0), (1.0, 0.0)), z=0.0)

    def test_prism_has_a_top_a_bottom_and_sides(self) -> None:
        points, faces = extrude_polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                                        base_z=0.0, height=2.0)
        assert len(points) == 8
        assert len(faces) == 2 + 4
        zs = {round(p[2], 6) for p in points}
        assert zs == {0.0, 2.0}

    def test_prism_rejects_non_positive_height(self) -> None:
        with pytest.raises(GeometryError, match="positive"):
            extrude_polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), base_z=0.0, height=0.0)

    def test_box_rejects_degenerate_extent(self) -> None:
        with pytest.raises(GeometryError, match="positive extent"):
            box_mesh((0, 0, 0), (0, 1, 1))

    def test_box_has_eight_points(self) -> None:
        points, faces = box_mesh((0, 0, 0), (1, 2, 3))
        assert len(points) == 8
        assert len(faces) == 6


class TestWallPanels:
    """The wall segmenter, against several openings at once.

    The shell had exactly one door and at most one window while this was written, so the code could
    have been -- and was -- "subtract the one opening". A room now gets up to three windows, and
    they are not twins: a wall may carry two with different sills. These are the cases that were
    never exercised, and the coverage identity is what catches a panel emitted twice.
    """

    WALL_LENGTH = 6.0
    WALL_HEIGHT = 2.8

    def _panels(self, *openings: WallOpening):
        return wall_panels(self.WALL_LENGTH, openings, wall_height_m=self.WALL_HEIGHT)

    @staticmethod
    def _panel_area(panels) -> float:
        return sum(
            (panel.end_m - panel.start_m) * (panel.top_m - panel.bottom_m)
            for panel in panels
            if not panel.is_degenerate
        )

    @staticmethod
    def _opening_area(openings) -> float:
        return sum(opening.width_m * opening.height_m for opening in openings)

    def _assert_covers(self, openings) -> None:
        """Panels cover the wall exactly once, minus the openings and nothing else.

        The identity, not a face count: a segmenter that emits a panel twice, or drops one, has the
        right *number* of pieces in many of the ways it can go wrong.
        """
        panels = self._panels(*openings)
        assert self._panel_area(panels) == pytest.approx(
            self.WALL_LENGTH * self.WALL_HEIGHT - self._opening_area(openings), abs=1e-9
        ), f"{len(panels)} panels: {[(p.start_m, p.end_m, p.bottom_m, p.top_m) for p in panels]}"

    def test_a_door_leaves_a_lintel_above_it(self) -> None:
        door = WallOpening("door_main", offset_along_m=3.0, width_m=0.9, height_m=2.05)
        panels = self._panels(door)
        assert len(panels) == 3
        above = [p for p in panels if p.bottom_m >= door.height_m - 1e-9]
        assert len(above) == 1, "a door needs no panel below it, and one above"
        self._assert_covers((door,))

    def test_a_window_leaves_a_wall_below_and_above_it(self) -> None:
        window = WallOpening("window_01", offset_along_m=3.0, width_m=1.2, height_m=1.2, sill_m=0.9)
        panels = self._panels(window)
        assert len(panels) == 4
        assert any(p.top_m <= window.sill_m + 1e-9 for p in panels), "no wall under the window"
        assert any(p.bottom_m >= window.sill_m + window.height_m - 1e-9 for p in panels)
        self._assert_covers((window,))

    def test_two_windows_with_different_sills_on_one_wall(self) -> None:
        """The case a room with three windows actually produces."""
        first = WallOpening("window_01", offset_along_m=1.5, width_m=1.1, height_m=1.2, sill_m=0.81)
        second = WallOpening("window_02", offset_along_m=4.5, width_m=1.3, height_m=1.2, sill_m=1.06)
        self._assert_covers((first, second))
        # Each window keeps its own sill: wall under it, wall above it, at *its* height. The two
        # sills differ, so a segmenter that read one opening's sill for both would show up here.
        panels = self._panels(first, second)
        for window in (first, second):
            under = [p for p in panels if abs(p.top_m - window.sill_m) < 1e-9
                     and p.start_m <= window.start_m + 1e-9 and p.end_m >= window.end_m - 1e-9]
            above = [p for p in panels if abs(p.bottom_m - window.top_m) < 1e-9
                     and p.start_m <= window.start_m + 1e-9 and p.end_m >= window.end_m - 1e-9]
            assert under, f"{window.opening_id}: no wall below it at its sill"
            assert above, f"{window.opening_id}: no wall above it"

    def test_a_door_and_a_window_on_one_wall(self) -> None:
        door = WallOpening("door_main", offset_along_m=1.2, width_m=0.9, height_m=2.05)
        window = WallOpening("window_01", offset_along_m=4.4, width_m=1.4, height_m=1.2, sill_m=0.9)
        self._assert_covers((door, window))

    def test_three_openings(self) -> None:
        self._assert_covers((
            WallOpening("door_main", offset_along_m=0.8, width_m=0.9, height_m=2.05),
            WallOpening("window_01", offset_along_m=3.0, width_m=1.0, height_m=1.2, sill_m=0.8),
            WallOpening("window_02", offset_along_m=5.0, width_m=0.8, height_m=1.2, sill_m=1.0),
        ))

    def test_an_opening_that_leaves_the_wall_is_refused(self) -> None:
        """The worker's named refusal, and the reason it must exist: this is where a render is
        spent. The shell checks the same span first, but a hand-written request never went through
        the shell."""
        with pytest.raises(GeometryError, match="the wall is only"):
            self._panels(WallOpening("window_01", offset_along_m=5.8, width_m=1.2,
                                     height_m=1.2, sill_m=0.9))

    def test_an_opening_taller_than_the_wall_is_refused(self) -> None:
        with pytest.raises(GeometryError, match="the wall is .* tall"):
            self._panels(WallOpening("window_01", offset_along_m=3.0, width_m=1.2,
                                     height_m=3.5, sill_m=0.9))

    def test_every_opening_is_matched_to_the_edge_it_is_on(self) -> None:
        """The bridge from world XY to a wall's own axis, over an outline with several walls."""
        outline = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))
        by_edge = match_openings_to_edges(outline, [
            ("door_main", 2.0, 0.0, 0.9, 2.05, 0.0),
            ("window_01", 0.0, 1.5, 1.2, 1.2, 0.9),
            ("window_02", 4.0, 1.5, 1.2, 1.2, 0.9),
        ])
        assert sorted(by_edge) == [0, 1, 3]
        assert {item.opening_id for item in by_edge[0]} == {"door_main"}
        assert all(0.0 <= item.start_m and item.end_m <= 4.0 for items in by_edge.values()
                   for item in items)


class TestUpAxisCorrection:
    def test_matrix_sends_plus_y_to_plus_z(self) -> None:
        """USD multiplies row vectors, so the point is transformed on the left."""
        matrix = y_up_to_z_up_matrix()
        vector = (0.0, 1.0, 0.0, 0.0)
        result = [sum(vector[i] * matrix[i][j] for i in range(4)) for j in range(4)]
        assert result[:3] == pytest.approx([0.0, 0.0, 1.0])

    def test_matrix_leaves_x_alone(self) -> None:
        matrix = y_up_to_z_up_matrix()
        vector = (1.0, 0.0, 0.0, 0.0)
        result = [sum(vector[i] * matrix[i][j] for i in range(4)) for j in range(4)]
        assert result[:3] == pytest.approx([1.0, 0.0, 0.0])

    def test_matrix_is_a_rotation(self) -> None:
        matrix = y_up_to_z_up_matrix()
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        assert determinant == pytest.approx(1.0)
