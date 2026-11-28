import unittest

import numpy as np
import trimesh

from surface import orient_faces


class PreviewSurfaceTests(unittest.TestCase):
    def test_inverted_shell_and_split_uv_vertices(self):
        box = trimesh.creation.box()
        points = box.vertices[box.faces].reshape(-1, 3)
        faces = np.arange(len(points)).reshape(-1, 3)[:, ::-1].copy()
        faces[2] = faces[2, ::-1]
        fixed, _ = orient_faces(points, faces)
        np.testing.assert_array_equal(np.sort(fixed, axis=1), np.sort(faces, axis=1))
        mesh = trimesh.Trimesh(points, fixed, process=True)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)

    def test_nearby_distinct_surfaces_are_not_welded(self):
        box = trimesh.creation.box(extents=(1, 1, .0005))
        faces, _ = orient_faces(box.vertices, box.faces)
        self.assertEqual(len(faces), len(box.faces))
        self.assertGreater(trimesh.Trimesh(box.vertices, faces).volume, 0)


if __name__ == "__main__":
    unittest.main()
