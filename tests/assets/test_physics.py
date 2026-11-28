# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

from rfw_w.assets.physics.mass_properties import mass_com_inertia


class PhysicsTest(unittest.TestCase):
    def test_mass_com_and_inertia_for_two_point_masses(self) -> None:
        centers = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        total, com, inertia = mass_com_inertia(
            centers, np.asarray([0, 1]), {0: 1.0, 1: 1.0}
        )
        self.assertAlmostEqual(total, 2.0)
        np.testing.assert_allclose(com, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(inertia, np.diag([0.0, 2.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
