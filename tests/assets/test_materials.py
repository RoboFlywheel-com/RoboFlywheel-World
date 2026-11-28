# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from rfw_w.assets.physics.materials import density_of, friction_of


class MaterialTableTest(unittest.TestCase):
    def test_subtype_takes_precedence(self) -> None:
        self.assertEqual(density_of("metal", "Aluminum_6061"), (2700.0, "subtype"))

    def test_unknown_material_falls_back(self) -> None:
        self.assertEqual(density_of("unobtainium"), (1000.0, "unknown"))
        self.assertEqual(friction_of("unobtainium"), (0.7, 0.5))
