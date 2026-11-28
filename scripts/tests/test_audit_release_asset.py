# SPDX-License-Identifier: Apache-2.0
"""Synthetic release fixtures; no production assets or engines required."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "audit_release_asset", Path(__file__).resolve().parents[1] / "audit_release_asset.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReleaseAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.leaf = self.root / "assets/0/0/0_0_0000"
        self.leaf.mkdir(parents=True)
        for directory in ("mesh", "mjcf", "affordance"):
            (self.leaf / directory).mkdir()
        for name, content in (("_SUCCESS", ""), ("part_physics.json", "{}"),
                              ("0_0_0000.usd", "#usda 1.0"), ("mesh/object.obj", "")):
            (self.leaf / name).write_text(content)
        for name in ("m_0_0_0000.urdf", "coacd_multi.urdf"):
            (self.leaf / name).write_text(
                '<robot><link><visual><geometry><mesh filename="mesh/object.obj"/>'
                '</geometry></visual></link></robot>')
        (self.leaf / "mjcf/m_0_0_0000.xml").write_text(
            '<mujoco><asset><mesh file="../mesh/object.obj"/></asset></mujoco>')

    def test_portable_dependencies(self):
        _, report = audit.inspect_asset(self.root, "0_0_0000")
        self.assertEqual(report["static_references"], "passed")
        self.assertEqual(report["stability"], "not_run")

    def test_missing_mesh_fails(self):
        (self.leaf / "mesh/object.obj").unlink()
        with self.assertRaises(FileNotFoundError):
            audit.inspect_asset(self.root, "0_0_0000")

    def test_external_dependency_fails(self):
        (self.leaf / "mjcf/m_0_0_0000.xml").write_text(
            '<mujoco><asset><mesh file="/outside.obj"/></asset></mujoco>')
        with self.assertRaises(ValueError):
            audit.inspect_asset(self.root, "0_0_0000")

    def test_old_object_key_fails(self):
        with self.assertRaises(ValueError):
            audit.inspect_asset(self.root, "old-object-id")


if __name__ == "__main__":
    unittest.main()
