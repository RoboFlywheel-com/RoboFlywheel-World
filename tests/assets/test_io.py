# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfw_w.assets.contracts.io import asset_path, load_manifest, part_dir, stable_seed


class ManifestIoTest(unittest.TestCase):
    def test_manifest_paths_and_seed_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "assets.jsonl"
            row = {"asset_id": "cup_01", "class_id": "cup", "prompt": "a cup"}
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            loaded = load_manifest(manifest)
            self.assertEqual(loaded, [row])
            self.assertEqual(asset_path("out", row, ".glb"), Path("out/cup/cup_01.glb"))
            self.assertEqual(part_dir("parts", row), Path("parts/cup/cup_01"))
            self.assertEqual(stable_seed(row, 2025), stable_seed(row, 2025))

    def test_required_fields_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "bad.jsonl"
            manifest.write_text('{"asset_id":"missing_class"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class_id"):
                load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
