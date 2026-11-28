# SPDX-License-Identifier: Apache-2.0
"""Release-scan regressions use synthetic markers in isolated repositories."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scan_sensitive.py"


class SensitiveScanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(SCRIPT, self.root / "scripts" / SCRIPT.name)

    def scan(self, *arguments):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        return subprocess.run(
            [sys.executable, "scripts/scan_sensitive.py", *arguments],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def keywords(self, value="synthetic-release-keyword"):
        (self.root / ".sanitize-keywords.local").write_text(value + "\n", encoding="utf-8")

    def test_missing_or_empty_keyword_file_is_blocked(self):
        for content in (None, "\n# no configured keywords\n"):
            with self.subTest(content=content):
                keyword_file = self.root / ".sanitize-keywords.local"
                if content is None:
                    keyword_file.unlink(missing_ok=True)
                else:
                    keyword_file.write_text(content, encoding="utf-8")
                result = self.scan()
                self.assertEqual(result.returncode, 2)
                self.assertIn("BLOCKED", result.stderr)
                self.assertNotIn("PASS", result.stderr)

    def test_all_publishable_directories_and_source_vendor_files_are_scanned(self):
        self.keywords()
        paths = [
            "assets/input.txt",
            "configs/runtime.json",
            "docs/guide.md",
            "examples/request.json",
            "scripts/helper.py",
            "src/package/module.py",
            "tests/test_example.py",
            "src/package/web/vendor/addon.js",
        ]
        for relative in paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("/home/" + "private/example/\n", encoding="utf-8")
        result = self.scan()
        self.assertEqual(result.returncode, 1)
        for relative in paths:
            self.assertIn(relative, result.stdout)

    def test_verbose_output_masks_the_configured_internal_keyword(self):
        marker = "synthetic-" + "internal-marker"
        self.keywords(marker)
        (self.root / "docs").mkdir()
        (self.root / "docs" / "marker.txt").write_text(marker + "\n", encoding="utf-8")
        result = self.scan("-v")
        self.assertEqual(result.returncode, 1)
        self.assertIn("internal_keyword docs/marker.txt:1", result.stdout)
        self.assertNotIn(marker, result.stdout)
        self.assertNotIn(marker, result.stderr)


if __name__ == "__main__":
    unittest.main()
