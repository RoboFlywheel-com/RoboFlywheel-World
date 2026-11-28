# SPDX-License-Identifier: Apache-2.0
"""Every publishable Python source carries the project license identifier."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_python_sources_have_apache_spdx_identifier():
    paths = [
        path
        for directory in ("src", "scripts", "tests")
        for path in (ROOT / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in paths
        if "# SPDX-License-Identifier: Apache-2.0" not in path.read_text(encoding="utf-8").splitlines()[:5]
    ]
    assert not missing, f"Python files without Apache-2.0 SPDX identifier: {missing}"
