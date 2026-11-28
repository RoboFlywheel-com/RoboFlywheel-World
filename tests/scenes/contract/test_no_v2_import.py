# SPDX-License-Identifier: Apache-2.0
"""Every module imports without the former external scene project.

No adapter exception is allowed: retrieval is owned by this repository.

This runs each check in a *subprocess* because ``sys.modules`` is process-global: once
some other test has imported the bridge, an in-process assertion here would pass or fail
depending on test order rather than on the property being claimed.
"""

from __future__ import annotations

import subprocess
import os
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(importlib.util.find_spec("rfw_w.scenes").submodule_search_locations[0])
SRC = PACKAGE_ROOT.parent

def _discover_modules() -> list[str]:
    """Every module, including retrieval and backend adapters.

    Discovered rather than listed: a hand-maintained list silently stops covering a module
    added later, which is exactly when an accidental v2 import would slip in.
    """
    root = PACKAGE_ROOT
    names: list[str] = []
    for path in sorted(root.rglob("*.py")):
        parts = ["rfw_w", "scenes", *path.relative_to(root).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        name = ".".join(parts)
        names.append(name)
    return names


V3_MODULES = _discover_modules()


def _run(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}},
    )


def _assert_import_is_isolated(module: str) -> None:
    program = (
        "import sys\n"
        f"import {module}\n"
        "leaked = sorted(name for name in sys.modules if name == 'pvscene' or name.startswith('pvscene.'))\n"
        "print(','.join(leaked))\n"
    )
    result = _run(program)
    assert result.returncode == 0, f"importing {module} failed:\n{result.stderr}"
    leaked = result.stdout.strip()
    assert leaked == "", (
        f"importing {module} pulled the external scene project into sys.modules via {leaked!r}; "
        "RFW-W modules must not import the external scene project"
    )


@pytest.mark.parametrize("module", V3_MODULES)
def test_v3_module_does_not_import_v2(module: str) -> None:
    _assert_import_is_isolated(module)


def test_core_is_importable_with_stdlib_only() -> None:
    """The package imports under an interpreter with no third-party packages at all.

    ``sys.executable`` here is the orchestrator interpreter, which happens to have numpy
    and torch. The blocklist below is what makes the claim meaningful: those modules must
    not have been needed.
    """
    program = (
        "import sys\n"
        "import rfw_w.scenes\n"
        "heavy = [name for name in ('numpy', 'torch', 'transformers', 'trimesh', 'mujoco', 'pxr')\n"
        "         if name in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    result = _run(program)
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    assert result.stdout.strip() == "", (
        f"importing rfw_w.scenes pulled in {result.stdout.strip()!r}; the core must stay stdlib-only"
    )


def test_discovery_covers_the_package() -> None:
    """The discovery walk must actually find modules, or the parametrised test proves nothing."""
    assert len(V3_MODULES) >= 10
    assert "rfw_w.scenes.contracts" in V3_MODULES
    assert "rfw_w.scenes.adapters.simready_retrieval" in V3_MODULES


def test_external_bridge_has_been_removed() -> None:
    """A legacy import bridge must not remain as an untested optional dependency."""
    assert not any(name.startswith("rfw_w.scenes.adapters.pvscene_v2") for name in V3_MODULES)
