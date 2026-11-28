#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove the v3 core imports without any third-party package.

Run this with a bare interpreter -- ``/opt/conda/bin/python`` has no numpy, torch, trimesh
or pxr -- so the check does not depend on pytest being installed. That is the whole point:
the claim is "the core needs nothing but the standard library", and it is only meaningful
when verified by an interpreter that has nothing else.

Every core module in the installed package is imported; scientific diagnostics,
preparation commands, Blender/multisim render tools and episode executables require
their optional runtime and are outside the core import contract., then every entry in ``sys.modules`` is
classified by where its file lives. Anything resolved from site-packages/dist-packages is
a third-party import and fails the run. Modules that live inside the source tree are fine;
stdlib modules resolve outside site-packages and are fine.

Retrieval and backend adapters are included; heavyweight runtimes must be imported lazily.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(importlib.util.find_spec("rfw_w.scenes").submodule_search_locations[0])
SRC = PACKAGE_ROOT.parent
PACKAGE = "rfw_w.scenes"

# Imported for their side effects by the interpreter itself or by the stdlib; their origin
# path is not meaningful evidence either way.
IGNORED = {"rfw_w"}


def _module_files(root: Path) -> list[str]:
    """Core importable module names, including lazy retrieval/backend adapters."""
    names: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] in {"validation", "visualization", "preparation", "commands"}:
            continue
        if relative.parts[:2] in {("workers", "blender"), ("workers", "multisim")} or relative.parts[:3] == ("workers", "isaac", "episodes"):
            continue
        rel = path.relative_to(root).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = PACKAGE + ("." + ".".join(parts) if parts else "")
        names.append(name)
    return names


def _is_third_party(module: object) -> str | None:
    """Return the offending path if this module came from an installed package."""
    origin = getattr(module, "__file__", None)
    if not origin:
        return None
    resolved = Path(origin)
    if PACKAGE_ROOT in resolved.parents or resolved == PACKAGE_ROOT:
        return None
    parts = set(resolved.parts)
    if "site-packages" in parts or "dist-packages" in parts:
        return str(resolved)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    names = _module_files(PACKAGE_ROOT)
    if not names:
        print(f"FAIL: no modules found under {PACKAGE_ROOT}", file=sys.stderr)
        return 1

    # Conda's sitecustomize/.pth files inject modules such as _distutils_hack at
    # interpreter startup, before any of our code runs. Only modules that appear *after*
    # this snapshot can be attributed to importing v3, so the baseline is subtracted.
    baseline = set(sys.modules)

    failures: list[tuple[str, str]] = []
    imported: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - report every failure, not the first
            failures.append((name, f"{type(error).__name__}: {error}"))
            continue
        imported.append(name)

    offenders: list[tuple[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        if name in baseline or name in IGNORED or name.split(".")[0] in IGNORED:
            continue
        origin = _is_third_party(module)
        if origin:
            offenders.append((name, origin))

    for name, detail in failures:
        print(f"FAIL import {name}: {detail}", file=sys.stderr)
    for name, origin in offenders:
        print(f"FAIL third-party {name} <- {origin}", file=sys.stderr)

    print(f"imported {len(imported)}/{len(names)} v3 modules under {sys.executable}")
    if failures or offenders:
        print(f"FAIL: {len(failures)} import error(s), {len(offenders)} third-party import(s)")
        return 1
    print(f"PASS: {PACKAGE} core is stdlib-only "
          f"(python {sys.version.split()[0]}, no third-party modules loaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
