# SPDX-License-Identifier: Apache-2.0
"""Installed data and worker source locations, independent of a repository checkout."""
from __future__ import annotations

import importlib.util
import json
from importlib.resources import files
from pathlib import Path, PurePosixPath


def data_path(relative: str) -> Path:
    """Locate package data in a normal installed wheel or editable installation.

    The worker protocols need a persistent filesystem path. Zip import is deliberately
    unsupported; installing a wheel extracts its contents and satisfies this contract.
    """
    name = PurePosixPath(relative)
    if name.is_absolute() or not name.parts or any(part in {"..", "."} for part in name.parts):
        raise ValueError("resource names must stay inside the package")
    result = files(__package__).joinpath(*name.parts)
    path = Path(str(result))
    if not path.exists():
        raise FileNotFoundError(f"package resource is unavailable: {relative}")
    return path


def load_json(relative: str):
    return json.loads(data_path(relative).read_text(encoding="utf-8"))


def module_path(module: str) -> Path:
    """Locate installed worker source without importing a heavy engine module."""
    if not module.startswith("rfw_w.scenes."):
        raise ValueError("worker module must belong to rfw_w.scenes")
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        raise ModuleNotFoundError(module)
    return Path(spec.origin).resolve()
