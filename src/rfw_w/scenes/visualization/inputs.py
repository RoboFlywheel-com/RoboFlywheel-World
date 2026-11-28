# SPDX-License-Identifier: Apache-2.0
"""Resolve explicit local inputs and references to installed example resources."""
from __future__ import annotations

from pathlib import Path

from rfw_w.scenes.resources import data_path


def resolve_input(value: str | Path, *, base: Path | None = None) -> Path:
    """Local paths keep their meaning; examples/... names packaged, read-only data.

    A local file wins when an explicit base is supplied or the invocation directory
    contains it. Other relative paths require that local file and never guess a source
    repository root.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    local = (base or Path.cwd()) / path
    if local.is_file():
        return local.resolve()
    if path.parts and path.parts[0] == "examples":
        return data_path(path.as_posix())
    return local.resolve()
