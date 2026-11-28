# SPDX-License-Identifier: Apache-2.0
"""Package-level invariants that every other module relies on."""

from __future__ import annotations

import re

import rfw_w.scenes


def test_schema_namespace_is_centralised() -> None:
    """Artefact schema ids are built from one namespace, not spelled out per module."""
    assert rfw_w.scenes.SCHEMA_NAMESPACE == "pvscene.v3"


def test_version_is_pep440_shaped() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([.\-+].*)?", rfw_w.scenes.__version__), rfw_w.scenes.__version__


def test_dunder_all_is_explicit() -> None:
    """``__all__`` must not be empty: it documents the public surface of the core."""
    assert rfw_w.scenes.__all__
    for name in rfw_w.scenes.__all__:
        assert hasattr(rfw_w.scenes, name), name
