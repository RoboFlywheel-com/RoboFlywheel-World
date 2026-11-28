# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures.

The one rule that matters here: a missing dependency must never look like a pass.
Tier B tests need real asset roots; when those are unset they skip *and say so*, and the
skip count stays visible in the summary line. Setting ``PVSCENE_REQUIRE_ASSETS=1`` turns
those skips into failures, which is what a release check should use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest


def _require_assets() -> bool:
    return os.environ.get("PVSCENE_REQUIRE_ASSETS", "0") == "1"


def _skip_or_fail(reason: str) -> None:
    """Skip a Tier B test for a missing prerequisite -- or fail, when the run forbids skipping.

    Promoted to a shared helper because it was not shared: the root check below honoured
    ``PVSCENE_REQUIRE_ASSETS`` while the guards in the integration tier called ``pytest.skip``
    directly, so ``--assets --require-assets`` could skip every test in that tier and still exit
    green. A run that claims to have checked the asset library has to fail when it did not reach
    one, and every guard in the tier has to be able to say so.
    """
    if _require_assets():
        pytest.fail(f"{reason} (PVSCENE_REQUIRE_ASSETS=1 forbids skipping)")
    pytest.skip(reason)


@dataclass(frozen=True, slots=True)
class AssetRoots:
    """Concrete paths to the real provider libraries, as configured for this run."""

    simready_index: Path
    simready_assets_root: Path
    hssd_root: Path

    @property
    def hssd_synset_index(self) -> Path:
        return self.hssd_root.parent / "preprocessed" / "hssd_wnsynsetkey_index.json"


def _resolve_roots() -> tuple[AssetRoots | None, list[str]]:
    """Return the configured roots, or the list of settings that are missing."""
    missing: list[str] = []
    values: dict[str, Path] = {}
    for var in ("PVSCENE_SIMREADY_INDEX", "PVSCENE_SIMREADY_ASSETS_ROOT", "PVSCENE_HSSD_ROOT"):
        raw = os.environ.get(var)
        if not raw:
            missing.append(var)
            continue
        path = Path(raw).expanduser()
        if not path.exists():
            missing.append(f"{var} (set to {raw!r} but that path does not exist)")
            continue
        values[var] = path
    if missing:
        return None, missing
    return (
        AssetRoots(
            simready_index=values["PVSCENE_SIMREADY_INDEX"],
            simready_assets_root=values["PVSCENE_SIMREADY_ASSETS_ROOT"],
            hssd_root=values["PVSCENE_HSSD_ROOT"],
        ),
        [],
    )


@pytest.fixture(scope="session")
def asset_roots() -> AssetRoots:
    roots, missing = _resolve_roots()
    if roots is None:
        _skip_or_fail("real asset roots are not configured: " + "; ".join(missing))
    return roots


@pytest.fixture(scope="session")
def require_assets() -> bool:
    """Whether the current run refuses to skip Tier B."""
    return _require_assets()


@pytest.fixture(scope="session")
def asset_gate():
    """The one way a Tier B test declines to run, in a tier that spreads over several files.

    Handed out as a fixture rather than imported, so a test module does not have to know where
    conftest lives to be honest about what it did not measure.
    """
    def gate(reason: str) -> None:
        _skip_or_fail(reason)

    return gate
