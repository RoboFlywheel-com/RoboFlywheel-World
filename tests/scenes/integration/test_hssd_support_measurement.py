# SPDX-License-Identifier: Apache-2.0
"""The room-free core, given a support surface that came out of the real rigid furniture library.

Tier B: needs ``PVSCENE_HSSD_ROOT`` (and its synset index). Without them these skip loudly; under
``PVSCENE_REQUIRE_ASSETS=1`` they fail, because a run that claims to have verified the core against
the library has to fail when it never reached one.

What is being proven here is narrow and worth stating: that the surface the core protects is the
surface an asset actually states, that the estimate it replaces is a genuinely different number, and
that the core invariants survive the substitution. Nothing here runs Isaac, a model, or the build.
What that leaves unrun -- the measured-asset path is no longer on the build path, and no run has
placed a scene on a library-stated support -- is recorded in ``docs/ARCHITECTURE.md`` §11.
"""

from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import gzip
import json
import os
from pathlib import Path

import pytest

from rfw_w.scenes.assets.providers.hssd import HssdConfig, candidates_for_synset
from rfw_w.scenes.assets.support_measurement import (
    SupportMeasurementError,
    measure_support_footprint,
)
from rfw_w.scenes.contracts.asset import AffordanceStatus
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.geometry.polygon import contains_point
from rfw_w.scenes.solvers.cell import solve_cell
from rfw_w.scenes.solvers.room import fit_room_around_cell

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.assets

# The synset the office experiment's desk is retrieved under, and the category envelope the core
# used before this change: ``CATEGORY_PROFILES["desk"].target_extent_m`` with a size-band midpoint
# for depth. The depth is the axis the comparison turns on, because it is the one the category
# profile guesses worst.
DESK_SYNSET = "desk.n.01"
ENVELOPE_DEPTH_M = 0.7425
ENVELOPE_WIDTH_M = 1.35

# How many assets the aggregate checks walk. The library is a mounted read-only export, so this
# sample is deterministic; 25 keeps the tier to seconds while still catching a rule that only holds
# for the first asset.
SAMPLE = 25


@pytest.fixture(scope="module")
def hssd_config(asset_roots, tmp_path_factory, asset_gate) -> HssdConfig:
    index = os.environ.get("PVSCENE_HSSD_SYNSET_INDEX")
    synset_index = Path(index).expanduser() if index else asset_roots.hssd_synset_index
    if not synset_index.is_file():
        asset_gate(
            f"no HSSD synset index at {synset_index}; set PVSCENE_HSSD_SYNSET_INDEX to the "
            "library's hssd_wnsynsetkey_index.json"
        )
    return HssdConfig(
        models_root=asset_roots.hssd_root,
        synset_index_path=synset_index,
        # Nothing on this path prepares a mesh, so this directory stays empty; it is required by the
        # config and deliberately kept out of the working tree.
        prepared_root=tmp_path_factory.mktemp("prepared"),
    )


@pytest.fixture(scope="module")
def desk_ids(hssd_config: HssdConfig) -> tuple[str, ...]:
    ids = candidates_for_synset(hssd_config, DESK_SYNSET)
    assert ids, f"the library returned no assets for {DESK_SYNSET}"
    return ids


def raw_support_surfaces(config: HssdConfig, asset_id: str) -> dict:
    """The sidecar as the dataset wrote it, read here without going through the helper."""
    path = config.support_surface_path(asset_id)
    return json.loads(gzip.decompress(path.read_bytes()))


def spec_with_seed(name: str, seed: int) -> SceneSpec:
    payload = json.loads((data_path('examples') / name / 'task.json').read_text())
    payload["seed"] = seed
    return SceneSpec.from_dict(payload)


class TestTheMeasurementIsReal:
    def test_a_real_desk_measures_a_footprint_the_envelope_does_not_predict(
        self, hssd_config: HssdConfig, desk_ids: tuple[str, ...],
    ) -> None:
        """The point of the whole change: the estimate and the asset disagree.

        The assertion is on depth because that is where the category profile guesses worst. Measured over
        the first 25 desks of this library, 14 differ from the envelope by more than 5 cm on *both*
        axes, so the first candidate is representative rather than lucky.
        """
        measurement = measure_support_footprint(hssd_config, desk_ids[0])
        assert measurement.provenance.status is AffordanceStatus.ANNOTATED
        assert measurement.provenance.method == "hssd_precomputed_support_surface_obb"
        assert measurement.width_m > 0.3
        assert 0.4 < measurement.support_height_m < 1.3
        assert abs(measurement.depth_m - ENVELOPE_DEPTH_M) > 0.05
        assert measurement.source_polygon_id.startswith(desk_ids[0])

    def test_the_measured_height_is_a_height_above_the_asset(
        self, hssd_config: HssdConfig, desk_ids: tuple[str, ...],
    ) -> None:
        """The height is cross-checked against the raw file, not against the helper's own answer.

        The surface is located by recomputing the selection rule from the sidecar bytes here, so a
        bug in the rule shows up as a disagreement rather than as two copies of the same mistake.
        """
        asset_id = desk_ids[0]
        measurement = measure_support_footprint(hssd_config, asset_id)
        payload = raw_support_surfaces(hssd_config, asset_id)
        surfaces = payload["supportSurfaces"]
        horizontal = [
            item for item in surfaces
            if abs(float(item["normal"].get("y", 0.0))) >= 0.5
            and float(item["normal"].get("y", 0.0)) == max(
                abs(float(item["normal"].get(axis, 0.0))) for axis in "xyz"
            )
        ]
        assert horizontal, "the asset states no horizontal surface; the choice cannot be checked"
        chosen = max(horizontal, key=lambda item: float(item["area"]))
        assert measurement.support_height_m == pytest.approx(float(chosen["obb"]["centroid"][1]))
        # And it cannot be above the asset itself.
        assert measurement.support_height_m <= float(payload["modelbbox"]["max"][1]) + 1e-6

    def test_every_sampled_desk_is_measured_or_fails_by_name(
        self, hssd_config: HssdConfig, desk_ids: tuple[str, ...],
    ) -> None:
        """No silent losses: each asset either measures or refuses with a reason that can be read.

        The count of refusals is compared against the count of assets whose sidecar is genuinely
        missing on disk, so the test states *why* anything was refused rather than assuming.
        """
        sample = desk_ids[:SAMPLE]
        measured = []
        refusals: dict[str, str] = {}
        for asset_id in sample:
            try:
                measured.append(measure_support_footprint(hssd_config, asset_id))
            except SupportMeasurementError as error:
                refusals[asset_id] = str(error)
        without_sidecar = [
            asset_id for asset_id in sample
            if not hssd_config.support_surface_path(asset_id).is_file()
        ]
        assert len(refusals) == len(without_sidecar), (
            f"{len(refusals)} refusal(s) for {len(without_sidecar)} asset(s) without a sidecar: "
            f"{refusals}"
        )
        for error_text in refusals.values():
            assert "HSSD_SUPPORT_" in error_text, error_text
        assert len(measured) == len(sample) - len(without_sidecar)
        assert len(measured) >= SAMPLE - 2

    def test_an_asset_without_a_surface_fails_by_name(
        self, hssd_config: HssdConfig, desk_ids: tuple[str, ...],
    ) -> None:
        """The one asset in this library that states no surface is refused, not defaulted.

        Asserted non-empty on purpose: with an empty list this test would pass while checking
        nothing, which is the defect this repository calls out by name.
        """
        without = [
            asset_id for asset_id in desk_ids
            if not hssd_config.support_surface_path(asset_id).is_file()
        ]
        assert without, "every desk states a surface; this refusal path was not exercised"
        for asset_id in without:
            with pytest.raises(SupportMeasurementError, match="HSSD_SUPPORT_SURFACE_ABSENT"):
                measure_support_footprint(hssd_config, asset_id)


class TestTheSelectionRule:
    def test_a_vertical_panel_is_never_chosen(
        self, hssd_config: HssdConfig, desk_ids: tuple[str, ...],
    ) -> None:
        """The measured defect, checked over the sample rather than argued.

        A panel has area and wins on area alone -- on this library it does, for 10 of 169 desks. The
        chosen surface is located here from the sidecar bytes by the rule the module documents
        (largest area among the horizontal ones) and must be the one the measurement returned.
        """
        panels_won = 0
        for asset_id in desk_ids[:SAMPLE]:
            if not hssd_config.support_surface_path(asset_id).is_file():
                continue
            surfaces = raw_support_surfaces(hssd_config, asset_id)["supportSurfaces"]

            def horizontal(item: dict) -> bool:
                y = abs(float(item["normal"].get("y", 0.0)))
                return y >= 0.5 and y == max(
                    abs(float(item["normal"].get(axis, 0.0))) for axis in "xyz"
                )

            eligible = [item for item in surfaces if horizontal(item)]
            assert eligible, f"{asset_id} states no horizontal surface"
            chosen = max(eligible, key=lambda item: float(item["area"]))
            largest_overall = max(surfaces, key=lambda item: float(item["area"]))
            if not horizontal(largest_overall):
                panels_won += 1
            measurement = measure_support_footprint(hssd_config, asset_id)
            assert measurement.support_height_m == pytest.approx(
                float(chosen["obb"]["centroid"][1])
            ), asset_id
        # The filter is only load-bearing where the overall-largest surface is a panel. On this
        # library that happens; if a refresh removes every such asset the count drops to zero and
        # this assertion says so rather than pretending the rule was exercised.
        assert panels_won > 0


class TestTheCoreWithARealSurface:
    def test_the_measured_cell_keeps_the_core_invariants(
        self, hssd_config: HssdConfig, desk_ids: tuple[str, ...],
    ) -> None:
        measurement = measure_support_footprint(hssd_config, desk_ids[0])
        spec = spec_with_seed("office_pick_cup", 11)
        cell = solve_cell(spec, seed=11, support_measurement=measurement)

        assert contains_point(cell.support_footprint, cell.target_offset)
        assert cell.standoffs
        assert cell.protected.area >= cell.support_footprint.area
        assert cell.support_height_m == measurement.support_height_m
        assert cell.support_geometry_status == "annotated"
        assert cell.support_geometry_method == (
            "asset_support_polygon:hssd_precomputed_support_surface_obb"
        )
        assert cell.core_hash != solve_cell(spec, seed=11).core_hash

    def test_the_frame_convention_is_pinned_numerically(
        self, hssd_config: HssdConfig, desk_ids: tuple[str, ...],
    ) -> None:
        """Front edge at y = 0, centred on x -- the frame the room fit will be built around."""
        measurement = measure_support_footprint(hssd_config, desk_ids[0])
        cell = solve_cell(spec_with_seed("office_pick_cup", 11), seed=11, support_measurement=measurement)
        min_x, min_y, max_x, max_y = cell.support_footprint.bounds()
        assert min_y == pytest.approx(0.0, abs=1e-9)
        assert min_x == pytest.approx(-max_x, abs=1e-9)
        assert (max_x - min_x) == pytest.approx(measurement.width_m, abs=1e-9)
        assert (max_y - min_y) == pytest.approx(measurement.depth_m, abs=1e-9)

    def test_the_measured_cell_is_still_room_independent(
        self, hssd_config: HssdConfig, desk_ids: tuple[str, ...],
    ) -> None:
        """The invariant that matters most, in the presence of the one new input."""
        measurement = measure_support_footprint(hssd_config, desk_ids[0])
        spec = spec_with_seed("office_pick_cup", 11)
        cell = solve_cell(spec, seed=11, support_measurement=measurement)

        core_hashes = set()
        shell_hashes = set()
        for room_seed in range(20):
            fitted = fit_room_around_cell(spec, cell, room_seed=room_seed)
            core_hashes.add(cell.core_hash)
            shell_hashes.add(tuple(fitted.shell.polygon_xy))
        assert len(core_hashes) == 1
        assert len(shell_hashes) > 1
