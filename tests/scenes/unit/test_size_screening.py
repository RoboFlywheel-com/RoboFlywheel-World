# SPDX-License-Identifier: Apache-2.0
"""The size bound, and the three ways a candidate can fail to satisfy it.

This is the check the furniture path never had. `rank_candidates_by_size` scores width alone and
rejects nothing, so a spec declaring a 0.12-0.35 m ceiling light was handed a 1.6692 m pendant --
the asset that is in the office delivery today -- and the placement hung it from the ceiling with
its shade 1.25 m above the floor, in the walking path. `geometry.in_bounds` cannot see that: the box
is inside the room, it is simply too low.

The SimReady path has always done this (`measured_size_violation`, `MEASURED_SIZE_OUT_OF_RANGE`), and
the rule now lives in one place so the two cannot drift again. What is tested here is the rule, the
frame it is applied in, and the three outcomes -- admitted, relaxed, unmeasurable -- because the
wrong behaviour on any of them is silent: the first deletes furniture, the second ships a
contradiction, and the third rejects an asset on a bound nobody measured.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from rfw_w.scenes.assets.providers.hssd import (
    HssdConfig,
    candidates_for_synset,
    clear_hssd_metadata_caches,
    measure_glb_extents,
    screen_candidates_by_size,
    size_bound_violation,
)
from rfw_w.scenes.contracts.scene_spec import Range1D
from rfw_w.scenes.contracts.sizing import midpoint_of, ratio_violation
from rfw_w.scenes.contracts.units import Vec3


def make_glb(path: Path, minimum: tuple[float, float, float],
             maximum: tuple[float, float, float]) -> None:
    """A minimal but real GLB: header, one JSON chunk, one POSITION accessor with bounds.

    Real enough that the reader's chunk walking is exercised -- a chunk is padded to a four-byte
    boundary, and a reader that forgets the padding mis-reads every chunk after the first.
    """
    document = {
        "asset": {"version": "2.0"},
        "accessors": [{"componentType": 5126, "count": 8, "type": "VEC3",
                       "min": list(minimum), "max": list(maximum)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }
    payload = json.dumps(document).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total = 12 + 8 + len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"glTF" + struct.pack("<II", 2, total)
        + struct.pack("<II", len(payload), 0x4E4F534A) + payload
    )


def ranged(low: float, high: float) -> Range1D:
    return Range1D(low, high)


# A ceiling light's declared range, from the office spec: 0.12 to 0.35 m, midpoint 0.235 m.
CEILING_LIGHT = (ranged(0.25, 0.55), ranged(0.25, 0.55), ranged(0.12, 0.35))


class TestTheSharedRule:
    def test_within_the_band_passes(self) -> None:
        assert ratio_violation([0.09, 0.09, 0.10], [0.09, 0.09, 0.105]) is None

    def test_a_factor_too_large_names_the_axis(self) -> None:
        reason = ratio_violation([0.10, 0.10, 0.31], [0.09, 0.09, 0.105])
        assert reason is not None and "axisz" in reason and "exceeds" in reason

    def test_a_factor_too_small_names_the_axis(self) -> None:
        reason = ratio_violation([0.01, 0.09, 0.10], [0.09, 0.09, 0.105])
        assert reason is not None and "axisx" in reason and "below" in reason

    def test_a_zero_target_axis_is_skipped_not_divided_by(self) -> None:
        """A caller that says it does not care about an axis must not have everything rejected."""
        assert ratio_violation([5.0, 0.09], [0.0, 0.09]) is None

    def test_the_midpoint_is_the_middle_not_the_maximum(self) -> None:
        """Aiming at the top of a range selects the biggest asset in it -- a bowl for a cup."""
        assert midpoint_of(CEILING_LIGHT) == pytest.approx((0.40, 0.40, 0.235))

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError):
            ratio_violation([0.1, 0.1], [0.1, 0.1, 0.1])


class TestAgainstDeclaredBounds:
    def test_an_undeclared_bound_is_not_a_violation(self) -> None:
        """Nothing to be outside of. The desk declares no bounds at all."""
        assert size_bound_violation(Vec3(9.0, 9.0, 9.0), None) is None

    def test_an_asset_within_the_band_passes(self) -> None:
        assert size_bound_violation(Vec3(0.30, 0.30, 0.30), CEILING_LIGHT) is None

    def test_the_pendant_that_shipped_is_rejected(self) -> None:
        """The regression witness: the exact asset in the office delivery, 7.1x the midpoint."""
        reason = size_bound_violation(Vec3(0.2578, 0.2578, 1.6692), CEILING_LIGHT)
        assert reason is not None and "1.6692" in reason and "axisz" in reason

    def test_the_under_height_lamp_is_NOT_caught_and_that_is_the_rule(self) -> None:
        """The honest limit of a 2x band, recorded so nobody assumes it is tighter than it is.

        `floor_lamp` declares 1.40-1.85 m. The delivered asset is 1.121 m. That is 0.69x the 1.625 m
        midpoint -- inside the band, so it passes. The rule catches the pendant (7x) and leaves this
        one alone, which is the trade the 2x limit makes: it rejects what is wrong by a factor, not
        what is merely small. Catching a 0.69x lamp would need a tighter band, and a tighter band
        rejects the chair (3 mm over) and the plant (1 cm over) too.
        """
        bounds = (ranged(0.25, 0.5), ranged(0.25, 0.5), ranged(1.4, 1.85))
        assert size_bound_violation(Vec3(0.236, 0.274, 1.121), bounds) is None


class TestTheFrameTheBoundIsMeasuredIn:
    """A wall-hung asset is authored lying down and installed upright, and the bound is about the
    installed piece. Getting this wrong is not a small error: it either deletes the picture from
    every spec or makes the whole category report 100% violations."""

    WALL_ART = (ranged(0.4, 0.9), ranged(0.02, 0.06), ranged(0.3, 0.7))
    # The asset as authored: 0.70 wide, 1.00 high, 0.005 thick, with the thickness on z.
    AUTHORED = Vec3(0.700, 1.000, 0.005)

    def test_in_the_asset_frame_it_looks_like_a_violation(self) -> None:
        reason = size_bound_violation(self.AUTHORED, self.WALL_ART, kind="")
        assert reason is not None, "the premise of the next test is that the raw frame rejects it"

    def test_in_the_installed_frame_it_passes(self) -> None:
        assert size_bound_violation(self.AUTHORED, self.WALL_ART, kind="wall_hung") is None

    def test_only_the_wall_hung_kind_is_permuted(self) -> None:
        """Every floor kind is authored upright already; permuting them would swap height for depth."""
        tall = Vec3(0.4, 0.4, 1.5)
        bounds = (ranged(0.3, 0.5), ranged(0.3, 0.5), ranged(1.2, 1.8))
        assert size_bound_violation(tall, bounds, kind="against_wall") is None
        assert size_bound_violation(tall, bounds, kind="corner") is None
        assert size_bound_violation(tall, bounds, kind="wall_hung") is not None


class TestAnAxisTooSmallToScreen:
    """A declared range under 5 cm is a thickness, not a size, and is not screened.

    Found by the office spec: `wall_art` declares 0.02-0.06 m of thickness while the real library's
    framed pictures are 1-5 mm, so the band around the 0.04 m midpoint is [0.02, 0.08] and *every*
    real picture failed. Screening that axis did not catch a bad asset -- it deleted the category.
    """

    def test_a_thin_axis_is_skipped(self) -> None:
        assert ratio_violation([0.005], [0.04], screen_below_m=0.05) is None
        assert ratio_violation([5.0], [0.04], screen_below_m=0.05) is None

    def test_an_axis_at_or_above_the_threshold_is_still_screened(self) -> None:
        assert ratio_violation([0.02], [0.235], screen_below_m=0.05) is not None

    def test_the_threshold_is_opt_in_not_a_default(self) -> None:
        """`retrieval` calls the bare rule; only the furniture screen passes a threshold."""
        assert ratio_violation([0.005], [0.04]) is not None

    def test_a_picture_passes_once_its_thickness_is_not_screened(self) -> None:
        bounds = (ranged(0.4, 0.9), ranged(0.02, 0.06), ranged(0.3, 0.7))
        assert size_bound_violation(Vec3(0.700, 1.000, 0.005), bounds, kind="wall_hung") is None

    def test_a_rug_is_not_screened_on_its_thickness(self) -> None:
        bounds = (ranged(1.4, 2.6), ranged(1.0, 2.0), ranged(0.005, 0.02))
        assert size_bound_violation(Vec3(2.438, 1.524, 0.015), bounds) is None


class TestMeasureGlbExtents:
    def test_an_unchanged_glb_is_not_read_twice(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "cached.glb"
        make_glb(path, (0.0, 0.0, 0.0), (0.5, 1.5, 0.25))
        clear_hssd_metadata_caches()
        original_open = Path.open
        reads = 0

        def counting_open(candidate, *args, **kwargs):
            nonlocal reads
            if candidate == path:
                reads += 1
            return original_open(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "open", counting_open)
        assert measure_glb_extents(path) == Vec3(0.5, 0.25, 1.5)
        assert measure_glb_extents(path) == Vec3(0.5, 0.25, 1.5)
        assert reads == 1


class TestCandidateCache:
    def test_asset_existence_is_scanned_once_per_synset(self, tmp_path, monkeypatch) -> None:
        models = tmp_path / "models"
        ids = ("alpha", "beta")
        for asset_id in ids:
            path = models / "objects" / asset_id[0] / f"{asset_id}.glb"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"glTF")
        index = tmp_path / "index.json"
        index.write_text(json.dumps({"chair.n.01": [{"id": item} for item in ids]}))
        config = HssdConfig(models_root=models, synset_index_path=index,
                            prepared_root=tmp_path / "prepared")
        clear_hssd_metadata_caches()
        original_is_file = Path.is_file
        checks = 0

        def counting_is_file(candidate):
            nonlocal checks
            if candidate.suffix == ".glb":
                checks += 1
            return original_is_file(candidate)

        monkeypatch.setattr(Path, "is_file", counting_is_file)
        assert candidates_for_synset(config, "chair.n.01") == ids
        assert candidates_for_synset(config, "chair.n.01") == ids
        assert checks == len(ids)

    def test_it_reads_the_accessor_bounds_and_converts_y_up_to_z_up(self, tmp_path) -> None:
        """A GLB says (x, y, z); the room says (x, z, y). This is the conversion, not a swap."""
        path = tmp_path / "a.glb"
        make_glb(path, (0.0, 0.0, 0.0), (0.5, 1.5, 0.25))
        assert measure_glb_extents(path) == Vec3(0.5, 0.25, 1.5)

    def test_a_missing_file_is_unmeasurable_not_a_violation(self, tmp_path) -> None:
        assert measure_glb_extents(tmp_path / "nope.glb") is None

    def test_a_non_glb_is_unmeasurable(self, tmp_path) -> None:
        path = tmp_path / "not.glb"
        path.write_bytes(b"this is not a mesh")
        assert measure_glb_extents(path) is None

    def test_a_glb_with_no_position_accessor_is_unmeasurable(self, tmp_path) -> None:
        path = tmp_path / "empty.glb"
        payload = json.dumps({"asset": {"version": "2.0"}, "meshes": []}).encode()
        payload += b" " * ((4 - len(payload) % 4) % 4)
        path.write_bytes(
            b"glTF" + struct.pack("<II", 2, 12 + 8 + len(payload))
            + struct.pack("<II", len(payload), 0x4E4F534A) + payload
        )
        assert measure_glb_extents(path) is None


class TestScreening:
    """The three outcomes, and what each one means for the candidate list."""

    BOUNDS = CEILING_LIGHT

    @pytest.fixture
    def config(self, tmp_path) -> HssdConfig:
        (tmp_path / "objects").mkdir()
        index = tmp_path / "index.json"
        index.write_text("{}")
        return HssdConfig(models_root=tmp_path, synset_index_path=index,
                          prepared_root=tmp_path / "prepared")

    def add(self, config: HssdConfig, asset_id: str, extents) -> None:
        """``extents`` is Z-up, as the room sees the piece; a GLB stores Y-up.

        The conversion is easy to forget and it makes a test pass on the wrong axis --
        `(0.26, 0.26, 0.50)` written straight into a GLB comes back as `(0.26, 0.50, 0.26)`, so a
        violation written for the height is measured against the depth and the assertion holds for
        a reason that has nothing to do with what it claims.
        """
        x, y, z = extents
        make_glb(config.glb_path(asset_id), (0.0, 0.0, 0.0), (x, z, y))

    def test_a_fitting_asset_is_admitted(self, config) -> None:
        self.add(config, "aaa", (0.30, 0.30, 0.30))
        screen = screen_candidates_by_size(config, ["aaa"], bounds=self.BOUNDS)
        assert screen.admitted == ("aaa",) and not screen.relaxed

    def test_the_pendant_is_relaxed_with_its_reason(self, config) -> None:
        self.add(config, "bbb", (0.2578, 0.2578, 1.6692))
        screen = screen_candidates_by_size(config, ["bbb"], bounds=self.BOUNDS)
        assert screen.admitted == ()
        assert len(screen.relaxed) == 1
        asset_id, reason = screen.relaxed[0]
        assert asset_id == "bbb" and "1.6692" in reason

    def test_an_unmeasurable_asset_is_kept_rather_than_rejected(self, config) -> None:
        """A bound nobody measured is not evidence against an asset."""
        (config.models_root / "objects" / "c").mkdir(parents=True, exist_ok=True)
        config.glb_path("ccc").write_bytes(b"corrupt")
        screen = screen_candidates_by_size(config, ["ccc"], bounds=self.BOUNDS)
        assert screen.unmeasurable == ("ccc",)
        assert "ccc" in screen.usable

    def test_nothing_fitting_takes_the_least_bad_first(self, config) -> None:
        """Ships the closest thing and says so, rather than the first candidate by rank order."""
        self.add(config, "mild", (0.26, 0.26, 0.50))     # ~2.1x the midpoint
        self.add(config, "wild", (0.26, 0.26, 2.50))     # ~10.6x the midpoint
        screen = screen_candidates_by_size(config, ["wild", "mild"], bounds=self.BOUNDS)
        assert [asset for asset, _ in screen.relaxed] == ["mild", "wild"]

    def test_no_declared_bound_admits_everything_measurable(self, config) -> None:
        self.add(config, "huge", (9.0, 9.0, 9.0))
        screen = screen_candidates_by_size(config, ["huge"], bounds=None)
        assert screen.admitted == ("huge",)

    def test_candidates_beyond_the_examined_window_are_kept_and_not_rejected(self, config) -> None:
        """The pool is not narrowed, only the effort -- the same rule the ranker follows.

        None of these five have a GLB, so the two inside the window are unmeasurable and the three
        outside were never reached. What the test pins is that none of them is *rejected*: a
        candidate nobody measured is not evidence against it, and it stays available to place.
        """
        candidates = [f"id{index:03d}" for index in range(5)]
        screen = screen_candidates_by_size(config, candidates, bounds=self.BOUNDS, max_examined=2)
        assert screen.unmeasurable == ("id000", "id001")
        assert screen.unexamined == ("id002", "id003", "id004")
        assert screen.admitted == ()
        assert screen.relaxed == ()
        assert set(screen.usable) == set(candidates)

    def test_usable_prefers_a_fit_then_unjudged_then_a_known_miss(self, config) -> None:
        """`usable` is the placement preference order, not the input order.

        A candidate measured inside the band beats one nobody judged, which beats one *known* to be
        outside it. Getting this backwards would place a known-bad asset ahead of an unexamined one
        that might fit.
        """
        self.add(config, "fits", (0.30, 0.30, 0.30))
        self.add(config, "misses", (0.26, 0.26, 2.50))
        screen = screen_candidates_by_size(
            config, ["misses", "missing", "fits"], bounds=self.BOUNDS, max_examined=3
        )
        assert screen.usable == ("fits", "missing", "misses")

    def test_reading_stops_once_enough_fit(self, config) -> None:
        """A synset can hold hundreds of assets and the build materialises three.

        Measuring all 24 candidates of nine entities read 216 GLBs per rebuild to decide an ordering
        whose only job is to make the first three plausible. The unexamined keep their order and are
        not rejected: not measuring an asset is not evidence against it.
        """
        self.add(config, "a", (0.30, 0.30, 0.30))
        self.add(config, "b", (0.31, 0.31, 0.31))
        self.add(config, "c", (0.29, 0.29, 0.29))
        self.add(config, "d", (0.26, 0.26, 9.00))   # would be rejected, never reached
        screen = screen_candidates_by_size(
            config, ["a", "b", "c", "d"], bounds=self.BOUNDS, max_examined=24, enough=3
        )
        assert screen.admitted == ("a", "b", "c")
        assert screen.unexamined == ("d",)
        assert screen.relaxed == (), "an unexamined candidate must not be rejected"
        assert screen.usable == ("a", "b", "c", "d")

    def test_the_usable_list_puts_admitted_before_unmeasurable_before_relaxed(self, config) -> None:
        """The order is the contract: a measured fit beats an unmeasured maybe beats a known miss."""
        self.add(config, "fits", (0.30, 0.30, 0.30))
        self.add(config, "misses", (0.26, 0.26, 2.50))
        screen = screen_candidates_by_size(
            config, ["missing", "misses", "fits"], bounds=self.BOUNDS, max_examined=3
        )
        assert screen.usable == ("fits", "missing", "misses")


class TestAgainstTheRealLibrary:
    """The cheap measurement against the expensive one. Skipped without the library configured."""

    @pytest.fixture(scope="module")
    def config(self):
        import os

        root = os.environ.get("PVSCENE_HSSD_ROOT")
        index = os.environ.get("PVSCENE_HSSD_SYNSET_INDEX")
        if not root or not index or not Path(root).is_dir():
            pytest.skip("PVSCENE_HSSD_ROOT / PVSCENE_HSSD_SYNSET_INDEX are not configured")
        return HssdConfig(models_root=Path(root), synset_index_path=Path(index),
                          prepared_root=Path(root) / "_prepared_probe")

    def test_the_reading_matches_the_prepared_mesh(self, config, tmp_path) -> None:
        """`prepare_furniture` exports meshes and returns `dimensions_m`; this reads a JSON chunk.

        The two must agree, or a candidate is screened on one number and placed on another. The
        ceiling lamp is the witness because it is the asset this whole check exists for.
        """
        from rfw_w.scenes.assets.providers.hssd import prepare_furniture

        asset_id = "0113c9f0a2eaa7dc65b0b9dbb3df9b8d6ea7f2e1"  # may not exist; pick from the index
        index = json.loads(config.synset_index_path.read_text())
        entries = index.get("ceiling_lamp.n.01") or []
        assert entries, "ceiling_lamp.n.01 is not in the index"
        disagreement = []
        for entry in entries[:3]:
            candidate = str(entry["id"])
            path = config.glb_path(candidate)
            if not path.is_file():
                continue
            cheap = measure_glb_extents(path)
            prepared = prepare_furniture(config, candidate)
            if cheap is None:
                continue
            for axis, (measured, reported) in enumerate(
                zip(cheap.to_list(), prepared.dimensions_m.to_list())
            ):
                if abs(measured - reported) > 0.002:
                    disagreement.append(
                        f"{candidate[:10]} axis{'xyz'[axis]}: read {measured:.4f}, "
                        f"prepared {reported:.4f}"
                    )
        assert not disagreement, disagreement
