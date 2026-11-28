# SPDX-License-Identifier: Apache-2.0
"""The library's own support surface, converted into the frame the task cell solves in.

No asset root is needed here: the sidecar is data, so a synthetic one is a real test of the reader,
the selection rule and the frame conversion. The library itself is exercised in
``tests/integration/test_hssd_support_measurement.py``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from rfw_w.scenes.assets.providers.hssd import (
    HssdConfig,
    PreparedFurniture,
    clear_hssd_metadata_caches,
    read_support_surfaces,
)
from rfw_w.scenes.assets.support_measurement import (
    SupportMeasurementError,
    assert_measurement_matches_prepared,
    measure_support_footprint,
    measure_support_from_readings,
    select_support_surface,
    support_polygon_of,
)
from rfw_w.scenes.contracts.asset import (
    AffordanceStatus,
    AssetKey,
    Provenance,
    SupportPolygon,
)
from rfw_w.scenes.contracts.units import Vec3

ASSET_ID = "0123456789abcdef0123456789abcdef01234567"


def surface(
    *,
    centroid=(0.4, 0.75, -0.2),
    axes=(1.2, 2e-06, 0.6),
    normal=(0.0, -1.0, 0.0),
    area=0.72,
) -> dict:
    """One entry of a real sidecar, as the dataset writes it."""
    return {
        "obb": {"centroid": list(centroid), "axesLengths": list(axes)},
        "normal": {"x": normal[0], "y": normal[1], "z": normal[2]},
        "area": area,
    }


def config_with(tmp_path: Path, surfaces: list[dict], asset_id: str = ASSET_ID) -> HssdConfig:
    models = tmp_path / "models"
    sidecar = models / "support-surfaces" / asset_id / f"{asset_id}.supportSurface.json.gz"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(gzip.compress(json.dumps({"supportSurfaces": surfaces}).encode()))
    index = tmp_path / "synset_index.json"
    index.write_text("{}")
    clear_hssd_metadata_caches()
    return HssdConfig(
        models_root=models, synset_index_path=index, prepared_root=tmp_path / "prepared",
    )


def prepared(**overrides: object) -> PreparedFurniture:
    payload: dict[str, object] = {
        "asset_key": AssetKey(provider="hssd", library_version="hssd-models", asset_id=ASSET_ID),
        "visual_obj": Path("/tmp/prepared/abc/abc.obj"),
        "collision_obj": Path("/tmp/prepared/abc/abc_collision.obj"),
        "dimensions_m": Vec3(1.2, 0.6, 0.75),
        "bounds_min_m": Vec3(-0.6, -0.3, 0.0),
        "bounds_max_m": Vec3(0.6, 0.3, 0.75),
        "mass_kg": 100.0,
        "support_polygons": (),
        "vertex_count": 100,
        "face_count": 50,
        "collision_face_count": 20,
        "provenance": Provenance(
            method="mesh_volume_times_density", version="v1", status=AffordanceStatus.ESTIMATED,
        ),
    }
    payload.update(overrides)
    return PreparedFurniture(**payload)  # type: ignore[arg-type]


class TestTheFrameConversion:
    def test_a_horizontal_surface_becomes_a_footprint_with_the_front_edge_at_zero(
        self, tmp_path: Path,
    ) -> None:
        """The one numeric pin on the conversion: extents, centring, height and provenance.

        The surface is published off-centre on purpose. A conversion that kept the asset's own
        coordinates would pass a test written around a centred surface and fail on every real desk.
        """
        measurement = measure_support_footprint(config_with(tmp_path, [surface()]), ASSET_ID)
        assert measurement.width_m == pytest.approx(1.2)
        assert measurement.depth_m == pytest.approx(0.6)
        assert measurement.support_height_m == pytest.approx(0.75)
        assert measurement.vertices_xy == (
            (-0.6, 0.0), (0.6, 0.0), (0.6, 0.6), (-0.6, 0.6)
        )
        assert measurement.provenance.status is AffordanceStatus.ANNOTATED
        assert measurement.provenance.method == "hssd_precomputed_support_surface_obb"
        assert measurement.source_polygon_id == f"{ASSET_ID}:surface0"
        assert measurement.front_edge_basis == "declared_min_y_faces_the_aisle"
        assert measurement.height_basis == "asset_local_base_plane"

    def test_the_assets_own_offset_does_not_move_the_cell_footprint(self, tmp_path: Path) -> None:
        """Where the asset publishes its surface must not reach the cell.

        HSSD publishes every surface twice, once in each half of a 180-degree ``modelToWorld``
        mirror. Anchoring on the surface's own rectangle is what makes that mirror irrelevant to the
        cell, and this is the test that says so rather than a comment claiming it.
        """
        here = measure_support_footprint(config_with(tmp_path, [surface()]), ASSET_ID)
        elsewhere = measure_support_footprint(
            config_with(tmp_path, [surface(centroid=(-0.9, 0.75, 1.4))]), ASSET_ID,
        )
        assert here.vertices_xy == elsewhere.vertices_xy
        assert here.depth_m == elsewhere.depth_m

    def test_a_base_offset_moves_the_height_and_nothing_else(self, tmp_path: Path) -> None:
        """The offset is the one thing that is not mirror-invariant, so it is an explicit knob."""
        readings = read_support_surfaces(config_with(tmp_path, [surface()]), ASSET_ID)
        raised = measure_support_from_readings(
            readings, asset_id=ASSET_ID, source_uri="u", base_offset_m=0.02,
        )
        assert raised.support_height_m == pytest.approx(0.77)
        assert raised.vertices_xy == measure_support_from_readings(
            readings, asset_id=ASSET_ID, source_uri="u",
        ).vertices_xy


class TestWhichSurfaceIsChosen:
    def test_a_vertical_panel_is_not_eligible(self, tmp_path: Path) -> None:
        """The measured defect this filter exists for.

        The reader keeps any reading whose flat axis matches its normal, and a normal may point
        along z -- so a panel qualifies. On the real library this changes the choice for 10 of 169
        desks, and picking a panel puts the target in mid-air.
        """
        panel = surface(centroid=(0.0, 0.445, 0.0), axes=(0.919, 0.6, 2e-06),
                        normal=(0.0, 0.0, -1.0), area=0.919)
        top = surface(area=0.694)
        readings = read_support_surfaces(config_with(tmp_path, [panel, top]), ASSET_ID)
        # Largest area first, so the panel leads -- and is skipped.
        assert readings[0].normal_axis == "z"
        assert select_support_surface(readings) == 1
        measurement = measure_support_from_readings(
            readings, asset_id=ASSET_ID, source_uri="u",
        )
        assert measurement.support_height_m == pytest.approx(0.75)
        assert measurement.source_polygon_id.endswith(":surface1")


class TestFailuresByName:
    def test_an_asset_without_a_sidecar_fails_by_name(self, tmp_path: Path) -> None:
        config = config_with(tmp_path, [])
        assert read_support_surfaces(config, ASSET_ID) == ()
        with pytest.raises(SupportMeasurementError, match="HSSD_SUPPORT_SURFACE_ABSENT"):
            measure_support_footprint(config, ASSET_ID)

    def test_a_sidecar_with_only_vertical_surfaces_fails_by_name(self, tmp_path: Path) -> None:
        """Not "the best of a bad set": a panel is not a surface, so there is nothing to protect."""
        panel = surface(axes=(0.919, 0.6, 2e-06), normal=(0.0, 0.0, -1.0), area=0.919)
        with pytest.raises(SupportMeasurementError, match="HSSD_SUPPORT_SURFACE_HORIZONTAL_ABSENT"):
            measure_support_footprint(config_with(tmp_path, [panel]), ASSET_ID)

    def test_a_sliver_surface_is_refused(self, tmp_path: Path) -> None:
        sliver = surface(axes=(0.0004, 2e-06, 0.6))
        with pytest.raises(SupportMeasurementError, match="HSSD_SUPPORT_SURFACE_DEGENERATE"):
            measure_support_footprint(config_with(tmp_path, [sliver]), ASSET_ID)

    def test_a_non_positive_height_is_refused(self, tmp_path: Path) -> None:
        at_the_floor = surface(centroid=(0.4, 0.0, -0.2))
        with pytest.raises(SupportMeasurementError, match="HSSD_SUPPORT_SURFACE_INVALID"):
            measure_support_footprint(config_with(tmp_path, [at_the_floor]), ASSET_ID)


class TestAgreementWithThePreparedAsset:
    def test_the_measurement_matches_the_polygon_the_scene_materialises(self, tmp_path: Path) -> None:
        measurement = measure_support_footprint(config_with(tmp_path, [surface()]), ASSET_ID)
        same = SupportPolygon(
            polygon_id="x",
            vertices_xy=((-0.2, -0.5), (1.0, -0.5), (1.0, 0.1), (-0.2, 0.1)),
            plane_z_m=0.75,
            provenance=measurement.provenance,
        )
        assert_measurement_matches_prepared(measurement, prepared(support_polygons=(same,)))

    def test_a_prepared_asset_that_materialises_another_surface_is_refused(self, tmp_path: Path) -> None:
        """The check that stops the cell protecting one rectangle while the scene places on another."""
        measurement = measure_support_footprint(config_with(tmp_path, [surface()]), ASSET_ID)
        narrower = SupportPolygon(
            polygon_id="x",
            vertices_xy=((-0.2, -0.5), (0.8, -0.5), (0.8, 0.1), (-0.2, 0.1)),
            plane_z_m=0.75,
            provenance=measurement.provenance,
        )
        with pytest.raises(SupportMeasurementError, match="PREPARED_SUPPORT_DISAGREES_WITH_MEASUREMENT"):
            assert_measurement_matches_prepared(measurement, prepared(support_polygons=(narrower,)))

    def test_a_prepared_asset_with_no_surface_is_refused(self, tmp_path: Path) -> None:
        measurement = measure_support_footprint(config_with(tmp_path, [surface()]), ASSET_ID)
        with pytest.raises(SupportMeasurementError, match="HSSD_SUPPORT_SURFACE_ABSENT"):
            assert_measurement_matches_prepared(measurement, prepared(support_polygons=()))


def test_the_record_converts_to_a_support_polygon(tmp_path: Path) -> None:
    """The bridge to the vocabulary the scene-side code already speaks."""
    measurement = measure_support_footprint(config_with(tmp_path, [surface()]), ASSET_ID)
    polygon = support_polygon_of(measurement)
    assert polygon.plane_z_m == measurement.support_height_m
    assert polygon.vertices_xy == measurement.vertices_xy
    assert polygon.is_rectangle


def test_a_reading_the_reader_keeps_always_has_a_positive_extent(tmp_path: Path) -> None:
    """Why ``source_polygon_id`` can be trusted to point at the surface that was measured.

    The provider silently drops a reading whose extent is zero, which would shift every later
    polygon label. For a *horizontal* reading that drop turns out to be unreachable, and the reason
    is worth pinning because the id's meaning depends on it: the flat axis is whichever extent is
    smallest, so a surface normal to y is only accepted when its x and z extents are strictly larger
    than the vertical one -- hence positive. Written as a test because "the reader happens to
    guarantee this" is exactly the kind of assumption that stops being true in silence.
    """
    zero_x = surface(axes=(0.0, 2e-06, 0.6))
    assert read_support_surfaces(config_with(tmp_path, [zero_x]), ASSET_ID) == ()
