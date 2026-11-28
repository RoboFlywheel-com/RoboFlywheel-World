# SPDX-License-Identifier: Apache-2.0
"""A prepared asset, and the one property the run depends on: it survives its own sidecar.

`prepare_furniture` is called once per entity per build and short-circuits when a preparation is
already on disk, so most builds in a repair loop read their assets from the sidecar rather than
re-deriving them. That makes the sidecar a *second implementation* of the same value: if a read
does not return what a fresh preparation computed, the scene changes for a reason the plan does not
contain.

That is not hypothetical. `to_dict` rounded its measurements on the way out while a fresh
preparation did not, so the two differed by ~1e-7 m in every dimension. Those numbers feed the
placement footprint, the recorded bounds and the scene version, so rebuilding the plan a repair
started from produced a *different scene* -- and the loop's restore path landed on a version no
gate had measured. The tests here are the ones that would have caught it, and they need no assets:
the invariant is a round trip.
"""

from __future__ import annotations

from pathlib import Path

from rfw_w.scenes.assets.providers.hssd import (
    STORED_DECIMALS,
    PreparedFurniture,
    _prepared_from_dict,
)
from rfw_w.scenes.contracts.asset import AssetKey, AffordanceStatus, Provenance
from rfw_w.scenes.contracts.units import Vec3


def prepared(**overrides: object) -> PreparedFurniture:
    """A preparation with full-precision measurements, as a fresh one has before rounding."""
    payload: dict[str, object] = {
        "asset_key": AssetKey(provider="hssd", library_version="hssd-models", asset_id="abc"),
        "visual_obj": Path("/tmp/prepared/abc/abc.obj"),
        "collision_obj": Path("/tmp/prepared/abc/abc_collision.obj"),
        "dimensions_m": Vec3(1.3700799942016602, 0.7602200508117676, 0.7499952153593767),
        "bounds_min_m": Vec3(-0.6850399971008301, -0.3801100254058838, 0.0),
        "bounds_max_m": Vec3(0.6850399971008301, 0.3801100254058838, 0.7499952153593767),
        "mass_kg": 312.4436893861899,
        "support_polygons": (),
        "vertex_count": 1971,
        "face_count": 1230,
        "collision_face_count": 68,
        "provenance": Provenance(
            method="mesh_volume_times_density", version="v1", status=AffordanceStatus.ESTIMATED
        ),
    }
    payload.update(overrides)
    return PreparedFurniture(**payload)  # type: ignore[arg-type]


class TestSidecarRoundTrip:
    def test_a_read_returns_what_a_fresh_preparation_computed(self) -> None:
        """The invariant. Everything else here is a detail of it."""
        fresh = prepared()
        read_back = _prepared_from_dict(fresh.to_dict(), directory=Path("/tmp/prepared/abc"))
        assert read_back.dimensions_m == fresh.dimensions_m
        assert read_back.bounds_min_m == fresh.bounds_min_m
        assert read_back.bounds_max_m == fresh.bounds_max_m
        assert read_back.mass_kg == fresh.mass_kg

    def test_the_measurements_are_held_at_the_stored_precision(self) -> None:
        # Rounding happens where the value is made, so the object and its sidecar are one
        # representation rather than two that have to be kept in step.
        fresh = prepared()
        assert fresh.dimensions_m.to_list() == [round(v, STORED_DECIMALS) for v in
                                               (1.3700799942016602, 0.7602200508117676,
                                                0.7499952153593767)]
        assert fresh.mass_kg == round(312.4436893861899, STORED_DECIMALS)

    def test_the_sidecar_does_not_round_a_second_time(self) -> None:
        # A second copy of the rule is the thing that drifts; if the stored numbers differed from
        # the held ones, the round trip above would fail -- so this only asserts the shape.
        payload = prepared().to_dict()
        assert payload["dimensions_m"] == prepared().dimensions_m.to_list()
        assert payload["mass_kg"] == prepared().mass_kg

    def test_the_whole_object_survives_the_round_trip(self) -> None:
        fresh = prepared()
        read_back = _prepared_from_dict(fresh.to_dict(), directory=Path("/tmp/prepared/abc"))
        assert read_back.asset_key == fresh.asset_key
        assert read_back.vertex_count == fresh.vertex_count
        assert read_back.collision_face_count == fresh.collision_face_count
        assert read_back.origin_convention == fresh.origin_convention
        assert read_back.provenance.status is fresh.provenance.status

    def test_the_sidecar_keeps_the_paths_portable(self) -> None:
        """Stored as names and re-joined to the directory, so a moved preparation directory does
        not silently point at the old one."""
        payload = prepared().to_dict()
        assert payload["visual_obj"] == "abc.obj"
        read_back = _prepared_from_dict(payload, directory=Path("/elsewhere/abc"))
        assert read_back.visual_obj == Path("/elsewhere/abc/abc.obj")
