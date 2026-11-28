# SPDX-License-Identifier: Apache-2.0
"""Asset record rules that decide whether a scene contains real objects.

Two rules carry the weight here:

* a missing affordance is neither false nor true, so a hard functional requirement against
  an unmeasured cavity cannot be quietly satisfied;
* a primitive may never stand in for a retrieved task asset, so a scene built from boxes
  cannot pass a contact predicate and be reported as containing the requested objects.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.asset import (
    AffordanceStatus,
    AssetError,
    AssetHandle,
    AssetKey,
    AssetRecordV3,
    CapabilityStatus,
    CollisionPart,
    NormalisationRecord,
    Provenance,
    Receptacle,
    RigidPhysics,
    SupportPolygon,
)
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.scene_spec import PhysicsType
from rfw_w.scenes.contracts.units import Transform, Quat, Vec3

SOURCE_HASH = content_hash({"asset": "mug_0001"})
IDENTITY = NormalisationRecord(
    applied_scale=1.0,
    applied_translation=Vec3.origin(),
    source_frame="asset",
    target_frame="world",
    method="none",
    version="v1",
)
MEASURED = Provenance(method="mesh_analysis", version="v1", status=AffordanceStatus.MEASURED)


def physics(mass: float = 0.18, inertia: tuple[float, ...] | None = None) -> RigidPhysics:
    return RigidPhysics(
        mass_kg=mass,
        center_of_mass_m=Vec3(0.0, 0.0, 0.04),
        inertia_kg_m2=inertia or (1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4),
        static_friction=0.6,
        dynamic_friction=0.5,
        restitution=0.05,
        provenance=MEASURED,
    )


def record(**overrides: object) -> AssetRecordV3:
    payload: dict[str, object] = {
        "key": AssetKey(provider="simready", library_version="v0.2", asset_id="0_0_0056"),
        "source_hash": SOURCE_HASH,
        "license": "CC-BY-4.0",
        "semantic": "mug",
        "super_category": "Mug",
        "physics_type": PhysicsType.RIGID,
        "dimensions_m": Vec3(0.09, 0.09, 0.10),
        "bounds_min_m": Vec3(-0.045, -0.045, 0.0),
        "bounds_max_m": Vec3(0.045, 0.045, 0.10),
        "canonical_frame": "world",
        "normalisation": IDENTITY,
        "visual_uri": "mesh/0_0_0056.obj",
        "collision_parts": (
            CollisionPart(
                part_id="main", source="collision_mesh", approximate=False,
                vertex_count=512, face_count=1024, provenance=MEASURED,
            ),
        ),
        "materials": ("ceramic",),
        "rigid_physics": physics(),
        "capabilities": {"isaac": CapabilityStatus.SUPPORTED},
        "prepared_version": "prep/v1",
    }
    payload.update(overrides)
    return AssetRecordV3(**payload)  # type: ignore[arg-type]


class TestIdentity:
    def test_asset_key_needs_all_three_parts(self) -> None:
        # Keying on asset_id alone lets two libraries that both have "0001" be swapped.
        for missing in ("provider", "library_version", "asset_id"):
            kwargs = {"provider": "simready", "library_version": "v0.2", "asset_id": "0_0_0056"}
            kwargs[missing] = ""
            with pytest.raises(AssetError, match=missing):
                AssetKey(**kwargs)

    def test_semantic_id_is_opaque(self) -> None:
        # What the model sees must not reveal the provider or the library version.
        key = AssetKey(provider="simready", library_version="v0.2", asset_id="0_0_0056")
        assert key.semantic_id.startswith("asset:")
        for leak in ("simready", "v0.2", "0_0_0056"):
            assert leak not in key.semantic_id

    def test_semantic_view_strips_paths_and_hashes(self) -> None:
        view = record().semantic_view()
        assert "visual_uri" not in view
        assert "source_hash" not in view
        assert "key" not in view
        assert set(view) == {
            "semantic_id", "semantic", "category", "physics_type", "dimensions_m",
            "materials", "has_support_surface", "has_receptacle",
        }


class TestNormalisation:
    def test_identity_allows_scale_one(self) -> None:
        IDENTITY.assert_not_double_applied(1.0)

    def test_refuses_a_second_scale(self) -> None:
        already = NormalisationRecord(
            applied_scale=0.01, applied_translation=Vec3.origin(),
            source_frame="cm", target_frame="m", method="unit_fix", version="v1",
        )
        with pytest.raises(AssetError, match="already been normalised"):
            already.assert_not_double_applied(1.0)

    def test_refuses_scaling_an_unscaled_asset(self) -> None:
        with pytest.raises(AssetError, match="scale=1"):
            IDENTITY.assert_not_double_applied(2.0)

    def test_rejects_non_positive_scale(self) -> None:
        with pytest.raises(AssetError, match="positive"):
            NormalisationRecord(
                applied_scale=0.0, applied_translation=Vec3.origin(),
                source_frame="a", target_frame="b", method="m", version="v1",
            )


class TestRigidPhysics:
    def test_rejects_non_positive_mass(self) -> None:
        with pytest.raises(AssetError, match="mass"):
            physics(mass=0.0)

    def test_recognises_spd_inertia(self) -> None:
        assert physics().inertia_is_spd
        physics().assert_simulable()

    def test_rejects_non_spd_inertia(self) -> None:
        # A negative eigenvalue makes a body explode or freeze rather than simulate.
        bad = physics(inertia=(-1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4))
        assert not bad.inertia_is_spd
        with pytest.raises(AssetError, match="not symmetric positive definite"):
            bad.assert_simulable()

    def test_rejects_out_of_range_restitution(self) -> None:
        with pytest.raises(AssetError, match="restitution"):
            RigidPhysics(
                mass_kg=1.0, center_of_mass_m=Vec3.origin(),
                inertia_kg_m2=(1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4),
                static_friction=0.5, dynamic_friction=0.4, restitution=1.5, provenance=MEASURED,
            )


class TestAffordances:
    def test_unknown_is_not_usable_for_a_hard_requirement(self) -> None:
        """A cavity nobody measured cannot certify an INSIDE relation.

        If `unknown` were falsy or defaulted to true, an unmeasured receptacle would pass a
        hard functional gate, which is precisely the conflation §5.2 forbids.
        """
        assert not AffordanceStatus.UNKNOWN.usable_for_hard_requirement
        assert not AffordanceStatus.ESTIMATED.usable_for_hard_requirement
        assert AffordanceStatus.MEASURED.usable_for_hard_requirement
        assert AffordanceStatus.ANNOTATED.usable_for_hard_requirement
        # `constructed` is usable for the same reason `measured` is, and for a different one:
        # the geometry is ours and was built to the dimension the task was solved against, so
        # there is nothing left for an estimate to be wrong about.
        assert AffordanceStatus.CONSTRUCTED.usable_for_hard_requirement

    def test_affordance_status_is_never_a_bool(self) -> None:
        # `bool(AffordanceStatus.UNKNOWN)` would be True for a plain Enum; the property above
        # is the only sanctioned way to ask, which keeps the distinction explicit.
        assert AffordanceStatus.UNKNOWN is not False
        assert AffordanceStatus.UNKNOWN is not True

    def test_support_polygon_needs_three_vertices(self) -> None:
        with pytest.raises(AssetError, match="at least 3 vertices"):
            SupportPolygon(
                polygon_id="top", vertices_xy=((0.0, 0.0), (1.0, 0.0)),
                plane_z_m=0.75, provenance=MEASURED,
            )

    def test_rectangle_detection(self) -> None:
        rect = SupportPolygon(
            polygon_id="top",
            vertices_xy=((0.0, 0.0), (1.2, 0.0), (1.2, 0.6), (0.0, 0.6)),
            plane_z_m=0.75, provenance=MEASURED,
        )
        assert rect.is_rectangle
        # An L-shaped support surface is not a rectangle, and a check that assumed otherwise
        # would place objects in the notch.
        notch = SupportPolygon(
            polygon_id="l",
            vertices_xy=((0.0, 0.0), (1.2, 0.0), (1.2, 0.3), (0.5, 0.3), (0.5, 0.6), (0.0, 0.6)),
            plane_z_m=0.75, provenance=MEASURED,
        )
        assert not notch.is_rectangle

    def test_only_a_measured_receptacle_is_certified(self) -> None:
        def cavity(status: AffordanceStatus) -> Receptacle:
            return Receptacle(
                cavity_id="inner",
                opening_frame=Transform.identity("world"),
                capacity_m3=0.002,
                opening_width_m=0.08,
                provenance=Provenance(method="sdf", version="v1", status=status),
            )

        assert cavity(AffordanceStatus.MEASURED).is_certified
        assert not cavity(AffordanceStatus.ESTIMATED).is_certified
        assert not cavity(AffordanceStatus.UNKNOWN).is_certified

    def test_receptacle_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(AssetError, match="capacity"):
            Receptacle(
                cavity_id="inner", opening_frame=Transform.identity("world"),
                capacity_m3=0.0, opening_width_m=0.08, provenance=MEASURED,
            )


class TestSubstitution:
    def test_primitive_collision_is_flagged(self) -> None:
        part = CollisionPart(
            part_id="box", source="primitive_box", approximate=True,
            vertex_count=8, face_count=12, provenance=MEASURED,
        )
        assert part.is_primitive_substitute

    def test_primitive_stand_in_is_rejected_for_a_task_target(self) -> None:
        """A box may proxy a static support; it may never be the requested cup."""
        rec = record(
            collision_parts=(
                CollisionPart(part_id="box", source="primitive_box", approximate=True,
                              vertex_count=8, face_count=12, provenance=MEASURED),
            )
        )
        assert rec.is_primitive_substitute
        with pytest.raises(AssetError, match="primitive"):
            rec.assert_usable_as_task_target()

    def test_real_collision_mesh_is_accepted(self) -> None:
        record().assert_usable_as_task_target()

    def test_asset_without_collision_geometry_is_rejected(self) -> None:
        rec = record(collision_parts=())
        with pytest.raises(AssetError, match="no collision geometry"):
            rec.assert_usable_as_task_target()

    def test_rigid_asset_must_carry_mass_properties(self) -> None:
        with pytest.raises(AssetError, match="no mass properties"):
            record(rigid_physics=None)


class TestCapabilityReporting:
    def test_unlisted_backend_defaults_to_untested(self) -> None:
        # Defaulting to supported would let an untested backend be reported as working.
        assert record().capability_for("mujoco") is CapabilityStatus.UNTESTED

    def test_declared_capability_is_reported(self) -> None:
        assert record().capability_for("isaac") is CapabilityStatus.SUPPORTED

    def test_license_is_required(self) -> None:
        with pytest.raises(AssetError, match="license"):
            record(license="")

    def test_source_hash_must_be_a_digest(self) -> None:
        with pytest.raises(AssetError, match="source_hash"):
            record(source_hash="deadbeef")


class TestHandle:
    def test_handle_requires_collision_geometry(self) -> None:
        with pytest.raises(AssetError, match="no collision"):
            AssetHandle(
                key=AssetKey(provider="simready", library_version="v0.2", asset_id="0_0_0056"),
                visual_uri="mesh/0_0_0056.obj", collision_uris=(),
                metadata_uri=None, package_hash=SOURCE_HASH,
            )

    def test_handle_carries_paths_but_exposes_only_the_semantic_id(self) -> None:
        handle = AssetHandle(
            key=AssetKey(provider="simready", library_version="v0.2", asset_id="0_0_0056"),
            visual_uri="/data/assets/mesh/0_0_0056.obj",
            collision_uris=("/data/assets/mesh/0_0_0056_collision.obj",),
            metadata_uri="/data/assets/0_0_0056.urdf",
            package_hash=SOURCE_HASH,
        )
        assert handle.semantic_id == handle.key.semantic_id
        assert "/data/assets" in handle.to_dict()["visual_uri"]
