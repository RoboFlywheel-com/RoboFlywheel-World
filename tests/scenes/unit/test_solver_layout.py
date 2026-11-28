# SPDX-License-Identifier: Apache-2.0
"""Placement search and the ON predicate.

The negative that matters is an object that cannot be placed: it must yield NO_PLACEMENT_CAPACITY
with the rejections showing how close it came, never a nudge into an illegal position. A solver
that silently shifts an object is one whose failures surface later as unexplained collisions.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.contracts.asset import AffordanceStatus, Provenance, Receptacle
from rfw_w.scenes.contracts.scene_spec import Predicate
from rfw_w.scenes.contracts.tool import EvidenceStatus
from rfw_w.scenes.contracts.units import Quat, Transform, Vec3
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.layout import (
    SPAWN_CLEARANCE_M,
    ApproachPreference,
    Footprint,
    LayoutError,
    OccupancyMap,
    SurfacePlacementConfig,
    place_in_zone,
    place_on_surface,
)
from rfw_w.scenes.geometry.robot import profile_for
from rfw_w.scenes.solvers.relation import (
    RelationRequest,
    evaluate_inside,
    evaluate_relation,
)
from rfw_w.scenes.solvers.support import PlacedInstance, Surface

VERSION = "a" * 64

DESK_TOP = Surface(
    surface_id="desk:top",
    owner_instance="desk",
    polygon=Polygon.rectangle(0.0, 0.0, 1.2, 0.6),
    plane_z_m=0.75,
    provenance=AffordanceStatus.MEASURED,
)
NOTCHED_TOP = Surface(
    surface_id="desk:top",
    owner_instance="desk",
    polygon=Polygon.from_sequence(
        [(0.0, 0.0), (1.2, 0.0), (1.2, 0.4), (0.9, 0.4), (0.9, 0.6), (0.0, 0.6)]
    ),
    plane_z_m=0.75,
    provenance=AffordanceStatus.MEASURED,
)
CUP = Footprint((0.05, 0.05))
BIG = Footprint((0.5, 0.4))


class TestFootprint:
    def test_dimensions_and_area(self) -> None:
        assert CUP.width_m == pytest.approx(0.1)
        assert CUP.area_m2 == pytest.approx(0.01)

    def test_rejects_degenerate_extents(self) -> None:
        with pytest.raises(LayoutError):
            Footprint((0.0, 0.05))

    def test_polygon_is_centred(self) -> None:
        bounds = CUP.at(0.5, 0.3).bounds()
        assert bounds == pytest.approx((0.45, 0.25, 0.55, 0.35))

    def test_from_extents_halves(self) -> None:
        assert Footprint.from_extents(Vec3(0.2, 0.1, 0.3)).half_extents == pytest.approx((0.1, 0.05))


class TestOccupancy:
    def test_records_and_lists(self) -> None:
        occupancy = OccupancyMap()
        occupancy.add("desk", Polygon.rectangle(0, 0, 1, 1))
        assert len(occupancy) == 1

    def test_owner_excludes_itself_from_its_own_blockers(self) -> None:
        # Placing a cup on a desk must not be obstructed by the desk.
        occupancy = OccupancyMap()
        occupancy.add("desk", Polygon.rectangle(0, 0, 1, 1))
        assert occupancy.blockers_for("desk") == ()
        assert len(occupancy.blockers_for("cup")) == 1

    def test_requires_an_owner(self) -> None:
        with pytest.raises(LayoutError, match="must name its owner"):
            OccupancyMap().add("", Polygon.rectangle(0, 0, 1, 1))


class TestSurfacePlacement:
    def test_places_a_cup_on_a_desk(self) -> None:
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
        )
        assert outcome.placed
        assert outcome.placement.surface_id == "desk:top"
        # Base just above the surface plane: half the height, plus the spawn clearance.
        assert outcome.placement.position_m.z == pytest.approx(0.8 + SPAWN_CLEARANCE_M)

    def test_placement_stays_inside_the_surface(self) -> None:
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
        )
        assert Polygon.rectangle(0, 0, 1.2, 0.6).bounds()

    def test_prefers_the_edge(self) -> None:
        """Outermost first, because objects against an edge are the common intent."""
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
        )
        centre = outcome.placement.position_m
        distance_to_edge = min(centre.x, 1.2 - centre.x, centre.y, 0.6 - centre.y)
        # Exactly the closest the margins allow: the 0.01 m edge margin plus the 0.05 m
        # half-extent. Anything further would mean the outermost-first ordering was not honoured.
        assert distance_to_edge == pytest.approx(0.06)

    def test_approach_scoring_is_opt_in_and_can_choose_a_different_candidate(self) -> None:
        baseline = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
            config=SurfacePlacementConfig(step_m=0.12),
        )
        scored = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
            config=SurfacePlacementConfig(step_m=0.12),
            approach=ApproachPreference(
                # A local work aisle along the front of the desk. No room shell is involved.
                region=Polygon.rectangle(-0.35, -1.0, 1.55, 0.0),
                profile=profile_for("franka_panda"),
            ),
        )
        assert baseline.placement.position_m.x == pytest.approx(0.06)
        assert scored.placed
        assert scored.placement.position_m.x != baseline.placement.position_m.x
        assert "approach direction" in scored.detail

    def test_approach_requirement_fails_when_no_base_position_exists(self) -> None:
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
            config=SurfacePlacementConfig(step_m=0.12),
            approach=ApproachPreference(
                region=Polygon.rectangle(0.0, 0.0, 0.2, 0.2),
                profile=profile_for("franka_panda"),
            ),
        )
        assert not outcome.placed
        assert outcome.failure_code == "NO_PLACEMENT_CAPACITY"
        assert any("approach direction" in rejection.reason for rejection in outcome.rejections)

    def test_too_big_for_the_surface_fails_without_moving_it(self) -> None:
        outcome = place_on_surface(
            instance_id="desk", entity_id="desk", footprint=BIG, surface=DESK_TOP,
            object_height_m=0.75, occupancy=OccupancyMap(),
        )
        assert not outcome.placed
        assert outcome.failure_code == "NO_PLACEMENT_CAPACITY"
        assert "does not fit" in outcome.detail

    def test_avoid_an_already_placed_object(self) -> None:
        occupancy = OccupancyMap()
        occupancy.add("mug_already_there", Polygon.rectangle(0.0, 0.0, 0.4, 0.6))
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=occupancy,
        )
        assert outcome.placed
        assert outcome.placement.position_m.x > 0.4

    def test_full_surface_reports_rejections_rather_than_forcing(self) -> None:
        occupancy = OccupancyMap()
        occupancy.add("blocker", Polygon.rectangle(-1.0, -1.0, 2.0, 2.0))
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=occupancy,
        )
        assert not outcome.placed
        assert outcome.failure_code == "NO_PLACEMENT_CAPACITY"
        assert outcome.rejections
        assert outcome.rejections[0].reason

    def test_candidate_budget_is_bounded(self) -> None:
        occupancy = OccupancyMap()
        occupancy.add("blocker", Polygon.rectangle(-1.0, -1.0, 2.0, 2.0))
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=occupancy,
            config=SurfacePlacementConfig(step_m=0.005, max_candidates=20),
        )
        assert len(outcome.rejections) <= 21

    def test_dynamic_objects_spawn_clear_of_their_support(self) -> None:
        """A body spawned exactly on its support lands a fraction of a millimetre inside it.

        Measured on a real scene: a cup 0.3 mm into a desk top was ejected half a metre in one step
        and ended on the floor, while a notebook 0.5 mm clear never moved. The clearance is what
        keeps a spawned body out of the solver's depenetration path, and it gives the settle
        something to actually settle.
        """
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
        )
        base = outcome.placement.position_m.z - 0.1 / 2.0
        assert base > DESK_TOP.plane_z_m

    def test_the_clearance_is_configurable(self) -> None:
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
            config=SurfacePlacementConfig(spawn_clearance_m=0.02),
        )
        assert outcome.placement.position_m.z == pytest.approx(0.75 + 0.05 + 0.02)

    def test_rejects_a_non_positive_object_height(self) -> None:
        with pytest.raises(LayoutError, match="height"):
            place_on_surface(
                instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
                object_height_m=0.0, occupancy=OccupancyMap(),
            )

    def test_occupancy_report_is_present(self) -> None:
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=OccupancyMap(),
        )
        payload = outcome.to_dict()
        assert payload["placed"] is True
        assert payload["placement"]["surface_id"] == "desk:top"


class TestZonePlacement:
    ZONE = Polygon.rectangle(0.2, 0.2, 3.0, 2.0)

    def test_places_furniture_in_a_zone(self) -> None:
        outcome = place_in_zone(
            instance_id="desk", entity_id="desk", footprint=Footprint((0.3, 0.6)),
            zone=self.ZONE, object_height_m=0.75, occupancy=OccupancyMap(),
        )
        assert outcome.placed
        assert outcome.placement.surface_id is None

    def test_leaves_a_gap_between_objects(self) -> None:
        occupancy = OccupancyMap()
        occupancy.add("cabinet", Polygon.rectangle(0.2, 0.2, 1.4, 2.0))
        outcome = place_in_zone(
            instance_id="desk", entity_id="desk", footprint=Footprint((0.3, 0.6)),
            zone=self.ZONE, object_height_m=0.75, occupancy=occupancy,
            config=SurfacePlacementConfig(step_m=0.05, object_gap_m=0.1),
        )
        assert outcome.placed
        assert outcome.placement.position_m.x > 1.4

    def test_impossible_zone_fails(self) -> None:
        outcome = place_in_zone(
            instance_id="desk", entity_id="desk", footprint=Footprint((5.0, 5.0)),
            zone=self.ZONE, object_height_m=0.75, occupancy=OccupancyMap(),
        )
        assert not outcome.placed
        assert outcome.failure_code == "NO_PLACEMENT_CAPACITY"

    def test_wall_bias_prefers_the_wall_without_excluding_the_zone(self) -> None:
        wall = Polygon.rectangle(0.2, 1.9, 3.0, 2.0)
        outcome = place_in_zone(
            instance_id="desk", entity_id="desk", footprint=Footprint((0.3, 0.3)),
            zone=self.ZONE, object_height_m=0.75, occupancy=OccupancyMap(), against_wall=wall,
        )
        assert outcome.placed
        assert outcome.placement.position_m.y > 1.4

    def test_zone_that_cannot_hold_the_footprint_fails_rather_than_overhanging(self) -> None:
        """A zone keeps furniture out of door sweeps, so overhanging it can block a door.

        Placing outside the zone and flagging it would trade a visible layout failure for an
        invisible functional one.
        """
        # A thin L whose arms are narrower than the footprint: the footprint fits the zone's
        # bounding box and none of the zone.
        thin_l = Polygon.from_sequence(
            [(0.0, 0.0), (2.0, 0.0), (2.0, 0.2), (0.2, 0.2), (0.2, 2.0), (0.0, 2.0)]
        )
        outcome = place_in_zone(
            instance_id="desk", entity_id="desk", footprint=Footprint((0.15, 0.15)),
            zone=thin_l, object_height_m=0.75, occupancy=OccupancyMap(),
        )
        assert not outcome.placed
        assert outcome.failure_code == "NO_PLACEMENT_CAPACITY"
        assert "leaves the zone" in outcome.rejections[0].reason
        assert "must change" in outcome.detail


class TestOnPredicate:
    def request(self, predicate: Predicate = Predicate.ON) -> RelationRequest:
        return RelationRequest(
            relation_id="r1", subject_instance="cup", predicate=predicate,
            object_instance="desk", scene_version=VERSION,
        )

    def subject(self, x: float, y: float, base_z: float) -> PlacedInstance:
        return PlacedInstance("cup", CUP.at(x, y), base_z, True)

    def test_on_the_desk_passes_with_a_witness(self) -> None:
        evidence = evaluate_relation(
            self.request(), subject=self.subject(0.4, 0.3, 0.75),
            support_instance="desk", surfaces=[DESK_TOP],
        )
        assert evidence.status is EvidenceStatus.PASS
        assert evidence.measured_value > 0.5
        assert evidence.unit
        assert evidence.threshold_profile
        assert evidence.witness_refs
        assert evidence.state_hash == VERSION

    def test_on_the_floor_fails_the_relation(self) -> None:
        """The runtime probe's negative case, as a predicate with a named owner."""
        floor = Surface("floor", None, Polygon.rectangle(-5, -5, 5, 5), 0.0, AffordanceStatus.MEASURED)
        evidence = evaluate_relation(
            self.request(), subject=self.subject(3.0, 3.0, 0.0),
            support_instance="desk", surfaces=[floor, DESK_TOP],
        )
        assert evidence.status is EvidenceStatus.FAIL
        assert "floor" in evidence.detail

    def test_floating_object_fails(self) -> None:
        evidence = evaluate_relation(
            self.request(), subject=self.subject(0.4, 0.3, 2.0),
            support_instance="desk", surfaces=[DESK_TOP],
        )
        assert evidence.status is EvidenceStatus.FAIL

    def test_notch_is_not_the_desk(self) -> None:
        """Resting at desk height over the notch rests on nothing, not on the desk."""
        evidence = evaluate_relation(
            self.request(), subject=self.subject(0.98, 0.5, 0.75),
            support_instance="desk", surfaces=[NOTCHED_TOP],
        )
        assert evidence.status is EvidenceStatus.FAIL

    def test_evidence_is_bound_to_the_scene_version(self) -> None:
        evidence = evaluate_relation(
            self.request(), subject=self.subject(0.4, 0.3, 0.75),
            support_instance="desk", surfaces=[DESK_TOP],
        )
        assert evidence.is_current_for(VERSION)
        assert not evidence.is_current_for("b" * 64)

    def test_wrong_predicate_is_refused(self) -> None:
        from rfw_w.scenes.contracts.scene_spec import Predicate as P
        from rfw_w.scenes.solvers.relation import evaluate_on

        with pytest.raises(Exception):
            evaluate_on(
                self.request(P.INSIDE), subject=self.subject(0.4, 0.3, 0.75),
                support_instance="desk", surfaces=[DESK_TOP],
            )


class TestInsidePredicate:
    def request(self) -> RelationRequest:
        return RelationRequest(
            relation_id="r2", subject_instance="cup", predicate=Predicate.INSIDE,
            object_instance="cupboard", scene_version=VERSION,
        )

    def test_uncertified_cavity_is_unsupported_not_failed(self) -> None:
        """The relation could not be decided, and blaming the scene for a missing measurement
        would be the wrong verdict."""
        receptacle = Receptacle(
            cavity_id="inner", opening_frame=Transform.identity("world"),
            capacity_m3=0.01, opening_width_m=0.2,
            provenance=Provenance(method="sdf", version="v1", status=AffordanceStatus.ESTIMATED),
        )
        evidence = evaluate_inside(self.request(), receptacle=receptacle, scene_version=VERSION)
        assert evidence.status is EvidenceStatus.UNSUPPORTED
        assert "estimated" in evidence.detail

    def test_no_receptacle_is_unsupported(self) -> None:
        evidence = evaluate_inside(self.request(), receptacle=None, scene_version=VERSION)
        assert evidence.status is EvidenceStatus.UNSUPPORTED
        assert "bounding box" in evidence.detail

    def test_certified_cavity_is_not_run_rather_than_assumed(self) -> None:
        """A measured cavity makes INSIDE expressible, but nothing has probed it yet, and a
        probe that was never run must not become a pass."""
        receptacle = Receptacle(
            cavity_id="inner", opening_frame=Transform.identity("world"),
            capacity_m3=0.01, opening_width_m=0.2,
            provenance=Provenance(method="sdf", version="v1", status=AffordanceStatus.MEASURED),
        )
        evidence = evaluate_inside(self.request(), receptacle=receptacle, scene_version=VERSION)
        assert evidence.status is EvidenceStatus.NOT_RUN
        assert not evidence.satisfies(VERSION)


class TestUnimplementedPredicates:
    def test_declared_but_unimplemented_is_unsupported(self) -> None:
        """CAPPING and the rest are in the schema and have no implementation; that is a missing
        capability, not an unexercised one, so it reports unsupported."""
        request = RelationRequest(
            relation_id="r3", subject_instance="pot", predicate=Predicate.CAPPING,
            object_instance="lid", scene_version=VERSION,
        )
        evidence = evaluate_relation(
            request,
            subject=PlacedInstance("pot", CUP.at(0.4, 0.3), 0.75, True),
            support_instance="desk", surfaces=[DESK_TOP],
        )
        assert evidence.status is EvidenceStatus.UNSUPPORTED
        assert "no implementation" in evidence.detail

    def test_relation_needs_two_instances(self) -> None:
        with pytest.raises(Exception):
            RelationRequest(
                relation_id="r4", subject_instance="cup", predicate=Predicate.ON,
                object_instance="cup", scene_version=VERSION,
            )


class TestOriginConvention:
    """Placing a mesh whose origin is not where the solver assumes puts it in the wrong place.

    The failure is silent: the scene renders, the depth is valid and the object settles, just
    somewhere else. It was found by a rendered cup being buried inside a desk.
    """

    ZONE = Polygon.rectangle(0.0, 0.0, 3.0, 3.0)

    def test_centred_mesh_is_lifted_by_half_its_height(self) -> None:
        outcome = place_in_zone(
            instance_id="desk", entity_id="desk", footprint=Footprint((0.3, 0.3)),
            zone=self.ZONE, object_height_m=0.8, occupancy=OccupancyMap(),
            origin_convention="centre",
        )
        assert outcome.placement.position_m.z == pytest.approx(0.4)

    def test_base_at_origin_mesh_sits_on_the_floor(self) -> None:
        outcome = place_in_zone(
            instance_id="desk", entity_id="desk", footprint=Footprint((0.3, 0.3)),
            zone=self.ZONE, object_height_m=0.8, occupancy=OccupancyMap(),
            origin_convention="base",
        )
        assert outcome.placement.position_m.z == pytest.approx(0.0)

    def test_the_two_conventions_differ_by_half_the_height(self) -> None:
        kwargs = {
            "instance_id": "desk", "entity_id": "desk", "footprint": Footprint((0.3, 0.3)),
            "zone": self.ZONE, "object_height_m": 0.8, "occupancy": OccupancyMap(),
        }
        centred = place_in_zone(**kwargs, origin_convention="centre").placement.position_m.z
        based = place_in_zone(**kwargs, origin_convention="base").placement.position_m.z
        assert centred - based == pytest.approx(0.4)

    def test_an_unknown_convention_is_refused(self) -> None:
        # Guessing would reintroduce exactly the silent misplacement this parameter prevents.
        with pytest.raises(LayoutError, match="unknown origin convention"):
            place_in_zone(
                instance_id="desk", entity_id="desk", footprint=Footprint((0.3, 0.3)),
                zone=self.ZONE, object_height_m=0.8, occupancy=OccupancyMap(),
                origin_convention="centroid_ish",
            )

    def test_a_support_owner_does_not_obstruct_things_placed_on_it(self) -> None:
        """Occupancy is planar, so the desk's floor footprint overlaps a cup on its top.

        Excluding the support's owner is the statement "the thing supporting you does not obstruct
        you"; without it every position on a surface is rejected.
        """
        occupancy = OccupancyMap()
        occupancy.add("desk", Polygon.rectangle(-1.0, -1.0, 1.0, 1.0))
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=occupancy,
        )
        assert outcome.placed

    def test_a_different_object_still_obstructs(self) -> None:
        occupancy = OccupancyMap()
        occupancy.add("other", Polygon.rectangle(-1.0, -1.0, 2.0, 2.0))
        outcome = place_on_surface(
            instance_id="cup", entity_id="cup", footprint=CUP, surface=DESK_TOP,
            object_height_m=0.1, occupancy=occupancy,
        )
        assert not outcome.placed
