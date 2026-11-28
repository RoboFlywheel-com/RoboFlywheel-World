# SPDX-License-Identifier: Apache-2.0
"""Isaac worker request and report validation.

The report check is the important one. Its job is to catch a worker that claims a pass while its
own captures say otherwise, because everything downstream trusts the report instead of looking at
the pixels. Each way that has actually happened or could happen is a test here.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.workers.isaac.protocol import (
    CameraRequest,
    CaptureReport,
    MeshInstance,
    ProtocolError,
    RoomGeometry,
    WorkerReport,
    WorkerRequest,
)

VERSION = "0" * 64


def camera(**overrides: object) -> CameraRequest:
    payload: dict[str, object] = {
        "camera_id": "overview", "purpose": "layout",
        "eye": (1.0, 1.0, 1.5), "target": (0.0, 0.0, 0.2),
    }
    payload.update(overrides)
    return CameraRequest(**payload)  # type: ignore[arg-type]


def capture(**overrides: object) -> CaptureReport:
    payload: dict[str, object] = {
        "camera_id": "overview", "rgb_path": "overview.png", "depth_path": "overview_depth_m.npy",
        "instance_id_path": "overview_instance_ids.npy", "resolution": (960, 640),
        "depth_convention": "camera_z_meters", "depth_valid_fraction": 0.97,
        "depth_min_m": 1.9, "depth_max_m": 2.2, "color_std": 13.4,
        "visible_instance_ids": (1, 2), "target_instance_ids": ("cup_0",),
        "instance_id_map": {"2": "cup_0"},
        "target_visible": True, "nonblank": True,
    }
    payload.update(overrides)
    return CaptureReport(**payload)  # type: ignore[arg-type]


def report(**overrides: object) -> WorkerReport:
    payload: dict[str, object] = {
        "scene_id": "s", "scene_version": VERSION, "status": "passed",
        "isaac_version": "4.5.0.rc.36", "captures": (capture(),),
        "instance_id_map": {"2": "cup_0", "3": "cup_1"},
        "instance_id_map_source": "annotator_idToLabels",
        "annotator_family": "instance_segmentation",
        "settle": {"performed": True}, "capture_state": "settled",
    }
    payload.update(overrides)
    return WorkerReport(**payload)  # type: ignore[arg-type]


class TestMeshInstance:
    def test_dynamic_requires_a_mass(self) -> None:
        """A dynamic body with no mass cannot be settled, so its settle result would describe
        nothing and the gate would pass on an object that never moved."""
        with pytest.raises(ProtocolError, match="positive mass"):
            MeshInstance(
                instance_id="cup", entity_id="cup", visual_uri="/a.obj", collision_uri=None,
                translation=(0, 0, 0), rotation_wxyz=(1, 0, 0, 0), dynamic=True,
            )

    def test_static_needs_no_mass(self) -> None:
        MeshInstance(
            instance_id="desk", entity_id="desk", visual_uri="/a.obj", collision_uri=None,
            translation=(0, 0, 0), rotation_wxyz=(1, 0, 0, 0),
        )

    def test_requires_a_visual_mesh(self) -> None:
        with pytest.raises(ProtocolError, match="visual mesh"):
            MeshInstance("cup", "cup", "", None, (0, 0, 0), (1, 0, 0, 0))

    def test_pose_shape_is_checked(self) -> None:
        with pytest.raises(ProtocolError, match="pose"):
            MeshInstance("cup", "cup", "/a.obj", None, (0, 0), (1, 0, 0, 0))

    def test_round_trips(self) -> None:
        instance = MeshInstance("cup", "cup", "/a.obj", "/b.obj", (1, 2, 3), (1, 0, 0, 0),
                                mass_kg=0.2, dynamic=True, normalisation="y_up_to_z_up")
        assert MeshInstance.from_dict(instance.to_dict()).to_dict() == instance.to_dict()


class TestCameraRequest:
    def test_eye_and_target_must_differ(self) -> None:
        # A zero-length view direction has no defined orientation.
        with pytest.raises(ProtocolError, match="coincide"):
            camera(eye=(1.0, 1.0, 1.5), target=(1.0, 1.0, 1.5))

    def test_unknown_purpose_is_rejected(self) -> None:
        with pytest.raises(ProtocolError, match="purpose"):
            camera(purpose="decorative")

    def test_resolution_must_be_positive(self) -> None:
        with pytest.raises(ProtocolError, match="resolution"):
            camera(resolution=(0, 640))

    def test_the_framing_round_trips(self) -> None:
        """Focal length *and* both apertures: the aperture is what the field of view follows."""
        framed = camera(focal_length_mm=6.378678048182431, horizontal_aperture_mm=20.955,
                        vertical_aperture_mm=15.2908)
        assert CameraRequest.from_dict(framed.to_dict()).to_dict() == framed.to_dict()

    def test_an_omitted_aperture_keeps_usds_own(self) -> None:
        """The defaults are USD's fallbacks, so existing requests frame exactly as before."""
        assert (camera().horizontal_aperture_mm, camera().vertical_aperture_mm) == (20.955, 15.2908)

    @pytest.mark.parametrize("kwargs", [{"horizontal_aperture_mm": 0.0},
                                        {"vertical_aperture_mm": -1.0}])
    def test_a_non_positive_aperture_is_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ProtocolError, match="apertures must be positive"):
            camera(**kwargs)


class TestWorkerRequest:
    def request(self, **overrides: object) -> WorkerRequest:
        payload: dict[str, object] = {
            "scene_id": "s", "scene_version": VERSION,
            "room": RoomGeometry(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0)), 2.8),
            "instances": (MeshInstance("cup", "cup", "/a.obj", None, (1, 1, 0.5), (1, 0, 0, 0)),),
            "cameras": (camera(),),
        }
        payload.update(overrides)
        return WorkerRequest(**payload)  # type: ignore[arg-type]

    def test_accepts_a_minimal_request(self) -> None:
        assert self.request().scene_id == "s"

    def test_scene_version_must_be_a_hash(self) -> None:
        with pytest.raises(ProtocolError, match="scene hash"):
            self.request(scene_version="latest")

    def test_camera_may_not_target_an_unknown_instance(self) -> None:
        with pytest.raises(ProtocolError, match="unknown instances"):
            self.request(cameras=(camera(target_instance_ids=("ghost",)),))

    def test_a_camera_is_required(self) -> None:
        with pytest.raises(ProtocolError, match="at least one camera"):
            self.request(cameras=())

    def test_instances_are_required(self) -> None:
        with pytest.raises(ProtocolError, match="at least one instance"):
            self.request(instances=())

    def test_room_polygon_needs_three_vertices(self) -> None:
        with pytest.raises(ProtocolError, match="at least 3"):
            RoomGeometry(((0.0, 0.0), (1.0, 0.0)), 2.8)

    def test_round_trips(self) -> None:
        original = self.request()
        assert WorkerRequest.from_dict(original.to_dict()).to_dict() == original.to_dict()

    def test_rejects_an_unknown_schema_version(self) -> None:
        payload = self.request().to_dict()
        payload["schema_version"] = "pvscene.isaac-worker-request/v0"
        with pytest.raises(ProtocolError, match="schema_version"):
            WorkerRequest.from_dict(payload)


class TestSelfConsistency:
    """Each assertion here is a way a run has claimed success while showing nothing."""

    def test_a_consistent_pass_is_accepted(self) -> None:
        report().assert_self_consistent()

    def test_a_failed_report_is_not_second_guessed(self) -> None:
        # Only a claimed pass is scrutinised; a failure is already honest.
        report(status="failed", captures=(), instance_id_map={}, instance_id_map_source="").assert_self_consistent()

    def test_a_pass_with_no_capture_is_rejected(self) -> None:
        with pytest.raises(ProtocolError, match="no capture"):
            report(captures=()).assert_self_consistent()

    def test_a_pass_without_an_isaac_version_is_rejected(self) -> None:
        # The manual is explicit that the actual build decides compatibility, so a report that
        # does not name it cannot support a claim about it.
        with pytest.raises(ProtocolError, match="Isaac version"):
            report(isaac_version="").assert_self_consistent()

    def test_a_blank_image_is_rejected(self) -> None:
        with pytest.raises(ProtocolError, match="blank image"):
            report(captures=(capture(nonblank=False),)).assert_self_consistent()

    def test_all_invalid_depth_is_rejected(self) -> None:
        with pytest.raises(ProtocolError, match="depth buffer"):
            report(captures=(capture(depth_valid_fraction=0.0),)).assert_self_consistent()

    def test_an_unexpected_depth_convention_is_rejected(self) -> None:
        """v2 wrote radial distance; this worker writes camera-z. Comparing them as equal is
        comparing different quantities."""
        with pytest.raises(ProtocolError, match="depth convention"):
            report(captures=(capture(depth_convention="distance_to_camera"),)).assert_self_consistent()

    def test_a_pass_with_invisible_required_targets_is_rejected(self) -> None:
        with pytest.raises(ProtocolError, match="not visible"):
            report(captures=(capture(target_visible=False),)).assert_self_consistent()

    def test_a_pass_with_no_instance_map_is_rejected(self) -> None:
        """Without a map a rendered id cannot be resolved to an entity, and a critic would have to
        guess which object is which -- the thing the architecture forbids."""
        with pytest.raises(ProtocolError, match="instance-id mapping"):
            report(instance_id_map={}).assert_self_consistent()

    def test_a_pass_with_an_assumed_instance_map_is_rejected(self) -> None:
        """A map seeded from the request's own ordering reads as correct and is not: the renderer
        assigns its own ids, so such a map resolves ids to the wrong objects confidently."""
        with pytest.raises(ProtocolError, match="assumption rather than an observation"):
            report(instance_id_map_source="request_instance_order").assert_self_consistent()

    def test_an_empty_capture_must_say_that_it_is_empty(self) -> None:
        """SPEC-015 R3 changed this rule on purpose.

        The old rule rejected *any* saved channel with an empty id map, which is what a view of a
        bare wall legitimately produces -- and it made two such views cost a whole delivery its
        render and its critique. What is still a contradiction is an empty capture that does not
        declare itself: a reader has to be able to tell "nothing was there" from "nothing was
        recorded".
        """
        with pytest.raises(ProtocolError, match="holds no entity and does not say so"):
            report(captures=(capture(instance_id_map={}),)).assert_self_consistent()

    def test_an_empty_capture_that_declares_itself_is_accepted(self) -> None:
        report(captures=(capture(instance_id_map={}, contains_nothing=True),)).assert_self_consistent()

    def test_a_channel_with_neither_map_nor_visible_ids_cannot_be_read(self) -> None:
        empty_capture = capture(instance_id_map={}, visible_instance_ids=())
        with pytest.raises(ProtocolError, match="cannot be read"):
            report(captures=(empty_capture,)).assert_self_consistent()

    def test_a_view_that_misses_its_declaration_must_name_it(self) -> None:
        with pytest.raises(ProtocolError, match="does not name it"):
            report(captures=(capture(instance_id_map={}, contains_nothing=True),),
                   view_coverage=({"camera_id": "overview", "visible_instances": 0,
                                   "minimum_visible_instances": 1},)).assert_self_consistent()

    def test_a_view_that_misses_its_declaration_and_names_it_passes(self) -> None:
        # The whole point of R4: a miss is the *view's* named gap, not the delivery's death.
        report(captures=(capture(instance_id_map={}, contains_nothing=True),),
               view_coverage=({"camera_id": "overview", "visible_instances": 0,
                               "minimum_visible_instances": 1},),
               view_gaps=("VIEW_EMPTY(overview)",)).assert_self_consistent()

    def test_settled_physics_must_be_rendered_after_settle(self) -> None:
        with pytest.raises(ProtocolError, match="settled state"):
            report(capture_state="authored").assert_self_consistent()

    def test_unknown_target_visibility_is_not_treated_as_invisible(self) -> None:
        # `None` means "no mapping was available to decide". Failing on it would be wrong; the
        # separate instance-map check is what catches the real problem.
        report(captures=(capture(target_visible=None),)).assert_self_consistent()

    def test_report_round_trips(self) -> None:
        original = report()
        restored = WorkerReport.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_a_pass_without_the_annotator_family_is_rejected(self) -> None:
        """The families have different id spaces, so which one ran is part of the result."""
        with pytest.raises(ProtocolError, match="annotator"):
            report(annotator_family="").assert_self_consistent()

    def test_report_serialises_the_map_source(self) -> None:
        payload = report().to_dict()
        assert payload["instance_id_map_source"] == "annotator_idToLabels"
        assert payload["annotator_family"] == "instance_segmentation"
        # The request's own ordering is kept separately, because it is not the renderer's id space.
        assert "instance_index_map" in payload

    def test_report_carries_cross_camera_conflicts(self) -> None:
        """A disagreement about what an id means has to survive a round trip, or a report could be
        read as though every id it holds had one meaning."""
        payload = report(instance_id_conflicts=("2", "7")).to_dict()
        assert payload["instance_id_conflicts"] == ["2", "7"]
        assert WorkerReport.from_dict(payload).to_dict() == payload


class TestCrossCameraLabels:
    """One global map built from the cameras' own maps.

    The live failure this exists for: `overview_top` saved an instance-id channel with an empty
    map, and the global map still looked complete because it had been filled from whichever
    cameras were read before it. A per-camera map is what the gate checks; this is about what the
    report may still claim *globally* once the cameras disagree.
    """

    def test_agreement_is_kept(self) -> None:
        from rfw_w.scenes.workers.isaac.render_worker import _merge_camera_labels

        merged, conflicts = _merge_camera_labels(
            [("a", {"2": "cup_0"}), ("b", {"2": "cup_0", "3": "desk_0"})]
        )
        assert merged == {"2": "cup_0", "3": "desk_0"}
        assert conflicts == ()

    def test_an_id_two_cameras_disagree_about_is_dropped_and_named(self) -> None:
        """Taking the last camera's answer would resolve that id to the wrong object and sound
        confident doing it, which is the failure the map's source check exists to prevent."""
        from rfw_w.scenes.workers.isaac.render_worker import _merge_camera_labels

        merged, conflicts = _merge_camera_labels([("a", {"2": "cup_0"}), ("b", {"2": "mug_0"})])
        assert merged == {}
        assert conflicts == ("2",)

    def test_a_later_agreement_does_not_revive_a_conflicted_id(self) -> None:
        # Once an id means two things, a third camera agreeing with one of them does not make it
        # mean one thing. Re-adding it would hide the disagreement behind a majority vote.
        from rfw_w.scenes.workers.isaac.render_worker import _merge_camera_labels

        merged, conflicts = _merge_camera_labels(
            [("a", {"2": "cup_0"}), ("b", {"2": "mug_0"}), ("c", {"2": "cup_0"})]
        )
        assert merged == {}
        assert conflicts == ("2",)

    def test_a_camera_that_produced_nothing_does_not_hide_the_others(self) -> None:
        from rfw_w.scenes.workers.isaac.render_worker import _merge_camera_labels

        merged, conflicts = _merge_camera_labels(
            [("overview_top", {}), ("task_closeup", {"2": "cup_0"})]
        )
        assert merged == {"2": "cup_0"}
        assert conflicts == ()


class TestInstanceLabels:
    """The annotator's label values, measured against a live run."""

    def test_background_and_unlabelled_are_not_instances(self) -> None:
        from rfw_w.scenes.workers.isaac.render_worker import _instance_id_from_label

        assert _instance_id_from_label("BACKGROUND") is None
        assert _instance_id_from_label("UNLABELLED") is None

    def test_a_prim_path_yields_its_last_component(self) -> None:
        # Measured live: idToLabels returns prim paths, not the semantic class string.
        from rfw_w.scenes.workers.isaac.render_worker import _instance_id_from_label

        assert _instance_id_from_label("/World/Instances/cup_0") == "cup_0"
        assert _instance_id_from_label("/World/B") == "B"

    def test_a_mapping_value_is_unwrapped(self) -> None:
        from rfw_w.scenes.workers.isaac.render_worker import _instance_id_from_label

        assert _instance_id_from_label({"class": "/World/Instances/cup_1"}) == "cup_1"


class TestAnUnmeasuredFieldIsNullAndNotADefault:
    """SPEC-015 R5: a failed report must not read as a measurement.

    Before this, the failure path left `openings_requested`/`openings_cut`/`shell_reference`/
    `lighting` at their dataclass defaults -- 0 and {} -- and the delivery manifest derived
    `door_cut: false` from `openings_cut`, so a run whose render never happened published an
    assertion about a room nothing had looked at.
    """

    def test_a_failed_report_publishes_nulls_for_what_it_never_measured(self):
        report = WorkerReport(
            scene_id="office_pick_cup", scene_version="v" * 64, status="failed",
            error_type="ProtocolError", error_detail="VIEW_EMPTY(room_eye_south)",
        )
        payload = report.to_dict()
        for field in ("openings_requested", "openings_cut", "shell_reference", "lighting"):
            assert payload[field] is None, field
        assert WorkerReport.from_dict(payload).openings_cut is None

    def test_a_passed_report_still_carries_its_measurements(self):
        report = WorkerReport(
            scene_id="office_pick_cup", scene_version="v" * 64, status="passed",
            openings_requested=4, openings_cut=4,
            shell_reference={"mode": "referenced"}, lighting={"dome_intensity": 866.4},
        )
        payload = report.to_dict()
        assert payload["openings_requested"] == 4 and payload["openings_cut"] == 4
        assert payload["shell_reference"] == {"mode": "referenced"}
        assert payload["lighting"] == {"dome_intensity": 866.4}
        round_tripped = WorkerReport.from_dict(payload)
        assert round_tripped.openings_cut == 4
        assert round_tripped.shell_reference == {"mode": "referenced"}

    def test_the_declaration_survives_a_round_trip(self):
        """It did not, for an hour: `from_dict` never read the field it was serialised into.

        The worker computes each view's gap from the *parsed* request, so a dropped declaration
        silently disabled the rule -- coverage read `min=0` for every camera while the request
        said 1, and the only symptom was a gap that never appeared.
        """
        original = camera(minimum_visible_instances=3)
        restored = CameraRequest.from_dict(original.to_dict())
        assert restored.minimum_visible_instances == 3
