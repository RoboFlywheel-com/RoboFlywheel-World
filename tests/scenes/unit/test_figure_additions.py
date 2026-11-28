# SPDX-License-Identifier: Apache-2.0
"""SPEC-016: adding robots and props to a figure request, and refusing to.

Every named rejection gets a negative test, because a placement gate that
cannot fail is the defect: these rules are the only thing between "the arm is
on the desk it works at" and "the arm hovers over the floor, half inside a
bookcase, reaching for a mug two metres away".
"""

from __future__ import annotations

import importlib
from rfw_w.scenes.resources import module_path, data_path
import json
import math
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = module_path('rfw_w.scenes.visualization.render_room_preview')
# The installed visualization module is the production entry under test.


preview = importlib.import_module('rfw_w.scenes.visualization.render_room_preview')


def write_box_obj(path: Path, half_x: float, half_y: float, height: float) -> Path:
    """A unit box resting on z = 0, written as an OBJ the placement code can read."""
    points = [(-half_x, -half_y, 0.0), (half_x, -half_y, 0.0), (half_x, half_y, 0.0),
              (-half_x, half_y, 0.0), (-half_x, -half_y, height), (half_x, -half_y, height),
              (half_x, half_y, height), (-half_x, half_y, height)]
    faces = [(1, 2, 3), (1, 3, 4), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]
    lines = [f"v {x} {y} {z}" for x, y, z in points]
    lines += ["f " + " ".join(str(i) for i in face) for face in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def base_request() -> dict:
    """A 4x4 m room with a desk against the north wall and a mug on it."""
    return {
        "room": {
            "polygon_xy": [[0, 0], [4, 0], [4, 4], [0, 4]],
            "floor_z_m": 0.0,
            "height_m": 2.7,
            "openings": [{"opening_id": "door_main", "kind": "door", "center_xy": [2.0, 0.0],
                          "width_m": 0.9, "height_m": 2.05, "sill_m": 0.0}],
        },
        "instances": [
            {"instance_id": "desk", "entity_id": "desk", "visual_uri": "desk.obj",
             "translation": [2.0, 3.5, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            {"instance_id": "mug", "entity_id": "mug", "visual_uri": "mug.obj",
             "translation": [2.3, 3.4, 0.75], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
        ],
    }


BOUNDS = {
    "desk": ([-0.7, -0.35, 0.0], [0.7, 0.35, 0.75]),
    "mug": ([-0.05, -0.05, 0.0], [0.05, 0.05, 0.09]),
}


def prop(tmp_path: Path, **overrides) -> dict:
    mesh = write_box_obj(tmp_path / f"{overrides.get('id', 'prop')}.obj", 0.05, 0.05, 0.12)
    item = {"id": "prop_box", "kind": "prop", "visual_uri": str(mesh), "on": "desk",
            "xy": [1.7, 3.5], "yaw_deg": 0.0}
    item.update(overrides)
    return item


class TestAdditionsPlacement:
    def test_prop_rests_on_the_support_top(self, tmp_path: Path) -> None:
        request = base_request()
        records = preview.apply_additions(request, [prop(tmp_path)], bounds_by_instance=BOUNDS,
                                          figure_asset_dir=tmp_path / "assets")
        assert len(records) == 1
        added = next(i for i in request["instances"] if i["instance_id"] == "prop_box")
        # The desk's top is at 0.75; the box's own base is at z = 0 in its frame.
        assert added["translation"][2] == pytest.approx(0.75)
        assert records[0]["on"] == "desk"

    def test_declared_height_that_disagrees_is_a_named_gap(self, tmp_path: Path) -> None:
        request = base_request()
        with pytest.raises(ValueError, match="ADDITION_FLOATING"):
            preview.apply_additions(request, [prop(tmp_path, z_m=1.2)],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_unknown_support_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ADDITION_UNKNOWN_SUPPORT"):
            preview.apply_additions(base_request(), [prop(tmp_path, on="sideboard")],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_hanging_off_the_support_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ADDITION_OFF_SUPPORT"):
            preview.apply_additions(base_request(), [prop(tmp_path, xy=[2.75, 3.5])],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_sharing_space_with_an_existing_instance_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ADDITION_OVERLAP"):
            preview.apply_additions(base_request(), [prop(tmp_path, xy=[2.3, 3.4])],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_two_additions_may_not_share_space(self, tmp_path: Path) -> None:
        items = [prop(tmp_path, id="prop_a", xy=[1.7, 3.5]),
                 prop(tmp_path, id="prop_b", xy=[1.72, 3.5])]
        with pytest.raises(ValueError, match="ADDITION_OVERLAP"):
            preview.apply_additions(base_request(), items, bounds_by_instance=BOUNDS,
                                    figure_asset_dir=tmp_path / "a")

    def test_a_floor_prop_may_not_stand_in_the_doorway(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ADDITION_BLOCKS_OPENING"):
            preview.apply_additions(base_request(),
                                    [prop(tmp_path, on="floor", xy=[2.0, 0.1])],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_a_prop_on_a_support_is_not_judged_by_the_opening_rule(self, tmp_path: Path) -> None:
        """The desk stands where the delivery put it; a mug on its top is not a blocked door.

        Measured on the office: the desk is against the window wall, so the
        opening strip refuses every object on its top if the rule is applied
        there.
        """
        request = base_request()
        request["instances"][0]["translation"] = [2.0, 0.4, 0.0]
        records = preview.apply_additions(request, [prop(tmp_path, xy=[2.0, 0.4])],
                                          bounds_by_instance=BOUNDS,
                                          figure_asset_dir=tmp_path / "a")
        assert records[0]["on"] == "desk"

    def test_outside_the_room_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ADDITION_OUTSIDE_ROOM"):
            preview.apply_additions(base_request(), [prop(tmp_path, on="floor", xy=[5.5, 2.0])],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_duplicate_id_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ADDITION_DUPLICATE_ID"):
            preview.apply_additions(base_request(), [prop(tmp_path, id="mug")],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_missing_mesh_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ADDITION_MESH_MISSING"):
            preview.apply_additions(base_request(),
                                    [prop(tmp_path, visual_uri=str(tmp_path / "nope.obj"))],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_unknown_kind_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ADDITION_KIND_UNKNOWN"):
            preview.apply_additions(base_request(), [prop(tmp_path, kind="hologram")],
                                    bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")


class TestRobotAdditions:
    """The robot path, with the bake stubbed: this tests placement, not URDF loading."""

    def robot_item(self, tmp_path: Path, monkeypatch, **overrides) -> dict:
        mesh = write_box_obj(tmp_path / "robot.obj", 0.22, 0.08, 0.61)
        import rfw_w.scenes.visualization.robot_asset as robot_asset
        def fake_bake(robot, pose, out):
            out.mkdir(parents=True, exist_ok=True)
            (out / "robot.obj").write_text(mesh.read_text(), encoding="utf-8")
            return {"source_urdf_sha256": "0" * 64, "visual_obj_sha256": "1" * 64,
                    "tcp_in_baked_frame_m": [0.49, -0.09, 0.26]}

        monkeypatch.setattr(robot_asset, "bake", fake_bake)
        pose = tmp_path / "pose.json"
        pose.write_text(json.dumps({"joint_degrees": {}}), encoding="utf-8")
        item = {"id": "arm", "kind": "robot", "robot": "piper", "pose": str(pose),
                "on": "desk", "xy": [1.6, 3.5], "yaw_deg": 0.0, "task_target": "mug"}
        item.update(overrides)
        return item

    def test_robot_is_placed_and_its_reach_recorded(self, tmp_path, monkeypatch) -> None:
        request = base_request()
        records = preview.apply_additions(
            request, [self.robot_item(tmp_path, monkeypatch)], bounds_by_instance=BOUNDS,
            figure_asset_dir=tmp_path / "assets")
        reach = records[0]["reach"]
        assert reach["profile"] == "piper_x"
        assert reach["horizontal_distance_m"] == pytest.approx(
            math.hypot(2.3 - 1.6, 3.4 - 3.5), abs=1e-4)
        assert reach["reach_m"] > reach["horizontal_distance_m"]
        assert records[0]["reach"]["claim"].startswith("reach proxy")

    def test_a_target_beyond_the_measured_reach_is_refused(self, tmp_path, monkeypatch) -> None:
        request = base_request()
        request["instances"][1]["translation"] = [3.9, 3.9, 0.75]
        # On the floor in the far corner: the arm's base is fine where it stands,
        # and the only thing wrong is the distance to what it is placed for.
        with pytest.raises(ValueError, match="ROBOT_TARGET_OUT_OF_REACH"):
            preview.apply_additions(
                request,
                [self.robot_item(tmp_path, monkeypatch, on="floor", xy=[0.6, 0.8])],
                bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_an_unknown_target_is_refused(self, tmp_path, monkeypatch) -> None:
        with pytest.raises(ValueError, match="ADDITION_UNKNOWN_TARGET"):
            preview.apply_additions(
                base_request(), [self.robot_item(tmp_path, monkeypatch, task_target="teapot")],
                bounds_by_instance=BOUNDS, figure_asset_dir=tmp_path / "a")

    def test_the_robot_is_judged_on_its_support_band(self, tmp_path, monkeypatch) -> None:
        """An arm reaching over the mug is the pose the figure is for.

        The robot mesh here is 0.61 m tall; only its lowest
        ROBOT_SUPPORT_BAND_M counts as what it stands on, and the declared task
        target is exempt from the overlap rule besides.
        """
        request = base_request()
        records = preview.apply_additions(
            request, [self.robot_item(tmp_path, monkeypatch)], bounds_by_instance=BOUNDS,
            figure_asset_dir=tmp_path / "assets")
        assert "support band" in records[0]["footprint_rule"]
        assert "mug" in records[0]["overlap_exempt"]


class TestShellMaterials:
    def test_declared_roles_reach_the_request(self) -> None:
        request = base_request()
        record = preview.apply_shell_materials(
            request, {"wall": {"colour": [0.8, 0.7, 0.6], "roughness": 0.9, "metallic": 0.0}})
        assert request["shell_material_override"] == ["wall"]
        assert request["room_materials"]["wall"]["colour"] == [0.8, 0.7, 0.6]
        assert record["roles"] == ["wall"]

    def test_a_role_with_no_surface_to_bind_from_is_refused(self) -> None:
        with pytest.raises(ValueError, match="SHELL_MATERIAL_ROLE_UNKNOWN"):
            preview.apply_shell_materials(base_request(), {"skirting": {"colour": [0, 0, 0]}})
