# SPDX-License-Identifier: Apache-2.0
"""SPEC-016: URDF forward kinematics and the baked figure robot.

The URDF here is written by the test, so nothing depends on a robot checkout
being present. What it pins is the behaviour that bit in practice: a tree whose
root is not called ``world``, joints listed out of order, a link with two
visuals that carry different materials, and the per-visual origin.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


kin = importlib.import_module('rfw_w.scenes.visualization.urdf_kinematics')


URDF = """<?xml version="1.0"?>
<robot name="stub">
  <material name="dark"><color rgba="0.2 0.2 0.2 1"/></material>
  <link name="base">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><mesh filename="base.stl"/></geometry>
      <material name="dark"/>
    </visual>
  </link>
  <link name="forearm">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="forearm.stl"/></geometry>
      <material name="paint"><color rgba="0.9 0.1 0.1 1"/></material>
    </visual>
    <visual>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
  </link>
  <link name="upper"/>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="forearm"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14"/>
  </joint>
</robot>
"""


@pytest.fixture()
def model(tmp_path: Path) -> dict:
    path = tmp_path / "stub.urdf"
    path.write_text(URDF, encoding="utf-8")
    return kin.parse_urdf(path)


class TestParsing:
    def test_visual_materials_belong_to_the_visual(self, model: dict) -> None:
        forearm = model["links"]["forearm"]
        assert forearm[0]["material_rgba"] == [0.9, 0.1, 0.1, 1.0]
        assert forearm[0]["origin_xyz"] == [0.0, 0.0, 0.0]
        assert model["links"]["base"][0]["material_name"] == "dark"
        assert model["materials"]["dark"][:3] == [0.2, 0.2, 0.2]

    def test_a_non_mesh_visual_is_recorded_not_dropped(self, model: dict) -> None:
        """A primitive visual is named as skipped, so a baked mesh never loses a
        part silently."""
        assert model["links"]["forearm"][1]["skipped"] == "non_mesh_geometry"

    def test_the_root_is_the_link_that_is_never_a_child(self, model: dict) -> None:
        assert kin.root_links(model) == ["base"]


class TestForwardKinematics:
    def test_joints_out_of_order_still_resolve(self, model: dict) -> None:
        """`elbow` is listed before its parent chain exists; the walk repeats
        until no new frame appears rather than assuming topological order."""
        frames = kin.fk(model, {})
        assert frames["forearm"][:3, 3] == pytest.approx([0.3, 0.0, 0.2])

    def test_a_revolute_joint_turns_its_subtree(self, model: dict) -> None:
        frames = kin.fk(model, {"shoulder": math.pi / 2})
        assert frames["forearm"][:3, 3] == pytest.approx([0.0, 0.3, 0.2], abs=1e-9)

    def test_the_same_vector_gives_the_same_frames(self, model: dict) -> None:
        first = kin.fk(model, {"shoulder": 0.3, "elbow": -0.2})
        second = kin.fk(model, {"shoulder": 0.3, "elbow": -0.2})
        for link, matrix in first.items():
            assert np.allclose(matrix, second[link])

    def test_a_link_hanging_off_nothing_is_a_named_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.urdf"
        path.write_text(URDF.replace('<parent link="base"/>', '<parent link="ghost"/>'),
                        encoding="utf-8")
        with pytest.raises(kin.UrdfError, match="URDF_DISCONNECTED"):
            kin.fk(kin.parse_urdf(path), {})


class TestRobotRegistry:
    def test_both_robots_declare_where_their_source_lives_and_front_faces(self) -> None:
        import rfw_w.scenes.visualization.robot_asset as robot_asset
        for name, spec in robot_asset.ROBOTS.items():
            assert spec["root_env"].startswith("PVSCENE_"), name
            # The declaration's yaw is a bearing; both robots front on +x, read
            # off their kinematics rather than off a render.
            assert spec["facing_offset_deg"] == 0.0, name
            assert spec["mount"] in {"floor", "surface"}, name

    def test_the_measured_profiles_are_the_ones_the_placement_rule_uses(self) -> None:
        from rfw_w.scenes.geometry.robot import profile_for

        import rfw_w.scenes.visualization.robot_asset as robot_asset
        from rfw_w.scenes.visualization.render_room_preview import ROBOT_PROFILE_OF

        for robot, profile_name in ROBOT_PROFILE_OF.items():
            profile = profile_for(profile_name)
            assert profile.reach_m > 0 and profile.footprint_m > 0, robot
