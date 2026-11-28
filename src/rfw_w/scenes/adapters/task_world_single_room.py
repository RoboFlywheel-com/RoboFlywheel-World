# SPDX-License-Identifier: Apache-2.0
"""Read-only link from a legacy single-room build to a task-world room ID.

This link records source identities and a declared frame. It does not merge the
SceneIR into the task graph or claim that any cross-room geometry was built.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from rfw_w.scenes.contracts.room_plan import RoomPlan
from rfw_w.scenes.contracts.state import SceneIR
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph


@dataclass(frozen=True, slots=True)
class SingleRoomLink:
    graph_hash: str
    graph_room_id: str
    plan_id: str
    plan_room_id: str
    plan_hash: str
    scene_id: str
    scene_hash: str
    room_to_world_xy_m: tuple[float, float]
    room_to_world_yaw_rad: float

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "reference_only",
            "graph_hash": self.graph_hash,
            "graph_room_id": self.graph_room_id,
            "plan_id": self.plan_id,
            "plan_room_id": self.plan_room_id,
            "plan_hash": self.plan_hash,
            "scene_id": self.scene_id,
            "scene_hash": self.scene_hash,
            "room_to_world": {
                "translation_xy_m": list(self.room_to_world_xy_m),
                "yaw_rad": self.room_to_world_yaw_rad,
            },
        }


def link_single_room(
    graph: TaskWorldGraph,
    graph_room_id: str,
    plan: RoomPlan,
    scene: SceneIR,
    *,
    room_to_world_xy_m: tuple[float, float],
    room_to_world_yaw_rad: float,
) -> SingleRoomLink:
    """Bind explicit IDs and immutable source hashes without changing either source."""
    rooms = [node for node in graph.to_dict()["nodes"] if node["kind"] == "room"]
    graph_room = next((node for node in rooms if node["id"] == graph_room_id), None)
    if graph_room is None:
        raise GraphError("GRAPH_DANGLING_REF", graph_room_id)
    if len(plan.rooms) != 1:
        raise GraphError("SINGLE_ROOM_PLAN_REQUIRED", plan.plan_id)
    plan_room = plan.rooms[0]
    if plan_room.kind != graph_room["room_kind"]:
        raise GraphError("SINGLE_ROOM_KIND_MISMATCH", graph_room_id)
    if (len(room_to_world_xy_m) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in room_to_world_xy_m)
            or type(room_to_world_yaw_rad) not in (int, float)
            or not math.isfinite(room_to_world_yaw_rad)):
        raise GraphError("GRAPH_FRAME_MISMATCH", graph_room_id)
    return SingleRoomLink(
        graph_hash=graph.content_hash,
        graph_room_id=graph_room_id,
        plan_id=plan.plan_id,
        plan_room_id=plan_room.room_id,
        plan_hash=plan.content_hash,
        scene_id=scene.scene_id,
        scene_hash=scene.content_hash(),
        room_to_world_xy_m=tuple(float(v) for v in room_to_world_xy_m),
        room_to_world_yaw_rad=float(room_to_world_yaw_rad),
    )


def check_single_room_link(
    link: SingleRoomLink, graph: TaskWorldGraph, plan: RoomPlan, scene: SceneIR,
) -> None:
    """Reject a link whose graph or single-room source has since changed."""
    if graph.content_hash != link.graph_hash:
        raise GraphError("GRAPH_VERSION_CONFLICT", link.graph_room_id)
    if plan.plan_id != link.plan_id or plan.content_hash != link.plan_hash:
        raise GraphError("SINGLE_ROOM_SOURCE_STALE", link.plan_id)
    if scene.scene_id != link.scene_id or scene.content_hash() != link.scene_hash:
        raise GraphError("SINGLE_ROOM_SOURCE_STALE", link.scene_id)
