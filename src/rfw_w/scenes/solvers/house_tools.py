# SPDX-License-Identifier: Apache-2.0
"""A measured, shared-wall two-room shell for the first task-world slice.

This module does not infer robot navigation from a doorway. Its corridor is a geometric
reservation in an empty built shell; furniture and robot motion have separate gates.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph

TOOL_VERSION = "house_shell/v1"
ROOM_SIDE_M = 4.0
WALL_THICKNESS_M = 0.1
WALL_HEIGHT_M = 2.8
DOOR_HEIGHT_M = 2.1


class HouseToolError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}({detail})")


def solve_house_layout(graph: TaskWorldGraph) -> dict[str, Any]:
    """Place the two ordered task rooms on either side of one common wall."""
    data = graph.to_dict()
    rooms = [n for n in data["nodes"] if n["kind"] == "room"]
    portals = [n for n in data["nodes"] if n["kind"] == "portal"]
    if (data.get("composition_schema") not in {"pvscene.room-composition/v1", "pvscene.room-composition/v2"}
            or len(rooms) != 2 or len(portals) != 1
            or set(portals[0]["room_ids"]) != {n["id"] for n in rooms}
            or portals[0]["initial_state"] != "open"):
        raise HouseToolError("HOUSE_TOPOLOGY_UNSUPPORTED", "two rooms, one open shared portal required")
    requested = portals[0]["requested_clearance_m"]
    if not isinstance(requested, (int, float)) or not math.isfinite(requested):
        raise HouseToolError("PORTAL_GEOMETRY_INVALID", portals[0]["id"])
    door_width = round(max(1.0, requested + 0.1), 6)
    if door_width > ROOM_SIDE_M - 1.0:
        raise HouseToolError("PORTAL_GEOMETRY_INVALID", portals[0]["id"])
    low = round((ROOM_SIDE_M - door_width) / 2, 6)
    high = round(low + door_width, 6)
    plan = {
        "schema_version": "pvscene.house-layout/v1",
        "source_graph_hash": graph.content_hash,
        "rooms": [
            {"id": rooms[0]["id"], "kind": rooms[0]["room_kind"],
             "bounds_xy_m": [-ROOM_SIDE_M, 0.0, 0.0, ROOM_SIDE_M]},
            {"id": rooms[1]["id"], "kind": rooms[1]["room_kind"],
             "bounds_xy_m": [0.0, 0.0, ROOM_SIDE_M, ROOM_SIDE_M]},
        ],
        "portal": {"id": portals[0]["id"], "room_ids": portals[0]["room_ids"],
                   "wall_x_m": 0.0, "span_y_m": [low, high],
                   "requested_clearance_m": requested, "height_m": DOOR_HEIGHT_M},
        "corridor": {"x_span_m": [-ROOM_SIDE_M + 0.5, ROOM_SIDE_M - 0.5],
                     "y_span_m": [low, high]},
        "wall_thickness_m": WALL_THICKNESS_M,
        "wall_height_m": WALL_HEIGHT_M,
    }
    plan["content_hash"] = content_hash(plan)
    return plan


def _box(lines: list[str], name: str, bounds: tuple[float, float, float, float, float, float],
         vertex_start: int) -> int:
    x0, y0, z0, x1, y1, z1 = bounds
    lines.append(f"g {name}")
    for x, y, z in ((x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                    (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)):
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    faces = ((1,2,3,4),(5,8,7,6),(1,5,6,2),(2,6,7,3),(3,7,8,4),(4,8,5,1))
    for face in faces:
        lines.append("f " + " ".join(str(vertex_start + i - 1) for i in face))
    return vertex_start + 8


def build_house_shell(plan: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Author one OBJ. The common wall has two segments around the door, never two copies."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise HouseToolError("HOUSE_SHELL_MISMATCH", "output directory already contains evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# pvscene shared-wall house shell v1"]
    vertex = 1
    boxes = _expected_boxes(plan)
    for name, bounds in boxes:
        vertex = _box(lines, name, bounds, vertex)
    mesh_bytes = ("\n".join(lines) + "\n").encode("ascii")
    (output_dir / "house.obj").write_bytes(mesh_bytes)
    (output_dir / "house_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return {"mesh_sha256": hashlib.sha256(mesh_bytes).hexdigest(),
            "plan_hash": plan["content_hash"], "groups": len(boxes)}


def _expected_boxes(plan: Mapping[str, Any]) -> list[tuple[str, tuple[float, float, float, float, float, float]]]:
    half = WALL_THICKNESS_M / 2
    h = WALL_HEIGHT_M
    low, high = plan["portal"]["span_y_m"]
    return [
        ("floor_left", (-4,0,-0.05,0,4,0)),
        ("floor_right", (0,0,-0.05,4,4,0)),
        ("wall_left", (-4-half,0,0,-4+half,4,h)),
        ("wall_right", (4-half,0,0,4+half,4,h)),
        ("wall_bottom_left", (-4,-half,0,0,half,h)),
        ("wall_bottom_right", (0,-half,0,4,half,h)),
        ("wall_top_left", (-4,4-half,0,0,4+half,h)),
        ("wall_top_right", (0,4-half,0,4,4+half,h)),
        ("wall_shared_low", (-half,0,0,half,low,h)),
        ("wall_shared_high", (-half,high,0,half,4,h)),
        ("wall_shared_lintel", (-half,low,DOOR_HEIGHT_M,half,high,h)),
    ]


def _read_obj_groups(path: Path) -> dict[str, tuple[float, float, float, float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    groups: dict[str, list[int]] = {}
    face_counts: dict[str, int] = {}
    current: str | None = None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("g "):
            current = line[2:]
            if current in groups:
                raise HouseToolError("HOUSE_SHELL_MISMATCH", "duplicate group " + current)
            groups[current] = []
            face_counts[current] = 0
        elif line.startswith("v "):
            vertices.append(tuple(float(s) for s in line.split()[1:4]))
        elif line.startswith("f "):
            if current is None:
                raise HouseToolError("HOUSE_SHELL_MISMATCH", "face without group")
            indices = [int(s.split("/")[0]) - 1 for s in line.split()[1:]]
            if len(indices) != 4:
                raise HouseToolError("HOUSE_SHELL_MISMATCH", "non-quad face " + current)
            groups[current].extend(indices)
            face_counts[current] += 1
    bounds = {}
    for name, indices in groups.items():
        if (face_counts[name] != 6 or len(set(indices)) != 8
                or any(i < 0 or i >= len(vertices) for i in indices)):
            raise HouseToolError("HOUSE_SHELL_MISMATCH", "invalid faces " + name)
        points = [vertices[i] for i in indices]
        bounds[name] = (min(p[0] for p in points), min(p[1] for p in points),
                        min(p[2] for p in points), max(p[0] for p in points),
                        max(p[1] for p in points), max(p[2] for p in points))
    return bounds


def measure_house_access(plan: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Read built mesh, measure actual wall gap at floor height, and reject stale/tampered plans."""
    try:
        stored = json.loads((output_dir / "house_plan.json").read_text())
        mesh = (output_dir / "house.obj").read_bytes()
    except (OSError, ValueError) as error:
        raise HouseToolError("HOUSE_SHELL_MISMATCH", "missing or unreadable build") from error
    if stored != plan or content_hash({k: v for k, v in stored.items() if k != "content_hash"}) != stored.get("content_hash"):
        raise HouseToolError("HOUSE_SHELL_MISMATCH", "plan changed")
    bounds = _read_obj_groups(output_dir / "house.obj")
    expected = dict(_expected_boxes(plan))
    if set(bounds) != set(expected):
        raise HouseToolError("HOUSE_SHELL_MISMATCH", "mesh groups")
    if any(any(abs(a-b) > 1e-6 for a,b in zip(bounds[name], wanted))
           for name, wanted in expected.items()):
        raise HouseToolError("HOUSE_SHELL_MISMATCH", "mesh bounds")
    lower = bounds["wall_shared_low"]
    upper = bounds["wall_shared_high"]
    lintel = bounds["wall_shared_lintel"]
    width = round(upper[1] - lower[4], 6)
    if (width <= 0 or lower[2] != 0 or upper[2] != 0
            or lintel[2] < DOOR_HEIGHT_M - 1e-6
            or abs(lintel[1] - lower[4]) > 1e-6
            or abs(lintel[4] - upper[1]) > 1e-6
            or abs(lower[4] - plan["portal"]["span_y_m"][0]) > 1e-6
            or abs(upper[1] - plan["portal"]["span_y_m"][1]) > 1e-6):
        raise HouseToolError("HOUSE_SHELL_MISMATCH", "shared wall or opening")
    if width < plan["portal"]["requested_clearance_m"]:
        raise HouseToolError("PORTAL_CLEARANCE_FAILED", plan["portal"]["id"])
    result = {"schema_version": "pvscene.house-measurement/v1",
              "source_graph_hash": plan["source_graph_hash"],
              "plan_hash": plan["content_hash"],
              "mesh_sha256": hashlib.sha256(mesh).hexdigest(),
              "room_count": 2, "portal_count": 1,
              "portal_id": plan["portal"]["id"],
              "measured_clearance_m": width,
              "corridor_width_m": width,
              "wall_groups": 9, "floor_groups": 2,
              "robot_navigation": "not_run"}
    return result
