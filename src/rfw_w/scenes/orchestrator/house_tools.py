# SPDX-License-Identifier: Apache-2.0
"""G2 tool transaction: stage artifacts, measure them, then commit one graph revision."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.contracts.tool import FailureCode, ToolRequest, ToolResult, ToolStatus
from rfw_w.scenes.solvers.house_tools import (
    TOOL_VERSION, HouseToolError, build_house_shell, measure_house_access, solve_house_layout,
)


def _request(graph: TaskWorldGraph, name: str, scope: tuple[str, ...], seed: int) -> ToolRequest:
    return ToolRequest(request_id=name + "_" + graph.content_hash[:20],
                       tool_version=name + "/v1", scene_version=graph.content_hash,
                       seed=seed, deadline_s=120.0, scope_ids=scope,
                       input_refs=(graph.content_hash,),
                       idempotency_key=name + ":" + graph.content_hash)


def run_house_tools(graph: TaskWorldGraph, output_dir: Path, *, seed: int = 7,
                    expected_hash: str | None = None) -> tuple[TaskWorldGraph, dict[str, Any]]:
    if expected_hash is not None and expected_hash != graph.content_hash:
        raise GraphError("GRAPH_VERSION_CONFLICT", "house tool input")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GraphError("HOUSE_OUTPUT_EXISTS", output_dir.name)
    room_ids = tuple(n["id"] for n in graph.to_dict()["nodes"] if n["kind"] == "room")
    portal_ids = tuple(n["id"] for n in graph.to_dict()["nodes"] if n["kind"] == "portal")
    events: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    def stage(name: str, scope: tuple[str, ...], fn, *, writer: bool = False):
        req = _request(graph, name, scope, seed)
        started = time.monotonic()
        try:
            value = fn()
        except (GraphError, HouseToolError) as error:
            failure_code = {
                "HOUSE_TOPOLOGY_UNSUPPORTED": FailureCode.BACKEND_UNSUPPORTED,
                "PORTAL_GEOMETRY_INVALID": FailureCode.CLEARANCE_VIOLATION,
                "PORTAL_CLEARANCE_FAILED": FailureCode.CLEARANCE_VIOLATION,
                "HOUSE_SHELL_MISMATCH": FailureCode.EXPORT_MISMATCH,
            }.get(error.code, FailureCode.SCHEMA_INVALID)
            failed = ToolResult(request_id=req.request_id, status=ToolStatus.FAILED,
                                tool_version=req.tool_version, base_scene_version=graph.content_hash,
                                read_only=True, elapsed_ms=(time.monotonic()-started)*1000,
                                failure_code=failure_code, metrics={"reason_code": error.code})
            events.append({"request": req.to_dict(), "result": failed.to_dict()})
            (output_dir / "failure.json").write_text(json.dumps(
                {"status": "failed", "reason_code": error.code, "events": events},
                indent=2, sort_keys=True) + "\n")
            raise
        result = ToolResult(request_id=req.request_id, status=ToolStatus.PASSED,
                            tool_version=req.tool_version, base_scene_version=graph.content_hash,
                            read_only=not writer, elapsed_ms=(time.monotonic()-started)*1000,
                            touched_ids=scope if writer else (),
                            proposed_delta={"built": value} if writer else None,
                            metrics={"artifact_status": "produced"})
        events.append({"request": req.to_dict(), "result": result.to_dict()})
        return value

    plan = stage("solve_house_layout", room_ids + portal_ids, lambda: solve_house_layout(graph))
    geometry_dir = output_dir / "geometry"
    build = stage("build_house_shell", room_ids + portal_ids,
                  lambda: build_house_shell(plan, geometry_dir))
    measure = stage("measure_house_access", room_ids + portal_ids,
                    lambda: measure_house_access(plan, geometry_dir))
    (geometry_dir / "house_measurement.json").write_text(
        json.dumps(measure, indent=2, sort_keys=True) + "\n")
    if build["mesh_sha256"] != measure["mesh_sha256"]:
        raise GraphError("HOUSE_SHELL_MISMATCH", "mesh changed after build")
    built = graph.to_dict()["built"]
    built["house_shell"] = {"status": "built", "provider": TOOL_VERSION,
                            "plan_hash": plan["content_hash"],
                            "mesh_sha256": measure["mesh_sha256"],
                            "geometry_ref": "geometry/house.obj",
                            "measured_clearance_m": measure["measured_clearance_m"],
                            "room_ids": list(room_ids), "portal_id": portal_ids[0]}
    stage("commit_house_shell", room_ids + portal_ids, lambda: built, writer=True)
    revised = graph.revise({**graph.to_dict(), "built": built,
                            "evidence_refs": []}, expected_hash=graph.content_hash)
    report = {"status": "passed", "source_graph_hash": graph.content_hash,
              "graph_hash": revised.content_hash, "rooms": len(room_ids), "portals": len(portal_ids),
              "measured_clearance_m": measure["measured_clearance_m"],
              "corridor_width_m": measure["corridor_width_m"],
              "mesh_sha256": measure["mesh_sha256"], "robot_navigation": "not_run",
              "events": events}
    (output_dir / "graph.json").write_text(json.dumps(revised.to_dict(), indent=2, sort_keys=True) + "\n")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return revised, report
