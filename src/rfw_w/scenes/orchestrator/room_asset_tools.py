# SPDX-License-Identifier: Apache-2.0
"""G3 transaction around real HSSD retrieval, preparation, placement and static checks."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from rfw_w.scenes.assets.room_tools import (
    RoomAssetError, asset_config, build_asset_ledger, verify_scene_assembly,
)
from rfw_w.scenes.agents.room_asset_visualization import render_room_assets_svg
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.contracts.tool import FailureCode, ToolRequest, ToolResult, ToolStatus
from rfw_w.scenes.solvers.house_tools import measure_house_access


def _event(graph: TaskWorldGraph, name: str, scope: tuple[str, ...], elapsed_ms: float,
           *, writer: bool = False, delta: dict[str, Any] | None = None,
           reason_code: str | None = None) -> dict[str, Any]:
    request = ToolRequest(request_id=name + "_" + graph.content_hash[:20],
                          tool_version=name + "/v1", scene_version=graph.content_hash,
                          seed=7, deadline_s=600.0, scope_ids=scope,
                          input_refs=(graph.content_hash,),
                          idempotency_key=name + ":" + graph.content_hash)
    failure = {
        "ASSET_ENV_MISSING": FailureCode.ASSET_IO_FAILURE,
        "ASSET_UNAVAILABLE": FailureCode.ASSET_UNAVAILABLE,
        "ASSET_UNCERTIFIED": FailureCode.ASSET_UNCERTIFIED,
        "SUPPORT_NOT_FOUND": FailureCode.SUPPORT_NOT_FOUND,
        "NO_PLACEMENT_CAPACITY": FailureCode.NO_PLACEMENT_CAPACITY,
        "CLEARANCE_VIOLATION": FailureCode.CLEARANCE_VIOLATION,
        "GEOMETRY_OUT_OF_BOUNDS": FailureCode.GEOMETRY_OUT_OF_BOUNDS,
        "GEOMETRY_PENETRATION": FailureCode.GEOMETRY_PENETRATION,
        "GRAPH_VERSION_CONFLICT": FailureCode.PRECONDITION_UNMET,
    }.get(reason_code, FailureCode.SCHEMA_INVALID) if reason_code else None
    result = ToolResult(request_id=request.request_id,
                        status=ToolStatus.FAILED if reason_code else ToolStatus.PASSED,
                        tool_version=request.tool_version, base_scene_version=graph.content_hash,
                        read_only=not writer or reason_code is not None, elapsed_ms=elapsed_ms,
                        touched_ids=scope if writer and reason_code is None else (),
                        proposed_delta=delta if reason_code is None else None,
                        failure_code=failure,
                        metrics={"reason_code": reason_code} if reason_code else {})
    return {"request": request.to_dict(), "result": result.to_dict()}


def run_room_asset_tools(graph: TaskWorldGraph, house_dir: Path, output_dir: Path,
                         *, expected_hash: str | None = None) -> tuple[TaskWorldGraph, dict[str, Any]]:
    if expected_hash is not None and expected_hash != graph.content_hash:
        raise GraphError("GRAPH_VERSION_CONFLICT", "asset tool input")
    if output_dir.exists():
        raise GraphError("ASSET_OUTPUT_EXISTS", output_dir.name)
    source = TaskWorldGraph.from_dict(json.loads((house_dir / "graph.json").read_text()))
    if source.content_hash != graph.content_hash:
        raise GraphError("GRAPH_VERSION_CONFLICT", "house source")
    if "house_shell" not in graph.to_dict()["built"]:
        raise GraphError("PRECONDITION_UNMET", "house shell")
    plan = json.loads((house_dir / "geometry/house_plan.json").read_text())
    source_measure = measure_house_access(plan, house_dir / "geometry")
    if source_measure["mesh_sha256"] != graph.to_dict()["built"]["house_shell"]["mesh_sha256"]:
        raise GraphError("HOUSE_SHELL_MISMATCH", "source geometry")
    output_dir.mkdir(parents=True)
    shutil.copytree(house_dir / "geometry", output_dir / "geometry")
    role_ids = tuple(n["id"] for n in graph.to_dict()["nodes"] if n["kind"] == "asset_role")
    events = []
    def on_stage(name: str, elapsed_ms: float, reason_code: str | None) -> None:
        events.append(_event(graph, name, role_ids, elapsed_ms, reason_code=reason_code))
        if reason_code is not None:
            (output_dir / "failure.json").write_text(json.dumps(
                {"status": "failed", "reason_code": reason_code, "events": events},
                indent=2, sort_keys=True) + "\n")
    try:
        config = asset_config(output_dir / "prepared")
    except RoomAssetError as error:
        on_stage("retrieve_role_candidates", 0.0, error.code)
        raise
    ledger = build_asset_ledger(graph, plan, config, output_dir, on_stage=on_stage)
    (output_dir / "asset_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    saved = json.loads((output_dir / "asset_ledger.json").read_text())
    started = time.monotonic()
    try:
        verified = verify_scene_assembly(graph, plan, saved, output_dir)
    except RoomAssetError as error:
        on_stage("verify_scene_assembly", (time.monotonic()-started)*1000, error.code)
        raise
    verify_elapsed = (time.monotonic()-started)*1000
    if measure_house_access(plan, output_dir / "geometry")["mesh_sha256"] != source_measure["mesh_sha256"]:
        raise GraphError("HOUSE_SHELL_MISMATCH", "copied geometry")
    raw = graph.to_dict()
    for item in ledger["instances"]:
        rid, iid = item["role_id"], item["instance_id"]
        role = next(n for n in raw["nodes"] if n["id"] == rid)
        parent = next(e["source_id"] for e in raw["edges"] if e["kind"] == "contains" and e["target_id"] == rid)
        fields = ({"purpose": role["purpose"], "placement_anchor": {"kind": role["placement_kind"], "ref": parent}}
                  if "purpose" in role else {})
        raw["nodes"].append({"id": iid, "kind": "asset_instance", "label": item["category"] + " instance",
                             "source": "inferred", "source_reason": "selected from local HSSD and statically placed",
                             "room_id": item["room_id"], "asset_ref": "hssd:" + item["asset_id"], **fields})
        raw["edges"].append({"id": "e_" + rid + "_" + iid, "kind": "contains",
                             "source_id": rid, "target_id": iid})
        raw["edges"].append({"id": "e_bind_" + rid + "_" + iid, "kind": "role_bound_to",
                             "source_id": rid, "target_id": iid})
    raw["built"]["asset_assembly"] = {
        "status": "built", "ledger_hash": ledger["content_hash"],
        "ledger_ref": "asset_ledger.json", "instance_ids": [i["instance_id"] for i in ledger["instances"]],
        "core_roles_bound": verified["core_roles_bound"],
        "optional_placed": verified["optional_placed"]}
    raw["evidence_refs"] = []
    revised = graph.revise(raw, expected_hash=graph.content_hash)
    events.extend([
        _event(graph, "verify_scene_assembly", role_ids, verify_elapsed),
        _event(graph, "commit_asset_assembly", role_ids, 0.0, writer=True,
               delta={"built": raw["built"]["asset_assembly"]}),
    ])
    report = {**verified, "source_graph_hash": graph.content_hash,
              "graph_hash": revised.content_hash, "house_remeasured": True,
              "asset_ledger_hash": ledger["content_hash"], "events": events}
    (output_dir / "graph.json").write_text(json.dumps(revised.to_dict(), indent=2, sort_keys=True) + "\n")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "scene.svg").write_text(render_room_assets_svg(plan, ledger))
    return revised, report
