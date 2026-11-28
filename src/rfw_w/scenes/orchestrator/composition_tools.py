# SPDX-License-Identifier: Apache-2.0
"""Persist successful and rejected semantic graph edits as typed transactions."""
from __future__ import annotations

import json
from pathlib import Path
import time

from rfw_w.scenes.agents.composition_revision import validate_composition_patch
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.contracts.tool import FailureCode, ToolRequest, ToolResult, ToolStatus


def commit_composition_patch(graph: TaskWorldGraph, patch: dict, output_dir: Path, *,
                             expected_hash: str, scope_ids: tuple[str, ...]) -> TaskWorldGraph:
    output_dir.mkdir(parents=True, exist_ok=False)
    request = ToolRequest(request_id="composition_" + graph.content_hash[:20],
                          tool_version="commit_composition_patch/v1", scene_version=expected_hash,
                          seed=7, deadline_s=30, scope_ids=scope_ids)
    started = time.monotonic()
    candidate = None
    error = None
    try:
        candidate = validate_composition_patch(graph, patch, expected_hash=expected_hash, scope_ids=scope_ids)
    except GraphError as caught:
        error = caught
    result = ToolResult(request_id=request.request_id, tool_version=request.tool_version,
                        base_scene_version=expected_hash, read_only=error is not None,
                        status=ToolStatus.FAILED if error else ToolStatus.PASSED,
                        elapsed_ms=(time.monotonic()-started)*1000,
                        touched_ids=scope_ids if not error else (),
                        proposed_delta={"graph_hash": candidate.content_hash} if candidate else None,
                        failure_code=FailureCode.SCHEMA_INVALID if error else None,
                        metrics={"reason_code": error.code} if error else {})
    (output_dir / "transaction.json").write_text(json.dumps({"request": request.to_dict(),
        "patch": patch, "result": result.to_dict()}, indent=2, sort_keys=True)+"\n")
    (output_dir / "source_graph.json").write_text(json.dumps(graph.to_dict(), indent=2)+"\n")
    if error:
        raise error
    (output_dir / "graph.json").write_text(json.dumps(candidate.to_dict(), indent=2)+"\n")
    return candidate
