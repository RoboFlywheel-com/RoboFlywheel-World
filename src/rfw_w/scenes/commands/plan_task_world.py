#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""First task-world tool: prompt -> validated intent graph -> readable SVG.

Offline --replay-response is for deterministic development. Live model calls require the
repository's explicit PVSCENE_ALLOW_EXTERNAL_VLM=1 gate. Each run gets a fresh directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


from rfw_w.scenes.agents.llm_client import LLMConfig, LLMError
from rfw_w.scenes.agents.task_world_planner import request_hash
from rfw_w.scenes.agents.task_world_visualization import render_task_world_svg
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.task_world_graph import GraphError
from rfw_w.scenes.orchestrator.plan_task_world import run_plan_task_world


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path, help="structured prompt + robot request JSON")
    source.add_argument("--prompt", help="task and scene description")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 task and scene description")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay-response", type=Path, help="recorded model JSON, no external call")
    mode.add_argument("--model", help="configured model for one live call")
    parser.add_argument("--graph-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot-id", default="robot")
    parser.add_argument("--robot-name", help="specific robot model; required without --request")
    parser.add_argument("--robot-mobility", choices=("fixed_arm", "mobile_manipulator"),
                        default="mobile_manipulator")
    parser.add_argument("--required-clearance-m", type=float, default=0.75)
    parser.add_argument("--deadline-s", type=float, default=240.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.request:
        req = json.loads(args.request.read_text(encoding="utf-8"))
    else:
        if not args.robot_name:
            parser.error("--robot-name is required with --prompt/--prompt-file")
        prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
        identifier = args.graph_id or "task_" + content_hash({"prompt": prompt, "seed": args.seed})[:12]
        req = {"graph_id": identifier, "seed": args.seed, "prompt": prompt,
               "robot": {"id": args.robot_id, "name": args.robot_name,
                         "mobility": args.robot_mobility,
                         "required_clearance_m": args.required_clearance_m}}

    if args.output_dir.exists():
        parser.error(f"output directory already exists; choose a fresh directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    record = {"status": "failed", "request_hash": None, "source": "replay" if args.replay_response else "live",
              "graph_hash": None, "robot_validation": "not_run", "built_geometry": "not_run"}
    try:
        record["request_hash"] = request_hash(req)
        config = None
        if args.model:
            config = LLMConfig(
                base_url=(os.environ.get("PVSCENE_VLM_BASE_URL")
                          or os.environ.get("SIMFOUNDRY_OPENAI_BASE_URL", "")),
                model=args.model,
                api_key_env=os.environ.get("PVSCENE_VLM_API_KEY_ENV", "DASHSCOPE_API_KEY"),
                wire=os.environ.get("PVSCENE_VLM_WIRE", "chat_completions"),
                timeout_s=float(os.environ.get("PVSCENE_VLM_TIMEOUT_S", "240")),
                max_tokens=8192,
            )
        result = run_plan_task_world(
            req,
            replay_response=(args.replay_response.read_text(encoding="utf-8")
                             if args.replay_response else None),
            config=config, deadline_s=args.deadline_s,
        )
        (args.output_dir / "tool_request.json").write_text(
            json.dumps(result.tool_request.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        (args.output_dir / "tool_result.json").write_text(
            json.dumps(result.tool_result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        record.update(result.to_dict())
        if result.graph is None:
            record["reason"] = result.reason_code
            (args.output_dir / "planning.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"planning failed: {record['reason']}", file=sys.stderr)
            return 1
        graph = result.graph
        (args.output_dir / "task_world_graph.json").write_text(
            json.dumps(graph.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "task_world_graph.svg").write_text(
            render_task_world_svg(graph), encoding="utf-8")
        record.update(status="planned", graph_hash=graph.content_hash,
                      rooms=sum(n["kind"] == "room" for n in graph.to_dict()["nodes"]),
                      steps=len(graph.to_dict()["steps"]),
                      unmet_obligations=len(graph.to_dict()["requirements"]))
    except (GraphError, LLMError, SystemExit, ValueError) as error:
        if isinstance(error, (GraphError, LLMError)):
            record["reason"] = error.code
        elif isinstance(error, SystemExit):
            record["reason"] = "EXTERNAL_CALL_NOT_ALLOWED"
        else:
            record["reason"] = type(error).__name__
        # Never serialize transport exception text: providers can echo sensitive headers or payloads.
        (args.output_dir / "planning.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"planning failed: {record['reason']}", file=sys.stderr)
        return 1
    (args.output_dir / "planning.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "request_id": record["request_id"],
                      "graph_hash": record["graph_hash"], "rooms": record["rooms"],
                      "steps": record["steps"], "unmet_obligations": record["unmet_obligations"],
                      "model_requests_used": record["budget"]["model_requests_used"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
