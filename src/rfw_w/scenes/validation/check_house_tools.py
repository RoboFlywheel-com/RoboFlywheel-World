#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run G2 on the checked semantic fixture and independently verify the saved OBJ."""
from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import argparse
import json
from pathlib import Path
import sys


from rfw_w.scenes.agents.task_world_assembly import compile_pick_place_slots
from rfw_w.scenes.contracts.task_world_graph import GraphError, TaskWorldGraph
from rfw_w.scenes.orchestrator.house_tools import run_house_tools
from rfw_w.scenes.solvers.house_tools import HouseToolError, measure_house_access, solve_house_layout


def fixture_graph() -> TaskWorldGraph:
    request = json.loads((data_path('fixtures/task_world/cross_room_request.json')).read_text())
    response = json.loads((data_path('fixtures/task_world/cross_room_slots_response.json')).read_text())
    return compile_pick_place_slots(response["slots"], request)


def svg(plan: dict) -> str:
    p = plan["portal"]
    low, high = p["span_y_m"]
    a, b = plan["rooms"]
    # World x in [-4,4] and y in [0,4], with y mirrored into SVG coordinates.
    def sx(x): return 440 + 90*x
    def sy(y): return 430 - 90*y
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="520" viewBox="0 0 880 520">'
            '<rect width="880" height="520" fill="#f5f7fa"/>'
            f'<rect x="{sx(-4)}" y="{sy(4)}" width="360" height="360" fill="#e0f3ed" stroke="#263238" stroke-width="4"/>'
            f'<rect x="{sx(0)}" y="{sy(4)}" width="360" height="360" fill="#e8ecff" stroke="#263238" stroke-width="4"/>'
            f'<line x1="440" y1="{sy(4)}" x2="440" y2="{sy(high)}" stroke="#263238" stroke-width="8"/>'
            f'<line x1="440" y1="{sy(low)}" x2="440" y2="{sy(0)}" stroke="#263238" stroke-width="8"/>'
            f'<text x="{sx(-2)}" y="{sy(3.3)}" text-anchor="middle" font-size="24">{a["id"]}</text>'
            f'<text x="{sx(2)}" y="{sy(3.3)}" text-anchor="middle" font-size="24">{b["id"]}</text>'
            f'<text x="440" y="{sy(high)-12}" text-anchor="middle" font-size="16">{p["id"]}: {high-low:.2f} m</text>'
            '<text x="440" y="490" text-anchor="middle" font-size="17">Built shell plan; robot navigation not run</text>'
            '</svg>')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("HOUSE_OUTPUT_EXISTS(output directory already exists)")
    graph = fixture_graph()
    revised, report = run_house_tools(graph, args.output_dir)
    saved = TaskWorldGraph.from_dict(json.loads((args.output_dir / "graph.json").read_text()))
    plan = json.loads((args.output_dir / "geometry/house_plan.json").read_text())
    measured = measure_house_access(plan, args.output_dir / "geometry")
    if saved.content_hash != revised.content_hash or measured["mesh_sha256"] != report["mesh_sha256"]:
        raise SystemExit("HOUSE_SHELL_MISMATCH(independent replay)")
    negatives: dict[str, str] = {}
    try:
        run_house_tools(graph, args.output_dir / "negative_stale", expected_hash="0"*64)
    except GraphError as error:
        negatives["stale_version"] = error.code
    bad = graph.to_dict()
    next(n for n in bad["nodes"] if n["kind"] == "portal")["requested_clearance_m"] = 10.0
    bad.pop("content_hash", None)
    try:
        solve_house_layout(TaskWorldGraph.from_dict(bad))
    except HouseToolError as error:
        negatives["bad_portal"] = error.code
    # A copy of the saved OBJ with a moved wall is checked in a fresh negative directory.
    tamper = args.output_dir / "negative_mesh"
    tamper.mkdir()
    (tamper / "house_plan.json").write_bytes((args.output_dir / "geometry/house_plan.json").read_bytes())
    mesh = (args.output_dir / "geometry/house.obj").read_text()
    (tamper / "house.obj").write_text(mesh.replace("1.500000", "1.400000"))
    try:
        measure_house_access(plan, tamper)
    except HouseToolError as error:
        negatives["tampered_mesh"] = error.code
    expected = {"stale_version": "GRAPH_VERSION_CONFLICT", "bad_portal": "PORTAL_GEOMETRY_INVALID",
                "tampered_mesh": "HOUSE_SHELL_MISMATCH"}
    if negatives != expected:
        raise SystemExit("HOUSE_NEGATIVE_FAILED(" + repr(negatives) + ")")
    (args.output_dir / "house.svg").write_text(svg(plan))
    report["independent_replay"] = True
    report["negative_results"] = negatives
    (args.output_dir / "acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "rooms", "portals", "measured_clearance_m",
                                                 "independent_replay", "negative_results")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
