#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a checked prompt-to-task-world overview as a standalone SVG."""
from __future__ import annotations

import argparse
from html import escape
import json
import sys
from pathlib import Path
from textwrap import wrap


from rfw_w.scenes.agents.task_world_planner import request_hash, validate_plan_graph
from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph


def text(x: int, y: int, value: str, *, size: int = 18, weight: int = 400,
         color: str = "#17283B") -> str:
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}">{escape(value)}</text>')


def lines(x: int, y: int, value: str, *, width: int = 25, leading: int = 29,
          size: int = 18) -> list[str]:
    return [text(x, y + i * leading, line, size=size)
            for i, line in enumerate(wrap(value, width=width, break_long_words=True))]


def card(x: int, y: int, width: int, height: int, title: str,
         color: str) -> list[str]:
    return [f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" '
            f'fill="#FFFFFF" stroke="{color}" stroke-width="2"/>',
            f'<rect x="{x}" y="{y}" width="{width}" height="48" rx="18" '
            f'fill="{color}"/>',
            text(x + 20, y + 32, title, size=21, weight=700, color="#FFFFFF")]


def build(request: dict, graph: TaskWorldGraph) -> str:
    data = graph.to_dict()
    nodes = {n["id"]: n for n in data["nodes"]}
    rooms = [n for n in data["nodes"] if n["kind"] == "room"]
    portals = [n for n in data["nodes"] if n["kind"] == "portal"]
    initial = data["initial_state"]
    goals = data["goals"]
    start_room_id = next(p["object"] for p in initial
                         if p["subject"] == request["robot"]["id"] and p["predicate"] == "IN_ROOM")
    goal = next(p for p in goals if p["predicate"] in {"ON", "INSIDE"})
    target = nodes[goal["subject"]]
    destination_support = nodes[goal["object"]]
    source_state = next(p for p in initial if p["subject"] == target["id"]
                        and p["predicate"] in {"ON", "INSIDE"})
    source_support = nodes[source_state["object"]]
    destination_room_id = destination_support["room_id"]
    start_room = nodes[start_room_id]
    destination_room = nodes[destination_room_id]
    portal = next(p for p in portals if set(p["room_ids"]) == {start_room_id, destination_room_id})

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="960" '
        'viewBox="0 0 1440 960" role="img" aria-label="Prompt to task world graph overview">',
        '<style>text{font-family:"Noto Sans CJK SC","Noto Sans",Arial,sans-serif}</style>',
        '<rect width="1440" height="960" fill="#F3F6FA"/>',
        text(40, 55, "从 Prompt 到跨房任务图", size=32, weight=700),
        text(41, 84, "真实模型语义提案 → 确定性图构造 → 图契约校验  ·  仅表示任务意图", size=16,
             color="#52677D"),
    ]
    parts += card(40, 115, 415, 252, "1  输入任务", "#345E92")
    parts += lines(61, 195, request["prompt"], width=23, leading=31, size=19)
    parts += [text(61, 309, "机器人：" + request["robot"]["name"], size=16, color="#52677D"),
              text(61, 339, "输入只含任务语义与必要能力", size=15, color="#52677D")]
    parts += card(512, 115, 415, 252, "2  模型抽取语义槽位", "#6B55A1")
    parts += [text(532, 191, f'来源房间  {start_room["label"]}', size=18),
              text(532, 225, f'目标房间  {destination_room["label"]}', size=18),
              text(532, 259, f'共享门户  {portal["label"]}', size=18),
              text(532, 293, f'目标物      {target["label"]}', size=18),
              text(532, 328, f'支撑面      {source_support["label"]} → {destination_support["label"]}',
                   size=17)]
    parts += card(984, 115, 416, 252, "3  校验后的图表征", "#1C806F")
    parts += [text(1005, 193, f'{len(rooms)} 个房间 · {len(portals)} 扇门 · {len(data["steps"])} 个动作',
                   size=19, weight=600),
              text(1005, 229, f'{len(data["nodes"])} 个节点 · {len(data["edges"])} 条关系边', size=18),
              text(1005, 265, f'{len(data["requirements"])} 项待验证义务', size=18),
              text(1005, 302, 'built = {}   ·   robot = not_run', size=16,
                   color="#A44D32"),
              text(1005, 339, 'graph hash  ' + graph.content_hash[:20] + '…', size=14,
                   color="#52677D")]
    parts += [text(469, 250, "→", size=36, weight=700, color="#708396"),
              text(941, 250, "→", size=36, weight=700, color="#708396")]

    parts += [text(40, 420, "空间拓扑 · 同一图的房间视图", size=23, weight=700),
              '<rect x="40" y="445" width="570" height="220" rx="18" '
              'fill="#EAF2FF" stroke="#9EBBE4" stroke-width="2"/>',
              '<rect x="830" y="445" width="570" height="220" rx="18" '
              'fill="#EAF7F3" stroke="#8AC6B8" stroke-width="2"/>',
              text(62, 483, start_room["label"] + '  [' + start_room_id + ']', size=24, weight=700),
              text(852, 483, destination_room["label"] + '  [' + destination_room_id + ']',
                   size=24, weight=700),
              text(62, 528, '机器人起点  ' + request["robot"]["id"], size=18),
              text(62, 566, '待取物  ' + target["label"], size=18),
              text(62, 604, '初态  ' + target["id"] + ' ON ' + source_support["id"],
                   size=16, color="#345E92"),
              text(852, 528, '目标支撑面  ' + destination_support["label"], size=18),
              text(852, 566, '目标态  ' + target["id"] + ' ON ' + destination_support["id"],
                   size=16, color="#1C806F"),
              text(852, 604, '工作区  work_destination', size=16, color="#52677D"),
              '<line x1="610" y1="553" x2="830" y2="553" stroke="#6B55A1" '
              'stroke-width="6"/>',
              '<rect x="618" y="516" width="205" height="74" rx="13" '
              'fill="#F2EAFE" stroke="#AC93D6" stroke-width="2"/>',
              text(631, 544, '共享门 · open', size=18, weight=700, color="#4F3787"),
              text(631, 572, '净宽待测  not_run', size=14, color="#6B55A1"),
    ]

    parts += [text(40, 710, "任务时序 · 初态与目标态分开", size=23, weight=700)]
    stage_colors = ("#DBEAFE", "#EEE7FA", "#D9F3E9")
    stage_titles = ("PICK  取物", "NAVIGATE  跨门", "PLACE  放置")
    stage_details = (f'{target["id"]}: ON → HELD_BY',
                     f'{request["robot"]["id"]}: {start_room_id} → {destination_room_id}',
                     f'{target["id"]}: HELD_BY → ON')
    for index, x in enumerate((40, 512, 984)):
        parts += [f'<rect x="{x}" y="731" width="416" height="105" rx="15" '
                  f'fill="{stage_colors[index]}"/>',
                  text(x + 20, 771, stage_titles[index], size=20, weight=700),
                  text(x + 20, 808, stage_details[index], size=16, color="#42556B")]
        if index < 2:
            parts.append(text(x + 427, 797, "→", size=31, weight=700, color="#708396"))
    parts += ['<rect x="40" y="865" width="1360" height="65" rx="13" '
              'fill="#FFF5E8" stroke="#E6C791"/>',
              text(60, 891, '待验证：门净宽 · 两端工作区 · 可抓取 · 携物路线 · 任务 rollout',
                   size=17, weight=600, color="#875B27"),
              text(60, 918, '这些是验证义务，不是已通过的物理或机器人结果。',
                   size=15, color="#875B27"),
              '</svg>']
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a fresh path")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    raw = json.loads(args.graph.read_text(encoding="utf-8"))
    request_hash(request)  # task-semantic boundary; refuses obvious credentials and local paths
    graph = validate_plan_graph(raw, request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(request, graph), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "graph_hash": graph.content_hash,
                      "rooms": 2, "steps": len(raw["steps"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
