# SPDX-License-Identifier: Apache-2.0
"""Readable SVG projections of a validated task-world graph. No model or renderer needed."""
from __future__ import annotations

from html import escape

from rfw_w.scenes.contracts.task_world_graph import TaskWorldGraph


def _text(x: float, y: float, value: str, *, size: int = 16, weight: int = 400,
          fill: str = "#243447") -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{escape(value)}</text>')


def render_task_world_svg(graph: TaskWorldGraph) -> str:
    data = graph.to_dict()
    nodes = data["nodes"]
    house = next(n for n in nodes if n["kind"] == "house")
    robot = next(n for n in nodes if n["kind"] == "robot")
    rooms = [n for n in nodes if n["kind"] == "room"]
    portals = [n for n in nodes if n["kind"] == "portal"]
    steps = data["steps"]
    requirements = data["requirements"]
    children: dict[str, list[dict]] = {}
    by_id = {n["id"]: n for n in nodes}
    for edge in data["edges"]:
        if edge["kind"] == "contains":
            children.setdefault(edge["source_id"], []).append(by_id[edge["target_id"]])

    def descendants(room_id: str) -> list[tuple[int, dict]]:
        result: list[tuple[int, dict]] = []
        seen: set[str] = set()

        def visit(parent_id: str, depth: int) -> None:
            for node in children.get(parent_id, []):
                if node["id"] in seen:
                    continue
                seen.add(node["id"])
                result.append((depth, node))
                visit(node["id"], depth + 1)

        visit(room_id, 0)
        # Legacy graphs may have room_id members without a contains edge.
        for node in nodes:
            if node.get("room_id") == room_id and node["id"] not in seen:
                result.append((0, node))
        return result

    room_entries = {room["id"]: descendants(room["id"]) for room in rooms}
    width = max(1200, 80 + 520 * len(rooms))
    cards_top = 145
    card_h = max(290, 120 + 31 * max((len(v) + 1 for v in room_entries.values()), default=0))
    portal_top = cards_top + card_h + 30
    task_top = portal_top + 105
    req_top = task_top + 72 + 58 * len(steps)
    height = req_top + 80 + 28 * len(requirements)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Task world graph">',
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        '<style>text{font-family:"Noto Sans CJK SC","Noto Sans",Arial,sans-serif}</style>',
        _text(42, 49, house["label"] + " · " + data["graph_id"], size=25, weight=700),
        _text(42, 77, "intent graph  |  revision " + str(data["revision"]) +
              "  |  " + graph.content_hash[:14], size=13, fill="#65758a"),
        _text(42, 101, "Initial: " + "; ".join(
            f'{p["subject"]} {p["predicate"]} {p["object"]}' for p in data["initial_state"]),
            size=14, fill="#466078"),
        _text(42, 123, "Goal: " + "; ".join(
            f'{p["subject"]} {p["predicate"]} {p["object"]}' for p in data["goals"]),
            size=14, weight=600, fill="#275b44"),
    ]
    room_x: dict[str, int] = {}
    for index, room in enumerate(rooms):
        x = 40 + index * 520
        room_x[room["id"]] = x
        parts += [
            f'<rect x="{x}" y="{cards_top}" width="500" height="{card_h}" rx="16" '
            'fill="#ffffff" stroke="#a7b9d2" stroke-width="2"/>',
            f'<rect x="{x}" y="{cards_top}" width="500" height="60" rx="16" fill="#dce9fb"/>',
            _text(x + 22, cards_top + 37, room["label"] + "  [" + room["id"] + "]", size=20, weight=700),
        ]
        members = room_entries[room["id"]]
        if any(p["subject"] == robot["id"] and p["predicate"] == "IN_ROOM"
               and p["object"] == room["id"] for p in data["initial_state"]):
            members = [(0, robot)] + members
        for row, (depth, member) in enumerate(members):
            kind = member["kind"]
            if kind == "distractor_policy":
                detail = (f'optional ≤{member["max_count"]}: '
                          + ", ".join(member["candidate_categories"]))
            elif kind == "wall_set":
                detail = "walls · segments not_run"
            elif kind == "floor":
                detail = "floor · geometry not_run"
            elif kind == "asset_role":
                detail = member.get("task_role", "asset") + " · " + member["label"]
                if member.get("priority") == "optional":
                    detail += " · optional"
                if "purpose" in member:
                    detail += " · " + member["purpose"]
            else:
                detail = kind + " · " + member["label"]
            label = f'{detail}  [{member["id"]}]'
            fill = ("#28736b" if member.get("purpose") == "background" else "#98632d" if kind == "distractor_policy" or
                    member.get("priority") == "optional" else "#243447")
            parts.append(_text(x + 22 + 19 * depth, cards_top + 91 + row * 31,
                               label, size=13 if kind in {"surface", "distractor_policy"}
                               else 14, fill=fill))
    for portal in portals:
        ends = portal["room_ids"]
        x0 = room_x.get(ends[0], 40) + 250
        x1 = room_x.get(ends[1], width - 280) + 250
        mid = (x0 + x1) / 2
        parts += [
            f'<line x1="{x0}" y1="{portal_top + 15}" x2="{x1}" y2="{portal_top + 15}" '
            'stroke="#5676a8" stroke-width="3"/>',
            f'<rect x="{mid - 160}" y="{portal_top - 9}" width="320" height="55" rx="12" '
            'fill="#e8f3e8" stroke="#85b48a"/>',
            _text(mid - 145, portal_top + 14,
                  f'{portal["label"]} [{portal["id"]}]  {portal["initial_state"]}', size=16, weight=700),
            _text(mid - 145, portal_top + 35,
                  f'requested clearance ≥ {portal["requested_clearance_m"]:.2f} m · not_run',
                  size=12, fill="#576b59"),
        ]
    parts.append(_text(42, task_top + 5, "Task timeline", size=21, weight=700))
    for index, step in enumerate(steps):
        y = task_top + 25 + index * 58
        parts += [
            f'<rect x="42" y="{y}" width="{width-84}" height="45" rx="9" fill="#ffffff" stroke="#c8d4e4"/>',
            _text(57, y + 29, f'{index + 1}. {step["id"]} · {step["verb"]}  →  '
                  f'{step["workspace_id"]}', size=16, weight=600),
        ]
        if step["route_portal_ids"]:
            parts.append(_text(width - 330, y + 29, "via " + ", ".join(step["route_portal_ids"]),
                               size=14, fill="#345a8a"))
    parts.append(_text(42, req_top + 5, "Verification obligations · all not_run",
                       size=21, weight=700))
    for index, req in enumerate(requirements):
        parts.append(_text(58, req_top + 39 + index * 28,
                           f'□ {req["id"]}: {req["kind"]}  [{", ".join(req["subject_ids"])}]  '
                           f'· {req["status"]}', size=14))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
