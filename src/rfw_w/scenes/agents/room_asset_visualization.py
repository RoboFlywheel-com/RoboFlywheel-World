# SPDX-License-Identifier: Apache-2.0
"""Read-only top view of the measured G2 shell and placed G3 asset ledger."""
from __future__ import annotations

from html import escape
from typing import Any, Mapping


def render_room_assets_svg(plan: Mapping[str, Any], ledger: Mapping[str, Any]) -> str:
    colours = {"furniture": "#287c71", "target": "#d45735", "optional": "#b08122"}
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="0 0 1000 620">',
             '<rect width="1000" height="620" fill="#f6f8fa"/>',
             '<text x="500" y="38" text-anchor="middle" font-size="23">Two-room asset assembly · static placement</text>']
    def sx(x): return 500 + x*105
    def sy(y): return 520 - y*105
    for room, fill in zip(plan["rooms"], ("#e2f2ec", "#e8eafd")):
        x0,y0,x1,y1=room["bounds_xy_m"]
        lines.append(f'<rect x="{sx(x0)}" y="{sy(y1)}" width="{(x1-x0)*105}" height="{(y1-y0)*105}" fill="{fill}" stroke="#24353b" stroke-width="4"/>')
        lines.append(f'<text x="{sx((x0+x1)/2)}" y="{sy(3.7)}" text-anchor="middle" font-size="20">{escape(room["id"])}</text>')
    low,high=plan["portal"]["span_y_m"]
    lines.append(f'<line x1="500" x2="500" y1="{sy(high)}" y2="{sy(low)}" stroke="#f6f8fa" stroke-width="12"/>')
    lines.append(f'<text x="500" y="{sy(high)-9}" text-anchor="middle" font-size="15">door {high-low:.2f} m</text>')
    for item in ledger["instances"]:
        x,y,_=item["pose_xyz_m"]
        w,d,_=item["dimensions_m"]
        kind = "furniture" if item["support_surface"] else ("optional" if item["priority"] == "optional" else "target")
        colour = colours[kind]
        lines.append(f'<rect x="{sx(x-w/2)}" y="{sy(y+d/2)}" width="{w*105:.2f}" height="{d*105:.2f}" rx="5" fill="{colour}" opacity="0.85"/>')
        if kind == "furniture":
            lines.append(f'<text x="{sx(x)}" y="{sy(y)+5}" text-anchor="middle" font-size="13" fill="white">{escape(item["role_id"])}</text>')
        else:
            lines.append(f'<line x1="{sx(x)}" y1="{sy(y)}" x2="{sx(x)+17}" y2="{sy(y)-15}" stroke="{colour}" stroke-width="2"/>')
            lines.append(f'<text x="{sx(x)+21}" y="{sy(y)-17}" font-size="14" fill="#24353b">{escape(item["category"])}</text>')
    core = sum(item["priority"] == "core" for item in ledger["instances"])
    optional = sum(item["priority"] == "optional" for item in ledger["instances"])
    lines.append(f'<text x="500" y="586" text-anchor="middle" font-size="17">{core} core · {optional} optional · robot grasp/IK/rollout not run</text>')
    lines.append('</svg>')
    return "\n".join(lines) + "\n"
