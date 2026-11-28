#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build a labelled contact sheet of a delivery's rendered views.

A delivery can hold several render rounds: the loop renders, the critic judges, and a
failed round is rolled back and rebuilt as a new `scene_version`.  Every round's images
are kept under `isaac/<scene_version>/attempt_*/`, so "which image did I get" needs the
manifest to answer -- only one of those rounds is the one the delivery ships.

This tool lays the rounds out as rows (chronological) and the camera views as columns,
and marks the row the manifest points at, so the images can be judged at a glance.
It reads only the delivery; it never writes into it.

    python -m rfw_w.scenes.visualization.contact_sheet \
        --delivery outputs/office_pick_cup/s5_office_full_20260924_02 \
        --output-dir outputs/_acceptance/sheets_20260924
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VIEW_ORDER = (
    "room_eye_north",
    "room_eye_south",
    "desk_workspace",
    "task_closeup",
    "overview_top",
    "overview_corner",
    "layout_top",
)
THUMB_WIDTH = 360
THUMB_HEIGHT = 203  # 16:9, padded if the render is not
LABEL_WIDTH = 340
HEADER_HEIGHT = 30
BACKGROUND = (22, 22, 26)
DRAWN = (240, 240, 240)
MUTED = (150, 150, 158)
DELIVERED = (120, 220, 150)
ROLLED_BACK = (235, 170, 90)


def views_of(request: Path | None, views: dict) -> tuple[str, ...]:
    """The camera order the run asked for, so a renamed or added view shows up as itself."""
    if request is not None and request.is_file():
        try:
            order = [camera["camera_id"] for camera in json.loads(request.read_text())["cameras"]]
            known = [name for name in order if name in views]
            if known:
                return tuple(known)
        except (KeyError, ValueError):
            pass
    return tuple(name for name in VIEW_ORDER if name in views)


def _rounds(delivery: Path, delivered: str | None) -> list[dict[str, object]]:
    """Every rendered attempt, oldest first, with its request and critic response.

    A scene version can hold more than one attempt: the loop re-runs the worker when a
    failure is classified as infrastructure, and every run keeps its own images. Syntax a
    version as several rows keeps them visible instead of hiding all but the last.
    """
    found: list[dict[str, object]] = []
    for version_dir in sorted((delivery / "isaac").glob("*")):
        attempts = sorted(version_dir.glob("attempt_*"))
        for attempt in attempts:
            views = {image.stem: image for image in sorted(attempt.glob("*.png"))}
            request = delivery / "isaac_requests" / version_dir.name / f"{attempt.name}.json"
            response = delivery / "critic_responses" / version_dir.name / f"{attempt.name}.txt"
            scores: dict[str, object] = {}
            if response.is_file():
                try:
                    scores = json.loads(response.read_text())
                except ValueError:
                    scores = {}
            found.append(
                {
                    "scene_version": version_dir.name,
                    "attempt_name": attempt.name,
                    "attempt_index": attempts.index(attempt) + 1,
                    "attempt_count": len(attempts),
                    "is_delivered": bool(
                        delivered and version_dir.name == delivered and attempt is attempts[-1]
                    ),
                    "views": views,
                    "request": request if request.is_file() else None,
                    "scores": (scores.get("scores") or {}),
                    "mtime": attempt.stat().st_mtime,
                }
            )
    return sorted(found, key=lambda row: row["mtime"])


def _label(rounds: list[dict[str, object]], index: int) -> tuple[str, tuple[int, int, int]]:
    import datetime

    row = rounds[index]
    stamp = datetime.datetime.fromtimestamp(row["mtime"]).strftime("%H:%M")
    scores = row["scores"]
    adherence = scores.get("prompt_adherence")
    if row["is_delivered"]:
        head = "delivered"
    elif int(row["attempt_count"]) > 1:
        head = "retried"
    else:
        head = "rolled back"
    colour = DELIVERED if row["is_delivered"] else ROLLED_BACK
    attempt = (f"attempt {row['attempt_index']}" if int(row["attempt_count"]) > 1 else "attempt 1")
    lines = [
        f"round {index + 1}   {stamp}",
        f"{str(row['scene_version'])[:8]}  {attempt}",
        head,
        f"prompt_adherence {adherence}" if adherence is not None else "no critic response",
    ]
    if scores:
        weakest = sorted(scores.items(), key=lambda item: item[1])[:2]
        lines.append(" ".join(f"{k.split('_')[0]} {v}" for k, v in weakest))
    return lines, colour


def _outcome(manifest: dict) -> list[str]:
    """What the delivery says about itself, short enough for the label column."""
    resolution = manifest.get("resolution") or {}
    failed = ", ".join(str(entry.get("gate_id")) for entry in (resolution.get("blockers") or []))
    return [
        f"achieved={resolution.get('achieved')}",
        f"meets_request={resolution.get('meets_request')}",
        f"failed={failed or 'none'}",
        f"not_run={', '.join(resolution.get('not_run_gates') or []) or 'none'}",
    ]


def sheet(delivery: Path, output: Path, views: tuple[str, ...], width: int) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    manifest_path = delivery / "delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    delivered = manifest.get("scene_version")
    rounds = _rounds(delivery, delivered)
    if not rounds:
        raise SystemExit(f"no rendered rounds under {delivery}/isaac")

    height = round(width * 9 / 16)
    room = manifest.get("room") or {}
    area = room.get("area_m2")
    header = [f"{manifest.get('scene_id', delivery.name)}  {delivery.name}"]
    if area:
        header.append(f"room {area:.2f} m2  {room.get('area_ratio_to_bounding_box')} of its box")
    header.extend(_outcome(manifest) if manifest else ["no delivery_manifest.json"])
    header_height = max(HEADER_HEIGHT, 6 + len(header) * 19)

    per_row_views = [views_of(row.get("request"), row["views"]) for row in rounds]
    column_count = max([len(views)] + [len(v) for v in per_row_views])
    image = Image.new(
        "RGB", (LABEL_WIDTH + column_count * width, header_height + len(rounds) * height), BACKGROUND
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    header_names = per_row_views[0] if per_row_views else views
    for column, view in enumerate(header_names):
        draw.text((LABEL_WIDTH + column * width + 6, 7), view, fill=MUTED, font=font)
    for line_number, text in enumerate(header):
        draw.text((8, 6 + line_number * 19), text[:58], fill=DRAWN, font=small)

    for index, row in enumerate(rounds):
        row_views = per_row_views[index]
        top = header_height + index * height
        lines, colour = _label(rounds, index)
        for line_number, text in enumerate(lines):
            draw.text((8, top + 8 + line_number * 19), text,
                      fill=colour if line_number >= 2 else DRAWN, font=small)
        for column, view in enumerate(row_views):
            left = LABEL_WIDTH + column * width
            path = row["views"].get(view)
            if not path:
                draw.text((left + 8, top + 8), "(missing)", fill=ROLLED_BACK, font=small)
                continue
            with Image.open(path) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((width, height))
                image.paste(thumb, (left, top))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pvscene_contact_sheets"))
    parser.add_argument("--thumb-width", type=int, default=THUMB_WIDTH)
    parser.add_argument("--views", default=",".join(VIEW_ORDER))
    args = parser.parse_args(argv)
    views = tuple(name for name in args.views.split(",") if name)

    for delivery in args.delivery:
        manifest_path = delivery / "delivery_manifest.json"
        name = delivery.name
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            name = f"{manifest.get('scene_id', delivery.name)}__{delivery.name}"
        written = sheet(delivery, args.output_dir / f"{name}.png", views, args.thumb_width)
        print(f"{delivery} -> {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
