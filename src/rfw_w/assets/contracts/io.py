# SPDX-License-Identifier: Apache-2.0
"""Manifest and path helpers shared by all pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def load_manifest(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            for field in ("asset_id", "class_id"):
                if field not in row:
                    raise ValueError(f"Missing {field!r} at {path}:{line_number}")
            from .request import identifier

            if not isinstance(row, dict):
                raise ValueError("Manifest rows must be objects")
            identifier(row["asset_id"])
            identifier(row["class_id"])
            if any(previous["asset_id"] == row["asset_id"] for previous in rows):
                raise ValueError("Duplicate asset_id in manifest")
            rows.append(row)
    return rows


def select_rows(rows: list[dict], start: int, end: int | None) -> list[dict]:
    return rows[start:end]


def asset_path(root: str | Path, row: dict, suffix: str) -> Path:
    return Path(root) / str(row["class_id"]) / f"{row['asset_id']}{suffix}"


def part_dir(root: str | Path, row: dict) -> Path:
    return Path(root) / str(row["class_id"]) / str(row["asset_id"])


def stable_seed(row: dict, base_seed: int) -> int:
    digest = hashlib.sha256(str(row["asset_id"]).encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def write_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def existing_inputs(
    rows: Iterable[dict], root: str | Path, suffix: str
) -> list[tuple[dict, Path]]:
    return [(row, asset_path(root, row, suffix)) for row in rows]
