# SPDX-License-Identifier: Apache-2.0
"""Structured part estimates, independently validated before physical calculation."""

from __future__ import annotations
import math
from pathlib import Path


def validate(raw, part_ids):
    from .palette import NAMES

    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("object"), dict)
        or not isinstance(raw.get("parts"), list)
    ):
        raise ValueError("annotations require object and parts")
    obj = raw["object"]
    dims = obj.get("dimension_cm")
    if (
        not isinstance(dims, list)
        or len(dims) != 3
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in dims
        )
    ):
        raise ValueError("invalid object dimensions")
    mass = obj.get("mass_g")
    if mass is not None and (
        type(mass) not in (int, float) or not math.isfinite(mass) or mass <= 0
    ):
        raise ValueError("invalid object mass")
    if any(p < 0 or p >= len(NAMES) for p in part_ids):
        raise ValueError("part IDs exceed unambiguous color palette")
    expected = {NAMES[p].lower() for p in part_ids}
    parts = {}
    for part in raw["parts"]:
        if not isinstance(part, dict):
            raise ValueError("invalid part annotation")
        name = str(part.get("mask_color", "")).lower()
        if name in parts or name not in expected:
            raise ValueError("duplicate or unknown part color")
        material = part.get("material")
        if not isinstance(material, dict) or not material.get("category"):
            raise ValueError("part material is required")
        for k in ("static_friction", "dynamic_friction"):
            v = part.get(k)
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError("finite nonnegative part friction is required")
        parts[name] = part
    if set(parts) != expected:
        raise ValueError("annotations do not cover every segmented part")
    return (
        [float(v) / 100 for v in dims],
        None if mass is None else mass / 1000,
        {p: parts[NAMES[p].lower()] for p in part_ids},
    )
