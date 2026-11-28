# SPDX-License-Identifier: Apache-2.0
"""Auditable material density and rubber-contact friction fallback tables."""

from __future__ import annotations

MATERIAL_DENSITY = {
    "metal": 7800,
    "steel": 7850,
    "iron": 7870,
    "cast iron": 7200,
    "stainless steel": 8000,
    "cobalt alloy": 8300,
    "nickel alloy": 8800,
    "bronze": 8800,
    "brass": 8500,
    "aluminum": 2700,
    "aluminium": 2700,
    "copper": 8960,
    "titanium": 4500,
    "zinc": 7140,
    "zinc alloy": 6600,
    "magnesium alloy": 1800,
    "lead": 11340,
    "carbon fiber": 1750,
    "fiberglass": 1900,
    "plastic": 1050,
    "abs": 1050,
    "pvc": 1400,
    "nylon": 1150,
    "polyethylene": 950,
    "polypropylene": 905,
    "acrylic": 1180,
    "resin": 1200,
    "polycarbonate": 1200,
    "pla": 1240,
    "pet": 1400,
    "hdpe": 950,
    "rubber": 1200,
    "silicone": 1100,
    "foam": 300,
    "wood": 650,
    "plywood": 600,
    "bamboo": 700,
    "cork": 240,
    "mdf": 750,
    "particleboard": 700,
    "oak": 750,
    "pine": 500,
    "glass": 2500,
    "ceramic": 2400,
    "porcelain": 2400,
    "stoneware": 2500,
    "borosilicate": 2230,
    "tempered glass": 2520,
    "stone": 2700,
    "granite": 2700,
    "marble": 2700,
    "concrete": 2300,
    "clay": 1900,
    "fabric": 300,
    "leather": 900,
    "paper": 800,
    "cardboard": 700,
    "food": 1000,
    "water": 1000,
    "ice": 917,
    "unknown": 1000,
}

MATERIAL_FRICTION = {
    "metal": (0.6, 0.5),
    "steel": (0.6, 0.5),
    "aluminum": (0.6, 0.5),
    "plastic": (0.7, 0.5),
    "rubber": (1.0, 0.9),
    "silicone": (0.9, 0.8),
    "wood": (0.7, 0.6),
    "glass": (0.7, 0.5),
    "ceramic": (0.7, 0.5),
    "fabric": (0.8, 0.7),
    "leather": (0.8, 0.6),
    "food": (0.7, 0.5),
    "unknown": (0.7, 0.5),
}


def _normalize(value: str) -> str:
    return (value or "unknown").strip().lower().replace("_", " ")


def density_of(category: str, subtype: str = "") -> tuple[float, str]:
    cat, sub = _normalize(category), _normalize(subtype)
    candidates = sorted(
        (key for key in MATERIAL_DENSITY if key != "unknown"), key=len, reverse=True
    )
    for key in candidates:
        if key == sub or (sub and key in sub):
            return float(MATERIAL_DENSITY[key]), "subtype"
    for key in candidates:
        if key == cat or key in cat:
            return float(MATERIAL_DENSITY[key]), "category"
    return float(MATERIAL_DENSITY["unknown"]), "unknown"


def friction_of(category: str, subtype: str = "") -> tuple[float, float]:
    value = f"{_normalize(category)} {_normalize(subtype)}"
    for key in sorted(MATERIAL_FRICTION, key=len, reverse=True):
        if key != "unknown" and key in value:
            return MATERIAL_FRICTION[key]
    return MATERIAL_FRICTION["unknown"]
