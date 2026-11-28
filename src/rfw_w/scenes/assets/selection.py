# SPDX-License-Identifier: Apache-2.0
"""Bounded, deterministic name and geometry ranking within the admitted HSSD vocabulary.

Names are retrieval hints, not visual evidence. No asset is claimed to have a handle or a
particular finish until rendered and reviewed. Full synset recall avoids asset-id prefix bias.
"""
from __future__ import annotations

import math
import re
from rfw_w.scenes.assets.providers.hssd import (
    HssdConfig, candidates_for_synset, load_synset_index, measure_glb_extents,
    size_bound_violation,
)
from rfw_w.scenes.contracts.scene_spec import EntitySpec


def name_score(category: str, name: str, material_tokens: tuple[str, ...] = ()) -> float:
    words = set(re.findall(r"[a-z]+", name.lower()))
    preferred = {
        "pen_holder": {"pen", "pencil", "spinning"},
        "notebook": {"notebook", "journal"},
        "side_table": {"side", "end", "bedside"},
        "chair": {"office", "desk", "swivel"},
    }.get(category, set())
    avoided = {
        "pen_holder": {"drawer", "tray", "letter"},
        "chair": {"outdoor", "dining", "adirondack"},
        "rug": {"hide", "cowhide", "sheepskin", "fur"},
    }.get(category, set())
    materials = set(material_tokens)
    if materials & {"oak", "wood", "walnut", "warm_wood"}:
        materials |= {"wood", "wooden", "oak", "walnut", "bamboo", "natural"}
    return 3.0 * len(words & preferred) - 4.0 * len(words & avoided) + len(words & materials)


def rank_for_entity(config: HssdConfig, entity: EntitySpec, *, synsets: tuple[str, ...],
                    target_width_m: float, kind: str = "", palette: tuple[str, ...] = (),
                    max_examined: int = 48) -> list[str]:
    index = load_synset_index(config)
    names = {str(row["id"]): str(row.get("name", ""))
             for synset in synsets for row in index.get(synset, []) if row.get("id")}
    candidates = sorted({asset for synset in synsets for asset in candidates_for_synset(config, synset)})
    tokens = entity.material_tokens or palette
    semantics = {asset: name_score(entity.category, names.get(asset, ""), tokens) for asset in candidates}
    candidates.sort(key=lambda asset: (-semantics[asset], asset))
    scored = []
    for asset in candidates[:max_examined]:
        extent = measure_glb_extents(config.glb_path(asset))
        if extent is None:
            scored.append((2, -semantics[asset], asset))
            continue
        violation = size_bound_violation(extent, entity.size_bounds_m, kind=kind)
        installed = extent.to_list()
        if kind == "wall_hung":
            installed = [installed[0], installed[2], installed[1]]
        outside_declared = bool(entity.size_bounds_m) and any(
            value < bound.minimum or value > bound.maximum
            for value, bound in zip(installed, entity.size_bounds_m or ()))
        # Width is a useful tie-break even for objects without annotated support surfaces.
        width_error = abs(math.log(max(extent.x, 1e-5) / max(target_width_m, 1e-5)))
        # A matching wood label must not promote an oversized cabinet ahead of furniture
        # actually inside the declared interval. The ratio screen remains a recorded fallback.
        band = 3 if violation else (1 if outside_declared else 0)
        scored.append((band, width_error - semantics[asset], asset))
    scored.sort()
    return [asset for _, _, asset in scored] + candidates[max_examined:]
