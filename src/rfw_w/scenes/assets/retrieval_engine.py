# SPDX-License-Identifier: Apache-2.0
"""Native metadata recall and ranking; geometry admission belongs to the adapter.

Catalogues remain raw mappings, so a full release is not duplicated into hundreds
of thousands of asset objects. Only the small recalled set is materialised.
"""

from dataclasses import dataclass
from itertools import permutations, zip_longest
import json
import math
from pathlib import Path
import re
from typing import Any

from rfw_w.scenes.contracts.scene_spec import PhysicsType
from rfw_w.scenes.contracts.units import Vec3


@dataclass(frozen=True)
class RetrievalGoal:
    goal_id: str
    semantic_class: str
    description: str
    physics_type: PhysicsType
    target_size_m: Vec3


@dataclass(frozen=True)
class RetrievalPolicy:
    recall_per_catalog: int = 64
    semantic_threshold: float = .35
    max_dimension_ratio: float = 2.
    dimension_match_mode: str = "axis_aligned"

    def __post_init__(self):
        if self.recall_per_catalog <= 0 or not 0 <= self.semantic_threshold <= 1:
            raise ValueError("invalid recall budget or semantic threshold")
        if not math.isfinite(self.max_dimension_ratio) or self.max_dimension_ratio <= 1:
            raise ValueError("max_dimension_ratio must be finite and greater than one")
        if self.dimension_match_mode not in ("axis_aligned", "permutation_invariant"):
            raise ValueError("unknown dimension matching mode")


@dataclass(frozen=True)
class CatalogAsset:
    asset_id: str
    semantic_class: str
    physics_type: PhysicsType
    dimensions_m: Vec3
    visual_uri: str
    collision_uri: str | None
    mass_kg: float | None
    material_label: str | None
    friction: float | None
    friction_secondary: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalEvidence:
    total_score: float
    semantic_score: float
    size_score: float
    readiness_score: float
    affordance_score: float


@dataclass(frozen=True)
class RetrievalCandidate:
    asset: CatalogAsset
    evidence: RetrievalEvidence


@dataclass(frozen=True)
class RetrievalAttempt:
    candidates: tuple[RetrievalCandidate, ...]
    rejected: tuple[tuple[str, str], ...]
    failure_code: str | None


def _normal_label(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _size_error(dimensions, target, mode):
    variants = permutations(dimensions) if mode == "permutation_invariant" else (dimensions,)
    errors = [tuple(abs(math.log(value / aim)) if aim > 0 else 0.
                    for value, aim in zip(variant, target)) for variant in variants]
    return min(errors, key=lambda values: (max(values), sum(values)))


class ReleaseCatalog:
    """Read versioned catalogue paths; never infer paths from a legacy object ID."""

    def __init__(self, config, *, scorer=None):
        self.config = config
        self.scorer = scorer
        self.records = json.loads(config.index_path.read_bytes())
        if not isinstance(self.records, dict) or "assets" in self.records:
            raise ValueError("catalogue must be keyed by release leaf")
        version = self.records.get("_meta", {}).get("library_version")
        if version is not None and version != config.library_version:
            raise ValueError("catalogue version does not match configured library")
        if config.library_version == "v1.3" and version is None:
            raise ValueError("v1.3 catalogue requires version metadata")
        self.by_class = {}
        for key, row in self.records.items():
            if key == "_meta":
                continue
            if not isinstance(row, dict):
                raise ValueError("catalogue record must be an object")
            prefix = Path(str(row.get("assets_prefix", "")))
            if prefix.is_absolute() or ".." in prefix.parts or not prefix.parts:
                raise ValueError("catalogue requires a relative assets_prefix")
            if config.library_version == "v1.3":
                if not re.fullmatch(r"[0-9]+_[0-9]+_[0-9]{4,}", key) or row.get("leaf", key) != key:
                    raise ValueError("invalid release leaf")
                if prefix.as_posix() != "/".join((*key.split("_")[:2], key)):
                    raise ValueError("release prefix disagrees with release leaf")
            semantic = str(row.get("super_category", "")).strip()
            if semantic:
                self.by_class.setdefault(semantic, []).append(key)

    def recall(self, goal, policy):
        terms = (goal.semantic_class, goal.description)
        scores = (self.scorer.score_many(terms, tuple(self.by_class))
                  if self.scorer else {label: self._lexical_score(terms, label) for label in self.by_class})
        if self.scorer is None:
            for label in scores:
                if _normal_label(label) == _normal_label(goal.semantic_class):
                    scores[label] = 1.
        ranked = sorted(scores, key=lambda label: (-scores[label], label))[:8]
        groups = []
        for label in ranked:
            rows = []
            for key in self.by_class[label]:
                row = self.records[key]
                dimensions = row.get("dimensions_m") or row.get("real_bbox")
                try:
                    dimensions = tuple(float(x) for x in dimensions)
                    if len(dimensions) != 3 or any(x <= 0 or not math.isfinite(x) for x in dimensions):
                        continue
                    dimensions = self._world_dimensions(dimensions)
                    errors = _size_error(dimensions, goal.target_size_m.to_list(), policy.dimension_match_mode)
                except (ValueError, TypeError):
                    continue
                rows.append((sum(errors), key, scores[label]))
            groups.append(sorted(rows))
        # Interleave classes so a populous adjacent class cannot starve the target.
        recalled = []
        for batch in zip_longest(*groups):
            recalled.extend(row for row in batch if row is not None)
            if len(recalled) >= policy.recall_per_catalog:
                break
        return [(self._asset(key), score) for _, key, score in recalled[:policy.recall_per_catalog]]

    @staticmethod
    def _lexical_score(terms, label):
        candidate = set(re.findall(r"\w+", label.lower()))
        scores = []
        for text in terms:
            query = set(re.findall(r"\w+", text.lower()))
            scores.append(len(candidate & query) / max(1, len(candidate | query)))
        return max(scores)

    def _asset(self, key):
        row = self.records[key]
        base = self.config.assets_root / row["assets_prefix"]
        visual = base / "mesh" / f"{key}.obj"
        collision = base / "mesh" / f"{key}_collision.obj"
        return CatalogAsset(key, row["super_category"], PhysicsType(row.get("physics_type", "rigid")),
                            Vec3(*self._world_dimensions(row.get("dimensions_m") or row["real_bbox"])), str(visual),
                            str(collision) if collision.is_file() else None,
                            _number(row.get("mass")), row.get("material"),
                            _number(row.get("mu1")), _number(row.get("mu2")), row)

    @staticmethod
    def _world_dimensions(dimensions):
        # Both declared library versions use a Y-up source mesh. Admission and
        # preparation apply the same recorded X rotation; ranking must agree.
        x, y, z = dimensions
        return (float(x), float(z), float(y))


class RetrievalEngine:
    def __init__(self, registry: ReleaseCatalog, *, policy=None):
        self.registry = registry
        self.policy = policy or RetrievalPolicy()

    def retrieve(self, goal, *, limit=16):
        if type(limit) is not int or limit <= 0:
            raise ValueError("retrieval limit must be positive")
        accepted, rejected = [], []
        for asset, semantic in self.registry.recall(goal, self.policy):
            if asset.physics_type != goal.physics_type:
                rejected.append((asset.asset_id, "PHYSICS_MISMATCH"))
                continue
            if semantic < self.policy.semantic_threshold:
                rejected.append((asset.asset_id, "SEMANTIC_BELOW_THRESHOLD"))
                continue
            errors = _size_error(asset.dimensions_m.to_list(), goal.target_size_m.to_list(),
                                 self.policy.dimension_match_mode)
            if max(errors) > math.log(self.policy.max_dimension_ratio):
                rejected.append((asset.asset_id, "SIZE_OUT_OF_RANGE"))
                continue
            size = math.exp(-2 * sum(errors) / 3)
            checks = (Path(asset.visual_uri).is_file(),
                      bool(asset.collision_uri) or self.registry.config.library_version == "v1.3",
                      asset.mass_kg is not None and asset.mass_kg > 0,
                      asset.friction is not None and asset.friction >= 0)
            readiness = sum(checks) / len(checks)
            evidence = RetrievalEvidence(.5 * semantic + .25 * size + .2 * readiness + .05,
                                         semantic, size, readiness, 1.)
            accepted.append(RetrievalCandidate(asset, evidence))
        accepted.sort(key=lambda hit: (-hit.evidence.total_score, hit.asset.asset_id))
        return RetrievalAttempt(tuple(accepted[:limit]), tuple(rejected),
                                None if accepted else "no_candidates")
