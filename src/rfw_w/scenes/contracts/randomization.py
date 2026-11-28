# SPDX-License-Identifier: Apache-2.0
"""Recorded domain randomisation that cannot hide changes to task structure."""

from __future__ import annotations

from dataclasses import dataclass

from rfw_w.scenes.contracts.hashing import is_digest


@dataclass(frozen=True, slots=True)
class RandomizedEntityChoice:
    entity_id: str
    asset_candidate_index: int

    def __post_init__(self) -> None:
        if not self.entity_id or self.asset_candidate_index < 0:
            raise ValueError("a randomised entity choice needs an id and non-negative index")

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "asset_candidate_index": self.asset_candidate_index,
        }


@dataclass(frozen=True, slots=True)
class AppearanceVariation:
    dome_intensity: float
    sun_intensity: float
    ceiling_fill_intensity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "dome_intensity": self.dome_intensity,
            "sun_intensity": self.sun_intensity,
            "ceiling_fill_intensity": self.ceiling_fill_intensity,
        }


@dataclass(frozen=True, slots=True)
class DomainRandomizationRecord:
    """The sampled domain plus the task closure it was forbidden to modify."""

    enabled: bool
    variant_seed: int
    invariant_entity_ids: tuple[str, ...]
    invariant_relation_ids: tuple[str, ...]
    invariant_spec_hash: str
    entity_choices: tuple[RandomizedEntityChoice, ...] = ()
    appearance: AppearanceVariation | None = None
    room_seed: int | None = None
    room_seed_attempts: tuple[int, ...] = ()
    rejected_attempts: tuple[str, ...] = ()
    policy_version: str = "task_invariant_domain_v1"

    def __post_init__(self) -> None:
        if not is_digest(self.invariant_spec_hash):
            raise ValueError("domain randomisation must carry a valid invariant spec hash")
        protected = set(self.invariant_entity_ids)
        touched = {choice.entity_id for choice in self.entity_choices}
        overlap = sorted(protected & touched)
        if overlap:
            raise ValueError(f"randomisation touches invariant entities: {overlap}")

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "variant_seed": self.variant_seed,
            "room_seed": self.room_seed,
            "room_seed_attempts": list(self.room_seed_attempts),
            "rejected_attempts": list(self.rejected_attempts),
            "policy_version": self.policy_version,
            "invariant_entity_ids": list(self.invariant_entity_ids),
            "invariant_relation_ids": list(self.invariant_relation_ids),
            "invariant_spec_hash": self.invariant_spec_hash,
            "entity_choices": [choice.to_dict() for choice in self.entity_choices],
            "appearance": None if self.appearance is None else self.appearance.to_dict(),
        }
