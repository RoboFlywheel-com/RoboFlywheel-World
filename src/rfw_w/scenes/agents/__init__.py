# SPDX-License-Identifier: Apache-2.0
"""Agents: the three places a model is consulted.

Architecture §4 allows at most five actual model requests per scene, and three by default: compile,
an optional layout critique, and a final critique. So these modules are small on purpose -- the
system's behaviour is meant to come from the solvers and validators, and a model is consulted where
semantics or aesthetics genuinely need judgement.

None of them is imported by the deterministic core, and the critic's parsing and gating are pure
functions so they can be tested without credentials or a network.
"""

from __future__ import annotations

from rfw_w.scenes.agents.visual_critic import (
    CRITERIA,
    Critique,
    CritiqueError,
    CritiqueStatus,
    VisualGate,
    VisualThresholds,
    evaluate_visual_gate,
    parse_critique,
    sanitised_payload,
)

__all__ = [
    "CRITERIA",
    "Critique",
    "CritiqueError",
    "CritiqueStatus",
    "VisualGate",
    "VisualThresholds",
    "evaluate_visual_gate",
    "parse_critique",
    "sanitised_payload",
]
