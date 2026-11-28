# SPDX-License-Identifier: Apache-2.0
"""Mesh quality filter: turn geometry features into accept/review/reject verdicts.

Geometry-only mode (default):
  python mesh_quality_filter.py --geom features.json --out scores.json

With optional VLM judgement:
  python mesh_quality_filter.py --geom features.json --vlm vlm.json --out scores.json

  Two VLM schemas are auto-detected per record:
    - boolean (current vlm_filter_kit): is_single_object / is_rigid_body /
      is_complete / no_floating_fragments / is_recognizable / sim_ready are bools.
      vlm_score = fraction of boolean checks passed; vlm_verdict = accept iff sim_ready.
    - legacy 1-5 numeric: is_complete_entity / no_floating_fragments /
      is_recognizable / sim_ready are floats in [1, 5]; vlm_score = mean / 5.

Outputs:
  scores.json  : per-mesh scores, verdicts, hit rules, raw signals
  scores.csv   : flat table
  scores.md    : human-readable summary
"""

import argparse

import csv

import json

import math

from pathlib import Path

import numpy as np

from rfw_w.assets.geometry.quality_policy import (
    RED_FLAGS,
    RED_FLAG_WEIGHTS,
    RED_FLAG_K,
    REJECT_QUALITY_THRESHOLD,
    REVIEW_QUALITY_THRESHOLD,
    GEOM_ONLY_CAP,
    VLM_WEIGHT,
    GEOM_WEIGHT,
)


def logistic_penalty(val, op, threshold, k):
    """Smooth 0..1 penalty crossing 0.5 at `threshold`."""
    if op == ">=":
        return 1.0 / (1.0 + math.exp(-k * (val - threshold)))
    else:  # '<='
        return 1.0 / (1.0 + math.exp(-k * (threshold - val)))


def hard_hit(val, op, threshold):
    return (op == ">=" and val >= threshold) or (op == "<=" and val <= threshold)


_VLM_BOOL_CHECKS = (
    "is_single_object",
    "is_rigid_body",
    "is_complete",
    "no_floating_fragments",
    "is_recognizable",
    "matches_domain_class",
)


def _vlm_score_and_verdict(v):
    """Return (vlm_score in [0,1] or None, vlm_verdict str or None, extras dict).

    Auto-detects the record schema:
      - boolean sim_ready  -> fraction of boolean checks passed; verdict from sim_ready
      - numeric  sim_ready -> legacy mean(sub-scores)/5; verdict from `verdict` field
    """
    if v is None:
        return None, None, {}

    sim_ready = v.get("sim_ready")
    extras = {}

    # Boolean schema (current vlm_filter_kit).
    if isinstance(sim_ready, bool):
        passed = 0
        total = 0
        for k in _VLM_BOOL_CHECKS:
            bv = v.get(k)
            if isinstance(bv, bool):
                total += 1
                passed += int(bv)
        vlm_score = (passed / total) if total else (1.0 if sim_ready else 0.0)
        vlm_verdict = "accept" if sim_ready else "reject"
        extras = {"vlm_sim_ready": sim_ready, "vlm_failure_step": v.get("failure_step")}
        return vlm_score, vlm_verdict, extras

    # Legacy 1-5 numeric schema.
    vlm_verdict = v.get("verdict")
    vlm_sub = {}
    for k in (
        "is_complete_entity",
        "no_floating_fragments",
        "is_recognizable",
        "sim_ready",
    ):
        sv = v.get(k)
        if (
            isinstance(sv, (int, float))
            and not isinstance(sv, bool)
            and math.isfinite(sv)
        ):
            vlm_sub[k] = sv
    vlm_score = (sum(vlm_sub.values()) / (5.0 * len(vlm_sub))) if vlm_sub else None
    return vlm_score, vlm_verdict, extras


def compute_score(g, v):
    """Return (mesh_quality, verdict, geom_score, geometric_risk, hit_rules,
    confidence, vlm_score, vlm_verdict, vlm_extras)."""
    flag_penalties = []
    hit_rules = []
    for metric, op, thr, short, _ in RED_FLAGS:
        val = g.get(metric)
        if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        k = RED_FLAG_K[metric]
        w = RED_FLAG_WEIGHTS[metric]
        p = logistic_penalty(float(val), op, thr, k)
        flag_penalties.append(w * p)
        if hard_hit(float(val), op, thr):
            hit_rules.append(metric)
    geometric_risk = min(1.0, sum(flag_penalties))
    geom_score = 1.0 - geometric_risk

    vlm_score, vlm_verdict, vlm_extras = _vlm_score_and_verdict(v)

    if vlm_score is None:
        mesh_quality = min(GEOM_ONLY_CAP, geom_score)
        confidence = "geom-only"
    else:
        mesh_quality = VLM_WEIGHT * vlm_score + GEOM_WEIGHT * geom_score
        confidence = "vlm+geom"

    if (
        vlm_verdict in ("reject", "borderline_drop")
        or mesh_quality < REJECT_QUALITY_THRESHOLD
    ):
        verdict = "reject"
    elif vlm_verdict == "borderline_keep" or mesh_quality < REVIEW_QUALITY_THRESHOLD:
        verdict = "review"
    else:
        verdict = "accept"

    return (
        mesh_quality,
        verdict,
        geom_score,
        geometric_risk,
        hit_rules,
        confidence,
        vlm_score,
        vlm_verdict,
        vlm_extras,
    )
