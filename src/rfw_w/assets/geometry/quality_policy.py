# SPDX-License-Identifier: Apache-2.0
"""Shared configuration for geometry scan and mesh-quality filter.

Edit these thresholds/weights if your downstream pipeline needs different
risk tolerance. All metrics are produced by geometry_scanner.py.
"""

# ---------------------------------------------------------------------------
# Red-flag rules used by both scanner reporting and mesh-quality scoring.
# (metric, op, hard_threshold, short_name, human_description)
# ---------------------------------------------------------------------------
RED_FLAGS = [
    (
        "boundary_edges_ratio",
        ">=",
        0.30,
        "bedge_ratio",
        "边界边占比。水密 mesh 应为 0;≥0.30 表示侧壁有大破口或物体只是半边扫描。",
    ),
    (
        "hull_solidity",
        "<=",
        0.20,
        "hull_sol",
        "凸包实心度。≤0.20 表示主体小但凸包被远处碎片撑大,存在漂浮碎片风险。",
    ),
    (
        "centroid_offset_norm",
        ">=",
        0.15,
        "cent_off",
        "质心相对 bbox 中心的归一化偏移。≥0.15 表示质心被碎片拉偏,物理 CoM 不可信。",
    ),
    (
        "aspect_ratio_max",
        ">=",
        10.0,
        "aspect",
        "bbox 最长/最短边比例。≥10 表示极薄长条或部分残片,接触求解器数值不稳。",
    ),
    (
        "top1_top2_gap",
        "<=",
        0.005,
        "top_gap",
        "最大/第二大连通分量占比之差。≤0.005 表示没有主导主体,可能是一堆等大小碎片。",
    ),
]

# ---------------------------------------------------------------------------
# Scoring weights for the five red flags (must sum to 1.0 ideally).
# ---------------------------------------------------------------------------
RED_FLAG_WEIGHTS = {
    "boundary_edges_ratio": 0.30,
    "hull_solidity": 0.30,
    "centroid_offset_norm": 0.20,
    "aspect_ratio_max": 0.15,
    "top1_top2_gap": 0.10,
}

# ---------------------------------------------------------------------------
# Logistic steepness k for each flag (higher = sharper cliff at threshold).
# ---------------------------------------------------------------------------
RED_FLAG_K = {
    "boundary_edges_ratio": 15.0,
    "hull_solidity": 25.0,
    "centroid_offset_norm": 20.0,
    "aspect_ratio_max": 0.5,
    "top1_top2_gap": 400.0,
}

# ---------------------------------------------------------------------------
# Verdict thresholds.
# ---------------------------------------------------------------------------
REJECT_QUALITY_THRESHOLD = 0.45
REVIEW_QUALITY_THRESHOLD = 0.70
GEOM_ONLY_CAP = 0.85

# VLM optional override (set to None if you never pass VLM results).
VLM_WEIGHT = 0.65
GEOM_WEIGHT = 0.35
