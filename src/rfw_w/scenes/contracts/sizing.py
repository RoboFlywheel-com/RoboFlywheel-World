# SPDX-License-Identifier: Apache-2.0
"""The one rule for whether a measured extent fits a requested size.

There are two retrieval paths in this project -- HSSD furniture and the SimReady rigid library -- and
they disagreed about this. SimReady measured the collision mesh and rejected a candidate more than
2x from the midpoint of the requested range, with the reason recorded. HSSD ranked candidates by
width alone and rejected nothing, so a spec declaring a 0.12-0.35 m ceiling light was handed a
1.67 m pendant that the placement then hung from the ceiling, leaving its shade 1.25 m above the
floor in the walking path.

The fix is not a new rule. It is this rule, applied to the path that skipped it. It lives here, in
the contract layer, so neither path can drift from the other without deleting a shared function --
which is harder to do by accident than editing one of two copies.
"""

from __future__ import annotations

from typing import Sequence

# 2x either way, against the midpoint of the requested range. Loose on purpose: exact containment
# rejects 8 of the 11 assets in a delivered office scene, including a chair over by 3 mm and a plant
# by 1 cm, and a rule that rejects almost everything is a rule that gets switched off. What it does
# catch is the asset that is wrong by a factor -- the pendant, the under-height lamp.
DEFAULT_RATIO_LIMIT = 2.0

# An axis whose declared range sits entirely below this is a thickness or a tolerance, not a size,
# and a factor-of-two band around it is not a meaningful test.
#
# This was found by the office spec. `wall_art` declares 0.02-0.06 m on its thickness axis, and the
# real library's framed pictures are 1-5 mm thick -- so the band around the 0.04 m midpoint is
# [0.02, 0.08] and every real picture fails it. Screening the axis did not catch a bad asset; it
# deleted a category. A rug's 5-20 mm and a notebook's 10-50 mm are the same shape of bound.
#
# 5 cm is the stated threshold: below it, an axis describes what a thing is made of rather than how
# big it is, and the design is not asserting a size there.
MIN_SCREENED_AXIS_M = 0.05


def midpoint_of(ranges: Sequence[object]) -> tuple[float, float, float]:
    """The middle of each declared range, which is what a measurement is compared against.

    Not the maximum: aiming at the top of a range systematically selects the biggest asset in it,
    which for a cup means picking a bowl. `retrieval.target_size_for` makes the same choice for the
    same reason.
    """
    if len(ranges) != 3:
        raise ValueError(f"a size needs three axis ranges, got {len(ranges)}")
    return tuple(  # type: ignore[return-value]
        (float(getattr(axis, "minimum")) + float(getattr(axis, "maximum"))) / 2.0 for axis in ranges
    )


def ratio_violation(
    measured: Sequence[float],
    target: Sequence[float],
    ratio_limit: float = DEFAULT_RATIO_LIMIT,
    *,
    axis_names: str = "xyz",
    screen_below_m: float | None = None,
) -> str | None:
    """Whether a measurement is outside the ratio band around a target, and by which axis.

    Returns a description rather than a boolean so a rejection says which axis and by how much --
    what a reviewer needs to tell a mis-measurement from a genuinely unsuitable asset.

    A target of zero on an axis is skipped rather than treated as a violation: the caller has said
    it does not care about that axis, and dividing by it would reject everything. ``screen_below_m``
    skips axes whose target is under that size, for the reason `MIN_SCREENED_AXIS_M` records.
    """
    if len(measured) != len(target):
        raise ValueError(f"{len(measured)} measured axes against {len(target)} targets")
    for axis, (value, wanted) in enumerate(zip(measured, target)):
        if wanted <= 0:
            continue
        if screen_below_m is not None and wanted < screen_below_m:
            continue
        ratio = value / wanted
        name = axis_names[axis]
        if ratio > ratio_limit:
            return (
                f"axis{name} {value:.4f} m exceeds {ratio_limit}x the requested {wanted:.4f} m"
            )
        if ratio < 1.0 / ratio_limit:
            return (
                f"axis{name} {value:.4f} m is below 1/{ratio_limit}x the requested {wanted:.4f} m"
            )
    return None
