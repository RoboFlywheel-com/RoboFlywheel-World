#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure how much the room shell actually varies, per experiment, as a number.

Why this exists
---------------

Room diversity is a claim about a population, and the original suite could not check it: the only shape assertions were
``len(families) > 1`` and ``len(hashes) == 20``, and the second passes for any generator that draws
one continuous number, because height is a continuous uniform draw. Both are satisfied by a
vocabulary of two shapes and a notch that never changes size.

So this measures the population instead, in the style of ``emit_room_shell_baseline.py``: a report
with ``--check`` floors, committed **before** the change that is meant to move it, so the change has
a before and an after rather than a description. Against the shell as it stood, two of the floors
below fail: ``distinct_topologies`` is 2 because the vocabulary publishes two families, and
``distinct_composites`` is 3 because there are two families and one bite size between them.

The first version of this measured "distinct notch fractions" at the shell's own 3-place resolution
and read **3** for a constant bite -- the two extents are rounded independently before they are
multiplied, so the exact ratio lands on both 0.062 and 0.063. The metric now quantises an order of
magnitude coarser than that noise, and reports the exact count beside it, because a measure that
mistakes arithmetic noise for variety is the failure this file was written to catch.

``neck_m`` is computed structurally from the family and its notches rather than by searching the
outline for a narrow passage: the number that matters is the width of the part a zone plan has to
inset into, and that is a function of what was asked for, not of what the polygon cleaner left.

Output is deterministic -- sorted keys, no timestamp, no paths -- so two runs produce byte-identical
files and a change is reviewed as a diff.

    python -m rfw_w.scenes.preparation.report_room_diversity --check     # non-zero if a floor is violated
    python -m rfw_w.scenes.preparation.report_room_diversity             # print the report
    python -m rfw_w.scenes.preparation.report_room_diversity --json      # the whole report as JSON
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import statistics
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any


from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.cell import CellError, solve_cell
from rfw_w.scenes.solvers.room import (
    ROOM_PROFILE,
    Family,
    RoomError,
    build_shell,
    fit_room_around_cell,
    generate_options,
)
from rfw_w.scenes.solvers.zones import ZoneError, plan_zones

SCHEMA = "pvscene.room-diversity-report/v1"

# The same 20 seeds the baseline fixture covers, so the report and the fixture describe one
# population rather than two that happen to look alike.
SEEDS_PER_EXPERIMENT = 20

# Outlines are also counted at a 5 cm quantisation, because the exact count is inflated by the
# continuous size draw: it is large for any generator, and therefore says nothing about shape.
OUTLINE_QUANTUM_M = 0.05

# What a vocabulary has to clear to be a vocabulary. Every number is derived from the population it
# measures, not chosen for taste:
#
#   distinct_bite_sizes  >= 3   How many different bite sizes the vocabulary produces, counted at
#                               BITE_QUANTUM. The bitten seeds are a third of the population and
#                               the rest are rectangles (0.0), so the frozen shell gives exactly 2
#                               -- which is why this floor fails before the change.
#   distinct_topologies  >= 3   (family, vertex count) -- rect 4, l 6, t/u 8. Two today.
#   distinct_composites  >= 4   (family, vertex count, bite size). Three today.
#   min_family_count     >= 4   Per family across the 60-shell aggregate. Under a fair draw the
#                               expected count is ~20 with sd 3.65, so 4 is about 4.4 sd below the
#                               mean; a fixed-family generator scores 0.
FLOORS: dict[str, int] = {
    "distinct_bite_sizes": 3,
    "distinct_topologies": 3,
    "distinct_composites": 4,
    "min_family_count": 4,
}


def experiments() -> list[Path]:
    return sorted((data_path('examples')).glob("*/task.json"))


def neck_m(family: str, notches: list[dict], width: float, depth: float) -> float | None:
    """The narrowest part a zone plan would have to inset into, from the request alone.

    ``None`` for a rectangle, which has no neck. Otherwise the minimum over the parts the family
    leaves behind:

    ``L``  one corner removed, so two arms survive -- ``width - bite_w`` and ``depth - bite_d``.
    ``T``  both corners of one edge removed, so that edge keeps its middle and the axis keeps a
           stem behind the bites. The two bites share the edge's budget.
    ``U``  an edge recess, so two flanks survive along the edge and a floor survives behind it.
    """
    if family == Family.RECT.value or not notches:
        return None
    if family == Family.L.value:
        bite = notches[0]
        return round(min(width - bite["width_m"], depth - bite["depth_m"]), 4)
    if family == Family.T.value:
        # Both bites are on one edge. Which edge decides which axis carries the neck.
        horizontal = all(
            notch["corner"] in ("bottom_left", "bottom_right", "top_left", "top_right")
            for notch in notches
        )
        along, inward = (width, depth) if horizontal else (depth, width)
        eaten = sum(notch["width_m"] for notch in notches)
        deepest = max(notch["depth_m"] for notch in notches)
        return round(min(along - eaten, inward - deepest), 4)
    if family == Family.U.value:
        recess = notches[0]
        if recess["edge"] in ("bottom", "top"):
            along, inward = width, depth
        elif recess["edge"] in ("left", "right"):
            along, inward = depth, width
        else:
            return None
        return round(min(along - recess["width_m"], inward - recess["depth_m"]), 4)
    return None


def notch_fraction(notches: list[dict], width: float, depth: float) -> float:
    """How much of the bounding box the notches take, at full precision.

    Deliberately not rounded to the shell's own 3-place draw resolution. Measured: with the bite a
    fixed 25 % of each axis, the *exact* ratio still comes out as both 0.062 and 0.063 across the
    population, because ``_clamp`` rounds the two extents independently before they are multiplied.
    Rounding this ratio as well would let that arithmetic noise read as variety -- which is exactly
    how a constant bite survived a test that claimed to check it. The distinct count below
    quantises on purpose, at a coarser step than the noise.
    """
    box = width * depth
    if box <= 0:
        return 0.0
    eaten = sum(notch["width_m"] * notch["depth_m"] for notch in notches)
    return eaten / box


# The step the distinct-bite count quantises at. Two places is an order of magnitude coarser than
# the noise above (0.0625 vs 0.062/0.063), so it counts bites rather than rounding artefacts.
BITE_QUANTUM = 0.01


def quantised_outline(polygon_xy: list[list[float]]) -> tuple:
    return tuple(sorted(
        (round(x / OUTLINE_QUANTUM_M), round(y / OUTLINE_QUANTUM_M)) for x, y in polygon_xy
    ))


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 4),
        "p25": round(ordered[len(ordered) // 4], 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(ordered[(3 * len(ordered)) // 4], 4),
        "max": round(ordered[-1], 4),
    }


def _histogram(values: list[float], bins: int = 10) -> dict[str, int]:
    if not values:
        return {}
    low, high = min(values), max(values)
    if high <= low:
        return {f"{low:.3f}": len(values)}
    step = (high - low) / bins
    counts: Counter[str] = Counter()
    for value in values:
        index = min(int((value - low) / step), bins - 1)
        counts[f"{low + index * step:.3f}"] += 1
    return dict(sorted(counts.items()))


def room_only_population(spec: SceneSpec, seeds: int = SEEDS_PER_EXPERIMENT) -> dict[str, Any]:
    """The population the baseline fixture covers: ``generate_options`` over ``seeds`` seeds.

    Everything the shell can vary is here, because this path takes the whole authored size band and
    is not fitted to a cell. A refusal is recorded rather than skipped -- a generator that crashes
    on the shapes it cannot draw would otherwise look *more* diverse for having drawn fewer.
    """
    shells: list[dict] = []
    refused: Counter[str] = Counter()
    for offset in range(seeds):
        seed = spec.seed + offset
        sampled = _with_seed(spec, seed)
        try:
            options = generate_options(sampled)
            shell = build_shell(sampled, options=options)
        except (RoomError, ValueError) as error:
            refused[type(error).__name__] += 1
            continue
        polygon = [[float(x), float(y)] for x, y in shell.polygon_xy]
        # The sampled dimensions are the outline's bounding box, so the report does not have to
        # re-open the size substream to know what was drawn.
        width = max(x for x, _ in polygon) - min(x for x, _ in polygon)
        depth = max(y for _, y in polygon) - min(y for _, y in polygon)
        notches = [dict(notch) for notch in options.to_dict()["notches"]]
        shells.append({
            "seed": seed,
            "family": options.family.value,
            "vertex_count": len(polygon),
            "width_m": round(width, 4),
            "depth_m": round(depth, 4),
            "openings": len(shell.openings),
            "notches": notches,
            "notch_area_fraction": notch_fraction(notches, width, depth),
            "neck_m": neck_m(options.family.value, notches, width, depth),
            "polygon_xy": polygon,
        })

    polygons = [Polygon.from_sequence([tuple(p) for p in shell["polygon_xy"]]) for shell in shells]
    fractions = [shell["notch_area_fraction"] for shell in shells]
    necks = [shell["neck_m"] for shell in shells if shell["neck_m"] is not None]
    topologies = {(shell["family"], shell["vertex_count"]) for shell in shells}
    # Quantised, so the rounding noise documented on `notch_fraction` cannot read as variety.
    bite_sizes = {round(shell["notch_area_fraction"] / BITE_QUANTUM) for shell in shells}
    composites = {
        (shell["family"], shell["vertex_count"], round(shell["notch_area_fraction"] / BITE_QUANTUM))
        for shell in shells
    }
    widths = [shell["width_m"] for shell in shells]
    depths = [shell["depth_m"] for shell in shells]

    return {
        "seeds_requested": seeds,
        "shells": len(shells),
        "refused": dict(sorted(refused.items())),
        "family_histogram": dict(sorted(Counter(s["family"] for s in shells).items())),
        "corner_histogram": dict(sorted(Counter(
            notch["corner"] for s in shells for notch in s["notches"] if notch["corner"]
        ).items())),
        "edge_histogram": dict(sorted(Counter(
            notch["edge"] for s in shells for notch in s["notches"] if notch["edge"]
        ).items())),
        "vertex_count_histogram": dict(sorted(Counter(
            s["vertex_count"] for s in shells
        ).items())),
        "notch_area_fraction": {
            "distinct_exact": len(set(fractions)),
            "distinct_bite_sizes": len(bite_sizes),
            "stats": _stats(fractions),
            "histogram": _histogram(fractions),
        },
        "neck_m": {"stats": _stats(necks)},
        "size_spread_m": {
            "x": None if not widths else round(max(widths) - min(widths), 4),
            "y": None if not depths else round(max(depths) - min(depths), 4),
        },
        "bbox_area_m2": _stats([s["width_m"] * s["depth_m"] for s in shells]),
        "distinct_outlines": {
            "exact": len({tuple(map(tuple, s["polygon_xy"])) for s in shells}),
            f"at_{OUTLINE_QUANTUM_M:.2f}m": len({quantised_outline(s["polygon_xy"]) for s in shells}),
        },
        "distinct_topologies": len(topologies),
        "distinct_composites": len(composites),
        "area_ratios": _stats([polygon.area_ratio_to_bounding_box() for polygon in polygons]),
        # Merged into the cross-experiment aggregate by ``build_report`` and dropped before the
        # report is serialised. The per-shell records are large and belong to the fixture, not here.
        "_contribution": {
            "families": Counter(s["family"] for s in shells),
            "bites": bite_sizes,
            "topologies": topologies,
            "composites": composites,
        },
    }


def _with_seed(spec: SceneSpec, seed: int) -> SceneSpec:
    """The same spec at a different seed, without reaching into ``SceneSpec``'s internals."""
    payload = spec.to_dict()
    payload["seed"] = seed
    return SceneSpec.from_dict(payload)


def fitted_population(spec: SceneSpec, seeds: int = SEEDS_PER_EXPERIMENT) -> dict[str, Any]:
    """The population the build actually uses, so the margins are the real ones.

    Separate from the room-only population above on purpose: there the shell takes the whole
    authored size band, here it is fitted around a cell, so the two answer different questions.
    ``size_margin_m`` is the number the diversity claim rests on -- how much room was left over
    once the task's protected region was accommodated -- and it is read off the fit rather than
    re-derived, so the report cannot disagree with what the build recorded.

    ``shape_fallback`` counts how many rooms had to give their shape up, and to what. Without it a
    vocabulary could look wider than it is: the seeds that fell back to a rectangle are rectangles.
    """
    try:
        cell = solve_cell(spec, seed=spec.seed)
    except CellError as error:
        return {"solved": False, "reason": str(error)}

    families: Counter[str] = Counter()
    refused: Counter[str] = Counter()
    margins: list[float] = []
    fallbacks: Counter[str | None] = Counter()
    for offset in range(seeds):
        seed = spec.seed + offset
        try:
            fitted = fit_room_around_cell(spec, cell, room_seed=seed)
        except (RoomError, ValueError) as error:
            refused[type(error).__name__] += 1
            continue
        families[fitted.options.family.value] += 1
        fallbacks[fitted.shape_fallback] += 1
        if fitted.size_margin_m is not None:
            margins.append(fitted.size_margin_m[0])

    return {
        "solved": True,
        "shells": sum(families.values()),
        "refused": dict(sorted(refused.items())),
        "family_histogram": dict(sorted(families.items())),
        "shape_fallback": {
            "none": fallbacks[None],
            "gave_ground": {
                str(label): count
                for label, count in sorted(fallbacks.items(), key=lambda item: str(item[0]))
                if label is not None
            },
        },
        "size_margin_m": _stats(margins),
    }


@contextmanager
def _forced_family(family: Family):
    """Make the solver draw one family, the way ``test_solver_room.py`` does.

    ``fit_room_around_cell`` reads the module-level ``ROOM_PROFILE`` at call time, so replacing it
    is enough and no solver signature has to grow a parameter for a measurement.
    """
    from rfw_w.scenes.solvers import room as room_module

    original = room_module.ROOM_PROFILE
    room_module.ROOM_PROFILE = room_module.RoomProfile(
        families=(family, Family.RECT), min_windows=1,
    )
    try:
        yield
    finally:
        room_module.ROOM_PROFILE = original


def family_capacity(spec: SceneSpec, seeds: int = SEEDS_PER_EXPERIMENT) -> dict[str, Any]:
    """What each shape family costs the floor the periphery gets to sample in.

    Why this is here at all
    -----------------------

    The decision to withhold the ``T`` family rested on a number that no committed tool produces:
    ``room.py`` records "9.00 m2 as a rectangle, 7.55 as an alcove, 7.20 as an L and **4.85 as a
    T** -- a 46 % cut". Re-measuring it here gives a different answer, and the difference matters
    because the number is the whole argument. A figure that decided a design choice has to be
    reproducible from a tool in the repository, or the next reader inherits an argument they cannot
    check.

    So each family is measured **on its own** -- forced as the only notched family, with ``RECT``
    as the always-available fallback -- and the numbers are read off the fit and the zone plan
    rather than re-derived, so the report cannot disagree with what the build recorded.

    ``periphery_area_m2`` is ``room - protected``: the floor the periphery may sample in, which is
    the quantity the loss is paid in. A ``T`` that costs placeable area costs it here.
    """
    try:
        cell = solve_cell(spec, seed=spec.seed)
    except CellError as error:
        return {"solved": False, "reason": str(error)}

    families: list[Family] = [Family.RECT, Family.L, Family.T, Family.U]
    table: dict[str, Any] = {}
    for family in families:
        realised: Counter[str] = Counter()
        fallbacks: Counter[str | None] = Counter()
        refused: Counter[str] = Counter()
        room_areas: list[float] = []
        protected_areas: list[float] = []
        periphery_areas: list[float] = []
        placeable_areas: list[float] = []
        with _forced_family(family):
            for offset in range(seeds):
                seed = spec.seed + offset
                try:
                    fitted = fit_room_around_cell(spec, cell, room_seed=seed)
                except (RoomError, ValueError) as error:
                    refused[type(error).__name__] += 1
                    continue
                realised[fitted.options.family.value] += 1
                fallbacks[fitted.shape_fallback] += 1
                room_area = Polygon(fitted.shell.polygon_xy).area
                protected_area = fitted.protected_in_room.area
                room_areas.append(room_area)
                protected_areas.append(protected_area)
                periphery_areas.append(room_area - protected_area)
                try:
                    graph = plan_zones(
                        fitted.shell, spec.style, cell_protected=fitted.protected_in_room,
                    )
                except ZoneError as error:
                    refused[f"ZoneError:{str(error).split('(')[0]}"] += 1
                    continue
                placeable_areas.append(graph.placeable_area_m2)

        table[family.value] = {
            "as_the_only_notched_family": dict(sorted(realised.items())),
            "shape_fallback": {
                "none": fallbacks[None],
                "gave_ground": {
                    str(label): count
                    for label, count in sorted(fallbacks.items(), key=lambda item: str(item[0]))
                    if label is not None
                },
            },
            "refused": dict(sorted(refused.items())),
            "room_area_m2": _stats(room_areas),
            "protected_area_m2": _stats(protected_areas),
            "periphery_area_m2": _stats(periphery_areas),
            "placeable_area_m2": _stats(placeable_areas),
        }
    return {"solved": True, "seeds": seeds, "families": table}


def build_report(
    seeds_per_experiment: int = SEEDS_PER_EXPERIMENT, *, capacity: bool = True
) -> dict:
    """The whole report, as a value, so the test and the CLI share one implementation.

    ``capacity`` exists because that section refits every family over every seed -- four profiles
    against the one the fit normally draws -- and costs about six times what the rest of the report
    does. A caller that is not reading the capacity numbers should not pay for them.
    """
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "note": (
            "How much the room shell varies across seeds, per experiment. Two populations are "
            "reported: the room-only one (generate_options over the authored size band, which is "
            "what tests/fixtures/room_shell_baseline.json freezes) and the fitted one the build "
            "uses. The floors in --check are set so the shape vocabulary as it stood before the "
            "diversity work fails them: two families were published and the bite was a constant "
            "fraction of the room, so there were two distinct topologies and three distinct "
            "composites.\n"
            "Generated by python -m rfw_w.scenes.preparation.report_room_diversity.\n"
            "Each experiment also carries a `family_capacity` section: every shape family forced "
            "alone, so the floor a family costs the periphery is measured before anyone decides "
            "whether to publish it. It refits four profiles over every seed and is about six times "
            "the cost of the rest, so `--no-capacity` skips it."
        ),
        "profile_families": [family.value for family in ROOM_PROFILE.families],
        "seeds_per_experiment": seeds_per_experiment,
        "floors": dict(sorted(FLOORS.items())),
        "aggregate": {},
        "experiments": {},
    }

    aggregate_families: Counter[str] = Counter()
    aggregate_bites: set[int] = set()
    aggregate_topologies: set[tuple] = set()
    aggregate_composites: set[tuple] = set()

    for path in experiments():
        spec = SceneSpec.from_dict(json.loads(path.read_text()))
        room_only = room_only_population(spec, seeds_per_experiment)
        fitted = fitted_population(spec, seeds_per_experiment)
        report["experiments"][path.parent.name] = {
            "room_only": room_only,
            "fitted": fitted,
            # Measured on the published vocabulary only. The families the solver *could* draw are
            # counted here whatever the profile publishes, because the question this answers is
            # "what would publishing this family cost", which is asked before the answer is yes.
            "family_capacity": (
                family_capacity(spec, seeds_per_experiment) if capacity else {"skipped": True}
            ),
        }
        contribution = room_only.pop("_contribution")
        aggregate_families.update(contribution["families"])
        aggregate_bites.update(contribution["bites"])
        aggregate_topologies.update(contribution["topologies"])
        aggregate_composites.update(contribution["composites"])

    report["aggregate"] = {
        "shells": sum(aggregate_families.values()),
        "family_histogram": dict(sorted(aggregate_families.items())),
        "distinct_bite_sizes": len(aggregate_bites),
        "distinct_topologies": len(aggregate_topologies),
        "distinct_composites": len(aggregate_composites),
        "min_family_count": min(aggregate_families.values()) if aggregate_families else 0,
    }
    return report


def violated_floors(report: dict) -> list[str]:
    aggregate = report["aggregate"]
    failures = []
    for name, floor in sorted(FLOORS.items()):
        actual = aggregate.get(name)
        if actual is None or actual < floor:
            failures.append(f"{name}={actual} is below floor {floor}")
    return failures


def serialise(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _print_summary(report: dict) -> None:
    print(f"profile families: {report['profile_families']}")
    for name, item in sorted(report["experiments"].items()):
        room_only = item["room_only"]
        print(f"\n{name}")
        print(f"  room-only   families={room_only['family_histogram']} "
              f"topologies={room_only['distinct_topologies']} "
              f"composites={room_only['distinct_composites']} "
              f"bite-sizes={room_only['notch_area_fraction']['distinct_bite_sizes']} "
              f"(exact {room_only['notch_area_fraction']['distinct_exact']})")
        print(f"              {room_only['distinct_outlines']}")
        print(f"              necks={room_only['neck_m']['stats']} "
              f"refused={room_only['refused'] or 'none'}")
        fitted = item["fitted"]
        print(f"  fitted      families={fitted.get('family_histogram', {})} "
              f"margin={fitted.get('size_margin_m')} refused={fitted.get('refused') or 'none'}")
        print(f"              gave ground={fitted.get('shape_fallback')}")
        capacity = item.get("family_capacity", {})
        if capacity.get("solved"):
            print("  capacity    each family forced alone -- medians in m2:")
            for family, row in sorted(capacity["families"].items()):
                placeable = (row["placeable_area_m2"] or {}).get("median")
                periphery = (row["periphery_area_m2"] or {}).get("median")
                print(f"              {family:<5} placeable={placeable} periphery={periphery} "
                      f"realised={row['as_the_only_notched_family']} "
                      f"fallback={row['shape_fallback']['gave_ground'] or 'none'} "
                      f"refused={row['refused'] or 'none'}")
    print(f"\naggregate: {report['aggregate']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=SEEDS_PER_EXPERIMENT,
                        help="seeds per experiment (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="print the whole report as JSON")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any floor is violated")
    parser.add_argument("--no-capacity", action="store_true",
                        help="skip the per-family capacity section (about 6x the rest of the report)")
    args = parser.parse_args()

    report = build_report(args.seeds, capacity=not args.no_capacity)
    if args.json:
        sys.stdout.write(serialise(report))
    else:
        _print_summary(report)

    if args.check:
        failures = violated_floors(report)
        if failures:
            sys.stdout.flush()
            print("\nthe shell vocabulary is below its floors:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1
        print("\nall floors met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
