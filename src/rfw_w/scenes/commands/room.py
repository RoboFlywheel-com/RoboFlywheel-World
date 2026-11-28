#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate and validate just the room shell, for debugging the geometry stage.

    python -m rfw_w.scenes.commands.room --spec experiments/office_pick_cup/task.json \
        --output-dir outputs/_probe/room_m1

Writes shell.json (the polygon, openings and validation evidence) and shell.svg so the outline
can be looked at. The SVG matters more than it sounds: a bounding-box reduction or a notch on
the wrong corner is obvious in a drawing and nearly invisible in a list of vertices.

Exit code is non-zero when validation fails, so this doubles as a check that a spec produces a
usable room.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path.cwd()

from rfw_w.scenes.contracts.room_plan import check_plan
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.contracts.shell_build import plan_gaps
from rfw_w.scenes.contracts.state import OpeningKind
from rfw_w.scenes.geometry.polygon import Polygon
from rfw_w.scenes.solvers.room import (
    Family,
    ShellOptions,
    _draw_notches,
    _pick,
    build_shell,
    generate_options,
    validate_shell,
)
from rfw_w.scenes.solvers.shell_providers import (
    PROVIDERS,
    ShellProviderError,
    minimum_door_width_m,
    provider_named,
)


def render_svg(shell, *, scale: float = 120.0, pad: float = 0.3,
               overlay: list | None = None) -> str:
    """A top-down drawing of the shell. Deliberately simple and dependency-free.

    ``overlay`` draws one more polygon over the floor -- the task cell's protected region, when the
    room was fitted around one. The room's *reason* is easier to see than its outline: a fitted room
    that does not visibly enclose the region it was drawn for is a bug you can look at.
    """
    polygon = Polygon(shell.polygon_xy)
    min_x, min_y, max_x, max_y = polygon.bounds()
    width = (max_x - min_x) + 2 * pad
    height = (max_y - min_y) + 2 * pad

    def tx(point: tuple[float, float]) -> tuple[float, float]:
        # Flip Y so the drawing matches a top-down view rather than maths convention.
        return ((point[0] - min_x + pad) * scale, (max_y - point[1] + pad) * scale)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * scale:.0f}" '
        f'height="{height * scale:.0f}" viewBox="0 0 {width * scale:.0f} {height * scale:.0f}">',
        '<rect width="100%" height="100%" fill="#101418"/>',
        '<polygon points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in map(tx, polygon.vertices))
        + '" fill="#2b3b4a" stroke="#8fd0ff" stroke-width="3"/>',
    ]
    if overlay:
        parts.append(
            '<polygon points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in map(tx, overlay))
            + '" fill="#c98a3a" fill-opacity="0.45" stroke="#ffd9a0" stroke-width="3"/>'
        )
    for opening in shell.openings:
        x, y = tx(opening.center_xy)
        colour = "#ffd166" if opening.kind is OpeningKind.DOOR else "#7fd1ae"
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{opening.width_m * scale / 2:.2f}" '
            f'fill="none" stroke="{colour}" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{y - 8:.2f}" fill="{colour}" font-size="16" '
            f'text-anchor="middle">{opening.opening_id} {opening.width_m:.2f}m</text>'
        )
    parts.append(
        f'<text x="12" y="{height * scale - 12:.0f}" fill="#8899a6" font-size="16">'
        f'{polygon.area:.2f} m2, {len(polygon.vertices)} vertices, '
        f'bbox fill {polygon.area_ratio_to_bounding_box():.3f}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def load_env_file(path: Path) -> None:
    """Machine-bound paths reach the code as environment variables, never as constants.

    A copy of the four-line reader the other entry points carry rather than an import from one of
    them: those are scripts, not a library, and importing `run_m1_acceptance` to get this would pull
    the whole pipeline in to read six lines.
    """
    import os

    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _draw_notches_for(family: Family, spec: SceneSpec) -> tuple:
    """Notches for a family the spec's seed did not choose, from its own substream.

    A substream keyed on the family rather than a replay of the shape one: replaying would have to
    reproduce exactly how far the main stream had been advanced, so any new draw added to
    ``generate_options`` would silently change what ``--family`` shows. Keying on the family keeps
    the viewer stable as the generator evolves.
    """
    rng = random.Random(f"{spec.seed}:shape:{family.value}")
    x_bounds, y_bounds, _ = spec.room.size_bounds_m
    width = _pick(rng, x_bounds)
    depth = _pick(rng, y_bounds)
    return _draw_notches(rng, family, width, depth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--door-width", type=float,
                        help="override the door width, for exercising the narrow-passage failure")
    parser.add_argument("--no-notch", action="store_true",
                        help="generate a plain rectangle, for comparison against the notched shell")
    parser.add_argument("--family", choices=[family.value for family in Family],
                        help="draw this shape family instead of the one the seed chose, for "
                             "looking at a room the run would not otherwise produce")
    parser.add_argument("--fit-cell", action="store_true",
                        help="draw the room the BUILD draws: solve the task cell and fit the shell "
                             "around its protected region. Without this you are looking at the "
                             "room-only derivation, which the pipeline no longer uses")
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        help="build the shell from the plan with this provider and run the "
                             "`shell.matches_plan` gate. `builtin` needs no external tree and "
                             "answers in a second; `infinigen` runs Blender for about 3 minutes. "
                             "Implies --fit-cell, because a provider is handed a plan and only the "
                             "fitted path produces one")
    args = parser.parse_args()

    spec = SceneSpec.from_dict(json.loads(args.spec.read_text()))
    if args.provider:
        load_env_file(REPO_ROOT / ".env.local")
        # A provider is handed a plan, and only the fitted path produces one. Implied rather than
        # required so `--provider builtin` is a one-word answer to "does the gate pass here"; a
        # silent switch to the *other* room would be worse than an error, and this is neither.
        args.fit_cell = True
    if args.fit_cell:
        # The viewer exists to be looked at, and since the reordering the interesting room is the
        # fitted one. A viewer that drew a different room from the build's would be worse than no
        # viewer: a notch in the wrong place is obvious in a drawing and invisible in a list of
        # vertices, which is the whole reason this tool exists.
        from rfw_w.scenes.solvers.cell import solve_cell
        from rfw_w.scenes.solvers.room import fit_room_around_cell
        from rfw_w.scenes.solvers.room_plan import plan_from_fitted

        cell = solve_cell(spec, seed=spec.seed)
        fitted = fit_room_around_cell(spec, cell, room_seed=spec.seed)
        shell, options = fitted.shell, fitted.options
        # The plan is written *beside* the shell, never folded into it: `shell.json` is compared
        # byte for byte against runs from before the plan existed, and that comparison is the
        # evidence that adding a plan did not move the room.
        plan = plan_from_fitted(spec, cell, fitted, room_seed=spec.seed)
        validation = validate_shell(
            shell, spec, options=options, protected_in_room=fitted.protected_in_room,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "shell.json").write_text(
            json.dumps(
                {
                    "schema_version": "pvscene.shell-run/v1",
                    "scene_id": spec.scene_id,
                    "seed": spec.seed,
                    "shell": shell.to_dict(),
                    "options": options.to_dict(),
                    "validation": validation.to_dict(),
                    # The task's floor, so the drawing shows what the room was fitted around and
                    # not only the outline that resulted.
                    "cell_translation_xy": list(fitted.cell_translation_xy),
                    "protected_xy": fitted.protected_in_room.to_list(),
                    # What the fit had to trade to get here, if anything. A shape that was given up
                    # is invisible in the drawing -- the room simply looks like a rectangle.
                    "shape_fallback": fitted.shape_fallback,
                    "size_margin_m": list(fitted.size_margin_m or ()),
                    "door_bearing_deg": fitted.door_bearing_deg,
                    "door_arc_margin_deg": fitted.door_arc_margin_deg,
                    "door_in_approach_arc": fitted.door_in_approach_arc,
                },
                indent=2, sort_keys=True,
            ) + "\n"
        )
        (args.output_dir / "shell.svg").write_text(
            render_svg(shell, overlay=fitted.protected_in_room.to_list()) + "\n"
        )
        (args.output_dir / "plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2) + "\n"
        )
        summary = {
            "valid": validation.ok, "gaps": list(validation.gaps),
            "measured": validation.measured, "output_dir": str(args.output_dir),
            "fitted": True, "cell_translation_xy": list(fitted.cell_translation_xy),
            "plan_hash": plan.content_hash, "plan_gaps": list(check_plan(plan)),
        }
        if args.provider:
            build_dir = args.output_dir / f"build_{args.provider}"
            try:
                build = provider_named(args.provider).build(plan, output_dir=build_dir)
            except ShellProviderError as error:
                # The provider's failure is a *result* here, not a crash: a shell nobody could
                # build is what `SHELL_BUILD_FAILED` is for, and printing it as a traceback would
                # make a named gap look like a bug in this script.
                #
                # No `built: false` here, and its absence is deliberate: a build that succeeded and
                # disagrees with its plan (`SHELL_BUILT_DOES_NOT_MATCH_PLAN`) and a build that never
                # produced geometry (`SHELL_BUILD_FAILED`) both land in this branch, and a single
                # boolean would report the first as the second. The code at the head of the message
                # is the distinction, and the build directory is named either way.
                summary |= {
                    "provider": args.provider,
                    "shell_matches_plan": False,
                    "provider_error": str(error),
                    "build_dir": str(build_dir),
                }
                print(json.dumps(summary, indent=2))
                return 1
            gate_gaps = plan_gaps(plan, build, minimum_door_width_m=minimum_door_width_m())
            (build_dir / "shell_build.json").write_text(
                json.dumps(build.to_dict(), indent=2, sort_keys=True) + "\n"
            )
            summary |= {
                "provider": args.provider,
                "built": True,
                "build_dir": str(build_dir),
                "shell_source": build.shell.source,
                "measurements": dict(build.measurements),
                "not_measured": list(build.not_measured),
                "not_compared": list(build.not_compared),
                "geometry_uris": list(build.geometry_uris),
                # Named `shell_matches_plan` rather than a bare `ok`, because this summary also
                # carries `valid` (the shell validator) and two booleans both meaning "fine" is how
                # a reader ends up quoting the wrong one.
                "shell_matches_plan": not gate_gaps,
                "shell_matches_plan_gaps": list(gate_gaps),
            }
            print(json.dumps(summary, indent=2))
            return 0 if (validation.ok and not gate_gaps) else 1
        print(json.dumps(summary, indent=2))
        return 0 if validation.ok else 1

    base = generate_options(spec)
    # Derived from the generated program rather than rebuilt field by field: the field-by-field
    # version silently dropped every field it did not name, which is how a viewer ends up drawing a
    # room the pipeline does not build. `--no-notch` and `--family` replace the shape instead.
    if args.no_notch:
        options = replace(base, family=Family.RECT, notches=())
    elif args.family is not None:
        options = replace(base, family=Family(args.family),
                          notches=_draw_notches_for(Family(args.family), spec))
    else:
        options = base
    if args.door_width is not None:
        options = replace(options, door_width_m=args.door_width)
    shell = build_shell(spec, options=options)
    validation = validate_shell(shell, spec, options=options)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shell.json").write_text(
        json.dumps(
            {
                "schema_version": "pvscene.shell-run/v1",
                "scene_id": spec.scene_id,
                "seed": spec.seed,
                "shell": shell.to_dict(),
                "options": options.to_dict(),
                "validation": validation.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output_dir / "shell.svg").write_text(render_svg(shell) + "\n")

    print(json.dumps({
        "valid": validation.ok,
        "gaps": list(validation.gaps),
        "measured": validation.measured,
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
