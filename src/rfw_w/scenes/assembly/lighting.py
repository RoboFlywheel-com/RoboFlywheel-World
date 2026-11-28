# SPDX-License-Identifier: Apache-2.0
"""Lighting plans and their calibration records."""

from __future__ import annotations
from dataclasses import replace
from rfw_w.scenes.solvers.furnish import AppearanceSettings

LIGHTING_PROFILES = ('mixed', 'daylight', 'warm_artificial')

def lighting_plan_for(scene: dict, profile_name: str):
    """The lighting this run renders with: the calibrated profile, times whatever moved it.

    SPEC-008 TASK 3's wiring, and the reason it is a function: the task target rendered
    *essentially unlit* (median Y 0.0020, 99.89% of its pixels under 0.01) because the request was
    built from `AppearanceSettings`' defaults, which were chosen for a different purpose and never
    calibrated against a picture. The calibrated profile is now the *base*, sized to the room it
    lights (`contracts.lighting.profile`).

    **The appearance is a multiplier on that base, not a replacement for it.** It was read as a
    replacement first, and the measurement said no: `--variant-seed` enables domain randomization,
    which *samples* the appearance (dome 720-1080, sun 2080-3120, fill 560-840), so every randomized
    run silently threw the calibrated profile away -- the office renders came out at 866/2158/595,
    which is one third of the calibrated level, with 75% of the target's pixels still dark where the
    threshold is 20%. Reading it as the ratio to its own defaults keeps both claims true at once:
    randomization still varies the picture (SPEC-006's record is unchanged), the repair still moves
    it (its step is already multiplicative -- `repair._adjust_appearance` scales by 1.35), and a run
    nobody varied renders the calibrated profile exactly.

    Returns ``(plan, source, room)``, where the source is `profile` or `profile x appearance` and
    the room is the extent the plan was sized for. The delivery records both.
    """
    from rfw_w.scenes.contracts.lighting import profile as plan_profile

    if profile_name not in LIGHTING_PROFILES:
        raise SystemExit(f"LIGHT_PROFILE_UNSUPPORTED({profile_name}): "
                         f"one of {', '.join(LIGHTING_PROFILES)}")
    polygon = scene["shell"].polygon_xy
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    room = {"room_width_m": max(xs) - min(xs), "room_depth_m": max(ys) - min(ys),
            "room_height_m": float(scene["shell"].height_m)}
    plan = plan_profile(profile_name, **room)
    appearance = scene["plan"].appearance
    unchanged = AppearanceSettings()
    factors = {
        "environment": float(appearance.dome_intensity) / unchanged.dome_intensity,
        "sun": float(appearance.sun_intensity) / unchanged.sun_intensity,
        "fill": float(appearance.ceiling_fill_intensity) / unchanged.ceiling_fill_intensity,
    }
    if all(abs(factor - 1.0) < 1e-9 for factor in factors.values()):
        return plan, "profile", room
    scaled = plan.with_sources(
        environment=(None if plan.environment is None else replace(
            plan.environment, intensity=plan.environment.intensity * factors["environment"])),
        sun=(None if plan.sun is None else replace(
            plan.sun, intensity=plan.sun.intensity * factors["sun"])),
        fill=(None if plan.fill is None else replace(
            plan.fill, intensity=plan.fill.intensity * factors["fill"])),
    )
    return scaled, "profile x appearance", room


def lighting_block_for(scene: dict, profile_name: str) -> dict:
    """The block the delivery publishes for a scene and a profile name.

    One call site shape rather than two: the backend asks for the block to report alongside the
    render, and the manifest asks again when it is written. They were written as
    `lighting_block(*lighting_plan_for(...))` at the manifest, which passes the room as a
    positional argument to a keyword-only parameter -- and it failed *after* the room had been
    built and rendered, on the two runs the lighting matrix needs. A shared helper is the only
    version of this the next change cannot break twice.
    """
    plan, source, room = lighting_plan_for(scene, profile_name)
    return lighting_block(plan, source, room=room)


def lighting_block(plan, source: str, *, room: dict[str, float]) -> dict:
    """What the delivery says about the light it was rendered with.

    A block rather than a sentence, for `background`'s reason: a reader asking "which profile, at
    what numbers, and what could not be carried" needs values. `named_gaps` is the profile's own
    account of the upstream nodes that do not survive the mapping (the Nishita gradient and sun
    disc, the emitter geometry), so the delivery says what the picture is *not*.
    """
    return {
        "profile": plan.profile, "profile_version": plan.profile_version,
        "source": source, "calibration": plan.calibration,
        "request": plan.to_request_fields(),
        "room_m": {key: round(float(value), 4) for key, value in room.items()},
        # Every source the plan carries, not just the extra fixtures: the ceiling fill is the one
        # SPEC-008 requirement 3 is about (its size follows the room), so the delivery has to show
        # where it was put and how big it is -- a block that only listed `fixtures` would report an
        # empty list for the calibrated profile and hide the very thing the profile is sized by.
        "sources": [{"kind": source_item.kind, "source": source_item.source,
                     "intensity": source_item.intensity, "colour": list(source_item.colour),
                     # Positions and sizes are rounded where they are *reported*: they are the
                     # room-fraction arithmetic (0.7 x a 3 m depth is 2.0999999999999996), and a
                     # delivery quoting twelve decimal places of a lamp's size is noise a reader
                     # has to filter. The request keeps the numbers the renderer got.
                     "position_m": ([round(float(v), 4) for v in source_item.position_m]
                                    if source_item.position_m else None),
                     "size_m": ([round(float(v), 4) for v in source_item.size_m]
                                if source_item.size_m else None),
                     "losses": list(source_item.losses)}
                    for source_item in plan.sources()],
        "named_gaps": list(plan.named_gaps()),
    }
