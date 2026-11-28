# SPDX-License-Identifier: Apache-2.0
"""Room surfaces, chosen from the spec's palette instead of from module constants.

Why this exists
---------------

``StylePlan.palette`` was parsed, validated, hashed and read by nothing, and the room's four
materials were module-level constants in the render worker. So two specs that ask for different
colour families -- ``["oak", "off_white", "muted_green", "brass"]`` against
``["off_white", "oak", "sage", "steel"]`` -- produced byte-identical rooms, which is the same
defect ``spec.prompt`` and ``spec.entities`` had before the furnishing plan was derived from the
spec. The three experiments already carry usable tokens; there was simply nothing reading them.

What the palette decides and what it does not
---------------------------------------------

The palette decides **colour**, and the seed decides **finish**. That split is deliberate: the
colour is the semantic content the spec asked for, so it must not drift with the seed, while
roughness is the kind of thing a domain variant may reasonably vary. A token that no entry covers
is recorded in :attr:`RoomMaterials.unmapped_tokens` rather than resolved to a default -- a
palette asking for a colour family this module does not have is a fact about the request, and the
manifest is where facts about the request belong.

Stdlib only, and walked by ``rfw_w.scenes.validation.check_stdlib_only``, because it sits in the solvers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

from rfw_w.scenes.contracts.scene_spec import StylePlan

# How much the seed may move a surface's roughness. Small on purpose: the finish is a variation,
# not a second decision, and a roughness that swung widely would read as a different material
# rather than the same one under different light.
ROUGHNESS_JITTER = 0.08


@dataclass(frozen=True, slots=True)
class MaterialExportProfile:
    """Explicit opt-in to the versioned architectural bake; native export stays available."""

    version: str = 'pbr-v1'
    resolution: int = 1024
    device: str = 'CPU'

    def __post_init__(self) -> None:
        if (self.version != 'pbr-v1' or self.resolution not in (256, 512, 1024, 2048, 4096)
                or self.device not in ('CPU', 'CUDA')):
            raise ValueError('MATERIAL_BAKE_UNAVAILABLE(unsupported profile)')


@dataclass(frozen=True, slots=True)
class Surface:
    """One flat surface: a colour, how rough it is, and whether it is metal."""

    colour: tuple[float, float, float]
    roughness: float
    metallic: float = 0.0

    def __post_init__(self) -> None:
        if len(self.colour) != 3:
            raise ValueError(f"colour must be three channels, got {self.colour}")
        for channel in self.colour:
            if not math.isfinite(channel) or not 0.0 <= channel <= 1.0:
                raise ValueError(f"colour channels must be within [0, 1], got {self.colour}")
        if not math.isfinite(self.roughness) or not 0.0 <= self.roughness <= 1.0:
            raise ValueError(f"roughness must be within [0, 1], got {self.roughness}")
        if not math.isfinite(self.metallic) or not 0.0 <= self.metallic <= 1.0:
            raise ValueError(f"metallic must be within [0, 1], got {self.metallic}")

    def to_dict(self) -> dict[str, Any]:
        return {"colour": list(self.colour), "roughness": self.roughness, "metallic": self.metallic}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Surface:
        return cls(
            colour=tuple(float(c) for c in payload["colour"]),
            roughness=float(payload["roughness"]),
            metallic=float(payload.get("metallic", 0.0)),
        )


# What a token is allowed to be used for. A palette is a list of colour families, not a list of
# roles, so the role has to come from somewhere; it comes from the token.
#
# There is no CEILING role. A ceiling is the wall colour, lighter -- a palette naming a plaster
# token is naming the walls -- and giving it its own role only made the two compete for the same
# token, which left one of them falling back to a default.
WALL = "wall"
FLOOR = "floor"
ACCENT = "accent"
# The surfaces of a *built* support -- the desk, the island, the nightstand. Its own role because
# the support is the one piece of furniture this pipeline constructs rather than retrieves, and
# because the three prompts name it and then ask for wood: without a role here the palette's wood
# tokens are all claimed by the floor and the wall, and the desk comes out the colour of whatever
# the renderer falls back to.
FURNITURE = "furniture"

# How far the ceiling is moved toward white from the wall colour. Halfway: distinguishable from the
# wall without becoming a second material, which is what a ceiling is.
CEILING_LIGHTEN = 0.5


@dataclass(frozen=True, slots=True)
class PaletteEntry:
    surface: Surface
    roles: frozenset[str]


def _entry(colour, roughness, roles, metallic=0.0) -> PaletteEntry:
    return PaletteEntry(Surface(colour, roughness, metallic), frozenset(roles))


# The tokens the three shipped experiments already name. Not a general colour system: a token that
# is not here is reported, not guessed at, and adding one is a line in this table.
PALETTE_LIBRARY: Mapping[str, PaletteEntry] = {
    # Woods.
    "oak": _entry((0.42, 0.30, 0.21), 0.60, (FLOOR, WALL, FURNITURE)),
    "walnut": _entry((0.29, 0.20, 0.15), 0.58, (FLOOR, WALL, FURNITURE)),
    # Neutrals.
    "off_white": _entry((0.82, 0.79, 0.74), 0.85, (WALL,)),
    "linen": _entry((0.87, 0.84, 0.78), 0.88, (WALL,)),
    # Greens.
    "muted_green": _entry((0.55, 0.62, 0.53), 0.80, (WALL, ACCENT)),
    "sage": _entry((0.60, 0.66, 0.56), 0.80, (WALL, ACCENT)),
    # Metals.
    "brass": _entry((0.72, 0.58, 0.32), 0.35, (ACCENT,), metallic=0.75),
    "steel": _entry((0.62, 0.64, 0.66), 0.30, (ACCENT,), metallic=0.80),
}


@dataclass(frozen=True, slots=True)
class RoomMaterials:
    """Every surface of the shell, as one value a repair can move.

    The defaults are exactly the four colours the render worker used to hardcode, so a request
    built without materials renders byte-identically to one built before this existed.
    """

    wall: Surface = Surface((0.82, 0.79, 0.74), 0.85)
    floor: Surface = Surface((0.42, 0.30, 0.21), 0.60)
    ceiling: Surface = Surface((0.90, 0.89, 0.87), 0.90)
    # What shows through a window and behind its pane.
    facade: Surface = Surface((0.55, 0.68, 0.85), 0.90)
    # Door and window frames. Dark and slightly metallic, which is what makes an opening read as
    # a frame rather than as the edge of a hole; the values are the ones v2 used for its window
    # jambs, mullions and latch.
    frame: Surface = Surface((0.055, 0.065, 0.07), 0.28, 0.72)
    glass: Surface = Surface((0.78, 0.84, 0.88), 0.12)
    # The built support. Defaults to the floor's wood rather than to a neutral, because a desk is
    # the piece every one of these prompts names and then asks to be wood -- and the *emitted* key
    # is conditional (see `to_dict`), so a spec whose palette names no furniture token still
    # produces the payload it produced before this existed.
    furniture: Surface = Surface((0.42, 0.30, 0.21), 0.60)
    # Semi-opaque rather than near-transparent: in headless RTX a near-transparent UsdPreviewSurface
    # renders as a mirror that hides the backdrop behind it, which is the trap v2 documented and
    # answered by shipping no glass geometry at all. 0.45 is a starting point, not a measurement.
    glass_opacity: float = 0.45
    glass_mode: str = "semi_transparent"
    palette_used: tuple[str, ...] = ()
    unmapped_tokens: tuple[str, ...] = ()
    # Roles the palette named nothing for, so the default above was used. Recorded for the same
    # reason as `unmapped_tokens`: a surface that fell back must not read as one that was chosen.
    roles_defaulted: tuple[str, ...] = ()
    # Tokens handed to a second role because nothing else could serve it. Recorded rather than
    # resolved silently, for the same reason as `unmapped_tokens`: a surface wearing a colour that
    # was chosen for something else is a fact about the request.
    palette_reuses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "wall": self.wall.to_dict(),
            "floor": self.floor.to_dict(),
            "ceiling": self.ceiling.to_dict(),
            "facade": self.facade.to_dict(),
            "frame": self.frame.to_dict(),
            "glass": self.glass.to_dict(),
            "glass_opacity": self.glass_opacity,
            "glass_mode": self.glass_mode,
        }
        if self.palette_used:
            payload["palette_used"] = list(self.palette_used)
        if self.unmapped_tokens:
            payload["unmapped_tokens"] = list(self.unmapped_tokens)
        if self.roles_defaulted:
            payload["roles_defaulted"] = list(self.roles_defaulted)
        if self.palette_reuses:
            payload["palette_reuses"] = list(self.palette_reuses)
        # Conditional, unlike the shell's six surfaces: a request whose palette names no furniture
        # token must still hash to what it hashed to before this role existed, and the shell
        # surfaces could not be made conditional without changing every payload ever written.
        if FURNITURE not in self.roles_defaulted:
            payload["furniture"] = self.furniture.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoomMaterials:
        base = cls()
        return cls(
            wall=Surface.from_dict(payload["wall"]) if "wall" in payload else base.wall,
            floor=Surface.from_dict(payload["floor"]) if "floor" in payload else base.floor,
            ceiling=Surface.from_dict(payload["ceiling"]) if "ceiling" in payload else base.ceiling,
            facade=Surface.from_dict(payload["facade"]) if "facade" in payload else base.facade,
            frame=Surface.from_dict(payload["frame"]) if "frame" in payload else base.frame,
            glass=Surface.from_dict(payload["glass"]) if "glass" in payload else base.glass,
            furniture=(
                Surface.from_dict(payload["furniture"]) if "furniture" in payload else base.furniture
            ),
            glass_opacity=float(payload.get("glass_opacity", base.glass_opacity)),
            glass_mode=str(payload.get("glass_mode", base.glass_mode)),
            palette_used=tuple(payload.get("palette_used", ())),
            unmapped_tokens=tuple(payload.get("unmapped_tokens", ())),
            roles_defaulted=tuple(payload.get("roles_defaulted", ())),
            palette_reuses=tuple(payload.get("palette_reuses", ())),
        )

    def surface_for(self, role: str) -> Surface:
        """The default surface for a role, for a palette that named nothing for it."""
        return {WALL: self.wall, FLOOR: self.floor, ACCENT: self.frame, FURNITURE: self.furniture}[role]


def _take(
    role: str, palette: tuple[str, ...], used: set[str], *, allow_reuse: bool = False,
) -> tuple[str, Surface] | None:
    """The token that serves this role, and its surface.

    Two rules, both of which the first version got wrong in a way that showed immediately:

    * **A token serves one role.** ``oak`` may be a floor or a wall; letting the wall take it
      first made the office's walls brown, because the palette lists ``oak`` before ``off_white``
      and nothing said the floor had a stronger claim.
    * **A token named for exactly this role beats one that merely tolerates it.** ``brass`` and
      ``steel`` exist only to be accents; ``muted_green`` is a wall that can also be an accent.
      Without this the office's door frames came out muted green and the ``brass`` it asked for
      went unused, which is the palette being read and then ignored.

    Within each rule the spec's own order decides, because the palette is a list a person wrote.

    A third rule, and it is the exception the first two would otherwise make impossible:

    * **A role that nothing unused can serve may borrow a token another role already has**, when the
      caller says so. The furniture role is the one that does: a palette of ``oak, off_white,
      muted_green, brass`` spends all its wood on the floor and the wall, and refusing to reuse one
      leaves the desk to the renderer's fallback colour -- measurably a lavender desk in a room whose
      prompt asked for warm wood. Borrowing is recorded (``palette_reuses``) rather than silent, and
      an unused token is still preferred, so a palette with a spare wood colour keeps it.
    """
    pool = palette if allow_reuse else [token for token in palette if token not in used]
    candidates = [
        token for token in pool
        if token in PALETTE_LIBRARY
        and role in PALETTE_LIBRARY[token].roles
    ]
    if allow_reuse:
        # Unused first, so borrowing is a last resort rather than the rule.
        candidates = ([t for t in candidates if t not in used] + [t for t in candidates if t in used])
    if not candidates:
        return None
    named_for_it = [token for token in candidates
                    if PALETTE_LIBRARY[token].roles == frozenset({role})]
    token = (named_for_it or candidates)[0]
    return token, PALETTE_LIBRARY[token].surface


def _lightened(surface: Surface, amount: float) -> Surface:
    """The same surface moved toward white, for the ceiling."""
    return Surface(
        colour=tuple(round(c + (1.0 - c) * amount, 6) for c in surface.colour),
        roughness=surface.roughness,
        metallic=surface.metallic,
    )


def _finish(surface: Surface, seed: int, role: str) -> Surface:
    """The same colour, with a roughness the seed may nudge. Never the colour."""
    rng = random.Random(f"{seed}:materials:{role}")
    drift = rng.uniform(-ROUGHNESS_JITTER, ROUGHNESS_JITTER)
    return Surface(
        colour=surface.colour,
        roughness=round(min(max(surface.roughness + drift, 0.0), 1.0), 4),
        metallic=surface.metallic,
    )


@dataclass(frozen=True, slots=True)
class _Resolution:
    """One pass of the room's role resolution.

    ``tokens`` is beside ``surfaces`` rather than derived from them because a role can end up on a
    default colour with no token at all, and "which palette name did the floor get" is a different
    question from "what colour is the floor".
    """

    tokens: dict[str, str | None]
    surfaces: dict[str, Surface]
    used: tuple[str, ...]
    reused: tuple[str, ...]
    defaulted: tuple[str, ...]


def _resolve(palette: tuple[str, ...]) -> _Resolution:
    """Which token serves which role, and the surfaces that follow.

    **The one implementation of the resolution order.** ``materials_for`` and
    :func:`resolve_roles` both go through here, because a second copy of this loop would let the
    plan's material references disagree with the surfaces that are actually rendered -- two answers
    to one question, which is the defect this module's rules exist to avoid.
    """
    base = RoomMaterials()
    defaulted: list[str] = []
    used: list[str] = []
    reused: list[str] = []
    surfaces: dict[str, Surface] = {}
    tokens: dict[str, str | None] = {}
    # The room's own roles first, then the support. Order matters: the support is allowed to borrow,
    # so it has to be resolved after the roles it might borrow from have claimed what they need.
    for role in (FLOOR, WALL, ACCENT, FURNITURE):
        taken = _take(role, palette, set(used), allow_reuse=(role == FURNITURE))
        if taken is None:
            defaulted.append(role)
            tokens[role] = None
            surfaces[role] = base.frame if role == ACCENT else base.surface_for(role)
        else:
            token, surface = taken
            tokens[role] = token
            if token in used:
                reused.append(token)
            else:
                used.append(token)
            surfaces[role] = surface
    return _Resolution(tokens, surfaces, tuple(used), tuple(reused), tuple(defaulted))


def resolve_roles(palette: "tuple[str, ...] | list[str]") -> dict[str, str | None]:
    """The palette token serving each room role, or ``None`` where the palette named none.

    Exists so a caller that needs to *name* the room's materials -- the room plan does -- can ask
    the same question the renderer asked, rather than resolving the palette a second time and
    hoping the two agree.
    """
    return dict(_resolve(tuple(palette)).tokens)


def materials_for(style: StylePlan, *, seed: int) -> RoomMaterials:
    """Resolve the shell's surfaces from a style plan and a seed.

    Deterministic in ``(style, seed)``: the same request gives the same room, which is what lets
    the manifest's material block be compared across runs rather than taken on trust.

    The resolution order is the room's own: the floor is what the room is *stood on* and the wall
    is what is seen most, so they claim their tokens first and the frames take what is left.
    """
    palette = tuple(style.palette)
    base = RoomMaterials()
    unmapped = tuple(token for token in palette if token not in PALETTE_LIBRARY)
    resolved = _resolve(palette)

    # A role on its default colour is *not* finished: it has no palette token to jitter, and the
    # first version of this only applied ``_finish`` on the branch that found one.
    surfaces = {
        role: surface if role in resolved.defaulted else _finish(surface, seed, role)
        for role, surface in resolved.surfaces.items()
    }

    wall = surfaces[WALL]
    # The glass tint and the facade are deliberately not driven by the palette: they are what is
    # *outside* the room, and a spec naming "oak" is not asking for a brown sky.
    return RoomMaterials(
        wall=wall,
        floor=surfaces[FLOOR],
        ceiling=_finish(_lightened(wall, CEILING_LIGHTEN), seed, "ceiling"),
        facade=base.facade,
        frame=surfaces[ACCENT],
        glass=base.glass,
        glass_opacity=base.glass_opacity,
        glass_mode=base.glass_mode,
        furniture=surfaces[FURNITURE],
        palette_used=resolved.used,
        unmapped_tokens=unmapped,
        roles_defaulted=resolved.defaulted,
        palette_reuses=resolved.reused,
    )
