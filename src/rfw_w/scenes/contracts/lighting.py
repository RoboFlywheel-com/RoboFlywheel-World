# SPDX-License-Identifier: Apache-2.0
"""The lighting a room is asked for, with every source's unit and mapping named.

It exists because three numbers on the renderer's request are not a plan. The request says how
bright the dome, the sun and the ceiling fill are; it does not say where they came from, what unit
they are in, what they were meant to stand for, or which branches of the source they cannot carry.
When the room came back dark, the numbers alone could not say whether the plan was wrong or the
mapping was -- and the answer turned out to be neither: the target was simply never in the light.

Two things are deliberately explicit here rather than implied:

- **Units.** Infinigen's sky is a World node with physical units; the renderer's dome and distant
  lights take its own relative intensities. The mapping is recorded per kind, and a kind with no
  mapping is refused instead of guessed.
- **Losses.** A uniform dome cannot carry a sky gradient or a sun disc, and a rect fill cannot
  carry Infinigen's emitter geometry. Those are named on the source that loses them, so a delivery
  can report them instead of claiming the look was preserved.

Intensities are per profile, and `calibration` says how far each one is from being measured:
`mixed` was calibrated against the exposure metric on a real delivery, the other two are still
proposed. SPEC-008 TASK 3's small/large matrix is what settles whether a level has to scale with
the room, since a wider rect fill spreads the same intensity over more area.
"""
from dataclasses import dataclass, replace
from typing import Any, Mapping

PROFILE_VERSION = "pvscene.lighting/v1"
PROFILES = ("daylight", "warm_artificial", "mixed")
# `spot` is recognised as a kind and deliberately has no mapping: a plan that asks for one gets a
# named gap rather than a guessed stand-in.
KINDS = ("environment", "sun", "rect", "point", "spot")

# What each kind's intensity is measured in, and how that reaches the renderer. A kind absent
# from here cannot be mapped, which is a named failure rather than a silent default.
UNIT_MAPPING = {
    "environment": ("relative-dome-radiance",
                    "Isaac DomeLight intensity; adds radiance uniformly over the sphere, so it "
                    "reaches the interior only through the openings"),
    "sun": ("relative-distant-irradiance",
            "Isaac DistantLight intensity; a parallel source, no disc and no angular size"),
    "rect": ("relative-rect-radiance",
             "Isaac RectLight intensity; a flat emitter with an explicit width and height"),
    "point": ("relative-point-intensity",
              "Isaac SphereLight intensity; used only where a plan asked for a bare bulb"),
}

# The branches a given representation cannot carry. Named per kind so a report can list what the
# render lost rather than only what it kept.
KNOWN_LOSSES = {
    "environment": ("nishita-sky-gradient", "nishita-sun-disc", "atmospheric-scattering"),
    "sun": ("angular-size",),
    "rect": ("infinigen-emitter-shape",),
    "point": (),
    "spot": (),
}

# The levels, and how far each is from being measured. `mixed` was calibrated against the
# exposure metric on a real delivery; the other two are still proposed, because the 18-case matrix
# that would exercise them has not run.
PROFILE_LEVELS = {
    "daylight": {"environment": 1500.0, "sun": 2600.0, "fill": 400.0},
    "warm_artificial": {"environment": 200.0, "sun": 0.0, "fill": 1800.0},
    "mixed": {"environment": 2700.0, "sun": 4500.0, "fill": 3600.0},
}
CALIBRATION = {"daylight": "not_run", "warm_artificial": "not_run",
               "mixed": "measured(2026-09-22)"}
# What the measurement was, so the numbers above are not a claim without a source. The run is
# `outputs/_probe/spec008_profile_trial_x3/measured/report.json`: on the SPEC-005 bedroom delivery
# (4.75 x 3.67 x 2.89 m) the task target went from 99.89 percent of pixels below Y=0.01 at the
# recorded lighting to 8.67 percent, with nothing above Y=0.98.
CALIBRATION_SCOPE = (
    "SPEC-008 TASK 3 ran the 18-group matrix (3 room types x small/large x 3 profiles) and the "
    "calibrated profile is inside the thresholds in all six rooms -- office small dark 0.0952 and "
    "large 0.0075, kitchen 0.0001 / 0.0000, bedroom 0.1668 / 0.0739, against a limit of 0.20, with "
    "the bright fraction 0.0000 everywhere -- so the level does **not** need to scale with room "
    "area: the fixture's size follows the room (`0.7 x width x depth`), the intensity is constant, "
    "and the larger room of each pair came out brighter, not darker. The bedroom at the small size "
    "is the tightest case (0.1668 of 0.20). Evidence: "
    "`outputs/_acceptance/spec008_matrix_20260923_03/report.json`, re-judged from the renders of "
    "`..._01`. One seed per room, camera `task_closeup`.",
)
# Kept as the name the module used before it had a measured profile, so nothing imports a ghost.
PROVISIONAL_LEVELS = PROFILE_LEVELS  # legacy alias
PROFILE_SUN_COLOUR = {"daylight": (1.0, 0.96, 0.88),
                      "warm_artificial": (1.0, 0.82, 0.62),
                      "mixed": (1.0, 0.90, 0.76)}
PROFILE_DOME_COLOUR = {"daylight": (0.92, 0.94, 1.0),
                       "warm_artificial": (0.90, 0.88, 0.88),
                       "mixed": (0.92, 0.93, 0.99)}


class LightingError(ValueError):
    """A plan that cannot be rendered as asked, named rather than approximated."""


@dataclass(frozen=True, slots=True)
class LightSource:
    """One source: what it is, where it came from, and what its representation loses."""

    kind: str
    source: str
    intensity: float
    colour: tuple[float, float, float]
    temperature_k: float | None = None
    position_m: tuple[float, float, float] | None = None
    size_m: tuple[float, float] | None = None
    direction: tuple[float, float, float] | None = None
    losses: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise LightingError(f"LIGHT_ENVIRONMENT_UNSUPPORTED({self.kind})")
        if self.kind not in UNIT_MAPPING:
            raise LightingError(f"LIGHT_UNIT_MAPPING_MISSING({self.kind})")
        if self.intensity < 0:
            raise LightingError(f"LIGHT_INTENSITY_INVALID({self.kind}:{self.intensity})")
        if len(self.colour) != 3:
            raise LightingError(f"LIGHT_COLOUR_INVALID({self.kind})")
        if self.temperature_k is not None and self.temperature_k <= 0:
            raise LightingError(f"LIGHT_TEMPERATURE_INVALID({self.kind}:{self.temperature_k})")

    @property
    def unit(self) -> str:
        return UNIT_MAPPING[self.kind][0]

    @property
    def mapping(self) -> str:
        return UNIT_MAPPING[self.kind][1]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "source": self.source, "intensity": self.intensity,
                "unit": self.unit, "mapping": self.mapping, "colour": list(self.colour),
                "temperature_k": self.temperature_k,
                "position_m": None if self.position_m is None else list(self.position_m),
                "size_m": None if self.size_m is None else list(self.size_m),
                "direction": None if self.direction is None else list(self.direction),
                "losses": list(self.losses), "note": self.note}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LightSource":
        def point(key: str):
            value = payload.get(key)
            return None if value is None else tuple(float(v) for v in value)  # type: ignore[return-value]

        return cls(kind=str(payload["kind"]), source=str(payload["source"]),
                   intensity=float(payload["intensity"]),
                   colour=tuple(float(v) for v in payload["colour"]),  # type: ignore[arg-type]
                   temperature_k=None if payload.get("temperature_k") is None
                   else float(payload["temperature_k"]),
                   position_m=point("position_m"), size_m=point("size_m"),
                   direction=point("direction"),
                   losses=tuple(str(v) for v in payload.get("losses", ())),
                   note=str(payload.get("note", "")))


@dataclass(frozen=True, slots=True)
class LightingPlan:
    """A profile's sources, with the exposure and colour management frozen alongside them."""

    profile: str
    environment: LightSource | None = None
    sun: LightSource | None = None
    fixtures: tuple[LightSource, ...] = ()
    fill: LightSource | None = None
    exposure: float = 0.0
    colour_management: str = "renderer-default"
    seed: int = 0
    calibration: str = "not_run"
    unsupported: tuple[str, ...] = ()
    profile_version: str = PROFILE_VERSION

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise LightingError(f"LIGHT_PROFILE_UNSUPPORTED({self.profile})")
        for source in self.sources():
            if source.kind not in UNIT_MAPPING:
                raise LightingError(f"LIGHT_UNIT_MAPPING_MISSING({source.kind})")
        if not -8.0 <= self.exposure <= 8.0:
            raise LightingError(f"LIGHT_EXPOSURE_INVALID({self.exposure})")

    def sources(self) -> tuple[LightSource, ...]:
        found = [source for source in (self.environment, self.sun, self.fill) if source]
        return tuple(found) + tuple(self.fixtures)

    def named_gaps(self) -> tuple[str, ...]:
        """Everything this plan knows it does not carry, in one tuple a report can quote."""
        gaps = list(self.unsupported)
        for source in self.sources():
            gaps.extend(f"{source.kind}:{loss}" for loss in source.losses)
        return tuple(gaps)

    def to_request_fields(self) -> dict[str, Any]:
        """The renderer's five fields. The names are `LightingRequest`'s, kept as data here so
        the core does not depend on the worker."""
        environment = self.environment
        sun = self.sun
        fill = self.fill
        if environment is None and fill is None:
            raise LightingError(f"LIGHT_PLAN_EMPTY({self.profile})")
        return {
            "dome_intensity": 0.0 if environment is None else environment.intensity,
            "sun_intensity": 0.0 if sun is None else sun.intensity,
            "ceiling_fill_intensity": 0.0 if fill is None else fill.intensity,
            "dome_color": list(environment.colour) if environment else [1.0, 1.0, 1.0],
            "sun_color": list(sun.colour) if sun else [1.0, 1.0, 1.0],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "profile_version": self.profile_version,
                "seed": self.seed, "exposure": self.exposure,
                "colour_management": self.colour_management, "calibration": self.calibration,
                "unsupported": list(self.unsupported), "named_gaps": list(self.named_gaps()),
                "environment": None if self.environment is None else self.environment.to_dict(),
                "sun": None if self.sun is None else self.sun.to_dict(),
                "fill": None if self.fill is None else self.fill.to_dict(),
                "fixtures": [source.to_dict() for source in self.fixtures]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LightingPlan":
        def source(key: str):
            value = payload.get(key)
            return None if value is None else LightSource.from_dict(value)

        return cls(profile=str(payload["profile"]),
                   profile_version=str(payload.get("profile_version", PROFILE_VERSION)),
                   environment=source("environment"), sun=source("sun"), fill=source("fill"),
                   fixtures=tuple(LightSource.from_dict(item)
                                  for item in payload.get("fixtures", ())),
                   exposure=float(payload.get("exposure", 0.0)),
                   colour_management=str(payload.get("colour_management", "renderer-default")),
                   seed=int(payload.get("seed", 0)),
                   calibration=str(payload.get("calibration", "not_run")),
                   unsupported=tuple(str(v) for v in payload.get("unsupported", ())))

    def with_sources(self, **changes: Any) -> "LightingPlan":
        """A changed plan as a new value, so a repair can hand one back."""
        return replace(self, **changes)


def profile(name: str, *, room_width_m: float, room_depth_m: float, room_height_m: float,
            seed: int = 0) -> LightingPlan:
    """A plan whose fixture sits in the room it was asked for rather than at a fixed point.

    SPEC-008 requirement 3: position and size follow the room, so a large room is not lit by a
    copy of a small room's central lamp.
    """
    if name not in PROFILE_LEVELS:
        raise LightingError(f"LIGHT_PROFILE_UNSUPPORTED({name})")
    for label, value in (("width", room_width_m), ("depth", room_depth_m),
                         ("height", room_height_m)):
        if value <= 0:
            raise LightingError(f"LIGHT_ROOM_INVALID({label}:{value})")
    levels = PROFILE_LEVELS[name]
    centre = (room_width_m / 2.0, room_depth_m / 2.0)
    environment = LightSource(
        kind="environment", source="infinigen-nishita", intensity=levels["environment"],
        colour=PROFILE_DOME_COLOUR[name], losses=KNOWN_LOSSES["environment"],
        note="mapped to a uniform dome; the upstream Nishita World node is not exported")
    sun = None
    if levels["sun"] > 0:
        sun = LightSource(kind="sun", source="infinigen-nishita", intensity=levels["sun"],
                          colour=PROFILE_SUN_COLOUR[name], losses=KNOWN_LOSSES["sun"],
                          direction=(0.0, 1.0, -0.6),
                          note="azimuth taken from the room's largest opening; not yet inferred")
    # The fill spans the room, so its size is a fraction of the room rather than a constant.
    fill = None
    if levels["fill"] > 0:
        fill = LightSource(kind="rect", source="plan", intensity=levels["fill"],
                           colour=PROFILE_DOME_COLOUR[name],
                           position_m=(centre[0], centre[1], room_height_m - 0.05),
                           size_m=(room_width_m * 0.7, room_depth_m * 0.7),
                           losses=KNOWN_LOSSES["rect"],
                           note="ceiling fill; the upstream emitter geometry is not carried")
    return LightingPlan(profile=name, environment=environment, sun=sun, fill=fill,
                        seed=seed, calibration=CALIBRATION[name],
                        unsupported=("nishita-world-node-not-exported",))


def from_appearance(appearance: Any, *, profile_name: str, room_width_m: float,
                    room_depth_m: float, room_height_m: float) -> LightingPlan:
    """Bridge a solved `AppearanceSettings` into a plan.

    Takes the object structurally: the appearance lives with the solvers and this is the plan
    contract, so importing it would point the dependency the wrong way.
    """
    if profile_name not in PROFILE_LEVELS:
        raise LightingError(f"LIGHT_PROFILE_UNSUPPORTED({profile_name})")
    base = profile(profile_name, room_width_m=room_width_m, room_depth_m=room_depth_m,
                   room_height_m=room_height_m)
    if base.environment is not None:
        base = base.with_sources(environment=replace(
            base.environment, intensity=float(appearance.dome_intensity),
            colour=tuple(float(c) for c in appearance.dome_color)))  # type: ignore[arg-type]
    if base.sun is not None:
        base = base.with_sources(sun=replace(
            base.sun, intensity=float(appearance.sun_intensity),
            colour=tuple(float(c) for c in appearance.sun_color)))  # type: ignore[arg-type]
    if base.fill is not None:
        base = base.with_sources(fill=replace(base.fill,
                                              intensity=float(appearance.ceiling_fill_intensity)))
    return base
