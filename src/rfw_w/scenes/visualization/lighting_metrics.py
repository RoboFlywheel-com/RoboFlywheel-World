# SPDX-License-Identifier: Apache-2.0
"""Whether a capture's task target is actually lit, in linear Rec.709 luminance.

Import-safe and dependency-free: the arithmetic is here so it can be checked against hand
values, and the array plumbing lives in `python -m rfw_w.scenes.validation.check_lighting`.

The thresholds are SPEC-008's proposed ones and are expected to move once the profile is frozen.
They live here as named constants rather than inside a comparison so that a revision shows up as
a number with a reason, not as a silently different operator.
"""
# SPEC-008 acceptance, proposed then to be frozen: among the pixels of the task target, at most
# a fifth may be darker than Y=0.01 and at most a twentieth brighter than Y=0.98.
UNDEREXPOSED_LIMIT = 0.20
OVEREXPOSED_LIMIT = 0.05
DARK_Y = 0.01
BRIGHT_Y = 0.98

# Rec.709 luma weights, on linear values. The weights are for *linear* light, which is why the
# decode below is not optional: averaging sRGB bytes weights the shadows far too heavily and
# would make a dim room look acceptable.
REC709 = (0.2126, 0.7152, 0.0722)


def srgb_to_linear(value):
    """One channel, 0..1 encoded, to 0..1 linear (IEC 61966-2-1)."""
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def luminance(r, g, b):
    """Rec.709 relative luminance of linear channels."""
    return REC709[0] * r + REC709[1] * g + REC709[2] * b


def luminance_from_srgb(r8, g8, b8):
    """Relative luminance straight from an 8-bit sRGB triple."""
    return luminance(srgb_to_linear(r8 / 255.0), srgb_to_linear(g8 / 255.0),
                     srgb_to_linear(b8 / 255.0))


def statistics(luminances):
    """Target-mask statistics for one camera. An empty mask is refused, not defaulted."""
    values = [float(value) for value in luminances]
    if not values:
        raise ValueError('CAMERA_TARGET_NOT_VISIBLE(mask empty)')
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (ordered[midpoint] if len(ordered) % 2
              else (ordered[midpoint - 1] + ordered[midpoint]) / 2)
    return {'target_pixels': len(values),
            'underexposed_fraction': sum(1 for v in values if v < DARK_Y) / len(values),
            'overexposed_fraction': sum(1 for v in values if v > BRIGHT_Y) / len(values),
            'median_y': median,
            'min_y': ordered[0], 'max_y': ordered[-1]}


def check_camera(camera_id, stats):
    """The named failure for a camera whose target is not lit as the SPEC requires."""
    if stats['underexposed_fraction'] > UNDEREXPOSED_LIMIT:
        raise ValueError(f'TARGET_UNDEREXPOSED({camera_id}:{stats["underexposed_fraction"]:.4f}'
                         f'>{UNDEREXPOSED_LIMIT})')
    if stats['overexposed_fraction'] > OVEREXPOSED_LIMIT:
        raise ValueError(f'TARGET_OVEREXPOSED({camera_id}:{stats["overexposed_fraction"]:.4f}'
                         f'>{OVEREXPOSED_LIMIT})')
    return stats


def exposure_summary(stats):
    """A one-line reading for a report, so a raw number is never quoted without its bands."""
    return (f"median Y {stats['median_y']:.4f}, "
            f"dark {stats['underexposed_fraction']:.4f} (limit {UNDEREXPOSED_LIMIT}), "
            f"bright {stats['overexposed_fraction']:.4f} (limit {OVEREXPOSED_LIMIT}), "
            f"n {stats['target_pixels']}")
