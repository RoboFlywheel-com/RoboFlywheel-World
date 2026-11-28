# SPDX-License-Identifier: Apache-2.0
"""Recover a renderer's effective focal length in pixels from its own depth map.

Why this is measured rather than assumed: Isaac's `_setup_camera` writes only `focalLength`
on the USD camera, so the horizontal aperture -- and with it the horizontal field of view --
is whatever the renderer applies to an attribute nobody authored. The Blender side meanwhile
hand-coded `sensor_width_mm=22.044444`, which is 119.883 degrees at the same focal length
against 117.334 for USD's default aperture. Whichever of those the two renderers actually
produced was never verified, and a material comparison across two different framings judges
texture scale against nothing.

The method needs no calibration target and adds no geometry: the shell's own depth map is
enough. Each pixel gives a depth `t` along the view axis, so under a candidate focal length in
pixels `f` it maps to a world point

    p = eye + t * ((u-cx)/f * x_cam - (v-cy)/f * y_cam - z_cam)

The minus on the `v` term is the image convention, not a sign slip: rows run downwards while
`y_cam` runs upwards, so the lower half of the frame is the part of the world below the camera
axis. Getting it backwards still passes a synthetic depth map traced with the same mistake --
which is why `tests/unit/test_camera_fov_probe.py` also pins one pixel against hand arithmetic,
and why the first run of this probe against a real Isaac depth map is what found the flip.

A horizontal room is explained only by the true `f`: the interior's surfaces are the floor
plane, the ceiling plane and the vertical walls, so the point cloud sits closest to one of the
two horizontal planes exactly when the framing is right.

The score maximised is *how much of the cloud the two horizontal planes explain*, not how far
the cloud is from them. The median distance was tried first and is nearly flat in `f` -- wall
pixels outnumber plane pixels and stay between the planes whatever the framing -- so it puts the
argmin within a few pixels of anywhere. The explained fraction peaks sharply instead.
"""
import argparse
import json
import math
from pathlib import Path

# The search brackets every aperture convention this project has used: 390 px for USD's
# 20.955 mm horizontal default, 370 px for the retired 22.044444 mm Blender constant, and
# 300 px if the renderer fits by USD's 15.2908 mm vertical aperture instead.
SEARCH_FRACTION = (0.2, 0.8)
TOLERANCE_FRACTION = 0.015
# A screen whose best-fit point cloud is not close to the horizontal planes has not been
# explained by this model -- a rendering with no floor in view, say -- and must say so rather
# than report the argmin of a flat score.
#
# Two conditions, not a fixed share of plane pixels: how much of the cloud the planes explain at
# best, and how far that peak stands above the rest of the search. A wide framing puts more wall
# pixels in frame and fewer plane pixels, so a threshold on the share alone would call a correct
# measurement inconclusive -- which is what 0.25 did to the vertical-fit case.
MIN_EXPLAINED_FRACTION = 0.1
PEAK_RATIO = 3.0
PLANE_BAND_M = 0.02
BAND_SIGMA_M = 0.01


def camera_basis(eye, target):
    """World-space camera axes, matching UsdGeom's `SetLookAt` with an up-vector of +Z."""
    def sub(a, b):
        return [a[i] - b[i] for i in range(3)]

    def norm(v):
        length = math.sqrt(sum(c * c for c in v))
        return [c / length for c in v]

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    z_cam = norm(sub(eye, target))  # points behind the camera; the view direction is -z_cam
    x_cam = norm(cross([0.0, 0.0, 1.0], z_cam))
    y_cam = cross(z_cam, x_cam)
    return x_cam, y_cam, z_cam


def world_points(depth, rig, focal_px):
    """The point cloud a candidate focal length implies, as a flat list of (x, y, z)."""
    import numpy as np

    eye = rig['eye']
    x_cam, y_cam, z_cam = camera_basis(eye, rig['target'])
    height, width = depth.shape
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
    t = depth[rows, cols].astype(float)
    u = (cols - cx) / focal_px
    v = (rows - cy) / focal_px
    points = np.empty((t.size, 3), dtype=float)
    for axis in range(3):
        direction = u * x_cam[axis] - v * y_cam[axis] - z_cam[axis]
        points[:, axis] = eye[axis] + t * direction
    return points


def _score(points, height):
    """How much of the cloud the horizontal planes explain, and the hard-band fraction."""
    import numpy as np

    distance = np.minimum(np.abs(points[:, 2]), np.abs(points[:, 2] - height))
    return (float(np.mean(np.exp(-(distance / BAND_SIGMA_M) ** 2))),
            float(np.mean(distance <= PLANE_BAND_M)),
            float(np.median(distance)))


def fit_camera(depth, rig, room_height, *, iterations=4):
    """Fit the focal length in pixels, coarse-to-fine, then report how well it explains the map.

    Returns the measurement and its quality. `conclusive` is false when even the best fit
    leaves the surfaces unexplained, because then the number is an artefact of the search.
    """
    width = int(rig['resolution'][0])
    import numpy as np

    if not np.any(np.isfinite(depth) & (depth > 0)):
        # Nothing to reconstruct: a framing reported from an empty map would be an artefact.
        return {'focal_px': float('nan'), 'explained_fraction': 0.0, 'plane_fraction': 0.0,
                'median_plane_distance_m': float('nan'), 'conclusive': False,
                'resolution': list(rig['resolution']),
                'method': 'depth-to-horizontal-planes', 'reason': 'no valid depth samples'}
    low, high = (SEARCH_FRACTION[0] * width, SEARCH_FRACTION[1] * width)
    best = None
    coarse = []
    for pass_index in range(iterations):
        step = (high - low) / 32.0
        for index in range(33):
            candidate = low + index * step
            explained, fraction, median = _score(world_points(depth, rig, candidate), room_height)
            if pass_index == 0:
                coarse.append(explained)
            if best is None or explained > best['explained_fraction']:
                best = {'focal_px': candidate, 'explained_fraction': explained,
                        'plane_fraction': fraction, 'median_plane_distance_m': median}
        low, high = best['focal_px'] - step, best['focal_px'] + step
    background = sorted(coarse)[len(coarse) // 2]
    best['search_background_fraction'] = background
    best['conclusive'] = (best['explained_fraction'] >= MIN_EXPLAINED_FRACTION
                          and best['explained_fraction'] >= PEAK_RATIO * background)
    best['resolution'] = [int(rig['resolution'][0]), int(rig['resolution'][1])]
    best['method'] = 'depth-to-horizontal-planes'
    return best


def hypotheses(rig):
    """What each aperture convention would predict, so the measurement can be read against them."""
    width, height_px = float(rig['resolution'][0]), float(rig['resolution'][1])
    focal, aperture = float(rig['focal_length_mm']), float(rig['aperture_mm'])
    usd_horizontal, usd_vertical = 20.955, 15.2908  # UsdGeom.Camera's defaults
    retired_blender_sensor = 22.0444444444          # what the recorded pair hard-coded
    return {
        'asserted_aperture_horizontal': round(focal * width / aperture, 4),
        'usd_default_horizontal': round(focal * width / usd_horizontal, 4),
        'usd_default_vertical_fit': round(focal * height_px / usd_vertical, 4),
        'retired_blender_sensor_width': round(focal * width / retired_blender_sensor, 4),
    }


def compare(rig, measured):
    """Which convention the measurement supports, and whether the fixture's value is the one used."""
    predicted = hypotheses(rig)
    nearest = min(predicted, key=lambda name: abs(predicted[name] - measured['focal_px']))
    return {'measured_focal_px': round(measured['focal_px'], 4),
            'nearest_hypothesis': nearest,
            'nearest_hypothesis_focal_px': predicted[nearest],
            'delta_fraction': round(abs(predicted[nearest] - measured['focal_px']) / measured['focal_px'], 6),
            'matches_asserted_aperture': abs(predicted['asserted_aperture_horizontal'] - measured['focal_px'])
                                         <= TOLERANCE_FRACTION * measured['focal_px'],
            'hypotheses': predicted}


def check(rig, measured):
    """Refuse a comparison whose framing is not the one the fixture declares."""
    result = compare(rig, measured)
    if not measured.get('conclusive'):
        raise ValueError(f"CAMERA_FOV_PROBE_INCONCLUSIVE(plane_fraction={measured.get('plane_fraction')})")
    if not result['matches_asserted_aperture']:
        raise ValueError('CAMERA_FOV_AUTHORED_APERTURE_MISMATCH'
                         f"(asserted={result['hypotheses']['asserted_aperture_horizontal']}"
                         f", measured={result['measured_focal_px']}"
                         f", nearest={result['nearest_hypothesis']})")
    return result


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth', type=Path, required=True, help='depth .npy from a capture')
    parser.add_argument('--rig', type=Path, required=True,
                        help="JSON with eye, target, focal_length_mm, aperture_mm, resolution")
    parser.add_argument('--room-height-m', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('output exists; use fresh evidence')
    rig = json.loads(args.rig.read_text())
    measured = fit_camera(np.load(args.depth), rig, args.room_height_m)
    report = {'rig': rig, 'room_height_m': args.room_height_m, 'measured': measured,
              'comparison': compare(rig, measured)}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['comparison'], indent=2))
    try:
        check(rig, measured)
    except ValueError as error:
        print(f'FAIL {error}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
