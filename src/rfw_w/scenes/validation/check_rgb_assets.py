#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent verifier for SPEC-014's five-backend RGB asset package.

``package_rgb_assets.py`` copies five renders and writes down what it copied.
That is the producer. This is a **separate process that recomputes the claims
from the published bytes**, so the package does not grade itself.

What it recomputes rather than reads:

* every published PNG and report is re-hashed and held against
  ``provenance.json``; a mismatch is ``MULTISIM_RGB_COPY_MISMATCH(name)``, and a
  missing file is ``MULTISIM_PROVENANCE_MISSING(detail)``;
* the exposure and noise on the shared light rig's calibration patch are
  measured again **from the PNG**, not taken from the backend's report, and
  compared against the report's own number. A backend that reported a figure it
  did not produce is ``MULTISIM_MEASUREMENT_MISMATCH(name:metric)``;
* the five reports must agree on scene version, camera and resolution, and must
  each carry ``status=passed`` with an empty ``gates_failed``.

The one thing it cannot do is decide whether the pictures are any good. User
visual review is always recorded ``not_run`` here; it is the user's call, and a
green machine verdict is not a substitute for it.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.validation.check_rgb_assets \\
      --assets outputs/_paper/spec014_five_sim_rgb_assets_NEW \\
      --lighting experiments/paper_multisim/lighting.json \\
      --output-dir outputs/_acceptance/spec014_assets_NEW
"""

from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


from rfw_w.scenes.workers.multisim.common import load_camera, load_light_rig, measure_exposure, measure_noise

BACKENDS = ("genesis", "isaac", "mujoco", "pybullet", "sapien")

# How far the verifier's own measurement may sit from the backend's reported one
# before the report is called wrong. Both compute the same formula on the same
# pixels, so this is tolerance for float formatting, not for disagreement.
MEASUREMENT_TOLERANCE = 0.01


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", required=True, type=Path,
                        help="the package written by package_rgb_assets.py")
    parser.add_argument("--lighting", type=Path,
                        default=data_path('examples/paper_multisim/lighting.json'),
                        help="the shared light rig the exposure/noise gates are defined by")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="a fresh directory; verification.json is the only thing written")
    args = parser.parse_args()

    assets = args.assets.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory "
                     "(evidence is never overwritten)")
    out.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    not_run: list[dict] = []
    checks: dict[str, dict] = {}
    measurements: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str, requirement: str) -> None:
        checks[name] = {"status": "passed" if ok else "failed",
                        "detail": detail, "requirement": requirement}

    provenance_path = assets / "provenance.json"
    camera_path = assets / "camera.json"
    for needed in (provenance_path, camera_path):
        if not needed.is_file():
            problems.append(f"MULTISIM_PROVENANCE_MISSING({needed.name})")
    if problems:
        payload = {"spec": "SPEC-014", "verifier": "python -m rfw_w.scenes.validation.check_rgb_assets",
                   "assets": str(assets), "verified": False, "problems": problems,
                   "machine_checks": checks, "not_run": not_run, "measurements": measurements,
                   "user_visual_review": "not_run"}
        (out / "verification.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    camera = load_camera(camera_path)
    rig = load_light_rig(args.lighting)

    record("camera_digest", digest(camera_path) == provenance.get("camera_sha256"),
           "camera.json matches the digest the package recorded for it", "SPEC-014 R3")
    if not checks["camera_digest"]["status"] == "passed":
        problems.append("MULTISIM_PROVENANCE_MISSING(camera.json digest)")

    declared_version = provenance.get("scene_version")
    declared_resolution = list(provenance.get("resolution_px", []))

    for name in BACKENDS:
        entry = (provenance.get("backends") or {}).get(name)
        if not entry:
            problems.append(f"MULTISIM_BACKEND_UNAVAILABLE({name})")
            record(f"present[{name}]", False, "no provenance entry", "SPEC-014 R1")
            continue

        rgb_path = assets / entry["asset"]
        report_path = assets / entry["report_asset"]
        if not rgb_path.is_file() or not report_path.is_file():
            problems.append(f"MULTISIM_PROVENANCE_MISSING({name}: published file absent)")
            record(f"present[{name}]", False, "asset or report file missing", "SPEC-014 R1")
            continue

        # 1. the bytes are the bytes the package says they are
        rgb_ok = digest(rgb_path) == entry["rgb_sha256"]
        report_ok = digest(report_path) == entry["report_sha256"]
        record(f"copy[{name}]", rgb_ok and report_ok,
               "published PNG and report re-hash to the recorded digests", "SPEC-014 R3")
        if not (rgb_ok and report_ok):
            problems.append(f"MULTISIM_RGB_COPY_MISMATCH({name})")

        report = json.loads(report_path.read_text(encoding="utf-8"))

        # 2. same scene, same camera, same frame
        version_ok = report.get("scene_version") == declared_version
        record(f"scene_version[{name}]", version_ok,
               "report's scene version equals the package's", "SPEC-014 R1")
        if not version_ok:
            problems.append(f"MULTISIM_SCENE_VERSION_MISMATCH({name})")

        reported_camera = report.get("camera") or {}
        camera_ok = all(
            np.allclose(np.asarray(reported_camera.get(key), dtype=float),
                        np.asarray(getattr(camera, key), dtype=float), atol=1e-9)
            for key in ("eye", "target")
        ) and abs(float(reported_camera.get("fx", 0)) - camera.fx) < 1e-6
        record(f"camera[{name}]", camera_ok,
               "report's eye/target/fx equal the package's frozen camera", "SPEC-014 R1")
        if not camera_ok:
            problems.append(f"MULTISIM_CAMERA_MISMATCH({name})")

        image = np.asarray(Image.open(rgb_path).convert("RGB"))
        size_ok = list(image.shape[1::-1]) == declared_resolution
        record(f"resolution[{name}]", size_ok,
               f"published PNG is {image.shape[1]}x{image.shape[0]}", "SPEC-014 R3")
        if not size_ok:
            problems.append(f"MULTISIM_RGB_INVALID({name},{image.shape[1]}x{image.shape[0]})")

        # 3. status and gates, as the backend itself published them
        status_ok = report.get("status") == "passed"
        gates = report.get("gates_failed") or []
        record(f"status[{name}]", status_ok and not gates,
               f"status={report.get('status')} gates_failed={gates}", "SPEC-014 R1")
        if not status_ok:
            problems.append(f"MULTISIM_BACKEND_UNAVAILABLE({name}: status={report.get('status')})")
        problems.extend(gates)

        # 4. recompute the appearance gates from the published pixels. This is
        #    the part that makes the check independent: a report could claim any
        #    exposure it liked, and only a fresh measurement catches it.
        exposure = measure_exposure(image, rig, camera)
        noise = measure_noise(image, rig, camera)
        measurements[name] = {
            "recomputed_mean_luminance": exposure["measured_mean_luminance"],
            "recomputed_noise": noise["high_frequency_residual"],
            "exposure_in_band": exposure["in_band"],
            "noise_within_limit": noise["within_limit"],
            "channel_deviation": float(
                np.abs(image.astype(float) - image.astype(float).mean(axis=2, keepdims=True)).mean()),
        }
        if not exposure["in_band"]:
            problems.append(
                f"MULTISIM_EXPOSURE_OUTSIDE_BAND({name},"
                f"{exposure['measured_mean_luminance']:.2f},{rig.target_mean_luminance:.2f})")
        if not noise["within_limit"]:
            problems.append(f"MULTISIM_RGB_NOISY({name},{noise['high_frequency_residual']:.3f})")
        # A grey frame is the defect the monochrome gap was named for.
        if measurements[name]["channel_deviation"] < 1.0:
            problems.append(f"MULTISIM_RGB_MONOCHROME({name})")

        # 5. hold the report to its own arithmetic
        reported = None
        # The five backends do not agree on where the block sits. MuJoCo nests its appearance
        # inside `conversion.lighting`, beside the MJCF notes it belongs with; the others put it
        # at the top level. Looking in only one place is not a neutral choice here -- it silently
        # downgraded MuJoCo to `not_run` on 2026-09-26, so the one backend whose materials this
        # figure reworked was also the one whose report nobody was checking.
        for block in (report.get("lighting"),
                      (report.get("conversion") or {}).get("lighting")):
            if isinstance(block, dict) and isinstance(block.get("exposure"), dict):
                reported = block["exposure"].get("measured_mean_luminance")
                if reported is not None:
                    break
        if reported is None:
            not_run.append({"check": f"report_agreement[{name}]",
                            "reason": "report does not carry a measured exposure to compare",
                            "gap": None, "blocks_verified": False})
        else:
            agrees = abs(float(reported) - exposure["measured_mean_luminance"]) <= MEASUREMENT_TOLERANCE
            record(f"report_agreement[{name}]", agrees,
                   f"report says {float(reported):.4f}, verifier measures "
                   f"{exposure['measured_mean_luminance']:.4f}", "SPEC-014 R4")
            if not agrees:
                problems.append(f"MULTISIM_MEASUREMENT_MISMATCH({name}:mean_luminance)")

    # The package's own claim about processing, held to the digests above.
    processing = provenance.get("image_processing")
    record("image_processing_declared", bool(processing),
           f"package declares image_processing={processing!r}", "SPEC-014 R3")
    if not processing:
        problems.append("MULTISIM_PROVENANCE_MISSING(image_processing)")

    verified = not problems and not any(entry["blocks_verified"] for entry in not_run)
    payload = {
        "spec": "SPEC-014",
        "verifier": "python -m rfw_w.scenes.validation.check_rgb_assets",
        "assets": str(assets),
        "lighting_rig": str(args.lighting),
        "verified": verified,
        "problems": sorted(set(problems)),
        "machine_checks": checks,
        "measurements": measurements,
        "exposure_target": {"target": rig.target_mean_luminance,
                            "tolerance": rig.luminance_tolerance,
                            "noise_max": rig.noise_max},
        "not_run": not_run,
        "user_visual_review": "not_run",
    }
    (out / "verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
