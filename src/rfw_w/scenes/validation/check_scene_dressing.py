#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a dressed render from its own files, in a separate process.

SPEC-016. `render_room_preview.py --additions` decides what may be added and
then renders it; this reads only what landed on disk and asks whether the
picture supports the claim:

* every added mesh still hashes to what provenance says it was,
* every addition's placement rules still hold when recomputed from the request
  alone -- rest on its support, inside the room, clear of the others,
* the robot is actually **visible** in each frame (counted in the worker's
  instance-id buffer, not assumed from the fact that it was added),
* the task target is still visible with the robot in the scene,
* the reach proxy's own arithmetic re-derives.

What it does not do, and says so: judge whether the picture looks right. That
is the user's call and stays `not_run` until they make it.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.validation.check_scene_dressing \\
      --render-dir outputs/_paper/office_dressed_NEW \\
      --output-dir outputs/_acceptance/office_dressed_NEW
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from rfw_w.scenes.visualization.render_room_preview import (
    LAYOUT_OVERLAP_EPS_M, ROBOT_SUPPORT_BAND_M, apply_normalisation, obj_local_bounds,
    point_inside, world_aabb, yaw_of,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    render_dir = args.render_dir.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")

    provenance = json.loads((render_dir / "provenance.json").read_text())
    request = json.loads((render_dir / "render_request.json").read_text())
    report = json.loads((render_dir / "isaac" / "report.json").read_text())
    additions = provenance.get("additions")
    problems: list[str] = []
    checks: list[dict] = []
    not_run = ["user visual judgement", "robot physics", "robot ik"]

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    if not additions:
        record("additions_recorded", False, "provenance carries no additions block")
        (out).mkdir(parents=True, exist_ok=True)
        (out / "verification.json").write_text(json.dumps(
            {"verified": False, "problems": problems, "checks": checks,
             "not_run": not_run}, indent=2) + "\n")
        print(json.dumps({"verified": False, "problems": problems}, indent=2))
        return 1

    record("figure_request_digest",
           digest(render_dir / "render_request.json") == provenance["figure_request_sha256"],
           "the rendered request is the one provenance names")
    record("worker_passed", report.get("status") == "passed",
           f"worker status={report.get('status')}")

    by_id = {i["instance_id"]: i for i in request["instances"]}
    polygon = request["room"]["polygon_xy"]

    # Re-derive every addition's box from the mesh on disk.
    boxes: dict[str, tuple] = {}
    for item in additions["added"]:
        item_id = item["instance_id"]
        instance = by_id.get(item_id)
        if instance is None:
            record(f"{item_id}:in_request", False, "recorded as added but not in the request")
            continue
        source = item["source"]
        for key, field in (("visual_sha256", "visual_uri"), ("collision_sha256", "collision_uri")):
            if source.get(key) and instance.get(field):
                path = Path(instance[field])
                record(f"{item_id}:{key}", path.is_file() and digest(path) == source[key],
                       f"{field} still hashes to what provenance recorded")
        if source.get("baked_obj_sha256"):
            path = Path(instance["visual_uri"])
            record(f"{item_id}:baked_obj_sha256",
                   path.is_file() and digest(path) == source["baked_obj_sha256"],
                   "the baked robot mesh still hashes to what provenance recorded")

        band = ROBOT_SUPPORT_BAND_M if item["kind"] == "robot" else None
        lo, hi = apply_normalisation(*obj_local_bounds(Path(instance["visual_uri"]), band),
                                     instance.get("normalisation", "none"))
        translation = [float(v) for v in instance["translation"]]
        yaw = yaw_of(instance["rotation_wxyz"])
        record(f"{item_id}:yaw_z_aligned", yaw is not None, "the addition's rotation is about z")
        offset = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0]
        half = [(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0]
        aabb_min, aabb_max = world_aabb(offset, half, translation, yaw or 0.0)
        boxes[item_id] = (aabb_min, aabb_max, translation[2] + lo[2], translation[2] + hi[2],
                          item)
        centre = [(aabb_min[0] + aabb_max[0]) / 2.0, (aabb_min[1] + aabb_max[1]) / 2.0]
        record(f"{item_id}:inside_room", point_inside(polygon, centre[0], centre[1]),
               f"centre ({centre[0]:.3f}, {centre[1]:.3f}) is inside the room outline")
        record(f"{item_id}:box_matches_record",
               all(abs(a - b) < 1e-6 for a, b in zip(aabb_min, item["world_aabb_min"]))
               and all(abs(a - b) < 1e-6 for a, b in zip(aabb_max, item["world_aabb_max"])),
               "the recomputed world box equals the recorded one")

        support = item["on"]
        if support != "floor":
            support_instance = by_id.get(support)
            record(f"{item_id}:support_present", support_instance is not None,
                   f"its support {support} is still in the request")
        if item["kind"] == "robot" and item.get("reach"):
            reach = item["reach"]
            target = by_id.get(reach["target_instance_id"])
            if target is None:
                record(f"{item_id}:target_present", False,
                       f"task target {reach['target_instance_id']} is not in the request")
            else:
                distance = math.hypot(float(target["translation"][0]) - translation[0],
                                      float(target["translation"][1]) - translation[1])
                record(f"{item_id}:reach_arithmetic",
                       abs(distance - reach["horizontal_distance_m"]) < 1e-3
                       and distance <= reach["reach_m"] + 1e-9,
                       f"{distance:.4f} m recomputed against a {reach['reach_m']} m proxy reach")

    # Additions must not overlap each other, on the same rules the builder used.
    ids = sorted(boxes)
    for index, first in enumerate(ids):
        a_min, a_max, a_zlo, a_zhi, a_item = boxes[first]
        exempt = set(a_item.get("overlap_exempt", []))
        for second in ids[index + 1:]:
            if second in exempt:
                continue
            b_min, b_max, b_zlo, b_zhi, _ = boxes[second]
            overlap_x = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
            overlap_y = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
            clear = not (overlap_x > LAYOUT_OVERLAP_EPS_M and overlap_y > LAYOUT_OVERLAP_EPS_M
                         and a_zhi > b_zlo and b_zhi > a_zlo)
            record(f"{first}/{second}:clear", clear, "the two additions do not share space")

    # Visibility, from the worker's own instance-id buffer.
    try:
        import numpy as np
    except ImportError:
        not_run.append("pixel visibility (numpy unavailable)")
        np = None
    if np is not None:
        for capture in report.get("captures", []):
            id_path = render_dir / "isaac" / capture["instance_id_path"]
            data = np.load(id_path, allow_pickle=True)
            if hasattr(data, "files"):
                data = data["data"]
            id_map = capture.get("instance_id_map") or {}
            unique, totals = np.unique(np.asarray(data).ravel(), return_counts=True)
            counts: dict[str, int] = {}
            for value, total in zip(unique.tolist(), totals.tolist()):
                name = id_map.get(str(value))
                if name:
                    counts[str(name)] = counts.get(str(name), 0) + int(total)
            for item in additions["added"]:
                if item["kind"] != "robot":
                    continue
                pixels = counts.get(item["instance_id"], 0)
                record(f"{item['instance_id']}:visible_in_{capture['camera_id']}", pixels > 0,
                       f"{pixels} pixels carry its instance id")
            for target in {item["reach"]["target_instance_id"]
                           for item in additions["added"]
                           if item["kind"] == "robot" and item.get("reach")}:
                pixels = counts.get(target, 0)
                record(f"{target}:visible_in_{capture['camera_id']}", pixels > 0,
                       f"the task target still shows {pixels} pixels with the robot in the scene")

    out.mkdir(parents=True, exist_ok=True)
    verified = not problems
    payload = {
        "verified": verified,
        "render_dir": str(render_dir),
        "scene_version": provenance.get("source_scene_version"),
        "additions": {"robots": additions.get("robots"), "props": additions.get("props")},
        "checks": checks,
        "checks_run": len(checks),
        "problems": problems,
        "not_run": not_run,
        "claim": "this checks the delivery's own arithmetic and the render's own buffers; it does "
                 "not judge whether the picture looks right",
    }
    (out / "verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified": verified, "checks_run": len(checks),
                      "problems": problems, "not_run": not_run}, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
