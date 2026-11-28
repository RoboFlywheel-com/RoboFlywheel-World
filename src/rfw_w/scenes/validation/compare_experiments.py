#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Put finished runs of different experiments side by side, against what each one asked for.

The question this answers is "did the room come out the way the description asked?". It reads
manifests rather than renders, so it says what was *built* -- which entities, which assets, which
room -- and prints the prompt each was built from next to them. Whether the result looks right is a
human judgement; this is the material for it.

    python -m rfw_w.scenes.validation.compare_experiments outputs/office_pick_cup/m1_010 \
        outputs/kitchen_fetch_jar/m1_011 outputs/bedroom_pick_alarm_clock/m1_012

Each run is summarised, then the runs are compared on the things that must differ between different
experiments and the things that must not. Two runs of *different* experiments sharing an identical
instance list would mean the spec is not driving the build -- which is exactly the defect this whole
area of work exists to remove, so it is checked rather than assumed.
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
from pathlib import Path

REPO_ROOT = Path.cwd()


def load(delivery: Path) -> dict:
    path = delivery / "delivery_manifest.json"
    if not path.is_file():
        raise SystemExit(f"no manifest at {path}")
    return json.loads(path.read_text())


def prompt_for(scene_id: str) -> str:
    """The description the spec carries, read from the experiment rather than the manifest.

    The manifest carries it too, but reading both lets a disagreement between them show up as a
    disagreement instead of being invisible.
    """
    for task in sorted((data_path('examples')).glob("*/task.json")):
        doc = json.loads(task.read_text())
        if doc.get("scene_id") == scene_id:
            return doc.get("prompt", "")
    return ""


def summarise(delivery: Path, manifest: dict) -> dict:
    instances = sorted(manifest.get("placement", {}).get("instances", {}))
    assets = {
        entry["entity_id"]: entry for entry in manifest.get("assets_used", ())
    }
    plan = manifest.get("plan", {})
    return {
        "delivery": str(delivery.relative_to(REPO_ROOT)) if REPO_ROOT in delivery.parents else str(delivery),
        "scene_id": manifest.get("scene_id", ""),
        "achieved": manifest.get("resolution", {}).get("achieved", ""),
        "meets_request": manifest.get("resolution", {}).get("meets_request"),
        "prompt": manifest.get("prompt", ""),
        "spec_prompt": prompt_for(manifest.get("scene_id", "")),
        "room": manifest.get("room", {}),
        "room_kind": manifest.get("zones", {}).get("room_kind", ""),
        "zones": [zone["kind"] for zone in manifest.get("zones", {}).get("zones", ())],
        "instances": instances,
        "assets": assets,
        "support": plan.get("support_id", ""),
        "target": plan.get("target_id", ""),
        "gaps": plan.get("derivation_gaps", []),
        "unplaced": manifest.get("furnishing", {}).get("unplaced", []) + [
            item for key in ("unplaced_core", "unplaced_task_closure", "unplaced_periphery", "periphery_dropped")
            for item in manifest.get("furnishing", {}).get(key, [])
        ],
        "size_rejections": manifest.get("furnishing", {}).get("size_bound_rejections", []),
        "scene_version": manifest.get("scene_version", ""),
    }


def render(rows: list[dict]) -> str:
    out: list[str] = []
    for row in rows:
        out.append("=" * 78)
        out.append(f"{row['delivery']}   scene_id={row['scene_id']}")
        out.append(f"  achieved={row['achieved']}  meets_request={row['meets_request']}")
        room = row["room"]
        out.append(
            f"  room: {room.get('area_m2')} m2, height {room.get('height_m')} m, "
            f"{room.get('vertices')} vertices, {room.get('openings')} opening(s)"
        )
        out.append(f"  THE ASK:  {row['prompt']}")
        if row["spec_prompt"] and row["spec_prompt"] != row["prompt"]:
            out.append(f"  !! the experiment file says something else: {row['spec_prompt']}")
        out.append(f"  support={row['support']!r}  target={row['target']!r}")
        out.append(f"  {len(row['instances'])} instances: {', '.join(row['instances'])}")
        for entity in row["instances"]:
            asset = row["assets"].get(entity)
            if asset is None:
                out.append(f"    {entity:14s} (no asset record)")
                continue
            dims = asset.get("dimensions_m", [0, 0, 0])
            out.append(
                f"    {entity:14s} {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} m"
                f"   {asset.get('asset_id', '')[:10]}  {asset.get('provider', '')}"
            )
        if row["gaps"]:
            out.append(f"  derivation gaps: {row['gaps']}")
        if row["unplaced"]:
            out.append(f"  UNPLACED: {row['unplaced']}")
        if row["size_rejections"]:
            per: dict[str, int] = {}
            for rejection in row["size_rejections"]:
                per[rejection["entity_id"]] = per.get(rejection["entity_id"], 0) + 1
            out.append(
                "  size-bound rejections: "
                + ", ".join(f"{name} {count}" for name, count in sorted(per.items()))
            )

    if len(rows) > 1:
        out.append("=" * 78)
        out.append("ACROSS THE RUNS")
        same = set(rows[0]["instances"])
        for row in rows[1:]:
            same &= set(row["instances"])
        out.append(f"  every run places: {sorted(same) or '(nothing in common)'}")
        for row in rows:
            out.append(
                f"  {row['scene_id']:26s} {len(row['instances'])} instances, "
                f"version {row['scene_version'][:12]}"
            )
        versions = {row["scene_version"] for row in rows}
        asset_sets = {frozenset(row["assets"]) for row in rows}
        out.append("")
        out.append(f"  scene versions differ: {len(versions) == len(rows)}")
        out.append(f"  entity sets differ:    {len(asset_sets) == len(rows)}")
        # Only a claim about DIFFERENT experiments. Two runs of one experiment are *supposed* to
        # place the same things -- that reproducibility is the point of a fresh run directory, and
        # warning about it would make the check noise that gets ignored.
        # Keyed on the experiment directory, not on `scene_id`: the id is a field of the spec and
        # changes when the spec does, so two runs of one experiment either side of such an edit
        # would look like two different experiments.
        distinct = {Path(row["delivery"]).parent for row in rows}
        if len(distinct) == len(rows) and len(asset_sets) < len(rows):
            out.append(
                "  !! two DIFFERENT experiments placed the same entity set, which means the spec "
                "is not driving the build"
            )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deliveries", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = [summarise(path.resolve(), load(path.resolve())) for path in args.deliveries]
    table = render(rows)
    print(table, flush=True)
    if args.out:
        args.out.write_text(table + "\n")
        print(f"\n[written] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
