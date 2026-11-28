"""Assemble reviewed candidate batches; never copy unselected models into the site."""

import argparse
import json
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", type=Path, nargs="+")
    parser.add_argument("--exclude", required=True, type=Path,
                        help="Private JSON mapping visually rejected display names to reasons")
    parser.add_argument("--featured", type=Path,
                        help="Ordered JSON list of reviewed candidates with name and public title")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--film", type=Path, help="Full project demo for older batches")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Choose a fresh external output directory")
    excluded = json.loads(args.exclude.read_text())
    groups = defaultdict(deque)
    qa = {"accepted": {}, "geometry_rejected": {}, "visually_rejected": excluded}
    out.mkdir(parents=True, exist_ok=False)
    prefix = "window.RFW_CATALOG = "
    for index, folder in enumerate(args.batches):
        catalog = json.loads((folder / "catalog.js").read_text().removeprefix(prefix).strip().removesuffix(";"))
        report = json.loads((folder / "review.json").read_text())
        qa["geometry_rejected"].update(report["rejected"])
        if index == 0:
            scenes = catalog["scenes"]
            for filename in ["viewer.js", "viewer-license.txt"]:
                shutil.copyfile(folder / filename, out / filename)
            shutil.copyfile(args.film or folder / "film.mp4", out / "film.mp4")
            for scene in scenes:
                for suffix in [".webp", ".mp4"]:
                    name = scene["id"] + suffix
                    shutil.copyfile(folder / name, out / name)
        for asset in catalog["assets"]:
            name = asset["name"]
            if name in excluded:
                continue
            if name in qa["accepted"] or name not in report["accepted"]:
                raise ValueError("Missing or duplicate asset review")
            for mode in ["visual", "collision"]:
                if not re.fullmatch("[a-f0-9]{32}", asset[mode]):
                    raise ValueError("Expected anonymous identifier")
                for suffix in [".webp", ".glb"]:
                    file = asset[mode] + suffix
                    shutil.copyfile(folder / file, out / file)
            qa["accepted"][name] = report["accepted"][name]
            groups[asset["family"]].append(asset)
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.popleft())
    if not ordered:
        raise ValueError("Empty final selection")
    if args.featured:
        by_name = {a["name"]: a for a in ordered}
        first = []
        entries = json.loads(args.featured.read_text())
        for entry in entries:
            # Only promote candidates already admitted by both review stages.
            item = by_name.pop(entry["name"])
            first.append(dict(item, name=entry["title"]))
        ordered = first + list(by_name.values())
        if len({a["name"] for a in ordered}) != len(ordered):
            raise ValueError("Duplicate public title")
        qa["featured"] = entries
    (out / "catalog.js").write_text(prefix + json.dumps(dict(assets=ordered, scenes=scenes)) + ";\n")
    (out / "review.json").write_text(json.dumps(qa, indent=2))
    print(f"{len(ordered)} assets, {len(groups)} families")


if __name__ == "__main__":
    main()
