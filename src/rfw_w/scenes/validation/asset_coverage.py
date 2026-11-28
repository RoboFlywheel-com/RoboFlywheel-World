#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit what the real asset libraries actually cover.

Produces outputs/_probe/m0_asset_coverage.json: per-category declared and evidenced counts, plus a
resolution of the categories a task requires against the providers that can serve them.

Nothing is downloaded and nothing is substituted. A category that is missing is reported
missing -- the acceptance manual is explicit that a block or a generated stand-in must never
fill a retrieval gap, and the only honest alternative is naming the gap.

    python -m rfw_w.scenes.validation.asset_coverage --env-file .env.local --output-dir outputs/_probe/m0_asset_coverage
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import sys
import time
from pathlib import Path


from rfw_w.scenes.assets.coverage import (
    CoverageError,
    FilesystemCheck,
    count_filesystem_assets,
    CoverageReport,
    ProviderCoverage,
    audit_hssd,
    audit_simready,
    resolve_requirements,
)

from rfw_w.scenes.validation.resources import load_env_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requirements", type=Path,
                        default=data_path('configs') / 'required_categories_v1.json')
    parser.add_argument("--verify-filesystem", action="store_true",
                        help="also count asset directories on disk and report the delta against "
                             "the index; slow (minutes) on a network mount")
    args = parser.parse_args()

    config = load_env_file(args.env_file) if args.env_file and args.env_file.is_file() else {}

    def configured(name: str) -> str | None:
        return config.get(name)

    providers: dict[str, ProviderCoverage] = {}
    absent: list[str] = []
    notes: list[str] = []
    started = time.monotonic()

    simready_index = configured("PVSCENE_SIMREADY_INDEX")
    if simready_index and Path(simready_index).is_file():
        # The library version is read from the path when it names one, so a report from v0.2
        # cannot be mistaken for a report on v0.4.
        version = next((part for part in Path(simready_index).parts if part.startswith("v")), "unknown")
        providers["simready"] = audit_simready(
            Path(simready_index), library_version=version,
            progress=lambda count: print(f"  ... {count} assets", file=sys.stderr),
        )
    else:
        absent.append("simready")
        notes.append(
            "simready catalogues index not configured or missing (PVSCENE_SIMREADY_INDEX); "
            "reported absent rather than skipped silently"
        )

    hssd_synset = configured("PVSCENE_HSSD_SYNSET_INDEX")
    if not hssd_synset and configured("PVSCENE_HSSD_ROOT"):
        hssd_synset = str(Path(configured("PVSCENE_HSSD_ROOT")).parent / "preprocessed" / "hssd_wnsynsetkey_index.json")
    if hssd_synset and Path(hssd_synset).is_file():
        providers["hssd"] = audit_hssd(Path(hssd_synset), library_version="hssd-models")
    else:
        absent.append("hssd")
        notes.append("HSSD synset index not configured or missing; reported absent")

    # The non-rigid libraries are named explicitly because their absence is a finding: the
    # architecture treats them as future providers, and "no local catalogue exists" is the
    # specific gap, not "the task was not attempted".
    for name, var in (("articraft", "PVSCENE_ARTICRAFT_CATALOG"), ("softsim", "PVSCENE_SOFT_SIM_CATALOG")):
        if not configured(var):
            absent.append(name)
            notes.append(
                f"{name}: no local catalogue exists and {var} is unset; articulated and deformable "
                "capabilities are therefore reported unsupported rather than untested"
            )

    requirements_payload = json.loads(args.requirements.read_text())
    resolutions = resolve_requirements(providers, requirements_payload.get("requirements", []))

    report = CoverageReport(providers=providers, resolutions=resolutions, absent_catalogues=tuple(sorted(absent)))
    payload = report.to_dict()
    payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Notes are finalised after the optional filesystem scan, and the elapsed figure is recorded
    # last: taking it first made the artifact report 4.5 s for a run that took over seven minutes,
    # which is an understated cost in the one field a reader uses to judge whether to re-run it.
    payload["notes"] = notes

    # The index is a summary and the export tree is the library; a count from the index alone says
    # nothing about directories it does not mention, and those assets are unreachable by retrieval.
    if args.verify_filesystem:
        assets_root = configured("PVSCENE_SIMREADY_ASSETS_ROOT")
        if not assets_root:
            notes.append("filesystem verification skipped: PVSCENE_SIMREADY_ASSETS_ROOT is unset")
            payload["notes"] = notes
        else:
            print("  counting asset directories on disk (this takes minutes)", file=sys.stderr)
            directories, domains, elapsed = count_filesystem_assets(Path(assets_root))
            index_entries = providers.get("simready").asset_count if "simready" in providers else 0
            check = FilesystemCheck(
                asset_directories=directories, domains=domains, index_entries=index_entries,
                elapsed_s=elapsed,
            )
            payload["filesystem_check"] = check.to_dict()
            if not check.consistent:
                notes.append(
                    f"the catalogue index and the export tree disagree by {check.delta} director(ies); "
                    "the index count alone is not the size of the searchable library"
                )
                payload["notes"] = notes

    payload["elapsed_seconds"] = round(time.monotonic() - started, 3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "asset_coverage.json"
    if target.exists():
        print(f"refusing to overwrite an existing report: {target}", file=sys.stderr)
        return 2
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "report": str(target),
        "providers": {name: {"assets": c.asset_count, "categories": c.category_count} for name, c in providers.items()},
        "absent_catalogues": list(report.absent_catalogues),
        "unsatisfied_requirements": list(report.unsatisfied_requirements),
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
