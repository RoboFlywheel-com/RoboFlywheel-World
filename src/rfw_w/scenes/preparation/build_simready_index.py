#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a v1.3 SimReady catalogue index from the release's own master table.

    python -m rfw_w.scenes.preparation.build_simready_index \
        --release-root /path/to/release/v1.3 \
        --out /path/to/simready_lib/index_v1.3.json --workers 48

The v0.2 index this project has been loading was built from the export's own directory walk. v1.3
publishes a different layout *and* a different naming scheme, so a walk would produce an index whose
entries resolve to nothing -- or, worse, to somebody else's asset: measured against the release's
`release_mapping.csv`, **99.4 % of leaves were renumbered** (521,310 of 524,611), and v1.3's
`0_0_0056` is the asset the old library called `0_0_0063`. So the paths come from the release's
`tables/master_table.sqlite`, which is the release's own authority on where each asset went.

What one entry looks like, and why:

* the key is the release's own ``<leaf>``, and ``assets_prefix`` is the asset's directory. The native
  catalogue and this project's adapter resolve a candidate as ``<assets_root>/<assets_prefix>/mesh/
  <asset_id>.obj``, which is then the mesh the *source* shipped, in its own frame -- the family this
  project loads every SimReady asset from. The release's normalised family (``m_<leaf>.obj``,
  ``m_<leaf>_collision.obj``) is deliberately not what the entry points at: its per-asset up-axis
  decision is not consistently upright, so the loader prepares its own pair from the source mesh
  instead. ``source_object_id`` is carried alongside so the entry still names the asset the release
  inherited it from.
* ``assets_prefix`` is ``<new_d>/<new_c>/<leaf>``, the asset's own directory under ``assets/``.
* ``dimensions_m`` is the visual extent the release recorded in ``up_axis_fix.vis_ext``, which is
  the extent of the *source* mesh (verified against both files). The catalogue uses it for its coarse
  size gate; the authoritative measurement is taken from the prepared mesh at retrieval time.
* ``mass``, ``mu1`` and ``mu2`` are read per asset from ``part_physics.json``, which the release's
  README names as the single authority for them. The master table carries no mass at all (every one
  of its 524,570 `total_mass` values is 0.0), so this is the only honest source.

A row whose files are missing, unreadable or without a mass is skipped *by name* and counted: the
counts go to stderr and into the index's ``_meta`` entry, because "the index has fewer assets than
the release" is a fact a reader needs rather than a rounding error.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

META_KEY = "_meta"
DEFAULT_TABLE = "tables/master_table.sqlite"


class BuildError(RuntimeError):
    """The index could not be built."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--sidecars-root", type=Path, default=None,
        help="a local mirror of the release holding the part_physics.json files, if one has been "
             "fetched; defaults to --release-root",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N rows, for a smoke run")
    parser.add_argument(
        "--only-domains", default="",
        help="comma-separated domain ids to build, for a smoke run against one slice of the "
             "release. The result is a partial index and says so in _meta: retrieval can only "
             "recall classes it can see, so a partial index measures the pipeline, not the library",
    )
    parser.add_argument(
        "--library-version", default="v1.3",
        help="the version string written into _meta and reported by the run",
    )
    return parser.parse_args(argv)


def rows(connection: sqlite3.Connection, limit: int, domains: Sequence[str] = ()) -> Iterable[tuple]:
    query = (
        "select new_leaf, asset_dir, new_d_name, new_c_name, old_object_id, material_guess, "
        "real_bbox, v13_status from assets"
    )
    if domains:
        quoted = ", ".join("?" for _ in domains)
        query = f"{query} where new_d in ({quoted})"
    query = f"{query} order by new_leaf"
    if limit:
        query = f"{query} limit {int(limit)}"
    return connection.execute(query, tuple(domains))


def read_physics(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def build_entry(asset_dir: str, row: tuple, sidecars_root: Path) -> tuple[str, dict[str, Any] | None, str]:
    """One catalogue entry, or a named reason it could not be built.

    ``asset_dir`` arrives separately because the export's ``_SUCCESS`` marker is the release's own
    statement that an asset finished, and an entry built from a half-written directory would resolve
    to paths that exist and geometry that is not the asset. It is the release's own relative path
    (``assets/<d>/<c>/<leaf>``), which is also what the entry's ``assets_prefix`` has to be.

    ``sidecars_root`` is where the asset directories are read from -- the release itself, or a
    pre-fetched copy of its ``part_physics.json`` and ``_SUCCESS`` files. Fetching them over the
    bucket runs at tens of files per second and that read is the whole cost of building this index,
    so being able to point it at a local mirror is the difference between twenty minutes and three
    hours. The mirror has to include ``_SUCCESS``: the two checks below are the release's own
    statement that an asset finished, and a builder that stopped making them because the files
    happened to be somewhere else would be a different builder.
    """
    leaf, _, domain, semantic, old_object_id, material, real_bbox, v13_status = row
    if v13_status != "ok":
        return leaf, None, f"v13_status={v13_status}"
    directory = sidecars_root / asset_dir
    if not (directory / "_SUCCESS").is_file():
        return leaf, None, "no _SUCCESS"
    physics = read_physics(directory / "part_physics.json")
    if physics is None:
        return leaf, None, "part_physics.json unreadable"
    mass = physics.get("total_mass")
    if mass is None or float(mass) <= 0:
        return leaf, None, "no positive mass"
    up_fix = physics.get("up_axis_fix") or {}
    extents = up_fix.get("vis_ext")
    if not extents or len(extents) != 3:
        return leaf, None, "no up_axis_fix.vis_ext"
    declared = extents
    if real_bbox:
        try:
            parsed = json.loads(real_bbox)
            if isinstance(parsed, list) and len(parsed) == 3:
                declared = parsed
        except ValueError:
            pass
    entry = {
        "assets_prefix": asset_dir.removeprefix("assets/"),
        "leaf": leaf,
        "source_object_id": old_object_id,
        "domain": str(physics.get("domain") or domain or ""),
        "super_category": str(physics.get("super_category") or semantic or ""),
        "physics_type": "rigid",
        # The release's own declared visual extent, in the asset's frame (`up_axis_fix.vis_ext` is
        # what its normalisation was computed from; measured against the meshes, it matches the
        # un-normalised one). The size gate reads it, and the world-frame extents that decide
        # admission are measured from the collision mesh at retrieval time -- the same division of
        # labour the v0.2 index had, whose `real_bbox` was in the asset frame for the same reason.
        "dimensions_m": [round(float(value), 6) for value in extents],
        "real_bbox": [round(float(value), 6) for value in declared],
        "max_dim_m": round(max(float(value) for value in extents), 6),
        "mass": round(float(mass), 6),
        "mu1": physics.get("whole_mu1"),
        "mu2": physics.get("whole_mu2"),
        "material": str(material or ""),
        "mass_check": str(physics.get("mass_check") or ""),
        "n_collision_hulls": physics.get("n_collision_hulls"),
        "up_axis_fix": {
            "axis_map": up_fix.get("axis_map"),
            "ambiguous": up_fix.get("ambiguous"),
        },
    }
    return leaf, entry, ""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.release_root.expanduser().resolve()
    table = (args.table or (root / DEFAULT_TABLE)).expanduser().resolve()
    if not table.is_file():
        raise BuildError(f"no master table at {table}")
    if not (root / "assets").is_dir():
        raise BuildError(f"no assets directory under {root}")

    connection = sqlite3.connect(f"file:{table}?mode=ro&immutable=1", uri=True)
    domains = tuple(d.strip() for d in args.only_domains.split(",") if d.strip())
    sidecars = (args.sidecars_root or root).expanduser().resolve()
    started = time.time()
    index: dict[str, Any] = {}
    skipped: dict[str, int] = {}
    total = 0

    def work(row: tuple) -> tuple[str, dict[str, Any] | None, str]:
        return build_entry(row[1], row, sidecars)

    with futures.ThreadPoolExecutor(args.workers) as pool:
        for leaf, entry, reason in pool.map(work, rows(connection, args.limit, domains), chunksize=64):
            total += 1
            if entry is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            entry["library_version"] = args.library_version
            index[str(leaf)] = entry
            if total % 25000 == 0:
                print(
                    f"  {total} rows -> {len(index)} entries "
                    f"({time.time() - started:.0f}s)",
                    file=sys.stderr, flush=True,
                )

    index[META_KEY] = {
        "library_version": args.library_version,
        "release_root": str(root),
        "sidecars_root": str(sidecars),
        "table": str(table),
        "rows": total,
        "only_domains": list(domains),
        "entries": len(index) - 1,
        "skipped": dict(sorted(skipped.items())),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder": "python -m rfw_w.scenes.preparation.build_simready_index",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index, sort_keys=True))
    print(
        f"rows={total} entries={len(index) - 1} skipped={sum(skipped.values())} "
        f"({time.time() - started:.0f}s) -> {args.out}",
        file=sys.stderr,
    )
    for reason, count in sorted(skipped.items()):
        print(f"  skipped {count:>6} : {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
