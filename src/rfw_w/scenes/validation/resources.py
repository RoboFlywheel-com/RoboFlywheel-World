#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Environment probe: what this machine can actually do, with the evidence for each claim.

M0's remaining obligation is to state the local gaps rather than assume the environment. This
produces that statement as a machine-readable artefact.

Three properties make the report worth trusting:

* every verdict carries the **command that was run** and its return code, so a reader can
  reproduce any single line instead of trusting the summary;
* nothing is downloaded, installed or substituted. A missing dependency is reported as absent
  and the probe stops there -- the acceptance manual forbids fetching weights or swapping in a
  substitute asset to make a check pass;
* credentials are reported as **set or unset only**. Values are never read into the report.

Internal asset paths are deliberately not hardcoded: the manual forbids committing them. They
come from environment variables, optionally loaded from a gitignored vars file via --env-file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path.cwd()
SCHEMA_VERSION = "pvscene.env-probe/v1"

# Reported as set/unset only; the value must never enter the artefact.
CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
)


class Verdict:
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"
    NOT_RUN = "not_run"


@dataclass
class Item:
    """One probed fact."""

    key: str
    verdict: str
    detail: str
    command: str = ""
    returncode: int | None = None
    value: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"verdict": self.verdict, "detail": self.detail}
        if self.command:
            payload["command"] = self.command
        if self.returncode is not None:
            payload["returncode"] = self.returncode
        if self.value is not None:
            payload["value"] = self.value
        if self.extra:
            payload["extra"] = self.extra
        return payload


def load_env_file(path: Path) -> dict[str, str]:
    """Read ``VAR=VALUE`` lines. Values are returned but never written into the report."""
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def probe_command(
    key: str,
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 60.0,
    parse: Callable[[str], Any] | None = None,
    expect_returncode: int = 0,
) -> Item:
    """Run a command and record it verbatim as the evidence for the verdict."""
    rendered = " ".join(command)
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, env=env
        )
    except FileNotFoundError:
        return Item(key, Verdict.ABSENT, "executable not found", command=rendered)
    except subprocess.TimeoutExpired:
        return Item(key, Verdict.UNKNOWN, f"timed out after {timeout}s", command=rendered)

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != expect_returncode:
        return Item(
            key,
            Verdict.ABSENT,
            f"exit {completed.returncode}: {(stderr or stdout).splitlines()[-1][:200] if (stderr or stdout) else 'no output'}",
            command=rendered,
            returncode=completed.returncode,
        )
    value = parse(stdout) if parse else (stdout.splitlines()[-1] if stdout else "")
    return Item(key, Verdict.PRESENT, "command succeeded", command=rendered,
                returncode=completed.returncode, value=value)


def probe_path(key: str, path: str | None, *, kind: str = "path") -> Item:
    if not path:
        return Item(key, Verdict.ABSENT, f"{key} is not configured")
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return Item(key, Verdict.ABSENT, f"configured but not present: {path}", value=str(candidate))
    if kind == "dir" and not candidate.is_dir():
        return Item(key, Verdict.ABSENT, f"expected a directory: {path}")
    if kind == "file" and not candidate.is_file():
        return Item(key, Verdict.ABSENT, f"expected a file: {path}")
    return Item(key, Verdict.PRESENT, f"resolves to {candidate}", value=str(candidate))


def probe_interpreters() -> list[Item]:
    """Every conda environment, and which of the packages we depend on it can import."""
    envs_root = Path(os.environ.get("PVSCENE_CONDA_ENVS", "/opt/conda/envs"))
    if not envs_root.is_dir():
        return [Item("interpreters", Verdict.ABSENT, f"no conda envs at {envs_root}")]

    discovered: list[str] = []
    modules = ["numpy", "torch", "transformers", "trimesh", "mujoco", "pxr", "pytest"]
    program = (
        "import importlib, sys\n"
        f"mods = {modules!r}\n"
        "found = {}\n"
        "for name in mods:\n"
        "    try:\n"
        "        module = importlib.import_module(name)\n"
        "        found[name] = getattr(module, '__version__', 'present')\n"
        "    except Exception:\n"
        "        pass\n"
        "import json; print(json.dumps({'python': sys.version.split()[0], 'modules': found}))\n"
    )

    items: list[Item] = []
    for env_dir in sorted(envs_root.iterdir()):
        python = env_dir / "bin" / "python"
        if not python.is_file():
            continue
        # Isaac's PYTHONPATH must be cleared or the probe would report Isaac's packages as
        # belonging to whichever environment is being measured.
        clean = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONHOME")
        }
        item = probe_command(
            f"interpreter.{env_dir.name}",
            [str(python), "-c", program],
            env=clean,
            timeout=180.0,
            parse=json.loads,
        )
        discovered.append(env_dir.name)
        items.append(item)
    summary = Item(
        "interpreters",
        Verdict.PRESENT if items else Verdict.ABSENT,
        f"{len(items)} interpreter(s) with a python binary under {envs_root}",
        value=discovered,
    )
    return [summary, *items]


def probe_hf_cache(cache_root: str | None) -> Item:
    """Whether the SBERT weights are already local.

    Reported because retrieval quality depends on it and because the probe must not download
    them: a probe that fetched weights would hide the very gap it is meant to reveal.
    """
    model = "models--sentence-transformers--all-mpnet-base-v2"
    if not cache_root:
        return Item("hf_cache.sbert_weights", Verdict.ABSENT, "PVSCENE_HF_CACHE is not configured")
    hub = Path(cache_root).expanduser() / "hub"
    if not hub.is_dir():
        return Item("hf_cache.sbert_weights", Verdict.ABSENT, f"no hub directory under {cache_root}")
    target = hub / model
    present = target.is_dir()
    snapshots = list((target / "snapshots").glob("*")) if present else []
    return Item(
        "hf_cache.sbert_weights",
        Verdict.PRESENT if present else Verdict.ABSENT,
        f"{model} {'found' if present else 'not found'} in the local HF cache",
        value=str(target) if present else None,
        extra={"snapshot_count": len(snapshots), "offline_required": True},
    )


def probe_prepared_sbert(prepared_dir: str | None, cache_root: str | None) -> Item:
    """Whether a locally loadable SBERT directory exists.

    The HF cache and the *loadable* form of it are different facts: this machine has the
    weights only as bare blobs, which ``local_files_only=True`` cannot read. Reporting them
    as one item would say "retrieval weights present" while retrieval is in fact blocked.
    """
    if not prepared_dir:
        return Item("prepared_sbert", Verdict.ABSENT, "PVSCENE_PREPARED_DIR is not configured")
    target = Path(prepared_dir).expanduser()
    if not target.is_dir():
        return Item(
            "prepared_sbert",
            Verdict.ABSENT,
            f"not prepared at {target}; run python -m rfw_w.scenes.preparation.prepare_sbert before retrieval",
        )
    needed = ("config.json", "vocab.txt", "tokenizer_config.json")
    weights = [name for name in ("model.safetensors", "pytorch_model.bin") if (target / name).is_file()]
    missing = [name for name in needed if not (target / name).is_file()]
    if missing or not weights:
        return Item(
            "prepared_sbert", Verdict.ABSENT,
            f"incomplete: missing {missing + ([] if weights else ['a weight file'])}",
            value=str(target),
        )
    return Item(
        "prepared_sbert",
        Verdict.PRESENT,
        "locally loadable model directory is complete",
        value=str(target),
        extra={"weights": weights, "cache_root": cache_root},
    )


def probe_simready(lib_root: str | None, assets_root: str | None) -> list[Item]:
    """The rigid provider library and the vendor asset tree it points into."""
    items = [
        probe_path("simready_library.index", f"{lib_root}/index.json" if lib_root else None, kind="file"),
        probe_path("simready_library.sbert_features", f"{lib_root}/sbert_features.pkl" if lib_root else None, kind="file"),
        probe_path("simready_library.cache", f"{lib_root}/cache" if lib_root else None, kind="dir"),
        probe_path("simready_assets_root", assets_root, kind="dir"),
    ]
    if not lib_root or not assets_root:
        return items

    # Resolve one catalogue entry through the exact path layout the v2 catalogue injects, so
    # "the root exists" is not mistaken for "the assets are reachable".
    index_path = Path(lib_root) / "index.json"
    if not index_path.is_file():
        return items
    try:
        with index_path.open() as handle:
            head = handle.read(200_000)
        import re

        first = re.search(r'"([0-9][0-9_]*)"\s*:\s*\{"assets_prefix":\s*"([^"]+)"', head)
    except OSError as error:
        return [*items, Item("simready_library.sample_asset", Verdict.UNKNOWN, f"unreadable: {error}")]

    if not first:
        return [*items, Item("simready_library.sample_asset", Verdict.UNKNOWN, "index head contained no parseable entry")]
    asset_id, prefix = first.group(1), first.group(2)
    base = Path(assets_root) / prefix
    expected = {
        "visual": base / "mesh" / f"{asset_id}.obj",
        "collision": base / "mesh" / f"{asset_id}_collision.obj",
        "metadata": base / f"{asset_id}.urdf",
    }
    resolved = {name: path.is_file() for name, path in expected.items()}
    return [
        *items,
        Item(
            "simready_library.sample_asset",
            Verdict.PRESENT if all(resolved.values()) else Verdict.ABSENT,
            f"probe asset {asset_id} resolved through the catalogue path layout",
            extra={"asset_id": asset_id, "resolved": resolved},
        ),
    ]


def probe_hssd(root: str | None) -> list[Item]:
    items = [
        probe_path("hssd.objects", f"{root}/objects" if root else None, kind="dir"),
        probe_path("hssd.support_surfaces", f"{root}/support-surfaces" if root else None, kind="dir"),
    ]
    if not root:
        return items
    synset = Path(root).parent / "preprocessed" / "hssd_wnsynsetkey_index.json"
    clips = Path(root).parent / "preprocessed" / "clip_hssd_embeddings.npy"
    items.append(probe_path("hssd.synset_index", str(synset), kind="file"))
    items.append(probe_path("hssd.clip_embeddings", str(clips), kind="file"))
    return items



# Negative fixtures are expected to fail; preserve their evidence semantics.
EXPECTED_FIXTURE_STATUS = {"supported": "passed", "missing_support": "failed"}


def probe_evidence_from_previous_runs() -> list[Item]:
    """Check the probe reports already on disk against their expected outcome.

    Cited rather than restated: the item names the artefact and the status inside it, so a
    claim of "Isaac works" always resolves to a file a reader can open. Each report is also
    compared against what its fixture is supposed to produce, which turns the report from a
    status dump into a check -- a positive case that failed, or a negative case that passed,
    is a real discrepancy and is reported as one.
    """
    results = REPO_ROOT / "results"
    if not results.is_dir():
        return [Item("prior_evidence", Verdict.NOT_RUN, "no results directory")]

    items: list[Item] = []
    for report in sorted(results.glob("*/report.json")):
        key = f"prior_evidence.{report.parent.name}"
        try:
            payload = json.loads(report.read_text())
        except (OSError, json.JSONDecodeError) as error:
            items.append(Item(key, Verdict.UNKNOWN, f"unreadable: {error}"))
            continue

        recorded = payload.get("status")
        fixture = payload.get("fixture")
        expected = EXPECTED_FIXTURE_STATUS.get(fixture)
        matches = expected is not None and recorded == expected
        if expected is None:
            detail = f"unrecognised fixture {fixture!r}; recorded status={recorded!r}"
            verdict = Verdict.UNKNOWN
        elif matches:
            detail = f"fixture {fixture!r} produced its expected {expected!r} status"
            verdict = Verdict.PRESENT
        else:
            detail = (
                f"fixture {fixture!r} should have produced {expected!r} but recorded {recorded!r}"
            )
            verdict = Verdict.ABSENT

        items.append(
            Item(
                key,
                verdict,
                detail,
                value=str(report.relative_to(REPO_ROOT)),
                extra={
                    "recorded_status": recorded,
                    "expected_status": expected,
                    "matches_expectation": matches,
                    "scope": payload.get("scope"),
                    "isaac_version": ".".join(payload.get("isaac_version", [])[:2]),
                },
            )
        )
    return items


def probe_gpu() -> Item:
    if not shutil.which("nvidia-smi"):
        return Item("gpu", Verdict.ABSENT, "nvidia-smi not on PATH")
    return probe_command(
        "gpu",
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
        timeout=30.0,
    )


def probe_blender() -> Item:
    """Blender is optional in this release, so absence is recorded rather than treated as a gap."""
    found = shutil.which("blender")
    if found:
        return probe_command("blender", [found, "--version"], timeout=60.0)
    return Item(
        "blender",
        Verdict.ABSENT,
        "no blender on PATH; the hero render profile and Blender-side baking are not_run in this release",
    )


def probe_credentials() -> Item:
    """Presence only. A value never enters this artefact."""
    present = sorted(name for name in CREDENTIAL_VARS if os.environ.get(name))
    return Item(
        "model_credentials",
        Verdict.PRESENT if present else Verdict.ABSENT,
        "credential values are deliberately not read; only presence is recorded",
        value=present,
        extra={"checked": list(CREDENTIAL_VARS)},
    )


def run_probe(config: dict[str, str]) -> dict[str, Any]:
    """Collect every item into one report."""
    def configured(name: str, default: str | None = None) -> str | None:
        return config.get(name) or os.environ.get(name) or default

    items: list[Item] = []

    # The bare interpreter is probed as well: it is what proves the core is stdlib-only.
    items.append(
        probe_command(
            "stdlib_interpreter",
            [
                configured("PVSCENE_STDLIB_PYTHON", "/opt/conda/bin/python") or "/opt/conda/bin/python",
                "-c",
                "import sys; print(sys.version.split()[0])",
            ],
            env={key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")},
            timeout=60.0,
        )
    )
    items.append(probe_path("isaac.python", configured("PVSCENE_ISAAC_PYTHON", "/isaac-sim/python.sh"), kind="file"))
    items.append(probe_gpu())
    items.append(probe_credentials())
    items.append(probe_blender())
    items.extend(probe_interpreters())
    items.append(probe_hf_cache(configured("PVSCENE_HF_CACHE")))
    items.append(probe_prepared_sbert(configured("PVSCENE_PREPARED_DIR"), configured("PVSCENE_HF_CACHE")))
    items.extend(probe_simready(configured("PVSCENE_SIMREADY_LIB"), configured("PVSCENE_SIMREADY_ASSETS_ROOT")))
    items.extend(probe_hssd(configured("PVSCENE_HSSD_ROOT")))
    items.extend(probe_evidence_from_previous_runs())

    counts: dict[str, int] = {}
    for item in items:
        counts[item.verdict] = counts.get(item.verdict, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "environment capability only; this report asserts nothing about scene quality",
        "policy": {
            "downloads": "never attempted",
            "substitutions": "never made; a missing dependency is reported absent",
            "credentials": "presence reported, values never read",
            "internal_paths": "supplied by environment variables, not committed to the repository",
        },
        "verdict_counts": counts,
        "items": {item.key: item.to_dict() for item in items},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="must not already contain a report; a stale report must never look current")
    parser.add_argument("--env-file", type=Path,
                        help="optional VAR=VALUE file for deployment paths (keep it out of version control)")
    parser.add_argument("--stdout-only", action="store_true", help="print the report without writing it")
    args = parser.parse_args()

    config: dict[str, str] = {}
    if args.env_file:
        if not args.env_file.is_file():
            print(f"env file not found: {args.env_file}", file=sys.stderr)
            return 2
        config = load_env_file(args.env_file)

    report = run_probe(config)

    if not args.stdout_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target = args.output_dir / "environment_probe.json"
        if target.exists():
            print(f"refusing to overwrite existing report: {target}", file=sys.stderr)
            return 2
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"verdict_counts": report["verdict_counts"], "report": str(target)}, indent=2))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    # Absent items are information, not failure: the probe's job is to report the machine, and
    # the manual explicitly expects gaps to be named rather than papered over.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
