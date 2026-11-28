# SPDX-License-Identifier: Apache-2.0
"""Run installed workers in their configured interpreter without shell evaluation."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerJob:
    python: str
    module: str
    arguments: tuple[str, ...] = ()
    timeout_s: float | None = None

    def __post_init__(self):
        if not self.python or not self.module or any(c in self.module for c in "/\\ \n"):
            raise ValueError("worker requires an interpreter and an importable module name")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("worker timeout must be positive")


def clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def run(job: WorkerJob) -> int:
    """Stream worker output and preserve its exit status; timeout is status 124."""
    try:
        result = subprocess.run(
            [job.python, "-m", job.module, *job.arguments],
            env=clean_environment(),
            timeout=job.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124
    return result.returncode


def installed_packages(python: str) -> dict[str, bool]:
    """Probe only package availability, without loading models or reading credentials."""
    import json

    source = (
        "import importlib.util,json; "
        "print(json.dumps({n:importlib.util.find_spec(n) is not None "
        "for n in ('rfw_w','rfw_w.assets','rfw_w.scenes')}))"
    )
    result = subprocess.run(
        [python, "-c", source], env=clean_environment(),
        capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode:
        raise RuntimeError("configured interpreter cannot run the package probe")
    return json.loads(result.stdout)
