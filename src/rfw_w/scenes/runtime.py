# SPDX-License-Identifier: Apache-2.0
"""Runtime configuration and explicit bundled example loading."""

from __future__ import annotations
import json
import os
from pathlib import Path
from rfw_w.scenes.contracts.scene_spec import SceneSpec

COMPILE_TIMEOUT_S = 600.0
DEFAULT_EXPERIMENT = 'office_pick_cup'
OUTPUTS_DIR = Path('outputs')

def load_env_file(path: Path) -> None:
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def step(name: str, detail: str = "") -> None:
    print(f"[stage] {name}{(' - ' + detail) if detail else ''}", flush=True)


def vlm_config(model: str):
    """The one place the critic's and the replanner's endpoint is chosen.

    It is a function rather than two inline constructions because both call sites had their own
    copy, and a copy is how a comparison run ends up with a critic on one endpoint and a replanner
    on another -- an A/B that silently measures two different things.

    The caller explicitly configures its provider and the environment variable holding its key.
    """
    from rfw_w.scenes.agents.llm_client import LLMConfig

    return LLMConfig(
        base_url=os.environ.get("PVSCENE_VLM_BASE_URL", ""),
        model=model,
        api_key_env=os.environ.get("PVSCENE_VLM_API_KEY_ENV", "OPENAI_API_KEY"),
        wire=os.environ.get("PVSCENE_VLM_WIRE", "chat_completions"),
        timeout_s=float(os.environ.get("PVSCENE_VLM_TIMEOUT_S", "240")),
    )


def _cache_file_token(path: Path) -> str:
    """Identify configured retrieval data without reading or exposing its contents."""
    stat = path.stat()
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


def experiment_path(experiment_id: str, *, root: Path | None = None) -> Path:
    """Where an experiment's spec lives. One layout, so a run and a test agree on it."""
    if not experiment_id or '/' in experiment_id or '\\' in experiment_id or experiment_id in ('.', '..'):
        raise ValueError('invalid example identity')
    if root is not None:
        return root / experiment_id / 'task.json'
    from rfw_w.scenes.resources import data_path
    return data_path(f'examples/{experiment_id}/task.json')


def load_spec(experiment_id: str = DEFAULT_EXPERIMENT, *, root: Path | None = None) -> SceneSpec:
    path = experiment_path(experiment_id, root=root)
    if not path.is_file():
        raise ValueError(f"no bundled example {experiment_id!r}; supply --spec")
    return SceneSpec.from_dict(json.loads(path.read_text()))
