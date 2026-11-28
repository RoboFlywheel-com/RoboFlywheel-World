# SPDX-License-Identifier: Apache-2.0
"""Explicit installed-process runtimes isolate model and simulator dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
import json, math, os, subprocess, time
from pathlib import Path


@dataclass
class Runtime:
    stages: dict = field(default_factory=dict)
    allow_remote_models: bool = False
    timeout_s: float = 3600

    def __post_init__(self):
        if (
            not isinstance(self.stages, dict)
            or type(self.allow_remote_models) is not bool
        ):
            raise ValueError(
                "runtime stages must be an object and model permission a boolean"
            )
        if (
            type(self.timeout_s) not in (float, int)
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("runtime timeout must be finite and positive")
        for name, cfg in self.stages.items():
            if not isinstance(name, str) or not isinstance(cfg, dict):
                raise ValueError("invalid stage runtime")
            if not isinstance(cfg.get("env", {}), dict) or not isinstance(
                cfg.get("options", {}), dict
            ):
                raise ValueError("stage env/options must be objects")
            reserved = {
                "manifest",
                "request",
                "out",
                "in-dir",
                "out-dir",
                "parts-dir",
                "renders-dir",
                "mesh",
                "visual-mesh",
                "face-ids",
                "annotations",
                "seed",
                "start",
                "end",
                "overwrite",
            }
            if reserved.intersection(cfg.get("options", {})):
                raise ValueError(
                    "stage options cannot override job inputs, outputs or seed"
                )
            for key in cfg.get("env", {}):
                if key in ("PYTHONPATH", "PYTHONHOME") or any(
                    token in key.upper()
                    for token in ("KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIAL")
                ):
                    raise ValueError(
                        "stage env cannot inject package paths or credential values"
                    )

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("invalid runtime configuration")
        return cls(
            data.get("stages", {}),
            data.get("allow_remote_models", False),
            data.get("timeout_s", 3600),
        )

    def options(self, stage):
        return dict(self.stages.get(stage, {}).get("options", {}))

    def environment(self, stage):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update(
            {str(k): str(v) for k, v in self.stages[stage].get("env", {}).items()}
        )
        return env

    def preflight(self, stages, formats=()):
        modules = {
            "image": "flux",
            "background": "background",
            "3d": "trellis",
            "segmentation": "segmentation",
            "render": "render",
            "annotation": "annotation",
            "physical": "physical",
        }
        for stage in stages:
            cfg = self.stages.get(stage)
            if not isinstance(cfg, dict) or not cfg.get("python"):
                raise ValueError("runtime stage requires explicit python: " + stage)
            py = Path(cfg["python"])
            if not py.is_file() or not os.access(py, os.X_OK):
                raise FileNotFoundError("runtime interpreter unavailable: " + stage)
            if stage == "annotation" and not self.allow_remote_models:
                raise ValueError("annotation model calls are disabled")
            options = cfg.get("options", {})
            required = {
                "image": ("ckpt",),
                "background": ("ckpt",),
                "3d": ("ckpt", "trellis-root"),
                "segmentation": ("ckpt", "p3sam-root"),
                "render": ("blender",),
            }.get(stage, ())
            for name in required:
                if not options.get(name):
                    raise ValueError(
                        "runtime stage requires resource: " + stage + ":" + name
                    )
            for name in ("ckpt", "trellis-root", "p3sam-root", "sonata-ckpt", "blender"):
                if name in options and not Path(str(options[name])).exists():
                    raise FileNotFoundError(
                        "runtime resource unavailable: " + stage + ":" + name
                    )
            if stage == "annotation":
                for name in ("endpoint", "model", "api-key-env"):
                    if not options.get(name):
                        raise ValueError("annotation requires " + name)
                if not os.environ.get(options["api-key-env"]):
                    raise ValueError("annotation credential is unset")
            dependencies = {
                "image": ("torch", "diffusers", "PIL"),
                "background": ("numpy", "torch", "torchvision", "transformers", "PIL"),
                "3d": ("torch", "PIL", "numpy"),
                "segmentation": ("torch", "numpy", "trimesh"),
                "render": (),
                "annotation": ("openai", "json_repair", "numpy", "trimesh", "PIL"),
                "physical": ("numpy", "scipy", "trimesh", "coacd", "psutil", "PIL"),
            }[stage]
            if stage == "physical" and "usd" in formats:
                dependencies = (*dependencies, "pxr")
            names = ("rfw_w.assets.workers." + modules[stage], *dependencies)
            code = "import importlib.util,json,sys; missing=[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; print(json.dumps(missing)); sys.exit(bool(missing))"
            result = subprocess.run(
                [str(py), "-c", code, *names],
                env=self.environment(stage),
                cwd="/tmp",
                capture_output=True,
                text=True,
                timeout=min(self.timeout_s, 60),
            )
            if result.returncode:
                try:
                    missing = json.loads(result.stdout)
                except (ValueError, TypeError):
                    missing = ["dependency probe failed"]
                raise RuntimeError(
                    "stage dependencies unavailable: "
                    + stage
                    + ": "
                    + ", ".join(missing)
                )

    def run(self, stage, module, args, out):
        cfg = self.stages[stage]
        command = [str(cfg["python"]), "-m", module, *map(str, args)]
        for k, v in cfg.get("options", {}).items():
            if isinstance(v, bool):
                if v:
                    command.append("--" + k)
            elif v is not None:
                command.extend(("--" + k, str(v)))
        log = Path(out) / ("stage-" + stage + ".log")
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=self.environment(stage),
                cwd=out,
                timeout=self.timeout_s,
            )
        if result.returncode:
            raise RuntimeError(
                "stage " + stage + " failed with exit " + str(result.returncode)
            )
        return {
            "status": "passed",
            "seconds": time.monotonic() - start,
            "log": log.name,
        }
