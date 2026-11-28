# SPDX-License-Identifier: Apache-2.0
"""Public RFW-W commands dispatch into separately installed pipeline environments."""
from __future__ import annotations

import argparse
import json
import os
import sys

from rfw_w.runtime import WorkerJob, installed_packages, run


COMMANDS = {
    ("asset", "generate"): "rfw_w.assets.commands.generate",
    ("asset", "batch"): "rfw_w.assets.commands.batch",
    ("asset", "catalog"): "rfw_w.assets.commands.catalog",
    ("asset", "validate"): "rfw_w.assets.commands.validate",
    ("scene", "generate"): "rfw_w.scenes.commands.generate",
    ("scene", "plan"): "rfw_w.scenes.commands.plan_task_world",
    ("scene", "world"): "rfw_w.scenes.commands.build_task_world",
    ("scene", "retrieve"): "rfw_w.scenes.commands.retrieve_asset",
    ("scene", "validate"): "rfw_w.scenes.validation.verify_delivery",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate rigid assets and task scenes in their installed environments.",
        epilog="Example: rfw-w asset generate --request request.json --config runtime.json --out out",
    )
    parser.add_argument("--python", help="worker interpreter; defaults to RFW_ASSET_PYTHON/RFW_SCENE_PYTHON or current Python")
    parser.add_argument("--timeout", type=float, help="whole worker timeout in seconds")
    parser.add_argument("pipeline", choices=("asset", "scene", "doctor"))
    parser.add_argument("command", nargs="?", help="generate, batch, catalog, validate; scene also plan, world, retrieve")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    variable = "RFW_ASSET_PYTHON" if args.pipeline == "asset" else "RFW_SCENE_PYTHON"
    python = args.python or os.environ.get(variable) or sys.executable
    try:
        if args.pipeline == "doctor":
            if args.command or args.arguments:
                parser.error("doctor takes no worker arguments")
            packages = installed_packages(python)
            print(json.dumps({"packages": packages, "models": "not_run", "simulators": "not_run"}, indent=2))
            return 0 if all(packages.values()) else 1
        key = (args.pipeline, args.command)
        if key not in COMMANDS:
            parser.error(f"unknown command for {args.pipeline}: {args.command}")
        return run(WorkerJob(python, COMMANDS[key], tuple(args.arguments), args.timeout))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"RFW-W command failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
