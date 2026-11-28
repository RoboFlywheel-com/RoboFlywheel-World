# SPDX-License-Identifier: Apache-2.0
"""Public RFW-W asset commands."""

import argparse, json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="rfw-w-asset")
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("generate", "batch"):
        c = sub.add_parser(cmd)
        c.add_argument("--request", required=True)
        c.add_argument("--config", required=True)
        c.add_argument("--out", required=True)
    sub.add_parser("catalog").add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.command == "catalog":
        from rfw_w.assets.catalog.server.app import main as serve

        serve()
        return 0
    from .runtime import Runtime

    runtime = Runtime.from_file(a.config)
    if a.command == "generate":
        from .application import generate

        report = generate(json.loads(Path(a.request).read_text()), runtime, a.out)
    else:
        from .batch import generate_batch

        data = [
            json.loads(line)
            for line in Path(a.request).read_text().splitlines()
            if line.strip()
        ]
        report = generate_batch(data, runtime, a.out)
    print(json.dumps(report, ensure_ascii=False))
    return (
        1
        if isinstance(report, list) and any(r["status"] == "failed" for r in report)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
