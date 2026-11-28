#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the configured Infinigen generator with local runtime fixes and constraints.

The generator is imported by its full module name to preserve gin registration.
Optional reservation/provenance arguments are removed before upstream parsing.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PATCH_PATH = Path(__file__).resolve().parent / "infinigen_constraint_patch.py"

#: Our arguments, and whether each takes a value.
OUR_ARGUMENTS = {"reservations": True, "provenance": True, "entry": True}


def split_our_arguments(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Take our arguments out and leave everything else exactly as it was, order included."""
    ours: dict[str, str] = {}
    rest = [argv[0]]
    index = 1
    while index < len(argv):
        item = argv[index]
        name = item[len("--pvscene-"):] if item.startswith("--pvscene-") else None
        if name is None or name not in OUR_ARGUMENTS:
            rest.append(item)
            index += 1
            continue
        takes_value = OUR_ARGUMENTS[name]
        ours[name] = argv[index + 1] if takes_value else "true"
        index += 2 if takes_value else 1
    return ours, rest


def load_patch():
    return importlib.import_module('rfw_w.scenes.workers.blender.infinigen_constraint_patch')


def main() -> int:
    ours, rest = split_our_arguments(list(sys.argv))
    sys.argv = rest
    # The registration the wrapper performs (`run_generate_indoors.py:21-30`): importing the module
    # with its full path is what avoids gin's "Ambiguous selector 'compose_indoors'" from __main__.
    from rfw_w.scenes.workers.blender.infinigen_runtime import (
        add_run_main_to_module,
        monkey_patch_infinigen,
    )

    monkey_patch_infinigen()

    import infinigen_examples.generate_indoors as gi

    patch = load_patch()
    if ours.get("reservations"):
        provenance = ours.get("provenance")
        if not provenance:
            raise SystemExit(
                "BACKGROUND_BUNDLE_INCOMPLETE(--pvscene-reservations needs "
                "--pvscene-provenance): a build asked for constraints must record whether it got "
                "them, or the record cannot say what ran")
        patch.install(Path(ours["reservations"]), Path(provenance),
                      entry_path=Path(ours.get("entry", __file__)).resolve())

    add_run_main_to_module(gi)
    gi._run_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
