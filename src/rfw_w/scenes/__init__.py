# SPDX-License-Identifier: Apache-2.0
"""RFW-W scene generation; under the unified rfw_w package.

This package is deliberately importable with nothing but the standard library.
Submodules are *not* imported here: importing a subpackage must never drag in a
heavy dependency, and the contract test ``tests/scenes/contract/test_no_v2_import.py``
asserts that no module pulls the former external ``pvscene`` package into ``sys.modules``.
"""

from __future__ import annotations

__all__ = ["__version__", "SCHEMA_NAMESPACE"]

__version__ = "0.3.0.dev0"

# Every JSON artefact this project writes carries a schema_version string of the
# form f"{SCHEMA_NAMESPACE}.<name>/v<N>". Artefacts are the interface between
# processes, so the namespace is centralised rather than repeated per module.
SCHEMA_NAMESPACE = "pvscene.v3"
