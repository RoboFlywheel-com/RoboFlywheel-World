# SPDX-License-Identifier: Apache-2.0
"""Provider adapters for the real asset libraries.

Neither adapter is imported by the core. Each one reaches for a different runtime -- v2 and
sentence-transformers for the rigid library, trimesh for the furniture -- so importing this package
pulls in nothing on its own.
"""
