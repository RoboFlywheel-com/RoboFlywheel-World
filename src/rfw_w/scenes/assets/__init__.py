# SPDX-License-Identifier: Apache-2.0
"""Lazy asset capabilities, provider adapters and prepared resources.

Core preflight can import this package without loading models or simulators.
"""

from __future__ import annotations

from rfw_w.scenes.assets.capabilities import (
    CapabilityEntry,
    CapabilityError,
    CapabilityGap,
    CapabilityRegistry,
    CapabilityReport,
    Subject,
    SupportLevel,
)

__all__ = [
    "CapabilityEntry",
    "CapabilityError",
    "CapabilityGap",
    "CapabilityRegistry",
    "CapabilityReport",
    "Subject",
    "SupportLevel",
]
