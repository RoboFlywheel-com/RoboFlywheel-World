# SPDX-License-Identifier: Apache-2.0
"""Content hashes for scene state.

Every piece of evidence carries a ``state_hash``, and the readiness rule is "pass *and*
hash matches" (architecture §11). That makes the hash the mechanism that stops a stale
approval from being reused after the scene changed, so it has to be:

* **meaning-preserving** -- recomputing it from the delivered artefacts must give the same
  value, which is exactly what the independent delivery verifier does;
* **stable** -- key order and incidental float formatting must not change it, or two runs
  of the same scene would disagree;
* **strict** -- NaN, infinity and non-JSON types are rejected rather than hashed, because a
  hash over a NaN is not reproducible and would quietly defeat the whole scheme.

Floats are rounded to :data:`FLOAT_DECIMALS` decimal places before hashing, so ``1.0`` and
``0.9999999999999999`` agree while a genuine 1e-10 change to a coordinate still changes the
hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

FLOAT_DECIMALS = 12
ALGORITHM = "sha256"
DIGEST_LENGTH = 64


class HashError(ValueError):
    """A value could not be canonically hashed."""


@runtime_checkable
class Canonicalisable(Protocol):
    """Anything that can render itself as a plain JSON structure."""

    def to_dict(self) -> dict[str, Any]: ...


def _normalise_float(value: float) -> float:
    if not math.isfinite(value):
        raise HashError(f"cannot hash non-finite float {value!r}")
    # round() collapses representation noise; adding 0.0 turns -0.0 into 0.0 so that a
    # negative zero does not hash differently from a positive one.
    return round(value, FLOAT_DECIMALS) + 0.0


def canonicalize(value: Any, *, path: str = "$") -> Any:
    """Return a JSON-safe, order-normalised copy of ``value``.

    ``bool`` is checked before ``int`` because ``bool`` is a subclass of ``int`` in Python;
    without that ordering ``True`` would be silently hashed as ``1``.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        try:
            return _normalise_float(value)
        except HashError as error:
            # Re-raise with the location attached: in a nested scene payload, "there is a
            # NaN somewhere" is not actionable, and the offender is usually one coordinate
            # in one transform.
            raise HashError(f"{path}: {error}") from error
    if isinstance(value, Canonicalisable):
        return canonicalize(value.to_dict(), path=path)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HashError(f"{path}: mapping keys must be strings, got {type(key).__name__}")
            out[key] = canonicalize(item, path=f"{path}.{key}")
        return {key: out[key] for key in sorted(out)}
    if isinstance(value, (list, tuple)):
        # Lists keep their order: order is semantic for placements, relation lists and
        # candidate rankings. Sequences are the one place where order is not normalised.
        return [canonicalize(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise HashError(f"{path}: unsupported type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Deterministic JSON text: sorted keys, fixed separators, ASCII-escaped."""
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    """Hex digest of the canonical form of ``value``."""
    return hashlib.new(ALGORITHM, canonical_json(value).encode("utf-8")).hexdigest()


def hash_text(text: str) -> str:
    """Digest of raw text, for hashing a source file without parsing it."""
    return hashlib.new(ALGORITHM, text.encode("utf-8")).hexdigest()


def is_digest(value: str) -> bool:
    """True when ``value`` looks like a digest this module produced."""
    if not isinstance(value, str) or len(value) != DIGEST_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class SceneHashes:
    """The six hashes that identify a scene version (architecture §5.3).

    ``version`` is derived from the other five, so it changes whenever any input to the
    scene changes. Evidence stores the ``version``; a mismatch means the evidence belongs
    to a different scene and must not be reused.
    """

    spec_hash: str
    assets_hash: str
    geometry_hash: str
    physics_hash: str
    appearance_hash: str
    version: str

    def __post_init__(self) -> None:
        for name in ("spec_hash", "assets_hash", "geometry_hash", "physics_hash",
                     "appearance_hash", "version"):
            value = getattr(self, name)
            if not is_digest(value):
                raise HashError(f"{name} must be a {DIGEST_LENGTH}-character hex digest, got {value!r}")

    @classmethod
    def from_components(
        cls,
        *,
        spec: Any,
        assets: Any,
        geometry: Any,
        physics: Any,
        appearance: Any,
    ) -> SceneHashes:
        spec_hash = content_hash(spec)
        assets_hash = content_hash(assets)
        geometry_hash = content_hash(geometry)
        physics_hash = content_hash(physics)
        appearance_hash = content_hash(appearance)
        version = content_hash(
            {
                "spec_hash": spec_hash,
                "assets_hash": assets_hash,
                "geometry_hash": geometry_hash,
                "physics_hash": physics_hash,
                "appearance_hash": appearance_hash,
            }
        )
        return cls(
            spec_hash=spec_hash,
            assets_hash=assets_hash,
            geometry_hash=geometry_hash,
            physics_hash=physics_hash,
            appearance_hash=appearance_hash,
            version=version,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "spec": self.spec_hash,
            "assets": self.assets_hash,
            "geometry": self.geometry_hash,
            "physics": self.physics_hash,
            "appearance": self.appearance_hash,
            "version": self.version,
        }

    def matches(self, other: SceneHashes) -> bool:
        """Two hash sets identify the same scene version."""
        return self.version == other.version
