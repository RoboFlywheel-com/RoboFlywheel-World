# SPDX-License-Identifier: Apache-2.0
"""Asset coverage audit: the real denominators behind every retrieval claim.

The acceptance manual asks for a coverage audit because catalogue claims and verified
capability are different things, and the counts that matter are the ones that were measured:

    "先做覆盖审计，公布具备完整证据的实际分母" / "catalog 声明 physics_type 不等于这个类型
     在每个 backend 的功能已通过"

So the audit reports, per category, what is present, what is *declared*, and what actually
carries the evidence a downstream gate depends on -- dimensions, mass, collision geometry,
materials. A category with 3000 declared assets and 12 with usable physics is a 12-asset
category for planning purposes, and only reporting the 3000 is how a library looks complete
while a scene cannot be built from it.

The simready catalogue is 143 MB of one object per asset id, so the parser **streams** it
rather than loading it: a coverage audit that needs 2 GB of RAM is one that quietly stops
running on the machine that needs it most.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

SCHEMA_VERSION = "pvscene.asset-coverage/v1"


class CoverageError(RuntimeError):
    """The catalogue could not be audited."""


# The buffer holds only the unconsumed tail, and it is re-sliced once per entry, so its size
# is a direct multiplier on the copying cost. Entries here are a few hundred bytes; 64 KiB
# keeps tens of them in flight while making the per-entry copy negligible. A 1 MiB buffer
# produced identical output but took 8x longer on the 143 MB index.
DEFAULT_CHUNK_BYTES = 64 * 1024


def iter_json_object_items(
    path: Path, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> Iterator[tuple[str, Any]]:
    """Yield ``(key, value)`` from a top-level JSON object without loading it whole.

    The decoder always reads at offset 0 of a buffer that holds only the not-yet-consumed
    tail, and the buffer grows until the current value parses. That keeps the index of a
    decode independent of how many refills have happened, which is the bug the obvious
    "remember the offset across refills" version has: once the buffer is rewritten the old
    offset means something else, and values that straddle a refill are read from the wrong
    place or skipped.

    Memory stays bounded by the largest single entry rather than by the file. A truncated file
    raises rather than ending the iteration quietly, so a half-written catalogue cannot be
    audited as if it were complete.
    """
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = handle.read(chunk_bytes)
        if not buffer:
            return
        opening = buffer.find("{")
        if opening < 0:
            raise CoverageError(f"{path} is not a JSON object")
        buffer = buffer[opening + 1:]

        def grow(needed: int) -> bool:
            """Extend the buffer past ``needed`` bytes. False at end of file."""
            nonlocal buffer
            if len(buffer) > needed:
                return True
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return False
            buffer += chunk
            return True

        while True:
            # Strip, and only leave this loop with a non-empty *stripped* buffer. Breaking
            # out right after a refill instead would hand the key decoder a buffer that starts
            # with whitespace, which raw_decode does not skip, and every subsequent read would
            # fail on that same leading whitespace until the file ran out.
            while True:
                buffer = buffer.lstrip(" \t\r\n,")
                if buffer:
                    break
                if not grow(0):
                    return
            if buffer[0] == "}":
                return

            while True:
                try:
                    key, end = decoder.raw_decode(buffer, 0)
                    break
                except ValueError:
                    if not grow(len(buffer)):
                        raise CoverageError(f"{path} ended in the middle of an object key")
            buffer = buffer[end:]

            while True:
                buffer = buffer.lstrip(" \t\r\n:")
                if buffer:
                    break
                if not grow(0):
                    raise CoverageError(f"{path} ended before a value followed key {key!r}")

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, 0)
                    break
                except ValueError:
                    if not grow(len(buffer)):
                        raise CoverageError(f"{path} ended in the middle of the value for {key!r}")
            buffer = buffer[end:]
            yield key, value


@dataclass(slots=True)
class CategoryCoverage:
    """What one category actually offers, separated into declared and evidenced.

    ``evidence`` is a free-form count per named property rather than a fixed set of fields,
    because the two libraries carry different metadata: simready has mass and materials, HSSD
    has orientation and support annotations. Forcing both into one schema would either drop
    the differences or invent values for them.

    ``usable_count`` is therefore computed by each audit under its own definition, and the
    definition is stated alongside it. A single cross-provider definition does not exist: a
    desk with no mass is unusable as a dynamic body and perfectly usable as fixed furniture,
    so the number is only meaningful together with the sentence explaining it.
    """

    category: str
    domain: str
    declared_count: int = 0
    usable_count: int = 0
    evidence: dict[str, int] = field(default_factory=dict)
    physics_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "declared_count": self.declared_count,
            "usable_count": self.usable_count,
            "evidence": dict(sorted(self.evidence.items())),
            "physics_types": dict(sorted(self.physics_types.items())),
        }


@dataclass(slots=True)
class ProviderCoverage:
    provider: str
    library_version: str
    source: str
    asset_count: int = 0
    categories: dict[str, CategoryCoverage] = field(default_factory=dict)
    unparsed_entries: int = 0
    # Stated next to the number it explains, so a usable_count never travels without it.
    usable_definition: str = ""
    notes: tuple[str, ...] = ()

    @property
    def category_count(self) -> int:
        return len(self.categories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "library_version": self.library_version,
            "source": self.source,
            "asset_count": self.asset_count,
            "category_count": self.category_count,
            "usable_definition": self.usable_definition,
            "unparsed_entries": self.unparsed_entries,
            "categories": {
                name: entry.to_dict() for name, entry in sorted(self.categories.items())
            },
            "notes": list(self.notes),
        }


def _is_bbox(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    if len(value) != 3:
        return False
    try:
        return all(float(component) > 0 for component in value)
    except (TypeError, ValueError):
        return False


def audit_simready(
    index_path: Path,
    *,
    library_version: str = "unknown",
    progress: Callable[[int], None] | None = None,
) -> ProviderCoverage:
    """Audit the simready rigid catalogue.

    Every field that a later gate depends on is counted separately rather than folded into a
    single "present" flag, because the interesting number is the smallest one. A category
    whose dimensions are known but whose mass is missing cannot be placed and settled.
    """
    if not index_path.is_file():
        raise CoverageError(f"catalogue index not found: {index_path}")

    coverage = ProviderCoverage(
        provider="simready",
        library_version=library_version,
        source=str(index_path),
    )

    for position, (asset_id, entry) in enumerate(iter_json_object_items(index_path), start=1):
        if not isinstance(entry, Mapping):
            coverage.unparsed_entries += 1
            continue
        category = str(entry.get("super_category") or "").strip()
        if not category:
            coverage.unparsed_entries += 1
            continue

        slot = coverage.categories.get(category)
        if slot is None:
            slot = CategoryCoverage(category=category, domain=str(entry.get("domain") or ""))
            coverage.categories[category] = slot

        coverage.asset_count += 1
        slot.declared_count += 1

        physics_type = str(entry.get("physics_type") or "rigid")
        slot.physics_types[physics_type] = slot.physics_types.get(physics_type, 0) + 1

        has_dimensions = _is_bbox(entry.get("real_bbox")) or (
            isinstance(entry.get("max_dim_m"), (int, float)) and float(entry["max_dim_m"]) > 0
        )
        mass = entry.get("mass")
        has_mass = isinstance(mass, (int, float)) and float(mass) > 0
        has_material = bool(str(entry.get("material") or "").strip())
        has_mass_check = str(entry.get("mass_check") or "") == "ok"
        for name, flag in (
            ("with_dimensions", has_dimensions),
            ("with_mass", has_mass),
            ("with_material", has_material),
            ("with_mass_check_ok", has_mass_check),
        ):
            if flag:
                slot.evidence[name] = slot.evidence.get(name, 0) + 1
        # A dynamic rigid body needs dimensions, a positive mass and a material together, so
        # the usable count is the intersection rather than any single count.
        if has_dimensions and has_mass and has_material:
            slot.usable_count += 1

        if progress is not None and position % 50_000 == 0:
            progress(position)

    # Record what the audit could and could not establish, so a reader knows the shape of the
    # evidence rather than only its totals.
    coverage.usable_definition = (
        "assets carrying dimensions, a positive mass and a material together -- what a dynamic "
        "rigid body needs to be placed and settled"
    )
    coverage.notes = (
        "counts are read from the catalogue index only; no asset file was opened",
        "mass_check is the catalogue's own verdict and is reported as declared, not re-measured",
        f"{coverage.unparsed_entries} entr(ies) carried no usable super_category and were counted separately",
    )
    return coverage


def audit_hssd(synset_index: Path, *, library_version: str = "unknown") -> ProviderCoverage:
    """Audit the HSSD furniture metadata.

    HSSD arrives as raw GLB with a synset index, so what matters here is not mass (there is
    none) but orientation and support annotations, because those decide which relations are
    even expressible. Report them explicitly: ``FACING`` is only meaningful where a front
    vector was annotated, and on this library most objects have none.
    """
    if not synset_index.is_file():
        raise CoverageError(f"HSSD synset index not found: {synset_index}")

    try:
        payload = json.loads(synset_index.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageError(f"unreadable HSSD synset index: {error}") from error
    if not isinstance(payload, Mapping):
        raise CoverageError("HSSD synset index must be a mapping of synset to assets")

    coverage = ProviderCoverage(
        provider="hssd",
        library_version=library_version,
        source=str(synset_index),
    )
    with_front = 0
    with_up = 0
    with_support_region = 0

    for synset, entries in payload.items():
        if not isinstance(entries, Sequence):
            coverage.unparsed_entries += 1
            continue
        slot = CategoryCoverage(category=str(synset), domain="hssd")
        for entry in entries:
            if not isinstance(entry, Mapping):
                coverage.unparsed_entries += 1
                continue
            slot.declared_count += 1
            coverage.asset_count += 1
            slot.usable_count += 1
            slot.physics_types["rigid"] = slot.physics_types.get("rigid", 0) + 1
            if str(entry.get("name") or "").strip():
                slot.evidence["with_name"] = slot.evidence.get("with_name", 0) + 1
            if str(entry.get("front") or "").strip():
                slot.evidence["with_front"] = slot.evidence.get("with_front", 0) + 1
                with_front += 1
            if str(entry.get("up") or "").strip():
                slot.evidence["with_up"] = slot.evidence.get("with_up", 0) + 1
                with_up += 1
            if entry.get("support_region"):
                slot.evidence["with_support_region"] = slot.evidence.get("with_support_region", 0) + 1
                with_support_region += 1
        coverage.categories[str(synset)] = slot

    coverage.usable_definition = (
        "every indexed object, because HSSD carries no mass or collision metadata: usability here "
        "means usable as fixed furniture after an offline collision preparation, not usable as a "
        "dynamic body"
    )
    coverage.notes = (
        "HSSD ships raw GLB: there are no mass, collision or scale measurements to count",
        "with_name counts objects carrying a human-readable name, which is the only text metadata",
        "with_front / with_up are orientation annotations; FACING is only expressible where a front exists",
        "with_support_region marks objects carrying a precomputed support surface",
        f"orientation annotations: front={with_front}, up={with_up}, support_region={with_support_region}",
        "a category here is a WordNet synset, not a vendor category",
    )
    return coverage


@dataclass(frozen=True, slots=True)
class RequirementCandidate:
    """One provider's coverage of one requested category."""

    provider: str
    matched_categories: tuple[str, ...]
    declared_count: int
    usable_count: int

    @property
    def satisfied(self) -> bool:
        # A category with declared assets but no evidenced ones does not satisfy a requirement:
        # it cannot be placed and settled, so it is a gap dressed as coverage.
        return self.usable_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "matched_categories": list(self.matched_categories),
            "declared_count": self.declared_count,
            "usable_count": self.usable_count,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True, slots=True)
class RequirementResolution:
    """Where a requested category can actually be sourced, and where it cannot."""

    requirement_id: str
    role: str
    candidates: tuple[RequirementCandidate, ...]

    @property
    def servable_providers(self) -> tuple[str, ...]:
        return tuple(candidate.provider for candidate in self.candidates if candidate.satisfied)

    @property
    def satisfied(self) -> bool:
        return bool(self.servable_providers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "role": self.role,
            "satisfied": self.satisfied,
            "servable_providers": list(self.servable_providers),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def resolve_requirements(
    providers: Mapping[str, ProviderCoverage],
    requirements: Sequence[Mapping[str, Any]],
) -> tuple[RequirementResolution, ...]:
    """Match each requirement against every provider's categories.

    ``candidates`` in a requirement names the per-provider category spellings to look for, so
    a vocabulary difference ("Mug" vs ``mug.n.01``) does not read as an empty library.
    """
    resolutions: list[RequirementResolution] = []
    for requirement in requirements:
        requirement_id = str(requirement.get("id") or "")
        if not requirement_id:
            raise CoverageError(f"requirement without an id: {dict(requirement)}")
        role = str(requirement.get("role") or "unspecified")
        declared = requirement.get("candidates") or {}
        if not isinstance(declared, Mapping):
            raise CoverageError(f"requirement {requirement_id}: candidates must be a mapping")

        candidates: list[RequirementCandidate] = []
        for provider in sorted(providers):
            wanted = declared.get(provider) or []
            coverage = providers[provider]
            matched: list[str] = []
            total_declared = 0
            total_usable = 0
            for category_name in wanted:
                slot = coverage.categories.get(str(category_name))
                if slot is None:
                    continue
                matched.append(str(category_name))
                total_declared += slot.declared_count
                total_usable += slot.usable_count
            candidates.append(
                RequirementCandidate(
                    provider=provider,
                    matched_categories=tuple(matched),
                    declared_count=total_declared,
                    usable_count=total_usable,
                )
            )
        resolutions.append(
            RequirementResolution(
                requirement_id=requirement_id, role=role, candidates=tuple(candidates)
            )
        )
    return tuple(resolutions)


@dataclass(slots=True)
class CoverageReport:
    """The whole audit: measured provider coverage plus requirement resolution."""

    providers: dict[str, ProviderCoverage]
    resolutions: tuple[RequirementResolution, ...]
    absent_catalogues: tuple[str, ...] = ()

    @property
    def unsatisfied_requirements(self) -> tuple[str, ...]:
        return tuple(r.requirement_id for r in self.resolutions if not r.satisfied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "scope": (
                "coverage counts read from catalogue metadata only; no asset file was opened and "
                "no physics was exercised"
            ),
            "declared_vs_measured": (
                "declared_count is what the catalogue claims; usable_count is the intersection of "
                "dimensions, mass and material actually recorded. Only the latter bounds what can "
                "be placed and settled"
            ),
            "providers": {name: coverage.to_dict() for name, coverage in sorted(self.providers.items())},
            "absent_catalogues": list(self.absent_catalogues),
            "requirements": [r.to_dict() for r in self.resolutions],
            "unsatisfied_requirements": list(self.unsatisfied_requirements),
        }


@dataclass(frozen=True, slots=True)
class FilesystemCheck:
    """What is on disk, against what the catalogue index accounts for.

    The index is a summary; the export tree is the library. They can disagree, and the direction
    that matters is directories the index does not mention: those assets exist, cannot be
    retrieved, and are invisible to a report that counts index entries. Reporting only the index
    count is therefore the same overclaim as reporting only declared physics -- it states the
    library is smaller or larger than the thing a retrieval actually searches.

    Slow by design and opt-in: counting half a million directories on a network mount takes minutes,
    and a coverage audit nobody runs is worse than one that is honest about its cost.
    """

    asset_directories: int
    domains: int
    index_entries: int
    elapsed_s: float

    @property
    def delta(self) -> int:
        return self.asset_directories - self.index_entries

    @property
    def consistent(self) -> bool:
        return self.delta == 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "asset_directories": self.asset_directories,
            "domains": self.domains,
            "index_entries": self.index_entries,
            "delta": self.delta,
            "elapsed_s": round(self.elapsed_s, 1),
        }
        if self.delta:
            payload["gap"] = (
                f"UNINDEXED_DIRS({self.delta})" if self.delta > 0 else f"MISSING_DIRS({-self.delta})"
            )
            payload["explanation"] = (
                "directories present on disk that the catalogue index does not mention: they exist "
                "but are unreachable by retrieval"
                if self.delta > 0
                else "index entries with no directory on disk: retrieval would resolve to missing files"
            )
        return payload


def count_filesystem_assets(root: Path) -> tuple[int, int, float]:
    """Count ``<root>/<domain>/<shard>/<asset>`` directories and the domains that hold them.

    Uses ``os.scandir`` rather than walking everything: only directories at exactly the asset
    depth are counted, so a nested cache inside an asset does not inflate the number.
    """
    import os
    import time as _time

    if not root.is_dir():
        raise CoverageError(f"assets root not found: {root}")
    started = _time.monotonic()
    domains = 0
    assets = 0
    for domain in os.scandir(root):
        if not domain.is_dir():
            continue
        domains += 1
        for shard in os.scandir(domain.path):
            if not shard.is_dir():
                continue
            for entry in os.scandir(shard.path):
                if entry.is_dir():
                    assets += 1
    return assets, domains, _time.monotonic() - started
