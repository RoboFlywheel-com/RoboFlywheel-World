# SPDX-License-Identifier: Apache-2.0
"""Retrieve assets from the simready library through the native retrieval engine.

The adapter keeps the recorded behaviour -- SBERT class recall, size filtering, explainable reranking --
and converts the result into v3 contracts. Three things it does *not* do:

* it does not expose provider URIs to the caller as anything but an
  :class:`~rfw_w.scenes.contracts.asset.AssetHandle`, because paths must not reach a model;
* it does not trust the catalogue's `real_bbox`. It measures the mesh, because on this library
  the catalogue disagrees with the geometry by enough to change where an object rests;
* it does not assume the vendor's declared up-axis. It is verified from the shapes, and the
  conversion is recorded on the asset so it cannot be applied twice.

A retrieval that finds nothing returns a typed failure code rather than an empty success, since
the difference between "no such category" and "the library is unreachable" decides whether the
request becomes unsupported or retried.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from rfw_w.scenes.assets.retrieval_engine import RetrievalEngine, RetrievalGoal, ReleaseCatalog
from rfw_w.scenes.assets.mesh_probe import (
    MeshProbeError,
    classify_up_axis,
    measure_obj,
    rotate_extents,
)
from rfw_w.scenes.contracts.asset import (
    AffordanceStatus,
    AssetHandle,
    AssetKey,
    AssetRecordV3,
    CollisionPart,
    NormalisationRecord,
    Provenance,
    RigidPhysics,
)
from rfw_w.scenes.contracts.hashing import content_hash, hash_text
from rfw_w.scenes.contracts.scene_spec import (
    EntityRole,
    EntitySpec,
    PhysicsType,
    Range1D,
)
from rfw_w.scenes.contracts.sizing import ratio_violation
from rfw_w.scenes.contracts.units import Vec3

TOOL_VERSION = "rfw.simready-retrieval/v1"
DEFAULT_TARGET_SIZE_M = Vec3(0.15, 0.15, 0.15)
# Absent prompt dimensions must not turn a mug or condiment jar into a 15 cm vessel.
# These are retrieval aims, not claims about the physical grasp or accepted mesh size.
CATEGORY_TARGET_SIZE_M = {"cup": Vec3(.09, .09, .105), "jar": Vec3(.08, .08, .12),
                          "alarm_clock": Vec3(.12, .07, .12)}

# The catalogue does not carry a licence field. Recording the absence is required by the
# contract, which refuses an asset with no licence at all: an unrecorded licence is a
# distribution problem, and silently writing "unknown" would hide it.
LICENCE_UNRECORDED = "unrecorded:provider_catalogue_has_no_license_field"


class RetrievalAdapterError(RuntimeError):
    """The adapter could not perform the retrieval."""


@dataclass(frozen=True, slots=True)
class LibraryFrame:
    """Which frame a library version's meshes arrive in, and how the world frame is reached.

    Not a convention to be learned from the geometry at read time: getting it wrong lays every
    object on its side, and the two versions this project has loaded disagree.

    * ``v0.2`` ships the GLB's own Y-up meshes -- ``mesh/<id>.obj``, verified from geometry -- so
      the world frame is a rotation about +X by 90 degrees, applied by the reader to what it
      measures and by the worker to the instance it authors. ``rotate_on_read``.
    * ``v1.3`` also ships the source's own frame, at ``mesh/<leaf>.obj``, and *additionally* a
      normalised family ``mesh/m_<leaf>.obj`` that its README recommends loading as-is. Measured
      against the release's files, that family is not reliably upright -- the jar's axis is moved to
      Z and the mug's stays on Y, so one of the two is sideways however it is loaded -- and the
      collision mesh exists only in it. So the pair is prepared here from the source's frame,
      already rotated once (``prepare``), and nothing downstream rotates it again:
      ``prepared_in_world_frame``. Getting *this* wrong is how the mug came out on its side a second
      time, with its 0.10 m handle span measured as its height.

    ``alignment`` is a name rather than a bool because the two rotations -- "the reader rotates what
    it measured" and "somebody already rotated it" -- were one bool for an afternoon, and both
    states of it are wrong together.
    """

    source_frame: str
    method: str
    alignment: str
    prepare: bool = False

    @property
    def rotate_on_read(self) -> bool:
        """Whether this version's meshes still need the Y-up correction applied by the reader."""
        return self.alignment == "rotate_on_read"


# The two ways a library's meshes can arrive. Named rather than boolean because both states of the
# old bool were wrong together: "who rotates this" and "has it been rotated" are one question only
# when there is exactly one answer, and there are two.
ALIGNMENTS = ("rotate_on_read", "prepared_in_world_frame")

LIBRARY_FRAMES: Mapping[str, LibraryFrame] = {
    "v0.2": LibraryFrame("simready_y_up", "rotate_x_plus_90_verified_from_geometry", "rotate_on_read"),
    "v1.3": LibraryFrame(
        "simready_y_up", "rotate_x_plus_90_applied_by_the_preparation",
        "prepared_in_world_frame", prepare=True,
    ),
}


def frame_for(library_version: str) -> LibraryFrame:
    """The declared frame for one library version.

    A version with no entry is refused rather than assumed: "assume the meshes are Y-up" is
    precisely the assumption that would have laid v1.3's library on its side without a single gate
    noticing, so an unknown version is a named failure instead.
    """
    try:
        frame = LIBRARY_FRAMES[library_version]
    except KeyError:
        raise RetrievalAdapterError(
            f"library version {library_version!r} has no declared frame; add it to LIBRARY_FRAMES "
            f"with the frame its meshes are actually in (known: {sorted(LIBRARY_FRAMES)})"
        ) from None
    if frame.alignment not in ALIGNMENTS:
        raise RetrievalAdapterError(
            f"library version {library_version!r} declares alignment {frame.alignment!r}, which is "
            f"not one of {list(ALIGNMENTS)}; a version whose meshes are rotated by nobody in "
            "particular renders sideways with every gate green"
        )
    return frame


def worker_normalisation(library_version: str) -> str:
    """The ``MeshInstance.normalisation`` the Isaac worker expects for a version's meshes.

    The worker owns one rotation (``y_up_to_z_up``) and applies it to whatever claims to be Y-up,
    so a mesh that is already Z-up has to say ``none`` here rather than be rotated twice.
    """
    return "y_up_to_z_up" if frame_for(library_version).rotate_on_read else "none"


@dataclass(frozen=True, slots=True)
class SimreadyConfig:
    """Local catalogue, asset, semantic-index and preparation resources."""

    index_path: Path
    assets_root: Path
    # The semantic class index is a separate artefact from the catalogue: the catalogue lists
    # assets, the semantic index holds the per-class text features. Passing the catalogue where
    # the index is expected fails with "unsupported semantic index schema: None", which is why
    # they are two required fields rather than one.
    semantic_index_path: Path | None = None
    provider: str = "simready"
    library_version: str = "unknown"
    # Where a version whose meshes need preparing (see `LibraryFrame.prepare`) writes them. Required
    # only for such a version, and named as such: a v0.2 deployment that has none must keep working.
    prepared_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.index_path.is_file():
            raise RetrievalAdapterError(f"catalogue index not found: {self.index_path}")
        if not self.assets_root.is_dir():
            raise RetrievalAdapterError(f"assets root not found: {self.assets_root}")
        if self.semantic_index_path is not None and not self.semantic_index_path.is_file():
            raise RetrievalAdapterError(
                f"semantic class index not found: {self.semantic_index_path}"
            )
        frame_for(self.library_version)



@dataclass(frozen=True, slots=True)
class CandidateView:
    """A retrieved candidate, with URIs withheld.

    ``asset_key`` and ``score`` are for the ranking and the audit trail; ``semantic_view()`` is
    what may be shown to a model. The URIs live on the handle, which is a separate type for
    exactly this reason.
    """

    asset_key: AssetKey
    semantic: str
    super_category: str
    dimensions_m: Vec3
    mass_kg: float | None
    material: str | None
    total_score: float
    semantic_score: float
    size_score: float
    readiness_score: float
    affordability_score: float
    up_axis: str | None
    # Internal: a catalogue-relative prefix used to resolve files at materialisation time. Not a
    # URI, and deliberately absent from both projections below.
    library_prefix: str = ""
    friction_static: float | None = None
    friction_dynamic: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset_key.to_dict(),
            "semantic": self.semantic,
            "super_category": self.super_category,
            "dimensions_m": [round(value, 6) for value in self.dimensions_m.to_list()],
            "mass_kg": self.mass_kg,
            "material": self.material,
            "score": round(self.total_score, 6),
            "score_breakdown": {
                "semantic": round(self.semantic_score, 6),
                "size": round(self.size_score, 6),
                "readiness": round(self.readiness_score, 6),
                "affordance": round(self.affordability_score, 6),
            },
            "up_axis": self.up_axis,
        }

    def semantic_view(self) -> dict[str, Any]:
        """Path-free projection for a model."""
        return {
            "semantic_id": self.asset_key.semantic_id,
            "semantic": self.semantic,
            "category": self.super_category,
            "dimensions_m": [round(value, 4) for value in self.dimensions_m.to_list()],
            "material": self.material,
        }


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """What a retrieval produced, including why it produced nothing."""

    entity_id: str
    candidates: tuple[CandidateView, ...]
    rejected: tuple[tuple[str, str], ...] = ()
    failure_code: str | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "candidate_count": len(self.candidates),
            "rejected_count": len(self.rejected),
            "rejection_reasons": sorted({reason for _, reason in self.rejected}),
            "failure_code": self.failure_code,
            "detail": self.detail,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _midpoint(bounds: Range1D) -> float:
    return (bounds.minimum + bounds.maximum) / 2.0


def target_size_for(entity: EntitySpec) -> Vec3:
    """The size the retrieval should aim for, from the entity's own bounds.

    Taken from the middle of the requested range rather than from its largest value: aiming at
    the maximum systematically selects the biggest asset in the range, which for a cup means
    picking a bowl.
    """
    if entity.size_bounds_m is None:
        return CATEGORY_TARGET_SIZE_M.get(entity.category, DEFAULT_TARGET_SIZE_M)
    return Vec3(*(_midpoint(axis) for axis in entity.size_bounds_m))


def build_goal(entity: EntitySpec, *, relation: str | None = None,
               support_id: str | None = None, support_goal_id: str | None = None):
    """Build the metadata-only retrieval request; placement is solved separately."""
    if relation and relation.lower() not in {"free", "on", "in", "inside", "under", "next_to"}:
        raise RetrievalAdapterError(f"unknown relation {relation!r}")
    return RetrievalGoal(entity.id, entity.category, entity.notes or entity.category,
                         entity.physics_type, target_size_for(entity))


def build_encoder(prepared_model_dir: Path | None):
    """Load the local SBERT runtime lazily, without fetching weights."""
    if prepared_model_dir is None or not prepared_model_dir.is_dir():
        raise RetrievalAdapterError("no prepared SBERT model directory; configure local weights")
    from rfw_w.scenes.assets.semantic_index import SentenceTransformerEncoder
    return SentenceTransformerEncoder(str(prepared_model_dir))


def build_engine(config: SimreadyConfig, *, encoder: Any = None, policy: Any = None,
                 scoring: str = "semantic"):
    """Build native retrieval. Lexical scoring must be explicitly requested."""
    scorer = None
    if scoring == "semantic":
        if encoder is None or config.semantic_index_path is None:
            raise RetrievalAdapterError("semantic retrieval requires an encoder and class index")
        from rfw_w.scenes.assets.semantic_index import load_semantic_class_index, SbertSemanticScorer
        scorer = SbertSemanticScorer(load_semantic_class_index(config.semantic_index_path), encoder=encoder)
    elif scoring != "lexical":
        raise RetrievalAdapterError(f"unknown scoring mode {scoring!r}")
    return RetrievalEngine(ReleaseCatalog(config, scorer=scorer), policy=policy)


def retrieve_entity(
    engine: Any,
    entity: EntitySpec,
    *,
    config: SimreadyConfig,
    relation: str | None = None,
    support_id: str | None = None,
    support_goal_id: str | None = None,
    limit: int = 8,
    measurements: list[tuple[str, Vec3]] | None = None,
    admitted_categories: Sequence[str] | None = None,
    allow_dimension_permutation: bool = False,
) -> RetrievalOutcome:
    """Retrieve candidates for one entity and describe them in v3 terms.

    ``measurements`` carries ``(super_category, extents)`` pairs accumulated from earlier
    retrievals so the up-axis verdict is reached from real shapes; with none supplied the axis is
    reported as undecided rather than assumed.
    """
    goal = build_goal(
        entity,
        relation=relation,
        support_id=support_id,
        support_goal_id=support_goal_id,
    )
    try:
        # Apply the output limit after category/mesh admission. Eight broad semantic hits
        # previously left just one cup, making a visual asset repair impossible.
        attempt = engine.retrieve(goal, limit=max(64, limit))
    except Exception as error:  # noqa: BLE001 - provider IO errors are typed at this interface
        return RetrievalOutcome(
            entity_id=entity.id,
            candidates=(),
            failure_code="ASSET_IO_FAILURE",
            detail=f"{type(error).__name__}: {error}",
        )

    if attempt.failure_code and not attempt.candidates:
        # Retrieval distinguishes an empty catalogue from a filtered-out one; keep that distinction.
        code = (
            "ASSET_UNAVAILABLE"
            if attempt.failure_code in ("no_candidates", "no_catalog", "empty")
            else "ASSET_IO_FAILURE"
        )
        return RetrievalOutcome(
            entity_id=entity.id,
            candidates=(),
            rejected=attempt.rejected,
            failure_code=code,
            detail=f"retrieval reported {attempt.failure_code}",
        )

    verdict = classify_up_axis(measurements or [])
    views: list[CandidateView] = []
    rejected: list[tuple[str, str]] = list(attempt.rejected)
    ratio_limit = getattr(engine.policy, "max_dimension_ratio", 2.0)
    for candidate in attempt.candidates:
        asset = candidate.asset
        measured = _measure_dimensions(asset, config)
        if measured is None:
            rejected.append((asset.asset_id, "collision mesh missing or unmeasurable"))
            continue
        extents, _ = measured

        # Re-check size against the *measured* extents. The coarse gate filters on the catalogue's
        # real_bbox, which on this library disagrees with the geometry by enough to matter: a
        # "coffee cup" catalogued as fitting passed that gate at 0.31 m tall. The architecture's
        # rule is that coarse catalogue values serve recall while the final fit uses the
        # canonical measurement, so the measured number has to decide.
        over = coarse_size_violation(extents, goal.target_size_m, ratio_limit,
                                     allow_dimension_permutation=allow_dimension_permutation)
        if over is not None:
            rejected.append((asset.asset_id, f"MEASURED_SIZE_OUT_OF_RANGE:{over}"))
            continue

        # Admission, not ranking. Semantic recall is deliberately broad -- a query for "desk"
        # returns Desk Bells and Desk Organizers, which are adjacent in embedding space and
        # useless as a desk. The acceptance manual allows a hard target to be swapped only
        # within synonyms the user's semantics permit, so the matched category is checked
        # against an explicit admitted set.
        category = str(getattr(asset, "metadata", {}).get("super_category", "")) or asset.semantic_class
        if admitted_categories is not None and category not in admitted_categories:
            rejected.append(
                (asset.asset_id, f"CATEGORY_NOT_ADMITTED:{category}")
            )
            continue

        views.append(
            CandidateView(
                asset_key=AssetKey(
                    provider=config.provider,
                    library_version=config.library_version,
                    asset_id=asset.asset_id,
                ),
                semantic=asset.semantic_class,
                super_category=category,
                dimensions_m=extents,
                mass_kg=asset.mass_kg,
                material=asset.material_label,
                total_score=float(candidate.evidence.total_score),
                semantic_score=float(candidate.evidence.semantic_score),
                size_score=float(candidate.evidence.size_score),
                readiness_score=float(candidate.evidence.readiness_score),
                affordability_score=float(candidate.evidence.affordance_score),
                up_axis=verdict.up_axis,
                library_prefix=str(
                    getattr(asset, "metadata", {}).get("assets_prefix", "")
                ).strip("/"),
                friction_static=_as_float(asset.friction),
                friction_dynamic=_as_float(asset.friction_secondary),
            )
        )

    if not views:
        reasons = sorted({reason.split(":")[0] for _, reason in rejected})
        detail = "every candidate was rejected; see rejection reasons"
        if any(reason.startswith("CATEGORY_NOT_ADMITTED") for _, reason in rejected):
            detail = (
                "no candidate belongs to an admitted category for this requirement; semantic "
                "recall matched adjacent categories only, and a hard target may not be replaced "
                "by one outside the permitted synonyms"
            )
        return RetrievalOutcome(
            entity_id=entity.id,
            candidates=(),
            rejected=tuple(rejected),
            failure_code="ASSET_UNAVAILABLE",
            detail=f"{detail} (rejection kinds: {reasons})",
        )
    if admitted_categories:
        priority = {category: index for index, category in enumerate(admitted_categories)}
        wanted = target_size_for(entity).to_list()
        views.sort(key=lambda view: (priority.get(view.super_category, len(priority)),
                                     sum(abs(math.log(value / target)) for value, target in
                                         zip(view.dimensions_m.to_list(), wanted)),
                                     -view.total_score, view.asset_key.asset_id))
    return RetrievalOutcome(entity_id=entity.id, candidates=tuple(views[:limit]), rejected=tuple(rejected))


def coarse_size_violation(extents: Vec3, target: Any, ratio_limit: float, *,
                          allow_dimension_permutation: bool = False) -> str | None:
    if not allow_dimension_permutation:
        return measured_size_violation(extents, target, ratio_limit)
    from itertools import permutations
    return None if any(measured_size_violation(Vec3(*dims), target, ratio_limit) is None
                       for dims in permutations(extents.to_list())) else "no admitted dimension permutation"


def measured_size_violation(extents: Vec3, target: Any, ratio_limit: float) -> str | None:
    """Whether the measured extents are outside the requested size range.

    The rule lives in `contracts.sizing` and is shared with furniture retrieval.

    Kept as a thin wrapper rather than deleted because callers and tests name this function, and
    because the ``Vec3`` unpacking belongs on this side of the boundary.
    """
    return ratio_violation(extents.to_list(), (target.x, target.y, target.z), ratio_limit)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@lru_cache(maxsize=16_384)
def _measure_obj_cached(path_text: str, _mtime_ns: int, _size: int) -> Any:
    """Cache the immutable OBJ measurement while still invalidating changed files."""
    return measure_obj(Path(path_text))


def _measure_path(path: Path) -> Any:
    stat = path.stat()
    return _measure_obj_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _measure_dimensions(asset: Any, config: SimreadyConfig) -> tuple[Vec3, Any] | None:
    """Measure the collision mesh, returning Z-up extents.

    Measured rather than read from the catalogue: the catalogue's `real_bbox` is rounded and, on
    this library, wrong by more than a centimetre on some assets.
    """
    uris = [getattr(asset, "collision_uri", None), getattr(asset, "visual_uri", None)]
    frame = frame_for(config.library_version)
    for uri in uris:
        if not uri:
            continue
        path = Path(uri)
        if not path.is_file():
            continue
        try:
            measurement = _measure_path(path)
        except (MeshProbeError, OSError):
            continue
        # Which frame the mesh is in is the library version's to declare: v0.2 ships the GLB's own
        # Y-up meshes and needs the rotation, v1.3 ships a normalised export that does not.
        source_mesh = path.name == f"{asset.asset_id}.obj"
        extents = rotate_extents(measurement.extents) if frame.rotate_on_read or source_mesh else measurement.extents
        return extents, measurement
    return None


def box_inertia(mass_kg: float, extents: Vec3) -> tuple[float, ...]:
    """Principal inertia of the equivalent solid box, as a flattened 3x3.

    An approximation, and labelled as one. The exact inertia needs the real mass distribution,
    which this library does not provide; a box inertia is symmetric positive definite so the body
    simulates correctly, and the provenance records that it was estimated so a functional
    decision is never made on it silently.
    """
    x, y, z = extents.x, extents.y, extents.z
    factor = mass_kg / 12.0
    return (
        factor * (y * y + z * z), 0.0, 0.0,
        0.0, factor * (x * x + z * z), 0.0,
        0.0, 0.0, factor * (x * x + y * y),
    )


def _prepare_candidate(view: CandidateView, config: SimreadyConfig):
    """Prepare one candidate's mesh pair from the library's source frame.

    Raises when the configuration has no prepared directory: a version that needs preparing cannot be
    used without one, and the alternative -- reading the library's own normalised family and hoping
    its per-asset heuristic got the up axis right -- is the defect this exists to avoid.
    """
    from rfw_w.scenes.assets.providers.simready import (
        SimreadyPrepareConfig,
        SimreadyPrepareError,
        prepare_rigid,
    )

    if config.prepared_root is None:
        raise RetrievalAdapterError(
            f"library version {config.library_version!r} needs its meshes prepared, but this "
            "configuration has no prepared_root; set it to a writable directory"
        )
    try:
        return prepare_rigid(
            SimreadyPrepareConfig(
                assets_root=config.assets_root, prepared_root=config.prepared_root,
            ),
            prefix=view.library_prefix,
            leaf=view.asset_key.asset_id,
            asset_id=view.asset_key.asset_id,
        )
    except SimreadyPrepareError as error:
        raise RetrievalAdapterError(str(error)) from error


def to_asset_record(
    view: CandidateView, config: SimreadyConfig
) -> tuple[AssetRecordV3, AssetHandle]:
    """Build a v3 record and handle for one candidate, measuring the mesh.

    Raises :class:`RetrievalAdapterError` when the collision mesh cannot be resolved: a record
    without collision geometry cannot be placed or settled, and inventing a proxy is the
    substitution the acceptance manual forbids.
    """
    if not view.library_prefix:
        raise RetrievalAdapterError(f"candidate {view.asset_key.asset_id} has no library prefix")
    base = config.assets_root / view.library_prefix
    visual = base / "mesh" / f"{view.asset_key.asset_id}.obj"
    collision = base / "mesh" / f"{view.asset_key.asset_id}_collision.obj"
    metadata = base / f"{view.asset_key.asset_id}.urdf"
    frame = frame_for(config.library_version)

    if frame.prepare:
        # The library's own pair is not one we can use: see `LibraryFrame.prepare`. The prepared
        # pair is built from the source mesh, and *its* measurements are the ones that count, so
        # this branch measures what it just wrote rather than the library's collision.
        prepared = _prepare_candidate(view, config)
        visual, collision = prepared.visual_obj, prepared.collision_obj
        measurement = _measure_path(collision)
    else:
        if not collision.is_file():
            raise RetrievalAdapterError(
                f"collision mesh missing for {view.asset_key.asset_id}: {collision}"
            )
        try:
            measurement = _measure_path(collision)
        except (MeshProbeError, OSError) as error:
            raise RetrievalAdapterError(
                f"could not measure {view.asset_key.asset_id}: {error}"
            ) from error
    if not visual.is_file():
        raise RetrievalAdapterError(f"visual mesh missing for {view.asset_key.asset_id}: {visual}")

    # The normalisation the asset carries, read from the library version's own declaration rather
    # than assumed: applying the Y-up rotation to an export that is already upright lays the object
    # on its side -- which is exactly what happened when v1.3's normalised family was taken at its
    # word, and the critique of that run said so.
    normalisation = NormalisationRecord(
        applied_scale=1.0,
        applied_translation=Vec3.origin(),
        source_frame=frame.source_frame,
        target_frame="world_z_up",
        method=frame.method,
        version="v1",
    )

    extents = rotate_extents(measurement.extents) if frame.rotate_on_read else measurement.extents
    bounds_min = rotate_extents(measurement.bounds_min) if frame.rotate_on_read else measurement.bounds_min
    bounds_max = rotate_extents(measurement.bounds_max) if frame.rotate_on_read else measurement.bounds_max

    measured = Provenance(
        method="obj_vertex_extents",
        version=TOOL_VERSION,
        status=AffordanceStatus.MEASURED,
        source_uri=str(collision),
    )
    estimated = Provenance(
        method="box_approximation_from_measured_bounds",
        version=TOOL_VERSION,
        status=AffordanceStatus.ESTIMATED,
        source_uri=str(collision),
    )

    mass = view.mass_kg
    if mass is None or mass <= 0:
        raise RetrievalAdapterError(
            f"{view.asset_key.asset_id} has no usable mass; a dynamic body cannot be built from it"
        )

    physics = RigidPhysics(
        mass_kg=float(mass),
        center_of_mass_m=Vec3(0.0, 0.0, 0.0),
        inertia_kg_m2=box_inertia(float(mass), extents),
        # Friction is recorded by the library; mineral oil-free contact needs a value, and a
        # missing one is filled with a documented default rather than left as zero, which would
        # make every object slide.
        static_friction=view.friction_static if view.friction_static is not None else 0.5,
        dynamic_friction=view.friction_dynamic if view.friction_dynamic is not None else 0.4,
        restitution=0.05,
        provenance=measured if view.friction_static is not None else estimated,
    )

    # Hash the files that define the asset, so evidence can be tied to the exact bytes.
    source_hash = hash_text(
        f"{collision.name}:{collision.stat().st_size}:{metadata.name if metadata.is_file() else ''}"
        f":{measurement.vertex_count}:{view.asset_key.asset_id}"
    )
    package_hash = content_hash(view.to_dict())

    record = AssetRecordV3(
        key=view.asset_key,
        source_hash=source_hash,
        license=LICENCE_UNRECORDED,
        semantic=view.semantic,
        super_category=view.super_category,
        physics_type=PhysicsType.RIGID,
        dimensions_m=extents,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        canonical_frame="world_z_up",
        normalisation=normalisation,
        visual_uri=str(visual),
        collision_parts=(
            CollisionPart(
                part_id="collision_mesh",
                source="collision_mesh",
                # The library's collision geometry is a convex export produced elsewhere; from
                # here it is a mesh whose fidelity was not audited, so it is marked approximate
                # rather than claimed exact.
                approximate=True,
                vertex_count=measurement.vertex_count,
                face_count=measurement.face_count,
                provenance=measured,
            ),
        ),
        materials=(view.material,) if view.material else (),
        rigid_physics=physics,
        capabilities={},  # default UNTESTED: this specific asset has not been run
        prepared_version=f"{config.library_version}+measured_v1",
    )
    handle = AssetHandle(
        key=view.asset_key,
        visual_uri=str(visual),
        collision_uris=(str(collision),),
        metadata_uri=str(metadata) if metadata.is_file() else None,
        package_hash=package_hash,
    )
    return record, handle


def load_admitted_categories(
    requirements_path: Path, *, provider: str = "simready"
) -> dict[str, tuple[str, ...]]:
    """Read the permitted category spellings per requirement from the requirements config.

    Kept as configuration rather than as a heuristic threshold on the semantic score: a "Desk
    Bell" can score highly against "desk" precisely because the words are close, so no threshold
    separates a synonym from a different object. Only an explicit list does.
    """
    import json

    if not requirements_path.is_file():
        raise RetrievalAdapterError(f"requirements config not found: {requirements_path}")
    payload = json.loads(requirements_path.read_text())
    admitted: dict[str, tuple[str, ...]] = {}
    for requirement in payload.get("requirements", []):
        requirement_id = str(requirement.get("id") or "")
        if not requirement_id:
            continue
        candidates = (requirement.get("candidates") or {}).get(provider) or []
        admitted[requirement_id] = tuple(str(name) for name in candidates)
    return admitted
