# SPDX-License-Identifier: Apache-2.0
"""Relation predicates, as evidence rather than as booleans.

A predicate that returns true or false cannot be reviewed, cannot be invalidated by a scene
change, and cannot be distinguished from one that was never evaluated. So each evaluation
produces an :class:`~rfw_w.scenes.contracts.tool.Evidence` carrying the scene hash it was measured
against, the numbers behind the verdict, and the witness that supports it.

The two failure modes the acceptance manual names are kept apart on purpose:

* an object that settled stably *somewhere else* is ``FUNCTION_PREDICATE_FAILED`` -- the relation
  was checked and is false;
* a predicate that was never evaluated is ``NOT_RUN``, and never satisfies a gate.

For INSIDE, only a *certified* rigid receptacle may carry the predicate. A point inside a
container's bounding box is not inside its cavity, and a sealed mesh makes an inside test
unreliable, so an uncertified cavity yields ``UNSUPPORTED`` rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from rfw_w.scenes.contracts.asset import Receptacle
from rfw_w.scenes.contracts.scene_spec import Predicate
from rfw_w.scenes.contracts.tool import Evidence, EvidenceStatus
from rfw_w.scenes.solvers.support import (
    PlacedInstance,
    Surface,
    assert_not_on_floor_when_support_expected,
    resolve_support,
)

TOOL_VERSION = "rfw_w.scenes.solvers.relation/v1"
BACKEND = "geometry"

# Thresholds are configured and reported alongside every measurement; these are the initial
# engineering values from the manual §6, to be calibrated on the development set.
ON_THRESHOLD_PROFILE = "on_engineering_v1"
ON_TOLERANCE_M = 0.003
ON_MIN_COVERAGE = 0.5


class RelationError(ValueError):
    """The relation request is not well formed."""


@dataclass(frozen=True, slots=True)
class RelationRequest:
    """One relation to evaluate, against a specific scene version."""

    relation_id: str
    subject_instance: str
    predicate: Predicate
    object_instance: str
    scene_version: str
    hard: bool = True

    def __post_init__(self) -> None:
        if not self.relation_id:
            raise RelationError("relation must have an id")
        if self.subject_instance == self.object_instance:
            raise RelationError(f"relation {self.relation_id}: a relation needs two instances")


def evaluate_on(
    request: RelationRequest,
    *,
    subject: PlacedInstance,
    support_instance: str,
    surfaces: Sequence[Surface],
    tolerance_m: float = ON_TOLERANCE_M,
    min_coverage: float = ON_MIN_COVERAGE,
) -> Evidence:
    """Evaluate ON(support_instance).

    The subject's own footprint decides which surface it rests on, and the surface's owner must be
    the named support. Stable-but-elsewhere is a failure of the predicate with the actual owner
    named, which is what makes "fell to the floor and stopped" distinguishable from "is on the
    desk" in the report rather than only in principle.
    """
    if request.predicate is not Predicate.ON:
        raise RelationError(f"evaluate_on called for predicate {request.predicate.value}")

    resolution = resolve_support(
        subject, surfaces, contact_tolerance_m=tolerance_m, min_coverage=min_coverage
    )
    guarded = assert_not_on_floor_when_support_expected(resolution, expected_owner=support_instance)

    witness_refs = () if guarded.witness is None else (f"surface:{guarded.witness.surface_id}",)
    if guarded.failure_code is None and guarded.witness is not None:
        return Evidence(
            gate_id="function.initial_relations",
            status=EvidenceStatus.PASS,
            state_hash=request.scene_version,
            backend=BACKEND,
            tool_version=TOOL_VERSION,
            method="support_polygon_coverage_and_contact_height",
            measured_value=guarded.witness.coverage_ratio,
            unit="fraction_of_footprint",
            threshold_profile=ON_THRESHOLD_PROFILE,
            witness_refs=witness_refs,
            detail=(
                f"{request.subject_instance} ON {support_instance}: "
                f"coverage {guarded.witness.coverage_ratio:.3f}, "
                f"base gap {guarded.witness.base_gap_m:.5f} m"
            ),
        )

    return Evidence(
        gate_id="function.initial_relations",
        status=EvidenceStatus.FAIL,
        state_hash=request.scene_version,
        backend=BACKEND,
        tool_version=TOOL_VERSION,
        method="support_polygon_coverage_and_contact_height",
        witness_refs=witness_refs,
        detail=guarded.detail
        or f"{request.subject_instance} does not rest on {support_instance}",
    )


def evaluate_inside(
    request: RelationRequest,
    *,
    receptacle: Receptacle | None,
    scene_version: str,
) -> Evidence:
    """Evaluate INSIDE for a receptacle, refusing to guess about an uncertified cavity.

    ``UNSUPPORTED`` rather than ``FAIL`` when the cavity is not certified: the relation could not
    be decided, and marking it failed would blame the scene for a missing measurement.
    """
    if request.predicate is not Predicate.INSIDE:
        raise RelationError(f"evaluate_inside called for predicate {request.predicate.value}")

    if receptacle is None:
        return Evidence(
            gate_id="function.initial_relations",
            status=EvidenceStatus.UNSUPPORTED,
            state_hash=scene_version,
            backend=BACKEND,
            tool_version=TOOL_VERSION,
            method="certified_rigid_receptacle_required",
            detail=(
                f"no receptacle is declared for {request.object_instance}; only a certified rigid "
                "receptacle may carry an INSIDE predicate, because a point inside a bounding box "
                "is not inside a cavity"
            ),
        )

    if not receptacle.is_certified:
        return Evidence(
            gate_id="function.initial_relations",
            status=EvidenceStatus.UNSUPPORTED,
            state_hash=scene_version,
            backend=BACKEND,
            tool_version=TOOL_VERSION,
            method="certified_rigid_receptacle_required",
            detail=(
                f"receptacle {receptacle.cavity_id} is {receptacle.provenance.status.value}, not "
                "measured; an estimated cavity cannot certify containment"
            ),
        )

    return Evidence(
        gate_id="function.initial_relations",
        status=EvidenceStatus.NOT_RUN,
        state_hash=scene_version,
        backend=BACKEND,
        tool_version=TOOL_VERSION,
        method="certified_rigid_receptacle_required",
        detail=(
            f"receptacle {receptacle.cavity_id} is certified but no containment probe has been "
            "run against it in this release"
        ),
    )


def evaluate_relation(
    request: RelationRequest,
    *,
    subject: PlacedInstance,
    support_instance: str,
    surfaces: Sequence[Surface],
    receptacle: Receptacle | None = None,
) -> Evidence:
    """Dispatch a relation to its predicate implementation."""
    if request.predicate is Predicate.ON:
        return evaluate_on(
            request,
            subject=subject,
            support_instance=support_instance,
            surfaces=surfaces,
        )
    if request.predicate is Predicate.INSIDE:
        return evaluate_inside(request, receptacle=receptacle, scene_version=request.scene_version)

    # Every other predicate is declared but not implemented in this release. Reported as
    # unsupported rather than not_run: the capability is absent, not merely unexercised, and the
    # capability registry says so too.
    return Evidence(
        gate_id="function.initial_relations",
        status=EvidenceStatus.UNSUPPORTED,
        state_hash=request.scene_version,
        backend=BACKEND,
        tool_version=TOOL_VERSION,
        method="predicate_not_implemented",
        detail=(
            f"predicate {request.predicate.value} is declared in the schema but has no "
            "implementation in this release"
        ),
    )
