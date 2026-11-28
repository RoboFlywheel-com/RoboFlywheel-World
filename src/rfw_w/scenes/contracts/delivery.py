# SPDX-License-Identifier: Apache-2.0
"""The delivery manifest, as a contract rather than as a dict assembled at the end of a run.

It was a plain dict built inline in ``run_m1_acceptance.py``, with twenty-four keys and no schema.
That is how it lost `zones` and `assets_used` once while being rewritten -- a change nobody noticed
until a downstream tool raised a KeyError, because nothing declared which keys were supposed to be
there. A delivery is the artefact this project ships; the shape of the document that describes it
should not be an accident of the last edit.

**Why the readiness fields are nested rather than repeated.** The manifest used to restate the
resolution by hand under names of its own (`requested_readiness`, `achieved_readiness`) while
``ReadinessResolution`` used `requested` and `achieved` -- two names for one fact, hand-copied, in a
file whose whole job is to be checkable. The resolution is embedded whole now, so a key cannot drift
from its source.

**The typed holes are declared.** Several fields are structures the pipeline builds ad hoc --
`placement`, `zones`, `plan`, `loop`. Typing them all would mean rewriting the producers, so they are
``Mapping[str, Any]`` here and the emitter records them under ``x-pvscene-untyped``. A silent open
object is a hole nobody can see; a listed one is a decision, and a test asserts the list is exactly
what it is expected to be so a new hole fails rather than passing unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rfw_w.scenes.contracts.readiness import ReadinessResolution
from rfw_w.scenes.contracts.randomization import DomainRandomizationRecord
from rfw_w.scenes.contracts.tool import Evidence

SCHEMA_VERSION = "pvscene.delivery-manifest/v2"


@dataclass(frozen=True, slots=True)
class RoomFitRecord:
    """How the room came to hold the task cell, as arithmetic rather than as a claim.

    The room is drawn *around* the cell's protected region, so the interesting question about a
    delivery is not only "what room did it get" but "did the room have to grow to take the task,
    and by how much". Without these numbers a reader can see the room and the task separately and
    not the relation between them -- which is the relation the whole ordering is about.

    ``needed_*`` is the protected region plus the standoff the fit keeps from the wall's face on
    both sides; ``sampled_*`` is what the seed then drew within the spec's bounds, never below the
    need. The gap between the two is the margin the room had left over for its own contents.
    """

    protected_width_m: float
    protected_depth_m: float
    needed_width_m: float
    needed_depth_m: float
    sampled_width_m: float
    sampled_depth_m: float
    # Two numbers rather than a pair, because this is a contract that reaches an emitted schema, and
    # the emitter writes a fixed-length tuple as JSON Schema's *tuple* form -- which draft 2020-12
    # no longer has, so the document it produced failed its own meta-schema.
    cell_x_m: float
    cell_y_m: float
    wall_standoff_m: float
    opening_attempts: int
    # What the room had left over once the cell and its standoff were accommodated, per axis. This
    # is the realised margin, and `wall_standoff_m` above is not it -- that records the margin the
    # fit *asks* for, which is a constant of the design.
    size_margin_x_m: float = 0.0
    size_margin_y_m: float = 0.0
    # The shape the seed drew, and the shape the room got, when they differ. Empty means the room
    # is the shape it asked to be; anything else names why it gave ground, because a seed that
    # traded its shape away must not read as one that never wanted a shape.
    shape_family: str = ""
    shape_fallback: str = ""
    # Where the way in is from the task's target, against the bearings the arm can work from. A
    # diagnostic rather than a gate: a hard gate on this refuses 40 % of office seeds.
    door_bearing_deg: float | None = None
    door_arc_margin_deg: float | None = None
    door_in_approach_arc: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protected_width_m": self.protected_width_m,
            "protected_depth_m": self.protected_depth_m,
            "needed_width_m": self.needed_width_m,
            "needed_depth_m": self.needed_depth_m,
            "sampled_width_m": self.sampled_width_m,
            "sampled_depth_m": self.sampled_depth_m,
            "cell_x_m": self.cell_x_m,
            "cell_y_m": self.cell_y_m,
            "wall_standoff_m": self.wall_standoff_m,
            "opening_attempts": self.opening_attempts,
            "size_margin_x_m": self.size_margin_x_m,
            "size_margin_y_m": self.size_margin_y_m,
            "shape_family": self.shape_family,
            "shape_fallback": self.shape_fallback,
            "door_bearing_deg": self.door_bearing_deg,
            "door_arc_margin_deg": self.door_arc_margin_deg,
            "door_in_approach_arc": self.door_in_approach_arc,
        }


@dataclass(frozen=True, slots=True)
class RoomSummary:
    """The shell, as a reader of the manifest needs it.

    Counts and one derived ratio rather than the polygon: the polygon is in `zones`, and repeating it
    here would give a reader two sources for the room's shape.
    """

    vertices: int
    area_m2: float
    area_ratio_to_bounding_box: float
    height_m: float
    openings: int
    #: None when the worker never measured it: a failed render publishes null here rather than a
    #: `false` derived from a default (SPEC-015 R5).
    door_cut: bool | None
    # Absent when the room was not fitted around a task cell -- the room-only path and the older
    # deliveries. Optional rather than defaulted to zeroes, because a fit that did not happen and a
    # fit of nothing are different facts.
    fit: RoomFitRecord | None = None


@dataclass(frozen=True, slots=True)
class RunBudget:
    """How long the run took and what it was allowed.

    Deliberately not `orchestrator.budgets.Budget.to_dict`: that one is the loop's own accounting of
    model requests and rollbacks, which the manifest carries under `loop`. This is the wall clock.
    """

    elapsed_s: float
    deadline_s: float


@dataclass(frozen=True, slots=True)
class VlmRecord:
    """Which model was asked, and whether a credential was configured.

    Recorded beside the critique because the critique's own `model_id` comes from the *response*: a
    gateway that rewrites the name leaves a delivery unable to say which model it was, and a run
    comparing two base models cannot be audited at all. The key itself is never carried -- only the
    name of the environment variable holding it.
    """

    base_url: str
    model: str
    api_key_env: str
    wire: str
    credential_present: bool

    @property
    def configured(self) -> bool:
        return bool(self.model)


@dataclass(frozen=True, slots=True)
class SceneGraphRecord:
    """The canonical graph artefact shipped with the delivery."""

    path: str
    content_hash: str
    state: str = "authored"


@dataclass(frozen=True, slots=True)
class DeliveryManifest:
    """Everything a delivery claims, in one document.

    ``schema_version`` is a field with a string default so the emitted schema pins it to a constant:
    a manifest that declares a version the schema does not describe should not validate.
    """

    scene_id: str
    prompt: str
    seed: int
    requested_backends: tuple[str, ...]
    scene_version: str
    resolution: ReadinessResolution
    gates: tuple[Evidence, ...]
    # Which gates the run declined to measure, and which it declared out of scope. Separate from
    # `gates` because a NOT_RUN evidence entry and a declared gap are different claims, and the
    # verifier reads these two lists to tell a missing measurement from an unsupported feature.
    not_run: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    isaac_version: str = ""
    isaac_artifact_dir: str = ""
    isaac_request_path: str = ""
    room: RoomSummary | None = None
    scene_graph: SceneGraphRecord | None = None
    randomization: DomainRandomizationRecord | None = None
    # The evidence a human reads rather than a checker. Each is the `to_dict` of its own structure,
    # which is why they are open here: `zones` comes from `ZoneGraph`, `plan` from `ScenePlan`, and
    # neither has a contract-level schema yet.
    assets_used: tuple[Mapping[str, Any], ...] = ()
    periphery_named_by_relations: tuple[str, ...] = ()
    compilation: Mapping[str, Any] = field(default_factory=dict)
    external_calls: Mapping[str, Any] = field(default_factory=dict)
    furnishing: Mapping[str, Any] = field(default_factory=dict)
    #: The background branch's opt-in, when a run took it (SPEC-011 requirement 7): the mode, the
    #: gap categories generation was allowed to fill, the policy it was asked to express, and a
    #: summary of the build's own records (the credentials, the rejection pool's ledger, the paths
    #: to `reservations.json` and `constraint_injection.json` beside the build). Empty means the
    #: default path -- the whole periphery retrieved, generation with no allowance anywhere -- which
    #: is also what `request_for()` in the SPEC-009 audit reads a delivery without this block as.
    background: Mapping[str, Any] = field(default_factory=dict)
    #: What the delivery was rendered with (SPEC-008 TASK 3): the lighting profile, the numbers that
    #: reached the renderer, the room they were sized for, and the upstream nodes the mapping could
    #: not carry. Empty on a delivery that skipped the backend -- nothing was rendered, so there is
    #: no lighting to report rather than a default worth guessing at.
    lighting: Mapping[str, Any] = field(default_factory=dict)
    zones: Mapping[str, Any] = field(default_factory=dict)
    plan: Mapping[str, Any] = field(default_factory=dict)
    placement: Mapping[str, Any] = field(default_factory=dict)
    visual_critique: Mapping[str, Any] | None = None
    vlm: VlmRecord | None = None
    agent: VlmRecord | None = None
    loop: Mapping[str, Any] | None = None
    budget: RunBudget | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.scene_version:
            raise ValueError(
                "a manifest must name a scene version; every gate's evidence is bound to it, and a "
                "delivery without one cannot be checked"
            )
        if not self.requested_backends:
            raise ValueError(
                "a manifest must record which backends were requested; inferring them from which "
                "backend gates exist would read a declared out-of-scope gap as a failure"
            )

    @property
    def achieved_readiness(self) -> str:
        return self.resolution.achieved.value

    @property
    def meets_request(self) -> bool:
        return self.resolution.meets_request

    def to_dict(self) -> dict[str, Any]:
        """The document as written to disk.

        ``resolution`` is embedded whole rather than flattened: the flattening was hand-written and
        had already drifted from the type it was copying.
        """
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "prompt": self.prompt,
            "seed": self.seed,
            "requested_backends": list(self.requested_backends),
            "scene_version": self.scene_version,
            "isaac_version": self.isaac_version,
            "isaac_artifact_dir": self.isaac_artifact_dir,
            "isaac_request_path": self.isaac_request_path,
            "resolution": self.resolution.to_dict(),
            "gates": [gate.to_dict() for gate in self.gates],
            "not_run": list(self.not_run),
            "unsupported": list(self.unsupported),
            "room": None if self.room is None else {
                "vertices": self.room.vertices,
                "area_m2": self.room.area_m2,
                "area_ratio_to_bounding_box": self.room.area_ratio_to_bounding_box,
                "height_m": self.room.height_m,
                "openings": self.room.openings,
                "door_cut": self.room.door_cut,
                "fit": None if self.room.fit is None else self.room.fit.to_dict(),
            },
            "scene_graph": None if self.scene_graph is None else {
                "path": self.scene_graph.path,
                "content_hash": self.scene_graph.content_hash,
                "state": self.scene_graph.state,
            },
            "randomization": (
                None if self.randomization is None else self.randomization.to_dict()
            ),
            "assets_used": [dict(entry) for entry in self.assets_used],
            "periphery_named_by_relations": list(self.periphery_named_by_relations),
            "compilation": dict(self.compilation),
            "external_calls": dict(self.external_calls),
            "furnishing": dict(self.furnishing),
            "background": dict(self.background),
            "lighting": dict(self.lighting),
            "zones": dict(self.zones),
            "plan": dict(self.plan),
            "placement": dict(self.placement),
            "visual_critique": None if self.visual_critique is None else dict(self.visual_critique),
            "vlm": None if self.vlm is None else {
                "base_url": self.vlm.base_url,
                "model": self.vlm.model,
                "api_key_env": self.vlm.api_key_env,
                "wire": self.vlm.wire,
                "credential_present": self.vlm.credential_present,
            },
            "agent": None if self.agent is None else {
                "base_url": self.agent.base_url,
                "model": self.agent.model,
                "api_key_env": self.agent.api_key_env,
                "wire": self.agent.wire,
                "credential_present": self.agent.credential_present,
            },
            "loop": None if self.loop is None else dict(self.loop),
            "budget": None if self.budget is None else {
                "elapsed_s": self.budget.elapsed_s,
                "deadline_s": self.budget.deadline_s,
            },
        }
