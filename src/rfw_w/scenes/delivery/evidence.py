# SPDX-License-Identifier: Apache-2.0
"""Typed evidence derived from actual runtime observations."""

from __future__ import annotations
from typing import Any
from rfw_w.scenes.contracts.tool import Evidence, EvidenceStatus
from typing import Mapping

TOOL_VERSION = 'rfw-w.scene-generation/v1'
BACKEND = 'isaac'

def evidence(
    gate_id: str,
    status: EvidenceStatus,
    scene_version: str,
    *,
    method: str,
    measured: float | None = None,
    unit: str | None = None,
    profile: str | None = None,
    detail: str = "",
    witnesses: tuple[str, ...] = (),
    backend: str = BACKEND,
) -> Evidence:
    return Evidence(
        gate_id=gate_id,
        status=status,
        state_hash=scene_version,
        backend=backend,
        tool_version=TOOL_VERSION,
        method=method,
        measured_value=measured,
        unit=unit,
        threshold_profile=profile,
        detail=detail,
        witness_refs=witnesses,
    )


def not_run_gates(scene_version: str) -> list[Evidence]:
    """Every gate this release legitimately does not exercise, reported rather than omitted.

    Each names why. A `not_run` with no reason is indistinguishable from an oversight, and the
    acceptance manual requires the gaps to be stated.
    """
    reasons = {
        "robot.nav": "no controller or navigation stack is wired in this release",
        "robot.ik": "no inverse-kinematics solver is wired; the reach proxy is a front-end filter only",
        "robot.motion": "no motion planner is wired in this release",
        "task.rollout": "no controller rollout exists, so `task_verified` is unreachable",
        "backend.mujoco.load": "dual-backend packaging is an M2 deliverable",
        "backend.mujoco.run": "dual-backend packaging is an M2 deliverable",
        "backend.mujoco.render": "dual-backend packaging is an M2 deliverable",
    }
    return [
        evidence(
            gate_id, EvidenceStatus.NOT_RUN, scene_version,
            method="declared_out_of_scope_for_this_milestone", detail=reason,
        )
        for gate_id, reason in sorted(reasons.items())
    ]


def unsupported_gates(scene_version: str) -> list[Evidence]:
    """Capabilities the schema declares and the local libraries cannot serve."""
    reasons = {
        "articulated.*": "no verified articulated asset exists locally; the coverage audit found no local catalogue",
        "deformable.*": "no verified deformable asset exists locally",
        "INSIDE": "no certified rigid receptacle is present in this scene",
        "FACING": "HSSD front annotations are empty for the furniture used here",
    }
    return [
        evidence(
            f"capability.{name}", EvidenceStatus.UNSUPPORTED, scene_version,
            method="capability_registry_preflight", detail=reason,
            backend="capability",
        )
        for name, reason in sorted(reasons.items())
    ]


def gates_from_worker(report: dict, *, scene_version: str, relation_evidence: Evidence,
                      target_instance: str) -> list[Evidence]:
    """Turn the worker's report into gate evidence.

    ``target_instance`` is passed rather than assumed: the settle gate is about the piece the task
    picks up, and this function used to look for a literal ``"cup"`` -- so a spec whose target was
    anything else would have reported the settle as passing on an object it never found.
    """
    evidence_list: list[Evidence] = [relation_evidence]

    settle = report.get("settle", {})
    objects = settle.get("objects", [])
    subject = next((item for item in objects if item.get("instance_id") == target_instance), None)
    if settle.get("performed") and subject and subject.get("readable"):
        drift = float(subject.get("last_second_drift_m", 0.0))
        drop = float(subject.get("drop_m", 0.0))
        # The manual: the clock must run the requested time, the last second must be stable, and a
        # settled object must be *on its support*, not merely still.
        evidence_list.append(evidence(
            "physics.clock_advanced",
            EvidenceStatus.PASS if abs(float(settle.get("observed_sim_seconds", 0)) - 5.0) < 0.05
            else EvidenceStatus.FAIL,
            scene_version, method="isaac_timeline_step_count",
            measured=float(settle.get("observed_sim_seconds", 0.0)), unit="s",
            profile="engineering_v1",
            detail=f"{settle.get('step_count')} steps at {settle.get('physics_hz')} Hz",
        ))
        stable = drift <= 0.002 and drop < 0.05
        evidence_list.append(evidence(
            "physics.settle_stability",
            EvidenceStatus.PASS if stable else EvidenceStatus.FAIL,
            scene_version, method="isaac_last_second_drift",
            measured=drift, unit="m", profile="engineering_v1",
            detail=f"last-second drift {drift:.6f} m, drop {drop:.6f} m",
        ))
    else:
        evidence_list.append(evidence(
            "physics.settle_stability", EvidenceStatus.NOT_RUN, scene_version,
            method="isaac_last_second_drift", detail="the worker did not perform a settle",
        ))
        evidence_list.append(evidence(
            "physics.clock_advanced", EvidenceStatus.NOT_RUN, scene_version,
            method="isaac_timeline_step_count", detail="the worker did not perform a settle",
        ))

    captures = report.get("captures", [])

    # The view's own declaration (SPEC-015 R3/R4). A view that framed fewer instances than it
    # declared is a named gap for *that view*: the render stands, the delivery continues, and this
    # gate says which view and what it contained. It is not the delivery's death, which is what
    # two such views cost a whole run before the declaration existed.
    required_ok, view_gaps = views_satisfied(report)
    evidence_list.append(evidence(
        "visual.required_views",
        EvidenceStatus.PASS if captures and required_ok else EvidenceStatus.FAIL,
        scene_version, method="isaac_render_channel_audit",
        measured=float(len(captures)), unit="views", profile="engineering_v1",
        detail=(
            "every view is non-blank, shows no invalid depth where depth was requested, no "
            "required target is missing, and every view met its declared minimum"
            if captures and required_ok
            else "at least one view is blank, has invalid depth where depth was requested, is "
                 "missing a required target, or missed its declared minimum"
                 + (": " + ", ".join(view_gaps) if view_gaps else "")
        ),
        witnesses=tuple(capture.get("rgb_path", "") for capture in captures) + view_gaps,
    ))

    if report.get("status") == "passed" and report.get("instance_id_map"):
        evidence_list.append(evidence(
            "backend.isaac.render", EvidenceStatus.PASS, scene_version,
            method="isaac_worker_report", measured=float(len(captures)), unit="views",
            profile="backend_smoke_v1",
            detail=f"isaac {report.get('isaac_version')}, instance map from {report.get('instance_id_map_source')}",
        ))
        evidence_list.append(evidence(
            "backend.isaac.run", EvidenceStatus.PASS, scene_version,
            method="isaac_worker_report", profile="backend_smoke_v1",
            detail="the worker ran and reported a pass",
        ))
        evidence_list.append(evidence(
            "backend.isaac.load", EvidenceStatus.PASS, scene_version,
            method="isaac_worker_report", profile="backend_smoke_v1",
            detail="the stage loaded and rendered",
        ))
    else:
        for gate in ("backend.isaac.load", "backend.isaac.run", "backend.isaac.render"):
            evidence_list.append(evidence(
                gate, EvidenceStatus.FAIL, scene_version, method="isaac_worker_report",
                detail=report.get("error_detail") or "the worker did not report a pass",
            ))
    return evidence_list


def view_usable(capture: Mapping[str, Any]) -> bool:
    """Non-blank, target not known-absent, and no invalid depth *where depth was requested*.

    The depth clause applies only to a capture that saved depth. Requiring it of every view failed
    a room-context view that deliberately does not save depth -- its valid fraction is zero because
    nothing was recorded, and the gate read that as a broken image and blocked a delivery whose
    renders were fine. A gate that fires on a view doing exactly what it was asked to do is worse
    than no gate: it teaches the reader to ignore the gate.
    """
    if not capture.get("nonblank"):
        return False
    if capture.get("target_visible") is False:
        return False
    if capture.get("depth_path") and float(capture.get("depth_valid_fraction", 0.0)) <= 0.1:
        return False
    return True


def views_satisfied(report: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Whether every delivered view is usable *and* met its own declaration, gaps named.

    Two things, deliberately separate: usable is about the picture (nonblank, depth, the target),
    the declaration is about the *promise* the camera made (SPEC-015 R3). A view that framed fewer
    instances than it declared is named in the report's `view_gaps` and this gate fails *by name*
    -- the delivery continues, which is the whole point of R4.
    """
    captures = report.get("captures") or ()
    coverage = {str(entry.get("camera_id")): entry for entry in report.get("view_coverage") or ()}
    gaps = tuple(str(gap) for gap in report.get("view_gaps") or ())
    for capture in captures:
        if not view_usable(capture):
            return False, gaps
        entry = coverage.get(str(capture.get("camera_id")))
        if entry is not None and not bool(entry.get("obligation_met", True)):
            return False, gaps
    return bool(captures), gaps


def door_cut_from_report(worker_report: Mapping[str, Any]) -> bool | None:
    """Whether a door was cut, or None when the run that produced this report never measured it.

    `bool(report.get("openings_cut"))` read a failed report's default as `false` and published it,
    so a delivery whose render never ran said its room had no door -- an assertion about geometry
    nothing had looked at (SPEC-015 R5).
    """
    cut = worker_report.get("openings_cut")
    return None if cut is None else bool(cut)
