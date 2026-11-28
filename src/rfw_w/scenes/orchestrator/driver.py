# SPDX-License-Identifier: Apache-2.0
"""Driving the bounded loop: segments, the escalation policy, and the two metrics.

:mod:`rfw_w.scenes.orchestrator.loop` is the interface — it decides when a run may commit, when it
has made no progress, and when a failure needs a semantic change. It deliberately does not call
models and does not know what a scene is. This module supplies both halves of that policy and drives
the loop to a terminal state.

**Why segments.** ``Orchestrator.run`` stops at ``AWAITING_MODEL`` rather than calling a model,
because spending a request is a policy the caller owns. The driver treats that stop as a segment
boundary: it may spend **one** request there, then starts a fresh orchestrator from the current scene
version and carries on. A fresh orchestrator rather than a resumed one because ``Orchestrator.run``
begins from ``pending = list(jobs)``, and re-running the plan's earlier jobs would re-commit a scene
that has not changed — which ``LoopState.advance`` correctly refuses, turning a legitimate resume
into a crash.

**What counts as progress.** The executable design is explicit (§10): progress is judged by the hard
failure set shrinking, a residual falling, or the pool of untried candidates changing — *"not merely
because the LLM says it is fixed"*. So every commit is measured against the failure set before and
after, and an atom that commits a change without shrinking it is recorded as **no improvement**. That
is what lets the no-progress rule terminate a run that is thrashing instead of letting it spend its
budget looking busy.

**What the host must provide.** The loop is generic and the pipeline is not, so the pipeline is
behind a small protocol: build a scene from a plan, run the deterministic gates, run the backend,
run the critique, and say whether the requested level is met. Everything else — the ladder, the
checkpoints, the budget, the metrics — is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from rfw_w.scenes.agents.replanner import (
    IMPLEMENTED_ACTIONS,
    SemanticPatch,
    application_requests,
    replan,
    scene_summary,
)
from rfw_w.scenes.contracts.tool import (
    EvidenceLedger,
    EvidenceStatus,
    FailureCode,
    ToolStatus,
)
from rfw_w.scenes.orchestrator.budgets import Budget
from rfw_w.scenes.orchestrator.checkpoints import CheckpointName, CheckpointStack
from rfw_w.scenes.orchestrator.escalation import (
    ATOMS,
    Atom,
    Ladder,
    Track,
    atom_for_stage,
    stage_names,
)
from rfw_w.scenes.orchestrator.events import FailureEvent, events_from_ledger
from rfw_w.scenes.orchestrator.loop import (
    Job,
    LoopState,
    Orchestrator,
    OrchestratorResult,
    StageOutcome,
    TerminalReason,
)
from rfw_w.scenes.orchestrator.repair import RepairAction, RepairOutcome, apply as apply_repair
from rfw_w.scenes.solvers.furnish import ScenePlan

SCHEMA_VERSION = "pvscene.driver/v1"


class DriverError(RuntimeError):
    """The driver or its host is misconfigured."""


class RepairUnavailable(RuntimeError):
    """A proposed rebuild has exhausted a finite resource, such as asset candidates."""


class LoopHost(Protocol):
    """What the driver needs from the pipeline it drives.

    Every method is an observation or a rebuild; none of them decides anything about escalation. The
    split is deliberate — the pipeline knows how to build and measure a scene, and the driver knows
    when to repair, roll back and escalate, and neither has to know the other's rules.
    """

    def current_plan(self) -> ScenePlan: ...

    def current_scene_version(self) -> str: ...

    def rebuild(self, plan: ScenePlan) -> str:
        """Build the scene from ``plan`` and return its new version.

        Must return a *different* version when the plan differs, and the same version when it does
        not: the loop's optimistic-concurrency check and its no-progress rule both read this.
        """
        ...

    def evidence_ledger(self) -> EvidenceLedger: ...

    def run_deterministic_gates(self, scene_version: str) -> None: ...

    def run_backend(self, scene_version: str) -> bool:
        """Render and settle. Returns whether the backend stage passed."""
        ...

    def run_critique(self, scene_version: str) -> bool:
        """Ask the visual critic. Returns whether the visual gate passed."""
        ...

    def last_critique(self) -> Any | None:
        """The critique most recently obtained, so the driver can read its repairs.

        The gate's pass/fail is not enough to choose a repair: the design has the critic say *what*
        should change, and which track the ladder starts on is decided by that. A driver given only
        a boolean would answer every visual complaint by moving furniture first.
        """

    def last_visual_gate(self) -> Any | None:
        """The verdict that came with that critique.

        Which criteria are below threshold is what decides the ladder's starting track -- a lighting
        suggestion about a criterion that passed is not a reason to start on the lighting track --
        so the driver needs the verdict, not just the suggestions. It is the host's verdict rather
        than one the driver re-evaluates, because the host is what chose the thresholds.
        """
        ...

    def gates_settled(self) -> bool:
        """Whether the loop has nothing left that would move it.

        Named for what it means rather than for the readiness ladder: a run that declined to render
        at all has not met its requested level, but it also has nothing further it can measure, and
        a loop that waited for an unreachable level would end every such run in a spurious
        exhaustion. What this must never do is report settled while a gate the configuration *can*
        measure is failing.
        """
        ...

    def wall_thickness_m(self) -> float: ...

    def synset_vocabulary(self) -> tuple[str, ...]: ...

    def goal(self) -> str: ...

    def achieved_level(self) -> str: ...


def _code_of(signature: str) -> str:
    """The typed failure code inside a stage signature.

    A stage's signature is ``<stage>:<code>``, and which stage reported a failure is not part of the
    failure's identity: a wall embedding is the same problem whether the gate found it or a repair
    rung failed to fix it. Reading the code back out is what lets the ladder continue on the failure
    it was repairing instead of treating each rung's own failure as a new problem and restarting at
    L0 forever.
    """
    return signature.rpartition(":")[2]


@dataclass(frozen=True, slots=True)
class LoopRun:
    """The loop's verdict, with the numbers the design measures it by."""

    terminal: str
    scene_version: str
    achieved_level: str
    segments: int
    events: tuple[FailureEvent, ...]
    patches: tuple[SemanticPatch, ...]
    removed: tuple[str, ...]
    checkpoints: tuple[dict[str, Any], ...]
    iterations: int
    budget: Mapping[str, Any]
    ladder: Mapping[str, Any]
    host_calls: Mapping[str, int]
    notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.terminal == TerminalReason.READY.value

    @property
    def agent_intervention_rate(self) -> float:
        """§28.2: LLM replans over scenes. One scene per run, so this is the replan count."""
        return float(len(self.patches))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "terminal": self.terminal,
            "ready": self.ready,
            "scene_version": self.scene_version,
            "achieved_level": self.achieved_level,
            "segments": self.segments,
            "iterations": self.iterations,
            "events": [event.to_dict() for event in self.events],
            "patches": [patch.to_dict() for patch in self.patches],
            "removed": list(self.removed),
            "checkpoints": list(self.checkpoints),
            "budget": dict(self.budget),
            "ladder": dict(self.ladder),
            "host_calls": dict(self.host_calls),
            "agent_intervention_rate": self.agent_intervention_rate,
            "notes": list(self.notes),
        }


# The stages every run has, in order. There is no "compile" stage: the host builds the initial scene
# before the driver is called, and the repair handlers do every rebuild after that. A compile stage
# whose only job was to rebuild what was just built would commit a version it did not change.
BASE_STAGES: tuple[tuple[str, bool], ...] = (
    ("gate.geometry", True),
    ("gate.task", True),
    ("backend.render", False),
    ("visual.critique", False),
    ("finalize", True),
)


@dataclass(slots=True)
class Driver:
    """Runs the loop against a host until it terminates, and records why."""

    host: LoopHost
    budget: Budget
    replan_config: Any = None
    # Whether a lighting/material repair is allowed to re-render and re-critique. Off by default
    # because a second render is a real cost and the caller should opt into it; when off, the
    # appearance track is still tried and its revalidation is reported as skipped.
    allow_appearance_revalidation: bool = True
    appearance_direction: str = "brighter"
    # How many distinct failures may be given a local repair job. Note that the orchestrator charges
    # its ``max_subgraph_rollbacks`` allowance for *every* local repair rather than only for a real
    # subgraph rollback -- ``loop.py`` cannot tell one repair from another, so it bounds them all
    # with the one counter. A caller sizing that allowance to the design's "at most 2 subgraph
    # rollbacks" therefore truncates the ladder after two failures, which is a confusing way to
    # fail. ``run`` refuses the configuration instead.
    max_repair_jobs: int = 6
    ladder: Ladder = field(default_factory=Ladder)
    checkpoints: CheckpointStack = field(default_factory=CheckpointStack)
    events: list[FailureEvent] = field(default_factory=list)
    patches: list[SemanticPatch] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    host_calls: dict[str, int] = field(default_factory=dict)

    # -- state the segment hooks read ------------------------------------------------------------

    _active: FailureEvent | None = field(default=None, repr=False)
    _by_signature: dict[str, FailureEvent] = field(default_factory=dict, repr=False)
    _track_order: dict[str, tuple[Track, ...]] = field(default_factory=dict, repr=False)
    _current: RepairOutcome | None = field(default=None, repr=False)
    _segments: int = 0
    _iterations: int = 0
    _no_replanner: bool = field(default=False, repr=False)

    # -- public entry point ----------------------------------------------------------------------

    def run(self, *, max_segments: int = 8) -> LoopRun:
        """Drive to a terminal state.

        ``max_segments`` bounds the number of model requests the run can make, independently of the
        budget: a driver that could segment forever would turn a bounded loop into an unbounded one
        whose bound lives in a different file.
        """
        if self.budget.max_subgraph_rollbacks < self.max_repair_jobs:
            raise DriverError(
                f"the budget allows {self.budget.max_subgraph_rollbacks} local repairs but this "
                f"driver may need {self.max_repair_jobs}; loop.py charges that allowance once per "
                "repair job, so a small value silently truncates the escalation ladder. Raise "
                "max_subgraph_rollbacks, or lower max_repair_jobs if fewer failures are expected."
            )

        terminal = TerminalReason.FAILED
        while True:
            if self.budget.expired():
                terminal = TerminalReason.BUDGET_EXHAUSTED
                self.notes.append("the deadline expired before another segment could start")
                break
            result = self._run_segment()
            if result.terminal is TerminalReason.READY:
                terminal = TerminalReason.READY
                break
            if result.terminal is not TerminalReason.AWAITING_MODEL:
                terminal = result.terminal
                self.notes.append(
                    f"the segment ended {result.terminal.value}: {len(result.failed_signatures)} "
                    f"failed signature(s), {len(result.committed)} commit(s)"
                )
                break
            if self._segments >= max_segments:
                terminal = TerminalReason.BUDGET_EXHAUSTED
                self.notes.append(f"the {max_segments}-segment limit is reached")
                break
            if not self._spend_model_request():
                # A ladder that ran out with nothing to escalate to is a repair exhaustion, not a
                # budget one. Reporting it as `budget_exhausted` would say the run ran out of time
                # when it ran out of repairs.
                terminal = (
                    TerminalReason.REPAIR_EXHAUSTED
                    if self._no_replanner else TerminalReason.BUDGET_EXHAUSTED
                )
                break

        return self._result(terminal)

    # -- one segment -----------------------------------------------------------------------------

    def _run_segment(self) -> OrchestratorResult:
        self._segments += 1
        # The repair stages are registered as handlers but deliberately **not** in the plan. A rung
        # is reached only through `repair_for`, which is what makes the ladder an escalation policy
        # rather than a list of jobs that all run in order -- put them in `stages` and every atom
        # fires on every run, including the ones for failures that never happened.
        stages = BASE_STAGES
        handlers = self._handlers()
        orchestrator = Orchestrator(
            budget=self.budget,
            stages=stages,
            handlers=handlers,
            initial_scene_version=self.host.current_scene_version(),
        )
        result = orchestrator.run(
            all_evidence_current=lambda version: self.host.gates_settled(),
            on_commit=self._on_commit,
            repair_for=self._repair_for,
            needs_semantic_change=self._needs_semantic_change,
        )
        self._iterations += result.iterations
        return result

    def _handlers(self) -> dict[str, Callable[[Job, LoopState], StageOutcome]]:
        handlers: dict[str, Callable[[Job, LoopState], StageOutcome]] = {
            "gate.geometry": self._stage_gate_geometry,
            "gate.task": self._stage_gate_task,
            "backend.render": self._stage_render,
            "visual.critique": self._stage_critique,
            "finalize": self._stage_finalize,
        }
        for stage in stage_names():
            handlers[stage] = self._stage_repair
        return handlers

    # -- observation stages ----------------------------------------------------------------------

    def _stage_gate_geometry(self, job: Job, state: LoopState) -> StageOutcome:
        self.host.run_deterministic_gates(state.scene_version)
        return self._gate_outcome(job, state, ("geometry.",), "the geometry gates")

    def _stage_gate_task(self, job: Job, state: LoopState) -> StageOutcome:
        self.host.run_deterministic_gates(state.scene_version)
        return self._gate_outcome(job, state, ("function.", "semantics."), "the task gates")

    def _gate_outcome(
        self, job: Job, state: LoopState, prefixes: Sequence[str], label: str
    ) -> StageOutcome:
        """Turn the ledger's state into a stage outcome.

        The failing gate's own typed code travels with the outcome, so the ladder picks a strategy
        from the failure rather than from prose. When several gates fail, the first in sorted order
        is reported: the ladder repairs one failure at a time and re-measures, which is cheaper and
        more legible than repairing all of them at once.
        """
        ledger = self.host.evidence_ledger()
        raised = events_from_ledger(ledger, scene_version=state.scene_version)
        relevant = [
            event for event in raised
            if any(event.source_gate.startswith(prefix) for prefix in prefixes)
        ]
        self._remember(raised)
        if not relevant:
            return StageOutcome(
                job_id=job.job_id, status=ToolStatus.PASSED,
                base_scene_version=state.scene_version, read_only=True, detail=f"{label} passed",
            )
        event = relevant[0]
        self._active = event
        return StageOutcome(
            job_id=job.job_id, status=ToolStatus.FAILED, base_scene_version=state.scene_version,
            failure_code=event.failure_code, detail=f"{event.source_gate}: {event.detail}"[:400],
            read_only=True,
        )

    def _stage_render(self, job: Job, state: LoopState) -> StageOutcome:
        self.host_calls["render"] = self.host_calls.get("render", 0) + 1
        passed = self.host.run_backend(state.scene_version)
        ledger = self.host.evidence_ledger()
        self._remember(events_from_ledger(ledger, scene_version=state.scene_version))
        if passed:
            return StageOutcome(
                job_id=job.job_id, status=ToolStatus.PASSED, base_scene_version=state.scene_version,
                read_only=True, detail="the backend stage passed",
            )
        # A backend that failed to load or crashed is an infrastructure failure, and the loop
        # retries it on its own allowance rather than spending a model request to replan a scene
        # that may be fine.
        return StageOutcome(
            job_id=job.job_id, status=ToolStatus.FAILED, base_scene_version=state.scene_version,
            failure_code=FailureCode.RENDER_FAILED, detail="the backend stage did not pass",
            read_only=True,
        )

    def _stage_critique(self, job: Job, state: LoopState) -> StageOutcome:
        self.host_calls["critique"] = self.host_calls.get("critique", 0) + 1
        passed = self.host.run_critique(state.scene_version)
        ledger = self.host.evidence_ledger()
        self._remember(events_from_ledger(ledger, scene_version=state.scene_version))
        if passed:
            return StageOutcome(
                job_id=job.job_id, status=ToolStatus.PASSED, base_scene_version=state.scene_version,
                read_only=True, detail="the visual gate passed",
            )
        self._activate(FailureCode.VISUAL_GATE_FAILED)
        self._record_critic_track()
        entry = ledger.get("visual.critique", state.scene_version)
        status = entry.status if entry is not None else EvidenceStatus.NOT_RUN
        detail = entry.detail[:400] if entry is not None else "no critique was recorded"
        if status is not EvidenceStatus.FAIL:
            # A critique that did not run is not a failed one. `not_run` never satisfies a gate, so
            # the readiness ladder already refuses to let an absent critique pass anything -- and
            # escalating on it instead would spend a replan repairing a scene whose only problem is
            # that no model was configured to look at it. That is a real path: a run without
            # `PVSCENE_VLM_MODEL` took the ladder all the way to a replan before this.
            return StageOutcome(
                job_id=job.job_id, status=ToolStatus.PASSED, base_scene_version=state.scene_version,
                read_only=True, detail=f"the visual gate did not run ({status.value}): {detail}",
            )
        return StageOutcome(
            job_id=job.job_id, status=ToolStatus.FAILED, base_scene_version=state.scene_version,
            failure_code=FailureCode.VISUAL_GATE_FAILED, detail=detail, read_only=True,
        )

    def _stage_finalize(self, job: Job, state: LoopState) -> StageOutcome:
        return StageOutcome(
            job_id=job.job_id, status=ToolStatus.PASSED, base_scene_version=state.scene_version,
            read_only=True, detail=f"achieved {self.host.achieved_level()}",
        )

    # -- the repair stage: one atom, applied, re-measured ----------------------------------------

    def _stage_repair(self, job: Job, state: LoopState) -> StageOutcome:
        atom = atom_for_stage(job.stage)
        if atom is None:
            raise DriverError(f"{job.stage} is not a repair stage")
        event = self._active
        if event is None:
            return StageOutcome(
                job_id=job.job_id, status=ToolStatus.FAILED, base_scene_version=state.scene_version,
                failure_code=FailureCode.NO_PROGRESS, detail="a repair ran with no active failure",
                read_only=False,
            )
        # **One job climbs the whole ladder for one failure, and is all-or-nothing.**
        #
        # Climbing here rather than returning one rung at a time is forced by the loop's shape: a
        # gate job runs once and is then popped, so nothing would ever re-enter `gate.geometry` to
        # notice that a rung had fixed it. One job is therefore "resolve this failure with the local
        # ladder", and the ladder still decides which rungs it may use and in what order.
        #
        # All-or-nothing because the executable design §10 accepts a soft patch only when it does
        # not harm a hard gate and improves a defined quality term, *"otherwise roll back"*. A rung
        # that changes the scene without clearing the failure is rolled back rather than committed,
        # so a run that cannot fix something locally does not leave a half-repaired scene behind it
        # -- and the version stays consistent with the evidence the loop has recorded.
        original_plan = self.host.current_plan()
        original_version = self.host.current_scene_version()
        sequence: list[str] = []

        while atom is not None:
            action = self._action_for(event, atom, job)
            outcome = apply_repair(
                self.host.current_plan(), action,
                wall_thickness_m=self.host.wall_thickness_m(),
                failure_code=event.failure_code.value if event.failure_code else "",
            )
            self.host_calls[f"repair:{atom.value}"] = (
                self.host_calls.get(f"repair:{atom.value}", 0) + 1
            )
            before = self._failure_set()
            sequence.append(atom.value)

            if not outcome.changed:
                # An atom that cannot act says so. The rung is recorded and the ladder moves up,
                # rather than the atom inventing a change so that it can report success.
                self.ladder.record(
                    event, atom, resolved=False, improved=False,
                    detail="; ".join(outcome.notes) or "no change was possible",
                )
                self.notes.append(f"{atom.value} could not act: {'; '.join(outcome.notes)}")
                atom = self._next_local_rung(event, atom)
                continue

            previous_plan = self.host.current_plan()
            try:
                new_version = self.host.rebuild(outcome.plan)
            except RepairUnavailable as error:
                # The host may have started adopting the proposal before discovering the gap.
                # Restore the last build; no new measurement or successful repair is claimed.
                self.host.rebuild(previous_plan)
                detail = str(error)[:400]
                self.ladder.record(event, atom, resolved=False, improved=False, detail=detail)
                self.notes.append(f"{atom.value} could not rebuild: {detail}")
                atom = self._next_local_rung(event, atom)
                continue
            if new_version == original_version:
                # The atom reported a change the scene version cannot see, which means the plan
                # carries a field its serialisation does not. Committing it would be refused by the
                # orchestrator -- a commit must produce a new version -- and the loop would crash
                # rather than repair. Recording it as no change is the honest reading *and* the
                # safe one: a field that is not hashed is a field no gate's evidence depends on.
                self.notes.append(
                    f"{atom.value} reported a change but the scene version did not move "
                    f"({new_version[:12]}); the plan has a field its to_dict does not carry"
                )
                self.ladder.record(
                    event, atom, resolved=False, improved=False,
                    detail="the scene version did not change",
                )
                atom = self._next_local_rung(event, atom)
                continue
            self._revalidate(outcome, new_version)
            after = self._failure_set()
            # The failure cleared when its own gate stops failing -- read from the gate the event
            # came from, not from a signature comparison, because the gate is what was measured.
            resolved = event.source_gate not in after
            improved = len(after) < len(before)
            self.ladder.record(
                event, atom, resolved=resolved, improved=improved,
                detail=f"{event.source_gate}: failure set {len(before)} -> {len(after)}",
            )

            if resolved:
                self._current = outcome
                return StageOutcome(
                    job_id=job.job_id, status=ToolStatus.PASSED,
                    base_scene_version=state.scene_version,
                    detail=f"{' then '.join(sequence)} cleared {event.source_gate}"[:400],
                    read_only=False,
                    proposed_delta={"atom": atom.value, "plan": "rebuilt"},
                    touched_ids=tuple(action.targets),
                )
            if not improved:
                self.notes.append(
                    f"{atom.value} changed the scene to {new_version[:12]} but the failure set did "
                    f"not shrink ({len(before)} -> {len(after)}); recorded as no improvement"
                )
            atom = self._next_local_rung(event, atom)

        # Every rung was tried and none of them cleared it. Restore the scene the job started from
        # so that the failure escalates from a consistent state rather than from whichever
        # half-repair happened to run last.
        self.host.rebuild(original_plan)
        if self.host.current_scene_version() != original_version:
            self.notes.append(
                "restoring the pre-repair plan produced a different scene version, which means the "
                "plan is not the whole of what the scene is built from"
            )
        self._current = None
        return StageOutcome(
            job_id=job.job_id, status=ToolStatus.FAILED, base_scene_version=state.scene_version,
            failure_code=event.failure_code,
            detail=(
                f"the local ladder is exhausted for {event.source_gate} "
                f"(tried {', '.join(sequence)}); the scene was restored and the failure escalates"
            )[:400],
            read_only=False,
        )

    def _in_plan(self, entity_id: str) -> bool:
        plan = self.host.current_plan()
        return bool(plan.furniture_for(entity_id) or plan.surface_for(entity_id))

    def _action_for(self, event: FailureEvent, atom: Atom, job: Job) -> RepairAction:
        """Build the action for this rung, narrowed to one target where the event names several.

        The target is the first one the *plan* contains, not the first one the gate named. A gate
        names every instance it looked at -- `geometry.in_bounds` lists all twelve -- and the
        alphabetically first of those is the cup, which is not a plan entity at all. Repairing
        therefore reported "no target in the plan" for an event that named eight pieces that are in
        it, and the L0 resample did nothing on a scene it had plenty to do on.
        """
        # The critic's own pairing first. `event.targets` is a *set* of objects, so the best it can
        # say is which of them the plan can move -- and it has to say it by taking the first one,
        # which means the alphabet chose the target. The critique still holds the pairing the
        # suggestion was made with, so when this atom is the one a suggestion named, the object it
        # named is the object the repair goes to.
        target = self._suggested_target_for(event, atom)
        if target is None:
            target = next(
                (name for name in event.targets if self._in_plan(name)), None
            ) if event.targets else None
        if target is None and event.targets:
            self.notes.append(
                f"none of {list(event.targets)} is an entity the plan can move; the repair is "
                "untargeted"
            )
        parameters: dict[str, Any] = {}
        if atom is Atom.OPTIMISE_POSE and event.failure_code:
            parameters["strategy"] = ""  # the repair module picks from the typed code
        if atom is Atom.ADJUST_LIGHTING_MATERIAL:
            # Which way to move the lighting comes from the critic that complained, when it said;
            # otherwise brighter, because a render that reads flat is the common complaint and the
            # opposite one is visible as blown highlights in the same critique.
            parameters["direction"] = self._appearance_direction()
        return self._repair_action(event, atom, target, parameters)

    def _suggested_target_for(self, event: FailureEvent, atom: Atom) -> str | None:
        """The object the critic named for this atom, when this is a complaint it raised.

        Only for ``origin == "critic"``: a geometry gate's failure is measured against the plan,
        and the last critique may describe a scene that has since moved. Reading a stale critique's
        target for a geometric event would aim a repair at an object the failure never mentioned.
        """
        if event.origin != "critic":
            return None
        from rfw_w.scenes.agents.visual_critic import critics_atom_for

        critique = self.host.last_critique()
        for item in getattr(critique, "suggestions", ()) or ():
            try:
                candidate = critics_atom_for(item.action)
            except (KeyError, ValueError):
                continue
            if candidate is atom:
                return item.target_id
        return None

    def _repair_action(
        self, event: FailureEvent, atom: Atom, target: str | None, parameters: Mapping[str, Any]
    ) -> RepairAction:
        from rfw_w.scenes.orchestrator.repair import action_for_event

        return action_for_event(
            event, atom=atom,
            source="critic" if event.origin == "critic" else "gate",
            target=target, parameters=parameters, rationale=event.detail,
        )

    def _appearance_direction(self) -> str:
        """Which way to move the lighting.

        Brighter by default, because a render that reads flat is the common complaint and the
        opposite complaint arrives with its own evidence -- blown highlights are visible in the same
        critique that would be asking for less light. The direction is a parameter rather than a
        decision so a caller who knows better can say so.
        """
        return self.appearance_direction

    def _revalidate(self, outcome: RepairOutcome, new_version: str) -> None:
        """Re-measure invalidated stages and bind cheap checks to the new global version.

        Appearance does not change geometry, but the ledger keys every check by the global
        scene hash. Re-running the cheap checks avoids inventing a pass for an unmeasured hash.
        Simulator and model calls still follow the atom's explicit invalidation set.
        """
        invalidated = outcome.invalidated_gates
        if not invalidated:
            return
        self.host.run_deterministic_gates(new_version)
        render_invalidated = any(gate.startswith("backend.") or gate == "visual.required_views"
                                 for gate in invalidated)
        visual_invalidated = "visual.critique" in invalidated
        if not render_invalidated and not visual_invalidated:
            return
        # Only re-measure what was actually measured. A geometry repair early in a run invalidates
        # the render and the critique in the sense that any evidence for them is now void -- but if
        # there is no such evidence yet, there is nothing to void and re-rendering would spend the
        # run's most expensive stage to produce a picture of a scene that is about to change again.
        # This is what keeps the geometry phase dense and the render sparse.
        ledger = self.host.evidence_ledger()
        # `latest_for`, not a lookup at one version: the question is whether a render has *ever*
        # happened, for any scene. If it has, re-measuring is worth the render; if it has not, the
        # run is spending its most expensive stage to photograph a scene that is about to change.
        if ledger.latest_for("visual.required_views") is None and ledger.latest_for("visual.critique") is None:
            self.notes.append(
                f"{outcome.atom.value} invalidated the render and critique, but neither had been "
                "measured yet, so nothing was re-measured"
            )
            return
        if not self.allow_appearance_revalidation:
            self.notes.append(
                f"the render and critique were invalidated by {outcome.atom.value} but revalidation "
                "is disabled, so the visual evidence is stale and cannot pass"
            )
            return
        self.host_calls["render"] = self.host_calls.get("render", 0) + 1
        if self.host.run_backend(new_version) and visual_invalidated:
            self.host_calls["critique"] = self.host_calls.get("critique", 0) + 1
            self.host.run_critique(new_version)

    # -- commit, ladder and model hooks ----------------------------------------------------------

    def _on_commit(self, job: Job, outcome: StageOutcome) -> str:
        """A repair's delta has been accepted: the scene it built is the current one."""
        if self._current is None:
            raise DriverError(f"{job.job_id} committed without a repair outcome")
        return self.host.current_scene_version()

    def _repair_for(self, signature: str) -> Job | None:
        """The next rung for a failure, or ``None`` when the ladder is exhausted.

        This is the executable design §10's ``repair_policy.next_untried(failure_signature)``, with
        the two policies the design fixes wrapped around it: the rung must be admissible for the
        event, and it must not be tried while a cheaper admissible rung is untried -- which the
        ladder enforces, so a critic's suggestion cannot promote itself.
        """
        event = self._event_for_signature(signature)
        if event is None:
            return None
        self._active = event
        for track in self._tracks(event):
            if track is Track.APPEARANCE and not self._appearance_is_admissible(event):
                continue
            atom = self.ladder.next_untried(event, track=track)
            if atom is None:
                continue
            info = ATOMS[atom]
            if info.needs_model:
                # L5 is not a job: the loop stops and the driver spends a request. Returning a job
                # here would put a model call inside a stage, where the budget cannot see it.
                #
                # `continue` rather than `return None`, so a track whose only untried rung is the
                # replan does not take the other track down with it: the appearance track has a
                # single rung for a critic complaint, so it reaches this line every time it is
                # tried second, and returning here made every rung after it unreachable.
                continue
            if info.track is Track.GEOMETRIC and not event.targets:
                # An untargeted geometric atom would rebuild the whole scene and be recorded as a
                # local repair. The ladder has no target to work on, so this level cannot help.
                continue
            return Job(job_id=f"{info.stage}@{len(self.ladder.attempts)}", stage=info.stage,
                       read_only=False, scope_ids=event.targets)
        return None

    def _appearance_is_admissible(self, event: FailureEvent) -> bool:
        """Whether the appearance track may be used for this failure.

        Only for a complaint the critic raised, and only where the ladder's appearance table has a
        rung. A geometric failure is never answered by moving the lighting: the atom cannot fix it,
        and trying would spend a render and a critique discovering that.
        """
        if event.origin != "critic":
            return False
        from rfw_w.scenes.orchestrator.escalation import admissible_atoms

        return bool(admissible_atoms(event, track=Track.APPEARANCE))

    def _needs_semantic_change(self, outcome: StageOutcome) -> bool:
        """Whether this failure has to become a replan.

        True only when the ladder has nothing left on either track. The design's rule is that the
        model is woken after local recovery is exhausted, so a failure that still has an untried
        rung must not reach here -- and the check is made against the ladder rather than assumed.
        """
        event = self._event_for_signature(outcome.signature)
        if event is None:
            return True
        for track in self._tracks(event):
            if track is Track.APPEARANCE and not self._appearance_is_admissible(event):
                continue
            atom = self.ladder.next_untried(event, track=track)
            # A rung that needs a model is not a local rung, so an untried L5 does not keep the
            # failure away from the replanner -- it is what sends it there.
            if atom is not None and not ATOMS[atom].needs_model:
                return False
        return True

    def _next_local_rung(self, event: FailureEvent, atom: Atom) -> Atom | None:
        """The next rung on this atom's track that a local job can actually perform.

        Stops at L5 rather than returning it, because a model request is spent by the driver at a
        segment boundary and not inside a stage. Returning it here would record an agent replan in
        the ladder's history that never happened, and the model-request metric is computed from
        exactly that history.
        """
        following = self.ladder.next_untried(event, track=ATOMS[atom].track)
        if following is not None and ATOMS[following].needs_model:
            return None
        return following

    def _spend_model_request(self) -> bool:
        """Spend one request on a replan, apply what survives, and report whether to continue."""
        if self.replan_config is None:
            self.notes.append(
                "the ladder is exhausted and no replanner is configured, so the failure stops here"
            )
            self._no_replanner = True
            return False
        if not self.budget.permits_model_request():
            self.notes.append("the ladder is exhausted and the model-request budget is spent")
            return False

        event = self._active
        summary = scene_summary(
            self.host.current_plan(), goal=self.host.goal(),
            stage=event.source_gate if event else "unknown",
            active_failure=(event.to_dict() if event else {"kind": "UNKNOWN"}),
            global_flags={"ladder_exhausted": True},
            suggested_repairs=self._tried_repair_verbs(),
            known_targets=tuple(
                sorted(set(self.host.current_plan().entity_ids) & set(event.targets))
                if event else ()
            ),
        )
        patch = replan(
            self.replan_config, plan=self.host.current_plan(), summary=summary,
            allowed_actions=IMPLEMENTED_ACTIONS,
            synset_vocabulary=self.host.synset_vocabulary(),
            on_request=lambda: self.budget.charge_model_request(purpose="replan"),
        )
        self.patches.append(patch)
        self.host_calls["replan"] = self.host_calls.get("replan", 0) + 1

        if patch.terminates:
            self.notes.append(
                "the replanner reported the task cannot be met: " + (patch.diagnosis or "no reason given")
            )
            return False
        if not patch.usable:
            self.notes.append(
                "the replan produced no usable operation: "
                + "; ".join(item.reason for item in patch.rejected)[:300]
            )
            return False

        plan = self.host.current_plan()
        for entity_id, request in application_requests(patch, plan):
            if request is None:
                plan = plan.without(entity_id)
                self.removed.append(entity_id)
            else:
                plan = plan.replaced_furniture(request)
        self.checkpoints.push(
            CheckpointName.FINAL_SCENE, plan,
            scene_version=self.host.current_scene_version(), note="after a replan",
        )
        self.host.rebuild(plan)
        self._active = None
        return True

    def _tried_repair_verbs(self) -> tuple[str, ...]:
        if self.ladder.attempts:
            return tuple(dict.fromkeys(attempt.atom.value for attempt in self.ladder.attempts))
        return ()

    # -- bookkeeping -----------------------------------------------------------------------------

    def _remember(self, raised: Sequence[FailureEvent]) -> None:
        for event in raised:
            self._by_signature[event.signature] = event
            if event.failure_code is not None:
                # Also keyed by the code alone, because that is all a stage signature carries once
                # the kind has been mapped away. Collisions are possible in principle and the
                # active-event fallback below resolves them in favour of the failure being worked on.
                self._by_signature.setdefault(event.failure_code.value, event)
            if event not in self.events:
                self.events.append(event)

    def _record_critic_track(self) -> None:
        """Decide which ladder track a critic's complaint starts on, from what the gate found.

        A complaint about lighting or materials is answered by moving the lighting; a complaint
        about how the scene works is answered by moving what is in the way. Which of the two it is
        comes from the criteria that are *below threshold*: a critic that also remarks on the
        lighting while failing `prompt_adherence` has not said the lighting is the problem.

        The event is also re-registered with ``origin="critic"``. That is what makes the appearance
        track reachable at all: it is admissible only for a complaint a critic raised, because a
        geometric gate's failure is never fixed by changing the lighting.
        """
        critique = self.host.last_critique()
        if critique is None:
            return
        event = self._active
        if event is None:
            return
        from dataclasses import replace as dataclass_replace

        event = dataclass_replace(event, origin="critic")
        self._active = event
        self._remember((event,))
        appearance = (Track.APPEARANCE, Track.GEOMETRIC)
        geometric = (Track.GEOMETRIC, Track.APPEARANCE)
        # Which tracks the *failing criteria* can be answered on, not which tracks the critic
        # happened to mention. The previous rule asked whether any suggestion at all was an
        # appearance one, so a single lighting remark among six sent the whole ladder down the
        # appearance track first -- and that track has one rung for this event kind, after which
        # its next entry is the replan and `_repair_for` stops. Measured: a critique that failed
        # `prompt_adherence` (answerable only by `swap_asset`, on the geometric track) alongside
        # `style_material_coherence` never reached the geometric track at all.
        tracks = self._tracks_for_failing_criteria()
        ordered = appearance if tracks == {Track.APPEARANCE} else geometric
        self._track_order[event.signature] = ordered

    def _tracks_for_failing_criteria(self) -> set[Track]:
        """Every track that can answer one of the criteria the gate found below threshold.

        A criterion below threshold whose admissible repairs are all on the appearance track is
        answerable by lighting; one that admits a geometric repair is not. An unreadable verdict --
        no gate, or no failing criteria in it -- returns both tracks, which reads as "either may
        try" rather than as "no track can", so a host that cannot say cannot pin the ladder.
        """
        from rfw_w.scenes.agents.visual_critic import CRITERION_REPAIRS, critics_atom_for

        gate = self.host.last_visual_gate()
        failures = tuple(getattr(gate, "failures", ()) or ())
        if not failures:
            return {Track.GEOMETRIC, Track.APPEARANCE}
        tracks: set[Track] = set()
        for criterion, _score, _minimum in failures:
            repairs = CRITERION_REPAIRS.get(criterion, ())
            if not repairs:
                return {Track.GEOMETRIC, Track.APPEARANCE}
            for repair in repairs:
                tracks.add(ATOMS[critics_atom_for(repair)].track)
        return tracks

    def _tracks(self, event: FailureEvent) -> tuple[Track, ...]:
        """The order the two tracks are tried in for this failure."""
        return self._track_order.get(event.signature, (Track.GEOMETRIC, Track.APPEARANCE))

    def _activate(self, code: FailureCode) -> None:
        """Make the failure with this code the one the ladder is working on, if it is known.

        Called by the stages that report a failure directly rather than through a gate's witnesses,
        so that the escalation ladder continues on one problem across its rungs instead of starting
        over for each one.
        """
        event = self._by_signature.get(code.value)
        if event is not None:
            self._active = event

    def _event_for_signature(self, signature: str) -> FailureEvent | None:
        """The event a stage failure belongs to, or ``None`` when it belongs to no known failure.

        Three lookups, narrowing: the exact signature, the typed code, and the failure currently
        being repaired. The last is what keeps the ladder on one problem across its rungs, and it is
        guarded by the code so a failure of a *different* kind cannot inherit the active event.
        """
        direct = self._by_signature.get(signature)
        if direct is not None:
            return direct
        # The active event comes before the by-code table, because that table keeps the *first*
        # event it saw for a code and the active one may be a later, more specific version of it --
        # a critic's complaint re-registers the visual failure with its origin recorded, and reading
        # the older copy would lose the fact that a critic raised it.
        active = self._active
        if active is not None and active.failure_code is not None:
            if active.failure_code.value == _code_of(signature):
                return active
        return self._by_signature.get(_code_of(signature))

    def _failure_set(self) -> frozenset[str]:
        """The hard failure set, which is what progress is measured by.

        Read from the ledger rather than from the events the driver has seen, so a failure that has
        been repaired stops counting the moment its gate passes at the current scene version.
        """
        ledger = self.host.evidence_ledger()
        # Deliberately **not** filtered by scene version. Filtering by it was a bug that made the
        # failure set read empty the moment a repair produced a new version: the failing gate's
        # evidence was then stale rather than current, and a stale failure counted as no failure at
        # all. That is the staleness trap in reverse -- the resolver treats a stale pass as no pass,
        # and this must treat a stale failure as no less of a failure. What shrinks the set is a gate
        # being *re-measured* and passing, not the version moving underneath it.
        failing = {
            gate_id for gate_id in ledger.gate_ids()
            if (entry := ledger.latest_for(gate_id)) is not None
            and entry.status is not EvidenceStatus.PASS
            and any(record.gate_id == gate_id and record.status is EvidenceStatus.FAIL
                    for record in ledger.entries.values())
        }
        return frozenset(failing)

    def _result(self, terminal: TerminalReason) -> LoopRun:
        return LoopRun(
            terminal=terminal.value,
            scene_version=self.host.current_scene_version(),
            achieved_level=self.host.achieved_level(),
            segments=self._segments,
            events=tuple(self.events),
            patches=tuple(self.patches),
            removed=tuple(self.removed),
            checkpoints=tuple(self.checkpoints.to_list()),
            iterations=self._iterations,
            budget=self.budget.report().to_dict(),
            ladder=self.ladder.to_dict(),
            host_calls=dict(self.host_calls),
            notes=tuple(self.notes),
        )
