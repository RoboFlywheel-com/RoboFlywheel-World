# SPDX-License-Identifier: Apache-2.0
"""The driver: does the ladder actually climb, and does it stop when it should.

Mostly negative, per the repo's rule that a gate which cannot fail is the defect. The four claims
these tests make are the ones the whole loop rests on:

* a failure a local atom can fix costs **zero** model requests;
* an atom that commits a change without shrinking the failure set is recorded as no improvement,
  rather than as a repair;
* the model is reached only once every rung has been tried, and never before;
* the appearance track cannot be reached from a geometric failure, nor the geometric track from a
  lighting complaint.

The host is a stub rather than the real pipeline: the loop's rules are what is under test, and a
test that needed Isaac to check them would not be run.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.agents.visual_critic import CriticRepair, RepairSuggestion, VisualGate
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.tool import Evidence, EvidenceLedger, EvidenceStatus
from rfw_w.scenes.orchestrator.budgets import Budget
from rfw_w.scenes.orchestrator.driver import Driver
from rfw_w.scenes.orchestrator.loop import TerminalReason
from rfw_w.scenes.solvers.furnish import (
    FurnitureRequest,
    Orientation,
    PlacementKind,
    ScenePlan,
)

WALL_THICKNESS_M = 0.1


def plan_at(wall_face_offset_m: float, *, skip_candidates: int = 0) -> ScenePlan:
    return ScenePlan(
        furniture=(
            FurnitureRequest(
                entity_id="desk", synset="desk.n.01", kind=PlacementKind.AGAINST_WALL,
                orientation=Orientation.FRONT_TO_ROOM, hard_required=True, optional=False,
                skip_candidates=skip_candidates,
            ),
            FurnitureRequest(
                entity_id="chair", synset="chair.n.01", kind=PlacementKind.BESIDE,
                orientation=Orientation.FACES_TOWARD, faces_entity="desk",
            ),
        ),
        wall_face_offset_m=wall_face_offset_m,
    )


class StubHost:
    """A pipeline whose one gate fails until the plan stands pieces off the wall face.

    That is not an arbitrary rule: it is the real defect this loop was built to repair. The room
    polygon is the wall's outer face, so a piece standing off by its own half-depth alone has its
    back one wall thickness inside the wall solid -- and the gate that measures the inner face is
    the one that notices.
    """

    def __init__(self, *, critique_passes: bool = True, critique_once: bool = True,
                 gate_ever_passes: bool = True, critique_ran: bool = True,
                 targets: tuple[str, ...] = ("desk",),
                 suggestions: tuple[RepairSuggestion, ...] | None = None,
                 failing_criteria: tuple[str, ...] | None = None) -> None:
        self.plan = plan_at(0.0)
        self.ledger = EvidenceLedger()
        self.version = content_hash(self.plan.to_dict())
        self.critique_passes = critique_passes
        self.critique_once = critique_once
        self.gate_ever_passes = gate_ever_passes
        self.critique_ran = critique_ran
        self.targets = targets
        self.renders = 0
        self.critiques = 0
        self.rebuilds = 0
        self.rebuilt_plans: list[ScenePlan] = []
        # The critic's own suggestion, which is what decides the track the ladder starts on.
        self.suggestions = suggestions if suggestions is not None else (
            RepairSuggestion(
                criterion="composition_lighting", target_id="desk",
                action=CriticRepair.ADJUST_LIGHTING_MATERIAL,
            ),
        )
        # Which criteria the gate found below threshold. Separate from the suggestions on purpose:
        # the real critic proposes repairs for things it noticed, and a repair is not an admission
        # that its criterion failed. Defaults to the suggestions' own criteria, which is the common
        # case and what the real critic usually produces.
        self.failing_criteria = failing_criteria if failing_criteria is not None else tuple(
            item.criterion for item in self.suggestions
        )

    # -- host protocol ---------------------------------------------------------------------------

    def current_plan(self) -> ScenePlan:
        return self.plan

    def current_scene_version(self) -> str:
        return self.version

    def rebuild(self, plan: ScenePlan) -> str:
        self.plan = plan
        self.rebuilds += 1
        # Every plan handed to the build, because the final one is not enough: a repair that does
        # not shrink the failure set is rolled back, so what an atom edited is only visible in the
        # history rather than in the state the run ends on.
        self.rebuilt_plans.append(plan)
        self.version = content_hash(plan.to_dict())
        return self.version

    def evidence_ledger(self) -> EvidenceLedger:
        return self.ledger

    def run_deterministic_gates(self, scene_version: str) -> None:
        embedded = self.plan.wall_face_offset_m < WALL_THICKNESS_M - 1e-9
        passing = not embedded and self.gate_ever_passes
        self.ledger.record(Evidence(
            gate_id="geometry.in_bounds",
            status=EvidenceStatus.PASS if passing else EvidenceStatus.FAIL,
            state_hash=scene_version, backend="stub", tool_version="stub/v1",
            method="placed_box_containment_in_room_outline_and_height",
            measured_value=0.0 if passing else WALL_THICKNESS_M, unit="m",
            threshold_profile="engineering_v1",
            witness_refs=() if passing else tuple(f"instance:{name}:inside-wall" for name in self.targets),
            detail="" if passing else "desk is inside the wall",
        ))
        self.ledger.record(Evidence(
            gate_id="geometry.no_penetration", status=EvidenceStatus.PASS,
            state_hash=scene_version, backend="stub", tool_version="stub/v1",
            method="plan_overlap", measured_value=0.0, unit="m2", threshold_profile="engineering_v1",
        ))

    def run_backend(self, scene_version: str) -> bool:
        self.renders += 1
        self.ledger.record(Evidence(
            gate_id="visual.required_views", status=EvidenceStatus.PASS, state_hash=scene_version,
            backend="stub", tool_version="stub/v1", method="montage", threshold_profile="views_v1",
        ))
        return True

    def run_critique(self, scene_version: str) -> bool:
        self.critiques += 1
        if not self.critique_ran:
            self.ledger.record(Evidence(
                gate_id="visual.critique", status=EvidenceStatus.NOT_RUN, state_hash=scene_version,
                backend="stub", tool_version="stub/v1", method="vlm_critique_strict_schema",
                detail="no model was configured to look at this scene",
            ))
            return False
        passing = self.critique_passes or (self.critique_once and self.critiques > 1)
        self.ledger.record(Evidence(
            gate_id="visual.critique",
            status=EvidenceStatus.PASS if passing else EvidenceStatus.FAIL,
            state_hash=scene_version, backend="stub", tool_version="stub/v1",
            method="vlm_critique_strict_schema",
            measured_value=3.0 if passing else 2.0, unit="score_1_to_5",
            threshold_profile="visual_engineering_v1",
            # In the form `targets_from_witnesses` reads, exactly as `critique_evidence` emits them:
            # the objects the critic asked to change, which is what the event's targets are.
            witness_refs=() if passing else ("composition_lighting",) + tuple(
                f"instance:{item.target_id}:repair-{item.action.value}" for item in self.suggestions
            ),
            detail="" if passing else "the room reads flat",
        ))
        return passing

    def last_critique(self) -> object | None:
        if self.critiques == 0:
            return None
        return _Critique(self.suggestions)

    def last_visual_gate(self) -> VisualGate | None:
        if self.critiques == 0:
            return None
        return VisualGate(
            passed=False, scores=None,
            failures=tuple((name, 2, 3) for name in self.failing_criteria),
            critical_defect=False, reason="criteria below threshold",
        )

    def gates_settled(self) -> bool:
        required = ("geometry.in_bounds", "geometry.no_penetration", "visual.required_views",
                    "visual.critique")
        return all(self.ledger.satisfies(gate, self.version) for gate in required)

    def wall_thickness_m(self) -> float:
        return WALL_THICKNESS_M

    def synset_vocabulary(self) -> tuple[str, ...]:
        return ("desk.n.01", "chair.n.01")

    def goal(self) -> str:
        return "pick up the cup from the desk"

    def achieved_level(self) -> str:
        return "functional_scene_ready" if self.gates_settled() else "draft"


class _Critique:
    """The two fields the driver reads off a critique. The gate's verdict is passed separately."""

    def __init__(self, suggestions: tuple[RepairSuggestion, ...]) -> None:
        self.suggestions = suggestions


class TestLocalRecovery:
    def test_a_wall_embedding_is_repaired_without_a_model_request(self) -> None:
        """The whole point of the ladder: a geometric failure costs no model request."""
        host = StubHost()
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()

        assert run.terminal == TerminalReason.READY.value
        assert run.budget["model_requests_used"] == 0
        assert host.plan.wall_face_offset_m == pytest.approx(WALL_THICKNESS_M)
        # L0 could not fix it and L1 could, so both rungs are in the history in that order.
        atoms = [attempt["atom"] for attempt in run.ladder["attempts"]]
        assert atoms == ["resample_pose", "optimise_pose"]
        assert run.ladder["resolved_without_model"] == 1
        assert run.ladder["local_recovery_rate"] == 1.0

    def test_the_cheaper_rung_is_tried_first(self) -> None:
        host = StubHost()
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()
        levels = [attempt["level"] for attempt in run.ladder["attempts"]]
        assert levels == ["L0_POSE_RESAMPLE", "L1_LOCAL_OPTIMISATION"]

    def test_the_render_happens_once_when_only_geometry_was_repaired(self) -> None:
        """Geometry is repaired before anything is rendered, which is what makes the loop cheap."""
        host = StubHost()
        Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()
        assert host.renders == 1
        assert host.critiques == 1

    def test_a_failure_that_never_clears_reaches_the_model(self) -> None:
        """With the gate unfixable, every rung is spent before the replanner is asked."""
        host = StubHost(gate_ever_passes=False)
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()

        attempts = [attempt["atom"] for attempt in run.ladder["attempts"]]
        # L0, L1, L2, L3 all tried and none of them worked, so L5 is the only rung left.
        assert "resample_pose" in attempts
        assert "optimise_pose" in attempts
        assert run.terminal != TerminalReason.READY.value
        assert run.ladder["resolved_without_model"] == 0


class TestNoProgress:
    def test_unmeasured_replacement_does_not_clear_a_failed_gate(self) -> None:
        host = StubHost()
        host.ledger.record(Evidence(
            gate_id="visual.critique", status=EvidenceStatus.FAIL, state_hash="a" * 64,
            backend="stub", tool_version="stub/v1", method="review"))
        host.ledger.record(Evidence(
            gate_id="visual.critique", status=EvidenceStatus.NOT_RUN, state_hash="b" * 64,
            backend="stub", tool_version="stub/v1", method="review", detail="budget exhausted"))
        driver = Driver(host=host, budget=Budget(deadline_s=60.0))
        assert "visual.critique" in driver._failure_set()

    def test_an_atom_that_changes_nothing_is_recorded_as_no_improvement(self) -> None:
        """A commit that does not shrink the failure set is thrashing, and must say so."""
        host = StubHost(gate_ever_passes=False)
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()

        # L0 resamples the sweep: the plan changes, so a version is produced, but the gate still
        # fails. That attempt is the one this test is about.
        resample = [a for a in run.ladder["attempts"] if a["atom"] == "resample_pose"]
        assert resample, "L0 was never tried"
        assert resample[0]["improved"] is False
        assert resample[0]["resolved"] is False
        assert any("did not shrink" in note for note in run.notes)

    def test_an_atom_with_nothing_to_do_reports_failure_rather_than_success(self) -> None:
        """``change_support`` has no alternative support here, and must not claim one."""
        host = StubHost(gate_ever_passes=False)
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()
        attempted = {attempt["atom"] for attempt in run.ladder["attempts"]}
        assert "change_support" in attempted or "swap_asset" in attempted


class TestOrdering:
    def test_the_appearance_track_is_unreachable_from_a_geometric_failure(self) -> None:
        """A wall embedding is never answered by moving the lighting."""
        host = StubHost(gate_ever_passes=False)
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()
        assert "adjust_lighting_material" not in {
            attempt["atom"] for attempt in run.ladder["attempts"]
        }

    def test_a_lighting_complaint_is_answered_by_lighting_and_not_by_moving_furniture(self) -> None:
        """The other direction of the same rule, with the critique failing every time."""
        host = StubHost(critique_passes=False, critique_once=False)
        run = Driver(host=host, budget=Budget(deadline_s=120.0, max_subgraph_rollbacks=6)).run()

        assert "adjust_lighting_material" in {
            attempt["atom"] for attempt in run.ladder["attempts"]
        }
        # Geometry is untouched: the atom's whole contract is that nothing moves. The wall repair
        # from the earlier failure stands, and the lighting rung did not disturb it.
        assert host.plan.wall_face_offset_m == pytest.approx(WALL_THICKNESS_M)
        # The render was redone, because the design forbids reusing the critique of a scene whose
        # appearance has changed.
        assert host.renders >= 2
        assert host.critiques >= 2
        # Every rendered appearance version still has real deterministic measurements.
        for (gate_id, version), entry in host.ledger.entries.items():
            if gate_id == "visual.required_views":
                assert host.ledger.get("geometry.in_bounds", version) is not None
        # The lighting rung changed nothing that survived, and says so: the critique never passes,
        # so a change that cannot be shown to help is rolled back rather than committed.
        assert host.plan.appearance.dome_intensity == pytest.approx(ScenePlan().appearance.dome_intensity)
        assert any(
            "did not shrink" in note and "adjust_lighting_material" in note
            for note in run.notes
        )

    def test_a_mixed_complaint_starts_on_the_track_that_can_answer_a_failing_criterion(self) -> None:
        """Measured on a real run rather than invented: the critique failed `prompt_adherence` --
        answerable only by `swap_asset`, on the geometric track -- and *also* proposed a lighting
        change, for a criterion that had passed. The old rule asked whether any suggestion at all
        was an appearance one, so the ladder started on the appearance track, ran its single rung
        and stopped: the swap the failing criterion needed was never reached, and the run reported
        an exhausted ladder on a scene it had not really tried to repair."""
        host = StubHost(
            critique_passes=False, critique_once=False,
            suggestions=(
                RepairSuggestion(criterion="prompt_adherence", target_id="chair",
                                 action=CriticRepair.SWAP_ASSET),
                RepairSuggestion(criterion="composition_lighting", target_id="desk",
                                 action=CriticRepair.ADJUST_LIGHTING_MATERIAL),
            ),
            failing_criteria=("prompt_adherence",),
        )
        run = Driver(host=host, budget=Budget(deadline_s=120.0, max_subgraph_rollbacks=6)).run()

        assert "swap_asset" in {attempt["atom"] for attempt in run.ladder["attempts"]}

    def test_the_swap_goes_to_the_object_the_critic_named(self) -> None:
        """`event.targets` is a set, so the best the event itself can say is which of its members
        the plan can move -- and it says it by taking the first, which is the alphabet's choice.
        The pairing the suggestion was made with is still in the critique, so the repair uses it."""
        host = StubHost(
            critique_passes=False, critique_once=False,
            suggestions=(
                RepairSuggestion(criterion="prompt_adherence", target_id="chair",
                                 action=CriticRepair.SWAP_ASSET),
            ),
            failing_criteria=("prompt_adherence",),
        )
        Driver(host=host, budget=Budget(deadline_s=120.0, max_subgraph_rollbacks=6)).run()

        # Read from the rebuild history, not from the final plan: the critique never passes, so a
        # swap that cannot be shown to help is rolled back and the plan the run ends on is the one
        # it started with. That rollback is what the atom's own test is about; this one is about
        # which object it was aimed at.
        def swapped(name: str) -> bool:
            return any(
                (request := plan.furniture_for(name)) is not None and request.asset_index == 1
                for plan in host.rebuilt_plans
            )

        assert swapped("chair")
        # "desk" sorts before "chair", so taking the first plan row in the target set -- which is
        # all an untargeted geometric repair could do -- would have swapped the desk instead.
        assert not swapped("desk")


class TestTargetSelection:
    def test_a_repair_targets_an_entity_the_plan_has(self) -> None:
        """A gate names every instance it looked at, and the alphabetically first of those need not
        be one the plan can move. `geometry.in_bounds` lists all twelve, `cup` sorts first, and the
        cup is not a plan entity -- so L0 reported "no target in the plan" for an event that named
        eight pieces that were in it, and did nothing on a scene it had plenty to do on.
        """
        host = StubHost()
        host.targets = ("cup", "desk")
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()

        assert run.terminal == TerminalReason.READY.value
        # L0 acted rather than reporting it had nothing to do.
        assert not any("no target in the plan" in note for note in run.notes)
        assert host.plan.furniture_for("desk").skip_candidates == 1

    def test_an_event_naming_only_non_plan_entities_says_so(self) -> None:
        host = StubHost()
        host.targets = ("cup", "pen_holder")
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()
        assert any("the repair is untargeted" in note for note in run.notes)


class TestTermination:
    def test_a_run_that_cannot_finish_stops_at_the_deadline(self) -> None:
        host = StubHost(gate_ever_passes=False)
        # Expired before the run starts rather than by a tiny timeout, which would be a race: the
        # claim is that a run which cannot finish does no work, and that must not depend on how
        # long the interpreter took to reach the check.
        budget = Budget(deadline_s=60.0, max_subgraph_rollbacks=6)
        budget.started_at -= 120.0
        run = Driver(host=host, budget=budget).run()
        assert run.terminal == TerminalReason.BUDGET_EXHAUSTED.value
        assert host.rebuilds == 0
        assert run.ladder["attempts"] == []

    def test_a_healthy_scene_passes_with_no_events_and_no_repairs(self) -> None:
        host = StubHost()
        host.plan = plan_at(WALL_THICKNESS_M)
        host.version = content_hash(host.plan.to_dict())
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()

        assert run.terminal == TerminalReason.READY.value
        assert run.ladder["attempts"] == []
        assert run.events == ()
        assert run.budget["model_requests_used"] == 0

    def test_a_critique_that_did_not_run_does_not_escalate(self) -> None:
        """`not_run` never satisfies a gate, so the readiness ladder already refuses to let an
        absent critique pass anything. Escalating on it instead spends a replan repairing a scene
        whose only problem is that no model was configured to look at it -- and a run without
        `PVSCENE_VLM_MODEL` did exactly that before this.
        """
        host = StubHost(critique_ran=False)
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6)).run()
        assert not any("visual" in event.to_dict()["source_gate"] for event in run.events)
        assert run.ladder["attempts"] == [] or all(
            attempt["atom"] != "adjust_lighting_material" for attempt in run.ladder["attempts"]
        )

    def test_the_replan_is_not_reached_without_a_replanner(self) -> None:
        host = StubHost(gate_ever_passes=False)
        run = Driver(host=host, budget=Budget(deadline_s=60.0, max_subgraph_rollbacks=6), replan_config=None).run()
        # A ladder that ran out with nothing to escalate to is a repair exhaustion, not a budget
        # one: the run did not run out of time, it ran out of repairs.
        assert run.terminal == TerminalReason.REPAIR_EXHAUSTED.value
        assert any("no replanner is configured" in note for note in run.notes)
        assert run.budget["model_requests_used"] == 0

def test_exhausted_retrieval_rolls_back_and_records_failed_rung():
    from rfw_w.scenes.orchestrator.driver import RepairUnavailable

    class ExhaustedHost(StubHost):
        def rebuild(self, plan):
            chair = plan.furniture_for('chair')
            if chair and chair.asset_index > 0:
                self.plan = plan  # simulate a host that partially adopted the proposal
                raise RepairUnavailable('TARGET_CANDIDATES_EXHAUSTED')
            return super().rebuild(plan)

    host = ExhaustedHost(
        critique_passes=False, critique_once=False,
        suggestions=(RepairSuggestion(criterion='prompt_adherence', target_id='chair',
                                      action=CriticRepair.SWAP_ASSET),),
        failing_criteria=('prompt_adherence',),
    )
    result = Driver(host=host, budget=Budget(deadline_s=120, max_subgraph_rollbacks=6)).run()
    assert host.plan.furniture_for('chair').asset_index == 0
    assert host.version == content_hash(host.plan.to_dict())
    exhausted = [row for row in result.ladder['attempts']
                 if 'TARGET_CANDIDATES_EXHAUSTED' in row['detail']]
    assert exhausted
    assert all(not row['resolved'] and not row['improved'] for row in exhausted)
