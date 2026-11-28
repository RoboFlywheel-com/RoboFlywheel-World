# SPDX-License-Identifier: Apache-2.0
"""The critic's repair suggestions: which are taken, and which are refused.

Mostly negative. The design gives the Visual Critic "permitted semantic modification suggestions",
and what makes them *permitted* is that four things are checked rather than trusted: the criterion
has to be one of the seven, the action has to be in the vocabulary the loop implements, the target
has to be an instance the critic was shown, and the action has to be one its criterion admits.

Every refusal is recorded. A critic that proposes actions the loop cannot take, or names objects
that are not in the scene, has stopped being grounded in what it was shown -- and a run that
silently ignored those suggestions would look exactly like a run whose critic had nothing to say.
"""

from __future__ import annotations

import json

import pytest

from rfw_w.scenes.agents.visual_critic import (
    CRITERIA,
    CRITERION_REPAIRS,
    CriticRepair,
    critics_atom_for,
    parse_critique,
)
from rfw_w.scenes.orchestrator.escalation import Atom, Track, ATOMS

VERSION = "a" * 64
SHOWN = ("desk", "chair")


def response(repairs, *, object_evidence=(), scores=None) -> str:
    return json.dumps({
        "scene_version": VERSION,
        "scores": scores or {criterion: 4 for criterion in CRITERIA},
        "defects": ["something"],
        "repairs": repairs,
        "object_evidence": list(object_evidence),
        "view_evidence": ["overview_top"],
    })


def parse(payload: str, *, targets=SHOWN):
    return parse_critique(
        payload, reviewed_scene_hash=VERSION, current_scene_hash=VERSION, known_targets=targets
    )


class TestAccepted:
    def test_a_well_formed_suggestion_survives(self) -> None:
        critique = parse(response([
            {"criterion": "composition_lighting", "target_id": "desk",
             "action": "adjust_lighting_material", "constraint": "too dark"},
        ]))
        assert critique.valid
        assert len(critique.suggestions) == 1
        assert critique.rejected_suggestions == ()

    def test_a_suggestion_names_the_criterion_target_and_action(self) -> None:
        critique = parse(response([
            {"criterion": "functionality", "target_id": "chair", "action": "optimise_pose"},
        ]))
        suggestion = critique.suggestions[0]
        assert suggestion.criterion == "functionality"
        assert suggestion.target_id == "chair"
        assert suggestion.action is CriticRepair.OPTIMISE_POSE


class TestRefused:
    def test_an_action_outside_the_vocabulary_is_dropped(self) -> None:
        """A critic may not reach for a subgraph rollback: those are the ladder's answers."""
        critique = parse(response([
            {"criterion": "functionality", "target_id": "desk", "action": "rollback_subgraph"},
        ]))
        assert critique.suggestions == ()
        assert critique.rejected_suggestions[0].reason == "unknown-action"

    def test_an_object_the_critic_was_not_shown_is_dropped(self) -> None:
        """A model may not invent an object: the only thing worse than no repair is a repair aimed
        at something that does not exist."""
        critique = parse(response([
            {"criterion": "functionality", "target_id": "ghost", "action": "optimise_pose"},
        ]))
        assert critique.suggestions == ()
        assert critique.rejected_suggestions[0].reason == "unknown-target"

    def test_an_action_its_criterion_does_not_admit_is_dropped(self) -> None:
        """A lighting complaint may not be answered by moving furniture."""
        critique = parse(response([
            {"criterion": "composition_lighting", "target_id": "desk", "action": "optimise_pose"},
        ]))
        assert critique.suggestions == ()
        assert critique.rejected_suggestions[0].reason == "action-not-admissible-for-criterion"

    def test_an_unknown_criterion_is_dropped(self) -> None:
        critique = parse(response([
            {"criterion": "vibes", "target_id": "desk", "action": "swap_asset"},
        ]))
        assert critique.suggestions == ()
        assert critique.rejected_suggestions[0].reason == "unknown-criterion"

    def test_no_target_list_means_no_suggestion_is_trusted(self) -> None:
        """Failing closed: a caller cannot get an unvalidated suggestion by forgetting to pass the
        instance ids, and is told which mistake it made."""
        critique = parse(response([
            {"criterion": "functionality", "target_id": "desk", "action": "optimise_pose"},
        ]), targets=())
        assert critique.suggestions == ()
        assert critique.rejected_suggestions[0].reason == "no-targets-supplied"

    def test_an_object_named_only_in_the_narrative_is_recorded_too(self) -> None:
        """The same class of problem as a repair aimed at a nonexistent object."""
        critique = parse(response(
            [],
            object_evidence=[{"object_id": "desk", "observation": "fine"},
                             {"object_id": "phantom", "observation": "floating"}],
        ))
        reasons = {item.reason for item in critique.rejected_suggestions}
        assert reasons == {"unknown-object-evidence"}

    def test_every_refusal_carries_a_reason_and_the_offending_values(self) -> None:
        critique = parse(response([
            {"criterion": "functionality", "target_id": "ghost", "action": "optimise_pose"},
        ]))
        item = critique.rejected_suggestions[0]
        assert item.action == "optimise_pose"
        assert item.target_id == "ghost"
        assert "desk" in item.detail and "chair" in item.detail

    def test_a_malformed_repairs_field_is_refused_without_killing_the_critique(self) -> None:
        """A bad suggestions list is not a bad critique: the scores are still usable."""
        payload = json.loads(response([]))
        payload["repairs"] = "please move the desk"
        critique = parse(json.dumps(payload))
        assert critique.valid
        assert critique.suggestions == ()
        assert critique.rejected_suggestions[0].reason == "not-a-list"

    def test_one_bad_suggestion_does_not_discard_a_good_one(self) -> None:
        critique = parse(response([
            {"criterion": "functionality", "target_id": "ghost", "action": "optimise_pose"},
            {"criterion": "functionality", "target_id": "chair", "action": "optimise_pose"},
        ]))
        assert len(critique.suggestions) == 1
        assert critique.suggestions[0].target_id == "chair"
        assert len(critique.rejected_suggestions) == 1


class TestVocabulary:
    def test_the_critic_cannot_ask_for_an_atom_the_ladder_keeps_for_itself(self) -> None:
        """The vocabulary is smaller than the ladder's atom set on purpose."""
        reachable = {critics_atom_for(action) for action in CriticRepair}
        assert Atom.ROLLBACK_SUBGRAPH not in reachable
        assert Atom.RESAMPLE_POSE not in reachable

    def test_every_critic_action_maps_to_a_real_atom(self) -> None:
        for action in CriticRepair:
            assert isinstance(critics_atom_for(action), Atom)

    def test_adding_a_context_object_is_a_replan(self) -> None:
        """There is no local atom that invents an object, and there should not be one."""
        assert critics_atom_for(CriticRepair.ADD_CONTEXT_OBJECT) is Atom.REPLAN

    def test_every_criterion_has_a_repair_row(self) -> None:
        assert set(CRITERION_REPAIRS) == set(CRITERIA)

    def test_every_row_names_at_least_one_action(self) -> None:
        for criterion, actions in CRITERION_REPAIRS.items():
            assert actions, f"{criterion} admits nothing, so a defect there can never be repaired"

    def test_the_lighting_criteria_admit_only_appearance_repairs(self) -> None:
        """The table that decides the loop's track from the VLM's judgement."""
        for criterion in ("composition_lighting", "style_material_coherence"):
            for action in CRITERION_REPAIRS[criterion]:
                assert ATOMS[critics_atom_for(action)].track is Track.APPEARANCE

    def test_the_functional_criteria_do_not_admit_appearance_repairs(self) -> None:
        for criterion in ("functionality", "workspace_and_passage"):
            for action in CRITERION_REPAIRS[criterion]:
                assert ATOMS[critics_atom_for(action)].track is Track.GEOMETRIC
