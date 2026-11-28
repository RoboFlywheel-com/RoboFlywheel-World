# SPDX-License-Identifier: Apache-2.0
"""T03 phase/gate protocol example. No simulator or hardware backend is included.

An adapter sends actuator commands and advances physics. A separate checker reads
measured state. Symbolic graph effects never satisfy these execution gates.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Phase:
    name: str
    command: str
    targets: tuple[str, ...]
    requires: tuple[str, ...]
    ensures: tuple[str, ...]


PHASES = (
    Phase('open_drawer', 'grasp_pull_release', ('drawer',),
          ('initial_scene_valid', 'robot_ready', 'drawer_closed', 'handle_accessible'),
          ('drawer_open', 'drawer_passively_held', 'hand_released')),
    Phase('stow_caliper', 'pick_transport_place_release', ('caliper', 'left_compartment'),
          ('drawer_open', 'drawer_passively_held', 'caliper_on_source', 'left_compartment_free'),
          ('caliper_inside_left', 'caliper_supported', 'hand_released')),
    Phase('stow_screwdriver', 'pick_transport_place_release', ('screwdriver', 'right_compartment'),
          ('drawer_open', 'drawer_passively_held', 'screwdriver_on_source', 'right_compartment_free'),
          ('screwdriver_inside_right', 'screwdriver_supported', 'hand_released')),
    Phase('close_drawer', 'grasp_push_release', ('drawer',),
          ('caliper_inside_left', 'screwdriver_inside_right', 'hand_clear_of_cavity', 'closure_sweep_clear'),
          ('drawer_closed', 'hand_released', 'contents_stable')),
    Phase('final_settle', 'observe_settle', ('drawer', 'caliper', 'screwdriver'),
          ('drawer_closed', 'hand_released'),
          ('caliper_inside_left', 'screwdriver_inside_right', 'contents_stable', 'episode_valid')),
)


class Runtime(Protocol):
    def execute(self, phase: Phase) -> None:
        """Run a closed-loop phase with bounded timeout and an emergency stop."""

    def stop(self) -> None:
        """Stop sending task motion commands when an execution gate fails."""


class Checker(Protocol):
    def measure(self, predicate: str) -> bool:
        """Recheck current measured state, with frozen thresholds and evidence."""


class TaskFailure(RuntimeError):
    pass


def run_reference(runtime: Runtime, checker: Checker) -> tuple[str, ...]:
    """Run once; no reset, retry, cached state effects or backend success claims.

    Returning completed phases is not an independent task validation report.
    The caller must save raw state/commands and invoke the independent verifier.
    """
    completed = []
    try:
        for phase in PHASES:
            for predicate in phase.requires:
                if not checker.measure(predicate):
                    raise TaskFailure(f'PRECONDITION_UNSATISFIED:{phase.name}:{predicate}')
            runtime.execute(phase)
            for predicate in phase.ensures:
                if not checker.measure(predicate):
                    raise TaskFailure(f'POSTCONDITION_UNSATISFIED:{phase.name}:{predicate}')
            completed.append(phase.name)
    except BaseException:
        runtime.stop()
        raise
    return tuple(completed)
