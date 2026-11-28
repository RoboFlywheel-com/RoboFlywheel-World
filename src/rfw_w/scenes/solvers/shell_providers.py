# SPDX-License-Identifier: Apache-2.0
"""Two ways to build a room shell, behind one interface.

Before this there was one way: extrude the outline. That is no longer a *fact about the pipeline*,
it is one implementation among two, and the difference between them is the whole point --

* ``builtin`` puts its walls exactly **on** the plan's lines, because it authors them from the plan.
* ``infinigen`` insets them by half a wall thickness, because a wall is a body with a thickness and
  the plan's polygon is its centreline. Its openings are sized by its own constants, not by the
  widths the plan asked for.

Both are handed the same `RoomPlan` and their `ShellBuild` goes through the same gate, so the
question a reader can answer is not "which builder ran" but "did it build what the plan says".
`builtin` is the control that keeps that gate honest: it is the one builder whose answer is known
in advance, so a gate that fails it has a defect of its own.

The optional Infinigen adapter uses the configured standalone checkout and records
its pinned version. Install this package into the Blender Python environment so
all generation, export and measurement workers use the same contracts.

**Nothing here decides whether the build is acceptable.** It reports what it built and what it
measured; `shell.matches_plan` judges it (SPEC-004). A provider that also judged could hide the
case where the provider itself was the problem.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from rfw_w.scenes.contracts.background_constraints import (
    BUNDLE_FILENAME,
    GENERATED_OBJECTS_MAX_PER_STAGE,
    INJECTION_APPLIED,
    INJECTION_FILENAME,
    SOLVE_STAGES,
    UNCALIBRATED_POLICY_NOTE,
    Reservation,
    ReservationBundle,
    ReservationPolicy,
    abort_param,
)
from rfw_w.scenes.contracts.room_plan import (
    RoomPlan,
    buildable_window_span_m,
    floor_plan_payload,
    opening_host_span,
    plan_to_shell,
)
from rfw_w.scenes.contracts.state import OpeningKind
from rfw_w.scenes.contracts.shell_build import (
    ShellBuild,
    ShellBuildError,
    shell_from_report,
)
from rfw_w.scenes.solvers.furnishing_plan import CATEGORY_PROFILES, background_gaps
from rfw_w.scenes.solvers.zones import ZoneGraph
from rfw_w.scenes.solvers.room import RoomError, ShellOptions
from rfw_w.scenes.solvers.materials import MaterialExportProfile

#: The commit this provider was written and measured against. Recorded in code rather than only in
#: the ADR because the ADR is not what runs: another tree on the same machine would produce a
#: different room and every measurement below would quietly describe a program nobody ran.
INFINIGEN_PIN = "892947be71c4ff102c6a558d088ffe149561c11f"

#: Standalone Infinigen checkout, supplied by local runtime configuration.
INFINIGEN_ROOT_ENV = "RFW_INFINIGEN_ROOT"

#: Wall thickness handed to Infinigen, in metres.
#:
#: This is the one number the plan cannot supply: `PlanRoom` carries an outline, a height and two
#: material tokens, and no thickness. The value matches `ShellOptions.wall_thickness_m` so both
#: providers are asked for the same room; it is *not* compared by the gate, and the reason is
#: recorded in SPEC-004 §4 R2 -- the measured quantity is the interior lining (0.05 m), which is
#: half of this, so a comparison would be against the wrong number by a factor of two.
INFINIGEN_WALL_THICKNESS_M = 0.1

#: Why the opt-in delivery's policy leaves `abort_unsatisfied_*` **off**, recorded rather than
#: implied: turning it on makes these rooms fail in ~20 s on upstream's *own* unsatisfied groups
#: (`lighting`, `desk`, `office_desk_chairs` -- SPEC-011 TASK 3 measured it), so a delivery could
#: never be built with the switch on. The audit reports the off switch as
#: `BACKGROUND_SOLVER_UNGUARDED` rather than passing it quietly.
ABORT_OFF_NOTE = ("abort_unsatisfied_* is off in this delivery because turning it on makes these "
                  "rooms fail on upstream's own unsatisfied groups (SPEC-011 TASK 3, measured: "
                  "20.2 s, violations={'lighting': 1}); the audit reports the off switch as "
                  "BACKGROUND_SOLVER_UNGUARDED")

#: How long one build may take. The measured cost is 2 min 52 s for a single room at `-t coarse`
#: (2026-09-21 probe); this is generous enough for a loaded machine and short enough that a wedged
#: Blender does not hold a run forever.
BUILD_TIMEOUT_S = 900

#: Stages the build turns on, and why each was off.
#:
#: Upstream's `custom_solve.gin` -- which the probe drove -- turns all three off, and ADR-002's
#: Consequences require them back: they are the architectural detail (a door in its frame, a
#: skirting board along the wall) that is the entire reason for adopting Infinigen instead of
#: extruding polygons.
#:
#: These are passed as gin **overrides** rather than as a gin file of our own. `apply_gin_configs`
#: (`infinigen/core/init.py:152`) resolves a config name by matching its *stem* inside
#: `infinigen_examples/configs_indoor` and `configs_nature`, and `generate_indoors.main` hardcodes
#: those two folders -- so a file of ours can only be loaded by wrapping that entry point or by
#: writing into a tree we do not own. Overrides are applied after every config file, so they win,
#: and they keep the enablement in our code next to this comment instead of in someone else's
#: directory.
INFINIGEN_STAGE_OVERRIDES: tuple[str, ...] = (
    "compose_indoors.room_doors_enabled=True",
    "compose_indoors.skirting_floor_enabled=True",
    "compose_indoors.skirting_ceiling_enabled=True",
)
#: And one switched **off**, which is a different kind of statement from the three above.
#:
#: Infinigen's `coarse` task does not stop at the shell: `compose_indoors` furnishes the room as
#: part of it (`p.run_stage("populate_assets", populate.populate_state_placeholders, state,
#: use_chance=False)` -- `generate_indoors.py:307`), and `use_chance=False` means no task gates it,
#: so there is no `-t` that leaves it out. We ask for a shell, so this is work we never wanted and
#: now pay for three times: it is minutes of constraint solving per build, the furniture is not
#: ours to place (core-first places it, `solvers/furnish.py`), and it makes a shell provider's
#: success depend on Infinigen's **asset** pipeline.
#:
#: Measured 2026-09-22, and this is what found it: with the kitchen's windows finally building, the
#: run died in that stage -- `blender_rock.py:27 require_blender_addon("extra_mesh_objects")`, an
#: addon this Blender tree does not carry and cannot fetch. A provider asked only for walls should
#: not be able to fail because a decorative rock needed Blender's extra-objects extension.
#:
#: `RandomStageExecutor._should_run_stage` (`pipeline.py:42`) reads `params[f"{name}_enabled"]`, and
#: `params` **is** the `compose_indoors` override dict (`generate_indoors.py:137` passes
#: `overrides` straight in) -- which is why this is a plain gin override and not a wrapper.
#:
#: Only the one stage, deliberately. The stages before it (the solve ladder) and after it
#: (`floating_objs` and the saves) are left as they are until a build gives evidence that they need
#: the same treatment; four `_enabled=False` overrides written on a suspicion would be a claim
#: nobody measured.
INFINIGEN_STAGE_OVERRIDES_OFF: tuple[str, ...] = (
    "compose_indoors.populate_assets_enabled=False",
    # And the last stage, explicitly, for a different reason: SPEC-009's audit found that
    # `floating_objs` left *no* evidence on 13 of 13 deliveries -- no line in the build's log and
    # no override in the command -- so "it did not run" was a claim nobody could make. Upstream's
    # stage runner is silent for a stage it does not run, so the only way that stage's state can be
    # a measurement is for our own command to say so. Unlike `populate_assets`, this one is not
    # switched off *by* this override -- 0.86 m of loose décor is not a shell and `populate_assets`
    # already carries the furniture decision -- it is recorded by it.
    "compose_indoors.floating_objs_enabled=False",
)

#: Our own build entry and the constraint patch it installs (SPEC-011 / ADR-009 decision 1). Both
#: live in this repository, and the build records their digests: a build that runs code we cannot
#: point at cannot say what it ran.
BACKGROUND_ENTRY_SCRIPT = "rfw_w.scenes.workers.blender.run_infinigen_build"
BACKGROUND_PATCH_MODULE = "rfw_w.scenes.workers.blender.infinigen_constraint_patch"


def _worker_path(module: str) -> Path:
    from rfw_w.scenes.resources import module_path
    return module_path(module)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class BackgroundConstraints:
    """The opt-in: which reservations to send into the build, and how hard (SPEC-011).

    Its absence *is* the default path -- a run without this records no reservation, sends no
    constraint and runs the third-party wrapper's own entry script, which is also what keeps the
    commands recorded in SPEC-004/005/009 evidence reproducible byte for byte. Requirement 7 in one
    type: the switch is a value a caller passes and a record can carry, not an environment variable
    that changes what a build means.
    """

    reservations: tuple[Reservation, ...]
    policy: ReservationPolicy = field(
        default_factory=lambda: ReservationPolicy(note=UNCALIBRATED_POLICY_NOTE))
    #: Whether this run also asks the build to *realise* background content, which is what the
    #: constraints exist to make safe. On by default, and the reason is a measurement: the first
    #: injected build left `populate_assets` switched off, so 32 generated objects became 1 and the
    #: zero intrusion it measured was a zero over an empty room -- a constraint nothing tested.
    generation: bool = True
    #: The categories this room allows generation to fill (SPEC-011 requirement 4, ADR-009 decision
    #: 4). Declared per run rather than derived, because "which decorations does this room want" is
    #: a decision about the task; *checked* against the retrieval vocabulary at request time
    #: (`solvers/furnishing_plan.py:background_gaps`), because "retrieval covers it" is a fact.
    gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reservations:
            raise ShellProviderError("BACKGROUND_BUNDLE_EMPTY(no reservations to inject)")
        if self.generation and not self.gaps:
            # The same refusal as the line above, one level down: generation with no gap declared is
            # a request for the pool to refuse everything the build realises, so the room would gain
            # nothing and the three minutes would have bought a slower way to build the same room.
            raise ShellProviderError(
                "BACKGROUND_BUNDLE_EMPTY(no gaps to fill): a run that asks for generation has to "
                "declare the categories it wants (ADR-009 decision 4)")
        # Validated here and not only in the audit: a request that declares a category retrieval
        # provides is a request the pipeline cannot honour, and the time to say so is before three
        # minutes of Blender rather than after.
        from rfw_w.scenes.solvers.furnishing_plan import PlanDerivationError, background_gaps

        try:
            background_gaps(self.gaps)
        except PlanDerivationError as error:
            raise ShellProviderError(str(error)) from error


def generation_provider(plan: RoomPlan, *, zones: "ZoneGraph", gaps: Sequence[str],
                        policy: ReservationPolicy | None = None,
                        infinigen_root: Path | None = None) -> "InfinigenShellProvider":
    """The opt-in delivery path: Infinigen, with the task's reservations injected and gaps declared.

    One function so a delivery run and a probe ask for the same thing: the reservations are
    re-derived from the plan and the zone plan this run drew (`reservations_from_plan`, the same
    call the probe makes), the gaps are the caller's declaration -- checked against the retrieval
    vocabulary here, before Blender starts -- and the policy travels into the build's record.

    It is a *provider*, not a mode: a caller that wants the default path keeps calling
    `provider_named("infinigen")`, which asks for no reservations and records nothing about them
    (SPEC-011 requirement 7 keeps that the default).
    """
    from rfw_w.scenes.solvers.room_plan import reservations_from_plan

    if not gaps:
        raise ShellProviderError(
            "BACKGROUND_BUNDLE_EMPTY(no gaps to fill): the opt-in path has to declare the "
            "categories generation may add (ADR-009 decision 4)")
    constraints = BackgroundConstraints(
        reservations=reservations_from_plan(plan, circulation=zones.circulation,
                                            door_sweep=zones.door_sweep),
        policy=policy or ReservationPolicy(note=UNCALIBRATED_POLICY_NOTE),
        gaps=tuple(str(item) for item in gaps))
    return InfinigenShellProvider(infinigen_root=infinigen_root,
                                  background_constraints=constraints)


#: What an opt-in delivery asks the build to express: the weights TASK 3's sweep measured as
#: protecting the region (6/2.5), with the abort reason recorded above.
BACKGROUND_GENERATION_POLICY = ReservationPolicy(
    abort_unsatisfied=False, band_penetration_weight=6.0, distance_weight=2.5,
    note="SPEC-011 TASK 4's opt-in delivery: the sweep's protecting setting (6/2.5), abort off "
         "(the reason is recorded beside this policy in the delivery's background block)")


def background_block(*, generation: bool, gaps: Sequence[str], out: Path,
                     policy: ReservationPolicy | None = None,
                     assembly: dict | None = None,
                     injection_record: Path | None = None) -> dict:
    """SPEC-011's opt-in as a delivery records it, read from the request and the build's records.

    Written at manifest time rather than stashed in the run's scene dict: a scene is rebuilt whenever
    the plan is restored or repaired, so a value written into it during the shell build does not
    survive to the manifest -- measured, on a run whose own log showed the opt-in and whose manifest
    came out empty. What travels instead is the request (these arguments) plus the build's records on
    disk, which is also the honest direction: the block reports what the run asked for and what the
    build wrote, not what a dict remembered.

    ``out`` is the delivery directory; the build's records live under ``shell_build/*/infinigen_build``
    inside it, so a delivery that took the opt-in *contains* its own evidence and this is only the
    reader's way in.
    """
    if not generation:
        return {}
    policy = policy or ReservationPolicy(note=UNCALIBRATED_POLICY_NOTE)
    block: dict = {
        "mode": "generated", "generation": True, "gaps": [str(item) for item in gaps],
        "policy": policy.to_dict(),
        "requested_by": "--background-generation (SPEC-011 requirement 7: a value the run carries, "
                        "not an environment variable)",
        "why_abort_is_off": ABORT_OFF_NOTE,
        }
    # The caller supplies the final scene version's immutable assembly record.
    if assembly is not None:
        block["assembly"] = assembly
        block["delivery_note"] = (
            "every kept generated object is a prepared asset and an instance of this delivery's "
            "scene_ir (`generated_<category>_<ordinal>`), judged by the same gates as the retrieved "
            "periphery; `assembly` names them and the pool ledger below says why the rest were "
            "refused -- the pool's refusals were measured inside the build, and "
            "`assembly.refused_at_assembly` names the ones refused here because they would have "
            "shared space with a piece the delivery already places"
        )
    else:
        block["delivery_note"] = (
            "the kept generated objects are in the shell build's scene and are named in the pool "
            "ledger below; they are not instances of this delivery's scene_ir -- the assembly step "
            "did not run or did not finish, and `assembly` is absent for that reason"
        )
    # A repaired run may contain several shell plans. Its final plan chooses the evidence;
    # filesystem ordering cannot choose a scene version.
    record_path = None if injection_record is None else Path(injection_record).resolve()
    if record_path is not None and not record_path.is_relative_to(Path(out).resolve()):
        raise ShellProviderError('BACKGROUND_INJECTION_FAILED(record outside delivery)')
    if record_path is None or not record_path.is_file():
        block["injection"] = "not-recorded"
        block["injection_reason"] = (
            "the run asked for background generation and the shell build left no record; "
            "BACKGROUND_PATCH_UNRECORDED is the failure name for this"
        )
        return block
    record = json.loads(record_path.read_text())
    pool = record.get("rejection_pool") or {}
    block.update({
        "injection": record.get("injection"),
        "injection_reason": record.get("injection_reason", ""),
        "upstream_pin": record.get("upstream_pin", ""),
        "patch_sha256": record.get("patch_sha256", ""),
        "entry_script_sha256": record.get("entry_script_sha256", ""),
        "budget": record.get("budget", {}),
        "rejection_pool": {key: pool.get(key) for key in
                           ("ran", "rule", "bounds", "kept", "rejected",
                            "rejected_categories", "rejection_reasons")},
        "rejected_state_keys": pool.get("rejected_state_keys", []),
        "record_path": record_path.resolve().relative_to(Path(out).resolve()).as_posix(),
        "bundle_path": record_path.with_name(BUNDLE_FILENAME).resolve().relative_to(Path(out).resolve()).as_posix(),
        "reservations": [item.get("region") for item in record.get("reservations", ())],
    })
    return block


def reservation_bundle_for(plan: RoomPlan, constraints: BackgroundConstraints, *,
                           entry_script: Path, patch_module: Path) -> ReservationBundle:
    """The request a build is handed: this plan's reservations, this policy, and the credentials.

    The reservations are the caller's (`solvers/room_plan.py:reservations_from_plan` reads them out
    of the plan and its zones): which regions this run must keep clear is a decision about the task,
    and a provider asked only for walls has no business deriving it.
    """
    return ReservationBundle(
        scene_id=plan.plan_id, core_hash=plan.core.core_hash,
        upstream_pin=INFINIGEN_PIN, entry_script=_repo_relative(entry_script),
        entry_script_sha256=_digest(entry_script), patch_module=_repo_relative(patch_module),
        patch_sha256=_digest(patch_module), reservations=constraints.reservations,
        policy=constraints.policy,
        # The vocabulary of the run, not of the day someone reads the record: the pool judges
        # "covered by retrieval" against the list the request carried.
        gaps=background_gaps(constraints.gaps),
        retrieval_categories=tuple(sorted(CATEGORY_PROFILES)),
        budget=dict(GENERATED_OBJECTS_MAX_PER_STAGE))


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    from importlib.util import find_spec
    package = Path(find_spec("rfw_w.scenes").submodule_search_locations[0])
    return "rfw_w/scenes/" + resolved.relative_to(package).as_posix() if package in resolved.parents else str(resolved)


class ShellProviderError(RoomError):
    """A provider could not produce a shell. The message leads with the failure code."""


@runtime_checkable
class RoomShellProvider(Protocol):
    """What the pipeline needs from a shell builder, and nothing else.

    Deliberately narrow: the caller has a plan and wants a shell it can check. A provider that
    also returned its own verdict, or that needed to know about Isaac, would make "which builder
    ran" leak into the gates instead of staying behind this boundary.
    """

    name: str

    def build(self, plan: RoomPlan, *, output_dir: Path) -> ShellBuild: ...


@dataclass(frozen=True, slots=True)
class BuiltinShellProvider:
    """The shell our own solver builds, measured against itself.

    It is the **control**, not leftover code. Its value is that its answer is known in advance: it
    authors the walls from the plan's polygon and the holes from the plan's openings, so walls sit
    on the plan's lines with a residual spread of exactly zero and every opening is the width the
    plan asked for. A gate that fails this build is reporting a defect in the gate.

    Its measurements are therefore exact rather than estimated -- and they are reported in the same
    fields the measured provider reports, so the two are comparable rather than merely adjacent.
    """

    name: str = "builtin"

    def build(self, plan: RoomPlan, *, output_dir: Path) -> ShellBuild:
        room = plan.rooms[0]
        # The conversion is `plan_to_shell`'s, and then only the `source` changes. That field is
        # how a reader tells one builder's shell from another's (ADR-001), and `plan:core` -- what
        # the conversion stamps -- would say this shell is the request. It is not: it is the room
        # this provider *built*, which happens to coincide with the plan because that is what it
        # was asked to build. The coincidence is the claim under test, so it must not be baked
        # into the label.
        shell = replace(plan_to_shell(plan), source=f"{self.name}:{plan.source}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "shell.json").write_text(
            json.dumps(shell.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        # `room.wall_line_offset_spread_m` is zero by construction and not merely by luck: the
        # polygon it reports *is* the polygon it built. Saying so in the same field the measured
        # provider fills is what lets one gate read both.
        measurements: dict[str, Any] = {
            "room.height_m": room.ceiling_height_m,
            "room.wall_line_offset_m": 0.0,
            "room.wall_line_offset_spread_m": 0.0,
            "wall.thickness_m": ShellOptions().wall_thickness_m,
            "openings.count": len(shell.openings),
            "openings.plan_count": len(plan.openings),
        }
        for opening in shell.openings:
            measurements[f"openings.{opening.opening_id}.width_m"] = opening.width_m
            measurements[f"openings.{opening.opening_id}.height_m"] = opening.height_m
            measurements[f"openings.{opening.opening_id}.sill_z_m"] = opening.sill_z_m
        return ShellBuild(
            shell=shell,
            provider=self.name,
            measurements=measurements,
            not_measured=(
                "room.floor_area_m2: the builder authors the floor from the plan's polygon and "
                "never re-derives it from a mesh, so there is no independent area to report",
            ),
            not_compared=_AUTHORED_NOT_COMPARED,
        )


@dataclass(frozen=True, slots=True)
class InfinigenShellProvider:
    """Infinigen builds the room from the plan, and then we measure what it built.

    The sequence is three steps and each one exists because of something Infinigen does not do:

    1. **Write the floor plan** in its own vocabulary (`floor_plan_payload`), as a file. The
       `eval` on the shape strings (`room/predefined.py:141`) means only this repository's
       serialiser may write it.
    2. **Build** with `-t coarse`, which is the stage that leaves a `scene.blend` behind.
    3. **Measure** the mesh with the ruler, because Infinigen does not hand its geometry back --
       every polygon in `solve_state.json` is the literal string ``"<not-serialized>"``.

    A build that fails raises `SHELL_BUILD_FAILED` carrying Blender's own last lines. The reason is
    not politeness: the preset path calls `eval` and unions room polygons, so a malformed plan ends
    inside the solidifier as a solver exception, and a reader needs that text to tell a bad plan
    from a bad tree.
    """

    name: str = "infinigen"
    infinigen_root: Path | None = None
    #: The opt-in. `None` is the default path: the third-party entry script, no reservations, and
    #: nothing recorded about constraints because none were asked for.
    background_constraints: BackgroundConstraints | None = None

    def _root(self) -> Path:
        root = self.infinigen_root
        if root is None:
            configured = os.environ.get(INFINIGEN_ROOT_ENV, "")
            if not configured:
                raise ShellProviderError(
                    f"SHELL_BUILD_FAILED({INFINIGEN_ROOT_ENV} is not set): the infinigen provider "
                    "drives a pinned checkout rather than whatever is installed, and the checkout's "
                    "location is machine-bound, so it comes from .env.local"
                )
            root = Path(configured)
        if not root.is_dir():
            raise ShellProviderError(
                f"SHELL_BUILD_FAILED({INFINIGEN_ROOT_ENV}={root} is not a directory)"
            )
        return root

    def command(self, plan: RoomPlan, *, run_dir: Path) -> list[str]:
        """The exact argv this provider runs. Public so a report can print what was executed.

        A recorded measurement is only reproducible if the command that produced it is visible, and
        the probe's numbers were once quoted from a scratch run for exactly as long as it took
        somebody to ask how they were produced.

        **Two entries, chosen by the opt-in.** Without `background_constraints` this is the
        third-party wrapper's own script, byte for byte as it has always been -- which is what keeps
        the commands recorded in SPEC-004/005/009 evidence reproducible. With it, our entry runs
        instead and is told where the reservation bundle and the record go; the abort switches that
        make an unsatisfied solve loud travel as gin overrides beside them.
        """
        root = self._root()
        infinigen = root
        room = plan.rooms[0]
        # Absolute, every one of them. The build runs with `cwd` set to the Infinigen checkout --
        # it has to, because the wrapper imports the tree by relative path -- so a relative
        # `--output_folder` or `floor_plan` is resolved against *that* tree. Measured: a relative
        # floor plan produced `FileNotFoundError` from `predefined.py:127` inside a traceback that
        # named our path, which is a confusing way to learn the file was looked for elsewhere.
        overrides = [
            f"Solver.floor_plan={(run_dir / 'floor_plan.json').resolve()}",
            f"RoomConstants.global_params.wall_height={room.ceiling_height_m}",
            f"RoomConstants.global_params.wall_thickness={INFINIGEN_WALL_THICKNESS_M}",
            *INFINIGEN_STAGE_OVERRIDES,
        ]
        entry = ["-m", BACKGROUND_ENTRY_SCRIPT]
        off_overrides = list(INFINIGEN_STAGE_OVERRIDES_OFF)
        if self.background_constraints is not None:
            policy = self.background_constraints.policy
            if self.background_constraints.generation:
                # Said explicitly rather than by leaving an override out: a command that does not
                # mention a stage is evidence of nothing (SPEC-009's finding about `floating_objs`),
                # and this is the stage whose product the constraints are there to keep in bounds.
                off_overrides = [item for item in off_overrides
                                 if item != "compose_indoors.populate_assets_enabled=False"]
                overrides.append("compose_indoors.populate_assets_enabled=True")
            entry = ["-m", BACKGROUND_ENTRY_SCRIPT,
                     "--pvscene-reservations", str((run_dir / BUNDLE_FILENAME).resolve()),
                     "--pvscene-provenance", str((run_dir / INJECTION_FILENAME).resolve()),
                     "--pvscene-entry", str((_worker_path(BACKGROUND_ENTRY_SCRIPT)).resolve())]
            if policy.abort_unsatisfied:
                # Requirement 5's spelling, passed where upstream reads it: an unsatisfied solve
                # must raise rather than return a scene that violates the constraints.
                overrides.extend(f"{abort_param(stage)}=True" for stage in SOLVE_STAGES)
        return [
            str(infinigen / "blender" / "4.2" / "python" / "bin" / "python3.11"),
            *entry,
            "--output_folder", str((run_dir / "scene").resolve()),
            "--seed", str(plan.provenance.seed),
            "-t", "coarse",
            "-g", "base_indoors", "custom_solve",
            "-p", *overrides, *off_overrides,
        ]

    # -- the reservation side of a build -------------------------------------------------------
    def write_bundle(self, plan: RoomPlan, *, run_dir: Path) -> ReservationBundle:
        """Write the request *before* the build, so the command has something to point at."""
        constraints = self.background_constraints
        assert constraints is not None  # guarded by the caller
        bundle = reservation_bundle_for(
            plan, constraints,
            entry_script=_worker_path(BACKGROUND_ENTRY_SCRIPT),
            patch_module=_worker_path(BACKGROUND_PATCH_MODULE))
        (run_dir / BUNDLE_FILENAME).write_text(json.dumps(bundle.to_dict(), indent=2) + "\n")
        return bundle

    def read_injection(self, bundle: ReservationBundle, *, run_dir: Path) -> dict[str, Any]:
        """Read the build's own record and refuse the ways it can be a build we did not ask for.

        Three refusals, and each is a different fact. A missing record means the patch never
        installed (so nothing was constrained). A record whose digests disagree with the bundle
        means the build ran code other than what was requested -- the record is what makes the
        constraint auditable, so a record that cannot be trusted is worse than none. A record whose
        reservation objects did not reach the export list is a constraint that was injected and then
        leaked, which is SPEC-011 requirement 9's failure.
        """
        record_path = run_dir / INJECTION_FILENAME
        if not record_path.is_file():
            raise ShellProviderError(
                f"BACKGROUND_INJECTION_FAILED(no {INJECTION_FILENAME} under {run_dir}): the build "
                "was asked for constraints and left no record of them")
        record = json.loads(record_path.read_text())
        if record.get("injection") != INJECTION_APPLIED:
            raise ShellProviderError(
                f"BACKGROUND_INJECTION_FAILED({record.get('injection')}: "
                f"{record.get('injection_reason') or record.get('error') or 'no reason recorded'})")
        for name in ("patch_sha256", "entry_script_sha256"):
            if record.get(name) != getattr(bundle, name):
                raise ShellProviderError(
                    f"BACKGROUND_PATCH_UNRECORDED({name}): the build recorded "
                    f"{record.get(name)} and the request carried {getattr(bundle, name)}")
        removed = record.get("objects_removed_before_export", {})
        if not removed.get("removed"):
            raise ShellProviderError(
                "BACKGROUND_RESERVATION_OBJECT_LEAKED(not removed): the reservation objects were "
                "still in the scene when the assets were realised, so they can reach the export")
        return record

    def _refuse_unbuildable_windows(self, plan: RoomPlan) -> None:
        """Refuse a plan whose window this builder would silently not build.

        `floor_plan_payload` already widens a window's span to `WINDOW_HOST_SPAN_M` and clamps it to
        the wall, so this can only fire on a wall shorter than that span -- a plan this repository's
        drawer never produces, since it puts a window only on a wall with
        `WINDOW_EDGE_INSET_M` to spare (`solvers/room.py::_windows_on_wall`), which is 1.4 m of wall
        for the default 1.2 m request. It is here for the plans the contract permits and the drawer
        does not make (`RoomPlan.source` may be ``external`` or ``bank``), and for the case where
        the two constants stop agreeing: `WINDOW_HOST_SPAN_M` is fixed and
        `buildable_window_span_m` moves with the wall thickness, so a future provider asking for
        much thinner walls would land here.

        The alternative is worse than a refusal. Infinigen does not reject a short window span, it
        drops it -- no cutter, no hole, and nothing in `scene.blend` to say one was asked for -- so
        the run would proceed to a room that is missing a window it was told to build, and the gate
        would report the absence without the reason. This is the one place that can name the reason.
        """
        needed = buildable_window_span_m(INFINIGEN_WALL_THICKNESS_M)
        for opening in plan.openings:
            if opening.kind != OpeningKind.WINDOW.value:
                continue
            span = opening_host_span(plan, opening)
            length = math.dist(span[0], span[1])
            if length <= needed:
                raise ShellProviderError(
                    f"SHELL_WINDOW_SPAN_UNBUILDABLE({opening.opening_id}, span_m={length:.4f}, "
                    f"needed_m={needed:.4f}): the plan's window sits on a wall that cannot hold a "
                    f"span longer than {needed:.4f} m ({INFINIGEN_WALL_THICKNESS_M} m walls), and "
                    "this builder drops a window it is given less than that"
                )

    def build(self, plan: RoomPlan, *, output_dir: Path) -> ShellBuild:
        root = self._root()
        run_dir = output_dir / "infinigen_build"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ShellProviderError(
                f"SHELL_BUILD_FAILED({run_dir} already has files): a build is evidence and a re-run "
                "writes a new directory; merging into one is how a stale measurement gets read as "
                "a fresh one"
            )
        self._refuse_unbuildable_windows(plan)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "floor_plan.json").write_text(
            json.dumps(floor_plan_payload(plan), indent=2) + "\n"
        )
        # Two files, and handing either one to the wrong consumer fails in a way that reads like a
        # bug in the consumer. `floor_plan.json` is Infinigen's vocabulary (`rooms` is a *mapping*
        # keyed `"<roomtype>_<storey>/<idx>"`) and is what the build is fed; `room_plan.json` is
        # this repository's (`rooms` is a list) and is what the ruler is fed. Measured: pointing
        # the ruler at the floor plan raised `KeyError: 0` from `plan["rooms"][0]`.
        (run_dir / "room_plan.json").write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
        bundle = None
        if self.background_constraints is not None:
            # Written before the command is built, because the command names it: a bundle created
            # after the run could describe reservations the build never saw.
            bundle = self.write_bundle(plan, run_dir=run_dir)

        argv = self.command(plan, run_dir=run_dir)
        # The wrapper is imported as a module inside Blender's Python, so the tree has to be on
        # `PYTHONPATH` -- and the inherited one has to go, because an Isaac shell leaves a Python
        # 3.10 stdlib there that surfaces as `AssertionError: SRE module mismatch` under 3.11.
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        # `src` joins the two trees so the patch can import the bundle's schema: the patch and the
        # provider read one contract rather than two spellings of it.
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(root)])
        environment.setdefault("OPENBLAS_NUM_THREADS", "1")

        (run_dir / "command.txt").write_text(" ".join(argv) + "\n")
        try:
            completed = subprocess.run(
                argv, cwd=root, env=environment, capture_output=True, text=True,
                timeout=BUILD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as expired:
            raise ShellProviderError(
                f"SHELL_BUILD_FAILED(the build did not finish within {BUILD_TIMEOUT_S} s): "
                f"{self.name} left {run_dir} for inspection"
            ) from expired
        (run_dir / "stdout.txt").write_text(completed.stdout or "")
        (run_dir / "stderr.txt").write_text(completed.stderr or "")
        if completed.returncode != 0:
            raise ShellProviderError(
                f"SHELL_BUILD_FAILED({self.name} exited {completed.returncode}): "
                + _tail(completed.stderr or completed.stdout or "")
            )

        blend = run_dir / "scene" / "scene.blend"
        if not blend.is_file():
            raise ShellProviderError(
                f"SHELL_BUILD_FAILED(no scene.blend under {run_dir / 'scene'}): the build reported "
                "success and left nothing to measure"
            )

        record = None
        if bundle is not None:
            # Before the ruler runs, not after: a build that was asked for constraints and did not
            # send them is refused here, so the measurement below is never quoted as if it came from
            # a constrained room.
            record = self.read_injection(bundle, run_dir=run_dir)

        report_path = run_dir / "measurement.json"
        measured = subprocess.run(
            [
                str(root / "blender" / "blender"),
                "-b", str(blend.resolve()),
                "-P", str(_worker_path("rfw_w.scenes.workers.blender.measure_built_shell")),
                "--",
                "--plan", str((run_dir / "room_plan.json").resolve()),
                "--out", str(report_path.resolve()),
            ],
            cwd=root, env=environment, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S,
        )
        (run_dir / "measure_stdout.txt").write_text(measured.stdout or "")
        (run_dir / "measure_stderr.txt").write_text(measured.stderr or "")
        if not report_path.is_file():
            # The exit code is **not** the signal here, and the wording says so: `blender -b -P`
            # returns 0 even when the script it ran raised, so a check on the code alone would
            # have reported a measurement that never happened as a clean run. The report being on
            # disk is the only thing that means the ruler finished -- which is why the check reads
            # the artefact rather than the status.
            raise ShellProviderError(
                f"SHELL_BUILD_FAILED(the ruler left no report, exiting {measured.returncode}; "
                "`blender -b -P` returns 0 even when the script raises): "
                + _tail(measured.stderr or measured.stdout or "")
            )

        report = json.loads(report_path.read_text())
        try:
            build = shell_from_report(plan, report, provider=self.name)
        except ShellBuildError as error:
            # A distinct code, because "the build failed" and "the build succeeded and is not the
            # planned room" are different facts and the ledger must not conflate them. The room
            # exists -- `scene.blend` and `measurement.json` are on disk and are named here -- it
            # just has an opening where the plan has none, or none where the plan has one.
            #
            # It is the *read-back* that refuses first (SPEC-003), not the gate: `shell_from_report`
            # will not build a shell from a report whose holes do not correspond to the plan,
            # because a hole on no planned segment has no id to be called by. The gate detects the
            # same disagreement when it is handed a shell, and the negative tests cover that path;
            # this one is the path a real build takes.
            raise ShellProviderError(
                f"SHELL_BUILT_DOES_NOT_MATCH_PLAN({self.name}, geometry={blend}): {error}"
            ) from error
        measurements = dict(build.measurements)
        if record is not None:
            # The provenance summary, in the same free-form field every other measurement lives in,
            # so a delivery manifest can answer "was this shell built with the reservations?" without
            # a second artefact to look up. The full record stays beside the build
            # (`infinigen_build/constraint_injection.json`), which is where the audit reads it.
            removed = record.get("objects_removed_before_export", {})
            measurements.update({
                "constraint.injection": record.get("injection"),
                "constraint.injection_record": str((run_dir / INJECTION_FILENAME).resolve()),
                "constraint.bundle": str((run_dir / BUNDLE_FILENAME).resolve()),
                "constraint.regions": sorted({item["region"]
                                              for item in record.get("reservation_objects", ())}),
                "constraint.reservation_objects_removed": bool(removed.get("removed")),
                "constraint.patch_sha256": record.get("patch_sha256"),
                "constraint.entry_script_sha256": record.get("entry_script_sha256"),
                "constraint.upstream_pin": record.get("upstream_pin"),
                "constraint.abort_unsatisfied": record.get("abort_unsatisfied", {}).get("values", {}),
                "constraint.policy": record.get("policy"),
                "constraint.terms": record.get("terms"),
            })
        return ShellBuild(
            shell=build.shell,
            provider=self.name,
            measurements=measurements,
            not_measured=build.not_measured,
            geometry_uris=(str(blend),),
            material_uris=(),
            not_compared=_MEASURED_NOT_COMPARED,
        )


#: The quantities the gate does **not** compare, whatever built the shell.
#:
#: A constant rather than a per-provider choice, because "what the gate compares" is a property of
#: the *gate* and not of the builder. Measured: `builtin` first reported an empty `not_compared`
#: while `infinigen` reported five entries, and a reader comparing the two reports would have
#: concluded that one builder's opening widths were held to the plan and the other's were not --
#: neither is true, and the reports have to be comparable for SPEC-004 §4 R2 to mean anything.
NOT_COMPARED_FIELDS: tuple[str, ...] = (
    "openings.*.width_m",
    "openings.*.height_m",
    "openings.*.sill_z_m",
    "openings.*.centre_xy",
    "wall.thickness_m",
)

#: Why each of those is outside the comparison set, for a shell **measured from a mesh**.
#:
#: This is not `not_measured`, and the distinction is the point: `not_measured` means "we did not
#: establish this", and putting a number we *did* measure in there would understate the evidence.
#: What these have in common is that the plan does not constrain them, so comparing them would
#: report a disagreement with something never agreed -- SPEC-001 R6 measured that a 0.9 m and a
#: 2.0 m door segment produce the same 0.929 m door.
_MEASURED_NOT_COMPARED: tuple[str, ...] = (
    "openings.*.width_m: measured from the mesh, but Infinigen sizes an opening from its own "
    "constants (solidifier.py:543 uses constants.door_width), not from the plan's wall segment",
    "openings.*.height_m: same mechanism; the door reads 2.0221 m against a plan that asked 2.05",
    "openings.*.sill_z_m: Infinigen has no expression for a per-window sill -- the probe measured "
    "all three office windows at 0.92 m against a plan of 0.815 / 1.064 / 1.007",
    "openings.*.centre_xy: the hole is placed anywhere within the host span the plan was turned "
    "into (`room_plan.opening_host_span`), so which wall it is on is compared and where along that "
    "wall is not",
    "wall.thickness_m: the measured 0.05 m is the interior lining, half of the 0.1 m the build was "
    "asked for, so a comparison would be against a different quantity of the same name",
)

#: The same fields, for a shell this repository authored. The reasons differ -- this builder
#: reproduces the plan's opening sizes rather than ignoring them -- and the *fields* do not, which
#: is the part that has to match.
_AUTHORED_NOT_COMPARED: tuple[str, ...] = (
    "openings.*.width_m: reproduced exactly, because this builder authors the hole from the plan's "
    "own width; still outside the gate's comparison set, which is the same set for every provider",
    "openings.*.height_m: same -- reproduced from the plan, not compared by the gate",
    "openings.*.sill_z_m: same -- reproduced from the plan, not compared by the gate",
    "openings.*.centre_xy: the plan's segment midpoint, which is what this builder was handed; "
    "which wall the opening is on is compared and where along that wall is not",
    "wall.thickness_m: authored as ShellOptions.wall_thickness_m, and not compared because the "
    "plan carries no thickness to compare it against",
)


def _tail(text: str, *, lines: int = 6) -> str:
    """The last few lines of a tool's output, which is where the actual error is."""
    kept = [line for line in text.strip().splitlines() if line.strip()]
    return " | ".join(kept[-lines:]) if kept else "no output"


#: The providers by name, so a caller can select one without importing the classes. Both are
#: always available to name: a provider that cannot run on this machine fails *by name* at build
#: time (`SHELL_BUILD_FAILED`), which is a fact a report can carry, rather than by being absent
#: from a registry, which is a fact it cannot.
PROVIDERS: dict[str, RoomShellProvider] = {
    "builtin": BuiltinShellProvider(),
    "infinigen": InfinigenShellProvider(),
}


def provider_named(name: str) -> RoomShellProvider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ShellProviderError(
            f"SHELL_PROVIDER_UNKNOWN({name}): known providers are {sorted(PROVIDERS)}"
        ) from None


def minimum_door_width_m() -> float:
    """The passage floor the shell policy already uses, from the one place it is written.

    Read from `ShellOptions` rather than restated here: `validate_shell` derives its door minimum
    the same way, and two copies of "how wide a door has to be" would drift. ADR-003 records that
    neither number yet comes from the robot's own `RobotProfile` -- that defect is unchanged by
    this function and is not made worse by it.
    """
    options = ShellOptions()
    return options.robot_footprint_m + options.clearance_margin_m

def export_shell(build: ShellBuild, *, output_dir: Path | None = None,
                 material_profile: MaterialExportProfile | None = None) -> Path:
    """Export the already measured source, never rebuild it with another random stream."""
    if build.provider != "infinigen" or not build.geometry_uris:
        raise ShellProviderError("EXPORT_BACKEND_UNAVAILABLE(no Infinigen source)")
    blend = Path(build.geometry_uris[0]).resolve()
    destination = (output_dir.resolve() if output_dir is not None else blend.parent.parent / "export") / "shell.usdc"
    if destination.parent.exists() and any(destination.parent.iterdir()):
        raise ShellProviderError("EXPORT_BACKEND_UNAVAILABLE(output already contains evidence)")
    root = InfinigenShellProvider()._root()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    material_args = []
    if material_profile is not None:
        material_args = ['--material-profile', material_profile.version,
                         '--room-plan', str(blend.parent.parent / 'room_plan.json'),
                         '--resolution', str(material_profile.resolution),
                         '--bake-device', material_profile.device]
    try:
        result = subprocess.run([
            str(root / "blender/blender"),
            "-b", str(blend), "--python-exit-code", "1",
            "-P", str(_worker_path("rfw_w.scenes.workers.blender.export_built_shell")),
            "--", "--out", str(destination),
            "--measurement", str(blend.parent.parent / "measurement.json"),
            *material_args,
        ], env=environment, cwd=root, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired as error:
        raise ShellProviderError("EXPORT_BACKEND_UNAVAILABLE(export timeout)") from error
    (destination.parent / "stdout.txt").write_text(result.stdout or "")
    (destination.parent / "stderr.txt").write_text(result.stderr or "")
    if result.returncode or not destination.with_suffix('.export.json').is_file():
        gap = re.search(r'MATERIAL_[A-Z_]+\([^\n]+\)', (result.stdout or '') + '\n' + (result.stderr or ''))
        if gap:
            raise ShellProviderError(gap.group(0))
        raise ShellProviderError("EXPORT_BACKEND_UNAVAILABLE(native USD export failed; see export logs)")
    from rfw_w.scenes.workers.isaac.shell_reference import inspect_shell
    inspect_shell(str(destination))
    return destination


#: The assembly the generated-furniture export writes, and the file the assembly step reads.
GENERATED_ASSEMBLY_FILENAME = "objects.json"


def export_generated_furniture(build: ShellBuild) -> Path:
    """Export the objects this build's rejection pool kept, in the frame a delivery instance needs.

    The counterpart of `export_shell` for the other half of the scene: the shell is the room, and
    these are the pieces the background branch realised and the pool allowed to stay. Before this
    existed they stopped at the build (SPEC-011's PLAN recorded the boundary): the ledger named
    them, the ruler measured them, and no delivery carried them.

    Runs Blender on the build's own saved scene, so the geometry and pose are the ones the room
    really has rather than a re-derivation. Idempotent per ledger: a rebuild of the same plan
    exports once and reuses the file, because the second export would be the same evidence written
    over itself -- and a *different* ledger in the same directory is refused instead.
    """
    if build.provider != "infinigen" or not build.geometry_uris:
        raise ShellProviderError("GENERATED_EXPORT_UNAVAILABLE(no Infinigen source)")
    blend = Path(build.geometry_uris[0]).resolve()
    build_dir = blend.parent.parent
    ledger = build_dir / INJECTION_FILENAME
    if not ledger.is_file():
        raise ShellProviderError(
            f"GENERATED_EXPORT_UNAVAILABLE(no {INJECTION_FILENAME} under {build_dir}): the pool's "
            f"ledger is the only place the kept objects are named, and a build without one has "
            f"nothing to export"
        )
    out = build_dir / "generated"
    target = out / GENERATED_ASSEMBLY_FILENAME
    ledger_sha256 = hashlib.sha256(ledger.read_bytes()).hexdigest()
    if target.is_file():
        existing = json.loads(target.read_text())
        if existing.get("ledger_sha256") == ledger_sha256:
            return target
        raise ShellProviderError(
            "GENERATED_EXPORT_UNAVAILABLE(output already contains evidence): the export directory "
            "holds an assembly of a different ledger"
        )
    if out.exists() and any(out.iterdir()):
        raise ShellProviderError("GENERATED_EXPORT_UNAVAILABLE(output already contains evidence)")
    record = json.loads(ledger.read_text())
    pin = str(record.get("upstream_pin") or INFINIGEN_PIN)
    root = InfinigenShellProvider()._root()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run([
            str(root / "blender/blender"),
            "-b", str(blend), "--python-exit-code", "1",
            "-P", str(_worker_path("rfw_w.scenes.workers.blender.export_generated_furniture")),
            "--", "--ledger", str(ledger), "--out", str(out), "--pin", pin,
        ], env=environment, cwd=root, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired as error:
        raise ShellProviderError("GENERATED_EXPORT_UNAVAILABLE(export timeout)") from error
    (out / "stdout.txt").write_text(result.stdout or "")
    (out / "stderr.txt").write_text(result.stderr or "")
    if result.returncode or not target.is_file():
        raise ShellProviderError(
            "GENERATED_EXPORT_UNAVAILABLE(the Blender export failed; see generated/stderr.txt "
            f"beside {build_dir})"
        )
    return target
