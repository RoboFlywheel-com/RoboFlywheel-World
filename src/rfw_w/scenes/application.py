# SPDX-License-Identifier: Apache-2.0
"""Scene generation and bounded repair with private per-run state."""

from __future__ import annotations
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence
from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.readiness import EvidenceLedger, ReadinessLevel, resolve_readiness
from rfw_w.scenes.contracts.scene_spec import SceneSpec
from rfw_w.scenes.contracts.tool import EvidenceStatus
from rfw_w.scenes.solvers.furnish import ScenePlan
from rfw_w.scenes.orchestrator.budgets import Budget, BudgetExhausted
from rfw_w.scenes.orchestrator.driver import Driver
from rfw_w.scenes.solvers.room import RoomError
from rfw_w.scenes.solvers.shell_providers import BACKGROUND_GENERATION_POLICY, ShellProviderError, generation_provider, provider_named
from rfw_w.scenes.solvers.scene_ir import scene_ir_from_build
from rfw_w.scenes.solvers.randomization import randomize_plan
from rfw_w.scenes.solvers.zones import ZoneError
from rfw_w.scenes.assembly.generated import assemble_generated
from rfw_w.scenes.assembly.validation import measure_shell_matches_plan, measure_deterministic_gates
from rfw_w.scenes.assembly.cameras import build_worker_request
from rfw_w.scenes.assembly.lighting import lighting_block_for, lighting_plan_for
from rfw_w.scenes.assembly.placement import BuildState, CoreUnplaced, build_plan, build_scene
from rfw_w.scenes.delivery.compose import write_delivery
from rfw_w.scenes.delivery.evidence import evidence, gates_from_worker, not_run_gates, unsupported_gates
from rfw_w.scenes.runtime import COMPILE_TIMEOUT_S, DEFAULT_EXPERIMENT, OUTPUTS_DIR, load_env_file, load_spec, step, vlm_config

BUILD_MEMORY = 64
SCENE_VERSION_PLACEHOLDER = '0' * 64

@dataclass(frozen=True)
class GenerationOptions:
    """Inputs and bounded execution choices for one fresh scene run."""
    experiment: str | None = None
    run: str | None = None
    spec: Path | None = None
    output_dir: Path | None = None
    prompt: str | None = None
    prompt_file: Path | None = None
    compile: bool = False
    no_compile: bool = False
    compile_only: bool = False
    compiler_model: str | None = None
    seed: int | None = None
    scene_id: str | None = None
    env_file: Path | None = None
    asset_bindings: Path | None = None
    asset_license: str | None = None
    active_gpu: int = 0
    domain_randomization: bool = False
    variant_seed: int | None = None
    max_variant_attempts: int = 8
    shell_provider: str = 'builtin'
    lighting_profile: str = 'mixed'
    room_bounds: str | None = None
    background_generation: bool = False
    background_gaps: str = ''
    skip_isaac: bool = False
    request_only: bool = False
    no_loop: bool = False
    deadline_s: float = 1800.0
    max_model_requests: int = 3
    max_local_repairs: int = 6
    no_replan: bool = False
    vlm_model: str | None = None
    agent_model: str | None = None

    def __post_init__(self) -> None:
        if self.asset_bindings is not None and (not self.asset_license or not self.asset_license.strip()):
            raise ValueError("--asset-bindings requires an explicit --asset-license")
        if self.asset_license is not None and self.asset_bindings is None:
            raise ValueError("--asset-license requires --asset-bindings")
        if self.spec is not None and self.experiment is not None:
            raise ValueError('--spec and --experiment are mutually exclusive')
        if self.prompt is not None and self.prompt_file is not None:
            raise ValueError('--prompt and --prompt-file are mutually exclusive')
        if self.compile and self.no_compile:
            raise ValueError('--compile and --no-compile are mutually exclusive')
        if self.no_compile and (self.prompt is not None or self.prompt_file is not None):
            raise ValueError('--no-compile needs a JSON spec')
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError('prompt must contain task text')
        if not math.isfinite(self.deadline_s) or self.deadline_s <= 0:
            raise ValueError('deadline must be finite and positive')
        if not 1 <= self.max_model_requests <= 5:
            raise ValueError('model request budget must be within 1..5')
        if self.max_variant_attempts < 1 or self.max_local_repairs < 0 or self.active_gpu < 0:
            raise ValueError('invalid execution budget or GPU index')
        if self.shell_provider not in ('builtin', 'infinigen'):
            raise ValueError('unknown shell provider')
        if self.lighting_profile not in ('mixed', 'daylight', 'warm_artificial'):
            raise ValueError('unknown lighting profile')
        if self.background_generation and (self.shell_provider != 'infinigen' or not self.background_gaps.strip()):
            raise ValueError('background generation requires Infinigen and explicit categories')
        for identity in (self.experiment, self.run):
            if identity is not None and (not identity or '/' in identity or '\\' in identity or identity in ('.', '..')):
                raise ValueError('example and run identities must be single path segments')

def _instance_drift(before: dict, after: dict) -> str:
    """Which placed instances moved between two builds of the same plan, and in what field."""

    def by_id(payload: dict) -> dict:
        return {str(item.get("instance_id")): item for item in payload.get("instances", ())}

    left, right = by_id(before), by_id(after)
    gone = sorted(set(left) - set(right))
    appeared = sorted(set(right) - set(left))
    moved: list[str] = []
    for name in sorted(set(left) & set(right)):
        if left[name] == right[name]:
            continue
        fields = sorted(
            key for key in set(left[name]) | set(right[name])
            if left[name].get(key) != right[name].get(key)
        )
        moved.append(f"{name} ({', '.join(fields)})")
    parts: list[str] = []
    if gone:
        parts.append("only in the first build: " + ", ".join(gone))
    if appeared:
        parts.append("only in the rebuild: " + ", ".join(appeared))
    if moved:
        parts.append("changed: " + ", ".join(moved[:8]))
    if not parts:
        # Same members, same contents, different order: the set is stable and its serialisation is
        # not, which is its own bug and would read as one here.
        parts.append("the same instances in a different order")
    return "instances that moved: " + " | ".join(parts)


class SceneHost:
    """One scene run behind the bounded repair driver protocol.

    A class rather than free functions because the loop needs the pipeline to be *resumable*: a
    repair hands back a changed plan, the scene is rebuilt from it, and every gate is re-measured
    against the new version. Holding the scene, the ledger and the version together is what makes
    that possible -- and the ledger is deliberately *not* rebuilt between iterations, because the
    evidence of what failed at the previous version is what shows the repair was needed.
    """

    def __init__(
        self,
        *,
        spec: SceneSpec,
        out: Path,
        args: GenerationOptions,
        requested: ReadinessLevel,
        requested_backends: Sequence[str],
    ) -> None:
        self.build_state = BuildState()
        from rfw_w.scenes.assets.providers.generated_package import load_bindings
        self.asset_bindings = (load_bindings(args.asset_bindings, license=args.asset_license)
                               if args.asset_bindings is not None else {})
        self.spec = spec
        self.out = out
        self.args = args
        self.model_budget: Budget | None = None
        self.requested = requested
        self.requested_backends = tuple(requested_backends)
        self.ledger = EvidenceLedger()
        # A plan belongs to this request; package imports never instantiate example scenes.
        self.plan: ScenePlan = build_plan(spec)
        self.support_id: str = self.plan.support_id
        self.target_id: str = self.plan.target_id
        self.scene: dict = {}
        self.version: str = SCENE_VERSION_PLACEHOLDER
        self.critique: Any = None
        # What a rebuild from an *unchanged* plan moved, if anything. `build` is supposed to be a
        # function of (spec, seed, plan): rebuilding the same plan has to produce the same version,
        # or the loop's restore path lands on a scene no gate has measured and reports `draft` with
        # everything stale. The generic note says that happened; this says where.
        self.build_diagnostics: list[str] = []
        # Every plan built in this run, against the version payload it produced. Keyed by the plan
        # rather than remembering only the last one, because the case this exists for is a *restore*
        # -- build(P), build something else, build(P) again -- and the rebuild of P is never adjacent
        # to the original.
        self._built: dict[str, dict] = {}
        # One shell build per *plan*, not per scene version. A repair changes the scene version
        # without changing the room, and `run_deterministic_gates` runs after every rung -- so
        # keying this on the version would re-run a three-minute Infinigen build for a room that
        # had not moved. The plan's own content hash is the right key precisely because the room
        # is a function of the plan and of nothing else.
        self._shell_builds: dict[str, Any] = {}
        # The verdict that came with it. Kept here rather than recomputed by the driver: which
        # criteria are below threshold is what decides the ladder's starting track, and a driver
        # holding only the critique would have to re-evaluate it against its own thresholds -- a
        # second opinion that is right today only because both use the default profile.
        self.visual_gate: Any = None
        self.worker_report: dict = {}
        # A loop may render several scene versions. Each version gets immutable request and output
        # paths; these point at the most recently measured version for critique and final delivery.
        self.worker_dir: Path | None = None
        self.worker_request_path: Path | None = None
        self.backend_attempts: dict[str, int] = {}
        self._backend_by_version: dict[str, tuple] = {}
        self._critique_by_version: dict[str, tuple] = {}
        self.randomization: Any = None
        self.generation_seed: int = spec.seed
        self.relation_evidence: Any = None
        self.wall_thickness_m_value: float = 0.0

    # -- building --------------------------------------------------------------------------------

    def _version_for(self, scene: dict, plan: ScenePlan) -> str:
        """The scene version: everything a gate, a render or a critique depends on.

        It used to hash only the spec, the shell, the desk and the cup. The comment above it claimed
        that "a change to any pose or asset produces a different version", which was false: moving
        the chair, the rug, the lamp or the picture produced the *same* version. That was survivable
        while nothing could move them. With the loop wired it is fatal twice over -- the orchestrator
        refuses to commit a change that does not change the version, and stale evidence would go on
        satisfying gates for a scene it never measured.

        The plan is hashed too, so a repair that changes a constraint without moving anything (a gap,
        the lighting) still produces a new version and still invalidates what depended on the old
        setting.

        A separate `"task_target"` entry used to sit beside these, holding the cup's asset and pose
        under a hardcoded key. It was redundant -- `instances` already carries every instance's
        asset and pose -- and the plan now names the support and the target, so which instance the
        task is about is part of the hash without a second copy that can disagree with it.
        """
        return content_hash(self._version_payload(scene, plan))

    def _version_payload(self, scene: dict, plan: ScenePlan) -> dict:
        """Everything the version covers, as the value that is hashed."""
        return {
            "scene_ir": scene["scene_ir"].content_payload(),
            "plan": plan.to_dict(),
        }

    def _adopt_version(self, scene: dict, plan: ScenePlan) -> str:
        """Take the version for a built scene, and record how it moved if the plan did not.

        Called from `build` and from the one place that recomputes a version without rebuilding
        (the post-reseed audit trail), so a drift between what the scene holds and what the version
        claims is noticed wherever it is made.
        """
        payload = self._version_payload(scene, plan)
        self._note_rebuild_difference(payload)
        return content_hash(payload)

    def _note_rebuild_difference(self, payload: dict) -> None:
        """Name what rebuilding an unchanged plan changed.

        The scene is meant to be a function of (spec, seed, plan), so a rebuild from the same plan
        must produce the same version. When it does not, something outside the plan is an input --
        a cached preparation that differs from a fresh one, a field mutated after the build -- and
        the loop's restore path silently lands on a version nothing has measured. Naming the part
        that moved is the difference between a diagnosis and a shrug; the driver's own note can only
        say that the plan is not the whole story.
        """
        key = json.dumps(payload["plan"], sort_keys=True)
        previous = self._built.get(key)
        self._built[key] = payload
        while len(self._built) > BUILD_MEMORY:
            # Bounded: a run may build hundreds of plans, and the one being restored is recent.
            self._built.pop(next(iter(self._built)))
        if previous is None or previous == payload:
            return
        before = previous
        for key in sorted(set(before) | set(payload)):
            if before.get(key) == payload.get(key):
                continue
            if key != "scene_ir":
                self.build_diagnostics.append(f"rebuilding an unchanged plan changed {key}")
                continue
            inner_before, inner_after = before["scene_ir"], payload["scene_ir"]
            moved = sorted(
                name for name in set(inner_before) | set(inner_after)
                if inner_before.get(name) != inner_after.get(name)
            )
            self.build_diagnostics.append(
                "rebuilding an unchanged plan changed scene_ir: " + ", ".join(moved)
            )
            if "instances" in moved:
                # The version moving is the symptom; *which piece moved* is the diagnosis, and it
                # is the difference between reading the next run's manifest and guessing at it.
                self.build_diagnostics.append(
                    _instance_drift(inner_before, inner_after)
                )

    def build(self, plan: ScenePlan) -> str:
        self.plan = plan
        # A repair hands back a changed plan, and the support or the target is exactly what an L3
        # re-support changes -- so they are re-read from the plan on every build rather than fixed
        # at construction.
        self.support_id, self.target_id = plan.support_id, plan.target_id
        if not self.support_id or not self.target_id:
            raise SystemExit(
                "the plan names no support and no task target, so the build cannot tell what the "
                "task is about; a plan must come from plan_from_spec"
            )
        self.scene = build_scene(
            self.spec, output_dir=self.out, seed=self.generation_seed, plan=plan, state=self.build_state,
            asset_bindings=self.asset_bindings, asset_license=self.args.asset_license
        )
        self.scene["generated_assembly"] = self._assemble_generated(self.scene)
        self.scene["randomization"] = self.randomization
        self.scene["scene_ir"] = scene_ir_from_build(self.spec, self.scene)
        self.wall_thickness_m_value = self.scene["wall_thickness_m"]
        self.version = self._adopt_version(self.scene, plan)
        self._restore_observations(self.version)
        return self.version

    def _restore_observations(self, scene_version: str) -> None:
        """A rollback restores the observations and artifact pointers of that exact version."""
        self.worker_report, self.worker_dir, self.worker_request_path = self._backend_by_version.get(
            scene_version, ({}, None, None))
        self.critique, self.visual_gate = self._critique_by_version.get(scene_version, (None, None))
        # Rebuild recreates scene without its derived export URI. A rejected visual repair
        # restores the old observations, so restore the same plan's shell for final delivery too.
        scene = getattr(self, "scene", {})
        scene.pop("shell_uri", None)
        room_plan = scene.get("room_plan")
        if room_plan is not None:
            exported = getattr(self, "_shell_exports", {}).get(f"infinigen:{room_plan.content_hash}")
            if isinstance(exported, Path):
                scene["shell_uri"] = str(exported)

    def rebuild(self, plan: ScenePlan) -> str:
        """Rebuild from a changed plan. Identical to ``build``: the pipeline has no cache to warm,
        and pretending rebuild and build differ would invite one of them to skip a step."""
        return self.build(plan)

    # -- the host protocol -----------------------------------------------------------------------

    def current_plan(self) -> ScenePlan:
        return self.plan

    def current_scene_version(self) -> str:
        return self.version

    def evidence_ledger(self) -> EvidenceLedger:
        return self.ledger

    def wall_thickness_m(self) -> float:
        return self.wall_thickness_m_value

    def goal(self) -> str:
        return self.spec.prompt

    def synset_vocabulary(self) -> tuple[str, ...]:
        """The synsets an L5 patch may add an object from.

        The plan's own synsets plus the spec's entities, and nothing else. A model choosing freely
        would be choosing what the request contains, which is the one decision the design keeps out
        of its hands.
        """
        vocabulary = {request.synset for request in self.plan.furniture}
        vocabulary.update(request.synset for request in self.plan.surfaces)
        return tuple(sorted(vocabulary))

    def last_critique(self) -> Any:
        return self.critique

    def last_visual_gate(self) -> Any:
        return self.visual_gate

    def achieved_level(self) -> str:
        return self._resolution().achieved.value

    def gates_settled(self) -> bool:
        if self._resolution().meets_request:
            return True
        # A `--skip-isaac` run cannot measure the render or the critique at all, so waiting for them
        # would end every such run in a spurious exhaustion. The claim it makes instead is narrow and
        # checkable: every gate that *can* be measured has passed, and the only unsatisfied ones are
        # the backend and visual gates this run declined to run. It is never true of a real run.
        if not self.args.skip_isaac:
            return False
        declined = {
            "visual.required_views", "visual.critique",
            "backend.isaac.load", "backend.isaac.run", "backend.isaac.render",
            "physics.settle_stability", "physics.clock_advanced",
        }
        # At the current version: the ledger keeps every version a gate was measured on, and a
        # pass from one this run has moved past is not a reason to call this scene settled.
        measured = {
            gate_id: entry
            for (gate_id, version), entry in self.ledger.entries.items()
            if version == self.version
            and gate_id.startswith(("geometry.", "function.", "semantics."))
        }
        if not measured:
            return False
        unsatisfied = {
            gate_id for gate_id, entry in measured.items()
            if not entry.status.satisfies_gate
        }
        return not unsatisfied and declined <= set(self.ledger.gate_ids())

    def _resolution(self):
        return resolve_readiness(
            self.ledger, scene_version=self.version, requested=self.requested,
            requested_backends=self.requested_backends,
        )

    # -- the deterministic gates -----------------------------------------------------------------

    def _shell_matches_plan(self, scene_version):
        return measure_shell_matches_plan(self, scene_version)

    def _assemble_generated(self, scene):
        return assemble_generated(self, scene)


    def _shell_build_lazy(self, room_plan):
        """This run's shell build: made on first need, cached, and a failure kept as a value.

        Extracted from the plan-comparison gate because the build is now first needed *before* the
        scene graph exists -- the generated-furniture assembly reads the build's own records, and
        those only exist once the room has been built -- while the gate still needs the same object
        later. One cache and one failure value, so the two sites cannot build the room twice or
        disagree about whether it was built.

        A provider that could not build is a *result*, not a crash: `SHELL_BUILD_FAILED` is a named
        gap and the run carries it into the ledger, where a reader can see that the room was never
        checked, rather than dying with a traceback that looks like a failed scene run.
        """
        provider_name = getattr(self.args, "shell_provider", "builtin")
        key = self._shell_key(room_plan)
        if key not in self._shell_builds:
            provider = self._shell_provider(provider_name, room_plan)
            try:
                self._shell_builds[key] = provider.build(
                    room_plan, output_dir=self.out / "shell_build" / room_plan.content_hash[:12]
                )
            except ShellProviderError as error:
                self._shell_builds[key] = error
        return self._shell_builds[key]

    def _shell_key(self, room_plan) -> str:
        """One key for one build: the provider, the plan, and whether the opt-in was asked for.

        Two sites look a build up (the plan-comparison gate and the export), and they had drifted
        once already -- the export kept an older key format, so the opt-in's build was looked up
        under a name nothing had written. A shared key is the only way the two cannot drift again.
        """
        return (f"{getattr(self.args, 'shell_provider', 'builtin')}:{room_plan.content_hash}"
                f":background={getattr(self.args, 'background_generation', False)}")

    def _shell_provider(self, provider_name: str, room_plan):
        """The shell provider this run asks for: the default one, or SPEC-011's opt-in.

        The opt-in is a value on this run rather than an environment variable (requirement 7):
        `--background-generation` with `--background-gaps` builds the shell through our own entry
        with the task's reservations injected, and the delivery records what was asked (the mode,
        the gaps, the policy, and where the build's own records are) under `background`.
        """
        if not getattr(self.args, "background_generation", False):
            return provider_named(provider_name)
        if provider_name != "infinigen":
            raise SystemExit(
                "BACKGROUND_GENERATION_NEEDS_INFINIGEN(--shell-provider "
                f"{provider_name}): only the Infinigen provider realises background content"
            )
        gaps = tuple(item.strip() for item in
                     str(getattr(self.args, "background_gaps", "")).split(",") if item.strip())
        policy = BACKGROUND_GENERATION_POLICY
        provider = generation_provider(room_plan, zones=self.scene["zones"], gaps=gaps,
                                       policy=policy)
        step("background",
             f"opt-in: {len(provider.background_constraints.reservations)} reservation(s), "
             f"gaps={list(gaps)}, abort off (see the record's why)")
        return provider

    def run_deterministic_gates(self, scene_version):
        return measure_deterministic_gates(self, scene_version)

    # -- the backend -----------------------------------------------------------------------------

    def prepare_shell_reference(self) -> None:
        """Export the measured build separately from the unchanged plan comparison gate."""
        if getattr(self.args, "shell_provider", "builtin") != "infinigen":
            self.scene.pop("shell_uri", None)
            return
        from rfw_w.scenes.solvers.shell_providers import export_shell
        self.scene.pop("shell_uri", None)
        key = self._shell_key(self.scene['room_plan'])
        build = self._shell_builds.get(key)
        if build is None or isinstance(build, Exception):
            raise ShellProviderError("EXPORT_BACKEND_UNAVAILABLE(no measured shell)")
        if not hasattr(self, "_shell_exports"):
            self._shell_exports = {}
        if key not in self._shell_exports:
            try:
                self._shell_exports[key] = export_shell(build)
            except (ShellProviderError, ValueError) as error:
                self._shell_exports[key] = error
        exported = self._shell_exports[key]
        if isinstance(exported, Exception):
            raise exported
        self.scene["shell_uri"] = str(exported)

    def run_backend(self, scene_version: str) -> bool:
        try:
            self.prepare_shell_reference()
        except (ShellProviderError, ValueError) as error:
            self.ledger.record(evidence(
                "backend.isaac.load", EvidenceStatus.FAIL, scene_version,
                method="shell_reference_preflight", detail=str(error),
            ))
            return False
        if self.args.skip_isaac:
            step("isaac", "skipped by request")
            for gate in (
                "physics.settle_stability", "physics.clock_advanced", "visual.required_views",
                "backend.isaac.load", "backend.isaac.run", "backend.isaac.render",
            ):
                self.ledger.record(evidence(
                    gate, EvidenceStatus.NOT_RUN, scene_version,
                    method="skipped_by_request", detail="--skip-isaac was passed",
                ))
            return True

        if getattr(self.args, "shell_provider", "builtin") == "infinigen" and not self.scene.get("shell_uri"):
            self.ledger.record(evidence(
                "backend.isaac.load", EvidenceStatus.FAIL, scene_version,
                method="shell_reference_preflight",
                detail="EXPORT_BACKEND_UNAVAILABLE(no successfully exported shell)",
            ))
            return False
        lighting, _lighting_source, _lighting_room = lighting_plan_for(
            self.scene, getattr(self.args, "lighting_profile", "mixed"))
        self.lighting_block = lighting_block_for(
            self.scene, getattr(self.args, "lighting_profile", "mixed"))
        request = build_worker_request(self.scene, self.spec, scene_version=scene_version,
                                       lighting=lighting)
        attempt = self.backend_attempts.get(scene_version, 0) + 1
        self.backend_attempts[scene_version] = attempt
        request_dir = self.out / "isaac_requests" / scene_version
        request_dir.mkdir(parents=True, exist_ok=True)
        request_path = request_dir / f"attempt_{attempt:03d}.json"
        payload = request.to_dict()
        if request.room.shell_uri:
            payload["room"]["shell_uri"] = os.path.relpath(request.room.shell_uri, request_dir)
        request_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        worker_dir = self.out / "isaac" / scene_version / f"attempt_{attempt:03d}"
        self.worker_request_path = request_path
        self.worker_dir = worker_dir
        step("isaac", f"running the worker into {worker_dir}")
        completed = subprocess.run(
            [
                sys.executable, "-m", "rfw_w.scenes.workers.launch",
                "--request", str(request_path), "--output-dir", str(worker_dir),
                "--active-gpu", str(self.args.active_gpu), "--timeout", "1200",
            ],
            capture_output=True, text=True,
        )
        print(completed.stdout[-2500:])
        if completed.stderr:
            print(completed.stderr[-1500:], file=sys.stderr)
        report_path = worker_dir / "report.json"
        if report_path.is_file():
            self.worker_report = json.loads(report_path.read_text())
        else:
            self.worker_report = {"status": "failed", "error_detail": "no worker report was written"}
        self._backend_by_version[scene_version] = (self.worker_report, self.worker_dir, self.worker_request_path)
        for item in gates_from_worker(
            self.worker_report, scene_version=scene_version,
            relation_evidence=self.relation_evidence, target_instance=self.target_id,
        ):
            # The relation evidence is already recorded at this version; re-recording it after
            # settle would be a different statement (post-settle rather than initial) and would
            # silently overwrite it.
            if item.gate_id != "function.initial_relations":
                self.ledger.record(item)
        step("simulate_settle / render_views", f"worker status={self.worker_report.get('status')}")
        return self.worker_report.get("status") == "passed"

    # -- the critic ------------------------------------------------------------------------------

    def run_critique(self, scene_version: str) -> bool:
        from rfw_w.scenes.agents.llm_client import (
            CRITIC_SYSTEM_PROMPT, LLMError, build_critic_prompt, call_vision, require_external_permission,
        )
        from rfw_w.scenes.agents.montage import MontageError, montage_data_url
        from rfw_w.scenes.agents.visual_critic import (
            critique_evidence, evaluate_visual_gate, parse_critique, sanitised_payload,
        )

        self.critique = None
        self.visual_gate = None
        vlm_model = self.args.vlm_model or __import__("os").environ.get("PVSCENE_VLM_MODEL")
        worker_dir = self.worker_dir
        images = sorted(worker_dir.glob("*.png")) if worker_dir is not None and worker_dir.is_dir() else []
        if not images or not vlm_model:
            reason = (
                "no rendered image is available" if not images else "PVSCENE_VLM_MODEL is not set"
            )
            self.ledger.record(evidence(
                "visual.critique", EvidenceStatus.NOT_RUN, scene_version,
                method="vlm_critique_strict_schema",
                detail=f"the visual critique did not run: {reason}", backend="visual",
            ))
            return False

        # One montage rather than five full frames: the doc asks for a montage, and five 1280x720
        # PNGs is a large enough payload that a real run timed out mid-request.
        try:
            montage_url, labelled = montage_data_url(
                [(image.stem, image) for image in images], columns=3
            )
        except MontageError as error:
            montage_url, labelled = "", []
            step("review_visual", f"montage failed: {error}")

        # The manual's rule is that no image means an invalid critique, not a cheaper one. Before
        # this, a failed montage fell through to `call_vision(images=[])`: the model was asked for
        # all seven scores from the text description alone, and its reply was accepted as VALID and
        # could pass. `InvalidReason.NO_IMAGE` existed and was never emitted.
        #
        # The renders are also bound to the scene here, which the old check could not do: it passed
        # the same value as both the reviewed and the current hash, so the comparison could only ask
        # whether the model echoed back the digest it had been handed.
        render_report = worker_dir / "report.json" if worker_dir is not None else None
        rendered_version = ""
        if render_report is not None and render_report.exists():
            rendered_version = json.loads(render_report.read_text()).get("scene_version", "")

        if not labelled:
            self.ledger.record(evidence(
                "visual.critique", EvidenceStatus.NOT_RUN, scene_version,
                method="vlm_critique_strict_schema", backend="visual",
                detail="no render could be montaged, so there is no image to critique",
            ))
            step("review_visual", "not_run: no image to critique")
            return False
        if rendered_version != scene_version:
            self.ledger.record(evidence(
                "visual.critique", EvidenceStatus.FAIL, scene_version,
                method="vlm_critique_strict_schema", backend="visual",
                detail=(
                    f"the renders on disk were produced for scene "
                    f"{rendered_version or 'an unrecorded version'}, not for {scene_version}; a "
                    f"critique of them cannot approve this scene"
                ),
            ))
            step("review_visual", "failed: the renders are not this scene's")
            return False

        # The instance ids the critic may name, so a suggestion it proposes can be checked against
        # the scene rather than taken on trust. A critic given no list is a critic that can invent
        # an object, and `parse_critique` refuses every suggestion in that case.
        known_targets = tuple(entry.instance_id for entry in self.scene["instances"])
        payload = sanitised_payload(
            scene_version=scene_version,
            views=[{"label": label} for label in labelled],
            targets=[
                {"semantic_id": instance_id, "category": self.spec.entity(instance_id).category}
                for instance_id in known_targets
            ],
            task_prompt=self.spec.prompt,
            task=self.spec.task.to_dict(),
        )
        # Everything above has already been measured and paid for, so a missing endpoint or
        # credential is recorded as a declared gap rather than allowed to abort the run.
        try:
            require_external_permission()
            config = vlm_config(vlm_model)
            if self.model_budget is not None:
                self.model_budget.assert_time_remains(stage="visual.critique")
                self.model_budget.charge_model_request(purpose="visual.critique")
                config = replace(config, timeout_s=min(config.timeout_s, self.model_budget.remaining_s))
            response = call_vision(
                config, system_prompt=CRITIC_SYSTEM_PROMPT,
                user_prompt=build_critic_prompt(payload), images=[montage_url],
            )
            # Preserve each actual response even if parsing or a later repair fails.
            response_dir = self.out / "critic_responses" / scene_version
            response_dir.mkdir(parents=True, exist_ok=True)
            response_path = response_dir / f"attempt_{len(list(response_dir.glob('attempt_*.txt'))) + 1:03d}.txt"
            response_path.write_text(response.text, encoding="utf-8")
            critique = parse_critique(
                response.text, reviewed_scene_hash=scene_version,
                current_scene_hash=scene_version, model_id=response.model_id,
                tokens_used=response.tokens_used, known_targets=known_targets,
            )
            gate = evaluate_visual_gate(critique)
            self.ledger.record(critique_evidence(critique, gate, scene_version=scene_version))
            self.critique = critique
            self.visual_gate = gate
            self._critique_by_version[scene_version] = (critique, gate)
            step("review_visual",
                 f"{critique.status.value}, gate passed={gate.passed}, "
                 f"{len(critique.suggestions)} repair(s) proposed, "
                 f"{len(critique.rejected_suggestions)} refused")
            return gate.passed
        except (LLMError, BudgetExhausted) as error:
            previous = self.ledger.get("visual.critique", scene_version)
            if previous is not None and previous.status in (EvidenceStatus.PASS, EvidenceStatus.FAIL):
                self.critique, self.visual_gate = self._critique_by_version.get(scene_version, (None, None))
                step("review_visual", "call unavailable; retaining the measured verdict for this scene")
                return previous.status is EvidenceStatus.PASS
            self.ledger.record(evidence(
                "visual.critique", EvidenceStatus.NOT_RUN, scene_version,
                method="vlm_critique_strict_schema", backend="visual",
                detail=f"the model call failed: {error}",
            ))
            step("review_visual", "the model call failed")
            return False



def generate(args: GenerationOptions) -> int:
    """Generate one scene, preserving immutable evidence for every attempted version."""
    if args.env_file:
        load_env_file(args.env_file)

    started = time.monotonic()
    if args.spec is not None and args.experiment is not None:
        raise ValueError("--spec and --experiment are mutually exclusive")
    explicit_prompt = args.prompt is not None or args.prompt_file is not None
    if args.no_compile and explicit_prompt:
        raise ValueError("--no-compile needs a JSON spec, not a prompt")
    do_compile = args.compile or explicit_prompt
    if not 1 <= args.max_model_requests <= 5:
        raise ValueError("--max-model-requests must be within 1..5")
    experiment_id = args.experiment or (None if args.spec or explicit_prompt else DEFAULT_EXPERIMENT)
    reference = (SceneSpec.from_dict(json.loads(args.spec.read_text())) if args.spec is not None
                 else load_spec(experiment_id) if experiment_id else None)
    if args.room_bounds:
        # A room *size* is part of the task's input, not a rendering choice: SPEC-008's matrix asks
        # whether one profile lights a small room and a large one, so the size has to be commandable
        # without forking three spec files per room type. The override lands in the effective spec,
        # which is what the delivery publishes, so the delivery says which room it was asked for.
        if args.spec is None:
            raise ValueError("--room-bounds overrides the JSON spec's room; give --spec as well")
        try:
            bounds = json.loads(args.room_bounds)
        except ValueError as error:
            raise ValueError(f"--room-bounds must be JSON: {error}")
        raw = json.loads(args.spec.read_text())
        raw.setdefault("room", {})["size_bounds_m"] = bounds
        reference = SceneSpec.from_dict(raw)
        step("run", f"room size_bounds_m overridden by the command line: {bounds}")
    if args.output_dir is not None:
        out = args.output_dir.resolve()
    elif args.run and experiment_id:
        out = (OUTPUTS_DIR / experiment_id / args.run).resolve()
    else:
        raise ValueError("give --run with --experiment, or --output-dir for prompt/spec input")
    if out.exists():
        raise ValueError("output directory already exists; every run needs a fresh directory")
    compiler_model = args.compiler_model or os.environ.get("PVSCENE_COMPILER_MODEL") or args.vlm_model or os.environ.get("PVSCENE_VLM_MODEL")
    critic_model = args.vlm_model or os.environ.get("PVSCENE_VLM_MODEL")
    agent_model = args.agent_model or os.environ.get("PVSCENE_AGENT_MODEL")
    if do_compile or (not args.compile_only and not args.request_only and (
        (critic_model and not args.skip_isaac) or (agent_model and not args.no_replan and not args.no_loop)
    )):
        from rfw_w.scenes.agents.llm_client import require_external_permission
        require_external_permission()
    if do_compile and not compiler_model:
        raise ValueError("prompt compilation requires --compiler-model or PVSCENE_COMPILER_MODEL")
    budget = Budget(deadline_s=args.deadline_s, max_model_requests=args.max_model_requests,
                    max_subgraph_rollbacks=max(args.max_local_repairs, 2), started_at=started)
    out.mkdir(parents=True, exist_ok=False)
    compilation = {"status": "not_run", "reason": "JSON_SPEC_INPUT", "model_requests": 0}
    spec = reference
    if do_compile:
        from rfw_w.scenes.agents.scene_compiler import compile_scene
        if args.prompt_file is not None:
            prompt = args.prompt_file.read_text(encoding="utf-8").strip()
            if not prompt:
                raise ValueError("prompt file must contain task text")
        elif args.prompt is not None:
            prompt = args.prompt
        else:
            prompt = reference.prompt
        seed = args.seed if args.seed is not None else reference.seed if reference else 11
        scene_id = args.scene_id or (reference.scene_id if reference else "prompt_scene")
        (out / "compile_request.json").write_text(json.dumps({
            "prompt": prompt, "seed": seed, "scene_id": scene_id,
        }, indent=2, ensure_ascii=False) + "\n")
        result = compile_scene(
            replace(
                vlm_config(compiler_model),
                max_tokens=12000,
                timeout_s=min(COMPILE_TIMEOUT_S, budget.remaining_s),
            ),
            prompt=prompt, seed=seed, scene_id=scene_id, reference=reference,
            on_request=lambda: budget.charge_model_request(purpose="compile"),
        )
        compilation = result.record
        spec = result.spec
        if spec is None and reference is not None:
            spec = reference
            compilation.update(status="fallback", fallback="supplied_spec", fallback_spec_hash=content_hash(spec.to_dict()))
            step("compile_failed", f"{compilation['reason']}; using the supplied reference spec")
        if spec is None:
            (out / "compilation.json").write_text(json.dumps(compilation, indent=2, ensure_ascii=False) + "\n")
            raise SystemExit(f"compile_failed({compilation['reason']}): no reference spec is available")
        if result.spec is not None:
            (out / "compiled_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n")
    elif args.seed is not None or args.scene_id is not None:
        raise ValueError("--seed and --scene-id are for prompt compilation; use --variant-seed for JSON runs")
    assert spec is not None
    compilation["effective_spec_hash"] = content_hash(spec.to_dict())
    (out / "effective_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n")
    if do_compile or args.compile_only:
        from rfw_w.scenes.agents.scene_preview import preview_scene
        try:
            svg, preview = preview_scene(spec)
            (out / "core_room.svg").write_text(svg)
        except (ValueError, RuntimeError) as error:
            preview = {"status": "failed", "reason": type(error).__name__, "detail": str(error)}
        compilation["preview"] = preview
    (out / "compilation.json").write_text(json.dumps(compilation, indent=2, ensure_ascii=False) + "\n")
    if args.compile_only:
        step("compile_only", f"status={compilation['status']}; outputs at {out}")
        return 0 if compilation["status"] != "fallback" and compilation.get("preview", {}).get("status") == "ran" else 1
    requested = ReadinessLevel(spec.requested_readiness)
    step("run", f"experiment={experiment_id or '-'} scene_id={spec.scene_id} -> {out}")

    host = SceneHost(
        spec=spec, out=out, args=args, requested=requested,
        requested_backends=spec.requested_backends,
    )
    host.model_budget = budget
    variant_seed = spec.seed if args.variant_seed is None else args.variant_seed
    randomization_enabled = args.domain_randomization or args.variant_seed is not None
    initial_plan, randomization = randomize_plan(
        spec,
        build_plan(spec),
        variant_seed=variant_seed,
        enabled=randomization_enabled,
    )
    attempts: list[int] = []
    rejected_attempts: list[str] = []
    candidate_seeds = (
        range(variant_seed, variant_seed + args.max_variant_attempts)
        if randomization_enabled else (spec.seed,)
    )
    for room_seed in candidate_seeds:
        attempts.append(room_seed)
        host.randomization = replace(
            randomization,
            room_seed=room_seed if randomization_enabled else None,
            room_seed_attempts=tuple(attempts),
            rejected_attempts=tuple(rejected_attempts),
        )
        host.generation_seed = room_seed
        try:
            host.build(initial_plan)
        except CoreUnplaced as error:
            (out / f"generation_attempt_{len(attempts):03d}.json").write_text(json.dumps({
                "room_seed": room_seed, "status": "failed", "unplaced_core": [error.item.to_dict()],
            }, indent=2) + "\n")
            if not randomization_enabled:
                raise
            rejected_attempts.append(f"room_seed={room_seed}: {str(error)[:300]}")
            continue
        except (RoomError, ZoneError) as error:
            # A seed can draw a room the rest of the pipeline refuses: the shell gains shape
            # families, so some seeds are T- or alcove-shaped rooms whose stems the zone scheme
            # cannot lay strips and a corridor into. That is a reason to sample another seed, not
            # to end the run -- and it is recorded, so a run that skipped three seeds says so
            # instead of looking like it drew a good one first. Measured over 300 seeds of each
            # experiment, 13 in 900 are refused this way.
            if not randomization_enabled:
                raise
            rejected_attempts.append(
                f"room_seed={room_seed}: {type(error).__name__}: {str(error)[:240]}"
            )
            continue
        (out / f"generation_attempt_{len(attempts):03d}.json").write_text(json.dumps({
            "room_seed": room_seed, "status": "built",
            **{key: host.scene[key] for key in (
                "unplaced_core", "unplaced_task_closure", "unplaced_periphery", "periphery_dropped",
            )},
        }, indent=2) + "\n")
        missing_task = [item["entity_id"] for key in ("unplaced_core", "unplaced_task_closure")
                        for item in host.scene[key]]
        if host.scene["unplaced_core"] or host.scene["unplaced_task_closure"]:
            rejected_attempts.append(
                f"room_seed={room_seed}: missing task closure {','.join(missing_task)}"
            )
            continue
        break
    else:
        raise SystemExit(
            f"no task-complete domain variant in {len(attempts)} deterministic "
            f"attempts from seed {variant_seed}: " + "; ".join(rejected_attempts)
        )

    # Attach the final audit trail and recompute the graph/version. The geometry is unchanged;
    # only the record now includes the rejected seeds that led to this accepted sample.
    host.randomization = replace(
        randomization,
        room_seed=host.generation_seed if randomization_enabled else None,
        room_seed_attempts=tuple(attempts),
        rejected_attempts=tuple(rejected_attempts),
    )
    host.scene["randomization"] = host.randomization
    host.scene["scene_ir"] = scene_ir_from_build(spec, host.scene)
    host.version = host._adopt_version(host.scene, host.plan)
    step(
        "randomization",
        f"enabled={host.randomization.enabled} variant_seed={host.randomization.variant_seed} "
        f"room_seed={host.randomization.room_seed} attempts={len(attempts)} "
        f"frozen_entities={len(host.randomization.invariant_entity_ids)} "
        f"sampled_entities={len(host.randomization.entity_choices)}",
    )
    step("plan", f"{len(host.plan.furniture)} furniture and {len(host.plan.surfaces)} surface "
                 f"objects, wall face offset {host.plan.wall_face_offset_m:.4f} m")

    if args.request_only:
        host._shell_matches_plan(host.version)
        host.prepare_shell_reference()
        if args.shell_provider == "infinigen" and not host.scene.get("shell_uri"):
            raise SystemExit("EXPORT_BACKEND_UNAVAILABLE(no referenced shell for request)")
        request = build_worker_request(host.scene, spec, scene_version=host.version)
        (out / "isaac_request.json").write_text(
            json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        (out / "scene_ir.json").write_text(
            json.dumps(host.scene["scene_ir"].to_dict(), indent=2, sort_keys=True) + "\n"
        )
        step("request_only", f"{len(request.instances)} instances, {len(request.cameras)} cameras")
        print(json.dumps({"request": str(out / "isaac_request.json"),
                          "placed": len(host.scene["instances"])}, indent=2))
        return 0

    loop_run = None
    if args.no_loop:
        # The straight-through form, with no ladder and no repair. Every gate is measured once, on
        # whatever the plan produced first.
        host.run_deterministic_gates(host.version)
        host.run_backend(host.version)
        host.run_critique(host.version)
        step("loop", "disabled by --no-loop; nothing was repaired")
    else:
        replan_config = None
        agent_model = args.agent_model or __import__("os").environ.get("PVSCENE_AGENT_MODEL")
        if agent_model and not args.no_replan:
            replan_config = vlm_config(agent_model)
        driver = Driver(
            host=host, budget=budget, replan_config=replan_config,
            max_repair_jobs=args.max_local_repairs,
        )
        loop_run = driver.run()
        step("loop", f"terminal={loop_run.terminal} segments={loop_run.segments} "
                     f"model_requests={loop_run.budget['model_requests_used']} "
                     f"local_recovery_rate={loop_run.ladder['local_recovery_rate']}")
        for event in loop_run.events:
            step("  event", f"{event.kind.value} at {event.source_gate} "
                            f"({event.failure_code.value if event.failure_code else '-'}) "
                            f"targets={list(event.targets)}")
        for attempt in loop_run.ladder["attempts"]:
            step("  repair", f"{attempt['level']} {attempt['atom']} "
                             f"resolved={attempt['resolved']} improved={attempt['improved']}")
        for note in loop_run.notes:
            step("  note", note)

    for item in not_run_gates(host.version):
        host.ledger.record(item)
    for item in unsupported_gates(host.version):
        host.ledger.record(item)

    resolution = host._resolution()
    step("finalize", f"achieved={resolution.achieved.value} outcome={resolution.outcome}")
    return write_delivery(host, args=args, compilation=compilation, loop_run=loop_run, budget=budget, started=started)
