#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a delivery from its artefacts alone, in a separate process.

    python -m rfw_w.scenes.validation.verify_delivery --delivery outputs/office_pick_cup/m1_009

This is the structural guarantee against overclaiming. It reads only what was delivered, re-runs
the readiness resolution on the recorded evidence, and exits non-zero if the manifest's claimed
level is not the one its own gates support.

It is a separate process on purpose. A verifier that shared state with the generator could inherit
an assumption that made the claim look true, and the whole point is that the claim survives leaving
the process that made it.

What it checks:

* every gate's evidence is bound to the manifest's scene version -- a pass from another version is
  stale and cannot count;
* no gate that never ran is treated as satisfied;
* the recomputed readiness equals the claimed readiness;
* a claimed `meets_request` is backed by zero blockers at the requested level;
* the artefacts a pass depends on actually exist on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from rfw_w.scenes.contracts.hashing import content_hash
from rfw_w.scenes.contracts.readiness import ReadinessLevel, resolve_readiness
from rfw_w.scenes.contracts.tool import (
    Evidence,
    EvidenceLedger,
    EvidenceStatus,
    ToolError,
)


class VerificationFailure(AssertionError):
    """The delivery does not support its own claim."""


def load_manifest(delivery: Path) -> dict:
    path = delivery / "delivery_manifest.json"
    if not path.is_file():
        raise VerificationFailure(f"no manifest at {path}; a delivery without one cannot be checked")
    return json.loads(path.read_text())


def rebuild_ledger(manifest: dict) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for payload in manifest.get("gates", []):
        try:
            ledger.record(Evidence.from_dict(payload))
        except ToolError as error:
            raise VerificationFailure(
                f"the recorded evidence for {payload.get('gate_id')!r} is not valid: {error}"
            ) from error
    return ledger


def verify_compilation(delivery: Path, manifest: dict) -> None:
    """Recheck compiler artifacts without loading a model or trusting its claimed hashes."""
    record = manifest.get("compilation") or {}
    if not record:
        return  # legacy deliveries did not have a compiler stage
    status = record.get("status")
    if status not in ("not_run", "compiled", "fallback"):
        raise VerificationFailure("unknown compilation status")
    if status == "not_run" and not record.get("effective_spec_hash"):
        return

    def read(name):
        try:
            return json.loads((delivery / name).read_text())
        except (OSError, ValueError) as error:
            raise VerificationFailure(f"compilation artifact {name} missing or invalid") from error

    effective = read("effective_spec.json")
    expected = record.get("effective_spec_hash") or record.get("compiled_spec_hash") or record.get("fallback_spec_hash")
    if content_hash(effective) != expected:
        raise VerificationFailure("effective spec content hash differs from the compilation record")
    for key in ("scene_id", "seed", "prompt"):
        if effective.get(key) != manifest.get(key):
            raise VerificationFailure(f"effective spec {key} differs from the delivery")
    if status == "not_run":
        return
    request = read("compile_request.json")
    if content_hash({"prompt": request.get("prompt"), "seed": request.get("seed")}) != record.get("compile_prompt_hash"):
        raise VerificationFailure("compile request content hash differs from the compilation record")
    if status == "compiled":
        compiled = read("compiled_spec.json")
        if content_hash(compiled) != record.get("compiled_spec_hash") or compiled != effective:
            raise VerificationFailure("compiled spec content hash or effective input differs")
        if any(compiled.get(key) != request.get(key) for key in ("scene_id", "seed", "prompt")):
            raise VerificationFailure("compiled spec changed the input identity")
    elif not record.get("reason") or content_hash(effective) != record.get("fallback_spec_hash"):
        raise VerificationFailure("compilation fallback has no reason or matching spec hash")


def verify(delivery: Path) -> dict:
    manifest = load_manifest(delivery)
    verify_compilation(delivery, manifest)
    verify_generated_assembly(delivery, manifest)
    scene_version = manifest.get("scene_version", "")
    if len(scene_version) != 64:
        raise VerificationFailure(f"the manifest names no scene version: {scene_version!r}")

    ledger = rebuild_ledger(manifest)

    # The resolution is embedded whole as of manifest v2. It used to be restated by hand under
    # names of the *manifest's* own (`requested_readiness`/`achieved_readiness`) while the type it
    # copied used `requested`/`achieved` -- a copy that could disagree with its source, in the file
    # whose whole purpose is to be checkable.
    resolution_block = manifest.get("resolution")
    if not isinstance(resolution_block, dict):
        raise VerificationFailure(
            "the manifest carries no `resolution` object; a delivery written before manifest v2 "
            "restated it under `requested_readiness`/`achieved_readiness`, and reading those here "
            "would re-introduce the hand-copy this shape exists to remove"
        )
    requested = ReadinessLevel(resolution_block["requested"])
    backends = manifest.get("requested_backends")
    if not backends:
        raise VerificationFailure(
            "the manifest does not record which backends were requested; inferring them from the "
            "gates present would mistake a declared out-of-scope gap for a requested backend"
        )
    resolution = resolve_readiness(
        ledger,
        scene_version=scene_version,
        requested=requested,
        requested_backends=tuple(backends),
    )

    # The specific checks run before the readiness comparison, because a stale pass and an
    # undeclared failure are more precise diagnoses than "the level came out lower than claimed".
    # Reporting the vague one first would send a reader looking for a missing gate when the real
    # problem is a gate that is present and wrong.

    # 1. A pass counts only for the scene it was measured on.
    #
    #    The ledger keeps every scene a gate was measured on, so a pass from another one is not an
    #    anomaly to throw on -- it is the record of a scene the run moved past, and the manifest is
    #    now a record of the run rather than a snapshot of its last state. What must still hold is
    #    the thing this check was always about: a delivery may not lean on one. A stale pass may be
    #    kept, but only beside an entry for the delivered scene; standing alone it would be the only
    #    thing recorded for that gate, and a reader would take it for this delivery's.
    for (gate_id, version), record in ledger.entries.items():
        if version != scene_version:
            if record.status is EvidenceStatus.PASS and (gate_id, scene_version) not in ledger.entries:
                raise VerificationFailure(
                    f"{gate_id} is recorded as a pass for scene {version[:12]} and has no entry for "
                    f"the delivered scene {scene_version[:12]}; a delivery has to say what it "
                    f"measured on the scene it ships"
                )
            continue
        if record.status is not EvidenceStatus.PASS and gate_id not in (
            manifest.get("not_run", []) + manifest.get("unsupported", [])
        ):
            # An unsatisfied gate that is neither declared not_run nor unsupported, and that is not
            # blocking the requested level, is an undeclared failure -- the thing a delivery must
            # not hide.
            if not any(blocker.gate_id == gate_id for blocker in resolution.blockers):
                raise VerificationFailure(
                    f"{gate_id} is {record.status.value} but is not declared as a blocker, "
                    "not_run or unsupported"
                )

    # 2. The canonical graph must exist and match the content hash recorded in the manifest.
    graph_record = manifest.get("scene_graph")
    if graph_record is not None:
        relative = Path(str(graph_record.get("path", "")))
        graph_path = (delivery / relative).resolve()
        if relative.is_absolute() or delivery.resolve() not in graph_path.parents:
            raise VerificationFailure("scene_graph.path must stay inside the delivery")
        if not graph_path.is_file():
            raise VerificationFailure(f"the manifest names {relative}, which is not on disk")
        graph = json.loads(graph_path.read_text())
        graph.pop("revision", None)
        graph.pop("hashes", None)
        observed = content_hash(graph)
        if observed != graph_record.get("content_hash"):
            raise VerificationFailure(
                "scene_ir.json does not match the content hash recorded in the manifest"
            )

    # 3. The artefacts a pass depends on must exist.
    isaac_relative = Path(str(manifest.get("isaac_artifact_dir") or "isaac"))
    isaac_dir = (delivery / isaac_relative).resolve()
    if isaac_relative.is_absolute() or delivery.resolve() not in isaac_dir.parents:
        raise VerificationFailure("isaac_artifact_dir must stay inside the delivery")
    report_path = isaac_dir / "report.json"
    shell_record = None
    shell_path = delivery / "shell_reference.json"
    if shell_path.is_file():
        from rfw_w.scenes.workers.isaac.shell_reference import inspect_shell, ShellReferenceError, reference_matches
        shell_record = json.loads(shell_path.read_text())
        try:
            actual = inspect_shell(str(delivery / shell_record["uri"]), delivery=delivery)
        except (ShellReferenceError, KeyError) as error:
            raise VerificationFailure(str(error)) from error
        if shell_record.get("scene_version") != scene_version:
            raise VerificationFailure("referenced shell belongs to another scene version")
        if not reference_matches(shell_record, actual):
            raise VerificationFailure("SHELL_URI_UNRESOLVABLE(shell evidence mismatch)")
    if ledger.satisfies("backend.isaac.render", scene_version):
        if not report_path.is_file():
            raise VerificationFailure("backend.isaac.render passed but no worker report exists")
        report = json.loads(report_path.read_text())
        if report.get("scene_version") != scene_version:
            raise VerificationFailure("worker report belongs to a different scene version")
        reference = report.get("shell_reference", {})
        if shell_record is not None or reference.get("mode") == "referenced":
            if shell_record is None or reference.get("mode") != "referenced":
                raise VerificationFailure("SHELL_URI_UNRESOLVABLE(reference evidence absent)")
            if not reference_matches(reference, actual):
                raise VerificationFailure("SHELL_URI_UNRESOLVABLE(render used another shell)")
            request_path = (delivery / manifest.get("isaac_request_path", "")).resolve()
            if not request_path.is_relative_to(delivery.resolve()) or not request_path.is_file():
                raise VerificationFailure("SHELL_URI_UNRESOLVABLE(worker request absent)")
            uri = json.loads(request_path.read_text()).get("room", {}).get("shell_uri")
            if not uri or (request_path.parent / uri).resolve() != (delivery / shell_record["uri"]).resolve():
                raise VerificationFailure("SHELL_URI_UNRESOLVABLE(request references another shell)")
        for capture in report.get("captures", []):
            for key in ("rgb_path", "depth_path", "instance_id_path"):
                name = capture.get(key)
                if name and not (isaac_dir / name).is_file():
                    raise VerificationFailure(f"a capture names {name}, which is not on disk")
        if report.get("instance_id_map_source") != "annotator_idToLabels":
            raise VerificationFailure(
                "the worker's instance-id map does not come from the renderer's own labels, so a "
                "rendered id cannot be resolved to an entity"
            )

    if ledger.satisfies("visual.critique", scene_version):
        critique = manifest.get("visual_critique") or {}
        if not critique.get("critique", {}).get("reviewed_scene_hash") == scene_version:
            raise VerificationFailure(
                "the visual gate passed but the critique is not recorded against this scene version"
            )

    # 4. The recomputed level must be the claimed one.
    claimed = resolution_block["achieved"]
    if resolution.achieved.value != claimed:
        raise VerificationFailure(
            f"the manifest claims {claimed!r} but its own gates support "
            f"{resolution.achieved.value!r}; blockers: "
            + "; ".join(f"{b.gate_id}={b.status.value}" for b in resolution.blockers)
        )

    # 5. A claim of meeting the request must be backed by zero blockers.
    if resolution_block.get("meets_request") and resolution.blockers:
        raise VerificationFailure(
            "the manifest claims the request was met while "
            f"{len(resolution.blockers)} required gate(s) are unsatisfied"
        )

    return {
        "verified": True,
        "delivery": str(delivery),
        "scene_version": scene_version,
        "claimed_readiness": claimed,
        "recomputed_readiness": resolution.achieved.value,
        "meets_request": resolution.meets_request,
        "gates_total": len(ledger.entries),
        "gates_passing": sum(
            1 for record in ledger.entries.values() if record.status is EvidenceStatus.PASS
        ),
        "not_run": sorted(manifest.get("not_run", [])),
        "unsupported": sorted(manifest.get("unsupported", [])),
    }


def verify_generated_assembly(delivery: Path, manifest: dict) -> None:
    """Re-read the selected immutable assembly instead of trusting its copied report."""
    assembly = (manifest.get("background") or {}).get("assembly")
    if not assembly:
        return
    if not isinstance(assembly, dict):
        raise VerificationFailure("generated assembly must be a record")
    root = delivery.resolve()

    def resource(name: str) -> Path:
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise VerificationFailure("generated assembly requires a relative artifact path")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise VerificationFailure("generated assembly artifact is missing or outside delivery")
        return path

    try:
        artifact = resource(assembly.get("artifact_path"))
        recorded = json.loads(artifact.read_text())
        if recorded != assembly:
            raise VerificationFailure("generated assembly differs from its immutable artifact")
        generation_hash = assembly.get("generation_hash", "")
        if len(generation_hash) != 64 or any(c not in "0123456789abcdef" for c in generation_hash):
            raise VerificationFailure("generated assembly has no valid generation hash")
        attempt = assembly.get("attempt")
        if type(attempt) is not int or attempt < 1:
            raise VerificationFailure("generated assembly has no valid attempt")
        expected = f"generated_furniture/{generation_hash}/attempt_{attempt:03d}/assembly.json"
        if assembly["artifact_path"] != expected:
            raise VerificationFailure("generated assembly path does not identify its version and attempt")
        if assembly.get("assembly"):
            exported = json.loads(resource(assembly["assembly"]).read_text())
            digest = exported.get("digest")
            if digest != assembly.get("assembly_digest"):
                raise VerificationFailure("generated export digest differs from the selected assembly")
            actual = content_hash({key: value for key, value in exported.items() if key != "digest"})
            if actual != digest:
                raise VerificationFailure("generated export content hash differs")
    except (OSError, ValueError, TypeError) as error:
        raise VerificationFailure("generated assembly artifact cannot be verified") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = verify(args.delivery.resolve())
    except VerificationFailure as error:
        print(json.dumps({"verified": False, "reason": str(error)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
