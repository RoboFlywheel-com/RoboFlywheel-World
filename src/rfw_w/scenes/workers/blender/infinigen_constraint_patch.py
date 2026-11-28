#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Put the task's reservations into Infinigen's constraint graph, so its furniture solver knows them.

SPEC-011's mechanism. Everything here is installed *in process*, from this file, into the pinned
upstream tree -- ADR-009 decision 1: the patch lives in our repository and the third-party tree gets
zero edits, so the change is reviewable and cannot be lost by a reinstall.

**Why the constraint graph and not a filter.** SPEC-009's probe measured it: with
`populate_assets` switched on, upstream's own furniture solver put a 2.01 m shelf 0.1265 m2 inside
the task's protected region. A post-hoc filter can only make that shelf *vanish*, and the next run
puts something else there -- the solver's state space has no notion of the reservation. The reserved
volume has to be a fact the solver optimises against.

**The three seams, and why each is that seam.**

1. `Solver.solve_rooms` (post-call). The state exists from the moment the rooms are solved, and
   every furniture stage after it (`solve_large`, `solve_medium`, `solve_small`) reads `state.objs`.
   One object per reservation is inserted there as an `ObjectState` with `generator=None` -- which is
   also what keeps `populate_state_placeholders` from trying to realise it (`populate.py:62` skips a
   generatorless entry) -- and each carries a `Touching` relation to its room. Without that relation
   `_is_active_room_object` (`greedy/active_for_stage.py:59-75`) marks the object inactive for every
   stage that binds a `room` variable, and both the evaluator (`evaluate.py:41`) and domain matching
   (`domain_contains.py:48`) then skip it: the failure mode would be a constraint that silently
   constrains nothing.
2. `checks.check_all` (post-call). Upstream's validation runs on exactly the graph upstream built,
   and our terms are appended afterwards. A reservation is a static object with no greedy stage, so
   it cannot pass `check_problem_greedy_coverage`, and that check is about *placeable* variables --
   running it unmodified over the unmodified graph keeps its meaning intact rather than weakening it.
3. `RandomStageExecutor.run_stage` for the **last** solve stage (post-call, in a `finally`).
   Deleting the reservations the moment `solve_small` finishes makes SPEC-011 requirement 9 true
   *by construction*: they cannot reach the export, the instance list or a collision proxy because
   they no longer exist when those are built. The first attempt hooked
   `populate.populate_state_placeholders` instead, on the reasoning that `populate_assets` is the
   next stage -- and it never fired, because `populate_assets` is a stage our own builds switch off
   (`INFINIGEN_STAGE_OVERRIDES_OFF`), so the hook that was supposed to remove them belonged to a
   stage that does not run. Measured; the provider's refusal is what caught it
   (`BACKGROUND_RESERVATION_OBJECT_LEAKED(not removed)`). The seam is now the ladder's own end, and
   `SOLVE_STAGES[-1]` is the name of that stage rather than a private copy of it.
4. `RandomStageExecutor.run_stage` for the **realising** stage (post-call), where the rejection pool
   runs (TASK 4). The objects exist there and nowhere else: `populate_state_placeholders` spawns
   them from the solver's own state (`populate.py:78-84`), which is also why the pool can select
   them by the solver's bookkeeping -- every state entry with a generator whose object is realised --
   rather than by guessing at names. The pool judges each candidate's whole object tree with the
   **same rule the ruler measures by** (`rfw_w.scenes.geometry.reservations`, shared on purpose) and
   deletes the tree when the piece may not stand there, so requirement 3's zero is true of the scene
   rather than of a filtered copy of it.

**The terms: two soft, one hard, and what the hard one is not.**

* `accessibility_cost(region, furniture, normal=+Z, dist=band_top)` -- upstream's own expression for
  this job (`home.py:592` uses it to keep doors clear). It extrudes the region's bounding box along
  +Z by the band and scores how deeply furniture penetrates it, so it is band-aware: a piece hanging
  above the band is not in it (`geometry.py:20-33`, `trimesh_geometry.py:991`).
* `distance(furniture, region).hinge(min, far).minimize(weight=w)` -- the 3D distance from the flat
  region mesh to the nearest furniture, penalised only when it is closer than `min`. A flat polygon
  at floor level makes this band-aware for free: a piece standing in the region is ~0 m away, a
  pendant 2.5 m above it is 2.5 m away. This is the term that actually pushes furniture out; the
  penetration cost mostly *reports* how deep something already is.
  (`hinge` returns a non-negative deviation, so it is `minimize` and the upper bound is effectively
  infinite -- `maximize` on a hinge, as `home.py:582` uses it, rewards being *outside* the range.)
* `accessibility_cost(...) in_range(0, hard_penetration_m)` as a **hard** constraint, per region. It
  is bounds-free on purpose and the reason matters: `constraint_bounding.py:158` returns no quantity
  bound for geometry costs, so the solver cannot *plan* with it -- it can only *judge*. The placement
  pressure is the soft terms'; this one is the tripwire that turns "something is standing in the
  task's region" into a counted violation, which `abort_unsatisfied_*` then makes loud
  (SPEC-011 requirement 5).

**What is deliberately not here.** `enable_collision_floating`/`enable_collision_solved` -- the
"turn collision checks on" half of ADR-009 decision 2 -- are read **only** by `place_floating`
(`generate_indoors.py:331-332`), and `floating_objs` is a stage our builds switch off
(`INFINIGEN_STAGE_OVERRIDES_OFF`). In this pin there is no solve-ladder collision check to turn on,
so the terms above are the whole of the mechanism; the record says so instead of leaving a reader to
wonder which half was implemented.

Weights and the hard threshold arrive in the bundle (`ReservationPolicy`) and are copied into the
record, because TASK 3 calibrates them: a weight that lives only in this file makes the calibration a
code edit instead of a measured number.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: The collection the reservation objects live in. Prefixed with our name so a reader of
#: `scene.blend` can see at a glance which objects are ours and are not room content.
COLLECTION = "pvscene:reservations"

#: An uncalibrated `distance` hinge has no meaningful upper bound: the penalty is for being *too
#: close*, and pulling furniture back toward the reservation would be the opposite of the intent.
HINGE_FAR_M = 1e6


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _contract():
    """The bundle's schema, from the contracts package.

    Imported here rather than re-declared: the request our provider writes and the record this patch
    reads have to agree, and one schema in one place is how they do. Needs `src` on `PYTHONPATH`,
    which the provider sets for every build it starts.
    """
    from rfw_w.scenes.contracts import background_constraints as contract
    return contract


# ---------------------------------------------------------------------------------------------
# The tag: how a constraint names the reservations
# ---------------------------------------------------------------------------------------------

def reservation_tag(region: str):
    """A tag carrying exactly one field: which reservation this object is.

    Built lazily so this module can be imported (and reviewed) outside Blender. Upstream's tag
    satisfaction is set algebra over tag objects with no registry and no hierarchy to register into
    (`tags.py:241-252`), so a tag defined here is selectable by `cl.scene()[{...}]` exactly like
    `Semantics.Room` -- and, equally important, it does *not* satisfy `Semantics.Object`, so the
    reservation can never be selected as furniture by upstream's own constraints.
    """
    if not hasattr(reservation_tag, "_cls"):
        from infinigen.core import tags as t

        @dataclass(frozen=True)
        class ReservationTag(t.Tag):
            region: str

        reservation_tag._cls = ReservationTag
    return reservation_tag._cls(region=region)


# ---------------------------------------------------------------------------------------------
# The installation
# ---------------------------------------------------------------------------------------------

class Injection:
    """One build's injection: the objects it built, the terms it asked for, what it removed."""

    def __init__(self, bundle: Any, *, provenance_path: Path, patch_path: Path,
                 entry_path: Path | None) -> None:
        self.bundle = bundle
        self.provenance_path = Path(provenance_path)
        self.patch_path = Path(patch_path)
        self.entry_path = None if entry_path is None else Path(entry_path)
        self.objects: list[dict[str, Any]] = []
        # Not `self.terms`: that name belongs to the method that builds them, and a field of the
        # same name shadows it -- measured, as `TypeError: 'list' object is not callable` from
        # inside `check_all` on the first injected build.
        self.term_records: list[str] = []
        self.state = None
        self.error = ""
        self.removed: dict[str, Any] = {"removed": False}
        self.pool: dict[str, Any] | None = None
        #: Why the pool has no record, when the realising stage ran and failed rather than never
        #: running. The two are different builds and the record says which one it was.
        self.pool_error = ""

    # -- seam 1 ------------------------------------------------------------------------------
    def insert(self, state) -> None:
        """Insert one invisible object per reservation, once the rooms have been solved."""
        import shapely.geometry
        from infinigen.core import tags as t
        from infinigen.core.constraints import constraint_language as cl
        from infinigen.core.constraints.example_solver import state_def
        from infinigen.core.constraints.example_solver.geometry import parse_scene
        from infinigen.core.util import blender as butil

        room_keys = [key for key, entry in state.objs.items() if t.Semantics.Room in entry.tags]
        if not room_keys:
            raise ValueError("BACKGROUND_INJECTION_NO_ROOM(the state has no room to reserve in)")
        collection = butil.get_collection(COLLECTION)
        for reservation in self.bundle.reservations:
            room_key = self._room_for(reservation, room_keys, state)
            obj = self._mesh_object(reservation.object_name, reservation.polygon_xy, collection)
            state.objs[reservation.object_name] = state_def.ObjectState(
                obj=obj, polygon=shapely.geometry.Polygon(reservation.polygon_xy),
                generator=None, tags={reservation_tag(reservation.region)},
                # The relation is not a geometric claim being tested: nothing moves this object, so
                # `all_relations_valid` (which raises on a relation it does not implement) is never
                # called on it. It is what makes the object a descendant of the room, which is the
                # condition a constraint must meet before the evaluator will look at it.
                relations=[state_def.RelationState(cl.Touching(), room_key)])
            # The geometry costs read `state.trimesh_scene` by *name*
            # (`impl_bindings.statenames_to_blnames`), so an object missing from it is a region the
            # solver scores but cannot measure -- and that failure would look exactly like "the
            # constraint did nothing".
            parse_scene.add_to_scene(state.trimesh_scene, obj)
            self._hide(obj)
            self.objects.append({
                "region": reservation.region, "object": reservation.object_name, "room": room_key,
                "band_top_m": reservation.band_top_m, "polygon_vertices": len(reservation.polygon_xy),
                "visible": False, "exported": False, "listed_as_instance": False,
                "trimesh_node": reservation.object_name})
            logger.info("pvscene: reservation object %s in room %s", reservation.object_name,
                        room_key)
        self.state = state

    def _room_for(self, reservation, room_keys: Sequence[str], state) -> str:
        """The room the reservation belongs to: the one whose polygon contains its centroid.

        One room today. The choice is by containment rather than by taking the first, because a
        region solved into the wrong room is a constraint that constrains nothing and reports
        success.
        """
        from shapely.geometry import Point

        ring = reservation.polygon_xy
        centroid = Point(sum(x for x, _ in ring) / len(ring), sum(y for _, y in ring) / len(ring))
        for key in room_keys:
            polygon = state.objs[key].polygon
            if polygon is not None and polygon.contains(centroid):
                return key
        return room_keys[0]

    @staticmethod
    def _mesh_object(name: str, polygon_xy, collection):
        """A flat polygon mesh at floor level -- the region's footprint, nothing extruded.

        Flat on purpose: the +Z extrusion distance in `accessibility_cost` *is* the band, and a
        floor-level polygon makes `distance` band-aware for free (the nearest point of a floor
        polygon to a piece hanging 2.5 m up is 2.5 m away).
        """
        import bpy

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(float(x), float(y), 0.0) for x, y in polygon_xy], [],
                         [list(range(len(polygon_xy)))])
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        return obj

    @staticmethod
    def _hide(obj) -> None:
        """Hidden and unselectable: machinery for the solver, not content.

        Called *after* the object joins `state.trimesh_scene`, and the order is measured:
        `parse_scene.add_to_scene` preprocesses the mesh in Blender's edit mode
        (`parse_scene.preprocess_obj`), and Blender refuses that on a hidden object --
        `RuntimeError: Operator bpy.ops.object.mode_set.poll() Cannot edit hidden object`. The
        stronger claim (requirement 9) is made later, by removing it before anything is exported.
        """
        obj.hide_viewport = True
        obj.hide_render = True
        obj.display_type = "WIRE"

    # -- seam 2 ------------------------------------------------------------------------------
    def terms(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """One soft pair and one hard term per region, over the reservation objects by name.

        Per region rather than over the union, because the bands differ: the protected region's band
        is the task's support height (0.575 m in the bedroom probe) while the corridor's is 1.8 m,
        and one extrusion height for both would either over-constrain above the support or let a
        wardrobe stand in the corridor.
        """
        from infinigen.core.constraints import constraint_language as cl
        from infinigen.core import tags as t
        from infinigen_examples.constraints import util as cu
        import numpy as np

        policy = self.bundle.policy
        rooms = cl.scene()[{t.Semantics.Room, -t.Semantics.Object}]
        # Furniture as the graph defines it (`home.py:513`). The reservation is not in this set: its
        # tag does not satisfy `Semantics.Object`.
        furniture = cl.scene()[{t.Semantics.Object}].related_to(rooms, cu.on_floor)
        constraints: dict[str, Any] = {}
        score_terms: dict[str, Any] = {}
        up = np.array([0.0, 0.0, 1.0])
        for reservation in self.bundle.reservations:
            region = cl.scene()[{reservation_tag(reservation.region)}]
            band = float(reservation.band_top_m)
            penetration = cl.accessibility_cost(region, furniture, up, band)
            score_terms[f"pvscene_reservation_{reservation.region}_penetration"] = (
                penetration.minimize(weight=policy.band_penetration_weight))
            score_terms[f"pvscene_reservation_{reservation.region}_distance"] = (
                cl.distance(furniture, region)
                .hinge(policy.distance_min_m, HINGE_FAR_M)
                .minimize(weight=policy.distance_weight))
            constraints[f"pvscene_reservation_{reservation.region}_penetration"] = (
                penetration.in_range(0.0, policy.hard_penetration_m))
            self.term_records.extend([
                f"score:{reservation.region}:accessibility_cost(+Z,{band:.4f}m)"
                f"*{policy.band_penetration_weight}",
                f"score:{reservation.region}:distance.hinge({policy.distance_min_m},{HINGE_FAR_M})"
                f"*{policy.distance_weight}",
                f"hard:{reservation.region}:accessibility_cost.in_range(0,"
                f"{policy.hard_penetration_m})"])
        return constraints, score_terms

    # -- seam 3 ------------------------------------------------------------------------------
    def remove(self) -> None:
        """Delete the reservation objects, so nothing downstream can export them."""
        import bpy
        from infinigen.core.util import blender as butil

        names = [entry["object"] for entry in self.objects]
        trimesh_scene = None if self.state is None else self.state.trimesh_scene
        for name in names:
            if self.state is not None:
                self.state.objs.pop(name, None)
            if trimesh_scene is not None:
                try:
                    trimesh_scene.delete_geometry(f"{name}_mesh")
                except Exception:  # noqa: BLE001 - absent is the state we are trying to reach
                    logger.debug("pvscene: no trimesh geometry for %s", name)
            obj = bpy.data.objects.get(name)
            if obj is not None:
                butil.delete(obj)
            logger.info("pvscene: removed reservation object %s", name)
        survivors_bpy = [name for name in names if name in bpy.data.objects]
        survivors_state = ([name for name in names if name in self.state.objs]
                           if self.state is not None else [])
        self.removed = {"removed": True, "names": names,
                        "when": f"{_contract().SOLVE_STAGES[-1]} finished",
                        "still_in_bpy": survivors_bpy, "still_in_state": survivors_state}

    def finish(self) -> None:
        """Remove the reservations and re-write the record with what actually happened.

        The record is written twice on purpose. At install time it can only say "the patch is in";
        whether the terms were appended and whether the objects were removed are facts that do not
        exist yet, and a record claiming either one then would be a claim written before the
        measurement. The second write -- after removal -- is the one that carries them, and it is
        also the one that survives if a later stage fails.
        """
        contract = _contract()
        self.remove()
        self.write_provenance(
            injection=contract.INJECTION_APPLIED,
            reason=f"reservations removed after {contract.SOLVE_STAGES[-1]}: the solve ladder is "
                   "the only part of the build that needs them")

    # -- seam 4: the rejection pool ------------------------------------------------------------
    def reject(self) -> None:
        """Refuse the realised objects that may not stand in the room, and record every verdict.

        ADR-009 decision 8, and the reason it is *here* rather than in a filter over the export:
        TASK 3 measured that no weight setting protects the reservation without emptying the room
        (60/25, 20/8 and 6/2.5 are the same room; 2/1 is the un-constrained room), and the office
        room under the most protective of those settings still kept two cabinets 0.0400 m^2 inside
        the task's protected region. A refusal is the only "do not stand there" the solver cannot
        outvote -- and doing it inside the build means the scene, not a filtered copy of it, is the
        room the ruler then measures.

        Selection is the solver's own bookkeeping, not a name pattern: `populate_state_placeholders`
        assigns each realised asset back into its state entry (`populate.py:78-84`), so the entries
        with a generator and a realised object *are* the generated content of this build. The name
        pattern is checked too, because an entry whose object is still a placeholder has not been
        realised and nothing has been gained by refusing a box.
        """
        contract = _contract()
        from rfw_w.scenes.geometry.polygon import Polygon
        from rfw_w.scenes.geometry.reservations import Region, measure_parts

        regions = [Region(name=item.region, polygon=Polygon(item.polygon_xy),
                          band_top_m=float(item.band_top_m))
                   for item in self.bundle.reservations]
        entries = self._realised_entries()
        candidates: list[dict[str, Any]] = []
        for key, state_obj in entries:
            parts = self._parts_of(state_obj.obj)
            measured = measure_parts(parts, regions)
            protected = max([part.overlaps_m2.get(contract.REGION_PROTECTED, 0.0)
                             for part in measured] or [0.0])
            # The corridor is reported the way the contract defines it (`corridor_overlap`):
            # circulation plus the door sweep, taken from the same part so the two numbers describe
            # one piece of furniture rather than two different ones.
            corridor = max([part.overlaps_m2.get(contract.REGION_CIRCULATION, 0.0)
                            + part.overlaps_m2.get(contract.REGION_DOOR_SWEEP, 0.0)
                            for part in measured] or [0.0])
            reasons = list(self._refusal_reasons(state_obj.obj.name, protected, corridor))
            entry = {
                "name": state_obj.obj.name, "factory": state_obj.obj.name.split("(")[0],
                "category": contract.category_of(state_obj.obj.name.split("(")[0]),
                "outcome": "rejected" if reasons else "kept", "attempt": 1,
                "parts": len(measured), "reasons": reasons,
                "overlaps_m2": {contract.REGION_PROTECTED: round(protected, 4),
                                contract.REGION_CIRCULATION: round(
                                    max([part.overlaps_m2.get(contract.REGION_CIRCULATION, 0.0)
                                         for part in measured] or [0.0]), 4),
                                contract.REGION_DOOR_SWEEP: round(
                                    max([part.overlaps_m2.get(contract.REGION_DOOR_SWEEP, 0.0)
                                         for part in measured] or [0.0]), 4)},
                "state_key": key,
            }
            candidates.append(entry)
            if reasons:
                self._delete_tree(state_obj.obj, key, reasons)
        # The record is built as the contract's own type and then serialised, so a pool that cannot
        # be read back cannot be written either: the audit reads this with the same schema, and a
        # second spelling of it here is how the two would come apart.
        record = contract.RejectionPool(
            stage=contract.REALISING_STAGE, ran=True,
            rule="rfw_w.scenes.geometry.reservations.measure_parts (hull footprint, region band, "
                 "m2) -- the same rule python -m rfw_w.scenes.workers.blender.measure_generated_furniture measures by",
            bounds=dict(self.bundle.budget),
            candidates=tuple(contract.PoolCandidate(
                name=item["name"], factory=item["factory"], category=item["category"],
                outcome=item["outcome"], attempt=item["attempt"], parts=item["parts"],
                overlaps_m2=item["overlaps_m2"], reasons=tuple(item["reasons"]))
                for item in candidates))
        self.pool = {
            **record.to_dict(),
            # The two facts the contract type does not carry because only the build can know them.
            "rejected_state_keys": [item["state_key"] for item in candidates
                                    if item["outcome"] == "rejected"],
            "attempts_per_candidate_max": 1,
            "why_bounded": "each candidate is judged once and either kept or deleted; the pool never "
                           "re-asks the solver (ADR-009 decision 8: no retrying until a better "
                           "picture comes out)",
        }
        logger.info("pvscene: rejection pool judged %d candidate(s), refused %d",
                    len(candidates), len(record.rejected()))
        self.write_provenance(
            injection=contract.INJECTION_APPLIED,
            reason=f"reservations removed after {contract.SOLVE_STAGES[-1]}; rejection pool ran "
                   f"after {contract.REALISING_STAGE}")

    def _realised_entries(self) -> list[tuple[str, Any]]:
        """The state entries whose generator realised an object: this build's generated content."""
        if self.state is None:
            return []
        return [(key, entry) for key, entry in self.state.objs.items()
                if entry.generator is not None and entry.obj is not None
                and "spawn_asset" in str(entry.obj.name)]

    @staticmethod
    def _parts_of(obj) -> list[Any]:
        """The object and its descendants: one factory can leave several Blender objects.

        The ruler measures each of them (they are separate meshes with separate footprints), so the
        pool has to look at each of them as well -- otherwise a child that hangs into the corridor
        would survive a pool that only ever measured its parent, and the ledger would disagree with
        the ruler about the room the pool left behind.
        """
        from infinigen.core.util import blender as butil

        return [item for item in butil.iter_object_tree(obj)
                if getattr(item, "type", None) == "MESH" and getattr(item, "data", None) is not None]

    def _refusal_reasons(self, name: str, protected: float, corridor: float) -> tuple[str, ...]:
        """Why this candidate may not stay: where it stands, and what category it would add.

        Both halves are requirement 4 and requirement 3 in one place. The category half is read
        against the *request's* vocabulary (which travels in the bundle) rather than against the
        patch's own idea of retrieval: a build that judged by a different vocabulary than the one it
        was sent would be reporting on a rule it was never given.
        """
        contract = _contract()
        factory = name.split("(")[0]
        category = contract.category_of(factory)
        reasons: list[str] = []
        if protected > 0.0:
            reasons.append(contract.overlap_reason(contract.REGION_PROTECTED, protected))
        if corridor > 0.0:
            reasons.append(contract.overlap_reason(contract.REGION_CIRCULATION, corridor))
        if not category:
            reasons.append(f"{contract.POOL_REASON_FACTORY_UNMAPPED}({factory})")
        elif category not in set(self.bundle.gaps):
            sub = (contract.NOT_A_GAP_COVERED if category in set(self.bundle.retrieval_categories)
                   else contract.NOT_A_GAP_NOT_DECLARED)
            reasons.append(f"{contract.POOL_REASON_NOT_A_GAP}({category}:{sub})")
        return tuple(reasons)

    def _delete_tree(self, obj, state_key: str, reasons: Sequence[str]) -> None:
        """Delete the whole object tree and its state entry: the two ways a piece can survive.

        The state entry is dropped as well as the Blender objects. `state.to_json` runs at the end
        of the build and would otherwise serialise an entry whose object no longer exists -- and
        `solve_state.json` is exactly where the *ruler* finds the generators it measures, so an
        entry left behind would have the ruler looking for a factory that is no longer in the room.
        """
        from infinigen.core.util import blender as butil

        tree = list(butil.iter_object_tree(obj))
        logger.info("pvscene: refusing %s (%d object(s)): %s", obj.name, len(tree),
                    ",".join(reasons))
        if self.state is not None:
            self.state.objs.pop(state_key, None)
        butil.delete(tree)

    # -- the record --------------------------------------------------------------------------
    def provenance(self, *, injection: str, reason: str = "") -> dict[str, Any]:
        contract = _contract()
        return {
            "version": contract.INJECTION_VERSION,
            "scene_id": self.bundle.scene_id, "core_hash": self.bundle.core_hash,
            "upstream_pin": self.bundle.upstream_pin,
            "entry_script": self.bundle.entry_script,
            "entry_script_sha256": (self._digest_or(self.entry_path,
                                                    self.bundle.entry_script_sha256)),
            "patch_module": self.bundle.patch_module,
            "patch_sha256": self._digest_or(self.patch_path, self.bundle.patch_sha256),
            "injection": injection, "injection_reason": reason,
            "reservations": [dict(reservation.to_dict(), object_name=reservation.object_name)
                             for reservation in self.bundle.reservations],
            "reservation_objects": self.objects,
            "terms": self.term_records,
            "objects_removed_before_export": self.removed,
            "rejection_pool": self.pool if self.pool is not None else {
                # Built through the contract's own type for the same reason the real pool is: the
                # absence is a record the audit reads, not a key it has to interpret.
                **contract.RejectionPool(
                    stage=contract.REALISING_STAGE, ran=False,
                    reason=(f"the realising stage failed, so the room was never judged: "
                            f"{self.pool_error}" if self.pool_error else
                            f"the rejection pool did not run: this build never reached "
                            f"{contract.REALISING_STAGE} (the stage is switched off in the default "
                            "path, and a build that died before it never got there)"),
                    bounds=dict(self.bundle.budget)).to_dict(),
                "rejected_state_keys": [], "attempts_per_candidate_max": 1,
                "why_bounded": "no candidate was judged"},
            "gaps": list(self.bundle.gaps),
            "budget": dict(self.bundle.budget),
            "policy": self.bundle.policy.to_dict(),
            "abort_unsatisfied": self._abort_switches(),
            "command_line": list(sys.argv),
            "collision_switches": {
                "used": False,
                "why": "enable_collision_floating/solved are read only by place_floating "
                       "(generate_indoors.py:331-332) and floating_objs is switched off in our "
                       "builds, so in this pin there is no solve-ladder collision check to turn on"},
            "error": self.error,
        }

    @staticmethod
    def _digest_or(path: Path | None, fallback: str) -> str:
        """The digest of the file as it is on disk *now*, so a record cannot agree with a stale plan."""
        if path is None or not Path(path).is_file():
            return fallback
        return _sha256(path)

    @staticmethod
    def _abort_switches() -> dict[str, Any]:
        """What the build's own command line carried, as command and record agree on the claim.

        Read from `sys.argv` rather than from gin, and the first version of this method is why:
        it called `gin.query_parameter("compose_indoors.abort_unsatisfied_large")`, which raised
        "Configurable 'compose_indoors' has no bound parameters" -- on a run whose
        `-p compose_indoors.abort_unsatisfied_large=True` had demonstrably taken effect, because
        the build died in `solve.py:214` with the abort message. A probe that reports "unbound" for
        a parameter that is bound is not evidence in either direction, so the record carries the
        command line instead and leaves the effect to the build's own log and exit status.
        """
        passed: dict[str, Any] = {}
        for group in ("large", "medium", "small"):
            name = f"compose_indoors.abort_unsatisfied_{group}"
            found = [item for item in sys.argv if item.startswith(f"{name}=")]
            passed[group] = (found[0].split("=", 1)[1] == "True" if found
                             else "not-passed(upstream default False: generate_indoors.py:202)")
        return {"values": passed, "evidence": "command-line(-p overrides in sys.argv)"}

    def write_provenance(self, *, injection: str, reason: str = "") -> None:
        self.provenance_path.write_text(
            json.dumps(self.provenance(injection=injection, reason=reason), indent=2) + "\n")


def install(bundle_path: Path, provenance_path: Path, *,
            entry_path: Path | None = None) -> Injection | None:
    """Install the three seams and write the record. Returns None when there is no bundle.

    A failed injection still writes its provenance: a build that quietly ran without the constraints
    it was asked for is exactly the failure this SPEC exists to make visible, and
    `injection: failed` with a reason is a fact the audit can report. A build with *no* bundle is
    not an injection attempt at all -- that is the default path -- so it writes nothing.
    """
    if not bundle_path:
        logger.info("pvscene: no reservation bundle; running unconstrained (default path)")
        return None
    contract = _contract()
    bundle = contract.ReservationBundle.from_dict(json.loads(Path(bundle_path).read_text()))
    injection = Injection(bundle, provenance_path=provenance_path,
                          patch_path=Path(__file__).resolve(), entry_path=entry_path)
    try:
        from infinigen.core.constraints import checks
        from infinigen.core.constraints.example_solver.solve import Solver
        from infinigen.core.util.pipeline import RandomStageExecutor

        original_solve_rooms = Solver.solve_rooms

        def solve_rooms(self, *args, **kwargs):
            state = original_solve_rooms(self, *args, **kwargs)
            injection.insert(state)
            return state

        original_check_all = checks.check_all

        def check_all(prob, greedy_stages, all_vars):
            original_check_all(prob, greedy_stages, all_vars)  # upstream's own graph, unchanged
            constraints, score_terms = injection.terms()
            prob.constraints.update(constraints)
            prob.score_terms.update(score_terms)
            logger.info("pvscene: appended %d constraint(s) and %d score term(s)",
                        len(constraints), len(score_terms))

        original_run_stage = RandomStageExecutor.run_stage

        def run_stage(self, name, *args, **kwargs):
            if name == contract.REALISING_STAGE:
                # Seam 4. Post-call on success: the pool's claim is about the room the stage left,
                # and a stage that raised has no room to judge. A failed stage still writes its
                # reason into the record (`provenance()`), so the absence is a fact and not silence.
                try:
                    result = original_run_stage(self, name, *args, **kwargs)
                except Exception as error:  # noqa: BLE001 - recorded, then re-raised unchanged
                    injection.pool_error = f"{type(error).__name__}: {error}"
                    injection.write_provenance(
                        injection=contract.INJECTION_APPLIED,
                        reason=f"reservations removed after {contract.SOLVE_STAGES[-1]}; "
                               f"{contract.REALISING_STAGE} failed")
                    raise
                injection.reject()
                return result
            if name != contract.SOLVE_STAGES[-1]:
                return original_run_stage(self, name, *args, **kwargs)
            try:
                return original_run_stage(self, name, *args, **kwargs)
            finally:
                # Also on failure: a build that dies in the last solve stage must not leave the
                # reservations in a blend that something might still read.
                injection.finish()

        Solver.solve_rooms = solve_rooms
        checks.check_all = check_all
        RandomStageExecutor.run_stage = run_stage
    except Exception as error:  # noqa: BLE001 - a patch that cannot install is a named failure
        injection.error = f"{type(error).__name__}: {error}"
        injection.write_provenance(injection=contract.INJECTION_FAILED, reason=injection.error)
        logger.error("pvscene: constraint patch failed to install\n%s", traceback.format_exc())
        return injection
    injection.write_provenance(injection=contract.INJECTION_APPLIED,
                               reason="installed before the build ran")
    logger.info("pvscene: constraint patch installed (%d reservations)",
                len(bundle.reservations))
    return injection
