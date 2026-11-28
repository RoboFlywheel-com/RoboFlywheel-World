#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim column of the SPEC-014 five-backend figure.

The other four columns rebuild the frozen scene from ``common_scene.json``'s
exported OBJ meshes. Isaac must not: it is the engine that *authored* the
delivery, so the faithful thing is to re-run the **production Isaac worker** on
the delivery's own request with the frozen figure camera substituted in. The
geometry is then the same scene version by construction, not by re-import, and
the only things this script changes are:

* ``cameras`` -- replaced by the one frozen figure camera (SPEC-014's shared K);
* ``simulate_seconds = 0.0`` -- SPEC-014 compares the *authored* geometry, and
  the production request carries 5 s of settle that would move the cup and the
  notebook after the extractor froze their poses;
* ``room.shell_uri`` -- resolved to an absolute path, because the output
  directory is not the request's directory;
* for the collision pass only, every instance's ``visual_uri`` -> its
  ``collision_uri``.

Two worker runs, two worker report directories, two requests:

    worker_visible/    the normal request           -> rgb.png, depth_visible
    worker_collision/  visual_uri := collision_uri  -> depth_collision

The two depth passes are never merged and never share a file name: the visible
pass says what the scene looks like, the collision pass says what the scene
collides as, and for a library whose collision meshes are convex exports those
are different shapes (a hull of a desk with legs is a block). The room shell is
left alone in both passes, because the shell's collision geometry *is* its
visible geometry.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.workers.multisim.render_isaac \\
      --delivery outputs/office_pick_cup/s5_office_full_20260924_02 \\
      --scene outputs/_paper/spec014_common_scene_20260924_01 \\
      --output-dir outputs/_paper/spec014_renders_20260924_01/isaac \\
      --active-gpu 2
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# python -m rfw_w.scenes.workers.multisim.render_isaac -> repo root
ROOT = Path.cwd()

from rfw_w.scenes.workers.multisim.common import (
    CommonScene,
    HIDDEN_SHELL_ROLES,
    SceneObject,
    appearance_gates,
    backend_report,
    digest,
    load_light_rig,
    save_depth_products,
    write_json,
)

VISIBLE_PREFIX = "depth_visible"
COLLISION_PREFIX = "depth_collision"

# --- depth self-check ------------------------------------------------------- #
# The alignment audit excludes a two-pixel silhouette band; here it is expressed
# as a depth-gradient bound, exactly as the other columns express it: a pixel is
# comparable only when its 3x3 neighbourhood is flat within this. On a surface
# seen edge-on the renderer evaluates the pixel *centre* while the sampled
# geometry sits up to half a pixel away, which is a sampling artefact and not a
# depth error.
EDGE_TOLERANCE_M = 0.01
# A deterministic grid of pixel centres (~4000 of them). A seeded random sample
# would make a quoted audit number depend on a seed.
SELF_CHECK_COLUMNS = 80
SELF_CHECK_ROWS = 50
# Triangles are bucketed into screen cells this many pixels on a side, so a
# sample pixel tests its cell's triangles rather than all ~480,000 of them.
SELF_CHECK_CELL_PX = 32
# The near plane the self-check clips against, in metres. The worker's authored
# camera clips at (0.01, 100), so matching its near plane keeps the self-check
# looking at the same geometry. This exists because a triangle that *straddles*
# the plane has no projection at all -- and the floor and every wall straddle it,
# since they extend behind the camera. Dropping those triangles (the obvious
# `all(z > 0)` rule) removes exactly the large interior surfaces: measured on this
# column, it left 1023 of 3434 sampled pixels with no covering geometry at all and
# a 6.8 cm p95 / 23 cm max residual against a render whose depth is exact.
NEAR_CLIP_M = 0.01


def png_size(path: Path) -> tuple[int, int]:
    """Width and height out of the PNG IHDR. Read here rather than trusted from
    the request, so a render that silently came back at another resolution is
    caught by the file and not by the field that asked for it."""
    header = path.open("rb").read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise SystemExit(f"MULTISIM_DEPTH_INVALID(isaac, {path.name} is not a PNG)")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def load_delivery(delivery: Path) -> tuple[dict, Path, dict]:
    """The delivery manifest, the request it points at, and that request."""
    manifest_path = delivery / "delivery_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"MULTISIM_PROVENANCE_MISSING(isaac, no manifest at {manifest_path})")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_request = delivery / manifest["isaac_request_path"]
    if not source_request.is_file():
        raise SystemExit(f"MULTISIM_PROVENANCE_MISSING(isaac, no request at {source_request})")
    request = json.loads(source_request.read_text(encoding="utf-8"))
    if request["scene_version"] != manifest["scene_version"]:
        raise SystemExit(
            "MULTISIM_SCENE_VERSION_MISMATCH(isaac, request="
            f"{request['scene_version'][:12]} manifest={manifest['scene_version'][:12]})"
        )
    return manifest, source_request, request


def resolve_shell(source_request: Path, request: dict) -> Path:
    shell_uri = request["room"].get("shell_uri")
    if not shell_uri:
        raise SystemExit("MULTISIM_ROOM_UNSUPPORTED(isaac, the request carries no shell_uri)")
    shell = (source_request.parent / shell_uri).resolve()
    if not shell.is_file():
        raise SystemExit(f"MULTISIM_ROOM_UNSUPPORTED(isaac, shell missing: {shell})")
    return shell


def figure_camera_record(camera) -> dict:
    """The frozen figure camera as the worker's camera record.

    Every numeric comes from ``camera.json``; nothing is left to the worker's
    own defaults. A request that omits the apertures renders against USD's
    20.955 x 15.2908, which is not the frustum the other four columns were
    given, and the deviation would be invisible in the report.
    """
    return {
        "camera_id": camera.name,
        "purpose": "export",
        "eye": [float(v) for v in camera.eye],
        "target": [float(v) for v in camera.target],
        "focal_length_mm": float(camera.focal_length_mm),
        "horizontal_aperture_mm": float(camera.horizontal_aperture_mm),
        "vertical_aperture_mm": float(camera.vertical_aperture_mm),
        "resolution": [int(camera.width), int(camera.height)],
        "target_instance_ids": [],
        "require_target_visible": False,
        "save_depth": True,
        "save_instance_ids": True,
    }


def build_request(
    source_request: Path, request: dict, camera, *, collision_pass: bool, rig=None
) -> tuple[dict, Path, list[dict], dict]:
    shell = resolve_shell(source_request, request)
    # Deep copy: the delivery's request is evidence and is never edited.
    figure = json.loads(json.dumps(request))
    figure["cameras"] = [figure_camera_record(camera)]
    figure["room"]["shell_uri"] = str(shell)
    figure["simulate_seconds"] = 0.0

    # Isaac is the one backend whose lighting is not expressed in the rig's
    # terms. The production worker takes SPEC-008's calibrated profile --
    # dome / sun / ceiling_fill intensities, with no direction field for the sun
    # at all -- so the rig's ambient triple and its directional vector have no
    # slot here. What DOES transfer is the exposure gain, which is the thing the
    # unified-exposure requirement actually needs. Applied as one scale across
    # the whole profile so the profile's internal balance is preserved, and the
    # parts that could not be applied are named rather than passed over.
    lighting_applied = {"rig": None, "not_applied": [], "why": None}
    if rig is not None:
        # The shared shell albedo. Isaac is the one backend that does not read the
        # bundle's OBJ shell: it runs the production worker against the delivery's
        # Infinigen USD, whose surfaces carry authored materials, including a dark
        # purple accent wall. Setting `room_materials` alone does NOT reach them --
        # `_author_room` hands off to `author_shell` and returns before a material
        # is built, so that whole block is the procedural fallback's. Measured
        # 2026-09-25: the wall the camera faces read near-black here and light grey
        # in the other four, no lighting change moved it, and raising the ceiling
        # fill 18x only revealed the purple. `shell_material_override` is the
        # worker's opt-in that rebinds every mesh under /World/Room to the wall
        # colour below, which is the same one-albedo-everywhere the MTL-less OBJ
        # shell gives the other four. It is off for every delivery.
        shell_rgb = list(rig.shell_albedo)
        materials = dict(figure.get("room_materials") or {})
        overridden = {}
        for surface in ("wall", "ceiling", "floor"):
            if isinstance(materials.get(surface), dict):
                overridden[surface] = list(materials[surface].get("colour", []))
                materials[surface] = {**materials[surface], "colour": shell_rgb}
        figure["room_materials"] = materials
        figure["shell_material_override"] = True
        gain = rig.gain_for("isaac")
        before = dict(figure.get("lighting", {}))
        # One factor across dome/sun/ceiling_fill, so SPEC-008's internal balance
        # survives. A per-channel scale was tried first, on the theory that the
        # black wall was the dome being starved; raising the dome 2.6x changed the
        # frame by nothing at all (mean 86.89 before and after), which is what sent
        # the search to the shell materials above. The room is sealed, so the dome
        # is largely outside it and the profile's balance is already tuned for that.
        scaled = {
            key: (value * gain if key.endswith("_intensity") else value)
            for key, value in before.items()
        }
        figure["lighting"] = scaled
        lighting_applied = {
            "rig": rig.name,
            "gain": gain,
            "profile_before": before,
            "profile_after": scaled,
            "shell_albedo": shell_rgb,
            "shell_albedo_replaced": overridden,
            "shell_material_override": True,
            "not_applied": [
                f"RIG_AMBIENT_UNSUPPORTED(isaac, {list(rig.ambient)})",
                f"RIG_DIRECTION_UNSUPPORTED(isaac, {list(rig.direction)})",
                f"RIG_POINT_LIGHTS_UNSUPPORTED(isaac, {len(rig.point_lights)})",
            ],
            "why": (
                "the Isaac path runs the production worker, whose request carries SPEC-008's "
                "calibrated dome/sun/ceiling_fill profile and has no field for a light "
                "direction or a point light. Only the exposure gain transfers; it is applied "
                "as a single scale on every *_intensity so the profile's internal balance is "
                "unchanged. The rig's geometry of light is NOT applied here, and that is why "
                "this column's shadows differ from the other four."
            ),
        }

    substitutions: list[dict] = []
    if collision_pass:
        for instance in figure["instances"]:
            collision = instance.get("collision_uri")
            if not collision:
                raise SystemExit(
                    "MULTISIM_GEOMETRY_MISMATCH(isaac, collision pass: "
                    f"{instance['instance_id']} has no collision_uri to substitute)"
                )
            collision_path = Path(collision)
            if not collision_path.is_file():
                raise SystemExit(
                    "MULTISIM_GEOMETRY_MISMATCH(isaac, collision pass: "
                    f"{instance['instance_id']} collision mesh missing at {collision})"
                )
            substitutions.append(
                {
                    "instance_id": instance["instance_id"],
                    "entity_id": instance["entity_id"],
                    "visual_uri_before": instance["visual_uri"],
                    "visual_uri_after": collision,
                    "collision_sha256": digest(collision_path),
                }
            )
            instance["visual_uri"] = collision
    return figure, shell, substitutions, lighting_applied


def run_worker(
    request_path: Path, worker_out: Path, *, active_gpu: int, timeout_s: int
) -> tuple[int, dict | None]:
    """Run the production worker entry point and read its report back.

    The directory is deliberately left absent: the worker refuses to run into an
    existing one, so a stale report can never be read as this run's result.
    """
    worker_out = worker_out.resolve()
    if worker_out.exists():
        raise SystemExit(
            f"MULTISIM_PROVENANCE_MISSING(isaac, {worker_out} already exists; a stale report "
            "must not be able to look like this run's result)"
        )
    command = [
        sys.executable,
        "-m", 'rfw_w.scenes.workers.launch',
        "--request",
        str(request_path),
        "--output-dir",
        str(worker_out),
        "--active-gpu",
        str(active_gpu),
        "--timeout",
        str(timeout_s),
    ]
    print("running: " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    report_path = worker_out / "report.json"
    if not report_path.is_file():
        # A process that exited without a report is a failure even at code 0: that is
        # exactly what Kit's shutdown path can produce.
        return completed.returncode or 1, None
    return completed.returncode, json.loads(report_path.read_text(encoding="utf-8"))


def capture_for(report: dict, camera_name: str, pass_name: str) -> dict:
    capture = next(
        (item for item in report.get("captures", ()) if item.get("camera_id") == camera_name), None
    )
    if capture is None:
        raise SystemExit(
            f"MULTISIM_DEPTH_INVALID(isaac, {pass_name}: the worker reported no capture for "
            f"{camera_name})"
        )
    return capture


def collect_depth(
    out_dir: Path,
    worker_out: Path,
    capture: dict,
    prefix: str,
    *,
    camera,
    pass_name: str,
) -> dict:
    source = worker_out / str(capture["depth_path"])
    if not source.is_file():
        raise SystemExit(
            f"MULTISIM_DEPTH_INVALID(isaac, {pass_name}: depth buffer missing at {source})"
        )
    depth = np.load(source)
    if depth.shape != (camera.height, camera.width):
        raise SystemExit(
            f"MULTISIM_DEPTH_INVALID(isaac, {pass_name}: depth is {depth.shape}, the camera asked "
            f"for {(camera.height, camera.width)})"
        )
    if capture.get("depth_convention") != "camera_z_meters":
        raise SystemExit(
            "MULTISIM_DEPTH_CONVENTION_MISMATCH(isaac, "
            f"{pass_name}: the worker declared {capture.get('depth_convention')!r})"
        )
    detail = save_depth_products(out_dir, depth, prefix=prefix)
    detail["worker_depth_valid_fraction"] = capture.get("depth_valid_fraction")
    detail["worker_depth_min_m"] = capture.get("depth_min_m")
    detail["worker_depth_max_m"] = capture.get("depth_max_m")
    detail["source_buffer"] = str(source)
    return detail


# --------------------------------------------------------------------------- #
# depth self-check: the published depth against the geometry that was placed
# --------------------------------------------------------------------------- #


def read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, triangles)`` out of an OBJ, in the mesh's own frame.

    Faces are fanned: a quad is two triangles. Materials and normals are ignored
    because the check is about geometry, and a face reference with more than one
    field (``f v/vt/vn``) is read by its vertex index.
    """
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            corners = [int(token.split("/")[0]) for token in line.split()[1:]]
            count = len(vertices)
            corners = [corner - 1 if corner > 0 else count + corner for corner in corners]
            for index in range(1, len(corners) - 1):
                faces.append((corners[0], corners[index], corners[index + 1]))
    return (
        np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
        np.asarray(faces, dtype=np.int64).reshape(-1, 3),
    )


def placed_geometry(scene_ir: dict, request: dict, scene: CommonScene) -> tuple[list[dict], dict]:
    """The world-space triangles the visible pass actually placed, with a ledger.

    Instances: the OBJ each instance's ``visual_uri`` names -- the file the
    worker loaded -- placed by the delivery's own ``scene_ir`` world transform.
    ``scene_ir`` is used rather than the request's pose because it is the
    delivery's own record of where the geometry ended up; the two are compared
    here and the gap is reported, so "which pose was used" cannot hide.

    Room shell: the common scene's exported shell meshes, which extract_scene
    already wrote in world space out of the same USD the worker rendered --
    including ``shell_exterior``, because the hidden rule is not applied on this
    column and the published depth therefore contains it.

    The worker applies a Y-up correction when ``normalisation`` is set. This
    check models the uncorrected case, so an instance that carries one is a named
    gap rather than a silently mis-posed mesh.
    """
    poses = {item["entity_id"]: item for item in scene_ir["instances"]}
    groups: list[dict] = []
    ledger: dict = {"pose_gap": 0.0, "mesh_bounds_gap_m": 0.0, "normalisations": [], "objects": []}
    for instance in request["instances"]:
        visual = Path(instance["visual_uri"])
        if not visual.is_file():
            raise SystemExit(
                "MULTISIM_GEOMETRY_MISMATCH(isaac, self-check: the visible mesh the request "
                f"names for {instance['instance_id']} is missing at {visual})"
            )
        normalisation = str(instance.get("normalisation", "none"))
        ledger["normalisations"].append(normalisation)
        if normalisation != "none":
            raise SystemExit(
                "MULTISIM_GEOMETRY_MISMATCH(isaac, self-check does not model "
                f"normalisation={normalisation!r} on {instance['instance_id']})"
            )
        record = poses.get(instance["entity_id"])
        if record is None:
            raise SystemExit(
                "MULTISIM_GEOMETRY_MISMATCH(isaac, self-check: "
                f"{instance['entity_id']} is not in scene_ir)"
            )
        world_transform = record["world_transform"]
        translation = np.asarray(world_transform["translation"], dtype=np.float64)
        rotation = np.asarray(world_transform["rotation_wxyz"], dtype=np.float64)
        ledger["pose_gap"] = max(
            ledger["pose_gap"],
            float(np.abs(translation - np.asarray(instance["translation"], dtype=np.float64)).max()),
            float(np.abs(rotation - np.asarray(instance["rotation_wxyz"], dtype=np.float64)).max()),
        )
        vertices, faces = read_obj_mesh(visual)
        # SceneObject.matrix_world() is the same transform every other column
        # applies to this instance, so this check cannot disagree with them about
        # where the object is.
        matrix = SceneObject(
            object_id=instance["instance_id"],
            kind="furniture",
            role=instance["entity_id"],
            visual_mesh=visual,
            collision_mesh=visual,
            translation=translation,
            rotation_wxyz=rotation,
            scale=1.0,
        ).matrix_world()
        world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        # scene_ir's bounds are the asset's own *local* box, so they are compared
        # in the mesh's own frame -- the OBJ's local AABB against the declared one.
        # Comparing the world AABB instead only measures the instance's rotation
        # and reports a rotation artefact as a mesh mismatch (0.305 m, measured,
        # on a scene whose local boxes all agree to 0.7 mm). This catches a
        # swapped or re-exported mesh; a moved instance is caught by `pose_gap`.
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        if record.get("bounds_min_m") is not None:
            declared_low = np.asarray(record["bounds_min_m"], dtype=np.float64)
            declared_high = np.asarray(record["bounds_max_m"], dtype=np.float64)
            ledger["mesh_bounds_gap_m"] = max(
                ledger["mesh_bounds_gap_m"],
                float(np.abs(low - declared_low).max()),
                float(np.abs(high - declared_high).max()),
            )
        groups.append(
            {
                "object_id": instance["instance_id"],
                "kind": "furniture",
                "source_path": str(visual),
                "world": world,
                "faces": faces,
                "vertices": int(vertices.shape[0]),
                "triangles": int(faces.shape[0]),
            }
        )
    for obj in scene.objects(include_hidden=True):
        if obj.kind != "shell":
            continue
        vertices, faces = read_obj_mesh(obj.visual_mesh)
        groups.append(
            {
                "object_id": obj.object_id,
                "kind": "shell",
                "role": obj.role,
                "source_path": str(obj.visual_mesh),
                "world": vertices,
                "faces": faces,
                "vertices": int(vertices.shape[0]),
                "triangles": int(faces.shape[0]),
            }
        )
    ledger["objects"] = [
        {
            "object_id": group["object_id"],
            "kind": group["kind"],
            "vertices": group["vertices"],
            "triangles": group["triangles"],
        }
        for group in groups
    ]
    ledger["meshes_read"] = {
        "from_the_common_scene": sorted(
            {group["source_path"] for group in groups if group["kind"] == "shell"}
        ),
        "from_the_delivery": sorted(
            {group["source_path"] for group in groups if group["kind"] != "shell"}
        ),
    }
    ledger["triangles_total"] = sum(group["triangles"] for group in groups)
    ledger["hidden_roles_not_hidden_here"] = list(HIDDEN_SHELL_ROLES)
    return groups, ledger


def bundle_digest_check(scene_root: Path, files: list[Path]) -> dict:
    """``bundle.sha256`` against the bundle files this column actually read.

    The bundle's digest list can only certify what reads the bundle, and on this
    column that is the self-check's shell meshes and nothing else: the renderer is
    driven by the delivery's own request and never opens the bundle. Reporting the
    count that matched -- rather than asserting the bundle is "verified" -- keeps
    the coverage of the check visible.
    """
    ledger_path = scene_root / "bundle.sha256"
    if not ledger_path.is_file():
        return {
            "bundle_sha256": None,
            "blocks_used": [],
            # The same keys as the checked path, so a caller never has to ask which
            # shape it got back -- and so a bundle without the file is a stated
            # absence rather than a KeyError two frames later.
            "scene_version_declared": None,
            "digests_checked": 0,
            "digests_matched": 0,
            "mismatched": [],
            "not_listed": [],
            "note": "the common scene carries no bundle.sha256 to check against",
        }
    document = json.loads(ledger_path.read_text(encoding="utf-8"))
    declared: dict[str, str] = {}
    blocks_used: list[str] = []
    for name, block in document.items():
        if isinstance(block, dict):
            declared.update(block)
            blocks_used.append(name)
    matched = 0
    mismatched: list[dict] = []
    not_listed: list[str] = []
    for path in files:
        key = path.relative_to(scene_root).as_posix()
        want = declared.get(key)
        if want is None:
            not_listed.append(key)
            continue
        actual = digest(path)
        if actual == want:
            matched += 1
        else:
            mismatched.append({"path": key, "declared": want, "actual": actual})
    return {
        "bundle_sha256": str(ledger_path),
        "blocks_used": blocks_used,
        # The same file carries the bundle's own scene version, so the version this
        # column binds to can be checked against the bundle the meshes came from
        # and not only against the delivery manifest.
        "scene_version_declared": document.get("scene_version"),
        "digests_checked": len(files),
        "digests_matched": matched,
        "mismatched": mismatched,
        "not_listed": not_listed,
    }


def _clip_to_near(triangles: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman against the single near plane: 3 corners in, 1-2 out.

    Vectorising this across a mixed batch means juggling three different output
    sizes for the same array, so the straddlers (a small minority) are clipped in
    Python and the ones entirely in front skip this function altogether.
    """
    polygons: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for corners, forward in zip(triangles, distance):
        polygon: list[np.ndarray] = []
        for index in range(3):
            here = corners[index]
            there = corners[(index + 1) % 3]
            here_front = float(forward[index]) >= NEAR_CLIP_M
            there_front = float(forward[(index + 1) % 3]) >= NEAR_CLIP_M
            if here_front:
                polygon.append(here)
            if here_front != there_front:
                share = (NEAR_CLIP_M - float(forward[index])) / (
                    float(forward[(index + 1) % 3]) - float(forward[index])
                )
                polygon.append(here + share * (there - here))
        for index in range(1, len(polygon) - 1):
            polygons.append((polygon[0], polygon[index], polygon[index + 1]))
    if not polygons:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.asarray(polygons, dtype=np.float64)


def placed_triangles(groups: list[dict], camera) -> tuple[np.ndarray, np.ndarray]:
    """Every placed triangle as world-space corners, clipped to the near plane.

    Returns ``(corners, owner)``: an ``(n, 3, 3)`` array and the group index each
    triangle came from. Clipping first is what makes ``camera.project`` valid for
    every returned corner, so the caller needs no "is it in front?" rule of its
    own — and no rule of its own to get wrong.
    """
    eye = camera.eye
    forward = camera.forward
    soups: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for index, group in enumerate(groups):
        faces = group["faces"]
        if faces.shape[0] == 0:
            continue
        corners = group["world"][faces]
        distance = (corners - eye) @ forward
        in_front = distance >= NEAR_CLIP_M
        entirely_in_front = in_front.all(axis=1)
        parts = [corners[entirely_in_front]]
        straddling = np.flatnonzero(~entirely_in_front & in_front.any(axis=1))
        if straddling.size:
            parts.append(_clip_to_near(corners[straddling], distance[straddling]))
        stacked = np.concatenate(parts, axis=0)
        if stacked.shape[0] == 0:
            continue
        soups.append(stacked)
        owners.append(np.full(stacked.shape[0], index, dtype=np.int64))
    if not soups:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.concatenate(soups, axis=0), np.concatenate(owners)


def depth_self_check(camera, depth: np.ndarray, groups: list[dict]) -> dict:
    """The published depth against the true ``camera_z`` of the placed geometry.

    For each sampled pixel centre, the ray through it is intersected with the
    geometry analytically: a triangle covers the pixel centre when its own
    projection does, and the depth of that intersection is the perspective-correct
    interpolation ``1 / (w0/z0 + w1/z1 + w2/z2)`` of the triangle's projected
    barycentric weights. The nearest covering triangle is the first hit, so this
    is a ray cast that needs no ray-casting library and no scene the worker did
    not place. Vertices alone would not do: the desk is 40 vertices and 30 faces,
    so most of its pixels contain no vertex at all and a nearest-vertex rule would
    score the wall behind it as the surface. The geometry is clipped to the near
    plane before projection (`placed_triangles`), which is what keeps the floor
    and the walls — the surfaces that extend behind the camera — in the test.

    Residuals are reported over the pixels whose published depth is valid and
    whose 3x3 neighbourhood is flat; the excluded population is counted and its
    worst residual is reported with it, so the exclusion is visible rather than
    implied. Both the camera-z and the ray-distance hypothesis are scored, so the
    report says which one the render agrees with instead of only that some number
    was small.
    """
    height, width = depth.shape
    flat_depth = np.asarray(depth, dtype=np.float64).reshape(-1)

    corners, triangle_owner = placed_triangles(groups, camera)
    if corners.shape[0] == 0:
        return {"method": "no geometry", "samples": 0, "not_run": ["depth_self_check_m(no geometry)"]}
    uv_flat, z_flat = camera.project(corners.reshape(-1, 3))
    triangle_uv = uv_flat.reshape(-1, 3, 2)
    triangle_z = z_flat.reshape(-1, 3)

    # The triangles arrive already clipped to the near plane (see
    # `placed_triangles`), so every vertex projects; a triangle seen exactly
    # edge-on still projects to zero area and cannot cover a pixel centre.
    corner_a = triangle_uv[:, 1] - triangle_uv[:, 0]
    corner_b = triangle_uv[:, 2] - triangle_uv[:, 0]
    area = corner_a[:, 0] * corner_b[:, 1] - corner_a[:, 1] * corner_b[:, 0]
    on_screen = (
        (triangle_uv[:, :, 0].max(axis=1) >= 0.0)
        & (triangle_uv[:, :, 0].min(axis=1) <= width)
        & (triangle_uv[:, :, 1].max(axis=1) >= 0.0)
        & (triangle_uv[:, :, 1].min(axis=1) <= height)
    )
    keep = on_screen & (np.abs(area) > 1e-12)
    triangle_uv = triangle_uv[keep]
    triangle_z = triangle_z[keep]
    triangle_owner = triangle_owner[keep]
    area = area[keep]
    if triangle_uv.shape[0] == 0:
        return {
            "method": "no triangle projects into the frame",
            "samples": 0,
            "not_run": ["depth_self_check_m(no triangle projects into the frame)"],
        }

    # Bucket the triangles into screen cells, then read the buckets back per
    # sample pixel. Vectorised both ways: a per-triangle Python loop over half a
    # million triangles is what this avoids.
    cell = SELF_CHECK_CELL_PX
    cells_x = int(np.ceil(width / cell))
    cells_y = int(np.ceil(height / cell))
    low_x = np.clip(np.floor(triangle_uv[:, :, 0].min(axis=1) / cell).astype(np.int64), 0, cells_x - 1)
    high_x = np.clip(np.floor(triangle_uv[:, :, 0].max(axis=1) / cell).astype(np.int64), 0, cells_x - 1)
    low_y = np.clip(np.floor(triangle_uv[:, :, 1].min(axis=1) / cell).astype(np.int64), 0, cells_y - 1)
    high_y = np.clip(np.floor(triangle_uv[:, :, 1].max(axis=1) / cell).astype(np.int64), 0, cells_y - 1)
    span_x = high_x - low_x + 1
    span_y = high_y - low_y + 1
    per_triangle = span_x * span_y
    total_entries = int(per_triangle.sum())
    entry_triangle = np.repeat(np.arange(triangle_uv.shape[0]), per_triangle)
    block_start = np.repeat(np.cumsum(per_triangle) - per_triangle, per_triangle)
    within = np.arange(total_entries) - block_start
    entry_cell = (
        (low_y[entry_triangle] + within // span_x[entry_triangle]) * cells_x
        + low_x[entry_triangle]
        + within % span_x[entry_triangle]
    )
    order = np.argsort(entry_cell, kind="stable")
    sorted_cells = entry_cell[order]
    cell_starts = np.searchsorted(sorted_cells, np.arange(cells_x * cells_y), side="left")
    cell_ends = np.searchsorted(sorted_cells, np.arange(cells_x * cells_y), side="right")

    step_u = max(1, width // SELF_CHECK_COLUMNS)
    step_v = max(1, height // SELF_CHECK_ROWS)
    columns = np.arange(step_u // 2, width, step_u)
    rows = np.arange(step_v // 2, height, step_v)

    # The same 3x3 flatness rule the other columns use, on the published array.
    # The background is `inf`, so a neighbourhood touching it computes `inf - inf`;
    # that is a NaN comparison whose answer is "not gentle", which is the wanted
    # answer, and the warning it raises is noise rather than a finding.
    finite = np.isfinite(depth) & (depth > 0)
    gentle = np.zeros_like(finite)
    centre = depth[1:-1, 1:-1]
    with np.errstate(invalid="ignore"):
        gentle[1:-1, 1:-1] = (
            (np.abs(depth[:-2, 1:-1] - centre) < EDGE_TOLERANCE_M)
            & (np.abs(depth[2:, 1:-1] - centre) < EDGE_TOLERANCE_M)
            & (np.abs(depth[1:-1, :-2] - centre) < EDGE_TOLERANCE_M)
            & (np.abs(depth[1:-1, 2:] - centre) < EDGE_TOLERANCE_M)
        )

    residuals: list[float] = []
    ray_residuals: list[float] = []
    samples: list[dict] = []
    rejected: list[float] = []
    no_depth = 0
    no_geometry = 0
    for row in rows:
        for column in columns:
            # Pixel centres: the renderer's depth for pixel (v, u) belongs to the
            # ray through (u + 0.5, v + 0.5).
            point = np.array([float(column) + 0.5, float(row) + 0.5])
            flat_index = int(row) * width + int(column)
            measured = float(flat_depth[flat_index])
            cell_index = int(row // cell) * cells_x + int(column // cell)
            candidates = order[cell_starts[cell_index]:cell_ends[cell_index]]
            if candidates.size == 0:
                no_geometry += 1
                continue
            if not (np.isfinite(measured) and measured > 0.0):
                no_depth += 1
                continue
            triangle_uvs = triangle_uv[entry_triangle[candidates]]
            weights_a = _cross2(triangle_uvs[:, 1] - point, triangle_uvs[:, 2] - point)
            weights_b = _cross2(triangle_uvs[:, 2] - point, triangle_uvs[:, 0] - point)
            weights_c = _cross2(triangle_uvs[:, 0] - point, triangle_uvs[:, 1] - point)
            triangle_areas = area[entry_triangle[candidates]]
            # Both windings are inside. Coverage is a property of the triangle,
            # not of which way it happens to face: the room shell's meshes are
            # wound opposite to the furniture OBJs, so a one-winding rule drops
            # every shell floor/ceiling triangle. It did, and the samples then
            # resolved to the facade 5 cm behind them -- a flat 5.6 cm median
            # against a render that was exact.
            inside = (
                ((weights_a >= -1e-9) & (weights_b >= -1e-9) & (weights_c >= -1e-9))
                | ((weights_a <= 1e-9) & (weights_b <= 1e-9) & (weights_c <= 1e-9))
            )
            if not inside.any():
                no_geometry += 1
                continue
            z0 = triangle_z[entry_triangle[candidates], 0][inside]
            z1 = triangle_z[entry_triangle[candidates], 1][inside]
            z2 = triangle_z[entry_triangle[candidates], 2][inside]
            wa = weights_a[inside] / triangle_areas[inside]
            wb = weights_b[inside] / triangle_areas[inside]
            wc = weights_c[inside] / triangle_areas[inside]
            z_at_pixel = 1.0 / (wa / z0 + wb / z1 + wc / z2)
            front = int(np.argmin(z_at_pixel))
            true_z = float(z_at_pixel[front])
            owner = int(triangle_owner[entry_triangle[candidates][inside][front]])
            residual = measured - true_z
            du = (point[0] - camera.cx) / camera.fx
            dv = (point[1] - camera.cy) / camera.fy
            ray_distance = true_z * math.sqrt(1.0 + du * du + dv * dv)
            if gentle[row, column]:
                residuals.append(abs(residual))
                ray_residuals.append(abs(measured - ray_distance))
                samples.append(
                    {
                        "pixel": [int(column), int(row)],
                        "object_id": groups[owner]["object_id"],
                        "rendered_depth_m": measured,
                        "true_camera_z_m": true_z,
                        "residual_m": residual,
                    }
                )
            else:
                rejected.append(abs(residual))

    array = np.asarray(residuals, dtype=np.float64)
    rejected_array = np.asarray(rejected, dtype=np.float64)
    ray_array = np.asarray(ray_residuals, dtype=np.float64)
    worst = sorted(samples, key=lambda item: abs(item["residual_m"]), reverse=True)[:5]
    median_abs = float(np.median(array)) if array.size else None
    median_ray = float(np.median(ray_array)) if ray_array.size else None
    check = {
        "method": "per sampled pixel centre, the analytic first hit through that pixel against the "
                  "geometry the worker placed (each triangle that projects onto the pixel centre, "
                  "nearest one kept, depth by perspective-correct barycentric interpolation) minus "
                  "the published depth_visible.npy",
        "units": "m",
        "sample_grid": f"{step_u}x{step_v} px stride over {width}x{height} pixel centres",
        "samples": int(array.size),
        "n_points": int(array.size),
        "median_m": median_abs,
        "mean_m": float(array.mean()) if array.size else None,
        "mean_signed_m": float(np.mean([item["residual_m"] for item in samples])) if samples else None,
        "p95_m": float(np.percentile(array, 95)) if array.size else None,
        "max_m": float(array.max()) if array.size else None,
        "fraction_within_1cm": float((array < 0.01).mean()) if array.size else None,
        "fraction_within_2cm": float((array < 0.02).mean()) if array.size else None,
        "edge_rule": f"3x3 depth neighbourhood flat within {EDGE_TOLERANCE_M} m (the two-pixel "
                     "silhouette band SPEC-014 excludes)",
        "edge_rejected_samples": int(rejected_array.size),
        "edge_rejected_max_m": float(rejected_array.max()) if rejected_array.size else None,
        "pixels_with_no_placed_geometry": no_geometry,
        "pixels_with_no_valid_depth": no_depth,
        "convention_hypothesis": {
            "median_abs_if_camera_z_m": median_abs,
            "median_abs_if_ray_distance_m": median_ray,
            "verdict": (
                "undetermined: no sample survived" if median_abs is None or median_ray is None
                else ("camera_z_meters" if median_abs <= median_ray else "ray_distance")
            ),
        },
        "worst_samples": [
            {
                "residual_m": item["residual_m"],
                "pixel": item["pixel"],
                "object_id": item["object_id"],
                "rendered_depth_m": item["rendered_depth_m"],
                "true_camera_z_m": item["true_camera_z_m"],
            }
            for item in worst
        ],
        "geometry_summary": {
            "triangles_tested": int(triangle_uv.shape[0]),
            "objects": len(groups),
            "source": "the delivery's own request (visual_uri) placed by scene_ir's world transforms, "
                      "plus the common scene's shell meshes, which are already world-space",
        },
        "not_run": [] if array.size else [
            "depth_self_check_m(no sampled pixel had both a valid published depth and placed "
            "geometry covering it)"
        ],
    }
    return check


def _cross2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """2D cross product of two (n, 2) arrays: the sign picks the winding."""
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path,
                        help="the frozen common scene directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lighting", type=Path,
                        default=data_path('examples')
                        / "paper_multisim" / "lighting.json",
                        help="the shared five-backend light rig (SPEC-014 R5)")
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per worker run, seconds")
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")

    started = time.monotonic()
    scene = CommonScene.load(args.scene)
    rig = load_light_rig(args.lighting)
    camera = scene.camera
    manifest, source_request, source_payload = load_delivery(delivery)

    if scene.scene_version != manifest["scene_version"]:
        raise SystemExit(
            "MULTISIM_SCENE_VERSION_MISMATCH(isaac, common scene="
            f"{scene.scene_version[:12]} delivery={manifest['scene_version'][:12]})"
        )

    out.mkdir(parents=True, exist_ok=True)

    visible_request, shell, _, lighting_applied = build_request(
        source_request, source_payload, camera, collision_pass=False, rig=rig
    )
    collision_request, _, substitutions, _ = build_request(
        source_request, source_payload, camera, collision_pass=True
    )
    visible_path = out / "request_visible.json"
    collision_path = out / "request_collision.json"
    for path, payload in ((visible_path, visible_request), (collision_path, collision_request)):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    visible_out = out / "worker_visible"
    collision_out = out / "worker_collision"
    visible_code, visible_report = run_worker(
        visible_path, visible_out, active_gpu=args.active_gpu, timeout_s=args.timeout
    )
    collision_code, collision_report = run_worker(
        collision_path, collision_out, active_gpu=args.active_gpu, timeout_s=args.timeout
    )
    elapsed = round(time.monotonic() - started, 2)

    failures: list[str] = []
    for name, code, report in (
        ("visible", visible_code, visible_report),
        ("collision", collision_code, collision_report),
    ):
        if report is None:
            failures.append(f"{name}: the worker wrote no report (exit {code})")
        elif report.get("status") != "passed":
            failures.append(
                f"{name}: worker status {report.get('status')!r} "
                f"({report.get('error_type')}: {report.get('error_detail')})"
            )

    warnings = [
        "the common scene's hidden rule "
        f"{list(HIDDEN_SHELL_ROLES)} is not applied to the Isaac pass: the worker renders the built "
        "shell as authored. The facade lies outside every wall, so it occludes rather than shows "
        "from this interior camera -- but through the room's openings it is what the window looks "
        "onto, where the other four backends show nothing. Disclosed rather than patched, because "
        "removing it would mean editing the shell the delivery's own render used.",
        "the collision depth describes the simready library's collision meshes, which are convex "
        "exports. A hull of a desk with legs is a block: the collision pass is a different shape "
        "from the visible pass by construction, and the two are kept in separate files.",
        "the figure camera's vertical aperture is derived from the horizontal one by the resolution "
        "aspect (square pixels), while the delivery's own camera record carries 20.955 / 15.2908 "
        "(ADR-011). Only Isaac can honour two apertures at once, so the deviation is recorded here "
        "and is the one all five columns share.",
        "simulate_seconds is 0.0, so nothing settles: these captures describe the authored geometry "
        "the extractor froze, not the settled state the delivery's own report describes.",
    ]
    missing_collision_visible_pass = [
        instance["instance_id"]
        for instance in source_payload["instances"]
        if not instance.get("collision_uri")
        or not Path(instance["collision_uri"]).is_file()
    ]
    if missing_collision_visible_pass:
        warnings.append(
            "the visible pass has no collision proxy for "
            f"{sorted(missing_collision_visible_pass)}: the worker skips a missing collision mesh "
            "silently, so those instances contribute nothing to the physics, though they render "
            "normally."
        )

    if failures:
        write_json(
            out / "report.json",
            backend_report(
                backend="isaac",
                engine_version="",
                scene=scene,
                camera=camera,
                extra={
                    "status": "failed",
                    "error": "MULTISIM_DEPTH_INVALID(isaac, " + "; ".join(failures) + ")",
                    "elapsed_s": elapsed,
                    "worker_runs": {
                        "visible": {"exit_code": visible_code},
                        "collision": {"exit_code": collision_code},
                    },
                    "warnings": warnings,
                },
            ),
        )
        raise SystemExit("MULTISIM_DEPTH_INVALID(isaac, " + "; ".join(failures) + ")")

    assert visible_report is not None and collision_report is not None
    visible_capture = capture_for(visible_report, camera.name, "visible")
    collision_capture = capture_for(collision_report, camera.name, "collision")

    rgb_source = visible_out / str(visible_capture["rgb_path"])
    if not rgb_source.is_file():
        raise SystemExit(f"MULTISIM_DEPTH_INVALID(isaac, visible: rgb missing at {rgb_source})")
    width, height = png_size(rgb_source)
    if (width, height) != (camera.width, camera.height):
        raise SystemExit(
            f"MULTISIM_CAMERA_MISMATCH(isaac, rgb.png is {width}x{height}, the camera asked for "
            f"{camera.width}x{camera.height})"
        )
    shutil.copyfile(rgb_source, out / "rgb.png")

    visible_detail = collect_depth(
        out, visible_out, visible_capture, VISIBLE_PREFIX, camera=camera, pass_name="visible"
    )
    collision_detail = collect_depth(
        out, collision_out, collision_capture, COLLISION_PREFIX, camera=camera, pass_name="collision"
    )

    capture_state = visible_report.get("capture_state")
    if capture_state != "authored":
        # The whole column is a claim about the authored geometry; a settled capture would make
        # every other backend's frozen pose a comparison against a different scene state.
        raise SystemExit(
            "MULTISIM_GEOMETRY_MISMATCH(isaac, the worker captured the "
            f"{capture_state!r} state, not the authored one; simulate_seconds={visible_request['simulate_seconds']})"
        )

    # The alignment audit's per-backend residual. It reads the *published*
    # depth_visible.npy rather than the array in memory, so the number describes
    # the artefact a reader will open, not a variable that happened to agree with
    # it.
    scene_ir = json.loads((delivery / "scene_ir.json").read_text(encoding="utf-8"))
    groups, geometry_ledger = placed_geometry(scene_ir, visible_request, scene)
    self_check = depth_self_check(camera, np.load(out / f"{VISIBLE_PREFIX}.npy"), groups)
    self_check["geometry_ledger"] = geometry_ledger

    # Does the common scene's mesh-digest list have anything to do with this
    # column? Asked explicitly because the answer is "for the render, no" -- and
    # a reader who assumed otherwise would be crediting this column's images with
    # a certification they never had.
    bundle_files = [Path(path) for path in geometry_ledger["meshes_read"]["from_the_common_scene"]]
    bundle_digests = bundle_digest_check(scene.root, bundle_files)
    scene_binding = {
        "question": "are the common scene's bundle.sha256 mesh digests a binding for this column?",
        "for_the_render": "no. The renderer is the production Isaac worker driven by the delivery's "
                          "own request: room.shell_uri is the delivery's own built shell USD and "
                          "every instance visual_uri/collision_uri is a file inside the delivery "
                          "tree. The worker never opens the common scene's meshes, so an in-place "
                          "edit of a bundle mesh cannot change these images, and these images "
                          "cannot certify a bundle mesh.",
        "what_the_render_is_bound_to": {
            "scene_version": scene.scene_version,
            "delivery_manifest_scene_version": manifest["scene_version"],
            "equal": scene.scene_version == manifest["scene_version"],
            "check": "CommonScene.load(...).scene_version against the delivery manifest's, compared "
                     "before a worker starts; a mismatch is a named refusal, not a warning",
            "shell_uri": str(shell),
            "shell_sha256": digest(shell),
            "instance_meshes": [
                {
                    "instance_id": instance["instance_id"],
                    "visual_uri": instance["visual_uri"],
                    "visual_sha256": digest(Path(instance["visual_uri"])),
                }
                for instance in source_payload["instances"]
            ],
            "camera": {
                "path": str(scene.root / "camera.json"),
                "sha256": digest(scene.root / "camera.json"),
            },
        },
        "where_the_bundle_does_enter": "the self-check. It measures the published depth against the "
                                       "common scene's exported shell meshes, so those files are the "
                                       "geometry record the residual is about -- and that is the only "
                                       "place the bundle's digests can bite.",
        "bundle_scene_version_matches_the_delivery": (
            bundle_digests["scene_version_declared"] == manifest["scene_version"]
        ),
        "bundle_digests": bundle_digests,
    }

    geometry_loaded = {
        "source": "the delivery's own Isaac request, re-run through the production worker",
        "why_not_rebuilt_from_obj": "Isaac authored this delivery; re-running its own request makes "
                                    "the geometry the same scene version by construction rather "
                                    "than by re-import",
        "scene_id": source_payload["scene_id"],
        "instance_count": len(source_payload["instances"]),
        "instances_requested": [
            {
                "instance_id": instance["instance_id"],
                "entity_id": instance["entity_id"],
                "visual_uri": instance["visual_uri"],
                "collision_uri": instance.get("collision_uri"),
                "dynamic": bool(instance.get("dynamic")),
            }
            for instance in source_payload["instances"]
        ],
        "shell_uri": str(shell),
        "shell_parts": {
            "hidden_roles_not_applied": list(HIDDEN_SHELL_ROLES),
            "common_scene_shell_meshes": len(
                [o for o in scene.objects(include_hidden=True) if o.kind == "shell"]
            ),
            "note": "the request's shell is the built Infinigen USD; the common scene's OBJ export "
                    "of the same build is what the other four backends load",
        },
        "openings_requested": visible_report.get("openings_requested"),
        "openings_cut": visible_report.get("openings_cut"),
        # A render that reaches a report at all has loaded every visual mesh: _author_instance
        # raises rather than skipping when one is missing. `asset_textures` is the per-instance
        # material record that proves the material beside each mesh was found too.
        "per_instance_material_record": visible_report.get("asset_textures"),
        "renderer_instance_id_map": visible_report.get("instance_id_map"),
        "renderer_instance_id_map_source": visible_report.get("instance_id_map_source"),
        "visible_instance_ids": visible_capture.get("visible_instance_ids"),
        "nonblank": visible_capture.get("nonblank"),
        "color_std": visible_capture.get("color_std"),
        "annotator_family": visible_report.get("annotator_family"),
    }

    collision_pass = {
        "substitution": "instance.visual_uri := instance.collision_uri, per instance",
        "why": "the visible geometry pass must come from the geometry the engine shows and the "
               "collision pass from the geometry it collides as; substituting the visual mesh with "
               "the collision mesh is how the second pass is asked for in an engine that renders "
               "the visible hierarchy",
        "room_shell": "unsubstituted: the shell's collision geometry is its visible geometry, so "
                      "the shell is identical in both passes",
        "instances_substituted": len(substitutions),
        "substitutions": substitutions,
        "request": collision_path.name,
        "worker_report": str((collision_out / "report.json").relative_to(out)),
        "rgb_not_published": "the collision pass also renders an RGB; it is left inside "
                             "worker_collision/ because the main figure's RGB row is the visible "
                             "geometry, and an image of colliders must not be mistaken for it",
        "renderer_instance_id_map": collision_report.get("instance_id_map"),
    }

    provenance = {
        "source_delivery": str(delivery),
        "source_scene_version": manifest["scene_version"],
        "source_request_path": str(source_request),
        "source_request_sha256": digest(source_request),
        "shell_uri": str(shell),
        "shell_sha256": digest(shell),
        "common_scene_dir": str(scene.root),
        "common_scene_version": scene.scene_version,
        "camera_path": str((scene.root / "camera.json")),
        "camera_sha256": digest(scene.root / "camera.json"),
        "figure_request_visible_path": visible_path.name,
        "figure_request_visible_sha256": digest(visible_path),
        "figure_request_collision_path": collision_path.name,
        "figure_request_collision_sha256": digest(collision_path),
        "worker_report_visible_path": str((visible_out / "report.json").relative_to(out)),
        "worker_report_visible_sha256": digest(visible_out / "report.json"),
        "worker_report_collision_path": str((collision_out / "report.json").relative_to(out)),
        "worker_report_collision_sha256": digest(collision_out / "report.json"),
        "rgb_path": "rgb.png",
        "rgb_sha256": digest(out / "rgb.png"),
        "rgb_source": str(rgb_source),
        "claim": "figure_only_static_render: the delivery's own worker, the delivery's own request, "
                 "the figure camera, no settle",
    }

    from PIL import Image as _Image
    appearance, appearance_gaps = appearance_gates(
        np.asarray(_Image.open(out / "rgb.png").convert("RGB")), rig, camera, "isaac")

    write_json(
        out / "report.json",
        backend_report(
            backend="isaac",
            engine_version=str(visible_report.get("isaac_version") or ""),
            scene=scene,
            camera=camera,
            extra={
                # backend_report already declares the four depth files under these names; this
                # says which of them came from which pass.
                "geometry_loaded": geometry_loaded,
                "capture_state": capture_state,
                "capture_state_source": "worker report",
                "collision_pass": collision_pass,
                "provenance": provenance,
                "depth_products_detail": {
                    "visible": visible_detail,
                    "collision": collision_detail,
                },
                # SPEC-014 Requirement 3: the residual between this column's own depth and the
                # true camera-z of the geometry it placed. A residual that is merely large stays
                # a number in the ledger; only a check that could not run at all is a failure,
                # because that is the difference between a measurement and a claimed one.
                "depth_self_check_m": self_check,
                "scene_binding": scene_binding,
                "lighting": {**appearance, "applied": lighting_applied},
                "gates_failed": appearance_gaps,
                "status": ("failed" if self_check["not_run"]
                           else "appearance_gate_failed" if appearance_gaps else "passed"),
                "status_reason": (
                    "MULTISIM_DEPTH_INVALID(isaac, " + "; ".join(self_check["not_run"]) + ")"
                    if self_check["not_run"] else
                    "both passes rendered, the depth is self-consistent with the geometry that was "
                    "placed, and every figure-only substitution is recorded"
                ),
                "requests": {
                    "visible": visible_path.name,
                    "collision": collision_path.name,
                    "only_differences_from_the_delivery_request": [
                        "cameras <- the one frozen figure camera",
                        "simulate_seconds <- 0.0",
                        "room.shell_uri <- absolute path",
                        "instances[i].visual_uri <- collision_uri (collision request only)",
                    ],
                },
                "worker_runs": {
                    "visible": {
                        "exit_code": visible_code,
                        "status": visible_report.get("status"),
                        "wall_seconds": visible_report.get("wall_seconds"),
                        "settle_performed": visible_report.get("settle", {}).get("performed"),
                        "output_dir": visible_out.name,
                    },
                    "collision": {
                        "exit_code": collision_code,
                        "status": collision_report.get("status"),
                        "wall_seconds": collision_report.get("wall_seconds"),
                        "settle_performed": collision_report.get("settle", {}).get("performed"),
                        "output_dir": collision_out.name,
                    },
                },
                "elapsed_s": elapsed,
                "warnings": warnings,
            },
        ),
    )

    print(json.dumps(
        {
            "output_dir": str(out),
            "engine_version": visible_report.get("isaac_version"),
            "scene_version": manifest["scene_version"],
            "capture_state": capture_state,
            "rgb": f"{width}x{height}",
            "depth_visible_valid_fraction": visible_detail["valid_pixel_fraction"],
            "depth_collision_valid_fraction": collision_detail["valid_pixel_fraction"],
            "instances_substituted": len(substitutions),
            "depth_self_check_m": {
                "samples": self_check["samples"],
                "median_m": self_check["median_m"],
                "p95_m": self_check["p95_m"],
                "max_m": self_check["max_m"],
                "edge_rejected_samples": self_check["edge_rejected_samples"],
                "convention_verdict": self_check["convention_hypothesis"]["verdict"],
                "not_run": self_check["not_run"],
            },
            "status": "failed" if self_check["not_run"] else "passed",
            "scene_binding": {
                "render_binding": "scene_version + the delivery's own request",
                "bundle_digests": scene_binding["bundle_digests"],
            },
            "elapsed_s": elapsed,
        },
        indent=2,
    ))
    return 1 if self_check["not_run"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
