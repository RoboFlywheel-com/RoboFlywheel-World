#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a cutaway copy of a delivery's built shell: no ceiling, no near wall.

Why a copy rather than a different camera: from inside the room no lens shows the
room well -- the view is either a wide-angle corner looking down at the floor, or
it is blocked by the wall the camera is standing against. Both were tried and
both were rejected on sight. The standard answer is to cut the near side away and
look in level from outside, which is what this builds.

The cut is made per **face**, not per prim: measured on the office shell, all four
walls and their openings arrive as a single USD mesh (118 verts, 162 faces) under
one prim, so deactivating prims would remove every wall rather than one.

Rules, applied to every mesh in the shell:

* ``ceiling``, ``ceiling_skirting`` and ``exterior`` are dropped whole -- the
  ceiling is what the level camera would otherwise look at, and the facade is
  outside every wall.
* ``wall``, ``skirtingboard`` and the opening prims are clipped by the cut plane:
  faces on the camera's side are dropped, the rest kept.
* ``floor`` is kept whole. Cutting it would leave the room floating.

``mode="roof"`` is the dollhouse variant: only the first rule applies. Walls,
skirting and openings are kept at **full height** -- no cut plane at all --
because a top-down oblique camera reads the room through the missing ceiling,
and a wall clipped at a horizontal plane looks like a broken model from above
rather than like an opened one.

``wall_clip`` is the classic dollhouse compromise, roof mode only: the two plan
edges incident to the camera's corner keep their wall faces only up to a height,
so the near walls become a low stub and the oblique camera sees the floor over
them, while the far walls stay full height as the backdrop. The cut lands on
face boundaries, not exactly at the requested height -- and it is conservative:
a near-wall face is dropped when its **highest vertex** is above the height, so
a face straddling the cut goes with the upper part. That is required, not just
cautious: the Infinigen wall mesh is triangulated with long diagonal triangles
spanning the full wall height (measured on the office shell
``/Shell/office_0_0_wall/Circle_002``: faces 0-3 run z 0.05->2.74 with centroid
z 0.95), and a centroid test keeps them below a 1.25 m cut -- they survive as
full-height slivers that render as long blades at grazing angle. The achieved
height recorded is the max vertex z over the kept near-wall faces.

Every kept prim keeps its original path and name, so whatever the render worker
binds by prim name still finds it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Roles dropped whole, independent of the cut plane.
DROPPED_ROLES = ("ceiling", "ceiling_skirting", "exterior")
# Roles clipped by the cut plane.
CLIPPED_ROLES = ("wall", "skirtingboard", "door", "window", "shell_other")
# The floor is never clipped: cutting it would leave the room floating.
KEPT_WHOLE_ROLES = ("floor",)
# How far inside the plan's edge the cut plane sits. The walls are authored as
# their *inner* surfaces on the polygon boundary (measured on the office shell:
# the near wall's face centroids sit at y = 0.017 while the plan edge is y = 0),
# so a plane exactly on the edge keeps every wall face and removes nothing. One
# wall thickness of inset takes the near wall and the openings in it, and trims
# the same depth off the two side walls where the cut is made.
CUT_INSET_M = 0.12
# How close a face centroid must sit to a wall-clip edge to count as part of
# that near wall. Only shell meshes are traversed -- no furniture is in this
# stage -- so the radius only needs to cover the authored inset of the inner
# wall surface (measured ~0.02 m on the office shell) plus skirting and window
# frames; 0.35 m keeps margin for those without reaching any other prim.
WALL_CLIP_NEAR_M = 0.35


def _point_segment_distance(px: float, py: float, edge) -> float:
    x1, y1, x2, y2 = edge
    ex, ey = x2 - x1, y2 - y1
    length = math.hypot(ex, ey)
    if length == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * ex + (py - y1) * ey) / (length * length)))
    return math.hypot(px - (x1 + t * ex), py - (y1 + t * ey))


def wall_clip_edges(polygon, corner: str) -> list[list[float]]:
    """The two plan edges incident to the polygon vertex nearest ``corner``.

    The corner names a bounding-box corner; the nearest polygon vertex is used so
    an L-shaped plan clips the edges of its actual outline rather than of its
    bbox. ``corner="all"`` returns every polygon edge instead -- the near-top
    dollhouse view has no near side, so every wall comes down to the stub.
    Returns ``[[x1, y1, x2, y2], ...]``.
    """
    if corner not in ("ne", "nw", "se", "sw", "all"):
        raise ValueError(f"WALL_CLIP_CORNER_INVALID({corner})")
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    if corner == "all":
        return [[xs[i], ys[i], xs[(i + 1) % len(xs)], ys[(i + 1) % len(xs)]]
                for i in range(len(xs))]
    target_x = max(xs) if "e" in corner else min(xs)
    target_y = max(ys) if "n" in corner else min(ys)
    nearest = min(range(len(polygon)),
                  key=lambda i: math.hypot(xs[i] - target_x, ys[i] - target_y))
    count = len(polygon)
    prev_v = (xs[(nearest - 1) % count], ys[(nearest - 1) % count])
    vertex = (xs[nearest], ys[nearest])
    next_v = (xs[(nearest + 1) % count], ys[(nearest + 1) % count])
    return [[prev_v[0], prev_v[1], vertex[0], vertex[1]],
            [vertex[0], vertex[1], next_v[0], next_v[1]]]


def wall_clip_for(room: dict, corner: str, height_m: float) -> dict:
    """The wall_clip dict for ``build_cutaway_shell``: corner, height, edges."""
    return {"corner": corner, "height_m": float(height_m),
            "edges": wall_clip_edges(room["polygon_xy"], corner)}


def role_of(prim_path: str) -> str:
    lowered = str(prim_path).lower()
    for hint, role in (
        ("exterior", "exterior"),
        ("skirtingboard_ceiling", "ceiling_skirting"),
        ("skirtingboard", "skirtingboard"),
        ("wall", "wall"),
        ("floor", "floor"),
        ("ceiling", "ceiling"),
        ("door", "door"),
        ("window", "window"),
    ):
        if hint in lowered:
            return role
    return "shell_other"


def choose_cut(room: dict) -> dict:
    """Which long side to open, and where the camera then stands.

    The room's longer axis decides: the wall removed is a long wall, and the
    camera looks across the room's short axis, level with the floor. Measured on
    the office plan (4.906 x 3.686 m) that opens the south wall and looks north,
    which is the side the task desk faces.
    """
    polygon = np.asarray(room["polygon_xy"], dtype=np.float64)
    min_x, min_y = polygon[:, 0].min(), polygon[:, 1].min()
    max_x, max_y = polygon[:, 0].max(), polygon[:, 1].max()
    centre = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    if (max_x - min_x) >= (max_y - min_y):
        return {
            "open_side": "south",
            "axis": "y",
            "keep_above": float(min_y),
            "eye_xy": (centre[0], min_y),
            "look_xy": (centre[0], centre[1]),
            "room_width_m": float(max_x - min_x),
        }
    return {
        "open_side": "west",
        "axis": "x",
        "keep_above": float(min_x),
        "eye_xy": (min_x, centre[1]),
        "look_xy": (centre[0], centre[1]),
        "room_width_m": float(max_y - min_y),
    }


def level_camera(room: dict, cut: dict, eye_height_m: float = 1.45,
                 focal_length_mm: float = 18.0, aperture_mm: float = 20.955,
                 width: int = 1600, height: int = 900,
                 stand_off_m: float | None = None) -> dict:
    """A level camera outside the opened side, framed on the room's width.

    ``eye`` and ``target`` share a z, so the view is horizontal: SPEC-015's
    lesson was that an unlevel camera is what made the first attempt read as odd.
    The stand-off follows from the lens and the room's width, so every room is
    framed the same way rather than each one guessed.
    """
    import math

    hfov = 2.0 * math.atan(aperture_mm / (2.0 * focal_length_mm))
    if stand_off_m is None:
        # Fit the room's width plus a margin, measured across the view.
        half_needed = cut["room_width_m"] / 2.0 * 1.08
        stand_off_m = half_needed / math.tan(hfov / 2.0)
    axis = cut["axis"]
    eye_xy, look_xy = cut["eye_xy"], cut["look_xy"]
    if axis == "y":
        offset = (0.0, -stand_off_m)
    else:
        offset = (-stand_off_m, 0.0)
    return {
        "eye": [eye_xy[0] + offset[0], eye_xy[1] + offset[1], eye_height_m],
        "target": [look_xy[0], look_xy[1], eye_height_m],
        "focal_length_mm": focal_length_mm,
        "horizontal_aperture_mm": aperture_mm,
        "vertical_aperture_mm": aperture_mm * height / width,
        "resolution": [width, height],
        "stand_off_m": stand_off_m,
        "eye_height_m": eye_height_m,
        "level": True,
    }


def build_cutaway_shell(shell_path: Path, out_path: Path, cut: dict | None = None,
                        mode: str = "cutaway", wall_clip: dict | None = None) -> dict:
    """Write the cut shell to ``out_path``; return a record of what was removed.

    The hierarchy is **cloned with** ``Sdf.CopySpec`` rather than rebuilt, so the
    material bindings and shader asset paths come along untouched; only the face
    lists of the clipped meshes change. Points stay in the prim's own local frame
    and no transform is altered, so a mesh that is kept is the same mesh.

    The worker validates a referenced shell before rendering it, and that check is
    deliberately strict: a default prim must exist, a sidecar must sit beside the
    USD with a matching ``usd_sha256`` and an integer ``openings_measured``, and
    any world-bounds keys it carries must match the evaluated geometry. A cutaway
    copy is a figure-only artifact, so its sidecar is written to describe **the
    copy**: its own digest, the same opening count, and no source/measurement
    paths (which would otherwise have to exist beside it). The record this
    function returns is what the figure's provenance cites.

    ``mode="roof"`` skips the cut plane entirely (``cut`` is then unused): walls,
    skirting and openings are kept at full height and only ``DROPPED_ROLES`` are
    removed. ``mode="cutaway"`` requires ``cut``. ``wall_clip`` (roof mode only,
    built with ``wall_clip_for``) additionally drops wall faces above
    ``height_m`` whose centroid sits near the two edges incident to ``corner``.
    """
    if mode not in ("cutaway", "roof"):
        raise ValueError(f"CUTAWAY_MODE_UNKNOWN({mode})")
    if mode == "cutaway" and cut is None:
        raise ValueError("CUTAWAY_CUT_MISSING(mode=cutaway needs a cut plane)")
    if wall_clip is not None:
        if mode != "roof":
            raise ValueError("CUTAWAY_WALL_CLIP_MODE(wall_clip requires mode=roof; "
                             "a cutaway already removes the near wall whole)")
        if "edges" not in wall_clip or "height_m" not in wall_clip:
            raise ValueError("CUTAWAY_WALL_CLIP_SHAPE(needs corner, height_m and edges; "
                             "build the dict with wall_clip_for)")
    from pxr import Usd, UsdGeom, Sdf

    source = Usd.Stage.Open(str(shell_path))
    if source is None:
        raise SystemExit(f"CUTAWAY_SHELL_UNREADABLE({shell_path})")
    root = source.GetDefaultPrim()
    if root is None:
        raise SystemExit(f"CUTAWAY_SHELL_UNREADABLE(no default prim in {shell_path})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        out_path.unlink()
    dest = Usd.Stage.CreateNew(str(out_path))
    Sdf.CopySpec(source.GetRootLayer(), root.GetPath(), dest.GetRootLayer(), root.GetPath())
    dest.SetDefaultPrim(dest.GetPrimAtPath(root.GetPath()))
    UsdGeom.SetStageUpAxis(dest, UsdGeom.GetStageUpAxis(source))
    UsdGeom.SetStageMetersPerUnit(dest, UsdGeom.GetStageMetersPerUnit(source))

    # Shader inputs keep their relative asset paths, which resolve against this
    # layer's directory now, so the textures have to sit beside the copy.
    textures = shell_path.parent / "textures"
    if textures.is_dir():
        link = out_path.parent / "textures"
        if not link.exists():
            link.symlink_to(textures.resolve(), target_is_directory=True)

    kept: list[dict] = []
    removed: list[dict] = []
    cache = UsdGeom.XformCache()
    clip_edges = wall_clip["edges"] if wall_clip is not None else None
    clip_height = float(wall_clip["height_m"]) if wall_clip is not None else None
    # Achieved clip height, recorded from the geometry: the max vertex z over the
    # kept near-wall faces, plus the lowest top of the dropped ones. Faces are
    # the granularity, so the cut lands on a face boundary, not on height_m --
    # provenance must say where it actually landed.
    clip_kept_near_z: list[float] = []
    clip_dropped_near_top_z: list[float] = []
    # Two passes: deciding needs the geometry, and applying removes prims, which
    # invalidates every other prim the traversal handed out.
    decisions: list[dict] = []
    for prim in dest.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.size == 0 or counts.size == 0:
            continue
        # Row-vector convention: p_world = p_local @ M. Reading it the other way
        # put every mesh at the origin once already; see extract_scene.py.
        xform = np.asarray(cache.GetLocalToWorldTransform(prim))
        world = points @ xform[:3, :3] + xform[3, :3]
        faces, cursor = [], 0
        for count in counts:
            faces.append(indices[cursor:cursor + int(count)])
            cursor += int(count)
        centroids = np.array([world[face].mean(axis=0) for face in faces])
        face_max_z = np.array([world[face][:, 2].max() for face in faces])
        role = role_of(prim.GetPath())
        if role in DROPPED_ROLES:
            keep = np.zeros(len(faces), dtype=bool)
            action = "dropped"
        elif clip_edges is not None and role in CLIPPED_ROLES:
            near = np.array([
                min(_point_segment_distance(c[0], c[1], edge) for edge in clip_edges)
                < WALL_CLIP_NEAR_M for c in centroids])
            # The height test is on the face's highest vertex, not its centroid:
            # Infinigen triangulates a wall with diagonal triangles spanning the
            # full height, and their centroids sit below the cut (see the module
            # docstring) -- a centroid test keeps them as full-height blades.
            # Straddling faces are dropped with the upper part; for a dollhouse
            # stub a conservative cut is the right one.
            keep = ~(near & (face_max_z > clip_height))
            action = "height_clipped" if not keep.all() else "kept"
            clip_kept_near_z.extend(
                float(z) for z, n, k in zip(face_max_z, near, keep) if n and k)
            clip_dropped_near_top_z.extend(
                float(z) for z, n, k in zip(face_max_z, near, keep) if n and not k)
        elif mode == "roof" or role in KEPT_WHOLE_ROLES:
            keep = np.ones(len(faces), dtype=bool)
            action = "kept"
        else:
            # Keep the far side of the cut plane, inset by one wall thickness so
            # the near wall's inner surface is actually on the removed side.
            axis = 1 if cut["axis"] == "y" else 0
            keep = centroids[:, axis] >= cut["keep_above"] + CUT_INSET_M
            action = "clipped"
        record = {
            "prim_path": str(prim.GetPath()),
            "role": role,
            "faces_in": len(faces),
            "faces_kept": int(keep.sum()),
            "action": action,
        }
        if not keep.any():
            removed.append(record)
            decisions.append({"path": prim.GetPath(), "drop": True})
            continue
        record["points_in"] = int(len(points))
        record["points_kept"] = int(len(points))
        kept.append(record)
        face_index = np.nonzero(keep)[0]
        # The export splits a mesh's faces across materials with GeomSubsets, whose
        # `indices` are **face indices**. Renumbering the faces without renumbering
        # these lists moves every material onto the wrong faces -- measured on this
        # shell it painted the office's white plaster wall with the facade's green
        # on the faces that had merely shifted. So each subset is remapped through
        # the same keep mask, and a subset left empty is removed.
        subsets = []
        for child in prim.GetChildren():
            if not child.IsA(UsdGeom.Subset):
                continue
            old = np.asarray(UsdGeom.Subset(child).GetIndicesAttr().Get(), dtype=np.int64)
            position = {int(old_face): new_face for new_face, old_face in enumerate(face_index)}
            remapped = sorted(position[int(i)] for i in old if int(i) in position)
            subsets.append({"path": child.GetPath(), "indices": remapped,
                            "faces_in": int(len(old)), "faces_kept": len(remapped)})
            record.setdefault("subsets", []).append(
                {"name": child.GetPath().name, "faces_in": int(len(old)),
                 "faces_kept": len(remapped)})
        decisions.append({
            "path": prim.GetPath(),
            "drop": False,
            "counts": [len(faces[i]) for i in face_index],
            "indices": [int(i) for i in face_index for i in faces[i]],
            "subsets": subsets,
        })

    for decision in decisions:
        if decision["drop"]:
            dest.RemovePrim(decision["path"])
            continue
        # Only the face list is rewritten. The points and every primvar stay as
        # the export wrote them, so the vertex indices remain valid and a mesh
        # that is kept is bit-for-bit the built one minus some faces -- no
        # renumbering, and nothing that could silently shift a UV seam.
        mesh = UsdGeom.Mesh(dest.GetPrimAtPath(decision["path"]))
        mesh.GetFaceVertexCountsAttr().Set(decision["counts"])
        mesh.GetFaceVertexIndicesAttr().Set(decision["indices"])
        for subset in decision["subsets"]:
            if not subset["indices"]:
                dest.RemovePrim(subset["path"])
                continue
            UsdGeom.Subset(dest.GetPrimAtPath(subset["path"])).GetIndicesAttr().Set(
                subset["indices"])

    dest.GetRootLayer().Save()
    sidecar = out_path.with_suffix(".export.json")
    original_sidecar = shell_path.with_suffix(".export.json")
    record_source = json.loads(original_sidecar.read_text()) if original_sidecar.is_file() else {}
    import hashlib
    note = {
        "cutaway": "a cutaway copy for the paper figure: the ceiling, the outer facade and the "
                   "near long wall are removed so a level camera can see in. ",
        "roof": ("a roofless copy for the dollhouse figure: the ceiling, the ceiling skirting "
                 "and the outer facade are removed so a top-down oblique camera can see in. "
                 + ("The two walls nearest the camera corner are clipped to a low stub; "
                    "every other wall stands at full height. " if wall_clip is not None else
                    "Every wall stands at full height. ")),
    }[mode] + ("Written to satisfy the shell validator honestly -- it describes this file, "
               "not the built shell.")
    sidecar_payload = {
        "figure_only": True,
        "mode": mode,
        "derived_from": str(shell_path),
        "derived_from_usd_sha256": hashlib.sha256(shell_path.read_bytes()).hexdigest(),
        "usd_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "openings_measured": int(record_source.get("openings_measured", 0)),
        # These two describe where the geometry came from, and the copy's geometry
        # does come from exactly there. Carried verbatim rather than invented.
        "source_sha256": record_source.get("source_sha256"),
        "materials": record_source.get("materials"),
        "lighting_policy": record_source.get("lighting_policy"),
        "note": note,
    }
    if wall_clip is not None:
        sidecar_payload["wall_clip"] = {
            "corner": wall_clip.get("corner"),
            "height_m": float(wall_clip["height_m"]),
            "edges": wall_clip["edges"],
        }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    result = {
        "source_shell": str(shell_path),
        "cutaway_shell": str(out_path),
        "cutaway_sidecar": str(sidecar),
        "mode": mode,
        "open_side": cut["open_side"] if cut else None,
        "cut_axis": cut["axis"] if cut else None,
        "cut_at_m": cut["keep_above"] if cut else None,
        "dropped_roles": list(DROPPED_ROLES),
        "prims": kept,
        "prims_removed_entirely": removed,
        "faces_in": sum(r["faces_in"] for r in kept + removed),
        "faces_kept": sum(r.get("faces_kept", 0) for r in kept),
    }
    if wall_clip is not None:
        result["wall_clip"] = {
            "corner": wall_clip.get("corner"),
            "height_m": float(wall_clip["height_m"]),
            "edges": wall_clip["edges"],
            "near_radius_m": WALL_CLIP_NEAR_M,
            # Face granularity and a max-vertex test: this is where the clip
            # actually landed, not the request.
            "achieved_max_kept_z_m": max(clip_kept_near_z) if clip_kept_near_z else None,
            "dropped_min_top_z_m": (min(clip_dropped_near_top_z)
                                    if clip_dropped_near_top_z else None),
        }
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("cutaway", "roof"), default="cutaway",
                        help="cutaway also clips the near long wall; roof removes only the "
                             "ceiling and facade and keeps every wall at full height")
    parser.add_argument("--wall-clip-height", type=float, default=None,
                        help="roof mode only: clip the two walls nearest --wall-clip-corner "
                             "to this height so an oblique camera sees over them")
    parser.add_argument("--wall-clip-corner", choices=("ne", "nw", "se", "sw"), default=None,
                        help="roof mode only: corner whose two adjacent walls are clipped")
    args = parser.parse_args()
    delivery = args.delivery.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")
    if args.wall_clip_height is not None or args.wall_clip_corner is not None:
        if args.mode != "roof":
            parser.error("--wall-clip-* is only valid with --mode roof")
        if args.wall_clip_height is None or args.wall_clip_corner is None:
            parser.error("--wall-clip-height and --wall-clip-corner must be given together")
    manifest = json.loads((delivery / "delivery_manifest.json").read_text())
    request = json.loads((delivery / manifest["isaac_request_path"]).read_text())
    shell = (delivery / manifest["isaac_request_path"]).parent / request["room"]["shell_uri"]
    cut = choose_cut(request["room"]) if args.mode == "cutaway" else None
    wall_clip = (wall_clip_for(request["room"], args.wall_clip_corner, args.wall_clip_height)
                 if args.wall_clip_height is not None else None)
    out.mkdir(parents=True, exist_ok=True)
    record = build_cutaway_shell(shell.resolve(), out / f"shell_{args.mode}.usdc", cut,
                                 mode=args.mode, wall_clip=wall_clip)
    if cut is not None:
        record["camera"] = level_camera(request["room"], cut)
    (out / "cutaway.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cutaway_shell": record["cutaway_shell"], "mode": record["mode"],
                      "open_side": record["open_side"],
                      "faces_in": record["faces_in"], "faces_kept": record["faces_kept"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
