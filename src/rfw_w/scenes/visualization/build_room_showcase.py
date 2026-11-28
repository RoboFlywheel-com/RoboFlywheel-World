#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compose a traceable one-room paper preview from a delivery and a render report.

The input may be a production Isaac render or a figure-only rerender made by
render_room_preview.py. Raw RGB/depth are read, never modified; only the displayed
depth colour mapping is applied. The output directory must be fresh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

MM = 1 / 25.4
ROOM = "#256d7b"
PLAN = "#546570"
TARGET = "#d45c3a"
SUPPORT = "#ba914d"
OTHER = "#9caab0"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_shell_measurement(delivery: Path) -> tuple[Path, Path]:
    for plan_path in sorted(delivery.glob("shell_build/**/room_plan.json")):
        measurement_path = plan_path.with_name("measurement.json")
        if measurement_path.is_file() and read(measurement_path).get("plan_hash") == read(plan_path).get("hash"):
            return plan_path, measurement_path
    raise ValueError("SHOWCASE_SHELL_PLAN_MISMATCH(missing paired plan/measurement)")


def yaw(quat: list[float]) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_matrix(quat: list[float]) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def footprint(instance: dict) -> list[tuple[float, float]]:
    lo, hi = instance["bounds_min_m"], instance["bounds_max_m"]
    tx, ty = instance["world_transform"]["translation"][:2]
    a = yaw(instance["world_transform"]["rotation_wxyz"])
    ca, sa = math.cos(a), math.sin(a)
    return [(tx + x * ca - y * sa, ty + x * sa + y * ca)
            for x, y in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]


def draw_plan(ax, scene: dict, plan: dict, measurement: dict, target_id: str, support_id: str,
              legend: bool = True) -> None:
    polygon = plan["rooms"][0]["polygon_xy"]
    ax.add_patch(Polygon(polygon, facecolor="#f3f5f4", edgecolor=PLAN,
                         linewidth=1.2, linestyle="--", label="RoomPlan"))
    first = True
    for wall in measurement["walls"]:
        if "point" not in wall or "end" not in wall:
            continue
        x1, y1 = wall["point"][:2]
        x2, y2 = wall["end"][:2]
        ax.plot([x1, x2], [y1, y2], color=ROOM, lw=2.0,
                label="Built wall" if first else None, zorder=3)
        first = False
    for opening in plan["openings"]:
        segment = opening.get("wall_segment_xy")
        if segment:
            ax.plot([p[0] for p in segment], [p[1] for p in segment],
                    color="#fffdf9" if opening["kind"] == "door" else "#70afc0",
                    lw=4.0, solid_capstyle="butt", zorder=4)
    for item in scene["instances"]:
        entity = item["entity_id"]
        if entity in (target_id, support_id):
            colour = TARGET if entity == target_id else SUPPORT
            ax.add_patch(Polygon(footprint(item), facecolor=colour, alpha=0.70,
                                 edgecolor="#29343b", lw=0.5, zorder=5))
            tx, ty = item["world_transform"]["translation"][:2]
            # Target above its centre, support below: a cup on a desk would otherwise
            # put both labels on the same pixel.
            ax.annotate(entity, (tx, ty), xytext=(0, 5 if entity == target_id else -9),
                        textcoords="offset points", ha="center",
                        fontsize=6.4, weight="bold", color="#25313a", zorder=7)
        elif item["bounds_min_m"][2] < 1.0:
            ax.add_patch(Polygon(footprint(item), facecolor=OTHER, alpha=0.28,
                                 edgecolor="#8c999d", lw=0.35, zorder=2))
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    ax.set_ylim(min(ys) - 0.35, max(ys) + 0.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # The legend is drawn once, under the first row: repeating it under every plan
    # put it on top of the figure's own footnote.
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.025), ncol=2,
                  frameon=False, fontsize=6.0, handlelength=1.6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path,
                        help="single-room mode: one delivery. Use --selection for the "
                             "three-row alignment/diversity figure.")
    parser.add_argument("--selection", type=Path,
                        help="three-row mode: a selection.json naming three rows of two seeds each")
    parser.add_argument("--render-dir", type=Path,
                        help="figure-only render root (containing isaac/report.json); default: delivery final Isaac artifact")
    parser.add_argument("--camera", default="paper_room_level")
    parser.add_argument("--detail-camera", default="paper_desk_detail")
    parser.add_argument("--depth-min-m", type=float, default=0.0)
    parser.add_argument("--depth-max-m", type=float, default=6.0,
                        help="fixed display colourbar maximum in metres; use the same value for compared rooms")
    parser.add_argument("--depth-cmap", choices=("turbo_r", "viridis", "plasma", "inferno", "gray"),
                        default="turbo_r", help="display mapping only; raw float depth is unchanged")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.depth_min_m < 0 or args.depth_max_m <= args.depth_min_m:
        parser.error("depth display range must satisfy 0 <= min < max")
    if args.selection and args.delivery:
        parser.error("--selection and --delivery are two different figures; give one")
    if args.selection:
        return build_selection(read(args.selection.resolve()), args.output_dir.resolve(),
                               args.camera, args.depth_min_m, args.depth_max_m, args.depth_cmap)
    if not args.delivery:
        parser.error("one of --delivery or --selection is required")
    delivery = args.delivery.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")
    manifest = read(delivery / "delivery_manifest.json")
    scene = read(delivery / "scene_ir.json")
    shell = read(delivery / "shell_reference.json")
    plan_path, measure_path = find_shell_measurement(delivery)
    plan, measurement = read(plan_path), read(measure_path)
    if shell["scene_version"] != manifest["scene_version"]:
        raise ValueError("SHOWCASE_SCENE_VERSION_MISMATCH(shell)")
    if scene["spec_hash"] != manifest["randomization"]["invariant_spec_hash"]:
        # SceneIR's spec hash may be the randomized spec in older deliveries; record
        # that distinction instead of assuming it proves the frozen task closure.
        spec_hash_match = False
    else:
        spec_hash_match = True
    render_root = args.render_dir.resolve() if args.render_dir else delivery
    image_dir = render_root / "isaac" if args.render_dir else delivery / manifest["isaac_artifact_dir"]
    figure_render_provenance = None
    if args.render_dir:
        figure_render_provenance = read(render_root / "provenance.json")
        if (Path(figure_render_provenance["source_delivery"]).resolve() != delivery or
                figure_render_provenance["source_scene_version"] != manifest["scene_version"]):
            raise ValueError("SHOWCASE_SCENE_VERSION_MISMATCH(figure render provenance)")
    report_path = image_dir / "report.json"
    report = read(report_path)
    if figure_render_provenance and figure_render_provenance["report_sha256"] != sha(report_path):
        raise ValueError("SHOWCASE_PROVENANCE_MISSING(render report digest mismatch)")
    if report["scene_version"] != manifest["scene_version"]:
        raise ValueError("SHOWCASE_SCENE_VERSION_MISMATCH(render)")
    if report.get("status") != "passed":
        raise ValueError("SHOWCASE_RENDER_MISSING(worker not passed)")
    captures = {c["camera_id"]: c for c in report["captures"]}
    if args.camera not in captures:
        raise ValueError(f"SHOWCASE_RENDER_MISSING({args.camera})")
    capture = captures[args.camera]
    if args.detail_camera not in captures or args.detail_camera == args.camera:
        raise ValueError(f"SHOWCASE_RENDER_MISSING({args.detail_camera})")
    detail_capture = captures[args.detail_camera]
    if capture["depth_convention"] != "camera_z_meters":
        raise ValueError("SHOWCASE_DEPTH_CONVENTION_MISMATCH")
    rgb_path = image_dir / capture["rgb_path"]
    depth_path = image_dir / capture["depth_path"]
    detail_path = image_dir / detail_capture["rgb_path"]
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    detail_rgb = np.asarray(Image.open(detail_path).convert("RGB"))
    depth = np.load(depth_path, allow_pickle=False)
    if depth.shape != rgb.shape[:2]:
        raise ValueError("SHOWCASE_DEPTH_SHAPE_MISMATCH")
    valid = np.isfinite(depth) & (depth > 0)
    if valid.mean() < 0.1:
        raise ValueError("SHOWCASE_DEPTH_INVALID")
    target_id = manifest["plan"]["target_id"]
    support_id = manifest["plan"]["support_id"]
    entity_ids = {i["entity_id"] for i in scene["instances"]}
    if target_id not in entity_ids or support_id not in entity_ids:
        raise ValueError("SHOWCASE_ENTITY_MISSING(target/support)")
    index = {int(key): value for key, value in report.get("instance_id_map", {}).items()}
    for camera_capture in (capture, detail_capture):
        visible = {index.get(int(value)) for value in camera_capture.get("visible_instance_ids", [])}
        if target_id not in visible or support_id not in visible:
            raise ValueError(f"SHOWCASE_ENTITY_NOT_VISIBLE({camera_capture['camera_id']},target/support)")
    out.mkdir(parents=True, exist_ok=True)
    panels = out / "panels"
    panels.mkdir()
    shutil.copy2(rgb_path, panels / "rgb.png")
    shutil.copy2(detail_path, panels / "detail_rgb.png")
    shutil.copy2(depth_path, panels / "depth_m.npy")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(183 * MM, 118 * MM), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.035, right=0.985,
                           bottom=0.13, top=0.89, wspace=0.12, hspace=0.27)
    ax_plan = fig.add_subplot(grid[0, 0])
    ax_rgb = fig.add_subplot(grid[0, 1])
    ax_detail = fig.add_subplot(grid[1, 0])
    ax_depth = fig.add_subplot(grid[1, 1])
    draw_plan(ax_plan, scene, plan, measurement, target_id, support_id)
    ax_rgb.imshow(rgb)
    ax_rgb.axis("off")
    ax_detail.imshow(detail_rgb)
    ax_detail.axis("off")
    depth_display = np.ma.masked_where(~valid, depth)
    cmap = plt.get_cmap(args.depth_cmap).copy()
    cmap.set_bad("#202832")
    ax_depth.imshow(depth_display, cmap=cmap, vmin=args.depth_min_m, vmax=args.depth_max_m,
                    interpolation="nearest")
    ax_depth.axis("off")
    for ax, title in ((ax_plan, "a  Plan / built shell"),
                      (ax_rgb, "b  Full-room RGB"),
                      (ax_detail, "c  Desk detail (same scene)"),
                      (ax_depth, "d  Camera-z depth")):
        ax.set_title(title, loc="left", fontsize=7.5, pad=5)
    bar = fig.colorbar(ScalarMappable(norm=colors.Normalize(args.depth_min_m, args.depth_max_m), cmap=cmap), ax=ax_depth,
                       orientation="horizontal", fraction=0.07, pad=0.04, aspect=25)
    bar.set_label("Depth (m)", fontsize=6.5, labelpad=2)
    bar.ax.tick_params(labelsize=6, length=2)
    fig.text(0.025, 0.95, f"{manifest['scene_id']}  |  room seed {manifest['randomization'].get('room_seed')}",
             fontsize=9, weight="bold", color="#243943", va="top")
    fig.text(0.035, 0.04, "Single-room preview · authored rigid geometry · depth from the full-room camera · readiness follows the delivery ledger",
             fontsize=6.1, color="#64737b")
    pdf = out / "figure_room_preview.pdf"
    png = out / "figure_room_preview.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    provenance = {
        "delivery": str(delivery), "scene_version": manifest["scene_version"],
        "camera": args.camera, "detail_camera": args.detail_camera,
        "capture_state": report.get("capture_state"),
        "room_seed": manifest["randomization"].get("room_seed"),
        "target_id": target_id, "support_id": support_id,
        "files": {"manifest": sha(delivery / "delivery_manifest.json"),
                  "scene_ir": sha(delivery / "scene_ir.json"), "room_plan": sha(plan_path),
                  "shell_measurement": sha(measure_path), "shell_reference": sha(delivery / "shell_reference.json"),
                  "render_report": sha(report_path), "rgb": sha(rgb_path),
                  "detail_rgb": sha(detail_path), "depth": sha(depth_path)},
        "depth": {"convention": "camera_z_meters", "display_range_m": [args.depth_min_m, args.depth_max_m],
                  "display_colormap": args.depth_cmap,
                  "valid_fraction": float(valid.mean()), "raw_shape": list(depth.shape)},
        "checks": {"source_scene_version_matched": True, "source_worker_passed": True,
                   "spec_hash_matches_invariant_hash": spec_hash_match,
                   "target_in_scene_ir": True, "support_in_scene_ir": True,
                   "target_and_support_visible_in_both_cameras": True},
        "source_resolution": manifest["resolution"],
    }
    if figure_render_provenance:
        provenance["files"]["figure_render_provenance"] = sha(render_root / "provenance.json")
        provenance["render_dir"] = str(render_root)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pdf": str(pdf), "png": str(png),
                      "depth_valid_fraction": provenance["depth"]["valid_fraction"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# SPEC-013 main figure: three rows (office / kitchen / bedroom), each row one
# frozen task shown across two independent room seeds.
#
# The single-room mode above answers "what does one room look like". This mode
# answers the question the paper actually asks: when the task core is frozen,
# what changes about the room, and does the task relation still land on it. The
# two are different figures and neither replaces the other, which is why this is
# a second mode rather than a rewrite.
# ---------------------------------------------------------------------------

SEED_A = "#256d7b"
SEED_B = "#a4552f"


def camera_of(render_dir: Path, camera_id: str) -> dict:
    """The camera the figure render actually used, from the request it sent."""
    request = read(render_dir / "render_request.json")
    for record in request["cameras"]:
        if record["camera_id"] == camera_id:
            return record
    raise ValueError(f"SHOWCASE_RENDER_MISSING({camera_id})")


def project(camera: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points -> pixels, for the same GL convention the worker uses."""
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = np.stack([right, up, -fwd], axis=0)
    width, height = camera["resolution"]
    fx = width * camera["focal_length_mm"] / camera["horizontal_aperture_mm"]
    fy = height * camera["focal_length_mm"] / camera["vertical_aperture_mm"]
    cam = (np.asarray(points, dtype=np.float64) - eye) @ rot.T
    z = -cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2.0 + fx * cam[:, 0] / z
        v = height / 2.0 - fy * cam[:, 1] / z
    return np.stack([u, v], axis=-1), z


def outline_mask(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask, by 4-neighbour erosion."""
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def draw_rgb_panel(ax, sample: dict, camera: dict, target_id: str, support_id: str,
                   seed_label: str, accent: str) -> None:
    """One full-room RGB with the task objects and the door marked on it.

    The marks come from the worker's own instance-id image, not from a guess about
    where the object is: a target the worker did not label cannot be marked, and
    the check that it is present is the same fact.
    """
    rgb = sample["rgb"]
    ids = sample["ids"]
    id_map = {int(k): v for k, v in sample["capture"]["instance_id_map"].items()}
    ax.imshow(rgb)
    # Target and support are labelled on opposite sides of their own outlines so
    # a cup sitting on a desk cannot have its label swallowed by the desk's.
    for entity, colour, offset in ((target_id, TARGET, (6, 10)), (support_id, SUPPORT, (6, -14))):
        code = next((k for k, v in id_map.items() if v == entity), None)
        if code is None:
            continue
        edge = outline_mask(ids == code)
        overlay = np.zeros_like(rgb)
        overlay[edge] = np.array([int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)])
        ax.imshow(np.where(edge[..., None], overlay, rgb))
        ys, xs = np.nonzero(edge)
        if len(xs):
            ax.annotate(entity, (xs.mean(), ys.mean()), xytext=offset,
                        textcoords="offset points", fontsize=6.0, weight="bold",
                        color="#ffffff", zorder=6, ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc=colour, ec="none", alpha=0.9))
    cut = sample.get("cutaway")
    for opening in sample["scene"]["room"]["openings"]:
        if opening["kind"] != "door":
            continue
        cx, cy = opening["center_xy"]
        if cut is not None:
            # A door in the wall this figure opened is not drawn, so marking it
            # would label furniture: on bedroom seed 37 the marker landed on the
            # wardrobe standing where the doorway used to be. Only doors that
            # survived the cut are marked.
            axis = 1 if cut["cut_axis"] == "y" else 0
            if (cx, cy)[axis] < cut["cut_at_m"] + 0.3:
                continue
        point = np.array([[cx, cy, opening["sill_z_m"] + opening["height_m"] / 2.0]])
        uv, z = project(camera, point)
        if z[0] <= 0 or not (0 <= uv[0, 0] < rgb.shape[1]):
            continue
        # Occlusion test against the render's own depth. A door that exists but
        # stands behind a wardrobe would otherwise be labelled on the wardrobe --
        # measured on bedroom seed 37, where the marker sat on the cabinet across
        # the doorway and read as if the cabinet were the door.
        row = int(round(uv[0, 1]))
        column = int(round(uv[0, 0]))
        if 0 <= row < sample["depth"].shape[0] and 0 <= column < sample["depth"].shape[1]:
            seen = float(sample["depth"][row, column])
            if not np.isfinite(seen) or seen <= 0 or abs(seen - z[0]) > 0.30:
                continue
        ax.plot(uv[0, 0], uv[0, 1], marker="o", ms=4.0, mfc="none", mec="#ffffff",
                mew=1.2, zorder=6)
        ax.annotate("door", (uv[0, 0], uv[0, 1]), xytext=(5, 4), textcoords="offset points",
                    fontsize=6.0, color="#ffffff", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="#2b3a42", ec="none", alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{seed_label}   room seed {sample['manifest']['randomization'].get('room_seed')}",
                 loc="left", fontsize=6.8, color=accent, pad=3.5)


def load_sample(delivery: Path, render_dir: Path, camera_id: str) -> dict:
    manifest = read(delivery / "delivery_manifest.json")
    scene = read(delivery / "scene_ir.json")
    report = read(render_dir / "isaac" / "report.json")
    if report["scene_version"] != manifest["scene_version"]:
        raise ValueError("SHOWCASE_SCENE_VERSION_MISMATCH(render)")
    if report.get("status") != "passed":
        raise ValueError("SHOWCASE_RENDER_MISSING(worker not passed)")
    capture = next((c for c in report["captures"] if c["camera_id"] == camera_id), None)
    if capture is None:
        raise ValueError(f"SHOWCASE_RENDER_MISSING({camera_id})")
    if capture["depth_convention"] != "camera_z_meters":
        raise ValueError("SHOWCASE_DEPTH_CONVENTION_MISMATCH")
    image_dir = render_dir / "isaac"
    provenance = read(render_dir / "provenance.json")
    if (Path(provenance["source_delivery"]).resolve() != delivery.resolve()
            or provenance["source_scene_version"] != manifest["scene_version"]):
        raise ValueError("SHOWCASE_SCENE_VERSION_MISMATCH(figure render provenance)")
    return {
        "delivery": delivery,
        "render_dir": render_dir,
        "manifest": manifest,
        "scene": scene,
        "plan": read(find_shell_measurement(delivery)[0]),
        "measurement": read(find_shell_measurement(delivery)[1]),
        "report": report,
        "capture": capture,
        "rgb": np.asarray(Image.open(image_dir / capture["rgb_path"]).convert("RGB")),
        "depth": np.load(image_dir / capture["depth_path"], allow_pickle=False),
        "ids": np.load(image_dir / capture["instance_id_path"], allow_pickle=False),
        "cutaway": read(render_dir / "provenance.json").get("cutaway"),
    }


def core_pose(sample: dict, entity: str) -> tuple:
    item = next((i for i in sample["scene"]["instances"] if i["entity_id"] == entity), None)
    if item is None:
        raise ValueError(f"SHOWCASE_ENTITY_MISSING({sample['delivery'].name},{entity})")
    return (tuple(round(v, 9) for v in item["world_transform"]["translation"]),
            tuple(round(v, 9) for v in item["world_transform"]["rotation_wxyz"]))


def measure_row(row: dict, samples: dict) -> dict:
    """The row's machine evidence: the frozen task closure held, the room moved.

    The core's **world** pose is expected to move, and asserting it constant would be
    wrong: the support is placed by the layout solver against whatever room the seed
    produced, so the whole core translates (and can rotate) with the room. What must
    not move is the closure SPEC-013 calls the core -- the target and support
    identities, the frozen relation between them, and the invariant spec hash the
    randomization promised to preserve. Measured on the three rows of the first
    figure: the support's world translation differs by 0.4-2.8 m between seeds while
    the invariant hash, the entity set and the ``ON`` relation are identical.

    The target's placement *within the support's frame* is reported separately. It is
    not part of the closure -- kitchen's jar sits on the opposite side of the island
    in the two seeds while still ON it -- so it is measured and disclosed rather than
    asserted equal.
    """
    a, b = samples["a"], samples["b"]
    target_id = a["manifest"]["plan"]["target_id"]
    support_id = a["manifest"]["plan"]["support_id"]
    measurement = {"target_id": target_id, "support_id": support_id,
                   "seed_a": a["manifest"]["randomization"].get("room_seed"),
                   "seed_b": b["manifest"]["randomization"].get("room_seed")}
    measurement["scene_version_a"] = a["manifest"]["scene_version"]
    measurement["scene_version_b"] = b["manifest"]["scene_version"]

    randomization_a = a["manifest"]["randomization"]
    randomization_b = b["manifest"]["randomization"]
    closure = {
        "invariant_spec_hash_equal":
            randomization_a.get("invariant_spec_hash") == randomization_b.get("invariant_spec_hash"),
        "invariant_entity_ids_equal":
            sorted(randomization_a.get("invariant_entity_ids", []))
            == sorted(randomization_b.get("invariant_entity_ids", [])),
        "invariant_relation_ids_equal":
            sorted(randomization_a.get("invariant_relation_ids", []))
            == sorted(randomization_b.get("invariant_relation_ids", [])),
    }
    relation = {}
    for side, sample in (("a", a), ("b", b)):
        found = [r for r in sample["scene"]["relations"]
                 if r["subject"] == target_id and r["object"] == support_id and r["hard"]]
        relation[side] = None if not found else {
            "predicate": found[0]["predicate"], "satisfied": found[0]["satisfied"]}
    closure["hard_relation_a"] = relation["a"]
    closure["hard_relation_b"] = relation["b"]
    closure["hard_relation_equal"] = relation["a"] == relation["b"] and relation["a"] is not None
    measurement["core_closure"] = closure
    measurement["core_closure_frozen"] = all(
        value for key, value in closure.items() if isinstance(value, bool))
    if not measurement["core_closure_frozen"]:
        raise ValueError(f"SHOWCASE_CORE_DRIFT({row['row_id']})")

    world_pose = {}
    for entity in (target_id, support_id):
        pose_a, pose_b = core_pose(a, entity), core_pose(b, entity)
        world_pose[entity] = {
            "pose_a_m": pose_a[0], "pose_b_m": pose_b[0],
            "translation_delta_m": max(abs(x - y) for x, y in zip(pose_a[0], pose_b[0])),
            "rotation_delta_max_abs": max(abs(x - y) for x, y in zip(pose_a[1], pose_b[1])),
        }
    measurement["core_world_pose"] = world_pose
    measurement["core_world_pose_moves_with_the_room"] = {
        entity: record["translation_delta_m"] > 1e-6 for entity, record in world_pose.items()}

    local = {}
    for side, sample in (("a", a), ("b", b)):
        support = next(i for i in sample["scene"]["instances"] if i["entity_id"] == support_id)
        target = next(i for i in sample["scene"]["instances"] if i["entity_id"] == target_id)
        rot = quat_matrix(support["world_transform"]["rotation_wxyz"])
        delta = np.asarray(target["world_transform"]["translation"], dtype=np.float64) \
            - np.asarray(support["world_transform"]["translation"], dtype=np.float64)
        local[side] = (rot.T @ delta)
    measurement["core_support_local_offset"] = {
        "a_m": [float(v) for v in local["a"]], "b_m": [float(v) for v in local["b"]],
        "delta_m": float(np.max(np.abs(local["a"] - local["b"]))),
    }
    measurement["core_placement_identical"] = bool(
        np.max(np.abs(local["a"] - local["b"])) <= 1e-6)

    polygon_a = np.asarray(a["scene"]["room"]["polygon_xy"], dtype=np.float64)
    polygon_b = np.asarray(b["scene"]["room"]["polygon_xy"], dtype=np.float64)
    measurement["room_area_m2"] = {
        "a": float(abs(np.dot(polygon_a[:, 0], np.roll(polygon_a[:, 1], -1))
                       - np.dot(polygon_a[:, 1], np.roll(polygon_a[:, 0], -1))) / 2.0),
        "b": float(abs(np.dot(polygon_b[:, 0], np.roll(polygon_b[:, 1], -1))
                       - np.dot(polygon_b[:, 1], np.roll(polygon_b[:, 0], -1))) / 2.0),
    }
    measurement["room_height_m"] = {"a": a["scene"]["room"]["height_m"],
                                    "b": b["scene"]["room"]["height_m"]}
    measurement["openings"] = {
        "a": sorted(o["opening_id"] for o in a["scene"]["room"]["openings"]),
        "b": sorted(o["opening_id"] for o in b["scene"]["room"]["openings"]),
    }
    entities = lambda s: sorted(i["entity_id"] for i in s["scene"]["instances"])
    measurement["instances"] = {"a": entities(a), "b": entities(b)}
    measurement["periphery_only_in_a"] = sorted(set(entities(a)) - set(entities(b)))
    measurement["periphery_only_in_b"] = sorted(set(entities(b)) - set(entities(a)))
    measurement["has_variation"] = not (
        abs(measurement["room_area_m2"]["a"] - measurement["room_area_m2"]["b"]) < 1e-6
        and measurement["openings"]["a"] == measurement["openings"]["b"]
        and not measurement["periphery_only_in_a"] and not measurement["periphery_only_in_b"])
    if not measurement["has_variation"]:
        raise ValueError(f"SHOWCASE_NO_VARIATION({row['row_id']})")
    measurement["depth_valid_fraction"] = {
        "a": float((np.isfinite(a["depth"]) & (a["depth"] > 0)).mean()),
        "b": float((np.isfinite(b["depth"]) & (b["depth"] > 0)).mean()),
    }
    candidates = {c["sample_id"]: c for c in row.get("candidates", [])}
    measurement["candidates_considered"] = len(candidates)
    measurement["excluded"] = row.get("excluded", [])
    measurement["readiness"] = {"a": a["manifest"]["resolution"], "b": b["manifest"]["resolution"]}
    return measurement


def build_selection(selection: dict, out: Path, camera_id: str, depth_vmin: float,
                    depth_vmax: float, cmap_name: str) -> int:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already has files; choose a fresh output directory")
    rows = selection["rows"]
    if len(rows) != 3:
        raise SystemExit(f"SHOWCASE_ROW_COUNT({len(rows)})")
    samples: dict[str, dict] = {}
    for row in rows:
        samples[row["row_id"]] = {
            "a": load_sample(Path(row["seed_a"]["delivery"]).resolve(),
                             Path(row["seed_a"]["render_dir"]).resolve(), camera_id),
            "b": load_sample(Path(row["seed_b"]["delivery"]).resolve(),
                             Path(row["seed_b"]["render_dir"]).resolve(), camera_id),
        }
    out.mkdir(parents=True, exist_ok=True)
    panels = out / "panels"
    panels.mkdir()
    measurements = {}
    for row in rows:
        row_id = row["row_id"]
        measurements[row_id] = measure_row(row, samples[row_id])
        for side in ("a", "b"):
            sample = samples[row_id][side]
            target = f"panels/{row_id}_{side}"
            Image.fromarray(sample["rgb"]).save(out / f"{target}_rgb.png")
            np.save(out / f"{target}_depth_m.npy", sample["depth"])
            depth_display = np.ma.masked_where(~(np.isfinite(sample["depth"]) & (sample["depth"] > 0)),
                                              sample["depth"])
            plt.imsave(out / f"{target}_depth_preview.png", depth_display,
                       cmap=plt.get_cmap(cmap_name).copy(), vmin=depth_vmin, vmax=depth_vmax)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(183 * MM, 162 * MM), facecolor="white")
    grid = fig.add_gridspec(3, 3, left=0.012, right=0.988, bottom=0.086, top=0.878,
                            width_ratios=(1.0, 1.32, 1.32), wspace=0.09, hspace=0.34)
    for index, row in enumerate(rows):
        row_id = row["row_id"]
        sample = samples[row_id]
        ax_plan = fig.add_subplot(grid[index, 0])
        draw_plan(ax_plan, sample["a"]["scene"], sample["a"]["plan"], sample["a"]["measurement"],
                  sample["a"]["manifest"]["plan"]["target_id"],
                  sample["a"]["manifest"]["plan"]["support_id"], legend=index == 0)
        objective = row.get("objective") or (sample["a"]["manifest"]["plan"]["target_id"]
                                            + " on " + sample["a"]["manifest"]["plan"]["support_id"])
        # Two lines only. The relation id used to be appended here and pushed the
        # bedroom row's second line into the neighbouring column at print width;
        # it is carried in measurements.json instead, where it is checked.
        ax_plan.set_title(f"{row['label']}\n{objective}", loc="left", fontsize=7.0,
                          color="#243943", pad=3.5)
        camera = camera_of(sample["a"]["render_dir"], camera_id)
        for column, side in ((1, "a"), (2, "b")):
            ax = fig.add_subplot(grid[index, column])
            draw_rgb_panel(ax, sample[side], camera_of(sample[side]["render_dir"], camera_id),
                           sample[side]["manifest"]["plan"]["target_id"],
                           sample[side]["manifest"]["plan"]["support_id"],
                           f"seed {side.upper()}", SEED_A if side == "a" else SEED_B)
    fig.text(0.012, 0.982, selection.get("title", "Frozen task core across independent room seeds"),
             fontsize=9.5, weight="bold", color="#243943", va="top")
    fig.text(0.012, 0.945,
             "Same frozen task closure in both columns of a row — target and support identity, the hard "
             "relation, and the invariant spec hash.",
             fontsize=6.2, color="#4d5f68", va="top")
    fig.text(0.012, 0.010,
             "Left: RoomPlan (dashed) against the built shell's walls, with the task objects' footprints.\n"
             "Right: full-room Isaac RGB of each seed from one shared figure camera; the outlines come from "
             "the worker's own instance ids, and the door is marked.\n"
             "Every panel is readiness=draft and every delivery's geometry.in_bounds gate is red; the figure says "
             "so rather than implying the rooms passed. Readiness follows each delivery's own ledger.",
             fontsize=5.6, color="#64737b", va="bottom")
    pdf = out / "figure_room_alignment.pdf"
    png = out / "figure_room_alignment.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # Companion: the same two seeds per row as camera-z depth, one fixed metre scale.
    fig = plt.figure(figsize=(183 * MM, 118 * MM), facecolor="white")
    grid = fig.add_gridspec(3, 2, left=0.035, right=0.985, bottom=0.10, top=0.90,
                            wspace=0.10, hspace=0.24)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#202832")
    for index, row in enumerate(rows):
        row_id = row["row_id"]
        for column, side in ((0, "a"), (1, "b")):
            depth = samples[row_id][side]["depth"]
            valid = np.isfinite(depth) & (depth > 0)
            ax = fig.add_subplot(grid[index, column])
            ax.imshow(np.ma.masked_where(~valid, depth), cmap=cmap, vmin=depth_vmin,
                      vmax=depth_vmax, interpolation="nearest")
            # Short title: the full row label overflows the panel at print width.
            room_name = row["label"].split(" \u00b7 ")[0]
            ax.set_title(f"{room_name} \u00b7 seed {side.upper()} \u00b7 room "
                         f"{samples[row_id][side]['manifest']['randomization'].get('room_seed')}",
                         loc="left", fontsize=6.8, color=SEED_A if side == "a" else SEED_B, pad=3.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    bar = fig.colorbar(ScalarMappable(norm=colors.Normalize(depth_vmin, depth_vmax), cmap=cmap),
                       ax=fig.axes, orientation="horizontal", fraction=0.03, pad=0.04, aspect=45)
    bar.set_label("Camera-z depth (m) — one scale for every panel", fontsize=6.2, labelpad=2)
    bar.ax.tick_params(labelsize=5.8, length=2)
    depth_pdf = out / "figure_room_alignment_depth.pdf"
    depth_png = out / "figure_room_alignment_depth.png"
    fig.savefig(depth_pdf, dpi=300)
    fig.savefig(depth_png, dpi=300)
    plt.close(fig)

    provenance = {
        "figure": "figure_room_alignment",
        "camera": camera_id,
        "depth": {"convention": "camera_z_meters", "display_range_m": [depth_vmin, depth_vmax],
                  "display_colormap": cmap_name},
        "rows": {
            row["row_id"]: {
                side: {
                    "delivery": str(samples[row["row_id"]][side]["delivery"]),
                    "render_dir": str(samples[row["row_id"]][side]["render_dir"]),
                    "scene_version": samples[row["row_id"]][side]["manifest"]["scene_version"],
                    "room_seed": samples[row["row_id"]][side]["manifest"]["randomization"].get("room_seed"),
                    "capture_state": samples[row["row_id"]][side]["report"].get("capture_state"),
                    "files": {
                        "manifest": sha(samples[row["row_id"]][side]["delivery"] / "delivery_manifest.json"),
                        "scene_ir": sha(samples[row["row_id"]][side]["delivery"] / "scene_ir.json"),
                        "render_report": sha(samples[row["row_id"]][side]["render_dir"] / "isaac" / "report.json"),
                        "rgb": sha(samples[row["row_id"]][side]["render_dir"] / "isaac"
                                   / samples[row["row_id"]][side]["capture"]["rgb_path"]),
                        "depth": sha(samples[row["row_id"]][side]["render_dir"] / "isaac"
                                     / samples[row["row_id"]][side]["capture"]["depth_path"]),
                    },
                } for side in ("a", "b")
            } for row in rows
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (out / "measurements.json").write_text(json.dumps(measurements, indent=2, sort_keys=True) + "\n")
    (out / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pdf": str(pdf), "depth_pdf": str(depth_pdf),
                      "rows": [r["row_id"] for r in rows],
                      "core_closure_frozen": {k: v["core_closure_frozen"] for k, v in measurements.items()},
                      "core_placement_identical": {k: v["core_placement_identical"]
                                                   for k, v in measurements.items()},
                      "has_variation": {k: v["has_variation"] for k, v in measurements.items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
