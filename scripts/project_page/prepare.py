"""Prepare a local-only visual review bundle from a read-only v1.3 release."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender as pr
import trimesh
from PIL import Image

from surface import orient_faces


ROOT = Path(__file__).resolve().parents[2]
SCENES = {
    "tea_service": ("Coffee corner", "Cup placement"),
    "insert_pencil": ("Study desk", "Pencil insertion"),
    "T12_pot": ("Garden balcony", "Pot repositioning"),
    "wash_station": ("Laundry room", "Wash station reset"),
}


def fetch(base, key, target, directory=False):
    flags = ["-r"] if directory else []
    subprocess.run(["ossutil", "cp", *flags, base + "/" + key, str(target)],
                   stdout=subprocess.DEVNULL, check=True)


def export_preview(meshes, visual, texture, center, radius, target):
    """Export normalized geometry, retaining the authored UV indices."""
    scene = trimesh.Scene()
    triangles = flipped = 0
    for i, src in enumerate(meshes):
        points = ((src.vertices - center) / radius)[:, [0, 2, 1]].copy()
        points[:, 2] *= -1  # A rotation from Z-up to Y-up, not a reflection.
        faces, changed = orient_faces(points, src.faces)
        flipped += changed
        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        if visual:
            uv = np.asarray(src.visual.uv)
            if uv.shape != (len(points), 2) or not np.isfinite(uv).all():
                raise ValueError("invalid UV coordinates")
            mat = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture, metallicFactor=0, roughnessFactor=.7, doubleSided=False)
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        else:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(
                ["#62ad9b", "#e7a25d", "#668fc2", "#c6a4cb", "#9ab76c"][i % 5])
        mesh.remove_unreferenced_vertices()
        # Recompute after winding changes; do not export stale OBJ normals.
        normals = np.asarray(mesh.vertex_normals)
        if not np.isfinite(normals).all() or (np.linalg.norm(normals, axis=1) < .99).any():
            raise ValueError("invalid recomputed normals")
        triangles += len(faces)
        scene.add_geometry(mesh)

    def scrub(tree):
        for section in ("nodes", "meshes", "materials", "images", "scenes"):
            for record in tree.get(section, []):
                record.pop("name", None)
                record.pop("extras", None)
    target.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True, tree_postprocessor=scrub))
    return triangles, flipped


def check_views(path, renderer):
    """Read back the delivery GLB and compare culled and two-sided depth."""
    loaded = trimesh.load(path, force="scene", process=False)
    scene = pr.Scene.from_trimesh_scene(loaded, bg_color=[.93,.94,.92,1], ambient_light=[.48]*3)
    camera = scene.add(pr.OrthographicCamera(xmag=1.12, ymag=1.12, znear=.1, zfar=10))
    light = scene.add(pr.DirectionalLight(color=np.ones(3), intensity=2.1))
    worst = 0.
    for i, (yaw, elevation) in enumerate([(42,32),(118,18),(194,-23),(264,33),(332,73),(68,-70)]):
        y, e = np.radians([yaw,elevation])
        z = np.array([np.cos(e)*np.sin(y), np.sin(e), np.cos(e)*np.cos(y)])
        x = trimesh.unitize(np.cross([0,1,0], z))
        pose = np.eye(4)
        pose[:3,:3] = np.column_stack((x, np.cross(z,x), z))
        pose[:3,3] = z * 4
        scene.set_pose(camera, pose)
        scene.set_pose(light, pose @ trimesh.transformations.rotation_matrix(-.55,[0,1,0]))
        rgb, depth = renderer.render(scene)
        both = renderer.render(scene, flags=pr.RenderFlags.SKIP_CULL_FACES | pr.RenderFlags.DEPTH_ONLY)
        mask = both > 0
        if mask.mean() < .001 or np.any(mask[[0,-1],:]) or np.any(mask[:,[0,-1]]):
            raise ValueError("empty, tiny or clipped view")
        loss = float((np.abs(depth[mask]-both[mask]) > .012).mean())
        worst = max(worst, loss)
        if loss > .08:
            raise ValueError("backface depth disagreement above 8 percent")
        if i == 0:
            Image.fromarray(rgb).save(path.with_suffix(".webp"), quality=90)
    return round(worst, 4)


def render_asset(folder, leaf, output, renderer):
    root = ET.parse(folder / f"m_{leaf}.xml").getroot()
    body = root.find("worldbody/body")
    meshes = {n.get("name"): n for n in root.findall("asset/mesh")}
    texture = Image.open(folder / root.find("asset/texture").attrib["file"]).convert("RGB")
    texture.thumbnail((1024, 1024))
    geometry = {"visual": [], "collision": []}
    if len(root.findall(".//body")) != 1 or set(body.attrib) - {"name"}:
        raise ValueError("unsupported body transform")
    for geom in body.findall("geom"):
        mesh = meshes[geom.attrib["mesh"]]
        if (set(mesh.attrib) - {"name", "file"} or
                {"pos", "quat", "euler", "axisangle", "fromto"} & geom.attrib.keys()):
            raise ValueError("unsupported mesh transform")
        obj = trimesh.load(folder / mesh.attrib["file"], force="mesh", process=False)
        visual = geom.get("contype") == "0"
        if visual and getattr(obj.visual, "uv", None) is None:
            raise ValueError("missing UVs")
        geometry["visual" if visual else "collision"].append(obj)

    if not all(geometry.values()):
        raise ValueError("missing representation")
    points = np.concatenate([m.vertices for m in geometry["visual"]])
    center = (points.min(0) + points.max(0)) / 2
    radius = np.linalg.norm(points - center, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("invalid bounds")
    public = {"size_cm": (np.ptp(points,axis=0)*100).round(2).tolist(),
              "hulls": len(geometry["collision"]), "texture_px": list(texture.size)}
    audit = {}
    for mode, group in geometry.items():
        public[mode] = uuid.uuid4().hex
        path = output / (public[mode] + ".glb")
        triangles, flips = export_preview(group, mode == "visual", texture, center, radius, path)
        loss = check_views(path, renderer)
        public[mode + "_triangles"] = triangles
        audit[mode] = {"triangles": triangles, "faces_reoriented": flips, "max_depth_loss": loss}
    return public, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--film", required=True, type=Path)
    parser.add_argument("--scene-catalog", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path, help="Private CSV: production,name,family")
    parser.add_argument("--viewer-package", required=True, type=Path, help="Extracted model-viewer npm package")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        parser.error("Use a fresh external output directory")
    base = os.environ["RFW_PAGE_RELEASE_URI"].rstrip("/")
    out.mkdir(parents=True, exist_ok=False)
    assets = []
    audit = {"accepted": {}, "rejected": {}}
    with args.selection.open() as handle:
        selection = list(csv.DictReader(handle))
    wanted = {s["production"] for s in selection}
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)
        fetch(base, "tables/release_mapping.csv", tmp / "map.csv")
        with (tmp / "map.csv").open() as handle:
            rows = {r["object_id"]: r for r in csv.DictReader(handle) if r["object_id"] in wanted}
        renderer = pr.OffscreenRenderer(560, 560)
        try:
            for sample in selection:
                row = rows[sample["production"]]
                leaf = row["new_leaf"]
                folder = tmp / leaf
                folder.mkdir()
                fetch(base, f"assets/{row['new_d']}/{row['new_c']}/{leaf}/mjcf/", folder, True)
                with tempfile.TemporaryDirectory(dir=tmp) as work:
                    try:
                        item, evidence = render_asset(folder, leaf, Path(work), renderer)
                    except ValueError as error:
                        audit["rejected"][sample["name"]] = str(error)
                        print(f"Excluded {sample['name']}: {error}", flush=True)
                        continue
                    for derivative in Path(work).iterdir():
                        shutil.move(derivative, out / derivative.name)
                audit["accepted"][sample["name"]] = evidence
                assets.append(dict(name=sample["name"], family=sample["family"], **item))
                print(f"Prepared {sample['name']}", flush=True)
        finally:
            renderer.delete()
    records = {s["id"]: s for s in json.loads(args.scene_catalog.read_text())["clips"]}
    scenes = []
    for key, (name, task) in SCENES.items():
        entry = records[key]
        poster = args.scene_catalog.parent / entry["poster"]
        token = uuid.uuid4().hex
        Image.open(poster).convert("RGB").save(out / f"{token}.webp", quality=92)
        shutil.copyfile(args.scene_catalog.parent / entry["video"], out / f"{token}.mp4")
        scenes.append(dict(id=token, name=name, task=task,
                           scope="Partial operation" if entry["scope"] == "局部操作" else "Recorded execution"))
    shutil.copyfile(args.film, out / "film.mp4")
    shutil.copyfile(args.viewer_package / "dist/model-viewer.min.js", out / "viewer.js")
    shutil.copyfile(args.viewer_package / "LICENSE", out / "viewer-license.txt")
    (out / "review.json").write_text(json.dumps(audit, indent=2))
    if not assets:
        raise ValueError("No assets passed review")
    payload = dict(assets=assets, scenes=scenes)
    (out / "catalog.js").write_text("window.RFW_CATALOG = " + json.dumps(payload) + ";\n")


if __name__ == "__main__":
    main()
