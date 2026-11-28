# SPDX-License-Identifier: Apache-2.0
"""Build aligned RGB and palette-mask grids for the unified part VLM call."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from rfw_w.assets.annotation.palette import COLORS

VIEW_PRESETS = {
    "front": (0.0, 0.45, 1.8),
    "back": (math.pi, 0.45, 1.8),
    "left": (-math.pi / 2, 0.45, 1.8),
    "right": (math.pi / 2, 0.45, 1.8),
    "bottom": (0.0, -math.pi / 2 + 0.1, 2.0),
}
LENS_MM, SENSOR_MM = 35.0, 36.0


def _camera(yaw, pitch, distance):
    eye = np.asarray(
        [
            distance * math.cos(pitch) * math.sin(yaw),
            distance * math.cos(pitch) * math.cos(yaw),
            distance * math.sin(pitch),
        ]
    )
    forward = -eye / np.linalg.norm(eye)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.99:
        up = np.asarray([0.0, 1.0, 0.0])
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    corrected_up = np.cross(side, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = side, corrected_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    focal = LENS_MM / (0.5 * SENSOR_MM)
    near, far = 0.01, 100.0
    projection = np.asarray(
        [
            [focal, 0, 0, 0],
            [0, focal, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ]
    )
    return view, projection


def _blender_frame(vertices):
    transformed = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
    lower, upper = transformed.min(axis=0), transformed.max(axis=0)
    return (transformed - (lower + upper) * 0.5) / max((upper - lower).max(), 1e-12)


def render_masks(seg_glb: str | Path, views: list[str], size: int) -> list[Image.Image]:
    mesh = trimesh.load(seg_glb, force="mesh", process=False)
    vertices, faces = _blender_frame(np.asarray(mesh.vertices)), np.asarray(mesh.faces)
    ids_path = Path(seg_glb).with_name("face_ids.npy")
    if not ids_path.is_file():
        raise ValueError("mask rendering requires explicit face assignments")
    ids = np.load(ids_path)
    if len(ids) != len(faces) or (ids < 0).any() or (ids >= len(COLORS)).any():
        raise ValueError("mask face assignments are invalid or ambiguous")
    colors = np.asarray([COLORS[int(part)] for part in ids], dtype=np.uint8)
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    images = []
    for view_name in views:
        view, projection = _camera(*VIEW_PRESETS[view_name])
        clip = (projection @ view @ homogeneous.T).T
        divisor = np.where(abs(clip[:, 3]) < 1e-12, 1e-12, clip[:, 3])
        ndc = clip[:, :2] / divisor[:, None]
        pixels = np.column_stack(
            [
                (ndc[:, 0] * 0.5 + 0.5) * size,
                (1 - (ndc[:, 1] * 0.5 + 0.5)) * size,
            ]
        )
        view_z = (view @ homogeneous.T).T[:, 2]
        order = np.argsort(view_z[faces].mean(axis=1))
        image = Image.new("RGB", (size, size), "white")
        drawer = ImageDraw.Draw(image)
        for face_index in order:
            drawer.polygon(
                [tuple(pixels[vertex]) for vertex in faces[face_index]],
                fill=tuple(int(value) for value in colors[face_index]),
            )
        images.append(image)
    return images


def _white_composite(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB").resize((size, size))


def _grid(images: list[Image.Image], size: int) -> Image.Image:
    output = Image.new("RGB", (size * len(images), size), "white")
    for index, image in enumerate(images):
        output.paste(image.resize((size, size)), (index * size, 0))
    return output


def build_grids(
    asset_id: str,
    seg_glb: Path,
    render_dir: Path,
    output_dir: Path,
    views: list[str],
    size: int = 512,
) -> tuple[Path, Path]:
    rgb_paths = [render_dir / f"{asset_id}_{view}.png" for view in views]
    missing = [str(path) for path in rgb_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RGB renders: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path, mask_path = output_dir / "rgb_grid.png", output_dir / "mask_grid.png"
    _grid([_white_composite(path, size) for path in rgb_paths], size).save(rgb_path)
    _grid(render_masks(seg_glb, views, size), size).save(mask_path)
    return rgb_path, mask_path


def palette_ids(face_ids) -> list[int]:
    return [int(value) for value in np.unique(face_ids) if 0 <= value < len(COLORS)]
