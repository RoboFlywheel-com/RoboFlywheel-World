#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Genesis column of the SPEC-014 five-backend figure.

Every backend rebuilds the frozen scene from ``common_scene.json``'s exported
OBJ meshes; Genesis is one of the four that do. Two passes, never merged:

    visual meshes      -> rgb.png, depth_visible.npy
    collision meshes   -> depth_collision.npy

Genesis is not a plain "import and render" engine on this machine. Two things
had to be established by measurement before it would draw anything, and both
are recorded in the report rather than hidden here:

1. **PyOpenGL has to be pointed at this interpreter's own glvnd.**
   Genesis rasterizes with a vendored pyrender over EGL. This container carries
   two glvnd installations -- the interpreter's
   (``$PREFIX/lib/libOpenGL.so.0``) and the system's
   (``/usr/lib/x86_64-linux-gnu/libOpenGL.so``, an older build) -- and they end
   up sharing one ``libGLdispatch.so.0``: the context is created through the
   system ``libEGL.so.1``, whose dispatch dependency resolves to the
   interpreter's dispatch library, while PyOpenGL's default search for
   ``libOpenGL.so`` finds the *system* one, whose dispatch stubs do not match it.
   The observable failure is not an error message but a SIGSEGV on the first
   vertex call (``glVertexAttribPointer`` reached from
   ``baseplatform._NullFunctionPointer.__call__``), after ``glGetString`` has
   already returned NULL. ``bind_gl_library_to_interpreter`` binds PyOpenGL to
   the interpreter's own ``libOpenGL.so.0``, so both sides agree; measured, that
   alone renders and no function-resolution patch is applied.

2. **Genesis's patched OBJ loader refuses faces with 8 columns** (a quad plus
   ``v/vt`` texture indices) -- ``genesis/ext/_trimesh_patch.py`` raises
   "Not supported yet" for ``columns == 8``. ``desk/visual.obj`` in this frozen
   scene is exactly that file; the scene's other 38 OBJ files load untouched.
   ``prepare_mesh`` fan-triangulates the n-gons into a scratch copy, carrying the
   ``.mtl`` and its textures beside it, so the surface that reaches the renderer
   is the same triangles the author exported.

Depth convention (measured, not assumed -- see ``depth_convention_found`` and
``depth_self_check_m`` in the report): Genesis's rasterizer hands back a
``float32`` image that is **linear view-space axial distance along the camera
forward axis, in metres, positive in front** -- i.e. exactly ``camera_z_meters``.
The evidence is ``genesis/ext/pyrender/jit_render.py::read_depth_buf``:

    depth_im = buf[::-1, :] * 2 - 1
    depth_im = (z_near * z_far) / (z_far + z_near - depth_im * (z_far - z_near)) * 2

That is the exact inverse of the GL projection row for ``d = -z_view``: solving
``z_ndc = (f+n)/(f-n) - 2fn/((f-n) d)`` for ``d`` gives
``d = 2fn / (f+n - z_ndc (f-n))``. No ray-distance conversion is needed. The
**invalid/background sentinel is exactly the far plane**: a cleared depth buffer
reads window depth 1.0, so ``z_ndc = 1`` and ``d = f``. Those pixels are finite
and positive, so ``save_depth_products``'s own ``isfinite & > 0`` mask would
count them as seen; they are turned into NaN here, using Genesis's own validity
rule from ``genesis/vis/camera.py`` (``(near < d) & (d < far * (1 - 1e-3))``)
so the reported valid fraction is geometry and not background.

Reading the source says what the buffer *should* be; ``depth_self_check`` tests
what it *is*, against geometry whose camera-z is known independently. Both the
axial and the ray-distance hypothesis are scored on the same 7623 in-frame mesh
vertices, and only the axial one survives: 2687 of them fit a rendered pixel to
within 1 cm, against 23 for ray distance (those 23 sit on the optical axis, where
the two hypotheses coincide). 97% of the pixels are closer to axial.

Usage:

    env -u PYTHONPATH -u PYTHONHOME \\
      python -m rfw_w.scenes.workers.multisim.render_genesis \\
      --scene outputs/_paper/spec014_common_scene_20260924_02 \\
      --output-dir outputs/_paper/spec014_renders_20260924_02/genesis
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import json
import math
import os
import platform as host_platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# python -m rfw_w.scenes.workers.multisim.render_genesis -> repo root
ROOT = Path.cwd()

from rfw_w.scenes.workers.multisim.common import (
    HIDDEN_SHELL_ROLES,
    CommonScene,
    appearance_gates,
    backend_report,
    digest,
    load_light_rig,
    resolves_to_no_material,
    look_at_quaternion_wxyz,
    quat_wxyz_to_matrix,
    save_depth_products,
    write_json,
)

VISIBLE_PREFIX = "depth_visible"
COLLISION_PREFIX = "depth_collision"

# The figure camera is pinned to square pixels, so one vertical field of view is
# the same quantity in all five backends (common.CameraSpec.fovy_rad).
# near/far are the rasterizer's clip planes. far is also the background sentinel
# Genesis writes, so it has to sit well past the room: this room is ~6.1 m
# across, and 20 m keeps the sentinel unambiguous while leaving the 24-bit depth
# buffer about 3e-5 m of resolution at 5 m.
# Genesis takes light intensity as a bare scalar, not in the rig's linear-colour
# units. These two map the rig onto that scale; one declared factor per light
# type rather than a hand-tuned number per light.
GENESIS_DIRECTIONAL_INTENSITY = 4.0
GENESIS_POINT_SCALE = 2.0

NEAR_M = 0.05
FAR_M = 20.0

# The depth self-check's tolerance: one pixel at this camera's focal length spans
# ~1.3 cm of depth at 5 m before any geometry slope is involved, so a residual
# below a centimetre is the same surface, not merely a similar number.
SELF_CHECK_TOL_M = 0.01
# Vertices kept per object. The shells have thousands and the point is a
# population, not a census; a few hundred is already far more than the 19 an
# object-per-sample scheme could offer.
SELF_CHECK_MAX_PER_OBJECT = 600
# The 3x3 flatness rule the per-object sample demands of its pixel, in metres of
# depth spread. Same rule and same number as the sibling PyBullet driver.
EDGE_RANGE_M = 0.02


def bind_gl_library_to_interpreter() -> dict:
    """Point PyOpenGL's GL library at the one belonging to this interpreter.

    Must run **before** ``import genesis``, so the raw GL modules -- which bind
    their functions against ``PLATFORM.GL`` -- see it. See the module docstring
    for the measurement behind this.

    A wrong pick here does not raise: it segfaults inside pyrender's first vertex
    call, which is a long way from the cause. The report therefore records the
    default that was found, the one that was bound, and -- after the first
    render -- what GL says about itself.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import ctypes

    import OpenGL.platform as opengl_platform
    import OpenGL.platform.egl as egl_platform
    from OpenGL.platform import ctypesloader

    record = {
        "what": "OpenGL.platform.egl.EGLPlatform.GL bound to the interpreter's libOpenGL.so.0",
        "why": "two glvnd installations share one libGLdispatch.so.0 here; the system "
               "libOpenGL.so's dispatch stubs do not match the dispatch library the EGL "
               "context is registered in, so every entry point reached through it is a "
               "SIGSEGV on first use",
        "pyopengl_platform": type(opengl_platform.PLATFORM).__name__,
    }

    # What PyOpenGL's own search would pick, resolved the way ``EGLPlatform.GL``
    # resolves it -- by name, not through the lazy property. Asking the property
    # would cache its answer on the instance, and an instance entry shadows the
    # class-level binding made below.
    try:
        default = ctypesloader.loadLibrary(ctypes.cdll, "OpenGL", mode=ctypes.RTLD_GLOBAL)
        record["pyopengl_default_gl_library"] = getattr(default, "_name", None)
    except OSError as err:
        record["pyopengl_default_gl_library"] = None
        record["pyopengl_default_error"] = f"{type(err).__name__}: {err}"

    candidate = Path(sys.prefix) / "lib" / "libOpenGL.so.0"
    record["interpreter_prefix"] = sys.prefix
    record["candidate"] = str(candidate)
    record["candidate_exists"] = candidate.is_file()
    if not candidate.is_file():
        record["bound"] = False
        record["fallback_reason"] = "this interpreter carries no libOpenGL.so.0"
        return record

    bound = ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
    egl_platform.EGLPlatform.GL = bound
    # Drop any answer the lazy property already cached on the instance.
    dropped = opengl_platform.PLATFORM.__dict__.pop("GL", None)
    record["instance_cache_dropped"] = getattr(dropped, "_name", None)
    record["bound"] = getattr(opengl_platform.PLATFORM.GL, "_name", None)
    return record


def gl_identity(cam=None) -> dict:
    """What the GL context actually is. Recorded, never assumed.

    Genesis uncurrents its offscreen context after every render
    (``rasterizer.render_camera``'s ``finally``), and a GL query with no current
    context answers NULL. The camera's own renderer is therefore asked to make
    its context current again first -- it is the object that owns it, and the
    alternative (a second EGL context) is a different context than the one the
    pixels came from.
    """
    out: dict = {}
    if cam is not None:
        renderer = getattr(getattr(cam, "_rasterizer", None), "_renderer", None)
        try:
            renderer.make_current()
            out["context_made_current"] = renderer is not None
        except Exception as err:  # pragma: no cover - diagnostic only
            out["context_made_current"] = f"{type(err).__name__}: {err}"
    try:
        import OpenGL.platform as opengl_platform
        from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString

        out["pyopengl_platform"] = getattr(
            opengl_platform.PLATFORM, "__name__", type(opengl_platform.PLATFORM).__name__
        )
        for label, enum in (("GL_VENDOR", GL_VENDOR), ("GL_RENDERER", GL_RENDERER),
                            ("GL_VERSION", GL_VERSION)):
            try:
                value = glGetString(enum)
                out[label] = value.decode("utf-8", "replace") if value is not None else None
            except Exception as err:  # pragma: no cover - diagnostic only
                out[label] = f"unavailable: {type(err).__name__}: {err}"
    except Exception as err:  # pragma: no cover - diagnostic only
        out["error"] = f"{type(err).__name__}: {err}"
    return out


def gpu_table() -> list[dict]:
    """Free memory and utilisation per GPU, so the device choice is visible."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        return [{"error": f"nvidia-smi unavailable: {type(err).__name__}: {err}"}]
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append({"index": int(parts[0]), "memory_used_mib": int(parts[1]),
                         "memory_total_mib": int(parts[2]), "utilization_percent": int(parts[3])})
        except ValueError:
            continue
    return rows


def pick_gpu(explicit: int | None, table: list[dict]) -> tuple[int | None, str]:
    usable = [row for row in table if "index" in row]
    if explicit is not None:
        return explicit, f"--active-gpu {explicit}"
    if not usable:
        return None, "no readable GPU table; the CUDA device was left to Genesis"
    best = min(usable, key=lambda row: (row["utilization_percent"], row["memory_used_mib"]))
    return best["index"], (
        f"freest of {len(usable)}: index {best['index']} at "
        f"{best['memory_used_mib']}/{best['memory_total_mib']} MiB and "
        f"{best['utilization_percent']}% utilisation"
    )


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def obj_stats(path: Path) -> dict:
    """Vertex and face counts straight out of the OBJ text.

    Read from the file rather than from a loader, so the number is a property of
    the geometry the renderer was given, independent of how any loader chose to
    merge it.
    """
    n_verts = n_faces = n_ngons = n_textured_faces = 0
    bbox_min = np.full(3, np.inf)
    bbox_max = np.full(3, -np.inf)
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                try:
                    xyz = [float(v) for v in line.split()[1:4]]
                except ValueError:
                    continue
                if len(xyz) != 3:
                    continue
                n_verts += 1
                bbox_min = np.minimum(bbox_min, xyz)
                bbox_max = np.maximum(bbox_max, xyz)
            elif line.startswith("f "):
                n_faces += 1
                refs = line.split()[1:]
                if len(refs) > 3:
                    n_ngons += 1
                if any("/" in ref for ref in refs):
                    n_textured_faces += 1
    return {
        "vertices": n_verts,
        "faces": n_faces,
        "ngon_faces": n_ngons,
        "faces_with_texture_indices": n_textured_faces,
        "bbox_min": bbox_min.tolist() if n_verts else None,
        "bbox_max": bbox_max.tolist() if n_verts else None,
    }


def obj_vertices(path: Path) -> np.ndarray:
    """(n,3) vertex positions from the OBJ text. Used only by the self-check,
    which must know where the geometry is without asking the renderer."""
    out = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                try:
                    out.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except (IndexError, ValueError):
                    continue
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def triangulate_obj(src: Path, dst_dir: Path) -> tuple[Path, int, int]:
    """Fan-triangulate n-gon faces into a scratch copy of an OBJ.

    Genesis's patched loader refuses 8-column faces (quad plus ``v/vt``). Every
    other line is copied verbatim and the ``.mtl`` and its textures are carried
    beside the copy, so the scratch file differs from the original only in how
    the same surface is spelled.
    """
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    n_ngons = n_tris = 0
    for line in lines:
        if line.startswith("f "):
            refs = line.split()[1:]
            if len(refs) > 3:
                n_ngons += 1
                for i in range(1, len(refs) - 1):
                    out.append(f"f {refs[0]} {refs[i]} {refs[i + 1]}")
                    n_tris += 1
                continue
        elif line.startswith("mtllib "):
            name = line.split(None, 1)[1].strip()
            mtl = src.parent / name
            if mtl.is_file():
                shutil.copyfile(mtl, dst_dir / name)
                for mtl_line in mtl.read_text(encoding="utf-8", errors="replace").splitlines():
                    stripped = mtl_line.strip()
                    if not (stripped.startswith("map_") or " map_" in stripped):
                        continue
                    texture = src.parent / stripped.split()[-1]
                    if texture.is_file():
                        shutil.copyfile(texture, dst_dir / texture.name)
        out.append(line)
    dst = dst_dir / src.name
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst, n_ngons, n_tris


def prepare_mesh(path: Path, scratch_root: Path) -> dict:
    """Hand Genesis a file its own loader will accept.

    The test is Genesis's own: ``genesis/utils/mesh.py`` loads an OBJ through
    ``trimesh.load(path, force="scene", group_material=..., process=False)``, so
    that is what is tried here. A file no loader can read is not a conversion
    this script may paper over -- it is raised.
    """
    import trimesh

    record = {"source": str(path), "used": str(path), "converted": False}
    try:
        trimesh.load(str(path), force="scene", group_material=True, process=False)
        return record
    except Exception as first:
        scratch = scratch_root / f"{path.parent.name}__{path.stem}"
        converted, n_ngons, n_tris = triangulate_obj(path, scratch)
        record["conversion_reason"] = f"{type(first).__name__}: {first}"
        record["converted"] = True
        record["scratch_dir"] = str(scratch)
        record["ngon_faces_split"] = n_ngons
        record["triangles_written"] = n_tris
        try:
            trimesh.load(str(converted), force="scene", group_material=True, process=False)
        except Exception as second:
            raise SystemExit(
                f"MULTISIM_BACKEND_UNAVAILABLE(genesis, OBJ mesh unreadable even after "
                f"triangulation: {path} -> {type(second).__name__}: {second})"
            ) from second
        record["used"] = str(converted)
        return record


# --------------------------------------------------------------------------- #
# depth
# --------------------------------------------------------------------------- #

def genesis_depth_to_camera_z(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Genesis depth -> ``camera_z_meters`` with background as NaN.

    The array is already axial metres (see the module docstring); the work here
    is the sentinel, which is exactly ``far`` and would otherwise be counted as
    a seen pixel by ``save_depth_products``'s ``isfinite & > 0`` mask.
    """
    raw = np.asarray(raw, dtype=np.float32)
    sentinel = FAR_M * (1.0 - 1e-3)
    valid = (raw > NEAR_M) & (raw < sentinel)
    out = np.where(valid, raw, np.float32("nan")).astype(np.float32)
    detail = {
        "raw_dtype": str(raw.dtype),
        "raw_shape": list(raw.shape),
        "raw_min_m": float(raw.min()) if raw.size else None,
        "raw_max_m": float(raw.max()) if raw.size else None,
        "background_sentinel_m": FAR_M,
        "pixels_at_or_past_sentinel": int((raw >= sentinel).sum()),
        "pixels_denormalised_to_nan": int((~valid).sum()),
        "validity_rule": "(near < d) & (d < far * (1 - 1e-3)), Genesis's own rule at "
                         "genesis/vis/camera.py",
    }
    return out, detail


def _flat_neighbourhood(depth: np.ndarray, u: int, v: int, camera, range_m: float) -> bool:
    """Is the depth locally flat around this pixel?

    A pixel on a silhouette or on an occlusion boundary reports whatever surface
    happens to be nearest at its centre, which is not where the vertex is -- the
    vertex sits *inside* the pixel but the rasterizer samples pixel centres, and
    one pixel at this focal length already spans ~1.3 cm of depth at 5 m before
    any slope is involved. Measuring there measures the geometry's edge, not the
    depth convention. ``python -m rfw_w.scenes.workers.multisim.render_pybullet``'s
    self-check rejects the same pixels with the same 2 cm rule over the same 3x3
    neighbourhood, so the two columns are comparable.
    """
    v0, v1 = max(v - 1, 0), min(v + 2, camera.height)
    u0, u1 = max(u - 1, 0), min(u + 2, camera.width)
    patch = depth[v0:v1, u0:u1].astype(np.float64)
    if patch.size < 9 or not np.isfinite(patch).all():
        return False
    return bool(patch.max() - patch.min() <= range_m)


def _residual_class(residual_m: float, tol_m: float) -> str:
    """Which failure mode a residual belongs to, if any. ``agrees`` is the only
    class that is evidence *for* the convention; the other two are properties of a
    z-buffer and of thin geometry, and naming them keeps the tail honest."""
    if residual_m < -tol_m:
        return "occluded"
    if residual_m > tol_m:
        return "off_surface"
    return "agrees"


def depth_self_check(camera, samples: list[dict], depth: np.ndarray,
                     tol_m: float = SELF_CHECK_TOL_M,
                     cap_per_object: int = SELF_CHECK_MAX_PER_OBJECT) -> dict:
    """Compare the rendered depth against the true ``camera_z`` of known geometry.

    Two populations, because they answer different questions.

    **One sample per object.**  The mesh vertex nearest the camera is projected
    with ``camera.project(...)`` and the rendered depth at that pixel is compared
    with the vertex's true camera-z. This is the number the task asks for: a real
    surface of a known object, at a pixel the render must have filled. Nearer
    vertices whose pixel sits on an edge (``_flat_neighbourhood``) are stepped
    over and counted, because a pixel on a silhouette reports the geometry's edge
    rather than the convention; every sample records how many were rejected, so
    that rule cannot quietly choose the answer.

    **Every in-frame vertex, as a convention discriminator.**  Both hypotheses are
    scored on the same vertices: the axial one, and the ray-distance one, which
    differs from it by ``sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)`` -- 1.0x on the
    optical axis and 1.35x at this camera's corners. The question is not which
    hypothesis gives a small median (an occluded pixel is far from both) but how
    many pixels each hypothesis can fit to within a centimetre. Only the true
    convention explains a large set of them.

    Two things move a residual away from zero and neither is a wrong convention,
    so both are counted rather than hidden:

    * **occluded** -- a z-buffer reports the nearest surface, so a vertex behind
      another object is *expected* to disagree, and its residual is negative by
      the gap to the occluder.
    * **off-surface** -- the pixel the vertex falls in is mostly surface *behind*
      it, which is what a grazing facet (a floor seen edge-on) and thin silhouette
      geometry do, where one pixel covers many centimetres of depth.

    The tail is reported rather than trimmed, so a reader can see which failure
    mode each outlier is.
    """
    # --- one sample per object ---------------------------------------------
    rows = []
    skipped = 0
    per_object = {}
    for sample in samples:
        points = sample["vertices"]
        if points.size == 0:
            skipped += 1
            continue
        uv, z = camera.project(points)
        inside = (
            (uv[:, 0] >= 0.5) & (uv[:, 0] < camera.width - 0.5)
            & (uv[:, 1] >= 0.5) & (uv[:, 1] < camera.height - 0.5)
            & (z > 0)
        )
        candidates = np.flatnonzero(inside)
        if candidates.size == 0:
            skipped += 1
            continue
        front = None
        rejected = 0
        for index in candidates[np.argsort(z[candidates], kind="stable")]:
            index = int(index)
            u, v = int(math.floor(uv[index, 0])), int(math.floor(uv[index, 1]))
            if not _flat_neighbourhood(depth, u, v, camera, EDGE_RANGE_M):
                rejected += 1
                continue
            front = index
            break
        fell_back_to_edge = front is None
        if fell_back_to_edge:
            # An object all of whose pixels are edges still gets a sample -- the
            # floor is seen at grazing incidence and is exactly that -- with the
            # caveat stated on the row rather than the object dropped.
            front = int(candidates[int(np.argmin(z[candidates]))])
        u, v = int(math.floor(uv[front, 0])), int(math.floor(uv[front, 1]))
        rendered = float(depth[v, u])
        true_z = float(z[front])
        du = (uv[front, 0] - camera.cx) / camera.fx
        dv = (uv[front, 1] - camera.cy) / camera.fy
        true_ray = true_z * math.sqrt(1.0 + du * du + dv * dv)
        residual = rendered - true_z
        row = {
            "pixel": [u, v],
            "rendered_depth_m": rendered,
            "true_camera_z_m": true_z,
            "residual_m": residual,
            "residual_class": _residual_class(residual, tol_m),
            "n_nearer_vertices_rejected": rejected,
            "every_pixel_of_this_object_is_an_edge": fell_back_to_edge,
        }
        if row["residual_class"] != "agrees":
            # Diagnostics, not a second measurement: a vertex sits inside pixel
            # (floor(u), floor(v)) but the rasterizer samples pixel *centres*, so
            # at a silhouette the containing pixel can be covered by whatever is
            # behind while the next pixel over is the surface itself. Recording
            # the best centre in the 3x3 neighbourhood says which of the two it
            # was, instead of asserting it.
            best = None
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    uu, vv = u + du, v + dv
                    if not (0 <= uu < camera.width and 0 <= vv < camera.height):
                        continue
                    neighbour = float(depth[vv, uu])
                    if not math.isfinite(neighbour):
                        continue
                    if best is None or abs(neighbour - true_z) < abs(best["residual_m"]):
                        best = {"pixel": [uu, vv], "rendered_depth_m": neighbour,
                                "residual_m": neighbour - true_z}
            row["neighbourhood_best_residual_m"] = None if best is None else best["residual_m"]
            row["neighbourhood_best"] = best
        per_object[sample["object_id"]] = row
        rows.append(row)

    if rows:
        abs_res = np.abs(np.asarray([row["residual_m"] for row in rows], dtype=np.float64))
        headline = {
            "n_samples": len(rows),
            "median_abs_m": float(np.median(abs_res)),
            "mean_abs_m": float(abs_res.mean()),
            "max_abs_m": float(abs_res.max()),
            "n_agreeing_within_1cm": int((abs_res <= tol_m).sum()),
            "n_occluded": sum(1 for row in rows if row["residual_class"] == "occluded"),
            "n_off_surface": sum(1 for row in rows if row["residual_class"] == "off_surface"),
            "n_outliers_explained_by_a_neighbouring_pixel": sum(
                1 for row in rows
                if row["residual_class"] != "agrees"
                and row.get("neighbourhood_best_residual_m") is not None
                and abs(row["neighbourhood_best_residual_m"]) <= tol_m
            ),
            "n_nearer_vertices_rejected_as_non_flat": sum(
                row["n_nearer_vertices_rejected"] for row in rows),
        }
    else:
        headline = {"n_samples": 0, "median_abs_m": None, "mean_abs_m": None, "max_abs_m": None,
                    "n_agreeing_within_1cm": 0, "n_occluded": 0, "n_off_surface": 0,
                    "n_outliers_explained_by_a_neighbouring_pixel": 0,
                    "n_nearer_vertices_rejected_as_non_flat": 0}

    # --- every in-frame vertex: which convention fits the pixels? -----------
    pop_res, pop_ray = [], []
    pop_object = {}
    pop_samples = []
    pop_skipped = 0
    for sample in samples:
        points = sample["vertices"]
        if points.size == 0:
            pop_skipped += 1
            continue
        uv, z = camera.project(points)
        inside = (
            (uv[:, 0] >= 0.5) & (uv[:, 0] < camera.width - 0.5)
            & (uv[:, 1] >= 0.5) & (uv[:, 1] < camera.height - 0.5)
            & (z > 0)
        )
        candidates = np.flatnonzero(inside)
        if candidates.size == 0:
            pop_skipped += 1
            continue
        # Even thinning over the file's vertex order, which carries no depth
        # information, so the subset is not biased towards near or far surfaces.
        if candidates.size > cap_per_object:
            picks = np.rint(np.linspace(0, candidates.size - 1, cap_per_object)).astype(np.int64)
            candidates = candidates[picks]
        u = np.floor(uv[candidates, 0]).astype(np.int64)
        v = np.floor(uv[candidates, 1]).astype(np.int64)
        rendered = depth[v, u].astype(np.float64)
        true_z = z[candidates].astype(np.float64)
        du = (uv[candidates, 0] - camera.cx) / camera.fx
        dv = (uv[candidates, 1] - camera.cy) / camera.fy
        true_ray = true_z * np.sqrt(1.0 + du * du + dv * dv)
        finite = np.isfinite(rendered)
        if not finite.any():
            pop_skipped += 1
            continue
        rendered, true_z, true_ray = rendered[finite], true_z[finite], true_ray[finite]
        residual = rendered - true_z
        ray_residual = rendered - true_ray
        abs_res = np.abs(residual)
        pop_object[sample["object_id"]] = {
            "n": int(residual.size),
            "median_abs_m": float(np.median(abs_res)),
            "n_within_1cm_if_axial": int((abs_res <= tol_m).sum()),
            "n_within_1cm_if_ray_distance": int((np.abs(ray_residual) <= tol_m).sum()),
            "n_occluded": int((residual < -tol_m).sum()),
            "n_off_surface": int((residual > tol_m).sum()),
        }
        pop_res.append(residual)
        pop_ray.append(ray_residual)
        for i in range(int(residual.size)):
            pop_samples.append({
                "object_id": sample["object_id"],
                "pixel": [int(u[finite][i]), int(v[finite][i])],
                "rendered_depth_m": float(rendered[i]),
                "true_camera_z_m": float(true_z[i]),
                "residual_m": float(residual[i]),
                "true_ray_distance_m": float(true_ray[i]),
                "residual_if_ray_distance_m": float(ray_residual[i]),
            })

    if pop_res:
        residual = np.concatenate(pop_res)
        ray_residual = np.concatenate(pop_ray)
        abs_res, abs_ray = np.abs(residual), np.abs(ray_residual)
        n = int(residual.size)
        axial_hits = int((abs_res <= tol_m).sum())
        ray_hits = int((abs_ray <= tol_m).sum())
        verdict = "undetermined: no hypothesis explains a large set of pixels"
        if axial_hits >= 100 and axial_hits >= 10 * max(ray_hits, 1):
            verdict = "camera_z_meters"
        elif ray_hits > axial_hits:
            verdict = "ray_distance"
        population = {
            "n_samples": n,
            "n_within_1cm_if_axial": axial_hits,
            "n_within_1cm_if_ray_distance": ray_hits,
            "fraction_within_1cm_if_axial": float(axial_hits / n),
            "fraction_within_1cm_if_ray_distance": float(ray_hits / n),
            "axial_median_abs_m": float(np.median(abs_res)),
            "ray_distance_median_abs_m": float(np.median(abs_ray)),
            "fraction_favouring_axial": float((abs_res < abs_ray).mean()),
            "occluded_fraction": float((residual < -tol_m).mean()),
            "off_surface_fraction": float((residual > tol_m).mean()),
            "objects_sampled": len(pop_object),
            "n_objects_without_a_usable_sample": pop_skipped,
            "per_object": pop_object,
            "largest_residual_samples": sorted(pop_samples,
                                               key=lambda row: -abs(row["residual_m"]))[:8],
        }
    else:
        population = {"n_samples": 0, "n_within_1cm_if_axial": 0,
                      "n_within_1cm_if_ray_distance": 0, "objects_sampled": 0,
                      "per_object": {}, "largest_residual_samples": []}
        verdict = "undetermined: no sample survived"

    headline.update({
        "per_object": {sample["object_id"]: per_object[sample["object_id"]]
                       for sample in samples if sample["object_id"] in per_object},
        "objects_total": len(samples),
        "objects_sampled": len(per_object),
        "n_objects_without_a_usable_sample": skipped,
        "method": "per object, the mesh vertex nearest the camera whose pixel is locally flat "
                  "(3x3 within "
                  f"{EDGE_RANGE_M} m) projected with camera.project(); rendered depth at that "
                  "pixel minus the vertex's true camera-z",
        "edge_rule": f"3x3 depth neighbourhood flat within {EDGE_RANGE_M} m; nearer vertices "
                     "failing it are skipped and counted per object",
        "tolerance_m": tol_m,
        "population": population,
        "convention_verdict": verdict,
        # The independent verifier reads `median_m` from every column's self-check.
        # Same number as `median_abs_m` above, published under the shared name so
        # one reader can compare all five -- and labelled, because this column's
        # median is over objects (one nearest visible vertex each), not over a
        # pixel grid the way MuJoCo's and PyBullet's are. A p95 over nineteen
        # objects would be a different quantity from the others' p95 over pixels
        # and is deliberately not published under that name.
        "median_m": headline.get("median_abs_m"),
        "metric": "per-object nearest visible mesh vertex; |rendered camera-z - true camera-z| "
                  "over one sample per instance (19 instances)",
    })
    return headline


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def build_pass(gs, camera, objs, mesh_paths: list[Path], rig=None):
    """One Genesis scene holding exactly the meshes of one pass.

    The two passes cannot share a scene: an entity's visualization geometry is
    fixed at build time, and SPEC-014 wants the visible depth and the collision
    depth to come from different geometry rather than from one geometry drawn
    twice. Two scenes are built back to back and both are rendered before either
    is released.
    """
    # The shared rig. Before it, this backend passed no lighting at all and
    # took Genesis's built-in default (ambient 0.1, one directional light at
    # intensity 5) -- which is why its column read as a different set from the
    # others. Genesis accepts both directional and point lights, so the whole
    # rig is expressible here.
    vis_options = None
    if rig is not None:
        gain = rig.gain_for("genesis")
        lights = [{"type": "directional", "dir": tuple(rig.direction),
                   "color": tuple(rig.directional_colour),
                   "intensity": GENESIS_DIRECTIONAL_INTENSITY * gain}]
        for point in rig.point_lights:
            lights.append({"type": "point", "pos": tuple(point["position"]),
                           "color": tuple(c / max(point["colour"]) for c in point["colour"]),
                           "intensity": max(point["colour"]) * GENESIS_POINT_SCALE * gain})
        vis_options = gs.options.VisOptions(
            ambient_light=tuple(c * gain for c in rig.ambient), lights=lights)
    gscene = gs.Scene(show_viewer=False, **({"vis_options": vis_options} if vis_options else {}))
    for obj, mesh_path in zip(objs, mesh_paths):
        # A mesh with no material library at all gets the rig's declared shell
        # albedo instead of Genesis's white default -- the same rule, on the
        # same bytes, that MuJoCo, PyBullet and SAPIEN apply.
        surface = None
        if rig is not None and resolves_to_no_material(mesh_path):
            surface = gs.surfaces.Default(color=tuple(rig.shell_albedo))
        gscene.add_entity(
            surface=surface,
            morph=gs.morphs.Mesh(
                file=str(mesh_path),
                pos=tuple(float(v) for v in obj.translation),
                quat=tuple(float(v) for v in obj.rotation_wxyz),
                scale=float(obj.scale),
                # These entities are scenery, never stepped.
                fixed=True,
                # Visualization only: this pass must not add collision geometry.
                visualization=True,
                collision=False,
                # No decimation, no convexification, no inertial re-anchoring.
                # Those rewrite the geometry, and this figure compares the
                # authored geometry between backends.
                decimate=False,
                convexify=False,
                align=False,
                # The exported OBJ files are Z-up metres, as the extractor wrote
                # them. Genesis defaults to True for non-GLTF formats; it is
                # stated here because the whole figure depends on it.
                file_meshes_are_zup=True,
                group_by_material=True,
            )
        )

    gcam = gscene.add_camera(
        res=(camera.width, camera.height),
        pos=tuple(float(v) for v in camera.eye),
        lookat=tuple(float(v) for v in camera.target),
        up=(0.0, 0.0, 1.0),
        fov=float(math.degrees(camera.fovy_rad)),
        near=NEAR_M,
        far=FAR_M,
        GUI=False,
    )
    gscene.build()

    # ``Scene.add_camera`` takes only pos/lookat/up, so the exact pose common.py
    # derives is applied here instead. Genesis's ``transform`` is camera-to-world
    # in the GL convention (``T_to_pos_lookat_up`` reads lookat as ``pos - Z``),
    # which is exactly what
    # ``quat_wxyz_to_matrix(look_at_quaternion_wxyz(camera))`` returns, so this
    # is the frozen camera rather than a look-at re-derived by hand. Both
    # readings are kept and their difference is reported.
    exact = np.eye(4)
    exact[:3, :3] = quat_wxyz_to_matrix(look_at_quaternion_wxyz(camera))
    exact[:3, 3] = camera.eye
    gcam.set_pose(transform=exact)
    return gscene, gcam, exact


def release_scene(gscene) -> dict:
    """Make the GL context uncurrent, then destroy the scene.

    Genesis's teardown calls ``rasterizer.destroy()`` -> ``make_current()``, which
    raises ``GenesisException`` when something is already current -- exactly the
    state a render leaves behind. Uncurrenting first turns the exit into a clean
    one. Each step is guarded on its own: ``make_uncurrent`` also raises when the
    renderer has no current context (a render already uncurrented it), and that
    must not stop the destroy. A failure here is recorded, never raised, because
    it is teardown and not evidence.
    """
    record = {"uncurrent": False, "destroyed": False}
    renderer = getattr(getattr(getattr(gscene, "_visualizer", None), "_rasterizer", None),
                       "_renderer", None)
    if renderer is not None:
        try:
            renderer.make_uncurrent()
            record["uncurrent"] = True
        except Exception as err:  # noqa: BLE001
            record["uncurrent_error"] = f"{type(err).__name__}: {err}"
    try:
        gscene.destroy()
        record["destroyed"] = True
    except Exception as err:  # noqa: BLE001
        record["destroy_error"] = f"{type(err).__name__}: {err}"
    return record


def camera_conversion_record(camera, gcam, exact: np.ndarray) -> dict:
    """Prove the camera that was rendered is the figure camera.

    Three readings, all against what was applied: the transform Genesis ended up
    with, its own intrinsics against the ones the camera spec declares, and a
    look-at computed by Genesis's own helper from the same eye/target/up, so the
    build-time pose and the applied pose can be compared rather than assumed
    equal.
    """
    import torch
    from genesis.utils.geom import pos_lookat_up_to_T

    applied = np.asarray(gcam.transform, dtype=np.float64)
    lookat_form = np.asarray(
        pos_lookat_up_to_T(
            torch.as_tensor(camera.eye, dtype=torch.float64),
            torch.as_tensor(camera.target, dtype=torch.float64),
            torch.as_tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        ).cpu(),
        dtype=np.float64,
    )
    intrinsics = np.asarray(gcam.intrinsics, dtype=np.float64)
    expected_intrinsics = np.array([[camera.fx, 0.0, camera.cx],
                                    [0.0, camera.fy, camera.cy],
                                    [0.0, 0.0, 1.0]])
    return {
        "form": "transform (camera-to-world, GL convention), applied with Camera.set_pose",
        "source": "quat_wxyz_to_matrix(look_at_quaternion_wxyz(camera)), eye as translation",
        "max_abs_difference_from_common_py_m": float(np.abs(applied - exact).max()),
        "max_abs_difference_pos_lookat_up_form_m": float(np.abs(lookat_form - exact).max()),
        "fov_degrees": float(math.degrees(camera.fovy_rad)),
        "resolution": [int(camera.width), int(camera.height)],
        "near_m": NEAR_M,
        "far_m": FAR_M,
        "fx": float(intrinsics[0, 0]),
        "fy": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
        "camera_fx": float(camera.fx),
        "camera_fy": float(camera.fy),
        "camera_cx": float(camera.cx),
        "camera_cy": float(camera.cy),
        "intrinsics_max_abs_difference_px": float(np.abs(intrinsics - expected_intrinsics).max()),
        "up_axis": [0.0, 0.0, 1.0],
    }


def png_size(path: Path) -> tuple[int, int]:
    """Width and height out of the PNG IHDR, so the resolution is read from the
    file that was written rather than trusted from the field that asked for it."""
    header = path.open("rb").read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise SystemExit(f"MULTISIM_DEPTH_INVALID(genesis, {path.name} is not a PNG)")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def write_rgb_png(path: Path, rgb: np.ndarray) -> dict:
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise SystemExit(
            f"MULTISIM_BACKEND_UNAVAILABLE(genesis, rgb came back as shape {arr.shape}, "
            "expected (h, w, 3) or (h, w, 4))"
        )
    if arr.dtype != np.uint8:
        raise SystemExit(
            f"MULTISIM_BACKEND_UNAVAILABLE(genesis, rgb came back as dtype {arr.dtype})"
        )
    dropped_alpha = arr.shape[2] == 4
    if dropped_alpha:
        arr = arr[:, :, :3]
    Image.fromarray(arr).save(path)
    return {
        "array_shape": list(arr.shape),
        "dtype": "uint8",
        "alpha_dropped": dropped_alpha,
        "mean": float(arr.mean()),
        "max": int(arr.max()),
        "nonblack_pixel_fraction": float((arr.max(axis=2) > 2).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", required=True, type=Path,
                        help="the frozen common scene directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lighting", type=Path,
                        default=data_path('examples')
                        / "paper_multisim" / "lighting.json",
                        help="the shared five-backend light rig (SPEC-014 R5)")
    parser.add_argument("--active-gpu", type=int, default=None,
                        help="GPU index to pin; default is the freest one nvidia-smi reports")
    parser.add_argument("--scratch-dir", type=Path, default=None,
                        help="where converted meshes go; default is a fresh temp directory")
    args = parser.parse_args()

    started = time.time()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already has files; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)

    scene_dir = args.scene.resolve()
    scene = CommonScene.load(scene_dir)
    rig = load_light_rig(args.lighting)
    camera = scene.camera
    visible = scene.objects()
    visible_ids = {obj.object_id for obj in visible}
    hidden = [obj.object_id for obj in scene.objects(include_hidden=True)
              if obj.object_id not in visible_ids]

    warnings: list[str] = []
    table = gpu_table()
    gpu_index, gpu_why = pick_gpu(args.active_gpu, table)
    if gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # pyrender's EGL platform selects its device by index too, and its ordering
    # is not guaranteed to match nvidia-smi's. It is therefore only pinned when
    # the device was named explicitly: unpinned, EGL tries the devices in order
    # and keeps the first that gives a context, which is the behaviour the
    # machine has already been measured to support.
    egl_pinned = args.active_gpu is not None
    if egl_pinned:
        os.environ["EGL_DEVICE_ID"] = str(gpu_index)

    gl_patch = bind_gl_library_to_interpreter()

    import genesis as gs

    backend_used = backend_reason = None
    init_error = None
    for backend, reason in ((gs.gpu, "a CUDA device is visible to the process"),
                            (gs.cpu, "gs.gpu was not usable")):
        if backend is gs.gpu and gpu_index is None:
            continue
        try:
            gs.init(backend=backend, logging_level="warning")
        except Exception as err:
            init_error = f"{type(err).__name__}: {err}"
            warnings.append(f"gs.init(backend={'gpu' if backend is gs.gpu else 'cpu'}) "
                            f"failed: {init_error}")
            continue
        backend_used = "gpu" if backend is gs.gpu else "cpu"
        backend_reason = reason
        break
    if backend_used is None:
        raise SystemExit(
            f"MULTISIM_BACKEND_UNAVAILABLE(genesis, no usable gs.init backend: {init_error})"
        )

    gl: dict = {}
    if "EGL" not in str(gl_patch.get("pyopengl_platform", "")):
        warnings.append(
            f"GL platform is {gl_patch.get('pyopengl_platform')!r}, not EGL; the library "
            "binding is only measured for EGL context creation"
        )

    scratch_root = Path(args.scratch_dir) if args.scratch_dir else Path(
        tempfile.mkdtemp(prefix="spec014_genesis_objfix_")
    )
    scratch_root.mkdir(parents=True, exist_ok=True)

    # --- what geometry is going in -----------------------------------------
    geometry_loaded = []
    conversions: dict[str, dict] = {}
    for obj in visible:
        entry = {
            "object_id": obj.object_id,
            "kind": obj.kind,
            "role": obj.role,
            "scale": float(obj.scale),
            "translation": [float(v) for v in obj.translation],
            "rotation_wxyz": [float(v) for v in obj.rotation_wxyz],
            "matrix_world": obj.matrix_world().tolist(),
            "visual_mesh": str(obj.visual_mesh.relative_to(scene_dir)),
            "collision_mesh": str(obj.collision_mesh.relative_to(scene_dir)),
            # The two digests are of the two *bundle* files named above, which is
            # what makes them comparable with the sibling backends' reports. The
            # file actually handed to Genesis is ``*_path_used`` -- the same path
            # unless the loader refused the file -- and its digest is recorded
            # separately, so no entry can claim a file it did not load.
            "visual_mesh_sha256": digest(obj.visual_mesh),
            "collision_mesh_sha256": digest(obj.collision_mesh),
            "visual_source_stats": obj_stats(obj.visual_mesh),
            "collision_source_stats": obj_stats(obj.collision_mesh),
        }
        for key, path in (("visual", obj.visual_mesh), ("collision", obj.collision_mesh)):
            record = prepare_mesh(path, scratch_root / key)
            entry[f"{key}_path_used"] = record["used"]
            entry[f"{key}_path_used_sha256"] = digest(Path(record["used"]))
            entry[f"{key}_path_used_is_the_bundle_file"] = (
                Path(record["used"]) == Path(path))
            if record["converted"]:
                conversions.setdefault(key, {})[obj.object_id] = record
                warnings.append(
                    f"{obj.object_id}/{key}: Genesis's OBJ loader refused the file "
                    f"({record['conversion_reason']}); {record['ngon_faces_split']} n-gon "
                    f"faces were fan-triangulated into a scratch copy"
                )
        geometry_loaded.append(entry)
    by_id = {entry["object_id"]: entry for entry in geometry_loaded}

    # --- pass 1: visual meshes, RGB + depth ---------------------------------
    t0 = time.time()
    visual_scene, visual_cam, exact_transform = build_pass(
        gs, camera, visible, [Path(by_id[obj.object_id]["visual_path_used"]) for obj in visible],
        rig=rig
    )
    build_visual_s = time.time() - t0
    camera_record = camera_conversion_record(camera, visual_cam, exact_transform)

    # What the built entities hold, read back off them: the render is only
    # evidence if what the renderer has is what the file said.
    for obj, entity in zip(visible, visual_scene.entities):
        link = entity.links[0]
        by_id[obj.object_id]["visual_loaded"] = {
            "vgeoms": int(link.n_vgeoms),
            "vertices": int(link.n_vverts),
            "faces": int(link.n_vfaces),
            "source_faces": by_id[obj.object_id]["visual_source_stats"]["faces"],
        }

    t0 = time.time()
    rgb, depth_raw, _, _ = visual_cam.render(rgb=True, depth=True)
    render_visual_s = time.time() - t0
    # The GL context exists from here on, so this is the first moment the strings
    # name the device the pixels came from rather than answering NULL.
    gl = gl_identity(visual_cam)
    if not gl.get("GL_RENDERER"):
        warnings.append("GL_RENDERER came back empty; the device behind the render is unnamed")
    depth_visible, visible_raw_detail = genesis_depth_to_camera_z(depth_raw)
    visible_detail = save_depth_products(out, depth_visible, prefix=VISIBLE_PREFIX)
    visible_detail["raw"] = visible_raw_detail
    rgb_detail = write_rgb_png(out / "rgb.png", rgb)
    width, height = png_size(out / "rgb.png")
    if (width, height) != (camera.width, camera.height):
        raise SystemExit(
            f"MULTISIM_CAMERA_MISMATCH(genesis, rgb.png is {width}x{height}, the camera asked "
            f"for {(camera.width, camera.height)})"
        )

    # --- pass 2: collision meshes, depth only -------------------------------
    t0 = time.time()
    collision_scene, collision_cam, _ = build_pass(
        gs, camera, visible, [Path(by_id[obj.object_id]["collision_path_used"]) for obj in visible]
    )
    build_collision_s = time.time() - t0
    for obj, entity in zip(visible, collision_scene.entities):
        link = entity.links[0]
        by_id[obj.object_id]["collision_loaded"] = {
            "vgeoms": int(link.n_vgeoms),
            "vertices": int(link.n_vverts),
            "faces": int(link.n_vfaces),
            "source_faces": by_id[obj.object_id]["collision_source_stats"]["faces"],
        }

    t0 = time.time()
    depth_collision_raw = collision_cam.render(rgb=False, depth=True)[1]
    render_collision_s = time.time() - t0
    depth_collision, collision_raw_detail = genesis_depth_to_camera_z(depth_collision_raw)
    collision_detail = save_depth_products(out, depth_collision, prefix=COLLISION_PREFIX)
    collision_detail["raw"] = collision_raw_detail
    collision_teardown = release_scene(collision_scene)

    # Two passes are only two passes if they disagree somewhere. The shell's
    # meshes are the same in both (this bundle ships no collision mesh for them),
    # so the disagreement has to come from the furniture -- and it has to exist,
    # or one of the two renders silently reused the other's geometry.
    both = np.isfinite(depth_visible) & np.isfinite(depth_collision)
    pass_diff = {
        "comparable_pixels": int(both.sum()),
        "pixels_differing_by_more_than_1mm": int((np.abs(depth_visible[both]
                                                          - depth_collision[both]) > 1e-3).sum()),
        "max_abs_difference_m": float(np.abs(depth_visible[both]
                                            - depth_collision[both]).max()) if both.any() else None,
        "note": "the shell objects ship visual.obj only, so their collision pass uses that same "
                "file; the furniture uses collision.obj, which is where the two images differ",
    }
    if pass_diff["pixels_differing_by_more_than_1mm"] == 0:
        warnings.append("the visible and collision depth passes are identical; one pass may have "
                        "reused the other's geometry")

    # --- the honest check ---------------------------------------------------
    samples = []
    for obj in visible:
        entry = by_id[obj.object_id]
        matrix = obj.matrix_world()
        samples.append({
            "object_id": obj.object_id,
            "vertices": obj_vertices(Path(entry["visual_path_used"])) @ matrix[:3, :3].T
                        + matrix[:3, 3],
        })
    check = depth_self_check(camera, samples, depth_visible)

    # Where the shell pieces sit relative to each other. Only needed because an
    # occluded self-check sample is otherwise unexplained: a shell object that
    # leaves its siblings' envelope is one the interior camera cannot see, and
    # that is a fact about the scene rather than about the depth convention.
    notes = []
    bboxes = {sample["object_id"]: (sample["vertices"].min(axis=0),
                                    sample["vertices"].max(axis=0)) for sample in samples}
    floor_box = bboxes.get("shell_floor")
    ceiling_box = bboxes.get("shell_ceiling")
    if floor_box is not None:
        # The room's own volume, taken from the shell that defines it: the floor's
        # footprint, carried up to the ceiling. The other shell pieces are compared
        # against this rather than against each other, because the wall's own
        # bounding box is the whole room and would contain anything.
        room_low = floor_box[0].copy()
        room_high = floor_box[1].copy()
        if ceiling_box is not None:
            room_high[2] = float(ceiling_box[1][2])
        outside_room = {}
        for oid in sorted(bboxes):
            # How far the box pokes out on each axis, which is not the same
            # question as whether the intervals are disjoint: a door leaf can
            # overlap the room and still stand 1.29 m past its wall.
            outside = np.maximum(room_low - bboxes[oid][0], bboxes[oid][1] - room_high)
            if not (outside > 0.05).any():
                continue
            row = check["per_object"].get(oid)
            outside_room[oid] = {
                "outside_by_axis_m": [round(float(v), 4) for v in outside],
                "bbox_min": [round(float(v), 3) for v in bboxes[oid][0]],
                "bbox_max": [round(float(v), 3) for v in bboxes[oid][1]],
                "self_check_residual_class": None if row is None else row["residual_class"],
                "self_check_residual_m": None if row is None else row["residual_m"],
                "reading": "measured placement; no mesh was moved to accommodate it",
            }
            if row is not None and row["residual_class"] != "agrees":
                warnings.append(
                    f"{oid} spends "
                    f"{float(outside.max()):.3f} m outside the room volume the floor and ceiling "
                    f"define, so the interior camera sees it through another surface: its "
                    f"self-check residual {row['residual_m']:.4f} m is that occlusion, not the "
                    "depth convention"
                )
        notes.append({"room_volume_m": {"min": [round(float(v), 3) for v in room_low],
                                        "max": [round(float(v), 3) for v in room_high]},
                      "objects_outside_room_volume": outside_room})

        # The convention verdict must not rest on an object the camera cannot
        # see: dropping everything that stands outside the room volume has to
        # leave the same answer.
        population = check["population"]
        dropped = [oid for oid in outside_room if oid in population.get("per_object", {})]
        if dropped:
            axial = population["n_within_1cm_if_axial"] - sum(
                population["per_object"][oid]["n_within_1cm_if_axial"] for oid in dropped)
            ray = population["n_within_1cm_if_ray_distance"] - sum(
                population["per_object"][oid]["n_within_1cm_if_ray_distance"] for oid in dropped)
            samples_left = population["n_samples"] - sum(
                population["per_object"][oid]["n"] for oid in dropped)
            population["verdict_excluding_objects_outside_the_room"] = {
                "dropped_object_ids": dropped,
                "n_samples": samples_left,
                "n_within_1cm_if_axial": axial,
                "n_within_1cm_if_ray_distance": ray,
                "same_verdict": axial >= 100 and axial >= 10 * max(ray, 1),
            }
            if not population["verdict_excluding_objects_outside_the_room"]["same_verdict"]:
                warnings.append(
                    "the depth convention verdict does not survive dropping the objects that "
                    "stand outside the room volume; it is being carried by geometry the camera "
                    "cannot see"
                )
    if (check["median_abs_m"] is not None and check["median_abs_m"] > 0.02) \
            or check["convention_verdict"] != "camera_z_meters":
        warnings.append(
            f"depth self-check did not confirm the axial convention: median residual "
            f"{check['median_abs_m']} m, verdict {check['convention_verdict']!r}; the depth "
            "convention recorded here should not be trusted"
        )
    if check["objects_sampled"] < len(visible):
        warnings.append(
            f"depth self-check sampled {check['objects_sampled']} of {len(visible)} objects; "
            f"{check['n_objects_without_a_usable_sample']} had no usable sample"
        )

    # "The mask is geometry, not background" has to be a measurement too: a
    # sentinel pixel inside the mask would be a background pixel reported as
    # something the camera saw.
    finite = np.isfinite(depth_visible)
    background_leaked = float((finite & (depth_visible >= FAR_M * (1.0 - 1e-3))).mean())
    if background_leaked > 0:
        warnings.append("background pixels reached the visible depth mask")

    elapsed = time.time() - started

    runtime = {
        "python_executable": sys.executable,
        "python_version": host_platform.python_version(),
        "python_full_version": sys.version.replace("\n", " "),
        "host": host_platform.node(),
        "platform": host_platform.platform(),
        "genesis_version": str(getattr(gs, "__version__", "unknown")),
        "genesis_backend": backend_used,
        "genesis_backend_reason": backend_reason,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_pinned": gpu_index,
        "gpu_pinned_reason": gpu_why,
        "egl_device_id_pinned": egl_pinned,
        "egl_device_id": os.environ.get("EGL_DEVICE_ID"),
        "gpu_table": table,
        "gl": gl,
        "gl_library_binding": gl_patch,
        "argv": sys.argv,
    }

    depth_convention_found = {
        "declared_by_engine": None,
        "engine_api": "genesis.vis.camera.Camera.render(rgb=..., depth=...) -> float32 (h, w)",
        "measured": "camera_z_meters",
        "definition": "linear view-space axial distance along the camera forward axis, metres, "
                      "positive in front of the camera",
        "is_ray_distance": False,
        "background_sentinel_m": FAR_M,
        "background_sentinel_rule": "a cleared depth buffer reads window depth 1.0, so z_ndc = 1 "
                                    "and d = z_far exactly",
        "background_pixels_found": visible_raw_detail["pixels_at_or_past_sentinel"],
        "validity_rule": "(near < d) & (d < far * (1 - 1e-3)); Genesis's own rule",
        "conversion_applied": "none to the values; background and clipped pixels set to NaN",
        "evidence": [
            "genesis/ext/pyrender/jit_render.py::read_depth_buf -- "
            "d = (z_near*z_far)/(z_far+z_near - z_ndc*(z_far-z_near)) * 2, the exact inverse of "
            "the GL projection row for d = -z_view",
            "genesis/vis/camera.py::get_point_cloud -- world_x = depth*(u+0.5-cx)/fx, "
            "world_z = depth: the engine itself reads the buffer as axial camera-z",
            f"depth_self_check_m in this report: of {check['population'].get('n_samples')} in-frame mesh "
            f"vertices, {check['population'].get('n_within_1cm_if_axial')} fit a rendered pixel to "
            f"within 1 cm under the axial hypothesis and "
            f"{check['population'].get('n_within_1cm_if_ray_distance')} under ray "
            f"distance -- the two differ by sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2), 1.0x on the "
            f"optical axis and 1.35x at the corners, so the pixels that do fit are where the "
            f"hypotheses are furthest apart",
            f"{check['n_agreeing_within_1cm']} of {check['n_samples']} objects agree to within 1 cm "
            f"at their nearest visible mesh vertex; the outliers are classified in the report "
            "(occluded / off-surface) rather than trimmed",
        ],
    }

    appearance, appearance_gaps = appearance_gates(rgb, rig, camera, "genesis")

    payload = backend_report(
        backend="genesis",
            engine_version=(
                f"genesis {getattr(gs, '__version__', 'unknown')} "
                f"(pyrender rasterizer over EGL, {backend_used} backend) "
                f"under {sys.executable} Python {host_platform.python_version()}"
            ),
            scene=scene,
            camera=camera,
            extra={
                "status": "passed" if not appearance_gaps else "appearance_gate_failed",
                "gates_failed": appearance_gaps,
                "lighting": {
                    **appearance,
                    "applied": {
                        "ambient": [c * rig.gain_for("genesis") for c in rig.ambient],
                        "directional_direction": list(rig.direction),
                        "directional_colour": list(rig.directional_colour),
                        "directional_intensity": GENESIS_DIRECTIONAL_INTENSITY
                                                 * rig.gain_for("genesis"),
                        "point_lights": len(rig.point_lights),
                        "point_intensity_scale": GENESIS_POINT_SCALE,
                        "gain": rig.gain_for("genesis"),
                    },
                    "why": (
                        "before the shared rig this backend passed no lighting and took "
                        "Genesis's default VisOptions (ambient 0.1, one directional light at "
                        "intensity 5). Genesis accepts both directional and point lights, so "
                        "the whole rig is expressible here, unlike PyBullet."
                    ),
                },
                "geometry_loaded": geometry_loaded,
                "conversion": {
                    "poses": "SceneObject.matrix_world(): translation applied after rotation and "
                             "scale; handed to gs.morphs.Mesh(pos=translation, "
                             "quat=rotation_wxyz, scale=scale)",
                    "up_axis": "Z, metres; file_meshes_are_zup=True (Genesis's default for OBJ, "
                               "stated explicitly because the whole figure depends on it)",
                    "quaternion_order": "wxyz, which SceneObject.rotation_wxyz already is",
                    "entity_flags": {"fixed": True, "visualization": True, "collision": False,
                                     "decimate": False, "convexify": False, "align": False,
                                     "group_by_material": True},
                    "obj_ngon_triangulation": conversions,
                    "scratch_root": str(scratch_root),
                    "scratch_root_kept": True,
                },
                "camera_conversion": camera_record,
                "hidden_objects_skipped": hidden,
                "hidden_shell_roles": list(HIDDEN_SHELL_ROLES),
                "hidden_rule": scene.hidden_rule(),
                "depth_convention_found": depth_convention_found,
                "depth_self_check_m": check,
                "depth_self_check_abs_residual_m": check["median_abs_m"],
                "visible_vs_collision": pass_diff,
                "scene_observations": notes,
                "depth_products_detail": {"visible": visible_detail, "collision": collision_detail},
                "passes": {
                    "visible": {
                        "geometry": "visual_mesh",
                        "outputs": ["rgb.png", f"{VISIBLE_PREFIX}.npy", f"{VISIBLE_PREFIX}_mask.npy"],
                        "build_s": build_visual_s,
                        "render_s": render_visual_s,
                    },
                    "collision": {
                        "geometry": "collision_mesh",
                        "outputs": [f"{COLLISION_PREFIX}.npy", f"{COLLISION_PREFIX}_mask.npy"],
                        "rgb": False,
                        "build_s": build_collision_s,
                        "render_s": render_collision_s,
                    },
                },
                "rgb": {"path": "rgb.png", "size": [width, height],
                        "sha256": digest(out / "rgb.png"), **rgb_detail},
                "background_pixels_in_visible_mask_fraction": background_leaked,
                "runtime": runtime,
                "elapsed_s": elapsed,
                "warnings": warnings,
                "claim": "figure_only_static_render: the frozen common scene's own OBJ meshes, "
                         "the frozen figure camera, no simulation step",
            },
    )
    # Release the GL context before the scene goes: Genesis's own teardown calls
    # ``make_current`` and raises when a context is already current, which turns a
    # successful run into a noisy one at exit. Recorded either way.
    payload["teardown"] = {"visible": release_scene(visual_scene), "collision": collision_teardown}
    write_json(out / "report.json", payload)

    print(json.dumps({
        "output_dir": str(out),
        "engine_version": runtime["genesis_version"],
        "python": f"{sys.executable} ({host_platform.python_version()})",
        "genesis_backend": backend_used,
        "gl_renderer": gl.get("GL_RENDERER"),
        "rgb": f"{width}x{height}",
        "depth_visible_valid_fraction": visible_detail["valid_pixel_fraction"],
        "depth_collision_valid_fraction": collision_detail["valid_pixel_fraction"],
        "depth_convention": depth_convention_found["measured"],
        "depth_self_check_median_abs_m": check["median_abs_m"],
        "depth_self_check_max_abs_m": check["max_abs_m"],
        "depth_self_check_n": check["n_samples"],
        "depth_self_check_verdict": check["convention_verdict"],
        "objects": len(visible),
        "warnings": len(warnings),
        "elapsed_s": round(elapsed, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
