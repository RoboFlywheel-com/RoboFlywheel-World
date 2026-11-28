# SPDX-License-Identifier: Apache-2.0
# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import logging
import os
import subprocess
import sys

import coacd
import numpy as np
import trimesh

logger = logging.getLogger(__name__)

__all__ = [
    "decompose_convex_coacd",
    "decompose_convex_mesh",
    "decompose_convex_mp",
]


def decompose_convex_coacd(
    filename: str,
    outfile: str,
    params: dict,
    verbose: bool = False,
    auto_scale: bool = True,
    scale_factor: float = 1.0,
) -> None:
    """Decomposes a mesh using CoACD and saves the result.

    This function loads a mesh from a file, runs the CoACD algorithm with the
    given parameters, optionally scales the resulting convex hulls to match the
    original mesh's bounding box, and exports the combined result to a file.

    Args:
        filename: Path to the input mesh file.
        outfile: Path to save the decomposed output mesh.
        params: A dictionary of parameters for the CoACD algorithm.
        verbose: If True, sets the CoACD log level to 'info'.
        auto_scale: If True, automatically computes a scale factor to match the
            decomposed mesh's bounding box to the visual mesh's bounding box.
        scale_factor: An additional scaling factor applied to the vertices of
            the decomposed mesh parts.
    """
    coacd.set_log_level("info" if verbose else "warn")

    mesh = trimesh.load(filename, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)

    result = coacd.run_coacd(mesh, **params)

    meshes = []
    for v, f in result:
        meshes.append(trimesh.Trimesh(v, f))

    # Compute collision_scale because convex decomposition usually makes the mesh larger.
    if auto_scale:
        all_mesh = sum([trimesh.Trimesh(*m) for m in result])
        convex_mesh_shape = np.ptp(all_mesh.vertices, axis=0)
        visual_mesh_shape = np.ptp(mesh.vertices, axis=0)
        scale_factor *= visual_mesh_shape / convex_mesh_shape

    combined = trimesh.Scene()
    for mesh_part in meshes:
        mesh_part.vertices *= scale_factor
        combined.add_geometry(mesh_part)

    combined.export(outfile)


def decompose_convex_mesh(
    filename: str,
    outfile: str,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    preprocess_mode: str = "auto",
    preprocess_resolution: int = 30,
    resolution: int = 2000,
    mcts_nodes: int = 20,
    mcts_iterations: int = 150,
    mcts_max_depth: int = 3,
    pca: bool = False,
    merge: bool = True,
    seed: int = 0,
    auto_scale: bool = True,
    scale_factor: float = 1.005,
    verbose: bool = False,
) -> str:
    """Decomposes a mesh into convex parts with retry logic.

    This function serves as a wrapper for `decompose_convex_coacd`, providing
    explicit parameters for the CoACD algorithm and implementing a retry
    mechanism. If the initial decomposition fails, it attempts again with
    `preprocess_mode` set to 'on'.

    Args:
        filename: Path to the input mesh file.
        outfile: Path to save the decomposed output mesh.
        threshold: CoACD parameter. See CoACD documentation for details.
        max_convex_hull: CoACD parameter. See CoACD documentation for details.
        preprocess_mode: CoACD parameter. See CoACD documentation for details.
        preprocess_resolution: CoACD parameter. See CoACD documentation for details.
        resolution: CoACD parameter. See CoACD documentation for details.
        mcts_nodes: CoACD parameter. See CoACD documentation for details.
        mcts_iterations: CoACD parameter. See CoACD documentation for details.
        mcts_max_depth: CoACD parameter. See CoACD documentation for details.
        pca: CoACD parameter. See CoACD documentation for details.
        merge: CoACD parameter. See CoACD documentation for details.
        seed: CoACD parameter. See CoACD documentation for details.
        auto_scale: If True, automatically scale the output to match the input
            bounding box.
        scale_factor: Additional scaling factor to apply.
        verbose: If True, enables detailed logging.

    Returns:
        The path to the output file if decomposition is successful.

    Raises:
        RuntimeError: If convex decomposition fails after all attempts.
    """
    coacd.set_log_level("info" if verbose else "warn")

    if os.path.exists(outfile):
        logger.warning(f"Output file {outfile} already exists, removing it.")
        os.remove(outfile)

    params = dict(
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        pca=pca,
        merge=merge,
        seed=seed,
    )

    try:
        decompose_convex_coacd(
            filename, outfile, params, verbose, auto_scale, scale_factor
        )
        if os.path.exists(outfile):
            return outfile
    except Exception as e:
        if verbose:
            print(f"Decompose convex first attempt failed: {e}.")

    if preprocess_mode != "on":
        try:
            params["preprocess_mode"] = "on"
            decompose_convex_coacd(
                filename, outfile, params, verbose, auto_scale, scale_factor
            )
            if os.path.exists(outfile):
                return outfile
        except Exception as e:
            if verbose:
                print(
                    f"Decompose convex second attempt with preprocess_mode='on' failed: {e}"
                )

    raise RuntimeError(f"Convex decomposition failed on {filename}")


# Child bootstrap, run as `python -c` by decompose_convex_mp. Imports happen
# INSIDE the try so import errors are printed (and thus captured in the log
# file the parent redirects to) instead of dying silently at unpickle time
# like a multiprocessing-spawn target would.
_COACD_CHILD_CODE = """
import json, os, sys, traceback
filename, outfile, params_json = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["OMP_NUM_THREADS"] = "4"
try:
    from rfw_w.assets.geometry.coacd import decompose_convex_coacd
    decompose_convex_coacd(filename, outfile, json.loads(params_json), verbose=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""


def decompose_convex_mp(
    filename: str,
    outfile: str,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    preprocess_mode: str = "auto",
    preprocess_resolution: int = 30,
    resolution: int = 2000,
    mcts_nodes: int = 20,
    mcts_iterations: int = 150,
    mcts_max_depth: int = 3,
    pca: bool = False,
    merge: bool = True,
    seed: int = 0,
    verbose: bool = False,
    auto_scale: bool = True,
    scale_factor: float = 1.005,
    timeout: float = 600.0,
) -> str:
    """Decomposes a mesh into convex parts in a separate process.

    Runs CoACD via ``subprocess`` (``python -c`` bootstrap) rather than
    multiprocessing.spawn: the child gets the parent's FULL import environment
    from the installed package in a clean environment, and its stdout +
    stderr are captured at the fd level into ``<outfile>.coacd.log`` -- so
    Python import errors, CoACD native logs AND native crashes (returncode
    -6/-11) are all diagnosable from the parent's log tail. A native crash or
    a ``timeout`` (child is then killed) counts as a failed attempt and falls
    through to the retry / RuntimeError, so one poison mesh can never take
    down the whole worker. (The previous spawn-based child died silently on
    distributed workers -- exit 1 with zero output -- because the re-import of the
    target's module failed before any user code ran.)

    See <upstream-project-docs> for details.

    Args:
        filename: Path to the input mesh file.
        outfile: Path to save the decomposed output mesh.
        threshold: CoACD parameter.
        max_convex_hull: CoACD parameter.
        preprocess_mode: CoACD parameter.
        preprocess_resolution: CoACD parameter.
        resolution: CoACD parameter.
        mcts_nodes: CoACD parameter.
        mcts_iterations: CoACD parameter.
        mcts_max_depth: CoACD parameter.
        pca: CoACD parameter.
        merge: CoACD parameter.
        seed: CoACD parameter.
        verbose: Unused (child always logs verbosely into the log file).
        auto_scale: If True, automatically scale the output.
        scale_factor: Additional scaling factor to apply.
        timeout: Per-attempt wall-clock limit in seconds; a child still alive
            past it is killed and counts as a failed attempt.

    Returns:
        The path to the output file if decomposition is successful.

    Raises:
        RuntimeError: If convex decomposition fails after all attempts.
    """
    import json

    params = dict(
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        pca=pca,
        merge=merge,
        seed=seed,
    )

    log_path = outfile + ".coacd.log"
    last_diag = ""
    # the child must import coacd/trimesh (runtime-installed target dir) and
    # the pipeline package exactly the way THIS process can
    child_env = dict(os.environ)
    child_env.pop("PYTHONPATH", None)
    child_env.pop("PYTHONHOME", None)
    child_env["OMP_NUM_THREADS"] = "4"

    def _attempt() -> bool:
        nonlocal last_diag
        cmd = [
            sys.executable,
            "-c",
            _COACD_CHILD_CODE,
            filename,
            outfile,
            json.dumps(params),
        ]
        try:
            with open(log_path, "w") as lf:
                r = subprocess.run(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    env=child_env,
                )
            rc = r.returncode
        except subprocess.TimeoutExpired:
            last_diag = f"timeout ({timeout}s), child killed"
            logger.warning(f"CoACD {last_diag} on {filename}")
            return False
        if rc != 0:
            tail = ""
            try:
                with open(log_path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    f.seek(max(0, f.tell() - 1500))
                    tail = f.read().decode("utf-8", "replace").strip()[-800:]
            except OSError:
                pass
            last_diag = f"child exit code {rc}; log tail: {tail}"
            logger.warning(f"CoACD child failed on {filename}: {last_diag}")
            return False
        return os.path.exists(outfile)

    if _attempt():
        return outfile

    if preprocess_mode != "on":
        params["preprocess_mode"] = "on"
        if _attempt():
            return outfile

    raise RuntimeError(f"Convex decomposition failed on {filename} ({last_diag})")
