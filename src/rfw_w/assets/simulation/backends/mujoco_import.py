# SPDX-License-Identifier: Apache-2.0
"""Acceptance test: load a (repaired) MJCF into MuJoCo and verify

  1. COLLISION  -- collision geom count == part_physics n_collision_hulls;
  2. PHYSICS    -- body mass / COM / inertia match part_physics.json;
                   geom friction == whole_mu1 (slide) / whole_mu2 (torsion);
  3. RENDER     -- offscreen frame is non-empty with the object visible.

Run with any python that has mujoco:

  python3 validate_mujoco_import.py \
      --xml <asset_dir>/mjcf/<key>_fixed.xml \
      --asset-dir <asset_dir> \
      --out /tmp/mujoco_acceptance/<key>

Exits 0 only if every check passes. Writes <out>/report.json + <out>/render.png.
Set MUJOCO_GL=egl (default) for headless offscreen rendering.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen


def run(xml_path, asset_dir, out_dir, res=(960, 720), teleop=False):
    import numpy as np
    import mujoco

    os.makedirs(out_dir, exist_ok=True)
    key = os.path.basename(asset_dir.rstrip("/"))
    report = {"key": key, "xml": xml_path, "checks": {}, "status": "ok"}
    phys = json.load(open(os.path.join(asset_dir, "part_physics.json")))

    # ------------------------------------------------------------- load
    spec = mujoco.MjSpec.from_file(xml_path)
    spec.visual.global_.offwidth = res[0]
    spec.visual.global_.offheight = res[1]
    model = spec.compile()
    data = mujoco.MjData(model)
    # object-only framing stats BEFORE the ground plane is added below
    center = np.array(model.stat.center)
    reach = float(max(model.stat.extent, 1e-3))
    report["checks"]["xml_loaded"] = {
        "pass": True,
        "ngeom": model.ngeom,
        "nmesh": model.nmesh,
    }

    # ------------------------------------------------- 1) collision hulls
    if teleop:
        sidecar = json.load(
            open(os.path.join(asset_dir, "mesh", f"{key}_teleop_collision.json"))
        )
        n_hull_expect = int(sidecar["n_hulls"])
    else:
        n_hull_expect = int(phys["n_collision_hulls"])
    coll = np.nonzero(model.geom_contype)[0]
    report["checks"]["collision_geom_count"] = {
        "got": int(len(coll)),
        "expect": n_hull_expect,
        "pass": int(len(coll)) == n_hull_expect,
    }

    # ------------------------------------------------------ 2) rigid physics
    bodies = [b for b in range(1, model.nbody) if model.body_mass[b] > 0]
    if len(bodies) == 1:
        b = bodies[0]
        exp_mass = float(phys["total_mass"])
        exp_com = np.array(phys["com"], dtype=float)
        exp_eigs = np.sort(np.linalg.eigvalsh(np.array(phys["inertia"], dtype=float)))
        mass_ok = abs(model.body_mass[b] - exp_mass) <= max(1e-4, 1e-3 * exp_mass)
        com_ok = bool(np.allclose(model.body_ipos[b], exp_com, atol=1e-4))
        inertia_ok = bool(
            np.allclose(np.sort(model.body_inertia[b]), exp_eigs, rtol=0.05, atol=1e-6)
        )
        report["checks"]["physics"] = {
            "mass": {
                "got": float(model.body_mass[b]),
                "expect": exp_mass,
                "pass": bool(mass_ok),
            },
            "com": {
                "got": model.body_ipos[b].tolist(),
                "expect": exp_com.tolist(),
                "pass": com_ok,
            },
            "inertia_eigs": {
                "got": np.sort(model.body_inertia[b]).tolist(),
                "expect": exp_eigs.tolist(),
                "pass": inertia_ok,
            },
        }
    else:
        report["checks"]["physics"] = {
            "pass": False,
            "error": f"{len(bodies)} mass-bearing bodies, expect 1",
        }

    # ------------------------------------------------- 3) friction material
    mu1 = float(phys.get("whole_mu1", 0.5))
    mu2 = float(phys.get("whole_mu2", 0.005))
    fr = model.geom_friction[coll] if len(coll) else np.zeros((0, 3))
    fr_ok = (
        len(coll) > 0
        and bool(np.allclose(fr[:, 0], mu1, atol=1e-6))
        and bool(np.allclose(fr[:, 1], mu2, atol=1e-6))
    )
    report["checks"]["friction"] = {
        "got_slide": None if len(coll) == 0 else float(fr[0, 0]),
        "got_torsion": None if len(coll) == 0 else float(fr[0, 1]),
        "expect": [mu1, mu2],
        "n_collision_geoms": int(len(coll)),
        "pass": fr_ok,
    }

    # ------------------------------------------------------------- 4) render
    # minimal studio: ground plane + directional light, camera from model stats
    grid = spec.add_material(name="ground", rgba=[0.9, 0.9, 0.92, 1.0])
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20.0, 20.0, 0.05], material="ground"
    )
    spec.worldbody.add_light(
        pos=[0.4, -0.3, 0.8],
        dir=[-0.5, 0.4, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.9, 0.87, 0.82],
        castshadow=True,
    )
    spec.visual.headlight.ambient = [0.3, 0.3, 0.32]
    spec.visual.headlight.diffuse = [0.7, 0.68, 0.64]
    model = spec.compile()
    data = mujoco.MjData(model)

    # rest pose: lift the free joint so the world-AABB bottom sits on the
    # ground. NOTE: geom_aabb is in each mesh's principal-inertia frame
    # (geom_xquat may be non-identity), so transform the 8 aabb corners to
    # the world frame instead of subtracting half-sizes axis-wise.
    def _world_aabb_min_z(m, d):
        corners = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
        mins = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue  # skip the infinite ground plane
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            rot = d.geom_xmat[g].reshape(3, 3)
            pts = (rot @ (c + corners * h).T).T + d.geom_xpos[g]
            mins.append(pts[:, 2].min())
        return float(min(mins))

    if model.njnt > 0 and model.nq >= 7:
        mujoco.mj_forward(model, data)
        data.qpos[2] = -_world_aabb_min_z(model, data) + 1e-4
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = 2.2 * reach
    cam.azimuth = 45.0
    cam.elevation = -25.0
    renderer = mujoco.Renderer(model, height=res[1], width=res[0])
    renderer.update_scene(data, camera=cam)
    img = renderer.render().copy()
    n_obj_px = int((img.mean(axis=2) < 200).sum())
    render_ok = float(img.std()) > 1.0 and n_obj_px > 500
    from PIL import Image

    Image.fromarray(img).save(os.path.join(out_dir, "render.png"))
    report["checks"]["render"] = {
        "pass": bool(render_ok),
        "frame_std": round(float(img.std()), 2),
        "object_pixels": n_obj_px,
        "png": os.path.join(out_dir, "render.png"),
    }

    # ---------------------------------------------------------------- verdict
    def _flatten_pass(d):
        out = []
        for v in d.values():
            if isinstance(v, dict):
                if "pass" in v:
                    out.append(bool(v["pass"]))
                else:
                    out.extend(_flatten_pass(v))
        return out

    all_pass = all(_flatten_pass(report["checks"]))
    report["status"] = "ok" if all_pass else "fail"
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--teleop",
        action="store_true",
        help="expect the teleop sidecar hull count instead of "
        "part_physics n_collision_hulls",
    )
    a = ap.parse_args()
    ok = run(a.xml, a.asset_dir, a.out, teleop=a.teleop)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
