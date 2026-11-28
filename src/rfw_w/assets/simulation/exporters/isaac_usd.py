# SPDX-License-Identifier: Apache-2.0
"""Isaac usd for one explicitly supplied asset directory."""

import argparse

import json

import os

import re

import sys


def _parse_obj(path):
    """Minimal OBJ reader: returns (points[f32 Nx3], face_counts, face_indices).

    Handles v / f with v, v/vt, v//vn, v/vt/vn index forms; triangulates fans.
    """
    import numpy as np

    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                else:  # fan-triangulate n-gons
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    pts = np.asarray(verts, dtype=np.float32) if verts else np.zeros((0, 3), np.float32)
    counts = np.full(len(faces), 3, dtype=np.int32)
    indices = (
        np.asarray(faces, dtype=np.int32).ravel() if faces else np.zeros(0, np.int32)
    )
    return pts, counts, indices


def _inertia_to_principal(inertia):
    """3x3 inertia tensor -> (diagonalInertia[3], principalAxes quat (w,x,y,z)).

    PhysX convention: I = R diag(I_diag) R^T with R from principalAxes.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    w, v = np.linalg.eigh(np.asarray(inertia, dtype=np.float64))
    if np.linalg.det(v) < 0:  # force right-handed
        v[:, 0] = -v[:, 0]
    q = Rotation.from_matrix(v).as_quat()  # x,y,z,w
    return w.tolist(), (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


class IsaacUsdBuilder:
    """Keeps one SimulationApp alive and converts many assets in-process."""

    def __init__(self, teleop=False):
        from isaacsim import SimulationApp

        self.app = SimulationApp({"headless": True})
        import omni.kit.app

        em = omni.kit.app.get_app().get_extension_manager()
        em.set_extension_enabled_immediate("omni.kit.asset_converter", True)
        # teleop: colliders come from the budgeted whole-mesh re-decomposition
        # (collision_budget) instead of the stock per-part hulls
        self.teleop = teleop

    # ------------------------------------------------------------- visual
    def _convert_visual(self, asset_dir, key):
        """OBJ -> mesh/<key>.visual.usd (skip if fresh). Returns usd path."""
        import asyncio
        import omni.kit.asset_converter as ac
        from rfw_w.assets.catalog.release import visual_mesh

        obj = str(visual_mesh(asset_dir, key))
        out = os.path.join(asset_dir, "mesh", f"{key}.visual.usd")
        if os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(obj):
            return out

        async def _go():
            ctx = ac.get_instance()
            task = ctx.create_converter_task(obj, out, None)
            return await task.wait_until_finished()

        ok = asyncio.get_event_loop().run_until_complete(_go())
        if not (ok and os.path.isfile(out)):
            raise RuntimeError(f"asset_converter failed for {obj}")
        self._strip_opacity(out)
        return out

    @staticmethod
    def _strip_opacity(visual_usd):
        """Kill texture-alpha-driven transparency on converter output.

        Two shader flavors show up in omni.kit.asset_converter output:
        - UsdPreviewSurface: opacity / opacityThreshold inputs
        - OmniPBR (id may be unauthored): enable_opacity=True +
          enable_opacity_texture=True wires the RGBA texture's alpha into
          opacity, rendering the whole mesh see-through in RTX.
        Returns the number of shaders patched.
        """
        from pxr import Usd, UsdShade

        stage = Usd.Stage.Open(visual_usd)
        n = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Shader):
                continue
            sh = UsdShade.Shader(prim)
            patched = False
            for name, value in (
                ("enable_opacity", False),
                ("enable_opacity_texture", False),
            ):
                inp = sh.GetInput(name)
                if inp:
                    inp.Set(value)
                    patched = True
            for name in ("opacity_texture", "opacity", "opacityThreshold"):
                inp = sh.GetInput(name)
                if not inp:
                    continue
                if inp.GetAttr().GetConnections():
                    inp.GetAttr().ClearConnections()
                prop = inp.GetAttr().GetName()
                if prim.HasProperty(prop):
                    prim.RemoveProperty(prop)
                patched = True
            n += int(patched)
        stage.GetRootLayer().Save()
        return n

    # -------------------------------------------------------------- robot
    def build_asset(self, asset_dir):
        """Build <key>.usd. Returns (key, status, detail)."""
        import numpy as np
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        key = os.path.basename(asset_dir.rstrip("/"))
        phys_path = os.path.join(asset_dir, "part_physics.json")
        sidecar_path = os.path.join(asset_dir, "mesh", f"{key}_teleop_collision.json")
        out_path = os.path.join(asset_dir, f"{key}.usd")
        fresh_sources = [phys_path]
        if self.teleop:
            if not os.path.isfile(sidecar_path):
                return key, "fail", "no teleop sidecar; run collision_budget first"
            fresh_sources.append(sidecar_path)
        if os.path.isfile(out_path) and os.path.getmtime(out_path) >= max(
            os.path.getmtime(p) for p in fresh_sources
        ):
            return key, "skip", "usd up-to-date"
        phys = json.load(open(phys_path))

        import glob

        if self.teleop:
            hulls = sorted(
                glob.glob(
                    os.path.join(
                        asset_dir, "mjcf", "mesh", f"{key}_teleop_collision_*.obj"
                    )
                ),
                key=lambda p: int(os.path.splitext(p)[0].rsplit("_", 1)[1]),
            )
            n_expect = int(json.load(open(sidecar_path))["n_hulls"])
        else:
            from rfw_w.assets.catalog.release import collision_meshes

            hulls = [str(path) for path in collision_meshes(asset_dir, key)]
            n_expect = int(phys.get("n_collision_hulls", len(hulls)))
        if not hulls:
            return key, "fail", "no hull objs"
        if len(hulls) != n_expect:
            return key, "fail", f"hull count {len(hulls)} != expected {n_expect}"

        visual_usd = self._convert_visual(asset_dir, key)

        # build into a temp file: a failed run must not leave an empty usd
        # that the up-to-date check above would then treat as valid
        tmp_path = out_path.removesuffix(".usd") + ".tmp.usd"
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        stage = Usd.Stage.CreateNew(tmp_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root_name = f"obj_{key}"
        root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{root_name}"))
        stage.SetDefaultPrim(root.GetPrim())

        # rigid body + mass
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass_api.CreateMassAttr(float(phys["total_mass"]))
        com = phys.get("com") or [0, 0, 0]
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(x) for x in com]))
        inertia = phys.get("inertia")
        if inertia:
            diag, quat = _inertia_to_principal(inertia)
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(x) for x in diag]))
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(*quat))

        # visual reference (relative path so the asset dir stays relocatable)
        vis = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{root_name}/visual"))
        vis.GetPrim().GetReferences().AddReference(
            f"./mesh/{os.path.basename(visual_usd)}"
        )

        # physics material
        mu1 = float(phys.get("whole_mu1", 0.5))
        mat = UsdPhysics.MaterialAPI.Apply(
            stage.DefinePrim(f"/{root_name}/PhysicsMaterial", "PhysicsMaterial")
        )
        mat.CreateStaticFrictionAttr(mu1)
        mat.CreateDynamicFrictionAttr(float(phys["whole_mu2"]))
        shd_mat = UsdShade.Material(mat.GetPrim())

        # colliders
        coll_scope = UsdGeom.Scope.Define(stage, Sdf.Path(f"/{root_name}/colliders"))
        for i, hp in enumerate(hulls):
            pts, counts, indices = _parse_obj(hp)
            if len(pts) == 0 or len(indices) == 0:
                return key, "fail", f"empty hull {hp}"
            m = UsdGeom.Mesh.Define(
                stage, Sdf.Path(f"/{root_name}/colliders/hull_{i:03d}")
            )
            m.CreatePointsAttr(pts)
            m.CreateFaceVertexCountsAttr(counts)
            m.CreateFaceVertexIndicesAttr(indices)
            m.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision-only, invisible
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
            mc.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
                shd_mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

        stage.GetRootLayer().Save()
        os.replace(tmp_path, out_path)
        return key, "ok", f"{len(hulls)} hulls -> {out_path}"

    def close(self):
        self.app.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--teleop", action="store_true")
    args = parser.parse_args()
    builder = IsaacUsdBuilder(teleop=args.teleop)
    try:
        key, status, detail = builder.build_asset(args.asset_dir)
        print(json.dumps({"key": key, "status": status, "detail": detail}))
        return 1 if status == "fail" else 0
    finally:
        builder.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
