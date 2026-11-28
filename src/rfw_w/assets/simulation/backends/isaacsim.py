# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim 4.5 driver: RTX-render a 360 orbit of an aligned asset resting on a
tabletop, and export a self-contained USD.

The native isaacsim URDF importer does NOT bring our per-part OBJ geometry into
the stage (it builds the scaffolding but no meshes -> "Used null prim"), so we
instead convert the textured visual OBJ to USD with omni.kit.asset_converter
(robust for OBJ+mtl+texture) and reference it into a hand-built scene. Physics
drop/settle is validated separately in Genesis; here the object is placed at
rest for a deterministic, well-framed RTX texture showcase.

Must be launched as a plain script (not `python -m`); boot SimulationApp first.

  python .../isaacsim_orbit.py /path/<uuid>.urdf --out <asset>/validation --frames 180
"""

import argparse
import json
import os
import sys
import time


def _look_at_matrix(eye, target, world_up=(0.0, 0.0, 1.0)):
    """world_from_camera Gf.Matrix4d for a USD camera (looks down local -Z)."""
    from pxr import Gf

    eye = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
    target = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
    up = Gf.Vec3d(*world_up)
    fwd = (target - eye).GetNormalized()
    right = Gf.Cross(fwd, up)
    if right.GetLength() < 1e-6:
        right = Gf.Vec3d(1.0, 0.0, 0.0)
    right = right.GetNormalized()
    up2 = Gf.Cross(right, fwd)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0.0))
    m.SetRow(1, Gf.Vec4d(up2[0], up2[1], up2[2], 0.0))
    m.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0.0))
    m.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1.0))
    return m


def _convert_obj_to_usd(obj_path, usd_path):
    """OBJ (+mtl+texture) -> USD via omni.kit.asset_converter. Returns bool."""
    import asyncio
    import omni.kit.asset_converter as ac

    async def _go():
        task = ac.get_instance().create_converter_task(obj_path, usd_path, None)
        return await task.wait_until_finished()

    ok = asyncio.get_event_loop().run_until_complete(_go())
    return bool(ok) and os.path.isfile(usd_path)


def run(
    urdf_path,
    out_dir,
    frames=180,
    fps=30,
    res=(1280, 960),
    settle_steps=0,
    spp_per_frame=16,
    path_tracing=False,
    subframes=None,
):
    from isaacsim import SimulationApp

    # RaytracedLighting (real-time RTX) is the stable high-quality path here.
    # PathTracing needs the OptiX denoiser weights (/usr/share/nvidia/
    # nvoptix.bin), absent in this container. Opt in with --pt where present.
    renderer = "PathTracing" if path_tracing else "RaytracedLighting"
    app = SimulationApp({"headless": True, "renderer": renderer})
    t0 = time.perf_counter()

    import numpy as np
    import carb
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    from rfw_w.assets.simulation import common

    meta = common.parse_urdf_meta(urdf_path)
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "sim": "isaacsim",
        "uuid": meta["uuid"],
        "mass": meta["mass"],
        "renderer": renderer,
        "placement": "static-rest",
    }

    if path_tracing:
        s = carb.settings.get_settings()
        s.set("/rtx/pathtracing/spp", spp_per_frame)
        s.set("/rtx/pathtracing/totalSpp", spp_per_frame)

    # ---- prep the visual mesh: copy OBJ+MTL+texture to a temp dir and drop the
    #      texture's ALPHA channel. Our textures are RGBA; the OBJ maps the PNG
    #      as map_Kd, so asset_converter wires the alpha into UsdPreviewSurface
    #      opacity -> RTX renders the object as dithered/see-through. RGB fixes it.
    import shutil
    from PIL import Image

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    mesh_dir = os.path.join(urdf_dir, "mesh")
    prep_dir = os.path.join(out_dir, "_isaac_src")
    if os.path.isdir(prep_dir):
        shutil.rmtree(prep_dir)
    os.makedirs(prep_dir)
    for f in os.listdir(mesh_dir):
        src = os.path.join(mesh_dir, f)
        if f.lower().endswith(".png"):
            Image.open(src).convert("RGB").save(os.path.join(prep_dir, f))
        elif os.path.isfile(src):
            shutil.copy(src, os.path.join(prep_dir, f))

    visual_obj = os.path.join(prep_dir, f"{meta['uuid']}.obj")
    conv_usd = os.path.join(out_dir, f"{meta['uuid']}_visual.usd")
    if not _convert_obj_to_usd(visual_obj, conv_usd):
        raise RuntimeError(f"OBJ->USD conversion failed for {visual_obj}")

    # ---- build a clean scene ----
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

    # white glossy floor (RTX reflects the object). Big flat box, top face z=0.
    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Ground"))
    ground.CreateSizeAttr(2.0)
    gm = Gf.Matrix4d(1.0)
    gm.SetScale(Gf.Vec3d(8.0, 8.0, 0.005))
    gm.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, -0.005))
    UsdGeom.Xformable(ground.GetPrim()).AddTransformOp().Set(gm)
    # glossy white PBR material for a soft mirror reflection
    mtl = UsdShade.Material.Define(stage, Sdf.Path("/World/GroundMat"))
    sh = UsdShade.Shader.Define(stage, Sdf.Path("/World/GroundMat/PBR"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.9, 0.9, 0.92)
    )
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.12)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.6)
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(ground.GetPrim()).Apply(ground.GetPrim())
    UsdShade.MaterialBindingAPI(ground.GetPrim()).Bind(mtl)

    # object: reference the converted mesh, rest its AABB bottom on the table
    obj = stage.DefinePrim("/World/obj", "Xform")
    obj.GetReferences().AddReference(conv_usd)
    rest_z = -float(meta["aabb_min"][2])
    UsdGeom.Xformable(obj).AddTransformOp().Set(
        Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(0.0, 0.0, rest_z))
    )

    # verify the mesh actually loaded (the whole point)
    app.update()
    n_mesh = sum(
        1
        for p in stage.Traverse(Usd.TraverseInstanceProxies())
        if p.GetTypeName() == "Mesh"
    )
    report["n_mesh"] = n_mesh
    if n_mesh == 0:
        raise RuntimeError("converted USD has no mesh geometry")

    # lights (kept modest so the tabletop stays light-gray, not blown to white)
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key.CreateIntensityAttr(900.0)
    key.CreateAngleAttr(2.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))  # warm key
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 0.0, 40.0))
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(250.0)

    # ---- camera (transform driven by a look-at matrix; USD -Z convention) ----
    framing = common.orbit_framing(meta, table_z=0.0)
    cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path("/World/orbit_cam"))
    aperture = 20.955  # mm
    focal = aperture / (2.0 * np.tan(np.radians(framing["fov_deg"]) / 2.0))
    cam_prim.CreateFocalLengthAttr(float(focal))
    cam_prim.CreateHorizontalApertureAttr(aperture)
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1000.0))
    cam_xf = UsdGeom.Xformable(cam_prim.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_op = cam_xf.AddTransformOp()

    import omni.replicator.core as rep

    rp = rep.create.render_product("/World/orbit_cam", (res[0], res[1]))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)

    def _aim(eye):
        cam_op.Set(_look_at_matrix(eye, framing["target"]))

    def _grab():
        """Step the render graph until the annotator yields a valid HxWx4.
        rt_subframes accumulates RTX samples per step -> higher = less grain."""
        for _ in range(30):
            rep.orchestrator.step(rt_subframes=subframes, pause_timeline=True)
            d = np.asarray(annot.get_data())
            if d.ndim == 3 and d.shape[0] > 1:
                return d[:, :, :3]
        return None

    # RaytracedLighting is a real-time integrator: it needs many accumulated
    # subframes per view or the still frame looks grainy / see-through. This is
    # the dominant per-frame cost, so it's tunable (default 96 for a single
    # hero render; batch runs drop it to keep 240-frame orbits tractable).
    if subframes is None:
        subframes = max(spp_per_frame, 8) if path_tracing else 96
    _aim(next(iter(common.orbit_positions(framing, 1))))
    _grab()  # prime the pipeline
    frames_rgb = []
    for eye in common.orbit_positions(framing, frames):
        _aim(eye)
        d = _grab()
        if d is not None:
            frames_rgb.append(d)

    mp4 = os.path.join(out_dir, "isaacsim_orbit.mp4")
    common.write_mp4(frames_rgb, mp4, fps=fps)
    report.update(status="ok", n_frames=len(frames_rgb), video=mp4)

    # ---- export a self-contained scene USD next to the asset ----
    try:
        usd_out = os.path.join(urdf_dir, f"{meta['uuid']}.usd")
        stage.Export(usd_out)
        report["usd"] = usd_out
    except Exception as e:
        print(f"  [isaac] USD export failed: {type(e).__name__}: {e}")

    report["seconds"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(out_dir, "isaacsim_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"  [isaac] {meta['uuid']} -> {mp4} "
        f"({renderer}, {len(frames_rgb)}f, meshes={n_mesh}, "
        f"{report['seconds']}s)"
    )

    app.close()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--settle", type=int, default=0)  # unused (static placement)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument(
        "--subframes",
        type=int,
        default=None,
        help="RTX accumulation steps/frame (default 96; lower=faster)",
    )
    ap.add_argument(
        "--pt", action="store_true", help="use PathTracing (needs OptiX nvoptix.bin)"
    )
    a = ap.parse_args()
    try:
        run(
            a.urdf,
            a.out,
            frames=a.frames,
            fps=a.fps,
            spp_per_frame=a.spp,
            path_tracing=a.pt,
            subframes=a.subframes,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "isaacsim_report.json"), "w") as f:
            json.dump(
                {
                    "sim": "isaacsim",
                    "status": "fail",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                },
                f,
                indent=2,
            )
        raise


if __name__ == "__main__":
    main()
