# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim RTX path-traced rendering of native measured USD animations."""


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import time

    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--oidn-library')
    p.add_argument('--limit-frames', type=int, help='Diagnostic prefix; never marked as a complete render')
    p.add_argument('--renderer', choices=['pathtracing','realtime'], default='pathtracing')
    a = p.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'PathTracing' if a.renderer=='pathtracing' else 'RaytracedLighting', 'width': 1920, 'height': 1080,
        'anti_aliasing': 0 if a.renderer=='pathtracing' else 2, 'multi_gpu': False, 'active_gpu': a.gpu, 'denoiser': False,
        'extra_args': ['--/rtx/pathtracing/optixDenoiser/enabled=false','--/app/renderer/skipWhileMinimized=false',
            '--/log/level=error','--/log/fileLogLevel=error','--/log/outputStreamLevel=error']})
    import carb
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, UsdLux, Gf
    from PIL import Image

    out = Path(a.stage).resolve()
    provenance = json.loads((out / 'stage_provenance.json').read_text())
    app.set_setting('/app/fastShutdown', False)
    validation=json.loads((out/'stage_validation.json').read_text())
    assert validation['status']=='passed' and all(validation['checks'].values())
    assert validation['scene_sha256']==hashlib.sha256((out/'scene.usdc').read_bytes()).hexdigest()
    settings = carb.settings.get_settings()
    settings.set('/rtx/rendermode', 'PathTracing')
    settings.set('/rtx/pathtracing/spp', a.samples)
    settings.set('/rtx/pathtracing/totalSpp', a.samples)
    settings.set('/rtx/pathtracing/maxBounces', 6)
    settings.set('/rtx/pathtracing/optixDenoiser/enabled', False)
    settings.set('/rtx/post/tonemap/op', 3)
    settings.set('/rtx/post/tonemap/filmIso', 400.0)
    settings.set('/rtx/post/tonemap/fNumber', 4.0)
    if a.renderer=='pathtracing':
        rep.settings.set_render_pathtraced(samples_per_pixel=a.samples)
    else:
        rep.settings.set_render_rtx_realtime(antialiasing='TAA')
        settings.set('/rtx/reflections/enabled',True)
        settings.set('/rtx/shadows/enabled',True)
        settings.set('/rtx/ambientOcclusion/enabled',True)
        settings.set('/rtx/indirectDiffuse/enabled',True)
        settings.set('/rtx/indirectDiffuse/fetchSampleCount',4)
        settings.set('/rtx/indirectDiffuse/maxBounces',4)
        settings.set('/rtx/sceneDb/ambientLightIntensity',.6)
    omni.usd.get_context().open_stage(str(out / 'scene.usdc'))
    stage = omni.usd.get_context().get_stage()
    # Blender can author an unsupported mirrored-ball world texture. Use the
    # explicit neutral studio lighting below rather than a renderer error color.
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            source_dome=UsdLux.DomeLight(prim)
            if source_dome.GetTextureFormatAttr().Get()=='mirroredBall':
                source_dome.GetTextureFileAttr().Clear()
                source_dome.CreateTextureFormatAttr('latlong')
                source_dome.CreateColorAttr(Gf.Vec3f(1,1,1))
    # Material compilation and texture upload must finish before frame capture.
    for _ in range(60):app.update()
    def find_camera(name):
        return next(pr for pr in stage.Traverse() if pr.IsA(UsdGeom.Camera) and name in str(pr.GetPath()).split('/'))
    camera = find_camera(provenance['camera_name'])
    wide = find_camera(provenance['wide_camera']) if provenance.get('wide_camera') else camera
    dome = UsdLux.DomeLight.Define(stage, '/UnifiedFill')
    dome.CreateIntensityAttr(1800)
    dome.CreateColorAttr(Gf.Vec3f(.88, .93, 1.0))
    anchor=UsdGeom.XformCache(1).GetLocalToWorldTransform(camera).ExtractTranslation()
    shift=Gf.Vec3d(anchor[0]-.67,anchor[1]+1.55,0) if provenance['id'].startswith('g1_') else Gf.Vec3d(0)
    for name,position,power in [('Key',(-1.8,-2.6,3.7),15000),('Fill',(.2,1.5,3.3),9000)]:
        light=UsdLux.SphereLight.Define(stage,'/Unified'+name)
        light.CreateRadiusAttr(.75)
        light.CreateIntensityAttr(power)
        location=anchor+Gf.Vec3d(-.2,0,.6 if name=='Key' else 1.2) if provenance['id'].endswith('_room') else shift+Gf.Vec3d(*position)
        UsdGeom.Xformable(light).AddTranslateOp().Set(location)
    render_camera=UsdGeom.Camera.Define(stage,'/UnifiedCamera')
    camera_transform=render_camera.AddTransformOp()
    product = rep.create.render_product(str(render_camera.GetPath()), (1920, 1080))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    timeline = __import__('omni.timeline', fromlist=['get_timeline_interface']).get_timeline_interface()
    timeline.set_time_codes_per_second(30)
    timeline.pause()
    frames = provenance['frames']
    indices = sorted(set([1, frames // 2, frames])) if a.preview else range(1, min(frames,a.limit_frames or frames) + 1)
    encoder = None
    started = time.monotonic()
    denoise = None
    if a.oidn_library:
        from rfw_w.scenes.workers.isaac.episodes.rtx_denoise import Denoiser
        denoise = Denoiser(a.oidn_library, 1920, 1080)
    try:
        if not a.preview:
            assert not (out / 'task.mp4').exists()
            encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080',
                '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(out / 'task.mp4')], stdin=subprocess.PIPE)
        for frame in indices:
            selected=wide if frame<=min(90,frames//8) else camera
            time_code=frame-1+provenance.get('time_code_start',1)
            source_camera=UsdGeom.Camera(selected)
            render_camera.SetFromCamera(source_camera.GetCamera(time_code))
            # GetCamera returns the world transform, including its parent xforms.
            camera_transform.Set(UsdGeom.XformCache(time_code).GetLocalToWorldTransform(selected))
            timeline.set_current_time((frame - 1 + provenance.get('time_code_start', 1)) / 30)
            timeline.commit()
            rep.orchestrator.step(rt_subframes=1 if a.renderer=='pathtracing' else 4, pause_timeline=True, delta_time=0.0)
            image = np.asarray(rgb.get_data())[:, :, :3].copy()
            assert image.shape == (1080, 1920, 3) and image.max() > 0
            if denoise:
                image = denoise(image)
            if a.preview or frame in (1, frames):
                Image.fromarray(image).save(out / (f'preview_{frame:05}.png' if a.preview else 'poster.png' if frame == 1 else 'result.png'))
            if encoder:
                encoder.stdin.write(image.tobytes())
            if frame % 30 == 0 or frame == 1:
                print('RTX_PROGRESS', provenance['id'], frame, '/', frames, flush=True)
        if encoder:
            encoder.stdin.close()
            assert encoder.wait() == 0
            provenance.update(renderer='Isaac Sim 4.5 RTX '+('PathTracing' if a.renderer=='pathtracing' else 'RaytracedLighting'), resolution=[1920,1080], samples_per_pixel=a.samples if a.renderer=='pathtracing' else None,
                denoiser='OIDN CPU' if denoise else False, elapsed_s=time.monotonic()-started, video_sha256=hashlib.sha256((out/'task.mp4').read_bytes()).hexdigest(), status='diagnostic_prefix' if a.limit_frames else 'rendered_pending_media_check',
                scene_sha256=hashlib.sha256((out/'scene.usdc').read_bytes()).hexdigest())
            (out/'render_provenance.json').write_text(json.dumps(provenance,indent=2))
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if encoder and encoder.poll() is None:
            encoder.kill()
        app.close()


if __name__ == "__main__":
    main()
