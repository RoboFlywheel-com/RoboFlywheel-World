# SPDX-License-Identifier: Apache-2.0
"""Native-state render/data export with streaming video and measured throughput."""
import hashlib
import json
from pathlib import Path
import subprocess
import time


def render(world, run, output, *, full=False, keep_frames=False):
    import mujoco as mj
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation
    started = time.monotonic()
    world, run, output = Path(world), Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((run/'trajectory.json').read_text()); rows = raw['samples']
    validation = json.loads((run/'task_validation.json').read_text())
    model = mj.MjModel.from_xml_path(str(world/'world.xml')); data = mj.MjData(model)
    model.vis.global_.offwidth = 1280; model.vis.global_.offheight = 720
    renderer = mj.Renderer(model, height=720, width=1280)
    profile = json.loads((world/'render_profile.json').read_text()) if (world/'render_profile.json').is_file() else {}
    task = json.loads((world/'task_profile.json').read_text()) if (world/'task_profile.json').is_file() else {}
    task_id = task.get('task_id', 'T03')

    def camera_pose(name, position, target):
        z = np.array(position, dtype=float)-np.array(target); z /= np.linalg.norm(z)
        x = np.cross([0,0,1],z); x /= np.linalg.norm(x); y = np.cross(z,x)
        cid = model.camera(name).id; model.cam_pos[cid] = position
        model.cam_quat[cid] = Rotation.from_matrix(np.column_stack([x,y,z])).as_quat()[[3,0,1,2]]
    defaults = {'overview': {'position': [.6,-2.6,2.3], 'target': [-.95,0,.85]},
                'detail': {'position': [-.35,-1.35,2.25], 'target': [-1.02,0,.88], 'fovy':44}}
    for name, default in defaults.items():
        camera = profile.get('cameras', {}).get(name, default)
        camera_pose(name, camera['position'], camera['target'])
        if 'fovy' in camera: model.cam_fovy[model.camera(name).id] = camera['fovy']
    model.vis.headlight.ambient[:] = profile.get('headlight_ambient', .4)
    model.vis.headlight.diffuse[:] = profile.get('headlight_diffuse', .7)
    visibility = []
    tracked = ["V13_" + obj['leaf'] for obj in task['objects']] if task else ['V13_11_2_0007','V13_9_6_0000','cabinet_drawer_1']
    tracked += ['left_finger', 'right_finger']
    body_geom_ids = {name: [i for i in range(model.ngeom) if model.geom_bodyid[i]==model.body(name).id]
                     for name in tracked}
    phases = {}
    for i,row in enumerate(rows): phases.setdefault(row['phase'], []).append(i)
    keys = sorted(set([0,len(rows)-1]+[v[len(v)//2] for v in phases.values()]))
    indices = range(len(rows)) if full else keys
    frames = output/'frames'
    if keep_frames or not full: frames.mkdir()
    sensors = output/'sensors'; sensors.mkdir()
    geom_object_type = int(mj.mjtObj.mjOBJ_GEOM)
    options = mj.MjvOption(); options.geomgroup[3]=0; options.geomgroup[2]=1
    encoder = None
    encode_log = None
    preparation_s = time.monotonic()-started
    render_s = write_s = flush_s = 0.
    try:
        if full and not keep_frames:
            encode_log = (output/'encoder.log').open('wb')
            encoder = subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24',
                '-s','1280x720','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-crf','20',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(output/'task.mp4')],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encode_log)
        for index in indices:
            tick = time.monotonic()
            row = rows[index]; data.qpos[:] = row['qpos']; mj.mj_forward(model,data)
            camera = 'overview' if row['phase'] in ('initial_settle','settle') else 'detail'
            renderer.update_scene(data,camera=camera,scene_option=options)
            array = renderer.render().copy(); img = Image.fromarray(array)
            renderer.enable_segmentation_rendering(); renderer.update_scene(data,camera=camera,scene_option=options)
            seg = renderer.render(); renderer.disable_segmentation_rendering()
            valid = (seg[:,:,1]==geom_object_type)&(seg[:,:,0]>=0)
            histogram = np.bincount(seg[:,:,0][valid].astype(np.int32), minlength=model.ngeom)
            visible = {name: sum(int(histogram[g]) for g in ids) for name,ids in body_geom_ids.items()}
            visibility.append(dict(t=row['t'],phase=row['phase'],camera=camera,pixels=visible))
            if index in keys:
                renderer.enable_depth_rendering(); renderer.update_scene(data,camera=camera,scene_option=options)
                depth = renderer.render().copy(); renderer.disable_depth_rendering()
                np.savez_compressed(sensors/f'{index:05d}.npz',depth_m=depth,instance_geom_id=seg[:,:,0],instance_object_type=seg[:,:,1])
                Image.fromarray(array).save(sensors/f'{index:05d}_rgb.png')
            draw = ImageDraw.Draw(img); draw.rectangle((0,0,1280,38),fill=(15,22,26))
            evidence_label = validation.get('display_label', validation['status'].upper())
            draw.text((16,10), f"{task_id} | {evidence_label} | {row['phase']} | {row['t']:.2f}s",fill='white')
            render_s += time.monotonic()-tick
            tick = time.monotonic()
            if encoder: encoder.stdin.write(img.tobytes())
            else: img.save(frames/f'{index:05d}.png')
            write_s += time.monotonic()-tick
        tick = time.monotonic()
        if encoder:
            encoder.stdin.close()
            if encoder.wait()!=0: raise RuntimeError('REFERENCE_VIDEO_ENCODER_FAILED')
        elif full:
            subprocess.run(['ffmpeg','-v','error','-framerate','30','-i',str(frames/'%05d.png'),
                            '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'task.mp4')],check=True)
        flush_s = time.monotonic()-tick
    except BaseException:
        if encoder and encoder.poll() is None:
            encoder.kill(); encoder.wait()
        raise
    finally:
        if encoder and not encoder.stdin.closed: encoder.stdin.close()
        if encode_log: encode_log.close()
        renderer.close()
    (output/'visibility.json').write_text(json.dumps(visibility,indent=2))
    calibration = []
    for name in ('overview','detail'):
        cid = model.camera(name).id
        fy = 720/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
        calibration.append(dict(name=name,position_world_m=model.cam_pos[cid].tolist(),quaternion_wxyz=model.cam_quat[cid].tolist(),
            intrinsics=[[fy,0,640],[0,fy,360],[0,0,1]],resolution=[1280,720],camera_axes='MuJoCo x right, y up, -z forward'))
    (output/'calibration.json').write_text(json.dumps(calibration,indent=2))
    timing = dict(preparation_s=preparation_s,render_sensor_s=render_s,frame_write_s=write_s,encoder_flush_s=flush_s,
                  elapsed_s=time.monotonic()-started,video_transport=('png_sequence' if keep_frames else 'raw_rgb_pipe') if full else 'phase_pngs',
                  png_frames_retained=keep_frames or not full)
    (output/'timing.json').write_text(json.dumps(timing,indent=2))
    (output/'media.json').write_text(json.dumps(dict(status='rendered_pending_media_check',
        task_status=validation['status'],source_trajectory_sha256=hashlib.sha256((run/'trajectory.json').read_bytes()).hexdigest(),
        source_world_sha256=hashlib.sha256((world/'world.xml').read_bytes()).hexdigest(),
        render_profile_sha256=hashlib.sha256((world/'render_profile.json').read_bytes()).hexdigest() if profile else None,
        task_id=task_id,tracked_objects=task.get('objects'),fps=30,frames=len(rows) if full else len(keys),native_state=True,cameras=['overview','detail'],visibility_ref='visibility.json',
        renderer='MuJoCo native OpenGL',renderer_version=mj.__version__,timing_ref='timing.json',
        rgb_fps=30 if full else 'phase_keyframes',depth_instance_sampling='phase_keyframes',calibration_ref='calibration.json'),indent=2))
    return timing
