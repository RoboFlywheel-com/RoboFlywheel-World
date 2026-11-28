# SPDX-License-Identifier: Apache-2.0
"""Blender (bpy) worker: render 360-degree turntable videos for exported OBJ assets.

Run with the scenesmith conda env python (bpy 4.5.x):
    CUDA_VISIBLE_DEVICES=0 python turntable_worker.py --shard shard_0.json --log worker_0.log

Each shard entry: {"asset_key": str, "obj": path, "out_mp4": path}
Frames are rendered to local /tmp and encoded to mp4 (libx264 via imageio-ffmpeg),
then the frames are deleted. Assets that fail are logged and skipped.
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback

import bpy
from mathutils import Vector

FPS = 24
N_FRAMES = int(os.environ.get("TT_FRAMES", "96"))  # 96 / 24 fps = 4.0 s per loop
RESOLUTION = int(os.environ.get("TT_RES", "1024"))
SAMPLES = int(os.environ.get("TT_SAMPLES", "32"))
TARGET_SIZE = 1.0  # longest bbox edge after normalization
ELEVATION_DEG = 20.0


def log(msg):
    print(f"[turntable {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "OPTIX"
    except Exception:
        prefs.compute_device_type = "CUDA"
    prefs.refresh_devices()
    devices = prefs.get_devices_for_type(prefs.compute_device_type)
    for dev in devices:
        dev.use = dev.type != "CPU"
    log("cycles devices: " + ", ".join(f"{d.name}(use={d.use})" for d in devices))


def import_and_normalize(obj_path):
    bpy.ops.wm.obj_import(filepath=obj_path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh in {obj_path}")
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lower = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
    upper = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
    size = upper - lower
    longest = max(size)
    if longest <= 0:
        raise RuntimeError(f"degenerate bounds in {obj_path}")
    scale = TARGET_SIZE / longest
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lower = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
    upper = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
    center = (lower + upper) * 0.5
    # center in XY, rest the bottom on the ground plane (z = 0)
    obj.location -= Vector((center.x, center.y, lower.z))
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lower = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
    upper = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
    return obj, lower, upper


def add_ground_plane():
    bpy.ops.mesh.primitive_plane_add(size=30.0, location=(0, 0, 0))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.82, 0.82, 0.84, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    plane.data.materials.append(mat)


def add_area_light(name, energy, size, location):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat("-Z", "Y").to_euler()


def setup_world_and_lights():
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.55, 0.55, 0.58, 1.0)
    bg.inputs[1].default_value = 0.5
    add_area_light("Key", 500, 2.0, (2.4, -2.0, 3.0))
    add_area_light("Fill", 250, 2.5, (-2.6, -1.4, 1.8))
    add_area_light("Rim", 400, 1.5, (0.2, 2.6, 2.4))


def setup_camera(lower, upper):
    center = (lower + upper) * 0.5
    diag = (upper - lower).length
    radius = diag * 0.5
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50.0
    half_fov = math.atan(cam_data.sensor_width / (2.0 * cam_data.lens))
    distance = radius / math.tan(half_fov) * 1.08
    elev = math.radians(ELEVATION_DEG)
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    # camera orbits the vertical axis through the bbox center
    pivot = bpy.data.objects.new("Pivot", None)
    bpy.context.collection.objects.link(pivot)
    pivot.location = center
    cam.location = (0.0, -distance * math.cos(elev), distance * math.sin(elev))
    cam.parent = pivot
    # aim at pivot
    direction = pivot.location - (pivot.location + cam.location)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    # 0 -> 360 deg between frame 1 and N_FRAMES+1 so the loop has no seam frame
    pivot.rotation_mode = "XYZ"
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    pivot.keyframe_insert("rotation_euler", frame=1, index=2)
    pivot.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    pivot.keyframe_insert("rotation_euler", frame=N_FRAMES + 1, index=2)
    action = pivot.animation_data.action
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    bpy.context.scene.camera = cam


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = (
        True  # OIDN (CPU); OptiX denoiser unavailable in bpy wheel
    )
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = N_FRAMES


def encode_video(frame_dir, out_mp4):
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    # ffmpeg's mp4 muxer seeks back to finalize the trailer; network-fs mounts
    # rejects that, so encode to local disk and copy the finished file over.
    fd, tmp_mp4 = tempfile.mkstemp(suffix=".mp4", prefix="tt_enc_")
    os.close(fd)
    writer = imageio.get_writer(
        tmp_mp4,
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    try:
        for f in range(1, N_FRAMES + 1):
            writer.append_data(imageio.imread(os.path.join(frame_dir, f"{f:04d}.png")))
    finally:
        writer.close()
    with open(tmp_mp4, "rb") as fh:
        if fh.read().find(b"moov") < 0:
            raise RuntimeError(f"encoded mp4 missing moov atom: {tmp_mp4}")
    shutil.copyfile(tmp_mp4, out_mp4)
    os.remove(tmp_mp4)
    with open(out_mp4, "rb") as fh:
        if fh.read().find(b"moov") < 0:
            raise RuntimeError(f"copied mp4 missing moov atom: {out_mp4}")


def render_asset(entry):
    reset_scene()
    configure_gpu()  # read_factory_settings wipes compute_device_type; re-pin every asset
    configure_render()
    _, lower, upper = import_and_normalize(entry["obj"])
    add_ground_plane()
    setup_world_and_lights()
    setup_camera(lower, upper)
    frame_dir = tempfile.mkdtemp(prefix=f"tt_{entry['asset_key']}_")
    try:
        for f in range(1, N_FRAMES + 1):
            bpy.context.scene.frame_set(f)
            bpy.context.scene.render.filepath = os.path.join(frame_dir, f"{f:04d}.png")
            bpy.ops.render.render(write_still=True)
        encode_video(frame_dir, entry["out_mp4"])
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--done-log", required=True)
    args = parser.parse_args()

    with open(args.shard) as f:
        entries = json.load(f)
    done = set()
    if os.path.exists(args.done_log):
        with open(args.done_log) as f:
            done = {line.strip() for line in f if line.strip()}

    for entry in entries:
        key = entry["asset_key"]
        if key in done:
            continue
        t0 = time.time()
        try:
            render_asset(entry)
        except Exception:
            log(f"FAIL {key}\n{traceback.format_exc()}")
            if os.path.exists(entry["out_mp4"]):
                os.remove(entry["out_mp4"])  # drop partial output so a retry re-encodes
            continue
        with open(args.done_log, "a") as f:
            f.write(key + "\n")
        log(f"OK {key} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
