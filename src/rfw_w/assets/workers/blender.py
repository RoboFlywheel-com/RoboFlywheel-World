# SPDX-License-Identifier: Apache-2.0
"""Blender-side script: render a normalized GLB from fixed camera views."""

import argparse
import math
import os
import sys

import bpy
import mathutils

VIEW_PRESETS = {
    "front": (0.0, 0.45, 1.8),
    "back": (math.pi, 0.45, 1.8),
    "left": (-math.pi / 2, 0.45, 1.8),
    "right": (math.pi / 2, 0.45, 1.8),
    "bottom": (0.0, -math.pi / 2 + 0.1, 2.0),
}


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def normalize_object(obj):
    bpy.context.view_layer.update()
    bounds = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    lower = mathutils.Vector(
        tuple(min(point[axis] for point in bounds) for axis in range(3))
    )
    upper = mathutils.Vector(
        tuple(max(point[axis] for point in bounds) for axis in range(3))
    )
    longest = max(upper - lower)
    if longest > 0:
        obj.location -= (lower + upper) * 0.5
        obj.scale *= 1.0 / longest
        bpy.context.view_layer.update()


def setup_camera(yaw, pitch, distance):
    old_camera = bpy.context.scene.camera
    if old_camera:
        bpy.data.objects.remove(old_camera, do_unlink=True)
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 35
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    x = distance * math.cos(pitch) * math.sin(yaw)
    y = distance * math.cos(pitch) * math.cos(yaw)
    z = distance * math.sin(pitch)
    camera.location = (x, y, z)
    direction = mathutils.Vector((-x, -y, -z)).normalized()
    up = mathutils.Vector((0, 0, 1))
    if abs(direction.dot(up)) > 0.99:
        up = mathutils.Vector((0, 1, 0))
    right = direction.cross(up).normalized()
    corrected_up = right.cross(direction).normalized()
    rotation = mathutils.Matrix(
        (
            (right.x, corrected_up.x, -direction.x),
            (right.y, corrected_up.y, -direction.y),
            (right.z, corrected_up.z, -direction.z),
        )
    )
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = rotation.to_quaternion()
    bpy.context.scene.camera = camera


def setup_scene(resolution, samples):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.compression = 15
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.8, 0.8, 0.8)
    for name, energy, size, location in (
        ("Key", 500, 2.0, (2.0, -2.0, 3.0)),
        ("Fill", 200, 2.5, (-2.5, -1.5, 2.0)),
        ("Rim", 300, 1.5, (0.0, 3.0, 2.5)),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy, light_data.shape, light_data.size = energy, "DISK", size
        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = location
        direction = -light.location
        light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--views", default="front,left,right")
    args = parser.parse_args(argv)
    clear_scene()
    setup_scene(args.resolution, args.samples)
    bpy.ops.import_scene.gltf(filepath=args.glb)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in {args.glb}")
    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    normalize_object(bpy.context.active_object)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.glb))[0]
    for view in args.views.split(","):
        view = view.strip()
        setup_camera(*VIEW_PRESETS[view])
        bpy.context.scene.render.filepath = os.path.join(
            args.out_dir, f"{stem}_{view}.png"
        )
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
