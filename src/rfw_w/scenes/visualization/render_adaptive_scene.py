# SPDX-License-Identifier: Apache-2.0
"""Read-only Blender preview of one saved G2/G3 workflow result.

Run under the scenesmith interpreter. This script does not change the source
bundle; its cutaway cameras only hide exterior wall groups in the render.
"""

from __future__ import annotations


def run():

    import hashlib
    import json
    from pathlib import Path
    import sys
    import time

    import bpy
    from mathutils import Vector
    from PIL import Image, ImageDraw, ImageFont


    SOURCE = Path(sys.argv[1]).resolve()
    OUTPUT = Path(sys.argv[2]).resolve()
    ASSETS = SOURCE / "assets" if (SOURCE / "assets").is_dir() else SOURCE


    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()


    def material(name: str, colour: tuple[float, float, float, float]):
        value = bpy.data.materials.new(name)
        value.diffuse_color = colour
        value.use_nodes = True
        principled = value.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = colour
        principled.inputs["Roughness"].default_value = 0.82
        return value


    def pbr_material(mid, value):
        mat = material(mid, tuple(value['base_color_rgba']))
        nodes = mat.node_tree.nodes; links = mat.node_tree.links
        principled = nodes.get('Principled BSDF')
        principled.inputs['Roughness'].default_value = value['roughness']
        for channel, info in value['channels'].items():
            texture = nodes.new('ShaderNodeTexImage')
            path = ASSETS / info['ref']
            assert digest(path) == info['sha256']
            texture.image = bpy.data.images.load(str(path), check_existing=False)
            texture.image.colorspace_settings.name = info['color_space']
            if channel == 'normal_gl':
                normal = nodes.new('ShaderNodeNormalMap')
                links.new(texture.outputs['Color'], normal.inputs['Color'])
                links.new(normal.outputs['Normal'], principled.inputs['Normal'])
            else:
                links.new(texture.outputs['Color'], principled.inputs['Base Color' if channel == 'base_color' else 'Roughness'])
        return mat


    def load_shell(path: Path, appearance=None):
        vertices: list[tuple[float, float, float]] = []
        groups: dict[str, list[list[int]]] = {}
        current = None
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "g":
                current = parts[1]
                groups[current] = []
            elif parts[0] == "f":
                groups[current].append([int(x.split("/")[0]) - 1 for x in parts[1:]])
        assert len(groups) == 11, f"unexpected house group count: {len(groups)}"
        floor_kitchen = material("presentation floor kitchen", (0.77, 0.84, 0.78, 1))
        floor_study = material("presentation floor study", (0.75, 0.80, 0.86, 1))
        wall = material("presentation walls", (0.93, 0.92, 0.88, 1))
        hidden = []; mapping = []
        pbr = {mid:pbr_material(mid,value) for mid,value in appearance['materials'].items()} if appearance else {}
        bindings = {b['surface_id']:b for b in appearance['bindings']} if appearance else {}
        for name, faces in groups.items():
            used = sorted({index for face in faces for index in face})
            remap = {global_index: local_index for local_index, global_index in enumerate(used)}
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata([vertices[index] for index in used], [],
                             [[remap[index] for index in face] for face in faces])
            mesh.update()
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.collection.objects.link(obj)
            obj.data.materials.append(floor_kitchen if name == "floor_left" else
                                      floor_study if name == "floor_right" else wall)
            if appearance:
                mesh.uv_layers.new(name='surface_meters')
                for surface in (s for s in appearance['surfaces'].values() if s['mesh_group'] == name):
                    binding = bindings.get(surface['surface_id'])
                    if not binding: continue
                    obj.data.materials.append(pbr[binding['material_id']])
                    material_index = len(obj.data.materials)-1
                    axis_index = {'x':0,'y':1,'z':2}[surface['axis']]
                    uv_axes = [{'x':0,'y':1,'z':2}[a] for a in surface['uv_axes']]
                    matched = []
                    for poly in mesh.polygons:
                        # The author's box winding is inverted; identify the exact plane,
                        # independent of face normal, and assign only this room's inner side.
                        if not all(abs(mesh.vertices[i].co[axis_index]-surface['plane_m'])<1e-5 for i in poly.vertices): continue
                        poly.material_index = material_index; matched.append(poly.index)
                        for loop_index in poly.loop_indices:
                            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
                            mesh.uv_layers.active.data[loop_index].uv = tuple((co[a]-origin)/binding['meters_per_tile']
                                for a,origin in zip(uv_axes,surface['uv_origin_m']))
                    assert matched, surface['surface_id']
                    mapping.append({'surface_id':surface['surface_id'],'material_id':binding['material_id'],
                                    'mesh_group':name,'polygon_indices':matched,'meters_per_tile':binding['meters_per_tile']})
            if name in {"wall_bottom_left", "wall_bottom_right", "wall_left", "wall_right"}:
                obj.hide_render = True
                hidden.append(name)
        return sorted(hidden), mapping


    def area(name, xyz, energy, size, target=(0, 1.5, 0)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = xyz
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


    def camera(name, eye, target, scale):
        data = bpy.data.cameras.new(name)
        data.type = "ORTHO"
        data.ortho_scale = scale
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = eye
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
        return obj


    def main():
        started = time.monotonic()
        assert OUTPUT.is_dir() and not any(OUTPUT.glob("*.png")), "fresh render output required"
        graph = json.loads((ASSETS / "graph.json").read_text())
        ledger = json.loads((ASSETS / "core_asset_ledger.json").read_text())
        background = json.loads((ASSETS / "background_ledger.json").read_text())
        ledger["instances"] += background["instances"]
        house_path = ASSETS / "geometry/house.obj"
        assert digest(house_path) == graph["built"]["house_shell"]["mesh_sha256"]

        bpy.ops.wm.read_factory_settings(use_empty=True)
        appearance = None
        if 'surface_appearance' in graph['built']:
            from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance
            verify_surface_appearance(ASSETS)
            appearance = json.loads((ASSETS/graph['built']['surface_appearance']['ledger_ref']).read_text())
        hidden, surface_mapping = load_shell(house_path, appearance)
        checked = []
        for item in ledger["instances"]:
            visual = ASSETS / item["visual_ref"]
            assert digest(visual) == item["visual_sha256"], item["instance_id"]
            bpy.ops.wm.obj_import(filepath=str(visual), forward_axis="Y", up_axis="Z")
            meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
            assert meshes, item["instance_id"]
            for obj in meshes:
                obj.location += Vector(item["pose_xyz_m"])
                obj.rotation_euler.z += item["yaw_rad"]
            checked.append({"instance_id": item["instance_id"], "role_id": item["role_id"],
                            "room_id": item["room_id"], "mesh_parts": len(meshes),
                            "visual_sha256": item["visual_sha256"]})

        covering_mapping = []
        if 'surface_coverings' in graph['built']:
            from rfw_w.scenes.orchestrator.surface_covering_tools import verify_surface_coverings
            verify_surface_coverings(ASSETS)
            covering = json.loads((ASSETS/'surface_coverings.json').read_text())
            mats = {mid:pbr_material(mid,value) for mid,value in covering['materials'].items()}
            for item in covering['instances']:
                mesh=bpy.data.meshes.new(item['instance_id'])
                mesh.from_pydata(item['vertices_m'],[],[[0,1,2,3]]);mesh.update()
                obj=bpy.data.objects.new(item['instance_id'],mesh);bpy.context.collection.objects.link(obj)
                mesh.materials.append(mats[item['material_id']]);mesh.uv_layers.new()
                for i,uv in enumerate(((0,0),(1,0),(1,1),(0,1))):mesh.uv_layers.active.data[i].uv=uv
                # Rectangles use cover UVs; normal orientation is also verified from the anchor.
                expected=Vector(item['normal'])
                if mesh.polygons[0].normal.dot(expected)<0:
                    import bmesh
                    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.reverse_faces(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
                covering_mapping.append({'instance_id':item['instance_id'],'surface_id':item['surface_id'],'material_id':item['material_id']})

        world = bpy.data.worlds.new("preview daylight")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs[0].default_value = (0.72, 0.79, 0.86, 1)
        world.node_tree.nodes["Background"].inputs[1].default_value = 0.7
        bpy.context.scene.world = world
        area("large softbox", (0, -2, 9), 1500, 8)
        area("back fill", (-4, 5, 7), 800, 7)

        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 24
        scene.cycles.use_denoising = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 16
        scene.view_settings.view_transform = "AgX"

        views = [
            ("whole_house.png", (7.0, -7.0, 11.0), (0, 1.7, 0.9), 11.2, 1400, 900),
            ("kitchen.png", (-5.3, -4.3, 7.7), (-2.0, 1.35, 0.4), 5.6, 1000, 850),
            ("study.png", (5.3, -4.3, 7.7), (2.0, 1.35, 0.4), 5.6, 1000, 850),
            ("kitchen_detail.png", (-4.8, -1.8, 3.2), (-2.9, 0.88, 0.75), 1.9, 1000, 700),
            ("study_detail.png", (4.8, -1.8, 3.2), (2.95, 0.58, 0.65), 1.9, 1000, 700),
        ]
        rendered = []
        for filename, eye, target, scale, width, height in views:
            scene.camera = camera(filename, eye, target, scale)
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            scene.render.filepath = str(OUTPUT / filename)
            bpy.ops.render.render(write_still=True)
            result = OUTPUT / filename
            assert result.is_file() and result.stat().st_size > 1000
            rendered.append({"file": filename, "sha256": digest(result), "size_px": [width, height],
                             "camera_eye_m": eye, "camera_target_m": target})

        sheet = Image.new("RGB", (1600, 1920), "#eef1f3")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        placements = [("whole_house.png", (180, 54, 1240, 797), "Whole house"),
                      ("kitchen.png", (60, 900, 710, 452), "Kitchen: task core and background"),
                      ("study.png", (830, 900, 710, 452), "Study: task core and background"),
                      ("kitchen_detail.png", (60, 1400, 710, 470), "Kitchen close-up"),
                      ("study_detail.png", (830, 1400, 710, 470), "Study close-up")]
        for filename, (x, y, width, height), title in placements:
            picture = Image.open(OUTPUT / filename).convert("RGB")
            picture.thumbnail((width, height - 28), Image.Resampling.LANCZOS)
            sheet.paste(picture, (x + (width - picture.width) // 2, y + 28))
            draw.text((x, y), title, fill="#25313b", font=font)
        draw.text((25, 18), "Adaptive scene Agent | static Blender preview", fill="#25313b", font=font)
        sheet.save(OUTPUT / "contact_sheet.png")

        report = {"status": "passed", "source_workflow_graph_hash": graph["content_hash"],
                  "house_mesh_sha256": digest(house_path), "asset_ledger_hash": ledger["content_hash"],
                  "render_engine": f"Blender Cycles CPU {bpy.app.version_string}",
                  "rendered": rendered, "asset_instances": checked,
                  "camera_only_hidden_wall_groups": hidden,
                  "floor_and_wall_colours": "bound PBR and explicit solids" if appearance else "preview-only presentation materials",
                  "surface_appearance_hash": appearance['content_hash'] if appearance else None,
                  "surface_binding_mapping": surface_mapping,
                  "covering_mapping": covering_mapping,
                  "asset_materials": "imported from prepared OBJ/MTL with local textures",
                  "dynamics": "not_run", "robot_navigation": "not_run", "robot_grasp": "not_run",
                  "elapsed_s": round(time.monotonic() - started, 3)}
        (OUTPUT / "render_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "render_engine", "elapsed_s")}, sort_keys=True))


    if __name__ == "__main__":
        main()


if __name__ == "__main__":
    run()
