# SPDX-License-Identifier: Apache-2.0
"""Explicit architectural mappings for Infinigen 892947be71c4ff102c6a558d088ffe149561c11f.

Blender-only adapter; importing it does not import bpy or the upstream tree.
"""
import hashlib
import json
import random

MAPPING_VERSION = 'pvscene.infinigen-materials/v2'
BAKE_VERSION = 'pvscene.architecture-bake/v3'
TOKENS = {'oak': 'wood', 'walnut': 'wood', 'off_white': 'plaster',
          'linen': 'plaster', 'muted_green': 'plaster', 'sage': 'plaster',
          'tile': 'tile', 'ceramic_tile': 'tile'}


def mapping(token, role):
    if token not in TOKENS or role not in ('floor', 'wall', 'ceiling'):
        raise ValueError(f'MATERIAL_INTENT_UNSUPPORTED({role}:{token})')
    return TOKENS[token]


def material_summary(material):
    """Parameter fingerprint independent of Blender's automatic material-name suffixes."""
    rows = []
    tree = material.node_tree
    for node in tree.nodes:
        inputs = {}
        for socket in node.inputs:
            if not hasattr(socket, 'default_value'):
                continue
            value = socket.default_value
            if not isinstance(value, (str, int, float, bool)):
                try:
                    value = list(value)
                except TypeError:
                    value = str(type(value))
            inputs[socket.identifier] = value
        row = {'name': node.name, 'type': node.bl_idname, 'inputs': inputs}
        for prop in ('operation', 'blend_type', 'space', 'uv_map'):
            if hasattr(node, prop):
                row[prop] = getattr(node, prop)
        if hasattr(node, 'color_ramp'):
            row['ramp'] = [(e.position, list(e.color)) for e in node.color_ramp.elements]
        rows.append(row)
    payload = {'nodes': rows, 'links': sorted((l.from_node.name, l.from_socket.identifier,
               l.to_node.name, l.to_socket.identifier) for l in tree.links)}
    return {'sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            'parameters': payload}


def make_material(token, role, seed):
    family = mapping(token, role)
    import numpy as np
    from infinigen.assets.materials.wood.hardwood_floor import HardwoodFloor
    from infinigen.assets.materials.ceramic.plaster import Plaster
    from infinigen.assets.materials.ceramic.tile import Tile
    from infinigen.assets.materials.ceramic.ceramic import shader_ceramic
    from rfw_w.scenes.solvers.materials import PALETTE_LIBRARY, CEILING_LIGHTEN

    derived = int.from_bytes(hashlib.sha256(f'{MAPPING_VERSION}:{seed}:{role}:{token}'.encode()).digest()[:4], 'big')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(derived)
        np.random.seed(derived)
        if family == 'wood':
            mat = HardwoodFloor().generate()
        elif family == 'plaster':
            mat = Plaster().generate(plaster_colored=False)
        else:
            mat = Tile().generate(shader_func=shader_ceramic, shape='square', scale=2, alternating=False)
        # Preserve grain/plaster variation while explicitly consuming the requested colour.
        colour = PALETTE_LIBRARY[token].surface.colour if token in PALETTE_LIBRARY else (0.8, 0.8, 0.8)
        if role == 'ceiling':
            colour = tuple(c + (1-c)*CEILING_LIGHTEN for c in colour)
        tree = mat.node_tree
        if family == 'wood':
            # Pinned wood.py passes its sampled bark colour to HueSaturation.Color and
            # Mix.B. Multiplying an already near-black bark colour cannot produce oak.
            # Replace those authored colour constants; leave grain/plank nodes intact.
            colour_nodes = [n for n in tree.nodes if n.type == 'HUE_SAT']
            if len(colour_nodes) != 1:
                raise ValueError('MATERIAL_CHANNEL_UNSUPPORTED(wood,base colour layout)')
            sampled = tuple(colour_nodes[0].inputs['Color'].default_value)
            for node in tree.nodes:
                for socket in node.inputs:
                    if socket.type == 'RGBA' and not socket.is_linked and tuple(socket.default_value) == sampled:
                        socket.default_value = (*colour, 1)
        for node in list(tree.nodes):
            if node.type != 'BSDF_PRINCIPLED' or family == 'wood':
                continue
            target = node.inputs['Base Color']
            if target.is_linked:
                source = target.links[0].from_socket
                mix = tree.nodes.new('ShaderNodeMixRGB')
                mix.blend_type = 'MULTIPLY'
                mix.inputs[0].default_value = 1
                mix.inputs[2].default_value = (*colour, 1)
                tree.links.new(source, mix.inputs[1])
                tree.links.new(mix.outputs[0], target)
            else:
                target.default_value = (*colour, 1)
        mat.name = f'{role}_{token}_{derived}'
        return mat, {'mapping_version': MAPPING_VERSION, 'token': token, 'role': role,
                     'family': family, 'seed': seed, 'derived_seed': derived,
                     'colour_linear': list(colour),
                     'colour_application': 'replace_source_colour' if family == 'wood' else 'multiply',
                     **material_summary(mat)}
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def convert_window_material(mat):
    """Recognise the pinned ray-dependent window graph, preserving camera reflection."""
    tree = mat.node_tree
    expected = {'Material Output': 'OUTPUT_MATERIAL', 'Transparent BSDF': 'BSDF_TRANSPARENT',
                'Glossy BSDF': 'BSDF_GLOSSY', 'Mix Shader': 'MIX_SHADER',
                'Light Path': 'LIGHT_PATH', 'Mix Shader.001': 'MIX_SHADER'}
    # Blender 4.2 exposes Glossy as anisotropic at the RNA class level, but keeps BSDF_GLOSSY.
    actual = {n.name: n.type for n in tree.nodes}
    links = {(l.from_node.name, l.from_socket.name, l.to_node.name, l.to_socket.identifier)
             for l in tree.links}
    expected_links = {
        ('Glossy BSDF', 'BSDF', 'Mix Shader', 'Shader_001'),
        ('Transparent BSDF', 'BSDF', 'Mix Shader', 'Shader'),
        ('Light Path', 'Is Camera Ray', 'Mix Shader.001', 'Fac'),
        ('Mix Shader', 'Shader', 'Mix Shader.001', 'Shader_001'),
        ('Transparent BSDF', 'BSDF', 'Mix Shader.001', 'Shader'),
        ('Mix Shader.001', 'Shader', 'Material Output', 'Surface')}
    if actual != expected or links != expected_links:
        raise ValueError(f'MATERIAL_CHANNEL_UNSUPPORTED({mat.name},window graph)')
    glossy = tree.nodes['Glossy BSDF']
    colour = tuple(glossy.inputs['Color'].default_value)
    roughness = glossy.inputs['Roughness'].default_value
    weight = tree.nodes['Mix Shader'].inputs[0].default_value
    converted = mat.copy()
    converted.name = 'pvscene_window_reflection'
    nodes = converted.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = colour
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = 1
    bsdf.inputs['Alpha'].default_value = weight
    output = nodes.new('ShaderNodeOutputMaterial')
    converted.node_tree.links.new(bsdf.outputs[0], output.inputs['Surface'])
    return converted, {'method': 'camera_reflection_alpha', 'opacity': weight,
                       'colour_linear': list(colour), 'roughness': roughness,
                       'limitation': 'ray-type branch and exact glossy Fresnel not preserved'}


def bake_roughness(obj, out, resolution, prefix):
    """Bake the input field, including when Cycles removes a zero-specular lobe."""
    import bpy
    from infinigen.tools import export

    overrides = []

    def override(tree, surface, material_name):
        if surface is None or len(surface.links) != 1:
            raise ValueError(f'MATERIAL_CHANNEL_UNSUPPORTED({material_name},roughness surface graph)')
        original = surface.links[0].from_socket
        shader = original.node
        if shader.type == 'GROUP':
            inner = shader.node_tree
            output = next((n for n in inner.nodes if n.type == 'GROUP_OUTPUT' and n.is_active_output), None)
            target = next((s for s in output.inputs if s.identifier == original.identifier), None) if output else None
            override(inner, target, material_name)
        elif shader.type == 'MIX_SHADER':
            # Pinned tile selects between ceramic shaders. Keep the authored mask
            # and sample each branch's scalar before mixing, just as for colour.
            for target in list(shader.inputs)[1:]:
                override(tree, target, material_name)
        elif shader.type == 'BSDF_PRINCIPLED':
            source = shader.inputs['Roughness']
            emit = tree.nodes.new('ShaderNodeEmission')
            overrides.append((tree, surface, original, emit))
            if source.is_linked:
                tree.links.new(source.links[0].from_socket, emit.inputs['Color'])
            else:
                value = source.default_value
                emit.inputs['Color'].default_value = (value, value, value, 1)
            tree.links.new(emit.outputs[0], surface)
        else:
            raise ValueError(f'MATERIAL_CHANNEL_UNSUPPORTED({material_name},roughness:{shader.type})')

    try:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            output = mat.node_tree.nodes.get('Material Output')
            override(mat.node_tree, output.inputs['Surface'] if output else None, mat.name)
        # Use the upstream image/node writer, but an EMIT pass preserves intrinsic
        # roughness even when the source shader has no specular contribution.
        export.bake_pass(obj, out, resolution, 'EMIT', True, prefix)
        (out / f'{prefix}_EMIT.png').rename(out / f'{prefix}_ROUGHNESS.png')
        for slot in obj.material_slots:
            if slot.material:
                node = slot.material.node_tree.nodes['EMIT_node']
                node.name = 'ROUGHNESS_node'
                node.image.filepath_raw = str(out / f'{prefix}_ROUGHNESS.png')
    finally:
        for tree, surface, original, emit in reversed(overrides):
            tree.nodes.remove(emit)
            tree.links.new(original, surface)


def bake_surface(obj, out, resolution, prefix):
    """Reuse pinned bake passes with a padded atlas and per-material parameter restore."""
    import bpy
    from infinigen.tools import export
    from infinigen.core.util import blender as butil

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # Smart-project's zero default margin lets filtering sample neighbouring islands.
    # Reserve 16 texels between islands for the 8-texel bake dilation on either side.
    obj.data.uv_layers.new(name='ExportUV')
    obj.data.uv_layers['ExportUV'].active = True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=0.7, island_margin=16/resolution,
                             margin_method='FRACTION')
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.render.bake.margin = 8
    with butil.SelectObjects(obj):
        for slot in obj.material_slots:
            if slot.material is not None:
                slot.material = export.deep_copy_material(slot.material)
        export.process_glass_materials(obj, True)
        # Bake intrinsic colour/normal, not coverage-weighted values. Restore coverage
        # on the final shader; otherwise transparent panes acquire near-black albedo.
        alpha_values = {}
        for slot in obj.material_slots:
            if slot.material is None:
                continue
            node = slot.material.node_tree.nodes.get('Principled BSDF')
            if node and not node.inputs['Alpha'].is_linked:
                alpha_values[slot.material.name] = node.inputs['Alpha'].default_value
                node.inputs['Alpha'].default_value = 1
        for channel in export.SPECIAL_BAKE:
            export.bake_special_emit(obj, out, resolution, True, channel, prefix)
        parameters = {}
        for slot in obj.material_slots:
            if slot.material is not None:
                parameters.update(export.remove_params(slot.material, slot.material.node_tree))
        for channel in export.BAKE_TYPES:
            if channel == 'ROUGHNESS':
                bake_roughness(obj, out, resolution, prefix)
            else:
                export.bake_pass(obj, out, resolution, channel, True, prefix)
        # Pinned bake_pass saves scalar channels in an sRGB image. Merely labelling
        # that PNG Non-Color later turns roughness .25 into .537 (regression probe 01).
        # Decode the stored transfer curve once, then write actual linear data texels.
        for channel in ('ROUGHNESS', 'METAL', 'TRANSMISSION'):
            path = out / f'{prefix}_{channel}.png'
            if not path.exists():
                continue
            encoded = bpy.data.images.load(str(path), check_existing=False)
            encoded.colorspace_settings.name = 'Non-Color'
            values = list(encoded.pixels)
            for offset in range(0, len(values), 4):
                for component in range(3):
                    value = values[offset+component]
                    values[offset+component] = value/12.92 if value <= .04045 else ((value+.055)/1.055)**2.4
            linear = bpy.data.images.new(f'{prefix}_{channel}_linear', resolution, resolution)
            linear.colorspace_settings.name = 'Non-Color'
            linear.pixels[:] = values
            linear.filepath_raw = str(path)
            linear.file_format = 'PNG'
            linear.save()
            for slot in obj.material_slots:
                if slot.material:
                    node = slot.material.node_tree.nodes.get(channel+'_node')
                    if node:
                        node.image = linear
        export.apply_baked_tex(obj, parameters)
        for slot in obj.material_slots:
            if slot.material and slot.material.name in alpha_values:
                slot.material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = alpha_values[slot.material.name]


def bake_architecture(meshes, plan, out, resolution=1024, device='CPU'):
    """Bake selected architecture in memory, keeping input geometry evidence untouched."""
    import bpy
    import time

    if resolution not in (256, 512, 1024, 2048, 4096) or device not in ('CPU', 'CUDA'):
        raise ValueError('MATERIAL_BAKE_UNAVAILABLE(unsupported profile)')
    if len(plan['rooms']) != 1:
        raise ValueError('MATERIAL_BINDING_MISMATCH(expected one room)')
    room = plan['rooms'][0]
    seed = plan['provenance']['seed']
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 8
    scene.cycles.device = 'CPU'
    if device == 'CUDA':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        cuda = [d for d in prefs.devices if d.type == 'CUDA']
        if not cuda:
            raise ValueError('MATERIAL_BAKE_UNAVAILABLE(no CUDA device)')
        for d in prefs.devices:
            d.use = d in cuda
        scene.cycles.device = 'GPU'
    rows = []
    roles = {'unique_assets:room_floor': 'floor', 'unique_assets:room_wall': 'wall',
             'unique_assets:room_exterior': 'wall',
             'unique_assets:room_ceiling': 'ceiling'}
    for obj in meshes:
        role = next((roles[c.name] for c in obj.users_collection if c.name in roles), 'detail')
        before = [slot.material.name if slot.material else None for slot in obj.material_slots]
        intent = None
        if role != 'detail':
            token = room['floor_material' if role == 'floor' else 'wall_material']
            material, intent = make_material(token, role, seed)
            obj.data.materials.clear()
            obj.data.materials.append(material)
            for face in obj.data.polygons:
                face.material_index = 0
        # Audit only used slots: unused template glass slots do not describe visible geometry.
        used = sorted({face.material_index for face in obj.data.polygons})
        summaries = []
        for index in used:
            if index >= len(obj.material_slots) or obj.material_slots[index].material is None:
                raise ValueError(f'MATERIAL_BINDING_MISMATCH({obj.name}:{index})')
            mat = obj.material_slots[index].material
            if not mat.use_nodes:
                raise ValueError(f'MATERIAL_CHANNEL_UNSUPPORTED({mat.name},non_node_shader)')
            displacement = False
            for node in mat.node_tree.nodes:
                if node.type == 'EMISSION':
                    raise ValueError(f'MATERIAL_CHANNEL_UNSUPPORTED({mat.name},emission)')
                if node.type == 'OUTPUT_MATERIAL' and node.inputs['Displacement'].is_linked:
                    displacement = True
                if node.type == 'BSDF_PRINCIPLED':
                    strength = node.inputs.get('Emission Strength')
                    colour = node.inputs.get('Emission Color')
                    if strength and colour and (strength.is_linked or
                        (strength.default_value > 0 and (colour.is_linked or any(colour.default_value[:3])))):
                        raise ValueError(f'MATERIAL_CHANNEL_UNSUPPORTED({mat.name},emission)')
            # Blender 4.2 moved this from cycles settings onto Material itself.
            mode = getattr(mat, 'displacement_method', 'UNKNOWN')
            if displacement and mode != 'BUMP':
                raise ValueError(f'MATERIAL_CHANNEL_UNSUPPORTED({mat.name},displacement:{mode})')
            summaries.append({'slot': index, 'shader': mat.name, 'summary': material_summary(mat),
                              'displacement': 'bump_to_tangent_normal' if displacement else 'none'})
        rows.append({'object': obj.name, 'role': role, 'source_bindings': before,
                     'intent': intent, 'mapped_bindings': summaries})
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'mapped.blend'))
    visibility = {o: o.hide_render for o in scene.objects}
    for index, (obj, row) in enumerate(zip(meshes, rows)):
        for other in scene.objects:
            other.hide_render = other != obj
        obj.hide_set(False)
        started = time.monotonic()
        prefix = f'surface_{index:03d}'
        window_conversions = []
        for slot in obj.material_slots:
            if slot.material and slot.material.name.startswith('shader_window_glass'):
                slot.material, conversion = convert_window_material(slot.material)
                window_conversions.append(conversion)
        if window_conversions:
            row['window_conversions'] = window_conversions
        bake_surface(obj, out, resolution, prefix)
        transmission = out / f'{prefix}_TRANSMISSION.png'
        if transmission.is_file():
            # UsdPreviewSurface has opacity but no transmission input. This is an explicit
            # thin-surface approximation, retained in the report for visual review.
            image = bpy.data.images.load(str(transmission), check_existing=False)
            image.colorspace_settings.name = 'Non-Color'
            pixels = list(image.pixels)
            for offset in range(0, len(pixels), 4):
                opacity = 1.0 - pixels[offset]
                pixels[offset:offset+4] = [opacity, opacity, opacity, 1.0]
            image.pixels[:] = pixels
            image.filepath_raw = str(out / f'{prefix}_OPACITY.png')
            image.file_format = 'PNG'
            image.save()
            for slot in obj.material_slots:
                if not slot.material:
                    continue
                tree = slot.material.node_tree
                for node in list(tree.nodes):
                    if node.type != 'BSDF_PRINCIPLED':
                        continue
                    socket = node.inputs['Transmission Weight']
                    for link in list(socket.links):
                        tree.links.remove(link)
                    socket.default_value = 0
                    texture = tree.nodes.new('ShaderNodeTexImage')
                    texture.image = image
                    tree.links.new(texture.outputs['Color'], node.inputs['Alpha'])
            row['glass_conversion'] = 'thin_surface_opacity=1-transmission; refraction not preserved'
        channels = {}
        for channel in ('DIFFUSE', 'ROUGHNESS', 'NORMAL', 'METAL', 'TRANSMISSION', 'OPACITY'):
            path = out / f'{prefix}_{channel}.png'
            if not path.is_file():
                if channel in ('DIFFUSE', 'ROUGHNESS', 'NORMAL'):
                    raise ValueError(f'MATERIAL_DEPENDENCY_MISSING({path.name})')
                continue
            images = [n.image for slot in obj.material_slots if slot.material
                      for n in slot.material.node_tree.nodes
                      if n.type == 'TEX_IMAGE' and n.image and
                      n.image.filepath and __import__('pathlib').Path(bpy.path.abspath(n.image.filepath)).resolve() == path.resolve()]
            if not images:
                raise ValueError(f'MATERIAL_BINDING_MISMATCH({obj.name}:{channel})')
            colour_space = 'sRGB' if channel == 'DIFFUSE' else 'Non-Color'
            for image in images:
                image.colorspace_settings.name = colour_space
            channels[channel] = {'uri': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'colour_space': colour_space, 'resolution': [resolution, resolution]}
            if channel == 'TRANSMISSION':
                channels[channel]['converted_to'] = 'OPACITY'
        row.update(channels=channels, uv_map='ExportUV', normal_convention='tangent +Y',
                   elapsed_s=round(time.monotonic()-started, 3))
    for obj, hidden in visibility.items():
        obj.hide_render = hidden
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'baked.blend'))
    return {'schema_version': 'pvscene.material-manifest/v1',
            'profile': {'version': 'pbr-v1', 'resolution': resolution, 'samples': 8,
                        'device': device, 'mapping': MAPPING_VERSION, 'bake': BAKE_VERSION,
                        'atlas_margin_texels': 16, 'bake_margin_texels': 8},
            'blender_version': bpy.app.version_string, 'surfaces': rows,
            'visual_review': 'not_run', 'spec_accepted': False}
