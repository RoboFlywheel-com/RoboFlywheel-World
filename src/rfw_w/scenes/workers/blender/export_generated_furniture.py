#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the generated objects a build kept, so each one can enter the delivery as an instance.

    blender -b <build>/scene/scene.blend -P python -m rfw_w.scenes.workers.blender.export_generated_furniture -- \
        --ledger <build>/constraint_injection.json --out <build>/generated --pin <commit>

Run under **Infinigen's bundled Blender Python**, on the build's own saved scene. It reads the
rejection pool's ledger -- the record the build wrote about what it kept and why -- and writes, for
every kept object, one raw OBJ in the object's own yaw-aligned frame with the footprint centre at
the origin and the base at z = 0, plus `objects.json`: the pose that puts it back, its size, its
category and factory.

**It measures nothing about the room and judges nothing.** Which objects may stay was decided
inside the build (SPEC-011 TASK 4); whether the kept ones are legal in the *delivery* is judged by
the run's gates (`geometry.in_bounds`, `geometry.no_penetration`, `geometry.task_cell_clear`) and
by the audit, on the instances this export makes possible. The two things this tool owes them are
the ones only the scene can answer: the real geometry, and the real pose.

The frame is the decision that matters, and it lives in `rfw_w.scenes.assets.generated_furniture`
so the Python side and this script cannot disagree about it:
`python -m rfw_w.scenes.workers.blender.measure_generated_furniture` measures the *hull footprint* of the same world
vertices, and a box that overstated a yawed object would hand the gates a shape the ruler never saw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path.cwd()

from rfw_w.scenes.assets.generated_furniture import (
    ASSEMBLY_FILENAME,
    ASSEMBLY_SCHEMA_VERSION,
    recentre,
)
from rfw_w.scenes.contracts.hashing import content_hash

#: Below this the object is treated as standing upright; above it the tilt is recorded, because a
#: yaw-aligned box then does not describe the object and a reader has to know that.
TILT_TOLERANCE_DEG = 1.0


def _tree(obj) -> list:
    """The object and its descendants, in a stable order: one factory can leave several meshes."""
    import bpy
    import mathutils
    found, stack = [], [obj]
    while stack:
        item = stack.pop()
        found.append(item)
        stack.extend(sorted(item.children, key=lambda child: child.name, reverse=True))
    return [item for item in found
            if getattr(item, 'type', None) == 'MESH' and getattr(item, 'data', None) is not None]


def _base_colour(material) -> tuple[float, float, float]:
    """A material's base colour: the Principled node's factor, else the viewport colour."""
    import bpy
    import mathutils
    try:
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                value = node.inputs['Base Color'].default_value
                return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:  # noqa: BLE001 - a material without a node tree is not a failure
        pass
    value = getattr(material, 'diffuse_color', (0.8, 0.8, 0.8, 1.0))
    return tuple(float(channel) for channel in value[:3])


def _has_texture(material) -> bool:
    import bpy
    import mathutils
    try:
        return any(node.type == 'TEX_IMAGE' and getattr(node, 'image', None) is not None
                   for node in material.node_tree.nodes)
    except Exception:  # noqa: BLE001
        return False


def _write_obj(frame, faces, path: Path, *, colour, material_name: str) -> None:
    """The raw mesh, in the frame `recentre` returned, with one base-colour material.

    Colours only, no maps: a generated object's textures are not baked (ADR-009 keeps material
    baking on SPEC-007's chain and grants no affordance certification through it), so the record
    says `has_texture` and the render gets the factor. Writing the geometry in the prepared frame
    here means the Python side has no centring of its own to get wrong.
    """
    import bpy
    import mathutils
    lines = [f"# generated object, frame footprint_centre_base_at_zero",
             f"mtllib {path.with_suffix('.mtl').name}", f"usemtl {material_name}"]
    for x, y, z in frame.points:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for face in faces:
        if len(face) >= 3:
            lines.append("f " + " ".join(str(index + 1) for index in face))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix('.mtl').write_text(
        "\n".join([f"newmtl {material_name}",
                   f"Kd {colour[0]:.6f} {colour[1]:.6f} {colour[2]:.6f}",
                   "Ks 0.000000 0.000000 0.000000", "Ns 100.000000", ""]),
        encoding="utf-8",
    )


def main() -> int:
    import bpy
    import mathutils
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pin', default='', help='the upstream commit the build ran')
    args = parser.parse_args(argv)

    ledger = json.loads(args.ledger.read_text())
    pool = ledger.get('rejection_pool') or {}
    kept = [item for item in pool.get('candidates', ()) if item.get('outcome') == 'kept']
    args.out.mkdir(parents=True, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records, skipped = [], []
    for index, candidate in enumerate(sorted(kept, key=lambda item: str(item.get('name', '')))):
        name = str(candidate.get('name', ''))
        obj = bpy.data.objects.get(name)
        if obj is None:
            # A kept object the scene does not have is a fact about this export, not a crash: the
            # record names it, and the reason it is missing is that this blend is not that build.
            skipped.append({'name': name, 'reason': 'absent-from-blend'})
            continue
        parts = _tree(obj)
        if not parts:
            skipped.append({'name': name, 'reason': 'no-mesh-in-tree'})
            continue
        world_points, faces, vertex_count = [], [], 0
        materials = []
        for part in parts:
            evaluated = part.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.transform(evaluated.matrix_world)
            offset = vertex_count
            world_points.extend(tuple(vertex.co) for vertex in mesh.vertices)
            vertex_count += len(mesh.vertices)
            faces.extend(tuple(index + offset for index in polygon.vertices)
                         for polygon in mesh.polygons)
            for slot in getattr(part, 'material_slots', ()):
                if slot.material is not None and slot.material not in materials:
                    materials.append(slot.material)
            evaluated.to_mesh_clear()
        matrix = obj.matrix_world
        yaw_rad = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
        up = matrix.to_3x3() @ mathutils.Vector((0.0, 0.0, 1.0))
        up = up.normalized()
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(up.z)))))
        frame = recentre(world_points, yaw_rad=yaw_rad)
        slug = f"{index:02d}_{candidate.get('factory', 'object')}"
        colour = _base_colour(materials[0]) if materials else (0.8, 0.8, 0.8)
        _write_obj(frame, faces, args.out / 'raw' / f"{slug}.obj", colour=colour,
                   material_name=f"generated_{slug}")
        records.append({
            'name': name, 'factory': str(candidate.get('factory', '')),
            'category': str(candidate.get('category', '')), 'outcome': 'kept',
            'slug': slug, 'raw_obj': str((args.out / 'raw' / f"{slug}.obj").name),
            'parts': len(parts), 'vertex_count': vertex_count, 'face_count': len(faces),
            'dimensions_m': [round(value, 6) for value in frame.dimensions_m.to_list()],
            'bounds_min_m': [round(value, 6) for value in frame.bounds_min.to_list()],
            'bounds_max_m': [round(value, 6) for value in frame.bounds_max.to_list()],
            'position_m': [round(value, 6) for value in frame.position_m.to_list()],
            'yaw_rad': round(frame.yaw_rad, 9),
            'tilt_deg': round(tilt_deg, 6),
            'materials': [material.name for material in materials],
            'has_texture': any(_has_texture(material) for material in materials),
            'origin_convention': 'footprint_centre_base_at_zero',
        })
    blend = Path(bpy.data.filepath)
    payload = {
        'schema_version': ASSEMBLY_SCHEMA_VERSION,
        'tool': 'python -m rfw_w.scenes.workers.blender.export_generated_furniture',
        'blend': str(blend),
        'blend_sha256': hashlib.sha256(blend.read_bytes()).hexdigest(),
        'ledger': str(args.ledger),
        'ledger_sha256': hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        'upstream_pin': args.pin,
        'kept_in_ledger': len(kept),
        'exported': len(records),
        'skipped': skipped,
        'tilt_tolerance_deg': TILT_TOLERANCE_DEG,
        'objects': records,
    }
    payload['digest'] = content_hash({key: value for key, value in payload.items()
                                      if key != 'digest'})
    (args.out / ASSEMBLY_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                              encoding='utf-8')
    print(f"exported {len(records)} of {len(kept)} kept object(s); skipped {len(skipped)} -> "
          f"{args.out / ASSEMBLY_FILENAME}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
