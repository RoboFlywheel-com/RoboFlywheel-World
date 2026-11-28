#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run under bundled Blender Python: tiny normal noise must not split a wall plane."""


def main():
    import importlib.util
    from pathlib import Path
    import bpy  # Registers Blender bundled modules, including mathutils.
    from mathutils import Vector

    from rfw_w.scenes.workers.blender import measure_built_shell as ruler
    keys=[]
    for v in [(0,1,0),(1e-7,-1,0),(-1e-7,1,0),(0,-1,0)]:
        n=ruler.canonical_plane(Vector(v))
        keys.append(tuple(round(x,ruler.PLANE_KEY_DECIMALS) for x in n))
    assert len(set(keys))==1, f'PLANE_SIGN_UNSTABLE({keys})'
    # A genuinely oblique face must remain a distinct orientation.
    n=ruler.canonical_plane(Vector((.1,1,0)))
    assert tuple(round(x,ruler.PLANE_KEY_DECIMALS) for x in n)!=keys[0]
    print('PASS: coplanar normal signs stable; oblique plane distinct')

    # Include world scale, ignore the downward-facing bottom of a slab.
    bpy.ops.mesh.primitive_cube_add(size=1)
    cube=bpy.context.object
    cube.scale=(2,3,.1)
    bpy.context.view_layer.update()
    assert abs(ruler.floor_area_m2(cube)-6.)<1e-6
    print('PASS: measured floor area includes world scale and excludes bottom')


if __name__ == "__main__":
    main()
