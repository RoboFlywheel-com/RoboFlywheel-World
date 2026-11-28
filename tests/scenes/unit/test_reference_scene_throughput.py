# SPDX-License-Identifier: Apache-2.0
"""Negative scene clearance and native-frame streaming regression evidence."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest


def minimal_world():
    root=ET.Element('mujoco');ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    for name in ('floor','rear_wall','side_wall','tool_board'):
        ET.SubElement(world,'geom',name=name,type='box',size='1 1 1')
    return root


def test_source_furniture_cannot_enter_manipulation_core(tmp_path):
    pytest.importorskip('trimesh');np=pytest.importorskip('numpy')
    from rfw_w.scenes.adapters.reference_scene import dress
    import trimesh
    source=tmp_path/'fixture.glb'
    trimesh.Scene(trimesh.creation.box(extents=[.2,.2,.2])).export(str(source))
    out=tmp_path/'world';out.mkdir()
    with pytest.raises(ValueError,match='REFERENCE_DRESSING_CORE_CLEARANCE'):
        dress(minimal_world(),out,{'assets':[dict(file=str(source),position_m=[0,0,0],id='synthetic_fixture',role='storage',license='synthetic')]})


def test_two_materials_with_same_texture_name_remain_distinct(tmp_path):
    pytest.importorskip('trimesh');np=pytest.importorskip('numpy');pytest.importorskip('PIL')
    import trimesh
    from PIL import Image
    from rfw_w.scenes.adapters.reference_scene import dress
    scene=trimesh.Scene()
    for i,color in enumerate(((255,0,0),(0,255,0))):
        mesh=trimesh.creation.box(extents=[.1,.1,.1]);mesh.apply_translation([i*.12,0,0])
        material=trimesh.visual.material.PBRMaterial(name='shared_name',baseColorTexture=Image.new('RGB',(8,8),color))
        mesh.visual=trimesh.visual.texture.TextureVisuals(uv=np.zeros((len(mesh.vertices),2)),material=material)
        scene.add_geometry(mesh,node_name=str(i))
    source=tmp_path/'fixture.glb';scene.export(str(source));out=tmp_path/'world';out.mkdir()
    root=minimal_world()
    report=dress(root,out,{'assets':[dict(file=str(source),position_m=[1.7,1.5,0],id='synthetic_fixture',role='storage',license='synthetic')]})
    files=[Path(t.get('file')) for t in root.find('asset').findall('texture') if t.get('name').startswith('tex_background')]
    assert len(files)==2 and files[0]!=files[1]
    assert list(Image.open(files[0]).getdata())[0]!=list(Image.open(files[1]).getdata())[0]
    assert report['assets'][0]['source_scale']==1
