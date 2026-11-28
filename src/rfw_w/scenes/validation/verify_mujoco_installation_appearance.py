# SPDX-License-Identifier: Apache-2.0
"""Independent appearance provenance and compiled MuJoCo physics equality ruler."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest


def require(ok,code):
    if not ok:raise ValueError(code)


def original_model(root):
    import mujoco
    tree=ET.parse(root/'scene.xml')
    for index,geom in enumerate(tree.findall('.//geom')):geom.set('name','original_'+str(index))
    for mesh in tree.findall('.//mesh'):mesh.set('file',str((root/mesh.get('file')).resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))


def physics(root):
    import mujoco,numpy as np
    source=original_model(root/'source');actual=mujoco.MjModel.from_xml_path(str(root/'appearance.xml'))
    for kind,fields,count in [('body',('body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat'),source.nbody),
                              ('joint',('jnt_type','jnt_range','jnt_pos','jnt_axis'),source.njnt),
                              ('actuator',('actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange'),source.nu)]:
        obj={'body':mujoco.mjtObj.mjOBJ_BODY,'joint':mujoco.mjtObj.mjOBJ_JOINT,'actuator':mujoco.mjtObj.mjOBJ_ACTUATOR}[kind]
        for i in range(count):
            name=mujoco.mj_id2name(source,obj,i)
            j=0 if kind=='body' and i==0 else mujoco.mj_name2id(actual,obj,name)
            require(j>=0,'APPEARANCE_PHYSICS_CHANGED')
            for field in fields:require(np.allclose(getattr(source,field)[i],getattr(actual,field)[j],atol=1e-9,rtol=1e-9),'APPEARANCE_PHYSICS_CHANGED')
    for i in range(source.ngeom):
        name=mujoco.mj_id2name(source,mujoco.mjtObj.mjOBJ_GEOM,i);j=mujoco.mj_name2id(actual,mujoco.mjtObj.mjOBJ_GEOM,name)
        require(j>=0,'APPEARANCE_PHYSICS_CHANGED')
        for field in ('geom_type','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp'):
            require(np.allclose(getattr(source,field)[i],getattr(actual,field)[j],atol=1e-9,rtol=1e-9),'APPEARANCE_PHYSICS_CHANGED')
        if source.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
            def mesh_vertices(model,g):
                mid=model.geom_dataid[g];start=model.mesh_vertadr[mid];return model.mesh_vert[start:start+model.mesh_vertnum[mid]]
            require(np.array_equal(mesh_vertices(source,i),mesh_vertices(actual,j)),'APPEARANCE_PHYSICS_CHANGED')
    ds=mujoco.MjData(source);da=mujoco.MjData(actual)
    config=json.loads((root/'source/installation.json').read_text())
    for model,data in ((source,ds),(actual,da)):
        for name,value in config['reference_joints'].items():
            j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);data.qpos[model.jnt_qposadr[j]]=value
        mujoco.mj_forward(model,data)
    def contacts(model,data):
        return sorted((mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),round(float(c.dist),9)) for c in data.contact)
    require(contacts(source,ds)==contacts(actual,da),'APPEARANCE_PHYSICS_CHANGED')
    return {'original_geoms':source.ngeom,'added_visual_geoms':actual.ngeom-source.ngeom,'contacts_unchanged':True,'body_mass_inertia_unchanged':True}


def verify(root):
    ledger=json.loads((root/'appearance_ledger.json').read_text())
    for ref,expected in ledger['artifacts'].items():
        path=(root/ref).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and digest(path)==expected,'APPEARANCE_ARTIFACT_MISMATCH')
    tree=ET.parse(root/'appearance.xml')
    for tag in ('mesh','texture'):
        for item in tree.findall('.//'+tag):require(item.get('file') in ledger['artifacts'],'APPEARANCE_MAPPING_MISSING')
    config=json.loads((root/'source/installation.json').read_text())
    mapped={r['instance_id'] for r in ledger['parts'] if r['kind']=='instance'}
    require(mapped=={i['instance_id'] for i in config['instances']},'APPEARANCE_MAPPING_MISSING')
    for item in config['instances']:
        require(digest(root/'source/scene'/item['visual_ref'])==item['visual_sha256'],'APPEARANCE_SOURCE_MISMATCH')
    from PIL import Image
    import numpy as np
    for material in ledger['materials']:
        for role,source in material['source_channels'].items():
            path=root/'source'/source['ref']
            require(digest(path)==source['sha256'],'APPEARANCE_SOURCE_MISMATCH')
            require(np.array_equal(np.asarray(Image.open(path).convert('RGB')),np.asarray(Image.open(root/material['channels'][role]).convert('RGB'))),'APPEARANCE_SOURCE_MISMATCH')
    source_faces={};exported_faces={}
    for part in ledger['parts']:
        mesh=root/part['mesh_ref'];lines=mesh.read_text().splitlines()
        require(sum(l.startswith('f ') for l in lines)==part['faces'],'APPEARANCE_MAPPING_MISSING')
        if part.get('texture_ref'):require(any(l.startswith('vt ') for l in lines),'APPEARANCE_MAPPING_MISSING')
        if part.get('source_mesh'):
            original=root/'source'/part['source_mesh'];require(digest(original)==part['source_sha256'],'APPEARANCE_SOURCE_MISMATCH')
            original_lines=original.read_text().splitlines()
            require([l for l in original_lines if l.startswith(('v ','vt ','vn '))]==[l for l in lines if l.startswith(('v ','vt ','vn '))],'APPEARANCE_MAPPING_MISSING')
            source_faces[part['source_mesh']]=sum(l.startswith('f ') for l in original_lines)
            exported_faces[part['source_mesh']]=exported_faces.get(part['source_mesh'],0)+part['faces']
    require(source_faces==exported_faces,'APPEARANCE_MAPPING_MISSING')
    return {'verified':True,'physics':physics(root),'parts':len(ledger['parts']),'surface_bindings':len(ledger['surfaces']),
            'renderer_limits':ledger['renderer_limits'],'robot_ik':'not_run','grasp':'not_run'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--delivery',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try:report=verify(a.delivery)
    except (ValueError,OSError,KeyError) as e:report={'verified':False,'failure_code':str(e)}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if report['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
