# SPDX-License-Identifier: Apache-2.0
"""Source-scale articulated URDF importer with explicit physics provenance.

MuJoCo primitives keep source collision shapes. Mesh collisions remain MuJoCo
convex approximations, recorded as such. No actuator or object attachment is added.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET



def values(v):
    return ' '.join(f'{float(x):.12g}' for x in v)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose(element):
    import numpy as np
    from scipy.spatial.transform import Rotation
    return (np.fromstring(element.get('xyz', '0 0 0'), sep=' '),
            Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' '))) if element is not None else (np.zeros(3), Rotation.identity())


def import_urdf(root, folder, *, prefix='cabinet', position=(0, 0, 0), yaw=0.):
    import numpy as np
    from scipy.spatial.transform import Rotation
    folder = Path(folder)
    source = ET.parse(folder / 'model.urdf').getroot()
    physics = json.loads((folder / 'physics.json').read_text())
    materials = {item['surface_key']: item['values'] for item in physics['bindings']
                 if not item.get('appearance_only')}
    links = {item.get('name'): item for item in source.findall('link')}
    joints = source.findall('joint')
    children = {j.find('child').get('link') for j in joints}
    roots = set(links) - children
    if len(roots) != 1:
        raise ValueError('ARTICULATED_TREE_INVALID')
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    sources = {'model.urdf': digest(folder / 'model.urdf'), 'physics.json': digest(folder / 'physics.json')}
    manifest = {'schema_version': 'pvscene.articulated-urdf/v1', 'scale': 1.,
                'root_fixed': True, 'links': {}, 'joints': {}, 'collisions': [], 'sources': sources,
                'contact_adapter': 'source dynamic sliding; torsion .005 rolling .0001; backend contact law',
                'mesh_collision_adapter': 'MuJoCo convex mesh', 'actuators_added': 0}

    def geometry(element, body, name, collision):
        pos, rot = pose(element.find('origin'))
        geom = element.find('geometry')[0]
        attrs = dict(name=name, pos=values(pos), quat=values(rot.as_quat()[[3, 0, 1, 2]]),
                     density='0', group='3' if collision else '2',
                     contype='1' if collision else '0', conaffinity='1' if collision else '0')
        if geom.tag == 'box':
            attrs.update(type='box', size=values(np.fromstring(geom.get('size'), sep=' ')/2))
        elif geom.tag == 'sphere':
            attrs.update(type='sphere', size=geom.get('radius'))
        elif geom.tag == 'cylinder':
            attrs.update(type='cylinder', size=f"{geom.get('radius')} {float(geom.get('length'))/2}")
        elif geom.tag == 'mesh':
            path = (folder / geom.get('filename')).resolve()
            sources[geom.get('filename')] = digest(path)
            ET.SubElement(asset, 'mesh', name=name+'_mesh', file=str(path), scale=geom.get('scale', '1 1 1'))
            attrs.update(type='mesh', mesh=name+'_mesh')
        else:
            raise ValueError('ARTICULATED_GEOMETRY_UNSUPPORTED:'+geom.tag)
        if collision:
            surface = body.get('name')[len(prefix)+1:]+'::'+element.get('name')
            if surface not in materials:
                raise ValueError('ARTICULATED_MATERIAL_MISSING:'+surface)
            attrs['friction'] = values([materials[surface]['dynamic_friction_coefficient'], .005, .0001])
            attrs['rgba'] = '.3 .3 .3 0'
            manifest['collisions'].append({'geom': name, 'surface': surface, 'geometry': geom.tag})
        else:
            attrs['rgba'] = '.52 .33 .17 1'
        ET.SubElement(body, 'geom', **attrs)

    def build(link, parent, joint=None):
        pos, rot = pose(joint.find('origin')) if joint is not None else (np.array(position), Rotation.from_euler('z', yaw))
        body = ET.SubElement(parent, 'body', name=prefix+'_'+link, pos=values(pos), quat=values(rot.as_quat()[[3, 0, 1, 2]]))
        inertial = links[link].find('inertial')
        if inertial is None:
            raise ValueError('ARTICULATED_INERTIA_MISSING:'+link)
        ipos, irot = pose(inertial.find('origin'))
        ii = inertial.find('inertia')
        inertia = np.array([[float(ii.get('ixx')), float(ii.get('ixy')), float(ii.get('ixz'))],
                            [float(ii.get('ixy')), float(ii.get('iyy')), float(ii.get('iyz'))],
                            [float(ii.get('ixz')), float(ii.get('iyz')), float(ii.get('izz'))]])
        inertia = irot.as_matrix() @ inertia @ irot.as_matrix().T
        mass = float(inertial.find('mass').get('value'))
        if mass <= 0 or np.linalg.eigvalsh(inertia).min() <= 0:
            raise ValueError('ARTICULATED_INERTIA_INVALID:'+link)
        ET.SubElement(body, 'inertial', mass=str(mass), pos=values(ipos),
                      fullinertia=values([*inertia.diagonal(), inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
        manifest['links'][link] = {'body': body.get('name'), 'mass_kg': mass, 'inertia': inertia.tolist()}
        if joint is not None and joint.get('type') != 'fixed':
            kind = joint.get('type')
            if kind not in ('prismatic', 'revolute', 'continuous'):
                raise ValueError('ARTICULATED_JOINT_UNSUPPORTED:'+kind)
            lim = joint.find('limit')
            jname = prefix+'_'+joint.get('name')
            attrs = dict(name=jname, type='slide' if kind == 'prismatic' else 'hinge',
                         axis=joint.find('axis').get('xyz'), damping='1')
            if kind != 'continuous':
                attrs.update(range=f"{lim.get('lower')} {lim.get('upper')}", limited='true')
            ET.SubElement(body, 'joint', **attrs)
            manifest['joints'][joint.get('name')] = {'joint': jname, 'type': kind,
                                                    'damping_adapter': 1., 'limit': dict(lim.attrib) if lim is not None else None}
        for index, visual in enumerate(links[link].findall('visual')):
            geometry(visual, body, prefix+'_'+link+'_visual_'+str(index), False)
        for index, collider in enumerate(links[link].findall('collision')):
            geometry(collider, body, prefix+'_'+link+'_collision_'+str(index), True)
        if not links[link].findall('collision'):
            raise ValueError('ARTICULATED_COLLISION_MISSING:'+link)
        for child in joints:
            if child.find('parent').get('link') == link:
                build(child.find('child').get('link'), body, child)
    build(next(iter(roots)), root.find('worldbody'))
    return manifest
