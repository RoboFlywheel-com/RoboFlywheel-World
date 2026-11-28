# SPDX-License-Identifier: Apache-2.0
"""Actual relocation, candidate tampering and independent FK/collision rejection."""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_mujoco_target_ik import verify, measure


def hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    import mujoco
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    before = hashes(a.delivery)
    results = {}
    relocated = a.output_dir / 'relocated'
    shutil.copytree(a.delivery, relocated)
    results['relocation'] = verify(relocated)
    for name in ('joint_limit', 'candidate_hash'):
        root = a.output_dir / name
        shutil.copytree(a.delivery, root)
        case = json.loads((root / 'ik_case.json').read_text())
        case['candidate']['joint1'] = 99 if name == 'joint_limit' else 0
        (root / 'ik_case.json').write_text(json.dumps(case, indent=2) + '\n')
        if name == 'joint_limit':
            integrity = json.loads((root / 'ik_integrity.json').read_text())
            integrity['ik_case.json'] = digest(root / 'ik_case.json')
            (root / 'ik_integrity.json').write_text(json.dumps(integrity, indent=2) + '\n')
        try:
            results[name] = verify(root)
        except ValueError as error:
            results[name] = {'verified': False, 'failure_code': str(error)}
    rules = json.loads((a.delivery / 'ik_rules.json').read_text())
    case = json.loads((a.delivery / 'ik_case.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(relocated / 'source/appearance.xml'))
    far = {**case, 'position_m': [10, 10, 10]}
    results['far_target'] = measure(model, rules, far)
    tree = ET.parse(relocated / 'source/appearance.xml')
    # Enclose the accepted TCP: actual mesh/box endpoint contacts, no mocked failure.
    ET.SubElement(tree.getroot().find('worldbody'), 'geom', name='probe_blocker', type='box',
                  pos=' '.join(map(str, case['position_m'])), size='.10 .10 .10')
    blocked_file = relocated / 'source/blocked_probe.xml'
    tree.write(blocked_file, encoding='utf-8')
    blocked = mujoco.MjModel.from_xml_path(str(blocked_file))
    results['blocked_endpoint'] = measure(blocked, rules, case)
    expected = {'joint_limit': 'TARGET_IK_JOINT_INVALID', 'candidate_hash': 'TARGET_IK_ARTIFACT_MISMATCH',
                'far_target': 'TARGET_IK_UNREACHABLE', 'blocked_endpoint': 'TARGET_IK_COLLISION'}
    report = {'verified': results['relocation']['verified'] and
              all(results[n].get('failure_code') == code for n, code in expected.items()) and before == hashes(a.delivery),
              'cases': results, 'source_unchanged': before == hashes(a.delivery), 'model_requests': 0}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
