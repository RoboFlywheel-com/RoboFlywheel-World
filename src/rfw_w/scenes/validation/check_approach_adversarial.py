# SPDX-License-Identifier: Apache-2.0
"""Real sampled path obstacle and tampered schedule/source probes."""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_mujoco_approach_path import verify, replay


def main():
    import mujoco
    p = argparse.ArgumentParser()
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    meta = json.loads((a.delivery / 'path.json').read_text())
    source = (a.delivery / meta['source_ref']).resolve()
    results = {}
    # Keep the actual source sibling while altering only the new path output.
    for name in ('missing_frame', 'source_hash'):
        target = a.output_dir / name
        target.mkdir()
        candidate = json.loads(json.dumps(meta))
        import os
        candidate['source_ref'] = os.path.relpath(source, target)
        # verifier rejects sources outside its parent; use portable copy in adversarial parent.
        copied = a.output_dir / 'ik'
        if not copied.exists():
            shutil.copytree(source, copied)
        candidate['source_ref'] = '../ik'
        if name == 'missing_frame':
            candidate['frames'].pop(len(candidate['frames']) // 2)
        else:
            candidate['source_hashes']['ik_case.json'] = '0' * 64
        (target / 'path.json').write_text(json.dumps(candidate, indent=2) + '\n')
        (target / 'path.sha256').write_text(digest(target / 'path.json') + '\n')
        try:
            results[name] = verify(target)
        except ValueError as e:
            results[name] = {'verified': False, 'failure_code': str(e)}
    model = mujoco.MjModel.from_xml_path(str(source / 'source/appearance.xml'))
    data = mujoco.MjData(model)
    frame = meta['frames'][len(meta['frames']) // 2]
    for n, v in frame.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        data.qpos[model.jnt_qposadr[j]] = v
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_base')
    tree = ET.parse(source / 'source/appearance.xml')
    # Use a fresh local copy so source assets remain immutable and references remain portable.
    obstacle_root = a.output_dir / 'blocked_source'
    shutil.copytree(source / 'source', obstacle_root)
    ET.SubElement(tree.getroot().find('worldbody'), 'geom', name='approach_blocker', type='box',
                  pos=' '.join(map(str, data.xpos[body])), size='.025 .025 .025')
    tree.write(obstacle_root / 'blocked.xml', encoding='utf-8')
    blocked = mujoco.MjModel.from_xml_path(str(obstacle_root / 'blocked.xml'))
    results['intermediate_obstacle'] = replay(blocked, meta['frames'])
    report = {'verified': results['missing_frame']['failure_code'] == 'APPROACH_PATH_INVALID' and
              results['source_hash']['failure_code'] == 'APPROACH_SOURCE_MISMATCH' and
              results['intermediate_obstacle']['failure_code'] == 'APPROACH_PATH_COLLISION', 'cases': results}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'verified': report['verified'], 'cases': {n: r.get('failure_code') for n, r in results.items()}}))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
