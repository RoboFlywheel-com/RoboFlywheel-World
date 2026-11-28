# SPDX-License-Identifier: Apache-2.0
"""Find actual collision-free initial posture whose direct approach collides."""
import argparse
import json
import shutil
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest
from rfw_w.scenes.validation.verify_mujoco_approach_path import sample, replay, source_hashes


def main():
    import mujoco
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--ik', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(a.ik / 'source/appearance.xml'))
    target = json.loads((a.ik / 'ik_case.json').read_text())['candidate']
    rng = np.random.default_rng(40)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    candidates = []
    accepted = None
    for _ in range(200):
        q = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        start = dict(zip(names, map(float, q)))
        # A closed jaw is a plausible initial state; the endpoint opens during approach.
        start.update(joint7=.015, joint8=-.015)
        initial = replay(model, [start])
        if not initial['verified']:
            candidates.append({'candidate': start, 'failure_code': 'INITIAL_COLLISION'})
            continue
        result = replay(model, sample(start, target))
        candidates.append({'candidate': start, 'path_verified': result['verified'], 'path_failure': result['failure_code']})
        if not result['verified']:
            accepted = {'reference_joints': start, 'path': result}
            break
    if accepted:
        robot = a.output_dir / 'robot'
        shutil.copytree(a.ik / 'source/source/robot', robot)
        meta = json.loads((robot / 'mujoco_runtime.json').read_text())
        meta['commands'] = accepted['reference_joints']
        (robot / 'mujoco_runtime.json').write_text(json.dumps(meta, indent=2) + '\n')
    report = {'verified': accepted is not None, 'accepted': accepted, 'attempts': candidates,
              'source_hashes': source_hashes(a.ik), 'fixture': 'actual noncolliding initial posture, colliding direct approach'}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'verified': accepted is not None, 'attempts': len(candidates)}))
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
