# SPDX-License-Identifier: Apache-2.0
"""Independent full sampled approach replay; no dynamics or swept-volume claim."""
import argparse
import json
import math
from pathlib import Path
from rfw_w.scenes.validation.verify_robot_runtime_model import digest


def source_hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in root.rglob('*') if p.is_file()}


def sample(start, end):
    # Sampling resolution is explicit; collision between samples is not certified.
    intervals = max(1, math.ceil(max(abs(end[n] - start[n]) / (.001 if n in ('joint7', 'joint8') else .01)
                                     for n in start)))
    return [{n: start[n] + (end[n] - start[n]) * k / intervals for n in start}
            for k in range(intervals + 1)]


def replay(model, frames):
    import mujoco
    data = mujoco.MjData(model)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(model.njnt)}
    measured = []
    for index, frame in enumerate(frames):
        if set(frame) != set(names):
            raise ValueError('APPROACH_PATH_INVALID')
        for name, value in frame.items():
            j = names[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not model.jnt_range[j, 0] <= value <= model.jnt_range[j, 1]:
                raise ValueError('APPROACH_PATH_INVALID')
            data.qpos[model.jnt_qposadr[j]] = value
        mujoco.mj_forward(model, data)
        penetrations = [{'geom1': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                         'geom2': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), 'distance_m': float(c.dist)}
                        for c in data.contact if c.dist < -.001]
        measured.append({'index': index, 'contacts_count': data.ncon, 'penetrations': penetrations})
        if penetrations:
            return {'verified': False, 'failure_code': 'APPROACH_PATH_COLLISION', 'frames': measured,
                    'frames_not_run': len(frames) - len(measured)}
    return {'verified': True, 'failure_code': None, 'frames': measured, 'frames_not_run': 0}


def verify(root):
    import mujoco
    from rfw_w.scenes.validation.verify_mujoco_target_ik import verify as verify_ik
    meta = json.loads((root / 'path.json').read_text())
    if digest(root / 'path.json') != (root / 'path.sha256').read_text().strip():
        raise ValueError('APPROACH_PATH_INVALID')
    source = (root / meta['source_ref']).resolve()
    # A tool references its preceding sibling output; portable parent-tree relocation is supported.
    if Path(meta['source_ref']).is_absolute() or not source.is_relative_to(root.resolve().parent):
        raise ValueError('APPROACH_SOURCE_MISMATCH')
    if source_hashes(source) != meta['source_hashes']:
        raise ValueError('APPROACH_SOURCE_MISMATCH')
    if not verify_ik(source)['verified']:
        raise ValueError('APPROACH_PATH_INVALID')
    start = json.loads((source / 'source/source/installation.json').read_text())['reference_joints']
    end = json.loads((source / 'ik_case.json').read_text())['candidate']
    expected = sample(start, end)
    if meta['frames'] != expected or meta['method'] != 'linear joint samples; max .01rad/.001m':
        raise ValueError('APPROACH_PATH_INVALID')
    model = mujoco.MjModel.from_xml_path(str(source / 'source/appearance.xml'))
    result = replay(model, meta['frames'])
    result.update(frame_count=len(expected), source_ref=meta['source_ref'], motion='not_run', grasp='not_run',
                  continuous_collision='not_run', model_requests=0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = verify(a.delivery)
    except (ValueError, OSError, KeyError, TypeError) as error:
        report = {'verified': False, 'failure_code': str(error)}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
