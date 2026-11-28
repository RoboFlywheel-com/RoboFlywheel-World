# SPDX-License-Identifier: Apache-2.0
"""Explicit MuJoCo process boundary for the stdlib robot layout Agent."""
import json
import subprocess
import sys
from pathlib import Path
from rfw_w.scenes.validation.verify_single_room_installation import verify as verify_install

ROOT = Path.cwd()


class RobotLayoutExecutor:
    def __init__(self, assets, robot, semantics, *, skip_initial=0):
        self.assets, self.robot, self.semantics = assets, robot, semantics
        self.skip_initial = skip_initial

    def bind_assets(self,assets):
        self.assets=assets.resolve()

    def call(self, script, args, step):
        try:
            completed = subprocess.run([sys.executable, '-m', 'rfw_w.scenes.validation.' + Path(script).stem, *map(str, args)],
                                       cwd=ROOT, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            (step / (script + '.stderr.txt')).write_text('ROBOT_TOOL_TIMEOUT\n')
            raise TimeoutError('ROBOT_TOOL_TIMEOUT') from None
        # Scripts print only public measurements; preserve diagnostic failures locally.
        (step / (script + '.stdout.txt')).write_text(completed.stdout)
        (step / (script + '.stderr.txt')).write_text(completed.stderr)
        return completed.returncode

    def __call__(self, name, state, step, output):
        if name in ('install_robot', 'revise_robot_installation'):
            revision = state.revision + (name == 'revise_robot_installation')
            args = ['--robot-bundle', self.robot, '--scene-bundle', self.assets,
                    '--room', self.semantics['room_id'], '--support', self.semantics['support_role_id'],
                    '--target', self.semantics['target_role_id'], '--skip-candidates', self.skip_initial + revision,
                    '--output-dir', step / 'installation']
            code = self.call('check_single_room_installation.py', args, step)
            if code:
                ref = step / 'installation/installation_report.json'
                report = json.loads(ref.read_text()) if ref.exists() else {}
                return {'status': 'failed', 'failure_code': report.get('failure_code', 'ROBOT_LAYOUT_INPUT_INVALID')}
            checked = verify_install(step / 'installation')
            (step / 'independent_installation.json').write_text(json.dumps(checked, indent=2) + '\n')
            if not checked['verified']:
                return {'status': 'failed', 'failure_code': checked['failure_code']}
            code = self.call('render_single_room_installation.py',
                             ['--delivery', step / 'installation', '--output-dir', step / 'appearance'], step)
            return ({'status': 'failed', 'failure_code': 'TARGET_IK_RENDER_FAILED'} if code else
                    {'status': 'passed', 'artifact_ref': step.relative_to(output).as_posix()})
        if name == 'check_target_reachability':
            source = output / state.install_ref / 'appearance'
            code = self.call('check_mujoco_target_ik.py', ['--delivery', source, '--output-dir', step / 'ik'], step)
            report_file = step / 'ik/report.json'
            report = json.loads(report_file.read_text()) if report_file.exists() else {}
            if code:
                return {'status': 'failed', 'failure_code': report.get('failure_code', 'ROBOT_LAYOUT_INPUT_INVALID')}
            from rfw_w.scenes.validation.verify_mujoco_target_ik import verify
            checked = verify(step / 'ik')
            (step / 'independent_ik.json').write_text(json.dumps(checked, indent=2) + '\n')
            return {'status': 'passed' if checked['verified'] else 'failed', 'failure_code': checked['failure_code'],
                    'artifact_ref': (step / 'ik').relative_to(output).as_posix()}
        if name == 'check_approach_path':
            # 039 uses sibling references: path and its IK source share the Agent tree parent.
            from rfw_w.scenes.validation.verify_mujoco_approach_path import sample, source_hashes, replay
            from rfw_w.scenes.validation.verify_robot_runtime_model import digest
            import os
            import mujoco
            source = output / state.ik_ref
            start = json.loads((source / 'source/source/installation.json').read_text())['reference_joints']
            end = json.loads((source / 'ik_case.json').read_text())['candidate']
            meta = {'source_ref': os.path.relpath(source, step), 'source_hashes': source_hashes(source),
                    'method': 'linear joint samples; max .01rad/.001m', 'frames': sample(start, end)}
            (step / 'path.json').write_text(json.dumps(meta, indent=2) + '\n')
            (step / 'path.sha256').write_text(digest(step / 'path.json') + '\n')
            model = mujoco.MjModel.from_xml_path(str(source / 'source/appearance.xml'))
            checked = replay(model, meta['frames'])
            (step / 'report.json').write_text(json.dumps(checked, indent=2) + '\n')
            return {'status': 'passed' if checked['verified'] else 'failed', 'failure_code': checked['failure_code'],
                    'artifact_ref': step.relative_to(output).as_posix()}
        raise ValueError('TOOL_NOT_AVAILABLE')
