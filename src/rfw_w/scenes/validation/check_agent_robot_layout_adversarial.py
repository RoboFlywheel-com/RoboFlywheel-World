# SPDX-License-Identifier: Apache-2.0
"""Actual altered Agent evidence and illegal finish replay rejection."""
import argparse
import json
import shutil
from pathlib import Path
from rfw_w.scenes.validation.verify_agent_robot_layout import verify
from rfw_w.scenes.orchestrator.robot_layout_agent import tree_hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    frozen = tree_hashes(a.delivery)
    results = {}
    for name in ('relocation', 'early_finish', 'stale_revision', 'source_changed'):
        root = a.output_dir / name
        shutil.copytree(a.delivery, root)
        events = json.loads((root / 'events.json').read_text())
        if name == 'early_finish':
            events[0]['decision']['tool_name'] = 'finish'
        elif name == 'stale_revision':
            events[1]['before']['revision'] = 9
        elif name == 'source_changed':
            scene = root / 'step_001_install_robot/installation/scene/graph.json'
            scene.write_text(scene.read_text() + '\n')
        if name in ('early_finish', 'stale_revision'):
            (root / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
        try:
            results[name] = verify(root)
        except (ValueError, OSError, KeyError) as error:
            results[name] = {'verified': False, 'failure_code': str(error)}
    report = {'verified': results['relocation']['verified'] and
              all(not results[n]['verified'] for n in ('early_finish', 'stale_revision', 'source_changed')) and
              frozen == tree_hashes(a.delivery), 'cases': results, 'source_unchanged': frozen == tree_hashes(a.delivery)}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
