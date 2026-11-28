# SPDX-License-Identifier: Apache-2.0
"""The selected GPU must reach the worker rather than silently defaulting to GPU zero."""
import importlib
import subprocess
import sys
from pathlib import Path


def test_selected_gpu_is_forwarded(tmp_path, monkeypatch):
    module = importlib.import_module('rfw_w.scenes.workers.launch')
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, '', '')
    monkeypatch.setattr(module.subprocess, 'run', run)
    (code, report) = module.invoke(tmp_path / 'request.json', tmp_path / 'new_output', isaac_python='/isaac-sim/python.sh', active_gpu=4, timeout_s=30)
    assert calls[0][-2:] == ['--active-gpu', '4']
    assert code == 1 and report is None
