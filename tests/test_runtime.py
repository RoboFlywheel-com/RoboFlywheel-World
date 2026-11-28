# SPDX-License-Identifier: Apache-2.0
"""Public runtime behavior with a real isolated Python process."""
import os
import sys

from rfw_w.runtime import WorkerJob, installed_packages, run


def test_worker_failure_exit_status_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "invalid-python-home"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "invalid-path"))
    assert run(WorkerJob(sys.executable, "module_that_does_not_exist")) == 1


def test_installed_packages_probe_does_not_inherit_isaac_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "invalid-python-home"))
    result = installed_packages(sys.executable)
    assert result["rfw_w"] is True
    assert set(result) == {"rfw_w", "rfw_w.assets", "rfw_w.scenes"}
