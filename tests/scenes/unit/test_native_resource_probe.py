# SPDX-License-Identifier: Apache-2.0
"""Resource precheck keeps positive/negative historical fixture semantics."""

import importlib
import json
from pathlib import Path
import sys


def test_prior_evidence_can_be_checked_after_external_bridge_removal(tmp_path, monkeypatch):
    probe = importlib.import_module('rfw_w.scenes.validation.resources')
    monkeypatch.setattr(probe, 'REPO_ROOT', tmp_path)
    for (fixture, status) in (('supported', 'passed'), ('missing_support', 'failed')):
        directory = tmp_path / 'results' / fixture
        directory.mkdir(parents=True)
        (directory / 'report.json').write_text(json.dumps({'fixture': fixture, 'status': status}))
    items = probe.probe_evidence_from_previous_runs()
    assert len(items) == 2
    assert all((item.verdict == probe.Verdict.PRESENT for item in items))
