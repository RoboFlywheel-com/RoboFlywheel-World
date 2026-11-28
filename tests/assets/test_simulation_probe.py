# SPDX-License-Identifier: Apache-2.0
"""Cross-engine acceptance must reject altered physical parameters."""

import json

import pytest

from rfw_w.assets.simulation.probe import check_properties, main


def test_mass_com_and_principal_inertia_are_independent_checks():
    physical = {"total_mass": 0.8, "com": [0.04, 0, 0],
                "inertia": [[0.01, 0.002, 0], [0.002, 0.02, 0], [0, 0, 0.025]]}
    import numpy as np

    values = np.linalg.eigvalsh(physical["inertia"])
    assert all(check_properties(physical, 0.8, [0.04, 0, 0], values).values())
    wrong = check_properties(physical, 0.7, [0, 0, 0], values * 2)
    assert not any(wrong.values())


def test_probe_missing_source_fails_with_portable_report(tmp_path):
    out = tmp_path / "probe"
    assert main(["--urdf", str(tmp_path / "missing.urdf"),
                 "--mjcf", str(tmp_path / "missing.xml"),
                 "--physics", str(tmp_path / "missing.json"),
                 "--backend", "mujoco", "--out", str(out)]) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "failed"
    assert str(tmp_path) not in json.dumps(report)
    assert report["settle"] == report["task_run"] == "not_run"
    with pytest.raises(SystemExit):
        main(["--urdf", "x", "--mjcf", "x", "--physics", "x", "--backend", "mujoco",
              "--out", str(out)])
