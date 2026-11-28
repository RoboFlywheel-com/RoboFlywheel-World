# SPDX-License-Identifier: Apache-2.0
"""Installed-wheel siblings must not be mistaken for the scene package itself."""
from types import SimpleNamespace


def test_stdlib_audit_rejects_other_installed_packages(monkeypatch, tmp_path):
    from rfw_w.scenes.validation import check_stdlib_only as audit
    site = tmp_path / "site-packages"
    package = site / "rfw_w.scenes"
    monkeypatch.setattr(audit, "PACKAGE_ROOT", package)
    monkeypatch.setattr(audit, "SRC", site)
    assert audit._is_third_party(SimpleNamespace(__file__=str(site / "numpy" / "__init__.py")))
    assert audit._is_third_party(SimpleNamespace(__file__=str(package / "contracts" / "asset.py"))) is None
