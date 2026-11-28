# SPDX-License-Identifier: Apache-2.0
"""Mapping failures must be visible before importing Blender or Infinigen."""
import importlib
from rfw_w.scenes.resources import module_path, data_path
from pathlib import Path

import pytest

_PATH = module_path('rfw_w.scenes.workers.blender.infinigen_materials')

_MODULE = importlib.import_module('rfw_w.scenes.workers.blender.infinigen_materials')


@pytest.mark.parametrize('token,role', [('unknown', 'floor'), ('', 'wall'), ('oak', 'furniture')])
def test_unsupported_intent_is_named(token, role):
    with pytest.raises(ValueError, match='MATERIAL_INTENT_UNSUPPORTED'):
        _MODULE.mapping(token, role)


@pytest.mark.parametrize('kwargs', [{'version': 'unknown'}, {'resolution': 0}, {'device': 'magic'}])
def test_invalid_export_profile_is_named(kwargs):
    from rfw_w.scenes.solvers.materials import MaterialExportProfile
    with pytest.raises(ValueError, match='MATERIAL_BAKE_UNAVAILABLE'):
        MaterialExportProfile(**kwargs)
