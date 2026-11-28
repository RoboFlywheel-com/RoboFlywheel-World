# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import trimesh

from rfw_w.assets.workers.segmentation import restore_prediction


def test_cleaned_prediction_keeps_face_labels_and_source_frame():
    clean = trimesh.creation.box(extents=[2, 1, 1])
    clean.apply_translation([4, 2, 1])
    source = trimesh.Trimesh(
        vertices=clean.vertices, faces=np.vstack([clean.faces, clean.faces[:1]]),
        process=False,
    )
    normalized = clean.copy()
    normalized.apply_scale(0.5)
    ids = np.arange(len(clean.faces))
    restored, actual_ids = restore_prediction(source, (None, ids, normalized), 2)
    assert len(restored.faces) == len(clean.faces) < len(source.faces)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_allclose(restored.bounds, source.bounds)


def test_prediction_mesh_must_match_labels_and_source_frame():
    source = trimesh.creation.box()
    with pytest.raises(ValueError, match="face assignments"):
        restore_prediction(source, (None, np.zeros(11), source), 1)
    moved = source.copy()
    moved.apply_translation([1, 0, 0])
    with pytest.raises(ValueError, match="coordinate frame"):
        restore_prediction(source, (None, np.zeros(12), moved), 1)
