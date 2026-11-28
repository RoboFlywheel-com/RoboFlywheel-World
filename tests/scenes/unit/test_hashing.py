# SPDX-License-Identifier: Apache-2.0
"""Hash stability and strictness.

Two properties are load-bearing and both are tested here:

* the same meaning hashes the same regardless of key order or float representation, or
  recomputation during delivery verification would spuriously disagree with the generator;
* a real change always changes the hash, or stale evidence would be accepted as current.
"""

from __future__ import annotations

import math

import pytest

from rfw_w.scenes.contracts.hashing import (
    HashError,
    SceneHashes,
    canonical_json,
    content_hash,
    hash_text,
    is_digest,
)


class TestCanonicalForm:
    def test_key_order_does_not_matter(self) -> None:
        assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})

    def test_nested_key_order_does_not_matter(self) -> None:
        left = {"outer": {"x": 1, "y": [{"p": 1, "q": 2}]}}
        right = {"outer": {"y": [{"q": 2, "p": 1}], "x": 1}}
        assert content_hash(left) == content_hash(right)

    def test_list_order_is_semantic(self) -> None:
        # Placement order and candidate ranking are meaningful; normalising them away would
        # let two genuinely different scenes share a hash.
        assert content_hash([1, 2, 3]) != content_hash([3, 2, 1])

    def test_bool_is_not_hashed_as_int(self) -> None:
        # bool subclasses int in Python; if it were not checked first, True and 1 would
        # collide and a flag change could go unnoticed.
        assert content_hash({"flag": True}) != content_hash({"flag": 1})

    def test_float_representation_noise_is_absorbed(self) -> None:
        assert content_hash({"x": 1.0}) == content_hash({"x": 0.9999999999999999})

    def test_negative_zero_equals_zero(self) -> None:
        assert content_hash({"x": -0.0}) == content_hash({"x": 0.0})

    def test_real_change_changes_the_hash(self) -> None:
        assert content_hash({"x": 1.0}) != content_hash({"x": 1.0001})

    def test_tiny_coordinate_change_still_changes_the_hash(self) -> None:
        # 1e-10 is below the renderer's notice but above the rounding threshold, and a
        # penetration-gate decision can hinge on exactly this magnitude.
        assert content_hash({"z": 0.79}) != content_hash({"z": 0.79 + 1e-10})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_are_rejected(self, bad: float) -> None:
        with pytest.raises(HashError):
            content_hash({"x": bad})

    def test_non_string_keys_are_rejected(self) -> None:
        with pytest.raises(HashError):
            content_hash({1: "x"})

    def test_unsupported_types_are_rejected(self) -> None:
        with pytest.raises(HashError):
            content_hash({"when": object()})

    def test_nested_rejection_reports_a_path(self) -> None:
        with pytest.raises(HashError, match=r"\$\.a\[0\]\.b"):
            content_hash({"a": [{"b": float("nan")}]})

    def test_canonical_json_is_compact_and_sorted(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_object_with_to_dict_is_accepted(self) -> None:
        class Payload:
            def to_dict(self) -> dict[str, object]:
                return {"y": 1, "x": [2, 3]}

        assert content_hash(Payload()) == content_hash({"x": [2, 3], "y": 1})


class TestDigests:
    def test_hash_text_is_stable(self) -> None:
        assert hash_text("hello") == hash_text("hello")
        assert hash_text("hello") != hash_text("hello ")

    def test_is_digest(self) -> None:
        assert is_digest(content_hash({"a": 1}))
        assert not is_digest("not-a-digest")
        assert not is_digest("Z" * 64)


class TestSceneHashes:
    @staticmethod
    def _hashes(**overrides: object) -> SceneHashes:
        components: dict[str, object] = {
            "spec": {"scene_id": "s"},
            "assets": {"desk": "a1"},
            "geometry": {"desk": [0, 0, 0]},
            "physics": {"desk": 12.0},
            "appearance": {"theme": "warm_minimal"},
        }
        components.update(overrides)
        return SceneHashes.from_components(**components)  # type: ignore[arg-type]

    def test_version_is_derived_from_the_other_five(self) -> None:
        hashes = self._hashes()
        assert is_digest(hashes.version)
        assert hashes.version != hashes.spec_hash

    def test_any_component_change_changes_the_version(self) -> None:
        baseline = self._hashes()
        for field in ("spec", "assets", "geometry", "physics", "appearance"):
            changed = self._hashes(**{field: {"changed": True}})
            assert changed.version != baseline.version, field

    def test_same_inputs_give_the_same_version(self) -> None:
        assert self._hashes().version == self._hashes().version

    def test_matches_compares_versions(self) -> None:
        assert self._hashes().matches(self._hashes())
        assert not self._hashes().matches(self._hashes(physics={"desk": 13.0}))

    def test_rejects_malformed_digests(self) -> None:
        with pytest.raises(HashError):
            SceneHashes("short", "a" * 64, "a" * 64, "a" * 64, "a" * 64, "a" * 64)

    def test_to_dict_exposes_all_six(self) -> None:
        assert set(self._hashes().to_dict()) == {
            "spec", "assets", "geometry", "physics", "appearance", "version",
        }
