# SPDX-License-Identifier: Apache-2.0
"""Coverage audit: streaming parse correctness and provider-specific usable counts.

The parser is the risky part. The simready catalogue is 143 MB, so it is parsed incrementally,
and the classic incremental-parser bug is a value that straddles a buffer refill. That case is
tested directly with a tiny chunk size rather than only at the default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfw_w.scenes.assets.coverage import (
    CoverageError,
    audit_hssd,
    audit_simready,
    iter_json_object_items,
    resolve_requirements,
)


def write_index(path: Path, entries: dict[str, object]) -> Path:
    path.write_text(json.dumps(entries, indent=1))
    return path


class TestStreamingParser:
    def test_reads_every_entry(self, tmp_path: Path) -> None:
        entries = {f"asset_{index:04d}": {"n": index} for index in range(500)}
        path = write_index(tmp_path / "index.json", entries)
        assert dict(iter_json_object_items(path)) == entries

    @pytest.mark.parametrize("chunk", [1, 2, 7, 13, 64, 4096])
    def test_survives_values_straddling_a_buffer_refill(self, tmp_path: Path, chunk: int) -> None:
        """A decoder that assumes a whole value is in the buffer fails on these chunk sizes."""
        entries = {
            "a": {"description": "x" * 500, "nested": {"deep": [1, 2, 3, {"k": "v"}]}},
            "b": {"value": 0.123456789},
            "c": "a string with } and { inside it",
        }
        path = write_index(tmp_path / "index.json", entries)
        assert dict(iter_json_object_items(path, chunk_bytes=chunk)) == entries

    def test_handles_empty_object(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        path.write_text("{}")
        assert list(iter_json_object_items(path)) == []

    def test_rejects_a_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(CoverageError, match="not a JSON object"):
            list(iter_json_object_items(path))

    def test_handles_unicode_and_escapes(self, tmp_path: Path) -> None:
        entries = {"a": "陶瓷汤碗", "b": "line\nbreak\ttab", "c": 'quote"inside'}
        path = write_index(tmp_path / "index.json", entries)
        assert dict(iter_json_object_items(path, chunk_bytes=8)) == entries


class TestSimreadyAudit:
    @staticmethod
    def index(tmp_path: Path) -> Path:
        return write_index(tmp_path / "index.json", {
            "a1": {"super_category": "Mug", "domain": "kitchen", "physics_type": "rigid",
                   "real_bbox": [0.09, 0.09, 0.1], "mass": 0.2, "material": "ceramic",
                   "mass_check": "ok"},
            "a2": {"super_category": "Mug", "domain": "kitchen", "physics_type": "rigid",
                   "real_bbox": [0.08, 0.08, 0.1], "mass": 0.18, "material": "ceramic",
                   "mass_check": "flag"},
            "a3": {"super_category": "Mug", "domain": "kitchen", "physics_type": "rigid",
                   "real_bbox": [0.1, 0.1, 0.12], "material": "ceramic"},  # no mass
            "b1": {"super_category": "Desk Bell", "domain": "office", "physics_type": "rigid",
                   "real_bbox": [0.1, 0.1, 0.15], "mass": 0.3, "material": "brass"},
            "c1": {"super_category": "", "domain": "office"},  # unusable entry
        })

    def test_counts_assets_and_categories(self, tmp_path: Path) -> None:
        coverage = audit_simready(self.index(tmp_path), library_version="v0.2")
        assert coverage.asset_count == 4
        assert coverage.unparsed_entries == 1
        assert set(coverage.categories) == {"Mug", "Desk Bell"}

    def test_declared_count_is_everything(self, tmp_path: Path) -> None:
        assert audit_simready(self.index(tmp_path)).categories["Mug"].declared_count == 3

    def test_usable_count_excludes_assets_missing_mass(self, tmp_path: Path) -> None:
        """The whole point of the audit: three mugs are declared, two are usable.

        A body with no mass cannot be settled, so reporting three would overstate the library.
        """
        assert audit_simready(self.index(tmp_path)).categories["Mug"].usable_count == 2

    def test_evidence_is_counted_separately(self, tmp_path: Path) -> None:
        evidence = audit_simready(self.index(tmp_path)).categories["Mug"].evidence
        assert evidence["with_dimensions"] == 3
        assert evidence["with_mass"] == 2
        assert evidence["with_material"] == 3
        assert evidence["with_mass_check_ok"] == 1

    def test_usable_definition_is_recorded_next_to_the_number(self, tmp_path: Path) -> None:
        coverage = audit_simready(self.index(tmp_path))
        assert "dynamic rigid body" in coverage.usable_definition

    def test_mass_check_is_reported_as_declared_not_remeasured(self, tmp_path: Path) -> None:
        coverage = audit_simready(self.index(tmp_path))
        assert any("declared" in note for note in coverage.notes)

    def test_missing_index_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(CoverageError, match="not found"):
            audit_simready(tmp_path / "nope.json")


class TestHssdAudit:
    @staticmethod
    def index(tmp_path: Path) -> Path:
        return write_index(tmp_path / "synsets.json", {
            "desk.n.01": [
                {"id": "a", "name": "Lismore Desk", "up": "", "front": "", "support_region": True},
                {"id": "b", "name": "Corner Desk", "up": "0,0,-1", "front": "0,1,0",
                 "support_region": False},
            ],
            "cup.n.01": [{"id": "c", "name": "Mug", "up": "", "front": "", "support_region": False}],
        })

    def test_counts_synsets_and_objects(self, tmp_path: Path) -> None:
        coverage = audit_hssd(self.index(tmp_path))
        assert coverage.category_count == 2
        assert coverage.asset_count == 3

    def test_all_indexed_objects_are_usable_as_fixed_furniture(self, tmp_path: Path) -> None:
        # HSSD carries no mass or collision metadata, so the usable definition is deliberately
        # different from simready's and is stated alongside the number.
        coverage = audit_hssd(self.index(tmp_path))
        assert coverage.categories["desk.n.01"].usable_count == 2
        assert "fixed furniture" in coverage.usable_definition

    def test_orientation_annotations_are_counted(self, tmp_path: Path) -> None:
        # These decide whether FACING is expressible at all.
        evidence = audit_hssd(self.index(tmp_path)).categories["desk.n.01"].evidence
        assert evidence["with_front"] == 1
        assert evidence["with_up"] == 1
        assert evidence["with_support_region"] == 1

    def test_notes_record_the_absence_of_physical_metadata(self, tmp_path: Path) -> None:
        coverage = audit_hssd(self.index(tmp_path))
        assert any("no mass" in note for note in coverage.notes)


class TestRequirementResolution:
    def test_matches_provider_specific_spellings(self, tmp_path: Path) -> None:
        """A vocabulary difference ("Mug" vs mug.n.01) must not read as an empty library."""
        simready = audit_simready(TestSimreadyAudit.index(tmp_path))
        hssd_path = write_index(tmp_path / "s.json", {"mug.n.04": [{"id": "m", "name": "Mug"}]})
        providers = {"simready": simready, "hssd": audit_hssd(hssd_path)}
        resolutions = resolve_requirements(providers, [
            {"id": "cup", "role": "task_target", "candidates": {"simready": ["Mug"], "hssd": ["mug.n.04"]}}
        ])
        assert resolutions[0].servable_providers == ("hssd", "simready")

    def test_reports_which_providers_cannot_serve(self, tmp_path: Path) -> None:
        simready = audit_simready(TestSimreadyAudit.index(tmp_path))
        resolutions = resolve_requirements({"simready": simready}, [
            {"id": "desk", "role": "support", "candidates": {"simready": ["Desk"]}}
        ])
        assert not resolutions[0].satisfied
        assert resolutions[0].candidates[0].matched_categories == ()

    def test_declared_without_evidence_does_not_satisfy(self, tmp_path: Path) -> None:
        """A category of assets that all lack mass is a gap dressed as coverage."""
        path = write_index(tmp_path / "i.json", {
            "x": {"super_category": "Widget", "real_bbox": [0.1, 0.1, 0.1], "material": "steel"},
        })
        resolutions = resolve_requirements({"simready": audit_simready(path)}, [
            {"id": "widget", "role": "task_target", "candidates": {"simready": ["Widget"]}}
        ])
        assert resolutions[0].candidates[0].declared_count == 1
        assert resolutions[0].candidates[0].usable_count == 0
        assert not resolutions[0].satisfied

    def test_requirement_without_an_id_is_rejected(self, tmp_path: Path) -> None:
        simready = audit_simready(TestSimreadyAudit.index(tmp_path))
        with pytest.raises(CoverageError, match="without an id"):
            resolve_requirements({"simready": simready}, [{"role": "support"}])


class TestFilesystemCheck:
    """The index is a summary; the export tree is the library, and they can disagree."""

    def test_consistent_counts_report_no_gap(self) -> None:
        from rfw_w.scenes.assets.coverage import FilesystemCheck

        payload = FilesystemCheck(asset_directories=100, domains=4, index_entries=100, elapsed_s=1.0).to_dict()
        assert "gap" not in payload
        assert payload["delta"] == 0

    def test_unindexed_directories_are_reported_as_a_named_gap(self) -> None:
        """Assets on disk that the index does not mention exist and cannot be retrieved.

        Reporting only the index count states the searchable library is smaller than it is, and
        hides a set of assets nobody can reach.
        """
        from rfw_w.scenes.assets.coverage import FilesystemCheck

        check = FilesystemCheck(asset_directories=524570, domains=31, index_entries=486595, elapsed_s=1.0)
        payload = check.to_dict()
        assert payload["gap"] == "UNINDEXED_DIRS(37975)"
        assert "unreachable by retrieval" in payload["explanation"]

    def test_missing_directories_are_reported_the_other_way(self) -> None:
        # Index entries with no directory mean retrieval would resolve to missing files.
        from rfw_w.scenes.assets.coverage import FilesystemCheck

        check = FilesystemCheck(asset_directories=90, domains=2, index_entries=100, elapsed_s=1.0)
        assert check.to_dict()["gap"] == "MISSING_DIRS(10)"

    def test_counting_uses_the_asset_depth(self, tmp_path: Path) -> None:
        """A nested directory inside an asset must not inflate the count."""
        from rfw_w.scenes.assets.coverage import count_filesystem_assets

        for domain in ("0", "1"):
            for shard in ("0", "1"):
                for asset in ("a", "b"):
                    nested = tmp_path / domain / shard / asset / "mesh"
                    nested.mkdir(parents=True)
        assets, domains, _ = count_filesystem_assets(tmp_path)
        assert assets == 8
        assert domains == 2

    def test_a_missing_root_is_an_error(self, tmp_path: Path) -> None:
        from rfw_w.scenes.assets.coverage import CoverageError, count_filesystem_assets

        with pytest.raises(CoverageError, match="not found"):
            count_filesystem_assets(tmp_path / "absent")
