# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json, sqlite3
from pathlib import Path
import pytest
from rfw_w.assets.catalog.index.build_catalog import build
from rfw_w.assets.catalog.release import resolve


@pytest.mark.parametrize("sequence", ["0000", "10000"])
def test_release_catalogue_fts_and_physics(tmp_path, sequence):
    release = tmp_path / "release"
    key = f"0_0_{sequence}"
    leaf = release / "assets/0/0" / key
    leaf.mkdir(parents=True)
    (leaf / "_SUCCESS").write_text("complete")
    (leaf / f"m_{key}.urdf").write_text("<robot/>")
    (leaf / "part_physics.json").write_text(
        json.dumps(
            {
                "dimensions_m": [0.1, 0.2, 0.3],
                "total_mass": 0.2,
                "whole_mu1": 0.7,
                "whole_mu2": 0.5,
                "com": [0, 0, 0],
                "inertia": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                "parts": {},
            }
        )
    )
    out = tmp_path / "catalog.db"
    assert (
        build(
            release,
            out,
            {
                "0/0": {
                    "class": "Ceramic Mug",
                    "class_zh": "马克杯",
                    "domain": "Kitchen",
                }
            },
        )
        == 1
    )
    con = sqlite3.connect(out)
    row = con.execute("SELECT key,export_rel,total_mass FROM assets").fetchone()
    assert row == (key, f"assets/0/0/{key}", 0.2)
    assert (
        con.execute("SELECT count(*) FROM fts_en WHERE fts_en MATCH 'Mug'").fetchone()[
            0
        ]
        == 1
    )
    assert resolve(release, key) == leaf
    with pytest.raises(FileExistsError):
        build(release, out)
    with pytest.raises(ValueError):
        resolve(release, "../other")


def test_query_numeric_filters_and_rank_fusion():
    from rfw_w.assets.catalog.server.query import parse, rrf

    parsed = parse("mug mass:<1.5 material:ceramic")
    assert "total_mass < ?" in parsed.where
    assert parsed.params == [1.5, "%ceramic%"]
    assert parsed.text == "mug"
    fused = rrf([[1, 2], [2, 3]])
    assert fused[2] > fused[1]
