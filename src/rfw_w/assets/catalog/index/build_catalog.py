# SPDX-License-Identifier: Apache-2.0
"""Build a local release catalogue from complete asset leaves; no cloud/table dependencies."""

from __future__ import annotations
import argparse, json, sqlite3, re
from pathlib import Path
from rfw_w.assets.catalog import config

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;

CREATE TABLE assets (
    rowid_          INTEGER PRIMARY KEY,   -- 与向量矩阵行号一致
    key             TEXT UNIQUE NOT NULL,  -- canonical release leaf
    orig_domain     INTEGER,
    orig_class      INTEGER,
    orig_num        INTEGER,
    final_domain    INTEGER,
    final_class     INTEGER,
    is_new_class    INTEGER DEFAULT 0,     -- final_class 为 NULL → 回退原始 class
    domain_name     TEXT,
    class_name      TEXT,
    domain_zh       TEXT,
    class_zh        TEXT,
    domain_group_zh TEXT,
    export_rel      TEXT,                  -- relative release leaf directory
    has_export      INTEGER DEFAULT 0,
    has_urdf        INTEGER DEFAULT 0,
    has_mjcf        INTEGER DEFAULT 0,
    has_affordance  INTEGER DEFAULT 0,
    has_render      INTEGER DEFAULT 0,
    -- 物理（整机）
    real_height     REAL,
    bbox_x          REAL,
    bbox_y          REAL,
    bbox_z          REAL,
    scale           REAL,
    n_parts         INTEGER,
    n_collision_hulls INTEGER,
    total_mass      REAL,
    composed_mass   REAL,
    vlm_mass        REAL,
    mass_ratio      REAL,
    mass_check      TEXT,
    mass_source     TEXT,
    mass_policy     TEXT,
    com_x           REAL,
    com_y           REAL,
    com_z           REAL,
    inertia         TEXT,
    whole_mu1       REAL,
    whole_mu2       REAL,
    material_source TEXT,
    coverage_ok     INTEGER,
    main_material   TEXT,                  -- 质量占比最大的部件材质
    main_material_subtype TEXT,
    -- VLM 语义
    object_guess    TEXT,
    observed_category TEXT,
    material_guess  TEXT,
    shape_summary   TEXT,
    surface_quality TEXT,
    key_features    TEXT,
    reasoning       TEXT,
    final_status    TEXT,
    sim_ready       INTEGER,
    breakage_score  REAL,
    is_multi        INTEGER,
    -- 部件名摘要（展示用）
    part_names      TEXT,
    part_n_graspable INTEGER,
    glb_rel         TEXT
);

CREATE INDEX idx_assets_domain  ON assets(final_domain);
CREATE INDEX idx_assets_class   ON assets(final_domain, final_class);
CREATE INDEX idx_assets_mass    ON assets(total_mass);
CREATE INDEX idx_assets_height  ON assets(real_height);
CREATE INDEX idx_assets_parts   ON assets(n_parts);
CREATE INDEX idx_assets_cname   ON assets(class_name);
CREATE INDEX idx_assets_status  ON assets(final_status);
-- 下面几个是筛选面板常用的等值/范围条件，缺了会让 COUNT 变成全表扫描
CREATE INDEX idx_assets_urdf    ON assets(has_urdf);
CREATE INDEX idx_assets_export  ON assets(has_export);
CREATE INDEX idx_assets_render  ON assets(has_render);
CREATE INDEX idx_assets_mu      ON assets(whole_mu1);
CREATE INDEX idx_assets_hulls   ON assets(n_collision_hulls);
CREATE INDEX idx_assets_dname   ON assets(domain_name);
CREATE INDEX idx_assets_mat     ON assets(main_material);

CREATE TABLE parts (
    key             TEXT NOT NULL,
    pid             INTEGER,
    color           TEXT,
    part_name       TEXT,
    material        TEXT,
    material_category TEXT,
    material_subtype TEXT,
    density         REAL,
    volume_m3       REAL,
    convex_volume   REAL,
    mass_kg         REAL,
    mu1             REAL,
    mu2             REAL,
    cx              REAL,
    cy              REAL,
    cz              REAL,
    hulls           INTEGER,
    priority_rank   INTEGER,
    graspable       INTEGER,
    functional_labels TEXT,
    basic_desc      TEXT,
    functional_desc TEXT,
    grasped_desc    TEXT,
    young_modulus_gpa REAL,
    hardness_hv     REAL,
    poisson_ratio   REAL
);
CREATE INDEX idx_parts_key     ON parts(key);
CREATE INDEX idx_parts_materia ON parts(material);

-- 关键词检索：rowid 与 assets.rowid_ 对齐
CREATE VIRTUAL TABLE fts_en USING fts5(text, tokenize='unicode61');

CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
"""


def build(release_root, out, taxonomy=None):
    release_root = Path(release_root).resolve()
    out = Path(out)
    if out.exists():
        raise FileExistsError("catalogue output must be new")
    out.parent.mkdir(parents=True, exist_ok=True)
    names = taxonomy or {}
    count = 0
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)
    columns = {row[1] for row in con.execute("PRAGMA table_info(assets)")}
    for marker in sorted(release_root.glob("assets/*/*/*/_SUCCESS")):
        leaf = marker.parent
        domain, category, key = leaf.relative_to(release_root / "assets").parts
        if not re.fullmatch(r"[0-9]+_[0-9]+_[0-9]{4,}", key) or key.split("_")[:2] != [
            domain,
            category,
        ]:
            raise ValueError("release leaf identity mismatch")
        physics = json.loads((leaf / "part_physics.json").read_text())
        dims = (
            physics.get("dimensions_m")
            or physics.get("up_axis_fix", {}).get("vis_ext")
            or physics.get("real_bbox")
        )
        if not isinstance(dims, list) or len(dims) != 3:
            raise ValueError("asset lacks measured dimensions")
        aff_path = leaf / "affordance/affordance_annot.json"
        aff = json.loads(aff_path.read_text()) if aff_path.is_file() else {}
        parts = aff.get("affordances", [])
        label = names.get(domain + "/" + category, {})
        row = {
            "rowid_": count,
            "key": key,
            "orig_domain": int(domain),
            "orig_class": int(category),
            "orig_num": int(key.rsplit("_", 1)[1]),
            "final_domain": int(domain),
            "final_class": int(category),
            "domain_name": label.get("domain", domain),
            "class_name": label.get("class", category),
            "domain_zh": label.get("domain_zh", ""),
            "class_zh": label.get("class_zh", ""),
            "export_rel": str(leaf.relative_to(release_root)),
            "has_export": 1,
            "has_urdf": int(bool(list(leaf.glob("*.urdf")))),
            "has_mjcf": int(bool(list((leaf / "mjcf").glob("*.xml")))),
            "has_affordance": int(aff_path.is_file()),
            "real_height": dims[2],
            "bbox_x": dims[0],
            "bbox_y": dims[1],
            "bbox_z": dims[2],
            "total_mass": physics["total_mass"],
            "n_parts": physics.get("n_parts", len(parts)),
            "n_collision_hulls": physics.get("n_collision_hulls", 0),
            "whole_mu1": physics["whole_mu1"],
            "whole_mu2": physics["whole_mu2"],
            "inertia": json.dumps(physics["inertia"]),
            "com_x": physics["com"][0],
            "com_y": physics["com"][1],
            "com_z": physics["com"][2],
            "final_status": "package_complete",
            "sim_ready": 1,
            "part_names": " ".join(
                item.get("part_name", item.get("name", "")) for item in parts
            ),
            "part_n_graspable": sum(item.get("graspable") is True for item in parts),
        }
        props = physics.get("parts", {})
        main = max(props.values(), key=lambda v: v.get("mass_kg", 0), default={})
        row.update(
            main_material=main.get(
                "material_category", main.get("material", "unknown")
            ),
            main_material_subtype=main.get("material_subtype", ""),
        )
        keys = list(row)
        con.execute(
            "INSERT INTO assets ("
            + ",".join(keys)
            + ") VALUES ("
            + ",".join("?" for _ in keys)
            + ")",
            [row[k] for k in keys],
        )
        text = " ".join(
            str(row.get(k, ""))
            for k in (
                "class_name",
                "class_zh",
                "domain_name",
                "main_material",
                "part_names",
            )
        )
        con.execute("INSERT INTO fts_en(rowid,text) VALUES (?,?)", (count, text))
        part_cols = {v[1] for v in con.execute("PRAGMA table_info(parts)")}
        for item in parts:
            physical = item.get("physics", {})
            pr = {
                "key": key,
                "pid": item.get("id"),
                "part_name": item.get("part_name", item.get("name", "")),
                "color": item.get("mask_color", ""),
                "graspable": int(item.get("graspable") is True),
                "priority_rank": item.get("priority_rank"),
                **{k: v for k, v in physical.items() if k in part_cols},
            }
            ks = list(pr)
            con.execute(
                "INSERT INTO parts ("
                + ",".join(ks)
                + ") VALUES ("
                + ",".join("?" for _ in ks)
                + ")",
                [pr[k] for k in ks],
            )
        count += 1
    con.execute("INSERT INTO meta VALUES (?,?)", ("release_version", "v1.3"))
    con.execute("INSERT INTO meta VALUES (?,?)", ("n_assets", str(count)))
    con.commit()
    con.close()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--taxonomy")
    args = parser.parse_args()
    taxonomy = json.loads(Path(args.taxonomy).read_text()) if args.taxonomy else None
    print(build(args.release_root, args.out, taxonomy))


if __name__ == "__main__":
    main()
