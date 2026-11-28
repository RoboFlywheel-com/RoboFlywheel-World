# SPDX-License-Identifier: Apache-2.0
"""SQLite 目录 + 向量矩阵的只读访问层。

向量矩阵按 rowid_ 行序存储（build_vectors 输出），所以「过滤 + 向量打分」
可以先用 SQL 求出允许的 rowid 集合，再只对这批行做内积 —— 带过滤的查询
通常只需算几千行，比全库 52 万行快两个数量级。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rfw_w.assets.catalog import config


class Store:
    def __init__(self, db: str | None = None, vectors: str | None = None):
        self.db_path = str(db or config.CATALOG_DB)
        self.vec_path = str(vectors or config.VECTORS_NPY)
        self._local = threading.local()
        self._vecs: np.ndarray | None = None
        self._keys: list[str] | None = None
        self._lock = threading.Lock()
        self._taxonomy: dict | None = None

    # ── 连接（每线程一个，sqlite3 连接不能跨线程共享） ─────────────────────
    @property
    def con(self) -> sqlite3.Connection:
        c = getattr(self._local, "con", None)
        if c is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA query_only = ON")
            c.execute("PRAGMA cache_size = -262144")  # 256MB page cache
            c.execute("PRAGMA mmap_size = 1073741824")
            self._local.con = c
        return c

    # ── 向量 ──────────────────────────────────────────────────────────────
    @property
    def vectors(self) -> np.ndarray | None:
        if self._vecs is None and Path(self.vec_path).exists():
            with self._lock:
                if self._vecs is None:
                    # 直接读进内存：52万×1024×4B = 2.1GB，本机 937GB 内存毫无压力。
                    # 用 mmap 时每次查询都要走一遍页表/缺页，全库内积反而更慢。
                    self._vecs = np.load(self.vec_path)
        return self._vecs

    @property
    def keys(self) -> list[str]:
        if self._keys is None:
            p = Path(config.VECTOR_KEYS)
            self._keys = json.loads(p.read_text()) if p.exists() else []
        return self._keys

    # ── 元信息 / 分类 ─────────────────────────────────────────────────────
    def meta(self) -> dict[str, str]:
        try:
            return {r["k"]: r["v"] for r in self.con.execute("SELECT k,v FROM meta")}
        except sqlite3.Error:
            return {}

    def taxonomy(self) -> dict:
        """中文分类名 -> [(class_en, domain_zh)]，供中文 query 精确匹配。"""
        if self._taxonomy is not None:
            return self._taxonomy
        m = self.meta()
        entries = []
        try:
            entries = json.loads(m.get("taxonomy_classes", "[]"))
        except Exception:
            pass
        by_zh: dict[str, list] = {}
        for e in entries:
            zh = (e.get("zh") or "").strip()
            if zh:
                by_zh.setdefault(zh, []).append((e.get("en"), e.get("domain_zh")))
        self._taxonomy = {"by_zh": by_zh, "entries": entries}
        return self._taxonomy

    def facets(self) -> dict:
        """筛选面板用的聚合：分类计数 + 数值分位。"""
        con = self.con
        domains = [
            dict(r)
            for r in con.execute(
                "SELECT final_domain AS id, domain_name, domain_zh, COUNT(*) AS n "
                "FROM assets GROUP BY final_domain ORDER BY n DESC"
            )
        ]
        classes = [
            dict(r)
            for r in con.execute(
                "SELECT final_domain AS d, final_class AS c, class_name, class_zh, COUNT(*) AS n "
                "FROM assets GROUP BY final_domain, final_class ORDER BY n DESC"
            )
        ]
        mats = [
            dict(r)
            for r in con.execute(
                "SELECT main_material AS m, COUNT(*) AS n FROM assets "
                "WHERE main_material IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 60"
            )
        ]
        stats = {}
        for col in (
            "total_mass",
            "real_height",
            "n_parts",
            "n_collision_hulls",
            "whole_mu1",
        ):
            r = con.execute(
                f"SELECT MIN({col}) a, MAX({col}) b, AVG({col}) c FROM assets "
                f"WHERE {col} IS NOT NULL"
            ).fetchone()
            if r and r["a"] is not None:
                stats[col] = {"min": r["a"], "max": r["b"], "avg": r["c"]}
        counts = {
            r["k"]: r["n"]
            for r in con.execute(
                "SELECT final_status AS k, COUNT(*) AS n FROM assets GROUP BY 1"
            )
        }
        return {
            "domains": domains,
            "classes": classes,
            "materials": mats,
            "stats": stats,
            "status_counts": counts,
            "n_assets": con.execute("SELECT COUNT(*) FROM assets").fetchone()[0],
        }

    # ── 行取用 ────────────────────────────────────────────────────────────
    def rows_by_ids(self, ids: Iterable[int]) -> dict[int, sqlite3.Row]:
        ids = list(ids)
        if not ids:
            return {}
        out: dict[int, sqlite3.Row] = {}
        # 分批避免 SQL 变量数上限
        for i in range(0, len(ids), 900):
            chunk = ids[i : i + 900]
            q = f"SELECT * FROM assets WHERE rowid_ IN ({','.join('?' * len(chunk))})"
            for r in self.con.execute(q, chunk):
                out[r["rowid_"]] = r
        return out

    def asset(self, key: str) -> dict | None:
        r = self.con.execute("SELECT * FROM assets WHERE key = ?", (key,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["parts"] = [
            dict(p)
            for p in self.con.execute(
                "SELECT * FROM parts WHERE key = ? ORDER BY pid", (key,)
            )
        ]
        return d

    def path_for(self, key: str) -> Path | None:
        rel = self.con.execute(
            "SELECT export_rel FROM assets WHERE key = ?", (key,)
        ).fetchone()
        if not rel:
            return None
        return config.EXPORT_ROOT / rel["export_rel"]


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
