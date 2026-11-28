# SPDX-License-Identifier: Apache-2.0
"""查询解析 + 混合检索。

查询串里 ``字段:值`` 会被解析成结构化条件，剩下的当自由文本：

    frying pan mass:<1.5 mu:>0.5 class:"Frying Pan" status:complete

三路召回后用 RRF 融合：

1. **FTS5 BM25** —— 英文关键字精确命中（标注全是英文）
2. **向量** —— bge-m3 跨语言语义，中文口语 query 的主力
3. **中文分类字典** —— 把 query 里的中文分类名映射到 ``class_name`` 精确命中

融合前先用 SQL 求出允许的 rowid 集合，向量打分只在允许的行上算 ——
带过滤的查询通常只剩几千行，远快于全库扫描。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: 结构化字段 -> (列名, 类型)。type 决定比较方式。
FIELDS: dict[str, tuple[str, str]] = {
    "class": ("class_name", "text"),
    "class_zh": ("class_zh", "text"),
    "domain": ("domain_name", "text"),
    "material": ("main_material", "text"),
    "subtype": ("main_material_subtype", "text"),
    "status": ("final_status", "text"),
    "mass": ("total_mass", "num"),
    "height": ("real_height", "num"),
    "width": ("bbox_x", "num"),
    "depth": ("bbox_y", "num"),
    "mu": ("whole_mu1", "num"),
    "mu2": ("whole_mu2", "num"),
    "parts": ("n_parts", "num"),
    "hulls": ("n_collision_hulls", "num"),
    "grasp": ("part_n_graspable", "num"),
    "scale": ("scale", "num"),
}

#: status 的语义别名 -> 原始 SQL 片段
STATUS_ALIASES = {
    "complete": "has_urdf = 1",
    "export": "has_export = 1",
    "render": "has_render = 1",
    "affordance": "has_affordance = 1",
    "new": "is_new_class = 1",
    "kept_original": "final_status = 'kept_original'",
    "reclassified": "final_status = 'reclassified'",
    "new_class": "final_status = 'new_class'",
}

_TOKEN = re.compile(r'(\w+):("[^"]*"|\S+)')
_NUM_OP = re.compile(r"^(>=|<=|>|<)?(-?\d+(?:\.\d+)?)(?:-(-?\d+(?:\.\d+)?))?$")
_CJK = re.compile(r"[一-鿿]")


@dataclass
class Parsed:
    text: str = ""
    where: list[str] = field(default_factory=list)
    params: list[Any] = field(default_factory=list)
    order: str | None = None
    notes: list[str] = field(default_factory=list)


def parse(q: str) -> Parsed:
    p = Parsed()
    rest = q
    for m in _TOKEN.finditer(q):
        key = m.group(1).lower()
        quoted = m.group(2).startswith('"')
        val = m.group(2).strip('"')
        if key == "sort":
            col = FIELDS.get(val.split(":")[0], (None, None))[0]
            desc = val.startswith("-")
            if col:
                p.order = f"{col} {'DESC' if desc else 'ASC'}"
            else:
                p.notes.append(f"未知排序字段 {val}")
            rest = rest.replace(m.group(0), " ")
            continue
        if key not in FIELDS:
            continue  # 不是已知字段 -> 留给自由文本
        col, typ = FIELDS[key]
        rest = rest.replace(m.group(0), " ")
        if typ == "text":
            if key == "status":
                frag = STATUS_ALIASES.get(val.lower())
                if frag:
                    p.where.append(frag)
                else:
                    p.where.append("final_status = ?")
                    p.params.append(val)
            elif quoted:
                # 加引号 = 精确匹配，能走索引；不加引号才退化成前缀模糊
                p.where.append(f"{col} = ?")
                p.params.append(val)
            else:
                p.where.append(f"{col} LIKE ?")
                p.params.append(f"%{val}%")
        else:
            mm = _NUM_OP.match(val)
            if not mm:
                p.notes.append(f"{key} 的值 {val!r} 不是数字/区间")
                continue
            op, lo, hi = mm.groups()
            if hi is not None:
                p.where.append(f"{col} BETWEEN ? AND ?")
                p.params.extend([float(lo), float(hi)])
            elif op == ">":
                p.where.append(f"{col} > ?")
                p.params.append(float(lo))
            elif op == ">=":
                p.where.append(f"{col} >= ?")
                p.params.append(float(lo))
            elif op == "<":
                p.where.append(f"{col} < ?")
                p.params.append(float(lo))
            elif op == "<=":
                p.where.append(f"{col} <= ?")
                p.params.append(float(lo))
            else:
                p.where.append(f"{col} = ?")
                p.params.append(float(lo))
    p.text = " ".join(rest.split())
    return p


def _where_sql(p: Parsed, prefix: str = "") -> tuple[str, list]:
    """拼 WHERE 片段（可带前缀列，如 'a.'）。"""
    if not p.where:
        return "", []
    return " AND ".join(p.where), list(p.params)


def _count_allowed(con: sqlite3.Connection, p: Parsed) -> int:
    if not p.where:
        return con.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    return con.execute(
        "SELECT COUNT(*) FROM assets WHERE " + " AND ".join(p.where), p.params
    ).fetchone()[0]


def _filter_candidates(con: sqlite3.Connection, p: Parsed, ids: list[int]) -> set[int]:
    """只对候选集应用结构化条件。

    早期实现是把满足条件的 rowid 全物化成数组再求交集，碰上
    ``mass:0.2-1.0 parts:>=2`` 这种命中 18 万行的条件，光建数组就要 1 秒多、
    再建 set 又要 1 秒。改成先召回（FTS/向量本来就只取几百条）再用
    ``rowid_ IN (...)`` 过滤，代价与召回规模成正比，与过滤条件宽严无关。
    """
    if not p.where or not ids:
        return set(ids)
    out: set[int] = set()
    where, params = _where_sql(p)
    for i in range(0, len(ids), 900):
        chunk = ids[i : i + 900]
        q = (
            f"SELECT rowid_ FROM assets WHERE rowid_ IN "
            f"({','.join('?' * len(chunk))}) AND ({where})"
        )
        out.update(r[0] for r in con.execute(q, chunk + params))
    return out


def _fts_match(con: sqlite3.Connection, text: str, k: int) -> list[int]:
    """英文 token 做前缀匹配的 BM25 召回。"""
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-]+", text)
    if not toks:
        return []
    # 去重保序，最多 12 个词，避免超长 OR 拖慢
    seen, uniq = set(), []
    for t in toks:
        if t.lower() not in seen:
            seen.add(t.lower())
            uniq.append(t)
    expr = " OR ".join(f'"{t}"*' for t in uniq[:12])
    try:
        rows = con.execute(
            "SELECT rowid FROM fts_en WHERE fts_en MATCH ? "
            "ORDER BY bm25(fts_en, 1.0) LIMIT ?",
            (expr, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


_CJK_SEG = re.compile(r"[一-鿿]+")


def _cjk_ngrams(text: str, max_n: int = 4):
    """把 query 里的中文切成长度递减的 n-gram。

    「带手柄的金属锅」要能命中「平底锅 / 炒锅 / 锅把手」这类分类名，只做整串
    子串匹配是不够的 —— 必须退到单字「锅」这一级。长的 n-gram 优先，
    这样「平底锅」会排在「锅」前面。
    """
    for seg in _CJK_SEG.findall(text or ""):
        for n in range(min(max_n, len(seg)), 0, -1):
            for i in range(len(seg) - n + 1):
                yield seg[i : i + n]


def _zh_class_ids(
    con: sqlite3.Connection, text: str, k: int
) -> list[tuple[list[int], list[str], float]]:
    """中文 query -> 候选分类，分成强弱两档返回。

    * **强匹配**（n-gram ≥2 字，如「平底锅」「手柄」）：走 taxonomy 通道，权重高
    * **弱匹配**（单字，如「锅」「刀」）：权重低，只用来兜底

    分档是必要的：「带手柄的金属锅」里「手柄」会命中『意式咖啡手柄/游戏手柄』
    这些明显无关的品类，而真正想要的「平底锅」只能靠单字「锅」召回。把两者
    放进同一个权重里，噪声会盖掉信号；分开之后噪声自然沉底。
    """
    if not _CJK_SEG.search(text or ""):
        return []
    names = _zh_class_names(con)
    best: dict[str, int] = {}
    for g in _cjk_ngrams(text):
        n = len(g)
        for zh in names:
            if best.get(zh, 0) >= n:
                continue
            if g in zh or zh in g:
                best[zh] = n
    if not best:
        return []

    out: list[tuple[list[int], list[str], float]] = []
    # 三档权重。「query 里原样写出的分类名」给最高权重：用户打「水杯」时，
    # 分类就叫「水杯」的那 286 个资产必须排第一，不能让向量召回的相似品类压过去。
    # 反过来「带手柄的金属锅」里没有哪个分类名被原样写出，就退化成向量主导，
    # 不会因为「手柄」这两个字把咖啡手柄、游戏手柄顶上来。
    for want, weight, pick, tag in (
        (8, 3.0, lambda zh, n: zh in text, ""),
        (6, 1.2, lambda zh, n: n >= 2 and zh not in text, ""),
        (8, 0.45, lambda zh, n: n == 1, "·单字"),
    ):
        cand = sorted(
            ((zh, n) for zh, n in best.items() if pick(zh, n)),
            key=lambda x: (-x[1], -len(x[0])),
        )[:want]
        ids: list[int] = []
        used: list[str] = []
        for zh, n in cand:
            rows = con.execute(
                "SELECT rowid_ FROM assets WHERE class_zh = ? LIMIT ?", (zh, k)
            ).fetchall()
            if rows:
                used.append(zh + tag)
                ids.extend(r[0] for r in rows)
        if ids:
            out.append((ids[:k], used, weight))
    return out


_ZH_CACHE: list[str] | None = None


def _zh_class_names(con: sqlite3.Connection) -> list[str]:
    global _ZH_CACHE
    if _ZH_CACHE is None:
        _ZH_CACHE = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT class_zh FROM assets WHERE class_zh IS NOT NULL"
            )
        ]
    return _ZH_CACHE


def rrf(
    lists: list[list[int]], weights: list[float] | None = None, k: int = 60
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：score = Σ w / (k + rank)。"""
    scores: dict[int, float] = {}
    weights = weights or [1.0] * len(lists)
    for lst, w in zip(lists, weights):
        for rank, rid in enumerate(lst):
            scores[rid] = scores.get(rid, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


@dataclass
class Hit:
    rowid: int
    score: float
    sources: list[str]


def search(store, q: str, limit: int = 40, offset: int = 0, embed=None) -> dict:
    """混合检索主入口。返回 {hits, total, parsed, debug}。"""
    con = store.con
    p = parse(q)
    debug: dict[str, Any] = {"notes": p.notes}

    if not p.text:
        # 无自由文本：直接交给 SQL 分页，不走向量也不走 FTS
        q = "SELECT rowid_ FROM assets"
        if p.where:
            q += " WHERE " + " AND ".join(p.where)
        q += f" ORDER BY {p.order or 'rowid_'} LIMIT ? OFFSET ?"
        page = [r[0] for r in con.execute(q, list(p.params) + [limit, offset])]
        return {
            "hits": [Hit(int(i), 0.0, ["filter"]) for i in page],
            "total": _count_allowed(con, p),
            "parsed": p.__dict__,
            "debug": debug,
        }

    R = max(200, limit * 8)
    src: list[list[int]] = []
    names: list[str] = []
    weights: list[float] = []

    fts_ids = _fts_match(con, p.text, R)
    if fts_ids:
        src.append(fts_ids)
        names.append("bm25")
        weights.append(1.0)
    debug["n_bm25"] = len(fts_ids)

    zh_tiers = _zh_class_ids(con, p.text, R)
    zh_used: list[str] = []
    for tier_ids, tier_names, w in zh_tiers:
        src.append(tier_ids)
        names.append("taxonomy")
        weights.append(w)
        zh_used.extend(tier_names)
    debug["zh_classes"] = zh_used

    vec_ids: list[int] = []
    if embed is not None and store.vectors is not None:
        try:
            qv = embed.encode_one(p.text)
            # 全库内积 52万×1024 ≈ 0.5 GFLOP：BLAS 下几毫秒到几十毫秒，
            # 精确检索用不着 faiss/milvus 这类近似索引
            sc = store.vectors @ qv
            top = np.argpartition(-sc, min(R, len(sc) - 1))[:R]
            top = top[np.argsort(-sc[top])]
            vec_ids = [int(i) + 1 for i in top]  # 0-based -> rowid_
        except Exception as e:
            debug["vector_error"] = str(e)[:200]
    if vec_ids:
        src.append(vec_ids)
        names.append("vector")
        weights.append(1.5)
    debug["n_vector"] = len(vec_ids)

    if not src:
        # 三路都没召回（例如纯符号 query）—— 退化成过滤分页
        debug["notes"].append("无召回，退化为过滤分页")
        q = "SELECT rowid_ FROM assets"
        if p.where:
            q += " WHERE " + " AND ".join(p.where)
        q += " ORDER BY rowid_ LIMIT ? OFFSET ?"
        page = [r[0] for r in con.execute(q, list(p.params) + [limit, offset])]
        return {
            "hits": [Hit(int(i), 0.0, ["filter"]) for i in page],
            "total": _count_allowed(con, p),
            "parsed": p.__dict__,
            "debug": debug,
        }

    fused = rrf(src, weights)
    if p.where:
        cand = [rid for rid, _ in fused]
        keep = _filter_candidates(con, p, cand)
        ranked = [(rid, s) for rid, s in fused if rid in keep]
        debug["n_after_filter"] = len(ranked)
    else:
        ranked = fused
    total = len(ranked)

    src_of: dict[int, list[str]] = {}
    for lst, nm in zip(src, names):
        for rid in lst:
            src_of.setdefault(rid, []).append(nm)

    page = ranked[offset : offset + limit]
    hits = [Hit(rid, s, src_of.get(rid, [])) for rid, s in page]
    return {"hits": hits, "total": total, "parsed": p.__dict__, "debug": debug}
