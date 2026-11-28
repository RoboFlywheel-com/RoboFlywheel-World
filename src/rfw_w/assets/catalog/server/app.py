# SPDX-License-Identifier: Apache-2.0
"""FastAPI 服务：资产检索和详情（仅 JSON API，不含网页前端）。

启动::

    python -m rfw_w.assets.catalog.server.app --port 8848
"""

from __future__ import annotations

import argparse
import json
import time

from fastapi import FastAPI, HTTPException

from rfw_w.assets.catalog import config  # noqa: E402
from rfw_w.assets.catalog.server import query  # noqa: E402
from rfw_w.assets.catalog.server.embedder import get_embedder  # noqa: E402
from rfw_w.assets.catalog.server.store import get_store  # noqa: E402

app = FastAPI(title="SimReady 资产搜索引擎", version="0.1")
store = get_store()


@app.on_event("startup")
def _warmup():
    """后台预热：向量矩阵（2.1GB）和 bge-m3 权重。

    不做的话第一次检索要等十几秒（读 2.1GB + 加载 568M 参数的模型），
    用户会以为服务卡死了。放到后台线程，不阻塞启动。
    """
    import threading

    def work():
        t = time.time()
        try:
            v = store.vectors
            if v is not None:
                print(f"[warmup] 向量 {v.shape} 就绪 {time.time()-t:.1f}s", flush=True)
        except Exception as e:
            print(f"[warmup] 向量加载失败：{e}", flush=True)
        t = time.time()
        try:
            e = get_embedder()
            if e.available:
                e.encode_one("warmup")
                print(
                    f"[warmup] bge-m3 就绪（{e.device}）{time.time()-t:.1f}s",
                    flush=True,
                )
            else:
                print("[warmup] bge-m3 不可用，检索将只有关键词和分类通道", flush=True)
        except Exception as e:
            print(f"[warmup] 模型加载失败：{e}", flush=True)

    threading.Thread(target=work, name="warmup", daemon=True).start()


# ── 序列化 ────────────────────────────────────────────────────────────────

CARD_FIELDS = (
    "key",
    "class_name",
    "class_zh",
    "domain_name",
    "domain_zh",
    "final_domain",
    "final_class",
    "final_status",
    "is_new_class",
    "total_mass",
    "real_height",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "n_parts",
    "n_collision_hulls",
    "part_n_graspable",
    "whole_mu1",
    "whole_mu2",
    "main_material",
    "main_material_subtype",
    "has_export",
    "has_urdf",
    "has_render",
    "has_affordance",
    "object_guess",
    "material_guess",
    "surface_quality",
    "shape_summary",
    "part_names",
)


def card(row, score: float = 0.0, sources: list[str] | None = None) -> dict:
    d = {k: row[k] for k in CARD_FIELDS}
    d["score"] = round(score, 6)
    d["sources"] = sources or []
    d["glb_rel"] = row["glb_rel"]
    return d


# ── 检索 ──────────────────────────────────────────────────────────────────


@app.get("/api/search")
def api_search(
    q: str = "",
    limit: int = config.DEFAULT_LIMIT,
    offset: int = 0,
    with_vectors: bool = True,
):
    limit = max(1, min(limit, config.MAX_LIMIT))
    embed = get_embedder() if with_vectors else None
    t0 = time.time()
    res = query.search(store, q, limit=limit, offset=offset, embed=embed)
    rows = store.rows_by_ids([h.rowid for h in res["hits"]])
    items = []
    for h in res["hits"]:
        r = rows.get(h.rowid)
        if r is not None:
            items.append(card(r, h.score, h.sources))
    return {
        "q": q,
        "items": items,
        "total": res["total"],
        "offset": offset,
        "limit": limit,
        "took_ms": round((time.time() - t0) * 1000, 1),
        "parsed": {
            "text": res["parsed"].get("text"),
            "filters": res["parsed"].get("where"),
        },
        "debug": res["debug"],
    }


@app.get("/api/asset/{key}")
def api_asset(key: str):
    a = store.asset(key)
    if a is None:
        raise HTTPException(404, f"unknown key {key}")
    a["sources"] = {"export_rel": a["export_rel"]}
    try:
        a["inertia"] = json.loads(a["inertia"]) if a.get("inertia") else None
    except Exception:
        pass
    return a


@app.get("/api/facets")
def api_facets():
    return store.facets()


@app.get("/api/taxonomy")
def api_taxonomy():
    return store.taxonomy()


@app.get("/api/health")
def api_health():
    m = store.meta()
    v = store.vectors
    return {
        "db": str(config.CATALOG_DB),
        "catalog_rows": m.get("n_rows"),
        "built_at": m.get("built_at"),
        "verified_at": m.get("verified_at"),
        "n_export": m.get("n_export"),
        "vectors": None if v is None else list(v.shape),
        "embed_model": config.EMB_MODEL,
        "data_root": str(config.DATA_ROOT),
        "export_root": str(config.EXPORT_ROOT),
        "export_root_exists": config.EXPORT_ROOT.exists(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SimReady 资产搜索服务")
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    if not config.CATALOG_DB.exists():
        print(
            f"[server] 目录索引不存在：{config.CATALOG_DB}\n"
            f"         先跑 python -m rfw_w.assets.catalog.index.build_catalog",
            flush=True,
        )
    import uvicorn

    uvicorn.run(
        "rfw_w.assets.catalog.server.app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
