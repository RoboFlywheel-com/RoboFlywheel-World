# SPDX-License-Identifier: Apache-2.0
"""构建向量索引 —— 用 bge-m3 把每个资产的语义文本编码成 1024 维向量。

用法::

    python -m rfw_w.assets.catalog.index.build_vectors --device cuda:0
    python -m rfw_w.assets.catalog.index.build_vectors --limit 5000 --out /tmp/v   # 试跑

要点：

* **bge-m3 是跨语言模型**，中文口语 query（"带手柄的金属锅"）能直接命中英文标注，
  这是纯 BM25 做不到的 —— VLM 标注全部是英文。
* 文档里刻意拼进**中文分类名**，给中文 query 一个额外的语义锚点。
* 池化用 CLS（bge-m3 dense 的标准做法），输出做 L2 归一化，检索就是内积。
* DashScope 的 text-embedding 系列在本账号返回 Model.AccessDenied，所以走本地推理。
* 断点续跑：已编码文本的哈希缓存在 ``emb_cache.jsonl``，重跑只补增量。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np


from rfw_w.assets.catalog import config  # noqa: E402


def doc_text(a: sqlite3.Row, parts: list[sqlite3.Row]) -> str:
    """资产 -> 待编码文本。

    顺序刻意让「分类名」靠前（模型对前部更敏感），中文名紧挨英文名。
    """
    bits: list[str] = []
    for k in ("class_name", "class_zh", "domain_name", "domain_zh"):
        if a[k]:
            bits.append(str(a[k]))
    for k in ("object_guess", "material_guess", "surface_quality"):
        if a[k]:
            bits.append(str(a[k]))
    if a["shape_summary"]:
        bits.append(str(a["shape_summary"]))
    if a["key_features"]:
        kf = a["key_features"]
        try:
            kf = "、".join(json.loads(kf)) if str(kf).startswith("[") else kf
        except Exception:
            pass
        bits.append(str(kf))
    if a["part_names"]:
        bits.append(str(a["part_names"]))
    # 部件描述：最能体现功能语义，取前 4 个避免过长
    for p in parts[:4]:
        for k in ("basic_desc", "functional_desc"):
            if p[k]:
                bits.append(str(p[k]))
    return " | ".join(bits)[: config.DOC_TEXT_LIMIT]


def load_docs(db: str, limit: int | None):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    q = (
        "SELECT rowid_, key, class_name, class_zh, domain_name, domain_zh, "
        "object_guess, material_guess, surface_quality, shape_summary, "
        "key_features, part_names FROM assets ORDER BY rowid_"
    )
    if limit:
        q += f" LIMIT {int(limit)}"
    assets = con.execute(q).fetchall()
    pcon = con.execute(
        "SELECT key, basic_desc, functional_desc FROM parts ORDER BY key"
    )
    by_key: dict[str, list] = {}
    for p in pcon:
        by_key.setdefault(p["key"], []).append(p)
    con.close()
    texts, metas = [], []
    for a in assets:
        t = doc_text(a, by_key.get(a["key"], []))
        texts.append(t)
        metas.append({"rowid": a["rowid_"], "key": a["key"]})
    return texts, metas


def load_model(device: str, offline: bool = False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    tok = AutoTokenizer.from_pretrained(
        config.EMB_MODEL, cache_dir=str(config.MODEL_DIR), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        config.EMB_MODEL, cache_dir=str(config.MODEL_DIR), local_files_only=True
    )
    model.eval()
    if device.startswith("cuda"):
        model.half()
    model.to(device)
    return tok, model, torch


def encode(
    texts: list[str], device: str, batch: int, max_len: int, progress_every: int = 2000
) -> np.ndarray:
    tok, model, torch = load_model(device)
    out = np.empty((len(texts), config.EMB_DIM), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            # bge-m3 dense = CLS token
            h = model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h.float(), p=2, dim=1)
            out[i : i + len(chunk)] = h.cpu().numpy()
            if (i // batch) % 20 == 0:
                done = i + len(chunk)
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                print(
                    f"[vectors] {done:,}/{len(texts):,} {rate:.0f}/s "
                    f"ETA {(len(texts)-done)/max(rate,1e-6)/60:.1f}min",
                    flush=True,
                )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="bge-m3 向量索引")
    ap.add_argument("--db", default=str(config.CATALOG_DB))
    ap.add_argument("--out", default=str(config.VECTORS_NPY))
    ap.add_argument("--keys", default=str(config.VECTOR_KEYS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=config.EMB_BATCH)
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="保留已有向量，只补缺失行（按 rowid 对齐）",
    )
    args = ap.parse_args()

    config.ensure_dirs()
    t0 = time.time()
    print(f"[vectors] 读目录 {args.db}", flush=True)
    texts, metas = load_docs(args.db, args.limit)
    n = len(texts)
    empties = sum(1 for t in texts if not t.strip())
    print(
        f"[vectors] {n:,} 条文档（空文本 {empties}），模型 {config.EMB_MODEL}",
        flush=True,
    )

    out = Path(args.out)
    done_mask = np.zeros(n, dtype=bool)
    if args.resume and out.exists():
        prev = np.load(out, mmap_mode="r")
        if prev.shape == (n, config.EMB_DIM):
            arr = np.array(prev)
            # 归一化后模长应接近 1；为 0 的行视为未编码
            done_mask = np.linalg.norm(arr, axis=1) > 1e-6
            print(f"[vectors] 续跑：已有 {done_mask.sum():,} 条", flush=True)
        else:
            arr = np.zeros((n, config.EMB_DIM), dtype=np.float32)
    else:
        arr = np.zeros((n, config.EMB_DIM), dtype=np.float32)

    todo = np.flatnonzero(~done_mask)
    if len(todo):
        sub = [texts[i] for i in todo]
        vecs = encode(sub, args.device, args.batch, config.EMB_MAX_LEN)
        arr[todo] = vecs
    else:
        print("[vectors] 无需编码", flush=True)

    # 空文本（理论上没有）也给一个单位向量，避免检索里出现零向量
    norms = np.linalg.norm(arr, axis=1)
    zero = norms < 1e-6
    if zero.any():
        arr[zero, 0] = 1.0

    # 传文件句柄而不是路径：np.save 见到不是 .npy 结尾的路径会自作主张补后缀，
    # 结果写出 vectors.f32.npy，后面按 VECTORS_NPY 找就找不到了
    with open(out, "wb") as f:
        np.save(f, arr)
    Path(args.keys).write_text(
        json.dumps([m["key"] for m in metas], ensure_ascii=False)
    )
    Path(config.EMB_META).write_text(
        json.dumps(
            {
                "model": config.EMB_MODEL,
                "dim": config.EMB_DIM,
                "n": n,
                "max_len": config.EMB_MAX_LEN,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": args.device,
            },
            ensure_ascii=False,
            indent=1,
        )
    )

    print(
        f"[vectors] 完成 {n:,} × {config.EMB_DIM} -> {out} "
        f"({out.stat().st_size/1e9:.2f}GB)，耗时 {(time.time()-t0)/60:.1f}min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
