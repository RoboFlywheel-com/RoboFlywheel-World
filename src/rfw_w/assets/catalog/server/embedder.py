# SPDX-License-Identifier: Apache-2.0
"""查询编码器 —— 与服务端同进程的 bge-m3 单例。

首次调用才加载模型（约 2.4GB fp16），之后常驻显存。默认 cuda:0，
显存不够可 ``SE_EMB_DEVICE=cpu`` 退回 CPU（查询编码单条 ~50ms，可接受）。
"""

from __future__ import annotations

import os
import threading

import numpy as np

from rfw_w.assets.catalog import config


class Embedder:
    def __init__(self, device: str | None = None):
        self.device = device or os.environ.get("SE_EMB_DEVICE", "cuda:0")
        self._model = None
        self._tok = None
        self._torch = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer

            dev = self.device
            if dev.startswith("cuda") and not torch.cuda.is_available():
                dev = "cpu"
            try:
                tok = AutoTokenizer.from_pretrained(
                    config.EMB_MODEL,
                    cache_dir=str(config.MODEL_DIR),
                    local_files_only=True,
                )
                model = AutoModel.from_pretrained(
                    config.EMB_MODEL,
                    cache_dir=str(config.MODEL_DIR),
                    local_files_only=True,
                )
            except Exception as e:  # 模型没下下来时不要拖垮整个服务
                raise RuntimeError(
                    f"未能加载 {config.EMB_MODEL}：{e}；"
                    f"请先跑 build_vectors 或在有网环境预热 {config.MODEL_DIR}"
                ) from e
            model.eval()
            if dev.startswith("cuda"):
                model.half()
            model.to(dev)
            self._tok, self._model, self._torch, self.device = tok, model, torch, dev

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        torch = self._torch
        with torch.no_grad():
            enc = self._tok(
                texts,
                padding=True,
                truncation=True,
                max_length=config.EMB_MAX_LEN,
                return_tensors="pt",
            ).to(self.device)
            h = self._model(**enc).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h.float(), p=2, dim=1)
            return h.cpu().numpy().astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_emb: Embedder | None = None


def get_embedder() -> Embedder:
    global _emb
    if _emb is None:
        _emb = Embedder()
    return _emb
