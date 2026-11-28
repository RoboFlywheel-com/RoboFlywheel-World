# SPDX-License-Identifier: Apache-2.0
"""Explicit local catalogue configuration; no source-tree or cloud defaults."""

from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SE_DATA_ROOT", "/path/to/release"))
EXPORT_ROOT = Path(os.environ.get("SE_EXPORT_ROOT", str(DATA_ROOT)))
INDEX_DIR = Path(os.environ.get("SE_INDEX_DIR", "/tmp/rfw-catalog/index"))
MODEL_DIR = Path(os.environ.get("SE_MODEL_DIR", "/path/to/models"))
CATALOG_DB = INDEX_DIR / "catalog.db"
VECTORS_NPY = INDEX_DIR / "vectors.f32"
VECTOR_KEYS = INDEX_DIR / "keys.json"
EMB_META = INDEX_DIR / "embed_meta.json"
TAXONOMY_CSV = Path(os.environ.get("SE_TAXONOMY_CSV", "/path/to/taxonomy.csv"))
EMB_MODEL = os.environ.get("SE_EMB_MODEL", str(MODEL_DIR / "bge-m3"))
EMB_DIM = int(os.environ.get("SE_EMB_DIM", "1024"))
EMB_BATCH = int(os.environ.get("SE_EMB_BATCH", "64"))
EMB_MAX_LEN = int(os.environ.get("SE_EMB_MAX_LEN", "256"))
DOC_TEXT_LIMIT = int(os.environ.get("SE_DOC_TEXT_LIMIT", "1200"))
HOST = os.environ.get("SE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SE_PORT", "8848"))
DEFAULT_LIMIT = 40
MAX_LIMIT = 200
SCAN_WORKERS = 16


def ensure_dirs():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
