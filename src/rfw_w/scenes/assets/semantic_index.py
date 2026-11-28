# SPDX-License-Identifier: Apache-2.0
"""SBERT class-vector indexes for semantic asset recall.

The core retrieval pipeline stays metadata-only. This module only loads class
vectors and lazily encodes query text; meshes remain untouched until a binding
has passed the downstream size and simulation-readiness gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import json
from math import sqrt, isfinite
from pathlib import Path
import pickle
import re
import threading
from typing import Any, Mapping, Protocol, Sequence


SEMANTIC_INDEX_SCHEMA_VERSION = "pvscene.semantic-class-index/v1"
DEFAULT_SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"


class TextEmbeddingEncoder(Protocol):
    model_id: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def encoder_model_identity(encoder: TextEmbeddingEncoder) -> str:
    """Canonical model name, independent of the local deployment path."""
    encoder_id = encoder.model_id
    if Path(encoder_id).is_dir():
        config_path = Path(encoder_id) / "config.json"
        config = json.loads(config_path.read_bytes()) if config_path.is_file() else {}
        encoder_id = config.get("rfw_model_id") or config.get("_name_or_path")
        if not encoder_id:
            raise ValueError("local encoder config requires rfw_model_id for index identity validation")
    return encoder_id


class SentenceTransformerEncoder:
    """Lazy all-mpnet-base-v2 encoder with a transformers-only fallback."""

    def __init__(
        self,
        model_id: str = DEFAULT_SBERT_MODEL,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        if not model_id or not device:
            raise ValueError("SBERT model_id and device must not be empty")
        self.model_id = model_id
        self.device = device
        self.local_files_only = local_files_only
        self._backend: tuple[str, Any, Any | None] | None = None
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        clean = tuple(str(text).strip() for text in texts)
        if not clean or any(not text for text in clean):
            raise ValueError("SBERT encoder requires non-empty text")
        backend, model, tokenizer = self._ensure_backend()
        if backend == "sentence_transformers":
            return model.encode(
                list(clean),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        import torch
        import torch.nn.functional as functional

        assert tokenizer is not None
        batch = tokenizer(
            list(clean),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return functional.normalize(pooled, p=2, dim=1).cpu().tolist()

    def _ensure_backend(self) -> tuple[str, Any, Any | None]:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            if importlib.util.find_spec("sentence_transformers") is not None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    local_files_only=self.local_files_only,
                )
                self._backend = ("sentence_transformers", model, None)
                return self._backend
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "SBERT retrieval requires sentence-transformers, or both "
                    "transformers and torch; install one runtime before using "
                    "--semantic-index"
                ) from error
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
            model = AutoModel.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            ).to(self.device)
            model.eval()
            self._backend = ("transformers", model, tokenizer)
            return self._backend


@dataclass(frozen=True, slots=True)
class SemanticClassIndex:
    classes: tuple[tuple[str, str], ...]
    vectors: Any
    model_id: str
    content_hash: str
    source_path: Path

    def __post_init__(self) -> None:
        if not self.classes or not self.model_id or not self.content_hash:
            raise ValueError("semantic index identity and classes must not be empty")
        if len(self.vectors) != len(self.classes):
            raise ValueError("semantic index class/vector count mismatch")
        dimensions = {len(vector) for vector in self.vectors}
        if len(dimensions) != 1 or next(iter(dimensions), 0) <= 0:
            raise ValueError("semantic index vectors must have one positive dimension")
        for vector in self.vectors:
            _normalized(vector)


class SbertSemanticScorer:
    """Score catalog semantic classes against an SBERT class-vector index."""

    def __init__(
        self,
        index: SemanticClassIndex,
        *,
        encoder: TextEmbeddingEncoder | None = None,
        lexical_fallback: Any | None = None,
    ) -> None:
        self.index = index
        self.encoder = encoder or SentenceTransformerEncoder(index.model_id)
        encoder_id = encoder_model_identity(self.encoder)
        if encoder_id != index.model_id:
            raise ValueError("encoder model identity does not match semantic index")
        self.lexical_fallback = lexical_fallback
        # The protocol must match across provider-specific indexes for paired
        # evaluation. The exact index hash is part of the catalog hash instead.
        self.fingerprint = f"sbert-class-cosine-v1:{index.model_id}"
        self.index_content_hash = index.content_hash
        by_label: dict[str, list[int]] = {}
        for index_position, (_, semantic_class) in enumerate(index.classes):
            by_label.setdefault(_normalize_label(semantic_class), []).append(
                index_position
            )
        self._by_label = {key: tuple(value) for key, value in by_label.items()}
        self._query_cache: dict[tuple[str, ...], tuple[Any, ...]] = {}
        self._score_cache: dict[tuple[str, ...], dict[str, float]] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        encoder: TextEmbeddingEncoder | None = None,
        lexical_fallback: Any | None = None,
    ) -> "SbertSemanticScorer":
        return cls(
            load_semantic_class_index(path),
            encoder=encoder,
            lexical_fallback=lexical_fallback,
        )

    def score(self, query_terms: Sequence[str], candidate: str) -> float:
        return self.score_many(query_terms, (candidate,))[candidate]

    def score_many(
        self,
        query_terms: Sequence[str],
        candidates: Sequence[str],
    ) -> dict[str, float]:
        query_key = tuple(str(item) for item in query_terms)
        with self._cache_lock:
            cached = self._score_cache.get(query_key)
        if cached is None:
            cached = self._all_class_scores(query_key)
            with self._cache_lock:
                self._score_cache[query_key] = cached
        output = {}
        for candidate in candidates:
            label = _normalize_label(candidate)
            output[candidate] = (
                cached[label]
                if label in cached
                else self._fallback(query_key, candidate)
            )
        return output

    def precompute(self, query_term_groups: Sequence[Sequence[str]]) -> None:
        """Encode all uncached frozen queries in one model batch."""

        groups = tuple(dict.fromkeys(tuple(str(item) for item in group) for group in query_term_groups))
        missing = [group for group in groups if group not in self._query_cache]
        if not missing:
            return
        spans = []
        texts = []
        for group in missing:
            group_texts = _query_texts(group)
            spans.append((len(texts), len(texts) + len(group_texts)))
            texts.extend(group_texts)
        vectors = self.encoder.encode(tuple(texts))
        if len(vectors) != len(texts):
            raise ValueError("SBERT encoder returned the wrong vector count")
        normalized = tuple(_normalized(vector) for vector in vectors)
        with self._cache_lock:
            for group, (start, end) in zip(missing, spans):
                self._query_cache[group] = normalized[start:end]

    def _all_class_scores(self, query_terms: tuple[str, ...]) -> dict[str, float]:
        queries = self._query_vectors(query_terms)
        try:
            import numpy as np

            query_matrix = np.asarray(queries, dtype=np.float32)
            class_matrix = np.asarray(self.index.vectors, dtype=np.float32)
            class_norms = np.linalg.norm(class_matrix, axis=1, keepdims=True)
            class_matrix = class_matrix / np.maximum(class_norms, 1e-12)
            similarities = query_matrix @ class_matrix.T
            raw_scores = similarities.max(axis=0).tolist()
        except (AttributeError, ImportError, TypeError, ValueError):
            raw_scores = [
                max(_cosine(query, vector) for query in queries)
                for vector in self.index.vectors
            ]
        output: dict[str, float] = {}
        for position, (_, semantic_class) in enumerate(self.index.classes):
            label = _normalize_label(semantic_class)
            output[label] = max(
                output.get(label, 0.0),
                max(0.0, min(1.0, float(raw_scores[position]))),
            )
        return output

    def _legacy_score(self, query_terms: Sequence[str], candidate: str) -> float:
        positions = self._by_label.get(_normalize_label(candidate), ())
        if not positions:
            return self._fallback(query_terms, candidate)
        queries = self._query_vectors(tuple(str(item) for item in query_terms))
        best = max(
            _cosine(query, self.index.vectors[position])
            for query in queries
            for position in positions
        )
        return max(0.0, min(1.0, best))

    def _query_vectors(self, query_terms: tuple[str, ...]) -> tuple[Any, ...]:
        with self._cache_lock:
            cached = self._query_cache.get(query_terms)
        if cached is not None:
            return cached
        texts = _query_texts(query_terms)
        vectors = tuple(self.encoder.encode(texts))
        if len(vectors) != len(texts):
            raise ValueError("SBERT encoder returned the wrong vector count")
        normalized = tuple(_normalized(vector) for vector in vectors)
        with self._cache_lock:
            self._query_cache[query_terms] = normalized
        return normalized

    def _fallback(self, query_terms: Sequence[str], candidate: str) -> float:
        if self.lexical_fallback is None:
            return 0.0
        return float(self.lexical_fallback.score(query_terms, candidate))


def load_semantic_class_index(path: str | Path) -> SemanticClassIndex:
    """Load the safe v1 format or the trusted legacy SBERT pickle."""

    source = Path(path).expanduser().resolve()
    raw_bytes = source.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if source.suffix in {".pkl", ".pickle"}:
        try:
            payload = pickle.loads(raw_bytes)
        except ModuleNotFoundError as error:
            raise RuntimeError(
                f"legacy semantic index {source} requires its array runtime: {error}"
            ) from error
        return _index_from_payload(
            payload,
            source=source,
            content_hash=digest,
            legacy=True,
        )
    if source.suffix == ".gz":
        payload = json.loads(gzip.decompress(raw_bytes))
    else:
        payload = json.loads(raw_bytes)
    return _index_from_payload(
        payload,
        source=source,
        content_hash=digest,
        legacy=False,
    )


def build_semantic_class_index(
    catalog_path: str | Path,
    output_path: str | Path,
    *,
    encoder: TextEmbeddingEncoder | None = None,
) -> Path:
    """Build a portable JSON/gzip class index from a legacy or v1 catalog."""

    catalog_source = Path(catalog_path).expanduser().resolve()
    raw_bytes = catalog_source.read_bytes()
    payload = json.loads(raw_bytes)
    pairs = _class_pairs(payload)
    if not pairs:
        raise ValueError(f"catalog has no semantic classes: {catalog_source}")
    text_encoder = encoder or SentenceTransformerEncoder()
    model_identity = encoder_model_identity(text_encoder)
    texts = tuple(
        f"{semantic_class}, {domain}" if domain else semantic_class
        for domain, semantic_class in pairs
    )
    vectors = text_encoder.encode(texts)
    if len(vectors) != len(pairs):
        raise ValueError("SBERT encoder returned the wrong vector count")
    output = {
        "schema_version": SEMANTIC_INDEX_SCHEMA_VERSION,
        "model_id": model_identity,
        "source_catalog": str(catalog_source),
        "source_catalog_hash": hashlib.sha256(raw_bytes).hexdigest(),
        "classes": [[domain, semantic] for domain, semantic in pairs],
        "text_features": [list(map(float, _normalized(vector))) for vector in vectors],
    }
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        output,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.tmp")
    if target.suffix == ".gz":
        temporary.write_bytes(gzip.compress(encoded, mtime=0))
    else:
        temporary.write_bytes(encoded)
    temporary.replace(target)
    return target


def _index_from_payload(
    payload: Any,
    *,
    source: Path,
    content_hash: str,
    legacy: bool,
) -> SemanticClassIndex:
    if not isinstance(payload, Mapping):
        raise ValueError("semantic index must be an object")
    if not legacy and payload.get("schema_version") != SEMANTIC_INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported semantic index schema: {payload.get('schema_version')!r}"
        )
    raw_classes = payload.get("classes")
    vectors = payload.get("text_features")
    if not isinstance(raw_classes, Sequence) or isinstance(raw_classes, (str, bytes)):
        raise ValueError("semantic index classes must be a list")
    if vectors is None or not hasattr(vectors, "__len__"):
        raise ValueError("semantic index text_features must be an array")
    classes = []
    for raw_class in raw_classes:
        if (
            not isinstance(raw_class, Sequence)
            or isinstance(raw_class, (str, bytes))
            or len(raw_class) != 2
        ):
            raise ValueError("semantic index class keys must be [domain, class]")
        domain, semantic_class = (str(value).strip() for value in raw_class)
        if not semantic_class:
            raise ValueError("semantic index class names must not be empty")
        classes.append((domain, semantic_class))
    model_id = str(payload.get("model_id") or DEFAULT_SBERT_MODEL)
    return SemanticClassIndex(
        classes=tuple(classes),
        vectors=vectors,
        model_id=model_id,
        content_hash=content_hash,
        source_path=source,
    )


def _class_pairs(payload: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, Mapping):
        raise ValueError("catalog JSON must be an object")
    raw_assets = payload.get("assets")
    records: Sequence[Any]
    if isinstance(raw_assets, list):
        records = raw_assets
    elif "assets" in payload:
        raise ValueError("catalog assets must be a list")
    else:
        records = list(payload.values())
    pairs = {
        (
            str(record.get("domain", "")).strip(),
            str(
                record.get("semantic_class") or record.get("super_category") or ""
            ).strip(),
        )
        for record in records
        if isinstance(record, Mapping)
    }
    return tuple(sorted(pair for pair in pairs if pair[1]))


def _query_texts(query_terms: tuple[str, ...]) -> tuple[str, ...]:
    clean = tuple(item.strip() for item in query_terms if item.strip())
    if not clean:
        raise ValueError("semantic query terms must not be empty")
    description = clean[-1]
    semantics = clean[:-1] or clean
    return tuple(dict.fromkeys(f"{semantic}. {description}" for semantic in semantics))


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _normalized(vector: Sequence[float]) -> Any:
    if any(not isfinite(float(value)) for value in vector):
        raise ValueError("semantic vectors must be finite")
    norm = sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 1e-12:
        raise ValueError("semantic vectors must not be zero")
    try:
        return vector / norm
    except TypeError:
        return tuple(float(value) / norm for value in vector)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("query and semantic index dimensions do not match")
    if any(not isfinite(float(value)) for value in (*left, *right)):
        raise ValueError("semantic vectors must be finite")
    left_norm = sqrt(sum(float(value) * float(value) for value in left))
    right_norm = sqrt(sum(float(value) * float(value) for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right)) / (
        left_norm * right_norm
    )
