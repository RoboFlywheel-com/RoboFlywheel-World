# SPDX-License-Identifier: Apache-2.0
"""Offline preparation of the retrieval runtime's local model files.

The SBERT weights for ``all-mpnet-base-v2`` are present on this machine, but only as a bare
HuggingFace ``blobs/`` directory: the ``snapshots/<commit>/`` tree that
``from_pretrained(local_files_only=True)`` reads is missing, so offline loading fails even
though the weights are sitting right there.

This module rebuilds a loadable model directory from those blobs, and it does so as an
explicit, recorded preparation step rather than a hidden workaround:

* files are identified by **content**, not by guessing from blob hashes, and an
  unrecognised blob is an error rather than a best-effort guess;
* nothing is downloaded. If a required file is absent the preparation fails and says which
  one, because the acceptance manual forbids hiding a weight fetch inside a generation run;
* the result carries a :class:`PreparationRecord` with the method, version and the exact
  blob each file came from, so a later reviewer can see where a number came from.

The rebuilt directory is deliberately written to a fresh location rather than back into the
shared cache, so a preparation cannot corrupt data other projects depend on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "pvscene.prepared-store/v1"

# Files a transformers-only load needs. `tokenizer.json` and the sentence-transformers
# module configs are carried through when present so that sentence-transformers can also
# load the same directory, but they are not required for the loading path this uses.
REQUIRED_FILES = (
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
)

# Exactly one weight file must be present, in whichever container the publisher used.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

# Torch checkpoints are zip archives or legacy pickle streams; safetensors is a length-prefixed
# JSON header followed by the tensor data.
_ZIP_MAGIC = b"PK\x03\x04"
_PICKLE_MAGIC = b"\x80"


class PreparationError(RuntimeError):
    """The local files could not be turned into a loadable model directory."""


@dataclass(frozen=True, slots=True)
class PreparationRecord:
    """Provenance for a prepared artefact."""

    subject: str
    method: str
    version: str
    source: str
    target: str
    files: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "subject": self.subject,
            "method": self.method,
            "version": self.version,
            "source": self.source,
            "target": self.target,
            "files": dict(sorted(self.files.items())),
        }


def _sniff_text(path: Path, limit: int = 4096) -> str | None:
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit)
        return raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _identify_weight(path: Path) -> str | None:
    """Name a weight file by its container format rather than by its size or name.

    Naming it from the size alone is how a safetensors blob gets called
    ``pytorch_model.bin``: the two are indistinguishable by extension and both are large, and
    the resulting load failure ("unable to load weights from pytorch checkpoint") points at
    the file rather than at the misnaming. So the format is read, not assumed.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
            if head[:4] == _ZIP_MAGIC or head[:1] == _PICKLE_MAGIC:
                return "pytorch_model.bin"
            length = int.from_bytes(head, "little")
            # A safetensors header is a bounded JSON object; a plausible length plus a JSON
            # opening brace is what distinguishes it from arbitrary binary.
            if 0 < length < 100_000_000:
                header = handle.read(min(length, 64))
                if header.lstrip().startswith(b"{"):
                    return "model.safetensors"
    except OSError:
        return None
    return None


def identify_blob(path: Path) -> str | None:
    """Name a single blob by inspecting its content.

    Returns ``None`` when the content matches nothing known. Rules are checked most specific
    first, because several of these files are JSON objects that share keys.
    """
    if _identify_weight(path) is not None:
        return _identify_weight(path)

    text = _sniff_text(path)
    if text is None:
        return None
    stripped = text.lstrip()

    if stripped.startswith("---") and "language:" in text[:200]:
        return "README.md"
    if stripped.startswith("<s> <pad>") or "\n[PAD]" in text:
        return "vocab.txt"

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return _identify_json(payload)
    return None


def _identify_json(payload: Any) -> str | None:
    """Name a JSON blob from its distinguishing keys."""
    if isinstance(payload, list):
        # modules.json is the only top-level list, and its entries name sentence-transformers
        # module classes.
        if payload and isinstance(payload[0], dict) and "type" in payload[0]:
            return "modules.json"
        return None

    if not isinstance(payload, dict):
        return None

    if "word_embedding_dimension" in payload:
        return "1_Pooling/config.json"
    if "max_seq_length" in payload and "do_lower_case" in payload:
        return "sentence_bert_config.json"
    if "__version__" in payload:
        return "config_sentence_transformers.json"
    if "architectures" in payload:
        return "config.json"
    if "truncation" in payload and "padding" in payload:
        return "tokenizer.json"
    if "do_lower_case" in payload and "bos_token" in payload:
        return "tokenizer_config.json"
    if "bos_token" in payload and "unk_token" in payload:
        return "special_tokens_map.json"
    return None


def identify_blobs(model_dir: Path) -> dict[str, Path]:
    """Map every blob in a HuggingFace model cache entry to a filename."""
    blobs = model_dir / "blobs"
    if not blobs.is_dir():
        raise PreparationError(f"no blobs directory under {model_dir}")

    identified: dict[str, Path] = {}
    unidentified: list[str] = []
    for blob in sorted(blobs.iterdir()):
        if not blob.is_file():
            continue
        name = identify_blob(blob)
        if name is None:
            unidentified.append(blob.name)
            continue
        if name in identified:
            raise PreparationError(
                f"two blobs both look like {name!r}: {identified[name].name} and {blob.name}; "
                "an ambiguous cache cannot be prepared reproducibly"
            )
        identified[name] = blob

    if unidentified:
        raise PreparationError(
            f"could not identify {len(unidentified)} blob(s) in {model_dir}: {sorted(unidentified)}; "
            "refusing to guess, because a mis-named file would load as a wrong model rather than fail"
        )
    return identified


def snapshot_tree_present(model_dir: Path) -> bool:
    """Whether the cache has a usable ``snapshots/<commit>/`` tree.

    Checked before preparing, because rebuilding over an intact cache would replace working
    symlinks with copies for no reason.
    """
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(entry.is_dir() and any(entry.iterdir()) for entry in snapshots.iterdir())


def materialize_local_model(model_dir: Path, target_dir: Path) -> PreparationRecord:
    """Build a loadable model directory from a blobs-only cache entry.

    Files are copied as symlinks where the filesystem allows it, so the several hundred
    megabytes of weights are not duplicated. A symlink that cannot be created falls back to a
    copy, and the record says which happened.
    """
    if snapshot_tree_present(model_dir):
        raise PreparationError(
            f"{model_dir} already has a snapshots tree; nothing to prepare. Point the runtime at "
            "the cache directly instead of duplicating it"
        )

    identified = identify_blobs(model_dir)
    missing = [name for name in REQUIRED_FILES if name not in identified]
    weights = [name for name in WEIGHT_FILES if name in identified]
    if not weights:
        missing.append(f"one of {list(WEIGHT_FILES)}")
    if missing:
        raise PreparationError(
            f"cache entry {model_dir} is missing required file(s) {missing}; only "
            f"{sorted(identified)} are present. Refusing to substitute or download"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, blob in sorted(identified.items()):
        destination = target_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        try:
            destination.symlink_to(blob.resolve())
            how = "symlink"
        except OSError:
            import shutil as _shutil

            _shutil.copy2(blob, destination)
            how = "copy"
        written[name] = f"{how} <- blobs/{blob.name}"

    # Add identity to the local copy, never to the shared cache blob.
    local_config = target_dir / "config.json"
    payload = json.loads(local_config.read_bytes())
    payload["rfw_model_id"] = "sentence-transformers/all-mpnet-base-v2"
    if local_config.is_symlink():
        local_config.unlink()
    local_config.write_text(json.dumps(payload, indent=2) + "\n")
    written["config.json"] = "copy <- " + written["config.json"].split(" <- ", 1)[1] + " (added rfw_model_id)"

    return PreparationRecord(
        subject="sentence-transformers/all-mpnet-base-v2",
        method="reconstruct_local_model_from_hf_blobs",
        version="v1",
        source=str(model_dir),
        target=str(target_dir),
        files=written,
    )


def prepare_if_needed(hf_cache: Path, target_dir: Path) -> tuple[PreparationRecord | None, str]:
    """Prepare the local model directory unless it already exists.

    Returns the record and a short status, so a caller can report "already prepared" without
    re-preparing and without pretending it did work it did not do.
    """
    model_id = "models--sentence-transformers--all-mpnet-base-v2"
    model_dir = hf_cache / "hub" / model_id
    if not model_dir.is_dir():
        raise PreparationError(f"no model cache entry at {model_dir}")

    marker = target_dir / "config.json"
    if marker.is_file():
        return None, f"already prepared at {target_dir}"

    if snapshot_tree_present(model_dir):
        return None, f"cache at {model_dir} is directly loadable; no preparation needed"

    return materialize_local_model(model_dir, target_dir), f"prepared into {target_dir}"
