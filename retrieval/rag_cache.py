from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from models.document_models import ChildChunkRecord, DocumentRecord, ParentDocumentRecord


def _stable_payload(documents: Sequence[DocumentRecord], window_size: int) -> str:
    parts: list[str] = [f"window_size={window_size}"]
    for document in documents:
        parts.append(f"{document.doc_id}|{document.source}|{document.category}|{document.text}")
    return "\n".join(parts)


def compute_cache_key(documents: Sequence[DocumentRecord], window_size: int) -> str:
    payload = _stable_payload(documents, window_size)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cache_paths(cache_dir: str | Path, cache_key: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return root / f"{cache_key}.npz", root / f"{cache_key}.json"


def load_cached_artifacts(cache_dir: str | Path, cache_key: str) -> Dict[str, Any] | None:
    vectors_path, metadata_path = get_cache_paths(cache_dir, cache_key)
    if not vectors_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(vectors_path, allow_pickle=False) as data:
        vectors = data["vectors"]

    return {"vectors": vectors, "metadata": metadata}


def save_cached_artifacts(
    cache_dir: str | Path,
    cache_key: str,
    vectors: np.ndarray,
    parents: Sequence[ParentDocumentRecord],
    chunks: Sequence[ChildChunkRecord],
    parent_sentences: Dict[int, list[str]],
) -> None:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    vectors_path, metadata_path = get_cache_paths(root, cache_key)
    np.savez_compressed(vectors_path, vectors=vectors.astype("float32"))
    metadata = {
        "parents": [asdict(parent) for parent in parents],
        "chunks": [asdict(chunk) for chunk in chunks],
        "parent_sentences": {str(parent_id): sentences for parent_id, sentences in parent_sentences.items()},
        "dimension": int(vectors.shape[1]),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")