from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from markitdown import MarkItDown

from models.document_models import DocumentRecord


def _iter_supported_files(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".pdf":
            yield path


def load_documents_from_directory(root_dir: str | Path) -> List[DocumentRecord]:
    """Convert local files with MarkItDown and return indexed document records."""

    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"input directory does not exist: {root_path}")

    converter = MarkItDown()
    documents: List[DocumentRecord] = []

    for doc_id, file_path in enumerate(_iter_supported_files(root_path), start=1):
        result = converter.convert(str(file_path))
        text = (result.text_content or "").strip()
        if not text:
            continue

        relative_source = file_path.relative_to(root_path).as_posix()
        category = file_path.parent.name if file_path.parent != root_path else root_path.name
        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                text=text,
                source=relative_source,
                category=category,
            )
        )

    return documents