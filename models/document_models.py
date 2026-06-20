from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DocumentRecord:
    doc_id: int
    text: str
    source: str
    category: str


@dataclass
class ParentDocumentRecord:
    parent_id: int
    source_id: int
    text: str
    source: str
    category: str
    section: str | None = None
    order: int = 0


@dataclass
class ChildChunkRecord:
    chunk_id: int
    parent_id: int
    source_id: int
    text: str
    source: str
    category: str
    sentence_index: int
    parent_start_sentence: int
    parent_end_sentence: int
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class CitationContextRecord:
    chunk_id: int
    parent_id: int
    source_id: int
    source: str
    category: str
    section: str | None
    sentence_index: int
    parent_start_sentence: int
    parent_end_sentence: int
    score: float
    chunk_text: str
    expanded_text: str


@dataclass
class RetrievedChunkRecord:
    score: float
    chunk_id: int
    metadata: Dict[str, Any]
    child_text: str
    parent_text: str
    expanded_text: str


@dataclass
class RagAnswerRecord:
    question: str
    answer: str
    results: List[RetrievedChunkRecord]
    context_bundle: List[Dict[str, Any]]
    prompt_context: str