from __future__ import annotations

import re
from typing import List, Sequence

from models.document_models import ChildChunkRecord, DocumentRecord, ParentDocumentRecord


def split_sentences(text: str) -> List[str]:
    segments = re.split(r"(?<=[.!?])\s+", text.strip())
    return [segment.strip() for segment in segments if segment and segment.strip()]


def sentence_offsets(text: str, sentences: Sequence[str]) -> List[tuple[int, int]]:
    offsets: List[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        match_index = text.find(sentence, cursor)
        if match_index == -1:
            match_index = cursor
        start = match_index
        end = start + len(sentence)
        offsets.append((start, end))
        cursor = end
    return offsets


def build_parent_child_records(
    documents: Sequence[DocumentRecord],
    window_size: int,
) -> tuple[
    list[ParentDocumentRecord],
    list[ChildChunkRecord],
    list[str],
    dict[int, list[str]],
]:
    parents: list[ParentDocumentRecord] = []
    chunks: list[ChildChunkRecord] = []
    texts_to_embed: list[str] = []
    parent_sentences: dict[int, list[str]] = {}

    for order, document in enumerate(documents):
        parent = ParentDocumentRecord(
            parent_id=document.doc_id,
            source_id=document.doc_id,
            text=document.text,
            source=document.source,
            category=document.category,
            order=order,
        )
        parents.append(parent)

        sentences = split_sentences(document.text)
        if not sentences:
            continue

        offsets = sentence_offsets(document.text, sentences)
        parent_sentences[parent.parent_id] = sentences

        for sentence_index, (sentence, (char_start, char_end)) in enumerate(zip(sentences, offsets)):
            chunk_id = parent.parent_id * 1000 + sentence_index + 1
            chunks.append(
                ChildChunkRecord(
                    chunk_id=chunk_id,
                    parent_id=parent.parent_id,
                    source_id=document.doc_id,
                    text=sentence,
                    source=document.source,
                    category=document.category,
                    sentence_index=sentence_index,
                    parent_start_sentence=max(0, sentence_index - window_size),
                    parent_end_sentence=min(len(sentences) - 1, sentence_index + window_size),
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            texts_to_embed.append(sentence)

    return parents, chunks, texts_to_embed, parent_sentences