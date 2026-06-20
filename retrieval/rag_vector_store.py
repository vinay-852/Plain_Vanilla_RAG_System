from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import faiss
import numpy as np
from langchain_core.documents import Document as LangChainDocument

from chunking.parent_child_chunking import build_parent_child_records
from retrieval.rag_cache import compute_cache_key, load_cached_artifacts, save_cached_artifacts
from llm import embed_texts, generate_response
from models.document_models import (
    ChildChunkRecord,
    CitationContextRecord,
    DocumentRecord,
    ParentDocumentRecord,
    RagAnswerRecord,
    RetrievedChunkRecord,
)

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """FAISS stores vectors only; document metadata lives alongside the index."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self.metadata: Dict[int, Dict[str, Any]] = {}

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, vectors: np.ndarray, docs: List[DocumentRecord]) -> None:
        logger.debug("Adding documents to FAISS: count=%d dimension=%d", len(docs), self.dimension)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"expected vectors with dimension {self.dimension}")
        if len(vectors) != len(docs):
            raise ValueError("number of vectors must match number of documents")

        vectors = self._normalize(vectors)
        ids = np.array([doc.doc_id for doc in docs], dtype="int64")

        self.index.add_with_ids(vectors, ids)
        for doc in docs:
            self.metadata[doc.doc_id] = asdict(doc)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        logger.debug("Searching FAISS: top_k=%d dimension=%d", top_k, self.dimension)
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D array")
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"expected query vector with dimension {self.dimension}")

        query_vector = query_vector.astype("float32").reshape(1, -1)
        query_vector = self._normalize(query_vector)

        scores, ids = self.index.search(query_vector, top_k)
        results: List[Dict[str, Any]] = []

        for score, doc_id in zip(scores[0], ids[0]):
            if doc_id == -1:
                continue
            metadata = self.metadata.get(int(doc_id), {})
            results.append({"score": float(score), "doc_id": int(doc_id), "metadata": metadata})

        return results

    def get_metadata(self, doc_id: int) -> Optional[Dict[str, Any]]:
        return self.metadata.get(doc_id)


class SmallToLargeFaissStore:
    """FAISS-backed small-to-large retrieval store with citation-ready metadata."""

    def __init__(self, dimension: int | None = None, window_size: int = 2) -> None:
        self.dimension = dimension
        self.window_size = window_size
        self.index: faiss.IndexIDMap2 | None = None
        self.chunk_metadata: Dict[int, Dict[str, Any]] = {}
        self.parents: Dict[int, ParentDocumentRecord] = {}
        self.parent_children: Dict[int, List[int]] = defaultdict(list)
        self.parent_sentences: Dict[int, List[str]] = {}
        self.child_vectors_ready = False

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        return vectors

    def _ensure_index(self, dimension: int) -> None:
        if self.dimension is None:
            self.dimension = dimension
        elif self.dimension != dimension:
            raise ValueError(f"expected vectors with dimension {self.dimension}")

        if self.index is None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))

    def _load_cached_state(self, cache_payload: Dict[str, Any]) -> List[ChildChunkRecord]:
        metadata = cache_payload["metadata"]
        vectors = cache_payload["vectors"]

        self.dimension = int(metadata["dimension"])
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))

        self.parents = {int(parent["parent_id"]): ParentDocumentRecord(**parent) for parent in metadata["parents"]}
        self.chunk_metadata = {int(chunk["chunk_id"]): chunk for chunk in metadata["chunks"]}
        self.parent_sentences = {int(parent_id): sentences for parent_id, sentences in metadata["parent_sentences"].items()}
        self.parent_children = defaultdict(list)

        chunks: List[ChildChunkRecord] = []
        for chunk in metadata["chunks"]:
            chunk_record = ChildChunkRecord(**chunk)
            chunks.append(chunk_record)
            self.parent_children[chunk_record.parent_id].append(chunk_record.chunk_id)

        ids = np.array([chunk.chunk_id for chunk in chunks], dtype="int64")
        self.index.add_with_ids(self._normalize(vectors), ids)
        self.child_vectors_ready = True
        logger.debug("Loaded cached embeddings: count=%d dimension=%d", len(chunks), self.dimension)
        return chunks

    def add_documents(self, documents: Sequence[DocumentRecord], cache_dir: str | Path | None = None) -> List[ChildChunkRecord]:
        logger.debug("Ingesting source documents: count=%d", len(documents))

        cache_key = None
        if cache_dir is not None:
            cache_key = compute_cache_key(documents, self.window_size)
            cached = load_cached_artifacts(cache_dir, cache_key)
            if cached is not None:
                return self._load_cached_state(cached)

        parents, chunks, texts_to_embed, parent_sentences = build_parent_child_records(documents, self.window_size)
        for parent in parents:
            self.parents[parent.parent_id] = parent
        self.parent_sentences.update(parent_sentences)

        for chunk in chunks:
            self.parent_children[chunk.parent_id].append(chunk.chunk_id)
            self.chunk_metadata[chunk.chunk_id] = asdict(chunk)

        if not chunks:
            raise ValueError("no indexable chunks were produced from the supplied documents")

        vectors = np.asarray(embed_texts(texts_to_embed), dtype="float32")
        if vectors.shape[0] != len(chunks):
            raise RuntimeError("embedding count did not match chunk count")

        self._ensure_index(vectors.shape[1])
        vectors = self._normalize(vectors)
        ids = np.array([chunk.chunk_id for chunk in chunks], dtype="int64")
        assert self.index is not None
        self.index.add_with_ids(vectors, ids)
        self.child_vectors_ready = True

        if cache_dir is not None and cache_key is not None:
            save_cached_artifacts(cache_dir, cache_key, vectors, parents, chunks, parent_sentences)

        logger.debug("Indexed child chunks: count=%d parents=%d dimension=%d", len(chunks), len(self.parents), self.dimension)
        return chunks

    def _get_parent_window(self, parent_id: int, sentence_index: int) -> str:
        sentences = self.parent_sentences.get(parent_id, [])
        if not sentences:
            return self.parents[parent_id].text
        start = max(0, sentence_index - self.window_size)
        end = min(len(sentences), sentence_index + self.window_size + 1)
        return " ".join(sentences[start:end])

    def _expand_metadata(self, chunk_id: int, score: float) -> CitationContextRecord:
        metadata = self.chunk_metadata[chunk_id]
        parent_id = int(metadata["parent_id"])
        sentence_index = int(metadata["sentence_index"])
        parent_text = self.parents[parent_id].text
        expanded_text = self._get_parent_window(parent_id, sentence_index)
        return CitationContextRecord(
            chunk_id=chunk_id,
            parent_id=parent_id,
            source_id=int(metadata["source_id"]),
            source=metadata["source"],
            category=metadata["category"],
            section=metadata.get("section"),
            sentence_index=sentence_index,
            parent_start_sentence=int(metadata["parent_start_sentence"]),
            parent_end_sentence=int(metadata["parent_end_sentence"]),
            score=score,
            chunk_text=metadata["text"],
            expanded_text=expanded_text if expanded_text else parent_text,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[RetrievedChunkRecord]:
        if not self.child_vectors_ready:
            raise RuntimeError("no vectors have been indexed yet")
        if self.index is None or self.dimension is None:
            raise RuntimeError("index has not been initialized")
        logger.debug("Searching child chunks: top_k=%d dimension=%d", top_k, self.dimension)
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D array")
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"expected query vector with dimension {self.dimension}")

        query_vector = query_vector.astype("float32").reshape(1, -1)
        query_vector = self._normalize(query_vector)

        scores, ids = self.index.search(query_vector, top_k)
        results: List[RetrievedChunkRecord] = []

        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id == -1:
                continue
            citation = self._expand_metadata(int(chunk_id), float(score))
            results.append(
                RetrievedChunkRecord(
                    score=float(score),
                    chunk_id=int(chunk_id),
                    metadata={**self.chunk_metadata[int(chunk_id)], "citation": asdict(citation)},
                    child_text=citation.chunk_text,
                    parent_text=self.parents[citation.parent_id].text,
                    expanded_text=citation.expanded_text,
                )
            )

        return results

    def get_citation_bundle(self, chunk_id: int) -> Dict[str, Any]:
        metadata = self.chunk_metadata[chunk_id]
        parent = self.parents[int(metadata["parent_id"])]
        return {
            "chunk": metadata,
            "parent": asdict(parent),
            "citation": {
                "source": metadata["source"],
                "parent_id": metadata["parent_id"],
                "chunk_id": metadata["chunk_id"],
                "sentence_index": metadata["sentence_index"],
                "parent_start_sentence": metadata["parent_start_sentence"],
                "parent_end_sentence": metadata["parent_end_sentence"],
                "section": metadata.get("section"),
                "category": metadata["category"],
            },
        }

    def build_context_bundle(self, results: Sequence[RetrievedChunkRecord]) -> List[Dict[str, Any]]:
        grouped: Dict[int, RetrievedChunkRecord] = {}
        for result in results:
            existing = grouped.get(result.metadata["parent_id"])
            if existing is None or result.score > existing.score:
                grouped[result.metadata["parent_id"]] = result

        context_bundle: List[Dict[str, Any]] = []
        for parent_id, result in grouped.items():
            citation = result.metadata["citation"]
            context_bundle.append(
                {
                    "parent_id": parent_id,
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "source": citation["source"],
                    "category": citation["category"],
                    "section": citation["section"],
                    "sentence_index": citation["sentence_index"],
                    "child_text": result.child_text,
                    "parent_text": result.parent_text,
                    "expanded_text": result.expanded_text,
                    "citation": citation,
                }
            )

        logger.debug("Built context bundle: groups=%d", len(context_bundle))
        return context_bundle

    @staticmethod
    def format_context_for_prompt(context_bundle: Sequence[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in context_bundle:
            citation = item["citation"]
            parts.append(
                "\n".join(
                    [
                        f"[source={citation['source']} parent_id={citation['parent_id']} chunk_id={citation['chunk_id']} score={item['score']:.4f}]",
                        f"child: {item['child_text']}",
                        f"expanded: {item['expanded_text']}",
                    ]
                )
            )
        return "\n\n".join(parts)

    def search_with_citations(self, query_vector: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        results = self.search(query_vector, top_k=top_k)
        context_bundle = self.build_context_bundle(results)
        return {
            "results": results,
            "context_bundle": context_bundle,
            "prompt_context": self.format_context_for_prompt(context_bundle),
        }

    @staticmethod
    def _context_bundle_to_documents(context_bundle: Sequence[Dict[str, Any]]) -> List[LangChainDocument]:
        documents: List[LangChainDocument] = []
        for item in context_bundle:
            documents.append(
                LangChainDocument(
                    page_content=item["expanded_text"],
                    metadata={
                        "source": item["source"],
                        "category": item["category"],
                        "parent_id": item["parent_id"],
                        "chunk_id": item["chunk_id"],
                        "sentence_index": item["sentence_index"],
                        "section": item["section"],
                        "score": item["score"],
                        "child_text": item["child_text"],
                        "expanded_text": item["expanded_text"],
                        "citation": item["citation"],
                    },
                )
            )
        return documents

    def answer_question(self, question: str, top_k: int = 3, min_score: float = 0.55) -> RagAnswerRecord:
        logger.debug("Answering question with RAG: question_chars=%d top_k=%d", len(question), top_k)
        query_vector = np.asarray(embed_texts([question]), dtype="float32")[0]
        retrieval = self.search_with_citations(query_vector, top_k=top_k)
        context_bundle = retrieval["context_bundle"]
        if not context_bundle or max(result.score for result in retrieval["results"]) < min_score:
            logger.debug("No sufficiently relevant PDF context found for question")
            return RagAnswerRecord(
                question=question,
                answer="Not found in PDF.",
                results=retrieval["results"],
                context_bundle=context_bundle,
                prompt_context=retrieval["prompt_context"],
            )

        documents = self._context_bundle_to_documents(context_bundle)
        answer = generate_response(question, documents)
        logger.debug("RAG answer generated: answer_chars=%d context_docs=%d", len(answer), len(documents))
        return RagAnswerRecord(
            question=question,
            answer=answer,
            results=retrieval["results"],
            context_bundle=context_bundle,
            prompt_context=retrieval["prompt_context"],
        )