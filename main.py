from __future__ import annotations

import argparse
from pathlib import Path

from ingestion.markitdown_loader import load_documents_from_directory
from llm import configure_logging
from retrieval.rag_vector_store import SmallToLargeFaissStore


def _format_citations(context_bundle: list[dict[str, object]]) -> str:
    if not context_bundle:
        return "- None"

    lines: list[str] = []
    for index, item in enumerate(context_bundle, start=1):
        citation = item.get("citation", {})
        source = citation.get("source", item.get("source", "unknown"))
        section = citation.get("section") or item.get("section") or "n/a"
        parent_id = citation.get("parent_id", item.get("parent_id", "n/a"))
        chunk_id = citation.get("chunk_id", item.get("chunk_id", "n/a"))
        sentence_index = citation.get("sentence_index", item.get("sentence_index", "n/a"))
        score = item.get("score", 0.0)
        lines.append(
            f"{index}. {source} | section: {section} | parent_id: {parent_id} | chunk_id: {chunk_id} | sentence: {sentence_index} | score: {score:.4f}"
        )

    return "\n".join(lines)


def _format_relevant_chunks(context_bundle: list[dict[str, object]]) -> str:
    if not context_bundle:
        return "- None"

    chunks: list[str] = []
    for index, item in enumerate(context_bundle, start=1):
        citation = item.get("citation", {})
        source = citation.get("source", item.get("source", "unknown"))
        chunk_text = item.get("child_text", "")
        expanded_text = item.get("expanded_text", "")
        chunks.append(
            "\n".join(
                [
                    f"{index}. {source}",
                    f"   Chunk: {chunk_text}",
                    f"   Expanded context: {expanded_text}",
                ]
            )
        )

    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over PDF files in the Dataset folder.")
    parser.add_argument("question", nargs="?", help="Optional one-off question to ask")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore cached embeddings and rebuild them")
    args = parser.parse_args()

    configure_logging()

    dataset_dir = Path(__file__).resolve().parent / "Dataset"
    documents = load_documents_from_directory(dataset_dir)
    if not documents:
        raise RuntimeError(f"No documents were loaded from {dataset_dir}")

    store = SmallToLargeFaissStore(window_size=2)
    cache_dir = None if args.rebuild_cache else Path(__file__).resolve().parent / ".rag_cache"
    child_chunks = store.add_documents(documents, cache_dir=cache_dir)
    if not child_chunks:
        raise RuntimeError("No child chunks were created for indexing")

    def ask_and_print(question: str) -> None:
        rag_answer = store.answer_question(question, top_k=3)
        print("\nQuestion:")
        print(question)
        print("\nAnswer:")
        print(rag_answer.answer)
        print("\nCitations:")
        print(_format_citations(rag_answer.context_bundle))
        print("\nRelevant Chunks:")
        print(_format_relevant_chunks(rag_answer.context_bundle))

    if args.question:
        ask_and_print(args.question)

    print("Loaded PDFs from Dataset. Type a question, or 'exit' to quit.")
    while True:
        question = input("Question: ").strip()
        if not question or question.lower() in {"exit", "quit", "q"}:
            break
        ask_and_print(question)


if __name__ == "__main__":
    main()