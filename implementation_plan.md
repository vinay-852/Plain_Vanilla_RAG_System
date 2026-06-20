# Implementation Plan

## Objective

Integrate Small-to-Large Retrieval into the existing LangChain + Gemini pipeline using Gemini Embedding 2 for indexing and Gemini chat for generation.

## Current State

- `llm.py` provides Gemini chat and embedding helpers.
- `faiss_vector_store.py` now contains a small-to-large FAISS store with parent and child metadata.
- Debug logging is controlled by `ENABLE_DEBUG_LOGGING`.

## Steps

### 1. Ingestion and Chunking

- Load source documents into a parent document structure.
- Split each parent into sentence-level child chunks.
- Compute sentence offsets and preserve `source`, `category`, `section`, and parent IDs.
- Store child chunks with deterministic IDs and parent references.

### 2. Embedding and Indexing

- Embed only child chunks with `gemini-embedding-2` through `llm.embed_texts`.
- Normalize embeddings before adding them to FAISS.
- Store chunk metadata in a side table keyed by `chunk_id`.
- Store parent documents separately so they can be expanded on retrieval.

### 3. Retrieval Expansion

- Embed the user query with the same embedding model.
- Retrieve top child chunks from FAISS.
- Expand each match to its parent paragraph or sentence window.
- Deduplicate by `parent_id` so the final prompt is not redundant.

### 4. Citation Output

- Return child metadata and a citation bundle with:
  - `source`
  - `parent_id`
  - `chunk_id`
  - `sentence_index`
  - `parent_start_sentence`
  - `parent_end_sentence`
  - `section`
  - `category`
- Include the expanded context text used for generation.

### 5. Prompt Assembly and Generation

- Convert the deduplicated expanded chunks into prompt context.
- Pass the context into the existing Gemini generation helper.
- Keep the system instruction grounded in retrieved context only.

### 6. Debugging and Validation

- Use `ENABLE_DEBUG_LOGGING` to trace ingestion, embedding, retrieval, and expansion.
- Validate that source and parent metadata are present in every result.
- Validate that prompt context includes the expanded window and citation details.

## Suggested Follow-Up

- Replace the demo document list with real file loading.
- Add tests for sentence splitting, parent expansion, and citation bundle structure.
- Add a higher-level retrieval API so downstream code can request answer text plus citations in one call.