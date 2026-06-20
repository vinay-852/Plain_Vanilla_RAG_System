# Small-to-Large Retrieval Spec

## Goal

Improve RAG context retention by indexing small, granular units for retrieval while expanding retrieved matches back to richer surrounding context at answer time.

This system uses a Small-to-Large Retrieval pattern, also called Parent Document Retrieval:

- Small chunks are embedded and indexed for precise matching.
- Retrieved chunks are expanded to a larger parent scope, such as the original paragraph or a neighboring sentence window.
- Metadata keeps every child chunk traceable to its source document and parent structure.

## Problem Statement

Current retrieval can miss important details when chunks are too large, or lose context when chunks are too small. The target design should:

- improve recall for specific facts,
- preserve enough surrounding context for grounded generation,
- keep provenance visible for debugging and citation,
- integrate into the existing LangChain + Gemini pipeline with minimal disruption.

## Core Requirements

### 1. Granularity

Documents must be split into small, indexable chunks.

Recommended default:

- sentence-level chunks for the retrieval index,
- optional fallback to clause or token-based splitting when a sentence is too long,
- stable chunk IDs so parent-child relationships remain deterministic.

### 2. Context Enrichment

When a child chunk is retrieved, the system must also retrieve richer context.

Supported enrichment modes:

- parent paragraph retrieval,
- sentence window retrieval using adjacent sentences, for example `-2/+2`,
- hybrid mode that returns both parent paragraph and a sentence window when available.

### 3. Metadata

Each child chunk must carry metadata pointing back to:

- original source document,
- parent paragraph or parent block ID,
- sentence index within the parent,
- document order,
- optional section heading or document title,
- optional character offsets for traceability.

### 4. Workflow

The full retrieval workflow must support:

1. document ingestion,
2. document normalization and sentence segmentation,
3. child chunk creation with metadata,
4. embedding and indexing of child chunks,
5. retrieval of top child chunks,
6. expansion to parent or window context,
7. prompt assembly,
8. Gemini response generation,
9. provenance output for debugging and inspection.

## Proposed Architecture

### Data Model

Each source document is split into one or more parent units, usually paragraphs.

Each parent unit is split into child sentence chunks.

Suggested child chunk fields:

- `chunk_id`: unique child identifier,
- `parent_id`: identifier of the parent paragraph or parent block,
- `source_id`: original document identifier,
- `source`: file name or logical origin,
- `text`: child sentence text,
- `sentence_index`: position of the sentence in the parent,
- `parent_start_sentence`: first sentence index in the parent window,
- `parent_end_sentence`: last sentence index in the parent window,
- `section`: optional section title,
- `category`: optional topical label,
- `char_start` and `char_end`: optional offsets in the source document.

Suggested parent fields:

- `parent_id`,
- `source_id`,
- `text`: original paragraph or expanded context block,
- `children`: list of child chunk IDs,
- `section`,
- `category`,
- `source`,
- optional offsets and ordering.

### Index Layout

Store embeddings only for child chunks.

Keep metadata in a side store keyed by `chunk_id`.

Recommended retrieval store contents:

- vector index: child chunk embeddings,
- chunk metadata store: child chunk provenance,
- parent store: full parent text or window text,
- adjacency store: ordered sibling relationships for window expansion.

### Retrieval Flow

1. User query is embedded.
2. Vector search returns top child chunks.
3. For each child chunk, the system resolves the parent ID.
4. The system expands the result using one of the configured enrichment modes:
   - parent paragraph,
   - sentence window,
   - hybrid.
5. The expanded text is deduplicated and ranked if needed.
6. The final context bundle is assembled for Gemini.
7. The response includes answer text and provenance metadata.

## Chunking Strategy

### Primary Split Unit

Use sentence segmentation as the default child chunking unit.

Rationale:

- better precision than paragraph-level indexing,
- more resilient for fact lookup,
- smaller embeddings improve semantic matching,
- easier to reconstruct context from neighboring sentences.

### Parent Unit

Use the original paragraph as the parent unit when the source is structured that way.

If the source does not have reliable paragraphs, use a sentence window as the parent unit.

Recommended parent window default:

- `window_size = 2` sentences on each side,
- do not cross document boundaries,
- optionally stop at section boundaries.

### Long Sentence Handling

If a sentence is unusually long, split it further into sub-sentence spans using one of:

- clause segmentation,
- token-based fallback splitting,
- recursive splitting with overlap.

This should only be used when the sentence would otherwise exceed embedding or prompt budgets.

## Context Expansion Rules

When a child chunk is retrieved:

- always include the child chunk itself,
- expand to the parent paragraph if available,
- otherwise expand to the configured sentence window,
- preserve original ordering,
- avoid duplicate text if multiple retrieved chunks point to the same parent.

Recommended deduplication rule:

- rank child chunks by score,
- group by `parent_id`,
- keep the highest-scoring child as the primary anchor,
- expand context once per parent group.

## Metadata Requirements

Every child chunk must include enough metadata to reconstruct provenance.

Minimum required fields:

- `chunk_id`,
- `parent_id`,
- `source_id`,
- `source`,
- `text`,
- `sentence_index`.

Strongly recommended fields:

- `section`,
- `category`,
- `parent_start_sentence`,
- `parent_end_sentence`,
- `char_start`,
- `char_end`.

Metadata must be returned with search results so the UI or debug logs can show:

- where the chunk came from,
- which parent it belongs to,
- what expanded context was attached,
- which chunk triggered the final answer.

## LangChain Integration

The existing pipeline should adapt as follows:

- ingestion stage creates parent and child records,
- embeddings are generated only for child records,
- FAISS stores child vectors and child metadata references,
- retrieval resolves parent context before prompt assembly,
- Gemini receives the enriched context bundle through the existing generation helpers.

Recommended LangChain responsibilities:

- document loading and normalization,
- sentence splitting and parent grouping,
- embedding generation,
- retrieval and re-ranking if needed,
- prompt assembly,
- final response generation.

## Debuggability and Observability

The system should support debug logging for:

- number of parent documents ingested,
- number of child chunks created,
- mapping from child chunk to parent ID,
- top retrieval results and scores,
- how much context was expanded for each retrieved item,
- final prompt size before generation.

The existing `ENABLE_DEBUG_LOGGING` switch should control these logs.

## Acceptance Criteria

The design is complete when:

- documents are indexed at sentence granularity,
- retrieval expands to parent or window context,
- child metadata includes source and parent references,
- duplicate parent context is deduplicated,
- generated answers use the expanded context,
- provenance can be inspected from the returned search output,
- logging can be enabled or disabled without code changes.

## Proposed Next Step

After review, add an `implementation_plan.md` that maps this spec onto the current codebase by defining:

- new chunking utilities,
- FAISS metadata shape updates,
- parent lookup storage,
- retrieval expansion logic,
- prompt assembly changes,
- validation tests.
