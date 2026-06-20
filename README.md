# Plain Vanilla RAG System

A Plain Vanilla Retrieval-Augmented Generation(RAG) system that implements the Small-to-Large Retrieval pattern using FAISS vector indexing, Google Gemini embeddings and chat models, and LangChain integration.

## Overview

This project improves context retention in RAG systems by:
- **Indexing small, granular chunks** (sentence-level) for precise retrieval matching
- **Expanding to richer context** (parent paragraphs or sentence windows) at answer time
- **Maintaining full provenance** for debugging and citation tracking

## Key Features

- **Small-to-Large Retrieval**: Index sentence-level child chunks while preserving parent document context
- **Flexible Expansion Modes**: 
  - Parent paragraph retrieval
  - Sentence window retrieval (e.g., -2/+2 adjacent sentences)
  - Hybrid mode combining both strategies
- **Rich Metadata**: Track source documents, parent blocks, and chunk positions
- **Gemini Integration**: Uses `gemini-embedding-2` (3072-dim vectors) for indexing and `gemini-1.5-flash` for generation
- **Multi-Format Document Support**: Load and process various document formats via MarkItDown

## Project Structure

```
.
├── main.py                    # Entry point for the RAG system
├── llm.py                     # Gemini chat and embedding helpers
├── rag_store.py               # Core RAG storage and retrieval logic
├── spec.md                    # Complete system specification
├── implementation_plan.md     # Development roadmap
├── requirements.txt           # Python dependencies
│
├── chunking/                  # Document chunking strategies
│   └── parent_child_chunking.py
│
├── ingestion/                 # Document loading and preprocessing
│   └── markitdown_loader.py
│
├── models/                    # Data models and structures
│   └── document_models.py
│
├── retrieval/                 # Retrieval and indexing
│   ├── rag_cache.py
│   └── rag_vector_store.py
│
└── Dataset/                   # Sample data and documents
```

## Requirements

- Python 3.10+
- FAISS (CPU or GPU version)
- Google Generative AI API key
- Dependencies in `requirements.txt`

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your Gemini API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

### Basic RAG Query

```python
from main import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline()

# Ingest documents
pipeline.ingest_documents("path/to/documents")

# Query
response = pipeline.query("Your question here")
print(response)
```

### Configuration

- **Debug Logging**: Enable via `ENABLE_DEBUG_LOGGING` environment variable
- **Embedding Model**: Uses `gemini-embedding-2` (3072 dimensions)
- **Chat Model**: Defaults to `gemini-1.5-flash`, falls back to `gemini-2.5-flash`
- **FAISS Index**: CPU-based on Windows

## Architecture

### Ingestion Pipeline
1. Load source documents (supports PDF, DOCX, TXT, etc.)
2. Split into sentence-level child chunks
3. Preserve metadata: source, category, section, parent IDs
4. Generate deterministic chunk IDs

### Indexing
1. Embed child chunks with `gemini-embedding-2`
2. Normalize embeddings
3. Store in FAISS index
4. Maintain metadata lookup tables
5. Keep parent documents for context expansion

### Retrieval & Generation
1. Embed user query with same embedding model
2. Retrieve top matching child chunks from FAISS
3. Expand each match to parent paragraph or sentence window
4. Deduplicate by parent ID
5. Generate answer using Gemini chat with expanded context
6. Include citation metadata in response

## Key Components

### `llm.py`
Provides LangChain-based helpers for:
- Chat generation with Gemini
- Text embedding with `gemini-embedding-2`
- Error handling and model fallback

### `rag_store.py`
Core RAG implementation with:
- Document ingestion and preprocessing
- Small-to-Large chunking strategies
- FAISS index management
- Metadata storage and retrieval

### `retrieval/rag_vector_store.py`
FAISS-based vector store with:
- Parent-child chunk relationships
- Multiple context enrichment modes
- Citation bundle generation

## Performance Notes

- **Vector Dimensions**: 3072 (gemini-embedding-2)
- **Chunk Strategy**: Sentence-level indexing with paragraph-level expansion
- **Platform**: Windows with `faiss-cpu`
- **API Model Fallback**: Automatic downgrade to `gemini-2.5-flash` if primary model unavailable

## Testing

Run test queries with:
```bash
python main.py
```

Test results are logged to `question_run.txt`

## Troubleshooting

- **API Key Issues**: Ensure `GOOGLE_API_KEY` is set in `.env`
- **FAISS Errors**: Install `faiss-cpu` or `faiss-gpu` as needed
- **Memory Issues**: Reduce chunk batch size or enable debug logging to identify large documents
- **Model 404 Errors**: System automatically falls back to `gemini-2.5-flash`

## Future Enhancements

- Adaptive chunking based on document structure
- Multi-modal retrieval (text + image)
- Real-time index updates
- Custom similarity scoring
- Distributed FAISS indexing

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure all tests pass and maintain the existing code structure.
