# Advanced RAG Pipeline

A production-style Retrieval-Augmented Generation system with hybrid retrieval, cross-encoder reranking, and NLI-based faithfulness detection. Built with FastAPI + Streamlit.

## Architecture

```
Query ──► Embedding ──► FAISS (Dense)  ──┐
     │                                    ├──► RRF Fusion ──► Cross-Encoder Rerank ──► LLM Generation
     └──────────────► BM25 (Sparse)  ──┘                                                    │
                                                                                             ▼
                                                                              NLI Faithfulness Check
                                                                              (is_grounded + score)
```

**Retrieval Pipeline:**
1. **Dense retrieval** — FAISS IndexFlatL2 with `all-MiniLM-L6-v2` embeddings (384-dim)
2. **Sparse retrieval** — BM25 (rank-bm25) for keyword matching
3. **Hybrid fusion** — Reciprocal Rank Fusion (RRF) combines both: `score += 1/(k + rank)`
4. **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` rescores top candidates
5. **Lost-in-the-middle reordering** — places best chunks at start/end of context window
6. **NLI faithfulness** — `nli-deberta-v3-small` checks entailment between answer and context

## Project Structure

```text
app/
  api/
    health.py            # Health check
    upload.py            # PDF upload (multi-doc sessions)
    query.py             # Query with mode selection
  core/
    config.py            # Environment settings
    exception.py         # Custom exceptions
    logger.py            # Loguru logging
    monitor.py           # Latency tracking
  ingestion/
    pdf_loader.py        # PyMuPDF PDF extraction
    chunker.py           # Recursive text splitting
    embedder.py          # SentenceTransformer embeddings
  retrieval/
    vector_store.py      # FAISS index
    sparse_retriever.py  # BM25 index
    hybrid_retriever.py  # RRF fusion
    reranker.py          # Cross-encoder reranker
    session_store.py     # Multi-document sessions
    rag_pipeline.py      # End-to-end RAG orchestration
  generation/
    prompt.py            # Prompt builder + lost-in-middle reorder
    llm.py               # LLM client (Groq/HuggingFace/fallback)
    hallucination.py     # NLI + similarity faithfulness scoring
ui/
  streamlit_app.py       # Interactive web UI
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```bash
LLM_API_KEY=your_groq_key
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
```

Without API keys, the app uses a local extractive fallback.

## Run

**Backend:**
```bash
uvicorn app.main:app --reload
```

**UI (separate terminal):**
```bash
streamlit run ui/streamlit_app.py
```

- API docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://localhost:8501

## API Endpoints

### POST `/api/upload`
Upload a PDF to a session. Returns `session_id` for subsequent queries.

```json
// Form data: file (PDF) + optional session_id
// Response:
{
  "session_id": "abc12345",
  "chunks_indexed": 42,
  "document_name": "report.pdf",
  "total_documents": 1
}
```

### POST `/api/query`
Query documents with configurable retrieval.

```json
// Request:
{
  "query": "What is the refund policy?",
  "session_id": "abc12345",
  "mode": "hybrid",
  "use_reranking": true,
  "top_k": 5
}

// Response:
{
  "answer": "The refund policy states...",
  "confidence": 0.82,
  "is_grounded": true,
  "faithfulness_score": 0.82,
  "retrieval_method": "hybrid+rerank",
  "sources": ["report.pdf"]
}
```

**Retrieval modes:** `dense` | `sparse` | `hybrid`

## Why These Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Dense retrieval | FAISS + MiniLM | Fast semantic search, small model footprint |
| Sparse retrieval | BM25 | Captures exact keyword matches that embeddings miss |
| Fusion | RRF (k=60) | Robust rank-based fusion, no score calibration needed |
| Reranker | ms-marco cross-encoder | Joint query-doc scoring for precise relevance |
| Faithfulness | NLI (DeBERTa) | Detects contradictions that similarity alone misses |
| Context order | Lost-in-middle | LLMs attend less to middle positions |

## Demo Flow

1. Start backend + UI
2. Upload 2 PDFs (creates a session)
3. Query in `dense`, then `hybrid`, then `hybrid+rerank`
4. Compare faithfulness scores and source quality
5. Toggle reranking to see relevance improvement
