# Enterprise Policy RAG

Production-grade Retrieval-Augmented Generation system for enterprise policy documents with evaluation and monitoring.

## Overview

Enterprise policies and compliance documents are difficult to search and reason over. This system enables natural language querying of policy documents using semantic search and LLM-powered answer generation with citations.

Built with production-ready evaluation metrics, hallucination detection, and performance monitoring to ensure reliable, grounded responses for compliance-critical use cases.

## Tech Stack

**Backend & API**

- FastAPI for REST endpoints
- Python 3.11+ with Pydantic for type safety

**RAG Pipeline**

- PyMuPDF for PDF parsing and text extraction
- LangChain for text splitting and chunking
- Sentence Transformers for embeddings
- FAISS and ChromaDB for vector storage
- LLM integration for answer generation

**Evaluation & Monitoring**

- Custom retrieval evaluation metrics
- Answer quality assessment
- Hallucination detection
- Latency tracking and performance monitoring

**Frontend & Deployment**

- Streamlit UI for document upload and querying
- Loguru for structured logging
- Pytest for testing

## Key Features

- Robust PDF ingestion with metadata preservation
- Semantic chunking optimized for retrieval
- Vector-based similarity search with configurable top-k
- Grounded answer generation with source citations
- Hallucination detection and quality safeguards
- Retrieval and answer evaluation pipelines
- Performance monitoring with latency tracking
- RESTful API for integration
- Interactive web interface

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI backend
uvicorn app.main:app --reload

# Run Streamlit UI (separate terminal)
streamlit run ui/streamlit_app.py

# Run tests
pytest tests/
```

## Docker

Build and run locally with Docker:

```bash
# build
docker build -t enterprise-policy-rag:local .
# run
docker run -p 8000:8000 -e HF_API_KEY=your_hf_key -v $(pwd)/data/processed:/data/processed enterprise-policy-rag:local
```

## Deploy to Render

Add the following GitHub secrets:
- `RENDER_API_KEY` (Render service API key)
- `RENDER_SERVICE_ID` (Render service ID)

You can either let GitHub Actions push the image to GHCR and trigger a Render deploy (configured in `.github/workflows/ci-cd.yml`), or connect Render directly to the GitHub repo and use the provided `.render.yaml` configuration.

## FAISS index persistence

You can persist the FAISS index in one of two ways:

1. Use Render / provider persistent disk (if available) and mount `/data/processed` as a volume.
2. Use S3/DigitalOcean Spaces by setting:

- `S3_BUCKET`, `S3_INDEX_KEY`, `S3_META_KEY`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `S3_ENDPOINT` (for Spaces)

On startup the app will try to download the index from S3 if `S3_BUCKET` is configured; otherwise it will use local `/data/processed/faiss.index`.

**API Endpoints**

- `POST /api/upload` - Upload and index PDF documents
- `POST /api/query` - Query indexed documents
- `GET /api/health` - Health check

## Project Structure

```
app/
├── ingestion/      # PDF loading, chunking, embeddings
├── retrieval/      # Vector store, retrieval, RAG pipeline
├── generation/     # LLM integration, prompts, hallucination detection
├── evaluation/     # Retrieval and answer quality metrics
├── api/            # FastAPI endpoints
└── core/           # Config, logging, monitoring, exceptions

ui/
└── streamlit_app.py   # Interactive web interface

tests/
└── test_*.py          # Unit and integration tests
```

## What This Project Demonstrates

**Machine Learning & AI**

- End-to-end RAG system design and implementation
- Semantic search with vector databases
- LLM integration with prompt engineering
- Hallucination detection techniques

**Software Engineering**

- Clean architecture with separation of concerns
- RESTful API design
- Error handling and custom exceptions
- Production logging and monitoring
- Type safety with Pydantic

**Data Engineering**

- Document processing pipelines
- Text chunking strategies
- Embedding generation and indexing
- Metadata management for traceability

**MLOps & Testing**

- Automated evaluation pipelines
- Retrieval and generation quality metrics
- Performance monitoring and latency tracking
- Unit testing for critical components
