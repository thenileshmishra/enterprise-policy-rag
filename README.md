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
