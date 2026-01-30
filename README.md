# Research Paper RAG Assistant

A **NotebookLLM-style** application that lets you chat with your research papers. Upload PDFs, ask questions, and get accurate answers with citations.

## What It Does

1. **Upload PDFs** → System extracts text, splits into chunks, and indexes them
2. **Ask Questions** → Finds relevant chunks using hybrid search (semantic + keyword)
3. **Get Answers** → LLM generates responses grounded in your documents with citations

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI                                    │
│                    (Upload PDFs, Chat, View Citations)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI BACKEND                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   INGESTION  │    │  RETRIEVAL   │    │  GENERATION  │                   │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤                   │
│  │ • PDF Parse  │    │ • Dense      │    │ • Llama LLM  │                   │
│  │ • Chunking   │───▶│   (FAISS)    │───▶│ • Citations  │                   │
│  │ • Embeddings │    │ • Sparse     │    │ • Grounding  │                   │
│  │ • Metadata   │    │   (BM25)     │    │   Check      │                   │
│  └──────────────┘    │ • Reranking  │    └──────────────┘                   │
│                      └──────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
             ┌───────────┐     ┌───────────┐     ┌───────────┐
             │   AWS S3  │     │  FAISS    │     │  MLflow   │
             │  (PDFs)   │     │  (Index)  │     │  (Metrics)│
             └───────────┘     └───────────┘     └───────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **Backend** | FastAPI, Python 3.11+ |
| **PDF Parsing** | Unstructured, PyMuPDF |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Store** | FAISS |
| **Sparse Search** | BM25 (rank-bm25) |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM) |
| **LLM** | Llama 3.3 70B via Groq / AWS SageMaker |
| **Grounding** | NLI Model (DeBERTa) |
| **Storage** | AWS S3 |
| **Evaluation** | RAGAS, MLflow |
| **Agents** | AutoGen |

## Key Features

### Ingestion
- **Smart Chunking**: Uses Unstructured library to preserve document structure
- **Rich Metadata**: Extracts page numbers, sections, and headings
- **S3 Storage**: Uploaded PDFs stored in AWS S3

### Retrieval
- **Hybrid Search**: Combines semantic (FAISS) + keyword (BM25) search
- **Reranking**: Cross-encoder reranks results for better accuracy
- **Lost-in-Middle**: Reorders context to prevent attention decay

### Generation
- **Grounded Answers**: NLI model verifies claims against source text
- **Citations**: Every answer includes source references with page numbers
- **Faithfulness Score**: Shows how well the answer is supported

### Evaluation (RAGAS Metrics)
- **Faithfulness**: Is the answer supported by the context?
- **Answer Relevance**: Does the answer address the question?
- **Context Precision**: Are retrieved chunks relevant?
- **Context Recall**: Are all needed chunks retrieved?

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Run the backend
uvicorn app.main:app --reload

# 4. Run the UI (new terminal)
streamlit run ui/streamlit_app.py
```

## Environment Variables

```bash
# Required
LLM_API_KEY=your_groq_api_key
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions

# Optional - AWS
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=your-bucket-name

# Optional - SageMaker
SAGEMAKER_ENDPOINT_NAME=your-llama-endpoint
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload and index a PDF |
| POST | `/api/query` | Ask a question |
| GET | `/api/sessions` | List active sessions |
| DELETE | `/api/session/{id}` | Delete a session |
| GET | `/api/health` | Health check |

### Example Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "session_id": "your-session-id",
    "use_hybrid": true,
    "use_reranking": true
  }'
```

Response includes:
- `answer`: Generated response
- `citations`: List of sources with page numbers
- `confidence`: Overall confidence score
- `faithfulness_score`: How grounded the answer is

## Project Structure

```
app/
├── ingestion/          # PDF loading, chunking, embeddings
│   ├── pdf_loader.py   # PyMuPDF + structure detection
│   ├── chunker.py      # Unstructured-based chunking
│   └── embedder.py     # Sentence Transformers
│
├── retrieval/          # Search and retrieval
│   ├── vector_store.py # FAISS index
│   ├── sparse_retriever.py  # BM25
│   ├── hybrid_retriever.py  # Dense + Sparse fusion
│   ├── reranker.py     # Cross-encoder reranking
│   └── session_store.py     # Session-based indexes
│
├── generation/         # LLM and response generation
│   ├── llm.py          # Groq/SageMaker client
│   ├── prompt.py       # Prompt templates
│   ├── hallucination.py     # NLI grounding check
│   └── citation.py     # Citation extraction
│
├── evaluation/         # Quality metrics
│   ├── ragas_eval.py   # RAGAS metrics
│   ├── mlflow_tracker.py    # Experiment tracking
│   └── synthetic_qa.py # Test data generation
│
├── agents/             # AutoGen orchestration
│   ├── rag_agent.py    # Main orchestrator
│   ├── retrieval_agent.py
│   └── grounding_agent.py
│
├── api/                # FastAPI routes
│   ├── upload.py
│   └── query.py
│
└── core/               # Config and utilities
    ├── config.py
    ├── logger.py
    └── monitor.py

ui/
└── streamlit_app.py    # Chat interface
```

## How It Works (Simple Explanation)

### 1. Upload Phase
```
PDF → Extract Text → Split into Chunks → Create Embeddings → Store in FAISS
```
- Each chunk keeps track of which page and section it came from

### 2. Query Phase
```
Question → Find Similar Chunks → Rerank → Build Prompt → Generate Answer → Verify
```
- **Hybrid Search**: Finds chunks that are semantically similar AND contain matching keywords
- **Reranking**: Uses a smarter model to pick the best chunks
- **Verification**: Checks if the answer is actually supported by the chunks

### 3. Session Management
- Each browser session gets its own vector index
- Upload multiple PDFs to the same session
- Ask questions across all your uploaded documents

## Docker

```bash
# Build
docker build -t research-rag:latest .

# Run
docker run -p 8000:8000 \
  -e LLM_API_KEY=your_key \
  research-rag:latest
```

## Evaluation

Run RAGAS evaluation on a test dataset:

```bash
python -m app.evaluation.ragas_eval --dataset data/eval/test_qa.json
```

View results in MLflow:
```bash
mlflow ui
# Open http://localhost:5000
```

## License

MIT
