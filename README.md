# Simple RAG (Beginner Friendly)

This project is intentionally minimal:

- Upload a PDF
- Split it into chunks
- Create embeddings
- Store embeddings in FAISS
- Ask questions and get grounded answers

No agents, no orchestration layer, no hybrid/reranking/session complexity.

## Project Structure

```text
app/
  api/
    health.py
    upload.py
    query.py
  core/
    config.py
    exception.py
    logger.py
    monitor.py
  ingestion/
    pdf_loader.py
    chunker.py
    embedder.py
  retrieval/
    vector_store.py
    retriever.py
    rag_pipeline.py
  generation/
    prompt.py
    llm.py
    hallucination.py
  main.py
ui/
  streamlit_app.py
requirements.txt
```

## 1) Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Run Backend (Local)

```bash
uvicorn app.main:app --reload
```

- API docs: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/api/health>

## 3) Run UI (Optional)

In a new terminal:

```bash
source venv/bin/activate
streamlit run ui/streamlit_app.py
```

## 4) Use It

1. Upload a PDF from `/api/upload` or the Streamlit UI.
2. Ask questions at `/api/query` or in the UI.

## Environment Variables (Optional)

Set these in `.env` for real LLM answers:

```bash
LLM_API_KEY=your_key
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
```

If not set, the app still works using a local fallback answer mode.
