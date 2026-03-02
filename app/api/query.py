"""Query API supporting dense, sparse, and hybrid retrieval with optional reranking."""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.core.logger import logger
from app.core.monitor import track_latency
from app.ingestion.embedder import get_query_embedding
from app.retrieval.session_store import get_session
from app.retrieval.hybrid_retriever import hybrid_retrieve
from app.retrieval.reranker import rerank
from app.retrieval.rag_pipeline import rag_answer

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(..., description="Session ID from upload")
    top_k: int = Field(5, ge=1, le=20)
    mode: str = Field("hybrid", description="Retrieval mode: dense | sparse | hybrid")
    use_reranking: bool = Field(False, description="Apply cross-encoder reranking")
    candidate_k: int = Field(15, ge=5, le=50, description="Candidates to fetch before reranking")


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    is_grounded: bool
    faithfulness_score: float
    retrieval_method: str
    sources: List[str]


@router.post("/query", response_model=QueryResponse)
@track_latency
async def ask_question(request: QueryRequest):
    """Retrieve chunks using selected mode, optionally rerank, then generate answer."""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        session = get_session(request.session_id)
        if not session or not session.all_chunks:
            raise HTTPException(status_code=404, detail="Session not found or empty. Upload documents first.")

        logger.info(f"Query [{request.mode}]: {query[:80]}...")

        query_embedding = get_query_embedding(query)
        fetch_k = request.candidate_k if request.use_reranking else request.top_k

        # Retrieve based on mode
        if request.mode == "dense":
            raw = session.vector_store.search(query_embedding, top_k=fetch_k)
            chunks = [{"text": m["text"], "source": m["source"], "page_number": m.get("page_number")} for m, _d in raw]
            method = "dense"

        elif request.mode == "sparse":
            raw = session.sparse_retriever.search(query, top_k=fetch_k)
            chunks = [{"text": m["text"], "source": m["source"], "page_number": m.get("page_number")} for m, _s in raw]
            method = "sparse"

        else:  # hybrid
            fused = hybrid_retrieve(
                query=query,
                vector_store=session.vector_store,
                sparse_retriever=session.sparse_retriever,
                query_embedding=query_embedding,
                top_k=fetch_k,
                candidate_k=fetch_k,
            )
            chunks = fused
            method = "hybrid"

        if not chunks:
            return QueryResponse(
                answer="No relevant information found in the documents.",
                confidence=0.0,
                is_grounded=False,
                faithfulness_score=0.0,
                retrieval_method=method,
                sources=[],
            )

        # Optional reranking
        if request.use_reranking and len(chunks) > 1:
            chunks = rerank(query, chunks, top_k=request.top_k)
            method += "+rerank"
        else:
            chunks = chunks[:request.top_k]

        result = rag_answer(query, chunks, retrieval_method=method)

        return QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            is_grounded=result["is_grounded"],
            faithfulness_score=result["faithfulness_score"],
            retrieval_method=result["retrieval_method"],
            sources=result["sources"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
