"""Simple query API for a minimal RAG flow."""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.core.logger import logger
from app.core.monitor import track_latency
from app.retrieval.retriever import retrieve_topk
from app.retrieval.rag_pipeline import rag_answer

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    confidence: float
    sources: List[str]


@router.post("/query", response_model=QueryResponse)
@track_latency
async def ask_question(request: QueryRequest):
    """Retrieve top chunks and generate an answer."""
    try:
        query = request.query.strip()

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {query[:100]}...")
        chunks = retrieve_topk(query, top_k=request.top_k)

        if not chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the documents to answer your question. Please make sure you have uploaded relevant documents.",
                confidence=0.0,
                sources=[],
            )

        result = rag_answer(query, chunks)

        return QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))
