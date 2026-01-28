"""
Query API endpoint with hybrid retrieval, reranking, and citations.
Supports session-based queries and enhanced response formatting.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.core.exception import CustomException
from app.core.logger import logger
from app.core.monitor import track_latency
from app.retrieval.retriever import retrieve_topk
from app.retrieval.rag_pipeline import rag_answer
from app.retrieval.session_store import get_session_store, search_session
from app.retrieval.hybrid_retriever import hybrid_search, get_hybrid_retriever
from app.retrieval.reranker import rerank_results
from app.generation.prompt import build_enhanced_rag_prompt
from app.generation.llm import generate_answer
from app.generation.hallucination import comprehensive_faithfulness_check
from app.generation.citation import extract_and_format_citations

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for multi-document queries")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    use_hybrid: bool = Field(True, description="Use hybrid (dense + sparse) retrieval")
    use_reranking: bool = Field(True, description="Use cross-encoder reranking")
    include_citations: bool = Field(True, description="Include detailed citations")
    dense_weight: float = Field(0.5, ge=0, le=1, description="Weight for dense retrieval")


class Citation(BaseModel):
    """Citation model for response."""
    id: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    confidence: float
    sources: List[str]
    citations: Optional[List[Citation]] = None
    faithfulness_score: Optional[float] = None
    is_grounded: Optional[bool] = None
    retrieval_method: str = "hybrid"


@router.post("/query", response_model=QueryResponse)
@track_latency
async def ask_question(request: QueryRequest):
    """
    Query the RAG system with enhanced retrieval and citations.

    - Supports session-based multi-document queries
    - Uses hybrid retrieval (dense + BM25 sparse)
    - Applies cross-encoder reranking
    - Includes faithfulness scoring
    - Returns structured citations
    """
    try:
        query = request.query.strip()

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing query: {query[:100]}...")

        # Retrieve documents based on session mode
        if request.session_id:
            # Session-based retrieval with hybrid search
            if request.use_hybrid:
                chunks = hybrid_search(
                    session_id=request.session_id,
                    query=query,
                    k=request.top_k * 3 if request.use_reranking else request.top_k,
                    dense_weight=request.dense_weight,
                    sparse_weight=1.0 - request.dense_weight,
                )
                retrieval_method = "hybrid"
            else:
                chunks = search_session(
                    session_id=request.session_id,
                    query=query,
                    k=request.top_k * 3 if request.use_reranking else request.top_k,
                )
                retrieval_method = "dense"
        else:
            # Global index retrieval (backward compatible)
            chunks = retrieve_topk(query, top_k=request.top_k * 3 if request.use_reranking else request.top_k)
            retrieval_method = "dense"

        if not chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the documents to answer your question. Please make sure you have uploaded relevant documents.",
                confidence=0.0,
                sources=[],
                citations=[],
                faithfulness_score=0.0,
                is_grounded=False,
                retrieval_method=retrieval_method,
            )

        # Apply reranking if enabled
        if request.use_reranking and len(chunks) > 1:
            chunks = rerank_results(
                query=query,
                results=chunks,
                top_k=request.top_k,
                use_diversity=False,
            )
            retrieval_method += "+rerank"
        else:
            chunks = chunks[:request.top_k]

        # Build prompt with lost-in-middle handling
        prompt = build_enhanced_rag_prompt(
            query=query,
            contexts=chunks,
            apply_lost_in_middle=True,
        )

        # Generate answer
        answer = generate_answer(prompt)

        # Compute faithfulness
        context_texts = [c.get("text", "") for c in chunks]
        faithfulness_result = comprehensive_faithfulness_check(
            answer=answer,
            contexts=context_texts,
            use_nli=True,
            use_similarity=True,
        )

        # Extract citations
        citations = []
        if request.include_citations:
            citation_result = extract_and_format_citations(answer, chunks)
            citations = [
                Citation(
                    id=str(i + 1),
                    source=cit.get("source", "Unknown"),
                    page=cit.get("page"),
                    section=cit.get("section"),
                    snippet=cit.get("snippet", "")[:200] if cit.get("snippet") else None,
                )
                for i, cit in enumerate(citation_result.get("citations", []))
            ]

        # If no citations extracted, create from sources
        if not citations:
            seen_sources = set()
            for i, chunk in enumerate(chunks):
                source = chunk.get("source", "Unknown")
                if source not in seen_sources:
                    citations.append(Citation(
                        id=str(len(citations) + 1),
                        source=source,
                        page=chunk.get("page_number"),
                        section=chunk.get("section"),
                        snippet=chunk.get("text", "")[:200],
                    ))
                    seen_sources.add(source)

        # Extract unique sources
        sources = list(set(c.get("source", "Unknown") for c in chunks))

        # Compute confidence (combination of faithfulness and retrieval quality)
        retrieval_confidence = sum(
            c.get("score", c.get("hybrid_score", c.get("rerank_score", 0.5)))
            for c in chunks
        ) / len(chunks) if chunks else 0

        confidence = (faithfulness_result["combined_score"] + retrieval_confidence) / 2

        logger.info(
            f"Query processed: faithfulness={faithfulness_result['combined_score']:.3f}, "
            f"confidence={confidence:.3f}, sources={len(sources)}"
        )

        return QueryResponse(
            answer=answer,
            confidence=round(confidence, 4),
            sources=sources,
            citations=citations,
            faithfulness_score=faithfulness_result["combined_score"],
            is_grounded=faithfulness_result["is_faithful"],
            retrieval_method=retrieval_method,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/simple")
@track_latency
async def simple_query(payload: dict):
    """
    Simple query endpoint for backward compatibility.
    """
    try:
        question = payload.get("query")

        if not question:
            raise HTTPException(status_code=400, detail="Query missing")

        if len(question) > 500:
            raise CustomException("Query too long")

        logger.info(f"Simple query: {question}")

        chunks = retrieve_topk(question, top_k=5)
        result = rag_answer(question, chunks)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/evaluate")
async def evaluate_query(
    query: str = Query(..., description="Query to evaluate"),
    session_id: Optional[str] = Query(None, description="Session ID"),
):
    """
    Evaluate retrieval quality for a query without generating an answer.
    Useful for debugging and tuning retrieval.
    """
    try:
        if session_id:
            dense_results = search_session(session_id, query, k=10)
            hybrid_results = hybrid_search(session_id, query, k=10)
        else:
            dense_results = retrieve_topk(query, top_k=10)
            hybrid_results = dense_results  # No hybrid for global index

        return {
            "query": query,
            "dense_results": [
                {
                    "source": r.get("source"),
                    "page": r.get("page_number"),
                    "score": r.get("score", r.get("distance")),
                    "text_preview": r.get("text", "")[:150],
                }
                for r in dense_results[:5]
            ],
            "hybrid_results": [
                {
                    "source": r.get("source"),
                    "page": r.get("page_number"),
                    "hybrid_score": r.get("hybrid_score"),
                    "text_preview": r.get("text", "")[:150],
                }
                for r in hybrid_results[:5]
            ],
            "num_dense": len(dense_results),
            "num_hybrid": len(hybrid_results),
        }

    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))
