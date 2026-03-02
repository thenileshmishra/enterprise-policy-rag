"""Cross-encoder reranker using sentence-transformers."""

from typing import List, Dict
from sentence_transformers import CrossEncoder
from app.core.logger import logger

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rerank chunks using a cross-encoder model.
    Each chunk must have a 'text' key.
    Returns top_k chunks sorted by cross-encoder relevance score.
    """
    if not chunks:
        return []

    model = get_reranker()
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(scores[i]), 4)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    logger.info(f"Reranked {len(chunks)} chunks, returning top {top_k}")
    return ranked[:top_k]
