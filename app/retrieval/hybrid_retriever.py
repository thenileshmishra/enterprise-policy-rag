"""Hybrid retriever: combines FAISS dense + BM25 sparse using Reciprocal Rank Fusion."""

from typing import List, Dict, Tuple
from app.core.logger import logger


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Dict, float]]],
    k: int = 60,
) -> List[Dict]:
    """
    Merge multiple ranked result lists using RRF.

    Formula: score(doc) = sum(1 / (k + rank)) across all lists.
    Each result list is a list of (metadata_dict, score) tuples.
    Uses 'chunk_id' or 'text' as document identity key.
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for results in result_lists:
        for rank, (meta, _score) in enumerate(results, start=1):
            doc_key = meta.get("chunk_id", meta["text"][:100])
            fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (k + rank)
            doc_map[doc_key] = meta

    # Sort by fused score descending
    sorted_keys = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)

    results = []
    for key in sorted_keys:
        doc = doc_map[key].copy()
        doc["rrf_score"] = round(fused_scores[key], 6)
        results.append(doc)

    return results


def hybrid_retrieve(
    query: str,
    vector_store,
    sparse_retriever,
    query_embedding,
    top_k: int = 5,
    candidate_k: int = 15,
) -> List[Dict]:
    """
    Run dense + sparse retrieval, fuse with RRF, return top_k results.
    """
    # Dense retrieval from FAISS
    dense_results = vector_store.search(query_embedding, top_k=candidate_k)
    logger.info(f"Dense retrieval returned {len(dense_results)} results")

    # Sparse retrieval from BM25
    sparse_results = sparse_retriever.search(query, top_k=candidate_k)
    logger.info(f"Sparse retrieval returned {len(sparse_results)} results")

    # Fuse using RRF
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    logger.info(f"RRF fusion produced {len(fused)} unique results")

    return fused[:top_k]
